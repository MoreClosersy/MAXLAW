import { ParsedChunk } from "./EmbeddingRetriever.js";
import { parseArticleNo } from "./articleNo.js";

export interface VectorStoreItem extends ParsedChunk {
    embedding: number[];
    /** articleNo 的**数值**形式，入库时算一次。条号比较一律走数值——理由见 articleNo.ts */
    articleNumber: number | null;
}

/** 查询侧的结构化信号，由 EmbeddingRetriever.parseQuery 产出 */
export interface QuerySignals {
    /** 查询点名的条号数值，如 [577] */
    articleNos: number[];
    /** 查询点名的法律名，如 ['民法典'] */
    lawNames: string[];
}

/**
 * 法律名命中的加权：**加在余弦相似度上的有界增量**。
 *
 * 原来是硬分层（`containsKeywords ? -1 : 1`），实测后果是一次排序事故：
 * 查询「民法典第577条规定了什么」，余弦只有 0.5630 的**文件标题 chunk**
 * （正文就是"中华人民共和国民法典"六个字）靠命中"民法典"三个字排到了第 1，
 * 把 0.7613 的真条文挤下去。0.2 的语义差距不该被一次字符串命中抹平。
 *
 * 分层排序还有个隐性代价：它让"命中与否"这个二值信号完全压过相似度，
 * 于是任何一条误命中（比如标题里的法律名）都会稳定污染 top-k，
 * 而且因为它是**布尔**的，调参时没有任何中间档位可用。
 */
const BOOST_SOURCE = 0.03;

export interface RetrievedChunk {
    document: string;
    source: string;
    articleNo: string | null;
    chapter: string | null;
    hash: string;
    /**
     * 喂给模型的那份文本（含编/章/节路径），与 `document` 分开——理由见 `ParsedChunk.embedText`。
     *
     * 为什么检索结果里也要带着它：开了精排之后，cross-encoder 读的就是这个字段。
     * **不再另起一个名字**（比如 rerankText）：同一个字符串起两个名字迟早会漂移，
     * 有人改了一个忘了另一个，而症状是"重排结果说不上哪里怪"。
     */
    embedText: string;
    /** 余弦相似度 */
    score: number;
    /**
     * 是否**精确命中**查询点名的条号（问的就是这一条，且库里真有）。
     *
     * 只认条号精确命中，不认法律名——这是给拒答逻辑用的"强信号"。
     * 法律名不够格：用户问"公司法对火星殖民怎么规定"同样会命中"公司法"，
     * 但那恰恰是该拒答的问题。
     */
    exactArticleHit: boolean;
}

export default class VectorStore {
    private vectorStore: VectorStoreItem[] = [];
    private readonly dimension: number;
    private seenHashes: Set<string> = new Set();
    /** 条号数值 → 条目。惰性构建、写入时失效。 */
    private articleIndex: Map<number, VectorStoreItem[]> | null = null;

    constructor(dimension: number) {
        this.dimension = dimension;
    }

    size(): number {
        return this.vectorStore.length;
    }

    /** 已入库的内容 hash，供上传去重 / 增量索引使用 */
    hasHash(hash: string): boolean {
        return this.seenHashes.has(hash);
    }

    async addEmbedding(embedding: number[], chunk: ParsedChunk) {
        // 维度断言：维度不一致的向量进了库，cosineSimilarity 会静默错序甚至算出 NaN，
        // 而且很难排查（表现为"检索结果看着不相关"而不是报错）。宁可在入库这一刻就拒绝。
        if (embedding.length !== this.dimension) {
            throw new Error(
                `[VectorStore] 向量维度不符：期望 ${this.dimension}，实际 ${embedding.length}（来源 ${chunk.source}${chunk.articleNo ? ' ' + chunk.articleNo : ''}）`
            );
        }
        this.vectorStore.push({ ...chunk, embedding, articleNumber: parseArticleNo(chunk.articleNo) });
        this.seenHashes.add(chunk.hash);
        this.articleIndex = null;
    }

    /** 按来源移除（删除某文档 / 重建索引时用） */
    removeBySource(source: string): number {
        const before = this.vectorStore.length;
        this.vectorStore = this.vectorStore.filter(item => item.source !== source);
        this.seenHashes = new Set(this.vectorStore.map(i => i.hash));
        this.articleIndex = null;
        return before - this.vectorStore.length;
    }

    async search(
        queryEmbedding: number[],
        topK: number = 5,
        signals: QuerySignals = { articleNos: [], lawNames: [] },
    ): Promise<RetrievedChunk[]> {
        if (queryEmbedding.length !== this.dimension) {
            throw new Error(
                `[VectorStore] 查询向量维度不符：期望 ${this.dimension}，实际 ${queryEmbedding.length}`
            );
        }

        const wanted = new Set(signals.articleNos);
        const lawNamed = signals.lawNames.length > 0;
        const scored: RetrievedChunk[] = [];
        /** 条号命中的条目，附带"是否来自查询点名的那部法律" */
        const numberHits: { chunk: RetrievedChunk; sourceHit: boolean }[] = [];

        for (const item of this.vectorStore) {
            // 标题不进检索池：它是结构信息，不是答案。理由见 splitIntoChunks 里 heading 的注释。
            if (item.kind === 'heading') continue;

            const sourceHit = lawNamed && signals.lawNames.some(name => item.source.includes(name));
            const numberHit = item.articleNumber !== null && wanted.has(item.articleNumber);

            const hit: RetrievedChunk = {
                document: item.document,
                source: item.source,
                articleNo: item.articleNo,
                chapter: item.chapter,
                hash: item.hash,
                embedText: item.embedText,
                score: this.cosineSimilarity(queryEmbedding, item.embedding)
                    + (sourceHit ? BOOST_SOURCE : 0),
                exactArticleHit: false,
            };
            if (numberHit) numberHits.push({ chunk: hit, sourceHit });
            // 条号命中的条目**同时**留在 scored 里参与常规排序：置顶是"加一个更好的位置"，
            // 不是"把别的候选踢出去"。曾经写成 else 分支，后果是静默丢弃——
            // 「合同法」在 KNOWN_LAW_NAMES 里，但语料里没有 合同法.md（合同法即民法典第三编），
            // 于是查询「合同法第107条」的 promoted 为空，而民法典#107 这个**正确答案**
            // 会被整个扔掉。置顶与常规排序用同一批对象引用，最后按引用去重
            // （不能用 hash 去重：语料里「第一章 一般规定」这类章标题在 6 个不同编下
            //   重复出现，hash 完全相同——实测 1681 个 chunk 只有 1670 个唯一 hash，
            //   3 组撞车全部是这类 heading。它们不进检索池，但按 hash 去重的写法会误伤）。
            scored.push(hit);
        }

        // 排序键是 score 本身，没有布尔分层（那是下面 exact 分层的反面，别混淆）。
        // 这是个**粗糙**的稀疏+稠密融合；P1 会换成 BM25 + RRF 的正规融合。
        scored.sort((a, b) => b.score - a.score);

        // 查询点名了条号、且语料里真有 → 这几条**直接置顶**，不参与相似度竞争。
        //
        // 这里的分层和上面拿掉的硬分层不是一回事，区别在于信号的确定性：
        //   ✗ 「命中任意关键词」——字符串命中，噪声大（法律名会命中标题，裸数字会撞正文）
        //   ✓ 「查询点名了第577条，且库里就是第577条」——结构化标识符的精确匹配，近乎确定
        //
        // 为什么必须置顶而不能只给个加分：实测「民法典第577条规定了什么」，
        // 第577条正文与查询的余弦只有 0.584，**低于**无关的第一百七十六条(0.742)。
        // 原因是查询里的条号对 embedding 模型是纯噪声，反而稀释了语义信号——
        // 相似度排序在这个场景里从原理上就问错了问题。用户点名了条号，那不是"检索"，
        // 是"查找"，答案唯一且已知。
        //
        // 条号必须**连同法律名**一起匹配：条号在法律之间并不唯一——
        // 民法典和公司法都有「第二十三条」。实测「公司法第二十三条的内容是什么」，
        // 民法典#23 排在公司法#23 前面（名次=2），因为查询里的"公司法"只值 0.03 的加分，
        // 压不过余弦差。用户点名了哪部法，就只在那部法里认这个条号。
        //
        // 查询点名的法律在语料里不存在时（如"刑法第232条"），promoted 会是空的，
        // 于是不置顶、退回语义检索——我们要的不是"换个法硬凑一条"，而是答不上来。
        const promoted = lawNamed ? numberHits.filter(h => h.sourceHit) : numberHits;
        if (promoted.length > 0) {
            const tier = promoted.map(h => ({ ...h.chunk, exactArticleHit: true }));
            tier.sort((a, b) => b.score - a.score);
            const inTier = new Set(promoted.map(h => h.chunk));
            return [...tier, ...scored.filter(c => !inTier.has(c))].slice(0, topK);
        }
        return scored.slice(0, topK);
    }


    /**
     * 按条号精确查回原文（字符串比对）。
     * 注意：语料里存的是「第五百七十七条」这种中文数字写法，而模型答案里
     * 阿拉伯数字和中文数字都可能出现，所以**引用校验不要用这个方法**，
     * 用下面的 findByArticleNumber()。
     */
    findByArticleNo(articleNo: string, source?: string): VectorStoreItem[] {
        return this.vectorStore.filter(item =>
            item.articleNo === articleNo && (source ? item.source === source : true)
        );
    }

    /**
     * 按条号**数值**查回原文，供引用校验器回查模型答案里的「第X条」。
     *
     * 数值比较是必须的：模型写「第110条」，语料存「第一百一十条」，
     * 字符串比对会得出"这条不存在"——一个把真条号报成编造的校验器，
     * 比没有校验器更糟，因为它会训练用户忽略告警。
     */
    findByArticleNumber(articleNo: number, source?: string): VectorStoreItem[] {
        if (!this.articleIndex) {
            this.articleIndex = new Map();
            for (const item of this.vectorStore) {
                const n = item.articleNumber;
                if (n === null) continue;
                const bucket = this.articleIndex.get(n);
                if (bucket) bucket.push(item);
                else this.articleIndex.set(n, [item]);
            }
        }
        const hits = this.articleIndex.get(articleNo) ?? [];
        return source ? hits.filter(h => h.source === source) : hits;
    }

    private cosineSimilarity(vecA: number[], vecB: number[]): number {
        // 到这里两边维度已由入库/查询断言保证一致
        let dot = 0, normA = 0, normB = 0;
        for (let i = 0; i < vecA.length; i++) {
            dot += vecA[i] * vecB[i];
            normA += vecA[i] * vecA[i];
            normB += vecB[i] * vecB[i];
        }
        if (normA === 0 || normB === 0) return 0;
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }
}
