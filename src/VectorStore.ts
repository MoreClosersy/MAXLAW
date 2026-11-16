import { ParsedChunk } from "./EmbeddingRetriever.js";
import { parseArticleNo } from "./articleNo.js";

export interface VectorStoreItem extends ParsedChunk {
    embedding: number[];
    /** articleNo 的**数值**形式，入库时算一次。条号比较一律走数值——理由见 articleNo.ts */
    articleNumber: number | null;
}

/** 查询侧的结构化信号，由 EmbeddingRetriever.parseQuery 产出 */
export interface QuerySignals {
    /** 没有绑定法律的条号数值，如 [577]（用户只说「第577条」时） */
    articleNos: number[];
    /** 查询点名的法律名，如 ['民法典'] */
    lawNames: string[];
    /**
     * 条号 → 它所属的法律（由查询的句法位置决定），如 [{law:'公司法', articleNo:23}]。
     *
     * 有它时置顶只认配上的那一条；缺省时退回旧语义（在**任一**被点名法律里认这个条号），
     * 这样直接调用 search 的老测试/老调用方行为不变。
     */
    articlePairs?: { law: string; articleNo: number }[];
    /**
     * 只在指定法律的文件里检索（二次检索用）。
     * 与 `lawNames` 不同：`lawNames` 只是加分，**不排除**别的法；这里是硬过滤。
     */
    lawFilter?: string;
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

    /**
     * 语料里是否有文件名包含该法律名的来源（如「民法典」→ 民法典.md）。
     * 引用校验器用它区分"库外法律"与"这部法在库里、只是没有这一条"——
     * 前者维持旧语义，后者要判 fabricated。
     */
    hasSource(name: string): boolean {
        return this.vectorStore.some(item => item.source.includes(name));
    }

    /** 语料里出现过的来源文件（去重）。调用方据此得到"库里有哪些法律"。 */
    sources(): string[] {
        return [...new Set(this.vectorStore.map(item => item.source))];
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

        const pairs = signals.articlePairs ?? [];
        /** 有配对信息时，置顶只认配上的那一条；`articleNos` 里只剩没绑上法律的条号。 */
        const paired = pairs.length > 0;
        const wanted = new Set(signals.articleNos);
        const lawNamed = signals.lawNames.length > 0;
        const scored: RetrievedChunk[] = [];
        /** 条号命中、应当置顶的条目 */
        const pinned: RetrievedChunk[] = [];

        for (const item of this.vectorStore) {
            // 标题不进检索池：它是结构信息，不是答案。理由见 splitIntoChunks 里 heading 的注释。
            if (item.kind === 'heading') continue;
            // 二次检索的硬过滤：只看指定的那部法。`lawNames` 做不到这件事（它只是加分）。
            if (signals.lawFilter && !item.source.includes(signals.lawFilter)) continue;

            const sourceHit = lawNamed && signals.lawNames.some(name => item.source.includes(name));
            // 条号必须**连同它所属的法律**一起匹配。条号跨法不唯一（民法典/公司法都有第二十三条），
            // 没配上的同名条号不是用户问的那一条——旧实现用两个独立集合做笛卡尔积，
            // 多法查询会把没被问到的同名条号一并置顶，并顺带跳过可答性判官。
            const pairHit = paired
                && pairs.some(p => p.articleNo === item.articleNumber && item.source.includes(p.law));
            const numberHit = item.articleNumber !== null && wanted.has(item.articleNumber);
            // 有配对时，`articleNos` 里只剩"用户没点名法律"的条号，它们在所有法律里置顶；
            // 没有配对时退回旧语义：点名了法律就只在那部法里认这个条号。
            const pinAsNumber = paired ? numberHit : numberHit && (!lawNamed || sourceHit);

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
            if (pairHit || pinAsNumber) pinned.push(hit);
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
        const tier = pinned.map(c => ({ ...c, exactArticleHit: true }));
        tier.sort((a, b) => b.score - a.score);
        const inTier = new Set(pinned);
        const ordered = [...tier, ...scored.filter(c => !inTier.has(c))];
        // 多法查询的"保底"：全局排序会让语义更强的那部法把另一部挤出 topK，
        // 那不是"没检索到"，是"被挤掉了"。见 ensureLawCoverage。
        return ensureLawCoverage(ordered.slice(0, topK), scored, signals.lawNames);
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

/**
 * 多法查询的**保底覆盖**：查询点名了两部以上的法律、而其中一部在结果里一条都没有时，
 * 把它的最佳候补换进尾部——**但只在那条候补本来就差一个位置时**。
 *
 * 为什么需要它：topK 是**全局**相似度排序，五个位置里没有"每部法至少一条"这回事。
 * 语义更强的那部法完全可以把另一部挤出前 5——而系统看起来一切正常，因为它确实
 * 检索到了"高分条文"。这不是召回失败，是排序把另一半挤掉了，指标（`hit` 只要
 * 命中任意一条即算成功）看不见它。
 *
 * 为什么必须加"名次差一个位置"这道闸（实测教训，不是防御性编程）：
 * 用 sem-09 的真实查询向量、把问题改成点名两部法去跑真实语料，民法典侧的最佳候选
 * 全局排第 **10**（第五百七十四条·提存，与问题无关），而第 5 名是公司法第五十三条。
 * 不设闸就会把这条无关条文塞进上下文——"保底"变成了"污染"。
 * 一部法如果连候补第一位都排不进，说明它本来就没有相关条文；正确做法是让判官/答案
 * 说"该法没有对应规定"，而不是硬凑一条。真正要召回它对侧的条文，需要按法改写查询
 * （查询分解），那是另一件事。
 *
 * 三条自我约束：
 *   1. 只换**尾部**、且只换非置顶项——用户点名第577条时，不能为了凑法律把 577 换出去；
 *   2. 不换另一部被点名法律的**唯一**代表——否则只是把缺口搬了个家；
 *   3. `lawNames` 少于两个、或那部法在库里根本没有 chunk（点名了合同法/刑法）时不动——
 *      "库里没有"是判官该拒答的情形，不是检索该硬凑的。
 *
 * 返回新数组，长度不变（除非无位置可换）。
 */
export function ensureLawCoverage(
    results: RetrievedChunk[],
    pool: RetrievedChunk[],
    lawNames: string[],
): RetrievedChunk[] {
    const distinct = [...new Set(lawNames)];
    if (results.length < 2 || distinct.length < 2) return results;

    const present = distinct.filter(law => pool.some(c => c.source.includes(law)));
    if (present.length < 2) return results;

    const key = (c: RetrievedChunk) => `${c.source}#${c.hash}`;
    const out = [...results];

    for (const law of present) {
        if (out.some(c => c.source.includes(law))) continue;
        const have = new Set(out.map(key));
        const candidate = pool.find(c => c.source.includes(law) && !have.has(key(c)));
        if (!candidate) continue;

        // 候补名次闸：只有"下一个就该轮到它"才算被挤掉（0 基下标 ≤ topK）。
        // 落后更多说明它本来就不相关，补进来是噪声——理由见上面的实测。
        const candidateRank = pool.indexOf(candidate);
        if (candidateRank < 0 || candidateRank > results.length) continue;

        let swapAt = -1;
        for (let i = out.length - 1; i >= 0; i--) {
            if (out[i].exactArticleHit) continue;
            // 另一部被点名法律的唯一代表不能换：换了只是把缺口搬了个家。
            if (out.filter(c => c.source === out[i].source).length <= 1) continue;
            swapAt = i;
            break;
        }
        if (swapAt === -1) continue;
        out[swapAt] = candidate;
    }
    return out;
}
