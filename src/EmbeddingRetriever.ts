import { logTitle } from "./utils.js";
import VectorStore, { RetrievedChunk, QuerySignals } from "./VectorStore.js";
import { RERANK_CANDIDATES, partitionPinned, RerankerLike } from "./Reranker.js";
import { extractArticleNumbers } from "./articleNo.js";
import { CACHE_DIR } from "./paths.js";
import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

/** 已知 embedding 模型的向量维度。用于断言真实响应、并让 fallback 维度与模型对齐。 */
const MODEL_DIMENSIONS: Record<string, number> = {
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2': 384,
    'BAAI/bge-small-zh-v1.5': 512,
    'BAAI/bge-base-zh-v1.5': 768,
    'BAAI/bge-large-zh-v1.5': 1024,
    'text-embedding-3-small': 1536,
    'text-embedding-3-large': 3072,
    'text-embedding-ada-002': 1536,
};

const DEFAULT_DIMENSION = 384;

/** 长索引任务每隔多少条落一次盘。见 embedDocument() 里的说明。 */
const FLUSH_EVERY = 200;

/**
 * 每次 HTTP 请求塞多少条文本。设为 1 即退回逐条请求。
 *
 * 逐条请求是索引构建的真实瓶颈：1681 个 chunk = 1681 次串行往返。实测（2026-09-18）
 * 走代理建 bge-base 索引时，进程 83 分钟只消耗 8.4 秒 CPU —— 时间全花在等网络上，
 * 而持续请求流会被代理限流（健康时 1.2–1.7s/条，被限流时 8.2s/条）。
 *
 * 批量化**不改变向量值**：已实测单条与批量返回逐元素最大差 1.2e-7、余弦精确为 1.0，
 * 即 float32 精度内的同一个向量。所以这不是"近似优化"，索引结果与逐条构建一致。
 */
const EMBED_BATCH_SIZE = Math.max(1, Math.floor(Number(process.env.EMBED_BATCH_SIZE) || 16));

export interface RetrievalStats {
    /** 实际去请求/命中缓存/走兜底的**文本条数**（批量化后与 HTTP 次数解耦） */
    embedCalls: number;
    /** 实际发出的 HTTP 请求次数。embedCalls / embedRequests ≈ 平均批量大小 */
    embedRequests: number;
    cacheHits: number;
    fallbackHits: number;
    dimensionMismatches: number;
}

export default class EmbeddingRetriever {
    private embeddingModel: string;
    private vectorStore: VectorStore;
    /** 仅存放**真实** embedding，会落盘 */
    private embeddingCache: Map<string, number[]> = new Map();
    /** 存放确定性 fallback 向量，**只在进程内有效、永不落盘**（见下方 embed() 注释） */
    private fallbackCache: Map<string, number[]> = new Map();
    /**
     * 本轮索引实际用到的缓存键。
     *
     * 只增不减地留着旧缓存会随每次改语料单调膨胀：删掉的文件、改过的条文，
     * 它们的向量永远不会再被查到（VectorStore 每次启动都从当前 knowledge/ 重建），
     * 却会一直躺在缓存文件里。一次语料替换实测留下 355 条死数据，约占 17%。
     */
    private activeKeys: Set<string> = new Set();
    /** 见 setStrict() */
    private strict = false;
    private cacheFile: string;
    private dimension: number;
    /** 精排器。null = 关闭，检索结果就是 bi-encoder 的余弦排序。 */
    private reranker: RerankerLike | null;
    private stats: RetrievalStats = { embedCalls: 0, embedRequests: 0, cacheHits: 0, fallbackHits: 0, dimensionMismatches: 0 };

    /**
     * @param reranker 精排器。**从构造参数进来，而不是事后 set**——
     *   配置里写着"开精排"而代码里忘了接，是最难发现的一类不一致：
     *   评测跑出来是个正常的分数，只是那个分数压根没走精排。
     *   构造参数让"开着精排却没有精排器"在类型上就写不出来。
     */
    constructor(embeddingModel: string, reranker: RerankerLike | null = null) {
        this.embeddingModel = embeddingModel;
        this.reranker = reranker;
        this.dimension = MODEL_DIMENSIONS[embeddingModel] ?? DEFAULT_DIMENSION;
        if (!MODEL_DIMENSIONS[embeddingModel]) {
            // 说清楚该怎么办。否则换模型的人会拿到 addEmbedding 抛的
            // 「向量维度不符：期望 384，实际 768」——那不是维度不符，是模型没注册，
            // 而这条报错会把人往"模型坏了"的方向带。
            console.warn(
                `[embedding] 未登记的模型 ${embeddingModel}，暂按 ${DEFAULT_DIMENSION} 维处理。` +
                `若它的真实维度不是 ${DEFAULT_DIMENSION}，入库时会抛维度断言错误——` +
                `把该模型的维度加进 EmbeddingRetriever.ts 的 MODEL_DIMENSIONS 表即可。`
            );
        }
        this.vectorStore = new VectorStore(this.dimension);
        this.cacheFile = path.join(CACHE_DIR, `${embeddingModel.replace(/\//g, '_')}_cache.json`);
        this.initializeCache();
        // 注意：这里**不注册** SIGINT/SIGTERM/exit 处理器。
        // 库不应该决定进程怎么退出（原实现 SIGINT -> process.exit(0) 会劫持宿主进程的关闭流程，
        // 且每实例化一次就泄漏一组 listener）。落盘改由调用方显式 flushCache()。
    }

    getStats(): RetrievalStats {
        return { ...this.stats };
    }

    /**
     * 严格模式：任何一次兜底都直接抛错。
     *
     * 服务器运行时**不能**开——一次网络抖动不该让服务起不来，那里要的是
     * "降级但可用 + 日志告警"。而 `pnpm embed` 恰恰相反：它的唯一产物就是索引，
     * 索引残缺等于这次运行毫无价值。
     */
    setStrict(strict: boolean): void {
        this.strict = strict;
    }

    /** 供引用校验器回查条号；检索本身不需要外部访问 store */
    getStore(): VectorStore {
        return this.vectorStore;
    }

    getDimension(): number {
        return this.dimension;
    }

    private initializeCache() {
        if (!fs.existsSync(CACHE_DIR)) {
            fs.mkdirSync(CACHE_DIR, { recursive: true });
        }
        if (!fs.existsSync(this.cacheFile)) return;
        try {
            const cacheData = JSON.parse(fs.readFileSync(this.cacheFile, 'utf-8'));
            let dropped = 0;
            for (const [key, vec] of Object.entries(cacheData as Record<string, number[]>)) {
                // 维度守门：历史缓存里可能混入过维度不符的向量（例如 1536 维的 fallback 落进
                // 384 维模型的缓存文件）。这类向量会让 cosineSimilarity 静默错序，
                // 且"重启不恢复、换成真 key 也不恢复"。加载期直接丢弃，不让它污染本次进程。
                if (Array.isArray(vec) && vec.length === this.dimension) {
                    this.embeddingCache.set(key, vec);
                } else {
                    dropped++;
                }
            }
            console.log(`[embedding] 载入缓存 ${this.embeddingCache.size} 条${dropped ? `，丢弃维度不符 ${dropped} 条` : ''}`);
            if (dropped > 0) {
                this.stats.dimensionMismatches += dropped;
                // 立即重写，把脏数据从磁盘上也清掉
                this.flushCache();
            }
        } catch (error) {
            console.error('[embedding] 缓存文件损坏，忽略:', error);
            this.embeddingCache = new Map();
        }
    }

    /**
     * 丢掉缓存里已不对应任何当前语料的条目，返回丢弃条数。
     * 没有跑过 embedDocument 时（activeKeys 为空）不动手——否则一次误调用会把整个缓存清空。
     */
    pruneCache(): number {
        if (this.activeKeys.size === 0) return 0;
        const before = this.embeddingCache.size;
        for (const key of [...this.embeddingCache.keys()]) {
            if (!this.activeKeys.has(key)) this.embeddingCache.delete(key);
        }
        return before - this.embeddingCache.size;
    }

    flushCache(quiet = false) {
        try {
            const cacheObj = Object.fromEntries(this.embeddingCache);
            fs.writeFileSync(this.cacheFile, JSON.stringify(cacheObj), 'utf-8');
            if (!quiet) console.log(`[embedding] 已保存缓存 ${this.embeddingCache.size} 条（fallback 向量不落盘）`);
        } catch (error) {
            console.error('[embedding] 保存缓存失败:', error);
        }
    }

    private getCacheKey(text: string): string {
        return crypto.createHash('md5').update(text).digest('hex');
    }

    /** 无 key / API 失败时的确定性兜底向量。维度跟随当前模型，不再硬编码 1536。 */
    private generateDeterministicEmbedding(text: string, dimension: number = this.dimension): number[] {
        let hash = 0;
        for (let i = 0; i < text.length; i++) {
            const char = text.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        const rng = (n: number) => {
            const x = Math.sin(n + hash) * 10000;
            return x - Math.floor(x);
        };
        const embedding = Array(dimension).fill(0).map((_, i) => rng(i));
        const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
        return embedding.map(val => val / magnitude);
    }

    /**
     * 索引一份文档。
     *
     * `onProgress` 不是装饰：全量民法典 + 公司法有 1600+ chunk，逐条打 embedding 接口
     * 是分钟级的活，没有进度输出就只能干等，也无法判断是否卡住。
     */
    async embedDocument(
        document: string,
        source: string = "unknown",
        onProgress?: (done: number, total: number) => void,
    ) {
        const chunks = splitIntoChunks(document, source);
        const total = chunks.length;
        const step = Math.max(1, Math.floor(total / 10));
        let lastFlush = 0;
        let lastReport = 0;

        // 按 EMBED_BATCH_SIZE 分组请求。embedBatch() 保证返回顺序与入参严格对应，
        // 所以下面按下标取用的写法是安全的（错位会是静默的，见 requestBatch 的校验）。
        for (let start = 0; start < total; start += EMBED_BATCH_SIZE) {
            const slice = chunks.slice(start, start + EMBED_BATCH_SIZE);
            // 嵌入 embedText 而不是 document：前者带章节路径，见 ParsedChunk.embedText
            const embeddings = await this.embedBatch(slice.map(c => c.embedText));
            for (let j = 0; j < slice.length; j++) {
                this.activeKeys.add(this.getCacheKey(slice[j].embedText));
                await this.vectorStore.addEmbedding(embeddings[j], slice[j]);
            }
            const done = start + slice.length;
            // 定期落盘，让长索引任务能断点续跑。
            //
            // 全量索引在网络抖动时中途失败是常态，而失败时内存里的向量从未落盘——
            // 重跑要从零开始，几十分钟的请求全部作废。embed.ts 的错误提示一直写着
            // "已成功嵌入的部分不会浪费，重跑会命中缓存"，在加上这行之前那句话是不成立的。
            // （批量化把请求数从 1681 降到约 105，但"中途失败"这件事不因批量而消失。）
            if (done - lastFlush >= FLUSH_EVERY) { this.flushCache(true); lastFlush = done; }
            if (onProgress && done - lastReport >= step) { onProgress(done, total); lastReport = done; }
        }

        // 文档末尾落盘。原来只在 FLUSH_EVERY 的整数倍上落，
        // 于是每份文档**最后不足 200 条**的那一段永远留在内存里——
        // 一次崩溃丢掉的是这部分，而 embed.ts 的提示语承诺的是"已成功嵌入的部分不会浪费"。
        // 批量化后 done 不再逐条递增，靠 % 判断会更不可靠，所以改成显式的"结束时落一次"。
        this.flushCache(true);
        onProgress?.(total, total);
        return total;
    }

    async embedQuery(query: string) {
        return this.embed(query);
    }

    private async embed(document: string): Promise<number[]> {
        this.stats.embedCalls++;
        const cacheKey = this.getCacheKey(document);

        const cached = this.embeddingCache.get(cacheKey);
        if (cached) {
            this.stats.cacheHits++;
            return cached;
        }
        const cachedFallback = this.fallbackCache.get(cacheKey);
        if (cachedFallback) {
            this.stats.cacheHits++;
            this.stats.fallbackHits++;
            return cachedFallback;
        }

        if (!process.env.EMBEDDING_API_KEY || process.env.EMBEDDING_API_KEY === 'none') {
            return this.useFallback(cacheKey, document, 'EMBEDDING_API_KEY 未配置或为 none');
        }

        for (let attempt = 1; ; attempt++) {
            try {
                const embedding = process.env.EMBEDDING_API_KEY.startsWith('hf_')
                    ? await this.embedWithHuggingFace(document)
                    : await this.embedWithOpenAI(document);

                if (embedding === null) {
                    // 接口有响应但内容不可用：重试不会变好，直接降级
                    return this.useFallback(cacheKey, document, 'embedding 接口返回不可用');
                }
                if (embedding.length !== this.dimension) {
                    // 真实响应维度和声明不一致：不写缓存，避免污染
                    this.stats.dimensionMismatches++;
                    console.error(`[embedding] 维度不符：期望 ${this.dimension}，实际 ${embedding.length}（模型 ${this.embeddingModel}）`);
                    return this.useFallback(cacheKey, document, '接口返回维度与模型不符');
                }
                this.embeddingCache.set(cacheKey, embedding);
                return embedding;
            } catch (error) {
                const transient = isTransientNetworkError(error);
                if (transient && attempt < EMBEDDING_MAX_ATTEMPTS) {
                    const delay = 500 * 4 ** (attempt - 1);   // 500ms → 2s
                    console.warn(`[embedding] 第 ${attempt} 次请求失败（${describeNetworkError(error)}），${delay}ms 后重试`);
                    await sleep(delay);
                    continue;
                }
                const timedOut = error instanceof Error && error.name === 'TimeoutError';
                console.error(timedOut
                    ? `[embedding] 请求超时（>${EMBEDDING_TIMEOUT_MS}ms），已重试 ${attempt - 1} 次，放弃本条`
                    : `[embedding] 请求异常（已重试 ${attempt - 1} 次）:`, timedOut ? '' : error);
                return this.useFallback(
                    cacheKey, document,
                    timedOut ? '请求超时' : attempt > 1 ? `请求失败（重试 ${attempt - 1} 次后）` : '请求抛出异常',
                );
            }
        }
    }

    /**
     * 走兜底向量。
     *
     * 关键点：fallback 向量**只进内存、不进持久缓存**。
     * 原实现把它 set 进同一个缓存 Map 再 flush 落盘，导致一次网络抖动就在磁盘上留下垃圾，
     * 重启不恢复、后来配上真 key 也不恢复——因为缓存命中优先于重新请求。
     */
    private useFallback(cacheKey: string, document: string, reason: string): number[] {
        this.stats.fallbackHits++;
        if (this.strict) throw new EmbeddingUnavailableError(reason);
        console.warn(`[embedding] 使用确定性兜底向量（${reason}）——检索结果将不可信`);
        const embedding = this.generateDeterministicEmbedding(document);
        this.fallbackCache.set(cacheKey, embedding);
        return embedding;
    }

    /**
     * 批量请求 + 解析，返回与 `documents` **一一对应**的向量数组。
     *
     * 失败返回 null（由调用方决定降级策略），**不**在这里做重试——重试语义与 embed() 一致，
     * 放在 embedBatch() 里统一处理，避免两处各写一遍超时/瞬时错误的判断。
     *
     * 形状校验是这里的重点：返回值条数必须**恰好等于**输入条数、且每条维度正确。
     * 少一条就会让后续向量与 chunk 整体错位——而错位的表现是"检索结果看着不相关"，
     * 不是报错。宁可判定整批不可用、退回逐条，也不要冒静默错位的风险。
     */
    private async requestBatch(documents: string[]): Promise<number[][] | null> {
        this.stats.embedRequests++;
        try {
            const vectors = process.env.EMBEDDING_API_KEY!.startsWith('hf_')
                ? await this.embedManyWithHuggingFace(documents)
                : await this.embedManyWithOpenAI(documents);
            if (vectors === null) return null;
            if (vectors.length !== documents.length) {
                console.error(`[embedding] 批量返回条数不符：请求 ${documents.length} 条，返回 ${vectors.length} 条，整批作废`);
                return null;
            }
            for (const v of vectors) {
                if (!Array.isArray(v) || v.length !== this.dimension) {
                    this.stats.dimensionMismatches++;
                    console.error(`[embedding] 批量中某条维度不符：期望 ${this.dimension}，实际 ${Array.isArray(v) ? v.length : typeof v}`);
                    return null;
                }
            }
            return vectors;
        } catch (error) {
            // 抛给 embedBatch 的重试循环处理
            throw error;
        }
    }

    /** 批量走 HuggingFace feature-extraction。inputs 传数组 → 返回 N×dim。 */
    private async embedManyWithHuggingFace(documents: string[]): Promise<number[][] | null> {
        const url = `${process.env.EMBEDDING_BASE_URL}/models/${this.embeddingModel}/pipeline/feature-extraction`;
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${process.env.EMBEDDING_API_KEY}`,
            },
            body: JSON.stringify({ inputs: documents }),
            signal: AbortSignal.timeout(EMBEDDING_TIMEOUT_MS),
        });

        if (!response.ok) {
            console.error(`[embedding] HuggingFace API error: ${response.status} ${response.statusText}`);
            return null;
        }
        let data: any;
        try {
            data = JSON.parse(await response.text());
        } catch {
            console.error('[embedding] HuggingFace JSON parse error');
            return null;
        }
        // 批量时返回 [[...], [...]]；若服务端把单元素数组压平成一维，这里也能认出来
        if (Array.isArray(data) && Array.isArray(data[0])) return data as number[][];
        if (documents.length === 1 && Array.isArray(data) && typeof data[0] === 'number') return [data];
        console.error('[embedding] HuggingFace 批量响应格式无法识别');
        return null;
    }

    /** 批量走 OpenAI 兼容 /embeddings。input 传数组 → data 带 index，必须按 index 还原顺序。 */
    private async embedManyWithOpenAI(documents: string[]): Promise<number[][] | null> {
        const response = await fetch(`${process.env.EMBEDDING_BASE_URL}/embeddings`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${process.env.EMBEDDING_API_KEY}`,
            },
            body: JSON.stringify({ model: this.embeddingModel, input: documents, encoding_format: 'float' }),
            signal: AbortSignal.timeout(EMBEDDING_TIMEOUT_MS),
        });

        if (!response.ok) {
            console.error(`[embedding] OpenAI API error: ${response.status} ${response.statusText}`);
            return null;
        }
        let data: any;
        try {
            data = JSON.parse(await response.text());
        } catch {
            console.error('[embedding] OpenAI JSON parse error');
            return null;
        }
        const items = data?.data;
        if (!Array.isArray(items)) {
            console.error('[embedding] OpenAI 批量响应格式无法识别');
            return null;
        }
        // 响应顺序**没有**契约保证（OpenAI 文档说明 data 可能乱序），按 index 显式还原。
        // 直接按数组下标取，就是上面说的那种静默错位。
        const out: number[][] = new Array(documents.length);
        for (let k = 0; k < items.length; k++) {
            const idx = typeof items[k]?.index === 'number' ? items[k].index : k;
            if (idx < 0 || idx >= documents.length || out[idx] !== undefined) {
                console.error(`[embedding] OpenAI 批量响应 index 异常（${idx}），整批作废`);
                return null;
            }
            out[idx] = items[k]?.embedding;
        }
        return out.every(v => Array.isArray(v)) ? out : null;
    }

    /**
     * 批量嵌入：缓存优先，未命中的按 EMBED_BATCH_SIZE 分组请求。
     *
     * 返回数组与 `documents` **顺序严格对应**——调用方按下标取用是安全的。
     *
     * 降级策略：某一批失败（网络/格式/维度）时，**退回逐条 `embed()`**，
     * 于是重试、strict 模式抛错、兜底向量这些既有语义原样保留，
     * 批量化对失败路径是纯增量的。批量只优化顺利路径。
     */
    async embedBatch(documents: string[]): Promise<number[][]> {
        const out: (number[] | null)[] = new Array(documents.length).fill(null);
        const misses: number[] = [];

        for (let i = 0; i < documents.length; i++) {
            const key = this.getCacheKey(documents[i]);
            const real = this.embeddingCache.get(key);
            if (real) { this.stats.cacheHits++; out[i] = real; continue; }
            const fb = this.fallbackCache.get(key);
            if (fb) { this.stats.cacheHits++; this.stats.fallbackHits++; out[i] = fb; continue; }
            misses.push(i);
        }
        if (misses.length === 0) return out as number[][];

        if (!process.env.EMBEDDING_API_KEY || process.env.EMBEDDING_API_KEY === 'none') {
            for (const i of misses) {
                out[i] = this.useFallback(this.getCacheKey(documents[i]), documents[i], 'EMBEDDING_API_KEY 未配置或为 none');
            }
            return out as number[][];
        }

        // 按批分组。最后一批可能不满。
        const groups: number[][] = [];
        for (let start = 0; start < misses.length; start += EMBED_BATCH_SIZE) {
            groups.push(misses.slice(start, start + EMBED_BATCH_SIZE));
        }

        for (const group of groups) {
            const texts = group.map(i => documents[i]);
            let vectors: number[][] | null = null;

            for (let attempt = 1; ; attempt++) {
                try {
                    vectors = await this.requestBatch(texts);
                    break;
                } catch (error) {
                    const transient = isTransientNetworkError(error);
                    if (transient && attempt < EMBEDDING_MAX_ATTEMPTS) {
                        const delay = 500 * 4 ** (attempt - 1);
                        console.warn(`[embedding] 批量请求（${texts.length} 条）第 ${attempt} 次失败（${describeNetworkError(error)}），${delay}ms 后重试`);
                        await sleep(delay);
                        continue;
                    }
                    console.warn(`[embedding] 批量请求失败（${texts.length} 条），退回逐条：${describeNetworkError(error)}`);
                    vectors = null;
                    break;
                }
            }

            if (vectors) {
                this.stats.embedCalls += group.length;
                for (let j = 0; j < group.length; j++) {
                    const vec = vectors[j];
                    this.embeddingCache.set(this.getCacheKey(documents[group[j]]), vec);
                    out[group[j]] = vec;
                }
            } else {
                // 退回逐条：保留重试/strict/兜底的全部既有语义
                for (const i of group) out[i] = await this.embed(documents[i]);
            }
        }
        return out as number[][];
    }

    private async embedWithHuggingFace(document: string): Promise<number[] | null> {
        const url = `${process.env.EMBEDDING_BASE_URL}/models/${this.embeddingModel}/pipeline/feature-extraction`;
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${process.env.EMBEDDING_API_KEY}`,
            },
            body: JSON.stringify({ inputs: document }),
            signal: AbortSignal.timeout(EMBEDDING_TIMEOUT_MS),
        });

        if (!response.ok) {
            console.error(`[embedding] HuggingFace API error: ${response.status} ${response.statusText}`);
            return null;
        }

        let data: any;
        try {
            data = JSON.parse(await response.text());
        } catch {
            console.error('[embedding] HuggingFace JSON parse error');
            return null;
        }

        // feature-extraction 批量为 1 时返回 [[n, n, ...]]
        if (Array.isArray(data) && Array.isArray(data[0])) return data[0];
        // 部分模型直接返回一维数组
        if (Array.isArray(data) && typeof data[0] === 'number') return data;
        console.error('[embedding] HuggingFace 响应格式无法识别');
        return null;
    }

    private async embedWithOpenAI(document: string): Promise<number[] | null> {
        const response = await fetch(`${process.env.EMBEDDING_BASE_URL}/embeddings`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${process.env.EMBEDDING_API_KEY}`,
            },
            body: JSON.stringify({ model: this.embeddingModel, input: document, encoding_format: 'float' }),
            signal: AbortSignal.timeout(EMBEDDING_TIMEOUT_MS),
        });

        if (!response.ok) {
            console.error(`[embedding] OpenAI API error: ${response.status} ${response.statusText}`);
            return null;
        }

        let data: any;
        try {
            data = JSON.parse(await response.text());
        } catch {
            console.error('[embedding] OpenAI JSON parse error');
            return null;
        }

        if (data?.data?.[0]?.embedding) return data.data[0].embedding;
        console.error('[embedding] OpenAI 响应格式无法识别');
        return null;
    }

    /** 返回结构化检索结果（带条号/章节元数据与分数），格式化交给调用方。 */
    async retrieve(query: string, topK: number = 5): Promise<RetrievedChunk[]> {
        const signals = parseQuery(query);
        const queryEmbedding = await this.embedQuery(query);

        // 召回池。默认就等于 topK，也就是**纯重排**：召回没有余量了（hit@5 = 1.000），
        // 开大池子只会让第 6~20 名里的错误条文有机会挤掉正确条文。理由详见 Reranker.ts。
        const pool = RERANK_CANDIDATES > topK ? RERANK_CANDIDATES : topK;
        const results = await this.vectorStore.search(queryEmbedding, this.reranker ? pool : topK, signals);
        const kept = results.filter(r => r.document.trim().length > 0);
        if (!this.reranker) return kept.slice(0, topK);

        // 置顶项不参与重排，理由见 partitionPinned：
        // 否则"点名条号"会先被置顶、再被 cross-encoder 以"不相关"为由踢下来——
        // 白折腾一趟，还倒退回修好之前。
        const { pinned, rerankable } = partitionPinned(kept);
        const reranked = await this.reranker.rerank(query, rerankable);
        return [...pinned, ...reranked].slice(0, topK);
    }

    size(): number {
        return this.vectorStore.size();
    }
}

// ---------------------------------------------------------------------------
// 切分与关键词提取：导出为纯函数，便于单测与 eval 直接调用
// ---------------------------------------------------------------------------

/**
 * chunk 的类型。
 * - article：一条法条正文
 * - heading：编/章/节标题，或文件标题
 * - content：前言、引用块、上传文档等其余内容
 */
export type ChunkKind = 'article' | 'heading' | 'content';

export interface ParsedChunk {
    document: string;
    source: string;
    /** 条号原文，如「第一百一十条」；章节标题等非条文 chunk 为 null */
    articleNo: string | null;
    /** 所属编/章/节路径，如「中华人民共和国民法典 > 第一编 总则 > 第五章 民事权利」 */
    chapter: string | null;
    kind: ChunkKind;
    /** 内容 hash，用于增量索引与上传去重 */
    hash: string;
    /**
     * **送去嵌入的文本**：正文前面拼上所属编/章/节路径。与 document 分开存。
     *
     * 为什么不直接改 document——document 是给用户看的：答案里引用的原文、
     * CitationVerifier 从中抽条号、eval 用它校对 ground truth，都读这个字段。
     * 往里塞「第二十五章 行纪合同」会原样漏进引用文本里。
     *
     * 为什么必须拼——民法典第960条与第966条只差一个条号：
     *   「第九百六十条 本章没有规定的，参照适用委托合同的有关规定。」
     *   「第九百六十六条 本章没有规定的，参照适用委托合同的有关规定。」
     * 26 个字里只有"六十六"那一个字的差别，而真正的区分信息全在所属章
     * （行纪合同 / 中介合同）上——章节名压根不在正文里。
     * 实测拿"中介合同这一章没有规定的…"去问，**错误的**960 以 0.8001 压过
     * **正确的**966 的 0.7950：系统对这类问题是稳定地答错。
     * 更糟的是 hit@1 一直显示 1.000——当时的答案键同时接受 960 和 966，
     * 指标在替错误打掩护。拼上章节路径后 966 反超 0.0826。
     */
    embedText: string;
}

/**
 * 单次 embedding 请求的超时（毫秒）。
 *
 * Node 的 undici 默认要 300 秒才放弃，而 VPN 抖动/中间设备丢包时连接会停在
 * ESTABLISHED 上一直不返回——表现为 `pnpm embed` 完全没输出、看起来像卡死，
 * 实际是卡在某一条的响应读取上。实测一次索引因此空转 5 分钟。
 * 30 秒足够一次正常的 feature-extraction（实测单条约 1 秒）。
 */
const EMBEDDING_TIMEOUT_MS = Number(process.env.EMBEDDING_TIMEOUT_MS ?? 30_000);

/**
 * 单次 embedding 的重试次数与退避。
 *
 * 实测：同一台机器、同一分钟里，curl 连续 5 次 200，而 node 连续 35 次
 * ECONNRESET（TLS 握手被重置）——这条链路是间歇性抖的，不是"通"或"不通"。
 * 一次索引要发 1680 个请求，不重试的话几乎必然踩到抖动，然后把整批索引
 * 悄悄降级成兜底向量（检索结果全错，但表面上"跑完了"）。
 */
const EMBEDDING_MAX_ATTEMPTS = Number(process.env.EMBEDDING_MAX_ATTEMPTS ?? 3);

/**
 * 只有网络层瞬时故障值得重试。
 * 接口返回 4xx/5xx 不重试——密钥错了、模型名错了，重试三次还是错，
 * 只会把一次明确的失败拖成三次。
 */
export function isTransientNetworkError(error: unknown): boolean {
    if (!(error instanceof Error)) return false;
    if (error.name === 'TimeoutError' || error.name === 'AbortError') return true;
    const code = (error.cause as { code?: string } | undefined)?.code;
    return code === 'ECONNRESET' || code === 'ECONNREFUSED' || code === 'ETIMEDOUT'
        || code === 'ENOTFOUND' || code === 'EAI_AGAIN' || code === 'EPIPE';
}

const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));

function describeNetworkError(error: unknown): string {
    if (error instanceof Error) {
        const code = (error.cause as { code?: string } | undefined)?.code;
        return code ?? error.name;
    }
    return String(error);
}

/** 严格模式下发生兜底即抛出，供 `pnpm embed` 用：索引残缺就该立刻停，而不是跑完 1680 条 */
export class EmbeddingUnavailableError extends Error {
    constructor(reason: string) {
        super(`embedding 不可用（${reason}），索引无法完整构建`);
        this.name = 'EmbeddingUnavailableError';
    }
}

const ARTICLE_RE = /^第[一二三四五六七八九十百千万零\d]+条/;
const ARTICLE_EN_RE = /^Article\s+\d+/;

/**
 * 以「第X条」为检索单元切分法律文本，并附带结构化元数据。
 *
 * 为什么不用通用的固定 token 窗口 + 重叠：
 * 法条是天然的语义与引用单元，条号本身就是 ground truth。固定窗口会在条文中间切断，
 * 也会让「条号 → 精确原文」的引用校验失去依据。
 * 只有在**没有条文格式**的文本上才退回按空行分段（并在段落间保留重叠）。
 */
export function splitIntoChunks(document: string, source: string): ParsedChunk[] {
    const rawParagraphs = document
        .split(/\n{2,}/)
        .map(p => p.trim())
        .filter(p => p.length > 0 && p !== '---');

    const chunks: ParsedChunk[] = [];
    const chapterStack: { level: number; title: string }[] = [];
    let buffer = '';
    let bufferArticleNo: string | null = null;
    let bufferChapter: string | null = null;

    const chapterPath = () => chapterStack.map(c => c.title).join(' > ') || null;

    const flush = () => {
        if (!buffer) return;
        chunks.push(makeChunk(buffer, source, bufferArticleNo, bufferChapter, bufferArticleNo ? 'article' : 'content'));
        buffer = '';
        bufferArticleNo = null;
    };

    for (const para of rawParagraphs) {
        const headerMatch = para.match(/^(#{1,6})\s+(.*)$/);
        const articleMatch = para.match(ARTICLE_RE) || para.match(ARTICLE_EN_RE);

        if (headerMatch) {
            flush();
            const level = headerMatch[1].length;
            const title = headerMatch[2].trim();
            while (chapterStack.length && chapterStack[chapterStack.length - 1].level >= level) {
                chapterStack.pop();
            }
            chapterStack.push({ level, title });
            // 标题仍然成 chunk，但标成 heading：它会被**排除出检索池**（见 VectorStore.search）。
            //
            // 原来是当成普通 chunk 检索的，理由是想回答"民法典有哪几编"这类结构问题。
            // 但代价实测无法接受：章标题的向量是整章语义的摘要，对任何牵涉该章的查询都拿高分
            // （"第八章 民事责任"对"第577条是什么"拿到 0.7613），系统性地挤掉真正的条文叶子。
            // 结构问题应该走 metadata（chapter 字段）或专门的结构化查询，不该占用检索位。
            chunks.push(makeChunk(para.replace(/^#+\s+/, ''), source, null, chapterPath(), 'heading'));
            continue;
        }

        if (articleMatch) {
            flush();
            buffer = para;
            bufferArticleNo = articleMatch[0];
            bufferChapter = chapterPath();
            continue;
        }

        // 引用块、前言，或长条文溢出的续段
        if (buffer) {
            buffer += '\n\n' + para;
        } else {
            chunks.push(makeChunk(para, source, null, chapterPath(), 'content'));
        }
    }
    flush();
    return chunks;
}

function makeChunk(document: string, source: string, articleNo: string | null, chapter: string | null, kind: ChunkKind): ParsedChunk {
    const prefix = chapterPrefix(chapter);
    return {
        document,
        source,
        articleNo,
        chapter,
        kind,
        hash: crypto.createHash('sha1').update(document).digest('hex').slice(0, 16),
        embedText: prefix ? `${prefix}\n${document}` : document,
    };
}

/**
 * 由章节路径生成拼进嵌入文本的前缀。
 *
 * 去掉首段（法律名，如"中华人民共和国民法典"）。这一条是量出来的，不是想出来的。
 * 同一个问题（near-02）下四种前缀的领先幅度：
 *   只留末级标题「第二十六章 中介合同」          0.0954
 *   编+章「第三编 合同 > 第二十六章 中介合同」    0.0826  ← 本规则
 *   完整路径（含"中华人民共和国民法典"）          0.0692
 *   法律名+末级标题                              0.0658
 * 法律名对同一部法律下的**所有**条文完全相同，
 * 只往共享分量上加码、不贡献任何区分度，等于把真正区分的那一段稀释掉。
 * 而"用户点名了哪部法律"这件事 parseQuery() 已经用 lawNames 做了结构化过滤，
 * 不需要嵌入空间再学一遍。
 *
 * 但**不能只留末级标题**（虽然它单看分数最高，是过拟合）：
 * 语料里「第一章 一般规定」「第一节 一般规定」各自出现在 6 条不同路径下，
 * 只留末级会让这些互不相干的章共享同一个前缀，反倒凭空制造相似度。
 * 所以保留法律名以下的整条路径——唯一性由编/章/节的层级保证。
 */
function chapterPrefix(chapter: string | null): string | null {
    if (!chapter) return null;
    const segs = chapter.split(' > ');
    // 只有一段时它就是全部（文件没有 H1 的情况），别把它也切掉
    const kept = segs.length > 1 ? segs.slice(1) : segs;
    return kept.join(' > ') || null;
}

/**
 * 从查询里抽结构化信号，交给 VectorStore 决定怎么加权。
 *
 * 原来这里返回 `string[]`，由 VectorStore 做 `document.includes(kw) || articleNo === kw`
 * 的字符串比对。两个坑：
 *   1. 语料存「第五百七十七条」，查询写「第577条」——字符串永不相等，
 *      于是"直接问某一条"这条**精度最高**的路径从来没有生效过
 *      （实测 article_lookup 的 hit@1 = 0.000，问第577条却召回章标题）。
 *   2. 抽出的裸数字（"577"）会去撞正文里的任意数字，制造无关命中。
 * 所以这里只输出归一后的语义：条号给**数值**，法律名给名字。数值比较是硬要求，
 * 同一条规矩在 articleNo.ts 里已经写过一次了。
 */
export function parseQuery(query: string): QuerySignals {
    return {
        articleNos: [...extractArticleNumbers(query).keys()],
        lawNames: KNOWN_LAW_NAMES.filter(name => query.includes(name)),
    };
}

/** 语料里可能出现的法律名。命中后作为 **source 过滤信号**，不再当正文字符串用。 */
const KNOWN_LAW_NAMES = [
    '民法典', '合同法', '公司法', '劳动法', '知识产权', '刑法',
    '行政法', '婚姻法', '继承法', '侵权责任法', '消费者权益保护法',
];
