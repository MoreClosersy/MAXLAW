import path from 'path';
import { CACHE_DIR } from './paths.js';

/**
 * Cross-encoder 精排（rerank）。
 *
 * **为什么需要它。** 召回已经到头了：hit@5 = 1.000，27/27 条 ground truth 全部进 top-5，
 * 一条都没漏。剩下的 6 条错在**排序**上——正确答案就在候选里，只是不在第一位。
 * 这是 bi-encoder 的固有短板：query 和 doc **各自独立**编码成一个向量，两者之间
 * 没有任何交互，编码「中介合同这一章没有规定的」时模型并不知道它将要和哪条法条比。
 * cross-encoder 把 (query, doc) 拼成一个序列做一次前向，注意力可以在两者之间直接流动，
 * 精度显著更高；代价是它不能预计算——每来一个候选就要跑一次前向。
 * 所以它只能用在精排阶段，用在召回阶段是算力上的自杀。
 *
 * **本地推理、零 API 成本。** 量化后的模型缓存在 `cache/models/`，首次运行联网下载，
 * 之后完全离线。跑在 CPU 上，不占 API 服务的额度，也不引入第二个 key。
 */

/** 换 reranker 模型跟换 embedding 模型一样，是配置不是代码 */
export const RERANK_MODEL = process.env.RERANK_MODEL ?? 'Xenova/bge-reranker-base';

/**
 * 默认**关闭**，`RERANK_ENABLED=1` 才加载。
 *
 * worktree 里那版是反的（默认开、`RERANK_DISABLED=1` 才关），这里刻意改过来：
 * 首次实例化要从 HuggingFace 拉模型，而"clone 下来就能跑起来"是 demo 的硬约束——
 * 一个还没配任何 key 的人，不该因为点了一次查询就卡在一个百兆下载上。
 * 想量 rerank 的效果时用 `pnpm eval:rerank`，那条路径显式打开它。
 */
export const RERANK_ENABLED = process.env.RERANK_ENABLED === '1';

/**
 * 召回多少条候选交给 cross-encoder。`0` 表示与 topK 相同——**这是默认值，而且是刻意的**。
 *
 * 直觉会说"召回 20 条再精排，正确答案更容易被捞回来"。但这里的召回**没有余量**：
 * hit@5 已经是 1.000，正确答案 100% 在前 5 里。把池子开到 20 的唯一效果，
 * 是让第 6~20 名里的错误条文获得一次挤掉正确条文的机会——纯风险，零收益。
 *
 * 还有一条更隐蔽的代价：拒答信号取的是 `Math.max(候选的余弦)`。
 * 池子一开大，进入最终 top-5 的**集合**就变了，这个最大值跟着变，
 * 于是"改善排序"这个动作会顺手改掉拒答行为（实测数字见 roadmap）。
 * 池子取 topK 时集合完全不变，拒答信号逐位不变——排序的归排序，拒答的归拒答。
 */
export const RERANK_CANDIDATES = Number(process.env.RERANK_CANDIDATES ?? 0);

export interface RerankCandidate {
    /**
     * 送去 cross-encoder 的文本。
     *
     * 与喂给 bi-encoder 的是**同一个字符串**（`ParsedChunk.embedText`，含编/章/节路径），
     * 不是干净的 `document`。原因是那类"只差一个条号"的条文：
     * 民法典第 960 条与第 966 条正文 26 个字里只差一个条号，区分信息全在所属章上
     * （行纪合同 / 中介合同），而章名压根不在正文里。只喂 document 就是让
     * cross-encoder 也瞎一次——它比 bi-encoder 强，但不会凭空知道这条属于哪一章。
     */
    embedText: string;
    /** 第一阶段 bi-encoder 的余弦，保留作对照。**不参与重排**，只用来事后看"重排把谁提上来了" */
    score?: number;
}

export interface RerankResult {
    /**
     * cross-encoder 打出的相关性 logit。
     *
     * **与余弦不是一个量纲**，不要和 `score` 比大小、也不要跨查询比较绝对值——
     * 它只在同一次 rerank 调用内部有排序意义。
     */
    rerankScore: number;
}

/**
 * 把"点名条号的精确命中"从重排里摘出来。
 *
 * 检索里有一类结果不是"相似度候选"，是**查找结果**：用户点名了第 577 条，
 * 而库里就有第 577 条。`VectorStore.search` 把它们置顶（理由见那里的注释：
 * 点名条号时余弦只有 0.584，反而低于无关条文的 0.742——相似度在这个场景里
 * 从原理上就问错了问题）。
 *
 * 那么精排就必须放过它们。理由不是"怕分数低"，是**这会退化成同一个 bug**：
 * 「民法典第577条规定了什么」这句话里，条号对任何模型都是纯噪声，
 * cross-encoder 照样可能觉得第 577 条正文和它不相关，把它判到第 4 位去。
 * 那样第 577 条会先被置顶、再被重排踢下来——白折腾一趟，还倒退回修好之前。
 * 确定性信号（条号精确匹配）与概率信号（相关性打分）的优先级不能被后者翻盘。
 *
 * 结构类型而不是 import `RetrievedChunk`：这个函数是纯的，不该把 reranker
 * 和向量库的类型绑在一起。
 */
export function partitionPinned<T extends { exactArticleHit?: boolean }>(
    candidates: readonly T[],
): { pinned: T[]; rerankable: T[] } {
    const pinned: T[] = [];
    const rerankable: T[] = [];
    for (const c of candidates) (c.exactArticleHit ? pinned : rerankable).push(c);
    return { pinned, rerankable };
}

/**
 * 精排器的最小接口。
 *
 * 抽出来不是为了"可替换实现"——是为了**可测**。"置顶项不被重排"这条不变量
 * 必须能在不下载 266MB 模型的前提下被单测钉住，所以测试要能塞进一个 stub。
 * 用结构类型而不是 class：`Reranker` 有私有字段，是标称类型，测试里造不出它的替身。
 */
export interface RerankerLike {
    rerank<T extends RerankCandidate>(
        query: string,
        candidates: readonly T[],
    ): Promise<(T & RerankResult)[]>;
}

export interface RerankStats {
    calls: number;
    /** 累计打分的 (query, doc) 对数量 */
    pairs: number;
    /** 累计精排耗时（不含模型加载） */
    rerankMs: number;
    /** 模型加载耗时；未加载为 0 */
    loadMs: number;
}

export default class Reranker implements RerankerLike {
    private tokenizer: any = null;
    private model: any = null;
    /**
     * 加载中的 Promise，**包括失败的那次**。
     *
     * 失败也要缓存：网络不通时不该每条查询都重试一遍下载——一次查询本来几十毫秒，
     * 重试下载会把它变成几十秒，而且日志会被同一个错误刷屏。
     * 失败后想恢复就重启进程，那是运维动作，不是请求路径上的事。
     */
    private loading: Promise<void> | null = null;
    private stats: RerankStats = { calls: 0, pairs: 0, rerankMs: 0, loadMs: 0 };

    getStats(): RerankStats {
        return { ...this.stats };
    }

    isReady(): boolean {
        return this.model !== null;
    }

    private load(): Promise<void> {
        if (!this.loading) {
            this.loading = (async () => {
                const t0 = Date.now();
                // 动态 import：没开 rerank 的进程不该为它付出任何加载成本
                // （transformers.js 拖进 onnxruntime，import 一次就是 1.5 秒）。
                const { AutoTokenizer, AutoModelForSequenceClassification, env } = await import('@huggingface/transformers');
                // 把模型缓存放进项目内，别污染 ~/.cache，也便于整个项目目录打包带走
                env.cacheDir = path.join(CACHE_DIR, 'models');
                console.log(`[rerank] 加载 ${RERANK_MODEL}（首次运行会从 HuggingFace 下载，之后离线可用）…`);
                this.tokenizer = await AutoTokenizer.from_pretrained(RERANK_MODEL);
                this.model = await AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL, {
                    // q8 = int8 量化：CPU 上快 2-4 倍，精度损失对排序几乎无感。
                    // 已对着安装的 transformers.js 源码核过：dtype 'q8' 映射到文件名后缀
                    // `_quantized`，也就是 Xenova/bge-reranker-base 仓库里的
                    // onnx/model_quantized.onnx——不是猜的。
                    dtype: 'q8',
                });
                this.stats.loadMs = Date.now() - t0;
                console.log(`[rerank] 就绪（${(this.stats.loadMs / 1000).toFixed(1)}s）`);
            })();
        }
        return this.loading;
    }

    /** 预加载，供启动时显式预热（不预热则首次查询多等一次模型加载） */
    async init(): Promise<void> {
        await this.load();
    }

    /**
     * 对候选逐对打 (query, doc) 相关性分，按分数降序返回**全部**候选。
     *
     * 不在这里截 topK：置顶项与重排项要合并之后再算预算，
     * 在这里截会让"置顶项占掉一个名额"变成"重排项少一个名额"这种错位。
     * 截断交给调用方（`EmbeddingRetriever.retrieve`）。
     */
    async rerank<T extends RerankCandidate>(
        query: string,
        candidates: readonly T[],
    ): Promise<(T & RerankResult)[]> {
        if (candidates.length === 0) return [];
        await this.load();

        const t0 = Date.now();
        // 同一个 query 配每个候选，组成 N 个 (query, doc) 对，一次前向批处理完
        const inputs = await this.tokenizer(new Array(candidates.length).fill(query), {
            text_pair: candidates.map(c => c.embedText),
            padding: true,
            truncation: true,
            max_length: 512,
        });
        const { logits } = await this.model(inputs);
        const scores = Array.from(logits.data as Float32Array);

        // 形状校验，和批量 embedding 那里同一个理由：一旦得分与候选错位，
        // 表现不是报错，而是"排序看起来有点怪"——最难查的一类 bug。
        if (scores.length !== candidates.length) {
            throw new Error(
                `[rerank] 得分数量与候选数不符：${scores.length} vs ${candidates.length}。` +
                `拒绝返回可能错位的重排结果。`
            );
        }

        this.stats.calls++;
        this.stats.pairs += candidates.length;
        this.stats.rerankMs += Date.now() - t0;

        return candidates
            .map((c, i) => ({ ...c, rerankScore: scores[i] }))
            .sort((a, b) => b.rerankScore - a.rerankScore);
    }
}
