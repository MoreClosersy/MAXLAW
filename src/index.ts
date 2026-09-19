import MCPClient from "./MCPClient.js";
import Agent, { AgentEvent } from "./Agent.js";
import EmbeddingRetriever from "./EmbeddingRetriever.js";
import Reranker, { RERANK_ENABLED, RERANK_MODEL, RERANK_CANDIDATES } from "./Reranker.js";
import { RetrievedChunk } from "./VectorStore.js";
import { loadPrompt } from "./prompts.js";
import { verifyCitations, summarizeCitations, CitationReport } from "./CitationVerifier.js";
import { OUTPUT_DIR } from "./paths.js";
import fs from "fs";
import { decideAbstention, shouldSkipJudge, mergeAbstention } from "./abstention.js";
import type { AbstentionDecision, AbstentionReason, JudgeOutcome } from "./abstention.js";
import {
    judgeAnswerability,
    skippedAccounting,
    ANSWERABILITY_JUDGE_ENABLED,
    type JudgeAccounting,
} from "./AnswerabilityJudge.js";

// 可被环境变量覆盖：换 embedding 模型是检索质量最大的一根杠杆，而它此前写死在源码里，
// 意味着"试一个模型"必须先改代码。注意换模型会改变**每一条**分数（语料指纹不变），
// 所以 eval 报告必须记录模型名——否则两次不同模型的评测会挂着同样的标签。
// 默认用中文特化模型。这不是偏好，是量出来的：同一份语料指纹、同一个数据集、同一份切分代码，
// **只换 embedding 模型**（miniLM 384 维 → bge-base-zh 768 维），整体 hit@1/hit@5/MRR
// 从 0.593/0.741/0.649 升到 0.778/1.000/0.862，semantic 的 hit@5 从 0.667 升到 1.000
// （7 条词汇鸿沟型失败清零）。语料指纹与数据集 SHA1 两趟完全一致，所以差异只可能来自模型。
// 之前那批"词汇鸿沟"型失败（口语化提问与法条措辞无共同词）是通用多语模型撑不住的。
// 出处：README 的 "The embedding model was chosen by measurement" 一节 + eval/report.json。
//
// 注意别把两次改动串成一次（这里此前就写错成 ".741 -> .963"）：0.963 是 bge 在**章节前缀
// 进入嵌入文本之前**的 hit@5，属于另一次改动；只换模型的终点是 1.000。
export const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL
    ?? "BAAI/bge-base-zh-v1.5";
export const CHAT_MODEL = process.env.CHAT_MODEL || "gpt-4o-mini";

/** 检索最高分低于此阈值，视为"没检索到相关条文"，走拒答分支而不是让模型编 */
export const RETRIEVAL_SCORE_THRESHOLD = Number(process.env.RETRIEVAL_SCORE_THRESHOLD ?? 0.35);
export const RETRIEVAL_TOP_K = Number(process.env.RETRIEVAL_TOP_K ?? 5);

/**
 * 构造检索器的**唯一入口**。
 *
 * 为什么要有这个函数，而不是各处在 `new EmbeddingRetriever(EMBEDDING_MODEL)` 后面
 * 自己接一段 if：server 和 eval 必须跑在**同一条**检索路径上。
 * 分开写的话，"评测里开了精排、线上忘了开"（或反过来）会变成一种两个都说得通、
 * 但数字对不上的状态——而离线的全部意义就在于它测的就是线上那条路。
 */
export function createRetriever(): EmbeddingRetriever {
    const reranker = RERANK_ENABLED ? new Reranker() : null;
    if (reranker) {
        console.log(
            `[rerank] 已启用：模型 ${RERANK_MODEL}，候选池 ` +
            (RERANK_CANDIDATES > 0 ? `${RERANK_CANDIDATES} 条` : '同 topK（纯重排）')
        );
    }
    return new EmbeddingRetriever(EMBEDDING_MODEL, reranker);
}

const PROMPT_FILE_VERSION = Number(process.env.PROMPT_VERSION ?? 3);
const { body: LEGAL_SYSTEM_PROMPT, version: PROMPT_VERSION } = loadPrompt('legal-system-prompt', PROMPT_FILE_VERSION);
export { LEGAL_SYSTEM_PROMPT, PROMPT_VERSION };

if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

// ---------------------------------------------------------------------------
// MCP 客户端：模块级单例 + 健康检查
//
// 原实现每次 /api/chat 都 new 两个 MCPClient，等于每个请求 spawn 两个子进程
// （uvx / npx，首次还要联网下载），握手串行、无并发上限。改为进程内复用。
// ---------------------------------------------------------------------------
let sharedClients: MCPClient[] | null = null;

function getMcpClients(): MCPClient[] {
    if (!sharedClients) {
        sharedClients = [
            new MCPClient("mcp-server-fetch", "uvx", ['mcp-server-fetch']),
            new MCPClient("mcp-server-file", "npx", ['-y', '@modelcontextprotocol/server-filesystem', OUTPUT_DIR]),
        ];
    }
    return sharedClients;
}

/** 供进程退出时统一回收子进程 */
export async function shutdownMcpClients() {
    if (!sharedClients) return;
    for (const c of sharedClients) {
        try { await c.close(); } catch { /* ignore */ }
    }
    sharedClients = null;
}

export interface QueryOptions {
    query: string;
    retriever: EmbeddingRetriever;
    history?: { role: 'user' | 'assistant'; content: string }[];
    onToken?: (token: string) => void;
    onEvent?: (event: AgentEvent) => void;
    signal?: AbortSignal;
}

export interface QueryResult {
    answer: string;
    retrieved: RetrievedChunk[];
    abstained: boolean;
    /** **仅生成模型**的开销。总成本 = `usage + judge.usage` */
    usage: { promptTokens: number; completionTokens: number };
    promptVersion: number;
    /**
     * 可答性判官的判定与开销。
     *
     * 为什么单独一个字段而不是并进 `usage`：`usage` 已经在 `server.ts` 发出去、前端已经在渲染。
     * 改它的含义就是造出"两个看起来是同一个测量的数"——判官上线后同一条查询的 token
     * 会凭空翻倍，而看板上没有任何东西会说明为什么。分开报，代价是每个读的人要知道
     * `total = usage + judge.usage`；这句话写在 README 的 Configuration 一节。
     */
    judge: JudgeAccounting;
    /**
     * 对答案里每个「第X条」的核对结果。
     * 注意与 `retrieved` 的区别：`retrieved` 是**检索到了什么**，
     * `citations` 是**模型实际引用了什么、引用站不站得住**。两者不一致才是要抓的东西。
     */
    citations: CitationReport;
}

/**
 * ## 拒答：实现已搬到 `./abstention.js`
 *
 * `decideAbstention` 和 `AbstentionDecision` 的实现现在住在 `src/abstention.ts`——
 * 纯逻辑、零副作用，因此**终于可以单测**。放在这个文件里的那些年它一行测试都没有，
 * 而原因就是这个文件顶层有 `mkdirSync` 和 `getMcpClients()`（会 spawn 子进程）：
 * 任何 import 它的测试都会顺带起两个子进程。
 *
 * 这里 re-export 只是为了保持公开面不变（`eval/run.ts` 按名字 import 它），
 * 门槛逻辑本身一个字都没改。
 *
 * ## 下面这张表是历史记录，别删——它是"为什么不能靠阈值"的全部证据
 *
 * ⚠️ 已知缺陷（`pnpm eval:signals` 实测）：**相似度阈值几乎不触发**。
 *
 * 默认阈值 0.35 在 69 题数据集（27 域内 / 42 域外）上拒掉 3/42 道域外题、误拒 0 道域内题。
 * 看起来"零误拒"很安全，但那是因为它同时基本什么都没拦——**它不是拒答机制，只是个
 * 明显的跑题过滤器**。
 *
 * 把所有候选信号量过一遍（AUC，域内 vs 域外，0.5 = 抛硬币）：
 *   cross-encoder 重排  0.938   ← 最好的一个，仍然不够
 *   bge top-1 余弦      0.891
 *   MiniLM 余弦         0.819
 *   bge 相对间隔        0.725
 *   MiniLM 相对间隔     0.450   ← 比抛硬币还差，**反相关**
 * 且"拦住全部 42 道域外"的阈值会误拒 21/27 道域内题；"零误拒"的阈值只能拦住 18/42。
 * 也就是说：通用 embedding 的余弦相似度在**同一语域内部**有个很高的地板，
 * 绝对分数不是可校准的相关性信号。
 *
 * 更根本的原因（已用两个实例确认）：这类信号的本质是**话题相关性，不是可答性**。
 * sem-02 的正确条文排在检索第 4 位但分数全库最低（问题口语化、条文书面化）；
 * ood-11 逐字命中了民法典第181条却分数最高（该条讲民事责任，问题问刑事责任）。
 * 于是它**先拒掉口语化的域内问题、最后才拒掉逐字命中的域外问题**——顺序正好反了。
 *
 * **结论（2026-09-19 已落地）**：拒答不再是任何相似度分数上的阈值。
 * 现在的链路是三段：相似度判定当"明显跑题过滤器" → 检索后一次 LLM 可答性判定
 * （要求判官**指名**哪一条回答了问题）→ 指名结果做确定性交叉校验，失败则退回相似度判定。
 * 见 `src/abstention.ts` 与 `src/AnswerabilityJudge.ts`。上面这个阈值仍然保留，
 * 因为它是判官调用失败时的回退目标——退回**已测量的现状**好过退回一个没人量过的状态。
 */
export { decideAbstention };
export type { AbstentionDecision, AbstentionReason };
export async function runQuery(opts: QueryOptions): Promise<QueryResult> {
    const { query, retriever } = opts;

    opts.onEvent?.({ type: 'iteration', message: '正在检索法条…' });
    const retrieved = await retriever.retrieve(query, RETRIEVAL_TOP_K);

    // 第一段：相似度。它拦不住该拦的东西（见 decideAbstention 上面那张 AUC 表），
    // 现在只负责两件事——挡掉明显跑题的，以及在判官故障时当回退目标。
    const sim = decideAbstention(retrieved, RETRIEVAL_SCORE_THRESHOLD);

    // 第二段：可答性判官。跳过规则是确定性的（纯函数，有单测），
    // 不花钱的情形不花钱，理由记进账单。
    const skip = shouldSkipJudge(retrieved, sim);
    let outcome: JudgeOutcome;
    let judge: JudgeAccounting;
    if (!ANSWERABILITY_JUDGE_ENABLED) {
        // 逃生阀优先于一切跳过规则：关掉就是没问，不去编造一个更具体的理由
        outcome = { kind: 'skipped', reason: 'disabled' };
        judge = skippedAccounting('disabled');
    } else if (skip.skip) {
        outcome = { kind: 'skipped', reason: skip.reason! };
        judge = skippedAccounting(skip.reason!);
    } else {
        // 没有这条 onEvent，SSE 在判官往返的 1-3 秒里完全静默，界面看起来是卡死的
        opts.onEvent?.({ type: 'iteration', message: '正在判定检索到的条文能否回答该问题…' });
        ({ outcome, accounting: judge } = await judgeAnswerability(query, retrieved, { signal: opts.signal }));
    }

    // 第三段：合并。判官失败时 mergeAbstention 原样返回 sim —— 退回**已测量的现状**，
    // 而不是退回"默认拒答"（判官故障不该导致误拒）或"默认回答"（解析失败不该等于可答）。
    const decision = mergeAbstention(sim, outcome);
    const { abstained, topScore } = decision;

    const context = formatContext(retrieved);
    const systemPrompt = abstained
        ? LEGAL_SYSTEM_PROMPT + '\n\n' + ABSTENTION_INSTRUCTION
        : LEGAL_SYSTEM_PROMPT;

    opts.onEvent?.({
        type: 'iteration',
        message: abstained
            ? abstentionEventMessage(decision.reason, topScore)
            : `检索到 ${retrieved.length} 条相关条文（最高分 ${topScore.toFixed(3)}）`,
    });

    const agent = new Agent({
        model: CHAT_MODEL,
        mcpClients: getMcpClients(),
        systemPrompt,
        context,
        history: opts.history,
        onToken: opts.onToken,
        onEvent: opts.onEvent,
        signal: opts.signal,
    });

    await agent.init();
    // 注意：这里不 close()——MCP 客户端是进程级复用的单例，
    // 请求结束就关掉会让下一个请求重新 spawn 子进程。进程退出时由 shutdownMcpClients() 统一回收。
    const answer = await agent.invoke(query);

    // 引用校验放在模型返回之后、返回给调用方之前。
    // 不在这里做"发现问题就重生成"的自动纠偏——那会掩盖模型的真实行为，
    // 让引用准确率这个指标失去意义。先如实报出来，纠偏策略等有了 eval 数据再定。
    const citations = verifyCitations(answer, retriever.getStore(), retrieved);
    const summary = summarizeCitations(citations);
    if (summary && citations.hasProblem) {
        console.warn(`[citation] ${summary}（拒答=${abstained}，prompt=v${PROMPT_VERSION}）`);
        for (const c of citations.checks) {
            if (c.status === 'verified') continue;
            console.warn(
                `[citation]   ${c.label}（原文「${c.raw}」）` +
                (c.status === 'fabricated'
                    ? ' —— 语料中不存在此条号'
                    : ` —— 条号真实存在（${c.sources.join('、')}），但未出现在本次检索结果中`)
            );
        }
    }

    return {
        answer,
        retrieved,
        abstained,
        usage: agent.getUsage(),
        promptVersion: PROMPT_VERSION,
        judge,
        citations,
    };
}

/**
 * 拒答时汇报给前端的那句话。
 *
 * 「没检索到」和「检索到了、但答不了」在产品上是两件不同的事——后者才是这一轮
 * 加判官的全部意义，把它说成前者会让整个机制在 demo 里看不见。
 * 判官**为什么**这么判不在这里说：那是未经审阅的模型理由，答案正文已经给了用户可读的说明。
 */
function abstentionEventMessage(reason: AbstentionReason, topScore: number): string {
    switch (reason) {
        case 'judge_unanswerable':
            return '判官判定：检索到的条文回答不了这个问题，进入拒答模式';
        case 'judge_parse_failure':
        case 'judge_error_fallback':
        case 'judge_binding_mismatch':
            return `判官未能给出可用判定（${reason}），退回相似度规则：最高分 ${topScore.toFixed(3)}`;
        default:
            return `检索最高分 ${topScore.toFixed(3)} 低于阈值 ${RETRIEVAL_SCORE_THRESHOLD}，进入拒答模式`;
    }
}

const ABSTENTION_INSTRUCTION = `## 本次检索结果不足（重要）

本次在本地法律知识库中**没有**检索到与用户问题足够相关的条文。你必须：
1. 开门见山地告诉用户"本地知识库中没有检索到与该问题直接相关的法律条文"；
2. **不得**凭记忆或常识编造任何法条条号、条文内容；
3. 可以说明该问题大致属于哪个法律领域、建议用什么关键词或条号重新提问；
4. 建议用户咨询专业律师或查阅官方法律数据库。

宁可明确回答"查不到"，也不要给出一个看似专业但无依据的回答。`;

/** 把结构化检索结果拼成给模型看的 context（保留条号与章节，便于模型准确标注出处） */
export function formatContext(chunks: RetrievedChunk[]): string {
    if (chunks.length === 0) return '（本地知识库未检索到相关条文）';
    return chunks.map((c, i) => {
        const head = [
            `[${i + 1}] 来源：${c.source}`,
            c.chapter ? `章节：${c.chapter}` : null,
            c.articleNo ? `条号：${c.articleNo}` : null,
            `相关度：${c.score.toFixed(3)}`,
        ].filter(Boolean).join(' | ');
        return `${head}\n${c.document}`;
    }).join('\n\n---\n\n');
}
