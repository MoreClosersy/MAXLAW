/**
 * 可答性判官：**检索之后那一次 LLM 调用**。
 *
 * 这个模块有副作用（网络、文件、环境变量），所以判定逻辑一个字都不在这里——
 * 全在 `src/abstention.ts`（纯逻辑、可单测）。这条分工是踩出来的：
 * `eval/judgeCore.ts` 之所以存在，就是因为 `eval/faithfulness.ts` 顶层有
 * `await loadKnowledge()`，第一次 `pnpm test` 直接挂了两分钟。
 *
 * ## 为什么需要它
 *
 * 拒答原先只是"检索最高分 < 0.35"，`pnpm eval:signals` 量出来它只拦下 42 道域外题里的 3 道。
 * 根因不是阈值没调好，而是**话题相关度不是可答性**：
 * `sem-02`（口语化问法）的正确条文被检索到了却是全库最低分；
 * `ood-11`（问刑事责任）逐字命中民法典第181条却是全部域外题的最高分。
 * 任何相似度阈值都会先拒掉口语化的域内问题、最后才拦下逐字命中的域外问题——顺序正好反了。
 *
 * ## 与 faithfulness 判官的关系
 *
 * 重试、缓存键、token 记账这几套照抄 `eval/faithfulness.ts` 的 `judge()`，但**环境变量刻意分开**
 * （`ANSWERABILITY_JUDGE_*` 而不是 `JUDGE_*`）：共用一个会让"改 faithfulness 判官模型"静默失效
 * 本模块的缓存、并改变拒答数字，把两个不相干的测量耦合到一起。
 *
 * 另一处**刻意不照抄**的是重试循环的异常处理。faithfulness 那版 catch 一切并重试三次——
 * 在评测脚本里没问题，在服务里意味着客户端断开后还要重试两次、然后为已经不存在的浏览器
 * 生成一整篇回答。所以这里把 abort 单独拎出来向上抛。
 */
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import OpenAI from 'openai';
import { loadPrompt } from './prompts.js';
import { CACHE_DIR } from './paths.js';
import type { RetrievedChunk } from './VectorStore.js';
import {
    formatJudgeContext,
    parseAnswerabilityOutput,
    crossCheckAnswerability,
    isParseError,
    type JudgeOutcome,
    type JudgeSkipReason,
} from './abstention.js';

/** 默认**开**。拒绝回答一个库外问题是正确性修复，不是一个可选优化。 */
export const ANSWERABILITY_JUDGE_ENABLED = process.env.ANSWERABILITY_JUDGE !== '0';
export const ANSWERABILITY_JUDGE_MODEL = process.env.ANSWERABILITY_JUDGE_MODEL ?? 'gpt-4o';
export const ANSWERABILITY_JUDGE_TIMEOUT_MS = Number(process.env.ANSWERABILITY_JUDGE_TIMEOUT_MS ?? 8_000);
export const ANSWERABILITY_JUDGE_PROMPT_VERSION = Number(process.env.ANSWERABILITY_JUDGE_PROMPT_VERSION ?? 1);

export interface JudgeAccounting {
    outcome: 'judged' | 'skipped' | 'binding_mismatch' | 'parse_failure' | 'error';
    skipReason: JudgeSkipReason | null;
    /** 仅 judged / binding_mismatch 有值 */
    answerable: boolean | null;
    cited: { law: string; articleNo: number } | null;
    binding: 'in_context' | 'not_in_context' | 'not_applicable' | null;
    /** 判官摘的原文是否真在所指数那条里。仅诊断，从不翻转判定。 */
    quoteVerified: boolean | null;
    /**
     * 判官指名的条文在**这次检索结果里**的分数。仅诊断，从不参与判定。
     * `eval/abstention.ts` 用它扫"判官之后再加一道检索分门还有没有增量信息"——
     * 那个门的形状是 `answerable && namedScore < t`，所以判的是**低分冒充答案**。
     * 不可答 / 指名不在检索结果里 / 硬跳过 → null（null 不是 0，不能混）。
     */
    namedScore: number | null;
    /** 判官原话，进日志与报告，**不进 UI**（未审阅的模型理由放在界面上是负债） */
    reason: string | null;
    cached: boolean;
    latencyMs: number;
    usage: { promptTokens: number; completionTokens: number };
    model: string;
    promptVersion: number;
    error?: string;
    /** 解析失败时模型的原话，截断后留档——§10.13 的教训：第一次校准能诊断全靠把判官原话打出来 */
    raw?: string;
}

const ZERO_USAGE = { promptTokens: 0, completionTokens: 0 };
const CACHE_PATH = path.join(CACHE_DIR, 'answerability-judge.json');
/** 上限。服务是长跑进程，不裁剪的话这个文件会一直长。 */
const CACHE_MAX_ENTRIES = 5000;
/** 落盘节流：每次调用都整文件写 2MB 不值得，攒 2 秒写一次。 */
const PERSIST_THROTTLE_MS = 2_000;
const MAX_RETRIES = 3;
/** 日志里模型原话的截断长度。够看出它是格式坏了还是压根在说别的事。 */
const RAW_LOG_CHARS = 200;

function truncate(s: string | null | undefined): string {
    const t = (s ?? '').replace(/\s+/g, ' ').trim();
    return t.length > RAW_LOG_CHARS ? `${t.slice(0, RAW_LOG_CHARS)}…` : t || '(空)';
}

interface CacheEntry {
    at: number;
    outcome: JudgeOutcome;
    usage: { promptTokens: number; completionTokens: number };
}

// ---------------------------------------------------------------------------
// 懒加载 + 记忆化
//
// 三样东西都不能在模块顶层构造：
// - `new OpenAI()` 在没有 API key 时会抛，顶层构造会让"没配 key 就起不来服务"；
// - `loadPrompt()` 在文件缺失时会抛，同理（import 本模块就炸）；
// - 缓存文件读盘不该发生在 import 期（eval 只是想 import 一个类型也会被牵连）。
// ---------------------------------------------------------------------------
let client: OpenAI | null = null;
let judgePrompt: string | null = null;
let cacheLoaded = false;
let cache: Record<string, CacheEntry> = {};
let lastPersist = 0;

function getClient(): OpenAI {
    if (!client) {
        client = new OpenAI({
            apiKey: process.env.OPENAI_API_KEY,
            baseURL: process.env.OPENAI_BASE_URL || undefined,
        });
    }
    return client;
}

function getPrompt(): string {
    if (judgePrompt === null) {
        judgePrompt = loadPrompt('answerability-judge', ANSWERABILITY_JUDGE_PROMPT_VERSION).body;
    }
    return judgePrompt;
}

function getCache(): Record<string, CacheEntry> {
    if (!cacheLoaded) {
        cacheLoaded = true;
        try {
            if (fs.existsSync(CACHE_PATH)) {
                cache = JSON.parse(fs.readFileSync(CACHE_PATH, 'utf-8'));
            }
        } catch {
            // 文件损坏不是致命错误，重头攒就行——缓存是省钱的中间态，不是事实来源
            cache = {};
        }
    }
    return cache;
}

/** 供评测脚本在结束前调用，保证最后几条也落盘（节流窗口内的那几条否则会丢）。 */
export function flushJudgeCache(): void {
    if (!cacheLoaded) return;
    persist(true);
}

function persist(force = false): void {
    if (!cacheLoaded) return;
    const now = Date.now();
    if (!force && now - lastPersist < PERSIST_THROTTLE_MS) return;
    lastPersist = now;
    try {
        if (!fs.existsSync(CACHE_DIR)) fs.mkdirSync(CACHE_DIR, { recursive: true });
        const keys = Object.keys(cache);
        if (keys.length > CACHE_MAX_ENTRIES) {
            // 按写入时间留最新的那一批
            const drop = keys.sort((a, b) => cache[a].at - cache[b].at).slice(0, keys.length - CACHE_MAX_ENTRIES);
            for (const k of drop) delete cache[k];
        }
        fs.writeFileSync(CACHE_PATH, JSON.stringify(cache));
    } catch (e) {
        console.warn(`[judge] 缓存落盘失败（不影响判定）：${(e as Error).message}`);
    }
}

// ---------------------------------------------------------------------------
// 进程内计数器
//
// 注意：这是**进程生命周期**状态，重启即清零，所以它不是评测的事实来源
// （`QueryResult.judge` 才是）。它只为了 /api/metrics 上能看见"判官是不是在失败"。
// 它给本模块加了可变状态，而可变状态正是让 src/index.ts 不可测的原因——
// 所以它必须限定为无 I/O 的计数器，且**永远不让可测的东西 import 本模块**。
// ---------------------------------------------------------------------------
const stats = { calls: 0, cacheHits: 0, errors: 0, parseFailures: 0, skipped: 0, bindingMismatches: 0, abstained: 0 };

export function getJudgeStats() {
    return { ...stats };
}

/** 跳过判官时的记账。调用方拿到的是"这次没问判官"，不是"判官说可答"。 */
export function skippedAccounting(reason: JudgeSkipReason, model: string = ANSWERABILITY_JUDGE_MODEL): JudgeAccounting {
    stats.skipped++;
    return {
        outcome: 'skipped',
        skipReason: reason,
        answerable: null,
        cited: null,
        binding: null,
        quoteVerified: null,
        namedScore: null,
        reason: null,
        cached: false,
        latencyMs: 0,
        usage: ZERO_USAGE,
        model,
        promptVersion: ANSWERABILITY_JUDGE_PROMPT_VERSION,
    };
}

function judgeKey(model: string, question: string, context: string): string {
    return crypto.createHash('sha1')
        .update(`${model}\u0000v${ANSWERABILITY_JUDGE_PROMPT_VERSION}\u0000${question}\u0000${context}`)
        .digest('hex');
}

/**
 * 判官指名的条文在检索结果里的分数。别名解析用的是 `check.matchedIndex`，
 * **不在这里重算一遍法律名归一化**——那会造出这套语义的第二个事实来源（§10.13 坑 1）。
 */
function namedScoreFor(outcome: JudgeOutcome, retrieved: RetrievedChunk[]): number | null {
    if (outcome.kind !== 'judged' && outcome.kind !== 'binding_mismatch') return null;
    const i = outcome.check.matchedIndex;
    return i === null ? null : retrieved[i]?.score ?? null;
}

function accountingFor(
    outcome: JudgeOutcome,
    model: string,
    extra: {
        cached: boolean; latencyMs: number;
        usage: { promptTokens: number; completionTokens: number };
        namedScore: number | null;
    },
): JudgeAccounting {
    const base = {
        skipReason: null as JudgeSkipReason | null,
        reason: null as string | null,
        answerable: null as boolean | null,
        cited: null as { law: string; articleNo: number } | null,
        binding: null as JudgeAccounting['binding'],
        quoteVerified: null as boolean | null,
        namedScore: extra.namedScore,
        cached: extra.cached,
        latencyMs: extra.latencyMs,
        usage: extra.usage,
        model,
        promptVersion: ANSWERABILITY_JUDGE_PROMPT_VERSION,
    };
    switch (outcome.kind) {
        case 'judged':
        case 'binding_mismatch':
            return {
                ...base,
                outcome: outcome.kind === 'judged' ? 'judged' as const : 'binding_mismatch' as const,
                answerable: outcome.parsed.answerable,
                cited: outcome.parsed.cited,
                binding: outcome.check.binding,
                quoteVerified: outcome.check.quoteVerified,
                reason: outcome.parsed.reason || null,
            };
        case 'parse_failure':
            return { ...base, outcome: 'parse_failure' as const, error: outcome.error, raw: outcome.raw };
        case 'error':
            return { ...base, outcome: 'error' as const, error: outcome.error };
        case 'skipped':
            return { ...base, outcome: 'skipped' as const, skipReason: outcome.reason };
    }
}

/**
 * 问一次判官：检索到的这些条文里，有没有哪一条回答了这个问题。
 *
 * 失败一律以 `{kind:'error'|'parse_failure'}` 返回，**绝不返回一个"可答"或"不可答"的判定**——
 * 把工具的故障记成判定结论是这条链上最容易犯的自欺。调用方 `mergeAbstention` 会把失败
 * 退回相似度判定，也就是退回**已经量过的现状**。
 *
 * 唯一例外是 **abort**：它向上抛。在服务里 abort 意味着客户端已经断开，
 * 这时候既不该重试，也不该继续生成。
 */
export async function judgeAnswerability(
    question: string,
    retrieved: RetrievedChunk[],
    opts: { signal?: AbortSignal; model?: string } = {},
): Promise<{ outcome: JudgeOutcome; accounting: JudgeAccounting }> {
    const context = formatJudgeContext(retrieved);
    // 模型是**每次调用**的参数，不是只读模块常量：评测要在同一个进程里跑两遍
    // （`--judge-model-b` 量判官身份对结论的影响），而 ESM 的模块只求值一次，
    // 靠改环境变量换模型在同一个进程里做不到。默认值仍然是环境变量。
    const model = opts.model ?? ANSWERABILITY_JUDGE_MODEL;
    const key = judgeKey(model, question, context);
    const store = getCache();

    const hit = store[key];
    if (hit) {
        stats.cacheHits++;
        if ((hit.outcome.kind === 'judged' || hit.outcome.kind === 'binding_mismatch') && !hit.outcome.parsed.answerable) {
            stats.abstained++;
        }
        return {
            outcome: hit.outcome,
            accounting: accountingFor(hit.outcome, model, {
                cached: true, latencyMs: 0, usage: hit.usage,
                namedScore: namedScoreFor(hit.outcome, retrieved),
            }),
        };
    }

    const user =
        `<question>\n${question}\n</question>\n\n` +
        `<retrieved_context>\n${context}\n</retrieved_context>`;

    let promptText: string;
    let api: OpenAI;
    try {
        promptText = getPrompt();
        api = getClient();
    } catch (e) {
        // 没配 key / prompt 文件缺失 —— 不是"不可答"，是判官不在
        stats.errors++;
        const outcome: JudgeOutcome = { kind: 'error', error: (e as Error).message };
        return {
            outcome,
            accounting: accountingFor(outcome, model, {
                cached: false, latencyMs: 0, usage: ZERO_USAGE, namedScore: null,
            }),
        };
    }

    const started = Date.now();
    let lastErr = '';
    let spent = { ...ZERO_USAGE };

    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        // 断开之后不要再试一次：那会为一个已经不存在的浏览器多花一次钱和一次延迟
        if (opts.signal?.aborted) throw new Error('aborted before judge attempt');

        // 超时必须真的取消底层请求。src/Agent.ts 那个私有的 withTimeout 只 reject 外层包装，
        // HTTP 请求还在跑——那正是超时要防的泄漏。AbortSignal.any 同时管断开和超时。
        const timeout = AbortSignal.timeout(ANSWERABILITY_JUDGE_TIMEOUT_MS);
        const signal = opts.signal ? AbortSignal.any([opts.signal, timeout]) : timeout;

        try {
            const resp = await api.chat.completions.create({
                model,
                messages: [
                    { role: 'system', content: promptText },
                    { role: 'user', content: user },
                ],
                // 判定要可复现：同一份输入判两次必须得到同一个结论
                temperature: 0,
                response_format: { type: 'json_object' },
            }, { signal });

            const usage = {
                promptTokens: resp.usage?.prompt_tokens ?? 0,
                completionTokens: resp.usage?.completion_tokens ?? 0,
            };
            // 在调用点累加（含重试）——从结果上 reduce 会漏掉重试和缓存未命中的那部分
            spent = {
                promptTokens: spent.promptTokens + usage.promptTokens,
                completionTokens: spent.completionTokens + usage.completionTokens,
            };

            const raw = resp.choices[0]?.message?.content ?? '';
            const parsed = parseAnswerabilityOutput(raw);
            let outcome: JudgeOutcome;
            if (isParseError(parsed)) {
                // 解析失败**要**缓存：同一个 prompt 再问一遍大概率还是坏格式。
                // 它与"网络错误"是两回事，后者不能缓存——那会把一次瞬时抖动冻进产物里。
                stats.parseFailures++;
                // 截断后再进缓存：这个字段会被写进 cache/answerability-judge.json，
                // 存全文等于让一个坏格式的模型输出把缓存文件撑大。
                outcome = { kind: 'parse_failure', error: parsed.error, raw: truncate(raw) };
                // 把模型原话打出来。§10.13 的第一次校准能诊断出问题，全靠这一步；
                // 只记一句"解析失败"的话，只能知道它坏了，永远不知道它坏在哪。
                console.warn(`[judge] parse_failure：${parsed.error}。原话：${truncate(raw)}`);
            } else {
                const check = crossCheckAnswerability(parsed, retrieved);
                outcome = check.binding === 'not_in_context'
                    ? { kind: 'binding_mismatch', parsed, check }
                    : { kind: 'judged', parsed, check };
                if (outcome.kind === 'binding_mismatch') stats.bindingMismatches++;
                if (!parsed.answerable) stats.abstained++;
                if (outcome.kind === 'binding_mismatch') {
                    // 判官说"可答"，但它指名的条号不在喂给它的检索结果里 —— 它在凭记忆答，不是在读东西。
                    // 这不是"可答"也不是"不可答"，是一次无法核验的判定，所以退回相似度。
                    console.warn(
                        `[judge] binding_mismatch：判官指名 ${parsed.cited?.law}第${parsed.cited?.articleNo}条，` +
                        `但检索结果里没有这一对。原话：${truncate(parsed.reason)}`,
                    );
                }
            }
            store[key] = { at: Date.now(), outcome, usage: spent };
            persist();
            stats.calls++;
            return {
                outcome,
                accounting: accountingFor(outcome, model, {
                    cached: false, latencyMs: Date.now() - started, usage: spent,
                    namedScore: namedScoreFor(outcome, retrieved),
                }),
            };
        } catch (e) {
            lastErr = (e as Error).message;
            // 客户端断开 → 立刻放弃，不重试、不回退
            if (opts.signal?.aborted) throw e;
            if (attempt < MAX_RETRIES) await new Promise(r => setTimeout(r, 1500 * attempt));
        }
    }

    stats.errors++;
    console.warn(`[judge] ${MAX_RETRIES} 次调用都失败，退回相似度判定：${lastErr}`);
    const outcome: JudgeOutcome = { kind: 'error', error: lastErr };
    return {
        outcome,
        accounting: accountingFor(outcome, model, {
            cached: false, latencyMs: Date.now() - started, usage: spent, namedScore: null,
        }),
    };
}
