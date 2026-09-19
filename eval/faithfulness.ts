/**
 * Faithfulness 评测：答案里的每一条法律断言，能不能在**检索到的条文**里找到依据。
 *
 * ## 这个指标和引用校验器不是一回事
 *
 * `CitationVerifier` 查的是**条号**：这个号存不存在、有没有出现在检索结果里。它是确定性的、
 * 免费的、跑在每一次请求上。但它看不见最要命的一类错误——**条号完全正确、条文内容也对，
 * 模型却拿它推出了一句它支持不了的结论**。「第188条规定诉讼时效三年」和
 * 「所以你这笔 2019 年的欠款已经要不回来了」是两件事，引用校验器对后者一言不发。
 *
 * faithfulness 补的就是这一段：把答案拆成断言，逐条问"检索到的条文里写没写这个"。
 *
 * ## 为什么判官要换一个模型
 *
 * 生成用的是 `CHAT_MODEL`（默认 gpt-4o-mini）。判官**默认用 gpt-4o**，不是为了"更准"这个
 * 模糊的理由，而是为了避开**自我偏好**：同一个模型给自己的输出打分，系统性地更宽松。
 * 这会正好把指标推向"看起来很好"，而这是最危险的方向。`JUDGE_MODEL` 可覆盖，
 * `--stability` 会用两个模型各判一遍同一批答案，把自我偏好**量出来**而不是假设它不存在。
 *
 * ## 为什么一定要校准
 *
 * 一个 LLM 判官报出的数字，如果不同时报出"它在已知好/已知坏的样本上表现如何"，
 * 那个数字是没有意义的——它可能只是模型在说"看起来还行"。所以本脚本的产出**永远**包含
 * 两部分：faithfulness 分数，以及判官在**注入缺陷**的样本上的检出率。
 * 分数和它的错误率必须一起出现，否则就是换个地方自欺。
 *
 * 注入的缺陷是**确定性的字符串操作**（不是让模型去改），所以每条变体的标签是已知的、
 * 不依赖任何模型判断。见 `mutate()`。
 *
 * ## 成本
 *
 * 要联网、要花钱：每题一次生成 + 每题一次判定（校准再乘 3～4 倍）。
 * `--limit=N` 先跑子集，`--calibrate` 只对前 N 题做变异校准。
 * 生成的答案会缓存到 `eval/.answers-cache.json`，重跑判分不必重新生成。
 */
import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import OpenAI from 'openai';
import { createRetriever, runQuery, formatContext, CHAT_MODEL, EMBEDDING_MODEL, RETRIEVAL_TOP_K, RETRIEVAL_SCORE_THRESHOLD } from '../src/index.js';
import { RERANK_ENABLED, RERANK_MODEL } from '../src/Reranker.js';
import { loadPrompt } from '../src/prompts.js';
import { loadKnowledge } from '../src/loadKnowledge.js';
import { EVAL_DIR, KNOWLEDGE_DIR } from '../src/paths.js';
import { parseJudgeOutput, mutate, caughtBy, type Claim, type Mutant } from './judgeCore.js';
export type { Claim } from './judgeCore.js';

// ---------------------------------------------------------------------------
// 参数
// ---------------------------------------------------------------------------
const argv = process.argv.slice(2);
const has = (f: string) => argv.includes(f);
const num = (flag: string, fallback: number): number => {
    const raw = argv.find(a => a.startsWith(`${flag}=`));
    if (raw === undefined) return fallback;
    const v = Number(raw.slice(flag.length + 1));
    if (!Number.isFinite(v)) { console.error(`❌ ${flag} 的值不是数字`); process.exit(2); }
    return v;
};

const LIMIT = num('--limit', 0);              // 0 = 全部
const CALIBRATE = has('--calibrate');
const STABILITY = has('--stability');
const VERBOSE = has('--verbose') || has('-v');
const ONLY = argv.find(a => a.startsWith('--id='))?.split('=')[1];

/**
 * 判官模型。默认与生成模型**不同**，理由见文件头。
 * 注意这里不读 CHAT_MODEL——判官跟着生成模型走就正好制造了自我偏好。
 */
const JUDGE_MODEL = process.env.JUDGE_MODEL ?? 'gpt-4o';
/** 用于 --stability：拿生成模型自己也判一遍，量自我偏好有多大 */
const JUDGE_MODEL_B = process.env.JUDGE_MODEL_B ?? CHAT_MODEL;
// 默认 v2。v1 在"条号对、内容不对"这一类上实测漏判 3/3（见 prompts/faithfulness-judge.v2.md
// 的 changelog），没有理由再让默认值停在有已知漏判的那一版上。
const JUDGE_PROMPT_VERSION = Number(process.env.JUDGE_PROMPT_VERSION ?? 2);

const { body: JUDGE_PROMPT, version: judgePromptVersion } = loadPrompt('faithfulness-judge', JUDGE_PROMPT_VERSION);

const client = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    baseURL: process.env.OPENAI_BASE_URL || undefined,
});

// ---------------------------------------------------------------------------
// 判官
// ---------------------------------------------------------------------------
export interface JudgeVerdict {
    claims: Claim[];
    summary: string;
    /** 解析失败时为 true——**绝不能**把解析失败当成"忠实"计入分母 */
    failed: boolean;
    error?: string;
    usage: { promptTokens: number; completionTokens: number };
}

const ZERO_USAGE = { promptTokens: 0, completionTokens: 0 };

/**
 * 判分缓存。键 = 模型 + prompt 版本 + (问题, context, 答案) 的 sha1。
 *
 * 加它的直接原因是**重跑太贵**：一轮 27 题要 27 次判定 + ~108 次校准 + 16 次稳定度，
 * 全是 gpt-4o。更要紧的是**可复现**——`temperature: 0` 只让抖动变小，不让它消失。
 * 没有缓存时，"这次 faithfulness 是 0.94、上次是 0.91"到底是代码改了还是判官抖了，
 * 没人分得清；有了缓存，同样的输入必然拿到同样的判定，两次报告之间的差异一定是代码造成的。
 *
 * 校准用的变体答案也走这个缓存，所以调完判官 prompt 想重跑校准时，只花改过的那部分钱。
 */
const JUDGE_CACHE_PATH = path.join(EVAL_DIR, '.judge-cache.json');
let judgeCache: Record<string, { claims: Claim[]; summary: string; failed: boolean; error?: string; usage: { promptTokens: number; completionTokens: number } }> = {};
try { if (fs.existsSync(JUDGE_CACHE_PATH)) judgeCache = JSON.parse(fs.readFileSync(JUDGE_CACHE_PATH, 'utf-8')); } catch { judgeCache = {}; }
let judgeCacheHits = 0, judgeCacheDirty = false;
/** 本次**实际花掉**的判分 token（缓存命中不计，因为没花钱）。
 *  必须在这里累加而不是从 `judged` 上 reduce：校准和稳定度也会调判官，
 *  只统计主流程会把成本报少一大截——而校准恰恰是最费钱的那部分。 */
let judgeTokensSpent = 0;
function judgeKey(model: string, question: string, context: string, answer: string): string {
    return crypto.createHash('sha1')
        .update(`${model}\u0000v${judgePromptVersion}\u0000${question}\u0000${context}\u0000${answer}`)
        .digest('hex');
}

export async function judge(
    question: string,
    context: string,
    answer: string,
    model: string = JUDGE_MODEL,
): Promise<JudgeVerdict> {
    const key = judgeKey(model, question, context, answer);
    const cached = judgeCache[key];
    if (cached) { judgeCacheHits++; return cached; }

    const user =
        `<question>\n${question}\n</question>\n\n` +
        `<retrieved_context>\n${context}\n</retrieved_context>\n\n` +
        `<answer>\n${answer}\n</answer>`;

    // 瞬时错误重试。一轮完整校准要一百多次调用，**一次 429 就足以在报告里留下
    // 一条"判定失败"并悄悄缩小分母**——而"剔除了却不报"正是这个脚本最想防的事。
    // 所以宁可在这里重试，也不让偶发错误走进汇总。
    let lastErr = '';
    for (let attempt = 1; attempt <= 3; attempt++) {
        try {
            const resp = await client.chat.completions.create({
                model,
                messages: [
                    { role: 'system', content: JUDGE_PROMPT },
                    { role: 'user', content: user },
                ],
                // 判定要可复现：同一份答案判两次必须得到同一个结论，否则
                // "分数变了"到底是答案变了还是判官抖了，没人分得清。
                temperature: 0,
                response_format: { type: 'json_object' },
            });
            const usage = {
                promptTokens: resp.usage?.prompt_tokens ?? 0,
                completionTokens: resp.usage?.completion_tokens ?? 0,
            };
            judgeTokensSpent += usage.promptTokens + usage.completionTokens;
            const parsed = parseJudgeOutput(resp.choices[0]?.message?.content ?? '');
            const out = 'error' in parsed
                ? { claims: [] as Claim[], summary: '', failed: true, error: parsed.error, usage }
                : { ...parsed, failed: false, usage };
            // 解析失败**要**缓存：同一个 prompt 再问一遍大概率还是坏格式，
            // 缓存下来才不会每轮重跑都重新踩一次。它与"网络错误"是两回事。
            judgeCache[key] = out; judgeCacheDirty = true;
            return out;
        } catch (e) {
            lastErr = (e as Error).message;
            if (attempt < 3) await new Promise(r => setTimeout(r, 1500 * attempt));
        }
    }
    // 三次都失败：不缓存（缓存下来会把一次偶发的服务端故障固化进报告）
    return { claims: [], summary: '', failed: true, error: lastErr, usage: ZERO_USAGE };
}

// ---------------------------------------------------------------------------
// 生成（带缓存）
// ---------------------------------------------------------------------------
interface GenRow {
    id: string; question: string; answer: string; context: string;
    abstained: boolean; usage: { promptTokens: number; completionTokens: number };
    citationsTotal: number; citationsVerified: number;
}
const CACHE_PATH = path.join(EVAL_DIR, '.answers-cache.json');

function loadCache(): Record<string, GenRow> {
    if (!fs.existsSync(CACHE_PATH)) return {};
    try { return JSON.parse(fs.readFileSync(CACHE_PATH, 'utf-8')); } catch { return {}; }
}

// ---------------------------------------------------------------------------
// 主流程
// ---------------------------------------------------------------------------
const datasetRaw = fs.readFileSync(path.join(EVAL_DIR, 'dataset.jsonl'), 'utf-8');
const datasetSha1 = crypto.createHash('sha1').update(datasetRaw).digest('hex').slice(0, 12);
let cases = datasetRaw.split('\n').filter(l => l.trim()).map(l => JSON.parse(l));
// 只判域内题：域外题应当拒答，拒答答案没有"法律断言"可判，
// 把它算进 faithfulness 会让分母里混进一堆空 claims，分数虚高。
cases = cases.filter(c => c.category !== 'out_of_domain');
if (ONLY) cases = cases.filter(c => c.id === ONLY);
if (LIMIT > 0) cases = cases.slice(0, LIMIT);

const manifest = JSON.parse(fs.readFileSync(path.join(KNOWLEDGE_DIR, 'CORPUS.json'), 'utf-8'));
const corpus = fs.readdirSync(KNOWLEDGE_DIR).filter(f => f.endsWith('.md'))
    .map(f => fs.readFileSync(path.join(KNOWLEDGE_DIR, f), 'utf-8')).join('\n');

const retriever = createRetriever();
retriever.setStrict(true);
await loadKnowledge(retriever);

const cache = loadCache();
const gens: GenRow[] = [];
for (const c of cases) {
    const hit = cache[c.id];
    if (hit && hit.question === c.question) {
        gens.push(hit);
        process.stdout.write(VERBOSE ? `\n  ${c.id} 用缓存` : '.');
        continue;
    }
    const t0 = Date.now();
    const r = await runQuery({ query: c.question, retriever });
    const row: GenRow = {
        id: c.id, question: c.question, answer: r.answer,
        context: formatContext(r.retrieved),
        abstained: r.abstained, usage: r.usage,
        citationsTotal: r.citations.total, citationsVerified: r.citations.verified,
    };
    cache[c.id] = row;
    gens.push(row);
    process.stdout.write(VERBOSE ? `\n  ${c.id} 生成 ${((Date.now() - t0) / 1000).toFixed(1)}s` : '.');
    fs.writeFileSync(CACHE_PATH, JSON.stringify(cache, null, 2), 'utf-8');
}
if (!VERBOSE) process.stdout.write('\n');

// --- 判分 -------------------------------------------------------------------
type Judged = GenRow & { verdict: JudgeVerdict; faithfulness: number | null };
const judged: Judged[] = [];
for (const g of gens) {
    const v = await judge(g.question, g.context, g.answer);
    const sup = v.claims.filter(c => c.verdict === 'supported').length;
    judged.push({
        ...g, verdict: v,
        // 没有任何 claim 时是 null 而不是 1.0：一个什么都没说的答案不该算"完全忠实"。
        faithfulness: v.failed || v.claims.length === 0 ? null : sup / v.claims.length,
    });
    process.stdout.write(VERBOSE ? `\n  ${g.id} → ${v.failed ? '判定失败' : `${sup}/${v.claims.length}`}` : '.');
}
if (!VERBOSE) process.stdout.write('\n');

// --- 校准：注入缺陷，看判官抓不抓得到 ---------------------------------------
interface CalibRow {
    id: string; kind: string; note: string; needle: string; caught: boolean; claims: number;
    /** 判官点出的 unsupported 里，有没有一条**提到被注入的标志物**。
     *  只作诊断用，不参与判定：字符串匹配被证明不可靠（判官的摘录天然会截断条号，
     *  数字写法也可能不同）。但"抓到了"远多于"点到了标志物"时，说明判官是在
     *  拿别的理由判 unsupported，那批样本需要人工看一眼——这是区分
     *  "判官真看见了注入"和"判官碰巧在别处挑刺"的唯一线索。 */
    needleMatched: boolean;
    /** 没抓到时判官究竟说了什么。**没有这个，"判官漏判"和"我造错了样本"分不开**——
     *  首轮冒烟测试就是栽在这里：3 条 article_swap 全报"没抓到"，而其中一条注入的
     *  根本不是缺陷（换过去的那条恰好也支持该断言）。看不到判官的原话就无从判断。 */
    missedClaims?: { text: string; verdict: string; evidence: string }[];
}
const calib: CalibRow[] = [];

// 校准的基底必须是**原本完全忠实**的答案。判据（见 judgeCore.caughtBy）是
// "变体里出现了 unsupported 断言"——只有基底本来是 0 条 unsupported，这个推理才成立：
// 变体里的 unsupported 只能来自注入。基底本身就不干净的话，原本就有的问题会被记成
// "抓到了"，检出率虚高。带 unsupported 的题**显式列出来**跳过，不静默剔除。
const cleanBase = judged.filter(j => !j.verdict.failed && j.verdict.claims.length > 0
    && j.verdict.claims.every(c => c.verdict === 'supported'));
const dirtyBase = judged.filter(j => !j.verdict.failed && j.verdict.claims.length > 0
    && j.verdict.claims.some(c => c.verdict === 'unsupported'));

if (CALIBRATE) {
    console.log('\n=== 校准：注入已知缺陷 ===');
    console.log(`基底：${cleanBase.length} 题完全忠实` +
        (dirtyBase.length ? `，跳过 ${dirtyBase.length} 题（原始答案已含 unsupported：${dirtyBase.map(j => j.id).join(' ')}）` : ''));
    for (const g of cleanBase) {
        for (const m of mutate(g.answer, g.context, corpus)) {
            const v = await judge(g.question, g.context, m.answer);
            // "抓到"= 变体里出现了 unsupported 断言（基底是干净的，所以它只能来自注入）。
            // **刻意不用** faithfulness 分数下降来判断：追加一条坏断言会让分母变大，
            // 分数完全可能不降反升，用分数判断会把"抓到了"误判成"没抓到"。
            const caught = caughtBy(v.claims);
            const unsup = v.claims.filter(c => c.verdict === 'unsupported');
            const row: CalibRow = {
                id: g.id, kind: m.kind, note: m.note, needle: m.needle,
                caught,
                needleMatched: unsup.some(c => c.text.includes(m.needle) || c.evidence.includes(m.needle)),
                claims: v.claims.length,
            };
            if (!caught) row.missedClaims = v.claims.map(c => ({ text: c.text, verdict: c.verdict, evidence: c.evidence }));
            calib.push(row);
            console.log(`  ${caught ? '✅' : '❌'} ${g.id.padEnd(11)} ${m.kind.padEnd(19)} ${m.note}`);
            if (!caught) {
                console.log(v.failed ? `        ↳ 判官解析失败：${v.error ?? ''}`
                    : `        ↳ 判官判为全 supported（${v.claims.length} 条断言）——需要人工判断是漏判还是样本本来就没缺陷`);
            } else if (!row.needleMatched) {
                console.log(`        ↳ 判 unsupported 但没点到标志物「${m.needle}」，判官的理由是别处的断言：`);
                for (const c of unsup.slice(0, 3)) console.log(`           · ${c.text}｜${c.evidence}`);
            }
        }
    }
}

// --- 稳定度：同一个答案判两次，以及换一个模型判 -----------------------------
/**
 * 稳定度。
 *
 * **比的是 faithfulness 分数，不是"断言数组是否逐位相同"。** 上一版用后者，
 * 于是判官把同一句话切成 2 条还是 3 条断言，都会被记成"不一致"——
 * 那量的是**断句习惯**，不是判定稳定性，两件事完全不同。而在冒烟测试里它报出
 * "gpt-4o-mini 一致 0/3"，看着像自我偏好巨大，实际很可能只是断句不同。
 *
 * 真正要回答的是：**换一个判官，分数会动多少？** 所以逐题算出两个模型各自的
 * faithfulness，报差值；另外单独报断言条数的差，让人能看出"分数没动但断句变了"
 * 这种情况。自我偏好如果存在，表现为**同一个模型的分数系统性偏高**。
 */
interface StabRow { id: string; faithA: number; faithB: number; claimsA: number; claimsB: number }
let stability: {
    repeats: number; model: string; rows: StabRow[];
    meanAbsDelta: number; modelBFavors: number; modelAFavors: number; identical: number;
    meanClaimDelta: number;
} | null = null;
if (STABILITY) {
    console.log('\n=== 稳定度（判官换成自己人会怎样）===');
    const faith = (v: JudgeVerdict): number => {
        if (v.failed || v.claims.length === 0) return NaN;   // 与主流程一致：无断言不是"完全忠实"
        return v.claims.filter(c => c.verdict === 'supported').length / v.claims.length;
    };
    const sample = gens.slice(0, Math.min(8, gens.length));
    const rows: StabRow[] = [];
    for (const g of sample) {
        const v1 = judged.find(x => x.id === g.id)!.verdict;
        const vB = await judge(g.question, g.context, g.answer, JUDGE_MODEL_B);
        const a = faith(v1), b = faith(vB);
        rows.push({ id: g.id, faithA: a, faithB: b, claimsA: v1.claims.length, claimsB: vB.claims.length });
        const fmt = (x: number) => Number.isNaN(x) ? '  n/a' : x.toFixed(3);
        console.log(`  ${g.id.padEnd(11)} ${JUDGE_MODEL.padEnd(10)} ${fmt(a)}   ${JUDGE_MODEL_B.padEnd(12)} ${fmt(b)}` +
            `   断言 ${v1.claims.length} vs ${vB.claims.length}`);
    }
    const valid = rows.filter(r => !Number.isNaN(r.faithA) && !Number.isNaN(r.faithB));
    const deltas = valid.map(r => r.faithB - r.faithA);
    const mean = (xs: number[]) => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
    stability = {
        repeats: sample.length, model: JUDGE_MODEL_B, rows,
        meanAbsDelta: mean(deltas.map(Math.abs)),
        // 正 = 换 JUDGE_MODEL_B 后分数更高 = 自己人给自己人打分更宽松
        modelBFavors: deltas.filter(d => d > 0).length,
        modelAFavors: deltas.filter(d => d < 0).length,
        identical: deltas.filter(d => d === 0).length,
        meanClaimDelta: mean(rows.map(r => r.claimsB - r.claimsA)),
    };
}

// --- 汇总 -------------------------------------------------------------------
const ok = judged.filter(j => !j.verdict.failed && j.verdict.claims.length > 0);
const totalClaims = ok.reduce((a, j) => a + j.verdict.claims.length, 0);
const totalSup = ok.reduce((a, j) => a + j.verdict.claims.filter(c => c.verdict === 'supported').length, 0);
const failedN = judged.filter(j => j.verdict.failed).length;
const emptyN = judged.filter(j => !j.verdict.failed && j.verdict.claims.length === 0).length;

const judgeTokens = judgeTokensSpent;
const genTokens = gens.reduce((a, g) => a + g.usage.promptTokens + g.usage.completionTokens, 0);

console.log(`\n=== Faithfulness（判官 ${JUDGE_MODEL}，prompt v${judgePromptVersion}）===`);
console.log(`判定成功 ${ok.length}/${judged.length} 题` +
    (failedN ? `，**判定失败 ${failedN} 题（已从分母剔除）**` : '') +
    (emptyN ? `，无断言 ${emptyN} 题` : ''));
console.log(`断言合计 ${totalClaims} 条：supported ${totalSup} / unsupported ${totalClaims - totalSup}`);
console.log(`**faithfulness = ${totalClaims ? (totalSup / totalClaims).toFixed(3) : '—'}**`);
console.log(`token：生成 ${genTokens}，判定 ${judgeTokens}` +
    (judgeCacheHits ? `（其中 ${judgeCacheHits} 次判定命中缓存，未重复计费）` : ''));
if (judgeCacheDirty) fs.writeFileSync(JUDGE_CACHE_PATH, JSON.stringify(judgeCache), 'utf-8');

const worst = ok.filter(j => j.faithfulness !== null && j.faithfulness < 1)
    .sort((a, b) => (a.faithfulness! - b.faithfulness!));
if (worst.length > 0) {
    console.log(`\n有 unsupported 断言的 ${worst.length} 题：`);
    for (const j of worst) {
        console.log(`  ${j.id} [${j.faithfulness!.toFixed(2)}] ${j.question}`);
        for (const c of j.verdict.claims.filter(c => c.verdict === 'unsupported')) {
            console.log(`      ✗ ${c.text}${c.evidence && c.evidence !== '无' ? `（引 ${c.evidence}）` : ''}`);
        }
    }
}

if (CALIBRATE) {
    const caught = calib.filter(c => c.caught).length;
    console.log(`\n=== 校准结果：判官检出率 ===`);
    // **零样本的类型必须显式列出。** 上一版只打印出现过的类型，
    // 于是 article_swap 一条都没生成时，汇总显示"7/7 = 1.000"——
    // 看起来四类注入全部检出，实际上有一类根本没测。
    // 这正是这个脚本要防的"静默剔除"，只不过发生在自己身上。
    const ALL_KINDS = ['fabricated_cite', 'distorted_number', 'article_swap', 'unsupported_append'];
    const present = [...new Set(calib.map(c => c.kind))];
    const absent = ALL_KINDS.filter(k => !present.includes(k));
    for (const k of ALL_KINDS) {
        const sub = calib.filter(c => c.kind === k);
        if (sub.length === 0) { console.log(`  ${k.padEnd(19)} ——  **本轮一条都没生成，这一类未被测量**`); continue; }
        console.log(`  ${k.padEnd(19)} ${sub.filter(c => c.caught).length}/${sub.length}`);
    }
    console.log(`  ${'合计'.padEnd(19)} ${caught}/${calib.length} = ${calib.length ? (caught / calib.length).toFixed(3) : '—'}`);
    const matched = calib.filter(c => c.needleMatched).length;
    if (caught > 0 && matched < caught) {
        console.log(`  其中点到注入标志物的 ${matched}/${caught}——差额部分是判官在别处挑的刺，建议抽查。`);
    }
    if (absent.length) {
        console.log(`\n  ⚠ 未测量的注入类型：${absent.join(', ')}——**不要**把它读成"判官在这一类上表现良好"，`);
        console.log(`    它只是没被测到。原因通常是数据不满足注入前提（如单法源 context、无可换的数目词）。`);
    }
    console.log(`\n  ⚠ 上表的 faithfulness 必须和这个检出率一起读：`);
    console.log(`    判官漏掉 ${calib.length - caught} 条已知缺陷，意味着真实 faithfulness 低于所报数字。`);
}
if (stability) {
    console.log(`\n稳定度：${stability.repeats} 题，${stability.model} 相对 ${JUDGE_MODEL} ` +
        `分数相同 ${stability.identical}、偏高 ${stability.modelBFavors}、偏低 ${stability.modelAFavors}；` +
        `平均绝对差 ${stability.meanAbsDelta.toFixed(3)}，平均断言条数差 ${stability.meanClaimDelta.toFixed(2)}`);
    console.log(`  ⚠ "分数相同"是按 faithfulness 比值算的，与断句无关。` +
        `断句本身会变（见断言条数差），但那是判官的表达习惯，不是判定结论。`);
}

const report = {
    generatedAt: new Date().toISOString(),
    datasetSha1,
    corpus: { version: manifest.version, hash: manifest.corpusHash },
    config: {
        generatorModel: CHAT_MODEL,
        judgeModel: JUDGE_MODEL,
        judgePromptVersion,
        embeddingModel: EMBEDDING_MODEL,
        rerank: RERANK_ENABLED ? RERANK_MODEL : null,
        topK: RETRIEVAL_TOP_K,
        threshold: RETRIEVAL_SCORE_THRESHOLD,
    },
    summary: {
        judged: judged.length, failedJudge: failedN, noClaims: emptyN,
        claims: totalClaims, supported: totalSup,
        faithfulness: totalClaims ? Number((totalSup / totalClaims).toFixed(4)) : null,
        tokens: { generate: genTokens, judge: judgeTokens },
        judgeCacheHits,
    },
    calibration: calib.length ? {
        base: { clean: cleanBase.length, skippedDirty: dirtyBase.map(j => j.id) },
        total: calib.length,
        caught: calib.filter(c => c.caught).length,
        byKind: Object.fromEntries([...new Set(calib.map(c => c.kind))].map(k => {
            const sub = calib.filter(c => c.kind === k);
            return [k, { caught: sub.filter(c => c.caught).length, n: sub.length }];
        })),
        rows: calib,
    } : null,
    stability,
    rows: judged.map(j => ({
        id: j.id, question: j.question, abstained: j.abstained,
        faithfulness: j.faithfulness,
        citations: { total: j.citationsTotal, verified: j.citationsVerified },
        judgeFailed: j.verdict.failed,
        judgeError: j.verdict.error ?? null,
        summary: j.verdict.summary,
        claims: j.verdict.claims,
        answer: j.answer,
    })),
};
fs.writeFileSync(path.join(EVAL_DIR, 'faithfulness-report.json'), JSON.stringify(report, null, 2), 'utf-8');
console.log(`\n完整结果 → eval/faithfulness-report.json`);
