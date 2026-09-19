/**
 * 拒答评测：`pnpm eval:abstention`。**花钱**（每题一次 LLM 判定）。
 *
 * ## 它和 `pnpm eval:signals` 的分工
 *
 * `abstentionSignals.ts` 是**免费的、离线的、确定性的**：它只量相似度信号的可分性，
 * 从不调模型，所以能随时重跑、也能进 CI。这个脚本要花钱、有缓存、有一整套逐题失败分类，
 * 混在一起会让 `pnpm eval:signals` 失去"随时能跑"这个属性。
 *
 * 它和 `pnpm eval` 分工也类似：那个测检索（确定性、进门禁），这个测拒答（随机、不进）。
 *
 * ## 为什么**不**调 `runQuery`
 *
 * `runQuery` 会连生成一起跑，等于每道题多付一次整篇回答的钱，而这一轮要量的是
 * **判定**不是**生成**。所以这里走 `retrieve() → decideAbstention → judgeAnswerability`，
 * 与 `runQuery` 共用同一个 `createRetriever()`（必须——评测要测的就是线上那条路）。
 *
 * ## 反事实：为什么 `sim_abstain` 在线上跳过、在这里不跳过
 *
 * 线上 `shouldSkipJudge` 有一条：相似度已经判拒答时不再问判官（省一次调用）。
 * 但**这一轮要回答的问题正是"那条跳过规则对不对"**——所以评测必须把它变成反事实：
 * 硬跳过（`no_hits` / `exact_article_hit`）之外一律跑判官，然后在报告里同时算两种策略：
 *
 *   - `policy=sim`：相似度的拒答胜出（= 线上默认行为）
 *   - `policy=judge`：判官说了算（= 如果去掉那条跳过规则会怎样）
 *
 * 两个数字都印在同一张表里。这不是"多跑一遍"，是**同一次运行的两列读数**。
 */
import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { createRetriever, EMBEDDING_MODEL, RETRIEVAL_TOP_K, RETRIEVAL_SCORE_THRESHOLD } from '../src/index.js';
import Reranker, { RERANK_ENABLED, RERANK_MODEL, RERANK_CANDIDATES } from '../src/Reranker.js';
import { loadKnowledge } from '../src/loadKnowledge.js';
import { decideAbstention, shouldSkipJudge } from '../src/abstention.js';
import type { JudgeSkipReason } from '../src/abstention.js';
import {
    judgeAnswerability,
    flushJudgeCache,
    ANSWERABILITY_JUDGE_MODEL,
    ANSWERABILITY_JUDGE_PROMPT_VERSION,
} from '../src/AnswerabilityJudge.js';
import type { JudgeAccounting } from '../src/AnswerabilityJudge.js';
import { auc, coverage, scanThresholds } from './abstentionCore.js';
import { EVAL_DIR, CACHE_DIR } from '../src/paths.js';

// ---------------------------------------------------------------------------
// 参数
// ---------------------------------------------------------------------------
const argv = process.argv.slice(2);
const has = (f: string) => argv.includes(f);
/** 同 eval/run.ts：取值写错就退出，不静默 fallback —— 一个写错的开关比没有开关更危险 */
const num = (flag: string, fallback: number): number => {
    const raw = argv.find(a => a.startsWith(`${flag}=`));
    if (raw === undefined) return fallback;
    const v = Number(raw.slice(flag.length + 1));
    if (!Number.isFinite(v)) {
        console.error(`❌ ${flag} 的值不是数字：${raw.slice(flag.length + 1)}`);
        process.exit(2);
    }
    return v;
};

const TOP_K = num('--top-k', RETRIEVAL_TOP_K);
const THRESHOLD = num('--threshold', RETRIEVAL_SCORE_THRESHOLD);
const LIMIT = num('--limit', Number.POSITIVE_INFINITY);
const VERBOSE = has('--verbose') || has('-v');
const NO_CACHE = has('--no-cache');
const ONLY_IDS = argv.find(a => a.startsWith('--id='))?.slice('--id='.length).split(',').filter(Boolean);
const MODEL_B = argv.find(a => a.startsWith('--judge-model-b='))?.slice('--judge-model-b='.length);
const POLICY = (argv.find(a => a.startsWith('--policy='))?.split('=')[1] ?? 'both') as 'judge' | 'sim' | 'both';
if (!['judge', 'sim', 'both'].includes(POLICY)) {
    console.error(`❌ --policy 只能是 judge | sim | both，收到：${POLICY}`);
    process.exit(2);
}
const POLICIES: ('judge' | 'sim')[] = POLICY === 'both' ? ['sim', 'judge'] : [POLICY];
const NO_SIGNALS = has('--no-signals');

const CACHE_PATH = path.join(CACHE_DIR, 'answerability-judge.json');
if (NO_CACHE) {
    // 只删这一份缓存。不做成"忽略缓存"是因为那需要给判官模块加一条只服务于评测的代码路径，
    // 而缓存本来就是可再生的中间态——删掉它没有任何不可逆的后果。
    if (fs.existsSync(CACHE_PATH)) fs.unlinkSync(CACHE_PATH);
    console.log('⚠ --no-cache：已删除判官缓存，本轮全部重新调用（会真的花钱）\n');
}

// ---------------------------------------------------------------------------
// 数据集
// ---------------------------------------------------------------------------
interface Expected { source: string; articleNo: number; snippet?: string }
interface Case {
    id: string; category: string; question: string;
    expectAbstain: boolean; expected?: Expected[];
    law?: string; keyTerms?: string[]; oodKind?: string;
}

const datasetRaw = fs.readFileSync(path.join(EVAL_DIR, 'dataset.jsonl'), 'utf-8');
const datasetSha1 = crypto.createHash('sha1').update(datasetRaw).digest('hex').slice(0, 12);
let cases: Case[] = datasetRaw.split('\n').filter(l => l.trim()).map(l => JSON.parse(l));

// 域外题必须带 oodKind，否则分层统计会**静默**漏掉它（Phase 0 之前 ood-01..04 就是这样：
// 汇总照常打印，分层里少 4 条，没有任何东西会说）。同一条守卫在 abstentionSignals.ts 里也有。
{
    const missing = cases.filter(c => c.category === 'out_of_domain' && !c.oodKind);
    if (missing.length > 0) {
        console.error(`✗ 数据集里有 ${missing.length} 道域外题没写 oodKind：${missing.map(c => c.id).join(' ')}`);
        console.error('  分层统计会静默漏掉它们。请在 eval/dataset.jsonl 里补上再跑。');
        process.exit(2);
    }
}

if (ONLY_IDS) cases = cases.filter(c => ONLY_IDS.includes(c.id));
if (LIMIT !== Number.POSITIVE_INFINITY) cases = cases.slice(0, LIMIT);
if (cases.length === 0) {
    console.error('❌ 筛选后没有题目可跑（检查 --id / --limit）');
    process.exit(2);
}

// ---------------------------------------------------------------------------
// 检索
// ---------------------------------------------------------------------------
const retriever = createRetriever();
retriever.setStrict(true);
await loadKnowledge(retriever);

// 重排分只作对照信号。加载失败就跳过这一行——诊断脚本不该因为一个模型下不下来就什么都报不出来。
// （HuggingFace 需要 VPN；这条降级路径让没 VPN 时评测照样能跑，只是少一列对照。）
let reranker: Reranker | null = null;
if (!NO_SIGNALS) {
    try {
        reranker = new Reranker();
        await reranker.rerank('预热', [{ embedText: '预热' }]);
    } catch (e) {
        console.warn(`⚠ 重排模型不可用（${(e as Error).message}），对照表里跳过该行\n`);
        reranker = null;
    }
}

// ---------------------------------------------------------------------------
// 逐题
// ---------------------------------------------------------------------------
type Kind = 'in_domain' | 'legal_adjacent' | 'unrelated' | 'unclassified';

interface Row {
    id: string; category: string; kind: Kind; question: string;
    expectAbstain: boolean;
    simAbstained: boolean; simTopScore: number; simReason: string;
    /** 硬跳过（no_hits / exact_article_hit）时判官根本没跑 */
    hardSkipReason: JudgeSkipReason | null;
    judge: JudgeAccounting;
    judgeB: JudgeAccounting | null;
    cov: number;
    rerank: number | null;
    expected: Expected[];
}

const rows: Row[] = [];
console.log(`判官 ${ANSWERABILITY_JUDGE_MODEL} · prompt v${ANSWERABILITY_JUDGE_PROMPT_VERSION} · ` +
    `策略 ${POLICY}${MODEL_B ? ` · 对照判官 ${MODEL_B}` : ''} · ${cases.length} 题\n`);

for (const c of cases) {
    const retrieved = await retriever.retrieve(c.question, TOP_K);
    const sim = decideAbstention(retrieved, THRESHOLD);
    const skip = shouldSkipJudge(retrieved, sim);
    // 只有硬跳过才真的不问判官。`sim_abstain` 在线上会跳过，但这里**必须问**——
    // 它正是这一轮要量其真伪的那条规则。
    const hardSkip = skip.skip && skip.reason !== 'sim_abstain';

    // policy=sim 时相似度的拒答胜出，那题就不必花钱问判官；其余策略（含 both 的反事实）都要问。
    const askJudge = !hardSkip && (POLICY !== 'sim' || !sim.abstained);
    let judge: JudgeAccounting;
    let judgeB: JudgeAccounting | null = null;

    if (!askJudge) {
        // 没问就是没问。answerable 留 null 而不是 false —— "没判定"绝不能被读成"判为不可答"。
        judge = {
            outcome: 'skipped', skipReason: hardSkip ? skip.reason : 'sim_abstain',
            answerable: null, cited: null, binding: null, quoteVerified: null, namedScore: null, reason: null,
            cached: false, latencyMs: 0, usage: { promptTokens: 0, completionTokens: 0 },
            model: ANSWERABILITY_JUDGE_MODEL, promptVersion: ANSWERABILITY_JUDGE_PROMPT_VERSION,
        };
    } else {
        ({ accounting: judge } = await judgeAnswerability(c.question, retrieved));
        if (MODEL_B) ({ accounting: judgeB } = await judgeAnswerability(c.question, retrieved, { model: MODEL_B }));
    }

    let rerankScore: number | null = null;
    if (reranker) {
        if (retrieved.some(x => x.exactArticleHit)) rerankScore = Number.POSITIVE_INFINITY;
        else {
            const out = await reranker.rerank(c.question, retrieved);
            rerankScore = Math.max(...out.map(x => x.rerankScore));
        }
    }

    // `?? 'unclassified'` 必须在 `as` **之前**：写成 `c.oodKind as Kind ?? 'unclassified'`
    // 是对类型系统撒谎（oodKind 可能是 undefined），而上面那道守卫就是为了让它不可能缺。
    const kind: Kind = c.category === 'out_of_domain'
        ? ((c.oodKind ?? 'unclassified') as Kind)
        : 'in_domain';

    rows.push({
        id: c.id, category: c.category, kind, question: c.question,
        expectAbstain: c.expectAbstain,
        simAbstained: sim.abstained, simTopScore: Number(sim.topScore.toFixed(4)), simReason: sim.reason,
        hardSkipReason: hardSkip ? skip.reason : null,
        judge, judgeB,
        cov: coverage(c.question, retrieved.map(x => x.document)),
        rerank: rerankScore,
        expected: c.expected ?? [],
    });

    const mark = judge.outcome === 'judged' ? (judge.answerable ? '可答' : '不可答') : judge.outcome;
    process.stdout.write(VERBOSE
        ? `  ${c.id.padEnd(10)} sim=${sim.abstained ? '拒' : '答'} 判官=${mark}` +
          `${judge.cited ? ` (${judge.cited.law}第${judge.cited.articleNo}条)` : ''}` +
          `${judge.cached ? ' [缓存]' : ''} ${judge.latencyMs}ms\n`
        : '.');
}
if (!VERBOSE) process.stdout.write('\n');
flushJudgeCache();

// ---------------------------------------------------------------------------
// 策略：把"判官判定"和"相似度判定"组合成两个真实可上线的操作点
// ---------------------------------------------------------------------------

/** 判官没给出可用判定（error / parse_failure / binding_mismatch / 跳过）→ 退回相似度 */
function decideWithJudge(row: Row, acct: JudgeAccounting, policy: 'judge' | 'sim'): boolean {
    if (row.hardSkipReason) return row.simAbstained;          // 判官根本没跑，相似度说了算
    if (policy === 'sim' && row.simAbstained) return true;    // 相似度的拒答胜出
    if (acct.outcome === 'judged' && acct.answerable !== null) return !acct.answerable;
    return row.simAbstained;                                   // 判官没判定 → 退回已测量的现状
}

const dom = rows.filter(r => r.kind === 'in_domain');
const ood = rows.filter(r => r.kind !== 'in_domain');
const stratify = (rs: Row[], k: Kind) => rs.filter(r => r.kind === k);

/** 显示宽度：中文按 2 列算，否则表格会歪 */
function padDisp(s: string, w: number): string {
    let width = 0;
    for (const ch of s) width += /[一-鿿　-〿＀-￯]/.test(ch) ? 2 : 1;
    return s + ' '.repeat(Math.max(0, w - width));
}

console.log(`\n=== 拒答信号对照（域内 ${dom.length} / 域外 ${ood.length}；域内=正类，AUC 0.5 = 抛硬币）===`);
console.log(`${padDisp('信号', 30)}${padDisp('零误拒能拦', 14)}${padDisp('全拦需误拒', 12)}` +
    `${padDisp('AUC(全部域外)', 16)}${padDisp('vs legal_adjacent', 19)}vs unrelated`);

/** 一个连续信号的三个读数。越小越该拒答。 */
function signalRow(name: string, get: (r: Row) => number | null): void {
    const usable = rows.filter(r => get(r) !== null);
    if (usable.length < rows.length) {
        console.log(`${padDisp(name, 30)}（有 ${rows.length - usable.length} 题取不到该信号，跳过）`);
        return;
    }
    const o = usable.filter(r => r.kind !== 'in_domain').map(r => get(r) as number);
    const d = usable.filter(r => r.kind === 'in_domain').map(r => get(r) as number);
    const scan = scanThresholds(o, d);
    const la = stratify(usable, 'legal_adjacent').map(r => get(r) as number);
    const un = stratify(usable, 'unrelated').map(r => get(r) as number);
    console.log(
        `${padDisp(name, 30)}` +
        `${padDisp(`${scan.zeroFalseRefusal?.caught ?? 0}/${o.length}`, 14)}` +
        `${padDisp(`${scan.allOod?.falseRefusals ?? 0}/${d.length}`, 12)}` +
        `${padDisp(auc(d, o).toFixed(3), 16)}` +
        `${padDisp(la.length ? `${auc(d, la).toFixed(3)} (n=${la.length})` : '—', 19)}` +
        `${un.length ? `${auc(d, un).toFixed(3)} (n=${un.length})` : '—'}`
    );
}

signalRow('top1 余弦', r => r.simTopScore);
signalRow('查询 bigram 覆盖率', r => r.cov);
if (reranker) signalRow('cross-encoder 重排分', r => r.rerank);

console.log('─'.repeat(96));

/**
 * 判官（二元信号）在每个策略下的一行。
 *
 * ⚠️ 这个 AUC **完全由同一行左边那两个错误率决定**，不携带额外信息：
 * 二元信号的 AUC ≡ 0.5 + (TPR − FPR)/2（Youden's J 的重标度），而 TPR/FPR 就是
 * 「域内答对率」和「域外答错率」。连续信号的 AUC 描述整条 ROC，二元信号只有一个操作点，
 * 所以**不要把这两行并排比较**。要比较，请比较错误率本身。
 * `src/tests/abstentionCore.test.ts` 把这条代数钉成了测试。
 */
function judgeRow(label: string, policy: 'judge' | 'sim', acctOf: (r: Row) => JudgeAccounting): void {
    const oodAbstained = (r: Row) => decideWithJudge(r, acctOf(r), policy);
    const caught = ood.filter(oodAbstained).length;
    const falseRefusals = dom.filter(oodAbstained).length;
    const posScore = dom.map(r => (oodAbstained(r) ? 0 : 1));   // 1 = 回答
    const negScore = ood.map(r => (oodAbstained(r) ? 0 : 1));
    const la = stratify(ood, 'legal_adjacent');
    const un = stratify(ood, 'unrelated');
    const laCaught = la.filter(oodAbstained).length;
    const unCaught = un.filter(oodAbstained).length;
    console.log(
        `${padDisp(label, 30)}` +
        `${padDisp(`${caught}/${ood.length}`, 14)}` +
        `${padDisp(`${falseRefusals}/${dom.length}`, 12)}` +
        `${padDisp(`${auc(posScore, negScore).toFixed(3)}*`, 16)}` +
        `${padDisp(la.length ? `${(laCaught / la.length).toFixed(3)} (n=${la.length})` : '—', 19)}` +
        `${un.length ? `${(unCaught / un.length).toFixed(3)} (n=${un.length})` : '—'}`
    );
}

for (const p of POLICIES) judgeRow(`LLM 可答性判官 [policy=${p}]`, p, r => r.judge);
if (MODEL_B) judgeRow(`  同上，判官换 ${MODEL_B}`, 'judge', r => r.judgeB ?? r.judge);

console.log(`\n  * 二元信号的 AUC ≡ 0.5 + (TPR − FPR)/2，TPR、FPR 就是左边两列的错误率，`);
console.log(`    所以这一列**完全由已打印的错误率决定，不携带额外信息**。`);
console.log(`    连续信号的 AUC 描述整条 ROC，判官只有一个操作点 —— 不要横向比较这两类行。`);

// ---------------------------------------------------------------------------
// 两个方向的错误率，并排 + 分层
// ---------------------------------------------------------------------------
for (const p of POLICIES) {
    const decided = (r: Row) => decideWithJudge(r, r.judge, p);
    console.log(`\n=== policy=${p}：两个方向的错误率（必须一起读）===`);
    console.log(`  域外被正确拒答（漏答越少越好）：${ood.filter(decided).length}/${ood.length}` +
        ` = ${(ood.filter(decided).length / ood.length).toFixed(3)}`);
    console.log(`  域内被误拒（误拒越少越好）    ：${dom.filter(decided).length}/${dom.length}` +
        ` = ${(dom.filter(decided).length / dom.length).toFixed(3)}` +
        (dom.filter(decided).length === 0 ? '   ← 零误拒' : ''));

    console.log('  按 oodKind 分层：');
    for (const k of ['legal_adjacent', 'unrelated', 'unclassified'] as Kind[]) {
        const s = stratify(ood, k);
        if (s.length === 0) continue;
        const caught = s.filter(decided).length;
        const warn = s.length < 10 ? '   ← n 太小，不要据此下结论' : '';
        console.log(`    ${k.padEnd(16)} 拒答 ${caught}/${s.length} = ${(caught / s.length).toFixed(3)} (n=${s.length})${warn}`);
    }
}

// ---------------------------------------------------------------------------
// 判官之后再叠一道"检索分门"：`answerable && namedScore < t` → 拒答
//
// 这是计划里"整条曲线"的第二个维度。判官是二元的、只有一个操作点，曲线只能从
// 别的可配置维度上来；这个维度扫的是一个**测量量**（检索分），不是模型的自我报告，
// 所以它不会引入未经校准的置信度（那是 §10.13 明确警告过的动作）。
//
// 它回答的问题：**在被证明不可校准的相似度信号之上，判官之后，相似度还有没有增量信息。**
// ---------------------------------------------------------------------------
const accepted = rows.filter(r => r.judge.outcome === 'judged' && r.judge.answerable === true);
const scored = accepted.filter(r => r.judge.namedScore !== null);
const gate = (r: Row, t: number): boolean => {
    if (decideWithJudge(r, r.judge, 'judge')) return true;
    const ns = r.judge.namedScore;
    return ns !== null && ns < t;
};

console.log(`\n=== 判官之后再叠一道检索分门（answerable && 指名条文的检索分 < t → 拒答）===`);
console.log(`  判官给了"可答"且有指名分的题：${scored.length} 条，` +
    `namedScore ${Math.min(...scored.map(r => r.judge.namedScore!)).toFixed(4)}` +
    ` … ${Math.max(...scored.map(r => r.judge.namedScore!)).toFixed(4)}`);
// 这一行是整个扫描为什么退化的原因，必须印出来，否则下面的表看着像"没扫出东西"。
console.log(`  域外一侧可答判定：${ood.filter(r => r.judge.answerable === true).length} 条` +
    ` —— 门只作用在"判官说可答"上，所以 t 抬高**只可能**误拒域内，不可能多拦域外。`);

const cands: number[] = [Number.NEGATIVE_INFINITY];
for (const v of [...new Set(scored.map(r => r.judge.namedScore!))].sort((a, b) => a - b)) cands.push(v);
console.log(`  ${padDisp('t', 12)}${padDisp('域外拒答', 12)}域内误拒`);
let shown = 0;
for (const t of cands) {
    if (shown >= 12) { console.log(`  …（其余 ${cands.length - shown} 个候选阈值全都不优于上一行，略）`); break; }
    const caught = ood.filter(r => gate(r, t)).length;
    const bad = dom.filter(r => gate(r, t)).length;
    console.log(`  ${padDisp(t === Number.NEGATIVE_INFINITY ? '-∞（纯判官）' : t.toFixed(4), 12)}` +
        `${padDisp(`${caught}/${ood.length}`, 12)}${bad}/${dom.length}`);
    shown++;
}
console.log(`  → 表里没有一行优于 t=-∞。**这道门在本轮数据上被纯判官支配**：它唯一能做的事是把`);
console.log(`    已经答对的域内题改成误拒。而它能防的那个失败模式（判官认可一条低分条文）本轮一次都没出现，`);
console.log(`    何况 §10.8 量到的唯一一个"骗过相似度"的样本 ood-11 拿的是**全库最高分**——`);
console.log(`    真正需要防的方向是高分为假，而这门判的是低分为假。**方向就是反的。**`);

// ---------------------------------------------------------------------------
// 两个被点名的硬样本：必须逐条看，而且要**看判官原话**
// ---------------------------------------------------------------------------
const HARD = [
    { id: 'sem-02', wantAbstain: false, why: '口语化提问，正确条文分数全库最低 —— 绝不能被误拒' },
    { id: 'ood-11', wantAbstain: true, why: '逐字命中民法典第181条，但它分配民事责任、问题问刑事责任 —— 必须被拒' },
];
console.log(`\n=== 两个硬样本（判定对了才算，理由也要人看）===`);
for (const h of HARD) {
    const r = rows.find(x => x.id === h.id);
    if (!r) { console.log(`  ${h.id}：本轮未跑到`); continue; }
    const abstained = decideWithJudge(r, r.judge, POLICIES[POLICIES.length - 1]);
    const ok = abstained === h.wantAbstain;
    console.log(`  ${ok ? '✅' : '❌'} ${h.id}  ${r.question}`);
    console.log(`     ${h.why}`);
    console.log(`     相似度：${r.simAbstained ? '拒答' : '回答'}（最高分 ${r.simTopScore}）` +
        `  判官：${r.judge.outcome}${r.judge.answerable === null ? '' : r.judge.answerable ? ' 可答' : ' 不可答'}` +
        `${r.judge.cited ? ` 指名 ${r.judge.cited.law}第${r.judge.cited.articleNo}条` : ''}` +
        `${r.judge.quoteVerified === null ? '' : ` 引文核对=${r.judge.quoteVerified}`}`);
    console.log(`     判官原话：${r.judge.reason ?? '（无）'}`);
    if (r.judge.cited && !abstained && r.expected.length > 0) {
        const want = r.expected[0];
        const hitExpected = r.judge.cited.articleNo === want.articleNo;
        console.log(`     ${hitExpected ? '✅' : '❌'} 指名的条号与 ground truth ${want.source}#${want.articleNo} ` +
            `${hitExpected ? '一致' : '不一致'}（不一致不算错，但要看一眼）`);
    }
}
console.log('  说明："理由错了也算 ❌"这条**由强制指名机制化**：可答必须指名到条，');
console.log('        所以 sem-02 的"理由对不对"就是它有没有指到第二十条。散文理由本身不做机器判定，');
console.log('        上面逐条打印出来供人复核 —— 我们不假装能自动核验一段自然语言。');

// ---------------------------------------------------------------------------
// 排除项：显式计数，并且两个分母都印
// ---------------------------------------------------------------------------
console.log(`\n=== 判官调用结果分类（绝不静默丢弃任何一类）===`);
const outcomes: Record<string, number> = {};
for (const r of rows) outcomes[r.judge.outcome] = (outcomes[r.judge.outcome] ?? 0) + 1;
for (const [k, v] of Object.entries(outcomes).sort()) console.log(`  ${k.padEnd(18)} ${v}`);
const failed = rows.filter(r => ['error', 'parse_failure', 'binding_mismatch'].includes(r.judge.outcome));
console.log(`  其中"没有给出可用判定"的共 ${failed.length} 条：${failed.map(r => r.id).join(' ') || '（无）'}`);
console.log(`  上面的比率用的是**全部** ${rows.length} 题做分母（失败按"退回相似度"计入，不当成可答也不当成不可答）。`);
if (failed.length > 0) {
    console.log(`  若只算判官给出判定的那些题，分母是 ${rows.length - failed.length} —— 两个都印，自己选一个读。`);
}

// ---------------------------------------------------------------------------
// 精度代理：判官指名的条号 vs ground truth
//
// 数据集每条域内题都带 expected[{source, articleNo}]，这是**免费的精度指纹**：
// 判官按定义必须指名一条，那名指得对不对是可确定性核对的。
// §10.13 惋惜过"手工标注的精度集是这套指标还缺的那块地基"，这个不花一分钱就补上了一部分。
// **不一致不算错**（README 的 Limitations 说过：ground truth 只是"一条"正确答案，
// sem-01 的承运人责任那条可能才是更好的答案），所以只作诊断打印、不计分。
// ---------------------------------------------------------------------------
console.log(`\n=== 判官指名的条号 vs ground truth（仅诊断，不计分）===`);
let namedTotal = 0, namedHit = 0;
const mismatches: string[] = [];
for (const r of rows) {
    if (r.expected.length === 0) continue;
    if (r.judge.outcome !== 'judged' || !r.judge.answerable || !r.judge.cited) continue;
    namedTotal++;
    const hit = r.expected.some(e => e.articleNo === r.judge.cited!.articleNo);
    if (hit) namedHit++;
    else mismatches.push(`${r.id} 判官指名 ${r.judge.cited.law}第${r.judge.cited.articleNo}条，` +
        `ground truth ${r.expected.map(e => `${e.source}#${e.articleNo}`).join('/')}`);
}
console.log(`  判官给了可答判定且有 ground truth 的题：${namedTotal} 条`);
console.log(`  其中指名命中的：${namedHit}/${namedTotal}${namedTotal ? ` = ${(namedHit / namedTotal).toFixed(3)}` : ''}`);
for (const m of mismatches) console.log(`    · ${m}`);

// ---------------------------------------------------------------------------
// 判官身份（模型 B）：翻转率
// ---------------------------------------------------------------------------
if (MODEL_B) {
    const both = rows.filter(r => r.judgeB && r.judge.outcome === 'judged' && r.judgeB.outcome === 'judged');
    const flips = both.filter(r => r.judge.answerable !== r.judgeB!.answerable);
    console.log(`\n=== 判官身份：${ANSWERABILITY_JUDGE_MODEL} vs ${MODEL_B}（单独一张表）===`);
    console.log(`  两个判官都给出判定的题：${both.length}`);
    console.log(`  在"可答/不可答"上分歧的：${flips.length}` +
        `${both.length ? ` = ${(flips.length / both.length).toFixed(3)}` : ''}`);
    for (const f of flips) {
        console.log(`    · ${f.id} ${ANSWERABILITY_JUDGE_MODEL}=${f.judge.answerable ? '可答' : '不可答'}` +
            ` / ${MODEL_B}=${f.judgeB!.answerable ? '可答' : '不可答'}`);
    }
    const decA = (r: Row) => decideWithJudge(r, r.judge, 'judge');
    const decB = (r: Row) => decideWithJudge(r, r.judgeB ?? r.judge, 'judge');
    console.log(`  policy=judge 下的结果差异：域外拒答 ${ood.filter(decA).length}/${ood.length}` +
        ` vs ${ood.filter(decB).length}/${ood.length}；` +
        `域内误拒 ${dom.filter(decA).length}/${dom.length} vs ${dom.filter(decB).length}/${dom.length}`);
    console.log(`  判官身份对指标的移动，通常比被测系统本身的效应还大 —— 换模型必须重量，不能假设。`);
}

// ---------------------------------------------------------------------------
// 成本与延迟
// ---------------------------------------------------------------------------
const called = rows.filter(r => r.judge.outcome !== 'skipped');
const uncached = called.filter(r => !r.judge.cached);
/**
 * 模型 B 的调用**必须和 A 一样计入成本**。
 * 上一版只累加 `r.judge`，于是跑 `--judge-model-b` 时 B 的几十次真实调用在"成本"一节里
 * 凭空消失——报告会显示"本轮新增 0 token"，而账面上明明花掉了一整轮。
 * **漏记的成本和漏记的失败一样，都是让数字变好看的方向。**
 */
const calledB = MODEL_B ? rows.filter(r => r.judgeB && r.judgeB.outcome !== 'skipped') : [];
const uncachedB = calledB.filter(r => !r.judgeB!.cached);
const sumTokens = (list: JudgeAccounting[]) => list.reduce((a, u) => ({
    p: a.p + u.usage.promptTokens, c: a.c + u.usage.completionTokens,
}), { p: 0, c: 0 });
const allT = sumTokens([...called.map(r => r.judge), ...calledB.map(r => r.judgeB!)]);
const newT = sumTokens([...uncached.map(r => r.judge), ...uncachedB.map(r => r.judgeB!)]);
const latOf = (list: Row[], pick: (r: Row) => JudgeAccounting) =>
    list.map(pick).map(a => a.latencyMs).sort((a, b) => a - b);
const pctOf = (arr: number[], p: number) =>
    arr.length ? arr[Math.min(arr.length - 1, Math.floor(arr.length * p))] : 0;
const lat = latOf(uncached, r => r.judge);

console.log(`\n=== 成本与延迟 ===`);
console.log(`  实际调用判官：${called.length} 题（硬跳过 ${rows.length - called.length} 题）` +
    (MODEL_B ? `；对照判官 ${calledB.length} 题` : ''));
console.log(`  命中缓存：${ANSWERABILITY_JUDGE_MODEL} ${called.filter(r => r.judge.cached).length} 次` +
    (MODEL_B ? ` / ${MODEL_B} ${calledB.filter(r => r.judgeB!.cached).length} 次` : '') +
    `（缓存命中不计费）`);
console.log(`  token：本轮新增 ${newT.p} prompt + ${newT.c} completion` +
    (MODEL_B ? `（两个判官合计）` : '') + `；含缓存回放共 ${allT.p} + ${allT.c}`);
/**
 * 延迟只在**本轮未命中缓存**的调用上有样本。热缓存重跑时样本数是 0——
 * 那时必须印"未测"，不能印 "0ms"：0 是一个测量结果，"没测"不是，
 * 而两者的区别正是这份脚本存在的理由。要延迟数字就报冷缓存那一轮。
 */
const latLine = (label: string, arr: number[]) => {
    if (arr.length === 0) {
        console.log(`  判定延迟 ${label}：**未测**（本轮 0 次未命中缓存的调用 —— 热缓存重跑不产生延迟样本）`);
        return;
    }
    console.log(`  判定延迟 ${label}（未命中缓存 ${arr.length} 次）：` +
        `p50 ${pctOf(arr, 0.5)}ms / p95 ${pctOf(arr, 0.95)}ms / 最大 ${arr[arr.length - 1]}ms`);
};
latLine(ANSWERABILITY_JUDGE_MODEL, lat);
if (MODEL_B) latLine(MODEL_B, latOf(uncachedB, r => r.judgeB!));

// ---------------------------------------------------------------------------
// 报告
// ---------------------------------------------------------------------------
const reportPath = path.join(EVAL_DIR, 'abstention-report.json');
const report = {
    // 身份块。判官的模型与 prompt 版本不是可选项——§10.13 量到判官身份对指标的移动
    // 比系统本身产生的任何效应都大，不记下来这份报告下周就没法解释。
    config: {
        datasetSha1,
        embeddingModel: EMBEDDING_MODEL,
        rerank: { enabled: RERANK_ENABLED, model: RERANK_MODEL, candidates: RERANK_CANDIDATES },
        topK: TOP_K,
        threshold: THRESHOLD,
        judgeModel: ANSWERABILITY_JUDGE_MODEL,
        judgeModelB: MODEL_B ?? null,
        judgePromptVersion: ANSWERABILITY_JUDGE_PROMPT_VERSION,
        policy: POLICY,
        cases: cases.length,
        /** 线上是"相似度拒答则跳过判官"。评测刻意不跳，好量这个反事实。 */
        note: 'judge was run even when similarity abstained, to measure the sim_abstain skip rule counterfactually',
    },
    policies: Object.fromEntries(POLICIES.map(p => {
        const decided = (r: Row) => decideWithJudge(r, r.judge, p);
        return [p, {
            oodCaught: ood.filter(decided).length, oodTotal: ood.length,
            domFalseRefusals: dom.filter(decided).length, domTotal: dom.length,
            byKind: Object.fromEntries((['legal_adjacent', 'unrelated', 'unclassified'] as Kind[])
                .map(k => {
                    const s = stratify(ood, k);
                    return [k, { n: s.length, caught: s.filter(decided).length }];
                }).filter(([, v]) => (v as { n: number }).n > 0)),
        }];
    })),
    exclusions: {
        byOutcome: outcomes,
        noUsableVerdict: failed.map(r => r.id),
        denominators: { all: rows.length, judgedOnly: rows.length - failed.length },
    },
    groundTruth: { namedTotal, namedHit, mismatches },
    // 分数门整条曲线。计划里"先量出整条曲线再定"的第二个维度——它退化的事实本身是结论。
    scoreGate: {
        acceptedWithScore: scored.length,
        namedScoreRange: scored.length
            ? [Math.min(...scored.map(r => r.judge.namedScore!)), Math.max(...scored.map(r => r.judge.namedScore!))]
            : null,
        oodAnswerableVerdicts: ood.filter(r => r.judge.answerable === true).length,
        curve: cands.map(t => ({
            t: t === Number.NEGATIVE_INFINITY ? null : t,
            oodCaught: ood.filter(r => gate(r, t)).length,
            domFalseRefusals: dom.filter(r => gate(r, t)).length,
        })),
    },
    modelB: MODEL_B ? {
        model: MODEL_B,
        flips: rows.filter(r => r.judgeB && r.judge.outcome === 'judged' && r.judgeB.outcome === 'judged')
            .filter(r => r.judge.answerable !== r.judgeB!.answerable).map(r => r.id),
    } : null,
    cost: {
        judgeCalls: called.length + calledB.length,
        judgeCallsByModel: { [ANSWERABILITY_JUDGE_MODEL]: called.length, ...(MODEL_B ? { [MODEL_B]: calledB.length } : {}) },
        cacheHits: called.filter(r => r.judge.cached).length + calledB.filter(r => r.judgeB!.cached).length,
        newTokens: { prompt: newT.p, completion: newT.c },
        totalTokensIncludingCacheReplay: { prompt: allT.p, completion: allT.c },
        // 延迟按判官分开记：把两个模型的延迟混成一个 p50，正是"两个看起来是同一个测量的数"。
        // ⚠️ samples === 0 表示**这一轮没测到延迟**（热缓存重跑），不是"延迟为 0"。
        latencyMs: {
            // 刻意**不**把上一轮的延迟数字抄进来：硬编码一个测量值进源码，
            // 下一份报告就会带着一个看起来属于自己、实际来自别处的数——
            // 那正是这份文件反复警告的"两个看起来是同一个测量的数字"。
            note: 'samples 是本轮未命中缓存的调用数；0 表示这一轮没测延迟，不是延迟为 0。要延迟就报冷缓存那一轮',
            byModel: {
                [ANSWERABILITY_JUDGE_MODEL]: { p50: pctOf(lat, 0.5), p95: pctOf(lat, 0.95), samples: lat.length },
                ...(MODEL_B ? { [MODEL_B]: (() => {
                    const lb = latOf(uncachedB, r => r.judgeB!);
                    return { p50: pctOf(lb, 0.5), p95: pctOf(lb, 0.95), samples: lb.length };
                })() } : {}),
            },
        },
    },
    rows: rows.map(r => ({
        id: r.id, kind: r.kind, question: r.question, expectAbstain: r.expectAbstain,
        simAbstained: r.simAbstained, simTopScore: r.simTopScore, simReason: r.simReason,
        hardSkipReason: r.hardSkipReason, cov: Number(r.cov.toFixed(4)), rerank: r.rerank,
        judge: r.judge,
        judgeB: r.judgeB,
    })),
    generatedAt: new Date().toISOString(),
};
fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), 'utf-8');
console.log(`\n完整结果 → ${path.relative(process.cwd(), reportPath)}`);
