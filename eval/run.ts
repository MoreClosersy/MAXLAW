/**
 * 评测入口：`pnpm eval`（加 `--generate` 连生成一起测）。
 *
 * ## 为什么先做检索指标
 *
 * 这条链路是 检索 → 拒答判定 → 生成。生成错可能是模型的锅，也可能是检索压根没
 * 给出正确条文——不把检索单独量出来，后面所有"答案不对"都无从归因。
 * 而且检索指标是**确定性**的：同样的语料 + 同样的查询，分数必须逐位一致，
 * 所以它能当 CI 门禁；生成指标有采样噪声，不适合卡门禁。
 *
 * ## ground truth 自检
 *
 * 数据集的每条 ground truth 都带一段条文正文的指纹片段。评测启动时先拿它
 * 对着当前语料核一遍，对不上就直接退出——**宁可评测跑不起来，也不要拿一份
 * 已经和语料对不上的 ground truth 算出一堆看似正常的分数**。
 * 这正是"扩语料必须是一次有版本号的离散动作"的落地点。
 */
import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { splitIntoChunks } from '../src/EmbeddingRetriever.js';
import { createRetriever, decideAbstention, EMBEDDING_MODEL, RETRIEVAL_TOP_K, RETRIEVAL_SCORE_THRESHOLD, runQuery } from '../src/index.js';
import { RERANK_ENABLED, RERANK_MODEL, RERANK_CANDIDATES } from '../src/Reranker.js';
import { loadKnowledge } from '../src/loadKnowledge.js';
import { KNOWLEDGE_DIR, EVAL_DIR } from '../src/paths.js';
import { parseArticleNo } from '../src/articleNo.js';
import { summarizeCitations } from '../src/CitationVerifier.js';

// ---------------------------------------------------------------------------
// 参数
// ---------------------------------------------------------------------------
const argv = process.argv.slice(2);
const has = (f: string) => argv.includes(f);
/**
 * 取数值参数。没传返回 fallback；**传了但不是数字就报错退出**。
 *
 * 这里刻意不用 `Number(x) || fallback`：那种写法会让 `--min-hit=abc` 静默变成 NaN，
 * 而 NaN 的比较恒为 false —— 门禁会永远显示通过。一个写错的门禁比没有门禁更危险，
 * 因为它会让 CI 一直绿着。退出码 2 用于区分"配置写错"与"指标不达标"（后者是 1）。
 */
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
const GENERATE = has('--generate');
const VERBOSE = has('--verbose') || has('-v');
const ONLY_CATEGORY = argv.find(a => a.startsWith('--category='))?.split('=')[1];
/**
 * CI 门禁。三类闸，都只在显式指定时判定（不传就只报数，避免默认行为随指标波动而变）：
 *
 *   --min-hit=<v>              整体 hit@K 下限 —— 闸**召回**
 *   --min-mrr=<v>              整体 MRR 下限   —— 闸**排序**
 *   --min-cat-hit=<类别>:<v>   分类别 hit@K 下限，可重复 —— 闸**单类塌陷**
 *
 * 为什么不止一个数：near_duplicate 曾经从 1.000 掉到 0.500，而整体 hit@1 只从 0.741
 * 掉到 0.704——**分类别的塌陷会被整体数字稀释掉**（roadmap §10.10）。
 * 只闸整体一个数，就是把这同一个教训在 CI 这一层再犯一次。
 * 而 MRR 必须单独闸：reranker 这类改动会只动排序、不动召回，hit@K 完全看不出来。
 */
const MIN_HIT = num('--min-hit', NaN);
const MIN_MRR = num('--min-mrr', NaN);
const MIN_CAT_HIT = argv
    .filter(a => a.startsWith('--min-cat-hit='))
    .map(a => a.slice('--min-cat-hit='.length))
    .map(spec => {
        const i = spec.lastIndexOf(':');
        return { category: spec.slice(0, i), floor: Number(spec.slice(i + 1)) };
    });

interface Expected { source: string; articleNo: number; snippet: string; }
interface Case {
    id: string;
    category: string;
    question: string;
    expected?: Expected[];
    expectAbstain: boolean;
}

const norm = (s: string) => s.replace(/\s+/g, '');

// ---------------------------------------------------------------------------
// 1. 语料指纹
// ---------------------------------------------------------------------------
interface Manifest { version: number; corpusHash: string; totalArticles: number; }
const manifestPath = path.join(KNOWLEDGE_DIR, 'CORPUS.json');
if (!fs.existsSync(manifestPath)) {
    console.error(`❌ 找不到 ${manifestPath}。先跑 pnpm corpus:manifest。`);
    process.exit(1);
}
const manifest: Manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));

// ---------------------------------------------------------------------------
// 2. 数据集 + ground truth 自检
// ---------------------------------------------------------------------------
const datasetPath = path.join(EVAL_DIR, 'dataset.jsonl');
if (!fs.existsSync(datasetPath)) {
    console.error(`❌ 找不到 ${datasetPath}`);
    process.exit(1);
}
const datasetRaw = fs.readFileSync(datasetPath, 'utf-8');
const datasetSha1 = crypto.createHash('sha1').update(datasetRaw).digest('hex').slice(0, 12);

let cases: Case[] = datasetRaw.split('\n').filter(l => l.trim()).map(l => JSON.parse(l));
if (ONLY_CATEGORY) cases = cases.filter(c => c.category === ONLY_CATEGORY);
if (cases.length === 0) {
    console.error(`❌ 没有选中任何用例${ONLY_CATEGORY ? `（--category=${ONLY_CATEGORY}）` : ''}`);
    process.exit(1);
}

// 把当前语料按**与检索完全相同的切分方式**摊平，用于核对 ground truth
const corpusArticles = new Map<string, string>();   // `source#条号` → 正文
for (const file of fs.readdirSync(KNOWLEDGE_DIR).filter(f => f.endsWith('.md'))) {
    const content = fs.readFileSync(path.join(KNOWLEDGE_DIR, file), 'utf-8');
    for (const chunk of splitIntoChunks(content, file)) {
        const n = parseArticleNo(chunk.articleNo);
        if (n !== null) corpusArticles.set(`${file}#${n}`, chunk.document);
    }
}

const stale: string[] = [];
for (const c of cases) {
    for (const e of c.expected ?? []) {
        const doc = corpusArticles.get(`${e.source}#${e.articleNo}`);
        if (!doc) { stale.push(`${c.id}: ${e.source} 第${e.articleNo}条 在语料中不存在`); continue; }
        if (!norm(doc).includes(norm(e.snippet))) {
            stale.push(`${c.id}: ${e.source} 第${e.articleNo}条 的指纹片段对不上（语料已改动？）`);
        }
    }
}
if (stale.length > 0) {
    console.error(`\n❌ ground truth 与语料 v${manifest.version} 不匹配，共 ${stale.length} 处：`);
    for (const s of stale) console.error(`   ${s}`);
    console.error(
        `\n   语料变更后必须重新校对 eval/dataset.jsonl 里对应的条号与指纹片段。\n` +
        `   recall@k / MRR 跨语料版本不可比——候选池变了，分母也就变了。\n` +
        `   确认语料没动过的话，就是数据集本身写错了。`
    );
    process.exit(1);
}

// ---------------------------------------------------------------------------
// 3. 建索引
// ---------------------------------------------------------------------------
const retriever = createRetriever();
retriever.setStrict(true);   // 索引不完整时不要算出"看起来正常"的分数
let loaded;
try {
    loaded = await loadKnowledge(retriever);
} catch (error) {
    console.error(`\n❌ 索引构建失败：${(error as Error).message}`);
    console.error('   评测数字在有兜底向量的索引上没有任何意义，因此直接退出。');
    process.exit(1);
}

// ---------------------------------------------------------------------------
// 4. 跑
// ---------------------------------------------------------------------------
interface Row {
    id: string; category: string; question: string;
    abstained: boolean; topScore: number; reason: string;
    ranks: number[];             // 每个 ground truth 条目在结果里的名次（1 起；0 表示没检索到）
    retrieved: string[];
    // --generate 才有的字段
    answer?: string;
    citationSummary?: string | null;
    citationsTotal?: number;
    citationsVerified?: number;
    citationsFabricated?: number;
    citationsNotInContext?: number;
}

const rows: Row[] = [];
for (const c of cases) {
    const retrieved = await retriever.retrieve(c.question, TOP_K);
    const decision = decideAbstention(retrieved, THRESHOLD);

    const keyed = retrieved.map(r => `${r.source}#${parseArticleNo(r.articleNo)}`);
    const ranks = (c.expected ?? []).map(e => {
        const idx = keyed.indexOf(`${e.source}#${e.articleNo}`);
        return idx < 0 ? 0 : idx + 1;
    });

    const row: Row = {
        id: c.id, category: c.category, question: c.question,
        abstained: decision.abstained, topScore: Number(decision.topScore.toFixed(4)),
        reason: decision.reason, ranks,
        retrieved: retrieved.map(r => `${r.source}#${parseArticleNo(r.articleNo)}`),
    };

    if (GENERATE) {
        const result = await runQuery({ query: c.question, retriever });
        row.answer = result.answer;
        row.citationSummary = summarizeCitations(result.citations);
        row.citationsTotal = result.citations.total;
        row.citationsVerified = result.citations.verified;
        row.citationsFabricated = result.citations.fabricated;
        row.citationsNotInContext = result.citations.notInContext;
        row.abstained = result.abstained;   // 以真实链路为准
    }
    rows.push(row);

    process.stdout.write(VERBOSE ? `\n  ${c.id} ${c.question.slice(0, 28)} → ${row.ranks.join(',') || '—'}\n` : '.');
}
if (!VERBOSE) process.stdout.write('\n');

// ---------------------------------------------------------------------------
// 5. 汇总
// ---------------------------------------------------------------------------
const inDomain = rows.filter(r => (r.ranks.length > 0));
const ood = rows.filter(r => r.ranks.length === 0);

const hit = (r: Row, k: number) => r.ranks.some(x => x > 0 && x <= k);
const rr = (r: Row) => { const f = r.ranks.filter(x => x > 0).sort((a, b) => a - b)[0]; return f ? 1 / f : 0; };
const rate = (n: number, d: number) => d === 0 ? '—' : (n / d).toFixed(3);

/** 显示宽度：CJK 字符占两列，padEnd 按码元数算会把表格错开 */
const width = (s: string) => [...s].reduce((a, ch) => a + (ch.charCodeAt(0) > 0x2e80 ? 2 : 1), 0);
const pad = (s: string, w: number) => s + ' '.repeat(Math.max(0, w - width(s)));

function table(subset: Row[], label: string) {
    const n = subset.length;
    if (n === 0) return null;
    // 没有 ground truth 的类别（域外问题）只报拒答率。
    // 给它们算 hit@k 会得到整齐的 0.000——那不是"没检索到"，是"本来就没有正确答案"，
    // 摆进表格只会让人误读成检索效果差。
    const hasGT = subset.some(r => r.ranks.length > 0);
    return {
        label, n, hasGT,
        hit1: hasGT ? rate(subset.filter(r => hit(r, 1)).length, n) : '—',
        hit3: hasGT ? rate(subset.filter(r => hit(r, 3)).length, n) : '—',
        hitK: hasGT ? rate(subset.filter(r => hit(r, TOP_K)).length, n) : '—',
        mrr: hasGT ? (subset.reduce((a, r) => a + rr(r), 0) / n).toFixed(3) : '—',
        abstainRate: rate(subset.filter(r => r.abstained).length, n),
    };
}

const categories = [...new Set(rows.map(r => r.category))];
const tables = [
    ...categories.map(c => table(rows.filter(r => r.category === c), c)),
    table(inDomain, '整体(有 GT)'),
].filter(Boolean) as ReturnType<typeof table>[];

console.log(`\n语料 v${manifest.version} · ${manifest.totalArticles} 条 · 指纹 ${manifest.corpusHash} · 数据集 ${datasetSha1}`);
console.log(`topK=${TOP_K}  拒答阈值=${THRESHOLD}  ${GENERATE ? '含生成' : '仅检索'}\n`);
console.log(pad('类别', 20) + ' 题数   hit@1   hit@3  hit@' + TOP_K + '    MRR   拒答率');
console.log('─'.repeat(72));
for (const t of tables!) {
    console.log(
        pad(t!.label, 20) + ' ' + String(t!.n).padStart(4) + '  ' +
        t!.hit1.padStart(6) + '  ' + t!.hit3.padStart(6) + '  ' + t!.hitK.padStart(6) + '  ' +
        t!.mrr.padStart(6) + '  ' + t!.abstainRate.padStart(6)
    );
}

if (ood.length > 0) {
    const correct = ood.filter(r => r.abstained).length;
    // 这一行刻意写"相似度判定下"：线上拒答早已不只是阈值，而是
    // 相似度 → LLM 可答性判官 → 交叉校验三段。这条默认路径**只跑第一段**，
    // 因为它免费、确定性、能进 CI 门禁（判官要钱、要 key、有采样噪声）。
    // 不写清楚的话，读者会以为下面这些"✗ 未拒答"就是产品的真实行为。
    console.log(`\n域外问题（相似度判定下）应全部拒答：${correct}/${ood.length} 正确`);
    for (const r of ood.filter(x => !x.abstained)) {
        console.log(`   ✗ 未拒答（最高分 ${r.topScore}）: ${r.question}`);
    }
    console.log(`   真正的拒答行为见 pnpm eval:abstention（会花钱，不进 CI）`);
}

const missed = inDomain.filter(r => !hit(r, TOP_K));
if (missed.length > 0) {
    console.log(`\n${missed.length} 条在 top-${TOP_K} 内没检索到 ground truth：`);
    for (const r of missed) console.log(`   ${r.id} [${r.topScore}] ${r.question}`);
}

if (GENERATE) {
    const cited = rows.filter(r => (r.citationsTotal ?? 0) > 0);
    const total = cited.reduce((a, r) => a + (r.citationsTotal ?? 0), 0);
    const verified = cited.reduce((a, r) => a + (r.citationsVerified ?? 0), 0);
    const fab = cited.reduce((a, r) => a + (r.citationsFabricated ?? 0), 0);
    const ooc = cited.reduce((a, r) => a + (r.citationsNotInContext ?? 0), 0);
    console.log(`\n引用校验：${cited.length}/${rows.length} 条回答含条号引用，共 ${total} 处`);
    console.log(`   核实通过 ${verified}  条号不存在 ${fab}  未出现在检索结果中 ${ooc}`);
    console.log(`   引用准确率 ${rate(verified, total)}`);
}

const report = {
    corpusVersion: manifest.version,
    corpusHash: manifest.corpusHash,
    totalArticles: manifest.totalArticles,
    datasetSha1,
    // 指标的身份必须完整：光有 corpusVersion + datasetSha1 不够。
    // 换 embedding 模型会改变**每一条**分数，而语料指纹一个字都没变——
    // 少了这两行，两次不同模型的评测会挂着完全相同的标签，数字却被拿去互相比较。
    config: {
        embeddingModel: EMBEDDING_MODEL,
        embeddingDimension: retriever.getDimension(),
        // 精排同样改变排序、同样一个字都不改语料指纹，所以它也属于"这次数字的身份"。
        // 漏了这三行，"开精排"和"没开精排"两次评测会挂着同一份配置标签。
        rerank: RERANK_ENABLED
            ? { model: RERANK_MODEL, candidates: RERANK_CANDIDATES > 0 ? RERANK_CANDIDATES : TOP_K }
            : null,
        topK: TOP_K,
        threshold: THRESHOLD,
        generate: GENERATE,
        // 这份报告里的 abstained 只反映**第一段**（相似度阈值），不是线上行为。
        // 不写这一行，"这份报告说拒答率 7%"会读成一个关于产品的陈述，而它只是
        // 关于一个已经知道拦不住东西的过滤器的陈述。真正的拒答数字见 abstention-report.json。
        abstention: { mode: 'similarity-only', judge: null },
    },
    generatedAt: new Date().toISOString(),
    metrics: tables,
    ood: {
        total: ood.length,
        abstainedCorrectly: ood.filter(r => r.abstained).length,
        // ⚠️ 仍然是"相似度判定下"的数字
        mode: 'similarity-only',
    },
    rows,
};
// ---------------------------------------------------------------------------
// CI 门禁
// ---------------------------------------------------------------------------
const overall = tables!.find(t => t!.label === '整体(有 GT)');
let exited = 0;
const gates: { label: string; actual: number; floor: number }[] = [];

if (overall) {
    if (!Number.isNaN(MIN_HIT)) gates.push({ label: `整体 hit@${TOP_K}`, actual: Number(overall.hitK), floor: MIN_HIT });
    if (!Number.isNaN(MIN_MRR)) gates.push({ label: '整体 MRR', actual: Number(overall.mrr), floor: MIN_MRR });
}

for (const { category, floor } of MIN_CAT_HIT) {
    const t = tables!.find(x => x!.label === category);
    // 类别名写错时**必须报错而不是静默跳过**——静默跳过等于给了一个永远通过的门禁，
    // 而"看起来在闸、其实没闸"比没闸更危险。
    if (!t) {
        console.error(`\n❌ --min-cat-hit 指定的类别不存在：${category}`);
        console.error(`   可选：${tables!.map(x => x!.label).join(' / ')}`);
        exited = 1;
        continue;
    }
    if (!t.hasGT) {
        console.error(`\n❌ 类别 ${category} 没有 ground truth（只有拒答率），不能作为门禁`);
        exited = 1;
        continue;
    }
    if (Number.isNaN(floor)) {
        console.error(`\n❌ --min-cat-hit=${category}:${floor} 的阈值不是数字`);
        exited = 1;
        continue;
    }
    gates.push({ label: `${category} hit@${TOP_K}`, actual: Number(t.hitK), floor });
}

if (gates.length > 0) console.log('');
for (const g of gates) {
    const ok = g.actual >= g.floor;
    console.log(`${ok ? '✅' : '❌'} ${g.label} = ${g.actual.toFixed(3)} ${ok ? '≥' : '<'} 门禁 ${g.floor}`);
    if (!ok) exited = 1;
}
if (gates.length > 0 && exited === 0) console.log('门禁全部通过。');

// 落盘查询向量：语料 1526 条的向量本来就在缓存里，但查询向量一直没保存，
// 于是每次跑 eval 都要把 31 条查询重新联网嵌一遍——网络一抖就整个评测跑不起来，
// 而这是一组**确定性**的检索指标，本该能离线复算。
//
// 注意与 pruneCache 的相互作用：查询向量和语料向量共用同一个缓存文件，而
// `pnpm embed` 的 pruneCache() 以 activeKeys（只含语料 chunk）为准，会把查询向量清掉。
// 也就是说重新索引之后第一次 eval 仍需联网——可以接受，缓存不会无限膨胀。
retriever.flushCache();

const reportPath = path.join(EVAL_DIR, 'report.json');
fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), 'utf-8');
console.log(`\n完整结果 → eval/report.json`);
process.exit(exited);
