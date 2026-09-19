/**
 * 拒答信号的实测：域外问题 vs 域内问题，四个候选信号各自的可分性。
 * 这是诊断脚本，不是产品代码。它存在的唯一目的是让 README 里那张拒答信号表**可复现**：
 * 表里的每个数字都来自这里，而不是从上一版结论里抄的。
 * 换 embedding 模型后检索到的文本本身会变，表就过期了——所以它得能一条命令重跑。
 *
 * ## 为什么要有 AUC
 *
 * "域外全拒要误拒几条 / 零误拒能拒掉几条"只描述**两端**，中间那段操作区间看不见，
 * 于是两个信号可能报出同样的两端、实际可分性差很远。AUC 把整条 ROC 压成一个数：
 * 随机抽一条域内 + 一条域外，信号把域内排在"更该回答"那一侧的概率。
 * **0.5 = 完全瞎猜**，这正是我们要防的——一个 AUC≈0.5 的信号配上精心挑的阈值，
 * 在样本上看起来能用，换个数据集立刻塌。
 *
 * ## 为什么按 oodKind 拆开看
 *
 * "域外"不是一个同质的东西。`unrelated`（番茄炒蛋、TypeScript 去重）和 `legal_adjacent`
 * （劳动合同法、商标法）对检索器是完全不同的难度，混在一起报一个平均 AUC
 * 会把"某类根本分不开"这个事实平均掉。数据集里存 oodKind 就是为了这里能拆。
 */
import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import EmbeddingRetriever from '../src/EmbeddingRetriever.js';
import Reranker from '../src/Reranker.js';
import { EMBEDDING_MODEL } from '../src/index.js';
import { loadKnowledge } from '../src/loadKnowledge.js';
import { EVAL_DIR } from '../src/paths.js';
import { coverage, auc, scanThresholds } from './abstentionCore.js';

const TOP_K = 5;
const retriever = new EmbeddingRetriever(EMBEDDING_MODEL);
retriever.setStrict(true);
await loadKnowledge(retriever);

const cases = fs.readFileSync(path.join(EVAL_DIR, 'dataset.jsonl'), 'utf-8')
    .split('\n').filter(l => l.trim()).map(l => JSON.parse(l));

// 域外题必须带 oodKind。以前这里是静默的 `c.oodKind ?? 'in_domain'`，
// 于是 ood-01..04（当时确实没写 oodKind）被当成域内题，既不在 legal_adjacent 也不在
// unrelated 分层里——**汇总照常打印，分层却少了 4 条，没有任何东西会说**。
// 一张看起来正常但少了几行的表比报错更难发现，所以这里直接退出。
{
    const missing = cases.filter(c => c.category === 'out_of_domain' && !c.oodKind);
    if (missing.length > 0) {
        console.error(`✗ 数据集里有 ${missing.length} 道域外题没写 oodKind：${missing.map(c => c.id).join(' ')}`);
        console.error('  分层统计会静默漏掉它们。请在 eval/dataset.jsonl 里补上 oodKind 再跑。');
        process.exit(2);
    }
}

interface Row {
    id: string; cat: string; kind: string; q: string;
    top1: number; margin: number; cov: number;
    rerank: number | null;   // null = 重排模型没跑起来（见下面的降级分支）
}
const rows: Row[] = [];

// 重排模型单独在这里加载，不看 RERANK_ENABLED —— 这里是**测量**某个信号有没有用，
// 不是在跑线上链路，所以不受那个开关约束。加载失败就降级跳过这个信号，
// 不能让整个诊断脚本因为一个模型下不下来就什么都报不出来。
let reranker: Reranker | null = null;
try {
    reranker = new Reranker();
    await reranker.rerank('预热', [{ embedText: '预热' }]);
} catch (e) {
    console.warn(`⚠ 重排模型不可用（${(e as Error).message}），跳过该信号\n`);
    reranker = null;
}

for (const c of cases) {
    const r = await retriever.retrieve(c.question, TOP_K);
    const scores = r.map(x => x.score);
    const top1 = Math.max(...scores);
    const top5 = scores.length >= TOP_K ? scores[TOP_K - 1] : Math.min(...scores);

    let rerank: number | null = null;
    if (reranker) {
        // 被"点名条号"钉住的候选**不参与重排**（理由见 src/Reranker.ts 的 partitionPinned），
        // 于是它们没有 rerankScore。这不是缺陷：查询能点出条号，就说明它按定义就是域内的，
        // 这类查询永远不该被拒答。所以这里给 +∞，而不是"剩下几条里最高的那个"——
        // 取后者会把 lookup 类问题算成一个很低的分，凭空造出一个假的可分性。
        if (r.some(x => x.exactArticleHit)) rerank = Number.POSITIVE_INFINITY;
        else {
            const out = await reranker.rerank(c.question, r);
            rerank = Math.max(...out.map(x => x.rerankScore));
        }
    }

    rows.push({
        id: c.id, cat: c.category, kind: c.oodKind ?? (c.category === 'out_of_domain' ? 'unclassified' : 'in_domain'), q: c.question,
        top1, margin: top1 - top5, cov: coverage(c.question, r.map(x => x.document)), rerank,
    });
}

const ood = rows.filter(r => r.cat === 'out_of_domain');
const dom = rows.filter(r => r.cat !== 'out_of_domain');

console.log(`模型 ${EMBEDDING_MODEL} · 域外 ${ood.length} 条 / 域内 ${dom.length} 条`);
console.log(reranker ? '精排模型已加载，第四个信号参与测量\n' : '');
console.log('=== 信号在域内/域外的取值区间 ===');

const SIGNALS = [
    ['top1 余弦', (r: Row) => r.top1],
    ['top1 − top5 间隔', (r: Row) => r.margin],
    ['查询 bigram 覆盖率', (r: Row) => r.cov],
    ['cross-encoder 重排分', (r: Row) => r.rerank],
] as const;

for (const [name, get] of SIGNALS) {
    const usable = rows.filter(r => get(r) !== null);
    if (usable.length < rows.length) continue;
    const o = usable.filter(r => r.cat === 'out_of_domain').map(r => get(r) as number).sort((a, b) => a - b);
    const d = usable.filter(r => r.cat !== 'out_of_domain').map(r => get(r) as number).sort((a, b) => a - b);
    console.log(`\n【${name}】 越小越该拒答`);
    console.log(`  域外: ${o.map(x => x.toFixed(3)).join(' ')}`);
    console.log(`  域内: 最小 ${d[0].toFixed(3)} / 中位 ${d[Math.floor(d.length / 2)].toFixed(3)} / 最大 ${d[d.length - 1].toFixed(3)}`);
    // 域内=正类：AUC 是"域内被排在更该回答那一侧"的概率。
    console.log(`  AUC(域内 vs 全部域外) = ${auc(d, o).toFixed(3)}`);
    // unclassified 正常恒为 0（上面的守卫已经拦住了缺 oodKind 的域外题），
    // 列在这里只是万一守卫被绕过时还能看见；n=0 时不打印，免得每轮多一行恒空的噪声
    for (const k of ['legal_adjacent', 'unrelated', 'unclassified']) {
        const neg = usable.filter(r => r.kind === k).map(r => get(r) as number);
        if (neg.length > 0) console.log(`      vs ${k.padEnd(16)} AUC = ${auc(d, neg).toFixed(3)}  (n=${neg.length})`);
    }
    // 全域扫描交给 abstentionCore.ts：新脚本 eval/abstention.ts 要和这里用**同一份**实现。
    // 两份实现迟早漂移，而症状是两张表对不上、却都"看起来正常"。
    const scan = scanThresholds(o, d);
    console.log(`  域外全拒: 阈值 ≥ ${scan.allOod?.threshold.toFixed(3)} 时误拒 ${scan.allOod?.falseRefusals}/${d.length}`);
    console.log(`  零误拒:   阈值 ${scan.zeroFalseRefusal?.threshold.toFixed(3)} 时拒掉 ${scan.zeroFalseRefusal?.caught}/${o.length} 域外`);
}
