/**
 * 多法查询改写的**端到端**评测（会花钱：每题一次 gpt-4o-mini 调用）。
 *
 * 只跑 `multi_law` 类别：比较原始检索与"按法改写 + 补位合并"的覆盖率。
 * 改写失败时退回原始检索（本脚本会如实计入"改写无效"）。
 *
 * 注意：这不是 CI 门禁的一部分。门禁只跑确定性检索；改写要调模型，不能进
 * 每次都要求同输入同输出的 job。
 */
import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import { createRetriever } from '../src/index.js';
import { loadKnowledge } from '../src/loadKnowledge.js';
import { rewriteForLaws, mergeRewrittenResults } from '../src/QueryRewriter.js';
import { parseQuery } from '../src/EmbeddingRetriever.js';
import { parseArticleNo } from '../src/articleNo.js';
import { EVAL_DIR } from '../src/paths.js';

interface Case { id: string; category: string; question: string; expected?: { source: string; articleNo: number }[] }

const key = (source: string, articleNo: number) => `${source}#${articleNo}`;

async function main(): Promise<void> {
    const raw = fs.readFileSync(path.join(EVAL_DIR, 'dataset.jsonl'), 'utf-8');
    const cases: Case[] = raw.split('\n').filter(l => l.trim()).map(l => JSON.parse(l))
        .filter(c => c.category === 'multi_law');
    if (cases.length === 0) { console.error('数据集里没有 multi_law 类别'); process.exit(1); }

    const retriever = createRetriever();
    await loadKnowledge(retriever);

    let baseHits = 0, mergedHits = 0, total = 0;
    console.log('id'.padEnd(10) + 'base  merged  laws');
    for (const c of cases) {
        const expected = (c.expected ?? []).map(e => key(e.source, e.articleNo));
        total += expected.length;
        const base = await retriever.retrieve(c.question, 5);
        const laws = parseQuery(c.question).lawNames;

        let merged = base;
        let note = laws.join('、') || '（未点名法律）';
        if (laws.length >= 2) {
            const rewrites = await rewriteForLaws(c.question, laws);
            if (rewrites.length > 0) {
                const extra = [];
                for (const r of rewrites) extra.push(...await retriever.retrieveInLaw(r.query, r.law, 2));
                merged = mergeRewrittenResults(base, extra, laws, 5);
            } else {
                note += '（改写失败，退回）';
            }
        } else {
            note += '（不触发）';
        }

        const hit = (list: typeof base) => expected.filter(e =>
            list.some(h => key(h.source, Number(parseArticleNo(h.articleNo))) === e)).length;
        const b = hit(base), m = hit(merged);
        baseHits += b; mergedHits += m;
        console.log(`${c.id.padEnd(10)}${`${b}/2`.padEnd(6)}${`${m}/2`.padEnd(8)}${note}`);
    }

    console.log(`\n覆盖率：原始 ${baseHits}/${total} = ${(baseHits / total).toFixed(3)}` +
        ` → 改写 ${mergedHits}/${total} = ${(mergedHits / total).toFixed(3)}`);
    console.log('（分母是 expected 条目数，不是题数；每题 2 条。）');
}

main().catch(e => { console.error(e); process.exit(1); });
