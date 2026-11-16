/**
 * 引用内容绑定的**校准**（离线、免费，不调用任何模型）。
 *
 * 它回答的问题不是"这个检查好不好"，而是"它在真实答案上误报多少、在已知坏样本上抓到多少"。
 * 没有这两个数，一个只会在标题上报警的检查比没有检查更糟——见 §Faithfulness 的教训。
 *
 * 数据来源：`eval/.answers-cache.json`（27 条真实答案，gitignored）与
 * `eval/judgeCore.ts` 的注入器（article_swap：条号真、内容错）。
 * 因此它只能在跑过 `pnpm eval:faithfulness` 的机器上运行；CI 里没有这份缓存。
 *
 * 判定规则本身在 `src/CitationVerifier.ts`（CONTENT_*），这里只复现它的两个读数：
 *   - 原始答案上被标 content_mismatch 的条数（误报）；
 *   - 注入换号样本里，被换上的那个条号被标出来的比例（检出）。
 */
import 'dotenv/config';
import fs from 'fs';
import { createRetriever, formatContext } from '../src/index.js';
import { loadKnowledge } from '../src/loadKnowledge.js';
import { verifyCitations } from '../src/CitationVerifier.js';
import { mutate } from './judgeCore.js';
import { cnToNumber } from '../src/articleNo.js';

const ANSWERS_PATH = 'eval/.answers-cache.json';

interface AnswerRow { id: string; question: string; answer: string; }

async function main(): Promise<void> {
    if (!fs.existsSync(ANSWERS_PATH)) {
        console.error(`缺少 ${ANSWERS_PATH}：先跑一次 pnpm eval:faithfulness（会花钱），这个校准才有数据。`);
        process.exit(1);
    }
    const answers: Record<string, AnswerRow> = JSON.parse(fs.readFileSync(ANSWERS_PATH, 'utf-8'));
    const corpus = fs.readFileSync('knowledge/民法典.md', 'utf-8')
        + '\n' + fs.readFileSync('knowledge/公司法.md', 'utf-8');

    const retriever = createRetriever();
    await loadKnowledge(retriever);
    const store = retriever.getStore();

    let originals = 0;
    let falsePositives = 0;
    const fpExamples: string[] = [];
    let swaps = 0;
    let detected = 0;
    let caughtAny = 0;
    const misses: string[] = [];

    for (const row of Object.values(answers)) {
        const retrieved = await retriever.retrieve(row.question, 5);

        const original = verifyCitations(row.answer, store, retrieved);
        originals += original.total;
        for (const c of original.checks.filter(x => x.status === 'content_mismatch')) {
            falsePositives++;
            fpExamples.push(`${row.id} ${c.label} own=${c.contentOverlap?.own.toFixed(3)} ` +
                `other=${c.contentOverlap?.bestOther.toFixed(3)}`);
        }

        for (const mutant of mutate(row.answer, formatContext(retrieved), corpus)) {
            if (mutant.kind !== 'article_swap') continue;
            swaps++;
            const needle = mutant.needle.match(/第\s*([0-9]+|[一二三四五六七八九十百千万零两]+)\s*条/);
            const swappedNo = needle ? cnToNumber(needle[1]) : NaN;
            const report = verifyCitations(mutant.answer, store, retrieved);
            const hit = report.checks.some(c =>
                c.articleNo === swappedNo && c.status === 'content_mismatch');
            if (hit) detected++;
            if (report.checks.some(c => c.articleNo === swappedNo && c.status !== 'verified')) caughtAny++;
            else {
                const c = report.checks.find(x => x.articleNo === swappedNo);
                misses.push(`${row.id} ${mutant.needle} own=${c?.contentOverlap?.own.toFixed(3) ?? '—'} ` +
                    `other=${c?.contentOverlap?.bestOther.toFixed(3) ?? '—'} status=${c?.status ?? '—'}`);
            }
        }
    }

    console.log(`原始答案：${originals} 处引用，被标 content_mismatch 的 ${falsePositives} 处（误报）`);
    for (const x of fpExamples) console.log(`  · ${x}`);
    console.log(`\n注入换号：${swaps} 个`);
    console.log(`  其中被标 content_mismatch：${detected} = ${swaps ? (detected / swaps).toFixed(3) : '—'}`);
    console.log(`  其中被任一非 verified 状态拦下（含 not_in_context/fabricated）：` +
        `${caughtAny} = ${swaps ? (caughtAny / swaps).toFixed(3) : '—'}`);
    for (const x of misses) console.log(`  · 漏检 ${x}`);
    console.log(`\n规则：bestOther - own ≥ 0.10 且 bestOther ≥ 0.25 且 own ≤ 0.35，` +
        `句子至少 8 个二元组。误报率按"原始引用"为分母，检出率按"注入换号"为分母。`);
}

main().catch(e => { console.error(e); process.exit(1); });
