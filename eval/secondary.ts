/**
 * 判官驱动二次检索的**端到端**验证（会花钱：2 次判官调用）。
 *
 * 构造方式：拿一道合法的跨法问题，把检索结果里的民法典条文**人为删掉**，
 * 模拟"另一半没召回"。然后走真实链路：
 *   1. 第一次判官判定（应当拒答，并指出缺民法典）；
 *   2. `planSecondaryRetrieval` 决定补哪部法；
 *   3. 在该法内再检索 2 条，重判一次；
 *   4. 看是否从"拒答"变成"可答"，且必答项覆盖两法。
 *
 * 这不是评测集的一部分（n=1，且是构造样本），只是证明机制能跑通、并量出它的成本。
 */
import 'dotenv/config';
import { createRetriever } from '../src/index.js';
import { loadKnowledge } from '../src/loadKnowledge.js';
import { judgeAnswerability } from '../src/AnswerabilityJudge.js';
import { planSecondaryRetrieval } from '../src/abstention.js';
import { normLawName } from '../src/laws.js';

const QUESTION = '法定代表人以公司名义从事民事活动，法律后果由谁承担？';

async function main(): Promise<void> {
    const retriever = createRetriever();
    await loadKnowledge(retriever);

    const full = await retriever.retrieve(QUESTION, 5);
    const half = full.filter(r => !r.source.includes('民法典'));
    console.log('完整检索：', full.map(r => `${r.source}#${r.articleNo}`).join(' '));
    console.log('删除民法典后：', half.map(r => `${r.source}#${r.articleNo}`).join(' '));

    const first = await judgeAnswerability(QUESTION, half);
    console.log(`\n第一次判官：${first.accounting.outcome} answerable=${first.accounting.answerable}` +
        ` (${first.accounting.latencyMs} ms, ${first.accounting.usage.promptTokens}+${first.accounting.usage.completionTokens} tokens)`);
    console.log(`理由：${first.accounting.reason ?? '（无）'}`);
    if (first.outcome.kind !== 'judged' && first.outcome.kind !== 'binding_mismatch') {
        console.log('第一次判定不可用，链路会退回相似度。');
        return;
    }

    const corpusLaws = [...new Set(retriever.getStore().sources().map(normLawName))];
    const plan = planSecondaryRetrieval(first.outcome.parsed, first.outcome.check, half, corpusLaws);
    console.log(`补检索计划：${plan.length ? plan.join('、') : '（不补）'}`);
    if (plan.length === 0) return;

    const extra = [];
    for (const law of plan) extra.push(...await retriever.retrieveInLaw(QUESTION, law, 2));
    const seen = new Set(half.map(r => `${r.source}#${r.hash}`));
    const expanded = [...half, ...extra.filter(r => !seen.has(`${r.source}#${r.hash}`))];
    console.log('补充后：', expanded.map(r => `${r.source}#${r.articleNo}`).join(' '));

    const second = await judgeAnswerability(QUESTION, expanded);
    console.log(`\n第二次判官：${second.accounting.outcome} answerable=${second.accounting.answerable}` +
        ` (${second.accounting.latencyMs} ms, ${second.accounting.usage.promptTokens}+${second.accounting.usage.completionTokens} tokens)`);
    console.log(`必答项：${second.accounting.requirements.map(r => `${r.law}#${r.articleNo}`).join('、') || '（无）'}`);
    console.log(`理由：${second.accounting.reason ?? '（无）'}`);
    console.log(`\n结果：${second.accounting.answerable ? '恢复为可答' : '仍然拒答'}`);
    console.log(`两次合计：${first.accounting.usage.promptTokens + second.accounting.usage.promptTokens} prompt + ` +
        `${first.accounting.usage.completionTokens + second.accounting.usage.completionTokens} completion tokens`);
}

main().catch(e => { console.error(e); process.exit(1); });
