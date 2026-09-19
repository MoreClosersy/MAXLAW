/**
 * 拒答判定的纯逻辑测试。
 *
 * 这组测试守三条不变量，每条都是踩过的坑：
 *
 * 1. **坏 JSON 不能默认成任何一边。** 判官返回无法解析的输出时，绝不能被记成"可答"
 *    或"不可答"——那两个方向都是把工具的故障记成判定结论。同一类自欺在 faithfulness
 *    判官那边已经出现过（见 `faithfulness.test.ts` 的头部注释）。
 * 2. **指名必须同时匹配法和条号。** 条号跨法不唯一（民法典第23条 / 公司法第23条），
 *    只比数字会放过"判官说的是公司法第23条、检索到的只有民法典第23条"——而那正是
 *    强制指名要拦住的东西。
 * 3. **失败一律退回相似度判定**，因为那是已测量的现状；退到任何一边都是在没量过的
 *    状态上做决策。
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { PROJECT_ROOT } from '../../src/paths.js';
import {
    decideAbstention,
    shouldSkipJudge,
    parseAnswerabilityOutput,
    crossCheckAnswerability,
    mergeAbstention,
    formatJudgeContext,
    isParseError,
    type JudgeOutcome,
} from '../../src/abstention.js';
import type { RetrievedChunk } from '../../src/VectorStore.js';
import type { ParsedAnswerability } from '../../src/abstention.js';

function chunk(over: Partial<RetrievedChunk> & { source: string; articleNo: string | null }): RetrievedChunk {
    return {
        document: '正文占位',
        chapter: null,
        hash: 'h',
        embedText: '正文占位',
        score: 0.5,
        exactArticleHit: false,
        ...over,
    };
}

const SIM_ANSWERED = { abstained: false, topScore: 0.9, hasExactArticleHit: false, reason: 'answered' as const };
const SIM_ABSTAINED = { abstained: true, topScore: 0.21, hasExactArticleHit: false, reason: 'below_threshold' as const };

// ---------------------------------------------------------------------------
// decideAbstention（这个函数此前零测试覆盖）
// ---------------------------------------------------------------------------
test('decideAbstention: 空检索 → no_hits，且 topScore 是 0 而不是 -Infinity', () => {
    const d = decideAbstention([], 0.35);
    assert.equal(d.abstained, true);
    assert.equal(d.reason, 'no_hits');
    // Math.max(...[]) 是 -Infinity，挡住它的是函数里那个 length > 0 的守卫。
    // 有人"简化"掉那个守卫，症状就是报告里冒出一个 -Infinity 分数。
    assert.equal(d.topScore, 0);
    assert.ok(Number.isFinite(d.topScore), 'topScore 必须是有限数');
});

test('decideAbstention: 低于阈值且无精确命中 → 拒答', () => {
    const d = decideAbstention([chunk({ source: '民法典.md', articleNo: '第一百八十八条', score: 0.2 })], 0.35);
    assert.equal(d.abstained, true);
    assert.equal(d.reason, 'below_threshold');
});

test('decideAbstention: 低于阈值但精确命中条号 → 仍然回答', () => {
    // 「第577条规定了什么」这类查询会命中低分（问题与条文用词重合少），
    // 但用户点名了条号，那就该答，不该拒。exactArticleHit 短路阈值就是为了这个。
    const d = decideAbstention(
        [chunk({ source: '民法典.md', articleNo: '第五百七十七条', score: 0.12, exactArticleHit: true })],
        0.35,
    );
    assert.equal(d.abstained, false);
    assert.equal(d.reason, 'answered');
});

test('decideAbstention: 高于阈值 → 回答', () => {
    const d = decideAbstention([chunk({ source: '民法典.md', articleNo: '第一百八十八条', score: 0.62 })], 0.35);
    assert.equal(d.abstained, false);
    assert.equal(d.reason, 'answered');
});

test('decideAbstention: topScore 取最大值，不是第一个元素', () => {
    const d = decideAbstention([
        chunk({ source: 'a.md', articleNo: '第一条', score: 0.1 }),
        chunk({ source: 'b.md', articleNo: '第二条', score: 0.88 }),
        chunk({ source: 'c.md', articleNo: '第三条', score: 0.3 }),
    ], 0.35);
    assert.equal(d.topScore, 0.88);
});

// ---------------------------------------------------------------------------
// shouldSkipJudge
// ---------------------------------------------------------------------------
test('shouldSkipJudge: 空检出不问判官', () => {
    assert.deepEqual(shouldSkipJudge([], SIM_ABSTAINED), { skip: true, reason: 'no_hits' });
});

test('shouldSkipJudge: 精确命中条号压过相似度拒答', () => {
    const sim = { abstained: true, topScore: 0.1, hasExactArticleHit: true, reason: 'below_threshold' as const };
    assert.deepEqual(shouldSkipJudge([chunk({ source: '民法典.md', articleNo: '第一条' })], sim), {
        skip: true, reason: 'exact_article_hit',
    });
});

test('shouldSkipJudge: 相似度已拒答则不花钱问判官', () => {
    const r = shouldSkipJudge([chunk({ source: '民法典.md', articleNo: '第一条', score: 0.2 })], SIM_ABSTAINED);
    assert.deepEqual(r, { skip: true, reason: 'sim_abstain' });
});

test('shouldSkipJudge: 高分但未精确命中 → 必须问判官（这正是 ood-11 的形状）', () => {
    // ood-11「正当防卫…要负刑事责任吗」逐字命中民法典第181条、分数是全部域外题里最高的，
    // 但那条讲民事责任。高分 + 未点名条号 = 恰恰是需要判官的情形，绝不能跳过。
    const r = shouldSkipJudge([chunk({ source: '民法典.md', articleNo: '第一百八十一条', score: 8.2 })], SIM_ANSWERED);
    assert.deepEqual(r, { skip: false, reason: null });
});

// ---------------------------------------------------------------------------
// parseAnswerabilityOutput —— 反自欺的核心
// ---------------------------------------------------------------------------
test('parseAnswerabilityOutput: 良构的可答输出', () => {
    const r = parseAnswerabilityOutput(JSON.stringify({
        answerable: true, law: '民法典', articleNo: 20,
        quote: '不满八周岁的未成年人为无民事行为能力人',
        reason: '第二十条规定…', missing: null,
    }));
    assert.ok(!isParseError(r));
    assert.equal(r.answerable, true);
    assert.deepEqual(r.cited, { law: '民法典', articleNo: 20 });
    assert.ok(r.quote && r.quote.length > 0);
});

test('parseAnswerabilityOutput: 良构的不可答输出', () => {
    const r = parseAnswerabilityOutput(JSON.stringify({
        answerable: false, law: null, articleNo: null, quote: null,
        reason: '民法典第181条分配的是民事责任，问题问的是刑事责任。',
        missing: '刑法 关于正当防卫超过必要限度应负刑事责任的规定',
    }));
    assert.ok(!isParseError(r));
    assert.equal(r.answerable, false);
    assert.equal(r.cited, null);
    assert.ok(r.missing && r.missing.length > 0);
});

test('parseAnswerabilityOutput: 剥掉 ```json 围栏', () => {
    const r = parseAnswerabilityOutput('```json\n{"answerable":true,"law":"民法典","articleNo":188,"reason":"x"}\n```');
    assert.ok(!isParseError(r));
    assert.deepEqual(r.cited, { law: '民法典', articleNo: 188 });
});

test('parseAnswerabilityOutput: 从前后散文里取出 JSON 对象', () => {
    const r = parseAnswerabilityOutput('好的，我的判断是：{"answerable":true,"law":"公司法","articleNo":23,"reason":"x"} 以上。');
    assert.ok(!isParseError(r));
    assert.deepEqual(r.cited, { law: '公司法', articleNo: 23 });
});

test('parseAnswerabilityOutput: 条号认中文数字写法（钉住 §10.13 的坑 1）', () => {
    // 「同一套条号语义有三个消费方…三处各写一份迟早会漂移」——src/articleNo.ts 的头注释。
    // 判官会写「第二十条」，语料存「第二十条」，模型答案写「第20条」，三种都得认。
    for (const v of ['第二十条', '第20条', '20', 20]) {
        const r = parseAnswerabilityOutput(JSON.stringify({ answerable: true, law: '民法典', articleNo: v, reason: 'x' }));
        assert.ok(!isParseError(r), `articleNo=${JSON.stringify(v)} 应当能解析`);
        assert.equal(r.cited!.articleNo, 20);
    }
});

test('parseAnswerabilityOutput: 坏输入一律 error，绝不默认成任何一边', () => {
    // 每条坏输入后面写的是"为什么它必须是 error"。删测试之前先读一遍：
    // 接受其中任何一条，都会让判官故障静默变成一条判定结论。
    const bad: [string, string][] = [
        ['', '空输出'],
        ['not json at all', '根本不是 JSON'],
        ['{}', '没有 answerable 字段'],
        ['null', 'null 不是对象'],
        ['[1,2,3]', '数组不是对象'],
        ['"可答"', '字符串不是对象'],
        ['{"answerable":"true","law":"民法典","articleNo":20}', 'answerable 写成了字符串，"含糊"不该被当成"干脆"'],
        ['{"answerable":1,"law":"民法典","articleNo":20}', 'answerable 是数字'],
        ['{"answerable":true}', '说了可答但给不出条号 —— 强制指名正是为了拦住这个'],
        ['{"answerable":true,"law":"","articleNo":20}', 'law 是空串'],
        ['{"answerable":true,"law":"民法典"}', '缺 articleNo'],
        ['{"answerable":true,"law":"民法典","articleNo":0}', '条号 0 不存在'],
        ['{"answerable":true,"law":"民法典","articleNo":-3}', '负条号'],
        ['{"answerable":true,"law":"民法典","articleNo":null}', 'articleNo 是 null'],
        ['{"answerable":true,"law":"民法典","articleNo":"不知道"}', '条号无法解析'],
    ];
    for (const [input, why] of bad) {
        const r = parseAnswerabilityOutput(input);
        assert.ok(isParseError(r), `应当报错（${why}），但得到了 ${JSON.stringify(r)}`);
    }
});

test('parseAnswerabilityOutput: 判了不可答却仍写了条号 → 通过，但记录契约违规', () => {
    const r = parseAnswerabilityOutput(JSON.stringify({
        answerable: false, law: '民法典', articleNo: 181, reason: '该条讲民事责任',
    }));
    assert.ok(!isParseError(r));
    assert.equal(r.answerable, false);
    assert.deepEqual(r.cited, { law: '民法典', articleNo: 181 }, '条号要留着，好让交叉校验记下这个违规');
    const check = crossCheckAnswerability(r, [chunk({ source: '民法典.md', articleNo: '第一百八十一条' })]);
    assert.equal(check.refusalNamedArticle, true);
    assert.equal(check.binding, 'not_applicable');
});

// ---------------------------------------------------------------------------
// crossCheckAnswerability —— 本文件最重要的一组
// ---------------------------------------------------------------------------
test('crossCheckAnswerability: 指名的条文在检索结果里 → in_context', () => {
    const parsed = parseAnswerabilityOutput(JSON.stringify({
        answerable: true, law: '民法典', articleNo: 20,
        quote: '不满八周岁的未成年人为无民事行为能力人', reason: 'x',
    }));
    assert.ok(!isParseError(parsed));
    const retrieved = [
        chunk({ source: '民法典.md', articleNo: '第一百八十八条' }),
        chunk({ source: '民法典.md', articleNo: '第二十条', document: '不满八周岁的未成年人为无民事行为能力人，由其法定代理人代理实施民事法律行为。' }),
    ];
    const check = crossCheckAnswerability(parsed, retrieved);
    assert.equal(check.binding, 'in_context');
    assert.equal(check.matchedIndex, 1);
    assert.equal(check.quoteVerified, true);
});

test('crossCheckAnswerability: 条号对但法是另一部 → not_in_context（只比数字的实现会在这里失败）', () => {
    // 民法典第23条和公司法第23条都存在。判官说公司法第23条，而检索到的只有民法典第23条：
    // 它指的不是给它的材料。只按条号数字匹配的实现会判成 in_context，整个强制指名就白做了。
    const parsed = parseAnswerabilityOutput(JSON.stringify({
        answerable: true, law: '公司法', articleNo: 23, reason: 'x',
    }));
    assert.ok(!isParseError(parsed));
    const retrieved = [chunk({ source: '民法典.md', articleNo: '第二十三条' })];
    const check = crossCheckAnswerability(parsed, retrieved);
    assert.equal(check.binding, 'not_in_context', '跨法的同名条号不算是同一个条文');
    assert.equal(check.matchedIndex, null);
});

test('crossCheckAnswerability: 法名带「中华人民共和国」/书名号/`.md` 都要能对上', () => {
    const retrieved = [chunk({ source: '民法典.md', articleNo: '第一百八十八条' })];
    for (const law of ['民法典', '中华人民共和国民法典', '《民法典》', '民法典.md']) {
        const parsed = parseAnswerabilityOutput(JSON.stringify({ answerable: true, law, articleNo: 188, reason: 'x' }));
        assert.ok(!isParseError(parsed));
        assert.equal(crossCheckAnswerability(parsed, retrieved).binding, 'in_context', `${law} 应当能对上`);
    }
});

test('crossCheckAnswerability: 检索结果里没有任何条号 → not_in_context', () => {
    const parsed = parseAnswerabilityOutput(JSON.stringify({ answerable: true, law: '民法典', articleNo: 20, reason: 'x' }));
    assert.ok(!isParseError(parsed));
    const check = crossCheckAnswerability(parsed, [chunk({ source: '民法典.md', articleNo: null })]);
    assert.equal(check.binding, 'not_in_context');
});

test('crossCheckAnswerability: 摘录对不上只记账，不改判定', () => {
    const parsed = parseAnswerabilityOutput(JSON.stringify({
        answerable: true, law: '民法典', articleNo: 20,
        quote: '这段话根本不在条文里，是判官编的', reason: 'x',
    }));
    assert.ok(!isParseError(parsed));
    const retrieved = [chunk({ source: '民法典.md', articleNo: '第二十条', document: '不满八周岁的未成年人为无民事行为能力人。' })];
    const check = crossCheckAnswerability(parsed, retrieved);
    assert.equal(check.binding, 'in_context');
    assert.equal(check.quoteVerified, false, '摘录对不上应当如实记为 false');
    // 但**判定不变**：字符串匹配是脆的（摘录会被截断、条号写法会不一致），
    // faithfulness 判官那边已经吃过这个亏，所以它只做诊断。
    const merged = mergeAbstention(SIM_ANSWERED, { kind: 'judged', parsed, check });
    assert.equal(merged.abstained, false, '摘录对不上不构成拒答理由');
    assert.equal(merged.reason, 'answered');
});

// ---------------------------------------------------------------------------
// mergeAbstention —— 真值表
// ---------------------------------------------------------------------------
function judged(answerable: boolean, retrieved: RetrievedChunk[]): JudgeOutcome {
    const parsed: ParsedAnswerability = {
        answerable,
        cited: answerable ? { law: '民法典', articleNo: 20 } : null,
        quote: null,
        reason: 'x',
        missing: null,
    };
    return { kind: 'judged', parsed, check: crossCheckAnswerability(parsed, retrieved) };
}

const IN_CTX = [chunk({ source: '民法典.md', articleNo: '第二十条' })];

test('mergeAbstention: 判官说可答 → 回答，即使相似度判定本来要拒答', () => {
    const m = mergeAbstention(SIM_ABSTAINED, judged(true, IN_CTX));
    assert.equal(m.abstained, false);
    assert.equal(m.reason, 'answered');
});

test('mergeAbstention: 判官说不可答 → 拒答', () => {
    const m = mergeAbstention(SIM_ANSWERED, judged(false, IN_CTX));
    assert.equal(m.abstained, true);
    assert.equal(m.reason, 'judge_unanswerable');
});

test('mergeAbstention: 说可答但指不出检索结果里的条文 → 完全退回相似度判定', () => {
    const m = mergeAbstention(SIM_ANSWERED, judged(true, [chunk({ source: '民法典.md', articleNo: '第一条' })]));
    assert.deepEqual(m, { ...SIM_ANSWERED, reason: 'judge_binding_mismatch' });
});

test('mergeAbstention: 调用失败 → 完全退回相似度判定', () => {
    const m = mergeAbstention(SIM_ANSWERED, { kind: 'error', error: 'timeout' });
    assert.deepEqual(m, { ...SIM_ANSWERED, reason: 'judge_error_fallback' });
});

test('mergeAbstention: 解析失败 → 完全退回相似度判定，且绝不能把"回答"变成"拒答"', () => {
    const m = mergeAbstention(SIM_ANSWERED, { kind: 'parse_failure', error: '坏 JSON', raw: 'not json' });
    // 这条断言是"无法核验的判定不算判定"在生产层的表述：
    // 判官坏了，系统必须维持它原本就会做的事，而不是替它编一个结论出来。
    assert.equal(m.abstained, SIM_ANSWERED.abstained, '解析失败不得改变 abstained');
    assert.equal(m.reason, 'judge_parse_failure');
});

test('mergeAbstention: 跳过时判定完全由相似度规则决定', () => {
    for (const reason of ['no_hits', 'exact_article_hit', 'sim_abstain'] as const) {
        assert.deepEqual(mergeAbstention(SIM_ANSWERED, { kind: 'skipped', reason }), SIM_ANSWERED);
        assert.deepEqual(mergeAbstention(SIM_ABSTAINED, { kind: 'skipped', reason }), SIM_ABSTAINED);
    }
});

// ---------------------------------------------------------------------------
// formatJudgeContext
// ---------------------------------------------------------------------------
test('formatJudgeContext: 编号块、且**不出现「相关度」**', () => {
    const ctx = formatJudgeContext([
        chunk({ source: '民法典.md', articleNo: '第一百八十八条', chapter: '第一编 总则', document: '向人民法院请求保护民事权利的诉讼时效期间为三年。', score: 0.612 }),
        chunk({ source: '公司法.md', articleNo: '第二十三条', document: '公司股东滥用法人独立地位…', score: 0.44 }),
    ]);
    assert.ok(ctx.includes('[1] 来源：民法典.md'));
    assert.ok(ctx.includes('[2] 来源：公司法.md'));
    assert.ok(ctx.includes('章节：第一编 总则'));
    assert.ok(ctx.includes('\n\n---\n\n'), '块之间用 --- 分隔');
    // 这是不复用 formatContext 的**全部意义**：把检索分喂给判官等于把已证明不可校准的
    // 信号从后门放回来，判官会锚在它上面。
    assert.ok(!ctx.includes('相关度'), `判官上下文里不得出现相关度，实际：${ctx}`);
    assert.ok(!ctx.includes('0.612') && !ctx.includes('0.44'), '分数值本身也不得出现');
});

test('formatJudgeContext: 空检出的兜底文案', () => {
    assert.equal(formatJudgeContext([]), '（本地知识库未检索到相关条文）');
});

// ---------------------------------------------------------------------------
// 模块纪律：纯逻辑不能住在有副作用的地方
// ---------------------------------------------------------------------------
test('src/abstention.ts 必须保持零副作用（不读 env、不 import index）', () => {
    // 这条不变量不是洁癖：`src/index.ts` 顶层有 mkdirSync 和会 spawn 子进程的
    // getMcpClients()，所以 decideAbstention 住在那里的时候一行测试都写不了。
    // eval 侧也吃过同款亏：eval/faithfulness.ts 顶层 await loadKnowledge()，
    // 第一次 pnpm test 直接挂了两分钟。
    const raw = fs.readFileSync(path.join(PROJECT_ROOT, 'src/abstention.ts'), 'utf8');
    // 先去注释再断言：注释里**讨论**这些词是好事（本文件的头部就在讨论为什么不能读 env），
    // 会被误伤的检查最后只会被人删掉。剥注释的手法很粗，但本文件里每个 // 都确实是注释。
    const src = raw.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
    assert.ok(!/process\.env/.test(src), 'src/abstention.ts 不得读 process.env');
    assert.ok(!/from\s+['"]\.\/index\.js['"]/.test(src), 'src/abstention.ts 不得 import ./index.js');
    assert.ok(!/from\s+['"](node:)?fs['"]/.test(src), 'src/abstention.ts 不得 import fs');
});
