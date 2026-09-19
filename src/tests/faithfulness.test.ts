/**
 * 判官输出的解析测试。
 *
 * 这组测试守的是一条**反自欺**的不变量：判官返回坏 JSON 时，绝不能被算成"忠实"。
 * 这类评测最常见的自欺方式是——模型偶尔不按格式输出，脚本 catch 住错误、
 * 把这条从分母里悄悄去掉，于是分数反而变好看了（分母小了一点点，但没人注意到）。
 * 所以这里逐条钉住：坏输入必须给出 `error`，而不是一个空 claims 的"成功"。
 *
 * `mutate()` 的测试守的是另一半：注入的缺陷必须**真的注入了**。
 * 目标字符串没匹配上就返回一个和原文一样的"变体"，会让校准的检出率虚高——
 * 判官判它"忠实"是对的，但会被记成"没抓到缺陷"；反过来若变体恰好碰对了，
 * 又会记成"抓到了"。两种都是假的。所以"没改到"必须是 null 而不是原样返回。
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseJudgeOutput, mutate, caughtBy, parseContextBlocks, similarity } from '../../eval/judgeCore.js';

// ---------------------------------------------------------------------------
// parseJudgeOutput
// ---------------------------------------------------------------------------
test('parseJudgeOutput: 正常 JSON', () => {
    const r = parseJudgeOutput(JSON.stringify({
        claims: [
            { text: '诉讼时效为三年', verdict: 'supported', evidence: '第188条' },
            { text: '超过时效就不能起诉了', verdict: 'unsupported', evidence: '无' },
        ],
        summary: '有一条断言没有依据',
    }));
    assert.ok(!('error' in r));
    assert.equal(r.claims.length, 2);
    assert.equal(r.claims[0].verdict, 'supported');
    assert.equal(r.claims[1].evidence, '无');
});

test('parseJudgeOutput: 剥掉 ```json 围栏', () => {
    const r = parseJudgeOutput('```json\n{"claims":[{"text":"甲","verdict":"supported","evidence":"第1条"}],"summary":"ok"}\n```');
    assert.ok(!('error' in r), '带围栏的输出应当能解析');
    assert.equal(r.claims.length, 1);
});

test('parseJudgeOutput: 前后有废话时截取花括号之间', () => {
    const r = parseJudgeOutput('好的，我的判断如下：\n{"claims":[],"summary":"没有断言"}\n希望有帮助。');
    assert.ok(!('error' in r));
    assert.equal(r.claims.length, 0);
});

test('parseJudgeOutput: 坏 JSON 必须报错，绝不能变成"忠实"', () => {
    // 这条是整个文件的核心。若这里返回了 { claims: [] }，
    // 调用方会把它当作"没有断言"而剔除出分母——分数悄悄变好看。
    for (const bad of ['', 'not json at all', '{"claims": "不是数组"}', '[1,2,3]', 'null']) {
        const r = parseJudgeOutput(bad);
        assert.ok('error' in r, `坏输入 ${JSON.stringify(bad)} 必须返回 error`);
    }
});

test('parseJudgeOutput: 丢弃 verdict 非法的条目，保留合法的', () => {
    const r = parseJudgeOutput(JSON.stringify({
        claims: [
            { text: '甲', verdict: 'supported', evidence: '第1条' },
            { text: '乙', verdict: 'MAYBE', evidence: '无' },       // 非法 verdict
            { text: '丙', verdict: 'unsupported', evidence: '无' },
        ],
        summary: 's',
    }));
    assert.ok(!('error' in r));
    assert.equal(r.claims.length, 2);
    assert.deepEqual(r.claims.map(c => c.verdict), ['supported', 'unsupported']);
});

// ---------------------------------------------------------------------------
// mutate
// ---------------------------------------------------------------------------
const CORPUS = '第一百八十八条 向人民法院请求保护民事权利的诉讼时效期间为三年。\n第五百七十七条 当事人一方不履行合同义务的，应当承担违约责任。';

// context 必须用 `formatContext()` 的真实格式（带 `来源：` 和 `---` 分隔）：
// `article_swap` 要靠"来源"判断换过去的是不是**另一部法**的条号。
// 第一版这里用了个手写的假格式，parseContextBlocks 解析不出块，
// article_swap 就会静默不生成——测试必须以生产格式为准。
const CTX = [
    '[1] 来源：民法典.md#第一百八十八条 | 条号：第188条 | 相关度：0.812',
    '向人民法院请求保护民事权利的诉讼时效期间为三年。',
    '',
    '---',
    '',
    '[2] 来源：公司法.md#第二十三条 | 条号：第23条 | 相关度：0.700',
    '公司股东滥用公司法人独立地位和股东有限责任，逃避债务，严重损害公司债权人利益的，应当对公司债务承担连带责任。',
].join('\n');

test('mutate: 四种注入都能生成，且确实改动了文本', () => {
    const answer = '根据第188条，诉讼时效期间为三年。';
    const context = CTX;
    const ms = mutate(answer, context, CORPUS);
    const kinds = ms.map(m => m.kind).sort();
    assert.deepEqual(kinds, ['article_swap', 'distorted_number', 'fabricated_cite', 'unsupported_append']);

    for (const m of ms) {
        assert.notEqual(m.answer, answer, `${m.kind} 没有真的改动答案`);
        assert.ok(m.answer.includes(m.needle) || m.kind === 'distorted_number',
            `${m.kind} 的 needle「${m.needle}」应当出现在变体里`);
    }
});

test('mutate: 编造的条号必须大于语料最大条号', () => {
    const ms = mutate('根据第188条，诉讼时效为三年。', '第188条 诉讼时效期间为三年。', CORPUS);
    const fab = ms.find(m => m.kind === 'fabricated_cite');
    assert.ok(fab, '应当生成 fabricated_cite');
    // 语料最大条号 577 → 编造 577+777 = 1354
    assert.equal(Number(fab!.needle), 577 + 777);
});

test('mutate: 答案与 context 没有共同数目词时不生成 distorted_number', () => {
    // 「八周岁」不在 context 里 → 改了也考不出"数字被篡改"，必须跳过而不是硬造
    const ms = mutate('孩子八周岁。', '第188条 诉讼时效期间为三年。', CORPUS);
    assert.equal(ms.filter(m => m.kind === 'distorted_number').length, 0);
});

test('mutate: 追加的断言其关键短语必须不在语料里', () => {
    const ms = mutate('甲。', '第188条 乙。', CORPUS);
    const app = ms.find(m => m.kind === 'unsupported_append');
    assert.ok(app, '应当生成 unsupported_append');
    assert.ok(!CORPUS.includes(app!.needle),
        `关键短语「${app!.needle}」不该在语料里，否则这条断言可能是有依据的`);
});

// ---------------------------------------------------------------------------
// caughtBy / parseContextBlocks
// ---------------------------------------------------------------------------
test('caughtBy: 只要出现一条 unsupported 就算抓到', () => {
    assert.equal(caughtBy([]), false, '一条断言都没有 → 没抓到');
    assert.equal(caughtBy([{ verdict: 'supported' }, { verdict: 'supported' }]), false);
    assert.equal(caughtBy([{ verdict: 'supported' }, { verdict: 'unsupported' }]), true);
});

test('parseContextBlocks: 解出每块的来源与条号', () => {
    const bs = parseContextBlocks(CTX);
    assert.equal(bs.length, 2);
    assert.equal(bs[0].source, '民法典.md#第一百八十八条');
    assert.equal(bs[0].articleNo, 188);
    assert.equal(bs[1].source, '公司法.md#第二十三条');
    assert.equal(bs[1].articleNo, 23);
});

test('mutate: 同法源且两条用词高度重合时不生成 article_swap', () => {
    // 换过去的那条如果和源条文说的大体是一件事，答案那句话**真的被 context 支持**，
    // "缺陷"就不是缺陷——lookup-02 正是这么翻车的（公司法第23条 → 第232条，
    // 两条都在讲"应当清算/连带责任"这类同一主题）。宁可**不生成**，
    // 也不能生成一个是非难辨的样本。
    const sameText = '公司股东滥用公司法人独立地位和股东有限责任，逃避债务，严重损害公司债权人利益的，应当对公司债务承担连带责任。';
    const oneLaw = [
        '[1] 来源：公司法.md#第二十三条 | 条号：第23条 | 相关度：0.812',
        sameText,
        '',
        '---',
        '',
        '[2] 来源：公司法.md#第二百三十二条 | 条号：第232条 | 相关度：0.700',
        sameText,
    ].join('\n');
    const ms = mutate('根据第23条，股东应当承担连带责任。', oneLaw, CORPUS);
    assert.equal(ms.filter(m => m.kind === 'article_swap').length, 0,
        '用词重合度高的同法源目标不该被选中');
});

test('mutate: 同法源但两条用词不重合时仍生成 article_swap', () => {
    // 兜底路径存在的理由：只在"另一部法"里找目标的话，单法源 context
    // （检索只召回了一部法，很常见）就永远量不到"条号对、内容不对"这一类——
    // 而 prompt 里明说这是**最隐蔽的一种**。量不到就等于没测。
    const oneLaw = [
        '[1] 来源：民法典.md#第一百八十八条 | 条号：第188条 | 相关度：0.812',
        '向人民法院请求保护民事权利的诉讼时效期间为三年。',
        '',
        '---',
        '',
        '[2] 来源：民法典.md#第五百七十七条 | 条号：第577条 | 相关度：0.700',
        '当事人一方不履行合同义务的，应当承担违约责任。',
    ].join('\n');
    const ms = mutate('根据第188条，诉讼时效期间为三年。', oneLaw, CORPUS);
    const sw = ms.find(m => m.kind === 'article_swap');
    assert.ok(sw, '用词不重合的同法源目标应当生成 article_swap');
    assert.ok(sw!.note.includes('同法源'), `note 应说明是兜底路径：${sw!.note}`);
});

test('similarity: 用词重合度', () => {
    assert.equal(similarity('诉讼时效期间为三年', '诉讼时效期间为三年'), 1);
    assert.equal(similarity('诉讼时效期间为三年', '当事人应当承担违约责任'), 0);
    const mid = similarity('公司股东应当承担连带责任', '公司股东可以提起代表诉讼');
    assert.ok(mid > 0 && mid < 1, `部分重合应落在 (0,1)：${mid}`);
});

test('mutate: article_swap 换的是另一部法的条号，且数字写法跟随原答案', () => {
    const ms = mutate('根据第188条，诉讼时效期间为三年。', CTX, CORPUS);
    const sw = ms.find(m => m.kind === 'article_swap');
    assert.ok(sw, '应当生成 article_swap');
    assert.ok(sw!.answer.includes('第23条'), '阿拉伯数字的引用要换成阿拉伯数字，不能突然变成中文数字');
    assert.ok(!sw!.answer.includes('第188条'));
    assert.ok(sw!.note.includes('公司法.md'), `note 应指出换到了另一部法：${sw!.note}`);
});

test('parseContextBlocks: 没有条号的块（章标题一类）不参与换号', () => {
    // `formatContext` 对没有条号的 chunk 不输出「条号：」字段。这些块不能当换号目标——
    // 换过去会变成一个"指向章标题"的引用，缺陷性质就变了（更像 fabricated_cite）。
    const withHeading = [
        '[1] 来源：民法典.md#总则 | 章节：第一章 基本规定 | 相关度：0.900',
        '第一章 基本规定',
        '',
        '---',
        '',
        '[2] 来源：民法典.md#第一百八十八条 | 条号：第188条 | 相关度：0.812',
        '向人民法院请求保护民事权利的诉讼时效期间为三年。',
        '',
        '---',
        '',
        '[3] 来源：民法典.md#第五百七十七条 | 条号：第577条 | 相关度：0.700',
        '当事人一方不履行合同义务的，应当承担违约责任。',
    ].join('\n');
    const bs = parseContextBlocks(withHeading);
    assert.equal(bs.length, 3);
    assert.equal(bs[0].articleNo, null, '章标题块没有条号');
    const sw = mutate('根据第188条，诉讼时效期间为三年。', withHeading, CORPUS)
        .find(m => m.kind === 'article_swap');
    assert.ok(sw, '应当生成 article_swap');
    assert.ok(sw!.answer.includes('第577条'), `换号目标应当是第577条，实际 note：${sw!.note}`);
});

test('caughtBy: 判官解析失败时 claims 为空 → 不算抓到（由调用方单独计数）', () => {
    // 解析失败会走 failed 分支、claims 为空，caughtBy 自然是 false。
    // 这条钉住的是"它不会被误当成抓到"——失败必须显式计入 failed 数，
    // 而不是被这条判据悄悄吸收掉。
    assert.equal(caughtBy([]), false);
});
