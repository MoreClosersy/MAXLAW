/**
 * 引用校验器的单元测试。
 *
 * 用 node:test 而不是引入 vitest/jest：这个项目的测试需求只有"跑几个纯函数断言"，
 * 为它拉一整套测试框架（含 ESM/TS 转译配置）不划算，而且会拖慢 CI。
 * `tsx --test` 直接跑 TypeScript，零额外依赖。
 *
 * 覆盖重点是**假阳性**：一个把真条号报成编造的校验器比没有校验器更糟，
 * 因为它会训练用户忽略告警。所以每种"看起来像幻觉但其实不是"的情况都要有测试。
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import VectorStore, { RetrievedChunk, VectorStoreItem } from '../VectorStore.js';
import { verifyCitations, summarizeCitations } from '../CitationVerifier.js';

const DIM = 2;
const dummy = [1, 0];

async function makeStore(entries: { articleNo: string; source: string; document?: string }[]): Promise<VectorStore> {
    const store = new VectorStore(DIM);
    for (const e of entries) {
        await store.addEmbedding(dummy, {
            document: e.document ?? `${e.articleNo} ……（条文正文）`,
            source: e.source,
            articleNo: e.articleNo,
            chapter: '第一编 总则',
            kind: 'article',
            hash: `${e.source}:${e.articleNo}`,
            // 这个测试只关心条号抽取，走的是 document；embedText 与嵌入无关，
            // 直接跟 document 相同即可（生产里它是 document 前面拼了章节路径）
            embedText: e.document ?? `${e.articleNo} ……（条文正文）`,
        });
    }
    return store;
}

function chunk(articleNo: string | null, source: string, document: string): RetrievedChunk {
    return {
        document, source, articleNo, chapter: null, hash: `${source}:${articleNo}`,
        // 这个测试只关心条号抽取，读的是 document；embedText 只影响嵌入与精排
        embedText: document, score: 0.9, exactArticleHit: false,
    };
}

const find = (report: ReturnType<typeof verifyCitations>, articleNo: number) =>
    report.checks.find(c => c.articleNo === articleNo);

test('阿拉伯数字与中文数字视为同一条 —— 校验器最关键的假阳性防线', async () => {
    const store = await makeStore([{ articleNo: '第一百一十条', source: '民法典.md' }]);
    const retrieved = [chunk('第一百一十条', '民法典.md', '第一百一十条 民事主体的人身权利……')];

    const report = verifyCitations('依据《民法典》第110条的规定，自然人享有名誉权。', store, retrieved);

    assert.equal(report.total, 1);
    assert.equal(find(report, 110)!.status, 'verified',
        '模型写「第110条」、语料存「第一百一十条」，必须判为同一条');
    assert.equal(report.hasProblem, false);
});

test('条号真实存在但未出现在检索结果中 → not_in_context（凭记忆引用）', async () => {
    const store = await makeStore([
        { articleNo: '第五百七十七条', source: '民法典.md' },
        { articleNo: '第一百一十条', source: '民法典.md' },
    ]);
    // 只检索到 110 条，模型却引了 577 条
    const retrieved = [chunk('第一百一十条', '民法典.md', '第一百一十条 自然人享有名誉权……')];

    const report = verifyCitations('根据《民法典》第577条，违约方应承担继续履行等责任。', store, retrieved);

    const c = find(report, 577)!;
    assert.equal(c.status, 'not_in_context');
    assert.equal(c.exists, true, '条号在语料里是存在的');
    assert.equal(c.inContext, false);
    assert.equal(c.sources[0], '民法典.md');
    assert.equal(report.fabricated, 0, '存在但没检索到，不等于编造');
    assert.equal(report.notInContext, 1);
    assert.equal(report.hasProblem, true);
});

test('条号在语料中不存在 → fabricated', async () => {
    const store = await makeStore([{ articleNo: '第一百一十条', source: '民法典.md' }]);
    const report = verifyCitations('根据《民法典》第九千九百九十九条的规定……', store, []);

    const c = find(report, 9999)!;
    assert.equal(c.status, 'fabricated');
    assert.equal(c.exists, false);
    assert.deepEqual(c.sources, []);
    assert.equal(report.fabricated, 1);
});

test('检索到的条文正文里的交叉引用，也算出现在 context 里', async () => {
    const store = await makeStore([
        { articleNo: '第五百七十七条', source: '民法典.md' },
        { articleNo: '第五百八十四条', source: '民法典.md' },   // 存在，但本次没检索到
    ]);
    // 第577条的正文里写着"依照本法第X条"这种交叉引用，模型顺着它往下写是正确的
    const retrieved = [chunk('第五百七十七条', '民法典.md',
        '第五百七十七条 当事人一方不履行合同义务的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。')];

    const report = verifyCitations(
        '《民法典》第577条规定了违约责任，具体的赔偿范围计算另见第584条。',
        store,
        retrieved,
    );

    // 577 命中 chunk 自身条号
    assert.equal(find(report, 577)!.status, 'verified');
    // 584 不在 context 里，也没被检索到 → 如实报 not_in_context
    assert.equal(find(report, 584)!.status, 'not_in_context');
});

test('答案里没有条号 → 不报问题（不能把"没引用"当成错误）', async () => {
    const store = await makeStore([{ articleNo: '第一条', source: '民法典.md' }]);
    const report = verifyCitations('这个问题的答案取决于具体合同条款，建议咨询律师。', store, []);

    assert.equal(report.total, 0);
    assert.equal(report.hasProblem, false);
    assert.equal(summarizeCitations(report), null);
});

test('同一章内异号同文的合法重复条款，两个条号都能被查到', async () => {
    // 民法典第960条（行纪合同）与第966条（中介合同）章末都是同一句兜底条款，
    // 这是法典本身的结构，不是抄录错误 —— 校验器必须两条都认。
    const body = '本章没有规定的，参照适用委托合同的有关规定。';
    const store = await makeStore([
        { articleNo: '第九百六十条', source: '民法典.md', document: `第九百六十条 ${body}` },
        { articleNo: '第九百六十六条', source: '民法典.md', document: `第九百六十六条 ${body}` },
    ]);
    const retrieved = [
        chunk('第九百六十条', '民法典.md', `第九百六十条 ${body}`),
        chunk('第九百六十六条', '民法典.md', `第九百六十六条 ${body}`),
    ];

    const report = verifyCitations('《民法典》第960条与第966条均规定：' + body, store, retrieved);

    assert.equal(report.total, 2);
    assert.equal(report.verified, 2);
    assert.equal(report.hasProblem, false);
});

test('同一答案里混有真条号与编造条号时，只报编造的那条', async () => {
    const store = await makeStore([
        { articleNo: '第五百七十七条', source: '民法典.md' },
        { articleNo: '第一百四十三条', source: '民法典.md' },
    ]);
    const retrieved = [
        chunk('第五百七十七条', '民法典.md', '第五百七十七条 当事人一方不履行合同义务的……'),
        chunk('第一百四十三条', '民法典.md', '第一百四十三条 具备下列条件的民事法律行为有效……'),
    ];

    const report = verifyCitations(
        '《民法典》第143条规定了民事法律行为的有效要件，第577条规定了违约责任，另见第8888条。',
        store, retrieved,
    );

    assert.equal(report.total, 3);
    assert.equal(report.verified, 2);
    assert.equal(report.fabricated, 1);
    // 编造的排在最前面，用户先看到最该看的
    assert.equal(report.checks[0].articleNo, 8888);
});

test('摘要文案区分两种失败原因', async () => {
    const store = await makeStore([
        { articleNo: '第一条', source: '民法典.md' },
        { articleNo: '第二条', source: '民法典.md' },   // 存在，但本次没检索到
    ]);
    const retrieved = [chunk('第一条', '民法典.md', '第一条 为了保护民事主体的合法权益……')];
    const report = verifyCitations('第1条、第2条、第7777条均有规定。', store, retrieved);

    const s = summarizeCitations(report)!;
    assert.match(s, /不存在/, '编造的要说"不存在"');
    assert.match(s, /未出现在检索结果中/, '凭记忆引用的要说"未出现在检索结果中"');
});

// ---------------------------------------------------------------------------
// 跨法同名条：只按条号匹配会把「凭记忆引用」判成通过
//
// 民法典和公司法都有第二十三条。旧实现只看条号在不在 context 里，所以答案写
// 「公司法第二十三条」、而上下文只有民法典第二十三条时，会报 verified ——
// 用户以为这条经过检索，其实没有。这一组把 (法, 条号) 绑在一起。
// ---------------------------------------------------------------------------

test('点名了法：上下文只有另一部法的同名条 → not_in_context，不是 verified', async () => {
    const store = await makeStore([
        { articleNo: '第二十三条', source: '民法典.md' },
        { articleNo: '第二十三条', source: '公司法.md' },
    ]);
    const retrieved = [chunk('第二十三条', '民法典.md', '第二十三条 民事主体……')];

    const report = verifyCitations('依据公司法第二十三条，股东滥用法人独立地位的……', store, retrieved);

    const c = find(report, 23)!;
    assert.equal(c.law, '公司法');
    assert.equal(c.status, 'not_in_context', '跨法的同名条不算是同一条');
    assert.equal(c.exists, true, '公司法#23 在语料里确实存在');
    assert.equal(c.inContext, false);
    assert.deepEqual(c.foundIn, []);
});

test('点名了法：该法同名条也在 context 里 → verified，且 foundIn 只记那部法', async () => {
    const store = await makeStore([
        { articleNo: '第二十三条', source: '民法典.md' },
        { articleNo: '第二十三条', source: '公司法.md' },
    ]);
    const retrieved = [
        chunk('第二十三条', '民法典.md', '第二十三条 民事主体……'),
        chunk('第二十三条', '公司法.md', '第二十三条 公司股东滥用……'),
    ];

    const report = verifyCitations('依据公司法第二十三条……', store, retrieved);

    const c = find(report, 23)!;
    assert.equal(c.status, 'verified');
    assert.deepEqual(c.foundIn, ['公司法.md']);
});

test('同一答案里两条引用分属两部法：各认各的，不互相串味', async () => {
    const store = await makeStore([
        { articleNo: '第五百七十七条', source: '民法典.md' },
        { articleNo: '第二十三条', source: '公司法.md' },
    ]);
    const retrieved = [
        chunk('第五百七十七条', '民法典.md', '第五百七十七条 当事人一方不履行……'),
        chunk('第二十三条', '公司法.md', '第二十三条 公司股东滥用……'),
    ];

    const report = verifyCitations(
        '《民法典》第五百七十七条规定违约责任；《公司法》第二十三条规定法人人格否认。',
        store, retrieved,
    );

    assert.equal(report.total, 2);
    assert.equal(report.verified, 2);
    const byNo = new Map(report.checks.map(c => [c.articleNo, c]));
    assert.equal(byNo.get(577)!.law, '民法典');
    assert.equal(byNo.get(23)!.law, '公司法');
});

test('没点名法律 + 条号跨法：仍按旧语义判 verified，但标出 ambiguousLaw', async () => {
    const store = await makeStore([
        { articleNo: '第二十三条', source: '民法典.md' },
        { articleNo: '第二十三条', source: '公司法.md' },
    ]);
    const retrieved = [chunk('第二十三条', '民法典.md', '第二十三条 民事主体……')];

    const report = verifyCitations('依据第二十三条……', store, retrieved);

    const c = find(report, 23)!;
    assert.equal(c.law, null);
    assert.equal(c.status, 'verified');
    assert.equal(c.ambiguousLaw, true, '条号跨法时 verified 的把握更小，要能看见');
});

test('点名了法：该法在库里、但没有这一条 → fabricated（条号在别的法里存在也不算）', async () => {
    const store = await makeStore([
        { articleNo: '第十一条', source: '公司法.md' },
        { articleNo: '第九十九条', source: '民法典.md' },   // 号码存在，但不在公司法
    ]);
    const retrieved = [chunk('第十一条', '公司法.md', '第十一条 法定代表人……')];

    const report = verifyCitations('依据公司法第九十九条……', store, retrieved);

    const c = find(report, 99)!;
    assert.equal(c.law, '公司法');
    assert.equal(c.status, 'fabricated');
    assert.equal(c.exists, false);
});

test('法名在上一句：不会把下一句的引用绑过去', async () => {
    const store = await makeStore([
        { articleNo: '第五百七十七条', source: '民法典.md' },
        { articleNo: '第二十三条', source: '公司法.md' },
    ]);
    const retrieved = [chunk('第二十三条', '公司法.md', '第二十三条 公司股东滥用……')];

    const report = verifyCitations(
        '《民法典》第五百七十七条讲了违约责任。另见第二十三条。',
        store, retrieved,
    );

    const c = find(report, 23)!;
    assert.equal(c.law, null, '第二十三条在另一个分句里，没有点名法律');
    assert.equal(c.status, 'verified');
});

// ---------------------------------------------------------------------------
// 内容绑定：条号真、也在 context 里，但引用旁边那段话其实属于另一条
//
// 这是 faithfulness 的 article_swap（判官只抓到 7/22）。确定性规则见
// CitationVerifier.ts 的 CONTENT_*：own 低、bestOther 明显更高时才报，
// 避免把"法律答案本来就会改写"误报成张冠李戴。
// ---------------------------------------------------------------------------

test('content_mismatch：句子讲的其实是另一条被检索到的条文', async () => {
    const breach = '第五百七十七条 当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。';
    const marital = '第一千零六十二条 夫妻在婚姻关系存续期间所得的下列财产，为夫妻的共同财产，归夫妻共同所有：（一）工资、奖金、劳务报酬。';
    const store = await makeStore([
        { articleNo: '第五百七十七条', source: '民法典.md', document: breach },
        { articleNo: '第一千零六十二条', source: '民法典.md', document: marital },
    ]);
    const retrieved = [
        chunk('第五百七十七条', '民法典.md', breach),
        chunk('第一千零六十二条', '民法典.md', marital),
    ];

    const report = verifyCitations(
        '根据民法典第五百七十七条，夫妻在婚姻关系存续期间所得的工资、奖金和劳务报酬，归夫妻共同所有。',
        store, retrieved,
    );

    const c = find(report, 577)!;
    assert.equal(c.status, 'content_mismatch');
    assert.ok(c.contentOverlap!.bestOther > c.contentOverlap!.own, '另一条的覆盖率应当更高');
    assert.equal(report.contentMismatch, 1);
    assert.equal(report.hasProblem, true);
    assert.match(summarizeCitations(report)!, /对不上/);
});

test('内容与所引条文一致 → verified，不报 content_mismatch', async () => {
    const breach = '第五百七十七条 当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。';
    const store = await makeStore([{ articleNo: '第五百七十七条', source: '民法典.md', document: breach }]);
    const retrieved = [chunk('第五百七十七条', '民法典.md', breach)];

    const report = verifyCitations(
        '根据民法典第五百七十七条，一方不履行合同义务的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。',
        store, retrieved,
    );
    assert.equal(find(report, 577)!.status, 'verified');
    assert.equal(report.contentMismatch, 0);
});

test('只有标题式引用 → 文本不足，不判 content_mismatch', async () => {
    const breach = '第五百七十七条 当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。';
    const store = await makeStore([{ articleNo: '第五百七十七条', source: '民法典.md', document: breach }]);
    const retrieved = [chunk('第五百七十七条', '民法典.md', breach)];

    const report = verifyCitations('### 民法典第五百七十七条\n\n（条文略）', store, retrieved);
    assert.equal(find(report, 577)!.status, 'verified');
});
