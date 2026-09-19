/**
 * 锁定「送去嵌入的文本」与「给用户看的正文」是**两个字段**这件事。
 *
 * 背景：民法典第960条（行纪合同章）与第966条（中介合同章）只差一个条号，
 * 26 个字里只有"六十六"那一个字的区别。真正的区分信息在所属章上，
 * 而章节名根本不在正文里——于是问"中介合同这一章没有规定的…"时，
 * 错误的 960 以 0.8001 压过正确的 966 的 0.7950。
 * 这类缺陷只能靠"嵌入文本必须带上章节路径"的断言挡住。
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { splitIntoChunks } from '../EmbeddingRetriever.js';
import { parseArticleNo } from '../articleNo.js';

/** 精简版民法典骨架，层级与真实语料一致：# 法律 / ## 编 / ### 章 / #### 节 */
const CORPUS = `# 测试民法典

## 第一编 总则

### 第一章 一般规定

第一条 为了保护民事主体的合法权益，制定本法。

## 第四编 人格权

### 第一章 一般规定

第九百八十九条 本编调整因人格权的享有和保护产生的民事关系。

## 第三编 合同

### 第二十五章 行纪合同

第九百六十条 本章没有规定的，参照适用委托合同的有关规定。

### 第二十六章 中介合同

第九百六十六条 本章没有规定的，参照适用委托合同的有关规定。
`;

const chunks = splitIntoChunks(CORPUS, '测试民法典.md');
const articles = chunks.filter(c => c.kind === 'article');
const byNo = (n: number) => articles.find(c => parseArticleNo(c.articleNo) === n)!;

test('embedText 带上编/章路径，document 保持干净', () => {
    const c = byNo(960);
    assert.equal(
        c.embedText,
        '第三编 合同 > 第二十五章 行纪合同\n第九百六十条 本章没有规定的，参照适用委托合同的有关规定。',
    );

    // 这是关键的另一半：章节路径**不能**漏进 document——它会被原样引用给用户，
    // 也会进 CitationVerifier 的条号抽取和 eval 的 ground truth 校对。
    assert.ok(!c.document.includes('第二十五章'), 'document 不能含章节名');
});

test('同一条正文挂在两个不同章下时，embedText 必须不同', () => {
    // 960/966 只差一个条号，所以这里直接构造"正文一字不差"的极端情况：
    // 同一段文字出现在两部不同的法律/章节下。靠 document 是无论如何分不开的。
    const same = splitIntoChunks(
        `# 甲法\n\n## 第一章 甲章\n\n第十条 内容完全一样的一段话。\n\n` +
        `# 乙法\n\n## 第九章 乙章\n\n第十条 内容完全一样的一段话。\n`,
        'x.md',
    ).filter(c => c.kind === 'article');

    assert.equal(same.length, 2);
    assert.equal(same[0].document, same[1].document, '前提：两段正文确实一字不差');
    assert.equal(same[0].hash, same[1].hash, '前提：hash 按正文算，所以也相同');
    assert.notEqual(same[0].embedText, same[1].embedText, '正文相同就只剩章节路径能区分了');

    // 也正因如此，检索结果**不能**按 hash 去重，只能按对象身份去重
    //（见 VectorStore.search 里那段注释）。
});

test('960/966：条号之外完全相同，差异全部来自章节路径', () => {
    const a = byNo(960), b = byNo(966);
    const strip = (s: string) => s.replace(/^第[^条]+条\s*/, '');
    assert.equal(strip(a.document), strip(b.document), '前提：去掉条号后正文一字不差');

    // 条号那一个字的差别不足以让 966 胜出（实测它是输的），
    // 起决定作用的是 embedText 里多出来的章名。
    assert.ok(a.embedText.includes('第二十五章 行纪合同'));
    assert.ok(b.embedText.includes('第二十六章 中介合同'));
});

test('去掉法律名，但保留到足以区分同名末级标题', () => {
    assert.ok(!byNo(960).embedText.includes('测试民法典'),
        '法律名不进嵌入文本（parseQuery 的 lawNames 已结构化处理）');

    // 「第一章 一般规定」在总则编和人格权编下各有一个。只留末级标题的话，
    // 这两条互不相干的条文会共享同一个前缀，凭空产生相似度——所以在真实语料里
    // 数了一遍：「第一章 一般规定」和「第一节 一般规定」各出现在 6 条不同路径下。
    const one = byNo(1), eight = byNo(989);
    assert.match(one.embedText, /^第一编 总则 > 第一章 一般规定\n/);
    assert.match(eight.embedText, /^第四编 人格权 > 第一章 一般规定\n/);
    assert.notEqual(one.embedText, eight.embedText);
});

test('没有章节归属的 chunk：embedText 退化为正文，不产生空行前缀', () => {
    const orphan = splitIntoChunks('第一条 光秃秃的一条，没有任何标题。', 'x.md')
        .filter(c => c.kind === 'article');
    assert.equal(orphan.length, 1);
    assert.equal(orphan[0].chapter, null);
    assert.equal(orphan[0].embedText, orphan[0].document);
    assert.ok(!orphan[0].embedText.startsWith('\n'));
});
