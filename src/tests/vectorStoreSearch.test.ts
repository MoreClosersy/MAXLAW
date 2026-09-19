/**
 * 检索排序的行为契约。
 *
 * 这四条全部对应评测第一次跑出来的真实 bug（见 roadmap §10.7），而且**都不是读代码能看出来的**：
 * 一个字符串比较、一个布尔排序键、一批当普通 chunk 建的标题。锁进单元测试，
 * 是因为它们只在特定查询形状下才暴露，靠人肉回归必然漏。
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import VectorStore from '../VectorStore.js';
import { ParsedChunk } from '../EmbeddingRetriever.js';

/** 查询向量固定为 [1,0]，于是每条文档的余弦就等于它的第一个分量。 */
const QUERY = [1, 0];

/**
 * 造一条 chunk。
 * `sim` 直接写成余弦值，好让"哪条更相似"一眼可见——用例里的数字取自真实评测，
 * 所以测试读起来就是那次的现场：第577条的余弦(0.584)确实**低于**无关条文的(1.0)。
 */
function chunk(
    source: string,
    articleNo: string | null,
    sim: number,
    kind: ParsedChunk['kind'] = 'article',
): ParsedChunk {
    const label = articleNo ?? source;
    return {
        document: articleNo ? `${articleNo} ……（正文）` : label,
        source,
        articleNo,
        chapter: '第一编 总则',
        kind,
        hash: `${source}:${label}`,
        embedText: articleNo ? `${articleNo} ……（正文）` : label,
    };
}

async function store(entries: { c: ParsedChunk; sim: number }[]): Promise<VectorStore> {
    const s = new VectorStore(2);
    for (const { c, sim } of entries) {
        // 余弦 = 第一分量，所以向量取 [sim, sqrt(1-sim²)]
        await s.addEmbedding([sim, Math.sqrt(1 - sim * sim)], c);
    }
    return s;
}

/** 民法典#176 余弦 1.0 —— 比真答案还高，这正是当初条号置顶要解决的问题 */
const MINFADIAN_176 = chunk('民法典.md', '第一百七十六条', 1.0);
/** 民法典#577 余弦 0.584 —— 实测值：它自己的正文与"关于它的问题"并不相似 */
const MINFADIAN_577 = chunk('民法典.md', '第五百七十七条', 0.584);
/** 公司法#23 与民法典#23 条号相同、法律不同 —— 条号在法律之间不唯一 */
const GONGSIFA_23 = chunk('公司法.md', '第二十三条', 0.5);
const MINFADIAN_23 = chunk('民法典.md', '第二十三条', 0.9);
/** 章标题：余弦 0.99，整章语义的摘要，对任何牵涉该章的查询都高分 */
const HEADING = chunk('民法典.md', null, 0.99, 'heading');
/** 合同法的内容其实在民法典第三编里，语料中没有 合同法.md */
const MINFADIAN_107 = chunk('民法典.md', '第一百零七条', 0.6);

const ALL = [
    { c: MINFADIAN_176, sim: 1.0 },
    { c: MINFADIAN_577, sim: 0.584 },
    { c: GONGSIFA_23, sim: 0.5 },
    { c: MINFADIAN_23, sim: 0.9 },
    { c: HEADING, sim: 0.99 },
    { c: MINFADIAN_107, sim: 0.6 },
];

const titles = (r: { articleNo: string | null; source: string }[]) =>
    r.map(x => `${x.source}#${x.articleNo ?? '(标题)'}`);

test('章标题不进检索池 —— 它是整章语义的摘要，会系统性挤掉真条文', async () => {
    const s = await store(ALL);
    const hits = await s.search(QUERY, 5, { articleNos: [], lawNames: [] });
    // 标题余弦 0.99 排第二高，若不是被排除，它一定出现在结果里
    assert.ok(!titles(hits).includes('民法典.md#(标题)'), `标题混进了结果：${titles(hits)}`);
});

test('精确条号查询置顶 —— 即使它自己的余弦低于无关条文', async () => {
    const s = await store(ALL);
    const hits = await s.search(QUERY, 5, { articleNos: [577], lawNames: ['民法典'] });
    assert.equal(hits[0].articleNo, '第五百七十七条', `实际首位是 ${hits[0].articleNo}`);
    assert.ok(hits[0].exactArticleHit);
    // 无关条文的余弦(1.0)最高，但必须排在真答案后面
    assert.equal(hits[1].articleNo, '第一百七十六条');
});

test('条号不唯一：点名了法律就只在那部法里认这个条号', async () => {
    const s = await store(ALL);
    const hits = await s.search(QUERY, 5, { articleNos: [23], lawNames: ['公司法'] });
    assert.equal(hits[0].source, '公司法.md', `实际首位来自 ${hits[0].source}`);
    assert.equal(hits[0].articleNo, '第二十三条');
});

test('置顶不是"把别的候选踢出去" —— 未置顶的条号命中仍留在结果里', async () => {
    const s = await store(ALL);
    const hits = await s.search(QUERY, 5, { articleNos: [23], lawNames: ['公司法'] });
    // 民法典#23 没被置顶（用户点名的是公司法），但它余弦最高，仍应作为普通候选出现。
    // 曾经写成 else 分支把它整个丢掉，而"静默消失的候选"是最难排查的一类问题。
    assert.ok(titles(hits).includes('民法典.md#第二十三条'), `民法典#23 被丢弃：${titles(hits)}`);
});

test('点名了语料里没有的法律时，不能把正确答案一起丢掉', async () => {
    const s = await store(ALL);
    // 「合同法」在 KNOWN_LAW_NAMES 里，但语料里没有 合同法.md——合同法即民法典第三编。
    // 若把"未置顶的条号命中"丢弃，民法典#107 这个正确答案就会消失。
    const hits = await s.search(QUERY, 5, { articleNos: [107], lawNames: ['合同法'] });
    assert.ok(titles(hits).includes('民法典.md#第一百零七条'), `正确答案被丢弃：${titles(hits)}`);
});

test('未点名条号时，排序就是余弦 —— 不做任何分层', async () => {
    const s = await store(ALL);
    const hits = await s.search(QUERY, 5, { articleNos: [], lawNames: [] });
    assert.deepEqual(titles(hits), [
        '民法典.md#第一百七十六条',
        '民法典.md#第二十三条',
        '民法典.md#第一百零七条',
        '民法典.md#第五百七十七条',
        '公司法.md#第二十三条',
    ]);
});
