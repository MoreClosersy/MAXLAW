/**
 * 多法问答的召回契约。
 *
 * 这两组用例对应两类缺陷，而且**都只在多法查询下暴露**——单法时它们被
 * 「查询只点名一部法律」这个前提掩盖了：
 *
 * 1. **条号置顶是「法名 × 条号」的笛卡尔积。** 旧实现把 lawNames 和 articleNos
 *    当两个独立集合，条件是「条号命中 ∩ 属于任一被点名法律」。问
 *    「民法典第577条和公司法第二十三条」，民法典第二十三条（用户没问）也会被置顶，
 *    还被标成 exactArticleHit，于是 `shouldSkipJudge` 跳过可答性判官。
 * 2. **topK 是全局排序，没有"每部法至少一条"。** 两个法都点名时，语义更强的那部
 *    可以把另一部整个挤出前 5；而 `hit@K` 只要命中任意一条就报 1.000，
 *    指标对"少了一半"完全沉默。
 *
 * 用 2 维向量把余弦写成第一分量，和 `vectorStoreSearch.test.ts` 同一套写法：
 * 用例里的数字读起来就是"谁更相似"，而不是一堆构造噪声。
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import VectorStore from '../VectorStore.js';
import { parseQuery } from '../EmbeddingRetriever.js';
import type { ParsedChunk } from '../EmbeddingRetriever.js';

/** 查询向量固定为 [1,0]，余弦 = 文档向量第一个分量。 */
const QUERY = [1, 0];

function chunk(source: string, articleNo: string, sim: number): ParsedChunk {
    return {
        document: `${articleNo} ……（正文）`,
        source,
        articleNo,
        chapter: '第一编 总则',
        kind: 'article',
        hash: `${source}:${articleNo}`,
        embedText: `${articleNo} ……（正文）`,
    };
}

async function store(entries: { c: ParsedChunk; sim: number }[]): Promise<VectorStore> {
    const s = new VectorStore(2);
    for (const { c, sim } of entries) {
        await s.addEmbedding([sim, Math.sqrt(1 - sim * sim)], c);
    }
    return s;
}

const titles = (r: { articleNo: string | null; source: string }[]) =>
    r.map(x => `${x.source}#${x.articleNo}`);

// ---------------------------------------------------------------------------
// parseQuery：条号必须跟着它所属的法律
// ---------------------------------------------------------------------------

test('parseQuery：条号归属它前面最近的法律，不跨法串味', () => {
    const signals = parseQuery('民法典第577条和公司法第二十三条分别怎么规定？');
    assert.deepEqual(signals.articlePairs, [
        { law: '民法典', articleNo: 577 },
        { law: '公司法', articleNo: 23 },
    ]);
    assert.deepEqual(signals.articleNos, []);
    assert.deepEqual(signals.lawNames, ['民法典', '公司法']);
});

test('parseQuery：并列的法律名共同管辖同一个条号', () => {
    // 「公司法和民法典第二十三条」= 两家的第二十三条，不是"公司法 + 民法典第二十三条"
    const signals = parseQuery('公司法和民法典第二十三条');
    assert.deepEqual(signals.articlePairs, [
        { law: '公司法', articleNo: 23 },
        { law: '民法典', articleNo: 23 },
    ]);
});

test('parseQuery：没点名法律的条号不猜法律', () => {
    const signals = parseQuery('第577条');
    assert.deepEqual(signals.articleNos, [577]);
    assert.deepEqual(signals.lawNames, []);
});

test('parseQuery：分句之后没再点名法律的条号，不会被上一句的法律绑走', () => {
    const signals = parseQuery('民法典第577条怎么规定？那第23条呢？');
    assert.deepEqual(signals.articlePairs, [{ law: '民法典', articleNo: 577 }]);
    // 第二句里没有法律名 → 退回"在所有法律里查找"，不硬绑民法典
    assert.deepEqual(signals.articleNos, [23]);
});

// ---------------------------------------------------------------------------
// search：配对置顶，不是笛卡尔积
// ---------------------------------------------------------------------------

test('search：多法查询不会把另一部法的同名条号一并置顶', async () => {
    const s = await store([
        { c: chunk('民法典.md', '第五百七十七条', 0.58), sim: 0.58 },
        { c: chunk('民法典.md', '第二十三条', 0.9), sim: 0.9 },
        { c: chunk('公司法.md', '第二十三条', 0.5), sim: 0.5 },
        { c: chunk('民法典.md', '第一百七十六条', 1.0), sim: 1.0 },
    ]);
    const hits = await s.search(QUERY, 5, parseQuery('民法典第577条和公司法第二十三条分别怎么规定？'));
    const byTitle = new Map(titles(hits).map((t, i) => [t, hits[i]]));

    assert.equal(byTitle.get('民法典.md#第五百七十七条')?.exactArticleHit, true);
    assert.equal(byTitle.get('公司法.md#第二十三条')?.exactArticleHit, true);
    // 关键回归：民法典#23 余弦最高(0.9)，但用户点名的是公司法#23，它不该被置顶。
    assert.equal(
        byTitle.get('民法典.md#第二十三条')?.exactArticleHit,
        false,
        `未被点名的同名条号被错误置顶：${titles(hits)}`,
    );
    assert.equal(hits[0].articleNo, '第五百七十七条');
});

// ---------------------------------------------------------------------------
// search / ensureLawCoverage：每部被点名的法律至少留一个位置
// ---------------------------------------------------------------------------

test('search：点名两部法时，另一部不会被全局排序整个挤出 topK', async () => {
    const s = await store([
        { c: chunk('民法典.md', '第一条', 1.0), sim: 1.0 },
        { c: chunk('民法典.md', '第二条', 0.95), sim: 0.95 },
        { c: chunk('民法典.md', '第三条', 0.9), sim: 0.9 },
        { c: chunk('民法典.md', '第四条', 0.85), sim: 0.85 },
        { c: chunk('民法典.md', '第五条', 0.8), sim: 0.8 },
        { c: chunk('公司法.md', '第十一条', 0.2), sim: 0.2 },
    ]);
    const hits = await s.search(QUERY, 5, { articleNos: [], lawNames: ['民法典', '公司法'] });

    assert.equal(hits.length, 5);
    assert.ok(
        hits.some(h => h.source === '公司法.md'),
        `公司法被挤出 topK：${titles(hits)}`,
    );
    // 保底只动尾部：最高分不能被换掉
    assert.equal(hits[0].source, '民法典.md');
});

test('search：另一部法落后太多时不硬凑 —— 那是"本来就不相关"，不是"被挤掉"', async () => {
    const entries: { c: ParsedChunk; sim: number }[] = [];
    for (let i = 0; i < 10; i++) {
        const sim = 1.0 - i * 0.02;
        entries.push({ c: chunk('民法典.md', `第${i + 1}条`, sim), sim });
    }
    // 公司法最佳候补全局第 11 名：连"候补第一位"都排不进，说明它本来就不相关。
    // 真实语料上实测过一次：sem-09 的查询向量下，民法典最佳候补是第 10 名的
    // 第五百七十四条（提存），与问题无关——补进来只会污染上下文。
    entries.push({ c: chunk('公司法.md', '第十一条', 0.1), sim: 0.1 });

    const s = await store(entries);
    const hits = await s.search(QUERY, 5, { articleNos: [], lawNames: ['民法典', '公司法'] });

    assert.equal(hits.length, 5);
    assert.ok(
        hits.every(h => h.source === '民法典.md'),
        `落后的候补被硬塞进来了：${titles(hits)}`,
    );
});

test('search：保底不会把用户点名的条号换出去', async () => {
    const s = await store([
        { c: chunk('民法典.md', '第五百七十七条', 0.1), sim: 0.1 },
        { c: chunk('民法典.md', '第一条', 1.0), sim: 1.0 },
        { c: chunk('民法典.md', '第二条', 0.95), sim: 0.95 },
        { c: chunk('民法典.md', '第三条', 0.9), sim: 0.9 },
        { c: chunk('民法典.md', '第四条', 0.85), sim: 0.85 },
        { c: chunk('公司法.md', '第十一条', 0.2), sim: 0.2 },
    ]);
    const hits = await s.search(QUERY, 5, parseQuery('民法典第577条和公司法'));

    assert.equal(hits[0].articleNo, '第五百七十七条', `置顶项被保底换掉了：${titles(hits)}`);
    assert.ok(hits.some(h => h.source === '公司法.md'), `公司法仍缺席：${titles(hits)}`);
});

test('search：只点名一部法时，保底不介入', async () => {
    const s = await store([
        { c: chunk('民法典.md', '第一条', 1.0), sim: 1.0 },
        { c: chunk('公司法.md', '第十一条', 0.2), sim: 0.2 },
    ]);
    const hits = await s.search(QUERY, 5, { articleNos: [], lawNames: ['民法典'] });
    assert.deepEqual(hits.map(h => h.source), ['民法典.md', '公司法.md']);
});

test('search：点名了语料里没有的法律时，保底不硬凑', async () => {
    const s = await store([
        { c: chunk('民法典.md', '第一条', 1.0), sim: 1.0 },
        { c: chunk('公司法.md', '第十一条', 0.2), sim: 0.2 },
    ]);
    // 合同法是已知法律名，但语料里没有 合同法.md——保底不该把"库里没有"变成"硬塞一条"
    const hits = await s.search(QUERY, 5, { articleNos: [], lawNames: ['合同法'] });
    assert.deepEqual(hits.map(h => h.source), ['民法典.md', '公司法.md']);
});
