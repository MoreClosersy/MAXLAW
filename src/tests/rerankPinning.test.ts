/**
 * 精排不能把"点名条号"的精确命中踢下去。
 *
 * 这条不变量必须被钉住，因为它是**两个各自都对的机制撞在一起**才出问题的：
 *   - `VectorStore.search` 把点名条号的条目置顶（理由：点名时余弦只有 0.584，
 *     反而低于无关条文的 0.742——相似度在这个场景里问错了问题）；
 *   - cross-encoder 按 (query, doc) 相关性重排。
 * 而「民法典第577条规定了什么」这句话里，条号对 cross-encoder 同样是纯噪声，
 * 它完全可能认为第 577 条正文与这句话不相关，把置顶项判到第 4 位。
 * 结果是第 577 条先被置顶、再被重排踢下来——白折腾一趟，还倒退回修好之前。
 *
 * 测试用的是一个**恶意 stub**：它把候选整个倒过来。置顶逻辑只要漏一点，
 * 第 577 条就会从第 1 位掉到最后一位，测试立刻红——不比"分数略有波动"那种软信号。
 *
 * 全部离线：不下载模型、不打 embedding API。
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import EmbeddingRetriever, { ParsedChunk } from '../EmbeddingRetriever.js';
import { RerankerLike, RerankCandidate, RerankResult, partitionPinned } from '../Reranker.js';

const DIM = 384;
/** 查询向量固定为 [1,0,0,…]，于是每条文档的余弦就等于它的第一个分量 */
const QUERY = [1, ...new Array(DIM - 1).fill(0)];
const vec = (sim: number) => [sim, Math.sqrt(1 - sim * sim), ...new Array(DIM - 2).fill(0)];

function chunk(source: string, articleNo: string | null, kind: ParsedChunk['kind'] = 'article'): ParsedChunk {
    return {
        document: articleNo ? `${articleNo} ……（正文）` : source,
        source,
        articleNo,
        chapter: '第一编 总则',
        kind,
        hash: `${source}:${articleNo ?? source}`,
        embedText: articleNo ? `${articleNo} ……（正文）` : source,
    };
}

/** 恶意 stub：把候选**整个倒过来**。任何没被保护的置顶项都会被扔到最后。 */
const adversarial: RerankerLike = {
    async rerank<T extends RerankCandidate>(_q: string, candidates: readonly T[]) {
        return candidates
            .map(c => ({ ...c, rerankScore: 0 }) as T & RerankResult)
            .reverse();
    },
};

/**
 * 建一个只走本地排序的 retriever。
 *
 * 覆写 `embedQuery`：这一步本来要打 embedding API，而本用例断言的是**排序**
 * 不是嵌入。用未登记的模型名让缓存文件不存在，构造就只剩内存操作。
 */
async function makeRetriever(entries: { c: ParsedChunk; sim: number }[]) {
    const r = new EmbeddingRetriever('test/not-a-registered-model', adversarial);
    r.embedQuery = async () => QUERY;
    for (const { c, sim } of entries) {
        await r.getStore().addEmbedding(vec(sim), c);
    }
    return r;
}

test('纯函数：置顶项与可重排项被分开', () => {
    const { pinned, rerankable } = partitionPinned([
        { exactArticleHit: true, n: 1 },
        { exactArticleHit: false, n: 2 },
        { exactArticleHit: true, n: 3 },
    ]);
    assert.deepEqual(pinned.map(x => x.n), [1, 3]);
    assert.deepEqual(rerankable.map(x => x.n), [2]);
});

test('点名条号时：哪怕精排把顺序整个倒过来，第 577 条仍然在第 1 位', async () => {
    const r = await makeRetriever([
        // 无关条文余弦更高（0.742）—— 这正是当初必须置顶的原因
        { c: chunk('民法典.md', '第一百七十六条'), sim: 0.742 },
        // 第 577 条自己的正文与"关于它的问题"并不相似，实测余弦 0.584
        { c: chunk('民法典.md', '第五百七十七条'), sim: 0.584 },
    ]);

    const out = await r.retrieve('民法典第577条规定了什么', 5);

    assert.equal(out[0].articleNo, '第五百七十七条',
        '置顶项被精排踢下去了 —— 点名的条号不是相似度候选，是查找结果');
    assert.equal(out[0].exactArticleHit, true);
    // 置顶项**没有**重排分数，因为按设计它压根没进精排器——不是"碰巧排第一"。
    // 这一条和下面那条一起，把"谁被精排了"钉死成可断言的事实。
    assert.ok(!('rerankScore' in out[0]), '置顶项不该有重排分数：它没进精排器');
    // 而其余候选确实走了精排（否则"没被踢下去"可能只是因为压根没开精排）
    assert.ok('rerankScore' in out[1], '非置顶项没带重排分数，说明精排路径压根没跑');
});

test('没点名条号时：精排说了算，余弦排序被它覆盖', async () => {
    const r = await makeRetriever([
        { c: chunk('民法典.md', '第一百七十六条'), sim: 0.742 },
        { c: chunk('民法典.md', '第五百七十七条'), sim: 0.584 },
    ]);

    const out = await r.retrieve('别人欠我钱一直不还怎么办', 5);

    // 没有置顶项，恶意 stub 的倒序就是最终顺序 —— 精排确实接管了排序
    assert.equal(out[0].articleNo, '第五百七十七条');
    assert.equal(out[0].exactArticleHit, false);
});

test('关闭精排时：返回顺序就是纯余弦，且不带 rerankScore', async () => {
    // reranker = null，模拟"没开精排"
    const off = new EmbeddingRetriever('test/not-a-registered-model', null);
    off.embedQuery = async () => QUERY;
    for (const { c, sim } of [
        { c: chunk('民法典.md', '第一百七十六条'), sim: 0.742 },
        { c: chunk('民法典.md', '第五百七十七条'), sim: 0.584 },
    ]) {
        await off.getStore().addEmbedding(vec(sim), c);
    }

    const out = await off.retrieve('民法典第577条规定了什么', 5);
    assert.equal(out[0].articleNo, '第五百七十七条', '置顶仍然生效（它不依赖精排）');
    assert.equal(out[1].articleNo, '第一百七十六条');
    assert.ok(!('rerankScore' in out[1]), '没开精排却带回了重排分数');
});
