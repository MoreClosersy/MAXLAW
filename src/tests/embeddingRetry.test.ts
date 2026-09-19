/**
 * embedding 客户端的重试与快速失败行为。
 *
 * 为什么这些断言值得写下来：一次全量索引要发 1680 个请求，而这条链路实测是
 * **间歇性抖动**的（同一分钟内 curl 5/5 成功、node 连续 35 次 TLS ECONNRESET）。
 * 重试逻辑写错的代价不是"慢一点"，而是整批索引被静默降级成兜底向量——
 * 检索结果全错，但日志上看起来"跑完了"。所以这里断言的是**行为契约**：
 * 哪些错该重试、哪些不该、重试用尽后发生什么。
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import EmbeddingRetriever, { isTransientNetworkError, EmbeddingUnavailableError } from '../EmbeddingRetriever.js';

const MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2';
const DIM = 384;

/** 造一个 undici 风格的 fetch 失败：错误本身是 TypeError，真实原因在 cause.code */
function netError(code: string): Error {
    const cause = Object.assign(new Error(code), { code });
    return Object.assign(new TypeError('fetch failed'), { cause });
}

/** 临时替换 globalThis.fetch，返回还原函数 */
function stubFetch(impl: typeof fetch): () => void {
    const original = globalThis.fetch;
    globalThis.fetch = impl;
    return () => { globalThis.fetch = original; };
}

function okResponse(): Response {
    return {
        ok: true,
        status: 200,
        statusText: 'OK',
        text: async () => JSON.stringify(Array.from({ length: DIM }, (_, i) => (i === 0 ? 1 : 0))),
    } as unknown as Response;
}

/** 每次 new 都会读一遍磁盘缓存，但测试只调 embedQuery，不 flushCache，不落盘 */
function makeRetriever() {
    process.env.EMBEDDING_API_KEY = 'hf_test';
    process.env.EMBEDDING_BASE_URL = 'https://example.invalid/hf-inference';
    return new EmbeddingRetriever(MODEL);
}

test('瞬时网络错误会重试，且重试成功后不产生兜底向量', async () => {
    let calls = 0;
    const restore = stubFetch((async () => {
        calls++;
        if (calls < 3) throw netError('ECONNRESET');
        return okResponse();
    }) as typeof fetch);

    try {
        const r = makeRetriever();
        const vec = await r.embedQuery('测试文本-' + Math.random());
        assert.equal(calls, 3, '前两次 ECONNRESET 应触发重试');
        assert.equal(vec.length, DIM);
        assert.equal(r.getStats().fallbackHits, 0, '重试成功就不该有兜底');
    } finally { restore(); }
});

test('重试用尽后降级为兜底向量（服务器运行时不能因为一次抖动就崩）', async () => {
    let calls = 0;
    const restore = stubFetch((async () => { calls++; throw netError('ECONNRESET'); }) as typeof fetch);

    try {
        const r = makeRetriever();
        const vec = await r.embedQuery('测试文本-' + Math.random());
        assert.equal(calls, 3, '重试次数应等于 EMBEDDING_MAX_ATTEMPTS');
        assert.equal(vec.length, DIM, '仍要返回一个可用的向量，让服务能起来');
        assert.equal(r.getStats().fallbackHits, 1);
    } finally { restore(); }
});

test('严格模式下兜底即抛错（pnpm embed 要的是"要么完整要么明确失败"）', async () => {
    const restore = stubFetch((async () => { throw netError('ECONNRESET'); }) as typeof fetch);

    try {
        const r = makeRetriever();
        r.setStrict(true);
        await assert.rejects(
            () => r.embedQuery('测试文本-' + Math.random()),
            (e: unknown) => e instanceof EmbeddingUnavailableError,
        );
    } finally { restore(); }
});

test('接口返回 4xx/5xx 不重试 —— 密钥错了重试三次还是错', async () => {
    let calls = 0;
    const restore = stubFetch((async () => {
        calls++;
        return { ok: false, status: 401, statusText: 'Unauthorized', text: async () => '' } as unknown as Response;
    }) as typeof fetch);

    try {
        const r = makeRetriever();
        await r.embedQuery('测试文本-' + Math.random());
        assert.equal(calls, 1, '有响应就不该重试，否则把一次明确失败拖成三次');
        assert.equal(r.getStats().fallbackHits, 1);
    } finally { restore(); }
});

test('isTransientNetworkError 只认网络层故障', () => {
    for (const code of ['ECONNRESET', 'ECONNREFUSED', 'ETIMEDOUT', 'ENOTFOUND', 'EAI_AGAIN', 'EPIPE']) {
        assert.equal(isTransientNetworkError(netError(code)), true, `${code} 应可重试`);
    }
    assert.equal(isTransientNetworkError(Object.assign(new Error(), { name: 'TimeoutError' })), true);
    assert.equal(isTransientNetworkError(new Error('业务逻辑错误')), false, '普通异常不该被当成网络抖动');
    assert.equal(isTransientNetworkError(null), false);
});
