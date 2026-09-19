/**
 * 批量嵌入的行为契约。
 *
 * 批量化把全量索引从 1681 次串行请求压到约 105 次（实测单条与批量返回逐元素最大差
 * 1.2e-7、余弦精确 1.0，是 float32 精度内的同一个向量）。它快，但也引入了一种
 * **静默**故障：向量与 chunk 错位。错位的表现不是报错，而是"检索结果看着不相关"——
 * 和当初那个维度不一致导致排序错乱的 bug 是同一类。所以这里断言的核心是**顺序**：
 * 响应少一条、乱序、维度不对，都必须被识别出来并退回逐条，而不是硬塞进去。
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import EmbeddingRetriever from '../EmbeddingRetriever.js';

const MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2';
const DIM = 384;

/** 用一个可辨认的向量代表第 n 条文本：只有第 (n % DIM) 位是 1。 */
function vecFor(n: number): number[] {
    const v = new Array(DIM).fill(0);
    v[n % DIM] = 1;
    return v;
}

function stubFetch(impl: typeof fetch): () => void {
    const original = globalThis.fetch;
    globalThis.fetch = impl;
    return () => { globalThis.fetch = original; };
}

function okResponse(body: unknown): Response {
    return {
        ok: true, status: 200, statusText: 'OK',
        text: async () => JSON.stringify(body),
    } as unknown as Response;
}

function makeRetriever() {
    process.env.EMBEDDING_API_KEY = 'hf_test';
    process.env.EMBEDDING_BASE_URL = 'https://example.invalid/hf-inference';
    return new EmbeddingRetriever(MODEL);
}

/** 唯一化文本，避免命中上一次测试留下的磁盘缓存 */
const uniq = (s: string) => `${s}-${Math.random()}`;

test('批量返回的向量与入参顺序严格对应', async () => {
    const texts = [uniq('甲'), uniq('乙'), uniq('丙')];
    let sentInputs: unknown = null;
    const restore = stubFetch((async (_url: string, init: any) => {
        sentInputs = JSON.parse(init.body).inputs;
        // 服务端按入参顺序返回 → 第 i 条给 vecFor(i)
        return okResponse(texts.map((_, i) => vecFor(i)));
    }) as unknown as typeof fetch);

    try {
        const r = makeRetriever();
        const out = await r.embedBatch(texts);
        assert.deepEqual(sentInputs, texts, '应一次请求发出全部文本');
        assert.equal(out.length, 3);
        for (let i = 0; i < 3; i++) {
            assert.equal(out[i][i], 1, `第 ${i} 条应拿到 vecFor(${i})——取错位就是静默故障`);
        }
        assert.equal(r.getStats().embedRequests, 1, '3 条应合成 1 次请求');
        assert.equal(r.getStats().embedCalls, 3);
    } finally { restore(); }
});

test('响应少一条时整批作废并退回逐条，而不是把后一条错位顶上', async () => {
    const texts = [uniq('甲'), uniq('乙'), uniq('丙')];
    let calls = 0;
    const restore = stubFetch((async (_url: string, init: any) => {
        calls++;
        const inputs = JSON.parse(init.body).inputs;
        // 批量请求故意只回 2 条（少一条）；逐条请求（inputs 是字符串）正常回一条
        if (Array.isArray(inputs)) return okResponse([vecFor(0), vecFor(1)]);
        return okResponse([vecFor(9)]);
    }) as unknown as typeof fetch);

    try {
        const r = makeRetriever();
        const out = await r.embedBatch(texts);
        assert.equal(calls, 4, '1 次批量失败 + 3 次逐条');
        for (let i = 0; i < 3; i++) {
            assert.equal(out[i][9], 1, `第 ${i} 条应来自逐条回退（vecFor(9)），实际 ${out[i].findIndex((x: number) => x === 1)}`);
        }
        assert.equal(r.getStats().dimensionMismatches, 0);
    } finally { restore(); }
});

test('OpenAI 兼容端点返回乱序 data 时按 index 还原', async () => {
    const texts = [uniq('甲'), uniq('乙'), uniq('丙')];
    const restore = stubFetch((async () => okResponse({
        data: [
            { index: 2, embedding: vecFor(2) },
            { index: 0, embedding: vecFor(0) },
            { index: 1, embedding: vecFor(1) },
        ],
    })) as unknown as typeof fetch);

    try {
        const r = makeRetriever();
        // 构造之后再切 key：requestBatch 是调用时才读环境变量，
        // 而 makeRetriever() 会把 key 重置成 hf_test —— 第一版测试就栽在这。
        process.env.EMBEDDING_API_KEY = 'sk_test';
        const out = await r.embedBatch(texts);
        for (let i = 0; i < 3; i++) {
            assert.equal(out[i][i], 1, `乱序响应下第 ${i} 条仍应拿到 vecFor(${i})`);
        }
    } finally { restore(); }
});

test('维度不符的批量响应不被写入缓存', async () => {
    const texts = [uniq('甲'), uniq('乙')];
    const restore = stubFetch((async (_url: string, init: any) => {
        const inputs = JSON.parse(init.body).inputs;
        if (Array.isArray(inputs)) return okResponse([vecFor(0), new Array(7).fill(0.1)]);
        return okResponse([vecFor(5)]);
    }) as unknown as typeof fetch);

    try {
        const r = makeRetriever();
        const out = await r.embedBatch(texts);
        // 整批作废 → 逐条回退
        assert.equal(out[0][5], 1);
        assert.equal(out[1][5], 1);
        assert.ok(r.getStats().dimensionMismatches >= 1, '维度不符必须被计数，否则线上无法发现');
    } finally { restore(); }
});

test('已缓存的文本不再发请求', async () => {
    const texts = [uniq('甲'), uniq('乙')];
    let calls = 0;
    const restore = stubFetch((async (_url: string, init: any) => {
        calls++;
        const inputs = JSON.parse(init.body).inputs;
        const n = Array.isArray(inputs) ? inputs.length : 1;
        return okResponse(Array.from({ length: n }, (_, i) => vecFor(i)));
    }) as unknown as typeof fetch);

    try {
        const r = makeRetriever();
        await r.embedBatch(texts);
        assert.equal(calls, 1);
        await r.embedBatch(texts);          // 第二次全部命中缓存
        assert.equal(calls, 1, '第二次不该再发请求');
        assert.equal(r.getStats().cacheHits, 2);
    } finally { restore(); }
});
