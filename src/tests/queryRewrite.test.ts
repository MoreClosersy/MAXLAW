import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseRewriteOutput, mergeRewrittenResults } from '../QueryRewriter.js';
import type { RetrievedChunk } from '../VectorStore.js';

const chunk = (source: string, articleNo: string, opts: { hash?: string; exact?: boolean } = {}): RetrievedChunk => ({
    document: `${articleNo} ……`,
    source,
    articleNo,
    chapter: null,
    hash: opts.hash ?? `${source}:${articleNo}`,
    embedText: `${articleNo} ……`,
    score: 0.5,
    exactArticleHit: opts.exact ?? false,
});

test('parseRewriteOutput: 合法输出按法保留', () => {
    const out = parseRewriteOutput(JSON.stringify({
        queries: [
            { law: '民法典', query: '越权代表行为的效力与相对人善意' },
            { law: '公司法', query: '公司为他人提供担保 董事会或者股东会决议' },
        ],
    }), ['民法典', '公司法']);
    assert.deepEqual(out, [
        { law: '民法典', query: '越权代表行为的效力与相对人善意' },
        { law: '公司法', query: '公司为他人提供担保 董事会或者股东会决议' },
    ]);
});

test('parseRewriteOutput: 坏 JSON / 缺 queries / 非数组 → 空数组，不抛错', () => {
    for (const raw of ['', 'not json', '{}', '{"queries":"x"}', '[]']) {
        assert.deepEqual(parseRewriteOutput(raw, ['民法典', '公司法']), []);
    }
});

test('parseRewriteOutput: 库外法律、过长/过短查询、重复项都被丢弃；每法最多 2 条', () => {
    const out = parseRewriteOutput(JSON.stringify({
        queries: [
            { law: '刑法', query: '故意伤害的构成要件' },
            { law: '民法典', query: '短' },
            { law: '民法典', query: 'x'.repeat(81) },
            { law: '民法典', query: '越权代表 相对人善意' },
            { law: '民法典', query: '越权代表 相对人善意' },
            { law: '民法典', query: '代表行为有效 相对人知道' },
            { law: '民法典', query: '第三条不该被采纳' },
        ],
    }), ['民法典', '公司法']);
    assert.equal(out.length, 2);
    assert.ok(out.every(x => x.law === '民法典'));
});

test('mergeRewrittenResults: 给缺的那部法补到 2 条，只从原始检索尾部换出', () => {
    const base = [
        chunk('公司法.md', '第十一条'),
        chunk('公司法.md', '第二十三条'),
        chunk('公司法.md', '第十条'),
        chunk('民法典.md', '第六十二条'),
        chunk('公司法.md', '第四条'),
    ];
    const extra = [chunk('民法典.md', '第五百零四条'), chunk('民法典.md', '第五百三十七条')];
    const merged = mergeRewrittenResults(base, extra, ['民法典', '公司法'], 5);

    assert.equal(merged.length, 5);
    assert.ok(merged.some(c => c.source === '民法典.md' && c.articleNo === '第五百零四条'));
    assert.ok(!merged.some(c => c.articleNo === '第四条'), '被换出的应当是尾部冗余项');
    assert.equal(merged.filter(c => c.source === '民法典.md').length, 2, '每部法最多 2 条');
});

test('mergeRewrittenResults: 不换出置顶项', () => {
    const base = [
        chunk('公司法.md', '第十一条'),
        chunk('公司法.md', '第二十三条'),
        chunk('公司法.md', '第十条'),
        chunk('公司法.md', '第四条'),
        chunk('民法典.md', '第六十一条', { exact: true }),
    ];
    const merged = mergeRewrittenResults(base, [chunk('民法典.md', '第五百零四条')], ['民法典', '公司法'], 5);
    assert.ok(merged.some(c => c.exactArticleHit && c.articleNo === '第六十一条'), '置顶项被换掉了');
});

test('mergeRewrittenResults: 已存在的条目不重复添加；空输入原样返回', () => {
    const base = [chunk('民法典.md', '第八十三条'), chunk('公司法.md', '第二十三条')];
    const dup = chunk('民法典.md', '第八十三条');
    assert.deepEqual(mergeRewrittenResults(base, [dup], ['民法典', '公司法'], 5), base);
    assert.equal(mergeRewrittenResults(base, [], ['民法典', '公司法'], 5), base);
});
