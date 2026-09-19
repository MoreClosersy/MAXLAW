/**
 * 引用校验器：检查模型答案里的每一个「第X条」。
 *
 * ## 为什么需要它
 *
 * 法条 RAG 最危险的失败不是"答得不准"，而是**答得很像样但条号是编的**。
 * 模型对「第577条」这种形式的记忆非常流利，检索没召回时它会顺着语感补一个条号出来，
 * 而条号一旦错，用户拿去引用的就是错的法条——比直接说"查不到"伤害大得多。
 *
 * 提示词（prompts/legal-system-prompt.v2.md:55）要求"引用的条号必须出现在检索到的
 * context 里"，但那只是一句祈使句，**没有任何机制保证它被遵守**。本模块就是那个机制。
 *
 * ## 三种状态，对应两种不同的失败
 *
 * - `fabricated`   —— 条号在语料里**根本不存在**。纯粹的编造。
 * - `not_in_context` —— 条号真实存在，但**没出现在本次检索结果里**。模型凭记忆引的。
 *   这一类比编造更隐蔽：条号是真的，条文内容大概率也是真的，所以人工抽查很难发现；
 *   但它同样是"没有依据的回答"，而且用户无从知道它没经过检索。
 * - `verified`     —— 条号存在，且确实出现在检索到的 context 里。
 *
 * 之所以要把后两者分开报，是因为它们的修法完全不同：`fabricated` 要靠提示词和
 * 拒答阈值压，`not_in_context` 要靠提高召回率压。
 */

import VectorStore, { RetrievedChunk } from './VectorStore.js';
import { extractArticleNumbers } from './articleNo.js';

export type CitationStatus = 'verified' | 'not_in_context' | 'fabricated';

export interface CitationCheck {
    /** 模型原文写法，如「第一百一十条」——保留它是因为模型偶尔会用非常规写法 */
    raw: string;
    /** 归一后的条号数值，如 110 */
    articleNo: number;
    /** 便于展示的规范写法，如「第110条」 */
    label: string;
    status: CitationStatus;
    exists: boolean;
    inContext: boolean;
    /** 该条号落在哪些语料文件里（民法典/公司法同号条文都存在时会有多个） */
    sources: string[];
    /** 本次检索结果中，哪些来源的文本里出现了这个条号 */
    foundIn: string[];
}

export interface CitationReport {
    checks: CitationCheck[];
    total: number;
    verified: number;
    notInContext: number;
    fabricated: number;
    /** 只要出现任一 fabricated / not_in_context 即为 true */
    hasProblem: boolean;
}

/**
 * 收集"本次检索到的 context 里出现过的所有条号"。
 *
 * 两个来源都要算：
 * 1. chunk 自身的 `articleNo`（被检索中的那一条）；
 * 2. chunk 正文里提到的其它条号——法条之间大量交叉引用（"依照本法第X条"），
 *    模型顺着这些交叉引用往下写是正确的，不能判成 hallucination。
 *
 * 漏掉第 2 点会让校验器频繁误报，而一个频繁误报的校验器等于没有校验器——
 * 用户会学会忽略它。
 */
function collectContextArticleNumbers(retrieved: RetrievedChunk[]): Map<number, Set<string>> {
    const found = new Map<number, Set<string>>();
    const add = (n: number, source: string) => {
        let bucket = found.get(n);
        if (!bucket) { bucket = new Set(); found.set(n, bucket); }
        bucket.add(source);
    };

    for (const chunk of retrieved) {
        const own = extractArticleNumbers(chunk.articleNo ?? '');
        for (const n of own.keys()) add(n, chunk.source);
        for (const n of extractArticleNumbers(chunk.document).keys()) add(n, chunk.source);
    }
    return found;
}

export function verifyCitations(
    answer: string,
    store: VectorStore,
    retrieved: RetrievedChunk[],
): CitationReport {
    const cited = extractArticleNumbers(answer);
    const inContext = collectContextArticleNumbers(retrieved);

    const checks: CitationCheck[] = [];
    for (const [articleNo, raw] of cited) {
        const hits = store.findByArticleNumber(articleNo);
        const exists = hits.length > 0;
        const contextHit = inContext.get(articleNo);

        checks.push({
            raw,
            articleNo,
            label: `第${articleNo}条`,
            status: !exists ? 'fabricated' : contextHit ? 'verified' : 'not_in_context',
            exists,
            inContext: !!contextHit,
            sources: [...new Set(hits.map(h => h.source))],
            foundIn: contextHit ? [...contextHit] : [],
        });
    }

    // 按严重程度排序：编造的排最前，用户先看到最该看的
    const rank: Record<CitationStatus, number> = { fabricated: 0, not_in_context: 1, verified: 2 };
    checks.sort((a, b) => rank[a.status] - rank[b.status] || a.articleNo - b.articleNo);

    const count = (s: CitationStatus) => checks.filter(c => c.status === s).length;
    return {
        checks,
        total: checks.length,
        verified: count('verified'),
        notInContext: count('not_in_context'),
        fabricated: count('fabricated'),
        hasProblem: checks.some(c => c.status !== 'verified'),
    };
}

/** 给日志/SSE 用的一句话摘要；无引用时返回 null */
export function summarizeCitations(report: CitationReport): string | null {
    if (report.total === 0) return null;
    if (!report.hasProblem) return `${report.total} 处引用全部核对通过`;
    const parts: string[] = [];
    if (report.fabricated > 0) parts.push(`${report.fabricated} 处条号在语料中不存在`);
    if (report.notInContext > 0) parts.push(`${report.notInContext} 处未出现在检索结果中`);
    return `${report.total} 处引用中 ${parts.join('、')}`;
}
