/**
 * 多法问题的**按法改写**：把一个问题拆成"每部法一条检索式"，再分别检索。
 *
 * 为什么需要它（实测，不是设计直觉）：`multi-01`（法人人格否认）和
 * `multi-03`（越权担保）在原始问句下，第二部法的最佳条文分别全局排第 9、第 7，
 * 被每法保底的名次闸挡在 top-5 外。按法改写成该法的规范术语后，民法典#83 /
 * #504 都能进该法前 2，合并后两道题的覆盖率从 1/2 变成 2/2。
 *
 * 边界：
 *   - 只在问题**明确点名 ≥2 部库内法律**时触发（隐式跨法如 multi-04 不触发）；
 *   - 只改检索式，不让模型答题（prompt 明令禁止结论）；
 *   - 改写失败/解析失败 → 返回空数组，退回原始检索，不影响可用性；
 *   - 改写结果只用于**补位**：每部法最多保留 2 条，只从原始检索的尾部换出，
 *     不动置顶项。
 */
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import OpenAI from 'openai';
import { loadPrompt } from './prompts.js';
import { CACHE_DIR } from './paths.js';
import { normLawName } from './laws.js';
import type { RetrievedChunk } from './VectorStore.js';

export const QUERY_REWRITE_ENABLED = process.env.QUERY_REWRITE !== '0';
export const QUERY_REWRITE_MODEL = process.env.QUERY_REWRITE_MODEL ?? 'gpt-4o-mini';
export const QUERY_REWRITE_PROMPT_VERSION = Number(process.env.QUERY_REWRITE_PROMPT_VERSION ?? 1);
const QUERY_REWRITE_TIMEOUT_MS = Number(process.env.QUERY_REWRITE_TIMEOUT_MS ?? 8_000);

export interface LawQuery {
    law: string;
    query: string;
}

const CACHE_PATH = path.join(CACHE_DIR, 'query-rewrite.json');
const MAX_CACHE = 2000;
let cache: Record<string, LawQuery[]> | null = null;
let client: OpenAI | null = null;
let promptBody: string | null = null;

function getCache(): Record<string, LawQuery[]> {
    if (cache === null) {
        try {
            cache = fs.existsSync(CACHE_PATH) ? JSON.parse(fs.readFileSync(CACHE_PATH, 'utf-8')) : {};
        } catch {
            cache = {};
        }
    }
    return cache!;
}

function persist(): void {
    const c = getCache();
    const keys = Object.keys(c);
    if (keys.length > MAX_CACHE) {
        for (const k of keys.slice(0, keys.length - MAX_CACHE)) delete c[k];
    }
    try {
        if (!fs.existsSync(CACHE_DIR)) fs.mkdirSync(CACHE_DIR, { recursive: true });
        fs.writeFileSync(CACHE_PATH, JSON.stringify(c));
    } catch {
        // 缓存落盘失败不影响判定，最多多花一次调用
    }
}

function getClient(): OpenAI {
    if (!client) {
        client = new OpenAI({
            apiKey: process.env.OPENAI_API_KEY,
            baseURL: process.env.OPENAI_BASE_URL || undefined,
        });
    }
    return client;
}

function cacheKey(question: string, laws: string[]): string {
    return crypto.createHash('sha1')
        .update(`${QUERY_REWRITE_MODEL}\u0000v${QUERY_REWRITE_PROMPT_VERSION}\u0000${question}\u0000${laws.join(',')}`)
        .digest('hex');
}

/**
 * 严格解析改写输出。**解析不出来就返回空数组**——这次没有改写，退回原始检索；
 * 不抛错、不重试：改写是增强，不是主路径。
 *
 * 约束：law 必须是允许的法律；query 4–80 字；每部法最多 2 条、总计最多 4 条；去重。
 */
export function parseRewriteOutput(raw: string, allowedLaws: string[]): LawQuery[] {
    let obj: unknown;
    try {
        obj = JSON.parse(raw);
    } catch {
        return [];
    }
    if (obj === null || typeof obj !== 'object') return [];
    const queries = (obj as Record<string, unknown>).queries;
    if (!Array.isArray(queries)) return [];

    const allowed = new Map(allowedLaws.map(l => [normLawName(l), l]));
    const perLaw = new Map<string, number>();
    const seen = new Set<string>();
    const out: LawQuery[] = [];

    for (const item of queries) {
        if (item === null || typeof item !== 'object') continue;
        const o = item as Record<string, unknown>;
        const rawLaw = typeof o.law === 'string' ? normLawName(o.law) : '';
        const law = allowed.get(rawLaw);
        const query = typeof o.query === 'string' ? o.query.trim() : '';
        if (!law || query.length < 4 || query.length > 80) continue;
        const count = perLaw.get(law) ?? 0;
        if (count >= 2) continue;
        const key = `${law}#${query}`;
        if (seen.has(key)) continue;
        seen.add(key);
        perLaw.set(law, count + 1);
        out.push({ law, query });
        if (out.length >= 4) break;
    }
    return out;
}

/** 为一组法律生成改写检索式；任何失败都返回 []（调用方退回原始检索）。 */
export async function rewriteForLaws(
    question: string,
    laws: string[],
    opts: { signal?: AbortSignal } = {},
): Promise<LawQuery[]> {
    const allowed = [...new Set(laws.map(normLawName))];
    if (allowed.length < 2) return [];

    const key = cacheKey(question, allowed);
    const store = getCache();
    if (store[key]) return store[key];

    try {
        if (promptBody === null) {
            promptBody = loadPrompt('query-rewrite', QUERY_REWRITE_PROMPT_VERSION).body;
        }
        const timeout = AbortSignal.timeout(QUERY_REWRITE_TIMEOUT_MS);
        const signal = opts.signal ? AbortSignal.any([opts.signal, timeout]) : timeout;
        const resp = await getClient().chat.completions.create({
            model: QUERY_REWRITE_MODEL,
            temperature: 0,
            response_format: { type: 'json_object' },
            messages: [
                { role: 'system', content: promptBody },
                {
                    role: 'user',
                    content: `用户问题：「${question}」\n需要检索的法律：${allowed.join('、')}`,
                },
            ],
        }, { signal });

        const parsed = parseRewriteOutput(resp.choices[0]?.message?.content ?? '', allowed);
        if (parsed.length === 0) return [];
        store[key] = parsed;
        persist();
        return parsed;
    } catch (e) {
        console.warn(`[rewrite] 改写失败，退回原始检索：${(e as Error).message}`);
        return [];
    }
}

/**
 * 把改写检索的结果补进原始 top-K。纯函数，便于单测。
 *
 * 规则刻意保守：
 *   - 每部被点名的法律最多保留 2 条（避免改写结果挤掉另一半）；
 *   - 只从**原始检索结果**的尾部换出，且只换"该法在结果里还有别的代表"的条目；
 *   - 不动置顶项，不重复添加。
 */
export function mergeRewrittenResults(
    base: RetrievedChunk[],
    extra: RetrievedChunk[],
    laws: string[],
    topK: number,
): RetrievedChunk[] {
    if (extra.length === 0) return base;
    const key = (c: RetrievedChunk) => `${c.source}#${c.hash}`;
    const out = [...base];
    const seen = new Set(out.map(key));
    const baseLen = base.length;

    for (const law of laws.map(normLawName)) {
        const candidates = extra.filter(e => normLawName(e.source).includes(law));
        let reps = out.filter(o => normLawName(o.source).includes(law)).length;
        for (const e of candidates) {
            if (reps >= 2) break;
            if (seen.has(key(e))) continue;
            if (out.length < topK) {
                out.push(e); seen.add(key(e)); reps++;
                continue;
            }
            let swapAt = -1;
            for (let i = baseLen - 1; i >= 0; i--) {
                const c = out[i];
                if (c.exactArticleHit) continue;
                if (out.filter(x => x.source === c.source).length <= 2) continue;
                swapAt = i;
                break;
            }
            if (swapAt < 0) break;
            seen.delete(key(out[swapAt]));
            out[swapAt] = e;
            seen.add(key(e));
            reps++;
        }
    }
    return out;
}
