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
 * - `fabricated`   —— 条号在语料里**根本不存在**（或点名的那部法里没有这一条）。纯粹的编造。
 * - `not_in_context` —— 条号真实存在，但**没出现在本次检索结果里**。模型凭记忆引的。
 *   这一类比编造更隐蔽：条号是真的，条文内容大概率也是真的，所以人工抽查很难发现；
 *   但它同样是"没有依据的回答"，而且用户无从知道它没经过检索。
 * - `verified`     —— 条号存在，且确实出现在检索到的 context 里。
 *
 * 之所以要把后两者分开报，是因为它们的修法完全不同：`fabricated` 要靠提示词和
 * 拒答阈值压，`not_in_context` 要靠提高召回率压。
 *
 * ## 为什么引用要和"它点名的那部法"一起匹配
 *
 * 条号跨法不唯一：民法典和公司法都有第二十三条。旧实现只按数值查 context，
 * 于是答案写「公司法第二十三条」、而上下文里只有民法典第二十三条时，会判 `verified`
 * ——这正是"把凭记忆的引用报成通过"。现在只要答案在同一分句里点了法名、且那部法
 * 确实在语料里，就只认该法下的这一条；跨法的同名条不再算数。
 *
 * 库外法律（刑法、劳动合同法…）**维持旧语义**：校验器不负责判断"库里有没有这部法"，
 * 那是拒答与提示词的事；在这里硬判 fabricated 会把真引用报成编造。
 */

import VectorStore, { RetrievedChunk } from './VectorStore.js';
import { cnToNumber, extractArticleNumbers, parseArticleNo } from './articleNo.js';
import { findLaws, normLawName, type LawMention } from './laws.js';

/**
 * `content_mismatch` —— 条号真实、也确实在检索结果里，但**引用旁边那段话不像这一条说的**：
 * 它与另一条被检索到的条文重合得多得多。这是 "article_swap"（条号真、内容错）的确定性抓手，
 * faithfulness 判官在注入测试里只抓到 7/22，而这个检查不需要模型。
 *
 * 它只能抓"句子里有实质内容、且那些内容明显属于另一条"的情形；标题式引用、
 * 纯列举、以及"参见/另见"式指针不会触发（没有足够文本可判）。所以它是**补强**，
 * 不是替代判官：校准见 README §Faithfulness 与 `pnpm eval:citations`。
 */
export type CitationStatus = 'verified' | 'not_in_context' | 'fabricated' | 'content_mismatch';

export interface CitationCheck {
    /** 模型原文写法，如「第一百一十条」——保留它是因为模型偶尔会用非常规写法 */
    raw: string;
    /** 归一后的条号数值，如 110 */
    articleNo: number;
    /** 便于展示的规范写法，如「第110条」 */
    label: string;
    /**
     * 答案在这条引用旁边点名了哪部法律（同一分句内最近的前置法名）；没点名时为 null。
     * 点名且该法在语料里时，`status` 是**(法, 条号)** 这一对的结论，不是条号单人的。
     */
    law: string | null;
    status: CitationStatus;
    exists: boolean;
    inContext: boolean;
    /** 该条号落在哪些语料文件里（民法典/公司法同号条文都存在时会有多个） */
    sources: string[];
    /** 本次检索结果中，哪些来源的文本里出现了这个条号 */
    foundIn: string[];
    /**
     * 诊断：没点名法律、且这个条号在语料里同时存在于多部法律。
     * 此时"在 context 里"只能证明某一部法的同名条在，无法证明是哪一部——不翻转判定，
     * 只让读日志的人知道这条 verified 的把握比别的小。
     */
    ambiguousLaw: boolean;
    /**
     * 内容绑定诊断：引用所在句与被引条文的二元组覆盖率（own），以及同一句与**其它**
     * 检索结果的最高覆盖率（bestOther）。只在 `verified` 的条目上计算；其余为 null。
     * 判定规则见 `CONTENT_*` 常量，校准数字见 README。
     */
    contentOverlap: { own: number; bestOther: number } | null;
}

export interface CitationReport {
    checks: CitationCheck[];
    total: number;
    verified: number;
    notInContext: number;
    fabricated: number;
    contentMismatch: number;
    /** 只要出现任一非 verified 的条目即为 true */
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

/** 答案里的条号，与 articleNo.ts 的 CITATION_RE 保持同一套写法。 */
const ARTICLE_RE = /第\s*([0-9]+|[一二三四五六七八九十百千万零两]+)\s*条/g;

/**
 * 分句边界：句号/分号/问号/叹号/换行。逗号不算——「根据《民法典》，第577条」
 * 仍是同一句；把逗号当边界会让常见写法丢掉法律归属。
 */
const CLAUSE_BOUNDARY = /[。；;！!？?\n]/g;

/** 同一分句内、条号之前最近的那个法律名；没有则 null。 */
function governingLaw(text: string, laws: LawMention[], position: number): string | null {
    let clauseStart = 0;
    for (const m of text.slice(0, position).matchAll(CLAUSE_BOUNDARY)) {
        clauseStart = (m.index ?? 0) + m[0].length;
    }
    let picked: string | null = null;
    for (const law of laws) {
        if (law.start >= clauseStart && law.end <= position) picked = law.name;
    }
    return picked;
}

interface CitationOccurrence {
    articleNo: number;
    raw: string;
    law: string | null;
    /** 引用所在的整句（分句边界见 CLAUSE_BOUNDARY），用于内容绑定检查 */
    sentence: string;
}

/**
 * 抽出答案里的每条引用，并尽量绑到同一分句里最近的前置法名。
 *
 * 用"最近的前置"而不是"全文任意法名"：答案常写成「《民法典》……；《公司法》……」，
 * 按全文取会让后一条引用错绑到前一部法，反而制造假告警。
 */
/** 引用所在整句：从上一个分句边界到下一个分句边界。 */
function sentenceAround(text: string, position: number): string {
    let start = 0;
    for (const m of text.slice(0, position).matchAll(CLAUSE_BOUNDARY)) {
        start = (m.index ?? 0) + m[0].length;
    }
    const boundary = new RegExp(CLAUSE_BOUNDARY.source, 'g');
    boundary.lastIndex = position;
    const end = boundary.exec(text)?.index ?? text.length;
    return text.slice(start, end);
}

function extractCitations(text: string): CitationOccurrence[] {
    const laws = findLaws(text);
    const out: CitationOccurrence[] = [];
    for (const m of text.matchAll(ARTICLE_RE)) {
        const n = cnToNumber(m[1]);
        if (!Number.isFinite(n) || n <= 0) continue;
        const position = m.index ?? 0;
        out.push({
            articleNo: n,
            raw: m[0].replace(/\s+/g, ''),
            law: governingLaw(text, laws, position),
            sentence: sentenceAround(text, position),
        });
    }
    return out;
}

// ---------------------------------------------------------------------------
// 内容绑定：引用旁边那句话，到底更像哪一条？
//
// 这是 "条号真、内容错"（faithfulness 的 article_swap）的确定性抓手。核心不是
// "和所引条文重合低就报警"——法律答案大量是改写，低重合是常态，那样会产生成片误报。
// 只有当**同一句话与另一条被检索到的条文重合得多得多**时才报：
// 那说明内容属于另一条，条号被张冠李戴了。
//
// 阈值来自 27 条真实答案（68 处引用）+ 25 个注入的跨法换号样本：
//   margin ≥ 0.10 且 bestOther ≥ 0.25 且 own ≤ 0.35 → 命中 8/25 个换号，误报 0/68。
// 检出率不高是诚实的：标题式引用、纯列举、改写幅度大的句子没有足够文本可判。
// 它是**补强**，不是替代判官（判官在同一批注入上抓到 7/22）。
// ---------------------------------------------------------------------------
const CONTENT_MARGIN = 0.10;
const CONTENT_BEST_OTHER_FLOOR = 0.25;
const CONTENT_OWN_CEIL = 0.35;
/** 少于这么多个二元组的句子没有信息量，不判（标题、编号、"另见第X条"）。 */
const CONTENT_MIN_BIGRAMS = 8;

/** 去掉标点与空白后的字符二元组。中文没有空格分词，二元组是够用的粗粒度。 */
function bigrams(text: string): Set<string> {
    const t = text.replace(/[\s，。；：、"“”()（）【】\[\]{}·—…!?！？]/g, '');
    const out = new Set<string>();
    for (let i = 0; i + 1 < t.length; i++) out.add(t.slice(i, i + 2));
    return out;
}

/** 句子里的二元组有多少比例出现在条文里。 */
function bigramCoverage(sentence: Set<string>, body: string): number {
    if (sentence.size === 0) return 0;
    const b = bigrams(body);
    if (b.size === 0) return 0;
    let inter = 0;
    for (const g of sentence) if (b.has(g)) inter++;
    return inter / sentence.size;
}

/** 同一对 (法, 条号) 只查一次；「第577条」重复出现不重复告警。 */
function dedupeCitations(items: CitationOccurrence[]): CitationOccurrence[] {
    const seen = new Set<string>();
    return items.filter(c => {
        const key = `${c.law ?? ''}#${c.articleNo}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
    });
}

export function verifyCitations(
    answer: string,
    store: VectorStore,
    retrieved: RetrievedChunk[],
): CitationReport {
    const citations = dedupeCitations(extractCitations(answer));
    const inContext = collectContextArticleNumbers(retrieved);

    const checks: CitationCheck[] = [];
    for (const c of citations) {
        const hits = store.findByArticleNumber(c.articleNo);
        const exists = hits.length > 0;
        const contextSources = inContext.get(c.articleNo) ?? new Set<string>();
        const wantLaw = c.law ? normLawName(c.law) : null;

        // 点名了法、且这部法在语料里存在 → 只认 (法, 条号) 这一对。
        // 库外法律维持旧语义：那不是引用校验该拦的东西（见文件头注释）。
        const lawInCorpus = wantLaw !== null && store.hasSource(wantLaw);
        const scoped = wantLaw !== null && lawInCorpus;
        const pairExists = scoped
            ? hits.some(h => normLawName(h.source) === wantLaw)
            : exists;

        const foundIn = scoped
            ? [...contextSources].filter(s => normLawName(s) === wantLaw)
            : [...contextSources];

        let status: CitationStatus = !pairExists
            ? 'fabricated'
            : foundIn.length > 0 ? 'verified' : 'not_in_context';

        // 内容绑定只在"条号对、也在 context 里"时有意义；其余两种失败已经说清楚了。
        let contentOverlap: { own: number; bestOther: number } | null = null;
        if (status === 'verified') {
            const ownChunk = retrieved.find(r =>
                parseArticleNo(r.articleNo) === c.articleNo
                && (wantLaw === null || normLawName(r.source) === wantLaw));
            if (ownChunk) {
                const grams = bigrams(c.sentence);
                const own = bigramCoverage(grams, ownChunk.document);
                const bestOther = Math.max(0, ...retrieved
                    .filter(r => r !== ownChunk)
                    .map(r => bigramCoverage(grams, r.document)));
                contentOverlap = { own, bestOther };

                const enoughText = grams.size >= CONTENT_MIN_BIGRAMS;
                if (enoughText
                    && bestOther - own >= CONTENT_MARGIN
                    && bestOther >= CONTENT_BEST_OTHER_FLOOR
                    && own <= CONTENT_OWN_CEIL) {
                    status = 'content_mismatch';
                }
            }
        }

        checks.push({
            raw: c.raw,
            articleNo: c.articleNo,
            label: `第${c.articleNo}条`,
            law: c.law,
            status,
            exists: pairExists,
            inContext: foundIn.length > 0,
            sources: [...new Set(hits.map(h => h.source))],
            foundIn,
            ambiguousLaw: !scoped && hits.length > 1,
            contentOverlap,
        });
    }

    // 按严重程度排序：编造的最先看，其次是"条号真但内容对不上"，再是凭记忆引用。
    const rank: Record<CitationStatus, number> = {
        fabricated: 0, content_mismatch: 1, not_in_context: 2, verified: 3,
    };
    checks.sort((a, b) => rank[a.status] - rank[b.status] || a.articleNo - b.articleNo);

    const count = (s: CitationStatus) => checks.filter(c => c.status === s).length;
    return {
        checks,
        total: checks.length,
        verified: count('verified'),
        notInContext: count('not_in_context'),
        fabricated: count('fabricated'),
        contentMismatch: count('content_mismatch'),
        hasProblem: checks.some(c => c.status !== 'verified'),
    };
}

/** 给日志/SSE 用的一句话摘要；无引用时返回 null。 */
export function summarizeCitations(report: CitationReport): string | null {
    if (report.total === 0) return null;
    if (!report.hasProblem) return `${report.total} 处引用全部核对通过`;
    const parts: string[] = [];
    if (report.fabricated > 0) parts.push(`${report.fabricated} 处条号在语料中不存在`);
    if (report.contentMismatch > 0) parts.push(`${report.contentMismatch} 处条号与相邻内容对不上`);
    if (report.notInContext > 0) parts.push(`${report.notInContext} 处未出现在检索结果中`);
    return `${report.total} 处引用中 ${parts.join('、')}`;
}
