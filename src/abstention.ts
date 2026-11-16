/**
 * 拒答判定：**纯逻辑，零副作用**。
 *
 * 这个模块里不读 `process.env`、不碰 `fs`、不 import `./index.js`。
 * 理由是踩过一次的：`src/index.ts` 顶层有 `mkdirSync` 和 `getMcpClients()`（会 spawn 子进程），
 * 所以住在那里的一切都不可测——`decideAbstention` 因此长期零测试覆盖。
 * 同样的教训在 eval 侧也出现过一次：`eval/faithfulness.ts` 顶层 `await loadKnowledge()`，
 * 第一次 `pnpm test` 直接挂了两分钟。**只要一个函数需要被测试，它就不能住在有副作用模块的顶层。**
 *
 * 所以：判定逻辑全在这里（可单测、离线），网络调用全在 `AnswerabilityJudge.ts`。
 *
 * ## 这个文件在解决什么问题
 *
 * 拒答原先只是"检索最高分 < 0.35"。`pnpm eval:signals` 量出来它在 42 道域外题里只拦下 3 道，
 * 而且原因不是阈值没调好：**话题相关度不是可答性**。两个样本把这件事钉死了——
 * `sem-02`（口语化问法）的正确条文第二十条被检索到了却是全库最低分；
 * `ood-11`（问刑事责任）逐字命中民法典第181条却是全部域外题的最高分，而那条讲的是民事责任。
 * 于是任何相似度阈值都会**先拒掉口语化的域内问题、最后才拦下逐字命中的域外问题**，顺序正好反了。
 *
 * 这里因此改成：检索之后问一次"这段文字里有没有答案"，而且**要求判官指名道姓说出哪一条**
 * ——多法问题时是"哪几条"，每一条都要在检索结果里（v2）。
 * 指名而不是 yes/no，是因为指名可以被确定性交叉校验（`crossCheckAnswerability`），
 * 而一个无法核验的判定不算判定。
 */
import { RetrievedChunk } from './VectorStore.js';
import { cnToNumber, extractArticleNumbers, parseArticleNo } from './articleNo.js';
import { normLawName } from './laws.js';

export interface AbstentionDecision {
    abstained: boolean;
    topScore: number;
    /** 是否精确命中查询点名的条号（唯一的"强信号"） */
    hasExactArticleHit: boolean;
    /** 拒答的原因，便于日志和评测归因 */
    reason: AbstentionReason;
}

export type AbstentionReason =
    | 'no_hits'
    | 'below_threshold'
    | 'answered'
    // —— 以下由判官路径产生 ——
    | 'judge_unanswerable'
    | 'judge_binding_mismatch'
    | 'judge_parse_failure'
    | 'judge_error_fallback';

/**
 * 相似度判定：**第一道，仍然保留，但不再是拒答机制**。
 *
 * 它现在的角色是"明显跑题过滤器"——0.35 在 69 题上拒掉 3/42 道域外题、误拒 0 道域内题。
 * 这既不是优点也不是缺点，它就是"基本什么都没拦"。真正的拒答在判官那一步。
 *
 * 之所以还留着它：(1) 检索为空时它是免费的硬拒答；(2) 判官调用失败时它是回退目标——
 * 退回已测量的现状，好过退化到一个没人量过的状态。
 *
 * `threshold` 是**必填**参数（以前是绑到环境变量的默认值）。默认参数让这个纯函数隐式依赖
 * 环境状态、让测试之间互相污染；环境变量留在 `src/index.ts`（所有配置旋钮都在那儿），显式传进来。
 */
export function decideAbstention(retrieved: RetrievedChunk[], threshold: number): AbstentionDecision {
    const topScore = retrieved.length > 0 ? Math.max(...retrieved.map(r => r.score)) : 0;
    const hasExactArticleHit = retrieved.some(r => r.exactArticleHit);
    if (retrieved.length === 0) return { abstained: true, topScore, hasExactArticleHit, reason: 'no_hits' };
    if (topScore < threshold && !hasExactArticleHit) return { abstained: true, topScore, hasExactArticleHit, reason: 'below_threshold' };
    return { abstained: false, topScore, hasExactArticleHit, reason: 'answered' };
}

// ---------------------------------------------------------------------------
// 跳过规则：什么时候不必花钱问判官
// ---------------------------------------------------------------------------

/**
 * `disabled` 是逃生阀（`ANSWERABILITY_JUDGE=0`），不是一条"问了也是白问"的规则：
 * 它是"这次根本没问"。两者在记账里必须分得开——把"判官关着"记成"判官说 no_hits"
 * 会让 eval 报告上的跳过原因变成一个没人能复核的断言。
 */
export type JudgeSkipReason = 'no_hits' | 'exact_article_hit' | 'sim_abstain' | 'disabled';

export interface JudgeSkip {
    skip: boolean;
    reason: JudgeSkipReason | null;
}

/**
 * 三条跳过规则，都是"问了也是白问"或"问了只会更糟"的情形。
 *
 * 顺序有讲究：`no_hits` 压过一切（没有候选可读），`exact_article_hit` 压过 `sim_abstain`
 * （点名条号是构造上的域内且可答，不该被低分推翻）。
 *
 * ⚠️ `exactArticleHit` **不等于**"检索结果里有答案"。它只表示"查询点名了某个条号，且库里真有这一条"。
 * `sem-02` 的正确条文第二十条在 rank 4，`exactArticleHit` 是 false——所以这条规则不会漏掉它。
 */
export function shouldSkipJudge(retrieved: RetrievedChunk[], sim: AbstentionDecision): JudgeSkip {
    if (retrieved.length === 0) return { skip: true, reason: 'no_hits' };
    if (sim.hasExactArticleHit) return { skip: true, reason: 'exact_article_hit' };
    // 阈值在域内是 0/27 误拒、域外只拦下 3/42。让它被判官推翻只会亏，所以它赢。
    // 记成独立的 reason 而不是并进上面两条，是为了让 eval 能量"如果让它被推翻会怎样"这个反事实。
    if (sim.abstained) return { skip: true, reason: 'sim_abstain' };
    return { skip: false, reason: null };
}

// ---------------------------------------------------------------------------
// 判官输出：解析
// ---------------------------------------------------------------------------

export interface JudgeRequirement {
    /** 法律名，如「民法典」 */
    law: string;
    /** 条号数值。数值比较是硬要求，理由见 articleNo.ts */
    articleNo: number;
    /** 判官摘录的原文（用于确定性核对"它到底读没读这一条"），可为 null */
    quote: string | null;
}

export interface ParsedAnswerability {
    answerable: boolean;
    /**
     * **必答项**：判官认为回答这个问题必须引用的条文。
     *
     * 为什么是数组而不是单条：一个问题可以合法地需要两部法各一条
     * （「法定代表人以公司名义…」= 公司法#11 + 民法典#61）。v1 的 `cited` 只有一个位置，
     * 于是"只召回了一半"会被判成可答——那正是多法问答最典型的失败形状。
     * 每一条都必须在检索结果里，否则整条判定作废（见 crossCheckAnswerability）。
     *
     * `answerable=true` 时至少一条；`false` 时通常为空——非空会被记成契约违规。
     */
    requirements: JudgeRequirement[];
    reason: string;
    /** 可答时为空；不可答时写出缺哪部法/哪类规定 */
    missing: string | null;
}

export interface ParseError {
    error: string;
}

export function isParseError(x: unknown): x is ParseError {
    return typeof x === 'object' && x !== null && 'error' in x;
}

/** 把判官给的条号归一成正整数；支持 20 / "20" / "第二十条" / "第20条"。 */
function toArticleNumber(v: unknown): number | null {
    if (typeof v === 'number') {
        return Number.isInteger(v) && v > 0 ? v : null;
    }
    if (typeof v === 'string') {
        const t = v.trim();
        if (t === '') return null;
        if (/^\d+$/.test(t)) {
            const n = Number(t);
            return n > 0 ? n : null;
        }
        // cnToNumber 内部会剥掉「第」和「条」，中文数字与阿拉伯数字都认
        const n = cnToNumber(t);
        if (Number.isFinite(n) && n > 0) return n;
        // 兜底：判官把整句写进这个字段（如「民法典第181条」）时，取出其中唯一的条号
        const found = [...extractArticleNumbers(t).keys()];
        return found.length === 1 ? found[0] : null;
    }
    return null;
}

function toNonEmptyString(v: unknown): string | null {
    return typeof v === 'string' && v.trim() !== '' ? v.trim() : null;
}

/** 判官最多能要求几条。正常法条问答不会需要一打条文；失控的数组会让"每条都必须在 context 里"永远对不上。 */
const MAX_JUDGE_REQUIREMENTS = 6;

function parseRequirement(v: unknown): JudgeRequirement | ParseError {
    if (v === null || typeof v !== 'object' || Array.isArray(v)) {
        return { error: `requirement 不是对象：${JSON.stringify(v)}` };
    }
    const o = v as Record<string, unknown>;
    const law = toNonEmptyString(o.law);
    if (!law) return { error: 'requirement 缺少 law' };
    const n = toArticleNumber(o.articleNo);
    if (n === null) return { error: `requirement 的 articleNo 无法解析（收到 ${JSON.stringify(o.articleNo)}）` };
    return { law, articleNo: n, quote: toNonEmptyString(o.quote) };
}

/**
 * 解析判官的严格 JSON 输出。**解析不出来一律返回 `{error}`，绝不默认成任何一边。**
 *
 * 这是整条链上最容易自欺的一环：如果坏 JSON 默认成"可答"，判官故障就会静默变成"回答"；
 * 默认成"不可答"，判官故障就会静默变成"拒答"。两个方向都是在把工具的故障记成判定结论。
 * 所以这里的契约是硬的：**无法核验的判定不算判定**，由调用方按"失败了"处理。
 * （同一条不变量在 faithfulness 判官那边是："解析失败不能算作忠实"。）
 *
 * 「说了可答但没给条号」也判为解析失败，这一条是刻意的：接受它就会得到一个
 * `crossCheckAnswerability` 无法审计的判定，而强制指名正是选它而不是 yes/no 的全部理由。
 *
 * 同时接受两种形状：
 *   v2 `{"answerable":true,"requirements":[{"law":"公司法","articleNo":11,...}, ...]}`
 *   v1 `{"answerable":true,"law":"民法典","articleNo":20,...}` —— 解析成单条 requirement。
 * 兼容 v1 不是为兼容而兼容：旧报告与旧缓存里全是 v1 形状，能回放才谈得上两版可比。
 */
export function parseAnswerabilityOutput(raw: string): ParsedAnswerability | ParseError {
    const text = typeof raw === 'string' ? raw.trim() : '';
    if (text === '') return { error: '输出为空' };

    // 剥掉 ```json 围栏；没有围栏就在正文里取第一个 { 到最后一个 } 的片段
    let body = text;
    const fence = text.match(/^```(?:json)?\s*\n?([\s\S]*?)\n?```$/);
    if (fence) {
        body = fence[1].trim();
    } else if (!text.startsWith('{')) {
        const start = text.indexOf('{');
        const end = text.lastIndexOf('}');
        if (start === -1 || end === -1 || end < start) return { error: `输出中没有 JSON 对象：${text.slice(0, 120)}` };
        body = text.slice(start, end + 1);
    }

    let obj: unknown;
    try {
        obj = JSON.parse(body);
    } catch (e) {
        return { error: `JSON 解析失败：${(e as Error).message}；原文：${body.slice(0, 120)}` };
    }

    if (obj === null || typeof obj !== 'object' || Array.isArray(obj)) {
        return { error: `输出不是 JSON 对象：${body.slice(0, 120)}` };
    }
    const o = obj as Record<string, unknown>;

    // 必须是字面量布尔。"true"（字符串）不接受——JSON 里写错类型是判官在含糊，
    // 而含糊的判定不该被当成干脆的判定。
    if (typeof o.answerable !== 'boolean') {
        return { error: `answerable 不是布尔值（收到 ${JSON.stringify(o.answerable)}）` };
    }

    const reason = toNonEmptyString(o.reason) ?? '';
    const missing = toNonEmptyString(o.missing);

    let requirements: JudgeRequirement[] = [];
    if (o.requirements !== undefined) {
        if (!Array.isArray(o.requirements)) {
            return { error: `requirements 不是数组（收到 ${JSON.stringify(o.requirements)}）` };
        }
        if (o.requirements.length > MAX_JUDGE_REQUIREMENTS) {
            return { error: `requirements 超过 ${MAX_JUDGE_REQUIREMENTS} 条（收到 ${o.requirements.length}）` };
        }
        for (const item of o.requirements) {
            const r = parseRequirement(item);
            if (isParseError(r)) return r;
            requirements.push(r);
        }
    } else {
        // v1 兼容：顶层单条。只给了 law 或只给了条号的"半条记录"不算 requirement——
        // v1 的"不可答却顺手写了条号"是契约违规记录，不是格式错误。
        const law = toNonEmptyString(o.law);
        const n = toArticleNumber(o.articleNo);
        if (law && n !== null) requirements.push({ law, articleNo: n, quote: toNonEmptyString(o.quote) });
    }

    // 同一对 (法, 条号) 重复出现只留第一条：重复会让"每条都在 context 里"的计数虚高。
    const seen = new Set<string>();
    requirements = requirements.filter(r => {
        const key = `${r.law}#${r.articleNo}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
    });

    if (o.answerable === true && requirements.length === 0) {
        return { error: 'answerable=true 但没有任何 requirement——强制指名不允许只给结论' };
    }
    return { answerable: o.answerable, requirements, reason, missing };
}

// ---------------------------------------------------------------------------
// 确定性交叉校验：判官指名的条号，在检索结果里吗？
// ---------------------------------------------------------------------------

export interface AnswerabilityCheck {
    /**
     * `in_context`    —— **每一条**指名的条文都在检索结果里
     * `not_in_context`—— 至少一条不在（它是在凭记忆答，不是在读给它的东西）
     * `not_applicable`—— 判的是不可答，没有可校验的指名
     */
    binding: 'in_context' | 'not_in_context' | 'not_applicable';
    /** 与 `requirements` 同序：每条命中的检索下标，未命中为 null */
    matchedIndexes: (number | null)[];
    /** 没能在检索结果里找到的必答项（法名+条号） */
    unmatched: { law: string; articleNo: number }[];
    /** 全部摘录都核对通过才是 true；一条可核对的摘录都没有时是 null。**只用于诊断，从不翻转判定。** */
    quoteVerified: boolean | null;
    /** 契约违规：判了不可答却还是写了条号。仅记录。 */
    refusalNamedArticle: boolean;
}

/** 「中华人民共和国民法典」「《民法典》」「民法典.md」→「民法典」 */
function normLaw(s: string): string {
    return s
        .replace(/中华人民共和国/g, '')
        .replace(/[《》\s]/g, '')
        .replace(/\.md$/i, '')
        .trim();
}

function normText(s: string): string {
    return s.replace(/\s+/g, '');
}

/** 取一条检索结果自身的条号数值。语料里存的是「第一百八十八条」这种写法。 */
function chunkArticleNumber(c: RetrievedChunk): number | null {
    const direct = parseArticleNo(c.articleNo);
    if (direct !== null) return direct;
    if (!c.articleNo) return null;
    const found = [...extractArticleNumbers(c.articleNo).keys()];
    return found.length > 0 ? found[0] : null;
}

/**
 * 把判官的**每一条**指名和检索结果对一遍。**必须同时匹配法和条号。**
 *
 * 只比条号是不够的：条号跨法不唯一（民法典第23条和公司法第23条都存在），
 * 只按数字匹配的实现会放过"判官说的是公司法第23条、而检索到的只有民法典第23条"——
 * 那正是强制指名要拦住的东西，放它过去整个设计就白做了。
 *
 * 多需求时是**合取**：任何一条对不上，整条判定就不能算数（binding=not_in_context），
 * 退回相似度。不能"对上一半就算可答"——那正是 v1 单条 schema 在多法问题上的漏洞。
 *
 * `quoteVerified` 是"判官指名了某条但其实没在读它"的确定性抓手，
 * 但**从不参与判定**：字符串匹配是脆的（摘录被截断、条号写法不一致），
 * faithfulness 判官那边已经吃过这个亏。它只作为诊断数字上报。
 */
export function crossCheckAnswerability(
    parsed: ParsedAnswerability,
    retrieved: RetrievedChunk[],
): AnswerabilityCheck {
    if (!parsed.answerable) {
        return {
            binding: 'not_applicable',
            matchedIndexes: [],
            unmatched: [],
            quoteVerified: null,
            refusalNamedArticle: parsed.requirements.length > 0,
        };
    }

    const matchedIndexes: (number | null)[] = [];
    const unmatched: { law: string; articleNo: number }[] = [];

    for (const req of parsed.requirements) {
        const wantLaw = normLaw(req.law);
        let matched: number | null = null;
        for (let i = 0; i < retrieved.length; i++) {
            const c = retrieved[i];
            const gotLaw = normLaw(c.source);
            const lawOk = wantLaw !== '' && gotLaw !== ''
                && (wantLaw === gotLaw || wantLaw.includes(gotLaw) || gotLaw.includes(wantLaw));
            if (!lawOk) continue;
            if (chunkArticleNumber(c) !== req.articleNo) continue;
            matched = i;
            break;
        }
        matchedIndexes.push(matched);
        if (matched === null) unmatched.push({ law: req.law, articleNo: req.articleNo });
    }

    if (unmatched.length > 0) {
        return { binding: 'not_in_context', matchedIndexes, unmatched, quoteVerified: null, refusalNamedArticle: false };
    }

    let anyCheckable = false;
    let allQuotesFound = true;
    for (let i = 0; i < parsed.requirements.length; i++) {
        const quote = parsed.requirements[i].quote;
        if (!quote) continue;
        const q = normText(quote);
        // 太短的摘录（如两个字的词）匹配上没有信息量，判为"无法核对"
        if (q.length < 6) continue;
        anyCheckable = true;
        const doc = retrieved[matchedIndexes[i] as number].document;
        if (!normText(doc).includes(q)) allQuotesFound = false;
    }

    return {
        binding: 'in_context',
        matchedIndexes,
        unmatched: [],
        quoteVerified: anyCheckable ? allQuotesFound : null,
        refusalNamedArticle: false,
    };
}

// ---------------------------------------------------------------------------
// 判官结局与合并
// ---------------------------------------------------------------------------

export type JudgeOutcome =
    | { kind: 'judged'; parsed: ParsedAnswerability; check: AnswerabilityCheck }
    | { kind: 'binding_mismatch'; parsed: ParsedAnswerability; check: AnswerabilityCheck }
    | { kind: 'skipped'; reason: JudgeSkipReason }
    | { kind: 'parse_failure'; error: string; raw: string }
    | { kind: 'error'; error: string };

/**
 * 把判官结局并进相似度判定。
 *
 * 失败路径一律**原样返回 `sim`**（只换 reason 便于归因）。为什么不是二选一：
 * - 默认拒答 ⇒ **判官**故障导致**拒答**，而误拒是这套系统里最贵的错误方向；
 * - 默认回答 ⇒ 解析失败静默变成"可答判定"，违反上面那条不变量；
 * - 退回 `sim` ⇒ 什么都不发明，而且它恰好就是今天的行为——
 *   判官宕机时退化到**已测量的现状**，而不是退化到一个没人量过的状态。
 */
export function mergeAbstention(sim: AbstentionDecision, outcome: JudgeOutcome): AbstentionDecision {
    switch (outcome.kind) {
        case 'judged':
            if (outcome.check.binding === 'not_in_context') {
                // 说了可答却**有**指不出的必答项 —— 至少有一半判断来自记忆，不是来自给它的材料。
                // 多需求是合取：缺一条就整条判定作废，不能"对上一半就算可答"。
                return { ...sim, reason: 'judge_binding_mismatch' };
            }
            return {
                abstained: !outcome.parsed.answerable,
                topScore: sim.topScore,
                hasExactArticleHit: sim.hasExactArticleHit,
                reason: outcome.parsed.answerable ? 'answered' : 'judge_unanswerable',
            };
        case 'binding_mismatch':
            return { ...sim, reason: 'judge_binding_mismatch' };
        case 'parse_failure':
            return { ...sim, reason: 'judge_parse_failure' };
        case 'error':
            return { ...sim, reason: 'judge_error_fallback' };
        case 'skipped':
            // 跳过时判定完全由相似度规则决定，sim 里已经是正确的结论
            return { ...sim };
    }
}

// ---------------------------------------------------------------------------
// 判官驱动二次检索：只补"判官明确说缺的那部法"
// ---------------------------------------------------------------------------

/**
 * 第一次判官否决后，决定**要不要**只针对缺失的法律再检索一次。
 *
 * 触发是确定性的，不让模型自由发挥：
 *   1. `binding_mismatch`：判官指名了某条，但那条没在上下文里 → 到该法范围内再捞一次；
 *   2. 判官判不可答，且 `missing`/`reason` 点名了某部**库内**法律，而本次检索里
 *      该法一条都没有 → 到该法范围内再捞一次。
 *
 * 库外法律（刑法、劳动合同法…）不在 `corpusLaws` 里，不会触发——拒答它们是正确的。
 * 返回的是法律名列表（最多 2 个），具体取哪几条由 `retrieveInLaw` 决定。
 */
export function planSecondaryRetrieval(
    parsed: ParsedAnswerability,
    check: AnswerabilityCheck,
    retrieved: RetrievedChunk[],
    corpusLaws: string[],
): string[] {
    const present = new Set(retrieved.map(r => normLawName(r.source)));
    const planned: string[] = [];
    const add = (law: string) => {
        const n = normLawName(law);
        if (corpusLaws.includes(n) && !planned.includes(n)) planned.push(n);
    };

    if (check.binding === 'not_in_context') {
        for (const u of check.unmatched) add(u.law);
    }
    if (!parsed.answerable) {
        const text = `${parsed.missing ?? ''} ${parsed.reason}`;
        for (const law of corpusLaws) {
            if (text.includes(law) && !present.has(law)) add(law);
        }
    }
    return planned.slice(0, 2);
}

// ---------------------------------------------------------------------------
// 判官看到的上下文
// ---------------------------------------------------------------------------

/**
 * 拼给判官看的检索结果。**与 `formatContext()` 分开，而且刻意不给 `相关度`。**
 *
 * 理由：把检索分喂给判官，等于把"已被证明不可校准的那个信号"从后门重新引进来，
 * 判官会锚在它上面——而这一整套东西存在的理由就是那个信号不可用。
 * 编号 `[1]…[n]` 让判官能引用"第几块"，但不让它看见那一块的分数。
 */
export function formatJudgeContext(chunks: RetrievedChunk[]): string {
    if (chunks.length === 0) return '（本地知识库未检索到相关条文）';
    return chunks.map((c, i) => {
        const head = [
            `[${i + 1}] 来源：${c.source}`,
            c.chapter ? `章节：${c.chapter}` : null,
            c.articleNo ? `条号：${c.articleNo}` : null,
        ].filter(Boolean).join(' | ');
        return `${head}\n${c.document}`;
    }).join('\n\n---\n\n');
}
