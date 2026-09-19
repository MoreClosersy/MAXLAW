/**
 * faithfulness 判官的**纯逻辑**部分：解析判官输出、构造已知缺陷的校准样本。
 *
 * ## 为什么单独一个文件
 *
 * 这些函数必须能被单测覆盖，而 `eval/faithfulness.ts` 是个有副作用的脚本——
 * 顶层就 `await loadKnowledge()` 建索引、跑 27 道题的生成。测试一旦 import 它，
 * 就会把整个评测跑一遍（第一次就是这么挂住的：`pnpm test` 卡在 2 分钟没动）。
 * 所以判据是：**只要一个函数需要被测试，它就不能住在有副作用模块的顶层。**
 *
 * 这里没有任何 import 副作用——不读 .env、不建索引、不发网络请求。
 */

import { extractArticleNumbers } from '../src/articleNo.js';

export interface Claim { text: string; verdict: 'supported' | 'unsupported'; evidence: string }

export interface ParsedJudge { claims: Claim[]; summary: string }

/**
 * 解析判官输出。
 *
 * **反自欺不变量：判官返回坏 JSON 时，绝不能被算成"忠实"。**
 * 这类评测最常见的自欺方式是——模型偶尔不按格式输出，脚本 catch 住错误、
 * 把这条从分母里悄悄去掉，于是分数反而好看了。所以坏输入必须走 `error` 分支，
 * 由调用方单独计数并**显式写进报告**：剔除不是问题，"剔除了却不报"才是。
 */
export function parseJudgeOutput(raw: string): ParsedJudge | { error: string } {
    let text = raw.trim();
    // 模型偶尔会无视"不要用代码块"的指令，套一层 ```json。剥掉。
    const fence = text.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/);
    if (fence) text = fence[1];

    let obj: unknown;
    try {
        obj = JSON.parse(text);
    } catch {
        // 兜底：从第一个 { 到最后一个 } 之间截一段再试一次
        const i = text.indexOf('{'); const j = text.lastIndexOf('}');
        if (i < 0 || j <= i) return { error: `不是 JSON：${raw.slice(0, 160)}` };
        try { obj = JSON.parse(text.slice(i, j + 1)); }
        catch { return { error: `不是 JSON：${raw.slice(0, 160)}` }; }
    }
    if (typeof obj !== 'object' || obj === null || Array.isArray(obj)) return { error: '顶层不是对象' };
    const o = obj as Record<string, unknown>;
    if (!Array.isArray(o.claims)) return { error: 'claims 不是数组' };

    const claims: Claim[] = [];
    for (const c of o.claims) {
        if (typeof c !== 'object' || c === null) continue;
        const cc = c as Record<string, unknown>;
        const verdict = cc.verdict;
        // verdict 非法的条目直接丢弃，但**不因为丢弃而报错**——只要还有合法条目，
        // 分母就是剩下的这些；丢了几个会体现在 claims.length 上，可复核。
        if (verdict !== 'supported' && verdict !== 'unsupported') continue;
        claims.push({ text: String(cc.text ?? ''), verdict, evidence: String(cc.evidence ?? '') });
    }
    return { claims, summary: String(o.summary ?? '') };
}

export interface Mutant {
    kind: string;
    /** 被注入进去的标志物。**只用于报告**，不用于判定"抓到了没有"，理由见 `caughtBy`。 */
    needle: string;
    answer: string;
    note: string;
}

/**
 * "抓到了没有"的判据：**变体里出现了任何一条 unsupported 断言**。
 *
 * 为什么不用"判官点出了 needle 字符串"：那个判据被证明是不可靠的。判官的 `text`
 * 是答案的**不超过 40 字摘录**，条号经常落在摘录之外；而同一个条号在答案里写「第577条」、
 * 在语料里写「第五百七十七条」，字符串根本对不上。首轮冒烟测试 3 道题的 `article_swap`
 * 全判"没抓到"，但其中至少有一例（把违约责任换成民法典第1164条"本编调整因侵害民事权益
 * 产生的民事关系"）判官几乎不可能真的认为它 supported——**是尺子在骗人，不是判官在漏判**。
 *
 * 改成结构判据的前提是：**校准样本只从"原本完全忠实"的答案里造**（见 `faithfulness.ts`
 * 里的筛选）。这样原始答案是 0 条 unsupported，变体里但凡出现 unsupported，
 * 就只能是注入进去的那个缺陷。判据因此不再依赖任何字符串匹配，也就不会因为
 * 模型的措辞或数字写法而飘。已经带 unsupported 的答案一律不参与校准——
 * 那会让"原本就有的问题"和"注入的问题"混在一起，分母变成噪声。
 */
export function caughtBy(claims: { verdict: string }[]): boolean {
    return claims.some(c => c.verdict === 'unsupported');
}

/** 出现即视为"被改到了"的数目词对。顺序即优先级——先匹配到的先用。 */
export const NUMBER_SWAPS: [string, string][] = [
    ['八周岁', '十八周岁'],
    ['三十日', '九十日'],
    ['六十日', '三十日'],
    ['十五日', '六十日'],
    ['三年', '十年'],
    ['二年', '五年'],
    ['一年', '三年'],
    ['六个月', '十二个月'],
];

/** 追加断言候选。第一个关键短语不在语料里的会被选中。 */
export const UNSUPPORTED_APPENDS: [string, string][] = [
    ['人民调解委员会调解', '另外需要说明的是，此类纠纷必须先经人民调解委员会调解，未经调解不得直接向人民法院提起诉讼。'],
    ['仲裁前置', '另外需要说明的是，此类争议属于仲裁前置事项，未经仲裁不得向人民法院起诉。'],
    ['公示催告', '另外需要说明的是，权利人应当先向人民法院申请公示催告，催告期满后方可主张权利。'],
];

/**
 * 把 `formatContext()` 的输出拆回"条"。
 *
 * 格式见 `src/index.ts` 的 `formatContext`：每块以
 * `[i] 来源：<file>#<anchor> | 章节：… | 条号：<no> | 相关度：…` 开头，块间用 `\n\n---\n\n` 分隔。
 * 这里只取两样东西：**来源**（用来判断是不是同一部法）和**条号**。
 */
export interface ContextBlock {
    /** 完整来源，形如 `民法典.md#第一百八十八条`（含锚点，便于人读） */
    source: string;
    /** 只取文件名。**判断"是不是同一部法"必须用这个**——
     *  锚点是逐条不同的，拿 `source` 比会把同一部法的每一条都当成不同法源，
     *  于是"跨法源换号"退化成"同法源乱换"，正好造出假缺陷。 */
    file: string;
    articleNo: number | null;
    articleNoRaw: string | null;
    /** 条文正文（不含头部那行）。用来判断"换过去的那条会不会碰巧也支持这句话"。 */
    body: string;
}

/** 字符二元组集合。中文没有空格分词，二元组是够用的粗粒度相似度。 */
function bigrams(text: string): Set<string> {
    const t = text.replace(/\s/g, '');
    const out = new Set<string>();
    for (let i = 0; i + 1 < t.length; i++) out.add(t.slice(i, i + 2));
    return out;
}

/** Jaccard 相似度。0 = 完全不共享用词，1 = 用词一样。 */
export function similarity(a: string, b: string): number {
    const A = bigrams(a), B = bigrams(b);
    if (A.size === 0 || B.size === 0) return 0;
    let inter = 0;
    for (const g of A) if (B.has(g)) inter++;
    return inter / (A.size + B.size - inter);
}

export function parseContextBlocks(context: string): ContextBlock[] {
    return context.split(/\n\n---\n\n/).map(block => {
        const head = block.split('\n')[0] ?? '';
        const source = head.match(/来源：([^\s|]+)/)?.[1] ?? '';
        const raw = head.match(/条号：([^\s|]+)/)?.[1] ?? null;
        const no = raw ? extractArticleNumbers(raw) : new Map();
        const body = block.split('\n').slice(1).join('\n').trim();
        return { source, file: source.split('#')[0], articleNo: [...no.keys()][0] ?? null, articleNoRaw: raw, body };
    }).filter(b => b.source !== '');
}

/**
 * 构造**已知缺陷**的校准样本。四种注入对应四类真实会发生的失败：
 *
 *   fabricated_cite     —— 编一个语料里根本不存在的条号（最粗暴，判官必须抓到）
 *   distorted_number    —— 把条文里的数字改掉（最隐蔽，最考验判官）
 *   article_swap        —— 引用一个在 context 里存在、但内容对不上的条号（"条号对、内容不对"）
 *   unsupported_append  —— 追加一句 context 里没有依据的断言
 *
 * 每个注入都要**确认真的改动了**：目标字符串没找到就跳过，而不是返回一个
 * "看起来变了其实没变"的样本。没改到的变体会让检出率变成噪声——判它忠实是对的，
 * 却会被记成"没抓到"。所以"没改到"必须是**不生成**，不能是原样返回。
 */
export function mutate(answer: string, context: string, corpus: string): Mutant[] {
    const out: Mutant[] = [];

    // 1) 编造条号。取语料最大条号再往上加，保证"不存在"这一点是确定的。
    //    条号用 `extractArticleNumbers` 而不是自己写正则——模型写「第188条」、
    //    语料写「第一百八十八条」，**两边的数字形式经常不一致**，
    //    这正是 `src/articleNo.ts` 存在的理由。第一版在这里自己写了一版只认中文数字的
    //    正则，于是 `article_swap` 一次都没生成出来（答案里是阿拉伯数字，一个都没匹配上），
    //    而校准会把"没生成"静默地当成"这一类注入不适用"。单测抓住了它。
    const maxNo = Math.max(0, ...extractArticleNumbers(corpus).keys());
    if (maxNo > 0) {
        const fake = maxNo + 777;
        out.push({
            kind: 'fabricated_cite',
            needle: String(fake),
            answer: answer + `\n\n此外，根据《中华人民共和国民法典》第${fake}条，该情形还应当承担相应的法律责任。`,
            note: `追加引用不存在的第${fake}条（语料最大条号为 ${maxNo}）`,
        });
    }

    // 2) 改数字。挑一个**答案和 context 里都出现**的数目词。
    //    两边都出现是必要条件：否则改完之后那句话本来就没依据，考不出"数字被篡改"。
    for (const [from, to] of NUMBER_SWAPS) {
        if (answer.includes(from) && context.includes(from)) {
            out.push({
                kind: 'distorted_number', needle: to,
                answer: answer.replaceAll(from, to),
                note: `把「${from}」改成「${to}」`,
            });
            break;
        }
    }

    // 3) 条号张冠李戴。找答案里引用的、且在 context 中真实存在的条号，
    //    换成 context 里**另一部法**的条号——句子没变，出处错了。
    //
    //    为什么必须是"另一部法"：第一版换成了 context 里随便另一条，结果 lookup-02 把
    //    《公司法》第23条换成了第232条，而第232条写的正是"公司因…解散的，应当清算"——
    //    答案那句话**真的被 context 支持**。判官判它 supported 是对的，却被记成"漏判"。
    //    这不是判官的问题，是我造了一个**根本不是缺陷的缺陷**。跨法源换号（公司法→民法典）
    //    才几乎必然错：民法典的条文不会支持一句关于公司清算的断言。
    //    context 里只有一部法时**不生成**，而不是退化成同法源乱换——见文件头的纪律：
    //    "没改到"必须是不生成。
    const blocks = parseContextBlocks(context);
    const inAnswer = extractArticleNumbers(answer);
    const fromNo = [...inAnswer.keys()].find(n => blocks.some(b => b.articleNo === n));
    const fromBlock = blocks.find(b => b.articleNo === fromNo);
    // 选目标：优先另一部法；没有别的法时，退而取**与源条文用词重合最少**的那条。
    // 重合度必须足够低才生成——lookup-02 的教训是换过去的那条（公司法第232条"应当清算"）
    // 恰好支持了原句，于是"缺陷"根本不是缺陷，判官判 supported 反倒是正确的。
    // 低重合不能**保证**不碰巧支持，但把概率压下去；剩下的是残留风险，
    // 靠 faithfulness.ts 在漏判时打印判官原话，让人能一眼看出是哪一种。
    const candidates = blocks.filter(b => b.articleNo !== fromNo && b.articleNo !== null);
    const crossLaw = candidates.filter(b => b.file !== fromBlock?.file);
    let toBlock = crossLaw[0];
    if (!toBlock && fromBlock) {
        const scored = candidates
            .map(b => ({ b, sim: similarity(fromBlock.body, b.body) }))
            .sort((x, y) => x.sim - y.sim);
        if (scored.length > 0 && scored[0].sim < 0.15) toBlock = scored[0].b;
    }
    if (fromNo !== undefined && fromBlock && toBlock) {
        const fromRaw = inAnswer.get(fromNo)!;
        // 数字写法跟原答案保持一致（答案写阿拉伯数字就还写阿拉伯数字），
        // 否则变体会因为"数字风格突变"而被判官一眼看出，考不出真正想考的东西。
        const toRaw = /^第\d+条$/.test(fromRaw) ? `第${toBlock.articleNo}条` : (toBlock.articleNoRaw ?? `第${toBlock.articleNo}条`);
        out.push({
            kind: 'article_swap', needle: toRaw,
            answer: answer.replaceAll(fromRaw, toRaw),
            note: `把「${fromRaw}」（${fromBlock.file}）换成「${toRaw}」（${toBlock.file}` +
                `${toBlock.file === fromBlock.file ? '，同法源但用词重合最低' : '，另一部法'}）`,
        });
    }

    // 4) 追加一句没有依据的断言。挑一句**确认不在语料里**的说法，
    //    否则它可能碰巧是有依据的，校准就失效了。
    for (const [phrase, sentence] of UNSUPPORTED_APPENDS) {
        if (!corpus.includes(phrase)) {
            out.push({
                kind: 'unsupported_append', needle: phrase,
                answer: answer + `\n\n${sentence}`,
                note: `追加无依据断言（关键短语「${phrase}」不在语料中）`,
            });
            break;
        }
    }

    return out;
}
