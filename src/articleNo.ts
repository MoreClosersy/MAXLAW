/**
 * 法条条号的解析与归一。
 *
 * 独立成模块的原因：**同一套条号语义有三个消费方**——`splitIntoChunks` 切分时要认条号，
 * `lintCorpus` 查重/查跳号时要算条号，引用校验器回查模型答案时也要算条号。
 * 三处各写一份迟早会漂移，而漂移的表现是"校验器说这个条号不存在、但它其实存在"，
 * 属于最不该出错的那类 bug。
 *
 * 关键点：**模型写的是阿拉伯数字还是中文数字，语料里存的是哪一种，都可能不一致**。
 * 所以对外一律用数值比较，不用字符串比较。
 */

const CN_DIGITS: Record<string, number> = {
    '零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
    '六': 6, '七': 7, '八': 8, '九': 9,
};

/** 中文数字转阿拉伯数字，支持到万位（法条条号最多四位，够用） */
export function cnToNumber(cn: string): number {
    const s = cn.replace(/^第/, '').replace(/条$/, '').trim();
    if (/^\d+$/.test(s)) return Number(s);
    let total = 0, section = 0, num = 0;
    for (const ch of s) {
        if (ch in CN_DIGITS) {
            num = CN_DIGITS[ch];
        } else if (ch === '十') {
            section += (num || 1) * 10; num = 0;
        } else if (ch === '百') {
            section += (num || 1) * 100; num = 0;
        } else if (ch === '千') {
            section += (num || 1) * 1000; num = 0;
        } else if (ch === '万') {
            total += (section + num) * 10000; section = 0; num = 0;
        } else {
            return NaN;
        }
    }
    return total + section + num;
}

/** 从「第X条」形式的字符串取出条号数值；不是条号则返回 null */
export function parseArticleNo(text: string | null | undefined): number | null {
    if (!text) return null;
    const m = text.match(/^第\s*([0-9]+|[一二三四五六七八九十百千万零两]+)\s*条/);
    if (!m) return null;
    const n = cnToNumber(m[1]);
    return Number.isFinite(n) ? n : null;
}

const CITATION_RE = /第\s*([0-9]+|[一二三四五六七八九十百千万零两]+)\s*条/g;

/**
 * 抽出文本里提到的全部条号。
 *
 * 注意是"全文扫描"而不是"只扫开头"：模型会在句子中间写「依据第577条」，
 * 而检索到的条文正文里也会出现「依照本法第X条」这类交叉引用——
 * 后者同样需要被识别出来，否则会把"引用原文里的交叉引用"误判成"编造条号"。
 *
 * 返回 Map<条号数值, 该条号首次出现时的原文写法>。
 */
export function extractArticleNumbers(text: string): Map<number, string> {
    const out = new Map<number, string>();
    for (const m of text.matchAll(CITATION_RE)) {
        const n = cnToNumber(m[1]);
        if (!Number.isFinite(n) || n <= 0) continue;
        if (!out.has(n)) out.set(n, m[0].replace(/\s+/g, ''));
    }
    return out;
}
