/**
 * 法律名的识别与归一。
 *
 * 抽成独立模块，是因为**查询侧和答案侧都要用同一套名字**：
 *   - `EmbeddingRetriever.parseQuery` 用它把条号绑到法律（多法检索）；
 *   - `CitationVerifier` 用它判断「公司法第二十三条」指的是哪部法。
 * 两处各写一份的典型症状是「民法典.md」在一处算民法典、在另一处不算——而校验器
 * 一旦把真引用报成问题，用户就会学会忽略告警（见 CitationVerifier 的头注释）。
 *
 * 只放纯函数和常量，不 import 任何有副作用的东西：这个模块会被单测直接加载。
 */

/** 语料里可能出现的法律名（含库外名字：库里没有，但识别出来才能正确判断"库外"）。 */
export const KNOWN_LAW_NAMES = [
    '民法典', '合同法', '公司法', '劳动法', '知识产权', '刑法',
    '行政法', '婚姻法', '继承法', '侵权责任法', '消费者权益保护法',
];

export interface LawMention {
    name: string;
    start: number;
    end: number;
}

/** 「中华人民共和国民法典」「《民法典》」「民法典.md」→「民法典」 */
export function normLawName(s: string): string {
    return s
        .replace(/中华人民共和国/g, '')
        .replace(/[《》\s]/g, '')
        .replace(/\.md$/i, '')
        .trim();
}

/** 长名优先：当前列表没有前缀重叠，但别把这一点当巧合依赖。 */
function lawNameRegex(): RegExp {
    return new RegExp([...KNOWN_LAW_NAMES].sort((a, b) => b.length - a.length).join('|'), 'g');
}

/** 找出文本里所有法律名及其位置。matchAll 会克隆正则，不污染 lastIndex。 */
export function findLaws(text: string): LawMention[] {
    return [...text.matchAll(lawNameRegex())].map(m => ({
        name: m[0],
        start: m.index ?? 0,
        end: (m.index ?? 0) + m[0].length,
    }));
}
