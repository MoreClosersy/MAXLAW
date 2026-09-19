/**
 * 语料结构校验（`pnpm lint:corpus`）。
 *
 * 这个工具不能判断"法条内容是否与官方文本一致"——那需要权威来源比对。
 * 但它能抓出一类**无需外部数据就能确定是错的**结构性缺陷：
 *   1. 同一部法律内条号重复；
 *   2. 不同条号内容完全相同（几乎必然是抄录错位）；
 *   3. 条号跳号 / 乱序；
 *   4. 条文正文异常短（疑似截断）。
 *
 * 这类缺陷对本项目危害特别大：eval 的 ground truth 是条号，引用校验器也按条号回查。
 * 语料错位 = 拿错的标准答案去判分，会把正确回答判成错的。
 */
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { KNOWLEDGE_DIR } from './paths.js';
import { cnToNumber } from './articleNo.js';

const ARTICLE_LINE_RE = /^(第[一二三四五六七八九十百千万零\d]+条)\s*([\s\S]*?)(?=\n\n|$)/;

interface Article {
    no: string;
    numeric: number;
    body: string;
    file: string;
    line: number;
    /** 所属编/章/节路径，用于区分"同章错位"与"跨章重复条款" */
    chapter: string;
}

function parseArticles(file: string): Article[] {
    const content = fs.readFileSync(path.join(KNOWLEDGE_DIR, file), 'utf-8');
    const blocks = content.split(/\n{2,}/);
    const out: Article[] = [];
    const stack: { level: number; title: string }[] = [];
    let lineNo = 1;
    for (const block of blocks) {
        const trimmed = block.trim();
        const heading = trimmed.match(/^(#{1,6})\s+(.*)$/);
        if (heading) {
            const level = heading[1].length;
            while (stack.length && stack[stack.length - 1].level >= level) stack.pop();
            stack.push({ level, title: heading[2].trim() });
        } else {
            const m = trimmed.match(ARTICLE_LINE_RE);
            if (m) {
                out.push({
                    no: m[1],
                    numeric: cnToNumber(m[1]),
                    body: trimmed.slice(m[1].length).trim(),
                    file,
                    line: lineNo,
                    chapter: stack.map(h => h.title).join(' > '),
                });
            }
        }
        lineNo += block.split('\n').length + 1;
    }
    return out;
}

interface Finding {
    severity: 'error' | 'warn';
    kind: string;
    message: string;
}

function lint(): Finding[] {
    const files = fs.readdirSync(KNOWLEDGE_DIR).filter(f => f.endsWith('.md')).sort();
    const findings: Finding[] = [];

    // 按"法律"分组：民法典系列的条号是同一套序列，公司法是另一套
    const groups = new Map<string, Article[]>();
    for (const file of files) {
        const law = file.startsWith('民法典') ? '民法典' : file.replace(/\.md$/, '');
        const arr = groups.get(law) ?? [];
        arr.push(...parseArticles(file));
        groups.set(law, arr);
    }

    for (const [law, articles] of groups) {
        // 1. 条号重复
        const byNo = new Map<string, Article[]>();
        for (const a of articles) {
            const arr = byNo.get(a.no) ?? [];
            arr.push(a);
            byNo.set(a.no, arr);
        }
        for (const [no, arr] of byNo) {
            if (arr.length > 1) {
                findings.push({
                    severity: 'error',
                    kind: 'duplicate-article-no',
                    message: `《${law}》${no} 重复出现 ${arr.length} 次：${arr.map(a => a.file).join(', ')}`,
                });
            }
        }

        // 2. 不同条号、正文完全相同
        const byBody = new Map<string, Article[]>();
        for (const a of articles) {
            if (a.body.length < 10) continue;
            const key = crypto.createHash('sha1').update(a.body).digest('hex');
            const arr = byBody.get(key) ?? [];
            arr.push(a);
            byBody.set(key, arr);
        }
        for (const arr of byBody.values()) {
            const distinct = Array.from(new Set(arr.map(a => a.no)));
            if (distinct.length < 2) continue;
            // 同章内异号同文 → 几乎必然是抄录错位，判 error。
            // 跨章同文 → 法典里本来就有重复条款（如民法典第960条「行纪合同」与
            // 第966条「中介合同」章末都是"本章没有规定的，参照适用委托合同的有关规定"），
            // 只提示复核，不判错。
            const chapters = Array.from(new Set(arr.map(a => a.chapter)));
            const sameChapter = chapters.length === 1;
            findings.push({
                severity: sameChapter ? 'error' : 'warn',
                kind: 'duplicate-article-body',
                message: sameChapter
                    ? `《${law}》${distinct.join(' / ')} 同属「${chapters[0]}」且正文完全相同——几乎必然是抄录错位：「${arr[0].body.slice(0, 40)}…」`
                    : `《${law}》${distinct.join(' / ')} 分属不同章节但正文相同（${chapters.join(' / ')}）——可能是法典本身的重复条款，请对照官方文本复核：「${arr[0].body.slice(0, 40)}…」`,
            });
        }

        // 3. 乱序 / 跳号
        const sorted = [...articles].sort((a, b) => a.line - b.line);
        for (let i = 1; i < sorted.length; i++) {
            if (sorted[i].file !== sorted[i - 1].file) continue;
            const delta = sorted[i].numeric - sorted[i - 1].numeric;
            if (delta <= 0) {
                findings.push({
                    severity: 'error',
                    kind: 'out-of-order',
                    message: `${sorted[i].file}: ${sorted[i - 1].no} 之后出现 ${sorted[i].no}（条号未递增）`,
                });
            } else if (delta > 1) {
                findings.push({
                    severity: 'warn',
                    kind: 'gap',
                    message: `${sorted[i].file}: ${sorted[i - 1].no} → ${sorted[i].no} 之间缺 ${delta - 1} 条`,
                });
            }
        }

        // 4. 异常短
        for (const a of articles) {
            if (a.body.length < 12) {
                findings.push({
                    severity: 'warn',
                    kind: 'suspiciously-short',
                    message: `${a.file}: ${a.no} 正文仅 ${a.body.length} 字，疑似截断：「${a.body}」`,
                });
            }
        }
    }

    return findings;
}

const findings = lint();
const errors = findings.filter(f => f.severity === 'error');
const warns = findings.filter(f => f.severity === 'warn');

console.log(`\n语料结构校验：${errors.length} 个 error，${warns.length} 个 warning\n`);
for (const f of errors) console.log(`  ❌ [${f.kind}] ${f.message}`);
if (errors.length && warns.length) console.log('');
for (const f of warns) console.log(`  ⚠️  [${f.kind}] ${f.message}`);

console.log(
    `\n本工具只验证**结构**（条号重复/错位/跳号/截断），不验证条文内容。\n` +
    `内容层面的权威性由 tools/build_corpus.py 保证：它从两个独立官方源抓取全文、逐条比对，\n` +
    `并把来源 URL、抓取件 sha256、校验项写入 knowledge/SOURCES.json，可独立复核。\n`
);

process.exit(errors.length > 0 ? 1 : 0);
