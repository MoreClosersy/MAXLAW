import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { KNOWLEDGE_DIR, PROJECT_ROOT } from './paths.js';

export interface CorpusFileInfo {
    file: string;
    articles: number;
    bytes: number;
    sha1: string;
}

export interface CorpusSourceRecord {
    /** 法名 */
    title: string;
    articles: number;
    /** 成品 markdown 的 sha1，与 tools/build_corpus.py 的输出对齐 */
    markdownSha1: string;
    verifiedAt: string;
    sources: { role: string; name: string; url: string; rawSha256: string }[];
    /** 构建时实际跑过的校验项 */
    verification: string[];
    /** 对原文做过的显式变换（未列出的即为逐字保留） */
    transformations: string[];
}

export interface CorpusManifest {
    /** 语料版本号，扩语料时手动 +1 */
    version: number;
    generatedAt: string;
    totalArticles: number;
    /** 全部文件内容的合并指纹——eval 报告记录它，就能判断两次跑分是否同一语料 */
    corpusHash: string;
    files: CorpusFileInfo[];
    provenance: string;
    /** 逐文件的权威来源与校验记录，来自 knowledge/SOURCES.json（由 tools/build_corpus.py 写出） */
    sources?: Record<string, CorpusSourceRecord>;
}

const ARTICLE_RE = /^第[一二三四五六七八九十百千万零\d]+条/gm;
const MANIFEST_PATH = path.join(PROJECT_ROOT, 'knowledge', 'CORPUS.json');
const SOURCES_PATH = path.join(PROJECT_ROOT, 'knowledge', 'SOURCES.json');

/** 读取权威来源记录。文件缺失返回 undefined——不编造，如实留空。 */
export function readSources(): Record<string, CorpusSourceRecord> | undefined {
    if (!fs.existsSync(SOURCES_PATH)) return undefined;
    try {
        return JSON.parse(fs.readFileSync(SOURCES_PATH, 'utf-8'));
    } catch {
        console.warn('[corpus] knowledge/SOURCES.json 解析失败，来源记录将缺失');
        return undefined;
    }
}

export function buildManifest(version: number, provenance: string): CorpusManifest {
    const files = fs.readdirSync(KNOWLEDGE_DIR).filter(f => f.endsWith('.md')).sort();
    const infos: CorpusFileInfo[] = [];
    const hasher = crypto.createHash('sha1');

    for (const file of files) {
        const content = fs.readFileSync(path.join(KNOWLEDGE_DIR, file), 'utf-8');
        hasher.update(file).update(content);
        infos.push({
            file,
            articles: (content.match(ARTICLE_RE) || []).length,
            bytes: Buffer.byteLength(content, 'utf-8'),
            sha1: crypto.createHash('sha1').update(content).digest('hex').slice(0, 12),
        });
    }

    return {
        version,
        generatedAt: new Date().toISOString().slice(0, 10),
        totalArticles: infos.reduce((s, i) => s + i.articles, 0),
        corpusHash: hasher.digest('hex').slice(0, 16),
        files: infos,
        provenance,
        sources: readSources(),
    };
}

export function writeManifest(m: CorpusManifest) {
    fs.writeFileSync(MANIFEST_PATH, JSON.stringify(m, null, 2) + '\n', 'utf-8');
}

export function readManifest(): CorpusManifest | null {
    if (!fs.existsSync(MANIFEST_PATH)) return null;
    try {
        return JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf-8'));
    } catch {
        return null;
    }
}
