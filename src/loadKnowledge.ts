/**
 * 把 knowledge/ 下的全部 .md 灌进检索器。
 *
 * 这段逻辑原本在 `server.ts` 和 `src/embed.ts` 里各有一份，评测框架要第三份——
 * 三份拷贝迟早漂移，而漂移的表现是"评测跑出来的召回率和服务实际行为对不上"，
 * 属于最难发现的那类不一致。抽成一份，三处调用。
 */
import fs from 'fs';
import path from 'path';
import EmbeddingRetriever from './EmbeddingRetriever.js';
import { KNOWLEDGE_DIR } from './paths.js';

export interface LoadKnowledgeResult {
    files: string[];
    chunks: number;
    /** 走了确定性兜底向量的次数；> 0 表示本次索引不可信 */
    fallbackHits: number;
    embedCalls: number;
}

export interface LoadKnowledgeHooks {
    /** 每个文件嵌入进度 */
    onProgress?: (file: string, done: number, total: number) => void;
    /** 每个文件完成 */
    onFileDone?: (file: string, chunks: number) => void;
    /** 加载前的提示（各调用方的措辞不同） */
    onStart?: (fileCount: number) => void;
}

export async function loadKnowledge(
    retriever: EmbeddingRetriever,
    hooks: LoadKnowledgeHooks = {},
): Promise<LoadKnowledgeResult> {
    if (!fs.existsSync(KNOWLEDGE_DIR)) {
        fs.mkdirSync(KNOWLEDGE_DIR, { recursive: true });
        throw new Error(`knowledge/ 目录不存在，已创建但内容为空：${KNOWLEDGE_DIR}`);
    }
    const files = fs.readdirSync(KNOWLEDGE_DIR).filter(f => f.endsWith('.md')).sort();
    if (files.length === 0) {
        throw new Error(`knowledge/ 下没有 .md 文件：${KNOWLEDGE_DIR}`);
    }

    hooks.onStart?.(files.length);
    let chunks = 0;
    for (const file of files) {
        const content = fs.readFileSync(path.join(KNOWLEDGE_DIR, file), 'utf-8');
        const n = await retriever.embedDocument(content, file, (done, total) => {
            hooks.onProgress?.(file, done, total);
        });
        chunks += n;
        hooks.onFileDone?.(file, n);
    }

    const stats = retriever.getStats();
    return { files, chunks, fallbackHits: stats.fallbackHits, embedCalls: stats.embedCalls };
}
