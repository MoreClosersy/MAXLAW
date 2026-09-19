import fs from 'fs';
import path from 'path';
import { PROMPTS_DIR } from './paths.js';

export interface LoadedPrompt {
    name: string;
    version: number;
    body: string;
}

/**
 * 加载带版本的 prompt 文件（`prompts/<name>.v<N>.md`）。
 *
 * Prompt 是系统架构的一部分，按配置管理而不是内联在代码里：
 * - 改 prompt 的 diff 不再和逻辑代码混在一起，review 时看得清；
 * - 版本号显式化，eval 报告可以记录"这次跑的是哪一版"；
 * - 版本控制直接复用 git 历史，不需要额外的 prompt 管理平台。
 *
 * 文件头部的 YAML front-matter 是元数据，不进入发给模型的正文。
 */
export function loadPrompt(name: string, version: number = 1): LoadedPrompt {
    const file = path.join(PROMPTS_DIR, `${name}.v${version}.md`);
    if (!fs.existsSync(file)) {
        throw new Error(`Prompt 文件不存在: ${file}`);
    }
    const raw = fs.readFileSync(file, 'utf-8');
    // 剥掉 front-matter（--- ... --- ），只保留正文
    const body = raw.startsWith('---')
        ? raw.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n/, '')
        : raw;
    return { name, version, body: body.trim() };
}
