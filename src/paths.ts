import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

/**
 * 项目根目录解析。
 *
 * 为什么不用 `__dirname` 或 `process.cwd()`：
 * - `__dirname` 在 `tsc` 编译后变成 `<root>/dist/src`，`path.join(__dirname, '../knowledge')`
 *   会指向 `<root>/dist/knowledge`（不存在）——这正是 `pnpm start` 曾经静默空转的原因；
 * - `process.cwd()` 依赖调用者在哪个目录敲命令，从子目录启动就错。
 *
 * 改为从当前模块位置向上找 `package.json`，dev（tsx 跑 src/）与 build（node 跑 dist/src/）
 * 两种布局都能落到同一个真实项目根。
 */
function findProjectRoot(): string {
    let dir = path.dirname(fileURLToPath(import.meta.url));
    for (let i = 0; i < 6; i++) {
        if (fs.existsSync(path.join(dir, 'package.json'))) return dir;
        const parent = path.dirname(dir);
        if (parent === dir) break;
        dir = parent;
    }
    // 兜底：退回 cwd，并明确告警，而不是静默用错路径
    console.warn('[paths] 未找到 package.json，回退到 process.cwd()；knowledge/prompts 可能加载失败');
    return process.cwd();
}

export const PROJECT_ROOT = findProjectRoot();
export const KNOWLEDGE_DIR = path.join(PROJECT_ROOT, 'knowledge');
export const PROMPTS_DIR = path.join(PROJECT_ROOT, 'prompts');
export const CACHE_DIR = path.join(PROJECT_ROOT, 'cache');
export const OUTPUT_DIR = path.join(PROJECT_ROOT, 'output');
export const FRONTEND_DIR = path.join(PROJECT_ROOT, 'frontend');
export const EVAL_DIR = path.join(PROJECT_ROOT, 'eval');
