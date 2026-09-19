/**
 * 语料清单刷新（`pnpm corpus:manifest`）。
 *
 * 独立成入口的原因：清单记录的是**语料本身**的事实（条数、指纹、权威来源），
 * 与"有没有成功算完 embedding"无关。原先它只挂在 `pnpm embed` 末尾，
 * 一旦 embedding 接口不通（网络被墙、key 失效），CORPUS.json 就会停留在旧语料上，
 * 而它恰恰是 eval 报告用来标注"这份分数对应哪一版语料"的依据——留着旧的就是在说谎。
 */
import 'dotenv/config';
import { buildManifest, writeManifest, readManifest } from './corpus.js';

export const PROVENANCE =
    '官方全文，经两独立官方源逐条交叉校验（见 knowledge/SOURCES.json 的 sources 与 verification 字段）；' +
    '构建脚本 tools/build_corpus.py，可复现';

/**
 * 重算清单并落盘。语料内容一变就自动 +1 —— 版本号必须与内容一一对应，
 * 否则 eval 报告里记的 version 指不回一份确定的语料，分数就不可追溯。
 */
export function refreshManifest() {
    const prev = readManifest();
    let manifest = buildManifest(prev?.version ?? 1, PROVENANCE);
    const changed = !!prev && prev.corpusHash !== manifest.corpusHash;
    if (changed) manifest = { ...manifest, version: (prev!.version ?? 1) + 1 };
    writeManifest(manifest);
    return { manifest, prev, changed };
}

if (process.argv[1]?.endsWith('manifest.ts') || process.argv[1]?.endsWith('manifest.js')) {
    const { manifest, prev, changed } = refreshManifest();
    console.log(`语料清单已刷新：v${manifest.version}，${manifest.totalArticles} 条，指纹 ${manifest.corpusHash}`);
    for (const f of manifest.files) {
        console.log(`  ${f.file} → ${f.articles} 条，${f.bytes} 字节，sha1 ${f.sha1}`);
    }
    if (changed) {
        console.warn(
            `\n⚠️  语料内容已变化（旧指纹 ${prev!.corpusHash} → 新 ${manifest.corpusHash}），` +
            `版本号 v${prev!.version} → v${manifest.version}。\n` +
            `   recall@k / MRR 在不同语料版本之间**不可比**：候选池变了，分母也就变了。\n` +
            `   已有 eval 数据集的 ground truth 条号必须重新校对后才能与旧分数对比。`
        );
    }
}
