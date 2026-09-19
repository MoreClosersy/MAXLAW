/**
 * `pnpm embed` 入口：把 knowledge/ 全量索引一遍、落盘 embedding 缓存、刷新语料清单。
 *
 * 原本这个功能挂在 index.ts 底部的
 *   `if (typeof require !== 'undefined' && require.main === module)`
 * 里——但 package.json 声明了 "type": "module"，ESM 下 `require` 永远 undefined，
 * 所以 `pnpm embed` 从来没有真正执行过。改成独立入口文件。
 */
import 'dotenv/config';
import EmbeddingRetriever, { EmbeddingUnavailableError } from './EmbeddingRetriever.js';
import { EMBEDDING_MODEL } from './index.js';
import { refreshManifest } from './manifest.js';
import { loadKnowledge } from './loadKnowledge.js';

async function main() {
    const retriever = new EmbeddingRetriever(EMBEDDING_MODEL);
    // 索引脚本要的是"要么完整、要么明确失败"，不是"降级但跑完"
    retriever.setStrict(true);

    // 落盘是**调用方**的责任（EmbeddingRetriever 有意不注册任何信号处理器），
    // 所以这里补上：Ctrl-C 或 kill 时先把已嵌入的部分写盘，再退出。
    // 不加这段的话，中断时内存里最后不足一个批次的向量会直接丢掉——
    // 而下面那句"已成功嵌入的部分不会浪费"就是空头支票。
    let interrupted = false;
    for (const sig of ['SIGINT', 'SIGTERM'] as const) {
        process.on(sig, () => {
            if (interrupted) process.exit(1);   // 第二次按下就硬退，别把用户困住
            interrupted = true;
            console.log(`\n  收到 ${sig}，正在保存已嵌入的部分…`);
            retriever.flushCache();
            console.log('  已保存。重跑 pnpm embed 会从这里续上。');
            process.exit(130);
        });
    }
    const result = await loadKnowledge(retriever, {
        onProgress: (file, done, all) => {
            process.stdout.write(`\r  ${file} … ${Math.floor((done / all) * 100)}%   `);
        },
        onFileDone: (file, n) => {
            console.log(`\r  ${file} → ${n} chunk${' '.repeat(28)}`);
        },
    });
    const total = result.chunks;

    const pruned = retriever.pruneCache();
    if (pruned > 0) console.log(`  清理语料变更遗留的陈旧向量 ${pruned} 条`);
    retriever.flushCache();

    const { manifest, prev, changed } = refreshManifest();

    const stats = retriever.getStats();
    console.log(`\n索引完成：${total} chunk，向量维度 ${retriever.getDimension()}`);
    if (stats.embedRequests > 0) {
        const avg = (stats.embedCalls / stats.embedRequests).toFixed(1);
        console.log(`请求数 ${stats.embedRequests} 次（平均每批 ${avg} 条，EMBED_BATCH_SIZE=${process.env.EMBED_BATCH_SIZE ?? 16}）`);
    }
    console.log(`语料版本 v${manifest.version}，${manifest.totalArticles} 条，指纹 ${manifest.corpusHash}`);
    if (changed) {
        console.warn(
            `\n⚠️  语料内容已变化（旧指纹 ${prev!.corpusHash} → 新 ${manifest.corpusHash}），` +
            `版本号 v${prev!.version} → v${manifest.version}。\n` +
            `   recall@k / MRR 在不同语料版本之间**不可比**：候选池变了，分母也就变了。\n` +
            `   已有 eval 数据集的 ground truth 条号必须重新校对后才能与旧分数对比。`
        );
    }
    if (stats.fallbackHits > 0) {
        console.error(
            `\n❌ ${stats.fallbackHits}/${stats.embedCalls} 次走了兜底向量，本次索引不可信。\n` +
            `   兜底向量不落盘，所以磁盘缓存是干净的；缺失的条目会在下次 pnpm embed 时重试。\n` +
            `   但这一次的索引是残缺的——用它跑出来的 eval 数字没有意义。`
        );
        // 用 exitCode 而不是 exit()：让 stdout/stderr 先 flush 完，否则上面的告警可能被吞掉
        process.exitCode = 1;
    }
}

main().catch(err => {
    if (err instanceof EmbeddingUnavailableError) {
        console.error(
            `\n❌ ${err.message}\n` +
            `   索引未写入：磁盘缓存里不含任何兜底向量，是干净的。\n` +
            `   已成功嵌入的部分不会浪费——重跑 pnpm embed 会命中缓存，只补没做成的那些。\n` +
            `   先确认网络：curl 一下 EMBEDDING_BASE_URL，再重跑。`
        );
    } else {
        console.error(err);
    }
    process.exit(1);
});
