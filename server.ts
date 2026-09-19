import { config } from 'dotenv';
config({ override: true });

import express, { Request, Response } from 'express';
import cors from 'cors';
import path from 'path';
import fs from 'fs';
import crypto from 'crypto';
import multer from 'multer';
import { runQuery, shutdownMcpClients, createRetriever } from './src/index.js';
import { getJudgeStats, ANSWERABILITY_JUDGE_ENABLED, ANSWERABILITY_JUDGE_MODEL } from './src/AnswerabilityJudge.js';
import { createSessionStore, SessionStore } from './src/SessionStore.js';
import { KNOWLEDGE_DIR, FRONTEND_DIR } from './src/paths.js';
import { loadKnowledge } from './src/loadKnowledge.js';

const app = express();
const PORT = Number(process.env.PORT ?? 3001);

// 走 createRetriever()：精排开关只有一处实现，server 与 eval 不会各开各的
const sharedRetriever = createRetriever();
let sessionStore: SessionStore;

async function initializeKnowledgeBase() {
  // 语料为空不是致命错误：服务照常起来，只是检索不到东西。
  // 但必须**大声**说出来——静默地零召回会被误当成"模型不会答"。
  let result;
  try {
    result = await loadKnowledge(sharedRetriever, {
      onStart: n => console.log(`[kb] 加载知识库：${n} 个文件`),
      onFileDone: (file, n) => console.log(`[kb]   ${file} → ${n} chunk`),
    });
  } catch (error) {
    console.warn(`[kb] ${(error as Error).message}——检索将没有任何结果`);
    return;
  }

  sharedRetriever.flushCache();
  console.log(`[kb] 就绪：${result.chunks} chunk，维度 ${sharedRetriever.getDimension()}`);
  if (result.fallbackHits > 0) {
    console.warn(
      `[kb] 警告：${result.fallbackHits}/${result.embedCalls} 次 embedding 走了确定性兜底向量——` +
      `检索结果不可信。请检查 EMBEDDING_API_KEY / EMBEDDING_BASE_URL。`
    );
  }
}

// 上传文件名由服务端生成，不使用 originalname（可含 ../ 或覆盖同名文件）
const storage = multer.diskStorage({
  destination(_req, _file, cb) {
    if (!fs.existsSync(KNOWLEDGE_DIR)) fs.mkdirSync(KNOWLEDGE_DIR, { recursive: true });
    cb(null, KNOWLEDGE_DIR);
  },
  filename(_req, file, cb) {
    const safeBase = path.basename(file.originalname, '.md')
      .replace(/[^\w一-龥-]/g, '_')
      .slice(0, 60) || 'upload';
    cb(null, `${safeBase}-${crypto.randomBytes(4).toString('hex')}.md`);
  },
});

const upload = multer({
  storage,
  fileFilter(_req, file, cb) {
    if (file.mimetype === 'text/markdown' || file.originalname.endsWith('.md')) cb(null, true);
    else cb(new Error('只接受 markdown (.md) 文件'));
  },
  limits: { fileSize: 5 * 1024 * 1024 },
});

app.use(cors());
app.use(express.json({ limit: '1mb' }));
app.use(express.static(FRONTEND_DIR));

function generateSessionId(): string {
  return crypto.randomBytes(9).toString('base64url');
}

/**
 * 流式对话（SSE）。
 *
 * 原实现是 res.json(...)：模型 token 只 process.stdout.write 到服务器终端，
 * 浏览器要等整个回答生成完才看到内容——"流式输出"对用户完全不可见。
 * 这里把 token 和工具事件都推给前端。
 */
app.post('/api/chat', async (req: Request, res: Response) => {
  const { message, sessionId = generateSessionId() } = req.body ?? {};
  if (typeof message !== 'string' || message.trim().length === 0) {
    return res.status(400).json({ error: 'message 不能为空' });
  }

  res.writeHead(200, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache, no-transform',
    'Connection': 'keep-alive',
    'X-Accel-Buffering': 'no',
  });

  const send = (event: string, data: unknown) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  // 客户端断连 → 取消下游 LLM 请求，别继续烧 token。
  //
  // 注意必须监听 res 而不是 req：express.json() 把请求体读完之后，
  // req 的 'close' 会**立即**触发（实测 0ms），用它做取消信号会在 LLM 请求发出前就中止。
  // res 的 'close' 才对应连接关闭；再用 writableEnded 排除"正常回完"的情况。
  const controller = new AbortController();
  res.on('close', () => {
    if (!res.writableEnded) {
      console.log('[chat] 客户端断连，中止本次生成');
      controller.abort();
    }
  });

  send('meta', { sessionId });

  try {
    const history = await sessionStore.getHistory(sessionId);
    const result = await runQuery({
      query: message,
      retriever: sharedRetriever,
      history,
      signal: controller.signal,
      onToken: (token) => send('token', { token }),
      onEvent: (event) => send('status', event),
    });

    await sessionStore.append(sessionId, [
      { role: 'user', content: message },
      { role: 'assistant', content: result.answer },
    ]);

    send('done', {
      sessionId,
      abstained: result.abstained,
      usage: result.usage,
      promptVersion: result.promptVersion,
      // 判官的判定与开销。**默认不发 reason**：把未经审阅的模型理由放进界面是负债，
      // 答案正文已经用自然语言说清了拒答原因，徽章只需要知道是"哪一类"拒答。
      // usage 仍只含生成模型，总成本 = usage + judge.usage（前端若要展示成本，两个都要加）。
      judge: {
        outcome: result.judge.outcome,
        skipReason: result.judge.skipReason,
        answerable: result.judge.answerable,
        cached: result.judge.cached,
        latencyMs: result.judge.latencyMs,
        usage: result.judge.usage,
        model: result.judge.model,
        promptVersion: result.judge.promptVersion,
      },
      // citationCheck：对模型**实际引用**的条号逐条核对（幻觉检测）。
      // 与下面的 citations（=检索到的条文列表）不是一回事，别混。
      citationCheck: result.citations,
      citations: result.retrieved.map(r => ({
        source: r.source,
        articleNo: r.articleNo,
        chapter: r.chapter,
        score: Number(r.score.toFixed(4)),
      })),
    });
  } catch (error) {
    if (controller.signal.aborted) {
      return res.end();
    }
    console.error('[chat] 请求失败:', error);
    // 统一错误响应：不把内部 error.message 回传给客户端
    send('error', { error: '服务处理失败，请稍后重试' });
  } finally {
    res.end();
  }
});

app.get('/api/knowledge', (_req: Request, res: Response) => {
  try {
    const files = fs.readdirSync(KNOWLEDGE_DIR).filter(f => f.endsWith('.md'));
    res.json({
      files: files.map(file => {
        const stats = fs.statSync(path.join(KNOWLEDGE_DIR, file));
        return { name: file, size: stats.size, lastModified: stats.mtime };
      }),
    });
  } catch (error) {
    console.error('[kb] 列目录失败:', error);
    res.status(500).json({ error: '读取知识库列表失败' });
  }
});

app.get('/api/knowledge/:filename', (req: Request, res: Response) => {
  const filename = req.params.filename;
  // 路径穿越防护：只允许纯文件名 + .md
  if (filename.includes('..') || filename.includes('/') || filename.includes('\\') || !filename.endsWith('.md')) {
    return res.status(400).json({ error: '文件名不合法' });
  }
  const filePath = path.join(KNOWLEDGE_DIR, filename);
  if (!filePath.startsWith(KNOWLEDGE_DIR) || !fs.existsSync(filePath)) {
    return res.status(404).json({ error: '文件不存在' });
  }
  try {
    res.json({ filename, content: fs.readFileSync(filePath, 'utf8'), format: 'markdown' });
  } catch (error) {
    console.error('[kb] 读文件失败:', error);
    res.status(500).json({ error: '读取文件失败' });
  }
});

app.post('/api/knowledge/upload', upload.single('file'), async (req: Request, res: Response) => {
  if (!req.file) return res.status(400).json({ error: '未收到文件' });
  try {
    const fileName = req.file.filename;
    const content = fs.readFileSync(req.file.path, 'utf-8');
    const added = await sharedRetriever.embedDocument(content, fileName);
    sharedRetriever.flushCache();
    console.log(`[kb] 已索引上传文件 ${fileName} → ${added} chunk`);
    res.status(200).json({ message: '上传并索引成功', filename: fileName, chunks: added });
  } catch (error) {
    console.error('[kb] 上传处理失败:', error);
    res.status(500).json({ error: '上传处理失败' });
  }
});

/** 轻量可观测端点：检索/缓存/兜底命中情况 */
app.get('/api/metrics', (_req: Request, res: Response) => {
  const stats = sharedRetriever.getStats();
  res.json({
    chunks: sharedRetriever.size(),
    dimension: sharedRetriever.getDimension(),
    embedding: stats,
    fallbackRate: stats.embedCalls > 0 ? stats.fallbackHits / stats.embedCalls : 0,
    // 判官计数器。⚠️ 这是**进程生命周期**状态，重启即清零，所以它**不是评测的事实来源**
    // （`QueryResult.judge` 才是）。它在这里只为了能一眼看出"判官是不是在失败"——
    // 尤其是 errors / parseFailures 两个数一旦持续增长，说明该查 prompt 或上游了。
    judge: { ...getJudgeStats(), enabled: ANSWERABILITY_JUDGE_ENABLED, model: ANSWERABILITY_JUDGE_MODEL },
    sessionBackend: sessionStore?.backend ?? 'unknown',
  });
});

app.get('/', (_req: Request, res: Response) => {
  res.sendFile(path.join(FRONTEND_DIR, 'index.html'));
});

async function start() {
  sessionStore = await createSessionStore();
  await initializeKnowledgeBase();
  const server = app.listen(PORT, () => {
    console.log(`[server] http://localhost:${PORT}`);
  });

  const shutdown = async (signal: string) => {
    console.log(`\n[server] 收到 ${signal}，正在关闭…`);
    server.close();
    sharedRetriever.flushCache();
    await shutdownMcpClients();
    await sessionStore.close();
    process.exit(0);
  };
  // 进程信号由应用入口处理，而不是由库（EmbeddingRetriever）劫持
  process.on('SIGINT', () => void shutdown('SIGINT'));
  process.on('SIGTERM', () => void shutdown('SIGTERM'));
}

start().catch(err => {
  console.error('[server] 启动失败:', err);
  process.exit(1);
});
