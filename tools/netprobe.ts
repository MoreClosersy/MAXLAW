import 'dotenv/config';
const targets = [
  ['chat  api.openai.com',      (process.env.OPENAI_BASE_URL || '') + '/models'],
  ['embed router.huggingface',  (process.env.EMBEDDING_BASE_URL || '') + '/models'],
  ['embed api.siliconflow.cn',  'https://api.siliconflow.cn/v1/models'],
];
for (const [name, url] of targets) {
  const t0 = Date.now();
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 15_000);
    const r = await fetch(url, { signal: ctrl.signal });
    clearTimeout(timer);
    console.log(`  ${name.padEnd(28)} ${r.status}  ${Date.now()-t0}ms`);
  } catch (e: any) {
    console.log(`  ${name.padEnd(28)} FAIL ${Date.now()-t0}ms  ${e?.cause?.code ?? e?.name ?? e}`);
  }
}
