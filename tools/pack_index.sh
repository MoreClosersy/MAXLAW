#!/usr/bin/env bash
#
# 打包预构建索引，供 CI 在冷缓存时下载（GitHub release `index-cache`）。
#
# 为什么需要它：`actions/cache` 在第一次运行、以及每次语料/INDEX_VERSION 变更后
# 都是冷的。冷缓存时让 CI 现场调 1681 次 embedding 接口需要 EMBEDDING_API_KEY，
# 而 fork PR 拿不到 secret——CI 就会在"没索引"上失败。把索引作为 release 资产
# 发一次，冷启动就不再依赖任何密钥。
#
# 用法：
#   pnpm embed            # 语料或 INDEX_VERSION 变更后重建（需要 EMBEDDING_API_KEY）
#   pnpm index:pack       # 生成 index-v<INDEX_VERSION>-<model>-<corpusHash>.json.gz
#
# 然后把产物上传到本仓库的 release，tag 固定为 `index-cache`（覆盖同名资产即可）。
# 注意：eval 会把 69 条查询向量也写进 cache，**打包前跑一次 `pnpm eval`** 能让
# 冷启动的第一轮评测也完全离线。
set -euo pipefail
cd "$(dirname "$0")/.."

INDEX_VERSION="${INDEX_VERSION:-1}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-base-zh-v1.5}"
MODEL_SLUG="${EMBEDDING_MODEL//\//_}"
SRC="cache/${MODEL_SLUG}_cache.json"
HASH=$(node -e 'console.log(require("./knowledge/CORPUS.json").corpusHash)')
OUT="index-v${INDEX_VERSION}-${MODEL_SLUG}-${HASH}.json.gz"

if [ ! -s "$SRC" ]; then
    echo "缺少 $SRC。先跑 pnpm embed（需要 EMBEDDING_API_KEY），再打包。" >&2
    exit 1
fi

COUNT=$(node -e 'const d=JSON.parse(require("fs").readFileSync(process.argv[1],"utf8")); console.log(Object.keys(d).length)' "$SRC")
gzip -c "$SRC" > "$OUT"

echo "语料指纹 ${HASH} · 索引条目 ${COUNT}"
echo "已生成 ${OUT}（$(du -h "$OUT" | cut -f1)）"
echo
echo "上传到：https://github.com/<owner>/<repo>/releases/tag/index-cache"
echo "CI 会从 .../releases/download/index-cache/${OUT} 下载；文件名必须完全一致。"
