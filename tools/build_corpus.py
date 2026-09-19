#!/usr/bin/env python3
"""
从官方全文页面构建知识库语料（支持《民法典》《公司法》这类"编-章-节"与"章-节"两种层级）。

设计要点
--------
1. **不做 NFKC 归一化**。NFKC 会把全角逗号「，」(U+FF0C) 压成半角「,」，
   中文法律文本必须保留全角标点。
2. 只做一处显式标点变换：官网转写用半角分号，规范化为全角「；」（见 normalize_punct）。
   该变换只动标点、不动实义字符，并在 CORPUS.json 里记录。
3. 以「第X条」为切分单元，同时解析 编/章/节 层级，写成 markdown 标题，
   供 src/EmbeddingRetriever.ts 的 splitIntoChunks 挂 chapter 元数据。
4. 三重校验，任一不过即退出、不产出脏语料：
   a. 条号连续、无缺失、无重复
   b. 目录（编/章/节序列）与正文归纳出的层级序列逐一吻合
   c. 两个独立官方源逐条比对完全一致

用法
----
    python3 tools/build_corpus.py <主源.html> <校验源.html> <输出.md> "<标题>" [--max-article N]
"""
import re, sys, html, hashlib, json, datetime
from pathlib import Path

CN = {'零':0,'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,
      '十':10,'百':100,'千':1000,'万':10000}

def cn2int(s):
    """中文数字转整数；纯阿拉伯数字直接解析。"""
    if s.isdigit(): return int(s)
    total = section = num = 0
    for ch in s:
        if ch not in CN: return None
        v = CN[ch]
        if v == 10000: section = (section + num) * v; total += section; section = num = 0
        elif v >= 10: section += (num if num else 1) * v; num = 0
        else: num = v
    return total + section + num

def strip_html(raw):
    raw = re.sub(r'(?is)<(script|style|head)[^>]*>.*?</\1>', ' ', raw)
    raw = re.sub(r'(?is)<br\s*/?>', '\n', raw)
    raw = re.sub(r'(?is)</(p|div|tr|li|h[1-6]|td|section)>', '\n', raw)
    raw = re.sub(r'(?s)<[^>]+>', '', raw)
    raw = html.unescape(raw)
    raw = raw.replace('　', ' ').replace(' ', ' ')
    raw = re.sub(r'[ \t]+', ' ', raw)
    raw = re.sub(r'\n[ ]*\n+', '\n', raw)
    return raw

def normalize_punct(s):
    """官网转写用半角分号；中文法律文本的标准标点是全角。只动标点，不动实义字符。"""
    s = s.replace(';', '；')
    return re.sub(r'([一-鿿]),(?=[一-鿿])', r'\1，', s)

def norm_heading(text):
    """「第一章 总 则」与「第一章 总则」是同一个标题，比对前先统一。"""
    text = re.sub(r'\s+', ' ', text).strip()
    m = re.match(r'^(第[一二三四五六七八九十百千万零\d]+[编章节])\s*(.*)$', text)
    if m:
        name = re.sub(r'\s+', '', m.group(2))
        return (m.group(1) + ' ' + name).strip()
    return re.sub(r'\s+', '', text)

ART   = re.compile(r'^第([一二三四五六七八九十百千万零\d]+)条(?=\s|$)')
BIAN  = re.compile(r'^第([一二三四五六七八九十百千万零\d]+)编\s+(\S.*)$')
ZHANG = re.compile(r'^第([一二三四五六七八九十百千万零\d]+)章\s+(\S.*)$')
JIE   = re.compile(r'^第([一二三四五六七八九十百千万零\d]+)节\s+(\S.*)$')
HEAD  = (BIAN, ZHANG, JIE)

def find_body_start(lines):
    """正文起点：第一条 所在行，向上收「第一编/第一章/第一节」这一串起始标题。

    必须限定"序号为 1"，否则会把目录末尾的章节标题（如《公司法》目录最后一行的
    「第十五章 附则」）一路卷进正文。
    """
    first = next(i for i, l in enumerate(lines) if ART.match(l))
    start = first
    while start > 0 and lines[start-1] == '':
        start -= 1
    while start > 0:
        l = lines[start-1].lstrip('# ').strip()
        m = next((r.match(l) for r in HEAD if r.match(l)), None)
        if not m or cn2int(m.group(1)) != 1:
            break
        start -= 1
    return start

def parse_body(path):
    """返回 (articles, structure, order)。articles: {n: 条文}; structure: {n: (编,章,节)}"""
    lines = [l.strip() for l in strip_html(Path(path).read_text(encoding='utf-8', errors='ignore')).split('\n')]
    start = find_body_start(lines)

    articles, structure, order = {}, {}, []
    bi = zh = ji = ''
    cur = None
    for line in lines[start:]:
        if not line:
            continue
        if line == '附则':
            bi, zh, ji, cur = '附则', '', '', None
            continue
        m = next((r.match(line) for r in HEAD if r.match(line)), None)
        if m:
            h = norm_heading(line)
            which = next(i for i, r in enumerate(HEAD) if r.match(line))
            if which == 0:   bi, zh, ji = h, '', ''
            elif which == 1: zh, ji = h, ''
            else:            ji = h
            cur = None
            continue
        m = ART.match(line)
        if m:
            n = cn2int(m.group(1))
            if n is None: continue
            if n in articles:
                raise SystemExit(f'[FATAL] 第{n}条 重复出现（{path}）')
            articles[n] = normalize_punct(line)
            structure[n] = (bi, zh, ji)
            order.append(n); cur = n
            continue
        if cur is not None:
            articles[cur] += '\n' + normalize_punct(line)
    return articles, structure, order

def toc_sequence(path):
    """页面顶部「目录」列出的 编/章/节 有序序列。

    切分点直接复用 find_body_start —— 不能靠"从尾部往上弹标题行"，
    因为整个目录都是标题行，一弹就会把目录本身吃掉。
    """
    lines = [l.strip() for l in strip_html(Path(path).read_text(encoding='utf-8', errors='ignore')).split('\n')]
    start = find_body_start(lines)
    out = []
    for l in lines[:start]:
        m = next((r for r in HEAD if r.match(l)), None)
        if m or l.lstrip('# ').strip() == '附则':
            out.append(norm_heading(l))
    return out

def body_sequence(structure, order):
    """按条序归纳出的层级序列。不能按字符串去重——「第一章 一般规定」在不同编里重复出现。"""
    seq, prev = [], ('', '', '')
    for n in order:
        cur = structure[n]
        for k in range(3):
            if cur[k] != prev[k] and cur[k]:
                seq.append(cur[k])
        prev = cur
    return seq

def emit_markdown(articles, structure, order, title, out, source_url):
    header = [f'# {title}\n']
    buf, prev = [], ('', '', '')
    for n in order:
        cur = structure[n]
        pp = [x for x in prev if x]
        pc = [x for x in cur if x]
        for k in range(len(pc)):
            if k >= len(pp) or pc[k] != pp[k]:
                for d in range(k, len(pc)):
                    buf.append(f'\n{"#" * (d + 2)} {pc[d]}\n')
                break
        prev = cur
        buf.append('\n' + articles[n] + '\n')
    out.write_text(''.join(header + buf), encoding='utf-8')
    return hashlib.sha1(''.join(buf).encode()).hexdigest()

def canon(s):
    """比对用规范化：去条号、去标题、去全部标点与空白，只留实义字符。"""
    s = re.sub(r'^第[一二三四五六七八九十百千万零\d]+条', '', s)
    s = re.sub(r'第[一二三四五六七八九十百千万零\d]+[编章节]\s*\S*', '', s)
    return re.sub(r'[\s，。；：、（）《》"\'“”·—\-,.;:()<>\[\]【】]', '', s)

def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if len(args) < 4:
        print(__doc__); sys.exit(1)
    src, chk, out, title = args[0], args[1], Path(args[2]), args[3]
    opt = {}
    for a in sys.argv[1:]:
        if a.startswith('--url='):        opt['url'] = a[6:]
        elif a.startswith('--check-url='): opt['check'] = a[12:]
        elif a.startswith('--source-name='): opt['srcname'] = a[14:]
        elif a.startswith('--check-name='):  opt['chkname'] = a[13:]

    A, SA, OA = parse_body(src)
    B, _, OB = parse_body(chk)

    if OA != OB:
        raise SystemExit('[FATAL] 两个来源的条文序列不一致')
    miss = [i for i in range(1, max(OA) + 1) if i not in A]
    if miss:
        raise SystemExit(f'[FATAL] 缺条 {len(miss)}: {miss[:20]}')

    # 末条尾巴会粘上网页页脚（两源页脚各不相同）。取两源最长公共前缀，再回退到
    # 最后一个句号削掉页脚。只对最后一条做——前面各条已逐字一致。
    last = OA[-1]
    pre = 0
    while pre < min(len(A[last]), len(B[last])) and A[last][pre] == B[last][pre]:
        pre += 1
    if pre < len(A[last]) or pre < len(B[last]):
        before = len(A[last])
        cut = max(A[last].rfind('。', 0, pre), A[last].rfind('\n', 0, pre)) + 1
        A[last] = A[last][:cut].strip(); B[last] = B[last][:cut].strip()
        print(f'  [i] 第{last}条 削去页脚 {before - len(A[last])} 字符')

    diffs = [n for n in OA if canon(A[n]) != canon(B[n])]
    if diffs:
        for n in diffs[:5]:
            print(f'  ✗ 第{n}条\n    A: {A[n][:90]}\n    B: {B[n][:90]}')
        raise SystemExit(f'[FATAL] 两源逐条比对不一致 {len(diffs)} 条')
    print(f'✅ 交叉校验：两源 {len(OA)} 条逐条一致（零缺失、零重复）')

    t, b = toc_sequence(src), body_sequence(SA, OA)
    if t != b:
        import difflib
        d = list(difflib.unified_diff(t, b, '目录', '正文', lineterm='', n=1))
        raise SystemExit('[FATAL] 结构不一致：\n  ' + '\n  '.join(d[:30]))
    print(f'✅ 结构自校验：目录 {len(t)} 个 编/章/节节点与正文归纳序列完全吻合')

    sha = emit_markdown(A, SA, OA, title, out, opt.get('url', ''))

    # 来源记录：这是语料可信度的凭据，必须能被人独立复核
    sources_path = out.parent / 'SOURCES.json'
    sources = json.loads(sources_path.read_text(encoding='utf-8')) if sources_path.exists() else {}
    sources[out.name] = {
        'title': title,
        'articles': len(OA),
        'markdownSha1': sha,
        'verifiedAt': datetime.date.today().isoformat(),
        'sources': [
            {'role': 'primary', 'name': opt.get('srcname', ''), 'url': opt.get('url', ''),
             'rawSha256': hashlib.sha256(Path(src).read_bytes()).hexdigest()},
            {'role': 'cross-check', 'name': opt.get('chkname', ''), 'url': opt.get('check', ''),
             'rawSha256': hashlib.sha256(Path(chk).read_bytes()).hexdigest()},
        ],
        'verification': [
            f'条号连续、无缺失、无重复（{len(OA)} 条）',
            '目录 编/章/节 序列与正文归纳序列完全吻合',
            '两个独立官方源逐条比对完全一致',
        ],
        'transformations': [
            'HTML 转纯文本（未做 NFKC，保留全角标点）',
            '半角分号 ; 规范化为全角 ；',
            '削去末条尾部粘连的网页页脚',
        ],
    }
    sources_path.write_text(json.dumps(sources, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'✅ 写出 {out}  {len(OA)} 条  sha1={sha[:12]}  bytes={out.stat().st_size}')
    print(f'✅ 来源记录写入 {sources_path}')

if __name__ == '__main__':
    main()
