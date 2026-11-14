# USITC の HTS の JSON（exportList）から木を作る（設計 2.1・3.1）。
# 行の親は「それより前の行のうち、字下げが自分より浅い一番近い行」。字下げは飛ぶことがある（2→4 など）ので、生の字下げを積んで読む。
# 葉は「下にコードのある行が無い、8 桁か 10 桁の行」。コードの無い題の行（I. CHEMICAL ELEMENTS など）が下に付いても葉のまま。
# 当てる対象（商品の葉）は第 1〜97 類の葉。8 桁の葉は統計番号 00 を足して 10 桁にする。
import json, re, time, urllib.request
from dataclasses import dataclass, field
from pathlib import Path

NON_PRODUCT = ('98', '99')
RELEASES_URL = 'https://hts.usitc.gov/reststop/releaseList'
EXPORT_URL = 'https://hts.usitc.gov/reststop/exportList?from=0101&to=9999&format=JSON&styles=false'
UA = {'User-Agent': 'hts-tiny-research/0.1'}


def digits(s):
    return re.sub(r'\D', '', s or '')


@dataclass
class Node:
    idx: int
    code: str            # 数字だけ。類は 2 桁、コードの無いまとめの行は ''
    depth: int           # 類は -1、項は 0、その下は字下げの深さ
    text: str
    parent: int | None
    children: list = field(default_factory=list)
    coded_desc: bool = False     # 下にコードのある行があるか


def class_code(n):
    """葉なら 10 桁のクラスのコード（8 桁の葉は 00 を足す）。葉でなければ None。"""
    if n.coded_desc:
        return None
    if len(n.code) == 10:
        return n.code
    if len(n.code) == 8:
        return n.code + '00'
    return None


class Tree:
    def __init__(self, release, title=''):
        self.release, self.title = release, title
        self.nodes = []
        self.by_code = {}      # コード（2・4・6・8・10 桁）→ 節の番号（最初に出た行）
        self.leaf_of = {}      # 商品の葉の 10 桁 → 節の番号

    def add(self, code, depth, text, parent):
        n = Node(len(self.nodes), code, depth, text, parent)
        self.nodes.append(n)
        if parent is not None:
            self.nodes[parent].children.append(n.idx)
        if code:
            self.by_code.setdefault(code, n.idx)
            p = parent
            while p is not None and not self.nodes[p].coded_desc:
                self.nodes[p].coded_desc = True
                p = self.nodes[p].parent
        return n.idx

    def ancestors(self, i):
        out, p = [], self.nodes[i].parent
        while p is not None:
            out.append(p)
            p = self.nodes[p].parent
        return out[::-1]

    def path_text(self, i, sep=' > '):
        return sep.join(t for t in [self.nodes[a].text for a in self.ancestors(i)] + [self.nodes[i].text] if t)

    def resolve(self, code):
        """コードの数字から、いまの木にある一番深い桁の (コード, 桁数) を返す。無ければ None（設計 3.1 の部分的な正解）。"""
        c = digits(code)
        if len(c) >= 10 and c[:10] in self.leaf_of:
            return c[:10], 10
        leaf8 = self.leaf_of.get(c[:8] + '00') if len(c) >= 8 else None
        if leaf8 is not None and len(self.nodes[leaf8].code) == 8:     # 8 桁の行そのものが葉（下に統計番号の行が無い）
            return c[:8] + '00', 10
        for n in (8, 6, 4, 2):
            if len(c) >= n and c[:n] in self.by_code:
                return c[:n], n
        return None

    def leaves_under(self, i):
        out, stack = [], [i]
        while stack:
            n = self.nodes[stack.pop()]
            cls = class_code(n)
            if cls in self.leaf_of:
                out.append(cls)
            stack.extend(reversed(n.children))
        return out

    def contrast_leaves(self, cls, k=10):
        """葉 cls と見分けるべき近い葉。兄弟から順に、足りなければ上の段へ。同じ段の中は木の順で、最大 k 個。"""
        out = []
        for a in reversed(self.ancestors(self.leaf_of[cls])):
            out += [c for c in self.leaves_under(a) if c != cls and c not in out]
            if len(out) >= k:
                break
        return out[:k]


def parse_export(records, release, title=''):
    t = Tree(release, title)
    stack, chapters, cur_ch = [], {}, ''          # stack: 祖先の (生の字下げ, 節の番号)
    for r in records:
        code = digits(r.get('htsno'))
        text = ' '.join((r.get('description') or '').split())
        raw = max(0, int(r.get('indent') or 0))
        while stack and stack[-1][0] >= raw:
            stack.pop()
        if stack:
            parent = stack[-1][1]
        else:                                     # 項（字下げ 0）。類の節の下に付ける
            cur_ch = code[:2] if len(code) >= 4 else cur_ch
            if cur_ch not in chapters:
                chapters[cur_ch] = t.add(cur_ch, -1, '', None)
            parent = chapters[cur_ch]
        i = t.add(code, len(stack), text, parent)
        stack.append((raw, i))
    for n in t.nodes:
        cls = class_code(n)
        if cls and n.depth >= 0 and cls[:2] not in NON_PRODUCT:
            t.leaf_of.setdefault(cls, n.idx)
    return t


def _get(url, timeout=120):
    req = urllib.request.Request(url, headers={**UA, 'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return res.read().decode('utf-8')


def fetch_current(out_dir):
    """現行の版の JSON を out_dir/<版>.json に取る（すでにあれば取らない）。過去の版の JSON は API で取れない（設計 2.1）。"""
    cur = next(r for r in json.loads(_get(RELEASES_URL)) if r.get('status') == 'current')
    p = Path(out_dir) / f'{cur["name"]}.json'
    if not p.exists():
        recs = json.loads(_get(EXPORT_URL, timeout=600))
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix('.part')
        tmp.write_text(json.dumps({'release': cur['name'], 'title': cur.get('title', ''),
                                   'fetchedAt': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'records': recs},
                                  ensure_ascii=False), encoding='utf-8')
        tmp.rename(p)
    return p


def load_tree(path):
    j = json.loads(Path(path).read_text(encoding='utf-8'))
    return parse_export(j['records'], j['release'], j.get('title', ''))
