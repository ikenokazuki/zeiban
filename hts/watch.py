# 版の見張り: USITC の現行の版が releases/ にまだ無ければ、JSON を取って gzip で保存し、前の版との差分を diffs/ に書く。
# 使い方: python3 hts/watch.py   （GitHub Actions の中では、結果を $GITHUB_OUTPUT と summary.md にも書く）
import gzip, hashlib, json, os, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hts import rev_diff as D, tree as T  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REL, DIFFS = ROOT / 'releases', ROOT / 'diffs'
INDEX = REL / 'index.json'
NAME = re.compile(r'^[0-9]{4}HTS[A-Za-z0-9]{1,12}$')      # 版の名前は外から来る文字列。ファイル名とコミットの文に使うので、形を確かめる
SMALL = 40                                                 # 葉の増減がこれ未満なら「小さい直し」
HEAD_0101 = 'Live horses, asses, mules and hinnies:'        # 項 0101 の行の文（load の補いにだけ使う）


def load(p):
    """保存した版（USITC の exportList の包み、または配布ファイルそのままの並び）→ 木。"""
    j = json.loads(gzip.open(p, 'rt', encoding='utf-8').read())
    recs = j['records'] if isinstance(j, dict) else j
    if T.digits(recs[0].get('htsno')) != '0101':            # 昔の写しの一部は 0101.21.00 から取ってあり、先頭の項 0101 の行が無い。補わないと、その下の 7 葉の道筋が「変わった」に見える
        recs = [{'htsno': '0101', 'indent': '0', 'description': HEAD_0101}] + recs
    return T.parse_export(recs, p.name.split('.')[0])


def kind(d):
    """A = 商品の行の変化なし、B = 小さい直し、C = 組み替え（葉の増減が SMALL 以上）。"""
    if not (d['born'] or d['gone'] or d['text'] or d['path']):
        return 'A'
    return 'B' if len(d['born']) + len(d['gone']) < SMALL else 'C'


def record(d, k, missed):
    """diffs/ に書く形。"""
    return {**d, 'kind': k, 'missed': missed}


def summary(d, k, missed, old, new, top=30):
    """人が読む要約（issue の本文）。"""
    A, B = D.leaves(old), D.leaves(new)
    code = lambda c: f'{c[:4]}.{c[4:6]}.{c[6:8]}.{c[8:]}'
    out = [f"# {d['old']} → {d['new']}（型 {k}）", '',
           f"- 商品の葉: {d['n_leaves_old']} → {d['n_leaves_new']}",
           f"- 増えた {len(d['born'])}・消えた {len(d['gone'])}・自分の行の文が変わった {len(d['text'])}・上の行の文だけ変わった {len(d['path'])}",
           f"- 型ごとの組の数: {d['kinds'] or 'なし'}",
           f"- クラスが変わった葉の割合: {d['share_class_changed']:.2%}"]
    if missed:
        out.append(f"- **取り逃がした版**（この差分に含まれる）: {', '.join(missed)}")
    for title, rows in (('増えた行', [(c, B[c][0]) for c in d['born']]), ('消えた行', [(c, f"{A[c][0]} → 行き先 {d['remap'][c][0] or 'なし'}（{d['remap'][c][2]}）") for c in d['gone']]),
                        ('文が変わった行', [(c, f'{A[c][0]} → {B[c][0]}') for c in d['text']])):
        if rows:
            out += ['', f'## {title}（{len(rows)}）', ''] + [f'- `{code(c)}` {t[:200]}' for c, t in rows[:top]] + ([f'- …ほか {len(rows) - top} 行（diffs/ の JSON に全部ある）'] if len(rows) > top else [])
    return '\n'.join(out) + '\n'


def main():
    index = json.loads(INDEX.read_text(encoding='utf-8'))
    rel = json.loads(T._get(T.RELEASES_URL))
    cur = next(r for r in rel if r.get('status') == 'current')
    name = cur['name']
    assert NAME.match(name), f'版の名前の形が想定と違う: {name!r}'
    if any(e['name'] == name for e in index):
        print(f'新しい版は無い（現行 {name}）')
        return
    recs = json.loads(T._get(T.EXPORT_URL, timeout=600))
    again = next(r for r in json.loads(T._get(T.RELEASES_URL)) if r.get('status') == 'current')
    assert again['name'] == name, '取っている間に版が替わった。次の回でやり直す'
    assert isinstance(recs, list) and len(recs) > 20000, f'行の数が少なすぎる: {len(recs) if isinstance(recs, list) else type(recs)}'
    raw = json.dumps({'release': name, 'title': cur.get('title', ''), 'fetchedAt': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'records': recs}, ensure_ascii=False)
    p = REL / f'{name}.json.gz'
    p.write_bytes(gzip.compress(raw.encode('utf-8'), mtime=0))
    prev = index[-1]
    old, new = load(REL / prev['file']), load(p)
    assert len(new.leaf_of) > 15000, f'商品の葉が少なすぎる: {len(new.leaf_of)}'
    d = D.diff(old, new)
    d['old'], d['new'] = prev['name'], name
    names = [r['name'] for r in rel]                       # 新しい順
    missed = [n for n in names[names.index(name) + 1:names.index(prev['name'])] if NAME.match(n)] if prev['name'] in names else []
    k = kind(d)
    (DIFFS / f"{prev['name']}__{name}.json").write_text(json.dumps(record(d, k, missed), ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    m = re.match(r'(\d\d)/(\d\d)/(\d{4})$', cur.get('releaseStartDate') or '')
    index.append({'name': name, 'date': f'{m.group(3)}-{m.group(1)}-{m.group(2)}' if m else '', 'file': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                  'source': 'USITC reststop/exportList', 'leaves': len(new.leaf_of), 'kind': k})
    INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    text = summary(d, k, missed, old, new)
    (ROOT / 'summary.md').write_text(text, encoding='utf-8')
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as f:
            f.write(f'new={name}\nkind={k}\n')
    print(text)


if __name__ == '__main__':
    main()
