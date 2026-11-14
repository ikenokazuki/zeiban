# 2 つの版の木の差分（設計 6 節の「差分」と「正解の付け替え」の材料）。第 98・99 類は見ない（商品の葉だけを比べる）。
# 出すもの: 増えた葉・消えた葉・自分の行の文が変わった葉・上の行の文だけ変わった葉、消えた葉の行き先（付け替え）、型ごとの数、変わった葉の割合。
# 使い方: 研究フォルダで  $PY -I hts/rev_diff.py <前の版.json> <新しい版.json> [書き出す.json]   ／   $PY -I hts/rev_diff.py demo
import collections, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hts import tree as T  # noqa: E402


def clean(s):
    """書式の印（<i>・<il>・<u>・<sup> など。exportList の styles=true や昔の配布ファイルに付く）を落として、空白をそろえる。印だけの違いは変化に数えない。"""
    return ' '.join(re.sub(r'</?[a-zA-Z]+[^>]*>', '', s).split())


def leaves(t):
    """商品の葉の 10 桁 → (自分の行の文, 親までの道筋の文)。"""
    return {c: (clean(t.nodes[i].text), clean(t.path_text(t.nodes[i].parent))) for c, i in t.leaf_of.items()}


def diff(old, new):
    A, B = leaves(old), leaves(new)
    gone, born = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    both = set(A) & set(B)
    text = sorted(c for c in both if A[c][0] != B[c][0])
    path = sorted(c for c in both if A[c][0] == B[c][0] and A[c][1] != B[c][1])
    # ponytail: 消えた葉と増えた葉は 8 桁の頭でしかまとめない。項をまたぐ付け替え（HS の大きな改訂）は「消えた」＋「増えた」に見える。
    #           HS 2028 の前に、USITC・WCO の対応表を読む形に上げる。
    g = collections.defaultdict(lambda: ([], []))
    for c in gone:
        g[c[:8]][0].append(c)
    for c in born:
        g[c[:8]][1].append(c)
    kinds, remap = collections.Counter(), {}
    for a, b in g.values():
        if a and b:
            if len(a) == 1 and len(b) == 1:
                kind = '番号だけ替わった' if A[a[0]] == B[b[0]] else '置き換わった'
            else:
                kind = '分かれた' if len(a) == 1 else 'まとめられた' if len(b) == 1 else '組み替えられた'
        else:
            kind = '消えた' if a else '増えた'
        kinds[kind] += 1
        for c in a:                                   # 消えた葉の行き先: 番号だけ替わったなら新しい 10 桁、ほかは残っている一番深い桁まで
            remap[c] = [b[0], 10, kind] if kind == '番号だけ替わった' else [*(new.resolve(c) or ('', 0)), kind]
    nodes = collections.Counter(len(c) for c, i in old.by_code.items()
                                if c[:2] not in T.NON_PRODUCT and c in new.by_code and clean(old.nodes[i].text) != clean(new.nodes[new.by_code[c]].text))
    n = len(A)
    return {'old': old.release, 'new': new.release, 'n_leaves_old': n, 'n_leaves_new': len(B),
            'born': born, 'gone': gone, 'text': text, 'path': path, 'kinds': dict(kinds), 'remap': remap,
            'node_text_changed_by_digits': dict(sorted(nodes.items())),
            'share_class_changed': (len(gone) + len(born)) / n, 'share_any_changed': (len(gone) + len(born) + len(text) + len(path)) / n}


def demo():
    R = lambda code, text, indent: {'htsno': code, 'description': text, 'indent': indent}
    old = [R('0101', 'Live horses', 0), R('0101.21.00', 'Purebred', 1), R('0101.29.00', 'Other', 1),
           R('0101.29.00.10', 'Imported for slaughter', 2), R('0101.29.00.90', 'Other', 2),
           R('0102', 'Live bovine animals', 0), R('0102.21.00', 'Purebred breeding animals', 1), R('0102.21.00.10', 'Dairy', 2), R('0102.21.00.20', 'Other', 2),
           R('9903.01.01', 'Temporary', 0)]
    new = [R('0101', 'Live horses', 0), R('0101.21.00', 'Purebred breeding animals', 1), R('0101.29.00', 'Other', 1),
           R('0101.29.00.15', 'Imported for slaughter', 2), R('0101.29.00.90', 'Other', 2),
           R('0102', 'Live bovine animals', 0), R('0102.21.00', 'Purebred breeding animals', 1), R('0102.21.00.10', 'Dairy', 2),
           R('0102.21.00.30', 'Male', 2), R('0102.21.00.40', 'Female', 2),
           R('0103', 'Live swine', 0), R('0103.10.00', 'Purebred breeding animals', 1),
           R('9903.01.01', 'Temporary, changed', 0), R('9903.01.02', 'New temporary', 0)]
    d = diff(T.parse_export(old, 'a'), T.parse_export(new, 'b'))
    assert d['text'] == ['0101210000'] and d['path'] == [], d
    assert clean('Koi (<i>Cyprinus  carpio</i>) m<sup>2</sup>') == 'Koi (Cyprinus carpio) m2'
    assert d['gone'] == ['0101290010', '0102210020'] and d['born'] == ['0101290015', '0102210030', '0102210040', '0103100000'], d
    assert d['kinds'] == {'番号だけ替わった': 1, '分かれた': 1, '増えた': 1}, d['kinds']
    assert d['remap'] == {'0101290010': ['0101290015', 10, '番号だけ替わった'], '0102210020': ['01022100', 8, '分かれた']}, d['remap']
    assert d['n_leaves_old'] == 5 and d['n_leaves_new'] == 7 and abs(d['share_class_changed'] - 6 / 5) < 1e-9      # 第 99 類の変化は数えない
    print('demo ok')


if __name__ == '__main__':
    if sys.argv[1:] == ['demo']:
        demo()
    else:
        d = diff(T.load_tree(sys.argv[1]), T.load_tree(sys.argv[2]))
        if len(sys.argv) > 3:
            Path(sys.argv[3]).write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
        print(json.dumps({k: (len(v) if isinstance(v, (list, dict)) and k not in ('kinds', 'node_text_changed_by_digits') else v) for k, v in d.items()}, ensure_ascii=False, indent=1))
