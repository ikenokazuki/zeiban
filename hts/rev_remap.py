# 付け替え: 古い版の 10 桁の正解を、新しい版の木での正解（残っている一番深い桁まで）に直す。
# 番号だけ替わった葉は新しい番号を追う。分かれた・まとめられた・消えた葉は、消えた時点で残っていた桁までに切る。
# 使い方: python3 hts/rev_remap.py demo
#         python3 hts/rev_remap.py check <古い版の名前> <新しい版の名前> [コードの列のある CSV  コードの列  日付の列]
import collections, csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hts import rev_diff as D, tree as T, watch as W  # noqa: E402

RENUM = '番号だけ替わった'


def remap(code, chain, new):
    """code: 古い版の葉の 10 桁。chain: 古い版から新しい版までの差分（古い順）。new: 新しい版の木。→ (コード, 桁数)。どの桁も残っていなければ None。"""
    cur = T.digits(code)
    for d in chain:
        e = d['remap'].get(cur) if len(cur) == 10 else None
        if e is None:
            continue
        cur = e[0]
        if e[2] != RENUM:                 # 切ったら、そこから先は番号を追わない（切った桁が後の版で消えた場合は、下の resolve がさらに上の桁に合わせる）
            break
    return new.resolve(cur) if cur else None


def chain_between(old_name, new_name):
    """releases/index.json の並びで、old_name から new_name までの差分を古い順に読む。"""
    names = [e['name'] for e in json.loads(W.INDEX.read_text(encoding='utf-8'))]
    i, j = names.index(old_name), names.index(new_name)
    assert i < j, '古い版・新しい版の順に書く'
    return [json.loads((W.DIFFS / f'{a}__{b}.json').read_text(encoding='utf-8')) for a, b in zip(names[i:j], names[i + 1:j + 1])]


def tally(codes, chain, new):
    """コードの並び → 付け替えのあとの形ごとの数。"""
    c = collections.Counter()
    for x in codes:
        r = remap(x, chain, new)
        c['どの桁も残らない' if r is None else '同じ 10 桁' if r == (x, 10) else '新しい番号の 10 桁' if r[1] == 10 else f'{r[1]} 桁まで'] += 1
    return dict(c)


def check(old_name, new_name, table=None, code_col=None, date_col=None):
    load = lambda n: W.load(W.REL / f'{n}.json.gz')
    old, new, chain = load(old_name), load(new_name), chain_between(old_name, new_name)
    out = {'old': old_name, 'new': new_name, 'steps': len(chain), 'leaves_old': len(old.leaf_of), 'leaves_new': len(new.leaf_of),
           'all_old_leaves': tally(sorted(old.leaf_of), chain, new)}
    direct = D.diff(old, new)                                   # 間の版を見ずに、両端だけで比べた付け替えと同じ結果になる葉の割合
    same = sum(remap(c, chain, new) == remap(c, [direct], new) for c in old.leaf_of)
    out['same_as_direct'] = same / len(old.leaf_of)
    for c in old.leaf_of:                                       # 行き先は、必ず新しい版の木にある行
        r = remap(c, chain, new)
        assert r is None or r[0] in new.leaf_of or r[0] in new.by_code or r[0][:8] + '00' in new.leaf_of, (c, r)
    if table:
        cut = next(e['date'] for e in json.loads(W.INDEX.read_text(encoding='utf-8')) if e['name'] == new_name)
        csv.field_size_limit(10**9)
        with open(table, encoding='utf-8', newline='') as f:
            rows = [T.digits(r[code_col]) for r in csv.DictReader(f) if (r[date_col] or '')[:10] < cut]
        rows = [c for c in rows if c in old.leaf_of]            # 古い版で葉だった正解だけ
        out['table'] = {'rows_before_new_release_and_leaf_in_old': len(rows), **tally(rows, chain, new)}
    return out


def demo():
    R = lambda code, text, indent: {'htsno': code, 'description': text, 'indent': indent}
    v1 = [R('0101', 'Live horses', 0), R('0101.21.00', 'Purebred', 1), R('0101.29.00', 'Other', 1), R('0101.29.00.10', 'Imported for slaughter', 2), R('0101.29.00.90', 'Other', 2),
          R('0102', 'Live bovine animals', 0), R('0102.21.00', 'Purebred breeding animals', 1), R('0102.21.00.10', 'Dairy', 2), R('0102.21.00.20', 'Other', 2)]
    v2 = [R('0101', 'Live horses', 0), R('0101.21.00', 'Purebred', 1), R('0101.29.00', 'Other', 1), R('0101.29.00.15', 'Imported for slaughter', 2), R('0101.29.00.90', 'Other', 2),
          R('0102', 'Live bovine animals', 0), R('0102.21.00', 'Purebred breeding animals', 1), R('0102.21.00.10', 'Dairy', 2), R('0102.21.00.30', 'Male', 2), R('0102.21.00.40', 'Female', 2)]
    v3 = [R('0101', 'Live horses', 0), R('0101.21.00', 'Purebred', 1), R('0101.29.00', 'Other', 1), R('0101.29.00.17', 'Imported for slaughter', 2), R('0101.29.00.90', 'Other', 2),
          R('0102', 'Live bovine animals', 0), R('0102.90.00', 'Other', 1)]
    a, b, c = (T.parse_export(v, n) for v, n in ((v1, 'a'), (v2, 'b'), (v3, 'c')))
    ab, bc = D.diff(a, b), D.diff(b, c)
    assert remap('0101210000', [ab, bc], c) == ('0101210000', 10)                # 変わらない葉
    assert remap('0101290010', [ab], b) == ('0101290015', 10)                    # 番号だけ替わった
    assert remap('0101290010', [ab, bc], c) == ('0101290017', 10)                # 2 回続けて番号が替わっても追える
    assert remap('0102210020', [ab], b) == ('01022100', 8)                       # 分かれた → 8 桁まで
    assert remap('0102210020', [ab, bc], c) == ('0102', 4)                       # 切った 8 桁が次の版で消えた → 残っている 4 桁まで
    assert remap('0102210010', [ab, bc], c) == ('0102', 4)                       # 2 つ目の版で消えた
    assert remap('9999999999', [ab, bc], c) is None                              # もともと木に無い番号
    assert tally(['0101210000', '0101290010', '0102210020'], [ab], b) == {'同じ 10 桁': 1, '新しい番号の 10 桁': 1, '8 桁まで': 1}
    print('demo ok')


if __name__ == '__main__':
    if sys.argv[1:] == ['demo']:
        demo()
    else:
        print(json.dumps(check(*sys.argv[2:7]), ensure_ascii=False, indent=1))
