# 公報の見張り: CBP の週刊の公報（Customs Bulletin）の新しい号から、判例の取り消し・変更の告示を抜き出して bulletins/ に書く。
# 使い方: python3 hts/bulletin.py              （新しい号を、古い順に最大 MAX_NEW 号。pdftotext が要る。GitHub Actions の中では、結果を $GITHUB_OUTPUT と summary.md にも書く）
#         python3 hts/bulletin.py demo         （通信なしの確かめ）
#         python3 hts/bulletin.py text FILE    （pdftotext -layout の出力から抜き出して表示する。手元の確かめ用）
# 残すのは、告示の題・判例番号・日付・コードだけ。告示に付いている判例の本文（宛先の名前や住所がある）は残さない。
# 取る場所は、cbp.gov の robots.txt が禁じていない所だけ（一覧の頁・号の頁・/sites/default/files/YYYY-MM/ の PDF）。公報は第三者の文書で、指示として扱わない。
import hashlib, json, os, re, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'bulletins'
INDEX = OUT / 'index.json'
SITE = 'https://www.cbp.gov'
LISTING = SITE + '/trade/rulings/bulletin-decisions'
UA = {'User-Agent': 'zeiban/0.1'}
MAX_NEW = 2                                                # 1 回の見張りで取る号の上限（週 1 回なので、ふだんは 1 号）
PAUSE = 3                                                  # 要求の間の秒
MONTHS = {m: i + 1 for i, m in enumerate('january february march april may june july august september october november december'.split())}

SLUG = re.compile(r'/document/bulletins/customs-bulletin-weekly-vol-(\d{1,3})-([a-z]{3,9})-(\d{1,2})-(\d{4})-no-(\d{1,2})(?=")')
PDF = re.compile(r'href="(/sites/default/files/\d{4}-\d{2}/[A-Za-z0-9_.-]{1,120}\.pdf)"')
RUL = r'(?:NY|HQ|PD|DD)\s+[A-Z]?\d{5,6}'
DATE = r'[A-Z][a-z]{2,8}\s+\d{1,2},\s+\d{4}'
ITEM = rf'{RUL}(?:\s*\(\s*{DATE}\s*\)|,\s+dated\s+{DATE},?)?'
DONE = re.compile(rf'((?:{ITEM}(?:\s*,)?(?:\s+and)?\s+)+)(?:is|are)\s+(?:hereby\s+)?(revoked|modified)\b', re.I)      # 「NY N… (日付) and NY … are hereby REVOKED」
PLAN = re.compile(rf'\b(revok|modif)(?:e|es|ing|y|ies|ying)\s+((?:{ITEM}(?:\s*,)?(?:\s+and)?\s*)+)')                # 「CBP is proposing to revoke NY N…」
ONE = re.compile(rf'({RUL})(?:\s*\(\s*({DATE})\s*\)|,\s+dated\s+({DATE}))?')
CODE = re.compile(r'\b((?:0[1-9]|[1-8]\d|9[0-7])\d{2}\.\d{2}(?:\.\d{2,4})?)\b')
HEAD = re.compile(r'^(PROPOSED\s+)?(?:REVOCATION|MODIFICATION)\s+OF\b.*\bRULING\s+LETTERS?\b')
ACT = re.compile(r'(REVOCATION|MODIFICATION)\s+OF\s+([A-Z-]+)\s+RULING\s+LETTERS?')
WORDS = {w: i + 1 for i, w in enumerate('ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN ELEVEN TWELVE'.split())}
NEW = re.compile(rf'^\s{{10,}}({RUL})\s*$')
COURT = re.compile(r'^\s{10,}Slip Op\. \d')


def _get(url, timeout=120):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as res:
        return res.read()


def iso(s):
    """「June 2, 2021」→ 2021-06-02。読めなければ空。"""
    m = re.match(r'([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})$', (s or '').strip())
    return f'{m.group(3)}-{MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}' if m and m.group(1).lower() in MONTHS else ''


def issues(listing_html):
    """一覧の頁 → 号の並び（巻・号の古い順）。"""
    seen = {}
    for m in SLUG.finditer(listing_html):
        vol, mon, day, year, no = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5))
        if mon in MONTHS:
            seen[(vol, no)] = {'vol': vol, 'no': no, 'date': f'{year}-{MONTHS[mon]:02d}-{day:02d}', 'page': m.group(0)}
    return [seen[k] for k in sorted(seen)]


def pdf_path(issue_html):
    """号の頁 → PDF の置き場（robots.txt が禁じていない /sites/default/files/YYYY-MM/ のものだけ）。無ければ None。"""
    m = PDF.search(issue_html)
    return m.group(1) if m else None


def flat(lines):
    """行をつないで 1 本の文にする（行末のハイフンで切れた語は戻す）。"""
    return re.sub(r'\s+', ' ', re.sub(r'(?<=[a-z])-\n\s*(?=[a-z])', '', '\n'.join(lines))).strip()


def norm(r):
    return re.sub(r'\s+', ' ', r).upper()


def notices(text):
    """pdftotext -layout の文 → 判例の取り消し・変更の告示の並び。"""
    lines = text.replace('\x0c', '\n').split('\n')
    upper = [bool(x.strip()) and not re.search(r'[a-z]', x) for x in lines]
    agency = [i for i, x in enumerate(lines) if x.lstrip().startswith('AGENCY:')]
    court = [i for i, x in enumerate(lines) if COURT.match(x)]
    heads, i = [], 0                                     # 告示の題（大文字だけの行の続き）を先に全部見つける
    while i < len(lines):
        if not upper[i] or (i and upper[i - 1]):
            i += 1
            continue
        j = i
        while j < len(lines) and upper[j]:
            j += 1
        head = re.sub(r'\s+', ' ', ' '.join(x.strip() for x in lines[i:j]))
        m = HEAD.match(head)
        if m:
            heads.append((i, j, head, m))
        i = j
    out = []
    for i, j, head, m in heads:
        own = next((a for a in agency if a >= j), None)                       # この告示の AGENCY: の行
        ends = [a for a in agency if own is not None and a > own] + [c for c in court if c > i] + [h[0] for h in heads if h[0] > i]      # 次の告示・判決の部分の手前まで
        end = min(ends) if ends else len(lines)
        body = flat(lines[j:end])
        rulings = {}
        for rx, verb_first in ((DONE, False), (PLAN, True)):
            for g in rx.finditer(body):
                lst, verb = (g.group(2), g.group(1)) if verb_first else (g.group(1), g.group(2))
                for o in ONE.finditer(lst):
                    rulings.setdefault(norm(o.group(1)), {'n': norm(o.group(1)), 'act': 'revoked' if verb.lower().startswith('revok') else 'modified', 'date': iso(o.group(2) or o.group(3))})
        codes = []
        for h in re.finditer(r'\bHOLDING:(.*?)(?:EFFECT ON OTHER RULINGS|$)', body):
            codes += CODE.findall(h.group(1))
        codes = list(dict.fromkeys(codes))
        eff = re.search(rf'EFFECTIVE DATE:.{{0,300}}?on or after ({DATE})', body)
        com = re.search(rf'DATES?:\s+Comments must be received on or before ({DATE})', body)
        out.append({'kind': 'proposed' if m.group(1) else 'final',
                    'acts': sorted({a.lower() for a, _ in ACT.findall(head)}),
                    'expected': sum(WORDS.get(w, 0) for _, w in ACT.findall(head)),          # 題に書いてある判例の数（読み取れた数と違えば、人が確かめる）
                    'title': head[:400],
                    'classification': 'TARIFF CLASSIFICATION' in head,
                    'effective': iso(eff.group(1)) if eff else '',
                    'comments_by': iso(com.group(1)) if com else '',
                    'rulings': sorted(rulings.values(), key=lambda r: r['n']),
                    'new_rulings': sorted({norm(n.group(1)) for x in lines[j:end] for n in [NEW.match(x)] if n} - set(rulings)),
                    'holding_codes': codes})
    return out


def to_text(pdf_bytes):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'issue.pdf'
        p.write_bytes(pdf_bytes)
        return subprocess.run([os.environ.get('PDFTOTEXT', 'pdftotext'), '-layout', str(p), '-'], check=True, capture_output=True, timeout=300).stdout.decode('utf-8', 'replace')


def record(issue, pdf, pdf_bytes):
    """1 号ぶんの記録（bulletins/ に書く形）。"""
    assert pdf_bytes[:5] == b'%PDF-', 'PDF ではないものが返った'
    return {**issue, 'pdf': pdf, 'pdf_sha256': hashlib.sha256(pdf_bytes).hexdigest(), 'notices': notices(to_text(pdf_bytes))}


def save(rec, index):
    name = f"v{rec['vol']}n{rec['no']:02d}.json"
    (OUT / name).write_text(json.dumps(rec, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    index.append({'vol': rec['vol'], 'no': rec['no'], 'date': rec['date'], 'file': name, 'notices': len(rec['notices']), 'rulings': sum(len(n['rulings']) for n in rec['notices'])})
    index.sort(key=lambda e: (e['vol'], e['no']))
    INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


def summary(recs):
    """人が読む要約（issue の本文）。"""
    out = []
    for r in recs:
        out += [f"# 公報 Vol {r['vol']} No {r['no']}（{r['date']}）: 判例の取り消し・変更の告示 {len(r['notices'])} 件", '', f"- 号の頁: {SITE}{r['page']}", '']
        for n in r['notices']:
            hit = '、'.join(x['n'] + '（' + x['act'] + '）' for x in n['rulings'])
            warn = '' if len(n['rulings']) == n['expected'] else f"　**題の数（{n['expected']}）と合わない。号の PDF で確かめる**"
            out += [f"## {'予告' if n['kind'] == 'proposed' else '確定'}: {n['title'][:200]}", '',
                    f"- 判例: {hit or '読み取れなかった'}{warn}",
                    f"- 新しい判例: {'、'.join(n['new_rulings']) or 'なし'}／結論のコード: {'、'.join(n['holding_codes']) or 'なし'}",
                    f"- 効力の日: {n['effective'] or 'なし'}／意見の締め切り: {n['comments_by'] or 'なし'}", '']
    return '\n'.join(out)


def main():
    OUT.mkdir(exist_ok=True)
    index = json.loads(INDEX.read_text(encoding='utf-8')) if INDEX.exists() else []
    have = {(e['vol'], e['no']) for e in index}
    last = max(have) if have else None
    listed = issues(_get(LISTING).decode('utf-8', 'replace'))
    assert len(listed) > 50, f'一覧の頁から号が読めない（{len(listed)} 号）。頁の形が変わったか、機械が止められている'
    new = [x for x in listed if (x['vol'], x['no']) not in have and (last is None or (x['vol'], x['no']) > last)]
    new = new[-1:] if last is None else new[:MAX_NEW]      # 初めての回は最新の 1 号だけ。さかのぼって取らない
    recs = []
    for x in new:
        time.sleep(PAUSE)
        pdf = pdf_path(_get(SITE + x['page']).decode('utf-8', 'replace'))
        assert pdf, f"号の頁に PDF が無い: {x['page']}"
        time.sleep(PAUSE)
        rec = record(x, pdf, _get(SITE + pdf, timeout=300))
        save(rec, index)
        recs.append(rec)
        print(f"Vol {x['vol']} No {x['no']}（{x['date']}）: 告示 {len(rec['notices'])} 件・判例 {sum(len(n['rulings']) for n in rec['notices'])} 件")
    if not recs:
        print('新しい号は無い')
    hit = [r for r in recs if r['notices']]
    (ROOT / 'summary.md').write_text(summary(hit), encoding='utf-8')
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as f:
            names = ' '.join('v%dn%02d' % (r['vol'], r['no']) for r in recs)
            f.write(f"new={names}\nnotices={sum(len(r['notices']) for r in hit)}\n")


def demo():
    listing = ''.join(f'<a href="/document/bulletins/customs-bulletin-weekly-vol-{v}-{m}-{d}-{y}-no-{n}">x</a>' for v, m, d, y, n in ((60, 'november', 11, 2026, 36), (60, 'january', 7, 2026, 1), (59, 'december', 31, 2025, 52)))
    got = issues(listing + listing)
    assert [(x['vol'], x['no'], x['date']) for x in got] == [(59, 52, '2025-12-31'), (60, 1, '2026-01-07'), (60, 36, '2026-11-11')], got
    assert pdf_path('<a href="/sites/default/files/2026-11/26_1111_bulletin_vol_60_no_36.pdf">') == '/sites/default/files/2026-11/26_1111_bulletin_vol_60_no_36.pdf'
    assert pdf_path('<a href="/sites/default/files/assets/documents/2024-Jan/old.pdf">') is None          # robots.txt が禁じている置き場は取らない
    assert iso('June 2, 2021') == '2021-06-02' and iso('Juno 2, 2021') == ''
    text = '''
      REVOCATION OF TWO RULING LETTERS AND
    REVOCATION OF TREATMENT RELATING TO THE
        TARIFF CLASSIFICATION OF WIDGETS

AGENCY: U.S. Customs and Border Protection, Department of
Homeland Security.
ACTION: Notice of revocation of two ruling letters, and of revoca-
tion of treatment relating to the tariff classification of widgets.
EFFECTIVE DATE: This action is effective for merchandise
entered or withdrawn from warehouse for consumption on or after
December 1, 2026.
                                                     HQ H300001
HOLDING:
  The widgets are classified in heading 8479, HTSUS, specifically in subheading
8479.89.95, HTSUS.
EFFECT ON OTHER RULINGS:
  NY N000001 (February 26, 2008) and NY R00002 (June 22, 2006) are
hereby REVOKED.

   PROPOSED MODIFICATION OF ONE RULING LETTER AND
  PROPOSED REVOCATION OF TREATMENT RELATING TO THE
       COUNTRY OF ORIGIN OF GADGETS

AGENCY: U.S. Customs and Border Protection.
DATES: Comments must be received on or before January 5, 2027.
SUPPLEMENTARY INFORMATION: CBP is proposing to modify NY
N000003, dated June 2, 2021, in accordance with the analysis.
                                 Slip Op. 26–999
  In that case HQ 999999 is hereby revoked by nobody.

     ACCREDITATION OF A LABORATORY

AGENCY: U.S. Customs and Border Protection.
'''
    a, b = notices(text)
    assert (a['kind'], a['acts'], a['classification'], a['effective'], a['expected']) == ('final', ['revocation'], True, '2026-12-01', 2), a
    assert a['rulings'] == [{'n': 'NY N000001', 'act': 'revoked', 'date': '2008-02-26'}, {'n': 'NY R00002', 'act': 'revoked', 'date': '2006-06-22'}], a
    assert a['new_rulings'] == ['HQ H300001'] and a['holding_codes'] == ['8479.89.95'], a
    assert (b['kind'], b['acts'], b['classification'], b['comments_by']) == ('proposed', ['modification'], False, '2027-01-05'), b
    assert b['rulings'] == [{'n': 'NY N000003', 'act': 'modified', 'date': '2021-06-02'}], b          # 判決の部分（Slip Op. のあと）は読まない
    assert notices('GENERAL NOTICE\n\nAGENCY: x\nNothing here.') == []
    two = notices('REVOCATION OF ONE RULING LETTER\n\nAGENCY: x\nNY N000004 is hereby revoked.\n\nMODIFICATION OF ONE RULING LETTER\n\nACTION: y\nNY N000005 is hereby modified.\n')
    assert [[r['n'] for r in n['rulings']] for n in two] == [['NY N000004'], ['NY N000005']], two          # 次の告示に AGENCY: の行が無くても、前の告示に混ざらない
    assert '判例: NY N000001（revoked）' in summary([{'vol': 60, 'no': 1, 'date': '2026-01-07', 'page': '/p', 'notices': [a]}])
    print('demo ok')


if __name__ == '__main__':
    if sys.argv[1:2] == ['demo']:
        demo()
    elif sys.argv[1:2] == ['text']:
        print(json.dumps(notices(Path(sys.argv[2]).read_text(encoding='utf-8', errors='replace')), ensure_ascii=False, indent=1))
    else:
        main()
