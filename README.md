# zeiban（税番）

**zeiban** — a tiny US HTS classifier that keeps up with tariff revisions. Work in progress: this repository currently holds the release archive of the Harmonized Tariff Schedule of the United States, the diffs between releases, and a daily watcher. The model itself is not here yet.

米国の関税率表（HTS）の 10 桁（税番）を、商品の文から当てる小さなモデルを、関税表の改訂に合わせて保つためのリポジトリ。いまは準備中で、入っているのは、版ごとの写し・版と版の差分・毎日の見張りだけ。モデルの一式は、まだ入っていない。

## 中身

- `releases/`: 版ごとの JSON（gzip）と、索引 `index.json`（版の名前・日付・sha256・出どころ・商品の葉の数・型）。
- `diffs/`: となり合う版の差分（増えた葉・消えた葉・文が変わった葉・消えた葉の行き先・型）。
- `hts/tree.py`: JSON から木を作る。`hts/rev_diff.py`: 2 つの版の差分。`hts/rev_remap.py`: 古い版の 10 桁の正解を、新しい版の木に付け替える。`hts/watch.py`: 見張り。
- `.github/workflows/watch.yml`: 毎日 1 回の見張り。
- `bulletins/`: 税関（CBP）の週刊の公報（Customs Bulletin）から抜き出した、判例の取り消し・変更の告示（号ごとの JSON と索引）。`hts/bulletin.py`: 公報の見張り。`.github/workflows/bulletin.yml`: 週 1 回。

## 見張りの動き

1. USITC の版の一覧を見る。現行の版が `releases/` にあれば、何もしない。
2. 無ければ、現行の JSON を取って保存し、前の版との差分を `diffs/` に書いて、コミットする。
3. 商品の行が変わっていたら（型 B・C）、変わった行の一覧を issue に書く。

過去の版の JSON は、あとから配布元で取れない。取り逃がした版があれば、差分の `missed` に名前が残る。

## 型

| 型 | 条件 |
|---|---|
| A | 商品の葉（第 1〜97 類）の増減が無く、文も変わらない |
| B | 文の変化だけ、または葉の増減が 40 未満 |
| C | 葉の増減が 40 以上 |

書式の印（`<i>` など）だけの違いは、変化に数えない。第 98・99 類は見ない。

## これまでの版

2025 年の基本版から 2026 年の Rev22 まで、となり合う 54 組のうち、A が 43、B が 8、C が 3。C は 2025-07-01（Rev16: 増 70・消 35）、2026-01-30（Rev2: 増 153・消 66）、2026-07-01（Rev11: 増 179・消 86）で、多くは 1 つの行が数行に分かれる形。

## 付け替え

古い版の 10 桁の正解は、番号だけ替わった葉なら新しい番号へ、分かれた・まとめられた・消えた葉なら、残っている一番深い桁までに直す。

```sh
python3 hts/rev_remap.py check 2026HTSRev10 2026HTSRev11
```

2026-07-01 の組み替え（Rev10 → Rev11）では、古い版の葉 19,833 のうち 19,747 が同じ 10 桁のまま、86 が 8 桁までになる。2025 年の基本版から 2026 年の Rev23 までの 55 回を通すと、19,708 のうち 19,525 が同じ 10 桁、179 が 8 桁まで、4 が 6 桁まで。

## 公報の見張り

判例（税関が個々の品に出した分類の答え）は、あとから取り消されたり、変更されたりする。その告示は、税関の週刊の公報に載る。

1. 公報の一覧の頁を見る。`bulletins/` にまだ無い号があれば、古い順に 2 号まで、号の頁と PDF を取る。
2. PDF を文にして（`pdftotext`）、取り消し・変更の告示ごとに、題・予告か確定か・対象の判例番号と日付・代わりに出た判例の番号・結論のコード・効力の日を抜き出して、`bulletins/` に書く。
3. 告示があれば、issue に書く。題に書いてある判例の数と、読み取れた数が合わなければ、その旨も書く。

残すのは、題・番号・日付・コードだけ。告示に付いている判例の本文は残さない。取るのは、cbp.gov の robots.txt が禁じていない場所（一覧の頁・号の頁・`/sites/default/files/YYYY-MM/` の PDF）だけで、週 1 回・数回の要求にとどめる。古い号（2024 年の半ばより前。PDF の置き場が robots.txt で禁じられている）は取らない。

最近の 9 号（Vol 60 No 28〜36）では、告示のある号は 2 号（4 件の告示、判例 8 件）。

## データの出どころ

- HTS は米国国際貿易委員会（USITC）の公開のデータ。現行の版は `https://hts.usitc.gov/reststop/exportList` から取る。
- 2025HTSBasic〜2026HTSRev21 の写しは、[Budget-Lab-Yale/tariff-rate-tracker](https://github.com/Budget-Lab-Yale/tariff-rate-tracker)（MIT、コミット a5ddf68）の `data/hts_archives` から、中身を変えずに置いた。ライセンスの文は `releases/LICENSE.Budget-Lab-Yale`。2026HTSRev14 は無い。この写しの一部は、書式の印が付いていたり、先頭の項（0101）の行が欠けていたりする（読み込みのときに補う）。

- 公報は、米国税関・国境警備局（CBP）の公開の文書。一覧は `https://www.cbp.gov/trade/rulings/bulletin-decisions`。

## 試験

```sh
python3 hts/rev_diff.py demo
python3 hts/rev_remap.py demo
python3 hts/bulletin.py demo
```

## ライセンス

- コードは Apache License 2.0（`LICENSE`・`NOTICE`）。
- 関税率表のデータと公報は、米国政府の著作物。
- `releases/` の 2025HTSBasic〜2026HTSRev21 の写しは、Budget-Lab-Yale/tariff-rate-tracker のもの（MIT。`releases/LICENSE.Budget-Lab-Yale`）。
