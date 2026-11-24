// zeiban: 商品の文から米国 HTS の 10 桁を出す実行ファイル。
// 一式（zeiban.bin と目録 zeiban.json。リリースで配る）をメモリにマップして読み、1 件ごとに全部を自分で行う:
//   語彙の点（NFKC・小文字 → 語の区切り → BMX の上位 64 葉）→ モデルの語の区切り（256 トークンまで）→ 3 層のエンコーダ
//   → 5 段の点（scale × cos + λ × BMX）→ 葉の点（段ごとの対数確率の和）。計算は、学習のときの Python の実装と同じ。
// 使い方: zeiban <一式のフォルダ> [k | debug | bench | ids]
//   標準入力に 1 行 1 件の JSON。2 通り:
//     {"name": 品名, "material": 素材, "categories": 分類（文か、文の並び）, "description": 説明}（name のほかは無くてよい）
//       → モデルに渡す文は「title: 品名 | material: 素材 | category: 分類 | description: 説明」（学習のときの EC の文と同じ形。素材・分類は品名のすぐ後に置く。
//         長い説明の後ろだと効きが落ちる）。語彙の点に使う文は、印（title: など）を除いて空白でつないだもの。
//     {"text": モデルに渡す文, "lex": 語彙の点に使う文（無ければ text）}（文を自分で組み立てる呼び手・照合用）
//     どちらにも "prefix": "6109" のように、もう分かっている頭の桁（2・4・6・8・10 桁）を足せる（質問に答えたあとの呼び直し。その下の葉だけから選ぶ。状態は持たない）。
//   → 標準出力に 1 行 1 件の JSON:
//     codes・scores: 上位 k 件（既定 5）の 10 桁と葉の点。
//     hsCode {code, description}・sixDigit・confidenceScore・sixDigitConfidence・dutyRate・alternates [{code, description, probabilityMass, dutyRate}]・htsRelease。
//     confidenceScore は 1 位の 10 桁が正しい確率、sixDigitConfidence は 6 桁までが正しい確率（調整用の行で合わせ込んだ値。目録の calib・calibPrefix）。
//       prefix つきの呼び直しでは、呼び手の答えは正しいとして出す（答えた段の確度は 1）。
//     dutyRate {general, special, other}: 関税表の基本の税率の欄（公式の文字のまま）。第 99 類の追加の関税（232 条・301 条など）・不当廉売の関税・脚注は入らない = 実際に払う額ではない。
//     questions: 質問（0 か 1 個）。確度が tau より低い最初の段を聞く（自信のある段は飛ばす）。候補は、1 位とその上の段まで同じ葉のうち点の高い 30 葉（QN。応答に出す k 件より広い）。
//       どの段も確度が tau 以上なら、上位 30 葉の頭の桁が分かれる最初の段（聞く目安は立たない）。
//       入力に "confirm": true を足すと、自信のある段も飛ばさず、いつも「上位 30 葉の頭の桁が分かれる最初の段」を聞く（上の段から順に全部を確かめたい呼び手のため）。{digits, ask, confidence, options: [{prefix, label, code, probabilityMass, dutyRate}]}。
//       選択肢の文（label）は関税表の公式の文。code は「これを選んだらこのコード」。confidence は 1 位の選択肢が正しい確率、ask は、それが目録の tau より低いとき true（聞く目安）。
//     probabilityMass は、上位 k 件の葉の点の softmax（目録の temp で割る）。候補の中での割合で、合わせ込みをした確率ではない。
//   debug: トークンの番号（ids）・BMX の上位（bmx: [葉の番号, 点]）・語彙の点の前処理をした文（norm）・計算の経路（path）も出す（Python の基準との照合用）。
//   ids: モデルに渡すトークンの番号だけを出す（モデルは動かさない。語の区切りの照合用）。
//   bench: 全部の行を読んでから 1 件ずつ時間を測り、1 行の JSON（読み込みの時間・常駐メモリ・スレッドの数・計算の経路・件ごとの [トークン数, マイクロ秒]）を出す。
// 計算は 2 スレッド（rayon の 2 本の中で全部を動かす）。エンコーダの計算の経路は、CPU に AVX2 と FMA があれば avx2、無ければ portable（どの CPU でも動く。enc.rs）。
// 環境変数 ZEIBAN_PORTABLE を置くと、いつも portable を使う。
// WebAssembly（wasm32-unknown-unknown。rs/build-wasm.sh）では、実行ファイルの代わりに、呼び出す口を出す（下の mod wasm）。
//   ファイルも標準入出力も使わない: 呼び手が、目録と一式の中身を wasm のメモリに書いて渡す。入力と応答は、上の JSON の 1 行と同じ。スレッドは 1 本。
mod enc;
mod enc_g;
mod tok;

use anyhow::{bail, Context, Result};
use enc::D;
use serde_json::{json, Value};
#[cfg(not(target_arch = "wasm32"))]
use std::{
    io::{BufRead, Write},
    path::Path,
    time::Instant,
};

#[cfg(target_endian = "big")]
compile_error!("一式の数はリトルエンディアンで書いてある（マップしたまま読むので、ビッグエンディアンの台では読めない）");

const NONE: u16 = 0xFFFF; // tok.term: 語彙の点に使わない片

pub struct Bundle {
    map: &'static [u8],
    pub man: Value,
}

pub trait Num: Copy {
    const DTYPE: &'static str;
}
macro_rules! num {
    ($($t:ty = $n:literal),*) => { $(impl Num for $t { const DTYPE: &'static str = $n; })* };
}
num!(i8 = "int8", u8 = "uint8", u16 = "uint16", u32 = "uint32", f32 = "float32", f64 = "float64");

impl Bundle {
    #[cfg(not(target_arch = "wasm32"))]
    fn open(dir: &Path) -> Result<&'static Self> {
        let man: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("zeiban.json"))?)?;
        let file = std::fs::File::open(dir.join(man["file"].as_str().context("目録の file")?))?;
        // SAFETY: 読むだけのマップ。動いている間にファイルを書き換えないこと
        let map: &'static [u8] = Box::leak(Box::new(unsafe { memmap2::Mmap::map(&file)? }));
        Self::from_parts(man, map)
    }

    // 目録と、一式の中身（64 バイトの境界にそろえて置いたもの）から作る。WebAssembly では、こちらだけを使う
    fn from_parts(man: Value, map: &'static [u8]) -> Result<&'static Self> {
        if man["bytes"].as_u64() != Some(map.len() as u64) {
            bail!("一式の大きさが目録と合わない");
        }
        Ok(Box::leak(Box::new(Self { map, man }))) // 終わりまで使うので、解放しない（区画の参照を 'static で持てる）
    }

    // 区画を型つきの並びとして読む（写さない）
    pub fn get<T: Num>(&self, name: &str) -> &[T] {
        let s = &self.man["sections"][name];
        assert!(s["dtype"] == T::DTYPE, "区画 {name} が無いか、型が違う");
        let (at, n) = (s["offset"].as_u64().unwrap() as usize, s["size"].as_u64().unwrap() as usize);
        // SAFETY: 数の型はどのビットの並びも有効。前後に余りが出ない（境界がそろっている）ことは下で確かめる
        let (a, b, c) = unsafe { self.map[at..at + n].align_to::<T>() };
        assert!(a.is_empty() && c.is_empty(), "区画 {name} の境界");
        b
    }
}

// numpy の和（pairwise sum）と同じ足し方。BMX の ebar を Python の実装と同じ値にする
fn np_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        return a.iter().fold(0.0, |s, &x| s + x);
    }
    if n > 128 {
        let h = (n / 2) & !7;
        return np_sum(&a[..h]) + np_sum(&a[h..]);
    }
    let mut r = [0f64; 8];
    r.copy_from_slice(&a[..8]);
    for c in a[8..].chunks_exact(8) {
        for j in 0..8 {
            r[j] += c[j];
        }
    }
    a[n - n % 8..].iter().fold(((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7])), |s, &x| s + x)
}

// BMX の索引（学習のときの Python の実装と同じ式・同じ計算の順。f64）。語 w の行は ptr[w]..ptr[w + 1]
struct Bmx {
    ptr: &'static [u32],
    doc: &'static [u16],
    tf: &'static [u8],   // 回数。0 は big（位置, 回数）にある（255 を超える回数）
    big: &'static [u32],
    len: &'static [u32],
    idf: &'static [f64],
    ent: &'static [f64],
    cst: &'static [f64], // avgdl・alpha・beta・norm1
    acc: Vec<[f64; 3]>,  // 文書ごとの (点, 当たった語の数, その語の e の和)
}

impl Bmx {
    // 問いの語の番号（重複なし・出た順）→ 上位 k 件の (葉の番号, 正規化した点)。点の高い順、同じ点は番号の小さい順
    fn top(&mut self, terms: &[u16], k: usize) -> Vec<(u16, f32)> {
        if terms.is_empty() {
            return Vec::new();
        }
        let (avgdl, a, beta, norm1) = (self.cst[0], self.cst[1], self.cst[2], self.cst[3]);
        let mx = terms.iter().map(|&w| self.ent[w as usize]).fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = terms.iter().map(|&w| if mx > 0.0 { self.ent[w as usize] / mx } else { 0.0 }).collect();
        let m = terms.len() as f64;
        let ebar = np_sum(&e) / m;
        let mut touched = Vec::new();
        for (&w, &ew) in terms.iter().zip(&e) {
            let idf = self.idf[w as usize];
            for p in self.ptr[w as usize] as usize..self.ptr[w as usize + 1] as usize {
                let d = self.doc[p] as usize;
                let f = match self.tf[p] {
                    0 => self.big.chunks_exact(2).find(|x| x[0] as usize == p).map_or(0.0, |x| x[1] as f64),
                    t => t as f64,
                };
                let x = &mut self.acc[d];
                if x[1] == 0.0 {
                    touched.push(d as u16);
                }
                x[0] += idf * f * (a + 1.0) / (f + a * self.len[d] as f64 / avgdl + a * ebar);
                x[1] += 1.0;
                x[2] += ew;
            }
        }
        let mut out: Vec<(f64, u16)> = Vec::with_capacity(touched.len());
        for d in touched {
            let x = std::mem::take(&mut self.acc[d as usize]);
            out.push((x[0] + beta * x[2] * x[1] / m, d));
        }
        out.sort_unstable_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
        out.truncate(k);
        out.into_iter().map(|(s, d)| (d, (s / (m * norm1)) as f32)).collect()
    }
}

struct Model {
    tok: tok::Tokenizer,
    nfkc: tok::Nfkc,
    term: &'static [u16], // 片の番号 → 語彙の点の語の番号
    bmx: Bmx,
    enc: Reader,
    node: &'static [i8],  // 節のベクトル（int8）。先頭 6 層の読み手の一式では空
    node4: &'static [u8], // 節のベクトル（int4。1 バイトに 2 個）。先頭 6 層の読み手の一式だけ
    node_s: &'static [f32], // 節ごとの数: 内積に掛けると、長さ 1 の節のベクトルとの内積になる
    n_nodes: Vec<usize>,
    scale: &'static [f32],
    lam: &'static [f32],
    leaf_path: &'static [u16],
    leaves: &'static [u8],
    q_len: usize,
    k: usize,
    release: String,
    labels: Vec<&'static str>, // 節ごと（段の順）の公式の文（親から先の分）。古い一式には無い（空）
    temp: f64,
    tau: f64,
    duty: Vec<&'static str>,      // 税率の 3 欄の組（一般・特恵・第 2 欄をタブで区切った行）。古い一式には無い（空）
    leaf_duty: &'static [u16],    // 葉 → duty の行の番号
    calib: Option<[[[f64; 4]; 5]; 2]>, // 確度の係数（段ごとの a・b・c・d）。[0] は最初の応答、[1] は prefix つきの呼び直し。無ければ、合わせ込みをしない
    alt: Option<Alt>,                  // 日本語の入力を読む 2 つ目の層（目録の route.ja。無い一式では、全部の入力をもとの層で読む）
}

// 文を読む層。目録の enc.kind が g6 の一式は、先頭 6 層の読み手（int4。語の区切りは BPE。文の頭に決まった文を付ける）。それ以外は、もとの 3 層の読み手
enum Reader {
    Mb(enc::Engine),
    G6 { eng: enc_g::Engine, bpe: tok::Bpe, pre: String },
}

// 2 つ目の層: 文を読む層・問いの射影・目盛り・確度の係数だけが別。語の埋め込み・節のベクトル・索引は、もとの層と同じものを使う
struct Alt {
    enc: enc::Engine,
    scale: &'static [f32],
    lam: &'static [f32],
    calib: Option<[[[f64; 4]; 5]; 2]>,
    share: f64, // 語の一致に使う文の、空白でない字のうち、かな・漢字がこの割合以上なら、この層で読む
}

// 切り替えの決まり: 空白でない字のうち、かな・漢字（ひらがな・カタカナ・半角カタカナ・漢字）が share 以上か
fn is_ja(s: &str, share: f64) -> bool {
    let (mut n, mut ja) = (0usize, 0usize);
    for c in s.chars() {
        if matches!(c, ' ' | '\t' | '\n' | '\r' | '\u{0b}' | '\u{0c}' | '\u{3000}') {
            continue;
        }
        n += 1;
        if matches!(c, '\u{3040}'..='\u{30ff}' | '\u{ff66}'..='\u{ff9f}' | '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}' | '\u{f900}'..='\u{faff}') {
            ja += 1;
        }
    }
    n > 0 && ja as f64 >= share * n as f64
}

// 質問を作る候補の葉の数。応答に出す上位 k 件（既定 5）とは別に、点の高い順にこの数まで見て、頭の桁が分かれる最初の段を質問にする。
// 5 件だけから作ると、正解の枝が選択肢に出ないことが多い（新しい判例の文で、質問 3 回までの 10 桁は 5 件 0.70・30 件 0.79）
const QN: usize = 30;

// 葉の点（-inf は対象の外）から、keep が真の葉の上位 n 件（点の高い順、同じ点は番号の小さい順）
fn pick(leaf: &[f32], n: usize, keep: impl Fn(usize) -> bool) -> Vec<(f32, usize)> {
    let mut top: Vec<(f32, usize)> = leaf.iter().enumerate().filter(|&(i, x)| *x > f32::NEG_INFINITY && keep(i)).map(|(i, x)| (*x, i)).collect();
    let cmp = |x: &(f32, usize), y: &(f32, usize)| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1));
    let n = n.min(top.len());
    if n < top.len() {
        top.select_nth_unstable_by(n, cmp);
        top.truncate(n);
    }
    top.sort_unstable_by(cmp);
    top
}

struct Pred {
    ids: Vec<u32>,
    bmx: Vec<(u16, f32)>,
    top: Vec<(f32, usize)>, // (葉の点, 葉の番号)。点の高い順に max(k, QN) 件まで
    leaf: Vec<f32>,         // 葉の点の全部（prefix の外の葉は -inf）。質問の候補を、枝の中から取り直すのに使う
    ja: bool,               // 2 つ目の層（日本語）で読んだか
}

impl Model {
    #[cfg(not(target_arch = "wasm32"))]
    fn load(dir: &Path) -> Result<Self> {
        Self::from_bundle(Bundle::open(dir)?)
    }

    fn from_bundle(b: &'static Bundle) -> Result<Self> {
        let num = |k: &str| b.man[k].as_u64().map(|x| x as usize).with_context(|| format!("目録の {k}"));
        let n_nodes: Vec<usize> = b.man["n_nodes"].as_array().context("目録の n_nodes")?.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
        let g6 = matches!(b.man["enc"]["kind"].as_str(), Some("g6" | "g24m"));
        let m = Self {
            tok: tok::Tokenizer::new(b)?,
            nfkc: tok::Nfkc::new(b),
            term: b.get("tok.term"),
            bmx: Bmx {
                ptr: b.get("bmx.ptr"),
                doc: b.get("bmx.doc"),
                tf: b.get("bmx.tf"),
                big: b.get("bmx.big"),
                len: b.get("bmx.len"),
                idf: b.get("bmx.idf"),
                ent: b.get("bmx.ent"),
                cst: b.get("bmx.const"),
                acc: vec![[0.0; 3]; n_nodes[4]],
            },
            enc: if g6 {
                Reader::G6 { eng: enc_g::Engine::new(b)?, bpe: tok::Bpe::new(b)?, pre: b.man["enc"]["text_prefix"].as_str().unwrap_or("").to_string() }
            } else {
                Reader::Mb(enc::Engine::new(b)?)
            },
            node: if g6 { &[] } else { b.get("node") },
            node4: if g6 { b.get("node4") } else { &[] },
            node_s: b.get(if g6 { "node4.n" } else { "node.s" }),
            scale: b.get(if g6 { "g.scale" } else { "scale" }),
            lam: b.get(if g6 { "g.lam" } else { "lam" }),
            leaf_path: b.get("leaf_path"),
            leaves: b.get("leaves"),
            q_len: num("q_len")?,
            k: num("k")?,
            release: b.man["htsRelease"].as_str().unwrap_or("").to_string(),
            labels: if b.man["sections"]["label"].is_object() { std::str::from_utf8(b.get::<u8>("label")).context("label 区画が UTF-8 でない")?.split('\n').collect() } else { Vec::new() },
            temp: b.man["temp"].as_f64().unwrap_or(2.0),
            tau: b.man["tau"].as_f64().unwrap_or(0.7),
            duty: if b.man["sections"]["duty"].is_object() { std::str::from_utf8(b.get::<u8>("duty")).context("duty 区画が UTF-8 でない")?.split('\n').collect() } else { Vec::new() },
            leaf_duty: if b.man["sections"]["leaf_duty"].is_object() { b.get("leaf_duty") } else { &[] },
            calib: match (&b.man["calib"], &b.man["calibPrefix"]) {
                (Value::Null, _) => None,
                (a, p) => Some([serde_json::from_value(a.clone()).context("目録の calib（段ごとの 4 つの数が 5 段ぶん）")?, serde_json::from_value(p.clone()).context("目録の calibPrefix")?]),
            },
            alt: match &b.man["route"]["ja"] {
                Value::Null => None,
                r => {
                    let pre = r["prefix"].as_str().context("目録の route.ja.prefix")?;
                    Some(Alt {
                        enc: enc::Engine::with_prefix(b, pre)?,
                        scale: b.get(&format!("{pre}scale")),
                        lam: b.get(&format!("{pre}lam")),
                        calib: match (&r["calib"], &r["calibPrefix"]) {
                            (Value::Null, _) => None,
                            (a, p) => Some([serde_json::from_value(a.clone()).context("目録の route.ja.calib")?, serde_json::from_value(p.clone()).context("目録の route.ja.calibPrefix")?]),
                        },
                        share: r["share"].as_f64().context("目録の route.ja.share")?,
                    })
                }
            },
            n_nodes,
        };
        let total: usize = m.n_nodes.iter().sum();
        if m.n_nodes.len() != 5 || m.node.len() + m.node4.len() * 2 != total * D || m.node_s.len() != total || m.leaf_path.len() != m.n_nodes[4] * 5 || m.leaves.len() != m.n_nodes[4] * 10 {
            bail!("節・葉の数が目録と合わない");
        }
        if !m.labels.is_empty() && m.labels.len() != total {
            bail!("公式の文の数が節の数と合わない");
        }
        if !m.duty.is_empty() && (m.leaf_duty.len() != m.n_nodes[4] || m.leaf_duty.iter().any(|&i| i as usize >= m.duty.len())) {
            bail!("税率の区画が葉の数と合わない");
        }
        Ok(m)
    }

    // モデルに渡すトークンの番号（頭と尻の印つき。長い文は後ろを切る）
    fn ids(&self, text: &str) -> Vec<u32> {
        match &self.enc {
            // 先頭 6 層の読み手: 決まった文 ＋ 文 を BPE で区切る
            Reader::G6 { bpe, pre, .. } => bpe.encode(&format!("{pre}{text}"), self.q_len),
            Reader::Mb(_) => {
                let mut ids = vec![self.tok.bos];
                ids.extend(self.tok.encode(text).into_iter().take(self.q_len - 2));
                ids.push(self.tok.eos);
                ids
            }
        }
    }

    fn predict(&mut self, text: &str, lex: &str, k: usize, prefix: &[u8]) -> Pred {
        // 語彙の点: 片 → 語（重複は最初の 1 つ）→ BMX の上位の葉
        let mut terms = Vec::new();
        for id in self.tok.encode(&self.nfkc.lex_norm(lex)) {
            let t = self.term[id as usize];
            // ponytail: contains は語の数の 2 乗。何万トークンもの文を受けるなら、見た語の印（語の数の長さの並び）に替える
            if t != NONE && !terms.contains(&t) {
                terms.push(t);
            }
        }
        let bmx = self.bmx.top(&terms, self.k);
        let (mut ja, mut scale, mut lam) = (false, self.scale, self.lam);
        let ids = self.ids(text);
        let q = match &self.enc {
            Reader::G6 { eng, .. } => eng.query(&ids),
            // 日本語の入力は 2 つ目の層で読む（ある一式だけ）。それ以外は、もとの層
            Reader::Mb(base) => {
                ja = self.alt.as_ref().is_some_and(|a| is_ja(lex, a.share));
                match &self.alt {
                    Some(a) if ja => {
                        (scale, lam) = (a.scale, a.lam);
                        a.enc.query(&ids)
                    }
                    _ => base.query(&ids),
                }
            }
        };
        // 段ごとの点 → 対数確率。節の BMX の点は、その下の葉の点の最大（無ければ 0）
        let total = self.node_s.len();
        let (mut s, mut bm) = (vec![0f32; total], vec![0f32; total]);
        let mut at = 0;
        let mut offs = [0usize; 5];
        for t in 0..5 {
            let n = self.n_nodes[t];
            offs[t] = at;
            if self.node4.is_empty() {
                enc::dots(&q[t * D..][..D], &self.node[at * D..][..n * D], &mut s[at..at + n]);
            } else {
                enc_g::dots4(&q[t * D..][..D], &self.node4[at * D / 2..][..n * D / 2], &mut s[at..at + n]);
            }
            for &(leaf, v) in &bmx {
                let j = at + self.leaf_path[leaf as usize * 5 + t] as usize;
                bm[j] = bm[j].max(v);
            }
            let mut mx = f32::NEG_INFINITY;
            for j in at..at + n {
                s[j] = scale[t] * (s[j] * self.node_s[j]) + lam[t] * bm[j];
                mx = mx.max(s[j]);
            }
            let lse = mx as f64 + s[at..at + n].iter().map(|&x| ((x - mx) as f64).exp()).sum::<f64>().ln();
            for x in &mut s[at..at + n] {
                *x = (*x as f64 - lse) as f32;
            }
            at += n;
        }
        // 葉の点 = 道の節の対数確率の和。上位 k 件（点の高い順、同じ点は番号の小さい順）
        // prefix があれば、その頭の桁の葉だけ（質問に答えたあとの呼び直し）
        let leaf: Vec<f32> = self.leaf_path.chunks_exact(5).enumerate().map(|(i, p)| if self.leaves[i * 10..].starts_with(prefix) { (0..5).fold(0f32, |a, t| a + s[offs[t] + p[t] as usize]) } else { f32::NEG_INFINITY }).collect();
        let top = pick(&leaf, k.max(QN), |_| true); // 質問を作るために、応答に出す k 件より広く取る（応答の欄は、呼び手の側で頭の k 件だけを使う）
        Pred { ids, bmx, top, leaf, ja }
    }
}

impl Model {
    fn code(&self, leaf: usize) -> &str {
        std::str::from_utf8(&self.leaves[leaf * 10..][..10]).unwrap_or("")
    }

    // 段 t の節の公式の文（親から先の分）
    fn label(&self, leaf: usize, t: usize) -> &str {
        let at: usize = self.n_nodes[..t].iter().sum();
        self.labels.get(at + self.leaf_path[leaf * 5 + t] as usize).copied().unwrap_or("")
    }

    // 葉の公式の文（類から 10 桁までの文をつないだもの）
    fn description(&self, leaf: usize) -> String {
        (0..5).map(|t| self.label(leaf, t)).filter(|x| !x.is_empty()).collect::<Vec<_>>().join(" > ")
    }

    // 葉の税率の欄（関税表の基本の税率。第 99 類の追加の関税・脚注は入らない）。古い一式では null
    fn duty_rate(&self, leaf: usize) -> Value {
        let Some(line) = self.leaf_duty.get(leaf).and_then(|&i| self.duty.get(i as usize)) else { return Value::Null };
        let mut c = line.split('\t');
        json!({ "general": c.next().unwrap_or(""), "special": c.next().unwrap_or(""), "other": c.next().unwrap_or("") })
    }

    // 上位の候補 → 応答の欄（hsCode・sixDigit・confidenceScore・sixDigitConfidence・dutyRate・alternates・questions）。
    // 確率（probabilityMass）は、候補の葉の点の softmax（temp で割る）。fixed は、呼び手がもう答えた段の数（prefix の桁 ÷ 2）
    // wide は、質問を作る広い候補（top を頭に含む、点の高い順の QN 葉）。leaf は、葉の点の全部
    // confirm は、自信のある段も飛ばさずに、上の段から順に聞く（呼び手が入力に "confirm": true を足したとき）
    fn respond(&self, top: &[(f32, usize)], wide: &[(f32, usize)], leaf: &[f32], fixed: usize, ja: bool, confirm: bool, o: &mut Value) {
        let Some(&(best, first)) = top.first() else { return };
        let w: Vec<f64> = top.iter().map(|x| ((x.0 as f64 - best as f64) / self.temp).exp()).collect();
        let sum: f64 = w.iter().sum();
        let p: Vec<f64> = w.iter().map(|x| x / sum).collect();
        // 段 t の確度 = 1 位の段 t までが正しい確率: σ(a + b·logit(m) + c·1 位の点 + d·答えた段の数)。m は、候補の確率のうち 1 位と同じ節の分（係数を合わせたときと同じ式）。
        // 上の桁の確度は、下の桁より低くしない。呼び手が答えた段は 1（答えは正しいとして出す）
        let calib = if ja { self.alt.as_ref().and_then(|a| a.calib) } else { self.calib };
        let conf = calib.map(|c| {
            let (c, mut v) = (c[(fixed > 0) as usize], [0f64; 5]);
            for t in (0..5).rev() {
                let m: f64 = top.iter().zip(&p).filter(|(x, _)| self.leaf_path[x.1 * 5 + t] == self.leaf_path[first * 5 + t]).map(|(_, q)| q).sum();
                let m = m.clamp(1e-3, 1.0 - 1e-3);
                let x = 1.0 / (1.0 + (-(c[t][0] + c[t][1] * (m.ln() - (-m).ln_1p()) + c[t][2] * best as f64 + c[t][3] * fixed as f64)).exp());
                v[t] = if t < 4 { x.max(v[t + 1]) } else { x };
            }
            v.iter_mut().take(fixed).for_each(|x| *x = 1.0);
            v
        });
        o["hsCode"] = json!({ "code": self.code(first), "description": self.description(first) });
        o["sixDigit"] = json!(&self.code(first)[..6]);
        o["confidenceScore"] = json!(conf.map_or(p[0], |v| v[4]));
        if let Some(v) = conf {
            o["sixDigitConfidence"] = json!(v[2]);
        }
        if !self.duty.is_empty() {
            o["dutyRate"] = self.duty_rate(first);
        }
        o["alternates"] = top.iter().zip(&p).skip(1).map(|(x, q)| json!({ "code": self.code(x.1), "description": self.description(x.1), "probabilityMass": q, "dutyRate": self.duty_rate(x.1) })).collect();
        // 質問: 確度が tau より低い最初の段 t0 を聞く（自信のある段は飛ばす）。候補は、1 位と t0 の上まで同じ葉のうち、点の高い QN 葉。その候補の頭の桁が分かれる最初の段が質問になる。
        // どの段も確度が tau 以上のとき（と、合わせ込みの無い一式）は、広い候補（上位 QN 葉）が分かれる最初の段（聞く目安は、その段の確度で決まる）。
        // 選択肢は、点の高い候補の順。確率は、候補の中での割合
        let split = |pool: &[(f32, usize)], from: usize| (from..5).find(|&t| pool.iter().any(|x| self.leaf_path[x.1 * 5 + t] != self.leaf_path[first * 5 + t]));
        let focus = conf.filter(|_| !confirm).and_then(|v| (fixed.min(5)..5).find(|&t| v[t] < self.tau)).and_then(|t0| {
            let pool = if t0 == 0 { wide.to_vec() } else { pick(leaf, QN, |i| self.leaf_path[i * 5 + t0 - 1] == self.leaf_path[first * 5 + t0 - 1]) };
            split(&pool, t0).map(|t| (t, pool))
        });
        let mut qs = Vec::new();
        if let Some((t, wide)) = focus.or_else(|| split(wide, 0).map(|t| (t, wide.to_vec()))) {
            let d = 2 * t + 2;
            let ww: Vec<f64> = wide.iter().map(|x| ((x.0 as f64 - best as f64) / self.temp).exp()).collect();
            let wsum: f64 = ww.iter().sum();
            let pw: Vec<f64> = ww.iter().map(|x| x / wsum).collect();
            let mut opts: Vec<(usize, f64)> = Vec::new(); // (その選択肢で一番点の高い葉, 確率の合計)
            for (x, q) in wide.iter().zip(&pw) {
                match opts.iter_mut().find(|y| self.leaf_path[y.0 * 5 + t] == self.leaf_path[x.1 * 5 + t]) {
                    Some(y) => y.1 += q,
                    None => opts.push((x.1, *q)),
                }
            }
            let options: Vec<Value> = opts.iter().map(|&(leaf, q)| json!({ "prefix": &self.code(leaf)[..d], "label": self.label(leaf, t), "code": self.code(leaf), "probabilityMass": q, "dutyRate": self.duty_rate(leaf) })).collect();
            // 聞く目安: 1 位の選択肢が正しい確率（確度。合わせ込みが無い一式では、確率の合計）が tau より低い
            let c = conf.map_or(opts[0].1, |v| v[t]);
            qs.push(json!({ "digits": d, "ask": c < self.tau, "confidence": c, "options": options }));
        }
        o["questions"] = json!(qs);
    }
}

// 入力の 1 件 → (モデルに渡す文, 語彙の点に使う文)。text があればそのまま。無ければ name・material・categories・description から組み立てる
fn build(v: &Value) -> Result<(String, String)> {
    let field = |k: &str| v[k].as_str().map(|x| x.trim().to_string()).filter(|x| !x.is_empty());
    if let Some(text) = v["text"].as_str() {
        return Ok((text.to_string(), v["lex"].as_str().unwrap_or(text).to_string()));
    }
    let cats = match &v["categories"] {
        Value::Array(a) => Some(a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(" ")).filter(|x| !x.is_empty()),
        _ => field("categories"),
    };
    let name = field("name").context("text か name が要る")?;
    let (mut text, mut lex) = (format!("title: {name}"), name);
    for (k, x) in [("material", field("material")), ("category", cats), ("description", field("description"))] {
        if let Some(x) = x {
            text += &format!(" | {k}: {x}");
            lex += " ";
            lex += &x;
        }
    }
    Ok((text, lex))
}

// /proc/self/status の (VmRSS, VmHWM, RssAnon（kB）, Threads)
#[cfg(not(target_arch = "wasm32"))]
fn status() -> [u64; 4] {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    ["VmRSS:", "VmHWM:", "RssAnon:", "Threads:"].map(|k| s.lines().find_map(|l| l.strip_prefix(k)).and_then(|v| v.split_whitespace().next()?.parse().ok()).unwrap_or(0))
}

// 1 件の入力（JSON の 1 行）→ 応答（JSON の 1 行）
fn answer(m: &mut Model, line: &str, k: usize, debug: bool) -> Result<String> {
    let v: Value = serde_json::from_str(line)?;
    let (text, lex) = build(&v)?;
    let prefix = v["prefix"].as_str().unwrap_or("");
    if !matches!(prefix.len(), 0 | 2 | 4 | 6 | 8 | 10) || !prefix.bytes().all(|c| c.is_ascii_digit()) {
        bail!("prefix は 2・4・6・8・10 桁の数字");
    }
    let p = m.predict(&text, &lex, k, prefix.as_bytes());
    let head = &p.top[..k.min(p.top.len())]; // 応答に出すのは上位 k 件。質問は、広い候補（p.top の全部）から作る
    let codes: Vec<&str> = head.iter().map(|&(_, i)| m.code(i)).collect();
    let mut o = json!({ "codes": codes, "scores": head.iter().map(|x| x.0).collect::<Vec<_>>(), "htsRelease": m.release });
    m.respond(head, &p.top, &p.leaf, prefix.len() / 2, p.ja, v["confirm"].as_bool().unwrap_or(false), &mut o);
    if p.ja {
        o["reader"] = json!("ja"); // 2 つ目の層（日本語）で読んだ印。もとの層で読んだ応答には、この欄を足さない（前の版と同じ形のまま）
    }
    if debug {
        o["ids"] = json!(p.ids);
        o["bmx"] = json!(p.bmx);
        o["norm"] = json!(m.nfkc.lex_norm(&lex));
        o["path"] = json!(enc::path());
    }
    Ok(o.to_string())
}

// WebAssembly の口。呼び手の手順:
//   1. zb_alloc で場所を 2 つ取り、目録（zeiban.json）と一式（zeiban.bin）の中身を書く
//   2. zb_load（0 なら成功。一式の場所は、そのまま持ち続けるので、解放しない。目録の場所は zb_free で返してよい）
//   3. 1 件ごとに: 入力の JSON を zb_alloc した場所に書く → zb_classify（0 なら成功）→ zb_out_ptr・zb_out_len の場所から応答（失敗のときは理由の文）を読む → 入力の場所を zb_free
// 応答の場所は、次に zb_load・zb_classify を呼ぶまで有効。呼び出しは 1 本のスレッドから順に行うこと。
#[cfg(target_arch = "wasm32")]
#[allow(static_mut_refs)]
mod wasm {
    use super::*;
    use std::alloc::{alloc, dealloc, Layout};

    static mut MODEL: Option<Model> = None;
    static mut OUT: Vec<u8> = Vec::new();

    // 一式の区画は 64 バイトの境界に置いてあるので、場所もその境界で取る
    fn layout(n: usize) -> Layout {
        Layout::from_size_align(n.max(1), 64).unwrap()
    }

    #[no_mangle]
    pub extern "C" fn zb_alloc(n: usize) -> *mut u8 {
        unsafe { alloc(layout(n)) }
    }

    /// # Safety
    /// p は、同じ n で zb_alloc が返した場所であること。
    #[no_mangle]
    pub unsafe extern "C" fn zb_free(p: *mut u8, n: usize) {
        dealloc(p, layout(n))
    }

    /// # Safety
    /// man・bin は、zb_alloc で取って中身を書いた場所であること。bin は、このあと解放しないこと。
    #[no_mangle]
    pub unsafe extern "C" fn zb_load(man: *const u8, man_len: usize, bin: *const u8, bin_len: usize) -> i32 {
        let load = || -> Result<Model> {
            let man: Value = serde_json::from_slice(std::slice::from_raw_parts(man, man_len))?;
            Model::from_bundle(Bundle::from_parts(man, std::slice::from_raw_parts(bin, bin_len))?)
        };
        match load() {
            Ok(m) => {
                MODEL = Some(m);
                OUT = Vec::new();
                0
            }
            Err(e) => {
                OUT = format!("{e:#}").into_bytes();
                1
            }
        }
    }

    /// # Safety
    /// req は、zb_alloc で取って入力の JSON（UTF-8）を書いた場所であること。
    #[no_mangle]
    pub unsafe extern "C" fn zb_classify(req: *const u8, len: usize, k: usize) -> i32 {
        let res = std::str::from_utf8(std::slice::from_raw_parts(req, len)).context("入力が UTF-8 でない").and_then(|line| answer(MODEL.as_mut().context("zb_load がまだ")?, line, k.max(1), false));
        let (code, s) = match res {
            Ok(s) => (0, s),
            Err(e) => (1, format!("{e:#}")),
        };
        OUT = s.into_bytes();
        code
    }

    #[no_mangle]
    pub extern "C" fn zb_out_ptr() -> *const u8 {
        unsafe { OUT.as_ptr() }
    }

    #[no_mangle]
    pub extern "C" fn zb_out_len() -> usize {
        unsafe { OUT.len() }
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn run() -> Result<()> {
    let t0 = Instant::now();
    let mut args = std::env::args().skip(1);
    let dir = args.next().context("使い方: zeiban <一式のフォルダ> [k | debug | bench]")?;
    let mode = args.next().unwrap_or_default();
    let k = mode.parse().unwrap_or(5);
    let mut m = Model::load(Path::new(&dir))?;
    let (load_ms, st_load) = (t0.elapsed().as_secs_f64() * 1e3, status());
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    let mut reqs = Vec::new();
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if mode == "bench" {
            reqs.push(build(&serde_json::from_str(&line)?)?);
            continue;
        }
        if mode == "ids" {
            // 語の区切りだけ（モデルは動かさない）: モデルに渡すトークンの番号を出す（Python の基準との照合用）
            let (text, _) = build(&serde_json::from_str(&line)?)?;
            writeln!(out, "{}", json!(m.ids(&text)))?;
            continue;
        }
        let o = answer(&mut m, &line, k, mode == "debug")?;
        writeln!(out, "{o}")?;
        out.flush()?;
    }
    if mode == "bench" {
        let mut items = Vec::with_capacity(reqs.len());
        for (text, lex) in &reqs {
            let t = Instant::now();
            let p = m.predict(text, lex, k, b"");
            items.push((p.ids.len(), t.elapsed().as_micros() as u64));
        }
        writeln!(out, "{}", json!({ "load_ms": load_ms, "rss_load_kb": st_load[..3], "rss_end_kb": status()[..3], "threads": st_load[3], "path": enc::path(), "items": items }))?;
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<()> {
    rayon::ThreadPoolBuilder::new().num_threads(2).build()?.install(run)
}
