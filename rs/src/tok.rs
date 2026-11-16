// 語の区切り（SentencePiece の Unigram）と、語彙の点の前処理（NFKC・小文字）。語の区切りは、前に書いた実装から写して、
// 語彙を tokenizer.json ではなく一式の区画から読む形にした（片の文字列はマップしたファイルを指す）。HF tokenizers 0.22 と同じトークンの番号を作る:
// - 追加トークン（<s> など）を本文から先に切り出す（最左・最長）。残りは半角空白を ▁ に置き換え、Unigram の Viterbi で分ける（同点は先に見つけた方）。
// - 1 文字の語が無い位置は未知語（点 = 最小の点 − 10）。続いた未知語は 1 つにまとめ、UTF-8 のバイトのトークン <0xXX> に置き換える。
// 対応する設定（normalizer なし・Metaspace (never, split なし)・Unigram・byte_fallback）であることは、一式を作る台本が確かめている。
// NFKC は、一式に入れた表（Python の unicodedata から作ったもの）で作る（学習のときの Python の前処理と同じ文にするため）。
use crate::Bundle;
use anyhow::{Context, Result};
use std::collections::HashMap;

pub struct Tokenizer {
    piece_ids: HashMap<&'static str, u32>,
    scores: &'static [f64],
    max_piece_chars: usize,
    unk: u32,
    unk_score: f64,
    bytes: Vec<u32>,
    added: Vec<(String, u32)>,
    pub bos: u32,
    pub eos: u32,
}

impl Tokenizer {
    pub fn new(b: &'static Bundle) -> Result<Self> {
        let (blob, off, scores) = (b.get::<u8>("tok.pieces"), b.get::<u32>("tok.off"), b.get::<f64>("tok.score"));
        let mut piece_ids = HashMap::with_capacity(scores.len());
        let mut max_piece_chars = 1;
        for i in 0..scores.len() {
            let p = std::str::from_utf8(&blob[off[i] as usize..off[i + 1] as usize])?;
            max_piece_chars = max_piece_chars.max(p.chars().count());
            piece_ids.insert(p, i as u32);
        }
        let bytes = (0..=255u8).map(|x| piece_ids.get(format!("<0x{x:02X}>").as_str()).copied()).collect::<Option<Vec<_>>>().context("バイトの片が語彙にない")?;
        let t = &b.man["tok"];
        let id = |k: &str| t[k].as_u64().map(|x| x as u32).with_context(|| format!("目録の tok.{k}"));
        let mut added = Vec::new();
        for a in t["added"].as_array().context("目録の tok.added")? {
            added.push((a[0].as_str().context("tok.added")?.to_string(), a[1].as_u64().context("tok.added")? as u32));
        }
        Ok(Self {
            piece_ids,
            scores,
            max_piece_chars,
            unk: id("unk")?,
            unk_score: scores.iter().copied().fold(f64::INFINITY, f64::min) - 10.0,
            bytes,
            added,
            bos: id("bos")?,
            eos: id("eos")?,
        })
    }

    // 1 つの文をトークンの番号の並びにする（頭と尻の印は付けない）
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let (mut seg, mut i) = (0, 0);
        while i < text.len() {
            let hit = self.added.iter().filter(|(p, _)| text[i..].starts_with(p.as_str())).max_by_key(|(p, _)| p.len());
            if let Some((p, id)) = hit {
                self.unigram(&text[seg..i].replace(' ', "▁"), &mut out);
                out.push(*id);
                i += p.len();
                seg = i;
            } else {
                i += text[i..].chars().next().map_or(1, char::len_utf8);
            }
        }
        self.unigram(&text[seg..].replace(' ', "▁"), &mut out);
        out
    }

    fn unigram(&self, s: &str, out: &mut Vec<u32>) {
        // best[e] = (位置 e で終わる最良の分け方の点, その最後の語の始まり, 語の番号)。始まりが usize::MAX なら未設定
        let mut best = vec![(0f64, usize::MAX, 0u32); s.len() + 1];
        let mut start = 0;
        while start < s.len() {
            let base = best[start].0;
            let first = s[start..].chars().next().map_or(1, char::len_utf8);
            let mut single = false;
            for (k, c) in s[start..].char_indices().take(self.max_piece_chars) {
                let end = start + k + c.len_utf8();
                if let Some(&id) = self.piece_ids.get(&s[start..end]) {
                    let score = base + self.scores[id as usize];
                    if best[end].1 == usize::MAX || score > best[end].0 {
                        best[end] = (score, start, id);
                    }
                    single |= end - start == first;
                }
            }
            if !single {
                let end = start + first;
                let score = base + self.unk_score;
                if best[end].1 == usize::MAX || score > best[end].0 {
                    best[end] = (score, start, self.unk);
                }
            }
            start += first;
        }
        // 後ろからたどる。続いた未知語は 1 つにまとめてからバイトのトークンにする（HF の fuse_unk と byte_fallback）
        let mut pieces = Vec::new();
        let mut end = s.len();
        while end > 0 {
            let (_, st, id) = best[end];
            match pieces.last_mut() {
                Some((pst, _, pid)) if id == self.unk && *pid == self.unk => *pst = st,
                _ => pieces.push((st, end, id)),
            }
            end = st;
        }
        for &(st, en, id) in pieces.iter().rev() {
            if id == self.unk {
                out.extend(s[st..en].bytes().map(|b| self.bytes[b as usize]));
            } else {
                out.push(id);
            }
        }
    }
}

const HANGUL: u32 = 0xAC00; // ハングルの音節（11,172 字 = 初声 19 × 中声 21 × 終声 28）は式で分解・合成する

pub struct Nfkc {
    dec_cp: &'static [u32],
    dec_off: &'static [u32],
    dec_seq: &'static [u32],
    ccc: &'static [u32],  // 字 << 8 | 結合クラス（0 でない字だけ。字の順）
    comp: &'static [u32], // (1 字目, 2 字目, 合成した字) の並び（1 字目・2 字目の順）
}

impl Nfkc {
    pub fn new(b: &'static Bundle) -> Self {
        Self { dec_cp: b.get("nfkc.dec_cp"), dec_off: b.get("nfkc.dec_off"), dec_seq: b.get("nfkc.dec_seq"), ccc: b.get("nfkc.ccc"), comp: b.get("nfkc.comp") }
    }

    fn class(&self, c: u32) -> u8 {
        match self.ccc.get(self.ccc.partition_point(|&e| e >> 8 < c)) {
            Some(&e) if e >> 8 == c => e as u8,
            _ => 0,
        }
    }

    fn compose(&self, a: u32, b: u32) -> Option<u32> {
        if (0x1100..0x1113).contains(&a) && (0x1161..0x1176).contains(&b) {
            return Some(HANGUL + ((a - 0x1100) * 21 + (b - 0x1161)) * 28);
        }
        if (HANGUL..HANGUL + 11172).contains(&a) && (a - HANGUL) % 28 == 0 && (0x11A8..0x11C3).contains(&b) {
            return Some(a + (b - 0x11A7));
        }
        let key = |i: usize| (self.comp[i * 3], self.comp[i * 3 + 1]);
        let (mut lo, mut hi) = (0, self.comp.len() / 3);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if key(mid) < (a, b) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        (lo < self.comp.len() / 3 && key(lo) == (a, b)).then(|| self.comp[lo * 3 + 2])
    }

    pub fn nfkc(&self, s: &str) -> String {
        if s.is_ascii() {
            return s.to_string();
        }
        // 分解（互換）
        let mut buf: Vec<(u32, u8)> = Vec::with_capacity(s.len());
        for c in s.chars().map(|c| c as u32) {
            if (HANGUL..HANGUL + 11172).contains(&c) {
                let si = c - HANGUL;
                buf.push((0x1100 + si / 588, 0));
                buf.push((0x1161 + si % 588 / 28, 0));
                if si % 28 != 0 {
                    buf.push((0x11A7 + si % 28, 0));
                }
            } else if let Ok(i) = self.dec_cp.binary_search(&c) {
                buf.extend(self.dec_seq[self.dec_off[i] as usize..self.dec_off[i + 1] as usize].iter().map(|&d| (d, self.class(d))));
            } else {
                buf.push((c, self.class(c)));
            }
        }
        // 結合文字の並びを、結合クラスの順にそろえる（同じクラスは元の順）
        let mut i = 0;
        while i < buf.len() {
            let j = i + buf[i..].iter().take_while(|x| x.1 != 0).count();
            buf[i..j].sort_by_key(|x| x.1);
            i = j + 1;
        }
        // 合成。直前の starter（結合クラス 0 の字）と合成できる字を合成する。間に、結合クラスが同じか大きい字があれば合成しない
        let mut out: Vec<u32> = Vec::with_capacity(buf.len());
        let (mut starter, mut last) = (None, 0u8);
        for &(c, class) in &buf {
            if let Some(sp) = starter {
                if out.len() == sp + 1 || last < class {
                    if let Some(p) = self.compose(out[sp], c) {
                        out[sp] = p;
                        continue;
                    }
                }
            }
            if class == 0 {
                starter = Some(out.len());
            }
            last = class;
            out.push(c);
        }
        out.into_iter().filter_map(char::from_u32).collect()
    }

    // 語彙の点に使う文（学習のときの Python と同じ前処理）: NFKC → 小文字 → 空白で分けて半角空白 1 つでつなぎ、先頭に空白を足す
    pub fn lex_norm(&self, text: &str) -> String {
        let t = self.nfkc(text).to_lowercase();
        let mut out = String::with_capacity(t.len() + 1);
        for w in t.split(|c: char| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)).filter(|w| !w.is_empty()) {
            // Python の str.split() は U+001C〜001F も空白に数える
            out.push(' ');
            out.push_str(w);
        }
        if out.is_empty() {
            out.push(' ');
        }
        out
    }
}
