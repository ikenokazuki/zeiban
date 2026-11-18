// 先頭 6 層の読み手（目録の enc.kind が g6。embeddinggemma-2 の文の部分の先頭 6 層: 幅 512・中間 2048）の CPU 推論。
// 重みは、一式の int4（1 個 4 ビット。値 −7〜7 に 8 を足した 1〜15）のまま使う（戻した写しを持たない。持つと常駐メモリが約 35 MB 増える）。
// 線形は、マップした int4 を掛ける所で fp32 に直し、出力ごとの目盛りを最後に掛ける。計算の中身は、学習のときの Python の実装（transformers）と同じ:
//   語の表 × √512 → 層ごとの入力（射影 × 512^-0.5 → 512 ごとに RMSNorm）→ 層 → RMSNorm → 射影（512 → 768）→ トークンの平均 → 問いの射影（5 段 × 256）
//   1 層: RMSNorm → 注意（問い 4 頭。問いと鍵に RMSNorm（重みあり）・値に RMSNorm（重みなし）・回転の位置づけ・点の倍率 1）→ RMSNorm → 足す
//         → RMSNorm → MLP（gelu の tanh 版(門) × 上げる射影 → 下げる射影）→ RMSNorm → 足す → 層ごとの入力の足し込み（gelu(門) × 層ごとの入力 → 射影 → RMSNorm → 足す）→ 層の倍率
//   最後の層だけ「頭の幅 512・鍵と値は 1 頭・θ = 1,000,000」。ほかは「頭の幅 256・鍵と値は 2 頭（問いの 2 頭ずつが共有）・θ = 10,000」。
// 行（トークン）を 2 つに分けて rayon::join で 2 スレッドに配る（注意は、鍵と値は全部の行、出力の行だけを分ける）。
// 計算の核は 2 通り（enc.rs と同じ分け方）: avx（x86-64 で AVX2 と FMA があるとき）と portable（素の Rust。どの CPU でも動く）。
use crate::enc::{avx_on, D};
use crate::Bundle;
use anyhow::{bail, Context, Result};

const H: usize = 512;
const HEADS: usize = 4;
const FF: usize = 2048;
const STATE: usize = 768;
const EPS: f32 = 1e-6;
const MAX_LEN: usize = 256;

// int4 の線形（バイアス無し）。重みは [16 出力ごとのかたまり][入力][8 バイト]（入力 1 つごとに、かたまりの 16 個の重み。下の 4 ビットが先）と、出力ごとの目盛り
pub struct P4 {
    inp: usize,
    blocks: usize,
    p: &'static [u8],
    s: &'static [f32],
}

impl P4 {
    fn load(b: &'static Bundle, name: &str, out: usize, inp: usize) -> Result<Self> {
        let (p, s) = (b.get::<u8>(name), b.get::<f32>(&format!("{name}.s")));
        if p.len() != out * inp / 2 || s.len() != out || out % 16 != 0 {
            bail!("{name}: 形が [{out}, {inp}] でない");
        }
        Ok(Self { inp, blocks: out / 16, p, s })
    }
}

macro_rules! kernels {
    ($($(#[$m:meta])* $v:vis fn $f:ident($($a:ident: $t:ty),*);)*) => {$(
        $(#[$m])*
        $v fn $f($($a: $t),*) {
            #[cfg(target_arch = "x86_64")]
            if avx_on() {
                // SAFETY: avx_on は、CPU に AVX2 と FMA があるときだけ真
                return unsafe { avx::$f($($a),*) };
            }
            portable::$f($($a),*)
        }
    )*};
}

kernels! {
    /// y += x W^T。x: [n][inp]、y: [n][out]
    fn linear(p: &P4, x: &[f32], n: usize, y: &mut [f32]);
    /// RMSNorm: 行（dim 個）ごとに、二乗の平均の平方根で割る。w があれば掛ける。x・out: [行][dim]
    fn rms(x: &[f32], w: &[f32], dim: usize, out: &mut [f32]);
    /// out = gelu(a) * b（gelu は tanh 版: x/2 * (1 + tanh(√(2/π) (x + 0.044715 x³)))）
    fn gelu_mul(a: &[f32], b: &[f32], out: &mut [f32]);
    /// 位置 from からの out.len() / (HEADS * hd) 行の注意の出力。q: [HEADS][位置][hd]、k・v: [kv][位置][hd]（q・k は回転済み）。問いの頭 h は、鍵と値の頭 h / (HEADS / kv) を見る
    fn attend(q: &[f32], k: &[f32], v: &[f32], n: usize, hd: usize, kv: usize, from: usize, out: &mut [f32]);
    /// 問いのベクトル q [D] と、int4 の節のベクトル node [行][D / 2 バイト] の内積（行ごと）。out: [行]
    pub fn dots4(q: &[f32], node: &[u8], out: &mut [f32]);
}

// どの CPU でも動く核。和は 8 本に分けて足す（enc.rs の portable と同じ考え方）
mod portable {
    use super::{EPS, HEADS, P4};

    fn hsum(s: [f32; 8]) -> f32 {
        ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7]))
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        let mut s = [0f32; 8];
        for (x, y) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
            for j in 0..8 {
                s[j] += x[j] * y[j];
            }
        }
        hsum(s)
    }

    // 8 バイト = int4 の 16 個 → fp32
    #[inline]
    pub fn w16(b: &[u8]) -> [f32; 16] {
        const M: u64 = 0x0f0f_0f0f_0f0f_0f0f;
        let v = u64::from_le_bytes(b[..8].try_into().unwrap());
        let (lo, hi) = ((v & M).to_le_bytes(), ((v >> 4) & M).to_le_bytes());
        let mut w = [0f32; 16];
        for j in 0..8 {
            w[2 * j] = lo[j] as f32 - 8.0;
            w[2 * j + 1] = hi[j] as f32 - 8.0;
        }
        w
    }

    const MC: usize = 48; // 一度に扱う行（x の 48 行を L2 に載せる）
    const RN: usize = 2; // まとめて掛ける行の数。4 と 8 も測った（WebAssembly・portable とも 2 がいちばん速い。差は 1 割以内）

    // R 行 × 16 出力: y += (Σ_l x_l × 重み_l) × 目盛り。wf は、かたまり 1 つぶんの重みを fp32 に直したもの [入力][16]
    #[inline]
    fn kern<const R: usize>(wf: &[f32], x: &[f32], inp: usize, s: &[f32], y: &mut [f32], out: usize) {
        let mut a = [[0f32; 16]; R];
        for (l, w) in wf.chunks_exact(16).enumerate() {
            for r in 0..R {
                let xv = x[r * inp + l];
                for c in 0..16 {
                    a[r][c] += xv * w[c];
                }
            }
        }
        for r in 0..R {
            for (c, yv) in y[r * out..][..16].iter_mut().enumerate() {
                *yv += a[r][c] * s[c];
            }
        }
    }

    // 重みを fp32 に直すのは、かたまり 1 つ × 行 MC 個で 1 回（行ごとに直すと、直す手間が掛け算と同じくらいかかる）
    pub fn linear(p: &P4, x: &[f32], n: usize, y: &mut [f32]) {
        let (inp, out) = (p.inp, p.blocks * 16);
        assert!(x.len() == n * inp && y.len() == n * out);
        let mut wf = vec![0f32; inp * 16];
        for r0 in (0..n).step_by(MC) {
            let r1 = (r0 + MC).min(n);
            for (b, w) in p.p.chunks_exact(inp * 8).enumerate() {
                for (dst, wl) in wf.chunks_exact_mut(16).zip(w.chunks_exact(8)) {
                    dst.copy_from_slice(&w16(wl));
                }
                let s = &p.s[b * 16..][..16];
                let mut r = r0;
                while r + RN <= r1 {
                    kern::<RN>(&wf, &x[r * inp..], inp, s, &mut y[r * out + b * 16..], out);
                    r += RN;
                }
                while r < r1 {
                    kern::<1>(&wf, &x[r * inp..], inp, s, &mut y[r * out + b * 16..], out);
                    r += 1;
                }
            }
        }
    }

    pub fn rms(x: &[f32], w: &[f32], dim: usize, out: &mut [f32]) {
        assert!(x.len() == out.len() && x.len() % dim == 0 && (w.is_empty() || w.len() == dim));
        for (r, y) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
            let sc = 1.0 / (dot(r, r) / dim as f32 + EPS).sqrt();
            if w.is_empty() {
                for (a, yv) in r.iter().zip(y) {
                    *yv = a * sc;
                }
            } else {
                for ((a, wv), yv) in r.iter().zip(w).zip(y) {
                    *yv = a * sc * wv;
                }
            }
        }
    }

    pub fn gelu(x: f32) -> f32 {
        0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
    }

    pub fn gelu_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
        for ((x, y), o) in a.iter().zip(b).zip(out) {
            *o = gelu(*x) * y;
        }
    }

    pub fn attend(q: &[f32], k: &[f32], v: &[f32], n: usize, hd: usize, kv: usize, from: usize, out: &mut [f32]) {
        let mut buf = vec![0f32; n];
        for (r, o) in out.chunks_exact_mut(HEADS * hd).enumerate() {
            let i = from + r;
            for h in 0..HEADS {
                let g = h / (HEADS / kv);
                let (kh, vh) = (&k[g * n * hd..][..n * hd], &v[g * n * hd..][..n * hd]);
                let qi = &q[(h * n + i) * hd..][..hd];
                // 点 q_i · k_j（倍率 1）→ softmax
                for (sj, kj) in buf.iter_mut().zip(kh.chunks_exact(hd)) {
                    *sj = dot(qi, kj);
                }
                let max = buf.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0f32;
                for sj in buf.iter_mut() {
                    *sj = (*sj - max).exp();
                    sum += *sj;
                }
                let isum = 1.0 / sum;
                // 混ぜる: out_i = Σ_j p_j v_j
                let acc = &mut o[h * hd..][..hd];
                acc.fill(0.0);
                for (sj, vj) in buf.iter().zip(vh.chunks_exact(hd)) {
                    let p = sj * isum;
                    if p > 0.0 {
                        for (a, x) in acc.iter_mut().zip(vj) {
                            *a += p * x;
                        }
                    }
                }
            }
        }
    }

    pub fn dots4(q: &[f32], node: &[u8], out: &mut [f32]) {
        let d = q.len();
        assert!(d % 16 == 0 && node.len() == out.len() * d / 2);
        for (r, o) in node.chunks_exact(d / 2).zip(out) {
            let mut s = [0f32; 16];
            for (x, y) in q.chunks_exact(16).zip(r.chunks_exact(8)) {
                let w = w16(y);
                for j in 0..16 {
                    s[j] += x[j] * w[j];
                }
            }
            *o = hsum([s[0] + s[8], s[1] + s[9], s[2] + s[10], s[3] + s[11], s[4] + s[12], s[5] + s[13], s[6] + s[14], s[7] + s[15]]);
        }
    }
}

// AVX2 と FMA の核。呼ぶ側（kernels!）が、CPU に AVX2 と FMA があることを確かめている
#[cfg(target_arch = "x86_64")]
mod avx {
    use super::{EPS, HEADS, P4};
    use std::arch::x86_64::*;

    const MC: usize = 96; // 外側で一度に扱う行（x の 96 行を L2 に載せる）
    const KC: usize = 256; // 入力の次元を分ける単位（重みの 16 x 256 を L1 に載せる）

    // 8 バイト = int4 の 16 個 → fp32 の 8 個 × 2
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn w16(p: *const u8) -> (__m256, __m256) {
        let v = _mm_loadl_epi64(p as *const __m128i);
        let m = _mm_set1_epi8(0x0f);
        let b = _mm_sub_epi8(_mm_unpacklo_epi8(_mm_and_si128(v, m), _mm_and_si128(_mm_srli_epi16(v, 4), m)), _mm_set1_epi8(8));
        (_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b)), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b, 8))))
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn linear(p: &P4, x: &[f32], n: usize, y: &mut [f32]) {
        let (inp, out, xp, yp) = (p.inp, p.blocks * 16, x.as_ptr(), y.as_mut_ptr());
        assert!(x.len() == n * inp && y.len() == n * out);
        for k0 in (0..inp).step_by(KC) {
            let kc = KC.min(inp - k0);
            for r0 in (0..n).step_by(MC) {
                let r1 = (r0 + MC).min(n);
                for b in 0..p.blocks {
                    let (w, s) = (p.p.as_ptr().add((b * inp + k0) * 8), p.s.as_ptr().add(b * 16));
                    let mut r = r0;
                    while r + 6 <= r1 {
                        kern6(kc, xp.add(r * inp + k0), inp, w, s, yp.add(r * out + b * 16), out);
                        r += 6;
                    }
                    while r < r1 {
                        kern1(kc, xp.add(r * inp + k0), w, s, yp.add(r * out + b * 16));
                        r += 1;
                    }
                }
            }
        }
    }

    // 6 行 × 16 出力: y += (Σ_l x_l × 整数の重み_l) × 目盛り
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn kern6(k: usize, x: *const f32, xs: usize, p: *const u8, s: *const f32, y: *mut f32, ys: usize) {
        let mut a = [[_mm256_setzero_ps(); 2]; 6];
        for l in 0..k {
            let (w0, w1) = w16(p.add(l * 8));
            for r in 0..6 {
                let b = _mm256_broadcast_ss(&*x.add(r * xs + l));
                a[r][0] = _mm256_fmadd_ps(b, w0, a[r][0]);
                a[r][1] = _mm256_fmadd_ps(b, w1, a[r][1]);
            }
        }
        let (s0, s1) = (_mm256_loadu_ps(s), _mm256_loadu_ps(s.add(8)));
        for r in 0..6 {
            _mm256_storeu_ps(y.add(r * ys), _mm256_fmadd_ps(a[r][0], s0, _mm256_loadu_ps(y.add(r * ys))));
            _mm256_storeu_ps(y.add(r * ys + 8), _mm256_fmadd_ps(a[r][1], s1, _mm256_loadu_ps(y.add(r * ys + 8))));
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn kern1(k: usize, x: *const f32, p: *const u8, s: *const f32, y: *mut f32) {
        let (mut a0, mut a1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        for l in 0..k {
            let b = _mm256_broadcast_ss(&*x.add(l));
            let (w0, w1) = w16(p.add(l * 8));
            a0 = _mm256_fmadd_ps(b, w0, a0);
            a1 = _mm256_fmadd_ps(b, w1, a1);
        }
        _mm256_storeu_ps(y, _mm256_fmadd_ps(a0, _mm256_loadu_ps(s), _mm256_loadu_ps(y)));
        _mm256_storeu_ps(y.add(8), _mm256_fmadd_ps(a1, _mm256_loadu_ps(s.add(8)), _mm256_loadu_ps(y.add(8))));
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)))
    }

    // 長さが 16 の倍数の内積
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot(a: *const f32, b: *const f32, n: usize) -> f32 {
        let (mut s0, mut s1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let mut j = 0;
        while j < n {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(j)), _mm256_loadu_ps(b.add(j)), s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(j + 8)), _mm256_loadu_ps(b.add(j + 8)), s1);
            j += 16;
        }
        hsum(_mm256_add_ps(s0, s1))
    }

    // Cephes の expf。x は [-87.3, 88.3] に切る（enc.rs の exp256 と同じ）
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn exp256(x: __m256) -> __m256 {
        let x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.3)), _mm256_set1_ps(88.3));
        let fx = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.442_695_f32)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693_359_4), x);
        let r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.121_944_4e-4), r);
        let z = _mm256_mul_ps(r, r);
        let mut y = _mm256_set1_ps(1.987_569_1e-4);
        for c in [1.398_199_9e-3f32, 8.333_452e-3, 4.166_579_6e-2, 1.666_666_5e-1, 5.000_000_1e-1] {
            y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(c));
        }
        y = _mm256_add_ps(_mm256_fmadd_ps(y, z, r), _mm256_set1_ps(1.0));
        let e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(0x7f)), 23);
        _mm256_mul_ps(y, _mm256_castsi256_ps(e))
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn rms(x: &[f32], w: &[f32], dim: usize, out: &mut [f32]) {
        assert!(x.len() == out.len() && x.len() % dim == 0 && dim % 16 == 0 && (w.is_empty() || w.len() == dim));
        for (r, y) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
            let sc = _mm256_set1_ps(1.0 / (dot(r.as_ptr(), r.as_ptr(), dim) / dim as f32 + EPS).sqrt());
            for j in (0..dim).step_by(8) {
                let v = _mm256_mul_ps(_mm256_loadu_ps(r.as_ptr().add(j)), sc);
                let v = if w.is_empty() { v } else { _mm256_mul_ps(v, _mm256_loadu_ps(w.as_ptr().add(j))) };
                _mm256_storeu_ps(y.as_mut_ptr().add(j), v);
            }
        }
    }

    // gelu の tanh 版。tanh(u) = 1 − 2 / (exp(2u) + 1)
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn gelu_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
        assert!(a.len() == b.len() && a.len() == out.len());
        let n8 = a.len() / 8 * 8;
        let (one, two) = (_mm256_set1_ps(1.0), _mm256_set1_ps(2.0));
        for j in (0..n8).step_by(8) {
            let x = _mm256_loadu_ps(a.as_ptr().add(j));
            let u = _mm256_mul_ps(_mm256_set1_ps(0.797_884_6), _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(0.044_715), _mm256_mul_ps(x, x)), x, x));
            let t = _mm256_sub_ps(one, _mm256_div_ps(two, _mm256_add_ps(exp256(_mm256_mul_ps(two, u)), one)));
            let g = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5), x), _mm256_add_ps(one, t));
            _mm256_storeu_ps(out.as_mut_ptr().add(j), _mm256_mul_ps(g, _mm256_loadu_ps(b.as_ptr().add(j))));
        }
        for j in n8..a.len() {
            out[j] = super::portable::gelu(a[j]) * b[j];
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn attend(q: &[f32], k: &[f32], v: &[f32], n: usize, hd: usize, kv: usize, from: usize, out: &mut [f32]) {
        assert!(hd % 16 == 0);
        let mut buf = vec![0f32; n.div_ceil(8) * 8];
        for (r, o) in out.chunks_exact_mut(HEADS * hd).enumerate() {
            let i = from + r;
            for h in 0..HEADS {
                let g = h / (HEADS / kv);
                let (kh, vh) = (k.as_ptr().add(g * n * hd), v.as_ptr().add(g * n * hd));
                let qi = q.as_ptr().add((h * n + i) * hd);
                let mut max = f32::NEG_INFINITY;
                for j in 0..n {
                    let s = dot(qi, kh.add(j * hd), hd);
                    buf[j] = s;
                    max = max.max(s);
                }
                buf[n..].fill(f32::NEG_INFINITY);
                let (mx, mut sum) = (_mm256_set1_ps(max), _mm256_setzero_ps());
                for j in (0..buf.len()).step_by(8) {
                    let e = exp256(_mm256_sub_ps(_mm256_loadu_ps(buf.as_ptr().add(j)), mx));
                    // 詰め物（-inf）は 0 にする（exp256 は下を -87.3 で切るので、0 にならない）
                    let e = _mm256_and_ps(e, _mm256_cmp_ps(_mm256_loadu_ps(buf.as_ptr().add(j)), _mm256_set1_ps(f32::NEG_INFINITY), _CMP_GT_OQ));
                    _mm256_storeu_ps(buf.as_mut_ptr().add(j), e);
                    sum = _mm256_add_ps(sum, e);
                }
                let isum = 1.0 / hsum(sum);
                let acc = o.as_mut_ptr().add(h * hd);
                for c in (0..hd).step_by(8) {
                    _mm256_storeu_ps(acc.add(c), _mm256_setzero_ps());
                }
                for j in 0..n {
                    let p = buf[j] * isum;
                    if p > 0.0 {
                        let pv = _mm256_set1_ps(p);
                        for c in (0..hd).step_by(8) {
                            _mm256_storeu_ps(acc.add(c), _mm256_fmadd_ps(pv, _mm256_loadu_ps(vh.add(j * hd + c)), _mm256_loadu_ps(acc.add(c))));
                        }
                    }
                }
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dots4(q: &[f32], node: &[u8], out: &mut [f32]) {
        let d = q.len();
        assert!(d % 16 == 0 && node.len() == out.len() * d / 2);
        for (r, o) in node.chunks_exact(d / 2).zip(out) {
            let (mut s0, mut s1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
            for j in (0..d).step_by(16) {
                let (w0, w1) = w16(r.as_ptr().add(j / 2));
                s0 = _mm256_fmadd_ps(_mm256_loadu_ps(q.as_ptr().add(j)), w0, s0);
                s1 = _mm256_fmadd_ps(_mm256_loadu_ps(q.as_ptr().add(j + 8)), w1, s1);
            }
            *o = hsum(_mm256_add_ps(s0, s1));
        }
    }
}

// 位置ごとの cos・sin（[位置][dim / 2]）
struct Rope {
    dim: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Rope {
    fn new(dim: usize, theta: f64) -> Self {
        let inv: Vec<f32> = (0..dim).step_by(2).map(|i| (1.0 / theta.powf(i as f64 / dim as f64)) as f32).collect();
        let (mut cos, mut sin) = (Vec::with_capacity(MAX_LEN * dim / 2), Vec::with_capacity(MAX_LEN * dim / 2));
        for t in 0..MAX_LEN {
            for f in &inv {
                let v = t as f32 * f;
                cos.push(v.cos());
                sin.push(v.sin());
            }
        }
        Self { dim, cos, sin }
    }

    // x: [頭][位置][dim]
    fn rotate(&self, x: &mut [f32], n: usize) {
        let hf = self.dim / 2;
        for (i, r) in x.chunks_exact_mut(self.dim).enumerate() {
            let t = i % n;
            let (c, s) = (&self.cos[t * hf..][..hf], &self.sin[t * hf..][..hf]);
            let (a, b) = r.split_at_mut(hf);
            for j in 0..hf {
                let (x1, x2) = (a[j], b[j]);
                a[j] = c[j] * x1 - s[j] * x2;
                b[j] = c[j] * x2 + s[j] * x1;
            }
        }
    }
}

struct Layer {
    hd: usize, // 頭の幅
    kv: usize, // 鍵と値の頭の数
    rope: usize,
    q: P4,
    k: P4,
    v: P4,
    o: P4,
    gate: P4,
    up: P4,
    down: P4,
    pg: P4,
    pp: P4,
    n_in: &'static [f32],
    n_q: &'static [f32],
    n_k: &'static [f32],
    n_att: &'static [f32],
    n_pre: &'static [f32],
    n_ff: &'static [f32],
    n_ple: &'static [f32],
    scalar: f32,
}

impl Layer {
    // 正規化 → 問い・鍵・値の射影。x: [n][H]、q: [n][HEADS * hd]、k・v: [n][kv * hd]（0 で初期化しておく）
    fn pre(&self, x: &[f32], q: &mut [f32], k: &mut [f32], v: &mut [f32]) {
        let n = x.len() / H;
        let mut h = vec![0f32; n * H];
        rms(x, self.n_in, H, &mut h);
        linear(&self.q, &h, n, q);
        linear(&self.k, &h, n, k);
        linear(&self.v, &h, n, v);
    }

    // 注意の出力の射影 → 正規化 → 足す → 正規化 → MLP → 正規化 → 足す → 層ごとの入力の足し込み → 層の倍率。x: [n][H]、attn: [n][HEADS * hd]、ple: [n][H]（この層の分）
    fn post(&self, x: &mut [f32], attn: &[f32], ple: &[f32]) {
        let n = x.len() / H;
        let (mut a, mut b) = (vec![0f32; n * H], vec![0f32; n * H]);
        linear(&self.o, attn, n, &mut a);
        rms(&a, self.n_att, H, &mut b);
        add(x, &b);
        rms(x, self.n_pre, H, &mut a);
        // MLP は ROWS 行ずつ（中間の幅 FF の配列を、行の全部の分は持たない。いちばん長い入力で、作業用のメモリが数 MB 減る）。
        // ROWS は、核が行をまとめる単位（portable は 48 行ごとに 2 行ずつ、avx は 96 行ごとに 6 行ずつ）の倍数なので、行のまとめ方も足す順も、分けないときと同じ（答えは 1 ビットも変わらない）
        const ROWS: usize = 48;
        let r = n.min(ROWS);
        let (mut g, mut u, mut m) = (vec![0f32; r * FF], vec![0f32; r * FF], vec![0f32; r * FF]);
        b.fill(0.0);
        for (ac, bc) in a.chunks(ROWS * H).zip(b.chunks_mut(ROWS * H)) {
            let r = ac.len() / H;
            let (g, u, m) = (&mut g[..r * FF], &mut u[..r * FF], &mut m[..r * FF]);
            g.fill(0.0);
            u.fill(0.0);
            linear(&self.gate, ac, r, g);
            linear(&self.up, ac, r, u);
            gelu_mul(g, u, m);
            linear(&self.down, m, r, bc);
        }
        drop((g, u, m));
        rms(&b, self.n_ff, H, &mut a);
        add(x, &a);
        a.fill(0.0);
        linear(&self.pg, x, n, &mut a);
        gelu_mul(&a, ple, &mut b);
        a.fill(0.0);
        linear(&self.pp, &b, n, &mut a);
        rms(&a, self.n_ple, H, &mut b);
        for (xv, bv) in x.iter_mut().zip(&b) {
            *xv = (*xv + bv) * self.scalar;
        }
    }
}

fn add(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}

pub struct Engine {
    emb: &'static [u8],
    emb_s: &'static [f32],
    ple: P4,
    ple_norm: &'static [f32],
    layers: Vec<Layer>,
    norm: &'static [f32],
    eproj: P4,
    q_proj: P4,
    q_bias: &'static [f32],
    ropes: [Rope; 2],
}

impl Engine {
    pub fn new(b: &'static Bundle) -> Result<Self> {
        avx_on(); // ここで経路を決める
        let nl = b.man["enc"]["layers"].as_u64().context("目録の enc.layers")? as usize;
        let mut layers = Vec::new();
        for i in 0..nl {
            let (hd, kv, rope) = if i == nl - 1 { (512, 1, 1) } else { (256, 2, 0) };
            let lin = |k: &str, out: usize, inp: usize| P4::load(b, &format!("g.l{i}.{k}"), out, inp);
            let vec = |k: &str, n: usize| -> Result<&'static [f32]> {
                let v = b.get::<f32>(&format!("g.l{i}.{k}"));
                if v.len() != n {
                    bail!("g.l{i}.{k}: 長さが {n} でない");
                }
                Ok(v)
            };
            layers.push(Layer {
                hd,
                kv,
                rope,
                q: lin("q", HEADS * hd, H)?,
                k: lin("k", kv * hd, H)?,
                v: lin("v", kv * hd, H)?,
                o: lin("o", H, HEADS * hd)?,
                gate: lin("gate", FF, H)?,
                up: lin("up", FF, H)?,
                down: lin("down", H, FF)?,
                pg: lin("pg", H, H)?,
                pp: lin("pp", H, H)?,
                n_in: vec("n_in", H)?,
                n_q: vec("n_q", hd)?,
                n_k: vec("n_k", hd)?,
                n_att: vec("n_att", H)?,
                n_pre: vec("n_pre", H)?,
                n_ff: vec("n_ff", H)?,
                n_ple: vec("n_ple", H)?,
                scalar: vec("scalar", 1)?[0],
            });
        }
        let e = Self {
            emb: b.get("g.emb"),
            emb_s: b.get("g.emb.s"),
            ple: P4::load(b, "g.ple", nl * H, H)?,
            ple_norm: b.get("g.ple.norm"),
            layers,
            norm: b.get("g.norm"),
            eproj: P4::load(b, "g.eproj", STATE, H)?,
            q_proj: P4::load(b, "g.q_proj", 5 * D, STATE)?,
            q_bias: b.get("g.q_proj.b"),
            ropes: [Rope::new(256, 10000.0), Rope::new(512, 1000000.0)],
        };
        if e.emb.len() != e.emb_s.len() * H / 2 || e.ple_norm.len() != H || e.norm.len() != H || e.q_bias.len() != 5 * D {
            bail!("読み手の区画の形が違う");
        }
        Ok(e)
    }

    // トークンの番号の並び（頭と尻の印つき。MAX_LEN まで）→ 5 段の問いのベクトル [5][D]（どれも長さ 1）
    pub fn query(&self, ids: &[u32]) -> Vec<f32> {
        let (n, nl) = (ids.len(), self.layers.len());
        assert!(n > 0 && n <= MAX_LEN);
        let root = (H as f32).sqrt();
        let mut x = vec![0f32; n * H];
        for (row, &id) in x.chunks_exact_mut(H).zip(ids) {
            let (r, s) = (&self.emb[id as usize * H / 2..][..H / 2], self.emb_s[id as usize]);
            for (o, b) in row.chunks_exact_mut(16).zip(r.chunks_exact(8)) {
                let w = portable::w16(b);
                for j in 0..16 {
                    o[j] = w[j] * s * root;
                }
            }
        }
        // 層ごとの入力: [n][層][H]
        let half = n / 2;
        let mut raw = vec![0f32; n * nl * H];
        {
            let ((xa, xb), (ra, rb)) = (x.split_at(half * H), raw.split_at_mut(half * nl * H));
            rayon::join(|| linear(&self.ple, xa, half, ra), || linear(&self.ple, xb, n - half, rb));
        }
        let sc = (H as f32).powf(-0.5);
        raw.iter_mut().for_each(|v| *v *= sc);
        let mut ple = vec![0f32; n * nl * H];
        rms(&raw, self.ple_norm, H, &mut ple);
        drop(raw);
        let mut pl = vec![0f32; n * H];
        for (i, l) in self.layers.iter().enumerate() {
            for (dst, src) in pl.chunks_exact_mut(H).zip(ple.chunks_exact(nl * H)) {
                dst.copy_from_slice(&src[i * H..][..H]);
            }
            self.layer(l, &mut x, &pl, n);
        }
        let mut xn = vec![0f32; n * H];
        rms(&x, self.norm, H, &mut xn);
        let mut y = vec![0f32; n * STATE];
        {
            let ((xa, xb), (ya, yb)) = (xn.split_at(half * H), y.split_at_mut(half * STATE));
            rayon::join(|| linear(&self.eproj, xa, half, ya), || linear(&self.eproj, xb, n - half, yb));
        }
        let mut state = vec![0f32; STATE];
        for r in y.chunks_exact(STATE) {
            add(&mut state, r);
        }
        state.iter_mut().for_each(|v| *v /= n as f32);
        let mut q = self.q_bias.to_vec();
        linear(&self.q_proj, &state, 1, &mut q);
        for v in q.chunks_exact_mut(D) {
            let len = v.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-12);
            v.iter_mut().for_each(|a| *a /= len);
        }
        q
    }

    // 1 層。行を前半と後半に分けて 2 スレッドで計算する
    fn layer(&self, l: &Layer, x: &mut [f32], ple: &[f32], n: usize) {
        let (h, hd, kv) = (n / 2, l.hd, l.kv);
        let (qw, kw) = (HEADS * hd, kv * hd);
        let (mut qp, mut kp, mut vp) = (vec![0f32; n * qw], vec![0f32; n * kw], vec![0f32; n * kw]);
        {
            let (xa, xb) = x.split_at(h * H);
            let ((qa, qb), (ka, kb), (va, vb)) = (qp.split_at_mut(h * qw), kp.split_at_mut(h * kw), vp.split_at_mut(h * kw));
            rayon::join(|| l.pre(xa, qa, ka, va), || l.pre(xb, qb, kb, vb));
        }
        // [位置][頭][hd] → [頭][位置][hd]。問いと鍵は RMSNorm（重みあり）→ 回転、値は RMSNorm（重みなし）
        let turn = |src: Vec<f32>, heads: usize, w: &[f32]| {
            let mut o = vec![0f32; src.len()];
            for (p, r) in src.chunks_exact(heads * hd).enumerate() {
                for g in 0..heads {
                    rms(&r[g * hd..][..hd], w, hd, &mut o[(g * n + p) * hd..][..hd]);
                }
            }
            o
        };
        let mut q = turn(qp, HEADS, l.n_q);
        let mut k = turn(kp, kv, l.n_k);
        let v = turn(vp, kv, &[]);
        let rope = &self.ropes[l.rope];
        rope.rotate(&mut q, n);
        rope.rotate(&mut k, n);
        let mut attn = vec![0f32; n * qw];
        {
            let (aa, ab) = attn.split_at_mut(h * qw);
            rayon::join(|| attend(&q, &k, &v, n, hd, kv, 0, aa), || attend(&q, &k, &v, n, hd, kv, h, ab));
        }
        drop((q, k, v));
        let ((xa, xb), (aa, ab), (pa, pb)) = (x.split_at_mut(h * H), attn.split_at(h * qw), ple.split_at(h * H));
        rayon::join(|| l.post(xa, aa, pa), || l.post(xb, ab, pb));
    }
}
