// 3 層の ModernBERT（ruri-v3-30m の形: 幅 256・頭 4・中間 1024）の CPU 推論。前に書いた実装（fp32 の経路）から写して、
// 重みを一式の int8（行ごとの目盛り）のまま使う形にした（fp32 の写しを持たない。持つと常駐メモリが約 14 MB 増える）。
// 線形は、マップした int8 を掛ける所で fp32 に直し、出力ごとの目盛りを最後に掛ける。語の埋め込みは使う行だけを戻し、節のベクトルは int8 のまま内積を取る。
// 行（トークン）を 2 つに分けて rayon::join で 2 スレッドに配る。層 0 は全体の注意、層 1・2 は前後 64 の注意（形は一式を作る台本が確かめている）。
// 計算の核（線形・正規化・GeGLU・注意・節との内積）は 2 通りある。読み込みのとき（Engine::new）に選ぶ:
// - avx: x86-64 で、CPU に AVX2 と FMA があるとき。手で書いた命令。
// - portable: 素の Rust（unsafe なし）。どの CPU でも動く。コンパイラがベクトルにできる形に書いてある。x86-64 でない台では、これだけを組み立てる。
// 環境変数 ZEIBAN_PORTABLE を置くと、AVX2 があっても portable を使う（試し用）。2 つの経路は丸めの順が少し違う（点の差は 1e-4 より小さい。Python の基準との照合で見る）。
use crate::Bundle;
use anyhow::{bail, Result};
use std::sync::OnceLock;

pub const D: usize = 256;
const HEADS: usize = 4;
const HS: usize = D / HEADS;
const QKV: usize = D * 3;
const FF: usize = 1024;
const EPS: f32 = 1e-5;
const WINDOW: usize = 64;
const MAX_LEN: usize = 256;

fn avx_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        if std::env::var_os("ZEIBAN_PORTABLE").is_none() && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return true;
        }
        false
    })
}

// 使っている経路の名前
pub fn path() -> &'static str {
    if avx_on() {
        "avx2"
    } else {
        "portable"
    }
}

// 線形（バイアス無し）。重みは int8 の [16 出力ごとのブロック][入力][16]（一式を作る台本がこの並びで書く）と、出力ごとの目盛り
struct Packed {
    inp: usize,
    blocks: usize,
    p: &'static [i8],
    s: &'static [f32],
}

impl Packed {
    fn load(b: &'static Bundle, name: &str, out: usize, inp: usize) -> Result<Self> {
        let (p, s) = (b.get::<i8>(name), b.get::<f32>(&format!("{name}.s")));
        if p.len() != out * inp || s.len() != out || out % 16 != 0 {
            bail!("{name}: 形が [{out}, {inp}] でない");
        }
        Ok(Self { inp, blocks: out / 16, p, s })
    }
}

// 核の振り分け: AVX2 と FMA があれば avx、無ければ（x86-64 でない台ではいつも）portable
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
    fn linear(p: &Packed, x: &[f32], n: usize, y: &mut [f32]);
    /// バイアス無しの LayerNorm。x・out: [n][D]
    fn layer_norm(x: &[f32], w: &[f32], out: &mut [f32]);
    fn add_assign(x: &mut [f32], y: &[f32]);
    /// GeGLU: 行 [2 × FF] の前半 a と後半 b から gelu(a) * b を作る。gelu(x) = x/2 * (1 + erf(x/√2))、erf は Abramowitz & Stegun 7.1.26（誤差 1.5e-7 以下）
    fn geglu(src: &[f32], dst: &mut [f32]);
    /// 位置 from からの out.len() / D 行の注意の出力。q・k・v: [頭][位置][HS]（q・k は回転済み）。window があれば、前後 window の鍵だけを見る
    fn attend(window: Option<usize>, q: &[f32], k: &[f32], v: &[f32], n: usize, from: usize, out: &mut [f32]);
    /// 問いのベクトル q [D] と、int8 の節のベクトル node [行][D] の内積（行ごと）。out: [行]
    pub fn dots(q: &[f32], node: &[i8], out: &mut [f32]);
}

// どの CPU でも動く核。和は 8 本に分けて足す（浮動小数の足し算は順を変えられないので、1 本の和はベクトルにならない）
mod portable {
    use super::{Packed, D, EPS, FF, HEADS, HS};

    // 8 本の和をまとめる
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

    // R 行をまとめて掛ける（重みを fp32 に直すのは R 行で 1 回）: y += (Σ_l x_l × 整数の重み_l) × 目盛り
    fn rows<const R: usize>(p: &Packed, x: &[f32], y: &mut [f32]) {
        let (inp, out) = (p.inp, p.blocks * 16);
        for (b, w) in p.p.chunks_exact(inp * 16).enumerate() {
            let mut a = [[0f32; 16]; R];
            for (l, wl) in w.chunks_exact(16).enumerate() {
                let wf: [f32; 16] = std::array::from_fn(|c| wl[c] as f32);
                for r in 0..R {
                    let xv = x[r * inp + l];
                    for c in 0..16 {
                        a[r][c] += xv * wf[c];
                    }
                }
            }
            let s = &p.s[b * 16..][..16];
            for r in 0..R {
                for (c, yv) in y[r * out + b * 16..][..16].iter_mut().enumerate() {
                    *yv += a[r][c] * s[c];
                }
            }
        }
    }

    pub fn linear(p: &Packed, x: &[f32], n: usize, y: &mut [f32]) {
        let (inp, out) = (p.inp, p.blocks * 16);
        assert!(x.len() == n * inp && y.len() == n * out);
        let n4 = n / 4 * 4;
        for (xr, yr) in x[..n4 * inp].chunks_exact(4 * inp).zip(y.chunks_exact_mut(4 * out)) {
            rows::<4>(p, xr, yr);
        }
        for (xr, yr) in x[n4 * inp..].chunks_exact(inp).zip(y[n4 * out..].chunks_exact_mut(out)) {
            rows::<1>(p, xr, yr);
        }
    }

    pub fn layer_norm(x: &[f32], w: &[f32], out: &mut [f32]) {
        for (r, y) in x.chunks_exact(D).zip(out.chunks_exact_mut(D)) {
            let (mut s, mut v) = ([0f32; 8], [0f32; 8]);
            for c in r.chunks_exact(8) {
                for j in 0..8 {
                    s[j] += c[j];
                }
            }
            let mean = hsum(s) / D as f32;
            for c in r.chunks_exact(8) {
                for j in 0..8 {
                    v[j] += (c[j] - mean) * (c[j] - mean);
                }
            }
            let sc = 1.0 / (hsum(v) / D as f32 + EPS).sqrt();
            for ((a, wv), yv) in r.iter().zip(w).zip(y) {
                *yv = (a - mean) * sc * wv;
            }
        }
    }

    pub fn add_assign(x: &mut [f32], y: &[f32]) {
        for (a, b) in x.iter_mut().zip(y) {
            *a += b;
        }
    }

    fn gelu(x: f32) -> f32 {
        let z = x * std::f32::consts::FRAC_1_SQRT_2;
        let a = z.abs();
        let t = 1.0 / (1.0 + 0.327_591_1 * a);
        let p = t * (0.254_829_6 + t * (-0.284_496_74 + t * (1.421_413_7 + t * (-1.453_152_1 + t * 1.061_405_4))));
        let erf = 1.0 - p * (-a * a).exp();
        0.5 * x * (1.0 + erf.copysign(z))
    }

    pub fn geglu(src: &[f32], dst: &mut [f32]) {
        for (r, d) in src.chunks_exact(2 * FF).zip(dst.chunks_exact_mut(FF)) {
            let (a, b) = r.split_at(FF);
            for ((av, bv), dv) in a.iter().zip(b).zip(d) {
                *dv = gelu(*av) * bv;
            }
        }
    }

    pub fn attend(window: Option<usize>, q: &[f32], k: &[f32], v: &[f32], n: usize, from: usize, out: &mut [f32]) {
        let scale = (HS as f32).powf(-0.5);
        let mut buf = vec![0f32; n];
        for (r, o) in out.chunks_exact_mut(D).enumerate() {
            let i = from + r;
            let (lo, hi) = match window {
                Some(w) => (i.saturating_sub(w), (i + w + 1).min(n)),
                None => (0, n),
            };
            for h in 0..HEADS {
                let (kh, vh) = (&k[(h * n + lo) * HS..(h * n + hi) * HS], &v[(h * n + lo) * HS..(h * n + hi) * HS]);
                let qi = &q[(h * n + i) * HS..][..HS];
                // 点 q_i · k_j → softmax
                let s = &mut buf[..hi - lo];
                for (sj, kj) in s.iter_mut().zip(kh.chunks_exact(HS)) {
                    *sj = dot(qi, kj);
                }
                let max = s.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0f32;
                for sj in s.iter_mut() {
                    *sj = ((*sj - max) * scale).exp();
                    sum += *sj;
                }
                let isum = 1.0 / sum;
                // 混ぜる: out_i = Σ_j p_j v_j
                let acc = &mut o[h * HS..][..HS];
                acc.fill(0.0);
                for (sj, vj) in s.iter().zip(vh.chunks_exact(HS)) {
                    let p = sj * isum;
                    for (a, x) in acc.iter_mut().zip(vj) {
                        *a += p * x;
                    }
                }
            }
        }
    }

    pub fn dots(q: &[f32], node: &[i8], out: &mut [f32]) {
        assert!(q.len() == D && node.len() == out.len() * D);
        for (r, o) in node.chunks_exact(D).zip(out) {
            let mut s = [0f32; 8];
            for (x, y) in q.chunks_exact(8).zip(r.chunks_exact(8)) {
                for j in 0..8 {
                    s[j] += x[j] * y[j] as f32;
                }
            }
            *o = hsum(s);
        }
    }
}

// AVX2 と FMA の核。呼ぶ側（kernels!）が、CPU に AVX2 と FMA があることを確かめている
#[cfg(target_arch = "x86_64")]
mod avx {
    use super::{Packed, D, EPS, FF, HEADS, HS};
    use std::arch::x86_64::*;

    const MC: usize = 96; // マイクロカーネルの外側で一度に扱う行（x の 96 行を L2 に載せる）
    const KC: usize = 256; // 入力の次元を分ける単位（重みの 16 x 256 を L1 に載せる）

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn linear(p: &Packed, x: &[f32], n: usize, y: &mut [f32]) {
        let (inp, out, xp, yp) = (p.inp, p.blocks * 16, x.as_ptr(), y.as_mut_ptr());
        assert!(x.len() == n * inp && y.len() == n * out);
        for k0 in (0..inp).step_by(KC) {
            let kc = KC.min(inp - k0);
            for r0 in (0..n).step_by(MC) {
                let r1 = (r0 + MC).min(n);
                for b in 0..p.blocks {
                    let (w, s) = (p.p.as_ptr().add((b * inp + k0) * 16), p.s.as_ptr().add(b * 16));
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

    // int8 の 8 個を fp32 にする
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn w8(p: *const i8) -> __m256 {
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(p as *const __m128i)))
    }

    // 6 行 × 16 出力: y += (Σ_l x_l × 整数の重み_l) × 目盛り
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn kern6(k: usize, x: *const f32, xs: usize, p: *const i8, s: *const f32, y: *mut f32, ys: usize) {
        let mut a = [[_mm256_setzero_ps(); 2]; 6];
        for l in 0..k {
            let (w0, w1) = (w8(p.add(l * 16)), w8(p.add(l * 16 + 8)));
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
    unsafe fn kern1(k: usize, x: *const f32, p: *const i8, s: *const f32, y: *mut f32) {
        let (mut a0, mut a1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        for l in 0..k {
            let b = _mm256_broadcast_ss(&*x.add(l));
            a0 = _mm256_fmadd_ps(b, w8(p.add(l * 16)), a0);
            a1 = _mm256_fmadd_ps(b, w8(p.add(l * 16 + 8)), a1);
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

    // Cephes の expf。x は [-87.3, 88.3] に切る
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

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn gelu256(x: __m256) -> __m256 {
        let z = _mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::FRAC_1_SQRT_2));
        let sign = _mm256_set1_ps(-0.0);
        let a = _mm256_andnot_ps(sign, z);
        let t = _mm256_div_ps(_mm256_set1_ps(1.0), _mm256_fmadd_ps(_mm256_set1_ps(0.327_591_1), a, _mm256_set1_ps(1.0)));
        let mut p = _mm256_set1_ps(1.061_405_4);
        for c in [-1.453_152_1f32, 1.421_413_7, -0.284_496_74, 0.254_829_6] {
            p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(c));
        }
        p = _mm256_mul_ps(p, t);
        let erf = _mm256_fnmadd_ps(p, exp256(_mm256_xor_ps(_mm256_mul_ps(a, a), sign)), _mm256_set1_ps(1.0));
        let serf = _mm256_or_ps(erf, _mm256_and_ps(z, sign));
        _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5), x), _mm256_add_ps(_mm256_set1_ps(1.0), serf))
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn layer_norm(x: &[f32], w: &[f32], out: &mut [f32]) {
        for (r, y) in x.chunks_exact(D).zip(out.chunks_exact_mut(D)) {
            let mut s = _mm256_setzero_ps();
            for c in r.chunks_exact(8) {
                s = _mm256_add_ps(s, _mm256_loadu_ps(c.as_ptr()));
            }
            let mv = _mm256_set1_ps(hsum(s) / D as f32);
            let mut v = _mm256_setzero_ps();
            for c in r.chunks_exact(8) {
                let a = _mm256_sub_ps(_mm256_loadu_ps(c.as_ptr()), mv);
                v = _mm256_fmadd_ps(a, a, v);
            }
            let sc = _mm256_set1_ps(1.0 / (hsum(v) / D as f32 + EPS).sqrt());
            for ((c, wc), yc) in r.chunks_exact(8).zip(w.chunks_exact(8)).zip(y.chunks_exact_mut(8)) {
                let a = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(c.as_ptr()), mv), sc);
                _mm256_storeu_ps(yc.as_mut_ptr(), _mm256_mul_ps(a, _mm256_loadu_ps(wc.as_ptr())));
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn add_assign(x: &mut [f32], y: &[f32]) {
        for (a, b) in x.chunks_exact_mut(8).zip(y.chunks_exact(8)) {
            _mm256_storeu_ps(a.as_mut_ptr(), _mm256_add_ps(_mm256_loadu_ps(a.as_ptr()), _mm256_loadu_ps(b.as_ptr())));
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn geglu(src: &[f32], dst: &mut [f32]) {
        for (r, d) in src.chunks_exact(2 * FF).zip(dst.chunks_exact_mut(FF)) {
            let (a, b) = r.split_at(FF);
            for ((ac, bc), dc) in a.chunks_exact(8).zip(b.chunks_exact(8)).zip(d.chunks_exact_mut(8)) {
                _mm256_storeu_ps(dc.as_mut_ptr(), _mm256_mul_ps(gelu256(_mm256_loadu_ps(ac.as_ptr())), _mm256_loadu_ps(bc.as_ptr())));
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn attend(window: Option<usize>, q: &[f32], k: &[f32], v: &[f32], n: usize, from: usize, out: &mut [f32]) {
        assert!(q.len() == n * D && k.len() == n * D && v.len() == n * D && from + out.len() / D <= n);
        let scale = _mm256_set1_ps((HS as f32).powf(-0.5));
        let mut s = vec![0f32; n + 8];
        for (r, o) in out.chunks_exact_mut(D).enumerate() {
            let i = from + r;
            let (lo, hi) = match window {
                Some(w) => (i.saturating_sub(w), (i + w + 1).min(n)),
                None => (0, n),
            };
            for h in 0..HEADS {
                let (kh, vh) = (k.as_ptr().add(h * n * HS), v.as_ptr().add(h * n * HS));
                let qi = q.as_ptr().add((h * n + i) * HS);
                let qv: [__m256; 8] = std::array::from_fn(|c| _mm256_loadu_ps(qi.add(c * 8)));
                // 点 q_i · k_j
                let w = hi - lo;
                for j in 0..w {
                    let kj = kh.add((lo + j) * HS);
                    let mut a = [_mm256_setzero_ps(); 4];
                    for c in 0..8 {
                        a[c % 4] = _mm256_fmadd_ps(qv[c], _mm256_loadu_ps(kj.add(c * 8)), a[c % 4]);
                    }
                    s[j] = hsum(_mm256_add_ps(_mm256_add_ps(a[0], a[1]), _mm256_add_ps(a[2], a[3])));
                }
                // softmax（w を 8 の倍数まで -inf で詰める）
                let w8 = (w + 7) & !7;
                s[w..w8].fill(f32::NEG_INFINITY);
                let max = _mm256_set1_ps(s[..w].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
                let mut sum = _mm256_setzero_ps();
                for c in s[..w8].chunks_exact_mut(8) {
                    let p = exp256(_mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(c.as_ptr()), max), scale));
                    _mm256_storeu_ps(c.as_mut_ptr(), p);
                    sum = _mm256_add_ps(sum, p);
                }
                let isum = 1.0 / hsum(sum);
                // 混ぜる: out_i = Σ_j p_j v_j
                let mut acc = [_mm256_setzero_ps(); 8];
                for j in 0..w {
                    let b = _mm256_set1_ps(s[j] * isum);
                    let vj = vh.add((lo + j) * HS);
                    for c in 0..8 {
                        acc[c] = _mm256_fmadd_ps(b, _mm256_loadu_ps(vj.add(c * 8)), acc[c]);
                    }
                }
                for c in 0..8 {
                    _mm256_storeu_ps(o.as_mut_ptr().add(h * HS + c * 8), acc[c]);
                }
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dots(q: &[f32], node: &[i8], out: &mut [f32]) {
        assert!(q.len() == D && node.len() == out.len() * D);
        let qv: [__m256; D / 8] = std::array::from_fn(|c| _mm256_loadu_ps(q.as_ptr().add(c * 8)));
        for (r, o) in node.chunks_exact(D).zip(out) {
            let mut a = [_mm256_setzero_ps(); 4];
            for c in 0..D / 8 {
                a[c % 4] = _mm256_fmadd_ps(qv[c], w8(r.as_ptr().add(c * 8)), a[c % 4]);
            }
            *o = hsum(_mm256_add_ps(_mm256_add_ps(a[0], a[1]), _mm256_add_ps(a[2], a[3])));
        }
    }
}

// 位置ごとの cos・sin（[位置][HS / 2]）
struct Rope {
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Rope {
    fn new(theta: f64) -> Self {
        let inv: Vec<f32> = (0..HS).step_by(2).map(|i| 1f32 / theta.powf(i as f64 / HS as f64) as f32).collect();
        let (mut cos, mut sin) = (Vec::with_capacity(MAX_LEN * HS / 2), Vec::with_capacity(MAX_LEN * HS / 2));
        for t in 0..MAX_LEN {
            for f in &inv {
                let v = t as f32 * f;
                cos.push(v.cos());
                sin.push(v.sin());
            }
        }
        Self { cos, sin }
    }

    // x: [頭][位置][HS]
    fn rotate(&self, x: &mut [f32], n: usize) {
        for (i, r) in x.chunks_exact_mut(HS).enumerate() {
            let t = i % n;
            let (c, s) = (&self.cos[t * HS / 2..][..HS / 2], &self.sin[t * HS / 2..][..HS / 2]);
            let (a, b) = r.split_at_mut(HS / 2);
            for j in 0..HS / 2 {
                let (x1, x2) = (a[j], b[j]);
                a[j] = c[j] * x1 - s[j] * x2;
                b[j] = c[j] * x2 + s[j] * x1;
            }
        }
    }
}

struct Layer {
    attn_norm: Option<&'static [f32]>,
    qkv: Packed,
    proj: Packed,
    mlp_norm: &'static [f32],
    wi: Packed,
    wo: Packed,
    window: Option<usize>,
}

impl Layer {
    // 正規化 → Wqkv。x: [n][D]、qkv: [n][QKV]（0 で初期化しておく）
    fn pre(&self, x: &[f32], qkv: &mut [f32]) {
        let n = x.len() / D;
        match self.attn_norm {
            Some(w) => {
                let mut a = vec![0f32; n * D];
                layer_norm(x, w, &mut a);
                linear(&self.qkv, &a, n, qkv);
            }
            None => linear(&self.qkv, x, n, qkv),
        }
    }

    // 注意の出力の射影を足し、正規化 → MLP を足す。x・attn: [n][D]
    fn post(&self, x: &mut [f32], attn: &[f32]) {
        let n = x.len() / D;
        let mut o = vec![0f32; n * D];
        linear(&self.proj, attn, n, &mut o);
        add_assign(x, &o);
        let mut m = vec![0f32; n * D];
        layer_norm(x, self.mlp_norm, &mut m);
        let mut wi = vec![0f32; n * 2 * FF];
        linear(&self.wi, &m, n, &mut wi);
        let mut g = vec![0f32; n * FF];
        geglu(&wi, &mut g);
        o.fill(0.0);
        linear(&self.wo, &g, n, &mut o);
        add_assign(x, &o);
    }
}

pub struct Engine {
    emb: &'static [i8],
    emb_s: &'static [f32],
    emb_norm: &'static [f32],
    layers: Vec<Layer>,
    final_norm: &'static [f32],
    q_proj: Packed,
    q_bias: &'static [f32],
    rope_global: Rope,
    rope_local: Rope,
}

impl Engine {
    pub fn new(b: &'static Bundle) -> Result<Self> {
        avx_on(); // ここで経路を決める
        let mut layers = Vec::new();
        for i in 0..3 {
            layers.push(Layer {
                attn_norm: (i > 0).then(|| b.get(&format!("l{i}.attn_norm"))),
                qkv: Packed::load(b, &format!("l{i}.qkv"), QKV, D)?,
                proj: Packed::load(b, &format!("l{i}.proj"), D, D)?,
                mlp_norm: b.get(&format!("l{i}.mlp_norm")),
                wi: Packed::load(b, &format!("l{i}.wi"), 2 * FF, D)?,
                wo: Packed::load(b, &format!("l{i}.wo"), D, FF)?,
                window: (i > 0).then_some(WINDOW),
            });
        }
        Ok(Self {
            emb: b.get("emb"),
            emb_s: b.get("emb.s"),
            emb_norm: b.get("emb_norm"),
            layers,
            final_norm: b.get("final_norm"),
            q_proj: Packed::load(b, "q_proj", 5 * D, D)?,
            q_bias: b.get("q_proj.b"),
            rope_global: Rope::new(160000.0),
            rope_local: Rope::new(10000.0),
        })
    }

    // トークンの番号の並び（頭と尻の印つき。MAX_LEN まで）→ 5 段の問いのベクトル [5][D]（どれも長さ 1）
    pub fn query(&self, ids: &[u32]) -> Vec<f32> {
        let n = ids.len();
        assert!(n > 0 && n <= MAX_LEN);
        let mut raw = vec![0f32; n * D];
        for (row, &id) in raw.chunks_exact_mut(D).zip(ids) {
            let (q, s) = (&self.emb[id as usize * D..][..D], self.emb_s[id as usize]);
            for (r, &v) in row.iter_mut().zip(q) {
                *r = v as f32 * s;
            }
        }
        let mut x = vec![0f32; n * D];
        layer_norm(&raw, self.emb_norm, &mut x);
        for l in &self.layers {
            self.layer(l, &mut x, n);
        }
        let mut cls = [0f32; D];
        layer_norm(&x[..D], self.final_norm, &mut cls);
        let mut q = self.q_bias.to_vec();
        linear(&self.q_proj, &cls, 1, &mut q);
        for v in q.chunks_exact_mut(D) {
            let len = v.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-12);
            v.iter_mut().for_each(|a| *a /= len);
        }
        q
    }

    // 1 層。行を前半と後半に分けて 2 スレッドで計算する（注意は、鍵と値は全部の行、出力の行だけを分ける）
    fn layer(&self, l: &Layer, x: &mut [f32], n: usize) {
        let h = n / 2;
        let mut qkv = vec![0f32; n * QKV];
        {
            let ((xa, xb), (qa, qb)) = (x.split_at(h * D), qkv.split_at_mut(h * QKV));
            rayon::join(|| l.pre(xa, qa), || l.pre(xb, qb));
        }
        // [位置][問い・鍵・値][頭][HS] → [頭][位置][HS]
        let (mut q, mut k, mut v) = (vec![0f32; n * D], vec![0f32; n * D], vec![0f32; n * D]);
        for (t, r) in qkv.chunks_exact(QKV).enumerate() {
            for hd in 0..HEADS {
                let at = (hd * n + t) * HS;
                q[at..at + HS].copy_from_slice(&r[hd * HS..][..HS]);
                k[at..at + HS].copy_from_slice(&r[D + hd * HS..][..HS]);
                v[at..at + HS].copy_from_slice(&r[2 * D + hd * HS..][..HS]);
            }
        }
        let rope = if l.window.is_some() { &self.rope_local } else { &self.rope_global };
        rope.rotate(&mut q, n);
        rope.rotate(&mut k, n);
        let mut attn = vec![0f32; n * D];
        {
            let (aa, ab) = attn.split_at_mut(h * D);
            rayon::join(|| attend(l.window, &q, &k, &v, n, 0, aa), || attend(l.window, &q, &k, &v, n, h, ab));
        }
        let ((xa, xb), (aa, ab)) = (x.split_at_mut(h * D), attn.split_at(h * D));
        rayon::join(|| l.post(xa, aa), || l.post(xb, ab));
    }
}
