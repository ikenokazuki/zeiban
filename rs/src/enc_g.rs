// 読み手の 1 つの形: 6 層 / 24 層混合 2/4-bit（enc.kind = "g6" / "g24m"）を AVX2 int8 (vpmaddubsw, K=64) / portable int8 で推論する。
// 行（トークン）を 2 つに分けて rayon::join で 2 スレッドに配る（注意は、鍵と値は全部の行、出力の行だけを分ける）。
// ZEIBAN_FP32=1 で FP32 参照経路、ZEIBAN_PORTABLE=1 で portable int8 経路、既定で AVX2 int8 経路を実行する。
use crate::enc::{avx_on, D};
use crate::Bundle;
use anyhow::{bail, Context, Result};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;

const H: usize = 512;
const HEADS: usize = 4;
const FF: usize = 2048;
const STATE: usize = 768;
const EPS: f32 = 1e-6;
const MAX_LEN: usize = 256;
const KGRP: usize = 64;

fn fp32_mode() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        static FP32: OnceLock<bool> = OnceLock::new();
        *FP32.get_or_init(|| std::env::var_os("ZEIBAN_FP32").is_some())
    }
}

// 4-bit block16 [out / 16][inp][8]
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
            bail!("{name}: P4 形が [{out}, {inp}] でない");
        }
        Ok(Self { inp, blocks: out / 16, p, s })
    }
}

// 2-bit block16 [out / 16][inp][4]
pub struct P2 {
    inp: usize,
    blocks: usize,
    p: &'static [u8],
    s: &'static [f32],
}

impl P2 {
    fn load(b: &'static Bundle, name: &str, out: usize, inp: usize) -> Result<Self> {
        let (p, s) = (b.get::<u8>(name), b.get::<f32>(&format!("{name}.s")));
        if p.len() != out * inp / 4 || s.len() != out || out % 16 != 0 {
            bail!("{name}: P2 形が [{out}, {inp}] でない");
        }
        Ok(Self { inp, blocks: out / 16, p, s })
    }
}

// k128_v1 2-bit SEQ: [out][inp / 128][32]
pub struct P2K {
    out: usize,
    inp: usize,
    p: &'static [u8],
    s: &'static [f32],
}

impl P2K {
    fn load(b: &'static Bundle, name: &str, out: usize, inp: usize) -> Result<Self> {
        let (p, s) = (b.get::<u8>(name), b.get::<f32>(&format!("{name}.s")));
        if p.len() != out * inp / 4 || s.len() != out || out % 4 != 0 || inp % 128 != 0 {
            bail!("{name}: P2K 形が [{out}, {inp}] でない");
        }
        Ok(Self { out, inp, p, s })
    }
}

// k128_v1 4-bit: [out][inp / 64][32]
pub struct P4K {
    out: usize,
    inp: usize,
    p: &'static [u8],
    s: &'static [f32],
}

impl P4K {
    fn load(b: &'static Bundle, name: &str, out: usize, inp: usize) -> Result<Self> {
        let (p, s) = (b.get::<u8>(name), b.get::<f32>(&format!("{name}.s")));
        if p.len() != out * inp / 2 || s.len() != out || out % 4 != 0 || inp % 64 != 0 {
            bail!("{name}: P4K 形が [{out}, {inp}] でない");
        }
        Ok(Self { out, inp, p, s })
    }
}

pub enum Mat {
    Q4(P4),
    Q2(P2),
    Q4K(P4K),
    Q2K(P2K),
}

impl Mat {
    fn load(b: &'static Bundle, name: &str, out: usize, inp: usize, bits: usize, k128: bool) -> Result<Self> {
        match (bits, k128) {
            (4, false) => Ok(Self::Q4(P4::load(b, name, out, inp)?)),
            (2, false) => Ok(Self::Q2(P2::load(b, name, out, inp)?)),
            (4, true) => Ok(Self::Q4K(P4K::load(b, name, out, inp)?)),
            (2, true) => Ok(Self::Q2K(P2K::load(b, name, out, inp)?)),
            _ => bail!("{name}: 非対応ビット幅 {bits}"),
        }
    }
}

pub struct QuantX {
    pub n: usize,
    pub inp: usize,
    pub groups: usize,
    pub q: Vec<i8>,
    pub scale: Vec<f32>,
    pub sum: Vec<i32>,
}

macro_rules! kernels {
    ($($(#[$m:meta])* $v:vis fn $f:ident($($a:ident: $t:ty),*) $(-> $ret:ty)?;)*) => {$(
        $(#[$m])*
        $v fn $f($($a: $t),*) $(-> $ret)? {
            #[cfg(target_arch = "x86_64")]
            if avx_on() {
                // SAFETY: avx_on は CPU に AVX2 と FMA があるときだけ真
                return unsafe { avx::$f($($a),*) };
            }
            portable::$f($($a),*)
        }
    )*};
}

kernels! {
    fn quantize_x_k64(x: &[f32], n: usize, inp: usize) -> QuantX;
    fn linear4(p: &P4, x: &[f32], n: usize, y: &mut [f32]);
    fn linear2(p: &P2, x: &[f32], n: usize, y: &mut [f32]);
    fn linear2_q8(p: &P2K, x: &QuantX, y: &mut [f32]);
    fn linear4_q8(p: &P4K, rows: usize, x: &QuantX, y: &mut [f32]);
    fn linear2_k_fp32(p: &P2K, x: &[f32], n: usize, y: &mut [f32]);
    fn linear4_k_fp32(p: &P4K, rows: usize, x: &[f32], n: usize, y: &mut [f32]);
    fn rms(x: &[f32], w: &[f32], dim: usize, out: &mut [f32]);
    fn gelu_mul(a: &[f32], b: &[f32], out: &mut [f32]);
    fn attend(q: &[f32], k: &[f32], v: &[f32], n: usize, hd: usize, kv: usize, from: usize, out: &mut [f32]);
    pub fn dots4(q: &[f32], node: &[u8], out: &mut [f32]);
}

#[inline]
fn linear_mat_q8(m: &Mat, qx: Option<&QuantX>, x_fp: &[f32], n: usize, y: &mut [f32]) {
    match m {
        Mat::Q4(p) => linear4(p, x_fp, n, y),
        Mat::Q2(p) => linear2(p, x_fp, n, y),
        Mat::Q4K(p) => {
            if fp32_mode() {
                linear4_k_fp32(p, p.out, x_fp, n, y);
            } else {
                linear4_q8(p, p.out, qx.expect("QuantX"), y);
            }
        }
        Mat::Q2K(p) => {
            if fp32_mode() {
                linear2_k_fp32(p, x_fp, n, y);
            } else {
                linear2_q8(p, qx.expect("QuantX"), y);
            }
        }
    }
}

#[inline]
fn linear_mat(m: &Mat, x: &[f32], n: usize, y: &mut [f32]) {
    let qx = match (m, fp32_mode()) {
        (Mat::Q2K(p), false) => Some(quantize_x_k64(x, n, p.inp)),
        (Mat::Q4K(p), false) => Some(quantize_x_k64(x, n, p.inp)),
        _ => None,
    };
    linear_mat_q8(m, qx.as_ref(), x, n, y);
}

mod portable {
    use super::{QuantX, EPS, HEADS, KGRP, P2, P2K, P4, P4K};

    pub fn quantize_x_k64(x: &[f32], n: usize, inp: usize) -> QuantX {
        assert!(x.len() == n * inp && inp % KGRP == 0);
        let groups = inp / KGRP;
        let mut q = vec![0i8; n * inp];
        let mut scale = vec![0f32; n * groups];
        let mut sum = vec![0i32; n * groups];
        for r in 0..n {
            let xr = &x[r * inp..][..inp];
            let qr = &mut q[r * inp..][..inp];
            for g in 0..groups {
                let src = &xr[g * KGRP..][..KGRP];
                let dst = &mut qr[g * KGRP..][..KGRP];
                let mut amax = 0f32;
                for &v in src {
                    amax = amax.max(v.abs());
                }
                let sc = (amax / 127.0).max(1e-12);
                let inv = 1.0 / sc;
                let mut acc = 0i32;
                for j in 0..KGRP {
                    let qi = (src[j] * inv).round_ties_even().clamp(-127.0, 127.0) as i8;
                    dst[j] = qi;
                    acc += qi as i32;
                }
                scale[r * groups + g] = sc;
                sum[r * groups + g] = acc;
            }
        }
        QuantX { n, inp, groups, q, scale, sum }
    }

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

    #[inline]
    pub fn w16_p2(b: &[u8]) -> [f32; 16] {
        let mut w = [0f32; 16];
        for j in 0..4 {
            let x = b[j];
            w[4 * j] = ((x & 3) as i8 * 2 - 3) as f32;
            w[4 * j + 1] = (((x >> 2) & 3) as i8 * 2 - 3) as f32;
            w[4 * j + 2] = (((x >> 4) & 3) as i8 * 2 - 3) as f32;
            w[4 * j + 3] = (((x >> 6) & 3) as i8 * 2 - 3) as f32;
        }
        w
    }

    const MC: usize = 48;
    const RN: usize = 2;

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

    fn linear_blocked<const BPC: usize>(raw: &[u8], scales: &[f32], blocks: usize, inp: usize, x: &[f32], n: usize, y: &mut [f32], unpack: fn(&[u8]) -> [f32; 16]) {
        let out = blocks * 16;
        assert!(x.len() == n * inp && y.len() == n * out);
        let mut wf = vec![0f32; inp * 16];
        for r0 in (0..n).step_by(MC) {
            let r1 = (r0 + MC).min(n);
            for (b, w) in raw.chunks_exact(inp * BPC).enumerate() {
                for (dst, wl) in wf.chunks_exact_mut(16).zip(w.chunks_exact(BPC)) {
                    dst.copy_from_slice(&unpack(wl));
                }
                let s = &scales[b * 16..][..16];
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

    pub fn linear4(p: &P4, x: &[f32], n: usize, y: &mut [f32]) {
        linear_blocked::<8>(p.p, p.s, p.blocks, p.inp, x, n, y, w16);
    }

    pub fn linear2(p: &P2, x: &[f32], n: usize, y: &mut [f32]) {
        linear_blocked::<4>(p.p, p.s, p.blocks, p.inp, x, n, y, w16_p2);
    }

    pub fn linear2_q8(p: &P2K, x: &QuantX, y: &mut [f32]) {
        let (n, inp, out, groups) = (x.n, p.inp, p.out, x.groups);
        assert!(inp == x.inp && y.len() == n * out);
        let blocks128 = inp / 128;
        for r in 0..n {
            let xr = &x.q[r * inp..][..inp];
            for c in 0..out {
                let wr = &p.p[c * blocks128 * 32..][..blocks128 * 32];
                let mut f_acc = 0f32;
                for b in 0..blocks128 {
                    let wb = &wr[b * 32..][..32];
                    let xb = &xr[b * 128..][..128];
                    for h in 0..2 {
                        let mut dot = 0i32;
                        for s in 0..2 {
                            let sh = 2 * (h * 2 + s);
                            let xc = &xb[(h * 2 + s) * 32..][..32];
                            for j in 0..32 {
                                dot += (((wb[j] >> sh) & 3) as i32) * (xc[j] as i32);
                            }
                        }
                        let gi = r * groups + b * 2 + h;
                        f_acc += ((2 * dot - 3 * x.sum[gi]) as f32) * x.scale[gi];
                    }
                }
                y[r * out + c] += f_acc * p.s[c];
            }
        }
    }

    pub fn linear4_q8(p: &P4K, rows: usize, x: &QuantX, y: &mut [f32]) {
        let (n, inp, groups) = (x.n, p.inp, x.groups);
        assert!(inp == x.inp && rows <= p.out && y.len() == n * rows);
        let row_bytes = inp / 2;
        for r in 0..n {
            let xr = &x.q[r * inp..][..inp];
            for c in 0..rows {
                let wr = &p.p[c * row_bytes..][..row_bytes];
                let mut f_acc = 0f32;
                for g in 0..groups {
                    let wh = &wr[g * 32..][..32];
                    let x0 = &xr[g * KGRP..][..32];
                    let x1 = &xr[g * KGRP + 32..][..32];
                    let mut dot = 0i32;
                    for j in 0..32 {
                        let u0 = (wh[j] & 15) as i32;
                        let u1 = (wh[j] >> 4) as i32;
                        dot += u0 * (x0[j] as i32) + u1 * (x1[j] as i32);
                    }
                    let gi = r * groups + g;
                    f_acc += ((dot - 8 * x.sum[gi]) as f32) * x.scale[gi];
                }
                y[r * rows + c] += f_acc * p.s[c];
            }
        }
    }

    pub fn linear2_k_fp32(p: &P2K, x: &[f32], n: usize, y: &mut [f32]) {
        let (inp, out, blocks128) = (p.inp, p.out, p.inp / 128);
        assert!(x.len() == n * inp && y.len() == n * out);
        for r in 0..n {
            let xr = &x[r * inp..][..inp];
            for c in 0..out {
                let wr = &p.p[c * blocks128 * 32..][..blocks128 * 32];
                let mut acc = 0f32;
                for b in 0..blocks128 {
                    let wg = &wr[b * 32..][..32];
                    let xg = &xr[b * 128..][..128];
                    for s in 0..4 {
                        let sh = 2 * s;
                        let xc = &xg[s * 32..][..32];
                        for j in 0..32 {
                            let w = (((wg[j] >> sh) & 3) as i8 * 2 - 3) as f32;
                            acc += xc[j] * w;
                        }
                    }
                }
                y[r * out + c] += acc * p.s[c];
            }
        }
    }

    pub fn linear4_k_fp32(p: &P4K, rows: usize, x: &[f32], n: usize, y: &mut [f32]) {
        let (inp, row_bytes) = (p.inp, p.inp / 2);
        assert!(x.len() == n * inp && rows <= p.out && y.len() == n * rows);
        let halves = inp / 64;
        for r in 0..n {
            let xr = &x[r * inp..][..inp];
            for c in 0..rows {
                let wr = &p.p[c * row_bytes..][..row_bytes];
                let mut acc = 0f32;
                for h in 0..halves {
                    let wh = &wr[h * 32..][..32];
                    let x0 = &xr[h * 64..][..32];
                    let x1 = &xr[h * 64 + 32..][..32];
                    for j in 0..32 {
                        let w0 = ((wh[j] & 15) as i8 - 8) as f32;
                        let w1 = ((wh[j] >> 4) as i8 - 8) as f32;
                        acc += x0[j] * w0 + x1[j] * w1;
                    }
                }
                y[r * rows + c] += acc * p.s[c];
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

#[cfg(target_arch = "x86_64")]
mod avx {
    use super::{QuantX, EPS, HEADS, KGRP, P2, P2K, P4, P4K};
    use std::arch::x86_64::*;

    const MC: usize = 96;
    const KC: usize = 256;

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hmax_ps(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let m128 = _mm_max_ps(lo, hi);
        let m64 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
        let m32 = _mm_max_ss(m64, _mm_shuffle_ps(m64, m64, 1));
        _mm_cvtss_f32(m32)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum_i32(v: __m256i) -> i32 {
        let s128 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        let s64 = _mm_add_epi32(s128, _mm_srli_si128(s128, 8));
        let s32 = _mm_add_epi32(s64, _mm_srli_si128(s64, 4));
        _mm_cvtsi128_si32(s32)
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn quantize_x_k64(x: &[f32], n: usize, inp: usize) -> QuantX {
        assert!(x.len() == n * inp && inp % KGRP == 0);
        let groups = inp / KGRP;
        let mut q = vec![0i8; n * inp];
        let mut scale = vec![0f32; n * groups];
        let mut sum = vec![0i32; n * groups];
        let absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffffu32 as i32));
        let vmin = _mm256_set1_ps(-127.0);
        let vmax = _mm256_set1_ps(127.0);
        let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        let xp = x.as_ptr();
        let qp = q.as_mut_ptr();

        for r in 0..n {
            for g in 0..groups {
                let sptr = xp.add(r * inp + g * KGRP);
                let dptr = qp.add(r * inp + g * KGRP);
                let mut mv0 = _mm256_and_ps(_mm256_loadu_ps(sptr), absmask);
                let mut mv1 = _mm256_and_ps(_mm256_loadu_ps(sptr.add(8)), absmask);
                for k in 1..4 {
                    mv0 = _mm256_max_ps(mv0, _mm256_and_ps(_mm256_loadu_ps(sptr.add(k * 16)), absmask));
                    mv1 = _mm256_max_ps(mv1, _mm256_and_ps(_mm256_loadu_ps(sptr.add(k * 16 + 8)), absmask));
                }
                let amax = hmax_ps(_mm256_max_ps(mv0, mv1));
                let sc = (amax / 127.0).max(1e-12);
                let vinv = _mm256_set1_ps(1.0 / sc);
                let mut sumv = _mm256_setzero_si256();
                for half in 0..2 {
                    let hp = sptr.add(half * 32);
                    let f0 = _mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(hp), vinv), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    let f1 = _mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(hp.add(8)), vinv), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    let f2 = _mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(hp.add(16)), vinv), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    let f3 = _mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(hp.add(24)), vinv), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    let i0 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(f0, vmin), vmax));
                    let i1 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(f1, vmin), vmax));
                    let i2 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(f2, vmin), vmax));
                    let i3 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(f3, vmin), vmax));
                    sumv = _mm256_add_epi32(sumv, _mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3)));
                    let p01 = _mm256_packs_epi32(i0, i1);
                    let p23 = _mm256_packs_epi32(i2, i3);
                    let packed = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(p01, p23), perm);
                    _mm256_storeu_si256(dptr.add(half * 32) as *mut __m256i, packed);
                }
                scale[r * groups + g] = sc;
                sum[r * groups + g] = hsum_i32(sumv);
            }
        }
        QuantX { n, inp, groups, q, scale, sum }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn w16(p: *const u8) -> (__m256, __m256) {
        let v = _mm_loadl_epi64(p as *const __m128i);
        let m = _mm_set1_epi8(0x0f);
        let b = _mm_sub_epi8(_mm_unpacklo_epi8(_mm_and_si128(v, m), _mm_and_si128(_mm_srli_epi16(v, 4), m)), _mm_set1_epi8(8));
        (_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b)), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b, 8))))
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn w16_p2(p: *const u8) -> (__m256, __m256) {
        let raw = _mm256_set1_epi32(std::ptr::read_unaligned(p as *const u32) as i32);
        let b01 = _mm256_shuffle_epi8(
            raw,
            _mm256_setr_epi8(0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1),
        );
        let b23 = _mm256_shuffle_epi8(
            raw,
            _mm256_setr_epi8(2, -1, -1, -1, 2, -1, -1, -1, 2, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1, 3, -1, -1, -1, 3, -1, -1, -1, 3, -1, -1, -1),
        );
        let sh = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
        let mask3 = _mm256_set1_epi32(3);
        let three = _mm256_set1_epi32(3);
        let c0 = _mm256_sub_epi32(_mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(b01, sh), mask3), 1), three);
        let c1 = _mm256_sub_epi32(_mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(b23, sh), mask3), 1), three);
        (_mm256_cvtepi32_ps(c0), _mm256_cvtepi32_ps(c1))
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn linear_blocked<const IS_P2: bool>(raw: &[u8], scales: &[f32], blocks: usize, inp: usize, x: &[f32], n: usize, y: &mut [f32]) {
        let bpc = if IS_P2 { 4 } else { 8 };
        let (out, xp, yp) = (blocks * 16, x.as_ptr(), y.as_mut_ptr());
        assert!(x.len() == n * inp && y.len() == n * out);
        for k0 in (0..inp).step_by(KC) {
            let kc = KC.min(inp - k0);
            for r0 in (0..n).step_by(MC) {
                let r1 = (r0 + MC).min(n);
                for b in 0..blocks {
                    let (w, s) = (raw.as_ptr().add((b * inp + k0) * bpc), scales.as_ptr().add(b * 16));
                    let mut r = r0;
                    while r + 6 <= r1 {
                        kern6::<IS_P2>(kc, xp.add(r * inp + k0), inp, w, s, yp.add(r * out + b * 16), out);
                        r += 6;
                    }
                    while r < r1 {
                        kern1::<IS_P2>(kc, xp.add(r * inp + k0), w, s, yp.add(r * out + b * 16));
                        r += 1;
                    }
                }
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn kern6<const IS_P2: bool>(k: usize, x: *const f32, xs: usize, p: *const u8, s: *const f32, y: *mut f32, ys: usize) {
        let bpc = if IS_P2 { 4 } else { 8 };
        let mut a = [[_mm256_setzero_ps(); 2]; 6];
        for l in 0..k {
            let (w0, w1) = if IS_P2 { w16_p2(p.add(l * bpc)) } else { w16(p.add(l * bpc)) };
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
    unsafe fn kern1<const IS_P2: bool>(k: usize, x: *const f32, p: *const u8, s: *const f32, y: *mut f32) {
        let bpc = if IS_P2 { 4 } else { 8 };
        let (mut a0, mut a1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        for l in 0..k {
            let b = _mm256_broadcast_ss(&*x.add(l));
            let (w0, w1) = if IS_P2 { w16_p2(p.add(l * bpc)) } else { w16(p.add(l * bpc)) };
            a0 = _mm256_fmadd_ps(b, w0, a0);
            a1 = _mm256_fmadd_ps(b, w1, a1);
        }
        _mm256_storeu_ps(y, _mm256_fmadd_ps(a0, _mm256_loadu_ps(s), _mm256_loadu_ps(y)));
        _mm256_storeu_ps(y.add(8), _mm256_fmadd_ps(a1, _mm256_loadu_ps(s.add(8)), _mm256_loadu_ps(y.add(8))));
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn linear4(p: &P4, x: &[f32], n: usize, y: &mut [f32]) {
        linear_blocked::<false>(p.p, p.s, p.blocks, p.inp, x, n, y);
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn linear2(p: &P2, x: &[f32], n: usize, y: &mut [f32]) {
        linear_blocked::<true>(p.p, p.s, p.blocks, p.inp, x, n, y);
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_u8_k64_c4<const MULT: i32, const OFF: i32>(
        u_buf: *const u8,
        x: &QuantX,
        r0: usize,
        r1: usize,
        s4: *const f32,
        yp: *mut f32,
        rows: usize,
        c: usize,
    ) {
        let (inp, groups) = (x.inp, x.groups);
        let ones16 = _mm256_set1_epi16(1);
        let xp = x.q.as_ptr();
        let scp = x.scale.as_ptr();
        let smp = x.sum.as_ptr();
        let sv = _mm_loadu_ps(s4);

        let mut r = r0;
        while r + 2 <= r1 {
            let mut f0 = _mm_setzero_ps();
            let mut f1 = _mm_setzero_ps();
            for g in 0..groups {
                let x0p = xp.add(r * inp + g * KGRP);
                let x1p = xp.add((r + 1) * inp + g * KGRP);
                let x0a = _mm256_loadu_si256(x0p as *const __m256i);
                let x0b = _mm256_loadu_si256(x0p.add(32) as *const __m256i);
                let x1a = _mm256_loadu_si256(x1p as *const __m256i);
                let x1b = _mm256_loadu_si256(x1p.add(32) as *const __m256i);

                let mut d0 = [_mm256_setzero_si256(); 4];
                let mut d1 = [_mm256_setzero_si256(); 4];
                for k in 0..4 {
                    let up = u_buf.add(k * inp + g * KGRP);
                    let ua = _mm256_loadu_si256(up as *const __m256i);
                    let ub = _mm256_loadu_si256(up.add(32) as *const __m256i);
                    d0[k] = _mm256_madd_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(ua, x0a), _mm256_maddubs_epi16(ub, x0b)), ones16);
                    d1[k] = _mm256_madd_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(ua, x1a), _mm256_maddubs_epi16(ub, x1b)), ones16);
                }
                let s0123_0 = _mm256_hadd_epi32(_mm256_hadd_epi32(d0[0], d0[1]), _mm256_hadd_epi32(d0[2], d0[3]));
                let s0123_1 = _mm256_hadd_epi32(_mm256_hadd_epi32(d1[0], d1[1]), _mm256_hadd_epi32(d1[2], d1[3]));
                let dots0 = _mm_add_epi32(_mm256_castsi256_si128(s0123_0), _mm256_extracti128_si256(s0123_0, 1));
                let dots1 = _mm_add_epi32(_mm256_castsi256_si128(s0123_1), _mm256_extracti128_si256(s0123_1, 1));

                let (dots0_m, dots1_m) = if MULT == 2 {
                    (_mm_slli_epi32(dots0, 1), _mm_slli_epi32(dots1, 1))
                } else {
                    (dots0, dots1)
                };
                let gi0 = r * groups + g;
                let gi1 = (r + 1) * groups + g;
                let adj0 = _mm_sub_epi32(dots0_m, _mm_set1_epi32(OFF * *smp.add(gi0)));
                let adj1 = _mm_sub_epi32(dots1_m, _mm_set1_epi32(OFF * *smp.add(gi1)));
                f0 = _mm_fmadd_ps(_mm_cvtepi32_ps(adj0), _mm_set1_ps(*scp.add(gi0)), f0);
                f1 = _mm_fmadd_ps(_mm_cvtepi32_ps(adj1), _mm_set1_ps(*scp.add(gi1)), f1);
            }
            let y0 = yp.add(r * rows + c);
            let y1 = yp.add((r + 1) * rows + c);
            _mm_storeu_ps(y0, _mm_fmadd_ps(f0, sv, _mm_loadu_ps(y0)));
            _mm_storeu_ps(y1, _mm_fmadd_ps(f1, sv, _mm_loadu_ps(y1)));
            r += 2;
        }
        while r < r1 {
            let mut f0 = _mm_setzero_ps();
            for g in 0..groups {
                let x0p = xp.add(r * inp + g * KGRP);
                let x0a = _mm256_loadu_si256(x0p as *const __m256i);
                let x0b = _mm256_loadu_si256(x0p.add(32) as *const __m256i);
                let mut d0 = [_mm256_setzero_si256(); 4];
                for k in 0..4 {
                    let up = u_buf.add(k * inp + g * KGRP);
                    let ua = _mm256_loadu_si256(up as *const __m256i);
                    let ub = _mm256_loadu_si256(up.add(32) as *const __m256i);
                    d0[k] = _mm256_madd_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(ua, x0a), _mm256_maddubs_epi16(ub, x0b)), ones16);
                }
                let s0123 = _mm256_hadd_epi32(_mm256_hadd_epi32(d0[0], d0[1]), _mm256_hadd_epi32(d0[2], d0[3]));
                let dots0 = _mm_add_epi32(_mm256_castsi256_si128(s0123), _mm256_extracti128_si256(s0123, 1));
                let dots0_m = if MULT == 2 { _mm_slli_epi32(dots0, 1) } else { dots0 };
                let gi0 = r * groups + g;
                let adj0 = _mm_sub_epi32(dots0_m, _mm_set1_epi32(OFF * *smp.add(gi0)));
                f0 = _mm_fmadd_ps(_mm_cvtepi32_ps(adj0), _mm_set1_ps(*scp.add(gi0)), f0);
            }
            let y0 = yp.add(r * rows + c);
            _mm_storeu_ps(y0, _mm_fmadd_ps(f0, sv, _mm_loadu_ps(y0)));
            r += 1;
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn linear2_q8(p: &P2K, x: &QuantX, y: &mut [f32]) {
        const TILE_M: usize = 48;
        let (n, inp, out) = (x.n, p.inp, p.out);
        assert!(inp == x.inp && y.len() == n * out && out % 4 == 0 && inp % 128 == 0);
        let blocks128 = inp / 128;
        let row_bytes = blocks128 * 32;
        let mask3 = _mm256_set1_epi8(0x03);
        let wp = p.p.as_ptr();
        let yp = y.as_mut_ptr();
        let mut u_buf = vec![0u8; 4 * inp];
        let ubp = u_buf.as_mut_ptr();

        let mut c = 0usize;
        while c + 4 <= out {
            for k in 0..4 {
                let wr = wp.add((c + k) * row_bytes);
                let dst_r = ubp.add(k * inp);
                for b in 0..blocks128 {
                    let raw = _mm256_loadu_si256(wr.add(b * 32) as *const __m256i);
                    let db = dst_r.add(b * 128);
                    _mm256_storeu_si256(db as *mut __m256i, _mm256_and_si256(raw, mask3));
                    _mm256_storeu_si256(db.add(32) as *mut __m256i, _mm256_and_si256(_mm256_srli_epi16(raw, 2), mask3));
                    _mm256_storeu_si256(db.add(64) as *mut __m256i, _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask3));
                    _mm256_storeu_si256(db.add(96) as *mut __m256i, _mm256_and_si256(_mm256_srli_epi16(raw, 6), mask3));
                }
            }
            for r0 in (0..n).step_by(TILE_M) {
                let r1 = (r0 + TILE_M).min(n);
                dot_u8_k64_c4::<2, 3>(ubp, x, r0, r1, p.s.as_ptr().add(c), yp, out, c);
            }
            c += 4;
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn linear4_q8(p: &P4K, rows: usize, x: &QuantX, y: &mut [f32]) {
        const TILE_M: usize = 48;
        let (n, inp, groups) = (x.n, p.inp, x.groups);
        assert!(inp == x.inp && rows <= p.out && y.len() == n * rows && rows % 4 == 0 && inp % 64 == 0);
        let row_bytes = inp / 2;
        let maskf = _mm256_set1_epi8(0x0f);
        let wp = p.p.as_ptr();
        let yp = y.as_mut_ptr();
        let mut u_buf = vec![0u8; 4 * inp];
        let ubp = u_buf.as_mut_ptr();

        let mut c = 0usize;
        while c + 4 <= rows {
            for k in 0..4 {
                let wr = wp.add((c + k) * row_bytes);
                let dst_r = ubp.add(k * inp);
                for g in 0..groups {
                    let raw = _mm256_loadu_si256(wr.add(g * 32) as *const __m256i);
                    let dg = dst_r.add(g * KGRP);
                    _mm256_storeu_si256(dg as *mut __m256i, _mm256_and_si256(raw, maskf));
                    _mm256_storeu_si256(dg.add(32) as *mut __m256i, _mm256_and_si256(_mm256_srli_epi16(raw, 4), maskf));
                }
            }
            for r0 in (0..n).step_by(TILE_M) {
                let r1 = (r0 + TILE_M).min(n);
                dot_u8_k64_c4::<1, 8>(ubp, x, r0, r1, p.s.as_ptr().add(c), yp, rows, c);
            }
            c += 4;
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn linear2_k_fp32(p: &P2K, x: &[f32], n: usize, y: &mut [f32]) {
        super::portable::linear2_k_fp32(p, x, n, y);
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn linear4_k_fp32(p: &P4K, rows: usize, x: &[f32], n: usize, y: &mut [f32]) {
        super::portable::linear4_k_fp32(p, rows, x, n, y);
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)))
    }

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

enum PleProj {
    B16(P4),
    K64(P4K),
}

struct Layer {
    hd: usize,
    kv: usize,
    rope: usize,
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    gate: Mat,
    up: Mat,
    down: Mat,
    pg: Mat,
    pp: Mat,
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
    fn pre(&self, x: &[f32], q: &mut [f32], k: &mut [f32], v: &mut [f32]) {
        let n = x.len() / H;
        let mut h = vec![0f32; n * H];
        rms(x, self.n_in, H, &mut h);
        let use_q8 = !fp32_mode() && matches!(self.q, Mat::Q2K(_) | Mat::Q4K(_));
        let qh = if use_q8 { Some(quantize_x_k64(&h, n, H)) } else { None };
        linear_mat_q8(&self.q, qh.as_ref(), &h, n, q);
        linear_mat_q8(&self.k, qh.as_ref(), &h, n, k);
        linear_mat_q8(&self.v, qh.as_ref(), &h, n, v);
    }

    fn post(&self, x: &mut [f32], attn: &[f32], ple: &[f32]) {
        let n = x.len() / H;
        let (mut a, mut b) = (vec![0f32; n * H], vec![0f32; n * H]);
        linear_mat(&self.o, attn, n, &mut a);
        rms(&a, self.n_att, H, &mut b);
        add(x, &b);
        rms(x, self.n_pre, H, &mut a);

        const ROWS: usize = 48;
        let r_max = n.min(ROWS);
        let (mut g, mut u, mut m) = (vec![0f32; r_max * FF], vec![0f32; r_max * FF], vec![0f32; r_max * FF]);
        b.fill(0.0);
        let use_q8 = !fp32_mode() && matches!(self.gate, Mat::Q2K(_) | Mat::Q4K(_));
        for (ac, bc) in a.chunks(ROWS * H).zip(b.chunks_mut(ROWS * H)) {
            let r = ac.len() / H;
            let (g, u, m) = (&mut g[..r * FF], &mut u[..r * FF], &mut m[..r * FF]);
            g.fill(0.0);
            u.fill(0.0);
            let qac = if use_q8 { Some(quantize_x_k64(ac, r, H)) } else { None };
            linear_mat_q8(&self.gate, qac.as_ref(), ac, r, g);
            linear_mat_q8(&self.up, qac.as_ref(), ac, r, u);
            gelu_mul(g, u, m);
            linear_mat(&self.down, m, r, bc);
        }
        drop((g, u, m));
        rms(&b, self.n_ff, H, &mut a);
        add(x, &a);

        a.fill(0.0);
        linear_mat(&self.pg, x, n, &mut a);
        gelu_mul(&a, ple, &mut b);
        a.fill(0.0);
        linear_mat(&self.pp, &b, n, &mut a);
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
    ple: PleProj,
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
        avx_on();
        let nl = b.man["enc"]["layers"].as_u64().context("目録の enc.layers")? as usize;
        let wlayout = b.man["enc"]["weight_layout"].as_str();
        let k128 = wlayout == Some("k128_v1");
        let hybrid = wlayout == Some("hybrid_k128_v1");
        let bits_arr: Vec<usize> = match b.man["enc"]["layer_bits"].as_array() {
            Some(a) => a.iter().map(|v| v.as_u64().unwrap_or(4) as usize).collect(),
            None => vec![b.man["enc"]["bits"].as_u64().unwrap_or(4) as usize; nl],
        };
        let mut full_attn = vec![false; nl];
        if let Some(fa) = b.man["enc"]["full_attn_layers"].as_array() {
            for v in fa {
                if let Some(i) = v.as_u64() {
                    if (i as usize) < nl {
                        full_attn[i as usize] = true;
                    }
                }
            }
        } else if nl > 0 {
            full_attn[nl - 1] = true;
        }
        if bits_arr.len() != nl {
            bail!("enc.layer_bits 長さ {} != {}", bits_arr.len(), nl);
        }

        let mut layers = Vec::with_capacity(nl);
        for i in 0..nl {
            let (hd, kv, rope) = if full_attn[i] { (512, 1, 1) } else { (256, 2, 0) };
            let bits = bits_arr[i];
            let use_k128 = k128 || (hybrid && bits == 2);
            let lin = |k: &str, out: usize, inp: usize| Mat::load(b, &format!("g.l{i}.{k}"), out, inp, bits, use_k128);
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
        let ple = if k128 {
            PleProj::K64(P4K::load(b, "g.ple", nl * H, H)?)
        } else {
            PleProj::B16(P4::load(b, "g.ple", nl * H, H)?)
        };
        let e = Self {
            emb: b.get("g.emb"),
            emb_s: b.get("g.emb.s"),
            ple,
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

    pub fn query(&self, ids: &[u32]) -> Vec<f32> {
        let n = ids.len();
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
        let half = n / 2;
        let ple_w = self.layers.len() * H;
        let mut raw = vec![0f32; n * ple_w];
        match &self.ple {
            PleProj::B16(p) => {
                let ((xa, xb), (ra, rb)) = (x.split_at(half * H), raw.split_at_mut(half * ple_w));
                rayon::join(|| linear4(p, xa, half, ra), || linear4(p, xb, n - half, rb));
            }
            PleProj::K64(p) => {
                let ((xa, xb), (ra, rb)) = (x.split_at(half * H), raw.split_at_mut(half * ple_w));
                if fp32_mode() {
                    rayon::join(
                        || linear4_k_fp32(p, ple_w, xa, half, ra),
                        || linear4_k_fp32(p, ple_w, xb, n - half, rb),
                    );
                } else {
                    let (qxa, qxb) = rayon::join(
                        || if half > 0 { Some(quantize_x_k64(xa, half, H)) } else { None },
                        || quantize_x_k64(xb, n - half, H),
                    );
                    rayon::join(
                        || {
                            if let Some(q) = &qxa {
                                linear4_q8(p, ple_w, q, ra);
                            }
                        },
                        || linear4_q8(p, ple_w, &qxb, rb),
                    );
                }
            }
        }
        let sc = (H as f32).powf(-0.5);
        raw.iter_mut().for_each(|v| *v *= sc);
        let mut ple = vec![0f32; n * ple_w];
        rms(&raw, self.ple_norm, H, &mut ple);
        drop(raw);

        let mut pl = vec![0f32; n * H];
        for (i, l) in self.layers.iter().enumerate() {
            for (dst, src) in pl.chunks_exact_mut(H).zip(ple.chunks_exact(ple_w)) {
                dst.copy_from_slice(&src[i * H..][..H]);
            }
            self.layer(l, &mut x, &pl, n);
        }

        let mut xn = vec![0f32; n * H];
        rms(&x, self.norm, H, &mut xn);
        let mut y = vec![0f32; n * STATE];
        {
            let ((xa, xb), (ya, yb)) = (xn.split_at(half * H), y.split_at_mut(half * STATE));
            rayon::join(|| linear4(&self.eproj, xa, half, ya), || linear4(&self.eproj, xb, n - half, yb));
        }
        let mut state = vec![0f32; STATE];
        for r in y.chunks_exact(STATE) {
            add(&mut state, r);
        }
        state.iter_mut().for_each(|v| *v /= n as f32);
        let mut q = self.q_bias.to_vec();
        linear4(&self.q_proj, &state, 1, &mut q);
        for v in q.chunks_exact_mut(D) {
            let len = v.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-12);
            v.iter_mut().for_each(|a| *a /= len);
        }
        q
    }

    fn layer(&self, l: &Layer, x: &mut [f32], ple: &[f32], n: usize) {
        let (h, hd, kv) = (n / 2, l.hd, l.kv);
        let (qw, kw) = (HEADS * hd, kv * hd);
        let (mut qp, mut kp, mut vp) = (vec![0f32; n * qw], vec![0f32; n * kw], vec![0f32; n * kw]);
        {
            let (xa, xb) = x.split_at(h * H);
            let ((qa, qb), (ka, kb), (va, vb)) = (qp.split_at_mut(h * qw), kp.split_at_mut(h * kw), vp.split_at_mut(h * kw));
            rayon::join(|| l.pre(xa, qa, ka, va), || l.pre(xb, qb, kb, vb));
        }
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

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_and_q8_k64_exact() {
        let n = 7usize;
        let inp = 512usize;
        let out = 64usize;
        let mut x = vec![0f32; n * inp];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i % 97) as f32 - 48.0) * 0.03;
        }
        let qp = portable::quantize_x_k64(&x, n, inp);
        let qa = unsafe { avx::quantize_x_k64(&x, n, inp) };
        assert_eq!(qp.q, qa.q);
        assert_eq!(qp.sum, qa.sum);
        for (a, b) in qp.scale.iter().zip(&qa.scale) {
            assert!((a - b).abs() < 1e-7);
        }

        let s: Vec<f32> = (0..out).map(|c| 0.01 + (c % 7) as f32 * 0.005).collect();
        let w2: Vec<u8> = (0..out * (inp / 4)).map(|i| ((i * 73 + 19) & 0xff) as u8).collect();
        let p2 = P2K { out, inp, p: Box::leak(w2.into_boxed_slice()), s: Box::leak(s.clone().into_boxed_slice()) };
        let (mut y2p, mut y2a) = (vec![0f32; n * out], vec![0f32; n * out]);
        portable::linear2_q8(&p2, &qp, &mut y2p);
        unsafe { avx::linear2_q8(&p2, &qa, &mut y2a) };
        let d2 = y2p.iter().zip(&y2a).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(d2 < 1e-4, "P2K max diff = {d2}");

        let w4: Vec<u8> = (0..out * (inp / 2)).map(|i| (((i * 37) % 15 + 1) | (((i * 53) % 15 + 1) << 4)) as u8).collect();
        let p4 = P4K { out, inp, p: Box::leak(w4.into_boxed_slice()), s: Box::leak(s.into_boxed_slice()) };
        let (mut y4p, mut y4a) = (vec![0f32; n * out], vec![0f32; n * out]);
        portable::linear4_q8(&p4, out, &qp, &mut y4p);
        unsafe { avx::linear4_q8(&p4, out, &qa, &mut y4a) };
        let d4 = y4p.iter().zip(&y4a).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(d4 < 1e-4, "P4K max diff = {d4}");

        let n2 = 48usize;
        let inp2 = 2048usize;
        let out2 = 512usize;
        let mut xb = vec![0f32; n2 * inp2];
        for (i, v) in xb.iter_mut().enumerate() {
            *v = if i % 13 == 0 { 12.7 } else if i % 13 == 1 { -12.7 } else { ((i % 255) as f32 - 127.0) * 0.08 };
        }
        let qbp = portable::quantize_x_k64(&xb, n2, inp2);
        let qba = unsafe { avx::quantize_x_k64(&xb, n2, inp2) };
        assert_eq!(qbp.q, qba.q);
        assert_eq!(qbp.sum, qba.sum);
        let sb: Vec<f32> = (0..out2).map(|c| 0.02 + (c % 5) as f32 * 0.01).collect();
        let wb2: Vec<u8> = (0..out2 * (inp2 / 4)).map(|i| ((i * 97 + 31) & 0xff) as u8).collect();
        let pb2 = P2K { out: out2, inp: inp2, p: Box::leak(wb2.into_boxed_slice()), s: Box::leak(sb.clone().into_boxed_slice()) };
        let (mut y2bp, mut y2ba) = (vec![0f32; n2 * out2], vec![0f32; n2 * out2]);
        portable::linear2_q8(&pb2, &qbp, &mut y2bp);
        unsafe { avx::linear2_q8(&pb2, &qba, &mut y2ba) };
        let d2b = y2bp.iter().zip(&y2ba).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(d2b < 1e-3, "P2K 2048 max diff = {d2b}");

        let wb4: Vec<u8> = (0..out2 * (inp2 / 2)).map(|i| (((i * 13) % 15 + 1) | (((i * 29) % 15 + 1) << 4)) as u8).collect();
        let pb4 = P4K { out: out2, inp: inp2, p: Box::leak(wb4.into_boxed_slice()), s: Box::leak(sb.into_boxed_slice()) };
        let (mut y4bp, mut y4ba) = (vec![0f32; n2 * out2], vec![0f32; n2 * out2]);
        portable::linear4_q8(&pb4, out2, &qbp, &mut y4bp);
        unsafe { avx::linear4_q8(&pb4, out2, &qba, &mut y4ba) };
        let d4b = y4bp.iter().zip(&y4ba).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(d4b < 1e-3, "P4K 2048 max diff = {d4b}");
    }
}
