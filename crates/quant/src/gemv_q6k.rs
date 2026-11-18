//! Fused `Q6_K` x `q8_K` dequant-GEMV kernels (single token and the dispatch).
//!
//! `q6k` holds the still-quantized `Q6_K` bytes (`[out_rows x in_dim]`, packed as
//! `Q6_K` super-blocks). The activation `x` is quantized to `q8_K` **once**, then
//! each output row is an integer dot of the 6-bit `q4` weights (signed `-32..31`,
//! reconstructed on the fly) against the int8 activation, scaled by the per-group
//! q6 scale, with the `-32*bsum` correction folded in: no f32 dequant expansion.
//! `attn_qkv` and `ffn_down` are `Q6_K` in this model, so this closes the bulk of
//! the remaining per-token weight traffic after the Q4_K integer kernel.

use super::{
	features, fp16_to_f32, hsum_float_8, par_rows, q6k_block_bytes, quantize_q8k, Q8K, QK_K,
};

/// Shuffle table for `get_scale_shuffle_k6` (llama.cpp). Reads at byte offset
/// `i*16` give a 16-byte mask selecting `scales[2*i]` for lanes 0-7 and
/// `scales[2*i+1]` for lanes 8-15 (pointer arithmetic is in `__m128i` units, so
/// `k_shuffle + i` is `i*16` bytes). Covers the 16 q6 sub-scales.
const fn build_scale_shuf_k6() -> [u8; 128] {
	let mut t = [0u8; 128];
	let mut i = 0usize;
	while i < 8 {
		let lo = (2 * i) as u8;
		let hi = (2 * i + 1) as u8;
		let mut j = 0usize;
		while j < 8 {
			t[i * 16 + j] = lo;
			t[i * 16 + 8 + j] = hi;
			j += 1;
		}
		i += 1;
	}
	t
}
static SCALE_SHUF_K6: [u8; 128] = build_scale_shuf_k6();

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) unsafe fn get_scale_shuffle_k6(i: usize) -> std::arch::x86_64::__m128i {
	std::arch::x86_64::_mm_loadu_si128(
		SCALE_SHUF_K6.as_ptr().add(i * 16) as *const std::arch::x86_64::__m128i
	)
}

/// Fused Q6_K x q8_K dequant-GEMV for one output row, scalar fallback. Faithful
/// port of llama.cpp `ggml_vec_dot_q6_K_q8_K_generic`: the 6-bit signed weight
/// `q = (low4 | high2<<4) - 32` is dotted against the int8 activation, scaled by
/// each per-16-group q6 scale. `q8` is the activation quantized once to `q8_K`.
pub(crate) fn q6k_dot_q8k_scalar(row: &[u8], nb: usize, q8: &Q8K) -> f32 {
	let blk = q6k_block_bytes();
	let mut sumf = 0.0f32;
	for i in 0..nb {
		let rb = &row[i * blk..i * blk + blk];
		let d = q8.ds[i] * fp16_to_f32(u16::from_le_bytes([rb[208], rb[209]]));
		let ql = &rb[0..128];
		let qh = &rb[128..192];
		let scales = &rb[192..208];
		let q8d = &q8.qs[i * QK_K..(i + 1) * QK_K];
		let mut aux = [0i8; 256];
		let mut a = 0usize;
		let mut q4 = 0usize;
		let mut qh_i = 0usize;
		for _jj in 0..2 {
			for l in 0..32 {
				let v0 = (ql[q4 + l] & 0xF) as i32 | (((qh[qh_i + l] & 3) << 4) as i32);
				let v1 = (ql[q4 + l + 32] & 0xF) as i32 | ((((qh[qh_i + l] >> 2) & 3) << 4) as i32);
				let v2 =
					((ql[q4 + l] >> 4) & 0xF) as i32 | ((((qh[qh_i + l] >> 4) & 3) << 4) as i32);
				let v3 = ((ql[q4 + l + 32] >> 4) & 0xF) as i32
					| ((((qh[qh_i + l] >> 6) & 3) << 4) as i32);
				aux[a + l] = (v0 - 32) as i8;
				aux[a + l + 32] = (v1 - 32) as i8;
				aux[a + l + 64] = (v2 - 32) as i8;
				aux[a + l + 96] = (v3 - 32) as i8;
			}
			a += 128;
			q4 += 64;
			qh_i += 32;
		}
		let mut aux32 = [0i32; 8];
		let mut is = 0usize;
		let mut q = 0usize;
		for _jj in 0..(QK_K / 16) {
			let scale = scales[is] as i8 as i32;
			is += 1;
			let mut aux16 = [0i16; 8];
			for l in 0..8 {
				aux16[l] = q8d[q + l] as i16 * aux[q + l] as i16;
			}
			for l in 0..8 {
				aux32[l] += scale * aux16[l] as i32;
			}
			q += 8;
			for l in 0..8 {
				aux16[l] = q8d[q + l] as i16 * aux[q + l] as i16;
			}
			for l in 0..8 {
				aux32[l] += scale * aux16[l] as i32;
			}
			q += 8;
		}
		let mut s = 0.0f32;
		for l in 0..8 {
			s += aux32[l] as f32;
		}
		sumf += d * s;
	}
	sumf
}

/// **Fused Q6_K dequant-GEMV** (llama.cpp `ggml_vec_dot_q6_K_q8_K`).
///
/// `q6k` holds the still-quantized `Q6_K` bytes (`[out_rows x in_dim]`, packed as
/// `Q6_K` super-blocks). The activation `x` is quantized to `q8_K` **once**, then
/// each output row is an integer dot of the 6-bit `q4` weights (signed `-32..31`,
/// reconstructed on the fly) against the int8 activation, scaled by the per-group
/// q6 scale, with the `-32*bsum` correction folded in: no f32 dequant expansion.
pub fn gemv_q6k(out_rows: usize, in_dim: usize, x: &[f32], q6k: &[u8], y: &mut [f32]) {
	assert_eq!(x.len(), in_dim);
	assert_eq!(y.len(), out_rows);
	let nb = in_dim / QK_K;
	let row_bytes = nb * q6k_block_bytes();
	assert_eq!(q6k.len(), out_rows * row_bytes);
	let q8 = quantize_q8k(x, in_dim);
	par_rows(out_rows, row_bytes, y, |rs, re, ysl| {
		q6k_rows(q6k, nb, &q8, rs, re, ysl);
	});
}

/// Rows `rs..re` of a Q6_K weight against an already-quantized activation,
/// written to `out[0..re-rs]`. Split out of [`gemv_q6k`] so several weights
/// that share one activation can be covered by a single pool dispatch.
pub(crate) fn q6k_rows(q6k: &[u8], nb: usize, q8: &Q8K, rs: usize, re: usize, out: &mut [f32]) {
	let row_bytes = nb * q6k_block_bytes();
	let f = features();
	for (i, yo) in out.iter_mut().enumerate().take(re - rs) {
		let o = rs + i;
		let rslice = &q6k[o * row_bytes..(o + 1) * row_bytes];
		*yo = if f.avxvnni {
			unsafe { q6k_dot_q8k_avxvnni(rslice, nb, q8) }
		} else if f.avx2 {
			unsafe { q6k_dot_q8k_avx2(rslice, nb, q8) }
		} else {
			q6k_dot_q8k_scalar(rslice, nb, q8)
		};
	}
}

/// Fused Q6_K x q8_K dequant-GEMV for one output row, AVX2. Direct port of
/// llama.cpp `ggml_vec_dot_q6_K_q8_K` (AVX2): integer `maddubs` (reconstructed
/// unsigned 0..63 q6 weight × s8 activation) accumulated per sub-block scale,
/// with the `-32*bsum` correction folded via `q8sclsub`. `q8` is the activation
/// quantized once to `q8_K` by [`quantize_q8k`].
///
/// Fused Q6_K x q8_K dequant-GEMV for one output row, AVX2 `maddubs` kernel
/// (port of llama.cpp `ggml_vec_dot_q6_K_q8_K`). Reconstructed on the fly; the
/// `-32` offset is folded into the `bsum` correction just like the scalar path.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q6k_dot_q8k_avx2(row: &[u8], nb: usize, q8: &Q8K) -> f32 {
	use std::arch::x86_64::*;
	let m3 = _mm256_set1_epi8(3);
	let m12 = _mm256_set1_epi8(12);
	let m15 = _mm256_set1_epi8(15);
	let m48 = _mm256_set1_epi8(48);
	let m192 = _mm256_set1_epi8(-64i8);
	let mut acc = _mm256_setzero_ps();
	let qs8_ptr = q8.qs.as_ptr();
	let bsums_ptr = q8.bsums.as_ptr();
	for i in 0..nb {
		let rb = &row[i * 210..i * 210 + 210];
		let d = q8.ds[i] * fp16_to_f32(u16::from_le_bytes([rb[208], rb[209]]));
		let mut ql = rb.as_ptr();
		let mut qh = rb.as_ptr().add(128);
		let scales_ptr = rb.as_ptr().add(192);
		let q8sums = _mm256_loadu_si256(bsums_ptr.add(i * (QK_K / 16)) as *const __m256i);
		let scales128 = _mm_loadu_si128(scales_ptr as *const __m128i);
		let scales_16 = _mm256_cvtepi8_epi16(scales128);
		let q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);
		let mut sumi = _mm256_setzero_si256();
		let mut is = 0usize;
		let mut q8p = qs8_ptr.add(i * QK_K);
		for _j in 0..(QK_K / 128) {
			let q4bits1 = _mm256_loadu_si256(ql as *const __m256i);
			ql = ql.add(32);
			let q4bits2 = _mm256_loadu_si256(ql as *const __m256i);
			ql = ql.add(32);
			let q4bits_h = _mm256_loadu_si256(qh as *const __m256i);
			qh = qh.add(32);
			let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m3), 4);
			let q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m12), 2);
			let q4h_2 = _mm256_and_si256(q4bits_h, m48);
			let q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bits_h, m192), 2);
			let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
			let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
			let q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
			let q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);
			let q8_0 = _mm256_loadu_si256(q8p as *const __m256i);
			q8p = q8p.add(32);
			let q8_1 = _mm256_loadu_si256(q8p as *const __m256i);
			q8p = q8p.add(32);
			let q8_2 = _mm256_loadu_si256(q8p as *const __m256i);
			q8p = q8p.add(32);
			let q8_3 = _mm256_loadu_si256(q8p as *const __m256i);
			q8p = q8p.add(32);
			let p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
			let p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
			let p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
			let p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);
			let scale_0 = _mm_shuffle_epi8(scales128, get_scale_shuffle_k6(is));
			let scale_1 = _mm_shuffle_epi8(scales128, get_scale_shuffle_k6(is + 1));
			let scale_2 = _mm_shuffle_epi8(scales128, get_scale_shuffle_k6(is + 2));
			let scale_3 = _mm_shuffle_epi8(scales128, get_scale_shuffle_k6(is + 3));
			is += 4;
			let p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
			let p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
			let p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
			let p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);
			sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
			sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
		}
		sumi = _mm256_sub_epi32(sumi, q8sclsub);
		acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
	}
	hsum_float_8(acc)
}

/// Build an 8-lane i32 scale vector `[lo,lo,lo,lo,hi,hi,hi,hi]` from a Q6_K
/// `get_scale_shuffle_k6` result, mirroring the AVX2 `cvtepi8_epi16(scale)`
/// layout: `lo` (byte 0) scales weights 0..15, `hi` (byte 8) scales 16..31.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn k6_scale_vec(s: std::arch::x86_64::__m128i) -> std::arch::x86_64::__m256i {
	use std::arch::x86_64::*;
	let lo = (_mm_cvtsi128_si32(s) & 0xFF) as i8 as i32;
	let hi = (_mm_cvtsi128_si32(_mm_srli_si128(s, 8)) & 0xFF) as i8 as i32;
	_mm256_setr_epi32(lo, lo, lo, lo, hi, hi, hi, hi)
}

/// Fused Q6_K x q8_K dequant-GEMV for one output row, **AVX-VNNI**. Mirrors
/// [`q6k_dot_q8k_avx2`] but replaces `maddubs` + `madd` with a single
/// `_mm256_dpbusd_avx_epi32` (int32 dot, no i16 saturation, VEX AVX-VNNI
/// encoding), the fastest path on
/// Arrow Lake. Called only after runtime `avxvnni` detection.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni")]
pub(crate) unsafe fn q6k_dot_q8k_avxvnni(row: &[u8], nb: usize, q8: &Q8K) -> f32 {
	use std::arch::x86_64::*;
	let m3 = _mm256_set1_epi8(3);
	let m12 = _mm256_set1_epi8(12);
	let m15 = _mm256_set1_epi8(15);
	let m48 = _mm256_set1_epi8(48);
	let m192 = _mm256_set1_epi8(-64i8);
	let mut acc = _mm256_setzero_ps();
	let qs8_ptr = q8.qs.as_ptr();
	let bsums_ptr = q8.bsums.as_ptr();
	for i in 0..nb {
		let rb = &row[i * 210..i * 210 + 210];
		let d = q8.ds[i] * fp16_to_f32(u16::from_le_bytes([rb[208], rb[209]]));
		let mut ql = rb.as_ptr();
		let mut qh = rb.as_ptr().add(128);
		let scales_ptr = rb.as_ptr().add(192);
		let q8sums = _mm256_loadu_si256(bsums_ptr.add(i * (QK_K / 16)) as *const __m256i);
		let scales128 = _mm_loadu_si128(scales_ptr as *const __m128i);
		let scales_16 = _mm256_cvtepi8_epi16(scales128);
		let q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);
		let mut sumi = _mm256_setzero_si256();
		let mut is = 0usize;
		let mut q8p = qs8_ptr.add(i * QK_K);
		for _j in 0..(QK_K / 128) {
			let q4bits1 = _mm256_loadu_si256(ql as *const __m256i);
			ql = ql.add(32);
			let q4bits2 = _mm256_loadu_si256(ql as *const __m256i);
			ql = ql.add(32);
			let q4bits_h = _mm256_loadu_si256(qh as *const __m256i);
			qh = qh.add(32);
			let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m3), 4);
			let q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m12), 2);
			let q4h_2 = _mm256_and_si256(q4bits_h, m48);
			let q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bits_h, m192), 2);
			let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
			let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
			let q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
			let q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);
			let q8_0 = _mm256_loadu_si256(q8p as *const __m256i);
			q8p = q8p.add(32);
			let q8_1 = _mm256_loadu_si256(q8p as *const __m256i);
			q8p = q8p.add(32);
			let q8_2 = _mm256_loadu_si256(q8p as *const __m256i);
			q8p = q8p.add(32);
			let q8_3 = _mm256_loadu_si256(q8p as *const __m256i);
			q8p = q8p.add(32);
			let mut d0 = _mm256_setzero_si256();
			d0 = _mm256_dpbusd_avx_epi32(d0, q4_0, q8_0);
			let mut d1 = _mm256_setzero_si256();
			d1 = _mm256_dpbusd_avx_epi32(d1, q4_1, q8_1);
			let mut d2 = _mm256_setzero_si256();
			d2 = _mm256_dpbusd_avx_epi32(d2, q4_2, q8_2);
			let mut d3 = _mm256_setzero_si256();
			d3 = _mm256_dpbusd_avx_epi32(d3, q4_3, q8_3);
			let scale_0 = _mm_shuffle_epi8(scales128, get_scale_shuffle_k6(is));
			let scale_1 = _mm_shuffle_epi8(scales128, get_scale_shuffle_k6(is + 1));
			let scale_2 = _mm_shuffle_epi8(scales128, get_scale_shuffle_k6(is + 2));
			let scale_3 = _mm_shuffle_epi8(scales128, get_scale_shuffle_k6(is + 3));
			is += 4;
			// Each `q4` vector spans 32 weights = two Q6_K sub-blocks, so it needs
			// two distinct scales: `lo=scales[2i]` for weights 0..15 and
			// `hi=scales[2i+1]` for weights 16..31. `dpbusd` returns 8 i32 lanes
			// (4 weights each) in that same order, so broadcast lo/hi to match the
			// AVX2 `cvtepi8_epi16`(scale_0) layout: [lo,lo,lo,lo,hi,hi,hi,hi].
			let scv0 = k6_scale_vec(scale_0);
			let scv1 = k6_scale_vec(scale_1);
			let scv2 = k6_scale_vec(scale_2);
			let scv3 = k6_scale_vec(scale_3);
			sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(scv0, d0));
			sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(scv1, d1));
			sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(scv2, d2));
			sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(scv3, d3));
		}
		sumi = _mm256_sub_epi32(sumi, q8sclsub);
		acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
	}
	hsum_float_8(acc)
}
