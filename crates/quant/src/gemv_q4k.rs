//! Fused `Q4_K` x `q8_K` dequant-GEMV kernels (single token and the dispatch).
//!
//! `q4k` holds the still-quantized `Q4_K` bytes (`[out_rows x in_dim]`, packed as
//! `Q4_K` super-blocks). The activation `x` is quantized to `q8_K` **once**, then
//! each output row is computed by an integer dot of the q4 nibbles (unsigned
//! 0..15) against the int8 activation, scaled by the per-sub-block q4 scale and
//! accumulated with the dmin/bsum correction: no f32 dequant expansion. This
//! reads ~4x less weight data and runs an integer micro-kernel, matching
//! llama.cpp's matvec path.

use super::{
	features, fp16_to_f32, get_scale_min_k4, par_rows, quantize_q8k, K_SCALE_SIZE, Q8K, QK_K,
};

/// Shuffle mask table for `get_scale_shuffle_k4` (llama.cpp). Row `i` broadcasts
/// the byte pair `(2*i, 2*i+1)` of the unpacked scales into every lane, so
/// `get_scale_shuffle_k4(k)` selects sub-block scale `k`.
const fn build_scale_shuf_k4() -> [u8; 256] {
	let mut t = [0u8; 256];
	let mut i = 0usize;
	while i < 8 {
		let lo = (2 * i) as u8;
		let hi = (2 * i + 1) as u8;
		let mut j = 0usize;
		while j < 16 {
			t[i * 32 + j * 2] = lo;
			t[i * 32 + j * 2 + 1] = hi;
			j += 1;
		}
		i += 1;
	}
	t
}
static SCALE_SHUF_K4: [u8; 256] = build_scale_shuf_k4();

/// **Fused Q4_K dequant-GEMV** (llama.cpp `ggml_vec_dot_q4_K_q8_K`).
///
/// `q4k` holds the still-quantized `Q4_K` bytes (`[out_rows x in_dim]`, packed as
/// `Q4_K` super-blocks). The activation `x` is quantized to `q8_K` **once**, then
/// each output row is computed by an integer dot of the q4 nibbles (unsigned
/// 0..15) against the int8 activation, scaled by the per-sub-block q4 scale and
/// accumulated with the dmin/bsum correction: no f32 dequant expansion. This
/// reads ~4x less weight data and runs an integer micro-kernel, matching
/// llama.cpp's matvec path.
pub fn gemv_q4k(out_rows: usize, in_dim: usize, x: &[f32], q4k: &[u8], y: &mut [f32]) {
	assert_eq!(x.len(), in_dim);
	assert_eq!(y.len(), out_rows);
	let nb = in_dim / QK_K;
	let blk = 2 + 2 + K_SCALE_SIZE + QK_K / 2;
	assert_eq!(q4k.len(), out_rows * nb * blk);
	let q8 = quantize_q8k(x, in_dim);
	let row_bytes = nb * blk;
	par_rows(out_rows, row_bytes, y, |rs, re, ysl| {
		q4k_rows(q4k, nb, &q8, rs, re, ysl);
	});
}

/// Rows `rs..re` of a Q4_K weight against an already-quantized activation,
/// written to `out[0..re-rs]`. Split out of [`gemv_q4k`] so several weights
/// sharing one activation can be covered by a single pool dispatch.
pub(crate) fn q4k_rows(q4k: &[u8], nb: usize, q8: &Q8K, rs: usize, re: usize, out: &mut [f32]) {
	let blk = 2 + 2 + K_SCALE_SIZE + QK_K / 2;
	let row_bytes = nb * blk;
	let f = features();
	for (i, yo) in out.iter_mut().enumerate().take(re - rs) {
		let o = rs + i;
		let rslice = &q4k[o * row_bytes..(o + 1) * row_bytes];
		*yo = if f.avxvnni {
			unsafe { q4k_dot_q8k_avxvnni(rslice, nb, q8) }
		} else if f.avx2 {
			unsafe { q4k_dot_q8k_avx2(rslice, nb, q8) }
		} else {
			q4k_dot_q8k_scalar(rslice, nb, q8)
		};
	}
}

/// Fused Q4_K x q8_K dequant-GEMV for one output row, scalar fallback. This is
/// a direct port of llama.cpp's `ggml_vec_dot_q4_K_q8_K` integer accumulation:
/// the q4 nibbles (unsigned 0..15) are multiplied (as integers) by the int8
/// activation and the per-sub-block q4 scale, with the `dmin * bsum` correction
/// for the quantized mins. `q8` is the activation already quantized to `q8_K`.
pub(crate) fn q4k_dot_q8k_scalar(row: &[u8], nb: usize, q8: &Q8K) -> f32 {
	let blk = 2 + 2 + K_SCALE_SIZE + QK_K / 2;
	let mut acc = 0.0f32;
	for i in 0..nb {
		let rb = &row[i * blk..i * blk + blk];
		let d = q8.ds[i] * fp16_to_f32(u16::from_le_bytes([rb[0], rb[1]]));
		let dmin = -q8.ds[i] * fp16_to_f32(u16::from_le_bytes([rb[2], rb[3]]));
		let scales = &rb[4..4 + K_SCALE_SIZE];
		let qs = &rb[4 + K_SCALE_SIZE..];
		let qs8 = &q8.qs[i * QK_K..(i + 1) * QK_K];
		let mut sumi = 0i32;
		let mut sum_m = 0i32;
		// 8 sub-blocks of 32 elements each.
		for s in 0..8 {
			let (sc, m) = get_scale_min_k4(s, scales);
			let base = (s / 2) * 32; // qs byte offset for this sub-block
			let shift = if s % 2 == 0 { 0 } else { 4 }; // low / high nibble
			let q8b = s * 32; // activation sub-block is contiguous
			let mut dot = 0i32;
			let mut q8sum = 0i32;
			for l in 0..32 {
				let nib = (qs[base + l] >> shift) & 0x0F;
				let qv = qs8[q8b + l] as i32;
				dot += nib as i32 * qv;
				q8sum += qv;
			}
			sumi += sc as i32 * dot;
			sum_m += m as i32 * q8sum;
		}
		acc += d * sumi as f32 + dmin * sum_m as f32;
	}
	acc
}

/// Fused Q4_K x q8_K dequant-GEMV for one output row, AVX2. Direct port of
/// llama.cpp `ggml_vec_dot_q4_K_q8_K`: integer `maddubs` (u8 nibble x s8 act)
/// accumulated per sub-block scale, with the `dmin * bsum` correction; no f32
/// dequant. `q8` is the activation quantized once to `q8_K` by [`quantize_q8k`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q4k_dot_q8k_avx2(row: &[u8], nb: usize, q8: &Q8K) -> f32 {
	use std::arch::x86_64::*;
	let blk = 2 + 2 + K_SCALE_SIZE + QK_K / 2;
	let m4 = _mm256_set1_epi8(0x0F_i8);
	let kmask1: u32 = 0x3f3f_3f3f;
	let kmask2: u32 = 0x0f0f_0f0f;
	let kmask3: u32 = 0x0303_0303;

	let mut acc = _mm256_setzero_ps();
	let mut acc_m = _mm_setzero_ps();

	let qs8_ptr = q8.qs.as_ptr();
	let bsums_ptr = q8.bsums.as_ptr();

	for i in 0..nb {
		let rb = &row[i * blk..i * blk + blk];
		let d = q8.ds[i] * fp16_to_f32(u16::from_le_bytes([rb[0], rb[1]]));
		let dmin = -q8.ds[i] * fp16_to_f32(u16::from_le_bytes([rb[2], rb[3]]));
		let scales = &rb[4..4 + K_SCALE_SIZE];
		let qs = rb.as_ptr().add(4 + K_SCALE_SIZE);

		// Unpack the 6-bit packed q4 scales into 8 scales + 8 mins (llama.cpp).
		let mut utmp = [0u32; 4];
		utmp[0] = u32::from_le_bytes([scales[0], scales[1], scales[2], scales[3]]);
		utmp[1] = u32::from_le_bytes([scales[4], scales[5], scales[6], scales[7]]);
		utmp[2] = u32::from_le_bytes([scales[8], scales[9], scales[10], scales[11]]);
		utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
		let uaux = utmp[1] & kmask1;
		utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
		utmp[2] = uaux;
		utmp[0] &= kmask1;
		let lo0 = utmp[0].to_le_bytes();
		let lo1 = utmp[1].to_le_bytes();
		let hi0 = utmp[2].to_le_bytes();
		let hi1 = utmp[3].to_le_bytes();
		let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
			i32::from_le_bytes([hi1[0], hi1[1], hi1[2], hi1[3]]),
			i32::from_le_bytes([hi0[0], hi0[1], hi0[2], hi0[3]]),
			i32::from_le_bytes([lo1[0], lo1[1], lo1[2], lo1[3]]),
			i32::from_le_bytes([lo0[0], lo0[1], lo0[2], lo0[3]]),
		));
		let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
		let scales256 = _mm256_set_m128i(sc128, sc128);
		let mins128 = _mm256_extracti128_si256(mins_and_scales, 1);

		let mut sumi = _mm256_setzero_si256();
		for j in 0..4 {
			let scale_l = _mm256_shuffle_epi8(scales256, get_scale_shuffle_k4(2 * j));
			let scale_h = _mm256_shuffle_epi8(scales256, get_scale_shuffle_k4(2 * j + 1));
			let q4bits = _mm256_loadu_si256(qs.add(j * 32) as *const __m256i);
			let q4l = _mm256_and_si256(q4bits, m4);
			let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
			let q8base = (i * QK_K + j * 64) as isize;
			let q8l = _mm256_loadu_si256(qs8_ptr.offset(q8base) as *const __m256i);
			let p16l = _mm256_maddubs_epi16(q4l, q8l);
			let p16l = _mm256_madd_epi16(scale_l, p16l);
			let q8h = _mm256_loadu_si256(qs8_ptr.offset(q8base + 32) as *const __m256i);
			let p16h = _mm256_maddubs_epi16(q4h, q8h);
			let p16h = _mm256_madd_epi16(scale_h, p16h);
			sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
		}

		// dmin * sum(m_sub * bsum_sub)
		let q8sums = _mm256_loadu_si256(bsums_ptr.add(i * (QK_K / 16)) as *const __m256i);
		let q8s = _mm_hadd_epi16(
			_mm256_extracti128_si256(q8sums, 0),
			_mm256_extracti128_si256(q8sums, 1),
		);
		let prod = _mm_madd_epi16(mins128, q8s);
		acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

		let vd = _mm256_set1_ps(d);
		acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
	}

	acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
	acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
	hsum_float_8(acc) + _mm_cvtss_f32(acc_m)
}

/// Fused Q4_K x q8_K dequant-GEMV for one output row, **AVX-VNNI**. Same
/// arithmetic as [`q4k_dot_q8k_avx2`] but the `maddubs` + `madd` pair is folded
/// into a single `_mm256_dpbusd_avx_epi32` (int32 dot, no i16 saturation, VEX
/// AVX-VNNI encoding). On
/// Arrow Lake (no AVX-512, but AVX-VNNI present) this is the fastest path and
/// matches the scalar reference to 1e-2. Called only after runtime detection of
/// the `avxvnni` feature.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni")]
pub(crate) unsafe fn q4k_dot_q8k_avxvnni(row: &[u8], nb: usize, q8: &Q8K) -> f32 {
	use std::arch::x86_64::*;
	let blk = 2 + 2 + K_SCALE_SIZE + QK_K / 2;
	let m4 = _mm256_set1_epi8(0x0F_i8);
	let kmask1: u32 = 0x3f3f_3f3f;
	let kmask2: u32 = 0x0f0f_0f0f;
	let kmask3: u32 = 0x0303_0303;

	let mut acc = _mm256_setzero_ps();
	let mut acc_m = _mm_setzero_ps();

	let qs8_ptr = q8.qs.as_ptr();
	let bsums_ptr = q8.bsums.as_ptr();

	for i in 0..nb {
		let rb = &row[i * blk..i * blk + blk];
		let d = q8.ds[i] * fp16_to_f32(u16::from_le_bytes([rb[0], rb[1]]));
		let dmin = -q8.ds[i] * fp16_to_f32(u16::from_le_bytes([rb[2], rb[3]]));
		let scales = &rb[4..4 + K_SCALE_SIZE];
		let qs = rb.as_ptr().add(4 + K_SCALE_SIZE);

		let mut utmp = [0u32; 4];
		utmp[0] = u32::from_le_bytes([scales[0], scales[1], scales[2], scales[3]]);
		utmp[1] = u32::from_le_bytes([scales[4], scales[5], scales[6], scales[7]]);
		utmp[2] = u32::from_le_bytes([scales[8], scales[9], scales[10], scales[11]]);
		utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
		let uaux = utmp[1] & kmask1;
		utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
		utmp[2] = uaux;
		utmp[0] &= kmask1;
		let lo0 = utmp[0].to_le_bytes();
		let lo1 = utmp[1].to_le_bytes();
		let hi0 = utmp[2].to_le_bytes();
		let hi1 = utmp[3].to_le_bytes();
		let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
			i32::from_le_bytes([hi1[0], hi1[1], hi1[2], hi1[3]]),
			i32::from_le_bytes([hi0[0], hi0[1], hi0[2], hi0[3]]),
			i32::from_le_bytes([lo1[0], lo1[1], lo1[2], lo1[3]]),
			i32::from_le_bytes([lo0[0], lo0[1], lo0[2], lo0[3]]),
		));
		let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
		let scales256 = _mm256_set_m128i(sc128, sc128);
		let mins128 = _mm256_extracti128_si256(mins_and_scales, 1);

		let mut sumi = _mm256_setzero_si256();
		for j in 0..4 {
			let scale_l = _mm256_shuffle_epi8(scales256, get_scale_shuffle_k4(2 * j));
			let scale_h = _mm256_shuffle_epi8(scales256, get_scale_shuffle_k4(2 * j + 1));
			let q4bits = _mm256_loadu_si256(qs.add(j * 32) as *const __m256i);
			let q4l = _mm256_and_si256(q4bits, m4);
			let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
			let q8base = (i * QK_K + j * 64) as isize;
			let q8l = _mm256_loadu_si256(qs8_ptr.offset(q8base) as *const __m256i);
			let q8h = _mm256_loadu_si256(qs8_ptr.offset(q8base + 32) as *const __m256i);
			let mut dotl = _mm256_setzero_si256();
			dotl = _mm256_dpbusd_avx_epi32(dotl, q4l, q8l);
			let mut doth = _mm256_setzero_si256();
			doth = _mm256_dpbusd_avx_epi32(doth, q4h, q8h);
			let scl = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(scale_l, 0));
			let sch = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(scale_h, 0));
			sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(scl, dotl));
			sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(sch, doth));
		}

		let q8sums = _mm256_loadu_si256(bsums_ptr.add(i * (QK_K / 16)) as *const __m256i);
		let q8s = _mm_hadd_epi16(
			_mm256_extracti128_si256(q8sums, 0),
			_mm256_extracti128_si256(q8sums, 1),
		);
		let prod = _mm_madd_epi16(mins128, q8s);
		acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

		let vd = _mm256_set1_ps(d);
		acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
	}

	acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
	acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
	hsum_float_8(acc) + _mm_cvtss_f32(acc_m)
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) unsafe fn hsum_float_8(x: std::arch::x86_64::__m256) -> f32 {
	let mut tmp = [0.0f32; 8];
	std::arch::x86_64::_mm256_storeu_ps(tmp.as_mut_ptr(), x);
	tmp.iter().sum::<f32>()
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn get_scale_shuffle_k4(i: usize) -> std::arch::x86_64::__m256i {
	std::arch::x86_64::_mm256_loadu_si256(
		SCALE_SHUF_K4.as_ptr().add(i * 32) as *const std::arch::x86_64::__m256i
	)
}

/// One Q4_K row against an **f32** activation: the exact `dequant(row) · x`,
/// in f32, with no q8_K quantization of `x`. `xsum[s]` is the sum of `x` over
/// 32-element sub-block `s` (for the `dmin * min` term), computed once per
/// activation by the caller. This is what the GPU's float GEMV computes, so a
/// matrix evaluated here agrees with its GPU-split rows.
pub(crate) fn q4k_dot_f32(row: &[u8], nb: usize, x: &[f32], xsum: &[f32]) -> f32 {
	#[cfg(target_arch = "x86_64")]
	if features().avx2 {
		return unsafe { q4k_dot_f32_avx2(row, nb, x, xsum) };
	}
	let mut buf = [0f32; QK_K];
	let mut s = 0f32;
	for i in 0..nb {
		super::dequant::dequantize_q4_k_block(&row[i * Q4K_BLK..], &mut buf);
		s += buf
			.iter()
			.zip(&x[i * QK_K..(i + 1) * QK_K])
			.map(|(a, b)| a * b)
			.sum::<f32>();
	}
	s
}

const Q4K_BLK: usize = 2 + 2 + K_SCALE_SIZE + QK_K / 2;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q4k_dot_f32_avx2(row: &[u8], nb: usize, x: &[f32], xsum: &[f32]) -> f32 {
	use std::arch::x86_64::*;
	debug_assert!(row.len() >= nb * Q4K_BLK && x.len() >= nb * QK_K && xsum.len() >= nb * 8);
	let mut acc = _mm256_setzero_ps();
	let mut mins = 0f32;
	let xp = x.as_ptr();
	for i in 0..nb {
		let rb = &row[i * Q4K_BLK..(i + 1) * Q4K_BLK];
		let d = fp16_to_f32(u16::from_le_bytes([rb[0], rb[1]]));
		let dmin = fp16_to_f32(u16::from_le_bytes([rb[2], rb[3]]));
		let scales = &rb[4..4 + K_SCALE_SIZE];
		let qs = rb.as_ptr().add(4 + K_SCALE_SIZE);
		for j in 0..4 {
			let (sc1, m1) = get_scale_min_k4(2 * j, scales);
			let (sc2, m2) = get_scale_min_k4(2 * j + 1, scales);
			mins += dmin * (m1 as f32 * xsum[i * 8 + 2 * j] + m2 as f32 * xsum[i * 8 + 2 * j + 1]);
			let xb = xp.add(i * QK_K + j * 64);
			let mut lo = _mm256_setzero_ps();
			let mut hi = _mm256_setzero_ps();
			for k in 0..4 {
				// 8 packed bytes: low nibbles are elements k*8.. of the first
				// sub-block, high nibbles the same elements of the second.
				let b = _mm_loadl_epi64(qs.add(j * 32 + k * 8) as *const __m128i);
				let w = _mm256_cvtepu8_epi32(b);
				let ql = _mm256_cvtepi32_ps(_mm256_and_si256(w, _mm256_set1_epi32(0x0F)));
				let qh = _mm256_cvtepi32_ps(_mm256_srli_epi32(w, 4));
				lo = _mm256_fmadd_ps(ql, _mm256_loadu_ps(xb.add(k * 8)), lo);
				hi = _mm256_fmadd_ps(qh, _mm256_loadu_ps(xb.add(32 + k * 8)), hi);
			}
			acc = _mm256_fmadd_ps(_mm256_set1_ps(d * sc1 as f32), lo, acc);
			acc = _mm256_fmadd_ps(_mm256_set1_ps(d * sc2 as f32), hi, acc);
		}
	}
	hsum_f32x8(acc) - mins
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum_f32x8(v: std::arch::x86_64::__m256) -> f32 {
	use std::arch::x86_64::*;
	let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
	let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
	_mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)))
}
