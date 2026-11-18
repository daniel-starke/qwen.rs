//! Intermediate `q8_K` quantization of the activation vector, and profiling of
//! the quantize cost.
//!
//! Layout mirrors llama.cpp `block_q8_K` (`d` float per 256-block, int8 `qs` per
//! element, int16 `bsums` per 16-group). The fused Q4_K / Q6_K dots consume
//! these with integer multiply-accumulate, so `x` is quantified to 8 bits
//! **once** per matmul instead of being streamed as f32 against dequantized
//! weights.

use super::{features, QK_K};
use std::cell::Cell;

thread_local! {
	static Q8_PROF_NS: Cell<u64> = const { Cell::new(0) };
}

/// Total time spent inside `quantize_q8k` since process start (profiling only).
pub fn q8_prof_ns() -> u64 {
	Q8_PROF_NS.with(|c| c.get())
}

/// Quantize one row **into caller-owned buffers**, allocating nothing.
///
/// [`quantize_q8k`] allocates three `Vec`s per call. That is invisible when it
/// is called once per matmul, and it is the limiter when prefill calls it once
/// per *token*: packing a `seq = 2048` x `in_dim = 17408` activation is 2048
/// calls x 3 allocations, and across the shared pool those serialize on the
/// global allocator hard enough to hold a 24-core memory-bound loop at ~12 GB/s.
///
/// `bsums` is a by-product the GPU pack path never reads, so it goes to a
/// per-thread scratch buffer rather than a fresh allocation.
pub fn quantize_q8k_into(x: &[f32], n: usize, ds: &mut [f32], qs: &mut [i8]) {
	debug_assert_eq!(n % QK_K, 0);
	let nb = n / QK_K;
	debug_assert!(ds.len() >= nb && qs.len() >= n);
	BSUMS.with(|b| {
		let mut b = b.borrow_mut();
		if b.len() < n / 16 {
			b.resize(n / 16, 0);
		}
		#[cfg(target_arch = "x86_64")]
		if features().avx2 {
			unsafe { quantize_q8k_avx2(x, nb, ds, qs, &mut b) };
			return;
		}
		let r = quantize_q8k_scalar(x, n);
		ds[..nb].copy_from_slice(&r.ds);
		qs[..n].copy_from_slice(&r.qs);
	});
}

thread_local! {
	/// Scratch for the `bsums` `quantize_q8k_into` computes and discards.
	static BSUMS: std::cell::RefCell<Vec<i16>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Intermediate `q8_K` quantization of the activation vector `x` (one per token).
pub struct Q8K {
	pub ds: Vec<f32>,           // per-256-block delta
	pub qs: Vec<i8>,            // per-element int8 (length n)
	pub(crate) bsums: Vec<i16>, // per-16-group sum of qs (length n/16)
}

/// Scalar reference for [`quantize_q8k`], kept callable so the AVX2 path can
/// be pinned to it bit-for-bit by `q8k_avx2_matches_scalar`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn quantize_q8k_scalar(x: &[f32], n: usize) -> Q8K {
	let nb = n / QK_K;
	let mut ds = vec![0.0f32; nb];
	let mut qs = vec![0i8; n];
	let mut bsums = vec![0i16; n / 16];
	for b in 0..nb {
		let xb = &x[b * QK_K..(b + 1) * QK_K];
		let mut amax = 0.0f32;
		for &v in xb {
			let a = v.abs();
			if a > amax {
				amax = a;
			}
		}
		if amax == 0.0 {
			continue;
		}
		let iscale = -127.0f32 / amax;
		for (j, &v) in xb.iter().enumerate() {
			qs[b * QK_K + j] = (iscale * v).round().clamp(-128.0, 127.0) as i32 as i8;
		}
		for g in 0..QK_K / 16 {
			let mut s = 0i32;
			for ii in 0..16 {
				s += qs[b * QK_K + g * 16 + ii] as i32;
			}
			bsums[b * (QK_K / 16) + g] = s as i16;
		}
		ds[b] = 1.0f32 / iscale;
	}
	Q8K { ds, qs, bsums }
}

pub fn quantize_q8k(x: &[f32], n: usize) -> Q8K {
	// NOTE: the profiling gate is a cached flag, not a per-call `env::var`;
	// decode calls this ~575 times per token.
	let t0 = if features().phase {
		Some(std::time::Instant::now())
	} else {
		None
	};
	debug_assert_eq!(n % QK_K, 0);
	let nb = n / QK_K;
	let mut ds = vec![0.0f32; nb];
	let mut qs = vec![0i8; n];
	let mut bsums = vec![0i16; n / 16];
	#[cfg(target_arch = "x86_64")]
	if features().avx2 {
		unsafe { quantize_q8k_avx2(x, nb, &mut ds, &mut qs, &mut bsums) };
		let r = Q8K { ds, qs, bsums };
		if let Some(t0) = t0 {
			Q8_PROF_NS.with(|c| c.set(c.get() + t0.elapsed().as_nanos() as u64));
		}
		return r;
	}
	for b in 0..nb {
		let xb = &x[b * QK_K..(b + 1) * QK_K];
		let mut amax = 0.0f32;
		for &v in xb {
			let a = v.abs();
			if a > amax {
				amax = a;
			}
		}
		if amax == 0.0 {
			continue; // ds/qs/bsums already zero
		}
		// llama.cpp quantize_row_q8_K_ref: iscale = -127/max, v = round(iscale*x) clamped to 127
		let iscale = -127.0f32 / amax;
		for (j, &v) in xb.iter().enumerate() {
			let q = (iscale * v).round().clamp(-128.0, 127.0) as i32 as i8;
			qs[b * QK_K + j] = q;
		}
		for g in 0..QK_K / 16 {
			let mut s = 0i32;
			for ii in 0..16 {
				s += qs[b * QK_K + g * 16 + ii] as i32;
			}
			bsums[b * (QK_K / 16) + g] = s as i16;
		}
		ds[b] = 1.0f32 / iscale;
	}
	let r = Q8K { ds, qs, bsums };
	if let Some(t0) = t0 {
		Q8_PROF_NS.with(|c| c.set(c.get() + t0.elapsed().as_nanos() as u64));
	}
	r
}

/// AVX2 [`quantize_q8k`]. Bit-identical to the scalar loop above: same
/// `-127/amax` scale, same round-half-away-from-zero (`_mm256_round_ps` with
/// `NEAREST_INT` matches `f32::round` for these magnitudes only if the tie
/// cases agree, so ties are pushed away from zero explicitly), same `bsums`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn quantize_q8k_avx2(
	x: &[f32],
	nb: usize,
	ds: &mut [f32],
	qs: &mut [i8],
	bsums: &mut [i16],
) {
	use std::arch::x86_64::*;
	let absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff_u32 as i32));
	for b in 0..nb {
		let xb = x.as_ptr().add(b * QK_K);
		// amax over the 256-element super-block.
		let mut m = _mm256_setzero_ps();
		for j in 0..QK_K / 8 {
			let v = _mm256_loadu_ps(xb.add(j * 8));
			m = _mm256_max_ps(m, _mm256_and_ps(v, absmask));
		}
		let mut lanes = [0.0f32; 8];
		_mm256_storeu_ps(lanes.as_mut_ptr(), m);
		let amax = lanes.iter().fold(0.0f32, |a, &v| if v > a { v } else { a });
		if amax == 0.0 {
			continue; // ds/qs/bsums already zero
		}
		let iscale = -127.0f32 / amax;
		let vscale = _mm256_set1_ps(iscale);
		// half-away-from-zero rounding: add copysign(0.5, v) then truncate.
		let half = _mm256_set1_ps(0.5);
		let signmask = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MIN));
		for g in 0..QK_K / 16 {
			let mut packed = [0i32; 16];
			for h in 0..2 {
				let v = _mm256_loadu_ps(xb.add(g * 16 + h * 8));
				let s = _mm256_mul_ps(v, vscale);
				let bias = _mm256_or_ps(half, _mm256_and_ps(s, signmask));
				let r = _mm256_cvttps_epi32(_mm256_add_ps(s, bias));
				_mm256_storeu_si256(packed.as_mut_ptr().add(h * 8) as *mut __m256i, r);
			}
			let mut sum = 0i32;
			for (k, &q) in packed.iter().enumerate() {
				let q = q.clamp(-128, 127) as i8;
				*qs.get_unchecked_mut(b * QK_K + g * 16 + k) = q;
				sum += q as i32;
			}
			*bsums.get_unchecked_mut(b * (QK_K / 16) + g) = sum as i16;
		}
		*ds.get_unchecked_mut(b) = 1.0f32 / iscale;
	}
}
