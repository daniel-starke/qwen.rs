//! CPU `f32` / `f16` GEMM and single-token GEMV kernels.
//!
//! The `out_rows` dimension is parallelized across the available cores. The
//! inner dot product uses an AVX2 kernel on x86_64 when available, with a
//! scalar fallback everywhere else. The `f16` variants widen weights to `f32`
//! on the fly inside the dot product so no full `f32` copy is materialized per
//! token (the matmul weights dominate the per-token cost, so this halves weight
//! memory traffic).

use super::fp16_to_f32;
use super::{features, par_rows};

/// Scalar `f32` dot product (fallback).
#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
	let mut s = 0.0f32;
	for i in 0..a.len() {
		s += a[i] * b[i];
	}
	s
}

/// AVX2 `f32` dot product (8-wide accumulation). Used on x86_64 when AVX2 is
/// present. The trailing 1..7 elements are finished with the scalar loop.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
	use std::arch::x86_64::*;
	let mut acc = _mm256_setzero_ps();
	let mut i = 0;
	while i + 8 <= a.len() {
		let va = _mm256_loadu_ps(a.as_ptr().add(i));
		let vb = _mm256_loadu_ps(b.as_ptr().add(i));
		acc = _mm256_fmadd_ps(va, vb, acc);
		i += 8;
	}
	let mut tmp = [0.0f32; 8];
	_mm256_storeu_ps(tmp.as_mut_ptr(), acc);
	let mut s = tmp.iter().sum::<f32>();
	while i < a.len() {
		s += a[i] * b[i];
		i += 1;
	}
	s
}

/// Compute `out = input @ weights^T`.
///
/// * `weights` is row-major `[out_rows, in_dim]` (already dequantized `f32`).
/// * `input`   is row-major `[seq_len, in_dim]`.
/// * `out`     receives `[out_rows, seq_len]` (i.e. `out[o * seq_len + t]`).
///
/// The `out_rows` dimension is parallelized across the available cores using
/// `std::thread` (no external dependency). The inner dot product uses an AVX2
/// kernel on x86_64 when available.
pub fn gemm(
	out_rows: usize,
	in_dim: usize,
	seq_len: usize,
	weights: &[f32],
	input: &[f32],
	out: &mut [f32],
) {
	assert_eq!(weights.len(), out_rows * in_dim);
	assert_eq!(input.len(), seq_len * in_dim);
	assert_eq!(out.len(), seq_len * out_rows);

	let use_avx2 = {
		#[cfg(target_arch = "x86_64")]
		{
			is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
		}
		#[cfg(not(target_arch = "x86_64"))]
		{
			false
		}
	};

	let dot = |a: &[f32], b: &[f32]| -> f32 {
		#[cfg(target_arch = "x86_64")]
		{
			if use_avx2 {
				unsafe { dot_avx2(a, b) }
			} else {
				dot_scalar(a, b)
			}
		}
		#[cfg(not(target_arch = "x86_64"))]
		{
			dot_scalar(a, b)
		}
	};

	// Parallelize over output rows. Each thread computes a contiguous stripe of
	// output rows into its own buffer, then the parent copies it back. This
	// avoids sharing `out` mutably across threads (which `std::thread::scope`
	// would reject).
	let threads = std::thread::available_parallelism()
		.map(|n| n.get())
		.unwrap_or(1)
		.max(1);
	let stripe = out_rows.div_ceil(threads);

	let ranges: Vec<(usize, usize)> = {
		let mut v = Vec::new();
		let mut row = 0;
		while row < out_rows {
			let end = (row + stripe).min(out_rows);
			v.push((row, end));
			row = end;
		}
		v
	};

	std::thread::scope(|s| {
		let mut handles = Vec::with_capacity(ranges.len());
		for (rs, re) in ranges {
			let h = s.spawn(move || {
				let mut chunk = vec![0.0f32; (re - rs) * seq_len];
				for (i, o) in (rs..re).enumerate() {
					let wrow = &weights[o * in_dim..(o + 1) * in_dim];
					for t in 0..seq_len {
						let irow = &input[t * in_dim..(t + 1) * in_dim];
						chunk[i * seq_len + t] = dot(wrow, irow);
					}
				}
				(rs, chunk)
			});
			handles.push(h);
		}
		for h in handles {
			let (rs, chunk) = h.join().unwrap();
			out[rs * seq_len..rs * seq_len + chunk.len()].copy_from_slice(&chunk);
		}
	});
}

/// `f16` (half-precision, stored as raw `u16`) variant of [`gemm`].
///
/// `weights` are `out_rows * in_dim` little-endian `f16` values; they are widened
/// to `f32` on the fly inside the dot product (a few cheap ops per element) so
/// the caller never materializes a full `f32` copy. This halves weight memory
/// traffic vs the `f32` `gemm` (the matmul weights dominate the per-token cost
/// of the 27B forward), which is why the layer cache stores weights as `f16`.
pub fn gemm_f16(
	out_rows: usize,
	in_dim: usize,
	seq_len: usize,
	weights: &[u16],
	input: &[f32],
	out: &mut [f32],
) {
	assert_eq!(weights.len(), out_rows * in_dim);
	assert_eq!(input.len(), seq_len * in_dim);
	assert_eq!(out.len(), seq_len * out_rows);

	let use_avx2 = {
		#[cfg(target_arch = "x86_64")]
		{
			is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
		}
		#[cfg(not(target_arch = "x86_64"))]
		{
			false
		}
	};

	// Convert a stripe of 8 `f16` weights to `f32` so we can reuse the AVX2
	// `f32` dot kernel; the widening is far cheaper than the 4-bit Q4_K decode
	// we otherwise re-run every token.
	let dot = |a: &[u16], b: &[f32]| -> f32 {
		let mut acc = 0.0f32;
		let mut i = 0;
		#[cfg(target_arch = "x86_64")]
		{
			if use_avx2 && i + 8 <= a.len() {
				let mut buf = [0.0f32; 8];
				while i + 8 <= a.len() {
					for k in 0..8 {
						buf[k] = fp16_to_f32(a[i + k]);
					}
					let va = unsafe { std::arch::x86_64::_mm256_loadu_ps(buf.as_ptr()) };
					let vb = unsafe { std::arch::x86_64::_mm256_loadu_ps(b.as_ptr().add(i)) };
					let mut tmp = [0.0f32; 8];
					unsafe {
						std::arch::x86_64::_mm256_storeu_ps(
							tmp.as_mut_ptr(),
							std::arch::x86_64::_mm256_fmadd_ps(
								va,
								vb,
								std::arch::x86_64::_mm256_setzero_ps(),
							),
						);
					}
					acc += tmp.iter().sum::<f32>();
					i += 8;
				}
			}
		}
		while i < a.len() {
			acc += fp16_to_f32(a[i]) * b[i];
			i += 1;
		}
		acc
	};

	let threads = std::thread::available_parallelism()
		.map(|n| n.get())
		.unwrap_or(1)
		.max(1);
	let stripe = out_rows.div_ceil(threads);

	let ranges: Vec<(usize, usize)> = {
		let mut v = Vec::new();
		let mut row = 0;
		while row < out_rows {
			let end = (row + stripe).min(out_rows);
			v.push((row, end));
			row = end;
		}
		v
	};

	std::thread::scope(|s| {
		let mut handles = Vec::with_capacity(ranges.len());
		for (rs, re) in ranges {
			let h = s.spawn(move || {
				let mut chunk = vec![0.0f32; (re - rs) * seq_len];
				for (i, o) in (rs..re).enumerate() {
					let wrow = &weights[o * in_dim..(o + 1) * in_dim];
					for t in 0..seq_len {
						let irow = &input[t * in_dim..(t + 1) * in_dim];
						chunk[i * seq_len + t] = dot(wrow, irow);
					}
				}
				(rs, chunk)
			});
			handles.push(h);
		}
		for h in handles {
			let (rs, chunk) = h.join().unwrap();
			out[rs * seq_len..rs * seq_len + chunk.len()].copy_from_slice(&chunk);
		}
	});
}

#[inline]
/// Attention scores for one head: `out[j] = dot(q, k_j) * scale`, where `k_j`
/// is the `q.len()`-long slice at `kbase[j * row_stride + off ..]`.
///
/// This exists instead of calling a plain dot per `j` for two reasons, both of
/// which made the naive form slower than the scalar code it replaced:
///
/// * a single-accumulator dot is bound by the **FMA dependency chain** (~4
///   cycles each), not by throughput, so `head_dim = 256` cost ~4x what it
///   should. Four independent accumulators break the chain.
/// * `#[target_feature]` functions cannot inline into a non-AVX2 caller, so a
///   per-`j` call paid real call overhead against ~32 FMAs of work. The `j`
///   loop now lives *inside* the AVX2 function.
///
/// Reassociates the sum, so it is not bit-identical to a serial dot.
pub fn attn_scores(
	q: &[f32],
	kbase: &[f32],
	row_stride: usize,
	off: usize,
	n: usize,
	scale: f32,
	out: &mut [f32],
) {
	#[cfg(target_arch = "x86_64")]
	{
		if super::features().avx2 {
			unsafe { attn_scores_avx2(q, kbase, row_stride, off, n, scale, out) };
			return;
		}
	}
	for j in 0..n {
		let k = &kbase[j * row_stride + off..j * row_stride + off + q.len()];
		out[j] = dot_scalar(q, k) * scale;
	}
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn attn_scores_avx2(
	q: &[f32],
	kbase: &[f32],
	row_stride: usize,
	off: usize,
	n: usize,
	scale: f32,
	out: &mut [f32],
) {
	use std::arch::x86_64::*;
	let hd = q.len();
	let qp = q.as_ptr();
	for j in 0..n {
		let kp = kbase.as_ptr().add(j * row_stride + off);
		let mut a0 = _mm256_setzero_ps();
		let mut a1 = _mm256_setzero_ps();
		let mut a2 = _mm256_setzero_ps();
		let mut a3 = _mm256_setzero_ps();
		let mut i = 0usize;
		while i + 32 <= hd {
			a0 = _mm256_fmadd_ps(_mm256_loadu_ps(qp.add(i)), _mm256_loadu_ps(kp.add(i)), a0);
			a1 = _mm256_fmadd_ps(
				_mm256_loadu_ps(qp.add(i + 8)),
				_mm256_loadu_ps(kp.add(i + 8)),
				a1,
			);
			a2 = _mm256_fmadd_ps(
				_mm256_loadu_ps(qp.add(i + 16)),
				_mm256_loadu_ps(kp.add(i + 16)),
				a2,
			);
			a3 = _mm256_fmadd_ps(
				_mm256_loadu_ps(qp.add(i + 24)),
				_mm256_loadu_ps(kp.add(i + 24)),
				a3,
			);
			i += 32;
		}
		while i + 8 <= hd {
			a0 = _mm256_fmadd_ps(_mm256_loadu_ps(qp.add(i)), _mm256_loadu_ps(kp.add(i)), a0);
			i += 8;
		}
		let acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
		let mut tmp = [0.0f32; 8];
		_mm256_storeu_ps(tmp.as_mut_ptr(), acc);
		let mut sfin = tmp.iter().sum::<f32>();
		while i < hd {
			sfin += *qp.add(i) * *kp.add(i);
			i += 1;
		}
		*out.get_unchecked_mut(j) = sfin * scale;
	}
}

fn dot_dispatch(a: &[f32], b: &[f32], use_avx2: bool) -> f32 {
	#[cfg(target_arch = "x86_64")]
	{
		if use_avx2 {
			unsafe { dot_avx2(a, b) }
		} else {
			dot_scalar(a, b)
		}
	}
	#[cfg(not(target_arch = "x86_64"))]
	{
		dot_scalar(a, b)
	}
}

#[inline]
fn f16_dot(a: &[u16], b: &[f32], use_avx2: bool) -> f32 {
	#[cfg(target_arch = "x86_64")]
	{
		if use_avx2 {
			unsafe { f16_dot_avx2(a, b) }
		} else {
			f16_dot_scalar(a, b)
		}
	}
	#[cfg(not(target_arch = "x86_64"))]
	{
		f16_dot_scalar(a, b)
	}
}

#[inline]
fn f16_dot_scalar(a: &[u16], b: &[f32]) -> f32 {
	let mut s = 0.0f32;
	for i in 0..a.len() {
		s += fp16_to_f32(a[i]) * b[i];
	}
	s
}

/// `f16`→`f32` widen + 8-wide FMA dot (AVX2). The 8 `f16` values are widened
/// with the verified `fp16_to_f32` and packed into one AVX register via
/// `_mm256_set_ps` (no store/load roundtrip), then FMAcc'd against `x`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn f16_dot_avx2(a: &[u16], b: &[f32]) -> f32 {
	use std::arch::x86_64::*;
	let mut acc = _mm256_setzero_ps();
	let mut i = 0;
	while i + 8 <= a.len() {
		let bv = _mm256_loadu_ps(b.as_ptr().add(i));
		let a0 = fp16_to_f32(a[i]);
		let a1 = fp16_to_f32(a[i + 1]);
		let a2 = fp16_to_f32(a[i + 2]);
		let a3 = fp16_to_f32(a[i + 3]);
		let a4 = fp16_to_f32(a[i + 4]);
		let a5 = fp16_to_f32(a[i + 5]);
		let a6 = fp16_to_f32(a[i + 6]);
		let a7 = fp16_to_f32(a[i + 7]);
		let av = _mm256_set_ps(a7, a6, a5, a4, a3, a2, a1, a0);
		acc = _mm256_fmadd_ps(av, bv, acc);
		i += 8;
	}
	let mut tmp = [0.0f32; 8];
	_mm256_storeu_ps(tmp.as_mut_ptr(), acc);
	let mut s = tmp.iter().sum::<f32>();
	while i < a.len() {
		s += fp16_to_f32(a[i]) * b[i];
		i += 1;
	}
	s
}

/// Compute `y = W @ x` for one token (`seq_len == 1`). `W` is row-major
/// `[out_rows, in_dim]` (f32). Output rows are parallelized over cores.
pub fn gemv(out_rows: usize, in_dim: usize, x: &[f32], w: &[f32], y: &mut [f32]) {
	assert_eq!(w.len(), out_rows * in_dim);
	assert_eq!(x.len(), in_dim);
	assert_eq!(y.len(), out_rows);
	let use_avx2 = features().avx2;
	par_rows(out_rows, in_dim * 4, y, |rs, _re, ysl| {
		for (i, yo) in ysl.iter_mut().enumerate() {
			let o = rs + i;
			*yo = dot_dispatch(&w[o * in_dim..(o + 1) * in_dim], x, use_avx2);
		}
	});
}

/// `f16` (half-precision) variant of [`gemv`]. The weights are widened to `f32`
/// on the fly inside the dot product (one cheap widen per element), so no full
/// `f32` copy is materialized per token.
pub fn gemv_f16(out_rows: usize, in_dim: usize, x: &[f32], w: &[u16], y: &mut [f32]) {
	assert_eq!(w.len(), out_rows * in_dim);
	assert_eq!(x.len(), in_dim);
	assert_eq!(y.len(), out_rows);
	let use_avx2 = features().avx2;
	par_rows(out_rows, in_dim * 2, y, |rs, _re, ysl| {
		for (i, yo) in ysl.iter_mut().enumerate() {
			let o = rs + i;
			*yo = f16_dot(&w[o * in_dim..(o + 1) * in_dim], x, use_avx2);
		}
	});
}
