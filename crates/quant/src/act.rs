//! Elementwise activations on the per-token paths: the SwiGLU gate, the SSM
//! output gate and the SSM block's causal conv1d + silu.
//!
//! AVX2 through ggml's `v_expf` (as `ggml_v_silu` does); scalar
//! `x / (1 + e^-x)` elsewhere and for the tail. ~10x the scalar `f32::exp`
//! loops they replace, which were most of a decode token's elementwise time
//! and too slow to run inline on a verify batch (see
//! `model::forward::small_rows`).

#[inline]
fn silu(x: f32) -> f32 {
	x / (1.0 + (-x).exp())
}

/// `out[i] = silu(g[i]) * u[i]`.
pub fn silu_mul(g: &[f32], u: &[f32], out: &mut [f32]) {
	let n = out.len();
	assert!(g.len() >= n && u.len() >= n);
	let mut i = 0;
	#[cfg(target_arch = "x86_64")]
	if super::features().avx2 {
		// SAFETY: AVX2 + FMA checked above; every index is below `n`.
		unsafe { silu_mul_avx2(g.as_ptr(), u.as_ptr(), out.as_mut_ptr(), n) };
		i = n / 8 * 8;
	}
	for i in i..n {
		out[i] = silu(g[i]) * u[i];
	}
}

/// `x[i] *= silu(g[i])`.
pub fn silu_mul_assign(g: &[f32], x: &mut [f32]) {
	let n = x.len();
	assert!(g.len() >= n);
	let mut i = 0;
	#[cfg(target_arch = "x86_64")]
	if super::features().avx2 {
		// SAFETY: as in `silu_mul`; `u` and `out` may alias (8 lanes are
		// loaded before they are stored).
		unsafe { silu_mul_avx2(g.as_ptr(), x.as_ptr(), x.as_mut_ptr(), n) };
		i = n / 8 * 8;
	}
	for i in i..n {
		x[i] *= silu(g[i]);
	}
}

/// Causal conv1d over one token, then silu: `out[c] = silu(sum_f wt[f * C +
/// c] * rows[f][c])` with `C = out.len()`. `wt` is the conv kernel
/// **tap-major** (`[ksz][C]`, the transpose of the GGUF's `[C][ksz]`), so the
/// sum vectorizes across channels; `rows[f]` is the window row for tap `f`.
pub fn conv_silu(rows: &[&[f32]], wt: &[f32], out: &mut [f32]) {
	let c_n = out.len();
	let ksz = rows.len();
	assert!(wt.len() >= ksz * c_n && rows.iter().all(|r| r.len() >= c_n));
	let mut c0 = 0;
	#[cfg(target_arch = "x86_64")]
	if super::features().avx2 {
		// SAFETY: AVX2 + FMA checked above; every index is below `c_n`.
		unsafe { conv_silu_avx2(rows, wt, out, c_n) };
		c0 = c_n / 8 * 8;
	}
	for c in c0..c_n {
		let mut s = 0.0f32;
		for f in 0..ksz {
			s += wt[f * c_n + c] * rows[f][c];
		}
		out[c] = silu(s);
	}
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn silu8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
	use std::arch::x86_64::*;
	let e = super::attn::v_expf(_mm256_xor_ps(x, _mm256_set1_ps(-0.0)));
	_mm256_div_ps(x, _mm256_add_ps(_mm256_set1_ps(1.0), e))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn silu_mul_avx2(g: *const f32, u: *const f32, out: *mut f32, n: usize) {
	use std::arch::x86_64::*;
	let mut i = 0;
	while i + 8 <= n {
		let s = silu8(_mm256_loadu_ps(g.add(i)));
		_mm256_storeu_ps(out.add(i), _mm256_mul_ps(s, _mm256_loadu_ps(u.add(i))));
		i += 8;
	}
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn conv_silu_avx2(rows: &[&[f32]], wt: &[f32], out: &mut [f32], c_n: usize) {
	use std::arch::x86_64::*;
	let mut c = 0;
	while c + 8 <= c_n {
		let mut acc = _mm256_setzero_ps();
		for (f, r) in rows.iter().enumerate() {
			acc = _mm256_fmadd_ps(
				_mm256_loadu_ps(wt.as_ptr().add(f * c_n + c)),
				_mm256_loadu_ps(r.as_ptr().add(c)),
				acc,
			);
		}
		_mm256_storeu_ps(out.as_mut_ptr().add(c), silu8(acc));
		c += 8;
	}
}
