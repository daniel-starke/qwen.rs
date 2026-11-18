//! The handful of dense-`f32` kernels the vision tower needs.
//!
//! The tower is ~0.46 B parameters against the text model's 27 B, and it runs
//! once per image rather than once per token, so it does **not** get the
//! K-quant GEMM treatment in `quant`. What it does need is (a) sequence-major
//! output, because every consumer here is per-patch, and (b) enough blocking
//! that the weight row is not re-read once per patch: at 2304 patches a naive
//! row-at-a-time loop re-streams the whole 0.9 GB of weights 2304 times.

use quant::par_for;

/// AVX2 dot product with four accumulator chains.
///
/// The chain count is the whole point: an FMA has ~4 cycles of latency and the
/// core can issue two per cycle, so a dot with two accumulators runs at 1/4 of
/// peak no matter how well the data caches. The tower's inner dimensions
/// (1152 / 4304 / 4608) are all multiples of 32, so the tails never run on the
/// real model.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
	use std::arch::x86_64::*;
	let n = a.len();
	let mut acc = [_mm256_setzero_ps(); 4];
	let mut i = 0;
	while i + 32 <= n {
		for (k, acc) in acc.iter_mut().enumerate() {
			*acc = _mm256_fmadd_ps(
				_mm256_loadu_ps(a.as_ptr().add(i + k * 8)),
				_mm256_loadu_ps(b.as_ptr().add(i + k * 8)),
				*acc,
			);
		}
		i += 32;
	}
	while i + 8 <= n {
		acc[0] = _mm256_fmadd_ps(
			_mm256_loadu_ps(a.as_ptr().add(i)),
			_mm256_loadu_ps(b.as_ptr().add(i)),
			acc[0],
		);
		i += 8;
	}
	let v = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
	let mut r = hsum256(v);
	while i < n {
		r += a[i] * b[i];
		i += 1;
	}
	r
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
	use std::arch::x86_64::*;
	let s = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
	let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
	let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
	_mm_cvtss_f32(s)
}

/// One weight row against **two** positions at once: eight independent FMA
/// chains, and the row is loaded once for both. This is the tower's inner
/// loop; the single-position form below is only the odd tail.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot2_avx2(w: &[f32], x0: &[f32], x1: &[f32]) -> (f32, f32) {
	use std::arch::x86_64::*;
	let n = w.len();
	let mut a = [_mm256_setzero_ps(); 4];
	let mut b = [_mm256_setzero_ps(); 4];
	let mut i = 0;
	while i + 32 <= n {
		for k in 0..4 {
			let wv = _mm256_loadu_ps(w.as_ptr().add(i + k * 8));
			a[k] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x0.as_ptr().add(i + k * 8)), a[k]);
			b[k] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x1.as_ptr().add(i + k * 8)), b[k]);
		}
		i += 32;
	}
	while i + 8 <= n {
		let wv = _mm256_loadu_ps(w.as_ptr().add(i));
		a[0] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x0.as_ptr().add(i)), a[0]);
		b[0] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x1.as_ptr().add(i)), b[0]);
		i += 8;
	}
	let av = _mm256_add_ps(_mm256_add_ps(a[0], a[1]), _mm256_add_ps(a[2], a[3]));
	let bv = _mm256_add_ps(_mm256_add_ps(b[0], b[1]), _mm256_add_ps(b[2], b[3]));
	let (mut ra, mut rb) = (hsum256(av), hsum256(bv));
	while i < n {
		ra += w[i] * x0[i];
		rb += w[i] * x1[i];
		i += 1;
	}
	(ra, rb)
}

fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
	let mut s = 0.0f32;
	for i in 0..a.len() {
		s += a[i] * b[i];
	}
	s
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
	debug_assert_eq!(a.len(), b.len());
	#[cfg(target_arch = "x86_64")]
	{
		if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
			return unsafe { dot_avx2(a, b) };
		}
	}
	dot_scalar(a, b)
}

/// Bytes of activation a position tile may occupy. The tile is what stays
/// resident while a weight row sweeps it, so it is sized to the L2 this host
/// has (2 MB) rather than to a fixed position count.
const TILE_BYTES: usize = 1 << 21;

/// `y[t, o] = dot(x[t, ..], w[o, ..]) + bias[o]`, with `w` row-major
/// `[out_dim, in_dim]` and `y` sequence-major `[n, out_dim]`.
///
/// Parallelized over **output rows**, not positions: each weight row is read
/// by exactly one worker, so the whole weight matrix is streamed once per
/// position tile. Splitting the positions across workers instead would make
/// every worker read every row: for a 256-patch image that was 16 passes over
/// the tower's 1.8 GB of weights, and it dominated the encode time.
pub fn matmul_bias(
	x: &[f32],
	n: usize,
	in_dim: usize,
	w: &[f32],
	bias: Option<&[f32]>,
	out_dim: usize,
) -> Vec<f32> {
	assert_eq!(x.len(), n * in_dim);
	assert_eq!(w.len(), out_dim * in_dim);
	if let Some(b) = bias {
		assert_eq!(b.len(), out_dim);
	}
	let mut y = quant::uninit_vec(n * out_dim);
	if n == 0 || out_dim == 0 {
		return y;
	}
	let ptile = (TILE_BYTES / (in_dim * 4)).clamp(1, n);
	let yp = y.as_mut_ptr() as usize;
	// Resolved once per call rather than once per dot: the AVX2 kernels carry
	// `#[target_feature]`, so a per-dot check also blocks inlining.
	#[cfg(target_arch = "x86_64")]
	let avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
	#[cfg(not(target_arch = "x86_64"))]
	let avx2 = false;

	let mut t0 = 0usize;
	while t0 < n {
		let t1 = (t0 + ptile).min(n);
		par_for(out_dim, |o| {
			let wrow = &w[o * in_dim..(o + 1) * in_dim];
			let b0 = bias.map(|b| b[o]).unwrap_or(0.0);
			// SAFETY (all writes below): within a tile each `o` is visited by
			// one worker and each `(t, o)` is written exactly once; `y` has
			// `n * out_dim` elements.
			let put = |t: usize, v: f32| unsafe { *(yp as *mut f32).add(t * out_dim + o) = v };
			let mut t = t0;
			#[cfg(target_arch = "x86_64")]
			if avx2 {
				while t + 2 <= t1 {
					let (a, b) = unsafe {
						dot2_avx2(
							wrow,
							&x[t * in_dim..(t + 1) * in_dim],
							&x[(t + 1) * in_dim..(t + 2) * in_dim],
						)
					};
					put(t, a + b0);
					put(t + 1, b + b0);
					t += 2;
				}
			}
			while t < t1 {
				let xt = &x[t * in_dim..(t + 1) * in_dim];
				let v = if avx2 {
					#[cfg(target_arch = "x86_64")]
					unsafe {
						dot_avx2(wrow, xt)
					}
					#[cfg(not(target_arch = "x86_64"))]
					dot_scalar(wrow, xt)
				} else {
					dot_scalar(wrow, xt)
				};
				put(t, v + b0);
				t += 1;
			}
		});
		t0 = t1;
	}
	y
}

/// LayerNorm with weight and bias, in place over each `dim`-long row.
pub fn layer_norm(x: &mut [f32], dim: usize, w: &[f32], b: &[f32], eps: f32) {
	assert_eq!(w.len(), dim);
	assert_eq!(b.len(), dim);
	let n = x.len() / dim;
	let xp = x.as_mut_ptr() as usize;
	par_for(n, |t| {
		// SAFETY: rows are disjoint and each index `t` is visited once.
		let row = unsafe { std::slice::from_raw_parts_mut((xp as *mut f32).add(t * dim), dim) };
		let mean = row.iter().sum::<f32>() / dim as f32;
		let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
		let inv = 1.0 / (var + eps).sqrt();
		for i in 0..dim {
			row[i] = (row[i] - mean) * inv * w[i] + b[i];
		}
	});
}

/// GELU, tanh approximation: the one `ggml_gelu` (and `gelu_pytorch_tanh`)
/// use, so the tower matches the reference implementation rather than the
/// slightly different erf form.
#[inline]
pub fn gelu(x: f32) -> f32 {
	const SQRT_2_OVER_PI: f32 = 0.797_884_6;
	0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
}

/// `erf`, Abramowitz & Stegun 7.1.26 in f64 (|error| < 1.5e-7).
pub fn erf(x: f64) -> f64 {
	let s = x.signum();
	let x = x.abs();
	let t = 1.0 / (1.0 + 0.327_591_1 * x);
	let y = 1.0
		- (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
			+ 0.254_829_592)
			* t * (-x * x).exp();
	s * y
}

/// GELU, exact (erf) form: torch's `nn.GELU()`.
#[inline]
pub fn gelu_erf(x: f32) -> f32 {
	(0.5 * x as f64 * (1.0 + erf(x as f64 / std::f64::consts::SQRT_2))) as f32
}

/// In-place softmax over one row, in the numerically stable max-subtracted
/// form.
pub fn softmax(row: &mut [f32]) {
	let mut m = f32::NEG_INFINITY;
	for &v in row.iter() {
		if v > m {
			m = v;
		}
	}
	let mut sum = 0.0f32;
	for v in row.iter_mut() {
		*v = (*v - m).exp();
		sum += *v;
	}
	let inv = 1.0 / sum;
	for v in row.iter_mut() {
		*v *= inv;
	}
}
