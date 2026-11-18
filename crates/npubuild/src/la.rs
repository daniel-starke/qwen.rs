//! Dense f32 linear algebra on the CPU cores for the calibration forward and
//! GPTQ: one tiled AVX2 GEMM (`C (-)= A B^T`), and on top of it a blocked
//! Cholesky factorization and a triangular inverse.

use quant::parallel::par_for;

/// What a GEMM does with its result.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Acc {
	Set,
	Add,
	Sub,
}

#[derive(Clone, Copy)]
struct Raw(*const f32, *const f32, *mut f32);
// SAFETY: the tasks of one GEMM write disjoint parts of C and only read A, B.
unsafe impl Send for Raw {}
unsafe impl Sync for Raw {}

/// Rows of A per tile: a tile of `PT x k` floats is read once per row of B.
const PT: usize = 16;
/// Rows of B per task.
const NB: usize = 64;

/// `C[m][n] (acc)= A[m][k] B[n][k]^T` on strided storage.
///
/// # Safety
/// `a`, `b`, `c` must hold `m` / `n` / `m` rows of `k` / `k` / `n` readable
/// (writable for `c`) floats at strides `lda` / `ldb` / `ldc`, and `c` must
/// not overlap `a` or `b`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_nt_raw(
	a: *const f32,
	lda: usize,
	m: usize,
	b: *const f32,
	ldb: usize,
	n: usize,
	k: usize,
	c: *mut f32,
	ldc: usize,
	acc: Acc,
) {
	if m == 0 || n == 0 {
		return;
	}
	let avx = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
	let (tiles, chunks) = (m.div_ceil(PT), n.div_ceil(NB));
	let raw = Raw(a, b, c);
	par_for(tiles * chunks, move |t| {
		let raw = raw;
		let (ti, ci) = (t / chunks, t % chunks);
		let (i0, i1) = (ti * PT, ((ti + 1) * PT).min(m));
		let (j0, j1) = (ci * NB, ((ci + 1) * NB).min(n));
		// SAFETY: per the caller's contract; this task owns C[i0..i1][j0..j1]
		unsafe {
			if avx {
				tile_avx2(
					raw.0.add(i0 * lda),
					lda,
					i1 - i0,
					raw.1.add(j0 * ldb),
					ldb,
					j1 - j0,
					k,
					raw.2.add(i0 * ldc + j0),
					ldc,
					acc,
				);
			} else {
				tile_scalar(
					raw.0.add(i0 * lda),
					lda,
					i1 - i0,
					raw.1.add(j0 * ldb),
					ldb,
					j1 - j0,
					k,
					raw.2.add(i0 * ldc + j0),
					ldc,
					acc,
				);
			}
		}
	});
}

#[inline(always)]
unsafe fn put(c: *mut f32, v: f32, acc: Acc) {
	match acc {
		Acc::Set => *c = v,
		Acc::Add => *c += v,
		Acc::Sub => *c -= v,
	}
}

#[allow(clippy::too_many_arguments)]
unsafe fn tile_scalar(
	a: *const f32,
	lda: usize,
	m: usize,
	b: *const f32,
	ldb: usize,
	n: usize,
	k: usize,
	c: *mut f32,
	ldc: usize,
	acc: Acc,
) {
	for j in 0..n {
		for i in 0..m {
			let mut s = 0.0f32;
			for x in 0..k {
				s += *a.add(i * lda + x) * *b.add(j * ldb + x);
			}
			put(c.add(i * ldc + j), s, acc);
		}
	}
}

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_avx2(
	a: *const f32,
	lda: usize,
	m: usize,
	b: *const f32,
	ldb: usize,
	n: usize,
	k: usize,
	c: *mut f32,
	ldc: usize,
	acc: Acc,
) {
	use std::arch::x86_64::*;
	#[inline(always)]
	unsafe fn hsum(v: __m256) -> f32 {
		let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
		let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
		_mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)))
	}
	let k16 = k / 16 * 16;
	for j in 0..n {
		let bj = b.add(j * ldb);
		let mut i = 0;
		// four rows of A against one row of B: the B loads are shared and
		// eight accumulators hide the FMA latency
		while i + 4 <= m {
			let (a0, a1, a2, a3) = (
				a.add(i * lda),
				a.add((i + 1) * lda),
				a.add((i + 2) * lda),
				a.add((i + 3) * lda),
			);
			let mut s = [_mm256_setzero_ps(); 8];
			let mut x = 0;
			while x < k16 {
				let (w0, w1) = (_mm256_loadu_ps(bj.add(x)), _mm256_loadu_ps(bj.add(x + 8)));
				s[0] = _mm256_fmadd_ps(_mm256_loadu_ps(a0.add(x)), w0, s[0]);
				s[1] = _mm256_fmadd_ps(_mm256_loadu_ps(a1.add(x)), w0, s[1]);
				s[2] = _mm256_fmadd_ps(_mm256_loadu_ps(a2.add(x)), w0, s[2]);
				s[3] = _mm256_fmadd_ps(_mm256_loadu_ps(a3.add(x)), w0, s[3]);
				s[4] = _mm256_fmadd_ps(_mm256_loadu_ps(a0.add(x + 8)), w1, s[4]);
				s[5] = _mm256_fmadd_ps(_mm256_loadu_ps(a1.add(x + 8)), w1, s[5]);
				s[6] = _mm256_fmadd_ps(_mm256_loadu_ps(a2.add(x + 8)), w1, s[6]);
				s[7] = _mm256_fmadd_ps(_mm256_loadu_ps(a3.add(x + 8)), w1, s[7]);
				x += 16;
			}
			let mut r = [
				hsum(_mm256_add_ps(s[0], s[4])),
				hsum(_mm256_add_ps(s[1], s[5])),
				hsum(_mm256_add_ps(s[2], s[6])),
				hsum(_mm256_add_ps(s[3], s[7])),
			];
			for x in k16..k {
				let w = *bj.add(x);
				r[0] += *a0.add(x) * w;
				r[1] += *a1.add(x) * w;
				r[2] += *a2.add(x) * w;
				r[3] += *a3.add(x) * w;
			}
			for (d, v) in r.iter().enumerate() {
				put(c.add((i + d) * ldc + j), *v, acc);
			}
			i += 4;
		}
		while i < m {
			let a0 = a.add(i * lda);
			let mut s = [_mm256_setzero_ps(); 4];
			let mut x = 0;
			let k32 = k / 32 * 32;
			while x < k32 {
				for (d, sd) in s.iter_mut().enumerate() {
					*sd = _mm256_fmadd_ps(
						_mm256_loadu_ps(a0.add(x + 8 * d)),
						_mm256_loadu_ps(bj.add(x + 8 * d)),
						*sd,
					);
				}
				x += 32;
			}
			let mut r = hsum(_mm256_add_ps(
				_mm256_add_ps(s[0], s[1]),
				_mm256_add_ps(s[2], s[3]),
			));
			for x in k32..k {
				r += *a0.add(x) * *bj.add(x);
			}
			put(c.add(i * ldc + j), r, acc);
			i += 1;
		}
	}
}

/// `y[s][n] = x[s][k] w[n][k]^T`.
pub fn linear(x: &[f32], s: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
	assert_eq!(x.len(), s * k);
	assert_eq!(w.len(), n * k);
	let mut y = vec![0f32; s * n];
	// SAFETY: the slices hold the stated rows; y is a fresh allocation
	unsafe {
		gemm_nt_raw(
			x.as_ptr(),
			k,
			s,
			w.as_ptr(),
			k,
			n,
			k,
			y.as_mut_ptr(),
			n,
			Acc::Set,
		)
	};
	y
}

/// `a^T` of an `[m][n]` matrix.
pub fn transpose(a: &[f32], m: usize, n: usize) -> Vec<f32> {
	assert_eq!(a.len(), m * n);
	let mut t = vec![0f32; m * n];
	let p = Raw(a.as_ptr(), a.as_ptr(), t.as_mut_ptr());
	const B: usize = 64;
	par_for(n.div_ceil(B), move |bj| {
		let p = p;
		for i0 in (0..m).step_by(B) {
			for j in bj * B..((bj + 1) * B).min(n) {
				for i in i0..(i0 + B).min(m) {
					// SAFETY: task bj owns rows [bj B, (bj + 1) B) of t
					unsafe { *p.2.add(j * m + i) = *p.0.add(i * n + j) };
				}
			}
		}
	});
	t
}

/// In-place Cholesky factorization `A = L L^T` of the symmetric positive
/// definite `[n][n]` matrix `a` (its lower triangle is read): on return the
/// lower triangle holds `L`, the strict upper triangle is undefined. `Err` on
/// a non-positive pivot.
pub fn cholesky_lower(a: &mut [f32], n: usize) -> Result<(), String> {
	assert_eq!(a.len(), n * n);
	const JB: usize = 256;
	const STRIP: usize = 1024;
	let mut j = 0;
	while j < n {
		let jb = JB.min(n - j);
		// the diagonal block, unblocked, accumulated in f64
		for c in j..j + jb {
			let mut d = a[c * n + c] as f64;
			for x in j..c {
				d -= (a[c * n + x] as f64) * (a[c * n + x] as f64);
			}
			if d <= 0.0 || !d.is_finite() {
				return Err(format!("cholesky: pivot {c} of {n} is {d:e}"));
			}
			let d = d.sqrt();
			a[c * n + c] = d as f32;
			for r in c + 1..j + jb {
				let mut s = a[r * n + c] as f64;
				for x in j..c {
					s -= (a[r * n + x] as f64) * (a[c * n + x] as f64);
				}
				a[r * n + c] = (s / d) as f32;
			}
		}
		let j2 = j + jb;
		if j2 == n {
			break;
		}
		// the panel below it: L21 L11^T = A21, a forward substitution per row
		let p = Raw(a.as_ptr(), a.as_ptr(), a.as_mut_ptr());
		par_for(n - j2, move |r| {
			let p = p;
			let r = j2 + r;
			// SAFETY: task r writes row r, columns [j, j2), and reads the
			// diagonal block (rows < j2), which no task writes
			unsafe {
				for c in j..j2 {
					let mut s = *p.2.add(r * n + c) as f64;
					for x in j..c {
						s -= (*p.2.add(r * n + x) as f64) * (*p.0.add(c * n + x) as f64);
					}
					*p.2.add(r * n + c) = (s / *p.0.add(c * n + c) as f64) as f32;
				}
			}
		});
		// the trailing lower triangle -= L21 L21^T, in strips of rows
		let mut r0 = j2;
		while r0 < n {
			let r1 = (r0 + STRIP).min(n);
			// SAFETY: reads columns [j, j2) of rows [j2, r1), writes columns
			// [j2, r1) of rows [r0, r1): disjoint column ranges
			unsafe {
				let base = a.as_mut_ptr();
				gemm_nt_raw(
					base.add(r0 * n + j),
					n,
					r1 - r0,
					base.add(j2 * n + j),
					n,
					r1 - j2,
					jb,
					base.add(r0 * n + j2),
					n,
					Acc::Sub,
				);
			}
			r0 = r1;
		}
		j = j2;
	}
	Ok(())
}

/// The inverse of the lower triangular `[n][n]` matrix in `l`'s lower
/// triangle, as a full `[n][n]` matrix (zero above the diagonal).
pub fn tri_inv_lower(l: &[f32], n: usize) -> Vec<f32> {
	assert_eq!(l.len(), n * n);
	let mut x = vec![0f32; n * n];
	inv_rec(l, n, 0, n, &mut x);
	x
}

fn inv_rec(l: &[f32], n: usize, o: usize, m: usize, x: &mut [f32]) {
	if m <= 128 {
		// unblocked: column by column, X L = I read row-wise (f64 sums)
		for i in o..o + m {
			let d = l[i * n + i] as f64;
			x[i * n + i] = (1.0 / d) as f32;
			for j in o..i {
				let mut s = 0f64;
				for t in j..i {
					s += (l[i * n + t] as f64) * (x[t * n + j] as f64);
				}
				x[i * n + j] = (-s / d) as f32;
			}
		}
		return;
	}
	let h = m / 2;
	let (o2, m2) = (o + h, m - h);
	inv_rec(l, n, o, h, x);
	inv_rec(l, n, o2, m2, x);
	// X21 = -X22 (L21 X11), each product as A B^T with a transposed copy
	let mut x11t = vec![0f32; h * h];
	for i in 0..h {
		for j in 0..=i {
			x11t[j * h + i] = x[(o + i) * n + o + j];
		}
	}
	// Both products have a triangular factor: only the non-zero part of
	// each block of columns (rows) is multiplied.
	const BLK: usize = 512;
	let mut t = vec![0f32; m2 * h];
	let mut j0 = 0;
	while j0 < h {
		let j1 = (j0 + BLK).min(h);
		// SAFETY: L21 is rows [o2, o2 + m2), columns [o, o + h) of l; columns
		// [j0, j1) of T read X11's rows >= j0 only (X11 is lower triangular)
		unsafe {
			gemm_nt_raw(
				l.as_ptr().add(o2 * n + o + j0),
				n,
				m2,
				x11t.as_ptr().add(j0 * h + j0),
				h,
				j1 - j0,
				h - j0,
				t.as_mut_ptr().add(j0),
				h,
				Acc::Set,
			)
		};
		j0 = j1;
	}
	drop(x11t);
	let tt = transpose(&t, m2, h);
	drop(t);
	let mut i0 = 0;
	while i0 < m2 {
		let i1 = (i0 + BLK).min(m2);
		// SAFETY: X22 is rows / columns [o2, o2 + m2) of x (read), X21 rows
		// [o2, o2 + m2), columns [o, o + h) (written): disjoint columns. Rows
		// [i0, i1) of X22 are zero from column i1 on.
		unsafe {
			let base = x.as_mut_ptr();
			gemm_nt_raw(
				base.add((o2 + i0) * n + o2),
				n,
				i1 - i0,
				tt.as_ptr(),
				m2,
				h,
				i1,
				base.add((o2 + i0) * n + o),
				n,
				Acc::Sub,
			);
		}
		i0 = i1;
	}
}

/// In-place fast Walsh-Hadamard transform of every `b`-block of `v`, scaled
/// by `1 / sqrt(b)`: `v_block @ H_b` for the Sylvester Hadamard matrix
/// (symmetric, so also `H_b v_block`).
pub fn fwht_blocks(v: &mut [f32], b: usize) {
	debug_assert!(b.is_power_of_two() && v.len().is_multiple_of(b));
	let sc = 1.0 / (b as f32).sqrt();
	for blk in v.chunks_exact_mut(b) {
		let mut h = 1;
		while h < b {
			for i in (0..b).step_by(2 * h) {
				for j in i..i + h {
					let (x, y) = (blk[j], blk[j + h]);
					blk[j] = x + y;
					blk[j + h] = x - y;
				}
			}
			h *= 2;
		}
		for x in blk {
			*x *= sc;
		}
	}
}

/// [`fwht_blocks`] on every row of an `[n][k]` matrix, in parallel.
pub fn fwht_rows(m: &mut [f32], k: usize, b: usize) {
	let p = Raw(m.as_ptr(), m.as_ptr(), m.as_mut_ptr());
	par_for(m.len() / k, move |r| {
		let p = p;
		// SAFETY: task r owns row r
		fwht_blocks(
			unsafe { std::slice::from_raw_parts_mut(p.2.add(r * k), k) },
			b,
		);
	});
}

#[cfg(test)]
mod tests {
	use super::*;

	fn rnd(n: usize, seed: u64) -> Vec<f32> {
		let mut s = seed | 1;
		(0..n)
			.map(|_| {
				s ^= s << 13;
				s ^= s >> 7;
				s ^= s << 17;
				((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
			})
			.collect()
	}

	#[test]
	fn gemm_matches_the_plain_loops() {
		for (m, n, k) in [(1, 1, 1), (5, 7, 33), (37, 70, 100), (16, 64, 48)] {
			let (a, b) = (rnd(m * k, 1), rnd(n * k, 2));
			let y = linear(&a, m, k, &b, n);
			for i in 0..m {
				for j in 0..n {
					let r: f32 = (0..k).map(|x| a[i * k + x] * b[j * k + x]).sum();
					assert!((y[i * n + j] - r).abs() < 1e-4, "{m} {n} {k}: {i} {j}");
				}
			}
		}
	}

	#[test]
	fn cholesky_and_inverse() {
		for n in [5usize, 300, 700, 1500] {
			// A = M M^T + n I
			let m = rnd(n * n, 3);
			let mut a = linear(&m, n, n, &m, n);
			for i in 0..n {
				a[i * n + i] += n as f32;
			}
			let a0 = a.clone();
			cholesky_lower(&mut a, n).unwrap();
			for i in 0..n {
				for j in i + 1..n {
					a[i * n + j] = 0.0;
				}
			}
			let llt = linear(&a, n, n, &a, n);
			let e = llt
				.iter()
				.zip(&a0)
				.map(|(x, y)| (x - y).abs())
				.fold(0f32, f32::max);
			assert!(e < 2e-3 * n as f32, "n {n}: |L L^T - A| {e}");
			let x = tri_inv_lower(&a, n);
			// X L = I
			let lt = transpose(&a, n, n);
			let p = linear(&x, n, n, &lt, n);
			let mut e = 0f32;
			for i in 0..n {
				for j in 0..n {
					e = e.max((p[i * n + j] - (i == j) as u8 as f32).abs());
				}
			}
			assert!(e < 1e-3, "n {n}: |X L - I| {e}");
		}
	}

	#[test]
	fn fwht_is_its_own_inverse() {
		let v0 = rnd(512, 5);
		let mut v = v0.clone();
		fwht_blocks(&mut v, 256);
		assert!((v[0] - v0[..256].iter().sum::<f32>() / 16.0).abs() < 1e-4);
		fwht_blocks(&mut v, 256);
		assert!(v.iter().zip(&v0).all(|(a, b)| (a - b).abs() < 1e-5));
	}
}
