//! The weight format of the NPU graph sets: int4 codes with one scale per
//! output row, of the matrix `W R` (`R`: a block-diagonal Hadamard rotation
//! with input signs, applied to the activation in the graph, so that
//! `x R (W R)^T = x W^T`), quantized by GPTQ (Frantar et al.) against the
//! Hessian `R^T (X^T X) R` of the matrix's input.

use crate::la::{self, Acc};
use quant::parallel::par_for;

/// Block of the rotation.
pub const ROTB: usize = 256;
/// Columns per lazy GPTQ block.
const BS: usize = 128;
/// Hessian damping, relative to the mean of the diagonal.
const DAMP: f64 = 0.01;

/// The rotation's input signs for a `k`-wide input: a fixed pseudo-random
/// +-1 sequence, the same prefix for every `k`.
pub fn signs(k: usize) -> Vec<f32> {
	let mut s = 0x9e37_79b9_7f4a_7c15u64;
	(0..k)
		.map(|_| {
			// splitmix64
			s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
			let mut z = s;
			z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
			z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
			if (z ^ (z >> 31)) >> 63 == 0 {
				1.0
			} else {
				-1.0
			}
		})
		.collect()
}

/// `m R` for every row of the `[n][k]` matrix `m`: sign multiply, then the
/// Hadamard transform of every [`ROTB`] block.
pub fn rotate_rows(m: &mut [f32], k: usize, sg: &[f32]) {
	rotate_rows_b(m, k, sg, ROTB);
}

/// [`rotate_rows`] with Hadamard blocks of `b` (a power of two dividing `k`).
pub fn rotate_rows_b(m: &mut [f32], k: usize, sg: &[f32], b: usize) {
	assert_eq!(sg.len(), k);
	assert_eq!(k % b, 0, "rotation block {b} does not divide {k}");
	for r in m.chunks_exact_mut(k) {
		r.iter_mut().zip(sg).for_each(|(v, s)| *v *= s);
	}
	la::fwht_rows(m, k, b);
}

/// `R^T h R` of the symmetric `[k][k]` matrix `h`.
pub fn rotate_sym(h: &mut Vec<f32>, k: usize, sg: &[f32]) {
	rotate_sym_b(h, k, sg, ROTB);
}

/// [`rotate_sym`] with Hadamard blocks of `b`.
pub fn rotate_sym_b(h: &mut Vec<f32>, k: usize, sg: &[f32], b: usize) {
	rotate_rows_b(h, k, sg, b);
	let mut t = la::transpose(h, k, k);
	rotate_rows_b(&mut t, k, sg, b);
	*h = t;
}

#[derive(Clone, Copy)]
struct Ptr<T>(*mut T);
// SAFETY: every use hands disjoint ranges to the tasks.
unsafe impl<T> Send for Ptr<T> {}
unsafe impl<T> Sync for Ptr<T> {}

/// One scale per row of `w` `[n][k]` for codes in `[-8, 7]`: the clip ratio
/// (of `max|w| / 7`, 21 steps from 0.5 to 1) with the least squared
/// round-to-nearest error.
pub fn row_scales(w: &[f32], n: usize, k: usize) -> Vec<f32> {
	assert_eq!(w.len(), n * k);
	let mut sc = vec![0f32; n];
	let p = Ptr(sc.as_mut_ptr());
	par_for(n, |r| {
		let p = p;
		let row = &w[r * k..(r + 1) * k];
		let amax = row.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
		let (mut best_e, mut best_s) = (f32::INFINITY, amax / 7.0);
		for g in 0..21 {
			let s = amax * (0.5 + 0.025 * g as f64) as f32 / 7.0;
			let e: f32 = row
				.iter()
				.map(|&v| {
					let d = (v / s).round_ties_even().clamp(-8.0, 7.0) * s - v;
					d * d
				})
				.sum();
			if e < best_e {
				(best_e, best_s) = (e, s);
			}
		}
		// SAFETY: task r owns sc[r]
		unsafe { *p.0.add(r) = best_s };
	});
	sc
}

/// int8 per row, round to nearest, scale `max|w| / 127`: (codes, scales).
pub fn quant_i8(w: &[f32], n: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
	let mut q = vec![0i8; n * k];
	let mut sc = vec![0f32; n];
	for r in 0..n {
		let row = &w[r * k..(r + 1) * k];
		let s = row.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12) / 127.0;
		sc[r] = s;
		for (o, v) in q[r * k..(r + 1) * k].iter_mut().zip(row) {
			*o = (v / s).round_ties_even().clamp(-127.0, 127.0) as i8;
		}
	}
	(q, sc)
}

/// Codes `[n][k]` in `[-8, 7]` packed two per byte, low nibble first.
pub fn pack4(q: &[i8]) -> Vec<u8> {
	q.chunks_exact(2)
		.map(|p| (p[0] as u8 & 15) | ((p[1] as u8 & 15) << 4))
		.collect()
}

pub fn unpack4(p: &[u8]) -> Vec<i8> {
	p.iter()
		.flat_map(|&b| [((b << 4) as i8) >> 4, (b as i8) >> 4])
		.collect()
}

/// What GPTQ needs of a Hessian: the column order and the upper factor `U`
/// of `H^-1 = U^T U` in that order, stored reversed (`x[k-1-i][k-1-j]` =
/// `U[i][j]`: the inverse of the Cholesky factor of the reversed matrix).
pub struct Factor {
	pub k: usize,
	pub perm: Vec<u32>,
	pub dead: Vec<bool>,
	pub x: Vec<f32>,
}

impl Factor {
	#[inline]
	pub fn u(&self, i: usize, j: usize) -> f32 {
		self.x[(self.k - 1 - i) * self.k + self.k - 1 - j]
	}
}

/// A Hessian ready to factor: columns in descending-diagonal order, both
/// axes reversed, damped; only the lower triangle of `hr` is filled.
pub struct Prepared {
	pub k: usize,
	pub perm: Vec<u32>,
	pub dead: Vec<bool>,
	pub hr: Vec<f32>,
}

/// Order and damp the `[k][k]` Hessian `h` (consumed): dead columns (zero
/// diagonal) get a unit diagonal, columns go in descending-diagonal order,
/// the diagonal is damped by [`DAMP`] of its mean.
pub fn prepare(mut h: Vec<f32>, k: usize) -> Prepared {
	assert_eq!(h.len(), k * k);
	let dead: Vec<bool> = (0..k).map(|i| h[i * k + i] == 0.0).collect();
	for i in 0..k {
		if dead[i] {
			h[i * k + i] = 1.0;
		}
	}
	let diag: Vec<f32> = (0..k).map(|i| h[i * k + i]).collect();
	let mut perm: Vec<u32> = (0..k as u32).collect();
	perm.sort_by(|&a, &b| diag[b as usize].total_cmp(&diag[a as usize]));
	let damp = (DAMP * diag.iter().map(|&d| d as f64).sum::<f64>() / k as f64) as f32;
	let mut hr = vec![0f32; k * k];
	let p = Ptr(hr.as_mut_ptr());
	let (hs, pm) = (&h, &perm);
	par_for(k, |i| {
		let p = p;
		let src = &hs[pm[k - 1 - i] as usize * k..][..k];
		// SAFETY: task i owns row i of hr
		let dst = unsafe { std::slice::from_raw_parts_mut(p.0.add(i * k), k) };
		for (j, d) in dst.iter_mut().enumerate().take(i + 1) {
			*d = src[pm[k - 1 - j] as usize];
		}
		dst[i] += damp;
	});
	Prepared { k, perm, dead, hr }
}

/// The inverse of the Cholesky factor of `hr` (its lower triangle), on the
/// CPU cores.
pub fn inv_chol(mut hr: Vec<f32>, k: usize) -> Result<Vec<f32>, String> {
	la::cholesky_lower(&mut hr, k)?;
	Ok(la::tri_inv_lower(&hr, k))
}

impl Prepared {
	/// With `x` = the inverse of the Cholesky factor of [`Prepared::hr`].
	pub fn finish(self, x: Vec<f32>) -> Factor {
		Factor {
			k: self.k,
			perm: self.perm,
			dead: self.dead,
			x,
		}
	}
}

/// Factor the `[k][k]` Hessian `h` (consumed) on the CPU cores.
pub fn factor(h: Vec<f32>, k: usize) -> Result<Factor, String> {
	let mut p = prepare(h, k);
	let x = inv_chol(std::mem::take(&mut p.hr), k)?;
	Ok(p.finish(x))
}

/// GPTQ of `w` `[n][k]` (consumed) with the fixed row scales `sc`: the codes
/// `[n][k]` in `[-8, 7]`.
pub fn gptq(mut w: Vec<f32>, n: usize, f: &Factor, sc: &[f32]) -> Vec<i8> {
	let k = f.k;
	assert_eq!(w.len(), n * k);
	assert_eq!(sc.len(), n);
	// columns in the factor's order, dead ones zero
	{
		let p = Ptr(w.as_mut_ptr());
		par_for(n, |r| {
			let p = p;
			// SAFETY: task r owns row r
			let row = unsafe { std::slice::from_raw_parts_mut(p.0.add(r * k), k) };
			let src = row.to_vec();
			for (c, d) in row.iter_mut().enumerate() {
				let o = f.perm[c] as usize;
				*d = if f.dead[o] { 0.0 } else { src[o] };
			}
		});
	}
	let mut q = vec![0i8; n * k];
	let mut e = vec![0f32; n * BS];
	let mut i1 = 0;
	while i1 < k {
		let i2 = (i1 + BS).min(k);
		let bs = i2 - i1;
		// U of the block, [bs][bs] row-major (upper)
		let mut u1 = vec![0f32; bs * bs];
		for i in 0..bs {
			for j in i..bs {
				u1[i * bs + j] = f.u(i1 + i, i1 + j);
			}
		}
		let (wp, qp, ep) = (
			Ptr(w.as_mut_ptr()),
			Ptr(q.as_mut_ptr()),
			Ptr(e.as_mut_ptr()),
		);
		let u1 = &u1;
		par_for(n.div_ceil(16), |t| {
			let (wp, qp, ep) = (wp, qp, ep);
			for r in t * 16..((t + 1) * 16).min(n) {
				// SAFETY: the task owns rows [16 t, 16 (t + 1)) of w, q and e
				let (wr, qr, er) = unsafe {
					(
						std::slice::from_raw_parts_mut(wp.0.add(r * k + i1), bs),
						std::slice::from_raw_parts_mut(qp.0.add(r * k + i1), bs),
						std::slice::from_raw_parts_mut(ep.0.add(r * BS), bs),
					)
				};
				let s = sc[r];
				for i in 0..bs {
					let c = (wr[i] / s).round_ties_even().clamp(-8.0, 7.0);
					qr[i] = c as i8;
					let err = (wr[i] - c * s) / u1[i * bs + i];
					er[i] = err;
					for (wv, uv) in wr[i..].iter_mut().zip(&u1[i * bs + i..(i + 1) * bs]) {
						*wv -= err * uv;
					}
				}
			}
		});
		if i2 < k {
			// W[:, i2..] -= E U[i1..i2, i2..]: as E B^T with B[j][t] = U[i1 + t][i2 + j]
			let m = k - i2;
			let mut b = vec![0f32; m * bs];
			let bp = Ptr(b.as_mut_ptr());
			par_for(m, |j| {
				let bp = bp;
				for t in 0..bs {
					// SAFETY: task j owns row j of b
					unsafe { *bp.0.add(j * bs + t) = f.u(i1 + t, i2 + j) };
				}
			});
			// SAFETY: E is [n][BS] (bs columns used), B [m][bs], the target
			// columns [i2, k) of w's n rows
			unsafe {
				la::gemm_nt_raw(
					e.as_ptr(),
					BS,
					n,
					b.as_ptr(),
					bs,
					m,
					bs,
					w.as_mut_ptr().add(i2),
					k,
					Acc::Sub,
				)
			};
		}
		i1 = i2;
	}
	drop(w);
	// back to the stored column order
	let mut out = vec![0i8; n * k];
	let op = Ptr(out.as_mut_ptr());
	let qs = &q;
	par_for(n, |r| {
		let op = op;
		for c in 0..k {
			// SAFETY: task r owns row r of out
			unsafe { *op.0.add(r * k + f.perm[c] as usize) = qs[r * k + c] };
		}
	});
	out
}

/// Round to nearest with the same scales (no Hessian).
pub fn rtn(w: &[f32], n: usize, k: usize, sc: &[f32]) -> Vec<i8> {
	let mut q = vec![0i8; n * k];
	for r in 0..n {
		for (o, v) in q[r * k..(r + 1) * k].iter_mut().zip(&w[r * k..(r + 1) * k]) {
			*o = (v / sc[r]).round_ties_even().clamp(-8.0, 7.0) as i8;
		}
	}
	q
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn packs_nibbles() {
		let q: Vec<i8> = vec![-8, 7, 0, -1, 3, -4];
		let p = pack4(&q);
		assert_eq!(p, [0x78, 0xf0, 0xc3]);
		assert_eq!(unpack4(&p), q);
	}

	#[test]
	fn rotation_is_orthogonal() {
		let k = 512;
		let sg = signs(k);
		assert_eq!(&signs(1024)[..k], &sg[..]);
		let w0: Vec<f32> = (0..2 * k)
			.map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
			.collect();
		let x0: Vec<f32> = (0..k)
			.map(|i| ((i * 13 % 59) as f32 - 29.0) / 29.0)
			.collect();
		let (mut w, mut x) = (w0.clone(), x0.clone());
		rotate_rows(&mut w, k, &sg);
		rotate_rows(&mut x, k, &sg);
		for r in 0..2 {
			let a: f32 = w0[r * k..(r + 1) * k]
				.iter()
				.zip(&x0)
				.map(|(a, b)| a * b)
				.sum();
			let b: f32 = w[r * k..(r + 1) * k]
				.iter()
				.zip(&x)
				.map(|(a, b)| a * b)
				.sum();
			assert!((a - b).abs() < 1e-3, "{a} {b}");
		}
	}

	/// GPTQ must beat round-to-nearest in the Hessian-weighted error.
	#[test]
	fn gptq_lowers_the_output_error() {
		let (n, k, s) = (24usize, 384usize, 600usize);
		let mut seed = 7u64;
		let mut rnd = || {
			seed ^= seed << 13;
			seed ^= seed >> 7;
			seed ^= seed << 17;
			((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
		};
		// correlated inputs: x = z M
		let m: Vec<f32> = (0..k * k).map(|_| rnd()).collect();
		let z: Vec<f32> = (0..s * k).map(|_| rnd()).collect();
		let mt = la::transpose(&m, k, k);
		let x = la::linear(&z, s, k, &mt, k);
		let xt = la::transpose(&x, s, k);
		let h = la::linear(&xt, k, s, &xt, k);
		let w: Vec<f32> = (0..n * k).map(|_| rnd()).collect();
		let sc = row_scales(&w, n, k);
		let f = factor(h, k).unwrap();
		let q = gptq(w.clone(), n, &f, &sc);
		let q0 = rtn(&w, n, k, &sc);
		let err = |q: &[i8]| -> f64 {
			let d: Vec<f32> = (0..n * k).map(|i| q[i] as f32 * sc[i / k] - w[i]).collect();
			la::linear(&x, s, k, &d, n)
				.iter()
				.map(|v| (*v as f64).powi(2))
				.sum()
		};
		let (eg, er) = (err(&q), err(&q0));
		assert!(eg < 0.7 * er, "gptq {eg} vs rtn {er}");
	}
}
