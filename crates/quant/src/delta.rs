//! Gated-delta-net state update for a block of value columns.
//!
//! The state of one head is `st[i * s + j]` (`i` = key dim, `j` = value dim),
//! and the recurrence only ever mixes along `i`: column `j` of the next state,
//! and output `o[j]`, depend on column `j` of the old state and nothing else in
//! it. So a head splits into independent column blocks, which is what lets the
//! prefill scan hand each worker a `(head, column block)` and have it walk every
//! token of the chunk on its own, instead of one pool dispatch per token.
//!
//! The AVX2 path is **bit-identical** to the scalar one: `mul` and `add` stay
//! separate (no `fma`), and every accumulator sums over `i` in the same order.

/// Columns `[j0, j0 + o.len())` of one gated-delta-net step on a head's
/// `s x s` state `st`: `st *= g`, `d = (v - st^T k) * beta`, `st += k (x) d`,
/// `o += st^T q * qscale`. `v` and `o` are the column block's slices; `k` and
/// `q` are the whole key vectors (length `s`).
#[allow(clippy::too_many_arguments)]
pub fn delta_net_cols(
	st: &mut [f32],
	s: usize,
	j0: usize,
	k: &[f32],
	q: &[f32],
	v: &[f32],
	beta: f32,
	g: f32,
	qscale: f32,
	o: &mut [f32],
) {
	assert!(st.len() >= s * s);
	delta_net_cols_ld(st, s, s, j0, k, q, v, beta, g, qscale, o);
}

/// [`delta_net_cols`] on a state whose row `i` starts at `st[i * ld]`, so a
/// column block can be scanned in a compact `s x jb` copy (`ld = jb`, `j0 =
/// 0`). Same arithmetic, bit for bit.
///
/// Why the copy matters: in the full `s x s` state a 32-column block is 128
/// rows of 128 bytes at a 512-byte stride, which maps onto 16 of the L1's 64
/// sets; 256 lines competing for 192 ways on a P-core and 128 on an E-core,
/// so the block was re-fetched from L2 on every pass of every token.
#[allow(clippy::too_many_arguments)]
pub fn delta_net_cols_ld(
	st: &mut [f32],
	ld: usize,
	s: usize,
	j0: usize,
	k: &[f32],
	q: &[f32],
	v: &[f32],
	beta: f32,
	g: f32,
	qscale: f32,
	o: &mut [f32],
) {
	let jb = o.len();
	assert!(v.len() == jb && j0 + jb <= ld && st.len() >= (s - 1) * ld + j0 + jb);
	assert!(k.len() >= s && q.len() >= s);
	let mut j = 0;
	#[cfg(target_arch = "x86_64")]
	if crate::features().avx2 {
		// SAFETY: AVX2+FMA detected; every access is inside the asserted bounds.
		unsafe {
			while j + 64 <= jb {
				cols_avx2::<8>(
					st,
					ld,
					s,
					j0 + j,
					k,
					q,
					&v[j..],
					beta,
					g,
					qscale,
					&mut o[j..],
				);
				j += 64;
			}
			while j + 32 <= jb {
				cols_avx2::<4>(
					st,
					ld,
					s,
					j0 + j,
					k,
					q,
					&v[j..],
					beta,
					g,
					qscale,
					&mut o[j..],
				);
				j += 32;
			}
			while j + 8 <= jb {
				cols_avx2::<1>(
					st,
					ld,
					s,
					j0 + j,
					k,
					q,
					&v[j..],
					beta,
					g,
					qscale,
					&mut o[j..],
				);
				j += 8;
			}
		}
	}
	if j < jb {
		cols_scalar(
			st,
			ld,
			s,
			j0 + j,
			k,
			q,
			&v[j..],
			beta,
			g,
			qscale,
			&mut o[j..],
		);
	}
}

#[allow(clippy::too_many_arguments)]
fn cols_scalar(
	st: &mut [f32],
	ld: usize,
	s: usize,
	j0: usize,
	k: &[f32],
	q: &[f32],
	v: &[f32],
	beta: f32,
	g: f32,
	qscale: f32,
	o: &mut [f32],
) {
	let jb = o.len();
	let mut sk = vec![0.0f32; jb];
	for i in 0..s {
		let ki = k[i];
		let row = &mut st[i * ld + j0..i * ld + j0 + jb];
		for j in 0..jb {
			let val = row[j] * g;
			row[j] = val;
			sk[j] += val * ki;
		}
	}
	for j in 0..jb {
		sk[j] = (v[j] - sk[j]) * beta;
	}
	for i in 0..s {
		let ki = k[i];
		let qi = q[i] * qscale;
		let row = &mut st[i * ld + j0..i * ld + j0 + jb];
		for j in 0..jb {
			let val = row[j] + ki * sk[j];
			row[j] = val;
			o[j] += val * qi;
		}
	}
}

/// `8 * NV` columns, the accumulators held in `NV` registers each.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn cols_avx2<const NV: usize>(
	st: &mut [f32],
	ld: usize,
	s: usize,
	j0: usize,
	k: &[f32],
	q: &[f32],
	v: &[f32],
	beta: f32,
	g: f32,
	qscale: f32,
	o: &mut [f32],
) {
	use std::arch::x86_64::*;
	let base = st.as_mut_ptr().add(j0);
	let gv = _mm256_set1_ps(g);
	let mut sk = [_mm256_setzero_ps(); NV];
	for i in 0..s {
		let ki = _mm256_set1_ps(*k.get_unchecked(i));
		let row = base.add(i * ld);
		for c in 0..NV {
			let val = _mm256_mul_ps(_mm256_loadu_ps(row.add(8 * c)), gv);
			_mm256_storeu_ps(row.add(8 * c), val);
			sk[c] = _mm256_add_ps(sk[c], _mm256_mul_ps(val, ki));
		}
	}
	let bv = _mm256_set1_ps(beta);
	let mut d = [_mm256_setzero_ps(); NV];
	let mut acc = [_mm256_setzero_ps(); NV];
	for c in 0..NV {
		d[c] = _mm256_mul_ps(
			_mm256_sub_ps(_mm256_loadu_ps(v.as_ptr().add(8 * c)), sk[c]),
			bv,
		);
		acc[c] = _mm256_loadu_ps(o.as_ptr().add(8 * c));
	}
	for i in 0..s {
		let ki = _mm256_set1_ps(*k.get_unchecked(i));
		let qi = _mm256_set1_ps(*q.get_unchecked(i) * qscale);
		let row = base.add(i * ld);
		for c in 0..NV {
			let val = _mm256_add_ps(_mm256_loadu_ps(row.add(8 * c)), _mm256_mul_ps(ki, d[c]));
			_mm256_storeu_ps(row.add(8 * c), val);
			acc[c] = _mm256_add_ps(acc[c], _mm256_mul_ps(val, qi));
		}
	}
	for c in 0..NV {
		_mm256_storeu_ps(o.as_mut_ptr().add(8 * c), acc[c]);
	}
}
