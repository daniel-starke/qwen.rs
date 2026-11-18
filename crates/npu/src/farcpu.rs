//! Far attention blocks on the CPU cores. During a far set's prefill the
//! cores are idle while the NPU and the scan device run the attention
//! layers' far blocks, and one block (256 queries x 1022 keys x 24 heads,
//! ~6.4 GFLOP) is about an NPU job's worth of work for them.
//!
//! The math and the result are the scan device's
//! (`shaders/attn_far_qk_npu.glsl` + `attn_far_npu.glsl`): unscaled scores
//! `q . k` (the pre graph scales the queries), keys `[nk, cap)` masked, and
//! per query row the unnormalized `o = sum exp(score - m) v`, the maximum
//! `m` and the sum `l`, which the engine merges with `merge_rows_f32`.
//!
//! Work unit: one KV group x a tile of positions (its 6 heads' rows), over
//! all of the call's blocks with a running (m, l, o) per row, key tiles of
//! `KT` converted to f32 once per unit (the keys transposed, so the scores
//! need no horizontal sums).

/// Head dimension, query heads per KV group, KV groups.
const HD: usize = 256;
const HPG: usize = 6;
const NG: usize = 4;
/// Keys per tile.
const KT: usize = 32;

/// Whether the cores can run it (AVX2, FMA, F16C).
pub fn supported() -> bool {
	#[cfg(target_arch = "x86_64")]
	{
		is_x86_feature_detected!("avx2")
			&& is_x86_feature_detected!("fma")
			&& is_x86_feature_detected!("f16c")
	}
	#[cfg(not(target_arch = "x86_64"))]
	{
		false
	}
}

/// `far_blocks` on its own thread while the caller drives the NPU: the
/// engine's share of a far attention call for the cores. Dropping it waits
/// for the thread (the blocks it reads stay in place until then).
pub struct Job {
	h: Option<std::thread::JoinHandle<(Vec<f32>, Vec<[f32; 2]>, f64)>>,
}

impl Job {
	/// Starts on `blocks` (the engine's host buffers, which outlive it).
	pub fn start(
		qg: &'static [u8],
		s: usize,
		nk: usize,
		blocks: Vec<(&'static [u8], &'static [u8])>,
	) -> Job {
		let h = std::thread::spawn(move || {
			let t = std::time::Instant::now();
			let rows = NG * HPG * s;
			let (mut o, mut ml) = (quant::uninit_vec(rows * HD), vec![[0f32; 2]; rows]);
			far_blocks(qg, s, nk, &blocks, &mut o, &mut ml);
			(o, ml, t.elapsed().as_secs_f64() * 1e3)
		});
		Job { h: Some(h) }
	}

	/// `o`, `ml` (as `far_blocks`) and the thread's time in ms.
	pub fn finish(mut self) -> Result<(Vec<f32>, Vec<[f32; 2]>, f64), String> {
		self.h
			.take()
			.unwrap()
			.join()
			.map_err(|_| "far attention on the cores panicked".to_string())
	}
}

impl Drop for Job {
	fn drop(&mut self) {
		if let Some(h) = self.h.take() {
			let _ = h.join();
		}
	}
}

struct Out(*mut f32, *mut [f32; 2]);
// SAFETY: each unit writes its own rows only (`far_blocks`).
unsafe impl Sync for Out {}

/// Rows `(g * 6 + h) * s + pos` of `qg` (`[4][6 s][256]` f16) against
/// `blocks` (K, V: `[cap][4][256]` f16, keys `[0, nk)`): `o` `[24 s][256]`
/// f32 (not normalized), `ml` `[24 s]` (max, sum). Panics without
/// [`supported`].
pub fn far_blocks(
	qg: &[u8],
	s: usize,
	nk: usize,
	blocks: &[(&[u8], &[u8])],
	o: &mut [f32],
	ml: &mut [[f32; 2]],
) {
	assert!(
		supported(),
		"far attention on the cores needs AVX2, FMA and F16C"
	);
	assert!(qg.len() >= NG * HPG * s * HD * 2);
	assert!(o.len() >= NG * HPG * s * HD && ml.len() >= NG * HPG * s);
	for (k, v) in blocks {
		assert!(k.len() >= nk * NG * HD * 2 && v.len() >= nk * NG * HD * 2);
	}
	// position tiles: enough units for every core at 64 positions too
	let pt = if s >= 128 { 16 } else { 4 };
	let tiles = s.div_ceil(pt);
	let out = Out(o.as_mut_ptr(), ml.as_mut_ptr());
	quant::parallel::par_for(NG * tiles, |u| {
		let (g, p0) = (u / tiles, (u % tiles) * pt);
		let pn = pt.min(s - p0);
		let out = &out;
		// SAFETY: features checked above; the unit's rows are its own
		unsafe { unit(qg, s, nk, blocks, g, p0, pn, out.0, out.1) };
	});
}

/// One KV group `g`, positions `p0..p0 + pn`: local row `h * pn + pp`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments)]
unsafe fn unit(
	qg: &[u8],
	s: usize,
	nk: usize,
	blocks: &[(&[u8], &[u8])],
	g: usize,
	p0: usize,
	pn: usize,
	o: *mut f32,
	ml: *mut [f32; 2],
) {
	use std::arch::x86_64::*;
	let rows = HPG * pn;
	let grow = |r: usize| (g * HPG + r / pn) * s + p0 + r % pn;
	let mut qb = Al::new(rows * HD);
	let q = qb.as_mut();
	for r in 0..rows {
		cvt_row(&qg[grow(r) * HD * 2..], &mut q[r * HD..(r + 1) * HD]);
	}
	let mut accb = Al::new(rows * HD);
	let acc = accb.as_mut();
	let mut m = vec![-1e30f32; rows];
	let mut l = vec![0f32; rows];
	// keys transposed: kt[d][j]; values as stored: vt[j][d]
	let (mut ktb, mut vtb, mut scb) = (Al::new(HD * KT), Al::new(KT * HD), Al::new(rows * KT));
	let (kt, vt, sc) = (ktb.as_mut(), vtb.as_mut(), scb.as_mut());
	for &(kb, vb) in blocks {
		let mut j0 = 0;
		while j0 < nk {
			let n = KT.min(nk - j0);
			for j in 0..KT {
				if j < n {
					let src = ((j0 + j) * NG + g) * HD * 2;
					cvt_row(&vb[src..], &mut vt[j * HD..(j + 1) * HD]);
				} else {
					vt[j * HD..(j + 1) * HD].fill(0.0);
				}
			}
			// keys transposed 8 x 8 at a time (absent keys as zeros)
			for j8 in (0..KT).step_by(8) {
				let src: [Option<*const u8>; 8] = std::array::from_fn(|i| {
					(j8 + i < n).then(|| kb.as_ptr().add(((j0 + j8 + i) * NG + g) * HD * 2))
				});
				for d8 in (0..HD).step_by(8) {
					let r: [__m256; 8] = std::array::from_fn(|i| match src[i] {
						Some(p) => {
							_mm256_cvtph_ps(_mm_loadu_si128(p.add(d8 * 2) as *const __m128i))
						}
						None => _mm256_setzero_ps(),
					});
					let t = transpose8(r);
					for (i, v) in t.iter().enumerate() {
						_mm256_storeu_ps(kt.as_mut_ptr().add((d8 + i) * KT + j8), *v);
					}
				}
			}
			// scores, 3 rows x 32 keys at a time
			let (qp, kp, sp) = (q.as_ptr(), kt.as_ptr(), sc.as_mut_ptr());
			for r in (0..rows).step_by(3) {
				let mut a = [_mm256_setzero_ps(); 12];
				// 12 accumulators + 3 broadcasts + 1 key vector: the 16
				// registers exactly (holding all 4 key vectors spilled)
				for d in 0..HD {
					let qv = [
						_mm256_set1_ps(*qp.add(r * HD + d)),
						_mm256_set1_ps(*qp.add((r + 1) * HD + d)),
						_mm256_set1_ps(*qp.add((r + 2) * HD + d)),
					];
					for c in 0..4 {
						let kv = _mm256_loadu_ps(kp.add(d * KT + c * 8));
						for i in 0..3 {
							a[i * 4 + c] = _mm256_fmadd_ps(qv[i], kv, a[i * 4 + c]);
						}
					}
				}
				for i in 0..3 {
					for c in 0..4 {
						_mm256_storeu_ps(sp.add((r + i) * KT + c * 8), a[i * 4 + c]);
					}
				}
			}
			// online softmax per row: sc -> probabilities
			for r in 0..rows {
				let row = &mut sc[r * KT..(r + 1) * KT];
				for x in row[n..].iter_mut() {
					*x = -1e30;
				}
				let rp = row.as_ptr();
				let mx = _mm256_max_ps(
					_mm256_max_ps(_mm256_loadu_ps(rp), _mm256_loadu_ps(rp.add(8))),
					_mm256_max_ps(_mm256_loadu_ps(rp.add(16)), _mm256_loadu_ps(rp.add(24))),
				);
				let mx = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
				let mx = _mm_max_ps(mx, _mm_movehl_ps(mx, mx));
				let tmax = _mm_cvtss_f32(_mm_max_ss(mx, _mm_movehdup_ps(mx)));
				if tmax > m[r] {
					let f = (m[r] - tmax).exp();
					if l[r] != 0.0 {
						for x in acc[r * HD..(r + 1) * HD].iter_mut() {
							*x *= f;
						}
					}
					l[r] *= f;
					m[r] = tmax;
				}
				let mv = _mm256_set1_ps(m[r]);
				let mut sum = _mm256_setzero_ps();
				let p = row.as_mut_ptr();
				for c in 0..KT / 8 {
					let e = quant::attn::v_expf(_mm256_sub_ps(_mm256_loadu_ps(p.add(c * 8)), mv));
					_mm256_storeu_ps(p.add(c * 8), e);
					sum = _mm256_add_ps(sum, e);
				}
				l[r] += hsum(sum);
			}
			// acc += P V, 3 rows x 32 dims at a time
			let (pp, vp, ap) = (sc.as_ptr(), vt.as_ptr(), acc.as_mut_ptr());
			for r in (0..rows).step_by(3) {
				for d0 in (0..HD).step_by(32) {
					let mut a = [_mm256_setzero_ps(); 12];
					for j in 0..n {
						let pv = [
							_mm256_set1_ps(*pp.add(r * KT + j)),
							_mm256_set1_ps(*pp.add((r + 1) * KT + j)),
							_mm256_set1_ps(*pp.add((r + 2) * KT + j)),
						];
						for c in 0..4 {
							let vv = _mm256_loadu_ps(vp.add(j * HD + d0 + c * 8));
							for i in 0..3 {
								a[i * 4 + c] = _mm256_fmadd_ps(pv[i], vv, a[i * 4 + c]);
							}
						}
					}
					for i in 0..3 {
						for c in 0..4 {
							let x = ap.add((r + i) * HD + d0 + c * 8);
							_mm256_storeu_ps(x, _mm256_add_ps(_mm256_loadu_ps(x), a[i * 4 + c]));
						}
					}
				}
			}
			j0 += n;
		}
	}
	for r in 0..rows {
		let gr = grow(r);
		std::ptr::copy_nonoverlapping(acc.as_ptr().add(r * HD), o.add(gr * HD), HD);
		*ml.add(gr) = [m[r], l[r]];
	}
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
unsafe fn unit(
	_: &[u8],
	_: usize,
	_: usize,
	_: &[(&[u8], &[u8])],
	_: usize,
	_: usize,
	_: usize,
	_: *mut f32,
	_: *mut [f32; 2],
) {
	unreachable!()
}

/// A zeroed f32 buffer aligned to 64 bytes (a cache line: the kernels'
/// 32-byte loads never split one).
struct Al(Vec<Line>, usize);

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Line([f32; 16]);

impl Al {
	fn new(n: usize) -> Al {
		Al(vec![Line([0.0; 16]); n.div_ceil(16)], n)
	}
	fn as_mut(&mut self) -> &mut [f32] {
		// SAFETY: a `Line` is 16 f32 with no padding
		unsafe { std::slice::from_raw_parts_mut(self.0.as_mut_ptr() as *mut f32, self.1) }
	}
}

/// Rows of an 8 x 8 f32 tile -> its columns.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn transpose8(r: [std::arch::x86_64::__m256; 8]) -> [std::arch::x86_64::__m256; 8] {
	use std::arch::x86_64::*;
	let t0 = _mm256_unpacklo_ps(r[0], r[1]);
	let t1 = _mm256_unpackhi_ps(r[0], r[1]);
	let t2 = _mm256_unpacklo_ps(r[2], r[3]);
	let t3 = _mm256_unpackhi_ps(r[2], r[3]);
	let t4 = _mm256_unpacklo_ps(r[4], r[5]);
	let t5 = _mm256_unpackhi_ps(r[4], r[5]);
	let t6 = _mm256_unpacklo_ps(r[6], r[7]);
	let t7 = _mm256_unpackhi_ps(r[6], r[7]);
	let u0 = _mm256_shuffle_ps::<0x44>(t0, t2);
	let u1 = _mm256_shuffle_ps::<0xee>(t0, t2);
	let u2 = _mm256_shuffle_ps::<0x44>(t1, t3);
	let u3 = _mm256_shuffle_ps::<0xee>(t1, t3);
	let u4 = _mm256_shuffle_ps::<0x44>(t4, t6);
	let u5 = _mm256_shuffle_ps::<0xee>(t4, t6);
	let u6 = _mm256_shuffle_ps::<0x44>(t5, t7);
	let u7 = _mm256_shuffle_ps::<0xee>(t5, t7);
	[
		_mm256_permute2f128_ps::<0x20>(u0, u4),
		_mm256_permute2f128_ps::<0x20>(u1, u5),
		_mm256_permute2f128_ps::<0x20>(u2, u6),
		_mm256_permute2f128_ps::<0x20>(u3, u7),
		_mm256_permute2f128_ps::<0x31>(u0, u4),
		_mm256_permute2f128_ps::<0x31>(u1, u5),
		_mm256_permute2f128_ps::<0x31>(u2, u6),
		_mm256_permute2f128_ps::<0x31>(u3, u7),
	]
}

/// 256 f16 (little endian bytes) -> f32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn cvt_row(src: &[u8], dst: &mut [f32]) {
	use std::arch::x86_64::*;
	debug_assert!(src.len() >= HD * 2 && dst.len() >= HD);
	for i in (0..HD).step_by(8) {
		let h = _mm_loadu_si128(src.as_ptr().add(i * 2) as *const __m128i);
		_mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_cvtph_ps(h));
	}
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum(v: std::arch::x86_64::__m256) -> f32 {
	use std::arch::x86_64::*;
	let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
	let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
	_mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)))
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::ir::{f16_bits, f16_to_f32};

	fn f16s(v: &[f32]) -> Vec<u8> {
		v.iter().flat_map(|&x| f16_bits(x).to_le_bytes()).collect()
	}

	fn rnd(n: usize, seed: u64, scale: f32) -> Vec<f32> {
		crate::noise(n * 4, seed)
			.chunks_exact(4)
			.map(|c| {
				(u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 / u32::MAX as f32 - 0.5)
					* scale
			})
			.collect()
	}

	/// The scan device's math in f64 on the f16-rounded inputs.
	fn reference(
		q: &[u8],
		s: usize,
		nk: usize,
		blocks: &[(Vec<u8>, Vec<u8>)],
	) -> (Vec<f64>, Vec<[f64; 2]>) {
		let rd =
			|b: &[u8], i: usize| f16_to_f32(u16::from_le_bytes([b[i * 2], b[i * 2 + 1]])) as f64;
		let rows = NG * HPG * s;
		let (mut o, mut ml) = (vec![0f64; rows * HD], vec![[0f64; 2]; rows]);
		for r in 0..rows {
			let g = r / (HPG * s);
			let mut sc = Vec::new();
			for (k, _) in blocks {
				for j in 0..nk {
					sc.push(
						(0..HD)
							.map(|d| rd(q, r * HD + d) * rd(k, (j * NG + g) * HD + d))
							.sum::<f64>(),
					);
				}
			}
			let m = sc.iter().cloned().fold(f64::MIN, f64::max);
			let mut l = 0.0;
			for (bi, (_, v)) in blocks.iter().enumerate() {
				for j in 0..nk {
					let p = (sc[bi * nk + j] - m).exp();
					l += p;
					for d in 0..HD {
						o[r * HD + d] += p * rd(v, (j * NG + g) * HD + d);
					}
				}
			}
			ml[r] = [m, l];
		}
		(o, ml)
	}

	fn check(s: usize, cap: usize, nk: usize, nb: usize) {
		if !supported() {
			return;
		}
		// queries scaled like the pre graph's (q / sqrt(256) folded in)
		let q = f16s(&rnd(NG * HPG * s * HD, 1, 0.5));
		let blocks: Vec<(Vec<u8>, Vec<u8>)> = (0..nb)
			.map(|b| {
				(
					f16s(&rnd(cap * NG * HD, 10 + b as u64, 4.0)),
					f16s(&rnd(cap * NG * HD, 20 + b as u64, 2.0)),
				)
			})
			.collect();
		let refs: Vec<(&[u8], &[u8])> = blocks.iter().map(|(k, v)| (&k[..], &v[..])).collect();
		let rows = NG * HPG * s;
		let (mut o, mut ml) = (vec![0f32; rows * HD], vec![[0f32; 2]; rows]);
		far_blocks(&q, s, nk, &refs, &mut o, &mut ml);
		let (ro, rml) = reference(&q, s, nk, &blocks);
		let mut worst = 0f64;
		for r in 0..rows {
			// compare the normalized outputs and the log sum exp
			let (lse, rlse) = (
				ml[r][0] as f64 + (ml[r][1] as f64).ln(),
				rml[r][0] + rml[r][1].ln(),
			);
			assert!((lse - rlse).abs() < 1e-3, "row {r}: lse {lse} vs {rlse}");
			for d in 0..HD {
				let (a, b) = (
					o[r * HD + d] as f64 / ml[r][1] as f64,
					ro[r * HD + d] / rml[r][1],
				);
				worst = worst.max((a - b).abs());
			}
		}
		assert!(worst < 1e-4, "max |o - ref| {worst}");
	}

	#[test]
	fn matches_the_scan_devices_math() {
		// a partial key tile at the end of each block (1022 = 31 * 32 + 30)
		check(16, 1024, 1022, 2);
	}

	#[test]
	fn short_chunks_and_odd_tiles() {
		// 20 positions: a tile of 4 positions and a remainder; 70 keys
		check(20, 72, 70, 3);
	}
}

#[cfg(test)]
mod bench {
	/// `cargo test --release -p npu farcpu::bench -- --ignored --nocapture`:
	/// ms per far block (256 positions, 1022 keys) on this host's cores.
	#[test]
	#[ignore]
	fn ms_per_block() {
		if !super::supported() {
			return;
		}
		let (s, cap, nb) = (256, 1024, 4);
		let q = vec![0x3c00u16.to_le_bytes(); 24 * s * 256].concat();
		let kv = vec![0x2c00u16.to_le_bytes(); cap * 4 * 256].concat();
		let blocks: Vec<(&[u8], &[u8])> = (0..nb).map(|_| (&kv[..], &kv[..])).collect();
		let (mut o, mut ml) = (vec![0f32; 24 * s * 256], vec![[0f32; 2]; 24 * s]);
		super::far_blocks(&q, s, cap - 2, &blocks, &mut o, &mut ml);
		let t = std::time::Instant::now();
		for _ in 0..3 {
			super::far_blocks(&q, s, cap - 2, &blocks, &mut o, &mut ml);
		}
		let ms = t.elapsed().as_secs_f64() * 1e3 / (3 * nb) as f64;
		let gf = 24.0 * s as f64 * (cap - 2) as f64 * 256.0 * 4.0 / ms / 1e6;
		eprintln!(
			"far block on {} threads: {ms:.1} ms ({gf:.0} GFLOP/s)",
			quant::parallel::threads()
		);
	}
}
