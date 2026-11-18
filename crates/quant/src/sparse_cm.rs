//! Column-major Q4_K for **activation-sparse** decode GEMVs.
//!
//! A row-major K-quant GEMV reads every weight whatever `x` holds, because a
//! row's 256 columns sit in one block. Here each input column's 4-bit values
//! for a tile of rows are contiguous, so a column whose activation is zero is
//! never read. The values are the original Q4_K ones, unchanged: the same
//! `d * sc * q - dmin * m` per element, so with a dense `x` this computes the
//! exact `dequant(W) · x` (in f32, without q8_K's activation rounding).
//!
//! **Not wired in: it does not pay on this host.**
//! `examples/sparsebench` on the CPU share of a decode `ffn_down` (3380 x
//! 17408): the row-major q8_K GEMV takes 0.57 ms (58 GB/s); this kernel 0.89
//! ms dense and 0.79 ms at 50 % activation sparsity. Its `touch` probe, which
//! reads exactly the needed bytes with no arithmetic, takes 0.66 ms at 50 %:
//! slower than reading everything sequentially. With 4x32 GB DDR5-4400 at two
//! DIMMs per channel, sub-page skips lose more DRAM efficiency (prefetcher
//! over-fetch, scattered bursts) than they save; it breaks even near 80 %
//! sparsity, which costs +6 % perplexity.
//!
//! Layout: rows are grouped in tiles of [`TILE`]; a *panel* is one tile times
//! one 256-column super-block:
//!
//! ```text
//! panel = d[TILE] f16 | dmin[TILE] f16 | sc[8][TILE] u8 | m[8][TILE] u8 | col[256][TILE/2]
//! ```
//!
//! Inside a column's `TILE/2` bytes, each 64-row group is 32 bytes: the low
//! nibbles are rows 0..32 of the group, the high nibbles rows 32..64. Panels
//! are tile-major, so one work unit (a tile and a run of super-blocks) reads
//! one contiguous region.

use super::dequant::get_scale_min_k4;
use super::{par_range, QK_K};

/// Rows per tile: each column of a tile is `TILE/2` contiguous bytes. 2048
/// (1 KB per column) was the best of 256 / 512 / 1024 / 2048 / 3584.
pub const TILE: usize = 2048;
const META: usize = TILE * (2 + 2 + 8 + 8);
const PANEL: usize = META + QK_K * TILE / 2;
/// Super-blocks per work unit.
const SB_PER_UNIT: usize = 1;

/// Rows `[r0, r1)` of a Q4_K matrix in the column-major sparse layout.
pub struct CmQ4K {
	pub rows: usize,
	pub in_dim: usize,
	tiles: usize,
	data: Vec<u8>,
}

impl CmQ4K {
	/// Repack rows `[r0, r1)` of the row-major Q4_K `raw` (`in_dim` columns).
	pub fn from_q4k(raw: &[u8], in_dim: usize, r0: usize, r1: usize) -> CmQ4K {
		assert!(in_dim.is_multiple_of(QK_K) && r1 >= r0);
		let nb = in_dim / QK_K;
		let rb = nb * 144;
		let rows = r1 - r0;
		let tiles = rows.div_ceil(TILE);
		let mut data = vec![0u8; tiles * nb * PANEL];
		let ptr = data.as_mut_ptr() as usize;
		par_range(tiles * nb, PANEL, |s, e| {
			for p in s..e {
				let (t, b) = (p / nb, p % nb);
				// SAFETY: panels are disjoint.
				let panel = unsafe {
					std::slice::from_raw_parts_mut((ptr as *mut u8).add(p * PANEL), PANEL)
				};
				for lr in 0..TILE {
					let r = t * TILE + lr;
					if r >= rows {
						break; // padding rows stay zero: d = dmin = 0
					}
					let blk = &raw[(r0 + r) * rb + b * 144..][..144];
					panel[2 * lr..2 * lr + 2].copy_from_slice(&blk[0..2]);
					panel[2 * TILE + 2 * lr..2 * TILE + 2 * lr + 2].copy_from_slice(&blk[2..4]);
					let scales = &blk[4..16];
					for s in 0..8 {
						let (sc, m) = get_scale_min_k4(s, scales);
						panel[4 * TILE + s * TILE + lr] = sc;
						panel[12 * TILE + s * TILE + lr] = m;
					}
					let qs = &blk[16..];
					let (g, k) = (lr / 64, lr % 64);
					for c in 0..QK_K {
						// Column c of the super-block: 64-wide chunk c/64,
						// low nibbles for the chunk's first 32 columns.
						let byte = qs[(c / 64) * 32 + c % 32];
						let q = if c % 64 < 32 { byte & 15 } else { byte >> 4 };
						let dst = META + c * (TILE / 2) + g * 32 + k % 32;
						panel[dst] |= if k < 32 { q } else { q << 4 };
					}
				}
			}
		});
		CmQ4K {
			rows,
			in_dim,
			tiles,
			data,
		}
	}

	pub fn bytes(&self) -> usize {
		self.data.len()
	}

	/// Bench probe: read exactly the bytes [`Self::gemv`] reads for this `x`
	/// (every panel's metadata, the active columns' nibbles) with no math.
	pub fn touch(&self, x: &[f32]) -> u64 {
		use std::arch::x86_64::*;
		let nb = self.in_dim / QK_K;
		let act: Vec<Vec<u16>> = (0..nb)
			.map(|b| {
				(0..QK_K)
					.filter(|&c| x[b * QK_K + c] != 0.0)
					.map(|c| c as u16)
					.collect()
			})
			.collect();
		let n_units = self.tiles * nb;
		let sink = std::sync::atomic::AtomicU64::new(0);
		par_range(n_units, PANEL / 2, |s, e| {
			let mut acc = unsafe { _mm256_setzero_si256() };
			for u in s..e {
				let b = u % nb;
				let panel = &self.data[u * PANEL..][..PANEL];
				unsafe {
					for o in (0..META).step_by(32) {
						acc = _mm256_xor_si256(
							acc,
							_mm256_loadu_si256(panel.as_ptr().add(o) as *const __m256i),
						);
					}
					for &c in &act[b] {
						let col = panel.as_ptr().add(META + c as usize * (TILE / 2));
						for o in (0..TILE / 2).step_by(32) {
							acc = _mm256_xor_si256(
								acc,
								_mm256_loadu_si256(col.add(o) as *const __m256i),
							);
						}
					}
				}
			}
			let v: [u64; 4] = unsafe { std::mem::transmute(acc) };
			sink.fetch_xor(
				v[0] ^ v[1] ^ v[2] ^ v[3],
				std::sync::atomic::Ordering::Relaxed,
			);
		});
		sink.into_inner()
	}

	/// `y = W · x` for the stored rows, reading only columns with `x != 0`.
	pub fn gemv(&self, x: &[f32], y: &mut [f32]) {
		assert_eq!(x.len(), self.in_dim);
		assert_eq!(y.len(), self.rows);
		let nb = self.in_dim / QK_K;
		// Active columns per 32-column sub-block, and the sub-block sums of x
		// (the `dmin * m` term multiplies the sum of x over the sub-block).
		let mut act: Vec<u16> = Vec::with_capacity(self.in_dim);
		let mut sub_start = vec![0u32; nb * 8 + 1];
		let mut xsum = vec![0.0f32; nb * 8];
		for s in 0..nb * 8 {
			let mut sum = 0.0f32;
			for j in 0..32 {
				let v = x[s * 32 + j];
				if v != 0.0 {
					act.push((s % 8 * 32 + j) as u16);
					sum += v;
				}
			}
			xsum[s] = sum;
			sub_start[s + 1] = act.len() as u32;
		}
		let units_per_tile = nb.div_ceil(SB_PER_UNIT);
		let n_units = self.tiles * units_per_tile;
		let mut part = vec![0.0f32; n_units * TILE];
		let pp = part.as_mut_ptr() as usize;
		let active_frac = act.len() as f64 / self.in_dim as f64;
		let unit_bytes =
			(SB_PER_UNIT as f64 * (META as f64 + active_frac * (QK_K * TILE / 2) as f64)) as usize;
		par_range(n_units, unit_bytes, |s, e| {
			for u in s..e {
				let (t, ub) = (u / units_per_tile, u % units_per_tile);
				// SAFETY: each unit owns its own `TILE` partial sums.
				let out =
					unsafe { std::slice::from_raw_parts_mut((pp as *mut f32).add(u * TILE), TILE) };
				for b in ub * SB_PER_UNIT..((ub + 1) * SB_PER_UNIT).min(nb) {
					let panel = &self.data[(t * nb + b) * PANEL..][..PANEL];
					let sb = b * 8;
					unsafe {
						panel_avx2(
							panel,
							x.as_ptr().add(b * QK_K),
							&act[sub_start[sb] as usize..sub_start[sb + 8] as usize],
							&sub_start[sb..sb + 9],
							&xsum[sb..sb + 8],
							out,
						)
					};
				}
			}
		});
		for t in 0..self.tiles {
			let n = TILE.min(self.rows - t * TILE);
			let yt = &mut y[t * TILE..t * TILE + n];
			yt.copy_from_slice(&part[t * units_per_tile * TILE..][..n]);
			for ub in 1..units_per_tile {
				let p = &part[(t * units_per_tile + ub) * TILE..][..n];
				for (a, b) in yt.iter_mut().zip(p) {
					*a += *b;
				}
			}
		}
	}
}

/// Accumulate one panel into `out[TILE]`. `act` holds the active columns of
/// this super-block (0..256), grouped by sub-block per `sub_start` (absolute
/// offsets; the slice starts at `sub_start[0]`).
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn panel_avx2(
	panel: &[u8],
	x: *const f32,
	act: &[u16],
	sub_start: &[u32],
	xsum: &[f32],
	out: &mut [f32],
) {
	use std::arch::x86_64::*;
	let base = sub_start[0];
	let mut acc = [0.0f32; TILE];
	let dp = panel.as_ptr();
	let mask = _mm256_set1_epi8(15);
	let cols = panel.as_ptr().add(META);
	for s in 0..8 {
		let (a0, a1) = (
			(sub_start[s] - base) as usize,
			(sub_start[s + 1] - base) as usize,
		);
		let xs = xsum[s];
		let sc = &panel[4 * TILE + s * TILE..][..TILE];
		let m = &panel[12 * TILE + s * TILE..][..TILE];
		if a0 == a1 {
			continue; // no active column: q·x = 0 and the sub-block's x sums to 0
		}
		for v in acc.iter_mut() {
			*v = 0.0;
		}
		for &c in &act[a0..a1] {
			let c = c as usize;
			let xv = _mm256_set1_ps(*x.add(c));
			let col = cols.add(c * (TILE / 2));
			for g in 0..TILE / 64 {
				let raw = _mm256_loadu_si256(col.add(g * 32) as *const __m256i);
				let lo = _mm256_and_si256(raw, mask);
				let hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask);
				let ap = acc.as_mut_ptr().add(g * 64);
				for (h, v) in [(0usize, lo), (32, hi)] {
					let l0 = _mm256_castsi256_si128(v);
					let l1 = _mm256_extracti128_si256(v, 1);
					let parts = [l0, _mm_srli_si128(l0, 8), l1, _mm_srli_si128(l1, 8)];
					for (i, p) in parts.into_iter().enumerate() {
						let w = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(p));
						let a = ap.add(h + i * 8);
						_mm256_storeu_ps(a, _mm256_fmadd_ps(xv, w, _mm256_loadu_ps(a)));
					}
				}
			}
		}
		let xsv = _mm256_set1_ps(xs);
		for r in (0..TILE).step_by(8) {
			let scv = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
				sc.as_ptr().add(r) as *const __m128i
			)));
			let mv = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
				m.as_ptr().add(r) as *const __m128i
			)));
			let d = _mm256_cvtph_ps(_mm_loadu_si128(dp.add(2 * r) as *const __m128i));
			let dmin = _mm256_cvtph_ps(_mm_loadu_si128(dp.add(2 * TILE + 2 * r) as *const __m128i));
			let dv = _mm256_mul_ps(d, scv);
			let mn = _mm256_mul_ps(dmin, mv);
			let o = out.as_mut_ptr().add(r);
			let v = _mm256_fmadd_ps(dv, _mm256_loadu_ps(acc.as_ptr().add(r)), _mm256_loadu_ps(o));
			_mm256_storeu_ps(o, _mm256_fnmadd_ps(mn, xsv, v));
		}
	}
}
