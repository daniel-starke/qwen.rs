//! Causal grouped-query attention, tiled ("flash attention" on the CPU).
//!
//! The per-(position, head) loop this replaces streamed the whole K and V
//! cache once per query row: at 10k context that is ~20 MB per group per row,
//! out of L3, and it ran at ~65 GFLOP/s; half of a 10.7k-token prefill was
//! attention. Here one task owns a tile of query positions × all heads of one
//! KV group, walks the keys in `BK`-row blocks, and every block is used by
//! `TQ * heads_per_group` query rows while it is cache-resident. Softmax is
//! the online (running max / running sum) form, so no score row is ever
//! materialized past one block.
//!
//! Short query batches (decode, MTP verify) have too few tiles to fill the
//! pool, so the key range is also split into segments whose partial
//! `(max, sum, acc)` are merged afterwards (split-K / "flash decoding").
//!
//! Layout (the model's): `q` and `out` are `[nq][n_head * hd]`, `k`/`v` are
//! `[T][n_kv * hd]`; query `t` sees keys `0 ..= t0 + t`. Heads
//! `g * hpg .. (g + 1) * hpg` share KV group `g`.
//!
//! Reassociates every sum relative to the two-pass reference, so it is not
//! bit-identical to it (see `attn_causal_matches_reference`).
//!
//! The cache may be stored as `f32` or as IEEE half bits (`u16`, see
//! [`KvElem`]); half rows are widened in registers (F16C) as they are loaded,
//! so a half cache reads half the bytes for the same arithmetic. Decode at
//! long context reads the whole cache once per token per layer, which is
//! where that matters.

use super::parallel::{par_for, threads};

/// Key rows per block. 64 rows x 256 x 4 B = 64 KB of K (and of V).
const BK: usize = 64;
/// Query positions per task (x `hpg` = 6 heads here = 48 rows).
const TQ: usize = 8;
/// Query positions per task with one head per KV group (blocked 6 at a
/// time). A task streams its head's whole K / V once, so the tile is what
/// amortizes it: 4096 x 4132 keys x 32 heads x 128 (the image transformer
/// at 1 MP) 12 / 48 / 96 / 192 / 384 / 768 positions = 295 / 959 / 1339 /
/// 1531 / 1596 / 1608 GFLOP/s (`bench_attn_block_mha`).
const TQ_MHA: usize = 384;
/// Smallest key segment worth a task of its own.
const SEG_MIN: usize = 512;

/// Storage type of K/V cache rows: `f32`, or `u16` holding IEEE half bits.
pub trait KvElem: Copy + Send + Sync + 'static {
	#[doc(hidden)]
	const HALF: bool;
	fn to_f32(self) -> f32;
}

impl KvElem for f32 {
	const HALF: bool = false;
	#[inline]
	fn to_f32(self) -> f32 {
		self
	}
}

impl KvElem for u16 {
	const HALF: bool = true;
	#[inline]
	fn to_f32(self) -> f32 {
		crate::fp16_to_f32(self)
	}
}

/// Largest finite half. Values are clamped to it on the way in: an `inf` in
/// a V row would turn every output that attends to it into NaN.
const HALF_MAX: f32 = 65504.0;

/// Append `src` to a half-precision cache row buffer (round to nearest even,
/// subnormals kept, magnitudes clamped to the largest finite half).
pub fn kv_extend(dst: &mut Vec<u16>, src: &[f32]) {
	let n0 = dst.len();
	dst.reserve(src.len());
	#[cfg(target_arch = "x86_64")]
	if super::features().f16c {
		// SAFETY: capacity reserved above; every element is written below
		// before the length is published.
		unsafe {
			to_half_f16c(src, dst.as_mut_ptr().add(n0));
			dst.set_len(n0 + src.len());
		}
		return;
	}
	dst.extend(
		src.iter()
			.map(|&x| crate::dequant::f32_to_fp16_subnormal(x.clamp(-HALF_MAX, HALF_MAX))),
	);
}

/// `dst[i] = half(src[i])`, rounded to nearest, clamped to the largest
/// half (as [`kv_extend`]), F16C where the CPU has it.
pub fn f32_to_f16_slice(src: &[f32], dst: &mut [u16]) {
	assert_eq!(src.len(), dst.len());
	#[cfg(target_arch = "x86_64")]
	if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c") {
		// SAFETY: features checked; dst holds src.len() halves
		unsafe { to_half_f16c(src, dst.as_mut_ptr()) };
		return;
	}
	for (d, &x) in dst.iter_mut().zip(src) {
		*d = crate::dequant::f32_to_fp16_subnormal(x.clamp(-HALF_MAX, HALF_MAX));
	}
}

/// `dst[i] = src[i]` widened from half, F16C where the CPU has it.
pub fn f16_to_f32_slice(src: &[u16], dst: &mut [f32]) {
	assert_eq!(src.len(), dst.len());
	#[cfg(target_arch = "x86_64")]
	if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c") {
		// SAFETY: features checked; both hold src.len() elements
		unsafe { from_half_f16c(src, dst) };
		return;
	}
	for (d, &h) in dst.iter_mut().zip(src) {
		*d = crate::fp16_to_f32(h);
	}
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn from_half_f16c(src: &[u16], dst: &mut [f32]) {
	use std::arch::x86_64::*;
	let n = src.len();
	let mut i = 0;
	while i + 8 <= n {
		let h = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
		_mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_cvtph_ps(h));
		i += 8;
	}
	while i < n {
		dst[i] = crate::fp16_to_f32(src[i]);
		i += 1;
	}
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn to_half_f16c(src: &[f32], dst: *mut u16) {
	use std::arch::x86_64::*;
	let (lo, hi) = (_mm256_set1_ps(-HALF_MAX), _mm256_set1_ps(HALF_MAX));
	let n = src.len();
	let mut i = 0;
	while i + 8 <= n {
		let x = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(src.as_ptr().add(i)), lo), hi);
		_mm_storeu_si128(
			dst.add(i) as *mut __m128i,
			_mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(x),
		);
		i += 8;
	}
	while i < n {
		*dst.add(i) = crate::dequant::f32_to_fp16_subnormal(src[i].clamp(-HALF_MAX, HALF_MAX));
		i += 1;
	}
}

/// See the module docs.
pub fn attn_causal<T: KvElem>(
	q: &[f32],
	k: &[T],
	v: &[T],
	nq: usize,
	t0: usize,
	n_head: usize,
	n_kv: usize,
	hd: usize,
	scale: f32,
	out: &mut [f32],
) {
	attn_causal_groups(q, k, v, nq, t0, n_head, n_kv, hd, scale, 0..n_kv, out)
}

/// [`attn_causal`] for KV groups `groups` only (and their query heads); the
/// rest of `out` is left untouched. The prefill path runs the other groups on
/// the GPU at the same time (`offload::OffloadManager::attn_prefill`).
#[allow(clippy::too_many_arguments)]
pub fn attn_causal_groups<T: KvElem>(
	q: &[f32],
	k: &[T],
	v: &[T],
	nq: usize,
	t0: usize,
	n_head: usize,
	n_kv: usize,
	hd: usize,
	scale: f32,
	groups: std::ops::Range<usize>,
	out: &mut [f32],
) {
	attn_impl(
		q, k, v, nq, t0, n_head, n_kv, hd, scale, groups, 0, None, out,
	)
}

/// Block-causal attention (an image block of a diffusion transformer):
/// query `t` of `nq` (sequence position `t0 + t`) sees the keys
/// `0 .. max(t0 + t + 1, t0 + nq)`, i.e. everything before the block and the
/// whole block itself. With `t0 = 0` and the block being the whole sequence
/// this is plain bidirectional attention. Same layout as [`attn_causal`].
#[allow(clippy::too_many_arguments)]
pub fn attn_block<T: KvElem>(
	q: &[f32],
	k: &[T],
	v: &[T],
	nq: usize,
	t0: usize,
	n_head: usize,
	n_kv: usize,
	hd: usize,
	scale: f32,
	out: &mut [f32],
) {
	attn_impl(
		q,
		k,
		v,
		nq,
		t0,
		n_head,
		n_kv,
		hd,
		scale,
		0..n_kv,
		t0 + nq,
		None,
		out,
	)
}

/// [`attn_block`] for a slice of a block's queries: query `t` of `nq`
/// (sequence position `t0 + t`) sees the keys `0 .. max(t0 + t + 1, full)`,
/// so `full` = the block's end runs `nq` of its rows on their own (the image
/// transformer's attention per 256-row chunk, overlapped with the NPU).
#[allow(clippy::too_many_arguments)]
pub fn attn_block_rows<T: KvElem>(
	q: &[f32],
	k: &[T],
	v: &[T],
	nq: usize,
	t0: usize,
	n_head: usize,
	n_kv: usize,
	hd: usize,
	scale: f32,
	full: usize,
	out: &mut [f32],
) {
	attn_impl(
		q,
		k,
		v,
		nq,
		t0,
		n_head,
		n_kv,
		hd,
		scale,
		0..n_kv,
		full,
		None,
		out,
	)
}

/// [`attn_block_rows`] with K as a [`K8`] (int8 Q.K on AVX-VNNI), multi-head
/// only (`n_head` heads of `hd`, V `[T][n_head * hd]` half). Falls back to
/// the float kernel on the half K `k` when the CPU lacks AVX-VNNI or `k8`
/// does not cover the keys.
#[allow(clippy::too_many_arguments)]
pub fn attn_block_rows_k8(
	q: &[f32],
	k: &[u16],
	k8: &K8,
	v: &[u16],
	nq: usize,
	t0: usize,
	n_head: usize,
	hd: usize,
	scale: f32,
	full: usize,
	out: &mut [f32],
) {
	let tk = (t0 + nq).max(full);
	let use8 = K8::supported(hd) && k8.nh == n_head && k8.hd == hd && k8.nk >= tk;
	attn_impl(
		q,
		k,
		v,
		nq,
		t0,
		n_head,
		n_head,
		hd,
		scale,
		0..n_head,
		full,
		use8.then_some(k8),
		out,
	)
}

/// Keys of one attention, all heads, as int8 for `vpdpbusd` (SageAttention's
/// recipe on the cores): each head's per-channel mean over the keys is
/// subtracted first (it shifts every score of a query by the same amount,
/// which softmax ignores for any subset of the keys, and takes K's large
/// constant channels out of the scale), and queries and keys are rotated per
/// head by the normalized Hadamard matrix (`H H^T = I`: the dots are
/// unchanged, a few large channels are spread over all), then each (key,
/// head) row is int8 at its own scale. Queries are int8 per (row, head),
/// offset to u8 (`+ 128`); the offset's term is removed with
/// `128 * sum(codes)` per key row. Measured on the image transformer's
/// blocks: the attention output's error 3.6-7.4e-3 (6-11e-3 unrotated).
///
/// Layout per head: groups of 8 keys, per group `hd / 4` vectors of 32
/// bytes, each 8 keys x 4 consecutive channels, so one `vpdpbusd` against a
/// broadcast of 4 query bytes gives 8 keys' partial dots in 8 lanes.
///
/// With [`K8::pack_v`] V is int8 too, at a scale per (key, head) that is
/// folded into the probabilities (`P'_j = P_j s_j`): each key block's `P'`
/// is u8 per query row at the row's block maximum, and P V is `vpdpbusd`
/// over 4 keys per lane (layout per head: quads of keys, `[hd][4]` each). A
/// scale per key keeps the attention sinks exact enough: the image
/// transformer's first blocks put ~95 % of the weight on a few text keys
/// whose V is ~14x smaller than the rest (a scale shared with other keys
/// gave a 5-9e-2 output error there, a per-key one 7e-3).
#[derive(Default)]
pub struct K8 {
	nh: usize,
	hd: usize,
	nk: usize,
	ng: usize,
	/// `[head][group][hd / 4][8][4]` codes
	q: Vec<i8>,
	/// `[head][ng * 8]` scale (0 for padding keys)
	s: Vec<f32>,
	/// `[head][ng * 8]` `128 * sum(codes)`
	c: Vec<i32>,
	/// `[head][hd]` the channel means the codes are centered at
	mean: Vec<f32>,
	/// `[head][hd]` the packed keys' raw channel sums ([`K8::key_mean`])
	sum: Vec<f64>,
	has_v: bool,
	/// `[head][ng * 2 quads][hd][4]` V codes
	vq: Vec<i8>,
	/// `[head][ng * 8]` V scale per key (0 for padding keys)
	vs: Vec<f32>,
}

/// In place normalized fast Walsh-Hadamard transform of `x` (length a power
/// of two): `x <- H x` with `H` symmetric and orthogonal.
fn fwht(x: &mut [f32]) {
	let n = x.len();
	debug_assert!(n.is_power_of_two());
	let mut h = 1;
	while h < n {
		for i in (0..n).step_by(2 * h) {
			for j in i..i + h {
				let (a, b) = (x[j], x[j + h]);
				x[j] = a + b;
				x[j + h] = a - b;
			}
		}
		h *= 2;
	}
	let s = 1.0 / (n as f32).sqrt();
	for v in x.iter_mut() {
		*v *= s;
	}
}

impl K8 {
	/// Whether [`attn_block_rows_k8`] takes the int8 path for head width `hd`.
	pub fn supported(hd: usize) -> bool {
		#[cfg(target_arch = "x86_64")]
		{
			let f = super::features();
			return f.avx2
				&& f.f16c && hd.is_multiple_of(32)
				&& hd.is_power_of_two()
				&& is_x86_feature_detected!("avxvnni");
		}
		#[allow(unreachable_code)]
		{
			let _ = hd;
			false
		}
	}

	/// Keys packed.
	pub fn len(&self) -> usize {
		self.nk
	}

	pub fn is_empty(&self) -> bool {
		self.nk == 0
	}

	/// Pack the first `nk` rows of `k` (`[T][nh * hd]` half), reusing the
	/// buffers, at the keys' own channel means.
	pub fn pack(&mut self, k: &[u16], nk: usize, nh: usize, hd: usize) {
		assert!(hd.is_multiple_of(32) && hd.is_power_of_two() && k.len() >= nk * nh * hd);
		self.alloc(nk, nh, hd, false);
		if nk == 0 {
			return;
		}
		let ks = nh * hd;
		// per-head channel sums (f64), one task per head
		let sb = self.sum.as_mut_ptr() as usize;
		par_for(nh, |h| {
			// SAFETY: head h's sums are written by this task only
			let acc = unsafe { std::slice::from_raw_parts_mut((sb as *mut f64).add(h * hd), hd) };
			let mut row = vec![0.0f32; hd];
			for j in 0..nk {
				f16_to_f32_slice(&k[j * ks + h * hd..j * ks + (h + 1) * hd], &mut row);
				for (a, &x) in acc.iter_mut().zip(&row) {
					*a += x as f64;
				}
			}
		});
		for (m, a) in self.mean.iter_mut().zip(&self.sum) {
			*m = (*a / nk as f64) as f32;
		}
		self.pack_range(k, 0, nk, false);
	}

	/// Start packing `nk` keys of `nh` heads of `hd` piece by piece
	/// ([`K8::pack_rows`], [`K8::pack_v_rows`] over disjoint ranges that
	/// cover `0 .. nk`), centered at the given channel means (`[nh][hd]`)
	/// instead of the keys' own: any per-(head, channel) offset shifts all
	/// scores of a query alike, so the means only center the int8 range
	/// (a previous step's keys' do). [`K8::key_mean`] then gives these keys'
	/// own means.
	pub fn begin(&mut self, nk: usize, nh: usize, hd: usize, mean: &[f32], with_v: bool) {
		assert!(hd.is_multiple_of(32) && hd.is_power_of_two() && mean.len() == nh * hd);
		self.alloc(nk, nh, hd, with_v);
		self.mean.copy_from_slice(mean);
	}

	/// Pack keys `j0 .. j1` of `k` (`[T][nh * hd]` half, from key 0) after
	/// [`K8::begin`], adding them to the channel sums.
	pub fn pack_rows(&mut self, k: &[u16], j0: usize, j1: usize) {
		assert!(j0 <= j1 && j1 <= self.nk && k.len() >= j1 * self.nh * self.hd);
		self.pack_range(k, j0, j1, true);
	}

	/// Pack V's keys `j0 .. j1` (`[T][nh * hd]` half, from key 0) after a
	/// [`K8::begin`] `with_v`.
	pub fn pack_v_rows(&mut self, v: &[u16], j0: usize, j1: usize) {
		assert!(self.has_v && j0 <= j1 && j1 <= self.nk);
		self.pack_v_range(v, j0, j1);
	}

	/// The packed keys' own channel means (`[nh][hd]`): those of
	/// [`K8::pack`], or of every range [`K8::pack_rows`] took.
	pub fn key_mean(&self) -> Vec<f32> {
		let nk = self.nk.max(1) as f64;
		self.sum.iter().map(|&a| (a / nk) as f32).collect()
	}

	fn alloc(&mut self, nk: usize, nh: usize, hd: usize, with_v: bool) {
		let ng = nk.div_ceil(8);
		self.has_v = with_v;
		self.nh = nh;
		self.hd = hd;
		self.nk = nk;
		self.ng = ng;
		// every key's codes, scale and sum are written by the packs; only
		// the last group's padding keys must read 0
		let gs = hd * 8;
		self.q.resize(nh * ng * gs, 0);
		self.s.resize(nh * ng * 8, 0.0);
		self.c.resize(nh * ng * 8, 0);
		self.mean.resize(nh * hd, 0.0);
		self.sum.clear();
		self.sum.resize(nh * hd, 0.0);
		let pad = ng * 8 - nk;
		for h in 0..nh {
			if pad > 0 {
				let g = (h * ng + ng - 1) * gs;
				for d in 0..hd / 4 {
					for jj in 8 - pad..8 {
						self.q[g + d * 32 + jj * 4..g + d * 32 + jj * 4 + 4].fill(0);
					}
				}
			}
			self.s[h * ng * 8 + nk..(h + 1) * ng * 8].fill(0.0);
			self.c[h * ng * 8 + nk..(h + 1) * ng * 8].fill(0);
		}
		if with_v {
			self.alloc_v();
		}
	}

	/// V's buffers for the current keys: every key's codes and scale are
	/// written by the packs; only the padding keys are zeroed here.
	fn alloc_v(&mut self) {
		let (nh, hd, nk) = (self.nh, self.hd, self.nk);
		let nq4 = self.ng * 2;
		self.vq.resize(nh * nq4 * hd * 4, 0);
		self.vs.resize(nh * nq4 * 4, 0.0);
		for h in 0..nh {
			let base = h * nq4 * hd * 4;
			for j in nk..nq4 * 4 {
				let q0 = base + (j / 4) * hd * 4 + j % 4;
				for d in 0..hd {
					self.vq[q0 + d * 4] = 0;
				}
			}
			self.vs[h * nq4 * 4 + nk..(h + 1) * nq4 * 4].fill(0.0);
		}
	}

	/// Keys `j0 .. j1` at [`Self::mean`]; with `sums` their raw values are
	/// added to [`Self::sum`] (per task, then in task order: deterministic).
	fn pack_range(&mut self, k: &[u16], j0: usize, j1: usize, sums: bool) {
		if j0 >= j1 {
			return;
		}
		let (nh, hd, ng) = (self.nh, self.hd, self.ng);
		let ks = nh * hd;
		let gs = hd * 8;
		// tasks of up to GT keys of one head
		const GT: usize = 512;
		let nt = (j1 - j0).div_ceil(GT);
		let mut part = if sums {
			vec![0.0f64; nh * nt * hd]
		} else {
			Vec::new()
		};
		let pb = part.as_mut_ptr() as usize;
		let (qb, sb, cb) = (
			self.q.as_mut_ptr() as usize,
			self.s.as_mut_ptr() as usize,
			self.c.as_mut_ptr() as usize,
		);
		let mean = &self.mean;
		par_for(nh * nt, |i| {
			let (h, t) = (i / nt, i % nt);
			let m = &mean[h * hd..(h + 1) * hd];
			let mut row = vec![0.0f32; hd];
			// SAFETY: task i owns partial slot i
			let mut acc = sums.then(|| unsafe {
				std::slice::from_raw_parts_mut((pb as *mut f64).add(i * hd), hd)
			});
			for j in j0 + t * GT..(j0 + (t + 1) * GT).min(j1) {
				let (g, jj) = (j / 8, j % 8);
				f16_to_f32_slice(&k[j * ks + h * hd..j * ks + (h + 1) * hd], &mut row);
				if let Some(a) = acc.as_deref_mut() {
					for (a, &x) in a.iter_mut().zip(&row) {
						*a += x as f64;
					}
				}
				for (x, &mu) in row.iter_mut().zip(m) {
					*x -= mu;
				}
				fwht(&mut row);
				let amax = row.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
				let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
				let mut sum = 0i32;
				// SAFETY: (head h, key j) is written by this task only (its
				// lane of the group: other keys' bytes are not touched)
				unsafe {
					let gq = (qb as *mut i8).add((h * ng + g) * gs + jj * 4);
					for d in 0..hd {
						let c = (row[d] * inv).round_ties_even().clamp(-127.0, 127.0) as i32;
						sum += c;
						*gq.add((d / 4) * 32 + d % 4) = c as i8;
					}
					*(sb as *mut f32).add(h * ng * 8 + j) = amax / 127.0;
					*(cb as *mut i32).add(h * ng * 8 + j) = 128 * sum;
				}
			}
		});
		if sums {
			for h in 0..nh {
				let s = &mut self.sum[h * hd..(h + 1) * hd];
				for t in 0..nt {
					let p = &part[(h * nt + t) * hd..(h * nt + t + 1) * hd];
					for (a, &x) in s.iter_mut().zip(p) {
						*a += x;
					}
				}
			}
		}
	}

	/// Pack V (`[T][nh * hd]` half, the keys [`K8::pack`] took) as int8
	/// too; until the next `pack` the attention takes P V in int8.
	pub fn pack_v(&mut self, v: &[u16]) {
		let (nh, hd, nk) = (self.nh, self.hd, self.nk);
		assert!(v.len() >= nk * nh * hd);
		// (not cleared: a whole-buffer zero fill per call, ~(keys x width)
		// bytes, that the pack below overwrites)
		self.alloc_v();
		self.has_v = true;
		self.pack_v_range(v, 0, nk);
	}

	fn pack_v_range(&mut self, v: &[u16], j0: usize, j1: usize) {
		let (nh, hd) = (self.nh, self.hd);
		assert!(v.len() >= j1 * nh * hd);
		if j0 >= j1 {
			return;
		}
		let nq4 = self.ng * 2;
		let ks = nh * hd;
		// tasks of up to GT keys of one head
		const GT: usize = 512;
		let nt = (j1 - j0).div_ceil(GT);
		let (qb, sb) = (self.vq.as_mut_ptr() as usize, self.vs.as_mut_ptr() as usize);
		par_for(nh * nt, |i| {
			let (h, t) = (i / nt, i % nt);
			let mut row = vec![0.0f32; hd];
			for j in j0 + t * GT..(j0 + (t + 1) * GT).min(j1) {
				f16_to_f32_slice(&v[j * ks + h * hd..j * ks + (h + 1) * hd], &mut row);
				let amax = row.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
				let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
				// SAFETY: (head h, key j) is written by this task only
				unsafe {
					let qj = (qb as *mut i8).add(h * nq4 * hd * 4 + (j / 4) * hd * 4 + j % 4);
					for d in 0..hd {
						*qj.add(d * 4) =
							(row[d] * inv).round_ties_even().clamp(-127.0, 127.0) as i8;
					}
					*(sb as *mut f32).add(h * nq4 * 4 + j) = amax / 127.0;
				}
			}
		});
	}
}

/// Keys `< full` are visible to every query in addition to its causal range.
#[allow(clippy::too_many_arguments)]
fn attn_impl<T: KvElem>(
	q: &[f32],
	k: &[T],
	v: &[T],
	nq: usize,
	t0: usize,
	n_head: usize,
	n_kv: usize,
	hd: usize,
	scale: f32,
	groups: std::ops::Range<usize>,
	full: usize,
	k8: Option<&K8>,
	out: &mut [f32],
) {
	if nq == 0 || groups.is_empty() {
		return;
	}
	assert!(groups.end <= n_kv);
	let (g0, ng) = (groups.start, groups.len());
	assert!(n_head.is_multiple_of(n_kv));
	let hpg = n_head / n_kv;
	let qs = n_head * hd;
	let ks = n_kv * hd;
	let tk = (t0 + nq).max(full);
	assert!(q.len() >= nq * qs && out.len() >= nq * qs);
	assert!(k.len() >= tk * ks && v.len() >= tk * ks);

	// one head per group (plain multi-head attention, the image model):
	// rows are positions, blocked 6 at a time by `Ctx::run`
	let tq = if hpg == 1 { TQ_MHA } else { TQ }.min(nq);
	let ntiles = nq.div_ceil(tq);
	let base = ng * ntiles;
	let nth = threads();
	let mut nseg = 1usize;
	if base < 2 * nth && tk >= 2 * SEG_MIN {
		nseg = (2 * nth).div_ceil(base).min(tk / SEG_MIN).max(1);
	}
	// Segment boundaries over [0, tk), BK-aligned.
	let seg_len = tk.div_ceil(nseg).div_ceil(BK) * BK;
	let ntask = base * nseg;
	let rows = tq * hpg;
	let simd = kernel_ok::<T>(hd);
	debug_assert!(k8.is_none() || (simd && hpg == 1));

	let ctx = Ctx {
		q,
		k,
		v,
		nq,
		t0,
		full,
		hd,
		hpg,
		qs,
		ks,
		scale,
		simd,
		k8,
	};
	let work = (ng * hpg) as u64 * nq as u64 * tk as u64 * hd as u64;
	let serial = ntask == 1 || work < (1 << 21);

	// Task i -> (tile, g, seg); heaviest (latest) tiles first.
	let decode = |i: usize| {
		let tile = ntiles - 1 - i / (ng * nseg);
		let r = i % (ng * nseg);
		(tile, g0 + r / nseg, r % nseg)
	};

	if nseg == 1 {
		let obase = out.as_mut_ptr() as usize;
		let run = |i: usize| {
			let (tile, g, _) = decode(i);
			let tb = tile * tq;
			let tl = tq.min(nq - tb);
			let mut st = State::new(tl * hpg, hd);
			ctx.run(&mut st, g, tb, tl, 0, tk);
			for p in 0..tl {
				for r in 0..hpg {
					let row = p * hpg + r;
					let inv = 1.0 / st.l[row];
					// SAFETY: (position tb + p, head g * hpg + r) is owned by
					// exactly this task.
					let o = unsafe {
						std::slice::from_raw_parts_mut(
							(obase as *mut f32).add((tb + p) * qs + (g * hpg + r) * hd),
							hd,
						)
					};
					let a = &st.acc[row * hd..(row + 1) * hd];
					for d in 0..hd {
						o[d] = a[d] * inv;
					}
				}
			}
		};
		if serial {
			(0..ntask).for_each(run);
		} else {
			par_for(ntask, run);
		}
		return;
	}

	// Split-K: partials per task, then merge per (tile, g).
	let mut pacc = vec![0.0f32; ntask * rows * hd];
	let mut pm = vec![f32::NEG_INFINITY; ntask * rows];
	let mut pl = vec![0.0f32; ntask * rows];
	{
		let (ab, mb, lb) = (
			pacc.as_mut_ptr() as usize,
			pm.as_mut_ptr() as usize,
			pl.as_mut_ptr() as usize,
		);
		let run = |i: usize| {
			let (tile, g, s) = decode(i);
			let tb = tile * tq;
			let tl = tq.min(nq - tb);
			let k0 = s * seg_len;
			let k1 = ((s + 1) * seg_len).min(tk);
			if k0 >= k1 {
				return;
			}
			let mut st = State::new(tl * hpg, hd);
			ctx.run(&mut st, g, tb, tl, k0, k1);
			// SAFETY: task `i` owns partial slot `i`.
			unsafe {
				let n = tl * hpg;
				std::ptr::copy_nonoverlapping(
					st.acc.as_ptr(),
					(ab as *mut f32).add(i * rows * hd),
					n * hd,
				);
				std::ptr::copy_nonoverlapping(st.m.as_ptr(), (mb as *mut f32).add(i * rows), n);
				std::ptr::copy_nonoverlapping(st.l.as_ptr(), (lb as *mut f32).add(i * rows), n);
			}
		};
		if serial {
			(0..ntask).for_each(run);
		} else {
			par_for(ntask, run);
		}
	}
	let obase = out.as_mut_ptr() as usize;
	let (pacc, pm, pl) = (&pacc, &pm, &pl);
	let merge = |i: usize| {
		// i indexes (tile, g) in task order with seg = 0.
		let (tile, g, _) = decode(i * nseg);
		let tb = tile * tq;
		let tl = tq.min(nq - tb);
		let mut o_acc = vec![0.0f32; hd];
		for row in 0..tl * hpg {
			let (p, r) = (row / hpg, row % hpg);
			let mut mx = f32::NEG_INFINITY;
			for s in 0..nseg {
				mx = mx.max(pm[(i * nseg + s) * rows + row]);
			}
			let mut lsum = 0.0f32;
			o_acc.fill(0.0);
			for s in 0..nseg {
				let t = i * nseg + s;
				let l = pl[t * rows + row];
				if l == 0.0 {
					continue;
				}
				let f = (pm[t * rows + row] - mx).exp();
				lsum += l * f;
				let a = &pacc[(t * rows + row) * hd..(t * rows + row + 1) * hd];
				for d in 0..hd {
					o_acc[d] += a[d] * f;
				}
			}
			let inv = 1.0 / lsum;
			// SAFETY: as in the unsplit path, (tb + p, g * hpg + r) is owned
			// by exactly this merge task.
			let o = unsafe {
				std::slice::from_raw_parts_mut(
					(obase as *mut f32).add((tb + p) * qs + (g * hpg + r) * hd),
					hd,
				)
			};
			for d in 0..hd {
				o[d] = o_acc[d] * inv;
			}
		}
	};
	if serial {
		(0..base).for_each(merge);
	} else {
		par_for(base, merge);
	}
}

struct Ctx<'a, T> {
	q: &'a [f32],
	k: &'a [T],
	v: &'a [T],
	nq: usize,
	t0: usize,
	full: usize,
	hd: usize,
	hpg: usize,
	qs: usize,
	ks: usize,
	scale: f32,
	simd: bool,
	k8: Option<&'a K8>,
}

/// One task's running softmax state; row = `p * hpg + r`.
struct State {
	acc: Vec<f32>,
	m: Vec<f32>,
	l: Vec<f32>,
	s: Vec<f32>,
	/// int8 path: the task's query rows as u8 (`code + 128`) and their
	/// scales times the attention scale
	q8: Vec<u8>,
	qsc: Vec<f32>,
	/// int8 V: a key block's probabilities as u8 per row and their scales
	p8: Vec<u8>,
	psc: Vec<f32>,
}

impl State {
	fn new(rows: usize, hd: usize) -> State {
		State {
			acc: vec![0.0; rows * hd],
			m: vec![f32::NEG_INFINITY; rows],
			l: vec![0.0; rows],
			s: vec![0.0; rows * BK],
			q8: Vec::new(),
			qsc: Vec::new(),
			p8: Vec::new(),
			psc: Vec::new(),
		}
	}
}

impl<T: KvElem> Ctx<'_, T> {
	/// Query positions `tb .. tb + tl`, KV group `g`, keys `k0 .. k1`
	/// (each position further clipped to its causal limit).
	fn run(&self, st: &mut State, g: usize, tb: usize, tl: usize, k0: usize, k1: usize) {
		let (hd, hpg, ks) = (self.hd, self.hpg, self.ks);
		debug_assert!(tb + tl <= self.nq);
		if self.k8.is_some() {
			// the task's query rows (head g) as u8 codes
			st.q8.resize(tl * hd, 0);
			st.qsc.resize(tl, 0.0);
			st.p8.resize(tl * BK, 0);
			st.psc.resize(tl, 0.0);
			let mut qr = vec![0.0f32; hd];
			for p in 0..tl {
				qr.copy_from_slice(&self.q[(tb + p) * self.qs + g * hd..][..hd]);
				fwht(&mut qr);
				let amax = qr.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
				let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
				st.qsc[p] = amax / 127.0 * self.scale;
				for (o, &x) in st.q8[p * hd..(p + 1) * hd].iter_mut().zip(&qr) {
					*o = ((x * inv).round_ties_even().clamp(-127.0, 127.0) as i32 + 128) as u8;
				}
			}
		}
		let mut j0 = k0;
		while j0 < k1 {
			let bk = BK.min(k1 - j0);
			let kp = self.k[j0 * ks + g * hd..].as_ptr();
			let vp = self.v[j0 * ks + g * hd..].as_ptr();
			if hpg == 1 {
				// rows are positions: up to 6 with the same visible keys share
				// each K / V load (q rows strided by `qs`)
				let nv_of = |p: usize| {
					let lim = (self.t0 + tb + p + 1).max(self.full);
					if lim <= j0 {
						0
					} else {
						bk.min(lim - j0)
					}
				};
				let mut p = 0;
				while p < tl {
					let nv = nv_of(p);
					if nv == 0 {
						p += 1;
						continue;
					}
					let mut rb = 1;
					while rb < 6 && p + rb < tl && nv_of(p + rb) == nv {
						rb += 1;
					}
					let s = unsafe { st.s.as_mut_ptr().add(p * BK) };
					match self.k8 {
						#[cfg(target_arch = "x86_64")]
						Some(k8) => unsafe {
							qk8(
								rb,
								st.q8.as_ptr().add(p * hd),
								st.qsc.as_ptr().add(p),
								k8,
								g,
								j0,
								nv,
								s,
							)
						},
						_ => {
							let q = self.q[(tb + p) * self.qs + g * hd..].as_ptr();
							self.qk(rb, q, self.qs, kp, nv, s);
						}
					}
					#[cfg(target_arch = "x86_64")]
					let k8v = self.k8.filter(|k8| k8.has_v);
					#[cfg(not(target_arch = "x86_64"))]
					let k8v: Option<&K8> = None;
					if k8v.is_none() {
						for row in p..p + rb {
							softmax_update(
								&mut st.s[row * BK..row * BK + nv],
								&mut st.m[row],
								&mut st.l[row],
								&mut st.acc[row * hd..(row + 1) * hd],
								self.simd,
							);
						}
					}
					let a = unsafe { st.acc.as_mut_ptr().add(p * hd) };
					match k8v {
						#[cfg(target_arch = "x86_64")]
						Some(k8) => unsafe {
							// softmax and P' = P x V's key scales in one pass,
							// rows as u8 at their own max, keys padded to 4
							let n4 = nv.div_ceil(4) * 4;
							let vs = k8.vs.as_ptr().add(g * k8.ng * 8 + j0);
							for row in p..p + rb {
								st.psc[row] = softmax_p8_row(
									st.s.as_ptr().add(row * BK),
									nv,
									&mut st.m[row],
									&mut st.l[row],
									&mut st.acc[row * hd..(row + 1) * hd],
									vs,
									st.p8.as_mut_ptr().add(row * BK),
								);
							}
							pv8(
								rb,
								st.p8.as_ptr().add(p * BK),
								st.psc.as_ptr().add(p),
								k8,
								g,
								j0,
								n4,
								a,
							)
						},
						_ => self.pv(rb, s, vp, nv, a),
					}
					p += rb;
				}
				j0 += bk;
				continue;
			}
			for p in 0..tl {
				let lim = (self.t0 + tb + p + 1).max(self.full); // keys < lim are visible
				if lim <= j0 {
					continue;
				}
				let nv = bk.min(lim - j0);
				let row0 = p * hpg;
				let qp = self.q[(tb + p) * self.qs + g * hpg * hd..].as_ptr();
				let mut r = 0;
				while r < hpg {
					let rb = (hpg - r).min(6);
					let q = unsafe { qp.add(r * hd) };
					let s = unsafe { st.s.as_mut_ptr().add((row0 + r) * BK) };
					self.qk(rb, q, hd, kp, nv, s);
					for rr in 0..rb {
						let row = row0 + r + rr;
						softmax_update(
							&mut st.s[row * BK..row * BK + nv],
							&mut st.m[row],
							&mut st.l[row],
							&mut st.acc[row * hd..(row + 1) * hd],
							self.simd,
						);
					}
					let a = unsafe { st.acc.as_mut_ptr().add((row0 + r) * hd) };
					self.pv(rb, s, vp, nv, a);
					r += rb;
				}
			}
			j0 += bk;
		}
	}

	/// `s[r][j] = q_r . k_j * scale`; query row `r` at `q + r * qrs`.
	fn qk(&self, rb: usize, q: *const f32, qrs: usize, k: *const T, nv: usize, s: *mut f32) {
		#[cfg(target_arch = "x86_64")]
		if self.simd {
			unsafe {
				let (hd, ks, sc) = (self.hd, self.ks, self.scale);
				match rb {
					6 => qk_avx2::<6, T>(q, qrs, k, ks, hd, nv, sc, s),
					5 => qk_avx2::<5, T>(q, qrs, k, ks, hd, nv, sc, s),
					4 => qk_avx2::<4, T>(q, qrs, k, ks, hd, nv, sc, s),
					3 => qk_avx2::<3, T>(q, qrs, k, ks, hd, nv, sc, s),
					2 => qk_avx2::<2, T>(q, qrs, k, ks, hd, nv, sc, s),
					_ => qk_avx2::<1, T>(q, qrs, k, ks, hd, nv, sc, s),
				}
			}
			return;
		}
		for r in 0..rb {
			for j in 0..nv {
				let mut a = 0.0f32;
				for d in 0..self.hd {
					unsafe { a += *q.add(r * qrs + d) * (*k.add(j * self.ks + d)).to_f32() };
				}
				unsafe { *s.add(r * BK + j) = a * self.scale };
			}
		}
	}

	fn pv(&self, rb: usize, p: *const f32, v: *const T, nv: usize, acc: *mut f32) {
		#[cfg(target_arch = "x86_64")]
		if self.simd {
			unsafe {
				let (hd, ks) = (self.hd, self.ks);
				match rb {
					6 => pv_avx2::<6, T>(p, v, ks, hd, nv, acc),
					5 => pv_avx2::<5, T>(p, v, ks, hd, nv, acc),
					4 => pv_avx2::<4, T>(p, v, ks, hd, nv, acc),
					3 => pv_avx2::<3, T>(p, v, ks, hd, nv, acc),
					2 => pv_avx2::<2, T>(p, v, ks, hd, nv, acc),
					_ => pv_avx2::<1, T>(p, v, ks, hd, nv, acc),
				}
			}
			return;
		}
		for r in 0..rb {
			for j in 0..nv {
				let pj = unsafe { *p.add(r * BK + j) };
				for d in 0..self.hd {
					unsafe { *acc.add(r * self.hd + d) += pj * (*v.add(j * self.ks + d)).to_f32() };
				}
			}
		}
	}
}

fn kernel_ok<T: KvElem>(hd: usize) -> bool {
	#[cfg(target_arch = "x86_64")]
	{
		let f = super::features();
		return f.avx2 && (f.f16c || !T::HALF) && hd.is_multiple_of(16); // `avx2` implies `fma` here
	}
	#[allow(unreachable_code)]
	{
		let _ = hd;
		false
	}
}

/// Fold one block of scores `s` (in place -> probabilities) into the row's
/// running max `m`, sum `l` and accumulator `acc`.
fn softmax_update(s: &mut [f32], m: &mut f32, l: &mut f32, acc: &mut [f32], simd: bool) {
	let mb = block_max(s, simd);
	let mn = m.max(mb);
	if mn > *m {
		let f = (*m - mn).exp(); // 0 on the first block (m = -inf)
		if *l != 0.0 {
			for a in acc.iter_mut() {
				*a *= f;
			}
		}
		*l *= f;
		*m = mn;
	}
	*l += exp_sub_sum(s, mn, simd);
}

/// `max(s)` (-inf when empty); the AVX2 form is exact, as the fold is
/// (`f32::max`'s NaN rule keeps the fold from vectorizing).
fn block_max(s: &[f32], simd: bool) -> f32 {
	#[cfg(target_arch = "x86_64")]
	if simd {
		return unsafe { block_max_avx2(s) };
	}
	let _ = simd;
	s.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn block_max_avx2(s: &[f32]) -> f32 {
	use std::arch::x86_64::*;
	let (n, p) = (s.len(), s.as_ptr());
	let mut mv = _mm256_set1_ps(f32::NEG_INFINITY);
	let mut i = 0;
	while i + 8 <= n {
		mv = _mm256_max_ps(mv, _mm256_loadu_ps(p.add(i)));
		i += 8;
	}
	let mut mb = hmax(mv);
	while i < n {
		mb = mb.max(*p.add(i));
		i += 1;
	}
	mb
}

/// `s[i] = exp(s[i] - m)`, returns the sum.
fn exp_sub_sum(s: &mut [f32], m: f32, simd: bool) -> f32 {
	#[cfg(target_arch = "x86_64")]
	if simd {
		return unsafe { exp_sub_sum_avx2(s, m) };
	}
	let _ = simd;
	let mut sum = 0.0f32;
	for x in s.iter_mut() {
		*x = (*x - m).exp();
		sum += *x;
	}
	sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_sub_sum_avx2(s: &mut [f32], m: f32) -> f32 {
	use std::arch::x86_64::*;
	let n = s.len();
	let p = s.as_mut_ptr();
	let mv = _mm256_set1_ps(m);
	let mut acc = _mm256_setzero_ps();
	let mut i = 0;
	while i + 8 <= n {
		let e = v_expf(_mm256_sub_ps(_mm256_loadu_ps(p.add(i)), mv));
		_mm256_storeu_ps(p.add(i), e);
		acc = _mm256_add_ps(acc, e);
		i += 8;
	}
	let mut sum = hsum(acc);
	while i < n {
		let e = (*p.add(i) - m).exp();
		*p.add(i) = e;
		sum += e;
		i += 1;
	}
	sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn hsum(v: std::arch::x86_64::__m256) -> f32 {
	use std::arch::x86_64::*;
	let lo = _mm256_castps256_ps128(v);
	let hi = _mm256_extractf128_ps(v, 1);
	let s = _mm_add_ps(lo, hi);
	let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
	let s = _mm_add_ss(s, _mm_movehdup_ps(s));
	_mm_cvtss_f32(s)
}

/// ggml's `ggml_v_expf` (AVX2): ~1.5 ulp, handles over/underflow.
///
/// # Safety
///
/// The CPU must support AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn v_expf(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
	use std::arch::x86_64::*;
	let r = _mm256_set1_ps(f32::from_bits(0x4b40_0000)); // 0x1.8p23
	let z = _mm256_fmadd_ps(x, _mm256_set1_ps(f32::from_bits(0x3fb8_aa3b)), r); // log2(e)
	let n = _mm256_sub_ps(z, r);
	let b = _mm256_fnmadd_ps(
		n,
		_mm256_set1_ps(f32::from_bits(0x35bf_be8e)), // 0x1.7f7d1cp-20
		_mm256_fnmadd_ps(n, _mm256_set1_ps(f32::from_bits(0x3f31_7200)), x), // 0x1.62e4p-1
	);
	let e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
	let k = _mm256_castsi256_ps(_mm256_add_epi32(
		e,
		_mm256_castps_si256(_mm256_set1_ps(1.0)),
	));
	let absn = _mm256_andnot_ps(_mm256_set1_ps(-0.0), n);
	let c = _mm256_castps_si256(_mm256_cmp_ps(absn, _mm256_set1_ps(126.0), _CMP_GT_OQ));
	let u = _mm256_mul_ps(b, b);
	let j = _mm256_fmadd_ps(
		_mm256_fmadd_ps(
			_mm256_fmadd_ps(
				_mm256_set1_ps(f32::from_bits(0x3c07_2010)),
				b,
				_mm256_set1_ps(f32::from_bits(0x3d2b_9f17)),
			), // 0x1.0e4020p-7, 0x1.573e2ep-5
			u,
			_mm256_fmadd_ps(
				_mm256_set1_ps(f32::from_bits(0x3e2a_af33)),
				b,
				_mm256_set1_ps(f32::from_bits(0x3eff_fedb)),
			), // 0x1.555e66p-3, 0x1.fffdb6p-2
		),
		u,
		_mm256_mul_ps(_mm256_set1_ps(f32::from_bits(0x3f7f_fff6)), b), // 0x1.ffffecp-1
	);
	if _mm256_movemask_ps(_mm256_castsi256_ps(c)) == 0 {
		return _mm256_fmadd_ps(j, k, k);
	}
	let g = _mm256_and_si256(
		_mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
		_mm256_set1_epi32(0x8200_0000u32 as i32),
	);
	let s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f00_0000)));
	let s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
	let d = _mm256_castps_si256(_mm256_cmp_ps(absn, _mm256_set1_ps(192.0), _CMP_GT_OQ));
	_mm256_or_ps(
		_mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
		_mm256_andnot_ps(
			_mm256_castsi256_ps(d),
			_mm256_or_ps(
				_mm256_and_ps(
					_mm256_castsi256_ps(c),
					_mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1),
				),
				_mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)),
			),
		),
	)
}

/// Eight cache elements starting at `p`, widened to f32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn ld8<T: KvElem>(p: *const T) -> std::arch::x86_64::__m256 {
	use std::arch::x86_64::*;
	if T::HALF {
		_mm256_cvtph_ps(_mm_loadu_si128(p as *const __m128i))
	} else {
		_mm256_loadu_ps(p as *const f32)
	}
}

/// `s[r * BK + j] = scale * dot(q_r, k_j)` for `RB` query rows (stride `hd`)
/// against `nv` keys (stride `ks`), two keys at a time: `2 * RB` accumulators.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments)]
unsafe fn qk_avx2<const RB: usize, T: KvElem>(
	q: *const f32,
	qrs: usize,
	k: *const T,
	ks: usize,
	hd: usize,
	nv: usize,
	scale: f32,
	s: *mut f32,
) {
	use std::arch::x86_64::*;
	let mut j = 0;
	while j + 2 <= nv {
		let k0 = k.add(j * ks);
		let k1 = k.add((j + 1) * ks);
		let mut a0 = [_mm256_setzero_ps(); RB];
		let mut a1 = [_mm256_setzero_ps(); RB];
		let mut d = 0;
		while d < hd {
			let kv0 = ld8(k0.add(d));
			let kv1 = ld8(k1.add(d));
			for r in 0..RB {
				let qv = _mm256_loadu_ps(q.add(r * qrs + d));
				a0[r] = _mm256_fmadd_ps(qv, kv0, a0[r]);
				a1[r] = _mm256_fmadd_ps(qv, kv1, a1[r]);
			}
			d += 8;
		}
		for r in 0..RB {
			*s.add(r * BK + j) = hsum(a0[r]) * scale;
			*s.add(r * BK + j + 1) = hsum(a1[r]) * scale;
		}
		j += 2;
	}
	if j < nv {
		let k0 = k.add(j * ks);
		let mut a0 = [_mm256_setzero_ps(); RB];
		let mut d = 0;
		while d < hd {
			let kv0 = ld8(k0.add(d));
			for r in 0..RB {
				a0[r] = _mm256_fmadd_ps(_mm256_loadu_ps(q.add(r * qrs + d)), kv0, a0[r]);
			}
			d += 8;
		}
		for r in 0..RB {
			*s.add(r * BK + j) = hsum(a0[r]) * scale;
		}
	}
}

/// `acc_r += sum_j p[r * BK + j] * v_j` for `RB` rows, 16 lanes of `d` at a
/// time: `2 * RB` accumulators, V loaded once per (j, d-chunk) for all rows.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn pv_avx2<const RB: usize, T: KvElem>(
	p: *const f32,
	v: *const T,
	ks: usize,
	hd: usize,
	nv: usize,
	acc: *mut f32,
) {
	use std::arch::x86_64::*;
	let mut d = 0;
	while d < hd {
		let mut a0 = [_mm256_setzero_ps(); RB];
		let mut a1 = [_mm256_setzero_ps(); RB];
		for r in 0..RB {
			a0[r] = _mm256_loadu_ps(acc.add(r * hd + d));
			a1[r] = _mm256_loadu_ps(acc.add(r * hd + d + 8));
		}
		for j in 0..nv {
			let vj = v.add(j * ks + d);
			let v0 = ld8(vj);
			let v1 = ld8(vj.add(8));
			for r in 0..RB {
				let pb = _mm256_broadcast_ss(&*p.add(r * BK + j));
				a0[r] = _mm256_fmadd_ps(pb, v0, a0[r]);
				a1[r] = _mm256_fmadd_ps(pb, v1, a1[r]);
			}
		}
		for r in 0..RB {
			_mm256_storeu_ps(acc.add(r * hd + d), a0[r]);
			_mm256_storeu_ps(acc.add(r * hd + d + 8), a1[r]);
		}
		d += 16;
	}
}

/// `s[r * BK + j] = qsc[r] * k8.s[j] * (dot(q8_r, k_j) - k8.c[j])` for `rb`
/// query rows (u8, stride `hd`) against keys `j0 .. j0 + nv` of head `h`
/// (`j0` a multiple of 8; written up to the next multiple of 8, at most BK).
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
unsafe fn qk8(
	rb: usize,
	q8: *const u8,
	qsc: *const f32,
	k8: &K8,
	h: usize,
	j0: usize,
	nv: usize,
	s: *mut f32,
) {
	debug_assert!(j0.is_multiple_of(8));
	let hd = k8.hd;
	let ngr = nv.div_ceil(8);
	let kq = k8.q.as_ptr().add((h * k8.ng + j0 / 8) * hd * 8);
	let ks = k8.s.as_ptr().add(h * k8.ng * 8 + j0);
	let kc = k8.c.as_ptr().add(h * k8.ng * 8 + j0);
	match rb {
		6 => qk8_vnni::<6>(q8, qsc, kq, ks, kc, hd, ngr, s),
		5 => qk8_vnni::<5>(q8, qsc, kq, ks, kc, hd, ngr, s),
		4 => qk8_vnni::<4>(q8, qsc, kq, ks, kc, hd, ngr, s),
		3 => qk8_vnni::<3>(q8, qsc, kq, ks, kc, hd, ngr, s),
		2 => qk8_vnni::<2>(q8, qsc, kq, ks, kc, hd, ngr, s),
		_ => qk8_vnni::<1>(q8, qsc, kq, ks, kc, hd, ngr, s),
	}
}

/// [`qk8`]'s kernel: two groups of 8 keys at a time, `2 * RB` i32
/// accumulators, one `vpdpbusd` per (row, group, 4 channels).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni")]
#[allow(clippy::too_many_arguments)]
unsafe fn qk8_vnni<const RB: usize>(
	q8: *const u8,
	qsc: *const f32,
	kq: *const i8,
	ks: *const f32,
	kc: *const i32,
	hd: usize,
	ngr: usize,
	s: *mut f32,
) {
	use std::arch::x86_64::*;
	let gs = hd * 8;
	let n4 = hd / 4;
	let mut g = 0;
	while g < ngr {
		let two = g + 2 <= ngr;
		let k0 = kq.add(g * gs);
		let k1 = if two { k0.add(gs) } else { k0 };
		let mut a0 = [_mm256_setzero_si256(); RB];
		let mut a1 = [_mm256_setzero_si256(); RB];
		for d4 in 0..n4 {
			let kv0 = _mm256_loadu_si256(k0.add(d4 * 32) as *const __m256i);
			let kv1 = _mm256_loadu_si256(k1.add(d4 * 32) as *const __m256i);
			for r in 0..RB {
				let qb =
					_mm256_set1_epi32((q8.add(r * hd + d4 * 4) as *const i32).read_unaligned());
				a0[r] = _mm256_dpbusd_avx_epi32(a0[r], qb, kv0);
				a1[r] = _mm256_dpbusd_avx_epi32(a1[r], qb, kv1);
			}
		}
		for (gg, a) in [(g, &a0), (g + 1, &a1)] {
			if gg >= ngr {
				break;
			}
			let c = _mm256_loadu_si256(kc.add(gg * 8) as *const __m256i);
			let ksv = _mm256_loadu_ps(ks.add(gg * 8));
			for r in 0..RB {
				let f = _mm256_cvtepi32_ps(_mm256_sub_epi32(a[r], c));
				let sc = _mm256_mul_ps(ksv, _mm256_set1_ps(*qsc.add(r)));
				_mm256_storeu_ps(s.add(r * BK + gg * 8), _mm256_mul_ps(f, sc));
			}
		}
		g += 2;
	}
}

/// `acc[r][d] += psc[r] * sum_j p8[r][j] * vq[j][d]` for `rb` rows
/// (u8 probabilities, stride BK) over keys `j0 .. j0 + n4` (`n4` a multiple
/// of 4, padding probabilities 0) of head `h`.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
unsafe fn pv8(
	rb: usize,
	p8: *const u8,
	psc: *const f32,
	k8: &K8,
	h: usize,
	j0: usize,
	n4: usize,
	acc: *mut f32,
) {
	debug_assert!(j0.is_multiple_of(4) && n4.is_multiple_of(4));
	let hd = k8.hd;
	let vq = k8.vq.as_ptr().add((h * k8.ng * 2 + j0 / 4) * hd * 4);
	match rb {
		6 => pv8_vnni::<6>(p8, psc, vq, hd, n4 / 4, acc),
		5 => pv8_vnni::<5>(p8, psc, vq, hd, n4 / 4, acc),
		4 => pv8_vnni::<4>(p8, psc, vq, hd, n4 / 4, acc),
		3 => pv8_vnni::<3>(p8, psc, vq, hd, n4 / 4, acc),
		2 => pv8_vnni::<2>(p8, psc, vq, hd, n4 / 4, acc),
		_ => pv8_vnni::<1>(p8, psc, vq, hd, n4 / 4, acc),
	}
}

/// [`pv8`]'s kernel: 16 channels at a time, `2 * RB` i32 accumulators, one
/// `vpdpbusd` per (row, 8 channels, 4 keys).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn pv8_vnni<const RB: usize>(
	p8: *const u8,
	psc: *const f32,
	vq: *const i8,
	hd: usize,
	nquad: usize,
	acc: *mut f32,
) {
	use std::arch::x86_64::*;
	let qs = hd * 4;
	let mut d = 0;
	while d < hd {
		let mut a0 = [_mm256_setzero_si256(); RB];
		let mut a1 = [_mm256_setzero_si256(); RB];
		for jq in 0..nquad {
			let vj = vq.add(jq * qs + d * 4);
			let v0 = _mm256_loadu_si256(vj as *const __m256i);
			let v1 = _mm256_loadu_si256(vj.add(32) as *const __m256i);
			for r in 0..RB {
				let pb =
					_mm256_set1_epi32((p8.add(r * BK + jq * 4) as *const i32).read_unaligned());
				a0[r] = _mm256_dpbusd_avx_epi32(a0[r], pb, v0);
				a1[r] = _mm256_dpbusd_avx_epi32(a1[r], pb, v1);
			}
		}
		for r in 0..RB {
			let ps = _mm256_set1_ps(*psc.add(r));
			let o = acc.add(r * hd + d);
			_mm256_storeu_ps(
				o,
				_mm256_fmadd_ps(_mm256_cvtepi32_ps(a0[r]), ps, _mm256_loadu_ps(o)),
			);
			_mm256_storeu_ps(
				o.add(8),
				_mm256_fmadd_ps(_mm256_cvtepi32_ps(a1[r]), ps, _mm256_loadu_ps(o.add(8))),
			);
		}
		d += 16;
	}
}

/// [`softmax_update`] then [`p8_row`] for one row of scores `s[..nv]`, in
/// one pass and bit for bit the same: the block max vectorized (`f32::max`'s
/// fold does not vectorize), the probabilities kept in registers instead of
/// written back to `s`. Returns the u8 scale.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn softmax_p8_row(
	s: *const f32,
	nv: usize,
	m: &mut f32,
	l: &mut f32,
	acc: &mut [f32],
	vs: *const f32,
	o: *mut u8,
) -> f32 {
	use std::arch::x86_64::*;
	const _: () = assert!(BK == 64);
	debug_assert!(nv > 0 && nv <= BK);
	let full = nv / 8;
	let tail = nv % 8;
	let mb = block_max_avx2(std::slice::from_raw_parts(s, nv));
	let mn = m.max(mb);
	if mn > *m {
		let f = (*m - mn).exp(); // 0 on the first block (m = -inf)
		if *l != 0.0 {
			for a in acc.iter_mut() {
				*a *= f;
			}
		}
		*l *= f;
		*m = mn;
	}
	// exp(s - mn) = 2^(s log2(e) - mn log2(e)): one FMA, then a cubic
	// (the probabilities become u8; 7.5e-5 relative error)
	let l2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
	let off = _mm256_set1_ps(-mn * std::f32::consts::LOG2_E);
	let mut sum8 = _mm256_setzero_ps();
	let mut x = [_mm256_setzero_ps(); 8];
	let mut mx = _mm256_setzero_ps();
	for i in 0..full {
		let e = v_exp2_p3(_mm256_fmadd_ps(_mm256_loadu_ps(s.add(i * 8)), l2e, off));
		sum8 = _mm256_add_ps(sum8, e);
		x[i] = _mm256_mul_ps(e, _mm256_loadu_ps(vs.add(i * 8)));
		mx = _mm256_max_ps(mx, x[i]);
	}
	if tail > 0 {
		// lanes past nv: 0 (scores padded with mn, vs padded to 8)
		let mut t = [mn; 8];
		std::ptr::copy_nonoverlapping(s.add(full * 8), t.as_mut_ptr(), tail);
		let lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
		let msk = _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(tail as i32), lane));
		let e = _mm256_and_ps(
			v_exp2_p3(_mm256_fmadd_ps(_mm256_loadu_ps(t.as_ptr()), l2e, off)),
			msk,
		);
		sum8 = _mm256_add_ps(sum8, e);
		x[full] = _mm256_mul_ps(e, _mm256_loadu_ps(vs.add(full * 8)));
		mx = _mm256_max_ps(mx, x[full]);
	}
	*l += hsum(sum8);
	let pm = hmax(mx);
	let inv = _mm256_set1_ps(if pm > 0.0 { 255.0 / pm } else { 0.0 });
	let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
	for h in 0..2 {
		let c = |i: usize| _mm256_cvtps_epi32(_mm256_mul_ps(x[h * 4 + i], inv));
		let w0 = _mm256_packus_epi32(c(0), c(1));
		let w1 = _mm256_packus_epi32(c(2), c(3));
		let b = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(w0, w1), perm);
		_mm256_storeu_si256(o.add(h * 32) as *mut __m256i, b);
	}
	pm / 255.0
}

/// `2^t` for `t <= 0` (clamped at -126, where the result is a tiny
/// non-negative value): round to the nearest integer `n`, a cubic minimax
/// fit of `2^f` on `f` in [-0.5, 0.5] (7.5e-5 relative error), `n` added to
/// the exponent.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn v_exp2_p3(t: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
	use std::arch::x86_64::*;
	let t = _mm256_max_ps(t, _mm256_set1_ps(-126.0));
	let r = _mm256_set1_ps(f32::from_bits(0x4b40_0000)); // 0x1.8p23
	let z = _mm256_add_ps(t, r);
	let n = _mm256_sub_ps(z, r);
	let f = _mm256_sub_ps(t, n);
	let p = _mm256_fmadd_ps(
		_mm256_fmadd_ps(
			_mm256_fmadd_ps(
				_mm256_set1_ps(0.055_171_598),
				f,
				_mm256_set1_ps(0.242_611_11),
			),
			f,
			_mm256_set1_ps(0.693_261),
		),
		f,
		_mm256_set1_ps(0.999_928_06),
	);
	let e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
	_mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(p), e))
}

/// One row of a key block's probabilities `p[..nv]` times V's key scales
/// `vs`, as u8 at the row's maximum into `o[..BK]` (0 past `nv`); returns
/// the scale (max / 255).
#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn p8_row(p: *const f32, vs: *const f32, nv: usize, o: *mut u8) -> f32 {
	use std::arch::x86_64::*;
	const _: () = assert!(BK == 64);
	let mut x = [_mm256_setzero_ps(); 8];
	let lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
	let mut mx = _mm256_setzero_ps();
	for i in 0..8 {
		if i * 8 >= nv {
			break;
		}
		// lanes past nv: 0 (the buffers hold BK floats; vs is padded to 8)
		let m = _mm256_cmpgt_epi32(_mm256_set1_epi32((nv - i * 8) as i32), lane);
		let v = _mm256_mul_ps(
			_mm256_loadu_ps(p.add(i * 8)),
			_mm256_loadu_ps(vs.add(i * 8)),
		);
		x[i] = _mm256_and_ps(v, _mm256_castsi256_ps(m));
		mx = _mm256_max_ps(mx, x[i]);
	}
	let pm = hmax(mx);
	let inv = _mm256_set1_ps(if pm > 0.0 { 255.0 / pm } else { 0.0 });
	let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
	for h in 0..2 {
		let c = |i: usize| _mm256_cvtps_epi32(_mm256_mul_ps(x[h * 4 + i], inv));
		let w0 = _mm256_packus_epi32(c(0), c(1));
		let w1 = _mm256_packus_epi32(c(2), c(3));
		let b = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(w0, w1), perm);
		_mm256_storeu_si256(o.add(h * 32) as *mut __m256i, b);
	}
	pm / 255.0
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn hmax(v: std::arch::x86_64::__m256) -> f32 {
	use std::arch::x86_64::*;
	let s = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
	let s = _mm_max_ps(s, _mm_movehl_ps(s, s));
	let s = _mm_max_ss(s, _mm_movehdup_ps(s));
	_mm_cvtss_f32(s)
}

#[cfg(all(test, target_arch = "x86_64"))]
mod fused_tests {
	use super::*;

	/// The fused softmax + u8 pass equals `softmax_update` then `p8_row` bit
	/// for bit, over every block width and both a first and a later block.
	#[test]
	fn softmax_p8_row_matches_unfused() {
		// also the cubic exp2 over its whole input range
		if crate::features().avx2 {
			let mut worst = 0f64;
			for i in 0..=200_000 {
				let t = -(i as f32) * 1e-4 * 13.0;
				let mut o = [0f32; 8];
				unsafe {
					use std::arch::x86_64::*;
					_mm256_storeu_ps(o.as_mut_ptr(), v_exp2_p3(_mm256_set1_ps(t)));
				}
				if t < -126.0 {
					assert!((0.0..1e-37).contains(&o[0]), "exp2({t}) = {}", o[0]);
					continue;
				}
				let want = (t as f64).exp2();
				worst = worst.max((o[0] as f64 / want - 1.0).abs());
			}
			assert!(worst < 8e-5, "exp2 rel. error {worst}");
		}
		let f = crate::features();
		if !f.avx2 {
			return;
		}
		let mut seed = 0x9e37_79b9_7f4a_7c15u64;
		let mut rnd = || {
			seed ^= seed << 13;
			seed ^= seed >> 7;
			seed ^= seed << 17;
			(seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
		};
		let hd = 32;
		for nv in 1..=BK {
			for &m0 in &[f32::NEG_INFINITY, 1.5, 40.0] {
				let s: Vec<f32> = (0..BK).map(|_| rnd() * 20.0).collect();
				let vs: Vec<f32> = (0..BK).map(|_| rnd().abs() + 0.01).collect();
				let acc0: Vec<f32> = (0..hd).map(|_| rnd()).collect();
				let l0 = if m0.is_finite() { 3.25 } else { 0.0 };
				let (mut ma, mut la, mut acca, mut sa) = (m0, l0, acc0.clone(), s.clone());
				let mut oa = [0u8; BK];
				let pa = unsafe {
					softmax_update(&mut sa[..nv], &mut ma, &mut la, &mut acca, true);
					p8_row(sa.as_ptr(), vs.as_ptr(), nv, oa.as_mut_ptr())
				};
				let (mut mb, mut lb, mut accb) = (m0, l0, acc0.clone());
				let mut ob = [0u8; BK];
				let pb = unsafe {
					softmax_p8_row(
						s.as_ptr(),
						nv,
						&mut mb,
						&mut lb,
						&mut accb,
						vs.as_ptr(),
						ob.as_mut_ptr(),
					)
				};
				// the fused pass' exp is a cubic (7.5e-5 relative error)
				let close = |a: f32, b: f32| (a - b).abs() <= 2e-4 * a.abs().max(b.abs());
				assert!(close(pa, pb), "nv {nv} m0 {m0}: {pa} {pb}");
				assert_eq!(ma.to_bits(), mb.to_bits());
				assert!(close(la, lb), "nv {nv} m0 {m0}: {la} {lb}");
				assert!(
					oa.iter().zip(&ob).all(|(a, b)| a.abs_diff(*b) <= 1),
					"nv {nv} m0 {m0}: {oa:?} {ob:?}"
				);
				assert!(acca
					.iter()
					.zip(&accb)
					.all(|(a, b)| a.to_bits() == b.to_bits()));
			}
		}
	}
}
