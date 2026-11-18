//! A prefill attention layer's block indexer scores on the NPU.
//!
//! WHY
//! ---
//! Past 512 blocks (2048 positions) every query of an attention layer
//! selects its 512 blocks of 4 keys by the indexer's score
//! `sum_h max(q_h . k_b, 0)` over every complete block. On the cores that
//! is n x blocks x 4 x 128 multiply-adds per sub-chunk, ~8.5 ms at 72k
//! positions, on the critical path (nothing else runs on the iGPU while
//! the host selects): a run that replayed the selection from disk
//! prefilled 82.6k tokens at 196.2 instead of 188.4 tok/s. Here the NPU
//! scores the blocks before the sub-chunk, in jobs of [`T`] blocks
//! (`npubuild`'s `idx_score`) queued right after the sub-chunk's
//! `attn_pre` (whose `iq` output they read in place), so the host finds
//! them done; the sub-chunk's own blocks (~64) stay on the host.
//!
//! The pooled block keys live here a second time as f16 tiles of [`T`]
//! rows per attention layer, appended as the engine pools them.

use crate::ze::{Batch, Graph, HostBuf, Npu};

/// Blocks per job (the graph's).
pub const T: usize = 4096;
const ID: usize = 128;
/// Queries of the graph.
const S: usize = 256;

/// Blocks before a sub-chunk from which the NPU scores them
/// (`QWEN_FLASH_IDX_MIN`, `0` = never; below it the host's scoring is
/// shorter than a job).
pub fn min_blocks() -> usize {
	static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
	*C.get_or_init(|| {
		std::env::var("QWEN_FLASH_IDX_MIN")
			.ok()
			.and_then(|v| v.trim().parse().ok())
			.map_or(1024, |v: usize| if v == 0 { usize::MAX } else { v })
	})
}

/// See the module docs.
pub struct NpuIdx {
	npu: &'static Npu,
	/// per job slot (dropped before the graph)
	jobs: Vec<Batch<'static>>,
	out: Vec<HostBuf<'static>>,
	_graph: Vec<Graph<'static>>,
	_blob: Vec<u8>,
	/// argument positions: iq, kidx
	ix: [usize; 2],
	/// per attention layer: the f16 tiles and the blocks they hold
	tiles: Vec<(Vec<HostBuf<'static>>, usize)>,
	/// in flight: (layer, blocks scored, jobs)
	pending: Option<(usize, usize, usize)>,
}

// SAFETY: used from the engine's thread only (the engine is Send).
unsafe impl Send for NpuIdx {}

impl NpuIdx {
	/// The graph `idx_score.blob` of `dir`, for `layers` attention layers;
	/// `None` when it is not there.
	pub fn load(
		npu: &'static Npu,
		dir: &std::path::Path,
		layers: usize,
	) -> Result<Option<NpuIdx>, String> {
		let f = dir.join("idx_score.blob");
		if !f.exists() {
			return Ok(None);
		}
		let blob = std::fs::read(&f).map_err(|e| format!("{}: {e}", f.display()))?;
		// SAFETY: the blob bytes live as long as the graph (both owned
		// here; the graph is dropped before `_blob`)
		let b: &'static [u8] = unsafe { std::slice::from_raw_parts(blob.as_ptr(), blob.len()) };
		let mut g = npu.load(b, 0)?;
		g.initialize()?;
		let ix = |n: &str| {
			g.args
				.iter()
				.position(|a| a.name == n && !a.is_output)
				.ok_or(format!("idx_score: no input {n}"))
		};
		let ix = [ix("iq")?, ix("kidx")?];
		let outs: Vec<usize> = (0..g.args.len()).filter(|&i| g.args[i].is_output).collect();
		if outs.len() != 1
			|| g.args[outs[0]].bytes != S * T * 4
			|| g.args[ix[1]].dims[1] as usize != T
		{
			return Err("idx_score: unexpected arguments".into());
		}
		Ok(Some(NpuIdx {
			npu,
			jobs: Vec::new(),
			out: Vec::new(),
			_graph: vec![g],
			_blob: blob,
			ix,
			tiles: (0..layers).map(|_| (Vec::new(), 0)).collect(),
			pending: None,
		}))
	}

	/// Forget every block (the cache was reset or replaced).
	pub fn invalidate(&mut self) {
		self.wait_all();
		for t in &mut self.tiles {
			t.1 = 0;
		}
	}

	/// The cache was cut to `nb` blocks.
	pub fn truncate(&mut self, nb: usize) {
		self.wait_all();
		for t in &mut self.tiles {
			t.1 = t.1.min(nb);
		}
	}

	fn wait_all(&mut self) {
		if let Some((_, _, nj)) = self.pending.take() {
			for j in &mut self.jobs[..nj] {
				let _ = j.wait();
			}
		}
	}

	/// The blocks layer `a` holds.
	pub fn blocks(&self, a: usize) -> usize {
		self.tiles[a].1
	}

	/// Append layer `a`'s blocks from [`Self::blocks`] on: `rows` f32
	/// `[..][128]`.
	pub fn append(&mut self, a: usize, rows: &[f32]) -> Result<(), String> {
		// (a job in flight may read the tile: these rows are past the
		// blocks it scores, whose outputs nobody reads)
		let (tiles, nb) = &mut self.tiles[a];
		for r in rows.chunks_exact(ID) {
			let (ti, i) = (*nb / T, *nb % T);
			if tiles.len() <= ti {
				let mut h = self.npu.alloc(T * ID * 2)?;
				h.as_mut_slice().fill(0);
				tiles.push(h);
			}
			let d = &mut tiles[ti].as_mut_slice()[i * ID * 2..][..ID * 2];
			for (d, &v) in d.chunks_exact_mut(2).zip(r) {
				d.copy_from_slice(&quant::f32_to_fp16(v).to_le_bytes());
			}
			*nb += 1;
		}
		Ok(())
	}

	/// Queue the scores of layer `a`'s blocks against the queries in `iq`
	/// (`attn_pre`'s output buffer, written by a list queued before).
	pub fn begin(&mut self, a: usize, iq: &HostBuf<'_>) -> Result<(), String> {
		// (a sub-chunk that failed may have left its jobs)
		self.wait_all();
		let nb = self.tiles[a].1;
		let nj = nb.div_ceil(T);
		while self.jobs.len() < nj {
			let g = &self._graph[0];
			let mut out = self.npu.alloc(S * T * 4)?;
			out.as_mut_slice().fill(0);
			let o = (0..g.args.len()).find(|&i| g.args[i].is_output).unwrap();
			let refs: Vec<&HostBuf<'_>> = (0..g.args.len())
				.map(|i| {
					if i == o {
						&out
					} else if i == self.ix[0] {
						iq
					} else {
						&self.tiles[a].0[0]
					}
				})
				.collect();
			// SAFETY: a mutable list over this graph; the graph outlives
			// the job (field order)
			let gs: &'static mut [Graph<'static>] = unsafe {
				std::slice::from_raw_parts_mut(
					&self._graph[0] as *const _ as *mut Graph<'static>,
					1,
				)
			};
			let b = self.npu.batch_plan_mutable(gs, &[(0, refs)])?;
			self.jobs.push(b);
			self.out.push(out);
		}
		for (j, job) in self.jobs[..nj].iter_mut().enumerate() {
			job.update(&[(0, self.ix[0], iq), (0, self.ix[1], &self.tiles[a].0[j])])?;
			job.submit()?;
		}
		self.pending = Some((a, nb, nj));
		Ok(())
	}

	/// The scores in flight for layer `a`, if any: wait for them; the
	/// result reads blocks `[0, nb)` of each of `n` queries straight from
	/// the jobs' output buffers (`nb` = 0 when nothing was in flight for
	/// `a`).
	pub fn end(&mut self, a: usize, n: usize, nbm: usize) -> Result<Scored<'_>, String> {
		let Some((pa, nb, nj)) = self.pending.take() else {
			return Ok(Scored::none());
		};
		for j in &mut self.jobs[..nj] {
			j.wait()?;
		}
		if pa != a || n > S {
			return Ok(Scored::none());
		}
		Ok(Scored {
			outs: self.out[..nj].iter().map(|o| o.as_slice()).collect(),
			cols: T,
			stride: T,
			nb: nb.min(nbm),
		})
	}
}

/// The block scores finished for a sub-chunk ([`NpuIdx::end`], or the
/// prefill attention's device: [`Scored::rows`]).
pub struct Scored<'a> {
	/// per buffer `cols` blocks of each query, rows `stride` apart
	outs: Vec<&'a [u8]>,
	cols: usize,
	stride: usize,
	/// blocks scored per query
	pub nb: usize,
}

impl<'a> Scored<'a> {
	/// Nothing scored.
	pub fn none() -> Scored<'static> {
		Scored {
			outs: Vec::new(),
			cols: T,
			stride: T,
			nb: 0,
		}
	}

	/// `nb` blocks per query in one buffer of rows `stride` apart.
	pub fn rows(s: &'a [u32], nb: usize, stride: usize) -> Scored<'a> {
		// SAFETY: plain integers viewed as bytes
		let b = unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4) };
		Scored {
			outs: vec![b],
			cols: stride.max(1),
			stride,
			nb,
		}
	}

	/// Query `t`'s scores of blocks `[0, d.len())` (at most [`Self::nb`])
	/// as order keys ([`okey`]).
	pub fn keys(&self, t: usize, d: &mut [u32]) {
		for (j, d) in d.chunks_mut(self.cols).enumerate() {
			let src = &self.outs[j][t * self.stride * 4..][..d.len() * 4];
			for (x, c) in d.iter_mut().zip(src.chunks_exact(4)) {
				*x = okey_bits(u32::from_le_bytes(c.try_into().unwrap()));
			}
		}
	}
}

/// A score's order key: the indexer's scores are sums of ReLUs, so their
/// f32 bits order as integers; anything with the sign bit set (-0) is 0.
pub fn okey(v: f32) -> u32 {
	okey_bits(v.to_bits())
}

fn okey_bits(b: u32) -> u32 {
	if (b as i32) < 0 {
		0
	} else {
		b
	}
}

/// The indices of the `m` largest of `k` (ties: the lower index) into
/// `out`, ascending; `cand` is scratch. One histogram pass over the top
/// 11 bits finds the bucket the `m`-th falls in, a second collects
/// everything above it and the bucket's members, of which only those are
/// selected.
pub fn top_m(k: &[u32], m: usize, cand: &mut Vec<u64>, out: &mut Vec<u32>) {
	out.clear();
	if k.len() <= m {
		out.extend(0..k.len() as u32);
		return;
	}
	let mut h = [0u32; 2048];
	for &x in k {
		h[(x >> 20) as usize] += 1;
	}
	let (mut above, mut bk) = (0usize, 2047usize);
	while above + (h[bk] as usize) < m {
		above += h[bk] as usize;
		bk -= 1;
	}
	cand.clear();
	for (i, &x) in k.iter().enumerate() {
		let b = (x >> 20) as usize;
		if b > bk {
			out.push(i as u32);
		} else if b == bk {
			cand.push(((x as u64) << 32) | (u32::MAX - (i as u32)) as u64);
		}
	}
	let r = m - above;
	if cand.len() > r {
		cand.select_nth_unstable_by(r - 1, |a, b| b.cmp(a));
	}
	out.extend(cand[..r].iter().map(|&c| u32::MAX - c as u32));
	out.sort_unstable();
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn top_m_matches_a_sort() {
		let mut s = 12345u64;
		let mut rnd = || {
			s = s
				.wrapping_mul(6364136223846793005)
				.wrapping_add(1442695040888963407);
			(s >> 33) as u32
		};
		let (mut cand, mut out) = (Vec::new(), Vec::new());
		for &(n, m, spread) in &[
			(100, 512, 1000),
			(5000, 512, 1000),
			(20000, 512, 7),
			(3000, 512, 1 << 20),
		] {
			let k: Vec<u32> = (0..n)
				.map(|_| {
					okey(rnd() as f32 / spread as f32 * if rnd() % 50 == 0 { 0.0 } else { 1.0 })
				})
				.collect();
			top_m(&k, m, &mut cand, &mut out);
			let mut r: Vec<u32> = (0..n as u32).collect();
			r.sort_by(|&a, &b| k[b as usize].cmp(&k[a as usize]).then(a.cmp(&b)));
			r.truncate(m);
			r.sort_unstable();
			assert_eq!(out, r, "n {n}");
		}
	}
}
