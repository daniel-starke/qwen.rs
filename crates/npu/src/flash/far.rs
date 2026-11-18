//! A prefill attention layer's keys before its sub-chunk on the NPU, while
//! the scan device (`offload::FlashAttn`) runs the rest.
//!
//! WHY
//! ---
//! An attention layer's sub-chunk is NPU-bound while its keys are few (the
//! `attn_pre` / `post_attn` graphs, ~16 ms) and bound by the iGPU's
//! attention past ~1500 keys (~35 ms at the indexer's 2048): 362 ms per
//! layer at 3.2k tokens, each device idle part of the time. The NPU runs an
//! attention block of 1020 keys in ~4.3 ms (7 with the selection mask), the
//! iGPU in ~15, so per sub-chunk the NPU takes the whole blocks
//! (`npubuild`'s `attn_far` / `attn_farm`) that balance the two by a cost
//! model ([`cost`]), the iGPU the rest, and the host merges both by their
//! log sum exp (the iGPU's kernel returns its `lse`). 3220 tokens: 363 ->
//! ~280 ms per attention layer, 190.3 -> 203.1 tok/s.
//!
//! A block is `[1024][2][256]` f16: 1020 keys (255 blocks of 4) and four
//! zero rows, the last two the `softmax_lse` kernel's (`-DKL=4`) max / sum
//! slots. A dense sub-chunk (every query sees every earlier key) gives the
//! NPU the first fixed ranges `[b 1020, (b + 1) 1020)` of the layer's
//! positions, in host buffers its jobs read in place (filled from the
//! engine's cache at first use), and the iGPU the keys from `kmin`. With
//! the indexer's selection the NPU's job gathers the 255 blocks of 4 the
//! most queries list (~50 % of the selected keys at 2-3k positions; a
//! fixed range held 32-47 %), masked per query unless every query lists
//! all of them, and the iGPU takes every other listed block.
//! `QWEN_FLASH_FAR_CHECK=1` compares each merge with the iGPU alone.

use super::engine::{u8s, u8sm, Kv};
use crate::ze::{Batch, Graph, HostBuf, Npu};
use quant::parallel::par_for;

#[derive(Clone, Copy)]
struct P<T>(*mut T);
unsafe impl<T> Send for P<T> {}
unsafe impl<T> Sync for P<T> {}

/// Key slots of a block (the graph's), keys of a block.
pub const CAP: usize = 1024;
pub const KEYS: usize = CAP - 4;
const HQ: usize = 24;
const HKV: usize = 2;
const DA: usize = 256;
const G: usize = HQ / HKV;
/// Queries of the graphs.
const S: usize = 256;
/// Most blocks a sub-chunk may give the NPU (job slots).
pub const MAX_SLOTS: usize = 8;

/// Blocks a sub-chunk may give the NPU (`QWEN_FLASH_FAR_SLOTS`, 1 to
/// [`MAX_SLOTS`]).
fn slots() -> usize {
	static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
	*C.get_or_init(|| {
		std::env::var("QWEN_FLASH_FAR_SLOTS")
			.ok()
			.and_then(|v| v.trim().parse().ok())
			.map_or(2, |v: usize| v.clamp(1, MAX_SLOTS))
	})
}

/// The split's cost model, ms (`QWEN_FLASH_FAR_COST` =
/// `base,dense,masked,dkey,skey`): the NPU's own work per sub-chunk
/// (pre(j + 1) + post(j - 1) and the host's staging), a dense / masked
/// block job, the iGPU per key of a dense / selected query (per 256
/// queries). `skey` is measured past 16k positions (`QWEN_FLASH_ATT_LOG`:
/// the iGPU's selected-key attention takes ~1.37x the 0.0169 fitted at
/// 3.2k, scattered blocks), which gives the NPU a second gathered job at
/// depth: 82.6k tokens 175-179 -> 185-186 tok/s.
fn cost() -> [f64; 5] {
	static C: std::sync::OnceLock<[f64; 5]> = std::sync::OnceLock::new();
	*C.get_or_init(|| {
		let mut c = [12.0, 4.6, 7.1, 0.0146, 0.023];
		if let Ok(v) = std::env::var("QWEN_FLASH_FAR_COST") {
			for (d, x) in c.iter_mut().zip(v.split(',')) {
				if let Ok(x) = x.trim().parse() {
					*d = x;
				}
			}
		}
		c
	})
}

/// f16 of -60000: a masked key's score offset (its weight underflows to 0;
/// a row with every key masked gets a log sum exp ~ -6e4, weight 0 in the
/// merge).
const MASKED: u16 = 0xfb53;

/// Blocks of 4 keys a gathered job holds.
const GB: usize = KEYS / 4;

/// The split of one sub-chunk: dense, the NPU's first `blocks` (whole
/// ranges, unmasked) and the iGPU's keys from `kmin`; with the selection,
/// per NPU job the blocks of 4 it gathers (the most selected ones, under a
/// mask unless every query lists them all) and the iGPU every other listed
/// block.
pub struct Plan {
	pub blocks: Vec<usize>,
	pub gather: Vec<(Vec<u32>, bool)>,
	pub kmin: usize,
	/// the cost model's estimate, ms: the NPU's side, the iGPU's
	pub est: [f64; 2],
}

impl Plan {
	pub fn jobs(&self) -> usize {
		self.blocks.len() + self.gather.len()
	}
}

struct Job {
	b: Batch<'static>,
}

/// See the module docs.
pub struct NpuFar {
	npu: &'static Npu,
	/// per slot: the dense and the masked graph's job (dropped before the
	/// graphs)
	jobs: Vec<[Job; 2]>,
	_graphs: Vec<Graph<'static>>,
	_blobs: Vec<Vec<u8>>,
	/// argument positions per graph: qg, kblk, vblk, (mask)
	kix: [[usize; 2]; 2],
	qg: HostBuf<'static>,
	mask: Vec<HostBuf<'static>>,
	o: Vec<HostBuf<'static>>,
	lse: Vec<HostBuf<'static>>,
	/// per job slot: the gathered K / V of a masked job
	gath: Vec<[HostBuf<'static>; 2]>,
	/// per attention layer, per block: K, V and whether it holds the cache's rows
	blocks: Vec<Vec<Option<([HostBuf<'static>; 2], bool)>>>,
	/// the jobs in flight: (slot, block)
	pending: Vec<(usize, usize)>,
	/// the queries of the sub-chunk in flight
	n: usize,
	/// the jobs in flight were waited for ([`Self::wait_jobs`])
	waited: bool,
}

// SAFETY: used from the engine's thread only (the engine is Send).
unsafe impl Send for NpuFar {}

impl NpuFar {
	/// The graphs `attn_far.blob` / `attn_farm.blob` of `dir`, for
	/// `layers` attention layers; `None` when they are not there.
	pub fn load(
		npu: &'static Npu,
		dir: &std::path::Path,
		layers: usize,
	) -> Result<Option<NpuFar>, String> {
		let (fd, fm) = (dir.join("attn_far.blob"), dir.join("attn_farm.blob"));
		if !fd.exists() || !fm.exists() {
			return Ok(None);
		}
		let rd =
			|p: &std::path::Path| std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()));
		let blobs = vec![rd(&fd)?, rd(&fm)?];
		let mut graphs = Vec::new();
		for b in &blobs {
			// SAFETY: the blob bytes live as long as the graph (both owned
			// here; the graphs are dropped before `_blobs`)
			let b: &'static [u8] = unsafe { std::slice::from_raw_parts(b.as_ptr(), b.len()) };
			let mut g = npu.load(b, 0)?;
			g.initialize()?;
			graphs.push(g);
		}
		let ix = |g: &Graph<'_>, n: &str| {
			g.args
				.iter()
				.position(|a| a.name == n && !a.is_output)
				.ok_or(format!("attn_far: no input {n}"))
		};
		let outs = |g: &Graph<'_>| {
			let mut o: Vec<usize> = (0..g.args.len()).filter(|&i| g.args[i].is_output).collect();
			o.sort_by_key(|&i| std::cmp::Reverse(g.args[i].bytes));
			o
		};
		let mut kix = [[0; 2]; 2];
		for (gi, g) in graphs.iter().enumerate() {
			kix[gi] = [ix(g, "kblk")?, ix(g, "vblk")?];
			if outs(g).len() != 2 || g.args[ix(g, "kblk")?].dims[1] as usize != CAP {
				return Err("attn_far: unexpected arguments".into());
			}
		}
		let qbytes = HKV * G * S * DA * 2;
		let qg = npu.alloc(qbytes)?;
		let bb = CAP * HKV * DA * 2;
		// placeholder K / V for recording (every job rebinds its block)
		let ph = [npu.alloc(bb)?, npu.alloc(bb)?];
		let (mut mask, mut o, mut lse, mut gath) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
		let mut jobs = Vec::new();
		for _ in 0..slots() {
			let gm = &graphs[1];
			let ob = gm.args[outs(gm)[0]].bytes;
			let lb = gm.args[outs(gm)[1]].bytes;
			mask.push(npu.alloc(S * CAP * 2)?);
			let mut g = [npu.alloc(bb)?, npu.alloc(bb)?];
			for h in &mut g {
				h.as_mut_slice().fill(0);
			}
			gath.push(g);
			o.push(npu.alloc(ob)?);
			lse.push(npu.alloc(lb)?);
		}
		for s in 0..slots() {
			let mut pair = Vec::new();
			for gi in 0..2 {
				let g = &graphs[gi];
				let (oo, ol) = (outs(g)[0], outs(g)[1]);
				let mut refs: Vec<&HostBuf<'_>> = Vec::new();
				for (i, a) in g.args.iter().enumerate() {
					refs.push(if i == oo {
						&o[s]
					} else if i == ol {
						&lse[s]
					} else if a.name == "qg" {
						&qg
					} else if a.name == "kblk" {
						&ph[0]
					} else if a.name == "vblk" {
						&ph[1]
					} else if a.name == "mask" {
						&mask[s]
					} else {
						return Err(format!("attn_far: unknown argument {}", a.name));
					});
				}
				// SAFETY: a mutable list over this graph; the graph outlives
				// the job (field order)
				let gs: &'static mut [Graph<'static>] = unsafe {
					std::slice::from_raw_parts_mut(
						&graphs[gi] as *const _ as *mut Graph<'static>,
						1,
					)
				};
				pair.push(Job {
					b: npu.batch_plan_mutable(gs, &[(0, refs)])?,
				});
			}
			let mut it = pair.into_iter();
			jobs.push([it.next().unwrap(), it.next().unwrap()]);
		}
		// the placeholders stay bound until a job's first update: keep them
		std::mem::forget(ph);
		Ok(Some(NpuFar {
			npu,
			jobs,
			_graphs: graphs,
			_blobs: blobs,
			kix,
			qg,
			mask,
			o,
			lse,
			gath,
			blocks: (0..layers).map(|_| Vec::new()).collect(),
			pending: Vec::new(),
			n: 0,
			waited: false,
		}))
	}

	/// Forget every filled block (the cache was reset or replaced).
	pub fn invalidate(&mut self) {
		self.wait_all();
		for l in &mut self.blocks {
			for b in l.iter_mut().flatten() {
				b.1 = false;
			}
		}
	}

	/// The cache was cut to `p` positions: blocks reaching past it are stale.
	pub fn truncate(&mut self, p: usize) {
		self.wait_all();
		for l in &mut self.blocks {
			for (i, b) in l.iter_mut().enumerate() {
				if let Some(b) = b {
					if (i + 1) * KEYS > p {
						b.1 = false;
					}
				}
			}
		}
	}

	fn wait_all(&mut self) {
		let w = std::mem::take(&mut self.waited);
		for (s, gi) in std::mem::take(&mut self.pending) {
			if !w {
				let _ = self.jobs[s][gi].b.wait();
			}
		}
	}

	/// Wait for the jobs in flight without merging them ([`Self::end`]
	/// merges).
	pub fn wait_jobs(&mut self) -> Result<(), String> {
		if !self.waited {
			for &(s, gi) in &self.pending {
				self.jobs[s][gi].b.wait()?;
			}
			self.waited = true;
		}
		Ok(())
	}

	/// The split of `n` queries at `p0` with key lists `keys` (each
	/// query's ascending keys: every earlier one, or the indexer's 512
	/// blocks then the tail).
	pub fn plan(&self, p0: usize, n: usize, keys: &[Vec<u32>]) -> Plan {
		Self::plan_with(p0, n, keys, self.jobs.len())
	}

	/// [`Self::plan`] with up to `slots` NPU jobs.
	pub fn plan_with(p0: usize, n: usize, keys: &[Vec<u32>], slots: usize) -> Plan {
		let [base, td, tm, dk, sk] = cost();
		let mut plan = Plan {
			blocks: Vec::new(),
			gather: Vec::new(),
			kmin: 0,
			est: [0.0; 2],
		};
		let sparse = (0..n).any(|t| (p0 + t + 1) / 4 > 512);
		let qs = n as f64 / S as f64;
		if !sparse {
			let full = p0 / KEYS;
			// keys per query on average, the first blocks to the NPU
			let avg = p0 as f64 + (n as f64 + 1.0) / 2.0;
			let mut best = (avg * dk * qs, 0);
			plan.est = [base, best.0];
			for nf in 1..=slots.min(full) {
				let e = [base + nf as f64 * td, (avg - (nf * KEYS) as f64) * dk * qs];
				let c = e[0].max(e[1]);
				if c < best.0 {
					best = (c, nf);
					plan.est = e;
				}
			}
			plan.blocks = (0..best.1).collect();
			plan.kmin = best.1 * KEYS;
			return plan;
		}
		// queries per complete block of 4 before the sub-chunk (the tail
		// and the sub-chunk's own blocks stay on the iGPU)
		let nb0 = p0 / 4;
		let tot: usize = keys.iter().map(|k| k.len()).sum();
		plan.est = [base, tot as f64 / n as f64 * sk * qs];
		if nb0 < GB {
			return plan;
		}
		let mut cnt = vec![0u32; nb0];
		for ks in keys {
			for c in ks.chunks_exact(4) {
				let b = c[0] as usize / 4;
				if c[0] % 4 == 0 && b < nb0 {
					cnt[b] += 1;
				}
			}
		}
		let mut order: Vec<u32> = (0..nb0 as u32).collect();
		order.sort_by_key(|&b| (std::cmp::Reverse(cnt[b as usize]), b));
		let mut best = (tot as f64 / n as f64 * sk * qs, 0);
		let mut left = tot;
		// a job whose blocks every query lists needs no mask
		let full = |j: usize| {
			order[j * GB..(j + 1) * GB]
				.iter()
				.all(|&b| cnt[b as usize] as usize == n)
		};
		let mut npu = base;
		for nf in 1..=slots.min(nb0 / GB) {
			left -= 4 * order[(nf - 1) * GB..nf * GB]
				.iter()
				.map(|&b| cnt[b as usize] as usize)
				.sum::<usize>();
			npu += if full(nf - 1) { td } else { tm };
			let e = [npu, left as f64 / n as f64 * sk * qs];
			let c = e[0].max(e[1]);
			if c < best.0 {
				best = (c, nf);
				plan.est = e;
			}
		}
		plan.gather = (0..best.1)
			.map(|j| {
				let mut g = order[j * GB..(j + 1) * GB].to_vec();
				g.sort_unstable();
				(g, !full(j))
			})
			.collect();
		plan
	}

	/// The iGPU's selection `sel` `[n][513]` for `plan`: dense queries from
	/// `kmin` (a dense plan), else every listed block of 4 the NPU does not
	/// gather.
	pub fn igpu_sel(plan: &Plan, p0: usize, n: usize, keys: &[Vec<u32>], sel: &mut [u32]) {
		const SEL: usize = 513;
		let mut on_npu = vec![false; p0 / 4 + 1];
		for (g, _) in &plan.gather {
			for &b in g {
				on_npu[b as usize] = true;
			}
		}
		for t in 0..n {
			let r = &mut sel[t * SEL..][..SEL];
			if plan.gather.is_empty() {
				r[0] = u32::MAX;
				continue;
			}
			let nb = (p0 + t + 1) / 4;
			// the complete blocks of 4 in the list (the tail follows them)
			let mut c = 0;
			for q in keys[t].chunks_exact(4) {
				let b = q[0] as usize / 4;
				if q[0] % 4 != 0 || b >= nb {
					break;
				}
				if b >= on_npu.len() || !on_npu[b] {
					r[1 + c] = b as u32;
					c += 1;
				}
			}
			r[0] = c as u32;
		}
	}

	/// Submit the NPU's jobs of `plan` for layer `a` (`l / 4`): `q` the
	/// sub-chunk's queries `[n][24][256]` f16, `kv` the layer's cache.
	pub(super) fn begin(
		&mut self,
		a: usize,
		plan: &Plan,
		n: usize,
		q: &[u16],
		keys: &[Vec<u32>],
		kv: &Kv,
	) -> Result<(), String> {
		if !self.pending.is_empty() {
			return Err("attn_far: jobs already in flight".into());
		}
		if plan.jobs() == 0 {
			return Ok(());
		}
		let row = HKV * DA;
		let t0 = std::time::Instant::now();
		let mut t_alloc = 0f64;
		// the blocks, filled from the cache
		let lb = &mut self.blocks[a];
		for &b in &plan.blocks {
			if lb.len() <= b {
				lb.resize_with(b + 1, || None);
			}
			if lb[b].is_none() {
				let ta = std::time::Instant::now();
				let bb = CAP * row * 2;
				let mut kb = self.npu.alloc(bb)?;
				let mut vb = self.npu.alloc(bb)?;
				for h in [&mut kb, &mut vb] {
					h.as_mut_slice()[KEYS * row * 2..].fill(0);
				}
				lb[b] = Some(([kb, vb], false));
				t_alloc += ta.elapsed().as_secs_f64();
			}
			let e = lb[b].as_mut().unwrap();
			if !e.1 {
				if kv.k.rows() < (b + 1) * KEYS {
					return Err(format!("attn_far: block {b} past the cache"));
				}
				for (h, src) in e.0.iter_mut().zip([&kv.k, &kv.v]) {
					let d = h.as_mut_slice();
					// SAFETY: the buffer holds CAP >= KEYS rows of f16
					let d = unsafe {
						std::slice::from_raw_parts_mut(d.as_mut_ptr() as *mut u16, KEYS * row)
					};
					src.copy_rows(b * KEYS, (b + 1) * KEYS, d);
				}
				e.1 = true;
			}
		}
		// the gathered blocks of 4, per job slot
		for (s, (g, _)) in plan.gather.iter().enumerate() {
			for (h, src) in self.gath[s].iter_mut().zip([&kv.k, &kv.v]) {
				let dp = P(h.as_mut_slice().as_mut_ptr() as *mut u16);
				let src = src.view();
				par_for(g.len().div_ceil(16), |c| {
					let dp = &dp;
					for (i, &b) in g.iter().enumerate().skip(c * 16).take(16) {
						for j in 0..4 {
							// SAFETY: task c owns slots [64 c, 64 c + 64)
							let d = unsafe {
								std::slice::from_raw_parts_mut(dp.0.add((i * 4 + j) * row), row)
							};
							d.copy_from_slice(src.row(b as usize * 4 + j));
						}
					}
				});
			}
		}
		let t_fill = t0.elapsed().as_secs_f64();
		// qg [2][S][12][256] (token major per KV head), prescaled by 1 / 16
		// (an exponent shift)
		{
			let qp = P(self.qg.as_mut_slice().as_mut_ptr() as *mut u16);
			par_for(HQ, |h| {
				let qp = &qp;
				let (kh, g) = (h / G, h % G);
				for t in 0..S {
					// SAFETY: task h owns its rows of qg
					let dst = unsafe {
						std::slice::from_raw_parts_mut(qp.0.add(((kh * S + t) * G + g) * DA), DA)
					};
					if t >= n {
						dst.fill(0);
						continue;
					}
					for (d, &x) in dst.iter_mut().zip(&q[(t * HQ + h) * DA..][..DA]) {
						*d = if x & 0x7c00 > 4 << 10 {
							x - (4 << 10)
						} else {
							quant::f32_to_fp16(quant::fp16_to_f32(x) / 16.0)
						};
					}
				}
			});
		}
		for (s, &b) in plan.blocks.iter().enumerate() {
			let [ik, iv] = self.kix[0];
			let e = self.blocks[a][b].as_ref().unwrap();
			let job = &mut self.jobs[s][0];
			job.b.update(&[(0, ik, &e.0[0]), (0, iv, &e.0[1])])?;
			job.b.submit()?;
			self.pending.push((s, 0));
		}
		for (s, (g, masked)) in plan.gather.iter().enumerate() {
			let gi = *masked as usize;
			let [ik, iv] = self.kix[gi];
			if !masked {
				let job = &mut self.jobs[s][0];
				job.b
					.update(&[(0, ik, &self.gath[s][0]), (0, iv, &self.gath[s][1])])?;
				job.b.submit()?;
				self.pending.push((s, 0));
				continue;
			}
			// mask row t: slot 4 i + j visible iff query t lists block g[i]
			let mut pos = vec![u16::MAX; g.last().map_or(0, |&b| b as usize + 1)];
			for (i, &b) in g.iter().enumerate() {
				pos[b as usize] = i as u16;
			}
			let mp = P(self.mask[s].as_mut_slice().as_mut_ptr() as *mut u16);
			par_for(S, |t| {
				let mp = &mp;
				// SAFETY: task t owns mask row t
				let r = unsafe { std::slice::from_raw_parts_mut(mp.0.add(t * CAP), CAP) };
				r.fill(MASKED);
				if t >= n {
					return;
				}
				for c in keys[t].chunks_exact(4) {
					let b = c[0] as usize / 4;
					if c[0] % 4 == 0 && b < pos.len() && pos[b] != u16::MAX {
						let i = pos[b] as usize * 4;
						r[i..i + 4].fill(0);
					}
				}
			});
			let job = &mut self.jobs[s][1];
			job.b
				.update(&[(0, ik, &self.gath[s][0]), (0, iv, &self.gath[s][1])])?;
			job.b.submit()?;
			self.pending.push((s, 1));
		}
		self.n = n;
		if std::env::var_os("QWEN_FLASH_ATT_LOG").is_some() {
			eprintln!(
				"[far] alloc {:.1} ms, fill {:.1} ms, all {:.1} ms",
				t_alloc * 1e3,
				t_fill * 1e3,
				t0.elapsed().as_secs_f64() * 1e3
			);
		}
		Ok(())
	}

	/// Whether jobs are in flight.
	pub fn busy(&self) -> bool {
		!self.pending.is_empty()
	}

	/// Wait for the jobs and merge them into the iGPU's `o` `[n][24][256]`
	/// f16 (normalized) with its `lse` `[n][24]`.
	pub fn end(&mut self, o: &mut [u16], lse: &[f32]) -> Result<(), String> {
		let pend = std::mem::take(&mut self.pending);
		if pend.is_empty() {
			return Ok(());
		}
		if !std::mem::take(&mut self.waited) {
			for &(s, gi) in &pend {
				self.jobs[s][gi].b.wait()?;
			}
		}
		let n = self.n;
		let slots: Vec<(&[u8], &[u8])> = pend
			.iter()
			.map(|&(s, _)| (self.o[s].as_slice(), self.lse[s].as_slice()))
			.collect();
		let op = P(o.as_mut_ptr());
		par_for(n, |t| {
			let op = &op;
			let h16 = |b: &[u8], i: usize| {
				quant::fp16_to_f32(u16::from_le_bytes([b[i * 2], b[i * 2 + 1]]))
			};
			let mut acc = [0f32; DA];
			for h in 0..HQ {
				let (kh, g) = (h / G, h % G);
				let r = (kh * S + t) * G + g;
				let lg = lse[t * HQ + h];
				let mut mx = lg;
				let mut mz = [(0f32, 0f32); MAX_SLOTS];
				for (i, (_, l)) in slots.iter().enumerate() {
					let (m, z) = (h16(l, r * 2), h16(l, r * 2 + 1));
					mz[i] = (m, z);
					mx = mx.max(m + z.ln());
				}
				// SAFETY: task t owns o[t][..]
				let dst =
					unsafe { std::slice::from_raw_parts_mut(op.0.add((t * HQ + h) * DA), DA) };
				let wg = (lg - mx).exp();
				let mut den = wg;
				// F16C rows (exact conversions; the f16 output rounds to
				// nearest even as `quant::f32_to_fp16`, subnormals kept)
				crate::ir::f16s_to_f32(u8s(dst), &mut acc, wg);
				let mut b = [0f32; DA];
				for (i, (ob, _)) in slots.iter().enumerate() {
					let (m, z) = mz[i];
					let w = (m - mx).exp();
					if w == 0.0 {
						continue;
					}
					den += w * z;
					crate::ir::f16s_to_f32(&ob[r * DA * 2..][..DA * 2], &mut b, 1.0);
					for (a, &x) in acc.iter_mut().zip(&b) {
						*a += w * x;
					}
				}
				for a in acc.iter_mut() {
					*a /= den;
				}
				crate::ir::f32s_to_f16(&acc, u8sm(dst), 1.0);
			}
		});
		Ok(())
	}
}
