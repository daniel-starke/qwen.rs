//! The transformer on the NPU (`image.npu` of `cli QWEN_IMAGE_DIR
//! --npu-build OUT`; the graphs: `npubuild::qimage`).
//!
//! A forward runs the 32 blocks layer by layer over all rows (graph
//! `dit_pre` for block 0, `dit_mid` for blocks 1-31: block l - 1 after its
//! attention plus block l up to its attention, `dit_post` after block 31).
//! Once every chunk of block l has its q / k / v, the cores run the
//! attention of chunk c (`quant::attn_block_rows`, f16 K / V read where the
//! NPU wrote them) and submit the next graph's chunk c at once, so the NPU
//! runs chunk c while the cores do chunk c + 1; K / V alternate between two
//! buffers so block l + 1's are written while block l's are read. The
//! submissions queue in order; the layer waits for all of them. The rows live in NPU host buffers
//! ([`Bufs`]): every graph argument is bound at its chunk's offset, so
//! nothing is copied between jobs. The prefix (text and condition images)
//! runs once per image through the same graphs at its own modulation
//! (`t = 0`); its K / V per block are kept ([`NpuDit::prefix_k`]) and put
//! in front of the target's at every step.

use super::config::DitCfg;
use super::dit::{Layout, Modulation, Rope};
use crate::npuhost::{as_u16, f16_row, narrow, npu, widen};
use npu::lock::{NpuLock, SharedNpuLock};
use npu::ze::{Graph, HostBuf, Npu};
use npubuild::qimage::{self as qi, D, HD, NH, S};
use std::path::{Path, PathBuf};

/// `image.npu`: `key = value` lines.
#[derive(Clone, Debug)]
pub struct NpuCfg {
	/// The BF16 checkpoint (text encoder, VAE, the transformer's small
	/// weights and configs).
	pub checkpoint: PathBuf,
	/// The compiled graphs (`dit_pre.blob`, `dit_mid.blob`, `dit_post.blob`).
	pub graphs: PathBuf,
	/// The blocks' files (`dit{L}.<name>.in`).
	pub weights: PathBuf,
	/// The VAE decoder's convolution graphs (`c{ci}_{co}.blob`; a set
	/// without them runs the decoder on the cores).
	pub vae: Option<PathBuf>,
	/// The text encoder's vision tower set (`vis.cfg`, `pos.f32`,
	/// `vis{i}.blob`; optional).
	pub vision: Option<PathBuf>,
}

impl NpuCfg {
	/// True when `path` is a file whose `arch` is `qwen-image`.
	pub fn is_image_npu(path: &Path) -> bool {
		path.is_file()
			&& std::fs::read_to_string(path).is_ok_and(|t| {
				t.lines().any(|l| {
					l.split_once('=')
						.is_some_and(|(k, v)| k.trim() == "arch" && v.trim() == "qwen-image")
				})
			})
	}

	pub fn parse(path: &Path) -> Result<NpuCfg, String> {
		let t = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
		let base = path.parent().unwrap_or(Path::new("."));
		let (mut ck, mut gr, mut w, mut arch, mut vae, mut vision) =
			(None, None, None, None, None, None);
		for line in t
			.lines()
			.map(str::trim)
			.filter(|l| !l.is_empty() && !l.starts_with('#'))
		{
			let (k, v) = line
				.split_once('=')
				.ok_or(format!("{}: bad line {line}", path.display()))?;
			let (k, v) = (k.trim(), v.trim());
			let p = || {
				let p = PathBuf::from(v);
				if p.is_absolute() {
					p
				} else {
					base.join(p)
				}
			};
			match k {
				"arch" => arch = Some(v.to_string()),
				"checkpoint" => ck = Some(p()),
				"graphs" => gr = Some(p()),
				"weights" => w = Some(p()),
				"vae" => vae = Some(p()),
				"vision" => vision = Some(p()),
				_ => return Err(format!("{}: unknown key {k}", path.display())),
			}
		}
		if arch.as_deref() != Some("qwen-image") {
			return Err(format!("{}: arch is not qwen-image", path.display()));
		}
		let need = |v: Option<PathBuf>, k: &str| v.ok_or(format!("{}: no {k}", path.display()));
		Ok(NpuCfg {
			checkpoint: need(ck, "checkpoint")?,
			graphs: need(gr, "graphs")?,
			weights: need(w, "weights")?,
			vae,
			vision,
		})
	}
}

const KINDS: [&str; 3] = ["dit_pre", "dit_mid", "dit_post"];

/// Load and initialize graph `kind` from `dir` (`{kind}.blob`).
fn load_graph(npu: &'static Npu, dir: &Path, kind: &str) -> Result<Graph<'static>, String> {
	let f = dir.join(format!("{kind}.blob"));
	let b = std::fs::read(&f).map_err(|e| format!("{}: {e}", f.display()))?;
	let mut g = npu.load(&b, 0)?;
	g.initialize()?;
	Ok(g)
}

/// What a graph argument is bound to.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Io {
	/// The residual rows the job reads / writes (`X[cur]` / `X[1 - cur]`).
	X,
	Xo,
	A,
	Q,
	K,
	V,
	Cos,
	Sin,
	/// The step's modulation: ms1, mg1, ms2, mg2.
	M(usize),
	/// A weight of the block before (false) or after (true) the cut.
	W(bool, usize),
}

/// The rows of one pass (the prefix, or the target), as NPU host buffers.
struct Bufs {
	/// Rows (whole chunks).
	rows: usize,
	/// The residual, ping-pong.
	x: [HostBuf<'static>; 2],
	a: HostBuf<'static>,
	q: HostBuf<'static>,
	/// `cos` / `sin` [rows][HD] of the pass' positions (halves layout).
	cos: HostBuf<'static>,
	sin: HostBuf<'static>,
	/// The same as f32 [rows][HD / 2], for the cores' RoPE ([`QkW`]).
	cosf: Vec<f32>,
	sinf: Vec<f32>,
	/// K / V of block `l` in `[l % 2]`: `[kv_rows][D]` (prefix first). Two
	/// so the NPU can write block `l + 1`'s while the cores still read
	/// block `l`'s.
	k: [HostBuf<'static>; 2],
	v: [HostBuf<'static>; 2],
	kv_rows: usize,
}

/// A block's q / k finish on the cores (the folded graphs,
/// `npubuild::qimage::fold`: the NPU writes q / k before their row
/// scales): row scales, head norm weights (permuted, as the files).
struct QkW {
	sq: Vec<f32>,
	sk: Vec<f32>,
	nq: Vec<f32>,
	nk: Vec<f32>,
}

/// `x` [rows][D] raw q or k -> `rms(x * s) * w`, then RoPE (halves) with
/// `cos` / `sin` [rows][HD / 2], in f32 (what the unfolded graphs do in
/// f16).
fn qk_finish(x: &mut [f32], s: &[f32], w: &[f32], cos: &[f32], sin: &[f32]) {
	let half = HD / 2;
	let eps = qi::EPS as f32;
	for (r, row) in x.chunks_exact_mut(D).enumerate() {
		let (c, sn) = (
			&cos[r * half..(r + 1) * half],
			&sin[r * half..(r + 1) * half],
		);
		for (h, t) in row.chunks_exact_mut(HD).enumerate() {
			let sh = &s[h * HD..(h + 1) * HD];
			let mut ss = 0f32;
			for (v, &f) in t.iter_mut().zip(sh) {
				*v *= f;
				ss += *v * *v;
			}
			let inv = 1.0 / (ss / HD as f32 + eps).sqrt();
			t.iter_mut().zip(w).for_each(|(v, &g)| *v *= inv * g);
			let (a, b) = t.split_at_mut(half);
			for i in 0..half {
				let (x1, x2) = (a[i], b[i]);
				a[i] = x1 * c[i] - x2 * sn[i];
				b[i] = x2 * c[i] + x1 * sn[i];
			}
		}
	}
}

/// [`qk_finish`] of the f16 rows `x` (`[rows][D]`, rows `0 ..` of the
/// pass' `cos` / `sin`), in place, on the pool.
fn qk_finish_f16(x: &mut [u16], s: &[f32], w: &[f32], cos: &[f32], sin: &[f32]) {
	const R: usize = 8;
	let rows = x.len() / D;
	let p = x.as_mut_ptr() as usize;
	let half = HD / 2;
	quant::par_for(rows.div_ceil(R), move |i| {
		let (r0, r1) = (i * R, ((i + 1) * R).min(rows));
		// SAFETY: task i owns rows [r0, r1)
		let d =
			unsafe { std::slice::from_raw_parts_mut((p as *mut u16).add(r0 * D), (r1 - r0) * D) };
		let mut f = vec![0f32; d.len()];
		quant::f16_to_f32_slice(d, &mut f);
		qk_finish(
			&mut f,
			s,
			w,
			&cos[r0 * half..r1 * half],
			&sin[r0 * half..r1 * half],
		);
		quant::f32_to_f16_slice(&f, d);
	});
}

/// [`qk_finish`] of f32 rows on the pool.
fn qk_finish_par(x: &mut [f32], s: &[f32], w: &[f32], cos: &[f32], sin: &[f32]) {
	const R: usize = 8;
	let rows = x.len() / D;
	let p = x.as_mut_ptr() as usize;
	let half = HD / 2;
	quant::par_for(rows.div_ceil(R), move |i| {
		let (r0, r1) = (i * R, ((i + 1) * R).min(rows));
		// SAFETY: task i owns rows [r0, r1)
		let d =
			unsafe { std::slice::from_raw_parts_mut((p as *mut f32).add(r0 * D), (r1 - r0) * D) };
		qk_finish(
			d,
			s,
			w,
			&cos[r0 * half..r1 * half],
			&sin[r0 * half..r1 * half],
		);
	});
}

pub struct NpuDit {
	npu: &'static Npu,
	/// Per block the q / k finish on the cores (folded graphs), else none.
	qk: Option<Vec<QkW>>,
	/// The graphs ([`KINDS`]); empty while unloaded
	/// ([`Self::unload_graphs`]), loaded again by the next pass.
	graphs: Vec<Graph<'static>>,
	/// Where they are read from.
	gdir: PathBuf,
	/// Per graph its arguments' bindings.
	io: Vec<Vec<Io>>,
	/// Per block its weight buffers, by the names of [`Self::wnames`].
	w: Vec<Vec<HostBuf<'static>>>,
	wnames: Vec<String>,
	m: [HostBuf<'static>; 4],
	layers: usize,
	bufs: Option<Bufs>,
	/// The prefix's K / V per block (`[len][D]`).
	pk: Vec<Kv>,
	pv: Vec<Kv>,
	/// Keep the next prefix's K / V as int8 (set per image by the caller,
	/// from the memory budget).
	pub kv8: bool,
	/// `QWEN_IMAGE_KV8_CHECK`: the other form of the prefix K / V
	/// ([`Self::swap_kv`]).
	alt: Option<(Vec<Kv>, Vec<Kv>)>,
	/// The iGPU's share of the target rows' attention ([`gpu_attn`]).
	#[cfg(feature = "vulkan")]
	ga: Option<offload::DitAttn>,
	/// The iGPU's `dit_mid` jobs ([`gpu_blk`]): [`Self::gb_chunks`] chunks
	/// of each block of the target ([`gpu_chunk`]).
	#[cfg(feature = "vulkan")]
	gb: Option<offload::DitBlk>,
	gb_chunks: usize,
	/// Seconds waiting for the iGPU since the last report.
	pub t_gpu: f64,
	/// Seconds on the NPU and in the attention since the last report.
	pub t_npu: f64,
	pub t_attn: f64,
	/// Seconds of the cores' work between a block's last job and its
	/// first chunk's attention (the NPU idles) since the last report.
	pub t_bound: f64,
	/// Per block the channel means of its target keys at the last step
	/// (empty until the first): the int8 keys of the next step are centered
	/// at them, so they are packed chunk by chunk as the jobs finish
	/// ([`kpack_early`]).
	kmean: Vec<Vec<f32>>,
	/// Run only the first this many blocks (checks against the reference).
	pub limit: Option<usize>,
	/// The block cache across steps ([`BlockCache`], `QWEN_IMAGE_CACHE`).
	cache: Option<BlockCache>,
	_lock: SharedNpuLock,
}

/// Block cache across steps (DBCache, cache-dit): blocks `0 .. front` and
/// `layers - back .. layers` run every step; on a cached step the middle
/// blocks are replaced by the residual change they made at the last full
/// step (added to the residual after block `front - 1`), and the target
/// resumes at block `layers - back` (its `dit_pre`).
///
/// `QWEN_IMAGE_CACHE=F,B,SPEC`: `SPEC` = `s2:4:6` (those steps cached) or
/// `tX[,R]` (cached while the front residual's accumulated relative L1
/// change since the last full step stays below `X`, at most `R` in a row).
/// `QWEN_IMAGE_CACHE_LOG=1` prints per step the indicator and, at full
/// steps, how much the middle blocks' change moved since the last one.
struct BlockCache {
	front: usize,
	back: usize,
	steps: Option<Vec<usize>>,
	thresh: f64,
	max_run: usize,
	log: bool,
	/// Step of the image (forward calls since [`NpuDit::reset`]).
	step: usize,
	acc: f64,
	run: usize,
	/// This step is cached.
	skip: bool,
	x_in: Vec<f32>,
	prev_r: Vec<f32>,
	h_in: Vec<f32>,
	delta: Vec<f32>,
	/// The full step's before `delta`, and the extrapolation weight
	/// (`QWEN_IMAGE_CACHE_EXT`): a cached step adds `delta + ext (delta -
	/// prev)`.
	prev: Vec<f32>,
	ext: f32,
}

/// Per 64k-element task of `n`: `f(a, b)` -> two sums, added up.
fn par_sum2(n: usize, f: impl Fn(usize, usize) -> (f64, f64) + Sync + Send) -> (f64, f64) {
	const C: usize = 1 << 16;
	let out = std::sync::Mutex::new((0f64, 0f64));
	quant::par_for(n.div_ceil(C), |i| {
		let (x, y) = f(i * C, ((i + 1) * C).min(n));
		let mut o = out.lock().unwrap();
		o.0 += x;
		o.1 += y;
	});
	out.into_inner().unwrap()
}

#[derive(Clone, Copy)]
struct PtrF(*mut f32);
// SAFETY: the tasks of [`par_sum2`] touch disjoint ranges
unsafe impl Send for PtrF {}
unsafe impl Sync for PtrF {}

impl BlockCache {
	fn from_env(layers: usize) -> Option<BlockCache> {
		let v = std::env::var("QWEN_IMAGE_CACHE").ok()?;
		if v.is_empty() || v == "0" || v == "off" {
			return None;
		}
		let p: Vec<&str> = v.split(',').collect();
		let front: usize = p.first()?.parse().ok()?;
		let back: usize = p.get(1)?.parse().ok()?;
		if front == 0 || front + back >= layers {
			eprintln!("QWEN_IMAGE_CACHE={v}: needs 0 < F, F + B < {layers}");
			return None;
		}
		let spec = p.get(2).copied().unwrap_or("t0.1");
		let (mut steps, mut thresh, mut max_run) = (None, 0.0, usize::MAX);
		if let Some(s) = spec.strip_prefix('s') {
			steps = Some(s.split(':').filter_map(|x| x.parse().ok()).collect());
		} else {
			let t = spec.strip_prefix('t')?;
			thresh = t.parse().ok()?;
			max_run = p.get(3).and_then(|r| r.parse().ok()).unwrap_or(usize::MAX);
		}
		Some(BlockCache {
			front,
			back,
			steps,
			thresh,
			max_run,
			log: std::env::var("QWEN_IMAGE_CACHE_LOG").is_ok_and(|v| v == "1"),
			step: 0,
			acc: 0.0,
			run: 0,
			skip: false,
			x_in: Vec::new(),
			prev_r: Vec::new(),
			h_in: Vec::new(),
			delta: Vec::new(),
			prev: Vec::new(),
			ext: std::env::var("QWEN_IMAGE_CACHE_EXT")
				.ok()
				.and_then(|v| v.parse().ok())
				.unwrap_or(0.0),
		})
	}

	fn reset(&mut self) {
		self.step = 0;
		self.acc = 0.0;
		self.run = 0;
		self.skip = false;
		self.prev_r.clear();
		self.h_in.clear();
		self.delta.clear();
		self.prev.clear();
	}

	/// The residual after block `front - 1` (f16, `x`): decide whether this
	/// step is cached; if so add the middle blocks' change to `x`.
	fn at_front(&mut self, x: &mut [u16]) -> bool {
		let n = x.len();
		let mut h = quant::uninit_vec(n);
		widen(x, &mut h);
		let first = self.prev_r.len() != n;
		if first {
			self.prev_r = vec![0f32; n];
		}
		let (hp, xp, rp) = (
			PtrF(h.as_mut_ptr()),
			PtrF(self.x_in.as_mut_ptr()),
			PtrF(self.prev_r.as_mut_ptr()),
		);
		// relative L1 change of the front residual h - x_in; prev_r := it
		let (d, s) = par_sum2(n, move |a, b| {
			let (hp, xp, rp) = (hp, xp, rp);
			let (mut d, mut s) = (0f64, 0f64);
			for i in a..b {
				// SAFETY: the task owns [a, b)
				unsafe {
					let r = *hp.0.add(i) - *xp.0.add(i);
					d += (r - *rp.0.add(i)).abs() as f64;
					s += (*rp.0.add(i)).abs() as f64;
					*rp.0.add(i) = r;
				}
			}
			(d, s)
		});
		let ind = if first {
			f64::INFINITY
		} else {
			d / s.max(1e-30)
		};
		let have = self.delta.len() == n;
		self.skip = have
			&& match &self.steps {
				Some(st) => st.contains(&self.step),
				None => self.acc + ind < self.thresh && self.run < self.max_run,
			};
		if self.log {
			eprintln!(
				"[cache] step {}: front change {ind:.3}, acc {:.3} -> {}",
				self.step,
				self.acc + ind,
				if self.skip { "cached" } else { "full" }
			);
		}
		if self.skip {
			self.acc += ind;
			self.run += 1;
			let dp = PtrF(self.delta.as_mut_ptr());
			let k = if self.prev.len() == n { self.ext } else { 0.0 };
			let pp = PtrF(if k != 0.0 {
				self.prev.as_mut_ptr()
			} else {
				self.delta.as_mut_ptr()
			});
			par_sum2(n, move |a, b| {
				let (hp, dp, pp) = (hp, dp, pp);
				for i in a..b {
					// SAFETY: the task owns [a, b)
					unsafe {
						let d = *dp.0.add(i);
						*hp.0.add(i) += d + k * (d - *pp.0.add(i));
					}
				}
				(0.0, 0.0)
			});
			// SAFETY: the same buffer as bytes
			let xb = unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut u8, n * 2) };
			narrow(&h, xb);
		} else {
			self.acc = 0.0;
			self.run = 0;
			self.h_in = h;
		}
		self.skip
	}

	/// The residual after block `layers - back - 1` of a full step.
	fn at_back(&mut self, x: &[u16]) {
		let n = x.len();
		let mut h = quant::uninit_vec(n);
		widen(x, &mut h);
		let old = self.delta.len() == n;
		let (hp, ip) = (PtrF(h.as_mut_ptr()), PtrF(self.h_in.as_mut_ptr()));
		let dp = PtrF(if old {
			self.delta.as_mut_ptr()
		} else {
			h.as_mut_ptr()
		});
		let two = old && self.prev.len() == n;
		let pp = PtrF(if two { self.prev.as_mut_ptr() } else { dp.0 });
		// h := h - h_in (the new change); the errors of predicting it by
		// delta + k (delta - prev), k = 0, 0.5, 1
		let errs = std::sync::Mutex::new([0f64; 4]);
		quant::par_for(n.div_ceil(1 << 16), |t| {
			let (hp, ip, dp, pp) = (hp, ip, dp, pp);
			let (a, b) = (t << 16, ((t + 1) << 16).min(n));
			let mut e = [0f64; 4];
			for i in a..b {
				// SAFETY: the task owns [a, b)
				unsafe {
					let v = *hp.0.add(i) - *ip.0.add(i);
					if old {
						let (d, p) = (*dp.0.add(i), *pp.0.add(i));
						e[0] += ((v - d) as f64).powi(2);
						e[1] += ((v - d - 0.5 * (d - p)) as f64).powi(2);
						e[2] += ((v - 2.0 * d + p) as f64).powi(2);
					}
					e[3] += (v as f64).powi(2);
					*hp.0.add(i) = v;
				}
			}
			let mut o = errs.lock().unwrap();
			for (o, e) in o.iter_mut().zip(e) {
				*o += e;
			}
		});
		let e = errs.into_inner().unwrap();
		if self.log && old {
			let r = |x: f64| (x / e[3].max(1e-30)).sqrt();
			eprintln!(
				"[cache] step {}: the middle blocks' change vs the last full step's (rel. L2): reused {:.3}{}",
				self.step,
				r(e[0]),
				if two {
					format!(", extrapolated k 0.5 {:.3}, k 1 {:.3}", r(e[1]), r(e[2]))
				} else {
					String::new()
				}
			);
		}
		self.prev = std::mem::replace(&mut self.delta, h);
	}
}

// SAFETY: the NPU objects are used from one thread at a time (the model
// is behind the server's job queue).
unsafe impl Send for NpuDit {}
unsafe impl Sync for NpuDit {}

/// One block's prefix K or V (`[rows][D]`): f16 bits, or int8 with a
/// scale per row and head (an edit with many reference images, whose f16
/// K / V would not fit the memory budget: [`NpuDit::kv8`]).
enum Kv {
	F16(Vec<u16>),
	I8(Vec<i8>, Vec<f32>),
}

/// The per-head rotation of int8 prefix K / V: `H (s * x)` per head (`H`
/// the normalized Hadamard matrix, `s` fixed random signs), undone when
/// written back. It spreads K's few large channels over the head, which
/// a scale per row and head would otherwise spend on them
/// (`QWEN_IMAGE_KV8_ROT=0` is the A/B).
fn kv_rot() -> Option<&'static [f32]> {
	static S: std::sync::OnceLock<Option<Vec<f32>>> = std::sync::OnceLock::new();
	S.get_or_init(|| {
		std::env::var("QWEN_IMAGE_KV8_ROT")
			.map_or(true, |v| v != "0")
			.then(|| npubuild::gptq::signs(HD))
	})
	.as_deref()
}

impl Kv {
	/// `src` (f16 bits, whole rows) as f16, or as int8 when `i8`.
	fn new(src: &[u16], i8: bool) -> Kv {
		if !i8 {
			return Kv::F16(src.to_vec());
		}
		let heads = src.len() / HD;
		let rot = kv_rot();
		let mut q = vec![0i8; src.len()];
		let mut s = vec![0f32; heads];
		let (qp, sp) = (q.as_mut_ptr() as usize, s.as_mut_ptr() as usize);
		const C: usize = 512;
		quant::par_for(heads.div_ceil(C), move |i| {
			let mut f = [0f32; HD];
			for h in i * C..((i + 1) * C).min(heads) {
				quant::f16_to_f32_slice(&src[h * HD..(h + 1) * HD], &mut f);
				if let Some(sg) = rot {
					f.iter_mut().zip(sg).for_each(|(v, &g)| *v *= g);
					npubuild::la::fwht_blocks(&mut f, HD);
				}
				let m = f.iter().fold(0f32, |a, &v| a.max(v.abs()));
				let sc = if m > 0.0 { m / 127.0 } else { 1.0 };
				let inv = 1.0 / sc;
				// SAFETY: task i owns heads [i C, (i + 1) C)
				unsafe {
					*(sp as *mut f32).add(h) = sc;
					let d = std::slice::from_raw_parts_mut((qp as *mut i8).add(h * HD), HD);
					for (o, &v) in d.iter_mut().zip(&f) {
						*o = (v * inv).round() as i8;
					}
				}
			}
		});
		Kv::I8(q, s)
	}

	/// Write the rows as f16 bits to `dst` (`len()` values).
	///
	/// # Safety
	/// `dst` must be valid for that many writes, and nothing else may
	/// access them meanwhile.
	unsafe fn write(&self, dst: *mut u16) {
		match self {
			Kv::F16(v) => std::ptr::copy_nonoverlapping(v.as_ptr(), dst, v.len()),
			Kv::I8(q, s) => {
				let (heads, dp, rot) = (s.len(), dst as usize, kv_rot());
				const C: usize = 512;
				quant::par_for(heads.div_ceil(C), move |i| {
					let mut f = [0f32; HD];
					for h in i * C..((i + 1) * C).min(heads) {
						let sc = s[h];
						for (o, &v) in f.iter_mut().zip(&q[h * HD..(h + 1) * HD]) {
							*o = v as f32 * sc;
						}
						if let Some(sg) = rot {
							npubuild::la::fwht_blocks(&mut f, HD);
							f.iter_mut().zip(sg).for_each(|(v, &g)| *v *= g);
						}
						// SAFETY: task i owns heads [i C, (i + 1) C) of dst
						let d = unsafe {
							std::slice::from_raw_parts_mut((dp as *mut u16).add(h * HD), HD)
						};
						quant::f32_to_f16_slice(&f, d);
					}
				});
			}
		}
	}

	fn len(&self) -> usize {
		match self {
			Kv::F16(v) => v.len(),
			Kv::I8(q, _) => q.len(),
		}
	}

	/// The rows as f16 bits, when they are kept so.
	fn f16(&self) -> Option<&[u16]> {
		match self {
			Kv::F16(v) => Some(v),
			Kv::I8(..) => None,
		}
	}
}

/// The iGPU for a share of the target rows' attention
/// (`offload::DitAttn`), unless `QWEN_IMAGE_ATTN_GPU=0` or there is none.
#[cfg(feature = "vulkan")]
fn gpu_attn() -> Option<offload::DitAttn> {
	if gpu_share() <= 0.0 {
		return None;
	}
	match offload::DitAttn::new(Some("intel")) {
		Ok(g) => Some(g),
		Err(e) => {
			eprintln!("[image] no iGPU for the attention ({e}): the cores run all of it");
			None
		}
	}
}

/// The share of each block's query chunks the iGPU takes
/// (`QWEN_IMAGE_ATTN_GPU`), default 0: measured slower. 1 MP edit,
/// transformer: cores only 151.7 s, 1/16 of the chunks on the iGPU 158.8,
/// 3/16 196.2; the kernel alone runs 0.47 TFLOP/s, but next to the
/// cores' attention it waits for memory and slows the cores (their
/// attention 114 -> 120 / 134 s for less work).
#[cfg(feature = "vulkan")]
/// `QWEN_IMAGE_GPU_BLK`: the iGPU's share of each block's chunks of the
/// target (`dit_mid` jobs; `offload::DitBlk`), as a fraction of the chunks
/// (default 0.3125: 5 of 16 at 1 MP; `0` = the NPU runs all of them).
fn gpu_blk_share() -> f64 {
	std::env::var("QWEN_IMAGE_GPU_BLK")
		.ok()
		.and_then(|v| v.parse().ok())
		.unwrap_or(0.3125)
}

/// Whether chunk `c` of a block's `chunks` is one of the iGPU's `nb`:
/// spread from chunk 1 on, so the iGPU starts after the block's second
/// chunk of attention and the NPU after its first (the first `nb` as one
/// submission left both idle for `nb` chunks of attention, ~40 ms of a
/// ~310 ms block at 1 MP).
fn gpu_chunk(c: usize, nb: usize, chunks: usize) -> bool {
	c >= 1 && nb > 0 && (0..nb).any(|i| 1 + i * (chunks - 1) / nb == c)
}

/// The iGPU for [`gpu_blk_share`] with block `l`'s vectors from the weight
/// files (folded sets only).
#[cfg(feature = "vulkan")]
fn gpu_blk(
	w: &[Vec<HostBuf<'static>>],
	wnames: &[String],
	folded: bool,
) -> Option<offload::DitBlk> {
	if gpu_blk_share() <= 0.0 || !folded {
		return None;
	}
	// the iGPU imports every block's int8 matrices (6.9 GB): next to a chat
	// engine that imports much already (Flash-Next's expert pools) that
	// lost the NPU for good, so the NPU runs every chunk then
	let mats: Vec<usize> = ["o.w", "gu.w", "d.w", "qkv.w"]
		.iter()
		.filter_map(|n| wnames.iter().position(|w| w == n))
		.collect();
	let need: usize = w
		.iter()
		.flat_map(|b| mats.iter().map(|&i| b[i].len()))
		.sum();
	let (have, cap) = (offload::host_imported(), offload::host_import_budget());
	if have + need > cap {
		eprintln!(
			"[image] the transformer's blocks all on the NPU: the iGPU holds {:.1} GB of imports, its share would add {:.1} GB (budget {:.1} GB, QWEN_GPU_IMPORT_MB)",
			have as f64 / 1e9,
			need as f64 / 1e9,
			cap as f64 / 1e9
		);
		return None;
	}
	let mut g = match offload::DitBlk::new(Some("intel"), w.len()) {
		Ok(g) => g,
		Err(e) => {
			eprintln!(
				"[image] no iGPU for the transformer's blocks ({e}): the NPU runs all of them"
			);
			return None;
		}
	};
	let idx = |n: &str| wnames.iter().position(|w| w == n);
	let names = ["o.pre", "gu.pre", "qkv.pre", "gu.sg", "o.s", "d.s"];
	let ix: Vec<usize> = names.iter().filter_map(|n| idx(n)).collect();
	if ix.len() != names.len()
		|| ["o.w", "gu.w", "d.w", "qkv.w"]
			.iter()
			.any(|n| idx(n).is_none())
	{
		return None;
	}
	for (l, b) in w.iter().enumerate() {
		let v: Vec<Vec<f32>> = ix.iter().map(|&i| qi::f16f(b[i].as_slice())).collect();
		g.set_block(l, [&v[0], &v[1], &v[2], &v[3], &v[4], &v[5]]);
	}
	Some(g)
}

#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
fn gpu_share() -> f64 {
	std::env::var("QWEN_IMAGE_ATTN_GPU")
		.ok()
		.and_then(|v| v.parse().ok())
		.unwrap_or(0.0)
}

/// Recorded per-chunk lists the pipeline keeps alive at most (each holds
/// descriptors; a 1 MP target has 16 chunks per block).
const MAX_LISTS: usize = 32;

/// The pool width of the attention overlapped with the NPU, for `n` rows
/// after `p` prefix keys (`QWEN_IMAGE_ATTN_THREADS` overrides; 0 = all).
/// The NPU's jobs slow down next to more than 16 busy cores (not from
/// bandwidth: cache-resident work does it too). With the int8 attention
/// (`int8`) 16 wins even where the attention is the longer part: 1 MP
/// edit with one reference 16 / 12 / 24 threads 110.7 / 110.8 / 126.0 s,
/// three references 147.9 / - / 170.0 s; text to image (NPU-bound) 8-16
/// threads 93.8-94.0 s. The float attention needs every core once an
/// edit's prefix makes it the longer part (one reference: 24 threads
/// 158.7 s, 16 183.6).
fn attn_threads(n: usize, p: usize, int8: bool) -> usize {
	std::env::var("QWEN_IMAGE_ATTN_THREADS")
		.ok()
		.and_then(|v| v.parse().ok())
		.unwrap_or(if !int8 && 3 * p > n { 0 } else { 16 })
}

/// The attention's precision on the cores (`QWEN_IMAGE_ATTN_I8`): `kv`
/// (default) Q.K and P V in int8 (`quant::K8` with V), `k` Q.K only, `0`
/// both in f32 on the half K / V (the A/B). Returns (Q.K int8, P V int8).
fn attn_i8() -> (bool, bool) {
	match std::env::var("QWEN_IMAGE_ATTN_I8").as_deref() {
		Ok("0") => (false, false),
		Ok("k") => (true, false),
		_ => (true, true),
	}
}

/// Pack a block's int8 keys and values chunk by chunk while the NPU runs
/// the block's later chunks, at the previous step's channel means
/// (`QWEN_IMAGE_KPACK_EARLY`, default on; `0` packs them all once the
/// block's last job is done, at their own means: the A/B).
fn kpack_early() -> bool {
	std::env::var("QWEN_IMAGE_KPACK_EARLY").map_or(true, |v| v != "0")
}

/// What the recorder thread reports: a `dit_mid` job of chunk `c` done, or
/// every job submitted before the sync done.
enum Ev {
	Chunk(usize),
	End(Result<(), String>),
}

/// What recording a job needs (the graphs mutable, for binding).
struct Rec<'a> {
	npu: &'static Npu,
	graphs: &'a mut [Graph<'static>],
	io: &'a [Vec<Io>],
	w: &'a [Vec<HostBuf<'static>>],
	m: &'a [HostBuf<'static>; 4],
	b: &'a Bufs,
}

// SAFETY: moved to the recorder thread, which alone binds the graphs; the
// driver's handles are thread safe; the buffers are only read through it.
unsafe impl Send for Rec<'_> {}

impl Rec<'_> {
	fn reborrow(&mut self) -> Rec<'_> {
		Rec {
			npu: self.npu,
			graphs: self.graphs,
			io: self.io,
			w: self.w,
			m: self.m,
			b: self.b,
		}
	}
}

/// Record graph `kind` of block `l` over chunks `c0 .. c1`: residual
/// `X[cur]` -> `X[1 - cur]`, K / V rows into `K[kv]` / `V[kv]` from `kofs`
/// on. The queue runs submissions in order.
#[allow(clippy::too_many_arguments)]
fn record(
	r: Rec<'_>,
	kind: usize,
	l: usize,
	c0: usize,
	c1: usize,
	cur: usize,
	kofs: usize,
	kv: usize,
) -> Result<npu::ze::Batch<'static>, String> {
	let (b, io, w, m) = (r.b, &r.io[kind], r.w, r.m);
	r.npu.batch_each(&mut r.graphs[kind], c1 - c0, &mut |g, j| {
		let c = c0 + j;
		let row = c * S * D * 2;
		for (i, x) in io.iter().enumerate() {
			match *x {
				Io::X => g.bind_at(i, &b.x[cur], row)?,
				Io::Xo => g.bind_at(i, &b.x[1 - cur], row)?,
				Io::A => g.bind_at(i, &b.a, row)?,
				Io::Q => g.bind_at(i, &b.q, row)?,
				Io::K => g.bind_at(i, &b.k[kv], (kofs + c * S) * D * 2)?,
				Io::V => g.bind_at(i, &b.v[kv], (kofs + c * S) * D * 2)?,
				Io::Cos => g.bind_at(i, &b.cos, c * S * HD * 2)?,
				Io::Sin => g.bind_at(i, &b.sin, c * S * HD * 2)?,
				Io::M(j) => g.bind(i, &m[j])?,
				Io::W(pre, j) => g.bind(i, &w[if pre { l } else { l - 1 }][j])?,
			}
		}
		Ok(())
	})
}

/// The prefix's K / V of block `l` (`pk[l]`, `pv[l]`, `p` rows) in front of
/// `K[l % 2]` / `V[l % 2]`.
///
/// # Safety
/// Nothing may read or write those rows meanwhile.
unsafe fn prefix_kv(b: &Bufs, pk: &[Kv], pv: &[Kv], l: usize, p: usize) {
	prefix_kv_parts(b, pk, pv, l, p, true, true);
}

/// [`prefix_kv`] of K only (`k`), V only (`v`), both or neither.
///
/// # Safety
/// As [`prefix_kv`].
unsafe fn prefix_kv_parts(b: &Bufs, pk: &[Kv], pv: &[Kv], l: usize, p: usize, k: bool, v: bool) {
	let kv = l % 2;
	for (on, dst, src) in [(k, &b.k[kv], &pk[l]), (v, &b.v[kv], &pv[l])] {
		debug_assert_eq!(src.len(), p * D);
		if on {
			src.write(dst.ptr() as *mut u16);
		}
	}
}

/// `QWEN_IMAGE_NANCHECK`: the non-finite values of block `l`'s input
/// residual and q / k / v.
fn nancheck_rows(b: &Bufs, l: usize, n: usize, p: usize, cur: usize) {
	let kv = l % 2;
	let bad = |v: &[u16]| v.iter().filter(|&&h| h & 0x7c00 == 0x7c00).count();
	let xr = as_u16(b.x[cur].as_slice());
	eprintln!(
		"  [nan] block {l}: x {} q {} k {} v {}",
		bad(&xr[..n * D]),
		bad(&as_u16(b.q.as_slice())[..n * D]),
		bad(&as_u16(b.k[kv].as_slice())[p * D..(p + n) * D]),
		bad(&as_u16(b.v[kv].as_slice())[p * D..(p + n) * D]),
	);
	let at: Vec<(usize, usize, f32)> = xr[..n * D]
		.iter()
		.enumerate()
		.filter(|(_, &h)| h & 0x7c00 == 0x7c00)
		.take(6)
		.map(|(i, _)| {
			let r = i / D;
			let mx = xr[r * D..(r + 1) * D]
				.iter()
				.filter(|&&h| h & 0x7c00 != 0x7c00)
				.fold(0f32, |a, &h| a.max(quant::fp16_to_f32(h).abs()));
			(r, i % D, mx)
		})
		.collect();
	if !at.is_empty() {
		eprintln!("  [nan] (row, channel, row max) {at:?}");
	}
}

impl NpuDit {
	pub fn load(
		c: &NpuCfg,
		cfg: &DitCfg,
		mut progress: impl FnMut(usize),
	) -> Result<NpuDit, String> {
		if (
			cfg.dim(),
			cfg.heads,
			cfg.head_dim,
			cfg.mlp_ratio * cfg.dim(),
		) != (D, NH, HD, qi::FF)
		{
			return Err("image.npu: the transformer's shape is not the graphs'".into());
		}
		let lock = NpuLock::shared()?;
		let npu = npu()?;
		let mut graphs = Vec::new();
		let mut io = Vec::new();
		let mut wnames: Vec<String> = Vec::new();
		for kind in KINDS {
			let g = load_graph(npu, &c.graphs, kind)?;
			let mut outs = qi::outputs(kind).iter();
			let mut v = Vec::new();
			for a in &g.args {
				let x = if a.is_output {
					match *outs
						.next()
						.ok_or(format!("{kind}: more outputs than expected"))?
					{
						"xo" => Io::Xo,
						"q" => Io::Q,
						"k" => Io::K,
						_ => Io::V,
					}
				} else {
					match a.name.as_str() {
						"x" => Io::X,
						"a" => Io::A,
						"cos" => Io::Cos,
						"sin" => Io::Sin,
						"ms1" => Io::M(0),
						"mg1" => Io::M(1),
						"ms2" => Io::M(2),
						"mg2" => Io::M(3),
						n => {
							let i = match wnames.iter().position(|w| w == n) {
								Some(i) => i,
								None => {
									wnames.push(n.to_string());
									wnames.len() - 1
								}
							};
							Io::W(qi::is_pre_weight(n), i)
						}
					}
				};
				v.push(x);
			}
			graphs.push(g);
			io.push(v);
		}
		let mut w = Vec::with_capacity(cfg.layers);
		for l in 0..cfg.layers {
			let mut v = Vec::with_capacity(wnames.len());
			for n in &wnames {
				use std::io::Read;
				let p = qi::wfile(&c.weights, l, n);
				let e = |e: std::io::Error| format!("{}: {e}", p.display());
				let mut f = std::fs::File::open(&p).map_err(e)?;
				let len = f.metadata().map_err(e)?.len() as usize;
				let mut hb = npu.alloc(len)?;
				f.read_exact(hb.as_mut_slice()).map_err(e)?;
				v.push(hb);
			}
			w.push(v);
			progress(l);
		}
		// check every weight's size against the graphs once
		for (k, g) in graphs.iter().enumerate() {
			for (a, x) in g.args.iter().zip(&io[k]) {
				if let Io::W(_, i) = *x {
					if w[0][i].len() != a.bytes {
						return Err(format!(
							"{}: {} has {} bytes, the graph wants {}",
							KINDS[k],
							wnames[i],
							w[0][i].len(),
							a.bytes
						));
					}
				}
			}
		}
		// folded graphs: q / k's row scales and head norms on the cores
		let qk = if wnames.iter().any(|n| n == "gu.sg") {
			let rd = |l: usize, n: &str| -> Result<Vec<f32>, String> {
				let p = qi::wfile(&c.weights, l, n);
				Ok(qi::f16f(
					&std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?,
				))
			};
			let mut v = Vec::with_capacity(cfg.layers);
			for l in 0..cfg.layers {
				let s = rd(l, "qkv.s")?;
				let (nq, nk) = (rd(l, "nq")?, rd(l, "nk")?);
				if s.len() != 3 * D || nq.len() != HD || nk.len() != HD {
					return Err(format!("image.npu: block {l}: bad qkv.s / nq / nk"));
				}
				v.push(QkW {
					sq: s[..D].to_vec(),
					sk: s[D..2 * D].to_vec(),
					nq,
					nk,
				});
			}
			Some(v)
		} else {
			None
		};
		let m = [
			npu.alloc(D * 2)?,
			npu.alloc(D * 2)?,
			npu.alloc(D * 2)?,
			npu.alloc(D * 2)?,
		];
		#[cfg(feature = "vulkan")]
		let gb = gpu_blk(&w, &wnames, qk.is_some());
		Ok(NpuDit {
			npu,
			qk,
			graphs,
			gdir: c.graphs.clone(),
			io,
			w,
			wnames,
			m,
			layers: cfg.layers,
			bufs: None,
			pk: Vec::new(),
			pv: Vec::new(),
			kv8: false,
			alt: None,
			#[cfg(feature = "vulkan")]
			ga: gpu_attn(),
			#[cfg(feature = "vulkan")]
			gb,
			gb_chunks: 0,
			t_gpu: 0.0,
			t_npu: 0.0,
			t_attn: 0.0,
			t_bound: 0.0,
			kmean: Vec::new(),
			limit: None,
			cache: BlockCache::from_env(cfg.layers),
			_lock: lock,
		})
	}

	/// Bytes of the weights in NPU host memory.
	pub fn resident_bytes(&self) -> usize {
		self.w.iter().flatten().map(|b| b.len()).sum()
	}

	/// Make the buffers hold `rows` rows of a pass and `kv_rows` K / V rows.
	fn ensure(&mut self, rows: usize, kv_rows: usize) -> Result<(), String> {
		self.load_graphs()?;
		let rows = rows.next_multiple_of(S);
		let kv_rows = kv_rows.max(rows);
		// exact sizes: a prefix of many reference images makes the prefix
		// pass' buffers far larger than the target's
		if self
			.bufs
			.as_ref()
			.is_some_and(|b| b.rows == rows && b.kv_rows == kv_rows)
		{
			return Ok(());
		}
		self.forget_gpu();
		self.bufs = None;
		let npu = self.npu;
		let a = |n: usize| -> Result<HostBuf<'static>, String> {
			let mut b = npu.alloc(n * 2)?;
			b.as_mut_slice().fill(0);
			Ok(b)
		};
		self.bufs = Some(Bufs {
			rows,
			x: [a(rows * D)?, a(rows * D)?],
			a: a(rows * D)?,
			q: a(rows * D)?,
			cos: a(rows * HD)?,
			sin: a(rows * HD)?,
			cosf: vec![0.0; rows * HD / 2],
			sinf: vec![0.0; rows * HD / 2],
			k: [a(kv_rows * D)?, a(kv_rows * D)?],
			v: [a(kv_rows * D)?, a(kv_rows * D)?],
			kv_rows,
		});
		Ok(())
	}

	fn set_mod(&mut self, m: &[f32]) {
		let d = D;
		#[cfg(feature = "vulkan")]
		if let Some(g) = self.gb.as_mut() {
			let p = |i: usize| -> Vec<f32> {
				let v = &m[i * d..(i + 1) * d];
				if i.is_multiple_of(2) {
					v.iter().map(|&s| 1.0 + s).collect()
				} else {
					v.to_vec()
				}
			};
			let v = [p(0), p(1), p(2), p(3)];
			g.set_mods([&v[0], &v[1], &v[2], &v[3]]);
		}
		for (i, part) in [0usize, 1, 2, 3].into_iter().enumerate() {
			let src = &m[part * d..(part + 1) * d];
			let v: Vec<f32> = if part % 2 == 0 {
				src.iter().map(|&s| 1.0 + s).collect()
			} else {
				src.to_vec()
			};
			f16_row(self.m[i].as_mut_slice(), &v);
		}
	}

	/// `cos` / `sin` of positions `p0 .. p0 + n` into the pass' buffers
	/// (pair `i` at lanes `i` and `64 + i`); the padding rows stay zero.
	fn set_rope(&mut self, rope: &Rope, p0: usize, n: usize) {
		let b = self.bufs.as_mut().unwrap();
		let half = HD / 2;
		for (f, src) in [(&mut b.cosf, &rope.cos), (&mut b.sinf, &rope.sin)] {
			f[..n * half].copy_from_slice(&src[p0 * half..(p0 + n) * half]);
			f[n * half..].fill(0.0);
		}
		for (buf, src) in [(&mut b.cos, &rope.cos), (&mut b.sin, &rope.sin)] {
			let s = buf.as_mut_slice();
			s[n * HD * 2..].fill(0);
			for r in 0..n {
				let row = &src[(p0 + r) * half..(p0 + r + 1) * half];
				let d = &mut s[r * HD * 2..(r + 1) * HD * 2];
				f16_row(&mut d[..half * 2], row);
				f16_row(&mut d[half * 2..], row);
			}
		}
	}

	/// [`record`] graph `kind` of block `l` over chunks `c0 .. c1` and
	/// submit it.
	#[allow(clippy::too_many_arguments)]
	fn submit(
		&mut self,
		kind: usize,
		l: usize,
		c0: usize,
		c1: usize,
		cur: usize,
		kofs: usize,
		kv: usize,
	) -> Result<npu::ze::Batch<'static>, String> {
		let r = Rec {
			npu: self.npu,
			graphs: &mut self.graphs,
			io: &self.io,
			w: &self.w,
			m: &self.m,
			b: self.bufs.as_ref().unwrap(),
		};
		let mut batch = record(r, kind, l, c0, c1, cur, kofs, kv)?;
		batch.submit()?;
		Ok(batch)
	}

	/// [`Self::submit`] over all chunks, and wait.
	fn run(
		&mut self,
		kind: usize,
		l: usize,
		chunks: usize,
		cur: usize,
		kofs: usize,
		kv: usize,
	) -> Result<(), String> {
		let t = std::time::Instant::now();
		self.submit(kind, l, 0, chunks, cur, kofs, kv)?.wait()?;
		self.t_npu += t.elapsed().as_secs_f64();
		Ok(())
	}

	/// Restart the block cache's steps (a new trajectory on the same
	/// prefix).
	pub fn reset_cache(&mut self) {
		if let Some(c) = self.cache.as_mut() {
			c.reset();
		}
	}

	/// Forget the prefix (a new image).
	pub fn reset(&mut self) {
		if let Some(c) = self.cache.as_mut() {
			c.reset();
		}
		self.pk.clear();
		self.pv.clear();
		self.alt = None;
		self.kmean.clear();
	}

	/// Swap the f16 prefix K / V with their int8 form (made on the first
	/// call): `QWEN_IMAGE_KV8_CHECK` runs each step with both.
	pub fn swap_kv(&mut self) {
		let alt = self.alt.take().unwrap_or_else(|| {
			let only = std::env::var("QWEN_IMAGE_KV8_ONLY").unwrap_or_default();
			let q = |v: &[Kv], on: bool| -> Vec<Kv> {
				v.iter()
					.map(|k| match k {
						Kv::F16(x) if only == "ulp" => Kv::F16(
							x.iter()
								.enumerate()
								.map(|(i, &h)| {
									if i % 2 == 1 && h & 0x7fff != 0 {
										h + 1
									} else {
										h
									}
								})
								.collect(),
						),
						Kv::F16(x) => Kv::new(x, on),
						Kv::I8(..) => panic!("swap_kv: the prefix is int8 already"),
					})
					.collect()
			};
			let alt = (q(&self.pk, only != "v"), q(&self.pv, only != "k"));
			// reconstruction error of the int8 form, per block
			let err = |a: &[Kv], b: &[Kv]| -> Vec<f64> {
				a.iter()
					.zip(b)
					.map(|(x, y)| {
						let n = x.len();
						let (mut u, mut w) = (vec![0u16; n], vec![0u16; n]);
						unsafe {
							x.write(u.as_mut_ptr());
							y.write(w.as_mut_ptr());
						}
						let (mut e, mut r) = (0f64, 0f64);
						for (&p, &q) in u.iter().zip(&w) {
							let (p, q) =
								(quant::fp16_to_f32(p) as f64, quant::fp16_to_f32(q) as f64);
							e += (p - q) * (p - q);
							r += p * p;
						}
						(e / r.max(1e-30)).sqrt()
					})
					.collect()
			};
			let (ek, ev) = (err(&self.pk, &alt.0), err(&self.pv, &alt.1));
			let f = |v: &[f64]| {
				v.iter()
					.map(|e| format!("{e:.1e}"))
					.collect::<Vec<_>>()
					.join(" ")
			};
			eprintln!("[kv8] K rel. error per block: {}", f(&ek));
			eprintln!("[kv8] V rel. error per block: {}", f(&ev));
			alt
		});
		let old = (
			std::mem::replace(&mut self.pk, alt.0),
			std::mem::replace(&mut self.pv, alt.1),
		);
		self.alt = Some(old);
	}

	/// True when the prefix of the current image has run.
	pub fn has_prefix(&self) -> bool {
		!self.pk.is_empty()
	}

	/// Run the prefix rows `txt` (`[len][D]`, the layout's first `len`
	/// positions) through the blocks at modulation `m` (`t = 0`) and keep
	/// their K / V per block.
	pub fn prefix(
		&mut self,
		txt: Vec<f32>,
		lay: &Layout,
		rope: &Rope,
		m: &Modulation,
	) -> Result<(), String> {
		let n = lay.prefix;
		self.reset();
		if n == 0 {
			self.pk = (0..self.layers).map(|_| Kv::F16(Vec::new())).collect();
			self.pv = (0..self.layers).map(|_| Kv::F16(Vec::new())).collect();
			return Ok(());
		}
		self.ensure(n, n)?;
		self.set_mod(&m.prefix);
		self.set_rope(rope, 0, n);
		{
			let b = self.bufs.as_mut().unwrap();
			let x = b.x[0].as_mut_slice();
			x[n * D * 2..].fill(0);
			narrow(&txt, &mut x[..n * D * 2]);
		}
		drop(txt);
		self.run(0, 0, n.div_ceil(S), 0, 0, 0)?;
		self.pipeline(n, 0, Some(&lay.segments), self.layers, None)?;
		Ok(())
	}

	/// The target rows' final residual (before `norm_out`) for the
	/// embedded target latents `x` (`img_in` rows, `[n][D]`) at modulation
	/// `m`; the prefix ([`Self::prefix`]) must have run.
	pub fn forward(
		&mut self,
		x: &[f32],
		lay: &Layout,
		rope: &Rope,
		m: &Modulation,
	) -> Result<Vec<f32>, String> {
		if !self.has_prefix() {
			return Err("npu transformer: the prefix has not run".into());
		}
		let p = lay.prefix;
		let n = x.len() / D;
		self.ensure(n, p + n.next_multiple_of(S))?;
		self.set_mod(&m.target);
		self.set_rope(rope, p, n);
		{
			let b = self.bufs.as_mut().unwrap();
			let xb = b.x[0].as_mut_slice();
			xb[n * D * 2..].fill(0);
			narrow(x, &mut xb[..n * D * 2]);
		}
		let layers = self.limit.unwrap_or(self.layers).clamp(1, self.layers);
		self.prefix_kv(0, p);
		self.run(0, 0, n.div_ceil(S), 0, p, 0)?;
		let mut bc = if self.limit.is_none() {
			self.cache.take()
		} else {
			None
		};
		if let Some(c) = bc.as_mut() {
			c.x_in.clear();
			c.x_in.extend_from_slice(x);
		}
		let r = self.pipeline(n, p, None, layers, bc.as_mut());
		if let Some(mut c) = bc {
			c.step += 1;
			c.skip = false;
			self.cache = Some(c);
		}
		let cur = r?;
		let b = self.bufs.as_ref().unwrap();
		let mut out = quant::uninit_vec(n * D);
		widen(&as_u16(b.x[cur].as_slice())[..n * D], &mut out);
		Ok(out)
	}

	/// Blocks `0 .. layers` over `n` rows whose block 0 up to its attention
	/// (`dit_pre`) has run: per block, once every chunk has its q / k / v,
	/// the cores run chunk c's attention and the next graph's chunk c goes
	/// to the NPU, which runs it while the cores do chunk c + 1.
	///
	/// The target (`segs` = `None`): its K / V rows from `kofs` (the
	/// prefix's in front, [`prefix_kv`]), every query seeing every key; the
	/// last block ends with `dit_post`. The prefix (`segs` = its segments,
	/// `kofs` = 0): causal text, block-causal images; each block's K / V
	/// are kept ([`Self::pk`]); the last block needs no attention. Returns
	/// the residual's buffer.
	fn pipeline(
		&mut self,
		n: usize,
		kofs: usize,
		segs: Option<&[(usize, usize, bool)]>,
		layers: usize,
		mut bc: Option<&mut BlockCache>,
	) -> Result<usize, String> {
		let chunks = n.div_ceil(S);
		let p = kofs;
		let mut qf = vec![0f32; S * D];
		let mut af = vec![0f32; S * D];
		let scale = 1.0 / (HD as f32).sqrt();
		let nancheck = std::env::var_os("QWEN_IMAGE_NANCHECK").is_some() && segs.is_none();
		// Appends go through a recorder thread: an append waits for the
		// NPU's running job (~13 ms of a 25 ms job; ~0.1 ms when idle), which
		// would stall the cores' attention of the next chunk.
		type Job = Option<(usize, usize, usize, usize, usize)>;
		let (tx, rx) = std::sync::mpsc::channel::<Job>();
		let (done_tx, done_rx) = std::sync::mpsc::channel::<Ev>();
		let rec = Rec {
			npu: self.npu,
			graphs: &mut self.graphs,
			io: &self.io,
			w: &self.w,
			m: &self.m,
			b: self.bufs.as_ref().unwrap(),
		};
		let b = rec.b;
		#[cfg_attr(not(feature = "vulkan"), allow(unused_variables))]
		let rec_w = rec.w;
		let (pk, pv, kv8) = (&mut self.pk, &mut self.pv, self.kv8);
		let qk = self.qk.as_deref();
		let (mut t_attn, mut t_wait, mut t_gpu, mut t_bound) = (0.0, 0.0, 0.0, 0.0);
		let mut kmean = std::mem::take(&mut self.kmean);
		let gone = || "npu recorder gone".to_string();
		// the iGPU's chunks: the last `ng` of each block's (target only)
		#[cfg(feature = "vulkan")]
		let mut ga = if segs.is_none() { self.ga.take() } else { None };
		#[cfg(feature = "vulkan")]
		// (not with the folded graphs: the iGPU would read raw q)
		let ng = match &ga {
			Some(_) if qk.is_none() => {
				((chunks as f64 * gpu_share()).round() as usize).min(chunks - 1)
			}
			_ => 0,
		};
		#[cfg(not(feature = "vulkan"))]
		let ng = 0usize;
		let cg = chunks - ng;
		// the iGPU's dit_mid chunks: the first `nb` of each block (target)
		#[cfg(feature = "vulkan")]
		let mut gb = if segs.is_none() && ng == 0 {
			self.gb.take()
		} else {
			None
		};
		#[cfg(feature = "vulkan")]
		// text to image and one-reference edits (prefix up to twice the
		// target rows; 1 MP edit 125.3 / 126.5 -> 100.3-104.9 s at 5 / 16);
		// edits with more references are attention-bound and unmeasured
		let nb = match &gb {
			Some(_) if p <= 2 * n => {
				((chunks as f64 * gpu_blk_share()).round() as usize).min(chunks - 1)
			}
			_ => 0,
		};
		#[cfg(not(feature = "vulkan"))]
		let nb = 0usize;
		self.gb_chunks = nb;
		#[cfg_attr(not(feature = "vulkan"), allow(unused_variables))]
		let w_ix: Option<[usize; 4]> = {
			let ix = |n: &str| self.wnames.iter().position(|w| w == n);
			match (ix("o.w"), ix("gu.w"), ix("d.w"), ix("qkv.w")) {
				(Some(a), Some(b), Some(c), Some(d)) => Some([a, b, c, d]),
				_ => None,
			}
		};
		let r = std::thread::scope(|sc| {
			sc.spawn(move || {
				let mut rec = rec;
				// (kind, chunk, batch)
				let mut pend: Vec<(usize, usize, npu::ze::Batch<'static>)> = Vec::new();
				let mut err: Option<String> = None;
				while let Ok(msg) = rx.recv() {
					match msg {
						Some((kind, l, c, cur, kv)) => {
							if err.is_some() {
								continue;
							}
							// at most MAX_LISTS recorded lists alive: the
							// descriptor heap does not hold a block's lists of
							// a long prefix (162 chunks at 10 references:
							// 0x70000002); the queue runs them in order, so
							// the oldest is the first done
							if pend.len() >= MAX_LISTS {
								let (k, c, mut bt) = pend.remove(0);
								if let Err(e) = bt.wait() {
									err = Some(e);
									continue;
								}
								if k == 1 {
									let _ = done_tx.send(Ev::Chunk(c));
								}
							}
							match record(rec.reborrow(), kind, l, c, c + 1, cur, p, kv)
								.and_then(|mut bt| bt.submit().map(|_| bt))
							{
								Ok(bt) => pend.push((kind, c, bt)),
								Err(e) => err = Some(e),
							}
						}
						None => {
							let mut r = err.take().map_or(Ok(()), Err);
							for (k, c, mut bt) in pend.drain(..) {
								match bt.wait() {
									Err(e) => r = r.and(Err(e)),
									Ok(_) if k == 1 && r.is_ok() => {
										let _ = done_tx.send(Ev::Chunk(c));
									}
									Ok(_) => {}
								}
							}
							if done_tx.send(Ev::End(r)).is_err() {
								break;
							}
						}
					}
				}
			});
			let (k8_on, v8_on) = attn_i8();
			let tw = attn_threads(n, p, k8_on);
			let mut k8 = quant::K8::default();
			let attn_check = std::env::var_os("QWEN_IMAGE_ATTN_CHECK").is_some();
			// the next block's keys packed during this block's wait
			let early_on = segs.is_none() && k8_on && bc.is_none() && kpack_early();
			if segs.is_none() {
				kmean.resize(layers, Vec::new());
			}
			let mut ahead: Option<usize> = None;
			let mut cur = 0;
			let mut l = 0;
			while l < layers {
				let kv = l % 2;
				let last = l + 1 == layers;
				// block l's K / V packed while its jobs ran
				let early = ahead.take() == Some(l);
				let t_b = std::time::Instant::now();
				if let Some(w) = qk.map(|q| &q[l]).filter(|_| !early) {
					// block l's K rows, raw from the NPU: scales, norm, RoPE
					let t = std::time::Instant::now();
					// SAFETY: block l's jobs are done, block l + 1's not
					// submitted: nothing else touches these rows
					let k = unsafe {
						std::slice::from_raw_parts_mut(
							(b.k[kv].ptr() as *mut u16).add(p * D),
							n * D,
						)
					};
					qk_finish_f16(k, &w.sk, &w.nk, &b.cosf, &b.sinf);
					t_attn += t.elapsed().as_secs_f64();
				}
				let kb = as_u16(b.k[kv].as_slice());
				let vb = as_u16(b.v[kv].as_slice());
				if segs.is_some() {
					pk.push(Kv::new(&kb[..n * D], kv8));
					pv.push(Kv::new(&vb[..n * D], kv8));
					if last {
						break;
					}
				}
				if k8_on && !early {
					// the block's keys (prefix + target) as int8, once per layer
					let t = std::time::Instant::now();
					k8.pack(&kb[..(p + n) * D], p + n, NH, HD);
					if v8_on {
						k8.pack_v(&vb[..(p + n) * D]);
					}
					t_attn += t.elapsed().as_secs_f64();
				}
				if k8_on && segs.is_none() {
					kmean[l] = k8.key_mean();
				}
				t_bound += t_b.elapsed().as_secs_f64();
				if nancheck {
					nancheck_rows(b, l, n, p, cur);
				}
				let mut bad = 0;
				#[cfg(feature = "vulkan")]
				if ng > 0 {
					let (r0, n) = (cg * S, p + n);
					let g = ga.as_mut().unwrap();
					// SAFETY: the iGPU writes rows r0 .. of `a`, which nothing
					// else touches until its jobs below
					let a = unsafe {
						std::slice::from_raw_parts(b.a.ptr() as *const u16, b.a.len() / 2)
					};
					g.begin(
						as_u16(b.q.as_slice()),
						r0,
						n - p - r0,
						kb,
						vb,
						n,
						a,
						r0,
						scale,
					)?;
				}
				for c in 0..cg {
					let (r0, r1) = (c * S, ((c + 1) * S).min(n));
					// a layer's first chunk, which the NPU waits for, on
					// every core
					let _w = quant::parallel::limit_width(if c == 0 { 0 } else { tw });
					let t = std::time::Instant::now();
					let q = &mut qf[..(r1 - r0) * D];
					widen(&as_u16(b.q.as_slice())[r0 * D..r1 * D], q);
					if let Some(w) = qk.map(|q| &q[l]) {
						let h = HD / 2;
						qk_finish_par(
							q,
							&w.sq,
							&w.nq,
							&b.cosf[r0 * h..r1 * h],
							&b.sinf[r0 * h..r1 * h],
						);
					}
					let o = &mut af[..(r1 - r0) * D];
					match segs {
						None if k8_on => {
							quant::attn_block_rows_k8(
								q,
								&kb[..(p + n) * D],
								&k8,
								&vb[..(p + n) * D],
								r1 - r0,
								p + r0,
								NH,
								HD,
								scale,
								p + n,
								o,
							);
							if attn_check && c == 0 {
								let mut want = vec![0f32; o.len()];
								quant::attn_block_rows(
									q,
									&kb[..(p + n) * D],
									&vb[..(p + n) * D],
									r1 - r0,
									p + r0,
									NH,
									NH,
									HD,
									scale,
									p + n,
									&mut want,
								);
								let (mut e, mut w) = (0f64, 0f64);
								for (a, b) in want.iter().zip(o.iter()) {
									e += ((a - b) * (a - b)) as f64;
									w += (a * a) as f64;
								}
								eprintln!("[attn8] block {l}: rel. error {:.3e}", (e / w).sqrt());
							}
						}
						None => quant::attn_block_rows(
							q,
							&kb[..(p + n) * D],
							&vb[..(p + n) * D],
							r1 - r0,
							p + r0,
							NH,
							NH,
							HD,
							scale,
							p + n,
							o,
						),
						Some(segs) => {
							for &(s0, s1, text) in segs {
								let (a0, a1) = (s0.max(r0), s1.min(r1));
								if a0 >= a1 {
									continue;
								}
								let (qs, os) = (
									&q[(a0 - r0) * D..(a1 - r0) * D],
									&mut o[(a0 - r0) * D..(a1 - r0) * D],
								);
								if text {
									let (ks, vs) = (&kb[..a1 * D], &vb[..a1 * D]);
									quant::attn_causal(
										qs,
										ks,
										vs,
										a1 - a0,
										a0,
										NH,
										NH,
										HD,
										scale,
										os,
									);
								} else {
									let (ks, vs) = (&kb[..s1 * D], &vb[..s1 * D]);
									if k8_on {
										quant::attn_block_rows_k8(
											qs,
											ks,
											&k8,
											vs,
											a1 - a0,
											a0,
											NH,
											HD,
											scale,
											s1,
											os,
										);
									} else {
										quant::attn_block_rows(
											qs,
											ks,
											vs,
											a1 - a0,
											a0,
											NH,
											NH,
											HD,
											scale,
											s1,
											os,
										);
									}
								}
							}
						}
					}
					// SAFETY: rows r0 .. r1 of `a` are read only by the job
					// submitted below
					let a = unsafe {
						std::slice::from_raw_parts_mut(
							(b.a.ptr() as *mut u8).add(r0 * D * 2),
							(r1 - r0) * D * 2,
						)
					};
					narrow(o, a);
					t_attn += t.elapsed().as_secs_f64();
					if c == 0 {
						t_bound += t.elapsed().as_secs_f64();
					}
					if nancheck {
						bad += o.iter().filter(|f| !f.is_finite()).count();
					}
					let job = if last {
						(2, layers, c, cur, 0)
					} else {
						(1, l + 1, c, cur, 1 - kv)
					};
					let on_gpu = !last && gpu_chunk(c, nb, cg);
					if !on_gpu {
						tx.send(Some(job)).map_err(|_| gone())?;
					}
					#[cfg(feature = "vulkan")]
					if on_gpu {
						// the iGPU's chunk, once its attention is done
						let wi = w_ix.unwrap();
						let (wp, wq) = (&rec_w[l], &rec_w[l + 1]);
						let j = offload::DitMid {
							l: l + 1,
							x: b.x[cur].as_slice(),
							a: b.a.as_slice(),
							xo: b.x[1 - cur].as_slice(),
							q: b.q.as_slice(),
							k: b.k[1 - kv].as_slice(),
							v: b.v[1 - kv].as_slice(),
							r0,
							r1,
							krow: p + r0,
							w_o: wp[wi[0]].as_slice(),
							w_gu: wp[wi[1]].as_slice(),
							w_d: wp[wi[2]].as_slice(),
							w_qkv: wq[wi[3]].as_slice(),
						};
						gb.as_mut().unwrap().mid_begin(&j)?;
					}
				}
				// block l + 1's prefix K / V and, at the last step's means, its
				// int8 keys chunk by chunk as the jobs finish
				let pre = early_on && !last && !kmean[l + 1].is_empty();
				let mut fin_done = vec![false; if pre { chunks } else { 0 }];
				let mut t_fin = 0.0;
				let kv1 = 1 - kv;
				let fin = |c: usize, k8: &mut quant::K8| -> f64 {
					let t = std::time::Instant::now();
					let (r0, r1) = (c * S, ((c + 1) * S).min(n));
					if let Some(w) = qk.map(|q| &q[l + 1]) {
						let h = HD / 2;
						// SAFETY: chunk c's job is done; no other job touches
						// its K rows, nothing reads them until block l + 1
						let k = unsafe {
							std::slice::from_raw_parts_mut(
								(b.k[kv1].ptr() as *mut u16).add((p + r0) * D),
								(r1 - r0) * D,
							)
						};
						qk_finish_f16(k, &w.sk, &w.nk, &b.cosf[r0 * h..], &b.sinf[r0 * h..]);
					}
					k8.pack_rows(as_u16(b.k[kv1].as_slice()), p + r0, p + r1);
					if v8_on {
						k8.pack_v_rows(as_u16(b.v[kv1].as_slice()), p + r0, p + r1);
					}
					t.elapsed().as_secs_f64()
				};
				if segs.is_none() && !last {
					let t = std::time::Instant::now();
					// Packed early, block l + 1's prefix keys (and values, int8
					// too) are packed from their kept f16 rows, and the int8
					// attention reads nothing else of them: no copy into the
					// pass' K / V (2 x prefix rows x 8 KiB per block and step).
					// The iGPU's attention and QWEN_IMAGE_ATTN_CHECK read the
					// half rows.
					let (sk, sv) = (pk[l + 1].f16(), pv[l + 1].f16());
					let direct = pre && ng == 0 && !attn_check;
					let copy_k = !(direct && sk.is_some());
					let copy_v = !(direct && v8_on && sv.is_some());
					// SAFETY: block l + 1's jobs write K / V [(l + 1) % 2]'s
					// target rows only; nothing reads them before block l + 1
					unsafe { prefix_kv_parts(b, pk, pv, l + 1, p, copy_k, copy_v) };
					if pre {
						k8.begin(p + n, NH, HD, &kmean[l + 1], v8_on);
						k8.pack_rows(sk.unwrap_or(as_u16(b.k[kv1].as_slice())), 0, p);
						if v8_on {
							k8.pack_v_rows(sv.unwrap_or(as_u16(b.v[kv1].as_slice())), 0, p);
						}
					}
					t_attn += t.elapsed().as_secs_f64();
				}
				#[cfg(feature = "vulkan")]
				if !last && nb > 0 {
					let t = std::time::Instant::now();
					gb.as_mut().unwrap().end()?;
					t_gpu += t.elapsed().as_secs_f64();
					if pre {
						for c in (0..cg).filter(|&c| gpu_chunk(c, nb, cg)) {
							t_fin += fin(c, &mut k8);
							fin_done[c] = true;
						}
					}
				}
				if ng > 0 {
					let t = std::time::Instant::now();
					#[cfg(feature = "vulkan")]
					ga.as_mut().unwrap().end()?;
					t_gpu += t.elapsed().as_secs_f64();
					for c in cg..chunks {
						let job = if last {
							(2, layers, c, cur, 0)
						} else {
							(1, l + 1, c, cur, 1 - kv)
						};
						tx.send(Some(job)).map_err(|_| gone())?;
					}
				}
				if nancheck {
					eprintln!("  [nan] block {l}: attn {bad}");
				}
				let t = std::time::Instant::now();
				tx.send(None).map_err(|_| gone())?;
				let t_fin0 = t_fin;
				loop {
					match done_rx.recv().map_err(|_| gone())? {
						Ev::Chunk(c) => {
							if pre && !fin_done[c] {
								t_fin += fin(c, &mut k8);
								fin_done[c] = true;
							}
						}
						Ev::End(r) => break r?,
					}
				}
				t_wait += t.elapsed().as_secs_f64() - (t_fin - t_fin0);
				if pre {
					// every job is done: any chunk not reported
					for c in 0..chunks {
						if !fin_done[c] {
							t_fin += fin(c, &mut k8);
						}
					}
					ahead = Some(l + 1);
				}
				t_attn += t_fin;
				cur = 1 - cur;
				let mut next = l + 1;
				if let Some(c) = bc.as_deref_mut() {
					let to = layers - c.back;
					// SAFETY: the block's jobs are done, the next block's not
					// submitted: nothing else touches the residual
					let x = unsafe {
						std::slice::from_raw_parts_mut(b.x[cur].ptr() as *mut u16, n * D)
					};
					if next == c.front {
						let t = std::time::Instant::now();
						let skip = c.at_front(x);
						t_attn += t.elapsed().as_secs_f64();
						if skip {
							// resume at block `to`: its prefix K / V, then its
							// dit_pre on the patched residual
							let t = std::time::Instant::now();
							// SAFETY: K / V [to % 2] are not in use (all jobs done)
							unsafe { prefix_kv(b, pk, pv, to, p) };
							for ch in 0..chunks {
								tx.send(Some((0, to, ch, cur, to % 2)))
									.map_err(|_| gone())?;
							}
							tx.send(None).map_err(|_| gone())?;
							loop {
								if let Ev::End(r) = done_rx.recv().map_err(|_| gone())? {
									break r?;
								}
							}
							t_wait += t.elapsed().as_secs_f64();
							next = to;
						}
					} else if next == to {
						let t = std::time::Instant::now();
						c.at_back(x);
						t_attn += t.elapsed().as_secs_f64();
					}
				}
				l = next;
			}
			drop(tx);
			Ok::<usize, String>(cur)
		});
		self.t_attn += t_attn;
		self.t_npu += t_wait;
		self.t_gpu += t_gpu;
		self.t_bound += t_bound;
		self.kmean = kmean;
		#[cfg(feature = "vulkan")]
		if segs.is_none() {
			if let Some(g) = ga {
				self.ga = Some(g);
			}
			if let Some(g) = gb {
				self.gb = Some(g);
			}
		}
		r
	}

	/// The prefix's K / V of block `l` in front of `K[l % 2]` / `V[l % 2]`.
	fn prefix_kv(&mut self, l: usize, p: usize) {
		let b = self.bufs.as_ref().unwrap();
		// SAFETY: no job runs
		unsafe { prefix_kv(b, &self.pk, &self.pv, l, p) };
	}

	/// Unload the graphs, keeping the weights: they hold driver memory that
	/// every NPU user of the process shares, and a chat engine on the NPU
	/// does not fit beside them (`0x70000002`). The next pass loads them
	/// again.
	pub fn unload_graphs(&mut self) {
		self.graphs.clear();
	}

	/// Load the graphs if [`Self::unload_graphs`] dropped them.
	fn load_graphs(&mut self) -> Result<(), String> {
		if !self.graphs.is_empty() {
			return Ok(());
		}
		for (k, kind) in KINDS.iter().enumerate() {
			let g = load_graph(self.npu, &self.gdir, kind)?;
			if g.args.len() != self.io[k].len() {
				self.graphs.clear();
				return Err(format!(
					"{kind}: the graph's arguments changed since the load"
				));
			}
			self.graphs.push(g);
		}
		Ok(())
	}

	/// Free the per-image buffers (the weights stay).
	pub fn release_buffers(&mut self) {
		self.forget_gpu();
		self.bufs = None;
		self.reset();
	}

	/// Drop the iGPU's imports of the pass buffers (before they are freed).
	fn forget_gpu(&mut self) {
		#[cfg(feature = "vulkan")]
		if let Some(g) = self.ga.as_mut() {
			g.forget();
		}
		#[cfg(feature = "vulkan")]
		if let Some(g) = self.gb.as_mut() {
			g.forget_rows();
		}
	}

	/// The weight names the graphs bind (for the log).
	pub fn weight_names(&self) -> &[String] {
		&self.wnames
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// `qk_finish` on permuted heads (evens, then odds) equals the
	/// checkpoint's form: row scale, RMS norm, weight, RoPE of adjacent
	/// pairs `(2i, 2i + 1)`.
	#[test]
	fn qk_finish_matches_adjacent_pairs() {
		let perm = qi::head_perm();
		let half = HD / 2;
		let rnd = |i: usize| ((i * 2654435761) % 1000) as f32 / 500.0 - 1.0;
		let x: Vec<f32> = (0..D).map(rnd).collect();
		let s: Vec<f32> = (0..D).map(|i| 0.5 + rnd(i + 7).abs()).collect();
		let w: Vec<f32> = (0..HD).map(|i| 1.0 + 0.1 * rnd(i + 3)).collect();
		let ang: Vec<f32> = (0..half).map(|i| 0.37 * i as f32).collect();
		let (cos, sin): (Vec<f32>, Vec<f32>) = ang.iter().map(|a| (a.cos(), a.sin())).unzip();
		// the checkpoint's order
		let mut want = vec![0f32; D];
		for h in 0..NH {
			let t: Vec<f32> = (0..HD).map(|j| x[h * HD + j] * s[h * HD + j]).collect();
			let r =
				1.0 / (t.iter().map(|v| v * v).sum::<f32>() / HD as f32 + qi::EPS as f32).sqrt();
			let n: Vec<f32> = (0..HD).map(|j| t[j] * r * w[j]).collect();
			for i in 0..half {
				let (a, b) = (n[2 * i], n[2 * i + 1]);
				want[h * HD + 2 * i] = a * cos[i] - b * sin[i];
				want[h * HD + 2 * i + 1] = b * cos[i] + a * sin[i];
			}
		}
		// permuted rows, scales and norm weight, as the files store them
		let pm = |v: &[f32]| -> Vec<f32> {
			(0..v.len())
				.map(|i| v[(i / HD) * HD + perm[i % HD]])
				.collect()
		};
		let (mut got, sp, wp) = (pm(&x), pm(&s), pm(&w));
		qk_finish(&mut got, &sp, &wp, &cos, &sin);
		let want = pm(&want);
		let e = got
			.iter()
			.zip(&want)
			.fold(0f32, |m, (a, b)| m.max((a - b).abs()));
		assert!(e < 1e-5, "max diff {e}");
	}

	#[test]
	fn image_npu_parses() {
		let d = std::env::temp_dir().join(format!("qwen-rs-imagenpu-{}", std::process::id()));
		std::fs::create_dir_all(&d).unwrap();
		let f = d.join("image.npu");
		std::fs::write(
			&f,
			"# a set\narch = qwen-image\ncheckpoint = C:/ck\ngraphs = graphs/dit\nweights = w\n",
		)
		.unwrap();
		assert!(NpuCfg::is_image_npu(&f));
		let c = NpuCfg::parse(&f).unwrap();
		assert_eq!(c.graphs, d.join("graphs/dit"));
		assert_eq!(c.weights, d.join("w"));
		std::fs::write(&f, "arch = flash\ngraphs = g\n").unwrap();
		assert!(!NpuCfg::is_image_npu(&f));
		assert!(NpuCfg::parse(&f).is_err());
		let _ = std::fs::remove_dir_all(&d);
	}
}
