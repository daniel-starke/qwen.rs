//! The transformer on the NPU (`image.npu` with `arch = h3-image`, built by
//! `cli H3_DIR --npu-build OUT`; the graphs: `npubuild::h3img`).
//!
//! A forward runs the 50 blocks layer by layer over every row of the
//! packed sequence (graph `pre` for block 0, `mid` for blocks 1-49: block
//! l - 1 after its attention plus block l up to its attention, `post`
//! after block 49). Once every chunk of block l has its q / k / v, the
//! cores run the attention of chunk c and submit the next graph's chunk c
//! at once, so the NPU runs chunk c while the cores do chunk c + 1. Appends
//! go through a recorder thread (an append waits for the running job).
//! A share of each block's `mid` chunks runs on the iGPU instead
//! (`offload::H3Blk`, the same math on the same buffers and weights).
//!
//! **Small chunks at a block's edges.** All of block l's keys must exist
//! before any of its attention runs, so the NPU idles during a block's
//! first chunk of attention and the cores during its last job; a block
//! starts and ends with [`S_SMALL`]-row chunks (the `*64` graphs) to keep
//! both short.
//!
//! **Rows are ordered by modulation entry.** A row's modulation is its
//! `(timestep, modality)` entry: text rows, image rows at the step's
//! timestep (the target and the references' vision rows in the text) and
//! the references (held at `REF_T`). Full attention does not depend on the
//! order of the keys, so the rows are grouped by entry, each group padded
//! to whole chunks, and a job binds its group's modulation. The
//! pad rows run through the graphs (all zeros, harmless) but are no keys:
//! the cores finish each chunk's valid K rows (row scales, head norm,
//! RoPE) into a compact copy and pack the valid V rows from where the NPU
//! wrote them.
//!
//! The residual lives in NPU host buffers as `x_l / c_l`
//! (`npubuild::h3img`: H3's residual exceeds f16), and the result is the
//! last block's scaled residual, which `norm_out` (an RMSNorm) takes as it
//! is.

use super::config::{DitCfg, MODALITIES, REF_T};
use super::dit::Rope;
use super::layout::Layout;
use crate::npuhost::{as_u16, narrow, npu, widen};
use npu::lock::{NpuLock, SharedNpuLock};
use npu::ze::{Graph, HostBuf, Npu};
use npubuild::h3img::{self as h3, DShape, MPARTS, S, S_SMALL};
use std::path::{Path, PathBuf};

/// `image.npu` with `arch = h3-image`: `key = value` lines.
#[derive(Clone, Debug)]
pub struct NpuCfg {
	/// The checkpoint (text encoder, VAE, the transformer's small weights).
	pub checkpoint: PathBuf,
	/// The transformer's compiled graphs (`pre.blob`, `mid.blob`,
	/// `post.blob`).
	pub graphs: PathBuf,
	/// The blocks' files (`h3dit{L}.<name>.in`, `h3scale.f32`, `h3ts.f32`).
	pub weights: PathBuf,
	/// The VAE decoder's graphs (`dec.blob`, `out.blob`).
	pub vae: Option<PathBuf>,
	/// The text encoder's files (`te{L}.*.in`, `te.scale.f32`) and graphs
	/// (`npubuild::h3te`).
	pub te: Option<PathBuf>,
	pub te_graphs: Option<PathBuf>,
}

fn arch_of(path: &Path) -> Option<String> {
	if !path.is_file() {
		return None;
	}
	let t = std::fs::read_to_string(path).ok()?;
	t.lines().find_map(|l| {
		l.split_once('=')
			.filter(|(k, _)| k.trim() == "arch")
			.map(|(_, v)| v.trim().to_string())
	})
}

impl NpuCfg {
	/// True when `path` is a file whose `arch` is `h3-image`.
	pub fn is_h3_npu(path: &Path) -> bool {
		arch_of(path).as_deref() == Some("h3-image")
	}

	pub fn parse(path: &Path) -> Result<NpuCfg, String> {
		let t = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
		let base = path.parent().unwrap_or(Path::new("."));
		let (mut ck, mut gr, mut w, mut vae, mut arch) = (None, None, None, None, None);
		let (mut te, mut teg) = (None, None);
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
				"te" => te = Some(p()),
				"te_graphs" => teg = Some(p()),
				_ => return Err(format!("{}: unknown key {k}", path.display())),
			}
		}
		if arch.as_deref() != Some("h3-image") {
			return Err(format!("{}: arch is not h3-image", path.display()));
		}
		let need = |v: Option<PathBuf>, k: &str| v.ok_or(format!("{}: no {k}", path.display()));
		Ok(NpuCfg {
			checkpoint: need(ck, "checkpoint")?,
			graphs: need(gr, "graphs")?,
			weights: need(w, "weights")?,
			vae,
			te,
			te_graphs: teg,
		})
	}
}

/// The graphs: [`S`] rows, then [`S_SMALL`] rows (optional: a set built
/// without them runs [`S`]-row chunks only).
const KINDS: [&str; 6] = ["pre", "mid", "post", "pre64", "mid64", "post64"];

/// Recorded lists alive at once (the driver's descriptor heap).
pub(super) const MAX_LISTS: usize = 32;

pub(super) fn load_graph(
	npu: &'static Npu,
	dir: &Path,
	kind: &str,
) -> Result<Graph<'static>, String> {
	let f = dir.join(format!("{kind}.blob"));
	let b = std::fs::read(&f).map_err(|e| format!("{}: {e}", f.display()))?;
	let mut g = npu.load(&b, 0)?;
	g.initialize()?;
	Ok(g)
}

/// What a graph argument is bound to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Io {
	X,
	Xo,
	A,
	Q,
	K,
	V,
	/// A modulation part of the block after (true) or before the cut.
	M(bool, usize),
	/// A weight of the block after (true) or before the cut.
	W(bool, usize),
}

/// A block's q / k finish on the cores: the row scales (`qkv.s`) and the
/// head norm weights.
pub(super) struct QkW {
	pub(super) sq: Vec<f32>,
	pub(super) sk: Vec<f32>,
	pub(super) nq: Vec<f32>,
	pub(super) nk: Vec<f32>,
}

/// `x` rows (`[rows][heads * hd]`) raw q or k -> `rms(x * s) * w`, then
/// the rotate-half RoPE over the first `2 half` channels of each head with
/// `cos` / `sin` `[rows][half]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn qk_finish(
	x: &mut [f32],
	s: &[f32],
	w: &[f32],
	cos: &[f32],
	sin: &[f32],
	half: usize,
	hd: usize,
	eps: f32,
) {
	let width = s.len();
	for (r, row) in x.chunks_exact_mut(width).enumerate() {
		let (c, sn) = (
			&cos[r * half..(r + 1) * half],
			&sin[r * half..(r + 1) * half],
		);
		for (h, t) in row.chunks_exact_mut(hd).enumerate() {
			let sh = &s[h * hd..(h + 1) * hd];
			let mut ss = 0f32;
			for (v, &f) in t.iter_mut().zip(sh) {
				*v *= f;
				ss += *v * *v;
			}
			let inv = 1.0 / (ss / hd as f32 + eps).sqrt();
			t.iter_mut().zip(w).for_each(|(v, &g)| *v *= inv * g);
			for i in 0..half {
				let (a, b) = (t[i], t[i + half]);
				t[i] = a * c[i] - b * sn[i];
				t[i + half] = b * c[i] + a * sn[i];
			}
		}
	}
}

/// One job's rows.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Chunk {
	/// Its modulation group.
	g: usize,
	/// Valid rows (from the chunk's start).
	n: usize,
	/// The compact (key) index of its first row.
	k0: usize,
	/// Its first padded row.
	r0: usize,
	/// [`S_SMALL`] rows instead of [`S`].
	small: bool,
}

impl Chunk {
	fn rows(&self) -> usize {
		if self.small {
			S_SMALL
		} else {
			S
		}
	}
}

/// `QWEN_H3_SMALL=0`: [`S`]-row chunks only (default: small chunks at a
/// block's start and end when the set has the small graphs).
fn small_chunks() -> bool {
	std::env::var("QWEN_H3_SMALL").map_or(true, |v| v != "0")
}

/// Small chunks a block starts with and ends with: the NPU idles during
/// the first chunk's attention and the cores during the last chunk's job,
/// both shorter for a small chunk.
const SMALL_FIRST: usize = 1;
const SMALL_LAST: usize = 2;

/// The rows of one job, ordered by modulation entry (module docs).
struct Plan {
	/// Padded rows (whole chunks).
	rows: usize,
	/// Valid rows (keys).
	nv: usize,
	/// The chunks in the order a block runs them.
	chunks: Vec<Chunk>,
	/// Per group: (reference rows, modality tag).
	groups: Vec<(bool, u8)>,
	/// The original row of each padded row (`usize::MAX`: padding).
	orig: Vec<usize>,
	/// `cos` / `sin` `[rows][half]` of the padded rows (padding: 1, 0).
	cos: Vec<f32>,
	sin: Vec<f32>,
	/// Padded positions of the target rows, in order.
	target: Vec<usize>,
}

impl Plan {
	/// With `small`, each group is padded to whole [`S_SMALL`]-row chunks
	/// (its rest past whole [`S`]-row chunks in small ones), and a block
	/// runs [`SMALL_FIRST`] small chunks first and [`SMALL_LAST`] last
	/// (splitting [`S`]-row chunks when the groups leave fewer).
	fn new(lay: &Layout, rope: &Rope, small: bool) -> Plan {
		let half = rope.half;
		let r0 = lay.text..lay.text + lay.refs;
		let key = |r: usize| (r0.contains(&r), lay.tags[r]);
		let mut groups: Vec<(bool, u8)> = (0..lay.len()).map(key).collect();
		groups.sort_unstable();
		groups.dedup();
		// per group its pieces: (rows, small)
		let mut pieces: Vec<Vec<(Vec<usize>, bool)>> = Vec::new();
		for &g in &groups {
			let rows: Vec<usize> = (0..lay.len()).filter(|&r| key(r) == g).collect();
			let mut v = Vec::new();
			let whole = if small {
				rows.len() / S * S
			} else {
				rows.len()
			};
			for part in rows[..whole].chunks(S) {
				v.push((part.to_vec(), false));
			}
			for part in rows[whole..].chunks(S_SMALL) {
				v.push((part.to_vec(), true));
			}
			pieces.push(v);
		}
		if small {
			// split S-row chunks (the largest group's first) until there are
			// enough small ones
			let want = SMALL_FIRST + SMALL_LAST;
			while pieces.iter().flatten().filter(|p| p.1).count() < want {
				let Some(gi) = (0..pieces.len())
					.filter(|&g| pieces[g].iter().any(|p| !p.1))
					.max_by_key(|&g| pieces[g].len())
				else {
					break;
				};
				let i = pieces[gi].iter().position(|p| !p.1).unwrap();
				let (rows, _) = pieces[gi].remove(i);
				for (j, part) in rows.chunks(S_SMALL).enumerate() {
					pieces[gi].insert(i + j, (part.to_vec(), true));
				}
			}
		}
		let mut orig = Vec::new();
		let mut chunks = Vec::new();
		let mut nv = 0;
		for (gi, v) in pieces.iter().enumerate() {
			for (part, sm) in v {
				let c = Chunk {
					g: gi,
					n: part.len(),
					k0: nv,
					r0: orig.len(),
					small: *sm,
				};
				nv += part.len();
				orig.extend_from_slice(part);
				orig.resize(c.r0 + c.rows(), usize::MAX);
				chunks.push(c);
			}
		}
		// the order a block runs them: small first and last
		let (sm, big): (Vec<Chunk>, Vec<Chunk>) = chunks.iter().partition(|c| c.small);
		let last = sm.len().saturating_sub(SMALL_FIRST).min(SMALL_LAST);
		let (head, tail) = sm.split_at(sm.len() - last);
		let chunks: Vec<Chunk> = head.iter().chain(&big).chain(tail).copied().collect();
		let rows = orig.len();
		let mut cos = vec![1f32; rows * half];
		let mut sin = vec![0f32; rows * half];
		for (p, &r) in orig.iter().enumerate() {
			if r != usize::MAX {
				cos[p * half..(p + 1) * half].copy_from_slice(&rope.cos[r * half..(r + 1) * half]);
				sin[p * half..(p + 1) * half].copy_from_slice(&rope.sin[r * half..(r + 1) * half]);
			}
		}
		let mut at = vec![usize::MAX; lay.len()];
		for (p, &r) in orig.iter().enumerate() {
			if r != usize::MAX {
				at[r] = p;
			}
		}
		let target = at[lay.target0()..].to_vec();
		Plan {
			rows,
			nv,
			chunks,
			groups,
			orig,
			cos,
			sin,
			target,
		}
	}
}

/// The NPU host buffers of a job's rows.
struct Bufs {
	rows: usize,
	/// The scaled residual, ping-pong.
	x: [HostBuf<'static>; 2],
	a: HostBuf<'static>,
	q: HostBuf<'static>,
	/// K / V as the NPU writes them (padded rows, raw).
	k: HostBuf<'static>,
	v: HostBuf<'static>,
}

pub struct NpuDit {
	/// The iGPU's `h3/mid` jobs ([`gpu_blk`]); first, so it drops (with its
	/// imports of the NPU buffers) before them.
	#[cfg(feature = "vulkan")]
	gb: Option<offload::H3Blk>,
	/// The discrete card's share of each block's attention ([`gpu_attn`]).
	#[cfg(feature = "vulkan")]
	ga: Option<offload::H3Attn>,
	/// Seconds per attention unit ([`attn_units`]) of the cores and of the
	/// card, learned and frozen (`quant::calib`): they size the card's share.
	attn_fit: quant::calib::Fit,
	npu: &'static Npu,
	shape: DShape,
	qk_eps: f32,
	graphs: Vec<Graph<'static>>,
	gdir: PathBuf,
	io: Vec<Vec<Io>>,
	/// Per block its weight buffers, by [`Self::wnames`].
	w: Vec<Vec<HostBuf<'static>>>,
	wnames: Vec<String>,
	/// Per block its modulation entries (`mod.in`).
	m: Vec<HostBuf<'static>>,
	qk: Vec<QkW>,
	/// The stored scale of the blocks' input, and of each block's output.
	c_in: f32,
	c: Vec<f32>,
	/// Run only the first this many blocks (checks against the reference).
	pub limit: Option<usize>,
	/// The modulation entries' timesteps.
	ts: Vec<f32>,
	bufs: Option<Bufs>,
	plan: Option<Plan>,
	/// Per block the previous step's key means (the int8 keys' centers).
	kmean: Vec<Vec<f32>>,
	/// Seconds waiting for the NPU, in the attention and K / V finishing,
	/// and at block boundaries (the NPU idle) since the last report.
	pub t_npu: f64,
	pub t_attn: f64,
	pub t_bound: f64,
	/// Seconds waiting for the iGPU since the last report, and its chunks
	/// of the last step.
	pub t_gpu: f64,
	pub gpu_chunks: usize,
	/// Seconds waiting for the discrete card since the last report, and its
	/// attention chunks of a block in the last step.
	pub t_gtx: f64,
	pub gtx_chunks: usize,
	_lock: SharedNpuLock,
}

/// `QWEN_H3_GPU_BLK`: the iGPU's share of each block's [`S`]-row chunks
/// (`h3/mid` jobs, `offload::H3Blk`), default 0.25 (4 of 16 at 4 MP: with
/// more the iGPU finishes each block last and the NPU idles); `0` = the NPU
/// runs all of them.
#[cfg(feature = "vulkan")]
fn gpu_blk_share() -> f64 {
	std::env::var("QWEN_H3_GPU_BLK")
		.ok()
		.and_then(|v| v.parse().ok())
		.unwrap_or(0.25)
}

/// The weights the iGPU's jobs read: `o.w`, `gu.w`, `d.w`, `qkv.w`, then
/// the vectors `o.pre`, `gu.pre`, `qkv.pre`, `gu.sg`.
const GPU_W: [&str; 8] = [
	"o.w", "gu.w", "d.w", "qkv.w", "o.pre", "gu.pre", "qkv.pre", "gu.sg",
];

/// The iGPU for [`gpu_blk_share`], with every block's vectors (a set with
/// static activations for `o`, `gu`, `qkv` and room in the import budget).
#[cfg(feature = "vulkan")]
fn gpu_blk(w: &[Vec<HostBuf<'static>>], wnames: &[String], s: &DShape) -> Option<offload::H3Blk> {
	if gpu_blk_share() <= 0.0 {
		return None;
	}
	let ix: Vec<usize> = GPU_W
		.iter()
		.filter_map(|n| wnames.iter().position(|w| w == n))
		.collect();
	if ix.len() != GPU_W.len() || w[0][ix[0]].len() != s.d * s.inner() {
		eprintln!("[h3] the transformer's blocks all on the NPU: the set's weights are not the iGPU kernels' form");
		return None;
	}
	let need: usize = w
		.iter()
		.flat_map(|b| ix[..4].iter().map(|&i| b[i].len()))
		.sum();
	let (have, cap) = (offload::host_imported(), offload::host_import_budget());
	if have + need > cap {
		eprintln!(
			"[h3] the transformer's blocks all on the NPU: the iGPU holds {:.1} GB of imports, its share would add {:.1} GB (budget {:.1} GB, QWEN_GPU_IMPORT_MB)",
			have as f64 / 1e9,
			need as f64 / 1e9,
			cap as f64 / 1e9
		);
		return None;
	}
	let mut g = match offload::H3Blk::new(Some("intel"), w.len(), s.d, s.inner(), s.kv, s.ff) {
		Ok(g) => g,
		Err(e) => {
			eprintln!("[h3] no iGPU for the transformer's blocks ({e}): the NPU runs all of them");
			return None;
		}
	};
	for (l, b) in w.iter().enumerate() {
		let v: Vec<Vec<f32>> = ix[4..]
			.iter()
			.map(|&i| npubuild::qimage::f16f(b[i].as_slice()))
			.collect();
		g.set_block(l, [&v[0], &v[1], &v[2], &v[3]]);
	}
	Some(g)
}

/// The iGPU's chunks of a block: `nb` of the [`S`]-row chunks, spread from
/// the second on (the NPU starts with the block's first chunks).
#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
fn gpu_chunks(chunks: &[Chunk], nb: usize) -> Vec<bool> {
	let big: Vec<usize> = (0..chunks.len()).filter(|&c| !chunks[c].small).collect();
	let mut on = vec![false; chunks.len()];
	let nb = nb.min(big.len().saturating_sub(1));
	for i in 0..nb {
		on[big[1 + i * (big.len() - 1) / nb]] = true;
	}
	on
}

#[cfg(feature = "vulkan")]
type Ga = offload::H3Attn;
#[cfg(not(feature = "vulkan"))]
type Ga = ();

/// Query rows of one submission to the discrete card (two [`S`]-row chunks).
#[cfg(feature = "vulkan")]
const GTX_ROWS: usize = 2 * S;

/// The attention of `rows` query rows over `nk` keys, `inner` wide, in
/// seconds at 1 TFLOP/s: the unit [`NpuDit::attn_fit`] measures in.
fn attn_units(rows: usize, nk: usize, inner: usize) -> f64 {
	(rows * nk * inner * 4) as f64 * 1e-12
}

/// The fit of the attention's seconds per unit: the cores (with the iGPU
/// and the NPU busy beside them), then the card.
const ATTN_FIT: (&str, [f64; 2]) = ("h3-attn", [1.0, 0.69]);
/// Blocks a fit takes (about three 4 MP steps).
const ATTN_FIT_NEED: u64 = 150;

/// The discrete card for the attention, unless `QWEN_H3_ATTN_GPU_SHARE=0`
/// or there is none (`QWEN_H3_ATTN_GPU`: the adapter, default the first
/// discrete one).
#[cfg(feature = "vulkan")]
fn gpu_attn(heads: usize) -> Option<offload::H3Attn> {
	if attn_share_env() == Some(0.0) {
		return None;
	}
	let pick = std::env::var("QWEN_H3_ATTN_GPU").ok();
	match offload::H3Attn::new(pick.as_deref(), heads, GTX_ROWS) {
		Ok(g) => Some(g),
		Err(e) => {
			eprintln!("[h3] no discrete card for the attention ({e}): the cores run all of it");
			None
		}
	}
}

/// `QWEN_H3_ATTN_GPU_SHARE`: the card's share of each block's query rows
/// (default: from [`ATTN_FIT`], the share that ends both together).
fn attn_share_env() -> Option<f64> {
	std::env::var("QWEN_H3_ATTN_GPU_SHARE")
		.ok()
		.and_then(|v| v.trim().parse::<f64>().ok())
		.map(|v| v.clamp(0.0, 1.0))
}

/// The card's chunks of a block: [`S`]-row chunks from the back (the cores
/// run the block's first chunk and its small last ones), as near `share`
/// of the valid rows as whole chunks get.
#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
fn gtx_chunks(chunks: &[Chunk], share: f64) -> Vec<bool> {
	let mut on = vec![false; chunks.len()];
	let total: usize = chunks.iter().map(|c| c.n).sum();
	let target = share * total as f64;
	let mut got = 0usize;
	for c in (1..chunks.len()).rev().filter(|&c| !chunks[c].small) {
		let n = chunks[c].n;
		if (got + n) as f64 - target > target - got as f64 {
			break;
		}
		on[c] = true;
		got += n;
	}
	on
}

/// The card's batches: its chunks in order, up to [`GTX_ROWS`] valid rows
/// a batch.
#[cfg(feature = "vulkan")]
fn gtx_batches(chunks: &[Chunk], on: &[bool]) -> Vec<Vec<usize>> {
	let mut v: Vec<Vec<usize>> = Vec::new();
	let mut rows = 0;
	for c in (0..chunks.len()).filter(|&c| on[c]) {
		let n = chunks[c].n;
		match v.last_mut() {
			Some(b) if rows + n <= GTX_ROWS => {
				b.push(c);
				rows += n;
			}
			_ => {
				v.push(vec![c]);
				rows = n;
			}
		}
	}
	v
}

/// A block's attention batches on the discrete card.
#[cfg(feature = "vulkan")]
struct GtxRun<'a> {
	g: &'a mut offload::H3Attn,
	/// Batches not submitted yet, in order.
	todo: std::collections::VecDeque<Vec<usize>>,
	/// Per slot its batch in flight, the submission's sequence number and
	/// attention units.
	slot: Vec<Option<(Vec<usize>, usize, f64)>>,
	seq: usize,
	/// The first submission takes the block's K / V up.
	upload: bool,
	/// Card seconds and attention units of the finished batches.
	secs: f64,
	units: f64,
}

#[cfg(feature = "vulkan")]
impl<'a> GtxRun<'a> {
	fn new(g: &'a mut offload::H3Attn, batches: &[Vec<usize>]) -> GtxRun<'a> {
		GtxRun {
			g,
			todo: batches.iter().cloned().collect(),
			slot: vec![None; offload::H3_ATTN_SLOTS],
			seq: 0,
			upload: true,
			secs: 0.0,
			units: 0.0,
		}
	}

	fn pending(&self) -> bool {
		!self.todo.is_empty() || self.slot.iter().any(Option::is_some)
	}

	/// Submit batches into the idle slots; `prep(batch, q)` writes the
	/// batch's finished query rows into `q` and returns their number.
	fn fill(
		&mut self,
		nk: usize,
		inner: usize,
		scale: f32,
		prep: &impl Fn(&[usize], &mut [f32]) -> usize,
	) -> Result<(), String> {
		for s in 0..self.slot.len() {
			if self.slot[s].is_some() {
				continue;
			}
			let Some(bt) = self.todo.pop_front() else {
				return Ok(());
			};
			let rows = prep(&bt, self.g.queries(s));
			self.g.submit(s, rows, nk, scale, self.upload)?;
			self.upload = false;
			self.seq += 1;
			self.slot[s] = Some((bt, self.seq, attn_units(rows, nk, inner)));
		}
		Ok(())
	}

	/// A finished batch, or with `wait` the oldest in flight once it is:
	/// its chunks and output rows (`[rows][inner]` f16).
	fn finished(&mut self, wait: bool) -> Result<Option<(Vec<usize>, &[u16])>, String> {
		let busy: Vec<usize> = (0..self.slot.len())
			.filter(|&s| self.slot[s].is_some())
			.collect();
		let s = match busy.iter().copied().find(|&s| self.g.done(s)) {
			Some(s) => s,
			None if wait => match busy
				.iter()
				.copied()
				.min_by_key(|&s| self.slot[s].as_ref().unwrap().1)
			{
				Some(s) => s,
				None => return Ok(None),
			},
			None => return Ok(None),
		};
		self.g.wait(s)?;
		let (bt, _, u) = self.slot[s].take().unwrap();
		if let Some(t) = self.g.seconds(s) {
			self.secs += t;
			self.units += u;
		}
		Ok(Some((bt, self.g.output(s))))
	}

	/// Refill the slots and take every finished batch (with `wait`, at
	/// least one unless nothing is pending), its rows through `place`:
	/// the chunks finished.
	fn step(
		&mut self,
		wait: bool,
		nk: usize,
		inner: usize,
		scale: f32,
		prep: &impl Fn(&[usize], &mut [f32]) -> usize,
		place: &impl Fn(&[usize], &[u16]),
	) -> Result<Vec<usize>, String> {
		self.fill(nk, inner, scale, prep)?;
		let (mut v, mut w) = (Vec::new(), wait);
		while let Some((bt, out)) = self.finished(w)? {
			place(&bt, out);
			v.extend(bt);
			w = false;
			self.fill(nk, inner, scale, prep)?;
		}
		Ok(v)
	}

	/// The chunks not finished on the card (after an error of its own),
	/// for the cores.
	fn abandon(&mut self) -> Vec<usize> {
		let mut v: Vec<usize> = self
			.slot
			.iter_mut()
			.filter_map(Option::take)
			.flat_map(|x| x.0)
			.collect();
		v.extend(self.todo.drain(..).flatten());
		v
	}
}

// SAFETY: the NPU objects are used from one thread at a time (behind the
// model's mutex; the driver's handles are thread safe).
unsafe impl Send for NpuDit {}
unsafe impl Sync for NpuDit {}

pub(super) fn read_f32(p: &Path) -> Result<Vec<f32>, String> {
	let b = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
	Ok(b.chunks_exact(4)
		.map(|c| f32::from_le_bytes(c.try_into().unwrap()))
		.collect())
}

/// What recording a job needs.
struct Rec<'a> {
	npu: &'static Npu,
	graphs: &'a mut [Graph<'static>],
	io: &'a [Vec<Io>],
	w: &'a [Vec<HostBuf<'static>>],
	m: &'a [HostBuf<'static>],
	b: &'a Bufs,
	chunks: &'a [Chunk],
	/// Per chunk its modulation entry.
	entry: &'a [usize],
	d: usize,
	inner: usize,
	ms: usize,
}

// SAFETY: moved to the recorder thread, which alone binds the graphs.
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
			chunks: self.chunks,
			entry: self.entry,
			d: self.d,
			inner: self.inner,
			ms: self.ms,
		}
	}
}

/// Record graph `kind` (`pre`, `mid`, `post`; its small form for a small
/// chunk) over chunk `c`, `l` the block of its pre part (`layers` for
/// `post`): residual `X[cur]` -> `X[1 - cur]`.
fn record(
	r: Rec<'_>,
	kind: usize,
	l: usize,
	c: usize,
	cur: usize,
) -> Result<npu::ze::Batch<'static>, String> {
	let ch = r.chunks[c];
	let kind = if ch.small { kind + 3 } else { kind };
	let (b, io, w, m) = (r.b, &r.io[kind], r.w, r.m);
	let (d, inner, ms, entry) = (r.d, r.inner, r.ms, r.entry);
	r.npu.batch_each(&mut r.graphs[kind], 1, &mut |g, _| {
		let (xr, ir) = (ch.r0 * d * 2, ch.r0 * inner * 2);
		for (i, x) in io.iter().enumerate() {
			match *x {
				Io::X => g.bind_at(i, &b.x[cur], xr)?,
				Io::Xo => g.bind_at(i, &b.x[1 - cur], xr)?,
				Io::A => g.bind_at(i, &b.a, ir)?,
				Io::Q => g.bind_at(i, &b.q, ir)?,
				Io::K => g.bind_at(i, &b.k, ir)?,
				Io::V => g.bind_at(i, &b.v, ir)?,
				Io::M(pre, p) => g.bind_at(
					i,
					&m[if pre { l } else { l - 1 }],
					(entry[c] * MPARTS + p) * ms,
				)?,
				Io::W(pre, k) => g.bind(i, &w[if pre { l } else { l - 1 }][k])?,
			}
		}
		Ok(())
	})
}

enum Ev {
	Chunk(usize),
	End(Result<(), String>),
}

impl NpuDit {
	pub fn load(
		c: &NpuCfg,
		cfg: &DitCfg,
		mut progress: impl FnMut(usize),
	) -> Result<NpuDit, String> {
		let shape = DShape {
			d: cfg.hidden,
			heads: cfg.heads,
			hd: cfg.head_dim,
			ff: cfg.ffn,
			layers: cfg.layers,
			kv: cfg.heads * cfg.head_dim,
		};
		let lock = NpuLock::shared()?;
		let npu = npu()?;
		let mut graphs = Vec::new();
		let mut io = Vec::new();
		let mut wnames: Vec<String> = Vec::new();
		for kind in KINDS {
			if kind.ends_with("64") && !c.graphs.join(format!("{kind}.blob")).exists() {
				break;
			}
			let g = load_graph(npu, &c.graphs, kind)?;
			let mut outs = h3::dit_outputs(&format!("h3/{kind}")).iter();
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
						n => match h3::mod_part(n) {
							Some(p) => Io::M(h3::is_pre(n), p),
							None => {
								let i = wnames.iter().position(|w| w == n).unwrap_or_else(|| {
									wnames.push(n.to_string());
									wnames.len() - 1
								});
								Io::W(h3::is_pre(n), i)
							}
						},
					}
				};
				v.push(x);
			}
			graphs.push(g);
			io.push(v);
		}
		let rd = |l: usize, n: &str| -> Result<Vec<u8>, String> {
			let p = h3::wfile(&c.weights, l, n);
			std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))
		};
		let to_buf = |b: &[u8]| -> Result<HostBuf<'static>, String> {
			let mut hb = npu.alloc(b.len())?;
			hb.as_mut_slice().copy_from_slice(b);
			Ok(hb)
		};
		let (mut w, mut m, mut qk) = (Vec::new(), Vec::new(), Vec::new());
		for l in 0..shape.layers {
			let mut v = Vec::with_capacity(wnames.len());
			for n in &wnames {
				use std::io::Read;
				let p = h3::wfile(&c.weights, l, n);
				let e = |e: std::io::Error| format!("{}: {e}", p.display());
				let mut f = std::fs::File::open(&p).map_err(e)?;
				let len = f.metadata().map_err(e)?.len() as usize;
				let mut hb = npu.alloc(len)?;
				f.read_exact(hb.as_mut_slice()).map_err(e)?;
				v.push(hb);
			}
			w.push(v);
			m.push(to_buf(&rd(l, "mod")?)?);
			let s = npubuild::qimage::f16f(&rd(l, "qkv.s")?);
			let f32s = |b: Vec<u8>| -> Vec<f32> {
				b.chunks_exact(4)
					.map(|c| f32::from_le_bytes(c.try_into().unwrap()))
					.collect()
			};
			let (nq, nk) = (f32s(rd(l, "nq")?), f32s(rd(l, "nk")?));
			let inner = shape.inner();
			if s.len() != 3 * inner || nq.len() != shape.hd || nk.len() != shape.hd {
				return Err(format!("image.npu: block {l}: bad qkv.s / nq / nk"));
			}
			qk.push(QkW {
				sq: s[..inner].to_vec(),
				sk: s[inner..2 * inner].to_vec(),
				nq,
				nk,
			});
			progress(l);
		}
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
		let sc = read_f32(&h3::scale_file(&c.weights))?;
		let ts = read_f32(&h3::ts_file(&c.weights))?;
		if sc.len() != shape.layers + 1 {
			return Err("image.npu: bad h3scale.f32".into());
		}
		let entries = ts.len() * MODALITIES;
		if m[0].len() < entries * MPARTS * h3::mstride(shape.d) {
			return Err("image.npu: mod.in is shorter than its entries".into());
		}
		#[cfg(feature = "vulkan")]
		let gb = gpu_blk(&w, &wnames, &shape);
		#[cfg(feature = "vulkan")]
		let ga = gpu_attn(shape.heads);
		let attn_fit = quant::calib::Fit::open(
			ATTN_FIT.0,
			quant::calib::fingerprint(&[&format!("heads={} hd={}", shape.heads, shape.hd)]),
			&ATTN_FIT.1,
			ATTN_FIT_NEED,
		);
		Ok(NpuDit {
			#[cfg(feature = "vulkan")]
			gb,
			#[cfg(feature = "vulkan")]
			ga,
			attn_fit,
			npu,
			shape,
			qk_eps: cfg.qk_eps,
			graphs,
			gdir: c.graphs.clone(),
			io,
			w,
			wnames,
			m,
			qk,
			c_in: sc[0],
			c: sc[1..].to_vec(),
			limit: None,
			ts,
			bufs: None,
			plan: None,
			kmean: Vec::new(),
			t_npu: 0.0,
			t_attn: 0.0,
			t_bound: 0.0,
			t_gpu: 0.0,
			gpu_chunks: 0,
			t_gtx: 0.0,
			gtx_chunks: 0,
			_lock: lock,
		})
	}

	/// Bytes of the weights in NPU host memory.
	pub fn resident_bytes(&self) -> usize {
		self.w
			.iter()
			.flatten()
			.chain(&self.m)
			.map(|b| b.len())
			.sum()
	}

	/// The stored scale of block `l`'s output (the result of a forward
	/// that ends there, times it, is the residual).
	pub fn scale(&self, l: usize) -> f32 {
		self.c[l]
	}

	/// The timesteps the modulation entries hold.
	pub fn timesteps(&self) -> &[f32] {
		&self.ts
	}

	/// Start a job: its rows' order and buffers.
	pub fn begin(&mut self, lay: &Layout, rope: &Rope) -> Result<(), String> {
		self.load_graphs()?;
		let p = Plan::new(lay, rope, self.io.len() == KINDS.len() && small_chunks());
		let (d, inner) = (self.shape.d, self.shape.inner());
		if self.bufs.as_ref().is_none_or(|b| b.rows != p.rows) {
			#[cfg(feature = "vulkan")]
			if let Some(g) = self.gb.as_mut() {
				g.forget_rows();
			}
			self.bufs = None;
			let npu = self.npu;
			let a = |n: usize| -> Result<HostBuf<'static>, String> {
				let mut b = npu.alloc(n * 2)?;
				b.as_mut_slice().fill(0);
				Ok(b)
			};
			self.bufs = Some(Bufs {
				rows: p.rows,
				x: [a(p.rows * d)?, a(p.rows * d)?],
				a: a(p.rows * inner)?,
				q: a(p.rows * inner)?,
				k: a(p.rows * inner)?,
				v: a(p.rows * inner)?,
			});
		} else if let Some(b) = self.bufs.as_mut() {
			let [x0, x1] = &mut b.x;
			for x in [x0, x1, &mut b.a] {
				x.as_mut_slice().fill(0);
			}
		}
		#[cfg(feature = "vulkan")]
		if let Some(g) = self.ga.as_mut() {
			if g.capacity() < p.nv {
				if let Err(e) = g.reserve(p.nv) {
					eprintln!("[h3] the attention on the cores alone this job: {e}");
				}
			}
		}
		self.plan = Some(p);
		self.kmean.clear();
		Ok(())
	}

	/// One step: the rows `x` (`[n][D]`, the layout's order: the refined
	/// text rows, then `proj_in` of the image rows) at the step's timestep
	/// `t` -> the target rows' last scaled residual (`[target][D]`).
	pub fn forward(&mut self, x: &[f32], t: f32) -> Result<Vec<f32>, String> {
		let plan = self.plan.as_ref().ok_or("npu transformer: no job begun")?;
		let d = self.shape.d;
		let ti = |t: f32| {
			self.ts.iter().position(|&x| x == t).ok_or(format!(
				"npu transformer: timestep {t} is not in the set's tables"
			))
		};
		let (it, ir) = (ti(t)?, ti(REF_T)?);
		let entry: Vec<usize> = plan
			.chunks
			.iter()
			.map(|ch| {
				let (r, tag) = plan.groups[ch.g];
				(if r { ir } else { it }) * MODALITIES + tag as usize
			})
			.collect();
		// the input rows, scaled, in the plan's order
		{
			let b = self.bufs.as_mut().unwrap();
			let xb = b.x[0].as_mut_slice().as_mut_ptr() as usize;
			let inv = 1.0 / self.c_in;
			let orig = &plan.orig;
			quant::par_for(plan.rows, |p| {
				// SAFETY: task p owns padded row p
				let dst = unsafe {
					std::slice::from_raw_parts_mut((xb as *mut u8).add(p * d * 2), d * 2)
				};
				let r = orig[p];
				if r == usize::MAX {
					dst.fill(0);
				} else {
					let row: Vec<f32> = x[r * d..(r + 1) * d].iter().map(|v| v * inv).collect();
					for (o, &v) in dst.chunks_exact_mut(2).zip(&row) {
						o.copy_from_slice(&quant::f32_to_fp16(v).to_le_bytes());
					}
				}
			});
		}
		// the iGPU's modulation slots: the step's distinct entries
		let mut ents: Vec<usize> = entry.clone();
		ents.sort_unstable();
		ents.dedup();
		let slot: Vec<usize> = entry
			.iter()
			.map(|e| ents.binary_search(e).unwrap())
			.collect();
		#[cfg(feature = "vulkan")]
		if let Some(g) = self.gb.as_mut() {
			if ents.len() > offload::H3_SLOTS {
				eprintln!(
					"[h3] {} modulation entries in a step: the NPU runs every chunk",
					ents.len()
				);
				self.gb = None;
			} else {
				let ms = h3::mstride(d);
				for (l, m) in self.m.iter().enumerate() {
					let mb = as_u16(m.as_slice());
					for (si, &e) in ents.iter().enumerate() {
						let mut parts = vec![0f32; MPARTS * d];
						for (p, dst) in parts.chunks_exact_mut(d).enumerate() {
							let o = (e * MPARTS + p) * ms / 2;
							quant::f16_to_f32_slice(&mb[o..o + d], dst);
						}
						g.set_entry(l, si, &parts);
					}
				}
			}
		}
		let cur = self.pipeline(&entry, &slot)?;
		let plan = self.plan.as_ref().unwrap();
		let b = self.bufs.as_ref().unwrap();
		let xb = as_u16(b.x[cur].as_slice());
		let mut out = vec![0f32; plan.target.len() * d];
		let op = crate::ops::Ptr(out.as_mut_ptr());
		quant::par_for(plan.target.len(), |i| {
			let op = op;
			let p = plan.target[i];
			// SAFETY: task i owns output row i
			let dst = unsafe { std::slice::from_raw_parts_mut(op.0.add(i * d), d) };
			quant::f16_to_f32_slice(&xb[p * d..(p + 1) * d], dst);
		});
		Ok(out)
	}

	/// The blocks over the job's rows (the input in `X[0]`); returns the
	/// residual's buffer.
	#[cfg_attr(not(feature = "vulkan"), allow(unused_variables))]
	fn pipeline(&mut self, entry: &[usize], slot: &[usize]) -> Result<usize, String> {
		let (d, inner, nh, hd) = (
			self.shape.d,
			self.shape.inner(),
			self.shape.heads,
			self.shape.hd,
		);
		let layers = self
			.limit
			.unwrap_or(self.shape.layers)
			.clamp(1, self.shape.layers);
		let nancheck = std::env::var("QWEN_IMAGE_NANCHECK").is_ok_and(|v| v == "1");
		let plan = self.plan.as_ref().unwrap();
		let chunks = plan.chunks.len();
		let nv = plan.nv;
		let half = plan.cos.len() / plan.rows;
		let scale = 1.0 / (hd as f32).sqrt();
		let eps = self.qk_eps;
		let int8 = quant::K8::supported(hd) && super::dit::attn_i8();
		let mut kc = vec![0u16; nv * inner];
		// the compact V for the float attention (no AVX-VNNI)
		let mut vc = if int8 {
			Vec::new()
		} else {
			vec![0u16; nv * inner]
		};
		// the iGPU's chunks of each block (`h3/mid` jobs only)
		#[cfg(feature = "vulkan")]
		let mut gb = self.gb.take();
		#[cfg(feature = "vulkan")]
		let on_gpu = match &gb {
			Some(_) => {
				let big = plan.chunks.iter().filter(|c| !c.small).count();
				gpu_chunks(
					&plan.chunks,
					(big as f64 * gpu_blk_share()).round() as usize,
				)
			}
			None => vec![false; chunks],
		};
		#[cfg(not(feature = "vulkan"))]
		let on_gpu = vec![false; chunks];
		self.gpu_chunks = on_gpu.iter().filter(|&&g| g).count();
		// the discrete card's attention chunks of each block
		#[cfg(feature = "vulkan")]
		let mut ga: Option<Ga> = if self.ga.as_ref().is_some_and(|g| g.capacity() >= nv) {
			self.ga.take()
		} else {
			None
		};
		#[cfg(not(feature = "vulkan"))]
		let mut ga: Option<Ga> = None;
		let share = attn_share_env().unwrap_or_else(|| {
			let (kc, kg) = (self.attn_fit.get(0), self.attn_fit.get(1));
			kc / (kc + kg)
		});
		let on_gtx = if ga.is_some() {
			gtx_chunks(&plan.chunks, share)
		} else {
			vec![false; chunks]
		};
		#[cfg(feature = "vulkan")]
		let batches = gtx_batches(&plan.chunks, &on_gtx);
		self.gtx_chunks = on_gtx.iter().filter(|&&g| g).count();
		let mut t_gtx = 0.0;
		#[allow(unused_mut)]
		let mut gtx_failed = false;
		let gw: Vec<usize> = GPU_W[..4]
			.iter()
			.map(|n| self.wnames.iter().position(|w| w == n).unwrap_or(0))
			.collect();
		let wall = &self.w;
		let mut t_gpu = 0.0;
		let rec = Rec {
			npu: self.npu,
			graphs: &mut self.graphs,
			io: &self.io,
			w: &self.w,
			m: &self.m,
			b: self.bufs.as_ref().unwrap(),
			chunks: &plan.chunks,
			entry,
			d,
			inner,
			ms: h3::mstride(d),
		};
		let b = rec.b;
		let qk = &self.qk;
		let kmean = &mut self.kmean;
		kmean.resize(layers, Vec::new());
		let attn_fit = &mut self.attn_fit;
		let (mut t_attn, mut t_wait, mut t_bound) = (0.0, 0.0, 0.0);
		let gone = || "npu recorder gone".to_string();
		// block l's valid K rows of chunk c, finished, into the compact copy
		// (and its V rows, without AVX-VNNI)
		let fin = |l: usize, c: usize, kc: &mut [u16], vc: &mut [u16], ga: &mut Option<Ga>| {
			let Chunk { n, k0, r0: p0, .. } = plan.chunks[c];
			let w = &qk[l];
			let src = &as_u16(b.k.as_slice())[p0 * inner..(p0 + n) * inner];
			let dp = kc.as_mut_ptr() as usize;
			const R: usize = 8;
			let (cs, sn) = (&plan.cos[p0 * half..], &plan.sin[p0 * half..]);
			quant::par_for(n.div_ceil(R), move |i| {
				let (r0, r1) = (i * R, ((i + 1) * R).min(n));
				let mut f = vec![0f32; (r1 - r0) * inner];
				quant::f16_to_f32_slice(&src[r0 * inner..r1 * inner], &mut f);
				qk_finish(
					&mut f,
					&w.sk,
					&w.nk,
					&cs[r0 * half..],
					&sn[r0 * half..],
					half,
					hd,
					eps,
				);
				// SAFETY: task i owns compact rows k0 + r0 .. k0 + r1
				let d = unsafe {
					std::slice::from_raw_parts_mut(
						(dp as *mut u16).add((k0 + r0) * inner),
						(r1 - r0) * inner,
					)
				};
				quant::f32_to_f16_slice(&f, d);
			});
			if !vc.is_empty() {
				vc[k0 * inner..(k0 + n) * inner]
					.copy_from_slice(&as_u16(b.v.as_slice())[p0 * inner..(p0 + n) * inner]);
			}
			// the card's copy, to go up with the block's first batch
			#[cfg(feature = "vulkan")]
			if let Some(g) = ga.as_mut() {
				g.stage_kv(
					k0,
					&kc[k0 * inner..(k0 + n) * inner],
					&as_u16(b.v.as_slice())[p0 * inner..(p0 + n) * inner],
				);
			}
			#[cfg(not(feature = "vulkan"))]
			let _ = ga;
		};
		let (tx, rx) = std::sync::mpsc::channel::<Option<(usize, usize, usize, usize)>>();
		let (done_tx, done_rx) = std::sync::mpsc::channel::<Ev>();
		let r = std::thread::scope(|sc| {
			sc.spawn(move || {
				let mut rec = rec;
				let mut pend: Vec<(usize, usize, npu::ze::Batch<'static>)> = Vec::new();
				let mut err: Option<String> = None;
				while let Ok(msg) = rx.recv() {
					match msg {
						Some((kind, l, c, cur)) => {
							if err.is_some() {
								continue;
							}
							if pend.len() >= MAX_LISTS {
								let (k, c, mut bt) = pend.remove(0);
								if let Err(e) = bt.wait() {
									err = Some(e);
									continue;
								}
								if k != 2 {
									let _ = done_tx.send(Ev::Chunk(c));
								}
							}
							match record(rec.reborrow(), kind, l, c, cur)
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
									Ok(_) if k != 2 && r.is_ok() => {
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
			// block 0 up to its attention, its K rows finished as the jobs end
			let t = std::time::Instant::now();
			for c in 0..chunks {
				tx.send(Some((0, 0, c, 0))).map_err(|_| gone())?;
			}
			tx.send(None).map_err(|_| gone())?;
			loop {
				match done_rx.recv().map_err(|_| gone())? {
					Ev::Chunk(c) => fin(0, c, &mut kc, &mut vc, &mut ga),
					Ev::End(r) => break r?,
				}
			}
			t_wait += t.elapsed().as_secs_f64();
			let mut k8 = quant::K8::default();
			let mut qf = vec![0f32; S * inner];
			let mut af = vec![0f32; S * inner];
			let mut cur = 0;
			for l in 0..layers {
				let last = l + 1 == layers;
				// block l's keys and values as int8 (its K rows are finished)
				let t = std::time::Instant::now();
				if int8 {
					// centered at the previous step's means, or the keys' own
					let mean = if kmean[l].len() == nh * hd {
						kmean[l].clone()
					} else {
						let mut s = vec![0f64; inner];
						for row in kc.chunks_exact(inner) {
							for (a, &h) in s.iter_mut().zip(row) {
								*a += quant::fp16_to_f32(h) as f64;
							}
						}
						s.iter().map(|&v| (v / nv as f64) as f32).collect()
					};
					k8.begin(nv, nh, hd, &mean, true);
					let vb = as_u16(b.v.as_slice());
					for &Chunk { n, k0, r0, .. } in &plan.chunks {
						k8.pack_rows(&kc, k0, k0 + n);
						// V rows of the chunk start at padded row r0, key k0
						k8.pack_v_rows(&vb[(r0 - k0) * inner..], k0, k0 + n);
					}
					kmean[l] = k8.key_mean();
				}
				let tb = t.elapsed().as_secs_f64();
				t_attn += tb;
				t_bound += tb;
				{
					let w = &qk[l];
					// chunk c's job (its attention rows in `a`)
					#[allow(unused_mut)]
					let mut send = |c: usize| -> Result<(), String> {
						if !last && on_gpu[c] {
							#[cfg(feature = "vulkan")]
							{
								let (wp, wq) = (&wall[l], &wall[l + 1]);
								let (r0, rows) = (plan.chunks[c].r0, plan.chunks[c].rows());
								gb.as_mut().unwrap().mid_begin(&offload::H3Mid {
									l: l + 1,
									x: b.x[cur].as_slice(),
									a: b.a.as_slice(),
									xo: b.x[1 - cur].as_slice(),
									q: b.q.as_slice(),
									k: b.k.as_slice(),
									v: b.v.as_slice(),
									r0,
									r1: r0 + rows,
									w_o: wp[gw[0]].as_slice(),
									w_gu: wp[gw[1]].as_slice(),
									w_d: wp[gw[2]].as_slice(),
									w_qkv: wq[gw[3]].as_slice(),
									slot: slot[c],
								})?;
							}
						} else {
							let job = if last {
								(2, layers, c, cur)
							} else {
								(1, l + 1, c, cur)
							};
							tx.send(Some(job)).map_err(|_| gone())?;
						}
						Ok(())
					};
					// chunk c's attention on the cores: seconds
					let mut cpu = |c: usize| -> f64 {
						let Chunk { n, r0, .. } = plan.chunks[c];
						let _w = quant::parallel::limit_width(if c == 0 { 0 } else { 16 });
						let t = std::time::Instant::now();
						let q = &mut qf[..n * inner];
						widen(&as_u16(b.q.as_slice())[r0 * inner..(r0 + n) * inner], q);
						qk_finish(
							q,
							&w.sq,
							&w.nq,
							&plan.cos[r0 * half..],
							&plan.sin[r0 * half..],
							half,
							hd,
							eps,
						);
						let o = &mut af[..n * inner];
						if int8 {
							// (V is read from `k8`; the slice only has to be long enough)
							quant::attn_block_rows_k8(
								q,
								&kc,
								&k8,
								as_u16(b.v.as_slice()),
								n,
								0,
								nh,
								hd,
								scale,
								nv,
								o,
							);
						} else {
							quant::attn_block_rows(q, &kc, &vc, n, 0, nh, nh, hd, scale, nv, o);
						}
						// SAFETY: rows r0 .. r0 + n of `a` are read only by the job below
						let a = unsafe {
							std::slice::from_raw_parts_mut(
								(b.a.ptr() as *mut u8).add(r0 * inner * 2),
								n * inner * 2,
							)
						};
						narrow(o, a);
						t.elapsed().as_secs_f64()
					};
					// the card's query rows of a batch, finished
					#[cfg(feature = "vulkan")]
					let prep = |bt: &[usize], dst: &mut [f32]| -> usize {
						let mut off = 0;
						for &c in bt {
							let Chunk { n, r0, .. } = plan.chunks[c];
							let src = &as_u16(b.q.as_slice())[r0 * inner..(r0 + n) * inner];
							let dp = dst[off * inner..(off + n) * inner].as_mut_ptr() as usize;
							let (cs, sn) = (&plan.cos[r0 * half..], &plan.sin[r0 * half..]);
							const R: usize = 8;
							quant::par_for(n.div_ceil(R), move |i| {
								let (a, z) = (i * R, ((i + 1) * R).min(n));
								let mut f = vec![0f32; (z - a) * inner];
								quant::f16_to_f32_slice(&src[a * inner..z * inner], &mut f);
								qk_finish(
									&mut f,
									&w.sq,
									&w.nq,
									&cs[a * half..],
									&sn[a * half..],
									half,
									hd,
									eps,
								);
								// SAFETY: task i owns rows a .. z of the chunk's part of `dst`
								unsafe {
									std::ptr::copy_nonoverlapping(
										f.as_ptr(),
										(dp as *mut f32).add(a * inner),
										f.len(),
									)
								};
							});
							off += n;
						}
						off
					};
					// a finished batch's rows into `a`
					#[cfg(feature = "vulkan")]
					let place = |bt: &[usize], out: &[u16]| {
						let mut off = 0;
						for &c in bt {
							let Chunk { n, r0, .. } = plan.chunks[c];
							// SAFETY: rows r0 .. r0 + n of `a` are read only by chunk c's job, sent after this
							let a = unsafe {
								std::slice::from_raw_parts_mut(
									(b.a.ptr() as *mut u16).add(r0 * inner),
									n * inner,
								)
							};
							a.copy_from_slice(&out[off * inner..(off + n) * inner]);
							off += n;
						}
					};
					#[cfg(feature = "vulkan")]
					let mut gr = ga.as_mut()
						.filter(|_| !batches.is_empty())
						.map(|g| GtxRun::new(g, &batches));
					// the chunks the card left after an error of its own
					#[allow(unused_mut)]
					let mut left: Vec<usize> = Vec::new();
					// refill the card and send the jobs of its finished chunks
					// (`true`: wait for at least one)
					#[cfg(feature = "vulkan")]
					macro_rules! gtx {
						($wait:expr) => {{
							let mut fin_c: Vec<usize> = Vec::new();
							let mut dead = false;
							if let Some(r) = gr.as_mut() {
								let t = std::time::Instant::now();
								match r.step($wait, nv, inner, scale, &prep, &place) {
									Ok(v) => fin_c = v,
									Err(e) => {
										eprintln!("[h3] the discrete card failed ({e}): the cores run the attention from here on");
										left.extend(r.abandon());
										dead = true;
									}
								}
								if $wait {
									t_gtx += t.elapsed().as_secs_f64();
								}
							}
							if dead {
								gr = None;
								gtx_failed = true;
							}
							for c in fin_c {
								send(c)?;
							}
						}};
					}
					#[cfg(feature = "vulkan")]
					gtx!(false);
					let (mut cpu_s, mut cpu_u) = (0.0, 0.0);
					for c in (0..chunks).filter(|&c| !on_gtx[c]) {
						let dt = cpu(c);
						t_attn += dt;
						if c == 0 {
							t_bound += dt;
						} else {
							cpu_s += dt;
							cpu_u += attn_units(plan.chunks[c].n, nv, inner);
						}
						send(c)?;
						#[cfg(feature = "vulkan")]
						gtx!(false);
					}
					#[cfg(feature = "vulkan")]
					while gr.as_ref().is_some_and(|r| r.pending()) {
						gtx!(true);
					}
					#[cfg(feature = "vulkan")]
					if let Some(r) = gr.as_ref() {
						if r.units > 0.0 && cpu_u > 0.0 {
							attn_fit.add(&[(cpu_s, cpu_u), (r.secs, r.units)]);
						}
					}
					#[cfg(not(feature = "vulkan"))]
					let _ = (cpu_s, cpu_u, &mut t_gtx, &attn_fit);
					for c in left {
						t_attn += cpu(c);
						send(c)?;
					}
				}
				let t = std::time::Instant::now();
				tx.send(None).map_err(|_| gone())?;
				let mut t_fin = 0.0;
				loop {
					match done_rx.recv().map_err(|_| gone())? {
						Ev::Chunk(c) => {
							let tf = std::time::Instant::now();
							fin(l + 1, c, &mut kc, &mut vc, &mut ga);
							t_fin += tf.elapsed().as_secs_f64();
						}
						Ev::End(r) => break r?,
					}
				}
				t_wait += t.elapsed().as_secs_f64() - t_fin;
				// the iGPU's chunks, then their K rows
				if !last && on_gpu.iter().any(|&g| g) {
					let t = std::time::Instant::now();
					#[cfg(feature = "vulkan")]
					gb.as_mut().unwrap().end()?;
					t_gpu += t.elapsed().as_secs_f64();
					let tf = std::time::Instant::now();
					for c in (0..chunks).filter(|&c| on_gpu[c]) {
						fin(l + 1, c, &mut kc, &mut vc, &mut ga);
					}
					t_fin += tf.elapsed().as_secs_f64();
				}
				t_attn += t_fin;
				cur = 1 - cur;
				if nancheck {
					let xb = as_u16(b.x[cur].as_slice());
					let bad = |v: &[u16]| v.iter().filter(|&&h| h & 0x7c00 == 0x7c00).count();
					let mx = xb
						.iter()
						.filter(|&&h| h & 0x7c00 != 0x7c00)
						.fold(0f32, |a, &h| a.max(quant::fp16_to_f32(h).abs()));
					eprintln!(
						"  [nan] block {l}: x {} (max {mx:.0}), q {}, k {}, v {}",
						bad(xb),
						bad(as_u16(b.q.as_slice())),
						bad(&kc),
						bad(as_u16(b.v.as_slice()))
					);
				}
			}
			drop(tx);
			Ok::<usize, String>(cur)
		});
		self.t_attn += t_attn;
		self.t_npu += t_wait;
		self.t_bound += t_bound;
		self.t_gpu += t_gpu;
		#[cfg(feature = "vulkan")]
		{
			self.gb = gb;
			if gtx_failed {
				self.ga = None;
			} else if ga.is_some() {
				self.ga = ga;
			}
		}
		self.t_gtx += t_gtx;
		r
	}

	/// Unload the graphs, keeping the weights (driver memory a chat engine
	/// on the NPU shares); the next job loads them again.
	pub fn unload_graphs(&mut self) {
		self.graphs.clear();
	}

	fn load_graphs(&mut self) -> Result<(), String> {
		if !self.graphs.is_empty() {
			return Ok(());
		}
		for kind in &KINDS[..self.io.len()] {
			self.graphs.push(load_graph(self.npu, &self.gdir, kind)?);
		}
		Ok(())
	}

	/// Free the job's buffers.
	pub fn release_buffers(&mut self) {
		#[cfg(feature = "vulkan")]
		if let Some(g) = self.gb.as_mut() {
			g.forget_rows();
		}
		#[cfg(feature = "vulkan")]
		if let Some(g) = self.ga.as_mut() {
			g.release();
		}
		self.bufs = None;
		self.plan = None;
	}

	pub fn weight_names(&self) -> &[String] {
		&self.wnames
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The finish of a raw row equals scale, RMSNorm with weight, then the
	/// rotate-half RoPE of the reference (`dit::Rope::apply`).
	#[test]
	fn qk_finish_matches_the_reference() {
		let (hd, heads, half) = (8, 2, 3);
		let w = heads * hd;
		let x0: Vec<f32> = (0..w).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
		let s: Vec<f32> = (0..w).map(|i| 0.5 + i as f32 * 0.01).collect();
		let nw: Vec<f32> = (0..hd).map(|i| 1.0 + i as f32 * 0.1).collect();
		let cos = vec![0.6f32, 0.8, -0.2];
		let sin = vec![0.8f32, -0.6, 0.97];
		let mut got = x0.clone();
		qk_finish(&mut got, &s, &nw, &cos, &sin, half, hd, 1e-5);
		let mut want: Vec<f32> = x0.iter().zip(&s).map(|(a, b)| a * b).collect();
		crate::ops::rms(&mut want, &nw, 1e-5, false);
		let rope = Rope { cos, sin, half };
		rope.apply(&mut want, heads, hd);
		for (a, b) in got.iter().zip(&want) {
			assert!((a - b).abs() < 1e-5, "{a} vs {b}");
		}
	}

	#[test]
	fn h3_npu_parses() {
		let d = std::env::temp_dir().join(format!("qwen-rs-h3npu-{}", std::process::id()));
		std::fs::create_dir_all(&d).unwrap();
		let p = d.join("image.npu");
		std::fs::write(&p, "arch = h3-image\ncheckpoint = C\ngraphs = graphs/h3\nweights = w\nvae = graphs/h3vae\n").unwrap();
		assert!(NpuCfg::is_h3_npu(&p));
		let c = NpuCfg::parse(&p).unwrap();
		assert_eq!(c.graphs, d.join("graphs/h3"));
		assert!(c.vae.is_some());
		std::fs::write(&p, "arch = qwen-image\n").unwrap();
		assert!(!NpuCfg::is_h3_npu(&p));
		let _ = std::fs::remove_dir_all(&d);
	}
}
