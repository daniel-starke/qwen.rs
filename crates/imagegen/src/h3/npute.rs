//! The text encoder on the NPU (`image.npu` keys `te`, `te_graphs`, built
//! by the `te` step of `cli H3_DIR --npu-build OUT`; the files and graphs:
//! `npubuild::h3te`).
//!
//! A layer is the transformer block's graphs without modulation (one entry)
//! and with grouped queries; the rows run in order in [`S`]-row chunks (the
//! rest in [`S_SMALL`]-row ones). The attention is causal, so chunk c's
//! attention in layer l needs the keys of chunks 0..=c of layer l only:
//! each chunk walks through the layers on its own (a wavefront), its next
//! job sent as soon as its attention is done, and no device waits for a
//! whole layer. The NPU runs the jobs in the order they are sent; a share of
//! the [`S`]-row chunks runs its `mid` jobs on the iGPU (`offload::H3Blk`,
//! the same math on the same int8 weights and host buffers) from a thread
//! of its own. The cores finish each chunk's keys (row scales, head norm,
//! MRoPE) into a compact copy per layer and run the attention.
//!
//! The residual is stored as `x_l / c_l` (`npubuild::h3img`); after the
//! layers that take a deepstack feature the job is cut at the layer's end
//! (`post`, the cores add the image rows' features, `pre`). The result is
//! the last layer's residual, unnormalized: `hidden_states[50]`.

use super::npudit::{load_graph, qk_finish, read_f32, Io, QkW, MAX_LISTS};
use crate::npuhost::{as_u16, narrow, npu, widen};
use npu::lock::{NpuLock, SharedNpuLock};
use npu::ze::{Graph, HostBuf, Npu};
#[cfg(feature = "vulkan")]
use npubuild::h3img::MPARTS;
use npubuild::h3img::{self as h3, DShape, S, S_SMALL};
use npubuild::h3te;
use std::path::{Path, PathBuf};

/// The graphs: pre, mid, post over [`S`] rows, then over [`S_SMALL`].
const KINDS: [&str; 6] = ["pre", "mid", "post", "pre64", "mid64", "post64"];
const PRE: usize = 0;
const MID: usize = 1;
const POST: usize = 2;

/// One chunk of rows: `r0 .. r0 + n` valid of `r0 .. r0 + rows`.
#[derive(Clone, Copy, Debug)]
struct Chunk {
	r0: usize,
	n: usize,
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

/// The chunks of `s` rows: [`S`]-row ones while more than `3 S_SMALL`
/// rows are left, then [`S_SMALL`]-row ones.
fn chunks(s: usize) -> Vec<Chunk> {
	let mut v = Vec::new();
	let mut r = 0;
	while r < s {
		let left = s - r;
		let small = left <= 3 * S_SMALL;
		let rows = if small { S_SMALL } else { S };
		v.push(Chunk {
			r0: r,
			n: left.min(rows),
			small,
		});
		r += rows;
	}
	v
}

/// `QWEN_H3_TE_GPU`: the iGPU's share of the [`S`]-row chunks (default
/// 0.25; `0` = the NPU runs every job).
#[cfg(feature = "vulkan")]
fn gpu_share() -> f64 {
	std::env::var("QWEN_H3_TE_GPU")
		.ok()
		.and_then(|v| v.parse().ok())
		.unwrap_or(0.25)
}

/// The weights the iGPU's jobs read (`npudit`'s `GPU_W`).
const GPU_W: [&str; 8] = [
	"o.w", "gu.w", "d.w", "qkv.w", "o.pre", "gu.pre", "qkv.pre", "gu.sg",
];

/// The NPU host buffers of a job.
struct Bufs {
	rows: usize,
	x: [HostBuf<'static>; 2],
	a: HostBuf<'static>,
	q: HostBuf<'static>,
	k: HostBuf<'static>,
	v: HostBuf<'static>,
}

/// What the NPU's recorder thread uses.
struct Side<'a> {
	graphs: &'a mut Vec<Graph<'static>>,
	io: &'a [Vec<Io>],
	w: &'a [Vec<HostBuf<'static>>],
	m: &'a [HostBuf<'static>],
	b: &'a Bufs,
	npu: &'static Npu,
}

// SAFETY: moved to the recorder thread, which alone binds the graphs; the
// buffers are only read by it (the driver reads and writes them).
unsafe impl Send for Side<'_> {}

/// What the iGPU's thread uses.
#[cfg(feature = "vulkan")]
struct GSide<'a> {
	g: &'a mut offload::H3Blk,
	w: &'a [Vec<HostBuf<'static>>],
	b: &'a Bufs,
}

// SAFETY: moved to the iGPU's thread, which alone uses the device; the
// buffers' rows of a job are touched by no other job meanwhile.
#[cfg(feature = "vulkan")]
unsafe impl Send for GSide<'_> {}

pub struct NpuTe {
	/// The iGPU's `mid` jobs; first, so it drops (with its imports of the
	/// NPU buffers) before them.
	#[cfg(feature = "vulkan")]
	gb: Option<offload::H3Blk>,
	npu: &'static Npu,
	shape: DShape,
	eps: f32,
	graphs: Vec<Graph<'static>>,
	gdir: PathBuf,
	io: Vec<Vec<Io>>,
	w: Vec<Vec<HostBuf<'static>>>,
	wnames: Vec<String>,
	m: Vec<HostBuf<'static>>,
	qk: Vec<QkW>,
	c_in: f32,
	c: Vec<f32>,
	bufs: Option<Bufs>,
	/// Seconds of the last [`Self::encode`]: waiting for a device, the cores'
	/// attention and key finishing; jobs on the iGPU.
	pub t_wait: f64,
	pub t_attn: f64,
	pub gpu_jobs: usize,
	_lock: SharedNpuLock,
}

// SAFETY: the NPU objects are used from one thread at a time (behind the
// model's mutex; the driver's handles are thread safe).
unsafe impl Send for NpuTe {}
unsafe impl Sync for NpuTe {}

/// What a job's completion reports.
enum Ev {
	/// Chunk `c`'s job of `kind` with pre block `l` is done.
	Done(usize, usize, usize),
	Err(String),
	/// The iGPU failed chunk `c`'s `mid` job with pre block `l` on
	/// residual buffer `cur` (or one sent after a failure): the NPU runs it.
	#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
	GpuErr(usize, usize, usize, String),
}

#[cfg(feature = "vulkan")]
fn gpu_blk(w: &[Vec<HostBuf<'static>>], wnames: &[String], s: &DShape) -> Option<offload::H3Blk> {
	if gpu_share() <= 0.0 {
		return None;
	}
	let ix: Vec<usize> = GPU_W
		.iter()
		.filter_map(|n| wnames.iter().position(|w| w == n))
		.collect();
	// int8 codes: one byte per weight (`d` with the `rowq` pad columns)
	let int8 = |i: usize, n: usize| w[0][ix[i]].len() >= n;
	if ix.len() != GPU_W.len()
		|| !(int8(0, s.d * s.inner())
			&& int8(1, 2 * s.ff * s.d)
			&& int8(2, s.d * s.ff)
			&& int8(3, s.qkv() * s.d))
	{
		eprintln!("[h3] the text encoder all on the NPU: its weights are not the iGPU kernels' form (int8)");
		return None;
	}
	let need: usize = w
		.iter()
		.flat_map(|b| ix[..4].iter().map(|&i| b[i].len()))
		.sum();
	let (have, cap) = (offload::host_imported(), offload::host_import_budget());
	if have + need > cap {
		eprintln!(
			"[h3] the text encoder all on the NPU: the iGPU holds {:.1} GB of imports, its share would add {:.1} GB (budget {:.1} GB, QWEN_GPU_IMPORT_MB)",
			have as f64 / 1e9,
			need as f64 / 1e9,
			cap as f64 / 1e9
		);
		return None;
	}
	let mut g = match offload::H3Blk::new(Some("intel"), w.len(), s.d, s.inner(), s.kv, s.ff) {
		Ok(g) => g,
		Err(e) => {
			eprintln!("[h3] no iGPU for the text encoder ({e}): the NPU runs every job");
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

impl NpuTe {
	/// The text encoder of `layers` layers of the shape `te` from the
	/// weights `wdir` (`te{L}.*.in`, `te.scale.f32`) and the graphs `gdir`.
	pub fn load(
		wdir: &Path,
		gdir: &Path,
		te: &crate::config::TeCfg,
		mut progress: impl FnMut(usize),
	) -> Result<NpuTe, String> {
		let shape = DShape {
			d: te.dim,
			heads: te.heads,
			hd: te.head_dim,
			ff: te.ffn,
			layers: te.layers,
			kv: te.kv_heads * te.head_dim,
		};
		let lock = NpuLock::shared()?;
		let npu = npu()?;
		let mut graphs = Vec::new();
		let mut io = Vec::new();
		let mut wnames: Vec<String> = Vec::new();
		for kind in KINDS {
			let g = load_graph(npu, gdir, kind)?;
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
			let p = h3te::wfile(wdir, l, n);
			std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))
		};
		let f32s = |b: Vec<u8>| -> Vec<f32> {
			b.chunks_exact(4)
				.map(|c| f32::from_le_bytes(c.try_into().unwrap()))
				.collect()
		};
		let (inner, kv) = (shape.inner(), shape.kv);
		let (mut w, mut m, mut qk) = (Vec::new(), Vec::new(), Vec::new());
		for l in 0..shape.layers {
			let mut v = Vec::with_capacity(wnames.len());
			for n in &wnames {
				use std::io::Read;
				let p = h3te::wfile(wdir, l, n);
				let e = |e: std::io::Error| format!("{}: {e}", p.display());
				let mut f = std::fs::File::open(&p).map_err(e)?;
				let len = f.metadata().map_err(e)?.len() as usize;
				let mut hb = npu.alloc(len)?;
				f.read_exact(hb.as_mut_slice()).map_err(e)?;
				v.push(hb);
			}
			w.push(v);
			let mb = rd(l, "mod")?;
			let mut hb = npu.alloc(mb.len())?;
			hb.as_mut_slice().copy_from_slice(&mb);
			m.push(hb);
			let s = npubuild::qimage::f16f(&rd(l, "qkv.s")?);
			let (nq, nk) = (f32s(rd(l, "nq")?), f32s(rd(l, "nk")?));
			if s.len() != inner + 2 * kv || nq.len() != shape.hd || nk.len() != shape.hd {
				return Err(format!(
					"image.npu: text encoder layer {l}: bad qkv.s / nq / nk"
				));
			}
			qk.push(QkW {
				sq: s[..inner].to_vec(),
				sk: s[inner..inner + kv].to_vec(),
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
							"text encoder {}: {} has {} bytes, the graph wants {}",
							KINDS[k],
							wnames[i],
							w[0][i].len(),
							a.bytes
						));
					}
				}
			}
		}
		let sc = read_f32(&h3te::scale_file(wdir))?;
		if sc.len() != shape.layers + 1 {
			return Err("image.npu: bad te.scale.f32".into());
		}
		#[cfg(feature = "vulkan")]
		let gb = gpu_blk(&w, &wnames, &shape);
		#[allow(unused_mut)]
		let mut t = NpuTe {
			#[cfg(feature = "vulkan")]
			gb,
			npu,
			shape,
			eps: te.eps,
			graphs,
			gdir: gdir.to_path_buf(),
			io,
			w,
			wnames,
			m,
			qk,
			c_in: sc[0],
			c: sc[1..].to_vec(),
			bufs: None,
			t_wait: 0.0,
			t_attn: 0.0,
			gpu_jobs: 0,
			_lock: lock,
		};
		#[cfg(feature = "vulkan")]
		t.set_gpu_entries();
		Ok(t)
	}

	/// The iGPU's modulation slot 0 of every layer: the layer's one entry.
	#[cfg(feature = "vulkan")]
	fn set_gpu_entries(&mut self) {
		let d = self.shape.d;
		let ms = h3::mstride(d);
		if let Some(g) = self.gb.as_mut() {
			for (l, m) in self.m.iter().enumerate() {
				let mb = as_u16(m.as_slice());
				let mut parts = vec![0f32; MPARTS * d];
				for (p, dst) in parts.chunks_exact_mut(d).enumerate() {
					quant::f16_to_f32_slice(&mb[p * ms / 2..p * ms / 2 + d], dst);
				}
				g.set_entry(l, 0, &parts);
			}
		}
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

	/// Unload the graphs, keeping the weights; the next encode loads them
	/// again.
	pub fn unload_graphs(&mut self) {
		self.graphs.clear();
	}

	fn load_graphs(&mut self) -> Result<(), String> {
		if self.graphs.is_empty() {
			for kind in KINDS {
				self.graphs.push(load_graph(self.npu, &self.gdir, kind)?);
			}
		}
		Ok(())
	}

	fn buffers(&mut self, rows: usize) -> Result<(), String> {
		if self.bufs.as_ref().is_some_and(|b| b.rows >= rows) {
			return Ok(());
		}
		#[cfg(feature = "vulkan")]
		if let Some(g) = self.gb.as_mut() {
			g.forget_rows();
		}
		self.bufs = None;
		let npu = self.npu;
		let (d, inner, kv) = (self.shape.d, self.shape.inner(), self.shape.kv);
		let a = |n: usize| -> Result<HostBuf<'static>, String> {
			let mut b = npu.alloc(n * 2)?;
			b.as_mut_slice().fill(0);
			Ok(b)
		};
		self.bufs = Some(Bufs {
			rows,
			x: [a(rows * d)?, a(rows * d)?],
			a: a(rows * inner)?,
			q: a(rows * inner)?,
			k: a(rows * kv)?,
			v: a(rows * kv)?,
		});
		Ok(())
	}

	/// The stack over the input rows `x` (`[s][d]`, the embeddings and the
	/// vision rows) with the RoPE tables `cos` / `sin` (`[s][hd / 2]`); the
	/// image rows `img` take `deep[j]` (`[img.len()][d]`) after layer j.
	/// Returns `hidden_states[layers]` (`[s][d]`).
	#[cfg_attr(not(feature = "vulkan"), allow(unused_variables, unused_mut))]
	pub fn encode(
		&mut self,
		x: &[f32],
		cos: &[f32],
		sin: &[f32],
		img: &[usize],
		deep: &[Vec<f32>],
	) -> Result<Vec<f32>, String> {
		let (d, inner, kv, hd) = (
			self.shape.d,
			self.shape.inner(),
			self.shape.kv,
			self.shape.hd,
		);
		let (heads, kvh) = (self.shape.heads, kv / hd);
		let layers = self.shape.layers;
		let s = x.len() / d;
		let half = hd / 2;
		if s == 0 || x.len() != s * d || cos.len() != s * half || sin.len() != s * half {
			return Err("npu text encoder: bad input".into());
		}
		if deep.iter().any(|f| f.len() != img.len() * d) {
			return Err("npu text encoder: bad deepstack features".into());
		}
		self.load_graphs()?;
		let ch = chunks(s);
		let rows = ch.last().map_or(0, |c| c.r0 + c.rows());
		self.buffers(rows)?;
		let nc = ch.len();
		// the input rows, scaled
		{
			let b = self.bufs.as_mut().unwrap();
			let inv = 1.0 / self.c_in;
			let xb = b.x[0].as_mut_slice();
			xb.fill(0);
			let dst = xb.as_mut_ptr() as usize;
			quant::par_for(s, |r| {
				let row: Vec<f32> = x[r * d..(r + 1) * d].iter().map(|v| v * inv).collect();
				// SAFETY: task r owns row r
				let o = unsafe { std::slice::from_raw_parts_mut((dst as *mut u16).add(r * d), d) };
				quant::f32_to_f16_slice(&row, o);
			});
		}
		// the image rows' index per row
		let mut img_of = vec![usize::MAX; s];
		for (j, &r) in img.iter().enumerate() {
			img_of[r] = j;
		}
		let ndeep = deep.len().min(layers - 1);
		#[cfg(feature = "vulkan")]
		let on_gpu: Vec<bool> = {
			let big: Vec<usize> = (0..nc).filter(|&c| !ch[c].small).collect();
			let ng = if self.gb.is_some() {
				(big.len() as f64 * gpu_share()).round() as usize
			} else {
				0
			};
			let ng = ng.min(big.len().saturating_sub(1));
			// the last big chunks: the others never wait for their keys
			let mut on = vec![false; nc];
			for &c in &big[big.len() - ng..] {
				on[c] = true;
			}
			on
		};
		#[cfg(not(feature = "vulkan"))]
		let on_gpu = vec![false; nc];
		self.gpu_jobs = 0;
		let ms = h3::mstride(d);
		let scale = 1.0 / (hd as f32).sqrt();
		let eps = self.eps;
		let (c_scale, qk) = (&self.c, &self.qk);
		let b = self.bufs.as_ref().unwrap();
		let (graphs, io, w, m, npu) = (&mut self.graphs, &self.io, &self.w, &self.m, self.npu);
		#[cfg(feature = "vulkan")]
		let mut gb = self.gb.take();
		let gw: Vec<usize> = GPU_W[..4]
			.iter()
			.map(|n| self.wnames.iter().position(|w| w == n).unwrap_or(0))
			.collect();
		let (mut t_wait, mut t_attn, mut gpu_jobs) = (0.0, 0.0, 0usize);
		let gone = || "npu text encoder: a device thread is gone".to_string();
		let r = std::thread::scope(|sc| -> Result<bool, String> {
			let (done_tx, done_rx) = std::sync::mpsc::channel::<Ev>();
			// the NPU's jobs in the order they are sent
			let (tx, rx) = std::sync::mpsc::channel::<(usize, usize, usize, usize)>();
			let dtx = done_tx.clone();
			let chn = &ch;
			let side = Side {
				graphs,
				io,
				w,
				m,
				b,
				npu,
			};
			sc.spawn(move || {
				let side = side;
				let Side {
					graphs,
					io,
					w,
					m,
					b,
					npu,
				} = side;
				let mut pend: std::collections::VecDeque<(
					usize,
					usize,
					usize,
					npu::ze::Batch<'static>,
				)> = Default::default();
				let mut open = true;
				loop {
					// take what was sent; wait for the oldest job when nothing is
					let msg = if pend.is_empty() {
						if !open {
							break;
						}
						match rx.recv() {
							Ok(m) => Some(m),
							Err(_) => break,
						}
					} else if pend.len() >= MAX_LISTS || !open {
						None
					} else {
						match rx.try_recv() {
							Ok(m) => Some(m),
							Err(std::sync::mpsc::TryRecvError::Empty) => None,
							Err(std::sync::mpsc::TryRecvError::Disconnected) => {
								open = false;
								None
							}
						}
					};
					match msg {
						Some((kind, l, c, cur)) => {
							let cc = chn[c];
							let k = if cc.small { kind + 3 } else { kind };
							let ios = &io[k];
							let r = npu
								.batch_each(&mut graphs[k], 1, &mut |g, _| {
									let (xr, ir, kr) =
										(cc.r0 * d * 2, cc.r0 * inner * 2, cc.r0 * kv * 2);
									for (i, x) in ios.iter().enumerate() {
										match *x {
											Io::X => g.bind_at(i, &b.x[cur], xr)?,
											Io::Xo => g.bind_at(i, &b.x[1 - cur], xr)?,
											Io::A => g.bind_at(i, &b.a, ir)?,
											Io::Q => g.bind_at(i, &b.q, ir)?,
											Io::K => g.bind_at(i, &b.k, kr)?,
											Io::V => g.bind_at(i, &b.v, kr)?,
											Io::M(pre, p) => g.bind_at(
												i,
												&m[if pre { l } else { l - 1 }],
												p * ms,
											)?,
											Io::W(pre, wi) => {
												g.bind(i, &w[if pre { l } else { l - 1 }][wi])?
											}
										}
									}
									Ok(())
								})
								.and_then(|mut bt| bt.submit().map(|_| bt));
							match r {
								Ok(bt) => pend.push_back((kind, l, c, bt)),
								Err(e) => {
									let _ = dtx.send(Ev::Err(e));
									return;
								}
							}
						}
						None => {
							let Some((kind, l, c, mut bt)) = pend.pop_front() else {
								continue;
							};
							let ev = match bt.wait() {
								Ok(_) => Ev::Done(kind, l, c),
								Err(e) => Ev::Err(e),
							};
							let stop = matches!(ev, Ev::Err(_));
							if dtx.send(ev).is_err() || stop {
								return;
							}
						}
					}
				}
			});
			// the iGPU's jobs, one at a time
			#[cfg(feature = "vulkan")]
			let gtx = gb.as_mut().map(|g| GSide { g, w, b }).map(|gs| {
				let (gtx, grx) = std::sync::mpsc::channel::<(usize, usize, usize)>();
				let dtx = done_tx.clone();
				let gw = &gw;
				sc.spawn(move || {
					let gs = gs;
					let GSide { g, w, b } = gs;
					let mut failed = false;
					while let Ok((l, c, cur)) = grx.recv() {
						if failed {
							if dtx.send(Ev::GpuErr(l, c, cur, String::new())).is_err() {
								return;
							}
							continue;
						}
						let cc = chn[c];
						let (wp, wq) = (&w[l - 1], &w[l]);
						let r = g
							.mid_begin(&offload::H3Mid {
								l,
								x: b.x[cur].as_slice(),
								a: b.a.as_slice(),
								xo: b.x[1 - cur].as_slice(),
								q: b.q.as_slice(),
								k: b.k.as_slice(),
								v: b.v.as_slice(),
								r0: cc.r0,
								r1: cc.r0 + cc.rows(),
								w_o: wp[gw[0]].as_slice(),
								w_gu: wp[gw[1]].as_slice(),
								w_d: wp[gw[2]].as_slice(),
								w_qkv: wq[gw[3]].as_slice(),
								slot: 0,
							})
							.and_then(|_| g.end());
						let ev = match r {
							Ok(()) => Ev::Done(MID, l, c),
							Err(e) => {
								failed = true;
								Ev::GpuErr(l, c, cur, e)
							}
						};
						if dtx.send(ev).is_err() {
							return;
						}
					}
				});
				gtx
			});
			drop(done_tx);
			// per chunk: the residual's buffer, finished
			let mut cur = vec![0usize; nc];
			let mut fin_c = vec![false; nc];
			// per layer: the compact finished K and V, chunks with keys, the next chunk to attend
			let mut kc: Vec<Vec<u16>> = vec![Vec::new(); layers];
			let mut vc: Vec<Vec<u16>> = vec![Vec::new(); layers];
			let mut have = vec![vec![false; nc]; layers];
			let mut next = vec![0usize; layers];
			let mut qf = vec![0f32; S * inner];
			let mut af = vec![0f32; S * inner];
			let gpu_dead = std::cell::Cell::new(false);
			let send =
				|kind: usize, l: usize, c: usize, cur: usize, gpu: bool| -> Result<(), String> {
					#[cfg(feature = "vulkan")]
					if gpu && kind == MID && !gpu_dead.get() {
						if let Some(t) = gtx.as_ref() {
							if t.send((l, c, cur)).is_ok() {
								return Ok(());
							}
							gpu_dead.set(true);
						}
					}
					let _ = gpu;
					tx.send((kind, l, c, cur)).map_err(|_| gone())
				};
			for c in 0..nc {
				send(PRE, 0, c, 0, false)?;
			}
			let mut left = nc;
			while left > 0 {
				let t = std::time::Instant::now();
				let ev = done_rx.recv().map_err(|_| gone())?;
				t_wait += t.elapsed().as_secs_f64();
				let t = std::time::Instant::now();
				let (kind, l, c) = match ev {
					Ev::Done(k, l, c) => (k, l, c),
					Ev::Err(e) => return Err(e),
					Ev::GpuErr(l, c, cur, e) => {
						if !gpu_dead.replace(true) {
							eprintln!("[h3] the iGPU failed ({e}): the NPU runs the text encoder's jobs from here on");
						}
						tx.send((MID, l, c, cur)).map_err(|_| gone())?;
						continue;
					}
				};
				let cc = ch[c];
				if kind != PRE {
					cur[c] = 1 - cur[c];
				}
				if kind == POST {
					if l == layers {
						fin_c[c] = true;
						left -= 1;
						t_attn += t.elapsed().as_secs_f64();
						continue;
					}
					// layer l - 1 ended: its deepstack features, then layer l up to its attention
					let f = &deep[l - 1];
					let inv = 1.0 / c_scale[l - 1];
					// SAFETY: chunk c's rows of X are touched by no job now
					let xb = unsafe {
						std::slice::from_raw_parts_mut(
							(b.x[cur[c]].ptr() as *mut u16).add(cc.r0 * d),
							cc.n * d,
						)
					};
					let mut row = vec![0f32; d];
					for r in 0..cc.n {
						let j = img_of[cc.r0 + r];
						if j == usize::MAX {
							continue;
						}
						let xr = &mut xb[r * d..(r + 1) * d];
						quant::f16_to_f32_slice(xr, &mut row);
						row.iter_mut()
							.zip(&f[j * d..(j + 1) * d])
							.for_each(|(a, &v)| *a += v * inv);
						quant::f32_to_f16_slice(&row, xr);
					}
					send(PRE, l, c, cur[c], false)?;
					t_attn += t.elapsed().as_secs_f64();
					continue;
				}
				if kind == MID && on_gpu[c] {
					gpu_jobs += 1;
				}
				// layer l's keys and values of chunk c, finished, into the compact copy
				{
					let wq = &qk[l];
					if kc[l].is_empty() {
						kc[l] = vec![0u16; s * kv];
						vc[l] = vec![0u16; s * kv];
					}
					let src = &as_u16(b.k.as_slice())[cc.r0 * kv..(cc.r0 + cc.n) * kv];
					let dp = kc[l].as_mut_ptr() as usize;
					let (cs, sn) = (&cos[cc.r0 * half..], &sin[cc.r0 * half..]);
					const R: usize = 8;
					let (n, r0) = (cc.n, cc.r0);
					quant::par_for(n.div_ceil(R), move |i| {
						let (a, z) = (i * R, ((i + 1) * R).min(n));
						let mut f = vec![0f32; (z - a) * kv];
						quant::f16_to_f32_slice(&src[a * kv..z * kv], &mut f);
						qk_finish(
							&mut f,
							&wq.sk,
							&wq.nk,
							&cs[a * half..],
							&sn[a * half..],
							half,
							hd,
							eps,
						);
						// SAFETY: task i owns compact rows r0 + a .. r0 + z
						let o = unsafe {
							std::slice::from_raw_parts_mut(
								(dp as *mut u16).add((r0 + a) * kv),
								(z - a) * kv,
							)
						};
						quant::f32_to_f16_slice(&f, o);
					});
					vc[l][cc.r0 * kv..(cc.r0 + cc.n) * kv]
						.copy_from_slice(&as_u16(b.v.as_slice())[cc.r0 * kv..(cc.r0 + cc.n) * kv]);
					have[l][c] = true;
				}
				// the attention of every chunk of layer l whose keys are all there
				while next[l] < nc && have[l][next[l]] {
					let c = next[l];
					let Chunk { r0, n, .. } = ch[c];
					let wq = &qk[l];
					let q = &mut qf[..n * inner];
					widen(&as_u16(b.q.as_slice())[r0 * inner..(r0 + n) * inner], q);
					qk_finish(
						q,
						&wq.sq,
						&wq.nq,
						&cos[r0 * half..],
						&sin[r0 * half..],
						half,
						hd,
						eps,
					);
					let o = &mut af[..n * inner];
					quant::attn_causal(
						q,
						&kc[l][..(r0 + n) * kv],
						&vc[l][..(r0 + n) * kv],
						n,
						r0,
						heads,
						kvh,
						hd,
						scale,
						o,
					);
					// SAFETY: rows r0 .. r0 + n of `a` are read only by the job sent below
					let a = unsafe {
						std::slice::from_raw_parts_mut(
							(b.a.ptr() as *mut u8).add(r0 * inner * 2),
							n * inner * 2,
						)
					};
					narrow(o, a);
					next[l] += 1;
					let (kind, nl) = if l + 1 == layers {
						(POST, layers)
					} else if l < ndeep {
						(POST, l + 1)
					} else {
						(MID, l + 1)
					};
					send(kind, nl, c, cur[c], on_gpu[c])?;
				}
				if next[l] == nc {
					kc[l] = Vec::new();
					vc[l] = Vec::new();
				}
				t_attn += t.elapsed().as_secs_f64();
			}
			drop(tx);
			debug_assert!(fin_c.iter().all(|&f| f));
			// every chunk's residual is in its own buffer; gather them in X[0]
			let x0 = b.x[0].ptr() as *mut u16;
			for (c, cc) in ch.iter().enumerate() {
				if cur[c] == 1 {
					// SAFETY: no job runs; the two buffers do not overlap
					unsafe {
						std::ptr::copy_nonoverlapping(
							(b.x[1].ptr() as *const u16).add(cc.r0 * d),
							x0.add(cc.r0 * d),
							cc.n * d,
						)
					};
				}
			}
			Ok(gpu_dead.get())
		});
		#[cfg(feature = "vulkan")]
		{
			if !matches!(r, Ok(true)) {
				self.gb = gb;
			}
		}
		self.t_wait = t_wait;
		self.t_attn = t_attn;
		self.gpu_jobs = gpu_jobs;
		r?;
		let b = self.bufs.as_ref().unwrap();
		let xb = as_u16(b.x[0].as_slice());
		let cl = self.c[layers - 1];
		let mut out = vec![0f32; s * d];
		widen(&xb[..s * d], &mut out);
		out.iter_mut().for_each(|v| *v *= cl);
		Ok(out)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn chunks_cover_the_rows() {
		for s in [1, 63, 64, 100, 192, 193, 256, 300, 448, 449, 1100, 4096] {
			let c = chunks(s);
			let mut r = 0;
			for x in &c {
				assert_eq!(x.r0, r);
				assert!(x.n > 0 && x.n <= x.rows());
				r += x.rows();
			}
			assert!(r >= s && r - c.last().unwrap().rows() < s, "{s}");
			assert_eq!(c.iter().map(|x| x.n).sum::<usize>(), s);
		}
		assert!(chunks(100).iter().all(|c| c.small));
		assert_eq!(chunks(1100).iter().filter(|c| !c.small).count(), 4);
	}
}
