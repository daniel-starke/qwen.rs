//! The VAE decoder's ViT on the NPU (`image.npu`'s `vae` graphs,
//! `npubuild::h3img`): every 256-pixel tile's `256 + 5` tokens through the
//! 24 layer graphs ([`h3::VB`] tiles a job) and the head graph, f16. The
//! cores make the tokens (`post_quant_conv`, `proj_in`, the registers),
//! turn the head's rows into pixels and blend the tiles. The weights are
//! f16 copies in NPU host memory made at load from the checkpoint, the
//! norms and residual scales folded in (`h3::vae_layer_weights`). A share
//! of the tiles runs on the iGPU at the same time (`offload::H3Vae`,
//! [`gpu_share`]), through every layer and the head, on the same buffers.

use super::vae::{Map, Vae};
use crate::npuhost::npu;
use npu::lock::{NpuLock, SharedNpuLock};
use npu::ze::{Batch, Graph, HostBuf, Npu};
use npubuild::h3img::{self as h3, VShape, VB, VS, VT};
use npubuild::st::Checkpoint;
use std::path::Path;

pub struct NpuVae {
	/// The iGPU's tiles; first, so it drops (with its imports of the NPU
	/// buffers) before them.
	#[cfg(feature = "vulkan")]
	gv: Option<offload::H3Vae>,
	npu: &'static Npu,
	shape: VShape,
	dec: Graph<'static>,
	out: Graph<'static>,
	/// `dec`'s argument indices: x, xo, then the weights in
	/// [`h3::VAE_WNAMES`] order.
	di: Vec<usize>,
	/// `out`'s: x, g, b, w, bo, y.
	oi: Vec<usize>,
	/// Per layer its weights ([`h3::VAE_WNAMES`] order).
	w: Vec<Vec<HostBuf<'static>>>,
	/// The head's: g, b, w, bo.
	ow: Vec<HostBuf<'static>>,
	/// Seconds waiting for the NPU and on the cores since the last report.
	pub t_wait: f64,
	pub t_cores: f64,
	/// Seconds waiting for the iGPU past the NPU, and its tiles of the last
	/// decode.
	pub t_gpu: f64,
	pub gpu_tiles: usize,
	_lock: SharedNpuLock,
}

/// `QWEN_H3_VAE_GPU`: the iGPU's share of the decoder's tiles (in whole NPU
/// jobs of [`VB`] tiles), default 0.3; `0` = the NPU decodes every tile.
#[cfg(feature = "vulkan")]
fn gpu_share() -> f64 {
	std::env::var("QWEN_H3_VAE_GPU")
		.ok()
		.and_then(|v| v.parse().ok())
		.unwrap_or(0.3)
}

/// The iGPU for [`gpu_share`] on the decoder's weights `w` and the head's
/// `ow`.
#[cfg(feature = "vulkan")]
fn gpu_vae(
	c: &VShape,
	w: &[Vec<HostBuf<'static>>],
	ow: &[HostBuf<'static>],
) -> Option<offload::H3Vae> {
	if gpu_share() <= 0.0 {
		return None;
	}
	let (cos, sin) = h3::vae_rope(c);
	let shape = offload::H3VaeShape {
		d: c.d(),
		ff: c.ff,
		s: c.s(),
		out: c.out,
		layers: c.layers,
		eps: (c.eps * (VS as f64) * (VS as f64)) as f32,
		eps_qk: c.eps.max(h3::EPS_G) as f32,
	};
	let lw: Vec<[&[u8]; 8]> = w
		.iter()
		.map(|l| std::array::from_fn(|i| l[i].as_slice()))
		.collect();
	let hw: [&[u8]; 4] = std::array::from_fn(|i| ow[i].as_slice());
	match offload::H3Vae::new(Some("intel"), shape, &cos, &sin, &lw, hw) {
		Ok(g) => Some(g),
		Err(e) => {
			eprintln!("[h3] no iGPU for the VAE decoder ({e}): the NPU decodes every tile");
			None
		}
	}
}

// SAFETY: used from one thread at a time (behind the model's mutex).
unsafe impl Send for NpuVae {}
unsafe impl Sync for NpuVae {}

fn f16_buf(npu: &'static Npu, v: &[f32]) -> Result<HostBuf<'static>, String> {
	let mut hb = npu.alloc(v.len() * 2)?;
	let s = hb.as_mut_slice();
	// SAFETY: host buffers are 4 KiB aligned; len = 2 v.len()
	let d = unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut u16, v.len()) };
	quant::f32_to_f16_slice(v, d);
	Ok(hb)
}

fn arg(g: &Graph<'_>, name: &str, out: bool, f: &Path) -> Result<usize, String> {
	g.args
		.iter()
		.position(|a| a.is_output == out && (out || a.name == name))
		.ok_or(format!("{}: no argument {name}", f.display()))
}

impl NpuVae {
	/// The graphs in `graphs` (`dec.blob`, `out.blob`) and the decoder's
	/// weights from the checkpoint directory `dir`.
	pub fn load(graphs: &Path, dir: &Path) -> Result<NpuVae, String> {
		let lock = NpuLock::shared()?;
		let npu = npu()?;
		let shape = VShape::read(dir)?;
		let ld = |n: &str| -> Result<(Graph<'static>, std::path::PathBuf), String> {
			let f = graphs.join(format!("{n}.blob"));
			let b = std::fs::read(&f).map_err(|e| format!("{}: {e}", f.display()))?;
			let mut g = npu.load(&b, 0)?;
			g.initialize()?;
			Ok((g, f))
		};
		let (dec, fd) = ld("dec")?;
		let (out, fo) = ld("out")?;
		let mut di = vec![arg(&dec, "x", false, &fd)?, arg(&dec, "xo", true, &fd)?];
		for n in h3::VAE_WNAMES {
			di.push(arg(&dec, n, false, &fd)?);
		}
		let mut oi = vec![arg(&out, "x", false, &fo)?];
		for n in ["g", "b", "w", "bo"] {
			oi.push(arg(&out, n, false, &fo)?);
		}
		oi.push(arg(&out, "y", true, &fo)?);
		let ck = Checkpoint::open_dir(&dir.join("vae_diffusers"))?;
		let mut w = Vec::with_capacity(shape.layers);
		for l in 0..shape.layers {
			let v = h3::vae_layer_weights(&ck, &shape, l)?;
			let bufs = v
				.iter()
				.map(|x| f16_buf(npu, x))
				.collect::<Result<Vec<_>, _>>()?;
			for (i, b) in bufs.iter().enumerate() {
				if b.len() != dec.args[di[2 + i]].bytes {
					return Err(format!(
						"{}: {} has {} bytes, the graph wants {}",
						fd.display(),
						h3::VAE_WNAMES[i],
						b.len(),
						dec.args[di[2 + i]].bytes
					));
				}
			}
			w.push(bufs);
		}
		let d = shape.d();
		let ow = vec![
			f16_buf(npu, &ck.vec("decoder.norm_out.weight", d)?)?,
			f16_buf(npu, &ck.vec("decoder.norm_out.bias", d)?)?,
			f16_buf(npu, &ck.mat("decoder.proj_out.weight", shape.out, d)?)?,
			f16_buf(npu, &ck.vec("decoder.proj_out.bias", shape.out)?)?,
		];
		#[cfg(feature = "vulkan")]
		let gv = gpu_vae(&shape, &w, &ow);
		Ok(NpuVae {
			#[cfg(feature = "vulkan")]
			gv,
			npu,
			shape,
			dec,
			out,
			di,
			oi,
			w,
			ow,
			t_wait: 0.0,
			t_cores: 0.0,
			t_gpu: 0.0,
			gpu_tiles: 0,
			_lock: lock,
		})
	}

	/// Bytes of the weights in NPU host memory.
	pub fn resident_bytes(&self) -> usize {
		self.w
			.iter()
			.flatten()
			.chain(&self.ow)
			.map(|b| b.len())
			.sum()
	}

	/// Decode denormalized latents `[latent][h][w]` like [`Vae::decode`];
	/// `None` when a tile is not the graphs' 16 x 16 latents (a canvas side
	/// under 256 pixels: the cores decode it).
	#[allow(clippy::type_complexity)]
	pub fn decode(
		&mut self,
		vae: &Vae,
		z: &[f32],
		h: usize,
		w: usize,
	) -> Result<Option<(Vec<f32>, usize, usize)>, String> {
		let (tiles, per_row, size, ov) = vae.tiles(z, h, w);
		if tiles.iter().any(|m| (m.h, m.w) != (VT, VT)) {
			return Ok(None);
		}
		let t = std::time::Instant::now();
		let c = self.shape;
		let (d, s, tok, outw) = (c.d(), c.s(), VT * VT, c.out);
		let jobs = tiles.len().div_ceil(VB);
		let npu = self.npu;
		let alloc = |n: usize| -> Result<HostBuf<'static>, String> {
			let mut b = npu.alloc(n)?;
			b.as_mut_slice().fill(0);
			Ok(b)
		};
		let xb = [alloc(jobs * VB * s * d * 2)?, alloc(jobs * VB * s * d * 2)?];
		let yb = alloc(jobs * VB * tok * outw * 2)?;
		let dw = vae.dec_weights_in()?;
		let xp = xb[0].ptr() as usize;
		let tiles_ref: &[Map] = &tiles;
		// one tile at a time: `tile_tokens` runs its GEMM on the pool
		(0..tiles.len()).for_each(|i| {
			let mut noop = |_: &str, _: &[f32]| {};
			let mut x = vae.tile_tokens(&dw, &tiles_ref[i], &mut noop);
			x.iter_mut().for_each(|v| *v *= VS);
			// SAFETY: task i owns tile i's rows
			let dst =
				unsafe { std::slice::from_raw_parts_mut((xp as *mut u16).add(i * s * d), s * d) };
			quant::f32_to_f16_slice(&x, dst);
		});
		self.t_cores += t.elapsed().as_secs_f64();
		let t = std::time::Instant::now();
		// the iGPU's tiles: the first `js` jobs' (all its submissions at once)
		#[cfg(feature = "vulkan")]
		let js = match self.gv.as_mut() {
			Some(g) => {
				let js = ((jobs as f64 * gpu_share()).round() as usize).min(jobs);
				if js > 0 {
					g.decode_begin(
						xb[0].as_slice(),
						yb.as_slice(),
						0,
						(js * VB).min(tiles.len()),
					)?;
				}
				js
			}
			None => 0,
		};
		#[cfg(not(feature = "vulkan"))]
		let js = 0usize;
		self.gpu_tiles = (js * VB).min(tiles.len());
		// lists of at most JOBS executions, at most ALIVE executions recorded
		// at once (the driver's descriptor memory: 0x70000002 beyond)
		const JOBS: usize = 32;
		const ALIVE: usize = 64;
		let mut pend: std::collections::VecDeque<(usize, Batch<'static>)> = Default::default();
		let mut alive = 0;
		let (di, oi) = (&self.di, &self.oi);
		let ow = &self.ow;
		let dbg = std::env::var("QWEN_IMAGE_TIMING").is_ok_and(|v| v == "1")
			&& std::env::var_os("QWEN_IMAGE_VAE_LOG").is_some();
		// (the iGPU is waited for before an error returns: it reads the buffers)
		let r = (|| -> Result<(), String> {
			for l in 0..=c.layers {
				for j0 in (js..jobs).step_by(JOBS) {
					let nj = JOBS.min(jobs - j0);
					while alive + nj > ALIVE {
						let (n, mut bt) = pend.pop_front().unwrap();
						bt.wait()?;
						alive -= n;
					}
					let mut bt = if l < c.layers {
						let (src, dst, wl) = (&xb[l % 2], &xb[(l + 1) % 2], &self.w[l]);
						npu.batch_each(&mut self.dec, nj, &mut |g, j| {
							let o = (j0 + j) * VB * s * d * 2;
							g.bind_at(di[0], src, o)?;
							g.bind_at(di[1], dst, o)?;
							for (k, b) in wl.iter().enumerate() {
								g.bind(di[2 + k], b)?;
							}
							Ok(())
						})?
					} else {
						let xl = &xb[c.layers % 2];
						npu.batch_each(&mut self.out, nj, &mut |g, j| {
							g.bind_at(oi[0], xl, (j0 + j) * VB * s * d * 2)?;
							for k in 0..4 {
								g.bind(oi[1 + k], &ow[k])?;
							}
							g.bind_at(oi[5], &yb, (j0 + j) * VB * tok * outw * 2)?;
							Ok(())
						})?
					};
					bt.submit()?;
					pend.push_back((nj, bt));
					alive += nj;
				}
				if dbg {
					eprintln!(
						"[h3 vae] layer {l} submitted at {:.2} s",
						t.elapsed().as_secs_f64()
					);
				}
			}
			for (_, mut bt) in pend {
				bt.wait()?;
			}
			Ok(())
		})();
		self.t_wait += t.elapsed().as_secs_f64();
		#[cfg(feature = "vulkan")]
		if let Some(g) = self.gv.as_mut() {
			let t = std::time::Instant::now();
			let rg = g.end();
			// the pass' buffers are freed at the end of this call
			g.forget_rows();
			rg?;
			self.t_gpu += t.elapsed().as_secs_f64();
		}
		r?;
		let t = std::time::Instant::now();
		let yv = yb.as_slice();
		// SAFETY: host buffers are 4 KiB aligned
		let y = unsafe { std::slice::from_raw_parts(yv.as_ptr() as *const u16, yv.len() / 2) };
		let imgs: Vec<std::sync::Mutex<Vec<f32>>> =
			(0..tiles.len()).map(|_| Default::default()).collect();
		quant::par_for(tiles.len(), |i| {
			let mut o = vec![0f32; tok * outw];
			quant::f16_to_f32_slice(&y[i * tok * outw..(i + 1) * tok * outw], &mut o);
			*imgs[i].lock().unwrap() = vae.tile_pixels(&o, VT, VT);
		});
		let imgs: Vec<Vec<f32>> = imgs.into_iter().map(|m| m.into_inner().unwrap()).collect();
		let r = Vae::blend(imgs, per_row, size, &ov);
		self.t_cores += t.elapsed().as_secs_f64();
		Ok(Some(r))
	}
}
