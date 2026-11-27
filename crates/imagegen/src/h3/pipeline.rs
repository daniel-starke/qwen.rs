//! Text to image and edits with MiniMax-H3-Image: references through the
//! VAE encoder (a posterior sample each) and the text encoder's vision
//! tower, the text encoder, the token refiner once, 6 transformer passes
//! over the whole packed sequence (one output head per step), the tiled VAE
//! decoder, and the Lanczos resize back to the requested size when the
//! canvas is larger.
//!
//! The model held between jobs ([`Model`]): the transformer resident
//! (quantized matmuls, the AdaLN tables) and the text encoder resident
//! while free memory allows: after encoding, a job drops it when the
//! available physical memory would fall below `QWEN_IMAGE_TE_RESERVE_MB`
//! (default 8192) during the denoising, and the next job loads it again,
//! from the weight cache when there is one ([`crate::wcache`]). The vision
//! tower and the VAE are read from the checkpoint per job.

use super::config::{bound_short_edge, canvas, Config, PIXEL_MEAN, PIXEL_STD, REF_T};
use super::dit::{Dit, Rope};
use super::layout;
use super::npudit::{NpuCfg, NpuDit};
use super::npute::NpuTe;
use super::npuvae::NpuVae;
use super::vae::{to_rgb8, Vae};
use crate::cond::Cond;
use crate::job::{ImageSrc, Options, Output};
use crate::ops::WFmt;
use crate::rng::TorchGen;
use crate::te::TextEncoder;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// The pipeline's default `min_generation_area` (4 MP): a smaller request
/// is generated on a canvas of this area and resized back.
pub const MIN_AREA: usize = 2048 * 2048;

/// The largest canvas a job may ask for (the model is trained up to ~20 MP).
pub const MAX_AREA: usize = 4096 * 4096;

/// The canvas floor in pixels when not 0 (the NPU build's calibration
/// images; `QWEN_IMAGE_MIN_AREA` likewise).
pub static MIN_AREA_OVERRIDE: AtomicUsize = AtomicUsize::new(0);

/// Most references an edit takes.
pub const MAX_IMAGES: usize = 16;

/// The default canvas side (2048 x 2048 = [`MIN_AREA`]).
pub const DEFAULT_SIDE: usize = 2048;

/// Short-edge bounds of a reference for the VAE and for the vision tower.
const VAE_EDGE: (usize, usize) = (1024, 2048);
const VISION_EDGE: (usize, usize) = (1024, 1024);

pub struct Model {
	pub cfg: Config,
	pub dir: PathBuf,
	pub fmt: WFmt,
	te: Mutex<Option<TextEncoder>>,
	/// The text encoder's weight cache file.
	te_file: Option<PathBuf>,
	/// The text encoder on the NPU (an `image.npu` with `te`; `te` above
	/// then holds only the tokenizer and the embeddings): its weights' and
	/// graphs' directories, the engine while loaded.
	npu_te: Option<(PathBuf, PathBuf, Mutex<Option<NpuTe>>)>,
	dit: Dit,
	/// The transformer's blocks on the NPU (an `image.npu`).
	npu: Option<Mutex<NpuDit>>,
	/// The VAE decoder on the NPU (an `image.npu` with `vae`), loaded on
	/// the first decode and dropped by [`Model::release_devices`].
	npu_vae: Option<(PathBuf, Mutex<Option<NpuVae>>)>,
	pub load_secs: f64,
}

/// One reference after the pipeline's preparation.
pub struct Reference {
	/// RGB `(w, h, bytes)` resized for the VAE (cropped to whole VAE cells
	/// by [`encode_reference`]).
	pub vae: (usize, usize, Vec<u8>),
	/// The vision tower's copy.
	pub vision: Cond,
}

impl Reference {
	pub fn new(src: &ImageSrc) -> Result<Reference, String> {
		let (w, h, rgba) = match src {
			ImageSrc::Path(p) => imgio::load_rgba(&p.to_string_lossy())
				.map_err(|e| format!("{}: {e}", p.display()))?,
			ImageSrc::Rgba(w, h, px) => (*w, *h, px.clone()),
		};
		if w == 0 || h == 0 || rgba.len() != w * h * 4 {
			return Err(format!(
				"reference image: {w}x{h} with {} bytes",
				rgba.len()
			));
		}
		// PIL's convert("RGB") drops the alpha channel
		let rgb: Vec<u8> = rgba
			.chunks_exact(4)
			.flat_map(|p| [p[0], p[1], p[2]])
			.collect();
		let resize = |(lo, hi): (usize, usize)| {
			let (nw, nh) = bound_short_edge(w, h, lo, hi);
			let px = if (nw, nh) == (w, h) {
				rgb.clone()
			} else {
				imgio::pil::resample(&rgb, 3, w, h, nw, nh, imgio::pil::Filter::Lanczos)
			};
			(nw, nh, px)
		};
		let vae = resize(VAE_EDGE);
		let (vw, vh, vpx) = resize(VISION_EDGE);
		let rgba: Vec<u8> = vpx
			.chunks_exact(3)
			.flat_map(|p| [p[0], p[1], p[2], 255])
			.collect();
		Ok(Reference {
			vae,
			vision: Cond {
				orig: (w, h),
				w: vw,
				h: vh,
				rgba,
			},
		})
	}
}

/// Bytes of physical memory available now (free + standby).
#[cfg(windows)]
pub fn avail_bytes() -> u64 {
	#[repr(C)]
	struct Ms {
		len: u32,
		load: u32,
		total: u64,
		avail: u64,
		total_pf: u64,
		avail_pf: u64,
		total_v: u64,
		avail_v: u64,
		avail_ext: u64,
	}
	extern "system" {
		fn GlobalMemoryStatusEx(m: *mut Ms) -> i32;
	}
	let mut m = Ms {
		len: std::mem::size_of::<Ms>() as u32,
		load: 0,
		total: 0,
		avail: 0,
		total_pf: 0,
		avail_pf: 0,
		total_v: 0,
		avail_v: 0,
		avail_ext: 0,
	};
	// SAFETY: a plain query into a struct of the documented layout
	if unsafe { GlobalMemoryStatusEx(&mut m) } == 0 {
		return u64::MAX;
	}
	m.avail
}

#[cfg(not(windows))]
pub fn avail_bytes() -> u64 {
	u64::MAX
}

fn timing() -> bool {
	std::env::var("QWEN_IMAGE_TIMING").is_ok_and(|v| v == "1")
}

impl Model {
	/// True when `path` is a MiniMax-H3-Image checkpoint or its
	/// `image.npu`.
	pub fn is_model(path: &Path) -> bool {
		super::config::is_checkpoint(path) || NpuCfg::is_h3_npu(path)
	}

	/// Free the NPU graphs and the VAE decoder's weights between jobs,
	/// keeping the transformer's weights: a chat engine on the NPU shares
	/// the driver's memory. The next job loads them again.
	pub fn release_devices(&self) {
		if let Some((_, m)) = &self.npu_vae {
			*m.lock().unwrap_or_else(|e| e.into_inner()) = None;
		}
		if let Some(n) = &self.npu {
			n.lock().unwrap_or_else(|e| e.into_inner()).unload_graphs();
		}
		if let Some((_, _, n)) = &self.npu_te {
			if let Some(n) = n.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
				n.unload_graphs();
			}
		}
	}

	/// The transformer (only its small weights on the NPU backend).
	pub fn dit(&self) -> &Dit {
		&self.dit
	}

	/// Load the checkpoint directory or an `image.npu` (the transformer's
	/// blocks and the VAE decoder on the NPU, the rest as the CPU backend):
	/// the text encoder and the transformer converted to `fmt` (or read from
	/// the weight cache), the AdaLN tables. `WFmt::F32` keeps nothing (the
	/// reference: every layer converted from BF16 per use).
	/// `progress(stage, step, steps)` per layer.
	pub fn load(
		path: &Path,
		fmt: WFmt,
		mut progress: impl FnMut(&str, usize, usize),
	) -> Result<Model, String> {
		let t = Instant::now();
		let nc = if NpuCfg::is_h3_npu(path) {
			Some(NpuCfg::parse(path)?)
		} else {
			None
		};
		let dir = nc.as_ref().map_or(path, |c| c.checkpoint.as_path());
		let cfg = Config::load(dir)?;
		let cache = crate::wcache::dir();
		let tag = format!("{fmt:?}").to_ascii_lowercase();
		let te_file = cache.as_ref().map(|d| d.join(format!("h3-te-{tag}.bin")));
		let dit_file = cache.as_ref().map(|d| d.join(format!("h3-dit-{tag}.bin")));
		// QWEN_H3_TE_NPU=0: the cores' text encoder with an `image.npu` that has one
		let npu_te = nc
			.as_ref()
			.and_then(|c| c.te.clone().zip(c.te_graphs.clone()))
			.filter(|_| std::env::var("QWEN_H3_TE_NPU").map_or(true, |v| v != "0"));
		let (te, npu_te) = match npu_te {
			Some((w, g)) => {
				let n = NpuTe::load(&w, &g, &cfg.te, |l| {
					progress("load text encoder (NPU)", l + 1, cfg.te.layers)
				})?;
				(
					TextEncoder::open(dir, &cfg.te)?,
					Some((w, g, Mutex::new(Some(n)))),
				)
			}
			None => (
				TextEncoder::load_cached(dir, &cfg.te, fmt, te_file.as_deref(), |l| {
					progress("load text encoder", l + 1, cfg.te.layers)
				})?,
				None,
			),
		};
		let (ts, _) = cfg.schedule();
		let (dit, npu) = match &nc {
			Some(c) => (
				Dit::load_small(dir, &cfg.dit, fmt, &ts)?,
				Some(Mutex::new(NpuDit::load(c, &cfg.dit, |l| {
					progress("load transformer (NPU)", l + 1, cfg.dit.layers)
				})?)),
			),
			None => (
				Dit::load(dir, &cfg.dit, fmt, &ts, dit_file.as_deref(), |l| {
					progress("load transformer", l + 1, cfg.dit.layers)
				})?,
				None,
			),
		};
		let npu_vae = nc
			.as_ref()
			.and_then(|c| c.vae.clone())
			.map(|g| (g, Mutex::new(None)));
		Ok(Model {
			dir: dir.to_path_buf(),
			fmt,
			te: Mutex::new(Some(te)),
			te_file,
			npu_te,
			dit,
			npu,
			npu_vae,
			load_secs: t.elapsed().as_secs_f64(),
			cfg,
		})
	}

	/// Bytes of the resident weights (text encoder, transformer).
	pub fn resident_bytes(&self) -> (usize, usize) {
		let te = self
			.te
			.lock()
			.unwrap_or_else(|e| e.into_inner())
			.as_ref()
			.map_or(0, |t| t.resident_bytes())
			+ self.npu_te.as_ref().map_or(0, |(_, _, n)| {
				n.lock()
					.unwrap_or_else(|e| e.into_inner())
					.as_ref()
					.map_or(0, |n| n.resident_bytes())
			});
		let npu = self.npu.as_ref().map_or(0, |n| {
			n.lock().unwrap_or_else(|e| e.into_inner()).resident_bytes()
		});
		let vae = self.npu_vae.as_ref().map_or(0, |(_, m)| {
			m.lock()
				.unwrap_or_else(|e| e.into_inner())
				.as_ref()
				.map_or(0, |v| v.resident_bytes())
		});
		(te, self.dit.resident_bytes() + npu + vae)
	}

	pub fn backend(&self) -> &'static str {
		if self.npu.is_some() {
			"NPU (int8) + cores"
		} else {
			"cores"
		}
	}

	/// The default size of a job: [`DEFAULT_SIDE`] square, or the last
	/// reference's aspect at [`MIN_AREA`] for an edit.
	pub fn default_size(&self, images: &[ImageSrc]) -> Result<(usize, usize), String> {
		let Some(last) = images.last() else {
			return Ok((DEFAULT_SIDE, DEFAULT_SIDE));
		};
		let (w, h) = match last {
			ImageSrc::Path(p) => {
				let (w, h, _) = imgio::load_rgba(&p.to_string_lossy())
					.map_err(|e| format!("{}: {e}", p.display()))?;
				(w, h)
			}
			ImageSrc::Rgba(w, h, _) => (*w, *h),
		};
		Ok(crate::cond::dims(MIN_AREA as f64, w as f64 / h as f64))
	}

	/// Generate one image. `progress(stage, step, steps)` before each
	/// denoising step and stage.
	pub fn generate(
		&self,
		o: &Options,
		mut progress: impl FnMut(&str, usize, usize),
	) -> Result<Output, String> {
		let (cfg, dir) = (&self.cfg, self.dir.as_path());
		let c = &cfg.dit;
		let canceled = || {
			if o.cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed)) {
				Err("canceled".to_string())
			} else {
				Ok(())
			}
		};
		if o.images.len() > MAX_IMAGES {
			return Err(format!(
				"{} reference images: an edit takes at most {MAX_IMAGES}",
				o.images.len()
			));
		}
		if o.steps != 0 && o.steps != c.pdd_heads {
			return Err(format!(
				"steps {}: MiniMax-H3-Image runs a fixed {} steps (one output head each)",
				o.steps, c.pdd_heads
			));
		}
		if o.noise.is_some() {
			return Err("MiniMax-H3-Image: explicit initial noise is not supported".into());
		}
		let refs: Vec<Reference> = o
			.images
			.iter()
			.map(Reference::new)
			.collect::<Result<_, _>>()?;
		let (width, height) = match (o.width, o.height) {
			(0, 0) => self.default_size(&o.images)?,
			(w, h) => (w, h),
		};
		// QWEN_IMAGE_MIN_AREA: the canvas floor in pixels (the reference dumps' small canvases)
		let min_area = match MIN_AREA_OVERRIDE.load(Ordering::Relaxed) {
			0 => std::env::var("QWEN_IMAGE_MIN_AREA")
				.ok()
				.and_then(|v| v.parse::<usize>().ok())
				.map_or(MIN_AREA, |a| a.clamp(1024, MAX_AREA)),
			a => a.clamp(1024, MAX_AREA),
		};
		let (ch, cw) = canvas(height, width, min_area);
		if ch * cw > MAX_AREA {
			return Err(format!("size {width}x{height}: at most {MAX_AREA} pixels"));
		}
		let ratio = cfg.vae.ratio();
		let (lh, lw) = (ch / ratio, cw / ratio);
		let mut gen = TorchGen::new(o.seed);

		// the text encoder, with the references' vision rows
		let t0 = Instant::now();
		let mut te = self.te.lock().unwrap_or_else(|e| e.into_inner());
		if te.is_none() {
			*te = Some(if self.npu_te.is_some() {
				TextEncoder::open(dir, &cfg.te)?
			} else {
				TextEncoder::load_cached(dir, &cfg.te, self.fmt, self.te_file.as_deref(), |l| {
					progress("load text encoder", l + 1, cfg.te.layers)
				})?
			});
		}
		let tenc = te.as_ref().unwrap();
		progress("text encoder", 0, 0);
		let vis = if refs.is_empty() {
			None
		} else {
			let conds: Vec<Cond> = refs
				.iter()
				.map(|r| Cond {
					orig: r.vision.orig,
					w: r.vision.w,
					h: r.vision.h,
					rgba: r.vision.rgba.clone(),
				})
				.collect();
			Some(crate::cond::vision_rows(dir, &tenc.ck, &conds, None)?)
		};
		let pads: Vec<usize> = vis
			.as_ref()
			.map_or_else(Vec::new, |v| v.grids.iter().map(|&(r, c)| r * c).collect());
		let (ids, tags) = super::te::presentation(&tenc.tok, &o.prompt, &pads)?;
		let hidden = match &self.npu_te {
			Some((wd, gd, m)) => {
				let mut g = m.lock().unwrap_or_else(|e| e.into_inner());
				if g.is_none() {
					*g = Some(NpuTe::load(wd, gd, &cfg.te, |l| {
						progress("load text encoder (NPU)", l + 1, cfg.te.layers)
					})?);
				}
				let n = g.as_mut().unwrap();
				let grids: &[(usize, usize)] = vis.as_ref().map_or(&[], |v| &v.grids);
				let pos = tenc.positions(&ids, grids)?;
				let (x, img) = tenc.inputs(&ids, vis.as_ref())?;
				let (cos, sin) = tenc.rope_tables(&pos);
				let deep: &[Vec<f32>] = vis.as_ref().map_or(&[], |v| &v.deep);
				let t = Instant::now();
				let h = n.encode(&x, &cos, &sin, &img, deep);
				// a used graph holds NPU driver memory every graph of the
				// process shares until it is unloaded: with the encoder's
				// and the transformer's both held, the VAE decoder's
				// appends fail (`0x70000002`); the next encode loads them
				n.unload_graphs();
				let h = h?;
				if timing() {
					eprintln!(
						"[te] NPU text encoder: {} rows in {:.2} s (device waits {:.2} s, attention and keys {:.2} s, iGPU jobs {})",
						ids.len(),
						t.elapsed().as_secs_f64(),
						n.t_wait,
						n.t_attn,
						n.gpu_jobs
					);
				}
				h
			}
			None => tenc.encode_ids(&ids, vis.as_ref(), &mut |_, _| {})?,
		};
		drop(vis);
		canceled()?;
		let t_te = t0.elapsed().as_secs_f64();

		// the references' latents (posterior samples, in order, off the job's generator)
		let t1 = Instant::now();
		let mut ref_lat: Vec<(Vec<f32>, usize, usize)> = Vec::with_capacity(refs.len());
		if !refs.is_empty() {
			progress("vae encoder", 0, 0);
			let vae = Vae::open(dir, &cfg.vae)?;
			for r in &refs {
				ref_lat.push(encode_reference(&vae, cfg, r, &mut gen)?);
				canceled()?;
			}
		}
		drop(refs);
		let t_venc = t1.elapsed().as_secs_f64();

		let t2 = Instant::now();
		let shapes: Vec<(usize, usize)> = ref_lat.iter().map(|&(_, h, w)| (h, w)).collect();
		let lay = layout::build(c, &tags, &shapes, (lh, lw));
		let rope = Rope::new(c, &lay.pos);
		let noise = gen.randn(c.in_ch * lh * lw);
		let target = layout::patchify(&noise, c.in_ch, lh, lw, c.patch);
		let mut rows = Vec::with_capacity((lay.refs + lay.target) * c.patch_dim());
		for (lat, h, w) in &ref_lat {
			let n = gen.randn(lat.len());
			let t = REF_T;
			let noised: Vec<f32> = lat
				.iter()
				.zip(&n)
				.map(|(&x, &e)| t * x + (1.0 - t) * e)
				.collect();
			rows.extend(layout::patchify(&noised, c.in_ch, *h, *w, c.patch));
		}
		drop(ref_lat);
		let r0 = rows.len();
		rows.extend(target);

		// keep the text encoder while the denoising leaves enough memory
		let need = (lay.len() * (6 * c.hidden + 4 * c.inner()) * 4) as u64 + (1 << 30);
		let reserve = std::env::var("QWEN_IMAGE_TE_RESERVE_MB")
			.ok()
			.and_then(|v| v.parse::<u64>().ok())
			.unwrap_or(8192)
			<< 20;
		let avail = avail_bytes();
		let drop_te = self.fmt != WFmt::F32 && avail < need + reserve;
		if timing() {
			eprintln!(
				"[h3] {} rows ({} text, {} reference, {} target); available {:.1} GB, need {:.1} GB{}",
				lay.len(),
				lay.text,
				lay.refs,
				lay.target,
				avail as f64 / 1e9,
				need as f64 / 1e9,
				if drop_te { ": text encoder dropped" } else { "" }
			);
		}
		if drop_te || self.fmt == WFmt::F32 {
			*te = None;
			if let Some((_, _, n)) = &self.npu_te {
				*n.lock().unwrap_or_else(|e| e.into_inner()) = None;
			}
		}
		drop(te);
		if drop_te {
			quant::alloc::trim();
		}

		let dit = &self.dit;
		let txt = dit.text_rows(&hidden, |_, _| {})?;
		drop(hidden);
		let (ts, sg) = cfg.schedule();
		let mut npu = self
			.npu
			.as_ref()
			.map(|n| n.lock().unwrap_or_else(|e| e.into_inner()));
		if let Some(n) = npu.as_mut() {
			n.begin(&lay, &rope)?;
			(n.t_npu, n.t_attn, n.t_bound, n.t_gpu, n.t_gtx) = (0.0, 0.0, 0.0, 0.0, 0.0);
		}
		let r = (|| {
			for (i, &tt) in ts.iter().enumerate() {
				canceled()?;
				progress("denoise", i + 1, ts.len());
				let ti = Instant::now();
				let v = match npu.as_mut() {
					Some(n) => {
						// the target rows' velocity
						let mut x = Vec::with_capacity(lay.len() * c.hidden);
						x.extend_from_slice(&txt);
						x.extend(dit.image_rows(&rows)?);
						dit.out_at(&n.forward(&x, tt)?, tt, i)?
					}
					None => {
						let (u, tidx) = lay.row_timesteps(tt);
						dit.forward(&lay, &rope, &txt, &rows, &u, &tidx, i, |_, _| {})?
					}
				};
				let dt = sg[i] - sg[i + 1];
				let vt = &v[v.len() - (rows.len() - r0)..];
				rows[r0..]
					.iter_mut()
					.zip(vt)
					.for_each(|(x, &vv)| *x += dt * vv);
				if timing() {
					eprintln!("[h3] step {i}: {:.1} s", ti.elapsed().as_secs_f64());
				}
			}
			Ok::<(), String>(())
		})();
		if let Some(n) = npu.as_mut() {
			if timing() {
				eprintln!(
					"[h3] NPU wait {:.1} s, iGPU wait {:.1} s ({} chunks a block), card wait {:.1} s ({} attention chunks a block), attention and K / V (cores) {:.1} s, block boundaries {:.1} s",
					n.t_npu, n.t_gpu, n.gpu_chunks, n.t_gtx, n.gtx_chunks, n.t_attn, n.t_bound
				);
			}
			n.release_buffers();
			// the graphs' driver memory, for the VAE decoder (as the text
			// encoder's above); the next job's `begin` loads them
			n.unload_graphs();
		}
		drop(npu);
		r?;
		drop(txt);
		let t_dit = t2.elapsed().as_secs_f64();
		canceled()?;

		progress("vae", 0, 0);
		let t3 = Instant::now();
		let mut z = layout::unpatchify(&rows[r0..], c.in_ch, lh, lw, c.patch);
		drop(rows);
		let p = lh * lw;
		for (chn, (&s, &m)) in cfg.vae.std.iter().zip(&cfg.vae.mean).enumerate() {
			z[chn * p..(chn + 1) * p]
				.iter_mut()
				.for_each(|v| *v = *v * s + m);
		}
		let vae = Vae::open(dir, &cfg.vae)?;
		let on_npu = match &self.npu_vae {
			Some((g, m)) => {
				let mut l = m.lock().unwrap_or_else(|e| e.into_inner());
				if l.is_none() {
					*l = Some(NpuVae::load(g, dir)?);
				}
				let n = l.as_mut().unwrap();
				(n.t_wait, n.t_cores, n.t_gpu) = (0.0, 0.0, 0.0);
				let r = n.decode(&vae, &z, lh, lw)?;
				if timing() {
					eprintln!(
						"[h3] VAE decoder: NPU wait {:.1} s, iGPU wait {:.1} s ({} tiles), cores {:.1} s",
						n.t_wait, n.t_gpu, n.gpu_tiles, n.t_cores
					);
				}
				r
			}
			None => None,
		};
		let (img, oh, ow) = match on_npu {
			Some(r) => r,
			None => vae.decode(&z, lh, lw, |_, _| {})?,
		};
		let mut rgb = to_rgb8(&img, oh, ow);
		if (oh, ow) != (height, width) {
			rgb = imgio::pil::resample(&rgb, 3, ow, oh, width, height, imgio::pil::Filter::Lanczos);
		}
		let t_vae = t3.elapsed().as_secs_f64();
		Ok(Output {
			width,
			height,
			rgba: rgb
				.chunks_exact(3)
				.flat_map(|p| [p[0], p[1], p[2], 255])
				.collect(),
			seed: o.seed,
			secs: [t_te, t_venc, t_dit, t_vae],
		})
	}
}

/// A reference's normalized latents `[latent][h][w]` (whole patches): the
/// VAE encoder's posterior sample off `gen`, rounded to f16.
pub fn encode_reference(
	vae: &Vae,
	cfg: &Config,
	r: &Reference,
	gen: &mut TorchGen,
) -> Result<(Vec<f32>, usize, usize), String> {
	let (chw, ph, pw) = reference_pixels(cfg, r);
	let (m, mh, mw) = vae.encode_moments(&chw, ph, pw, |_, _| {})?;
	Ok(sample_latents(cfg, &m, mh, mw, gen))
}

/// The VAE encoder's input of a reference: cropped at the top left to
/// whole VAE cells, ImageNet-normalized, channel-major `[3][h][w]`.
pub fn reference_pixels(cfg: &Config, r: &Reference) -> (Vec<f32>, usize, usize) {
	let ratio = cfg.vae.ratio();
	let (w, h, px) = (r.vae.0, r.vae.1, &r.vae.2);
	let (ph, pw) = (h / ratio * ratio, w / ratio * ratio);
	let mut chw = vec![0f32; 3 * ph * pw];
	for y in 0..ph {
		for x in 0..pw {
			for c in 0..3 {
				chw[(c * ph + y) * pw + x] =
					(px[(y * w + x) * 3 + c] as f32 / 255.0 - PIXEL_MEAN[c]) / PIXEL_STD[c];
			}
		}
	}
	(chw, ph, pw)
}

/// The normalized latents of moments `m` (`[2 latent][mh][mw]`): a
/// posterior sample off `gen` (drawn over the whole latent), rounded to f16,
/// cropped to whole patches.
pub fn sample_latents(
	cfg: &Config,
	m: &[f32],
	mh: usize,
	mw: usize,
	gen: &mut TorchGen,
) -> (Vec<f32>, usize, usize) {
	let z = cfg.vae.latent;
	let n = gen.randn(z * mh * mw);
	let p = cfg.dit.patch;
	let (lh, lw) = (mh / p * p, mw / p * p);
	let mut lat = vec![0f32; z * lh * lw];
	for c in 0..z {
		for y in 0..lh {
			for x in 0..lw {
				let i = (c * mh + y) * mw + x;
				let mean = m[i];
				let logvar = m[z * mh * mw + i].clamp(-30.0, 20.0);
				let s = mean + (0.5 * logvar).exp() * n[i];
				let s = quant::fp16_to_f32(quant::f32_to_fp16(s));
				lat[(c * lh + y) * lw + x] = (s - cfg.vae.mean[c]) / cfg.vae.std[c];
			}
		}
	}
	(lat, lh, lw)
}

/// Load the model in the format `QWEN_IMAGE_WEIGHTS` names (default
/// `q6k`; `f32` = the reference) and generate one image.
pub fn generate(
	dir: &Path,
	o: &Options,
	mut progress: impl FnMut(&str, usize, usize),
) -> Result<Output, String> {
	Model::load(dir, WFmt::from_env(), &mut progress)?.generate(o, progress)
}
