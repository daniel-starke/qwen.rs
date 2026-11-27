//! Text to image and edits: encode the prompt (with the condition images
//! through the vision tower), encode the condition images with the VAE,
//! denoise the latents over the schedule (prefix K / V once, the target
//! rows per step), decode.

use super::cond;
use super::config::Config;
use super::dit::{layout, Dit, Prefix, Rope};
use super::npudit::{NpuCfg, NpuDit};
use super::npuvae::NpuVae;
use super::sched::Schedule;
use super::vae::{to_rgba8, Vae};
use crate::cond::Cond;
use crate::job::{mem_gb, ImageSrc, Options, Output};
use crate::ops::WFmt;
use crate::te::TextEncoder;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Instant;

/// The seed's initial noise as the pipeline draws it (`randn` of shape
/// `(1, 1, z, h, w)`), token-major `[h w][z]`.
pub fn noise(seed: u64, z: usize, h: usize, w: usize) -> Vec<f32> {
	let n = crate::rng::torch_randn(seed, z * h * w);
	let mut t = vec![0f32; n.len()];
	for c in 0..z {
		for p in 0..h * w {
			t[p * z + c] = n[c * h * w + p];
		}
	}
	t
}

/// The model held between jobs: the text encoder's and the transformer's
/// matmuls quantized once (`WFmt::F32`: nothing resident, the M1 / M2
/// reference); the vision tower and the VAE are read from the checkpoint
/// per job (small, and only the decoder runs for text to image).
pub struct Model {
	pub cfg: Config,
	/// The checkpoint directory.
	pub dir: PathBuf,
	pub fmt: WFmt,
	/// The text encoder; dropped for the denoising of an edit whose prefix
	/// needs its memory ([`prefix_plan`]), loaded again by the next job.
	te: std::sync::Mutex<Option<TextEncoder>>,
	dit: Dit,
	/// The transformer's blocks on the NPU (an `image.npu`); `dit` then
	/// holds no blocks (its small weights only).
	npu: Option<std::sync::Mutex<NpuDit>>,
	/// The VAE decoder's convolutions on the NPU (an `image.npu` with
	/// `vae`).
	/// (its graphs directory; `None` while dropped for the denoising of an
	/// edit with many references, loaded again for the decoder).
	npu_vae: Option<(PathBuf, std::sync::Mutex<Option<NpuVae>>)>,
	/// The vision tower's NPU set (an `image.npu` with `vision`), loaded on
	/// the first edit and kept.
	npu_vis: Option<(PathBuf, std::sync::Mutex<VisSlot>)>,
	/// Seconds the load took.
	pub load_secs: f64,
}

/// The NPU vision tower, loaded on the first edit.
struct VisSlot(Option<npu::vis::NpuVision>);

// SAFETY: the NPU objects are used from one thread at a time (behind the
// mutex; the driver's handles are thread safe).
unsafe impl Send for VisSlot {}

impl Model {
	/// Free the NPU graphs between jobs, keeping the transformer's weights:
	/// the transformer's graphs, the VAE decoder and the vision tower hold
	/// driver memory that a chat engine on the NPU shares, and its next
	/// prefill fails beside them (`0x70000002`). The next job loads them
	/// again.
	pub fn release_devices(&self) {
		if let Some((_, m)) = &self.npu_vis {
			m.lock().unwrap_or_else(|e| e.into_inner()).0 = None;
		}
		if let Some((_, m)) = &self.npu_vae {
			*m.lock().unwrap_or_else(|e| e.into_inner()) = None;
		}
		if let Some(n) = &self.npu {
			n.lock().unwrap_or_else(|e| e.into_inner()).unload_graphs();
		}
	}

	/// True when `path` is something [`Self::load`] takes: a checkpoint
	/// directory or an `image.npu`.
	pub fn is_model(path: &Path) -> bool {
		NpuCfg::is_image_npu(path) || npubuild::qimage::is_qwen_image(path)
	}

	/// The checkpoint directory (CPU backend) or an `image.npu` (the
	/// transformer on the NPU, the rest as the CPU backend).
	/// `progress(stage, step, steps)` per layer.
	pub fn load(
		path: &Path,
		fmt: WFmt,
		mut progress: impl FnMut(&str, usize, usize),
	) -> Result<Model, String> {
		let t = Instant::now();
		let nc = if NpuCfg::is_image_npu(path) {
			Some(NpuCfg::parse(path)?)
		} else {
			None
		};
		let dir = nc.as_ref().map_or(path, |c| c.checkpoint.as_path());
		let cfg = Config::load(dir)?;
		let te = TextEncoder::load(dir, &cfg.te, fmt, |l| {
			progress("load text encoder", l + 1, cfg.te.layers)
		})?;
		let (dit, npu) = match &nc {
			Some(c) => (
				Dit::open(dir, &cfg.dit)?,
				Some(std::sync::Mutex::new(NpuDit::load(c, &cfg.dit, |l| {
					progress("load transformer (NPU)", l + 1, cfg.dit.layers)
				})?)),
			),
			None => (
				Dit::load(dir, &cfg.dit, fmt, |l| {
					progress("load transformer", l + 1, cfg.dit.layers)
				})?,
				None,
			),
		};
		let npu_vae = match nc.as_ref().and_then(|c| c.vae.as_ref()) {
			Some(g) => {
				progress("load VAE decoder (NPU)", 0, 0);
				Some((
					g.clone(),
					std::sync::Mutex::new(Some(NpuVae::load(g, dir)?)),
				))
			}
			None => None,
		};
		let npu_vis = nc
			.as_ref()
			.and_then(|c| c.vision.clone())
			.map(|p| (p, std::sync::Mutex::new(VisSlot(None))));
		Ok(Model {
			dir: dir.to_path_buf(),
			fmt,
			te: std::sync::Mutex::new(Some(te)),
			dit,
			npu,
			npu_vae,
			npu_vis,
			load_secs: t.elapsed().as_secs_f64(),
			cfg,
		})
	}

	/// Bytes of the resident weights (text encoder, transformer).
	pub fn resident_bytes(&self) -> (usize, usize) {
		let npu = self
			.npu
			.as_ref()
			.map_or(0, |n| n.lock().unwrap().resident_bytes());
		let vae = self.npu_vae.as_ref().map_or(0, |n| {
			n.1.lock()
				.unwrap()
				.as_ref()
				.map_or(0, |v| v.resident_bytes())
		});
		let te = self
			.te
			.lock()
			.unwrap()
			.as_ref()
			.map_or(0, |t| t.resident_bytes());
		(te, self.dit.resident_bytes() + npu + vae)
	}

	/// The NPU VAE (an `image.npu` with `vae`), loaded if it was dropped.
	fn npu_vae(&self) -> Result<Option<std::sync::MutexGuard<'_, Option<NpuVae>>>, String> {
		let Some((g, m)) = &self.npu_vae else {
			return Ok(None);
		};
		let mut l = m.lock().unwrap();
		if l.is_none() {
			*l = Some(NpuVae::load(g, &self.dir)?);
		}
		Ok(Some(l))
	}

	/// What runs the transformer, for the log.
	pub fn backend(&self) -> &'static str {
		if self.npu.is_some() {
			"NPU (int8) + cores"
		} else {
			"cores"
		}
	}

	/// Generate one image. `progress(stage, step, steps)` is called before
	/// each denoising step and stage.
	pub fn generate(
		&self,
		o: &Options,
		mut progress: impl FnMut(&str, usize, usize),
	) -> Result<Output, String> {
		let (cfg, dir) = (&self.cfg, self.dir.as_path());
		let canceled = || {
			if o.cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed)) {
				Err("canceled".to_string())
			} else {
				Ok(())
			}
		};
		if o.images.len() > MAX_IMAGES {
			return Err(format!(
				"{} condition images: an edit takes at most {MAX_IMAGES}",
				o.images.len()
			));
		}
		let conds: Vec<Cond> = o
			.images
			.iter()
			.map(|src| match src {
				ImageSrc::Path(p) => {
					Cond::load(p, o.resolution).map_err(|e| format!("{}: {e}", p.display()))
				}
				ImageSrc::Rgba(w, h, px) => Cond::new(*w, *h, px, o.resolution),
			})
			.collect::<Result<_, _>>()?;
		let (width, height) = match (o.width, o.height) {
			(0, 0) => cond::target(&conds, o.resolution).unwrap_or((o.resolution, o.resolution)),
			(w, h) => (w, h),
		};
		if width % 32 != 0 || height % 32 != 0 || width == 0 || height == 0 {
			return Err(format!(
				"size {width}x{height}: sides must be multiples of 32"
			));
		}
		let (h, w) = (height / 16, width / 16);
		let z = cfg.dit.in_ch;

		if std::env::var("QWEN_IMAGE_TIMING").is_ok_and(|v| v == "1") {
			eprintln!(
				"[mem] job start: ws / private {:.2} / {:.2} GB",
				mem_gb().0,
				mem_gb().1
			);
		}
		let mut te = self.te.lock().unwrap();
		if te.is_none() {
			*te = Some(TextEncoder::load(dir, &cfg.te, self.fmt, |l| {
				progress("load text encoder", l + 1, cfg.te.layers)
			})?);
		}
		let tenc = te.as_ref().unwrap();
		progress("text encoder", 0, 0);
		let t0 = Instant::now();
		let vis = if conds.is_empty() {
			None
		} else {
			match &self.npu_vis {
				Some((p, m)) => {
					let mut g = m.lock().unwrap();
					if g.0.is_none() {
						let npu = crate::npuhost::npu()?;
						let t = Instant::now();
						g.0 = Some(npu::vis::NpuVision::load(npu, &p.to_string_lossy())?);
						if std::env::var("QWEN_IMAGE_TIMING").is_ok_and(|v| v == "1") {
							eprintln!(
								"[te] vision tower (NPU) loaded in {:.1} s",
								t.elapsed().as_secs_f64()
							);
						}
					}
					Some(crate::cond::vision_rows(
						dir,
						&tenc.ck,
						&conds,
						g.0.as_mut(),
					)?)
				}
				None => Some(crate::cond::vision_rows(dir, &tenc.ck, &conds, None)?),
			}
		};
		let t_vis = t0.elapsed().as_secs_f64();
		let enc = tenc.encode_images(&o.prompt, vis.as_ref(), |_, _| {})?;
		if std::env::var("QWEN_IMAGE_TIMING").is_ok_and(|v| v == "1") {
			eprintln!(
				"[te] vision tower {t_vis:.1} s, text encoder {:.1} s ({} tokens)",
				t0.elapsed().as_secs_f64() - t_vis,
				enc.ids.len()
			);
		}
		drop(vis);
		let t_te = t0.elapsed().as_secs_f64();
		canceled()?;

		let t1 = Instant::now();
		let mut refs = Vec::with_capacity(conds.len());
		if !conds.is_empty() {
			progress("vae encoder", 0, 0);
			let vae = Vae::open(dir, &cfg.vae)?;
			let mut nv = self.npu_vae()?;
			for c in &conds {
				refs.push(vae.encode_with(
					&c.vae_pixels(),
					nv.as_mut().and_then(|g| g.as_mut()),
					|_, _| {},
				)?);
			}
			if let Some(n) = nv.as_mut().and_then(|g| g.as_mut()) {
				n.release_buffers();
			}
		}
		let shapes: Vec<(usize, usize)> = conds.iter().map(|c| (c.h / 16, c.w / 16)).collect();
		drop(conds);
		let t_venc = t1.elapsed().as_secs_f64();
		canceled()?;

		let t2 = Instant::now();
		let dit = &self.dit;
		let lay = layout(&enc.image_pad, &shapes, (h, w))?;
		let (kv8, drop_te) = prefix_plan(cfg.dit.layers, cfg.dit.dim(), lay.prefix);
		let timing = std::env::var("QWEN_IMAGE_TIMING").is_ok_and(|v| v == "1");
		if timing {
			eprintln!(
				"[mem] after the encoders: ws / private {:.2} / {:.2} GB",
				mem_gb().0,
				mem_gb().1
			);
		}
		if drop_te && self.npu.is_some() {
			*te = None;
			// the NPU vision tower too (reloaded in ~0.4 s by the next edit)
			if let Some((_, m)) = &self.npu_vis {
				m.lock().unwrap().0 = None;
			}
			// and the VAE's (loaded again for the decoder)
			if let Some((_, m)) = &self.npu_vae {
				*m.lock().unwrap() = None;
			}
		}
		drop(te);
		let mut txt = Some(dit.prefix_rows(&enc.hidden, &lay, &refs)?);
		drop((enc, refs));
		if kv8 {
			// the text encoder's and the tower's freed blocks, kept by the
			// recycling allocator (up to QWEN_ALLOC_CACHE_MB), before the
			// prefix K / V grow
			quant::alloc::trim();
		}
		if timing {
			eprintln!(
				"[mem] before the prefix: ws / private {:.2} / {:.2} GB",
				mem_gb().0,
				mem_gb().1
			);
		}
		let rope = Rope::new(&cfg.dit, &lay.ids);
		let mut pre = Prefix::new(&lay);
		let mut sched =
			Schedule::for_steps(&cfg.sched, cfg.sample_sigmas.as_deref(), o.steps, h * w);
		// an experiment: the sigmas as given (a terminal 0 appended)
		if let Ok(v) = std::env::var("QWEN_IMAGE_SIGMAS") {
			let mut s: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
			if s.last() != Some(&0.0) {
				s.push(0.0);
			}
			sched = Schedule { sigmas: s };
		}
		let mut x = match &o.noise {
			Some(n) if n.len() == h * w * z => n.clone(),
			Some(n) => return Err(format!("noise: {} values, expected {}", n.len(), h * w * z)),
			None => noise(o.seed, z, h, w),
		};
		let kv8_check =
			std::env::var("QWEN_IMAGE_KV8_CHECK").is_ok_and(|v| v == "1" || v == "2") && !kv8;
		let mut npu = self.npu.as_ref().map(|n| n.lock().unwrap());
		if let Some(n) = npu.as_mut() {
			n.reset();
			n.kv8 = kv8;
			if std::env::var("QWEN_IMAGE_TIMING").is_ok_and(|v| v == "1") {
				eprintln!(
					"[dit] prefix {} rows, K / V {}{}",
					lay.prefix,
					if kv8 { "int8" } else { "f16" },
					if drop_te {
						", text encoder dropped"
					} else {
						""
					}
				);
			}
			(n.t_npu, n.t_attn, n.t_gpu, n.t_bound) = (0.0, 0.0, 0.0, 0.0);
		}
		let r = (|| {
			let ts: Vec<f32> = (0..sched.steps())
				.map(|i| sched.t(i, cfg.sched.train_steps))
				.collect();
			let mods = dit.modulations(&ts)?;
			for (i, m) in mods.iter().enumerate() {
				canceled()?;
				progress("denoise", i + 1, sched.steps());
				let v = match npu.as_mut() {
					Some(n) => {
						if let Some(t) = txt.take() {
							n.prefix(t, &lay, &rope, m)?;
						}
						let xin = dit.img_in(&x)?;
						let v = dit.out(&n.forward(&xin, &lay, &rope, m)?, m)?;
						if kv8_check {
							let same =
								std::env::var("QWEN_IMAGE_KV8_CHECK").is_ok_and(|v| v == "2");
							if !same {
								n.swap_kv();
							}
							let v8 = dit.out(&n.forward(&xin, &lay, &rope, m)?, m)?;
							if !same {
								n.swap_kv();
							}
							let (mut e, mut r) = (0f64, 0f64);
							for (a, b) in v8.iter().zip(&v) {
								e += ((a - b) as f64).powi(2);
								r += (*b as f64).powi(2);
							}
							eprintln!("[kv8] step {i}: velocity rel. error {:.3e}", (e / r).sqrt());
						}
						v
					}
					None => dit.forward(&x, &mut txt, &mut pre, &rope, m, |_, _, _| {})?,
				};
				sched.step(i, &mut x, &v);
				if i == 0 && std::env::var("QWEN_IMAGE_TIMING").is_ok_and(|v| v == "1") {
					eprintln!(
						"[mem] after step 0: ws / private {:.2} / {:.2} GB",
						mem_gb().0,
						mem_gb().1
					);
				}
			}
			Ok::<(), String>(())
		})();
		if let Some(n) = npu.as_mut() {
			if std::env::var("QWEN_IMAGE_TIMING").is_ok_and(|v| v == "1") {
				eprintln!(
					"[dit] NPU wait {:.1} s, attention (cores) {:.1} s, iGPU wait {:.1} s, block boundaries {:.1} s",
					n.t_npu, n.t_attn, n.t_gpu, n.t_bound
				);
			}
			n.release_buffers();
		}
		drop(npu);
		r?;
		drop(pre);
		let t_dit = t2.elapsed().as_secs_f64();
		canceled()?;

		progress("vae", 0, 0);
		let t3 = Instant::now();
		let vae = Vae::open(dir, &cfg.vae)?;
		let timing = std::env::var("QWEN_IMAGE_TIMING").is_ok_and(|v| v == "1");
		let mut tl = Instant::now();
		let mut nv = self.npu_vae()?;
		if let Some(n) = nv.as_mut().and_then(|g| g.as_mut()) {
			(n.t_wait, n.t_cores) = (0.0, 0.0);
		}
		let img = vae.decode_with(&x, h, w, nv.as_mut().and_then(|g| g.as_mut()), |name, m| {
			if timing {
				eprintln!(
					"[vae] {name} {}x{}x{}: {:.2} s",
					m.w,
					m.h,
					m.c,
					tl.elapsed().as_secs_f64()
				);
				tl = Instant::now();
			}
		});
		if let Some(n) = nv.as_mut().and_then(|g| g.as_mut()) {
			if timing {
				eprintln!(
					"[vae] NPU waited {:.2} s, cores {:.2} s",
					n.t_wait, n.t_cores
				);
			}
			n.release_buffers();
		}
		drop(nv);
		let img = img?;
		let t_vae = t3.elapsed().as_secs_f64();
		Ok(Output {
			width: img.w,
			height: img.h,
			rgba: to_rgba8(&img),
			seed: o.seed,
			secs: [t_te, t_venc, t_dit, t_vae],
		})
	}
}

/// Most condition images an edit takes (every interface refuses more).
pub const MAX_IMAGES: usize = 10;

/// How an NPU transformer keeps a prefix of `rows` rows (`layers` blocks
/// of width `dim`): `(kv8, drop_te)`. Its K / V (f16: `layers` x 2 x rows
/// x dim x 2 bytes, ~2.2 GB per 1 MP reference image) and the prefix
/// pass' buffers may take `QWEN_IMAGE_PREFIX_MB` (default 6144: the
/// 24 GB budget less the ~16.3 GB of resident weights and the VAE
/// decoder's peak) next to the text encoder; beyond that the K / V are
/// int8 with a scale per row and head (~2 refs and less stay f16,
/// bit-identical), and when even those do not fit, the text encoder is
/// dropped for the denoising (~5 refs and more; reloaded by the next job).
fn prefix_plan(layers: usize, dim: usize, rows: usize) -> (bool, bool) {
	let budget = std::env::var("QWEN_IMAGE_PREFIX_MB")
		.ok()
		.and_then(|v| v.parse::<usize>().ok())
		.unwrap_or(6144)
		<< 20;
	let rows = rows.next_multiple_of(npubuild::qimage::S);
	// x (2), a, q, K (2), V (2) of the prefix pass, f16
	let bufs = 8 * rows * dim * 2;
	let f16 = layers * 2 * rows * dim * 2;
	let i8 = f16 / 2 + layers * 2 * rows * (dim / 128) * 4;
	if f16 + bufs <= budget {
		(false, false)
	} else {
		(true, i8 + bufs > budget)
	}
}

/// Load the model in the format `QWEN_IMAGE_WEIGHTS` names (default
/// `q6k`) and generate one image.
pub fn generate(
	dir: &Path,
	o: &Options,
	mut progress: impl FnMut(&str, usize, usize),
) -> Result<Output, String> {
	Model::load(dir, WFmt::from_env(), &mut progress)?.generate(o, progress)
}
