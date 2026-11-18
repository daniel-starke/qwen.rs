//! Reference data points at every stage of text to image, for checking a
//! backend (or an optimization of one) stage by stage.
//!
//! ```text
//! stageref record REF QWEN_IMAGE_DIR [--size N] [--seed N] [--prompt P]
//! stageref check  REF MODEL [--free]
//! ```
//!
//! `record` runs the f32 reference (`WFmt::F32`: weights widened from BF16
//! per use) and writes into `REF` (raw little-endian f32 unless noted):
//! `meta.txt` (prompt, size, seed, token counts), `te.f32` (text encoder
//! output, kept rows), `prefix.f32` (`txt_in` rows), per step `i`
//! `mod{i}.f32` (target modulation, prefix modulation, out scale), `x{i}.f32`
//! (the latents the step starts from), `v{i}.f32` (velocity); the target
//! rows after every block of the first and the last step
//! (`blk{step}_{l}.f16`, f16 bits), `latents.f32` (after the last step) and
//! `image.png` (the VAE's output). Size 256 by default (256 target rows:
//! one NPU chunk; ~2 min, 0.3 GB).
//!
//! `check` runs `MODEL` (a checkpoint directory: the Q6_K CPU backend, or
//! an `image.npu`) on the recorded inputs and prints the relative error of
//! each stage: text encoder, prefix rows, every block of the first and the
//! last step (each forward fed the recorded latents), every step's
//! velocity, the VAE decoder on the recorded final latents against
//! `image.png` (PSNR), the NPU VAE encoder against the cores' on
//! `image.png` (an `image.npu` with `vae`), and with `--free` the
//! free-running image (its own latents from the recorded noise) against
//! `image.png` (PSNR). Times per stage.
//!
//! The free-running PSNR is chaotic on the NPU backend: a 1-ulp change of
//! the attention's f32 output moved it from 32.2 to 29.1 dB at 512x512;
//! compare the per-step numbers, and the free-running one only for jumps
//! beyond that band.

use imagegen::dit::{layout, Dit, Modulation, Prefix, Rope};
use imagegen::npudit::{NpuCfg, NpuDit};
use imagegen::npuvae::NpuVae;
use imagegen::sched::Schedule;
use imagegen::te::TextEncoder;
use imagegen::vae::{to_rgba8, Vae};
use imagegen::{Config, WFmt};
use std::path::{Path, PathBuf};
use std::time::Instant;

const PROMPT: &str = "A red fox sitting in a snowy forest at dawn, soft golden light, photograph";

fn wf32(p: &Path, v: &[f32]) -> Result<(), String> {
	let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
	std::fs::write(p, b).map_err(|e| format!("{}: {e}", p.display()))
}

fn rf32(p: &Path) -> Result<Vec<f32>, String> {
	Ok(std::fs::read(p)
		.map_err(|e| format!("{}: {e}", p.display()))?
		.chunks_exact(4)
		.map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
		.collect())
}

/// PSNR of two 8-bit images.
fn psnr(got: &[u8], want: &[u8]) -> f64 {
	let mse = got
		.iter()
		.zip(want)
		.map(|(&p, &q)| (p as f64 - q as f64).powi(2))
		.sum::<f64>()
		/ got.len() as f64;
	10.0 * (255.0f64 * 255.0 / mse.max(1e-12)).log10()
}

fn wf16(p: &Path, v: &[f32]) -> Result<(), String> {
	let b: Vec<u8> = v
		.iter()
		.flat_map(|&x| quant::f32_to_fp16(x).to_le_bytes())
		.collect();
	std::fs::write(p, b).map_err(|e| format!("{}: {e}", p.display()))
}

fn rf16(p: &Path) -> Result<Vec<f32>, String> {
	Ok(std::fs::read(p)
		.map_err(|e| format!("{}: {e}", p.display()))?
		.chunks_exact(2)
		.map(|c| quant::fp16_to_f32(u16::from_le_bytes([c[0], c[1]])))
		.collect())
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
	if a.len() != b.len() {
		return f64::NAN;
	}
	let (mut d, mut n) = (0f64, 0f64);
	for (x, y) in a.iter().zip(b) {
		d += (*x as f64 - *y as f64).powi(2);
		n += (*y as f64).powi(2);
	}
	(d / n.max(1e-30)).sqrt()
}

fn flag(a: &[String], k: &str) -> Option<String> {
	a.iter()
		.position(|x| x == k)
		.and_then(|i| a.get(i + 1))
		.cloned()
}

fn mod_rows(m: &Modulation) -> Vec<f32> {
	[&m.target[..], &m.prefix, &m.out_scale].concat()
}

fn mod_from(v: &[f32], d: usize) -> Modulation {
	Modulation {
		target: v[..4 * d].to_vec(),
		prefix: v[4 * d..8 * d].to_vec(),
		out_scale: v[8 * d..9 * d].to_vec(),
	}
}

struct Meta {
	dir: PathBuf,
	prompt: String,
	size: usize,
	seed: u64,
	steps: usize,
}

fn read_meta(r: &Path) -> Result<Meta, String> {
	let p = r.join("meta.txt");
	let t = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
	let get = |k: &str| {
		t.lines()
			.find_map(|l| l.strip_prefix(&format!("{k} = ")))
			.map(str::to_string)
			.ok_or(format!("{}: no {k}", p.display()))
	};
	let num = |k: &str| -> Result<u64, String> {
		get(k)?
			.parse()
			.map_err(|_| format!("{}: bad {k}", p.display()))
	};
	Ok(Meta {
		dir: get("checkpoint")?.into(),
		prompt: get("prompt")?,
		size: num("size")? as usize,
		seed: num("seed")?,
		steps: num("steps")? as usize,
	})
}

fn record(a: &[String]) -> Result<(), String> {
	let r = PathBuf::from(a.get(1).ok_or("record REF QWEN_IMAGE_DIR")?);
	let dir = PathBuf::from(a.get(2).ok_or("record REF QWEN_IMAGE_DIR")?);
	let size: usize = flag(a, "--size")
		.and_then(|s| s.parse().ok())
		.unwrap_or(256);
	let seed: u64 = flag(a, "--seed").and_then(|s| s.parse().ok()).unwrap_or(42);
	let prompt = flag(a, "--prompt").unwrap_or(PROMPT.to_string());
	std::fs::create_dir_all(&r).map_err(|e| e.to_string())?;
	let cfg = Config::load(&dir)?;
	let d = cfg.dit.dim();
	let t = Instant::now();
	let te = TextEncoder::load(&dir, &cfg.te, WFmt::F32, |_| {})?;
	let enc = te.encode(&prompt, |_, _| {})?;
	eprintln!(
		"text encoder {:.1} s ({} tokens, {} kept)",
		t.elapsed().as_secs_f64(),
		enc.ids.len(),
		enc.ids.len() - enc.drop
	);
	wf32(&r.join("te.f32"), &enc.hidden)?;
	let dit = Dit::open(&dir, &cfg.dit)?;
	let (h, w) = (size / 16, size / 16);
	let lay = layout(&enc.image_pad, &[], (h, w))?;
	let txt = dit.prefix_rows(&enc.hidden, &lay, &[])?;
	wf32(&r.join("prefix.f32"), &txt)?;
	let rope = Rope::new(&cfg.dit, &lay.ids);
	let sched = Schedule::new(&cfg.sched, cfg.sample_sigmas.as_deref(), 8, h * w);
	let mut x = imagegen::pipeline::noise(seed, cfg.dit.in_ch, h, w);
	let mut pre = Prefix::new(&lay);
	let mut t_opt = Some(txt);
	let n = sched.steps();
	for i in 0..n {
		let t = Instant::now();
		let (m, _, _) = dit.modulation(sched.t(i, cfg.sched.train_steps))?;
		wf32(&r.join(format!("mod{i}.f32")), &mod_rows(&m))?;
		wf32(&r.join(format!("x{i}.f32")), &x)?;
		let mut err = Ok(());
		let v = dit.forward(&x, &mut t_opt, &mut pre, &rope, &m, |l, _, xs| {
			if (i == 0 || i == n - 1) && err.is_ok() {
				err = wf16(&r.join(format!("blk{i}_{l}.f16")), xs);
			}
		})?;
		err?;
		wf32(&r.join(format!("v{i}.f32")), &v)?;
		sched.step(i, &mut x, &v);
		eprintln!("step {i}: {:.1} s", t.elapsed().as_secs_f64());
	}
	wf32(&r.join("latents.f32"), &x)?;
	let img = Vae::open(&dir, &cfg.vae)?.decode(&x, h, w, |_, _| {})?;
	imgio::save_png_rgba(
		&r.join("image.png").to_string_lossy(),
		img.w,
		img.h,
		&to_rgba8(&img),
	)?;
	std::fs::write(
		r.join("meta.txt"),
		format!(
			"checkpoint = {}\nprompt = {prompt}\nsize = {size}\nseed = {seed}\nsteps = {n}\ntokens = {}\nkept = {}\nprefix = {}\ndim = {d}\n",
			dir.display(),
			enc.ids.len(),
			enc.ids.len() - enc.drop,
			lay.prefix
		),
	)
	.map_err(|e| e.to_string())?;
	eprintln!(
		"recorded {} in {:.0} s",
		r.display(),
		t.elapsed().as_secs_f64()
	);
	Ok(())
}

fn check(a: &[String]) -> Result<(), String> {
	let r = PathBuf::from(a.get(1).ok_or("check REF MODEL")?);
	let model = PathBuf::from(a.get(2).ok_or("check REF MODEL")?);
	let free = a.iter().any(|x| x == "--free");
	let meta = read_meta(&r)?;
	let nc = if NpuCfg::is_image_npu(&model) {
		Some(NpuCfg::parse(&model)?)
	} else {
		None
	};
	let dir = nc.as_ref().map_or(model.clone(), |c| c.checkpoint.clone());
	let cfg = Config::load(&dir)?;
	let (d, layers) = (cfg.dit.dim(), cfg.dit.layers);
	let t = Instant::now();
	let te = TextEncoder::load(&dir, &cfg.te, WFmt::Q6K, |_| {})?;
	println!(
		"load text encoder (Q6_K): {:.1} s",
		t.elapsed().as_secs_f64()
	);
	let t = Instant::now();
	let enc = te.encode(&meta.prompt, |_, _| {})?;
	println!(
		"text encoder: rel. error {:.3e} ({:.2} s)",
		rel(&enc.hidden, &rf32(&r.join("te.f32"))?),
		t.elapsed().as_secs_f64()
	);
	drop(te);
	let t = Instant::now();
	let (dit, mut npu) = match &nc {
		Some(c) => (
			Dit::open(&dir, &cfg.dit)?,
			Some(NpuDit::load(c, &cfg.dit, |_| {})?),
		),
		None => (Dit::load(&dir, &cfg.dit, WFmt::Q6K, |_| {})?, None),
	};
	println!(
		"load transformer ({}): {:.1} s",
		if npu.is_some() { "NPU int8" } else { "Q6_K" },
		t.elapsed().as_secs_f64()
	);
	// the recorded text encoder output from here on: each stage on the
	// reference's inputs
	let te_ref = rf32(&r.join("te.f32"))?;
	let (h, w) = (meta.size / 16, meta.size / 16);
	let slots = vec![false; te_ref.len() / cfg.dit.ctx_dim];
	let lay = layout(&slots, &[], (h, w))?;
	let txt = dit.prefix_rows(&te_ref, &lay, &[])?;
	println!(
		"prefix rows: rel. error {:.3e}",
		rel(&txt, &rf32(&r.join("prefix.f32"))?)
	);
	let txt = rf32(&r.join("prefix.f32"))?;
	let rope = Rope::new(&cfg.dit, &lay.ids);
	let sched = Schedule::new(&cfg.sched, cfg.sample_sigmas.as_deref(), 8, h * w);
	let n = meta.steps;
	let m0 = mod_from(&rf32(&r.join("mod0.f32"))?, d);
	let (mm, _, _) = dit.modulation(sched.t(0, cfg.sched.train_steps))?;
	println!(
		"modulation: rel. error {:.3e}",
		rel(&mod_rows(&mm), &mod_rows(&m0))
	);
	let mut pre = Prefix::new(&lay);
	let t = Instant::now();
	if let Some(np) = npu.as_mut() {
		np.prefix(txt.clone(), &lay, &rope, &m0)?;
	}
	let t_pre = t.elapsed().as_secs_f64();
	// per-block errors at the first and last step
	for i in [0, n - 1] {
		let x = rf32(&r.join(format!("x{i}.f32")))?;
		let m = mod_from(&rf32(&r.join(format!("mod{i}.f32")))?, d);
		let mut errs = vec![f64::NAN; layers];
		match npu.as_mut() {
			Some(np) => {
				for (l, e) in errs.iter_mut().enumerate() {
					np.limit = Some(l + 1);
					let got = np.forward(&dit.img_in(&x)?, &lay, &rope, &m)?;
					*e = rel(&got, &rf16(&r.join(format!("blk{i}_{l}.f16")))?);
				}
				np.limit = None;
			}
			None => {
				let mut t_opt = (i == 0).then(|| txt.clone());
				let mut err = Ok(());
				dit.forward(&x, &mut t_opt, &mut pre, &rope, &m, |l, _, xs| {
					match rf16(&r.join(format!("blk{i}_{l}.f16"))) {
						Ok(want) => errs[l] = rel(xs, &want),
						Err(e) => err = Err(e),
					}
				})?;
				err?;
			}
		}
		let s: Vec<String> = errs.iter().map(|e| format!("{e:.1e}")).collect();
		println!("step {i} blocks: {}", s.join(" "));
	}
	// velocity per step on the recorded latents
	let mut t_dit = 0.0;
	if let Some(np) = npu.as_mut() {
		np.reset_cache();
	}
	for i in 0..n {
		let x = rf32(&r.join(format!("x{i}.f32")))?;
		let m = mod_from(&rf32(&r.join(format!("mod{i}.f32")))?, d);
		let t = Instant::now();
		let v = match npu.as_mut() {
			Some(np) => dit.out(&np.forward(&dit.img_in(&x)?, &lay, &rope, &m)?, &m)?,
			None => {
				let mut none = None;
				dit.forward(&x, &mut none, &mut pre, &rope, &m, |_, _, _| {})?
			}
		};
		let dt = t.elapsed().as_secs_f64();
		t_dit += dt;
		println!(
			"step {i} velocity: rel. error {:.3e} ({dt:.2} s)",
			rel(&v, &rf32(&r.join(format!("v{i}.f32")))?)
		);
	}
	println!(
		"transformer: prefix {t_pre:.2} s, {:.2} s per step",
		t_dit / n as f64
	);
	// the VAE decoder alone, on the recorded latents
	let vae = Vae::open(&dir, &cfg.vae)?;
	let mut nv = match nc.as_ref().and_then(|c| c.vae.as_ref()) {
		Some(g) => Some(NpuVae::load(g, &dir)?),
		None => None,
	};
	let want_img = imgio::load_rgba(&r.join("image.png").to_string_lossy())?.2;
	let lat = rf32(&r.join("latents.f32"))?;
	let t = Instant::now();
	let img = vae.decode_with(&lat, h, w, nv.as_mut(), |_, _| {})?;
	let t1 = t.elapsed().as_secs_f64();
	let t = Instant::now();
	let img2 = vae.decode_with(&lat, h, w, nv.as_mut(), |_, _| {})?;
	println!(
		"vae decoder ({}): PSNR {:.1} dB vs the reference on its latents ({t1:.1} s, again {:.1} s)",
		if nv.is_some() { "NPU f16 + cores" } else { "cores" },
		psnr(&to_rgba8(&img), &want_img),
		t.elapsed().as_secs_f64()
	);
	drop((img, img2));
	// the VAE encoder (edits) on the reference image: the backend's against
	// the cores' f32 one
	if let Some(n) = nv.as_mut() {
		let px = imagegen::vae::Map {
			h: meta.size,
			w: meta.size,
			c: 4,
			d: want_img.iter().map(|&b| b as f32 / 127.5 - 1.0).collect(),
		};
		let t = Instant::now();
		let want = vae.encode(&px, |_, _| {})?;
		let t_cpu = t.elapsed().as_secs_f64();
		let t = Instant::now();
		let got = vae.encode_with(&px, Some(n), |_, _| {})?;
		println!(
			"vae encoder (NPU f16 + cores): rel. error {:.3e} vs the cores' ({:.2} s, cores {t_cpu:.2} s)",
			rel(&got, &want),
			t.elapsed().as_secs_f64()
		);
		n.release_buffers();
	}
	if free {
		let mut x = rf32(&r.join("x0.f32"))?;
		if let Some(np) = npu.as_mut() {
			np.reset_cache();
		}
		for i in 0..n {
			let (m, _, _) = dit.modulation(sched.t(i, cfg.sched.train_steps))?;
			let v = match npu.as_mut() {
				Some(np) => dit.out(&np.forward(&dit.img_in(&x)?, &lay, &rope, &m)?, &m)?,
				None => {
					let mut none = None;
					dit.forward(&x, &mut none, &mut pre, &rope, &m, |_, _, _| {})?
				}
			};
			sched.step(i, &mut x, &v);
		}
		println!(
			"free-running latents: rel. error {:.3e}",
			rel(&x, &rf32(&r.join("latents.f32"))?)
		);
		let t = Instant::now();
		let img = vae.decode_with(&x, h, w, nv.as_mut(), |_, _| {})?;
		println!(
			"free-running image: PSNR {:.1} dB vs the reference (vae {:.1} s)",
			psnr(&to_rgba8(&img), &want_img),
			t.elapsed().as_secs_f64()
		);
	}
	let _ = meta.dir;
	let _ = meta.seed;
	Ok(())
}

fn main() {
	let a: Vec<String> = std::env::args().skip(1).collect();
	let r = match a.first().map(String::as_str) {
		Some("record") => record(&a),
		Some("check") => check(&a),
		_ => Err("usage: stageref record REF QWEN_IMAGE_DIR [--size N] [--seed N] [--prompt P] | check REF MODEL [--free]".into()),
	};
	if let Err(e) = r {
		eprintln!("stageref: {e}");
		std::process::exit(1);
	}
}
