//! `--generate`, `--interactive` and `--serve` on Qwen3.8-Flash-Next's NPU
//! engine (`cli <file>.npu` with `arch = flash`, `npu::flash::engine`);
//! the server has the prefix cache. With `vision = DIR` in the `.npu` file
//! it takes images (`--image`, `/image`, image blocks): the tower as on the
//! 27B's NPU engine (`npuchat`'s helpers), out width 2560. With the dense layers on a
//! GPU (`dense = ...`) decode verifies up to 8 tokens per step (context
//! drafts, `model::suffix`), through `npuchat`'s decode loop.
//!
//! `QWEN_NPU_IDS=FILE` feeds those token ids instead of the prompt's;
//! `QWEN_FLASH_DUMP_ALL=FILE` writes every prompt position's logits (raw
//! f32 `[n][vocab]`) for a KL check against truth logits and stops;
//! `QWEN_FLASH_DUMP_LAST=FILE` writes the prompt's last logits;
//! `QWEN_FLASH_SNAPCHECK=N` feeds all but the last N prompt tokens,
//! snapshots, runs the rest, then restores the snapshot into a reset engine,
//! runs the rest again and compares the logits (a prefix-cache check).

use std::io::Write;
use std::path::Path;

use npu::flash::engine::{Config, FlashEngine};
use sampler::Rng;
use tokenizer::{StreamDecoder, Tokenizer};

use crate::chat::GenOptions;
use crate::npuchat::{
	decode, encode_image, feed_images, first_rows, first_rows_mm, spec_note, vis_attn, SpecEngine,
	VisEngine,
};

/// Flash-Next's context: the model's 262144 positions.
const CTX: usize = 262144;

impl SpecEngine for FlashEngine {
	fn spec_k(&self) -> usize {
		FlashEngine::spec_k(self)
	}
	fn verify(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>, String> {
		FlashEngine::verify(self, ids)
	}
	fn commit(&mut self, n: usize) -> Result<(), String> {
		FlashEngine::commit(self, n)
	}
	/// The MTP head's drafts (`mtp = DIR`; none without one: only context
	/// drafts then).
	fn drafts(&mut self, next: u32) -> Result<Vec<u32>, String> {
		FlashEngine::drafts(self, next)
	}
	fn forward(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
		FlashEngine::forward(self, ids)
	}
	fn feed(&mut self, ids: &[u32]) -> Result<(), String> {
		FlashEngine::feed(self, ids)
	}
	fn pos(&self) -> usize {
		FlashEngine::pos(self)
	}
	fn context(&self) -> usize {
		CTX
	}
	fn step_costs(&self) -> Option<Vec<f32>> {
		FlashEngine::step_costs(self)
	}
	fn set_coupling(&mut self, c: Option<sampler::Coupling>) {
		FlashEngine::set_coupling(self, c)
	}
}

impl VisEngine for FlashEngine {
	fn vision(&self) -> Option<(&npu::vis::VisCfg, &[f32])> {
		FlashEngine::vision(self)
	}
	fn encode_image(
		&mut self,
		patches: &[f32],
		pos: &[f32],
		hw: &[(u32, u32)],
	) -> Result<Vec<f32>, String> {
		FlashEngine::encode_image(self, patches, pos, hw)
	}
	fn feed_mm(&mut self, ids: &[u32], spans: &[npu::engine::ImageSpan<'_>]) -> Result<(), String> {
		FlashEngine::feed_mm(self, ids, spans)
	}
}

pub fn is_flash(path: &str) -> bool {
	Config::is_flash(path)
}

/// What a loader thread hands back: Vulkan devices (externally
/// synchronized, used by one thread at a time) and a tokenizer.
struct SendBox<T>(T);
// SAFETY: the devices are made on the loader thread and only used by the
// engine's thread afterwards, never by two threads at once.
unsafe impl<T> Send for SendBox<T> {}

fn load(path: &str) -> Result<(FlashEngine, Tokenizer, Vec<u32>), Box<dyn std::error::Error>> {
	let cfg = Config::parse(path)?;
	let dense = cfg.dense.clone().map(dense_gpu);
	// `QWEN_FLASH_EXP_GPU=0`: every routed expert of a prefill on the NPU
	let exp_dev = cfg.scan.clone().filter(|_| {
		std::env::var("QWEN_FLASH_EXP_GPU")
			.map_or(true, |v| v.parse::<f64>().map_or(true, |f| f > 0.0))
	});
	let exp_name = exp_dev.clone();
	// The scan adapter's devices and the tokenizer are made on a thread
	// while the engine loads (whose critical path is reading the expert
	// pools from disk); the experts' device goes first, since the engine
	// imports each layer's pools into it as they are read.
	let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
	let (tx_exp, rx_exp) = std::sync::mpsc::channel::<
		Result<Box<dyn npu::flash::engine::PrefillExp + Send>, String>,
	>();
	let (eng, devs) = std::thread::scope(|sc| {
		let cfg = &cfg;
		let devs = sc.spawn(move || {
			let exp_dev = exp_dev;
			// after the NPU is open: opening it next to the first Vulkan
			// device took 0.5 s instead of 0.09, and the expert reads wait
			// for it
			let _ = go_rx.recv();
			if let Some(dev) = &exp_dev {
				let _sp = quant::ltime::span(|| "flash: prefill experts device".into());
				let _ = tx_exp.send(prefill_exp(dev));
			}
			drop(tx_exp);
			let sp = quant::ltime::span(|| "flash: scan device".into());
			let scan = cfg.scan.as_deref().map(crate::npuchat::scan_device);
			sp.end();
			// `QWEN_FLASH_ATTN_GPU=0`: prefill attention on the cores (the A/B)
			let attn = cfg
				.scan
				.as_deref()
				.filter(|_| std::env::var("QWEN_FLASH_ATTN_GPU").map_or(true, |v| v != "0"))
				.map(|dev| {
					let _sp = quant::ltime::span(|| "flash: prefill attention device".into());
					prefill_attn(dev)
				});
			// made whether the engine takes them or not (it says after its
			// load); an error counts only where it does
			let dev = cfg.scan.as_deref().unwrap_or("intel");
			let hc = cfg.scan.as_ref().map(|_| {
				(
					prefill_hc(dev, &cfg.weights, "hca"),
					prefill_hc(dev, &cfg.weights, "hcm"),
				)
			});
			// the tower's attention (the engine loads the tower)
			let va = cfg.vision.as_ref().map(|d| {
				let _sp = quant::ltime::span(|| "flash: vision attention device".into());
				npu::vis::VisCfg::parse(&d.join("vis.cfg"))
					.map(|c| (vis_attn(cfg.scan.as_deref(), &c), c))
			});
			let sp = quant::ltime::span(|| "flash: tokenizer".into());
			let tok = Tokenizer::from_hf_json(&Path::new(&cfg.tokenizer).join("tokenizer.json"))
				.map_err(|e| format!("tokenizer: {e}"));
			sp.end();
			SendBox((scan, attn, hc, tok, va))
		});
		let exp: Option<npu::flash::engine::PrefillExpNew> = exp_name.is_some().then(|| {
			Box::new(move || {
				let _ = go_tx.send(());
				rx_exp
					.recv()
					.map_err(|_| "flash: the experts' device was not made".to_string())?
			}) as npu::flash::engine::PrefillExpNew
		});
		let sp = quant::ltime::span(|| "flash: engine".into());
		let eng = FlashEngine::load_with(cfg, None, dense, exp);
		sp.end();
		let devs = devs
			.join()
			.map_err(|_| "flash: the device thread panicked".to_string());
		(eng, devs)
	});
	let mut eng = eng?;
	let _sp = quant::ltime::span(|| "flash: after the engine".into());
	let SendBox((scan, attn, hc, tok, va)) = devs?;
	if let Some(s) = scan {
		eng.set_scan(s?)?;
	}
	if let Some(a) = attn {
		eng.set_prefill_attn(a?);
	}
	if let Some(dev) = &exp_name {
		eprintln!("flash: a share of the prefill's routed experts on '{dev}'");
	}
	if let Some((hca, hcm)) = hc {
		if eng.wants_prefill_hc() {
			eng.set_prefill_hc(hca?);
		}
		if eng.wants_prefill_hcm() {
			eng.set_prefill_hcm(hcm?);
		}
	}
	if let Some(c) = eng.vision().map(|v| v.0.clone()) {
		// (made on the device thread from the same `vis.cfg`)
		let a = match va {
			Some(Ok((a, _))) => a,
			_ => vis_attn(cfg.scan.as_deref(), &c),
		};
		eprintln!(
			"flash: vision tower: {} blocks on the NPU, their attention on {}",
			c.layers,
			a.name()
		);
		eng.set_vis_attn(a);
	}
	let tok = tok?;
	let stop: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
		.iter()
		.filter_map(|s| tok.token_id(s))
		.collect();
	Ok((eng, tok, stop))
}

/// Prefill attention on the scan device's adapter (`offload::FlashAttn`).
#[cfg(feature = "vulkan")]
fn prefill_attn(dev: &str) -> Result<Box<dyn npu::flash::engine::PrefillAttn>, String> {
	struct A(offload::FlashAttn);
	impl npu::flash::engine::PrefillAttn for A {
		fn attend_begin(
			&mut self,
			a: usize,
			k: &[&[u16]],
			v: &[&[u16]],
			p0: usize,
			n: usize,
			q: &[u16],
			sel: &[u32],
			kmin: usize,
		) -> Result<(), String> {
			self.0.attend_begin(a, k, v, p0, n, q, sel, kmin)
		}
		fn attend_end(&mut self, out: &mut [u16], lse: Option<&mut [f32]>) -> Result<(), String> {
			self.0.attend_end(out, lse)
		}
		fn invalidate(&mut self) {
			self.0.invalidate()
		}
		fn take_fill(&mut self) -> (f64, u64) {
			self.0.take_fill()
		}
		fn idx_len(&self, a: usize) -> Option<usize> {
			Some(self.0.idx_len(a))
		}
		fn idx_append(&mut self, a: usize, at: usize, rows: &[f32]) -> Result<(), String> {
			self.0.idx_append(a, at, rows)
		}
		fn score_begin(&mut self, a: usize, iq: &[u16], n: usize, nb: usize) -> Result<(), String> {
			self.0.score_begin(a, iq, n, nb)
		}
		fn score_end(&mut self) -> Result<Option<(usize, usize, usize, usize, &[u32])>, String> {
			self.0.score_end()
		}
	}
	let mut fa = offload::FlashAttn::new(Some(dev), 12)?;
	// (the mirrors are resident system memory: see `npu::flash::engine`'s
	// `exp_gpu_ne`)
	fa.set_max_rows(CTX);
	Ok(Box::new(A(fa)))
}

/// A share of the prefill's routed experts on the scan device's adapter
/// (`offload::Experts`, the pools imported zero-copy).
#[cfg(feature = "vulkan")]
fn prefill_exp(dev: &str) -> Result<Box<dyn npu::flash::engine::PrefillExp + Send>, String> {
	struct X(offload::Experts);
	// SAFETY: a Vulkan device and its buffers, used by one thread at a time
	// (made on a loader thread, then the engine's)
	unsafe impl Send for X {}
	impl npu::flash::engine::PrefillExp for X {
		fn add_layer(
			&mut self,
			l: usize,
			gu_w: &[u8],
			gu_s: &[u16],
			d_w: &[u8],
			d_s: &[u16],
			d_pre: &[u16],
		) -> Result<(), String> {
			self.0.add_layer(l, gu_w, gu_s, d_w, d_s, d_pre)
		}
		fn begin(&mut self, l: usize, x: &[u8], slots: &[(usize, &[u32])]) -> Result<(), String> {
			self.0.begin(l, x, slots)
		}
		fn end(&mut self) -> Result<&[u16], String> {
			self.0.end()
		}
		fn last_ms(&self) -> Option<f64> {
			self.0.last_ms()
		}
	}
	// a piece: `QWEN_FLASH_KSUP` sub-chunks of 256 tokens, 10 experts each
	let ksup = std::env::var("QWEN_FLASH_KSUP")
		.ok()
		.and_then(|v| v.parse().ok())
		.unwrap_or(npu::flash::engine::KSUP)
		.max(1);
	let tok = ksup * npu::flash::engine::S;
	Ok(Box::new(X(offload::Experts::new(
		Some(dev),
		npu::flash::NL,
		tok,
		tok * 10,
	)?)))
}

#[cfg(not(feature = "vulkan"))]
fn prefill_exp(_dev: &str) -> Result<Box<dyn npu::flash::engine::PrefillExp + Send>, String> {
	Err("scan = ... needs a build with --features vulkan".into())
}

/// The delta layers' hyper-connection `p` (`hca`: attention, `hcm`: MoE)
/// in prefill on the scan device's adapter (`offload::HcPre`), its weights
/// from `wdir`.
#[cfg(feature = "vulkan")]
fn prefill_hc(
	dev: &str,
	wdir: &Path,
	p: &str,
) -> Result<Box<dyn npu::flash::engine::PrefillHc>, String> {
	struct H(offload::HcPre);
	impl npu::flash::engine::PrefillHc for H {
		fn begin(&mut self, di: usize, x: &[u16], n: usize) -> Result<(), String> {
			self.0.begin(di, x, n)
		}
		fn end(&mut self, mixed: &mut [u16], inj: &mut [u16]) -> Result<(), String> {
			self.0.end(mixed, inj)
		}
	}
	let t = std::time::Instant::now();
	let _sp = quant::ltime::span(|| format!("flash: prefill {p} device"));
	let nl = npu::flash::NL;
	let delta: Vec<usize> = (0..nl).filter(|l| l % 4 != 3).collect();
	let mut h = offload::HcPre::new(Some(dev), delta.len())?;
	h.set_tag(if p == "hcm" { "hcm" } else { "hca" });
	let f16 = |b: Vec<u8>| -> Vec<f32> {
		b.chunks_exact(2)
			.map(|c| quant::fp16_to_f32(u16::from_le_bytes([c[0], c[1]])))
			.collect()
	};
	for (di, &l) in delta.iter().enumerate() {
		let rd = |n: &str| {
			let p = wdir.join(format!("fl{l}.{p}.{n}.in"));
			std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))
		};
		// the graphs' int8 codes x 2^-7 x the row scale
		let sc = |n: &str| -> Result<Vec<f32>, String> {
			Ok(f16(rd(n)?).iter().map(|v| v / 128.0).collect())
		};
		h.add_layer(
			di,
			&f16(rd("norm")?),
			&rd("dinj.w")?,
			&sc("dinj.s")?,
			&rd("up.w")?,
			&sc("up.s")?,
		)?;
	}
	eprintln!(
		"flash: delta layers' {} hyper-connection on '{dev}' ({:.1} s)",
		if p == "hcm" { "MoE" } else { "attention" },
		t.elapsed().as_secs_f64()
	);
	Ok(Box::new(H(h)))
}

#[cfg(not(feature = "vulkan"))]
fn prefill_hc(
	_dev: &str,
	_wdir: &Path,
	_p: &str,
) -> Result<Box<dyn npu::flash::engine::PrefillHc>, String> {
	Err("delta_pre_m needs a build with --features vulkan (or QWEN_FLASH_HC_GPU=0)".into())
}

#[cfg(not(feature = "vulkan"))]
fn prefill_attn(_dev: &str) -> Result<Box<dyn npu::flash::engine::PrefillAttn>, String> {
	Err("scan = ... needs a build with --features vulkan".into())
}

/// The dense layers' GEMVs on the adapter `dense = ADAPTER` names
/// (`offload::Gemv`). `QWEN_FLASH_DENSE_RESERVE_MB`: VRAM left free (256).
#[cfg(feature = "vulkan")]
fn dense_gpu(dev: String) -> npu::flash::gpudec::DenseGpuNew {
	use npu::flash::gpudec::{DenseGpu, Mat};
	struct G(offload::Gemv);
	impl DenseGpu for G {
		fn add(
			&mut self,
			m: Mat<'_>,
			n: usize,
			k: usize,
			xoff: usize,
			yoff: usize,
		) -> Result<usize, String> {
			let m = match m {
				Mat::I8(w, s) => offload::GemvMat::I8 { w, s },
				Mat::Q8(w, s) => offload::GemvMat::Q8 { w, s },
				Mat::F16(w) => offload::GemvMat::F16 { w },
			};
			self.0.add(m, n, k, xoff, yoff)
		}
		fn run(&mut self, x: &[f32], ids: &[usize]) -> Result<(), String> {
			self.0.run(x, ids)
		}
		fn act(
			&mut self,
			mode: u32,
			n: usize,
			src: usize,
			dst: usize,
			xsrc: usize,
			hc: usize,
			a: f32,
		) -> Result<usize, String> {
			self.0.add_act(mode, n, src, dst, xsrc, hc, a)
		}
		fn vec(
			&mut self,
			mode: u32,
			n: usize,
			src: usize,
			dst: usize,
			aux: usize,
			hc: usize,
			a: f32,
			blk: usize,
			flags: u32,
			ycopy: Option<usize>,
			extra: Option<usize>,
			w: Option<&[f32]>,
		) -> Result<usize, String> {
			self.0
				.add_vec(mode, n, src, dst, aux, hc, a, blk, flags, ycopy, extra, w)
		}
		fn set_silu(&mut self, id: usize, a: f32, xdst: usize) -> Result<bool, String> {
			self.0.set_silu(id, a, xdst).map(|()| true)
		}
		fn run_chain(&mut self, x: &[f32], stages: &[&[usize]]) -> Result<(), String> {
			self.0.run_chain(x, stages)
		}
		fn run_chain_n(&mut self, xs: &[&[f32]], stages: &[&[usize]]) -> Result<(), String> {
			self.0.run_chain_n(xs, stages)
		}
		fn y(&self) -> &[f32] {
			self.0.y()
		}
		fn y_at(&self, p: usize) -> &[f32] {
			self.0.y_at(p)
		}
		fn max_np(&self) -> usize {
			offload::GEMV_NP
		}
		fn io_next(&mut self, xr: &[std::ops::Range<usize>], yr: &[std::ops::Range<usize>]) {
			self.0.io_next(xr, yr)
		}
		fn run_chain_n_begin(&mut self, xs: &[&[f32]], stages: &[&[usize]]) -> Result<(), String> {
			self.0.run_chain_n_begin(xs, stages)
		}
		fn run_wait(&mut self) -> Result<(), String> {
			self.0.run_wait()
		}
		fn has_waves(&self) -> bool {
			true
		}
		fn set_wave(&mut self, w: usize, p0: usize) {
			self.0.set_wave(w, p0)
		}
		fn delta_init(&mut self, layers: usize) -> Result<(), String> {
			self.0.delta_init(layers)
		}
		fn delta(
			&mut self,
			mode: u32,
			flags: u32,
			layer: usize,
			yz: usize,
			yab: usize,
			ya: usize,
			w: &[f32],
		) -> Result<usize, String> {
			self.0.add_delta(mode, flags, layer, yz, yab, ya, w)
		}
		fn delta_put(&mut self, layer: usize, st: &[f32], win: &[f32]) -> Result<(), String> {
			self.0.delta_put(layer, st, win)
		}
		fn delta_get(
			&mut self,
			layer: usize,
			st: &mut [f32],
			win: &mut [f32],
		) -> Result<(), String> {
			self.0.delta_get(layer, st, win)
		}
	}
	Box::new(move |need| {
		let reserve: u64 = std::env::var("QWEN_FLASH_DENSE_RESERVE_MB")
			.ok()
			.and_then(|v| v.parse().ok())
			.unwrap_or(256);
		let g = offload::Gemv::new(Some(&dev), need, reserve << 20)?;
		Ok(Box::new(G(g)) as Box<dyn DenseGpu>)
	})
}

#[cfg(not(feature = "vulkan"))]
fn dense_gpu(_dev: String) -> npu::flash::gpudec::DenseGpuNew {
	Box::new(|_| Err("dense = ... needs a build with --features vulkan".into()))
}

pub fn generate(
	path: &str,
	prompt: &str,
	chat_mode: bool,
	system: Option<&str>,
	images: &[String],
	opts: GenOptions,
) -> Result<(), Box<dyn std::error::Error>> {
	if opts.mtp > 0 || opts.dflash.is_some() {
		return Err("--mtp / --dflash are not available on Flash-Next (its MTP head is the .npu file's `mtp = DIR`)".into());
	}
	println!("== generate (flash, npu)");
	let (mut eng, tok, stop) = load(path)?;
	let mut text = String::new();
	let mut imgs = Vec::new();
	// Templated when asked for, and whenever there is an image.
	let templated = chat_mode || system.is_some() || !images.is_empty();
	if templated {
		text.push_str(&Tokenizer::system_turn(system, opts.think));
		let mut content = String::new();
		for f in images {
			let enc = encode_image(&mut eng, &imgio::load(f)?, opts.max_image_tokens)?;
			content.push_str(&vision::placeholder(enc.1.tokens()));
			imgs.push(enc);
		}
		content.push_str(prompt);
		text.push_str(&Tokenizer::chat_turn("user", &content));
		text.push_str(opts.think.open());
	} else {
		text.push_str(prompt);
	}
	let mut ids = tok.encode(&text);
	if let Some(f) = std::env::var_os("QWEN_NPU_IDS") {
		ids = std::fs::read_to_string(&f)?
			.split_whitespace()
			.map(|x| x.parse::<u32>())
			.collect::<Result<_, _>>()?;
	}
	if let Some(f) = std::env::var_os("QWEN_FLASH_DUMP_ALL") {
		let t = std::time::Instant::now();
		let all = eng.forward_all(&ids)?;
		eprintln!(
			"  [flash] {} positions in {:.1} s",
			ids.len(),
			t.elapsed().as_secs_f64()
		);
		let b: Vec<u8> = all.iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
		std::fs::write(f, b)?;
		return Ok(());
	}
	if let Some(k) = std::env::var("QWEN_FLASH_SNAPCHECK")
		.ok()
		.and_then(|v| v.parse::<usize>().ok())
	{
		let cut = ids.len().saturating_sub(k.max(1)).max(1);
		eng.feed(&ids[..cut])?;
		let t = std::time::Instant::now();
		let snap = eng.snapshot()?;
		let bytes: usize = snap.iter().flatten().map(|c| c.len()).sum();
		eprintln!(
			"  [flash] snapshot at {cut}: {:.1} MB in {:.0} ms",
			bytes as f64 / 1e6,
			t.elapsed().as_secs_f64() * 1e3
		);
		let a = eng.forward(&ids[cut..])?;
		let a2 = eng.forward(&ids[..1])?;
		eng.reset()?;
		let t = std::time::Instant::now();
		eng.restore(cut, &snap)?;
		eprintln!(
			"  [flash] restored in {:.0} ms",
			t.elapsed().as_secs_f64() * 1e3
		);
		let b = eng.forward(&ids[cut..])?;
		let b2 = eng.forward(&ids[..1])?;
		let d = |x: &[f32], y: &[f32]| x.iter().zip(y).fold(0f32, |m, (p, q)| m.max((p - q).abs()));
		eprintln!(
			"  [flash] snapcheck: max |a - b| {:.3e} (rest of prompt), {:.3e} (one more step)",
			d(&a, &b),
			d(&a2, &b2)
		);
		return Ok(());
	}
	print!("{prompt}");
	std::io::stdout().flush().ok();
	if templated {
		crate::chat::print_open(opts.think);
	}
	let m_prefill = quant::cputime::Meter::start();
	let rows = first_rows_mm(&mut eng, &tok, &ids, &imgs)?;
	let (prefill_s, prefill_use) = m_prefill.read();
	if let Some(f) = std::env::var_os("QWEN_FLASH_DUMP_LAST") {
		let b: Vec<u8> = rows[0].iter().flat_map(|x| x.to_le_bytes()).collect();
		std::fs::write(f, b)?;
	}
	let nv = tok.vocab_size();
	let mut rng = Rng::new(opts.seed);
	let mut history = ids.clone();
	let mut sd = StreamDecoder::new();
	let _ = (quant::parallel::stats(), quant::parallel::stats_sites());
	let m_decode = quant::cputime::Meter::start();
	let (produced, steps) = decode(
		&mut eng,
		rows,
		&mut history,
		&stop,
		&opts,
		&mut rng,
		nv,
		|id| {
			let piece = sd.push(&tok, id, true);
			if !piece.is_empty() {
				print!("{piece}");
				std::io::stdout().flush().ok();
			}
		},
	)?;
	print!("{}", sd.flush());
	println!();
	let (decode_s, decode_use) = m_decode.read();
	let pstats = quant::parallel::stats();
	let psites = quant::parallel::stats_sites();
	eprintln!(
		"  [perf] flash prefill={} tok in {:.1} ms ({:.2} tok/s)  decode={produced} tok in {:.1} ms ({:.2} tok/s{})",
		ids.len(),
		prefill_s * 1000.0,
		ids.len() as f64 / prefill_s.max(1e-9),
		decode_s * 1000.0,
		produced as f64 / decode_s.max(1e-9),
		spec_note(&eng, produced, steps),
	);
	eprintln!(
		"  [eff] prefill {}  decode {}",
		quant::cputime::describe(prefill_s, prefill_use, ids.len()),
		quant::cputime::describe(decode_s, decode_use, produced)
	);
	if pstats.3 > 0 {
		eprintln!(
			"  [pool] decode work {:.2} s, spin {:.2} s, join {:.2} s, {} dispatches, {} parks",
			pstats.0, pstats.1, pstats.2, pstats.3, pstats.4
		);
		for (site, work, wall, n) in psites.iter().take(12) {
			eprintln!(
				"  [pool]   {work:8.2} s shard, {wall:7.2} s wall, {:5.1} x, {n:6} calls  {site}",
				work / wall.max(1e-9)
			);
		}
	}
	Ok(())
}

/// The engine as a `server::Engine`: logits cut to the tokenizer's vocabulary.
struct FlashServerEngine {
	eng: FlashEngine,
	nv: usize,
	max_image_tokens: usize,
}

impl server::Engine for FlashServerEngine {
	fn reset(&mut self) -> Result<(), String> {
		self.eng.reset()
	}

	fn forward(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
		let mut l = self.eng.forward(ids)?;
		l.truncate(self.nv);
		Ok(l)
	}

	fn feed(&mut self, ids: &[u32]) -> Result<(), String> {
		self.eng.feed(ids)
	}

	fn release_devices(&mut self) {
		self.eng.release_npu();
	}

	fn can_snapshot(&self) -> Option<bool> {
		Some(true)
	}

	fn piece_len(&self) -> usize {
		self.eng.piece_len()
	}

	fn step_costs(&self) -> Option<Vec<f32>> {
		self.eng.step_costs()
	}

	fn snapshot(&mut self) -> Option<server::EngineSnap> {
		match self.eng.snapshot() {
			Ok(parts) => Some(server::EngineSnap {
				pos: self.eng.pos(),
				parts,
				layout: Vec::new(),
			}),
			Err(e) => {
				eprintln!("flash: snapshot failed: {e}");
				None
			}
		}
	}

	fn restore(&mut self, snap: &server::EngineSnap) -> Result<(), String> {
		self.eng.restore(snap.pos, &snap.parts)
	}

	fn spec_k(&self) -> usize {
		self.eng.spec_k()
	}

	fn verify(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>, String> {
		let mut rows = self.eng.verify(ids)?;
		for l in rows.iter_mut() {
			l.truncate(self.nv);
		}
		Ok(rows)
	}

	fn accept(&mut self, n: usize, next: Option<u32>) -> Result<Vec<u32>, String> {
		self.eng.commit(n)?;
		match next {
			Some(t) => self.eng.drafts(t),
			None => Ok(Vec::new()),
		}
	}

	fn set_coupling(&mut self, c: Option<sampler::Coupling>) {
		self.eng.set_coupling(c);
	}

	fn feed_images(
		&mut self,
		ids: &[u32],
		images: &[(usize, &imgio::Image)],
	) -> Result<(), String> {
		let mut enc = Vec::with_capacity(images.len());
		for (_, img) in images {
			enc.push(encode_image(&mut self.eng, img, self.max_image_tokens)?);
		}
		let spans: Vec<npu::engine::ImageSpan<'_>> = images
			.iter()
			.zip(&enc)
			.map(|((start, _), (emb, g))| npu::engine::ImageSpan {
				start: *start,
				grid_w: g.mw,
				grid_h: g.mh,
				emb,
			})
			.collect();
		self.eng.feed_mm(ids, &spans)
	}
}

/// `cli <file>.npu --serve` with `arch = flash`: with a cache budget a
/// prefix cache of engine snapshots (~113 MB of recurrent state plus the
/// KV rows since the last sealed segment), with a directory its disk tier.
pub fn serve(
	path: &str,
	host: &str,
	port: u16,
	cache: &server::CacheOpts,
	max_image_tokens: usize,
) -> Result<(), Box<dyn std::error::Error>> {
	let (eng, tok, stop) = load(path)?;
	let image = eng.vision().map(|(c, _)| server::ImageGeom {
		patch: c.patch,
		merge: c.merge,
		max_tokens: max_image_tokens,
	});
	// a snapshot is only valid for the graphs and weights the `.npu` file
	// names
	let tag = format!(
		"flash:{}",
		std::fs::read_to_string(path)?
			.split_whitespace()
			.collect::<Vec<_>>()
			.join(" ")
	);
	let info = server::EngineInfo {
		name: "flash (Qwen3.8-Flash-Next, NPU)".into(),
		model: "Qwen3.8-Flash-Next".into(),
		ctx: CTX,
		ctx_train: CTX,
		n_vocab: tok.vocab_size(),
		n_embd: npu::flash::W,
		stop,
		tag,
		image,
	};
	let nv = tok.vocab_size();
	server::serve_engine(
		Box::new(FlashServerEngine {
			eng,
			nv,
			max_image_tokens,
		}),
		info,
		tok,
		host,
		port,
		cache,
		190.0,
	);
	Ok(())
}

/// `cli <file>.npu --interactive` with `arch = flash`: a multi-turn chat;
/// each turn feeds only what it adds.
pub fn interactive(
	path: &str,
	system: Option<&str>,
	initial_images: &[String],
	opts: GenOptions,
) -> Result<(), Box<dyn std::error::Error>> {
	use std::io::BufRead;
	if opts.mtp > 0 || opts.dflash.is_some() {
		return Err("--mtp / --dflash are not available on Flash-Next (its MTP head is the .npu file's `mtp = DIR`)".into());
	}
	let (mut eng, tok, stop) = load(path)?;
	println!(
		"== interactive chat (flash, npu; /reset starts over, /exit quits{})",
		if eng.vision().is_some() {
			", /image <file> attaches an image"
		} else {
			""
		}
	);
	let mut pending: Vec<String> = initial_images.to_vec();
	let think = opts.think;
	let preamble = Tokenizer::system_turn(system, think);
	let (head, open_rest) = crate::chat::open_parts(think);
	// `--no-preserve-thinking`: the state just past the last reply's
	// assistant header (pos, snapshot, history length) and its answer; the
	// next turn restores it and feeds the answer alone. (Flash-Next's
	// snapshot shares its sealed KV segments, so this costs ~the recurrent
	// state.)
	let mut last: Option<(usize, Vec<Vec<quant::snap::Chunk>>, usize, String)> = None;
	let nv = tok.vocab_size();
	let mut rng = Rng::new(opts.seed);
	let mut history: Vec<u32> = Vec::new();
	let mut lines = std::io::stdin().lock().lines();
	loop {
		print!(
			"
> "
		);
		std::io::stdout().flush().ok();
		let Some(Ok(line)) = lines.next() else { break };
		let line = line.trim().to_string();
		match line.as_str() {
			"/exit" | "/quit" => break,
			"/reset" => {
				eng.reset()?;
				history.clear();
				last = None;
				println!("  (new conversation)");
				continue;
			}
			"" => continue,
			_ => {}
		}
		if let Some(f) = line.strip_prefix("/image ") {
			let f = f.trim().to_string();
			match imgio::load(&f) {
				Ok(_) if eng.vision().is_none() => {
					println!("  this .npu file names no vision tower (vision = DIR)")
				}
				Ok(_) => {
					println!("  attached {f} (sent with your next message)");
					pending.push(f);
				}
				Err(e) => println!("  {e}"),
			}
			continue;
		}
		let mut content = String::new();
		let mut imgs = Vec::new();
		let mut failed = false;
		for f in pending.drain(..) {
			match imgio::load(&f)
				.and_then(|img| encode_image(&mut eng, &img, opts.max_image_tokens))
			{
				Ok(enc) => {
					content.push_str(&vision::placeholder(enc.1.tokens()));
					imgs.push(enc);
				}
				Err(e) => {
					println!("  {f}: {e}");
					failed = true;
				}
			}
		}
		if failed {
			continue;
		}
		content.push_str(&line);
		let mut text = String::new();
		if history.is_empty() {
			text.push_str(&preamble);
		} else {
			if let Some((pos, parts, hl, answer)) = last.take() {
				match eng.restore(pos, &parts) {
					Ok(()) => {
						history.truncate(hl);
						text.push_str(&answer);
					}
					Err(e) => println!("  (reasoning kept: restore failed: {e})"),
				}
			}
			// the reply ended on a stop token that was never fed
			text.push_str(
				"<|im_end|>
",
			);
		}
		text.push_str(&Tokenizer::chat_turn("user", &content));
		text.push_str(head);
		if think.preserve {
			text.push_str(open_rest);
		}
		let mut ids = tok.encode(&text);
		let t = std::time::Instant::now();
		let mut snap = None;
		let r = if think.preserve {
			first_rows_mm(&mut eng, &tok, &ids, &imgs)
		} else {
			(|| -> Result<Vec<Vec<f32>>, String> {
				feed_images(&mut eng, &tok, &ids, &imgs)?;
				snap = Some((eng.pos(), eng.snapshot()?, history.len() + ids.len()));
				let rest = tok.encode(open_rest);
				let rows = first_rows(&mut eng, &rest)?;
				ids.extend_from_slice(&rest);
				Ok(rows)
			})()
		};
		let rows = match r {
			Ok(l) => l,
			Err(e) => {
				println!("  {e}");
				continue;
			}
		};
		history.extend_from_slice(&ids);
		let prefill_s = t.elapsed().as_secs_f64();
		crate::chat::print_open(think);
		let t = std::time::Instant::now();
		let mut sd = StreamDecoder::new();
		let mut reply = String::new();
		let (produced, steps) = decode(
			&mut eng,
			rows,
			&mut history,
			&stop,
			&opts,
			&mut rng,
			nv,
			|id| {
				let piece = sd.push(&tok, id, true);
				if !piece.is_empty() {
					print!("{piece}");
					std::io::stdout().flush().ok();
					reply.push_str(&piece);
				}
			},
		)?;
		let tail = sd.flush();
		print!("{tail}");
		reply.push_str(&tail);
		println!();
		if let Some((pos, parts, hl)) = snap {
			let answer = crate::chat::answer_of(&reply, think).to_string();
			last = Some((pos, parts, hl, answer));
		}
		let decode_s = t.elapsed().as_secs_f64();
		eprintln!(
			"  [perf] flash prefill={} tok in {:.1} ms ({:.2} tok/s)  decode={produced} tok in {:.1} ms ({:.2} tok/s{})  context {}",
			ids.len(),
			prefill_s * 1000.0,
			ids.len() as f64 / prefill_s.max(1e-9),
			decode_s * 1000.0,
			produced as f64 / decode_s.max(1e-9),
			spec_note(&eng, produced, steps),
			eng.pos(),
		);
	}
	Ok(())
}
