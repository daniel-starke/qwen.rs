//! `--generate` / `--serve` on the NPU-only engine (`cli <file>.npu …`):
//! the `.npu` file names the lean prefill graph sets, the s = 1 decode set,
//! the embedding table and the directory whose `tokenizer.json` is used
//! (`npu::engine::Config`; `cli BF16_DIR --npu-build OUT` writes all of it). One request at a time; sampling
//! is the same `sampler` the GGUF path uses. With `vision = DIR` in the
//! `.npu` file the engine takes images: the tower's per-patch work as NPU
//! graphs, its attention on the `scan` adapter (the iGPU), else the cores
//! ([`vis_attn`]); the image's preprocessing is `vision`'s.

use std::io::Write;
use std::path::Path;

use model::suffix::{
	best_draft, ctx_policy, suffix_enabled, suffix_ev, suffix_min_engine, SuffixIndex, GRAM,
};
use npu::engine::{Config, NpuEngine};
use sampler::{sample, Rng};
use tokenizer::{StreamDecoder, Tokenizer};

use crate::chat::GenOptions;

/// True when `path` is an NPU engine description.
pub fn is_npu(path: &str) -> bool {
	path.ends_with(".npu")
}

/// What [`first_rows`] and [`decode`] need of an engine: the 27B's NPU
/// engine and Flash-Next's (`flashchat`).
pub(crate) trait SpecEngine {
	fn spec_k(&self) -> usize;
	fn verify(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>, String>;
	fn commit(&mut self, n: usize) -> Result<(), String>;
	fn drafts(&mut self, next: u32) -> Result<Vec<u32>, String>;
	fn forward(&mut self, ids: &[u32]) -> Result<Vec<f32>, String>;
	fn feed(&mut self, ids: &[u32]) -> Result<(), String>;
	fn pos(&self) -> usize;
	fn context(&self) -> usize;
	/// The sampler's shared noise for the drafts (`server::coupled`).
	fn set_coupling(&mut self, _c: Option<sampler::Coupling>) {}
	/// `server::Engine::step_costs`.
	fn step_costs(&self) -> Option<Vec<f32>> {
		None
	}
}

impl SpecEngine for NpuEngine {
	fn spec_k(&self) -> usize {
		NpuEngine::spec_k(self)
	}
	fn verify(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>, String> {
		NpuEngine::verify(self, ids)
	}
	fn commit(&mut self, n: usize) -> Result<(), String> {
		NpuEngine::commit(self, n)
	}
	fn drafts(&mut self, next: u32) -> Result<Vec<u32>, String> {
		NpuEngine::drafts(self, next)
	}
	fn forward(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
		NpuEngine::forward(self, ids)
	}
	fn feed(&mut self, ids: &[u32]) -> Result<(), String> {
		NpuEngine::feed(self, ids)
	}
	fn pos(&self) -> usize {
		NpuEngine::pos(self)
	}
	fn context(&self) -> usize {
		NpuEngine::context(self)
	}
}

/// What the image helpers need of an engine with a vision tower: the 27B's
/// NPU engine and Flash-Next's.
pub(crate) trait VisEngine: SpecEngine {
	fn vision(&self) -> Option<(&npu::vis::VisCfg, &[f32])>;
	fn encode_image(
		&mut self,
		patches: &[f32],
		pos: &[f32],
		hw: &[(u32, u32)],
	) -> Result<Vec<f32>, String>;
	fn feed_mm(&mut self, ids: &[u32], spans: &[npu::engine::ImageSpan<'_>]) -> Result<(), String>;
}

impl VisEngine for NpuEngine {
	fn vision(&self) -> Option<(&npu::vis::VisCfg, &[f32])> {
		NpuEngine::vision(self)
	}
	fn encode_image(
		&mut self,
		patches: &[f32],
		pos: &[f32],
		hw: &[(u32, u32)],
	) -> Result<Vec<f32>, String> {
		NpuEngine::encode_image(self, patches, pos, hw)
	}
	fn feed_mm(&mut self, ids: &[u32], spans: &[npu::engine::ImageSpan<'_>]) -> Result<(), String> {
		NpuEngine::feed_mm(self, ids, spans)
	}
}

/// Feed `ids` and return the logits that follow: of the last token, or,
/// on an engine that drafts, of the verify step that fed the last token
/// (its drafts come from [`decode`]'s first accept).
pub(crate) fn first_rows<E: SpecEngine>(eng: &mut E, ids: &[u32]) -> Result<Vec<Vec<f32>>, String> {
	if eng.spec_k() > 1 {
		if ids.len() > 1 {
			eng.feed(&ids[..ids.len() - 1])?;
		}
		eng.verify(&ids[ids.len() - 1..])
	} else {
		eng.forward(ids).map(|l| vec![l])
	}
}

/// Sample up to `max_new` tokens after `rows` (from [`first_rows`]), each
/// passed to `emit` and pushed on `history`, until a stop token. On an engine
/// that drafts (`spec_k` > 1) each step verifies the sampled token plus the
/// MTP drafts and keeps those the samples reproduce; the last token sampled
/// is not fed (as after a stop token). Returns (tokens, steps).
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode<E: SpecEngine>(
	eng: &mut E,
	mut rows: Vec<Vec<f32>>,
	history: &mut Vec<u32>,
	stop: &[u32],
	opts: &GenOptions,
	rng: &mut Rng,
	nv: usize,
	mut emit: impl FnMut(u32),
) -> Result<(usize, usize), String> {
	let k = eng.spec_k();
	let mut fed: Vec<u32> = history[history.len() - 1..].to_vec();
	let (mut produced, mut steps) = (0usize, 0usize);
	// context drafts (`model::suffix`) in place of the MTP's on a long match
	let mut ix = (k > 1 && suffix_enabled()).then(|| SuffixIndex::new(history));
	let mut bar = quant::progress::Decode::start(quant::progress::Show::Title);
	// the penalties see this reply's tokens only
	let start = history.len();
	let coupling =
		(k > 1 && opts.params.temperature > 0.0 && server::coupled()).then(|| sampler::Coupling {
			seed: rng.next_u64(),
			params: opts.params,
		});
	eng.set_coupling(coupling);
	// whether the last step's drafts were the model drafter's
	let mut model_last: Option<bool> = None;
	ctx_policy()
		.lock()
		.unwrap_or_else(|e| e.into_inner())
		.begin();
	loop {
		let (mut next, mut n, mut done) = (0, 0, false);
		for (j, l) in rows.iter().enumerate() {
			let l = &l[..l.len().min(nv)];
			next = match &coupling {
				Some(c) => sampler::sample_coupled(
					l,
					&history[start..],
					&opts.params,
					c,
					history.len() as u64,
				),
				None => sample(l, &history[start..], &opts.params, rng),
			};
			n = j + 1;
			if stop.contains(&next) {
				done = true;
				break;
			}
			history.push(next);
			if let Some(ix) = ix.as_mut() {
				ix.push(next);
			}
			produced += 1;
			bar.tokens(1);
			emit(next);
			if produced == opts.max_new {
				done = true;
				break;
			}
			if fed.get(j + 1) != Some(&next) {
				break;
			}
		}
		steps += 1;
		if k > 1 {
			bar.step();
		}
		if std::env::var_os("QWEN_NPU_SPEC_DEBUG").is_some() {
			eprintln!(
				"  [spec] pos {} fed {fed:?} -> kept {n}, next {next}",
				eng.pos()
			);
		}
		if k > 1 {
			eng.commit(n)?;
			if done {
				return Ok((produced, steps));
			}
			let room = (k - 1).min(eng.context().saturating_sub(eng.pos() + 1));
			let costs = eng.step_costs().filter(|_| suffix_ev());
			if let (Some(c), Some(true)) = (costs.as_deref(), model_last) {
				ctx_policy()
					.lock()
					.unwrap_or_else(|e| e.into_inner())
					.model_step(fed.len() - 1, n - 1, c);
			}
			let min = if costs.is_some() {
				GRAM
			} else {
				suffix_min_engine()
			};
			let ctxd = ix
				.as_ref()
				.map(|ix| best_draft(ix, None, room))
				.filter(|(d, m)| *m >= min && !d.is_empty());
			// by expected tokens per step cost (`model::suffix::CtxPolicy`)
			let ctxd = match (costs.as_deref(), ix.as_ref()) {
				(Some(c), Some(ix)) => {
					let mut p = ctx_policy().lock().unwrap_or_else(|e| e.into_inner());
					p.observe(ix.tail(ix.len()));
					ctxd.and_then(|(d, m)| p.choose(ix.len(), &d, m, c).map(|k| d[..k].to_vec()))
				}
				_ => ctxd.map(|(d, _)| d),
			};
			model_last = Some(ctxd.is_none());
			let drafts = match ctxd {
				Some(d) => d,
				None => eng.drafts(next)?,
			};
			fed = std::iter::once(next).chain(drafts).take(k).collect();
			rows = eng.verify(&fed)?;
		} else {
			if done {
				return Ok((produced, steps));
			}
			fed = vec![next];
			rows = vec![eng.forward(&fed)?];
		}
	}
}

pub(crate) fn spec_note<E: SpecEngine>(eng: &E, produced: usize, steps: usize) -> String {
	if eng.spec_k() > 1 {
		format!(", {:.2} tok/step", produced as f64 / steps.max(1) as f64)
	} else {
		String::new()
	}
}

/// The recurrence of cut delta layers on a Vulkan adapter (`scan = intel`
/// in the `.npu` file: the iGPU), `offload::DeltaScan`.
#[cfg(feature = "vulkan")]
pub(crate) fn scan_device(dev: &str) -> Result<Box<dyn npu::engine::Scan>, String> {
	struct GpuScan(offload::DeltaScan);
	impl npu::engine::Scan for GpuScan {
		fn set_state(&mut self, di: usize, st: Option<&[f32]>) -> Result<(), String> {
			self.0.set_state(di, st)
		}
		fn state(&mut self, di: usize) -> Result<Vec<f32>, String> {
			self.0.state(di)
		}
		fn set_states(
			&mut self,
			n: usize,
			_len: usize,
			fill: &mut dyn FnMut(usize, &mut [f32]) -> bool,
		) -> Result<(), String> {
			self.0.set_states(n, fill)
		}
		fn get_states(
			&mut self,
			n: usize,
			take: &mut dyn FnMut(usize, &[f32]),
		) -> Result<(), String> {
			self.0.get_states(n, take)
		}
		fn scan(
			&mut self,
			di: usize,
			n: usize,
			qn: &[u8],
			kn: &[u8],
			v: &[u8],
			gb: &[u8],
			o: &mut [u8],
			oscale: f32,
		) -> Result<(), String> {
			self.0.scan(di, n, qn, kn, v, gb, o, oscale)
		}
		fn scan_begin(
			&mut self,
			di: usize,
			n: usize,
			qn: &[u8],
			kn: &[u8],
			v: &[u8],
			gb: &[u8],
			o: &mut [u8],
			oscale: f32,
		) -> Result<(), String> {
			self.0.scan_begin(di, n, qn, kn, v, gb, o, oscale)
		}
		fn scan_end(&mut self) -> Result<(), String> {
			self.0.scan_end()
		}
		fn set_gnorm(&mut self, di: usize, w: &[f32]) -> Result<bool, String> {
			self.0.set_gnorm(di, w).map(|_| true)
		}
		fn scan_begin_gn(
			&mut self,
			di: usize,
			n: usize,
			qn: &[u8],
			kn: &[u8],
			v: &[u8],
			gb: &[u8],
			z: &[u8],
			o: &mut [u8],
			oscale: f32,
		) -> Result<(), String> {
			self.0.scan_begin_gn(di, n, qn, kn, v, gb, z, o, oscale)
		}
		fn set_conv(&mut self, di: usize, w: &[f32]) -> Result<bool, String> {
			self.0.set_conv(di, w).map(|_| true)
		}
		fn conv_next(
			&mut self,
			di: usize,
			n: usize,
			cpast: &[u8],
			qkv: &[u8],
			qn: &mut [u8],
			kn: &mut [u8],
			v: &mut [u8],
		) -> Result<(), String> {
			self.0.conv_next(di, n, cpast, qkv, qn, kn, v)
		}
		fn poke(&mut self) -> Result<(), String> {
			self.0.poke()
		}
		fn far_begin(
			&mut self,
			s: usize,
			nk: usize,
			qg: &[u8],
			blocks: &[(&[u8], &[u8])],
			fresh: bool,
		) -> Result<bool, String> {
			self.0.far_begin(s, nk, qg, blocks, fresh).map(|_| true)
		}
		fn far_poll(&mut self) -> Result<bool, String> {
			self.0.far_poll()
		}
		fn far_wait(&mut self) -> Result<(), String> {
			self.0.far_wait()
		}
		fn far_out(&self) -> (&[f32], &[[f32; 2]]) {
			self.0.far_out()
		}
	}
	Ok(Box::new(GpuScan(offload::DeltaScan::new(Some(dev), 48)?)))
}

#[cfg(not(feature = "vulkan"))]
pub(crate) fn scan_device(_dev: &str) -> Result<Box<dyn npu::engine::Scan>, String> {
	Err("scan = ... needs a build with --features vulkan".into())
}

/// The MLPs' share on the Vulkan adapter `dev` names (`mlpgpu = ADAPTER`
/// in the `.npu` file: a discrete card), `offload::MlpSplit`.
/// `QWEN_NPU_MLP_GPU_ROWS`: rows per layer at most (default all the budget
/// holds, less `QWEN_NPU_MLP_GPU_RESERVE_MB`, default 512: room for other
/// applications, e.g. an editor started later). `QWEN_NPU_MLP_GPU_FRAC=F2,F4`: the
/// share of those rows the GPU computes at 2 and at 3-4 positions (default
/// 0.9,0.8).
#[cfg(feature = "vulkan")]
fn mlp_gpu(dev: String) -> npu::engine::MlpGpuNew {
	struct Split(offload::MlpSplit, [f64; 2]);
	impl npu::mlp::MlpGpu for Split {
		fn rows(&self, s: usize) -> usize {
			// more positions cost the GPU more than the cores: it keeps
			// fewer rows then (balanced on `QWEN_NPU_MLP_TIMING=1`'s `gpu wait`)
			let f = if s == 1 { 1.0 } else { self.1[(s - 2).min(1)] };
			((self.0.rows() as f64 * f / 1024.0).round() as usize * 1024).clamp(1024, self.0.rows())
		}
		fn max_s(&self) -> usize {
			4
		}
		fn begin(&mut self, l: usize, s: usize, ab: [npu::mlp::Act<'_>; 2]) -> Result<(), String> {
			self.0.begin(
				l,
				s,
				self.rows(s),
				ab.map(|a| (a.even, a.odd, a.inv, a.bias)),
			)
		}
		fn finish(&mut self, s: usize, y: &mut [f32]) -> Result<(), String> {
			self.0.finish(s, y)
		}
	}
	Box::new(move |mlps| {
		let env = |k: &str, d: usize| {
			std::env::var(k)
				.ok()
				.and_then(|v| v.parse().ok())
				.unwrap_or(d)
		};
		if mlps.iter().any(|m| !m.hadamard()) {
			return Err("the set's rotations are not Hadamard transforms".into());
		}
		let mut g = offload::MlpSplit::new(
			Some(&dev),
			mlps.len(),
			env("QWEN_NPU_MLP_GPU_ROWS", 17408),
			(env("QWEN_NPU_MLP_GPU_RESERVE_MB", 512) as u64) << 20,
		)?;
		let fr: Vec<f64> = std::env::var("QWEN_NPU_MLP_GPU_FRAC")
			.ok()
			.map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
			.unwrap_or_default();
		let fr = if fr.len() == 2 {
			[fr[0], fr[1]]
		} else {
			[0.9, 0.8]
		};
		// a sliver of a share while something else holds the VRAM is not
		// worth its paging: wait for the next attempt
		if g.rows() < 4096 {
			return Err(format!("VRAM for {} rows per layer only", g.rows()));
		}
		// the budget does not show what another application holds: uploads
		// that crawl (~25 ms per layer otherwise) do
		let t = std::time::Instant::now();
		for (l, m) in mlps.iter().enumerate() {
			let (gate, up, down, sg, su, d3, sd) = m.gpu_layer();
			g.add_layer(gate, up, down, sg, su, d3, sd)?;
			if t.elapsed().as_secs_f64() > 0.5 + 0.05 * (l + 1) as f64 {
				return Err(format!(
					"{} layers took {:.1} s to upload: the VRAM is in use elsewhere",
					l + 1,
					t.elapsed().as_secs_f64()
				));
			}
		}
		Ok(Box::new(Split(g, fr)) as Box<dyn npu::mlp::MlpGpu>)
	})
}

#[cfg(not(feature = "vulkan"))]
fn mlp_gpu(_dev: String) -> npu::engine::MlpGpuNew {
	Box::new(|_| Err("mlpgpu needs a build with --features vulkan".into()))
}

/// The vision tower's configuration as `vision`'s (for its preprocessing).
fn vis_cfg(c: &npu::vis::VisCfg) -> vision::VisionConfig {
	vision::VisionConfig {
		image_size: 0,
		patch_size: c.patch,
		n_embd: c.embd,
		n_ff: 0,
		n_layer: c.layers,
		n_head: c.heads,
		proj_dim: c.proj,
		merge_size: c.merge,
		eps: 1e-6,
		mean: c.mean,
		std: c.std,
		n_pos: c.n_pos,
	}
}

/// The tower's attention on the cores (`vision::tower`): without a `scan`
/// adapter, or when the adapter's kernels do not fit the tower.
struct CoreAttn(vision::VisionConfig);

impl npu::vis::VisAttn for CoreAttn {
	fn attend(
		&mut self,
		_n: usize,
		qkv: &[u8],
		hw: &[(u32, u32)],
		out: &mut [u8],
	) -> Result<(), String> {
		vision::tower::attend_f16(&self.0, hw, qkv, out);
		Ok(())
	}
	fn name(&self) -> String {
		"cores".into()
	}
}

/// What runs the tower's attention: the Vulkan adapter `dev` names (`scan =
/// intel` of the `.npu` file: the iGPU, `offload::VisAttn`), else the cores.
#[cfg(feature = "vulkan")]
pub(crate) fn vis_attn(dev: Option<&str>, c: &npu::vis::VisCfg) -> Box<dyn npu::vis::VisAttn> {
	struct Gpu(offload::VisAttn, usize, String);
	impl npu::vis::VisAttn for Gpu {
		fn attend(
			&mut self,
			_n: usize,
			qkv: &[u8],
			hw: &[(u32, u32)],
			out: &mut [u8],
		) -> Result<(), String> {
			self.0.attend(self.1, hw, qkv, out)
		}
		fn name(&self) -> String {
			self.2.clone()
		}
	}
	// (the kernels are written for heads of 72 dims)
	match dev
		.filter(|_| c.embd / c.heads == 72)
		.map(|d| (d, offload::VisAttn::new(Some(d))))
	{
		Some((d, Ok(a))) => return Box::new(Gpu(a, c.heads, format!("Vulkan adapter '{d}'"))),
		Some((d, Err(e))) => {
			eprintln!("npu: vision attention on '{d}' failed ({e}); the cores take it")
		}
		None => {}
	}
	Box::new(CoreAttn(vis_cfg(c)))
}

#[cfg(not(feature = "vulkan"))]
pub(crate) fn vis_attn(_dev: Option<&str>, c: &npu::vis::VisCfg) -> Box<dyn npu::vis::VisAttn> {
	Box::new(CoreAttn(vis_cfg(c)))
}

/// The grid an image is resized to, at most `max_tokens` tokens.
fn image_grid(
	c: &npu::vis::VisCfg,
	img: &imgio::Image,
	max_tokens: usize,
) -> Result<vision::Grid, String> {
	vision::smart_resize(
		img.width,
		img.height,
		c.patch,
		c.merge,
		vision::DEFAULT_MIN_TOKENS,
		max_tokens.max(1),
	)
}

/// One image through the engine's tower: its embeddings (`[tokens][width]`)
/// and its grid.
pub(crate) fn encode_image<E: VisEngine>(
	eng: &mut E,
	img: &imgio::Image,
	max_tokens: usize,
) -> Result<(Vec<f32>, vision::Grid), String> {
	let (c, pos_w) = eng
		.vision()
		.ok_or("this .npu file names no vision tower (vision = DIR), so the engine is text only")?;
	let (c, vc) = (c.clone(), vis_cfg(c));
	let grid = image_grid(&c, img, max_tokens)?;
	let t = std::time::Instant::now();
	let patches = vision::patchify(img, &grid, c.patch, c.merge, c.mean, c.std);
	let pos = vision::tower::position_embedding(&vc, pos_w, &grid);
	let hw = vision::patch_positions(&grid, c.merge);
	let emb = eng.encode_image(&patches, &pos, &hw)?;
	let s = t.elapsed().as_secs_f64();
	eprintln!(
		"  [image] {}x{} px -> {}x{} patches -> {} tokens in {:.2} s ({:.0} tok/s)",
		grid.width,
		grid.height,
		grid.gw,
		grid.gh,
		grid.tokens(),
		s,
		grid.tokens() as f64 / s.max(1e-9)
	);
	Ok((emb, grid))
}

/// Feed `ids`, whose runs of `<|image_pad|>` stand for `images` (embeddings
/// and grid, in order).
pub(crate) fn feed_images<E: VisEngine>(
	eng: &mut E,
	tok: &Tokenizer,
	ids: &[u32],
	images: &[(Vec<f32>, vision::Grid)],
) -> Result<(), String> {
	if images.is_empty() {
		return eng.feed(ids);
	}
	let pad = vision::VisionTokens::resolve(tok)?.pad;
	let mut spans = Vec::with_capacity(images.len());
	let mut i = 0;
	for (emb, g) in images {
		while i < ids.len() && ids[i] != pad {
			i += 1;
		}
		let run = ids[i..].iter().take_while(|&&x| x == pad).count();
		if run != g.tokens() {
			return Err(format!("prompt: an image needs {} placeholder tokens, the encoded prompt has a run of {run}", g.tokens()));
		}
		spans.push(npu::engine::ImageSpan {
			start: i,
			grid_w: g.mw,
			grid_h: g.mh,
			emb,
		});
		i += run;
	}
	if ids[i..].contains(&pad) {
		return Err("prompt: more placeholder runs than images".into());
	}
	eng.feed_mm(ids, &spans)
}

/// [`first_rows`] for a prompt with images.
pub(crate) fn first_rows_mm<E: VisEngine>(
	eng: &mut E,
	tok: &Tokenizer,
	ids: &[u32],
	images: &[(Vec<f32>, vision::Grid)],
) -> Result<Vec<Vec<f32>>, String> {
	if images.is_empty() {
		return first_rows(eng, ids);
	}
	// (the last token is the assistant header's, never an image's)
	feed_images(eng, tok, &ids[..ids.len() - 1], images)?;
	first_rows(eng, &ids[ids.len() - 1..])
}

fn load(path: &str) -> Result<(NpuEngine, Tokenizer, Vec<u32>), Box<dyn std::error::Error>> {
	let cfg = Config::parse(path)?;
	let tok = Tokenizer::from_hf_json(&Path::new(&cfg.tokenizer).join("tokenizer.json"))
		.map_err(|e| format!("tokenizer: {e}"))?;
	let scan = match &cfg.scan {
		Some(dev) => Some(scan_device(dev)?),
		None => None,
	};
	let mut eng = NpuEngine::load_with_scan(&cfg, scan)?;
	if let Some(dev) = cfg.mlpgpu.clone() {
		eng.set_mlp_gpu(mlp_gpu(dev));
	}
	if let Some(c) = eng.vision().map(|v| v.0.clone()) {
		let a = vis_attn(cfg.scan.as_deref(), &c);
		eprintln!(
			"npu: vision tower: {} blocks on the NPU, their attention on {}",
			c.layers,
			a.name()
		);
		eng.set_vis_attn(a);
	}
	let mut stop = vec![tok.eos()];
	stop.extend(tok.token_id("<|endoftext|>"));
	Ok((eng, tok, stop))
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
		return Err("--mtp / --dflash are not available on the NPU engine".into());
	}
	println!("== generate (npu)");
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
	// `QWEN_NPU_IDS=FILE` (diagnostic): these token ids (whitespace-
	// separated) instead of the prompt's, e.g. an eval text's `ids.txt`.
	if let Some(f) = std::env::var_os("QWEN_NPU_IDS") {
		let t = std::fs::read_to_string(&f)?;
		ids = t
			.split_whitespace()
			.map(|x| x.parse::<u32>())
			.collect::<Result<_, _>>()?;
	}
	print!("{prompt}");
	std::io::stdout().flush().ok();
	if templated {
		crate::chat::print_open(opts.think);
	}
	let t = std::time::Instant::now();
	// `QWEN_NPU_CONT=k[,r]` (diagnostic): feed the first k prompt tokens,
	// snapshot, reset, restore (`r` = 0 skips the round trip), then the rest;
	// the prefix cache's path, to compare against one prefill.
	let cont = std::env::var("QWEN_NPU_CONT").ok().map(|v| {
		let mut it = v.split(',').map(|x| x.trim().parse::<usize>().unwrap_or(0));
		(it.next().unwrap_or(0), it.next().unwrap_or(1) != 0)
	});
	let rows = match cont {
		Some((k, rt)) if k > 0 && k < ids.len() => {
			eng.feed(&ids[..k])?;
			if rt {
				let (pos, parts) = (eng.pos(), eng.snapshot()?);
				eng.reset()?;
				eng.restore(pos, &parts.iter().map(|p| &p[..]).collect::<Vec<_>>())?;
			}
			first_rows(&mut eng, &ids[k..])?
		}
		_ => first_rows_mm(&mut eng, &tok, &ids, &imgs)?,
	};
	{
		let l = &rows[0][..rows[0].len().min(tok.vocab_size())];
		let mut top: Vec<(usize, f32)> = l.iter().copied().enumerate().collect();
		top.sort_by(|a, b| b.1.total_cmp(&a.1));
		eprintln!("  [npu] first-token top: {:?}", &top[..3]);
		// `QWEN_NPU_DUMP=FILE`: those logits, raw f32.
		if let Some(f) = std::env::var_os("QWEN_NPU_DUMP") {
			std::fs::write(
				f,
				l.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
			)?;
		}
	}
	let prefill_s = t.elapsed().as_secs_f64();
	let nv = tok.vocab_size();
	let mut rng = Rng::new(opts.seed);
	let mut history = ids.clone();
	let mut sd = StreamDecoder::new();
	let t = std::time::Instant::now();
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
	let decode_s = t.elapsed().as_secs_f64();
	eprintln!(
		"  [perf] npu prefill={} tok in {:.1} ms ({:.2} tok/s)  decode={produced} tok in {:.1} ms ({:.2} tok/s{})",
		ids.len(),
		prefill_s * 1000.0,
		ids.len() as f64 / prefill_s.max(1e-9),
		decode_s * 1000.0,
		produced as f64 / decode_s.max(1e-9),
		spec_note(&eng, produced, steps),
	);
	Ok(())
}

/// The engine as a `server::Engine`: logits cut to the tokenizer's vocabulary.
struct NpuServerEngine {
	eng: NpuEngine,
	nv: usize,
	max_image_tokens: usize,
}

impl server::Engine for NpuServerEngine {
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

	fn snapshot(&mut self) -> Option<server::EngineSnap> {
		match self.eng.snapshot_rows() {
			// (the KV parts as rows: the prefix cache pages them and shares
			// the pages between checkpoints)
			Ok((parts, rows)) => Some(server::EngineSnap {
				pos: self.eng.pos(),
				parts: parts.into_iter().map(quant::snap::part).collect(),
				layout: rows
					.into_iter()
					.map(|r| match r {
						Some((streams, rows)) => server::Layout::Rows { streams, rows },
						None => server::Layout::Opaque,
					})
					.collect(),
			}),
			Err(e) => {
				eprintln!("npu: snapshot failed: {e}");
				None
			}
		}
	}

	fn restore(&mut self, snap: &server::EngineSnap) -> Result<(), String> {
		// (one chunk per part as made above, or as read back from disk)
		let parts: Vec<_> = snap.parts.iter().map(|p| quant::snap::concat(p)).collect();
		self.eng
			.restore(snap.pos, &parts.iter().map(|p| &p[..]).collect::<Vec<_>>())
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

/// `cli <file>.npu --serve`: with a cache budget a prefix cache of engine
/// snapshots (~75 MB of recurrent state per checkpoint, 64 KB per position
/// shared between a conversation's checkpoints), with a directory its disk
/// tier.
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
	// A snapshot is only valid for the graph sets and weights it was made
	// with: the `.npu` file names them.
	let tag = format!(
		"npu:{}",
		std::fs::read_to_string(path)?
			.split_whitespace()
			.collect::<Vec<_>>()
			.join(" ")
	);
	let info = server::EngineInfo {
		name: "npu (Level Zero, lean graphs)".into(),
		model: server::DEFAULT_MODEL.into(),
		ctx: eng.context(),
		ctx_train: 262144,
		n_vocab: tok.vocab_size(),
		n_embd: 5120,
		stop,
		tag,
		image,
	};
	let nv = tok.vocab_size();
	server::serve_engine(
		Box::new(NpuServerEngine {
			eng,
			nv,
			max_image_tokens,
		}),
		info,
		tok,
		host,
		port,
		cache,
		130.0,
	);
	Ok(())
}

/// `cli <file>.npu --interactive`: a multi-turn chat. Each turn feeds only
/// what it adds (the previous reply's end, the user turn, the assistant
/// header), through the prefill graphs from the conversation's state.
pub fn interactive(
	path: &str,
	system: Option<&str>,
	initial_images: &[String],
	opts: GenOptions,
) -> Result<(), Box<dyn std::error::Error>> {
	use std::io::BufRead;
	if opts.mtp > 0 || opts.dflash.is_some() {
		return Err("--mtp / --dflash are not available on the NPU engine".into());
	}
	let (mut eng, tok, stop) = load(path)?;
	println!(
		"== interactive chat (npu; /reset starts over, /exit quits{})",
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
	// next turn restores it and feeds the answer alone.
	let mut last: Option<(usize, Vec<Vec<u8>>, usize, String)> = None;
	let nv = tok.vocab_size();
	let mut rng = Rng::new(opts.seed);
	let mut history: Vec<u32> = Vec::new();
	let mut lines = std::io::stdin().lock().lines();
	let mut turn = 0usize;
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
				turn = 0;
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
		if turn == 0 {
			text.push_str(&preamble);
		} else {
			if let Some((pos, parts, hl, answer)) = last.take() {
				let r = eng.restore(pos, &parts.iter().map(|p| &p[..]).collect::<Vec<_>>());
				match r {
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
			"  [perf] npu prefill={} tok in {:.1} ms ({:.2} tok/s)  decode={produced} tok in {:.1} ms ({:.2} tok/s{})  context {}",
			ids.len(),
			prefill_s * 1000.0,
			ids.len() as f64 / prefill_s.max(1e-9),
			decode_s * 1000.0,
			produced as f64 / decode_s.max(1e-9),
			spec_note(&eng, produced, steps),
			eng.pos(),
		);
		turn += 1;
	}
	println!();
	Ok(())
}
