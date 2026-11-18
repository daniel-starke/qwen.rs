//! Generation modes: one-shot `--generate`, the multi-turn `--interactive`
//! session (the M6 leftover), and the `--image-probe` diagnostic.
//!
//! Both generation paths share one session type so image handling, incremental
//! prefill and sampling are written once. The session keeps a single
//! `RunState` for its whole life: every turn prefills **only the tokens it
//! adds**, so an N-turn conversation costs one prefill of the conversation, not
//! N of them.

use std::io::Write;

use model::Model;
use sampler::{sample, Rng, SampleParams};
use tokenizer::{StreamDecoder, Thinking, Tokenizer};
use vision::VisionModel;

/// Everything a generation run needs beyond the model itself.
pub struct GenOptions {
	pub max_new: usize,
	pub params: SampleParams,
	pub seed: u64,
	/// Cap on merged tokens per image.
	pub max_image_tokens: usize,
	/// Tokens the MTP head drafts per step; 0 disables speculative decoding.
	pub mtp: usize,
	/// DFlash2 drafter (directory or `model.safetensors`); exclusive with `mtp`.
	pub dflash: Option<String>,
	/// The chat template's thinking switches (`--think`,
	/// `--no-preserve-thinking`).
	pub think: Thinking,
}

/// What a chat turn adds after the user turn, split where `--interactive`
/// with `preserve: false` rewinds to: the assistant header, then the rest of
/// the generation prompt (the open `<think>\n`, or the closed empty block).
pub fn open_parts(think: Thinking) -> (&'static str, &'static str) {
	let open = think.open();
	(
		Tokenizer::ASSISTANT_OPEN,
		&open[Tokenizer::ASSISTANT_OPEN.len()..],
	)
}

/// A reply's answer as the template replays it without its reasoning
/// (`preserve_thinking: false`): what follows the last `</think>`, trimmed.
/// A reply that never closed its think block (cut off by `--max-new`) has
/// none.
pub fn answer_of(reply: &str, think: Thinking) -> &str {
	match reply.rfind("</think>") {
		Some(i) => reply[i + "</think>".len()..].trim(),
		None if think.enabled() => "",
		None => reply.trim(),
	}
}

/// The line `--think` prints after the prompt, so the transcript shows the
/// block the template opened.
pub fn print_open(think: Thinking) {
	if think.enabled() {
		println!("<think>");
		std::io::stdout().flush().ok();
	}
}

/// MTP totals over a session, for the `[perf]` line.
pub use model::MtpStats;

/// A live conversation against one `RunState`.
pub struct Session<'a> {
	model: &'a Model,
	tok: &'a Tokenizer,
	vm: Option<&'a VisionModel>,
	rs: model::RunState,
	opts: GenOptions,
	rng: Rng,
	/// Every token fed so far, for the repetition penalty.
	history: Vec<u32>,
	logits: Vec<f32>,
	primed: bool,
	/// Speculative-decoding counters, summed over every `generate`.
	pub mtp: MtpStats,
}

impl<'a> Session<'a> {
	pub fn new(
		model: &'a Model,
		tok: &'a Tokenizer,
		vm: Option<&'a VisionModel>,
		opts: GenOptions,
	) -> Session<'a> {
		let rs = model.new_run_state();
		let rng = Rng::new(opts.seed);
		Session {
			logits: vec![0.0; model.hp.n_vocab],
			model,
			tok,
			vm,
			rs,
			opts,
			rng,
			history: Vec::new(),
			primed: false,
			mtp: MtpStats::default(),
		}
	}

	/// Feed `text` (already rendered, including any `<|image_pad|>` runs) plus
	/// the images those runs stand for. Only the new tokens are prefilled.
	pub fn feed(
		&mut self,
		text: &str,
		images: Vec<(usize, usize, Vec<f32>)>,
	) -> Result<usize, String> {
		let ids = self.tok.encode(text);
		if ids.is_empty() {
			return Ok(0);
		}
		let n = ids.len();
		let logits = if images.is_empty() {
			self.model.forward_prefill(&mut self.rs, &ids, None).0
		} else {
			// `start_pos` is the MRoPE clock, not `rs.pos`; after an image the
			// two are permanently different.
			let start = self.rs.mpos[0];
			let mm = vision::build_raw(self.tok, ids.clone(), images, start)?;
			let mut scratch = Vec::new();
			let input = mm.input(&mut scratch, self.model.hp.n_embd);
			self.model.forward_prefill_mm(&mut self.rs, input, None).0
		};
		self.history.extend_from_slice(&ids);
		self.logits = logits;
		self.primed = true;
		Ok(n)
	}

	/// Render one image into a placeholder run and its projector output.
	pub fn encode_image(
		&self,
		img: &imgio::Image,
	) -> Result<(String, (usize, usize, Vec<f32>)), String> {
		let vm = self
			.vm
			.ok_or("images need --mmproj <mmproj-*.gguf>; none was given")?;
		let (emb, grid) = vm.encode(img, self.opts.max_image_tokens)?;
		eprintln!(
			"  [image] {}x{} px -> {}x{} patches -> {} tokens",
			grid.width,
			grid.height,
			grid.gw,
			grid.gh,
			grid.tokens()
		);
		Ok((vision::placeholder(grid.tokens()), (grid.mw, grid.mh, emb)))
	}

	/// Sample until EOS or `max_new`, streaming each piece to stdout. Returns
	/// the decoded text and how many tokens it took.
	pub fn generate(&mut self) -> (String, usize) {
		assert!(self.primed, "generate() before feed()");
		let mut out = String::new();
		let mut produced = 0usize;
		// Byte-level BPE splits multi-byte characters across tokens, so the
		// pieces have to be reassembled rather than decoded one at a time.
		let mut sd = StreamDecoder::new();
		// With MTP the decoder drafts and verifies several positions per pass;
		// sampling below is unchanged, one token at a time from the main
		// model's logits.
		let mut dec = model::Decoder::with_context(std::mem::take(&mut self.logits), &self.history);
		let mut bar = quant::progress::Decode::start(quant::progress::Show::Title);
		// the penalties see this reply's tokens only
		let start = self.history.len();
		for _ in 0..self.opts.max_new {
			let next = sample(
				dec.logits(),
				&self.history[start..],
				&self.opts.params,
				&mut self.rng,
			);
			if next == self.tok.eos() {
				break;
			}
			self.history.push(next);
			produced += 1;
			bar.tokens(1);
			let piece = sd.push(self.tok, next, true);
			if !piece.is_empty() {
				print!("{piece}");
				std::io::stdout().flush().ok();
				out.push_str(&piece);
			}
			dec.feed(self.model, &mut self.rs, next);
		}
		self.logits = dec.logits().to_vec();
		dec.finish(self.model, &mut self.rs);
		self.mtp.add(&dec.stats);
		let tail = sd.flush();
		if !tail.is_empty() {
			print!("{tail}");
			std::io::stdout().flush().ok();
			out.push_str(&tail);
		}
		(out, produced)
	}

	pub fn pos(&self) -> usize {
		self.rs.pos
	}

	/// A point [`Session::rewind`] can return to.
	pub fn checkpoint(&self) -> (model::Checkpoint, usize) {
		(self.rs.checkpoint(), self.history.len())
	}

	/// Forget every token since `cp`; the next [`Session::feed`] continues
	/// from there.
	pub fn rewind(&mut self, cp: (model::Checkpoint, usize)) -> bool {
		if !self.rs.rewind(cp.0) {
			return false;
		}
		self.history.truncate(cp.1);
		true
	}

	/// `QWEN_FEAT_DUMP=DIR`: write the whole token history to `DIR/tokens.bin`
	/// (`u32`) and re-prefill it from scratch with [`model::set_feature_dump`]
	/// on, so `DIR/feat_{il}.bin` holds every position's hidden state at the
	/// layers a DFlash2 drafter reads. Offline acceptance measurement only.
	pub fn dump_features(&self, dir: &str) -> std::io::Result<()> {
		let dir = std::path::PathBuf::from(dir);
		std::fs::create_dir_all(&dir)?;
		for il in [5, 19, 33, 47, 61] {
			let _ = std::fs::remove_file(dir.join(format!("feat_{il}.bin")));
		}
		let bytes: Vec<u8> = self.history.iter().flat_map(|t| t.to_le_bytes()).collect();
		std::fs::write(dir.join("tokens.bin"), bytes)?;
		let mut rs = self.model.new_run_state();
		model::set_feature_dump(Some(dir));
		self.model.forward_prefill(&mut rs, &self.history, None);
		model::set_feature_dump(None);
		Ok(())
	}
}

/// Load the model, tokenizer and (optionally) the vision tower.
pub struct Loaded {
	pub model: Model,
	pub tok: Tokenizer,
	pub vm: Option<VisionModel>,
}

pub fn load(
	path: &str,
	mmproj: Option<&str>,
	n_gpu_layers: usize,
	ctx: usize,
	n_batch: usize,
	mtp: usize,
	dflash: Option<&str>,
) -> Result<Loaded, Box<dyn std::error::Error>> {
	let mut model = Model::load(path)?;
	model.set_ctx(ctx);
	model.set_n_batch(n_batch);
	if mtp > 0 {
		model.enable_mtp(mtp)?;
		eprintln!("  mtp: drafting {mtp} token(s) per step with the MTP head");
	}
	if let Some(p) = dflash {
		model.enable_dflash(p, model::dflash_n_draft())?;
	}
	let t_warm = std::time::Instant::now();
	model.warm();
	eprintln!(
		"  [perf] warmed {} layers in {:.1} ms",
		model.hp.n_layer_main,
		t_warm.elapsed().as_secs_f64() * 1000.0
	);
	if n_gpu_layers > 0 || model::gpu_prefill_requested() {
		model.init_offload(n_gpu_layers);
	}
	let gguf = gguf::parse_file(path)?;
	let tok = Tokenizer::from_gguf(&gguf).map_err(|e| format!("tokenizer: {e}"))?;
	let vm = match mmproj {
		Some(p) => {
			let t = std::time::Instant::now();
			let vm = VisionModel::load(p, Some(model.hp.n_embd))?;
			eprintln!(
				"  [perf] mmproj loaded in {:.1} ms ({} blocks, {}x{} px grid, {} -> {} dims)",
				t.elapsed().as_secs_f64() * 1000.0,
				vm.cfg.n_layer,
				vm.cfg.image_size,
				vm.cfg.image_size,
				vm.cfg.n_embd,
				vm.cfg.proj_dim,
			);
			Some(vm)
		}
		None => None,
	};
	Ok(Loaded { model, tok, vm })
}

/// `--image-probe`: run the tower alone on one image and report what it
/// produced. Loads no text model, so it is the fast way to check an mmproj,
/// an image format or a token budget.
pub fn image_probe(
	mmproj: &str,
	images: &[String],
	max_image_tokens: usize,
	dump: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
	if images.is_empty() {
		return Err("--image-probe needs at least one --image <file>".into());
	}
	let t = std::time::Instant::now();
	let vm = VisionModel::load(mmproj, None)?;
	println!(
		"== mmproj {mmproj}\n  loaded in {:.1} ms: {} blocks, embd {}, ff {}, heads {}, patch {}, merge {}, proj {}",
		t.elapsed().as_secs_f64() * 1000.0,
		vm.cfg.n_layer,
		vm.cfg.n_embd,
		vm.cfg.n_ff,
		vm.cfg.n_head,
		vm.cfg.patch_size,
		vm.cfg.merge_size,
		vm.cfg.proj_dim,
	);
	for path in images {
		let img = imgio::load(path)?;
		let grid = vm.grid_for(img.width, img.height, max_image_tokens)?;
		println!(
			"-- {path}: {}x{} px -> {}x{} px, {}x{} patches, {} tokens",
			img.width,
			img.height,
			grid.width,
			grid.height,
			grid.gw,
			grid.gh,
			grid.tokens()
		);
		if let Some(d) = dump {
			std::fs::write(d, img.resized(grid.width, grid.height).to_ppm())?;
			println!("   resized pixels written to {d}");
		}
		let t = std::time::Instant::now();
		let (emb, _) = vm.encode(&img, max_image_tokens)?;
		let ms = t.elapsed().as_secs_f64() * 1000.0;
		let n = emb.len() as f64;
		let mean = emb.iter().map(|&v| v as f64).sum::<f64>() / n;
		let rms = (emb.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n).sqrt();
		let mut min = f32::INFINITY;
		let mut max = f32::NEG_INFINITY;
		let mut nan = 0usize;
		for &v in &emb {
			if v.is_nan() {
				nan += 1;
				continue;
			}
			min = min.min(v);
			max = max.max(v);
		}
		println!(
			"   encoded in {ms:.1} ms ({:.1} tok/s): {} x {} embeddings  mean={mean:.5} rms={rms:.5} min={min:.4} max={max:.4} nan={nan}",
			grid.tokens() as f64 / ms * 1000.0,
			grid.tokens(),
			vm.cfg.proj_dim,
		);
		if nan > 0 {
			return Err("vision tower produced NaNs".into());
		}
	}
	Ok(())
}
