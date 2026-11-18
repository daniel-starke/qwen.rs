//! `--generate` / `--interactive` on an OpenVINO export directory
//! (`cli <dir> --generate …`): the model, its weights and its tokenizer all
//! come from the export, and the forward pass is OpenVINO's (see `crates/ov`).
//! Text only; sampling is the same `sampler` the GGUF path uses.

use std::io::Write;
use std::path::Path;

use sampler::{sample, Rng};
use tokenizer::{StreamDecoder, Tokenizer};

use crate::chat::GenOptions;

/// True when `path` is an OpenVINO export rather than a GGUF.
pub fn is_export(path: &str) -> bool {
	Path::new(path)
		.join("openvino_language_model.xml")
		.is_file()
}

struct Session {
	llm: ov::OvLlm,
	tok: Tokenizer,
	opts: GenOptions,
	rng: Rng,
	history: Vec<u32>,
	logits: Vec<f32>,
	/// Tokens that end generation: the tokenizer's EOS (`<|im_end|>`) and
	/// `<|endoftext|>` (`generation_config.json` lists both).
	stop: Vec<u32>,
}

impl Session {
	fn load(
		dir: &str,
		opts: GenOptions,
		o: &ov::Options,
	) -> Result<Session, Box<dyn std::error::Error>> {
		if opts.mtp > 0 || opts.dflash.is_some() {
			return Err("--mtp / --dflash are not available on the OpenVINO backend".into());
		}
		let dir = Path::new(dir);
		let t = std::time::Instant::now();
		let tok = Tokenizer::from_hf_json(&dir.join("tokenizer.json"))
			.map_err(|e| format!("tokenizer: {e}"))?;
		eprintln!(
			"  [perf] tokenizer.json loaded in {:.1} ms ({} tokens)",
			t.elapsed().as_secs_f64() * 1000.0,
			tok.vocab_size()
		);
		let llm = ov::OvLlm::load(dir, o)?;
		let mut stop = vec![tok.eos()];
		stop.extend(tok.token_id("<|endoftext|>"));
		Ok(Session {
			rng: Rng::new(opts.seed),
			llm,
			tok,
			opts,
			history: Vec::new(),
			logits: Vec::new(),
			stop,
		})
	}

	fn feed(&mut self, text: &str) -> Result<usize, String> {
		let ids = self.tok.encode(text);
		if ids.is_empty() {
			return Ok(0);
		}
		self.logits = self.llm.forward(&ids)?;
		self.history.extend_from_slice(&ids);
		Ok(ids.len())
	}

	fn generate(&mut self) -> Result<(String, usize), String> {
		let mut out = String::new();
		let mut produced = 0usize;
		let mut sd = StreamDecoder::new();
		let nv = self.tok.vocab_size();
		// `QWEN_OV_DUMP=FILE`: every sampled position's logits, raw f32, one
		// vocab row after another; how two devices are compared.
		let mut dump = std::env::var_os("QWEN_OV_DUMP").and_then(|f| std::fs::File::create(f).ok());
		// the penalties see this reply's tokens only
		let start = self.history.len();
		for _ in 0..self.opts.max_new {
			// The lm_head is padded past the tokenizer's vocab; those rows
			// name no token.
			let l = &self.logits[..self.logits.len().min(nv)];
			if let Some(f) = dump.as_mut() {
				let b: Vec<u8> = l.iter().flat_map(|x| x.to_le_bytes()).collect();
				f.write_all(&b).map_err(|e| format!("QWEN_OV_DUMP: {e}"))?;
			}
			let next = sample(l, &self.history[start..], &self.opts.params, &mut self.rng);
			if self.stop.contains(&next) {
				break;
			}
			self.history.push(next);
			produced += 1;
			let piece = sd.push(&self.tok, next, true);
			if !piece.is_empty() {
				print!("{piece}");
				std::io::stdout().flush().ok();
				out.push_str(&piece);
			}
			self.logits = self.llm.forward(&[next])?;
		}
		let tail = sd.flush();
		if !tail.is_empty() {
			print!("{tail}");
			std::io::stdout().flush().ok();
			out.push_str(&tail);
		}
		Ok((out, produced))
	}
}

fn perf(prefill_tok: usize, prefill_s: f64, produced: usize, decode_s: f64, device: &str) {
	eprintln!(
		"  [perf] openvino {device} prefill={prefill_tok} tok in {:.1} ms ({:.2} tok/s)  decode={produced} tok in {:.1} ms ({:.2} tok/s)",
		prefill_s * 1000.0,
		prefill_tok as f64 / prefill_s.max(1e-9),
		decode_s * 1000.0,
		produced as f64 / decode_s.max(1e-9),
	);
}

pub fn generate(
	dir: &str,
	prompt: &str,
	chat_mode: bool,
	system: Option<&str>,
	opts: GenOptions,
	o: &ov::Options,
) -> Result<(), Box<dyn std::error::Error>> {
	println!("== generate (openvino)");
	let max_new = opts.max_new;
	let think = opts.think;
	let mut s = Session::load(dir, opts, o)?;
	let mut text = String::new();
	let templated = chat_mode || system.is_some();
	if templated {
		text.push_str(&Tokenizer::system_turn(system, think));
		text.push_str(&Tokenizer::chat_turn("user", prompt));
		text.push_str(think.open());
	} else {
		text.push_str(prompt);
	}
	print!("{prompt}");
	std::io::stdout().flush().ok();
	if templated {
		crate::chat::print_open(think);
	}
	let t = std::time::Instant::now();
	let n = s.feed(&text)?;
	let prefill_s = t.elapsed().as_secs_f64();
	let t = std::time::Instant::now();
	let (_, produced) = s.generate()?;
	let decode_s = t.elapsed().as_secs_f64();
	println!();
	if produced < max_new {
		eprintln!("  [eos after {produced} tokens]");
	}
	perf(n, prefill_s, produced, decode_s, &s.llm.device);
	Ok(())
}

/// The export as a `server::Engine`: OpenVINO's logits cut to the
/// tokenizer's vocabulary (the lm_head is padded past it).
struct OvEngine {
	llm: ov::OvLlm,
	nv: usize,
}

impl server::Engine for OvEngine {
	fn reset(&mut self) -> Result<(), String> {
		self.llm.reset()
	}

	fn forward(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
		let mut l = self.llm.forward(ids)?;
		l.truncate(self.nv);
		Ok(l)
	}
}

/// `max_position_embeddings` from the export's `config.json` (under
/// `text_config` for the VL architecture).
fn trained_ctx(dir: &Path) -> Option<usize> {
	let j = server::Json::parse(&std::fs::read(dir.join("config.json")).ok()?).ok()?;
	let key = "max_position_embeddings";
	let v = j
		.get("text_config")
		.and_then(|t| t.get(key))
		.or_else(|| j.get(key))?;
	v.as_f64().map(|n| n as usize)
}

/// `cli <export-dir> --serve`: the same HTTP API as the GGUF path, on the
/// export's tokenizer and OpenVINO. The context is `o.ctx` (`--ov-ctx`, the
/// NPU's static KV capacity; on CPU / GPU the server caps at it too).
pub fn serve(
	dir: &str,
	o: &ov::Options,
	host: &str,
	port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
	let path = Path::new(dir);
	let tok = Tokenizer::from_hf_json(&path.join("tokenizer.json"))
		.map_err(|e| format!("tokenizer: {e}"))?;
	let llm = ov::OvLlm::load(path, o)?;
	let mut stop = vec![tok.eos()];
	stop.extend(tok.token_id("<|endoftext|>"));
	let info = server::EngineInfo {
		name: match std::env::var("QWEN_OV_STATIC_DEVICE") {
			Ok(d) if llm.device == "NPU" && d != "NPU" => format!("openvino NPU engine on {d}"),
			_ => format!("openvino {}", llm.device),
		},
		model: server::DEFAULT_MODEL.into(),
		ctx: o.ctx,
		ctx_train: trained_ctx(path).unwrap_or(0),
		n_vocab: tok.vocab_size(),
		n_embd: ov::npu::HIDDEN as usize,
		stop,
		tag: format!("ov:{dir}"),
		image: None,
	};
	let nv = tok.vocab_size();
	server::serve_engine(
		Box::new(OvEngine { llm, nv }),
		info,
		tok,
		host,
		port,
		&server::CacheOpts::default(),
		11.0,
	);
	Ok(())
}

pub fn interactive(
	dir: &str,
	system: Option<&str>,
	opts: GenOptions,
	o: &ov::Options,
) -> Result<(), Box<dyn std::error::Error>> {
	use std::io::BufRead;
	let think = opts.think;
	if !think.preserve {
		// this engine's state cannot go back to before a reply
		return Err("--no-preserve-thinking is not available on the OpenVINO backend".into());
	}
	let mut s = Session::load(dir, opts, o)?;
	println!(
		"== interactive chat (openvino {}; /reset starts over, /exit quits)",
		s.llm.device
	);
	let preamble = Tokenizer::system_turn(system, think);
	let stdin = std::io::stdin();
	let mut lines = stdin.lock().lines();
	let mut turn = 0usize;
	loop {
		print!("\n> ");
		std::io::stdout().flush().ok();
		let Some(Ok(line)) = lines.next() else { break };
		let line = line.trim().to_string();
		match line.as_str() {
			"/exit" | "/quit" => break,
			"/reset" => {
				s.llm.reset()?;
				s.history.clear();
				turn = 0;
				println!("  (new conversation)");
				continue;
			}
			"" => continue,
			_ => {}
		}
		let mut text = String::new();
		if turn == 0 {
			text.push_str(&preamble);
		} else {
			text.push_str("<|im_end|>\n");
		}
		text.push_str(&Tokenizer::chat_turn("user", &line));
		text.push_str(think.open());
		let t = std::time::Instant::now();
		let n = match s.feed(&text) {
			Ok(n) => n,
			Err(e) => {
				println!("  {e}");
				continue;
			}
		};
		crate::chat::print_open(think);
		let prefill_s = t.elapsed().as_secs_f64();
		let t = std::time::Instant::now();
		let (_, produced) = s.generate()?;
		let decode_s = t.elapsed().as_secs_f64();
		println!();
		perf(n, prefill_s, produced, decode_s, &s.llm.device);
		eprintln!("  [perf] turn {turn}: context now {} tokens", s.llm.pos());
		turn += 1;
	}
	println!();
	Ok(())
}
