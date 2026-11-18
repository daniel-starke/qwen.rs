//! `cli`: the qwen.rs front-end binary.
//!
//! Modes:
//!   * `qwen-rs <model.gguf>`: dump metadata + tensor index (M1)
//!   * `qwen-rs <model.gguf> --dequant <name>`: read & dequantize a real
//!     tensor, printing its shape and value statistics (M2 sanity check)
//!
//! Argument parsing is handled by the custom `getopt` module.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::io::Write;
use std::process::exit;
use std::rc::Rc;

/// Large prefill buffers are recycled instead of re-faulted per layer; see
/// `quant::alloc`.
#[global_allocator]
static ALLOC: quant::alloc::RecyclingAlloc = quant::alloc::RecyclingAlloc::new();

mod awake;
mod chat;
use quant::cputime;
mod flashchat;
mod getopt;
mod imagecmd;
mod kld;
mod mcpstdio;
mod npubuildcmd;
mod npuchat;
mod ovchat;
mod tst;

use chat::{GenOptions, Session};
use getopt::{Arg, Command, Config, Error, Parser};
use tokenizer::{Effort, Thinking, Tokenizer};

fn dump(path: &str) -> Result<(), Box<dyn std::error::Error>> {
	println!("== GGUF: {path}");
	let file = gguf::parse_file(path)?;

	println!(
		"version={} n_tensors={} n_kv={} data_offset={}",
		file.version, file.n_tensors, file.n_kv, file.data_offset
	);

	println!("\n-- metadata ({}) --", file.metadata.len());
	for e in &file.metadata {
		println!("  {} = {}", e.key, e.value.display());
	}

	// Surface arch-relevant keys to make the gate decision obvious.
	let interesting = [
		"general.architecture",
		"general.name",
		"qwen35.block_count",
		"qwen35.context_length",
		"qwen35.embedding_length",
		"qwen35.attention.head_count",
		"qwen35.attention.head_count_kv",
		"qwen35.expert_count",
		"qwen35.expert_used_count",
		"qwen35.rope.dimension_count",
		"qwen35.rope.freq_scale",
		"qwen35.ssm.conv_kernel",
		"qwen35.ssm.state_size",
		"qwen35.vocab_size",
	];
	println!("\n-- arch highlights --");
	for key in interesting {
		if let Some(e) = file.metadata.iter().find(|e| e.key == key) {
			println!("  {} = {}", e.key, e.value.display());
		}
	}

	println!("\n-- tensors ({}) --", file.tensors.len());
	for t in &file.tensors {
		let shape = t
			.dims
			.iter()
			.map(|d| d.to_string())
			.collect::<Vec<_>>()
			.join("x");
		println!(
			"  {:>8}  {:<48} [{}]  off={}",
			t.ggml_type.name(),
			t.name,
			shape,
			t.offset
		);
	}

	Ok(())
}

fn dequant_tensor(path: &str, name: &str) -> Result<(), Box<dyn std::error::Error>> {
	println!("== dequant: {name}");
	let file = gguf::parse_file(path)?;
	let t = file
		.tensors
		.iter()
		.find(|t| t.name == name)
		.ok_or_else(|| {
			format!(
				"tensor '{name}' not found ({} tensors total)",
				file.tensors.len()
			)
		})?;

	println!(
		"  type={} shape=[{}] elements={} bytes={}",
		t.ggml_type.name(),
		t.dims
			.iter()
			.map(|d| d.to_string())
			.collect::<Vec<_>>()
			.join("x"),
		t.num_elements(),
		t.byte_length()
	);

	let raw = file.read_tensor_bytes(path, t)?;
	let deq = quant::dequantize(t.ggml_type, &raw, t.num_elements());

	let mut min = f32::INFINITY;
	let mut max = f32::NEG_INFINITY;
	let mut sum = 0.0f64;
	let mut abs_sum = 0.0f64;
	let mut nan = 0usize;
	for &v in &deq {
		if v.is_nan() {
			nan += 1;
			continue;
		}
		if v < min {
			min = v;
		}
		if v > max {
			max = v;
		}
		sum += v as f64;
		abs_sum += v.abs() as f64;
	}
	let n = deq.len() as f64;
	println!(
		"  dequantized elements={}  min={min:.6} max={max:.6} mean={:.6} absmean={:.6} nan={nan}",
		deq.len(),
		sum / n,
		abs_sum / n,
	);
	// A few sample values for eyeballing.
	let samples: Vec<String> = deq.iter().take(8).map(|v| format!("{v:.6}")).collect();
	println!("  samples[0..8] = [{}]", samples.join(", "));
	Ok(())
}

fn forward(
	path: &str,
	tokens: &[u32],
	max_layers: Option<usize>,
	n_gpu_layers: usize,
	ctx: usize,
	n_batch: usize,
) -> Result<(), Box<dyn std::error::Error>> {
	println!("== forward: {} tokens", tokens.len());
	let mut model = model::Model::load(path)?;
	model.set_ctx(ctx);
	model.set_n_batch(n_batch);
	model.warm();
	if n_gpu_layers > 0 || model::gpu_prefill_requested() {
		model.init_offload(n_gpu_layers);
	}
	println!(
		"  n_embd={} n_head={} head_dim={} n_vocab={} layers={} (full-attn every {}; total full={})",
		model.hp.n_embd,
		model.hp.n_head,
		model.hp.head_dim,
		model.hp.n_vocab,
		model.hp.n_layer_main,
		model.hp.full_attn_interval,
		model.full_attn_layers.iter().filter(|&&b| b).count(),
	);

	let (logits, logits_dc) = model.forward(tokens, max_layers);
	// dump full logits to binary for comparison
	{
		let dump_path =
			std::env::var("QWEN_DUMP").unwrap_or_else(|_| "/tmp/kilo/qwen_logits.bin".to_string());
		if let Ok(mut f) = std::fs::File::create(&dump_path) {
			let _ = f.write_all(unsafe {
				std::slice::from_raw_parts(logits.as_ptr() as *const u8, logits.len() * 4)
			});
		}
	}
	let n = logits.len();
	let sum: f64 = logits.iter().map(|&v| v as f64).sum();
	println!("SUM {:.6}", sum);
	{
		let sum: f64 = logits_dc.iter().map(|&v| v as f64).sum();
		let mut idx: Vec<usize> = (0..n).collect();
		idx.sort_by(|&a, &b| logits_dc[b].partial_cmp(&logits_dc[a]).unwrap());
		let am = idx[0];
		println!(
			"DCR SUM {:.6} ARGMAX {} {:.6} TOP0 {} TOP1 {}",
			sum, am, logits_dc[am], logits_dc[idx[0]], logits_dc[idx[1]]
		);
	}
	let l = |i: usize| if i < n { logits[i] } else { 0.0 };
	println!(
		"L0 {:.6} L1 {:.6} L10 {:.6} L100 {:.6} L1000 {:.6}",
		l(0),
		l(1),
		l(10),
		l(100),
		l(1000)
	);
	let mut idx: Vec<usize> = (0..n).collect();
	idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
	let am = idx[0];
	println!("ARGMAX {} {:.6}", am, logits[am]);
	println!("-- top 20 logits --");
	for &i in idx.iter().take(20) {
		println!("TOP {} {:.6}", i, logits[i]);
	}
	Ok(())
}

fn tokmeta(path: &str) -> Result<(), Box<dyn std::error::Error>> {
	println!("== tokenizer metadata: {path}");
	let file = gguf::parse_file(path)?;
	for e in &file.metadata {
		if !e.key.starts_with("tokenizer.") && !e.key.starts_with("general.") {
			continue;
		}
		match &e.value {
			gguf::MetadataValue::Array { ty, values } => {
				// Do not dump huge arrays in full.
				if e.key == "tokenizer.ggml.tokens" || e.key == "tokenizer.ggml.merges" {
					let head: Vec<String> = values.iter().take(6).map(|v| v.display()).collect();
					println!(
						"  {} [{ty:?} x {}] head=[{}]",
						e.key,
						values.len(),
						head.join(", ")
					);
				} else if values.len() <= 24 {
					println!(
						"  {} [{ty:?} x {}] = {}",
						e.key,
						values.len(),
						e.value.display()
					);
				} else {
					let head: Vec<String> = values.iter().take(5).map(|v| v.display()).collect();
					let tail: Vec<String> = values
						.iter()
						.rev()
						.take(5)
						.rev()
						.map(|v| v.display())
						.collect();
					println!(
						"  {} [{ty:?} x {}] head=[{}] ... tail=[{}]",
						e.key,
						values.len(),
						head.join(", "),
						tail.join(", ")
					);
				}
			}
			_ => println!("  {} = {}", e.key, e.value.display()),
		}
	}
	Ok(())
}

/// Autoregressive text generation (M6/M9 integration).
///
/// `chat` renders the prompt through the Qwen chat template. Images force it
/// on regardless: the template's `<|vision_start|>` framing is how the model
/// was trained to see them, and a bare completion prompt with patch embeddings
/// spliced in produces confident nonsense.
#[allow(clippy::too_many_arguments)]
fn generate(
	path: &str,
	prompt: &str,
	chat_mode: bool,
	system: Option<&str>,
	images: &[String],
	mmproj: Option<&str>,
	opts: GenOptions,
	n_gpu_layers: usize,
	ctx: usize,
	n_batch: usize,
) -> Result<(), Box<dyn std::error::Error>> {
	println!("== generate");
	let l = chat::load(
		path,
		mmproj,
		n_gpu_layers,
		ctx,
		n_batch,
		opts.mtp,
		opts.dflash.as_deref(),
	)?;
	let max_new = opts.max_new;
	let think = opts.think;
	let mut sess = Session::new(&l.model, &l.tok, l.vm.as_ref(), opts);

	// Templated when asked for, and whenever there is an image.
	let templated = chat_mode || !images.is_empty() || system.is_some();
	let mut text = String::new();
	let mut imgs = Vec::new();
	if templated {
		text.push_str(&Tokenizer::system_turn(system, think));
		let mut content = String::new();
		for f in images {
			let img = imgio::load(f)?;
			let (placeholder, enc) = sess.encode_image(&img)?;
			content.push_str(&placeholder);
			imgs.push(enc);
		}
		content.push_str(prompt);
		text.push_str(&Tokenizer::chat_turn("user", &content));
		text.push_str(think.open());
	} else {
		text.push_str(prompt);
	}

	print!("{prompt}");
	std::io::stdout().flush().ok();
	if templated {
		chat::print_open(think);
	}

	let m_prefill = cputime::Meter::start();
	let _ = (quant::parallel::stats(), quant::parallel::stats_sites());
	let prefill_tok = sess.feed(&text, imgs)?;
	let (prefill_s, prefill_cpu) = m_prefill.read();
	let pstats = quant::parallel::stats();
	let psites = quant::parallel::stats_sites();
	let prefill_ms = prefill_s * 1000.0;

	let m_decode = cputime::Meter::start();
	let (_, produced) = sess.generate();
	let (decode_s, decode_cpu) = m_decode.read();
	let decode_ms = decode_s * 1000.0;
	println!();

	report(
		n_gpu_layers,
		prefill_tok,
		prefill_ms,
		produced,
		decode_ms,
		max_new,
	);
	eprintln!(
		"  [eff] prefill {}  decode {}",
		cputime::describe(prefill_s, prefill_cpu, prefill_tok),
		cputime::describe(decode_s, decode_cpu, produced)
	);
	if pstats.3 > 0 {
		eprintln!(
			"  [pool] prefill work {:.2} s, spin {:.2} s, join {:.2} s, {} dispatches, {} parks",
			pstats.0, pstats.1, pstats.2, pstats.3, pstats.4
		);
		for (site, work, wall, n) in psites.iter().take(25) {
			eprintln!(
				"  [pool]   {work:8.2} s shard, {wall:7.2} s wall, {:5.1} x, {n:6} calls  {site}",
				work / wall.max(1e-9)
			);
		}
	}
	if sess.mtp.steps > 0 {
		eprintln!("  [perf] {}", sess.mtp.summary());
	}
	if let Ok(dir) = std::env::var("QWEN_FEAT_DUMP") {
		sess.dump_features(&dir)?;
		eprintln!("  [feat] dumped {} positions to {dir}", sess.pos());
	}
	Ok(())
}

/// Multi-turn interactive chat (the M6 leftover).
///
/// One `RunState` lives for the whole session, so turn *k* only prefills the
/// tokens turn *k* adds: an hour-long conversation never re-reads the 16.8 GB
/// of weights for text it has already seen.
#[allow(clippy::too_many_arguments)]
fn interactive(
	path: &str,
	system: Option<&str>,
	initial_images: &[String],
	mmproj: Option<&str>,
	opts: GenOptions,
	n_gpu_layers: usize,
	ctx: usize,
	n_batch: usize,
) -> Result<(), Box<dyn std::error::Error>> {
	use std::io::BufRead;

	let l = chat::load(
		path,
		mmproj,
		n_gpu_layers,
		ctx,
		n_batch,
		opts.mtp,
		opts.dflash.as_deref(),
	)?;
	let think = opts.think;
	let mut sess = Session::new(&l.model, &l.tok, l.vm.as_ref(), opts);

	println!("== interactive chat  (/image <file> attaches an image, /exit quits)");
	let mut pending: Vec<String> = initial_images.to_vec();
	let preamble = Tokenizer::system_turn(system, think);
	// `--no-preserve-thinking`: where the last reply began (just past its
	// assistant header) and its answer. The next turn rewinds there and
	// feeds the answer alone, so earlier turns carry no reasoning, as the
	// template renders them.
	let mut last: Option<((model::Checkpoint, usize), String)> = None;
	let (head, open_rest) = chat::open_parts(think);

	let stdin = std::io::stdin();
	let mut lines = stdin.lock().lines();
	let mut turn = 0usize;
	loop {
		print!("\n> ");
		std::io::stdout().flush().ok();
		let line = match lines.next() {
			Some(Ok(l)) => l,
			_ => break,
		};
		let line = line.trim().to_string();
		if line == "/exit" || line == "/quit" {
			break;
		}
		if let Some(f) = line.strip_prefix("/image ") {
			let f = f.trim().to_string();
			match imgio::load(&f) {
				Ok(_) => {
					println!("  attached {f} (sent with your next message)");
					pending.push(f);
				}
				Err(e) => println!("  {e}"),
			}
			continue;
		}
		if line.is_empty() {
			continue;
		}

		let mut content = String::new();
		let mut imgs = Vec::new();
		for f in pending.drain(..) {
			let img = imgio::load(&f)?;
			let (placeholder, enc) = sess.encode_image(&img)?;
			content.push_str(&placeholder);
			imgs.push(enc);
		}
		content.push_str(&line);

		// Close the previous assistant turn before opening the new user one.
		// Generation stops *at* EOS without feeding it, so the `<|im_end|>`
		// here is what the model sees either way.
		let mut text = String::new();
		if turn == 0 {
			text.push_str(&preamble);
		} else {
			if let Some((cp, answer)) = last.take() {
				if sess.rewind(cp) {
					text.push_str(&answer);
				}
			}
			text.push_str("<|im_end|>\n");
		}
		text.push_str(&Tokenizer::chat_turn("user", &content));
		text.push_str(head);
		if think.preserve {
			text.push_str(open_rest);
		}

		let t_prefill = std::time::Instant::now();
		let mut prefill_tok = sess.feed(&text, imgs)?;
		let cp = (!think.preserve).then(|| sess.checkpoint());
		if cp.is_some() {
			prefill_tok += sess.feed(open_rest, Vec::new())?;
		}
		let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
		chat::print_open(think);
		let t_decode = std::time::Instant::now();
		let (reply, produced) = sess.generate();
		if let Some(cp) = cp {
			last = Some((cp, chat::answer_of(&reply, think).to_string()));
		}
		let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
		println!();
		eprintln!(
			"  [perf] turn {turn}: prefill {prefill_tok} tok in {prefill_ms:.1} ms, decode {produced} tok in {decode_ms:.1} ms, context now {} tokens",
			sess.pos()
		);
		if sess.mtp.steps > 0 {
			eprintln!("  [perf] {}", sess.mtp.summary());
		}
		turn += 1;
	}
	println!();
	Ok(())
}

/// The throughput line the `--n-gpu-layers` sweep keys on.
fn report(
	n_gpu_layers: usize,
	prefill_tok: usize,
	prefill_ms: f64,
	produced: usize,
	decode_ms: f64,
	max_new: usize,
) {
	let pps = if prefill_ms > 0.0 {
		prefill_tok as f64 / prefill_ms * 1000.0
	} else {
		0.0
	};
	let dpps = if decode_ms > 0.0 {
		produced as f64 / decode_ms * 1000.0
	} else {
		0.0
	};
	if produced < max_new {
		eprintln!("  [eos after {produced} tokens]");
	}
	eprintln!(
		"  [perf] n_gpu_layers={} prefill={} tok in {:.1} ms ({:.2} tok/s)  decode={} tok in {:.1} ms ({:.2} tok/s)",
		n_gpu_layers, prefill_tok, prefill_ms, pps, produced, decode_ms, dpps,
	);
}

fn embed_check(path: &str, toks: &[u32]) -> Result<(), Box<dyn std::error::Error>> {
	println!("== embed check");
	let model = model::Model::load(path)?;
	for &tok in toks {
		let e = model.embedding(tok);
		let mut min = f32::INFINITY;
		let mut max = f32::NEG_INFINITY;
		let mut ss = 0.0f64;
		for &v in &e {
			if v < min {
				min = v;
			}
			if v > max {
				max = v;
			}
			ss += (v as f64) * (v as f64);
		}
		let nrm = ss.sqrt();
		let samples: Vec<String> = e.iter().take(8).map(|v| format!("{v:.6}")).collect();
		println!(
			"  tok {tok}: norm={nrm:.5} min={min:.5} max={max:.5} samples[0..8]=[{}]",
			samples.join(", ")
		);
	}
	Ok(())
}

/// Print what the tokenizer does to a string, id by id.
///
/// Added while chasing a prompt that made the model emit EOS immediately: the
/// text was demonstrably right, so the question was whether the *ids* were, and
/// there was no way to look. A literal backslash-n in the argument is expanded,
/// so a multi-line chat frame can be passed as one shell word.
fn tokenize_dump(path: &str, text: &str) -> Result<(), Box<dyn std::error::Error>> {
	// `@FILE` reads the text from a file, verbatim, as `--generate` does.
	let text = match text.strip_prefix('@') {
		Some(f) => std::fs::read_to_string(f)?,
		None => text.replace("\\n", "\n"),
	};
	let gguf = gguf::parse_file(path)?;
	let tk = Tokenizer::from_gguf(&gguf).map_err(|e| format!("tokenizer: {e}"))?;
	let ids = tk.encode(&text);
	println!("{} tokens", ids.len());
	for id in &ids {
		println!("  {:>6}  {:?}", id, tk.decode(&[*id], false));
	}
	Ok(())
}

/// Measure the prefill CPU/GPU row split on this host and recommend its
/// fractions (`model::tune`). `seqs` is a comma list of chunk lengths; the
/// longest one sets the base fractions.
fn tune_split(
	path: &str,
	seqs: &str,
	rounds: usize,
	n_gpu_layers: usize,
) -> Result<(), Box<dyn std::error::Error>> {
	let seqs: Vec<usize> = seqs
		.split(',')
		.filter(|s| !s.trim().is_empty())
		.map(|s| s.trim().parse::<usize>())
		.collect::<Result<_, _>>()
		.map_err(|e| format!("--tune-split {seqs}: {e}"))?;
	let mut model = model::Model::load(path)?;
	model.init_offload(n_gpu_layers);
	model.tune_split(&seqs, rounds)?;
	Ok(())
}

/// Modes selected by the command-line flags.
enum Mode {
	Dump,
	Dequant,
	Forward,
	TokMeta,
	Generate,
	Interactive,
	ImageProbe,
	EmbedTok,
	Tokenize,
	Serve,
	KldEval,
	TuneSplit,
	TuneProfile,
}

/// Argument source that lets us re-inject a token that was consumed as an
/// optional value but is actually a following flag (e.g. `--generate --temp`).
struct InjectIter {
	queue: Rc<RefCell<VecDeque<String>>>,
	inner: std::vec::IntoIter<String>,
}

impl Iterator for InjectIter {
	type Item = String;

	fn next(&mut self) -> Option<String> {
		if let Some(s) = self.queue.borrow_mut().pop_front() {
			return Some(s);
		}
		self.inner.next()
	}
}

fn main() {
	quant::ltime::start();
	let args: Vec<String> = std::env::args().collect();
	if args.len() < 2 {
		eprintln!(
			"usage: {0} --version
       {0} <model.gguf> [--dequant <tensor> | --forward <t0,t1,...> [--layers N] | --generate [<prompt>] | --interactive | --tokmeta | --embedtok <ids> | --tokenize <text> | --serve [PORT]]\n\
             \n\
             generation : [--chat] [--system <text>] [--think off|low|medium|xhigh] [--no-preserve-thinking] [--progress] [--no-keep-awake] [--max-new N --temp T --top-k K --top-p P --min-p P --presence-penalty X --rep R --seed S --ctx N --batch N] [--mtp [N] | --dflash <dir>]\n\
             profile    : [--profile FILE]  (sampling and decode knobs per thinking mode and depth; else QWEN_PROFILE, else <model>.profile beside the model, else built in)\n\
             vision     : [--mmproj <mmproj.gguf>] [--image <file>]... [--image-max-tokens N] [--image-dump <file.ppm>]\n\
             diagnostics: {0} <any.gguf> --image-probe --mmproj <mmproj.gguf> --image <file>\n\
             tuning     : {0} <model.gguf> --tune-split [SEQ,SEQ,...] [--tune-rounds N]  (measure and recommend the prefill GPU split fractions)\n\
             profiles   : {0} <model> --tune-profile OUT [--ctx N] [--mtp [N] | --dflash <dir>] [--tune-depths D,D,..] [--tune-tokens N] [--tune-rounds N]\n             \
             [--tune-modes think,nothink] [--tune-corpus DIR] [--tune-quick]  (measure this machine's decode knobs per mode and depth, write a profile)\n\
             server     : --serve [PORT] [--host H --ctx N --batch N] [--mmproj <mmproj.gguf>] [--mtp [N] | --dflash <dir>] [--client-ignore PREFIX=FIELD[,FIELD...]|off]...  (drop request fields of clients whose User-Agent starts with PREFIX, * = all; default *=temperature) [--record-requests DIR]  (save each POST, for `examples/replay.rs`)\n\
             openvino   : {0} <openvino-export-dir> --generate|--interactive|--serve [PORT] [--host H] [--ov-device NPU|CPU|GPU] [--ov-lib DIR] [--ov-cache DIR]\n             \
             [--ov-ctx N] [--ov-chunk N] [--ov-prop KEY=VALUE]...\n\
             npu        : {0} <model.npu> --generate|--interactive|--serve [PORT] [--host H] [--prompt-cache-mb N] [--prompt-cache-dir DIR [--prompt-cache-disk-mb N] [--prompt-cache-spill]]\n             \
             [--image <png|jpg|bmp|ppm>]... [--image-max-tokens N]  (with `vision = DIR` in the .npu file)\n\
             image      : {0} <qwen-image-dir|image.npu> --txt2img PROMPT -o FILE.png [--image FILE]... [--resolution N] [--size WxH] [--seed N] [--steps N] [--noise FILE.f32] [--progress]
             image server: any --serve above, or {0} <qwen-image-dir|image.npu> --serve [PORT] (images only), with [--image-model DIR|image.npu] [--image-out-dir DIR]... [--image-idle SECS]  (MCP at /mcp, /v1/images/*)
             mcp stdio  : {0} --mcp-stdio [URL]  (stdio <-> URL/mcp bridge for clients that only start commands; default http://127.0.0.1:8080)
             npu build  : {0} <bf16-checkpoint-dir> --npu-build OUT [--npu-calib FILE]... [--npu-calib-chunks N] [--npu-steps a,b,..] [--npu-compiler DIR] [--npu-jobs N] [--npu-no-gpu]",
			args.first().map(|s| s.as_str()).unwrap_or("qwen-rs")
		);
		exit(2);
	}
	if args[1] == "--version" || args[1] == "-V" {
		println!("qwen.rs {}", env!("CARGO_PKG_VERSION"));
		return;
	}
	if args[1] == "--mcp-stdio" {
		exit(mcpstdio::main(
			args.get(2).map_or("http://127.0.0.1:8080", |s| s.as_str()),
		));
	}
	let path = args[1].clone();
	if args.iter().any(|a| a == "--npu-build") {
		exit(npubuildcmd::main(&path, &args[2..]));
	}
	if args.iter().any(|a| a == "--txt2img") {
		exit(imagecmd::main(&path, &args[2..]));
	}

	let opts = vec![
		Config::new_long("dump", Arg::None),
		Config::new_long("dequant", Arg::Required),
		Config::new_long("forward", Arg::Required),
		Config::new_long("tokmeta", Arg::None),
		Config::new_long("generate", Arg::Optional),
		Config::new_long("embedtok", Arg::Required),
		Config::new_long("tokenize", Arg::Required),
		Config::new_long("serve", Arg::Optional),
		Config::new_long("host", Arg::Required),
		Config::new_long("chat", Arg::None),
		Config::new_long("progress", Arg::None),
		Config::new_long("no-keep-awake", Arg::None),
		Config::new_long("interactive", Arg::None),
		Config::new_long("system", Arg::Required),
		Config::new_long("think", Arg::Required),
		Config::new_long("no-preserve-thinking", Arg::None),
		Config::new_long("client-ignore", Arg::Required),
		Config::new_long("record-requests", Arg::Required),
		Config::new_long("mmproj", Arg::Required),
		Config::new_long("image", Arg::Required),
		Config::new_long("image-max-tokens", Arg::Required),
		Config::new_long("image-dump", Arg::Required),
		Config::new_long("image-probe", Arg::None),
		Config::new_long("layers", Arg::Required),
		Config::new_long("n-gpu-layers", Arg::Required),
		Config::new_long("max-new", Arg::Required),
		Config::new_long("temp", Arg::Required),
		Config::new_long("top-k", Arg::Required),
		Config::new_long("top-p", Arg::Required),
		Config::new_long("rep", Arg::Required),
		Config::new_long("seed", Arg::Required),
		Config::new_long("ctx", Arg::Required),
		Config::new_long("batch", Arg::Required),
		Config::new_long("prompt-cache-mb", Arg::Required),
		Config::new_long("prompt-cache-dir", Arg::Required),
		Config::new_long("prompt-cache-disk-mb", Arg::Required),
		Config::new_long("prompt-cache-spill", Arg::None),
		Config::new_long("mtp", Arg::Optional),
		Config::new_long("dflash", Arg::Required),
		Config::new_long("kld-eval", Arg::Required),
		Config::new_long("kld-tokens", Arg::Required),
		Config::new_long("sparse-sweep", Arg::Required),
		Config::new_long("tune-split", Arg::Optional),
		Config::new_long("tune-rounds", Arg::Required),
		Config::new_long("tune-profile", Arg::Required),
		Config::new_long("tune-depths", Arg::Required),
		Config::new_long("tune-tokens", Arg::Required),
		Config::new_long("tune-modes", Arg::Required),
		Config::new_long("tune-corpus", Arg::Required),
		Config::new_long("tune-quick", Arg::None),
		Config::new_long("profile", Arg::Required),
		Config::new_long("min-p", Arg::Required),
		Config::new_long("presence-penalty", Arg::Required),
		Config::new_long("ov-device", Arg::Required),
		Config::new_long("ov-lib", Arg::Required),
		Config::new_long("ov-cache", Arg::Required),
		Config::new_long("ov-ctx", Arg::Required),
		Config::new_long("ov-chunk", Arg::Required),
		Config::new_long("ov-prop", Arg::Required),
		Config::new_long("image-model", Arg::Required),
		Config::new_long("image-out-dir", Arg::Required),
		Config::new_long("image-idle", Arg::Required),
	];

	let queue: Rc<RefCell<VecDeque<String>>> = Rc::new(RefCell::new(VecDeque::new()));
	let inner: Vec<String> = args.iter().skip(2).cloned().collect();
	let iter = InjectIter {
		queue: queue.clone(),
		inner: inner.into_iter(),
	};
	let mut parser = Parser::new(opts, iter);

	let mut mode = Mode::Dump;
	let mut dequant_name = String::new();
	let mut forward_tokens = String::new();
	let mut gen_prompt = String::new();
	let mut embed_ids = String::new();
	let mut tokenize_text = String::new();
	let mut max_layers: Option<usize> = None;
	let mut n_gpu_layers: usize = 0;
	let mut host = String::from("127.0.0.1");
	let mut port = 8080u16;
	let mut chat_mode = false;
	let mut keep_awake = awake::from_env();
	let mut system: Option<String> = None;
	// `--think LEVEL` / `--no-preserve-thinking`: the chat template's
	// `reasoning_effort` (`enable_thinking: false` for `off`) and
	// `preserve_thinking`.
	let mut think: Option<Thinking> = None;
	let mut preserve_thinking = true;
	// `--client-ignore`: `None` keeps the server's defaults.
	let mut client_ignore: Option<Vec<server::ClientIgnore>> = None;
	let mut mmproj: Option<String> = None;
	let mut image_model: Option<String> = None;
	let mut image_out: Vec<std::path::PathBuf> = Vec::new();
	let mut image_idle = 300u64;
	let mut images: Vec<String> = Vec::new();
	let mut image_max_tokens = vision::DEFAULT_MAX_TOKENS;
	let mut image_dump: Option<String> = None;
	let mut max_new = 16usize;
	// Sampling: the profile's for the thinking mode unless given (the
	// built-in profile is the model cards' sets; `server::profile`).
	let mut temperature: Option<f32> = None;
	let mut top_k: Option<usize> = None;
	let mut top_p: Option<f32> = None;
	let mut min_p: Option<f32> = None;
	let mut presence: Option<f32> = None;
	let mut rep: Option<f32> = None;
	let mut seed = 1234u64;
	let mut profile_path: Option<String> = None;
	// Context window / prefill chunking. `ctx == 0` means "model native
	// context_length"; `n_batch` defaults to llama.cpp's 2048.
	let mut ctx: usize = 0;
	let mut n_batch: usize = 2048;
	// Server prefix cache budget (MiB of run state kept across requests).
	let mut prompt_cache_mb: usize = 32 * 1024;
	// Where the prefix cache persists its shared entries (off unless given).
	let mut prompt_cache_dir: Option<String> = None;
	let mut prompt_cache_disk_mb: Option<usize> = None;
	let mut prompt_cache_spill = false;
	// Speculative decoding with the GGUF's MTP head: drafts per step (0 = off;
	// a bare `--mtp` is 3, cut short by `QWEN_MTP_PMIN`).
	let mut mtp: usize = 0;
	// a bare `--mtp`: the profile's largest `QWEN_MTP_N`, else 3
	let mut mtp_bare = false;
	// DFlash2 block drafter (z-lab/Qwen3.8-27B-DFlash2 directory).
	let mut dflash: Option<String> = None;
	// --kld-eval: text file, positions, and the variants to compare.
	let mut kld_file = String::new();
	let mut kld_tokens: usize = 256;
	let mut sparse_sweep = String::new();
	// --tune-split: chunk lengths to measure and passes per point.
	let mut tune_seqs = String::from("2048,160,100");
	let mut tune_rounds: Option<usize> = None;
	// --tune-profile: where to write it, and what to measure.
	let mut tune_out: Option<String> = None;
	let mut tune_depths: Vec<usize> = Vec::new();
	let mut tune_tokens: usize = 96;
	let mut tune_modes: Vec<bool> = vec![false, true];
	let mut tune_corpus: Option<String> = None;
	let mut tune_quick = false;
	// OpenVINO backend (when the model path is an OpenVINO export directory).
	let mut ov_opts = ov::Options::default();
	// Whether `--ov-ctx` was given (else `--ctx` sets the export's context).
	let mut ov_ctx_set = false;

	loop {
		match parser.next_opt() {
			Ok(Command::Rest(_)) => break,
			Ok(Command::Opt(cfg, val)) => match cfg.long.as_str() {
				"dump" => mode = Mode::Dump,
				"dequant" => {
					mode = Mode::Dequant;
					dequant_name = val.unwrap_or_default();
				}
				"forward" => {
					mode = Mode::Forward;
					forward_tokens = val.unwrap_or_default();
				}
				"tokmeta" => mode = Mode::TokMeta,
				"generate" => {
					mode = Mode::Generate;
					match val {
						Some(v) if v.starts_with('-') => {
							// A flag was consumed as the optional prompt; re-inject it.
							gen_prompt.clear();
							queue.borrow_mut().push_back(v);
						}
						// `@FILE` reads the prompt from a file: a long prompt
						// does not fit a Windows command line (8191 chars).
						Some(v) if v.starts_with('@') => {
							gen_prompt = std::fs::read_to_string(&v[1..]).unwrap_or_else(|e| {
								eprintln!("--generate {v}: {e}");
								exit(2)
							})
						}
						Some(v) => gen_prompt = v,
						None => gen_prompt.clear(),
					}
				}
				"embedtok" => {
					mode = Mode::EmbedTok;
					embed_ids = val.unwrap_or_default();
				}
				"tokenize" => {
					mode = Mode::Tokenize;
					tokenize_text = val.unwrap_or_default();
				}
				"chat" => chat_mode = true,
				"progress" => quant::progress::set_enabled(true),
				"no-keep-awake" => keep_awake = false,
				"interactive" => {
					mode = Mode::Interactive;
					chat_mode = true;
				}
				"system" => system = Some(val.expect("system")),
				"think" => {
					let v = val.expect("think");
					match Thinking::parse(&v) {
						Some(t) => think = Some(t),
						None => {
							eprintln!("--think wants off, low, medium or xhigh, got {v}");
							exit(2)
						}
					}
					// a thinking level is a chat-template switch
					chat_mode = true;
				}
				"no-preserve-thinking" => preserve_thinking = false,
				// The first one replaces the defaults; `off` clears the list.
				"client-ignore" => {
					let v = val.expect("client-ignore");
					let rules = client_ignore.get_or_insert_with(Vec::new);
					if v == "off" {
						rules.clear();
					} else {
						match server::ClientIgnore::parse(&v) {
							Ok(r) => rules.push(r),
							Err(e) => {
								eprintln!("{e}");
								exit(2);
							}
						}
					}
				}
				"record-requests" => {
					let d = std::path::PathBuf::from(val.expect("record-requests"));
					if let Err(e) = server::set_record_requests(d.clone()) {
						eprintln!("error: --record-requests {}: {e}", d.display());
						exit(2);
					}
				}
				"mmproj" => mmproj = Some(val.expect("mmproj")),
				"image-model" => image_model = Some(val.expect("image-model")),
				"image-out-dir" => {
					let d = std::path::PathBuf::from(val.expect("image-out-dir"));
					let d = std::fs::canonicalize(&d).unwrap_or_else(|e| {
						eprintln!("error: --image-out-dir {}: {e}", d.display());
						exit(2)
					});
					// canonicalize gives \\?\C:\..., which clients never send
					let s = d.to_string_lossy().to_string();
					image_out.push(s.strip_prefix(r"\\?\").unwrap_or(&s).into());
				}
				"image-idle" => image_idle = val.expect("image-idle").parse().expect("image-idle"),
				"image" => images.push(val.expect("image")),
				"image-max-tokens" => {
					image_max_tokens = val
						.expect("image-max-tokens")
						.parse()
						.expect("image-max-tokens")
				}
				"image-dump" => image_dump = Some(val.expect("image-dump")),
				"image-probe" => mode = Mode::ImageProbe,
				"host" => host = val.expect("host").to_string(),
				"serve" => {
					mode = Mode::Serve;
					match val {
						Some(v) if v.starts_with('-') => {
							// A flag was consumed as the optional port; re-inject it.
							queue.borrow_mut().push_back(v);
						}
						Some(v) => port = v.parse().expect("port"),
						None => {}
					}
				}
				"layers" => max_layers = Some(val.expect("layers").parse().expect("layers")),
				"n-gpu-layers" => {
					n_gpu_layers = val.expect("n-gpu-layers").parse().expect("n-gpu-layers")
				}
				"max-new" => max_new = val.expect("max-new").parse().expect("max-new"),
				"temp" => temperature = Some(val.expect("temp").parse().expect("temp")),
				"top-k" => top_k = Some(val.expect("top-k").parse().expect("top-k")),
				"top-p" => top_p = Some(val.expect("top-p").parse().expect("top-p")),
				"min-p" => min_p = Some(val.expect("min-p").parse().expect("min-p")),
				"presence-penalty" => {
					presence = Some(
						val.expect("presence-penalty")
							.parse()
							.expect("presence-penalty"),
					)
				}
				"rep" => rep = Some(val.expect("rep").parse().expect("rep")),
				"profile" => profile_path = Some(val.expect("profile")),
				"seed" => seed = val.expect("seed").parse().expect("seed"),
				"ctx" => ctx = val.expect("ctx").parse().expect("ctx"),
				"batch" => n_batch = val.expect("batch").parse().expect("batch"),
				"prompt-cache-mb" => {
					prompt_cache_mb = val
						.expect("prompt-cache-mb")
						.parse()
						.expect("prompt-cache-mb")
				}
				"prompt-cache-dir" => prompt_cache_dir = Some(val.expect("prompt-cache-dir")),
				"prompt-cache-disk-mb" => {
					prompt_cache_disk_mb = Some(
						val.expect("prompt-cache-disk-mb")
							.parse()
							.expect("prompt-cache-disk-mb"),
					)
				}
				"prompt-cache-spill" => prompt_cache_spill = true,
				"mtp" => match val {
					Some(v) if v.starts_with('-') => {
						// A flag was consumed as the optional count; re-inject it.
						mtp_bare = true;
						queue.borrow_mut().push_back(v);
					}
					Some(v) => mtp = v.parse().expect("mtp"),
					None => mtp_bare = true,
				},
				"dflash" => dflash = Some(val.expect("dflash")),
				"kld-eval" => {
					mode = Mode::KldEval;
					kld_file = val.expect("kld-eval");
				}
				"kld-tokens" => kld_tokens = val.expect("kld-tokens").parse().expect("kld-tokens"),
				"sparse-sweep" => sparse_sweep = val.expect("sparse-sweep"),
				"tune-split" => {
					mode = Mode::TuneSplit;
					match val {
						Some(v) if v.starts_with('-') => queue.borrow_mut().push_back(v),
						Some(v) => tune_seqs = v,
						None => {}
					}
				}
				"tune-rounds" => {
					tune_rounds = Some(val.expect("tune-rounds").parse().expect("tune-rounds"))
				}
				"tune-profile" => {
					mode = Mode::TuneProfile;
					tune_out = Some(val.expect("tune-profile"));
				}
				"tune-depths" => {
					tune_depths = val
						.expect("tune-depths")
						.split(',')
						.filter(|s| !s.trim().is_empty())
						.map(|s| s.trim().parse().expect("tune-depths"))
						.collect()
				}
				"tune-tokens" => {
					tune_tokens = val.expect("tune-tokens").parse().expect("tune-tokens")
				}
				"tune-modes" => {
					tune_modes = Vec::new();
					for m in val.expect("tune-modes").split(',') {
						match m.trim() {
							"think" => tune_modes.push(true),
							"nothink" => tune_modes.push(false),
							o => {
								eprintln!("--tune-modes wants think and / or nothink, got {o}");
								exit(2)
							}
						}
					}
				}
				"tune-corpus" => tune_corpus = Some(val.expect("tune-corpus")),
				"tune-quick" => tune_quick = true,
				"ov-device" => ov_opts.device = val.expect("ov-device"),
				"ov-lib" => ov_opts.lib_dir = Some(val.expect("ov-lib").into()),
				"ov-cache" => ov_opts.cache_dir = Some(val.expect("ov-cache").into()),
				"ov-ctx" => {
					ov_opts.ctx = val.expect("ov-ctx").parse().expect("ov-ctx");
					ov_ctx_set = true;
				}
				"ov-chunk" => ov_opts.chunk = val.expect("ov-chunk").parse().expect("ov-chunk"),
				"ov-prop" => {
					let v = val.expect("ov-prop");
					let Some((k, v)) = v.split_once('=') else {
						eprintln!("--ov-prop wants KEY=VALUE, got {v}");
						exit(2)
					};
					ov_opts.props.push((k.to_string(), v.to_string()));
				}
				_ => {}
			},
			Err(Error::MissingArgument(cmd)) => {
				eprintln!("error: missing required argument for {cmd}");
				exit(2);
			}
			Err(Error::InvalidCommand(cmd)) => {
				eprintln!("error: invalid command {cmd}");
				exit(2);
			}
		}
	}
	if keep_awake {
		if awake::keep_awake() {
			eprintln!("keep-awake: display and system held on until exit");
		} else {
			eprintln!("keep-awake: not available here");
		}
	}
	// The profile: `--profile`, else `QWEN_PROFILE`, else `<model>.profile`
	// beside the model, else the built-in one. The shell's `QWEN_*`
	// variables win over its knobs (not over a tuner's trials).
	let tuning = matches!(mode, Mode::TuneProfile);
	if !tuning {
		quant::knob::freeze_user_env();
	}
	let profile_file = profile_path
		.map(std::path::PathBuf::from)
		.or_else(|| std::env::var_os("QWEN_PROFILE").map(Into::into))
		.or_else(|| server::profile::Profile::beside(&path));
	let profile = match &profile_file {
		Some(f) => match server::profile::Profile::read(f) {
			Ok(p) => p,
			Err(e) => {
				eprintln!("error: {e}");
				exit(2)
			}
		},
		None => server::profile::Profile::default(),
	};
	profile.apply_load();
	if mtp_bare {
		// the tuner sweeps every draft count a verify batch holds
		mtp = if tuning {
			offload::SPEC_MAX_SEQ - 1
		} else {
			profile.max_knob("QWEN_MTP_N").unwrap_or(3)
		};
	}
	server::set_profile(profile.clone());
	// On an OpenVINO export `--ctx` is `--ov-ctx` unless that is given too.
	if ctx > 0 && !ov_ctx_set {
		ov_opts.ctx = ctx;
	}
	// `--serve`: the default of requests that do not say (off, preserving,
	// unless given). Generation: medium (the template's open `<think>` with
	// no instruction) unless given.
	if let Some(rules) = client_ignore {
		server::set_client_ignore(rules);
	}
	if think.is_some() || !preserve_thinking {
		server::set_default_thinking(Thinking {
			preserve: preserve_thinking,
			..think.unwrap_or(Thinking::OFF)
		});
	}
	let think = Thinking {
		preserve: preserve_thinking,
		..think.unwrap_or(Thinking::on(Effort::Medium))
	};

	// a Qwen-Image checkpoint (or its `image.npu`) as the model path serves images only
	let image_only = imagegen::Model::is_model(std::path::Path::new(&path));
	if image_only && image_model.is_none() {
		image_model = Some(path.clone());
	}
	if let (Mode::Serve, Some(dir)) = (&mode, &image_model) {
		server::images::configure(server::images::ImageCfg {
			dir: dir.into(),
			out_dirs: image_out.clone(),
			idle_secs: image_idle,
			fmt: imagegen::WFmt::from_env(),
		});
	}
	if let (Mode::Serve, true) = (&mode, image_only) {
		server::serve_images(&host, port);
		exit(0);
	}
	let cache_opts = server::CacheOpts {
		mb: prompt_cache_mb,
		dir: prompt_cache_dir.clone(),
		disk_mb: prompt_cache_disk_mb,
		spill: prompt_cache_spill,
	};
	let result = match mode {
		Mode::Dump => dump(&path),
		Mode::Dequant => dequant_tensor(&path, &dequant_name),
		Mode::Forward => {
			let tokens: Vec<u32> = forward_tokens
				.split(',')
				.filter(|s| !s.trim().is_empty())
				.map(|s| s.trim().parse::<u32>().expect("token id"))
				.collect();
			forward(&path, &tokens, max_layers, n_gpu_layers, ctx, n_batch)
		}
		Mode::TokMeta => tokmeta(&path),
		Mode::Generate | Mode::Interactive => {
			// a completion without the template: the non-thinking set
			let templated = chat_mode || !images.is_empty() || system.is_some();
			let r = profile.resolve(think.enabled() && templated, 0);
			let mut params = r.params;
			params.temperature = temperature.unwrap_or(params.temperature);
			params.top_k = top_k.unwrap_or(params.top_k);
			params.top_p = top_p.unwrap_or(params.top_p);
			params.min_p = min_p.unwrap_or(params.min_p);
			params.presence_penalty = presence.unwrap_or(params.presence_penalty);
			params.repeat_penalty = rep.unwrap_or(params.repeat_penalty);
			if temperature.is_some_and(|t| t <= 0.0) {
				server::profile::greedy_penalties(&mut params, &|k| match k {
					"presence_penalty" => presence.is_some(),
					"repetition_penalty" => rep.is_some(),
					_ => false,
				});
			}
			eprintln!("{}", profile.describe());
			// the profile's decode knobs, for the whole run on this thread
			let _knobs = quant::knob::Scope::new(&r.knobs);
			let opts = GenOptions {
				max_new,
				params,
				seed,
				max_image_tokens: image_max_tokens,
				mtp,
				dflash: dflash.clone(),
				think,
			};
			if npuchat::is_npu(&path) {
				if mmproj.is_some() {
					eprintln!("error: the NPU engine's vision tower is the one its .npu file names (vision = DIR), not an --mmproj");
					exit(2);
				}
				if flashchat::is_flash(&path) {
					if matches!(mode, Mode::Interactive) {
						flashchat::interactive(&path, system.as_deref(), &images, opts)
					} else {
						flashchat::generate(
							&path,
							&gen_prompt,
							chat_mode,
							system.as_deref(),
							&images,
							opts,
						)
					}
				} else if matches!(mode, Mode::Interactive) {
					npuchat::interactive(&path, system.as_deref(), &images, opts)
				} else {
					npuchat::generate(
						&path,
						&gen_prompt,
						chat_mode,
						system.as_deref(),
						&images,
						opts,
					)
				}
			} else if ovchat::is_export(&path) {
				if !images.is_empty() || mmproj.is_some() {
					eprintln!("error: images are not supported on the OpenVINO backend");
					exit(2);
				}
				if matches!(mode, Mode::Interactive) {
					ovchat::interactive(&path, system.as_deref(), opts, &ov_opts)
				} else {
					ovchat::generate(
						&path,
						&gen_prompt,
						chat_mode,
						system.as_deref(),
						opts,
						&ov_opts,
					)
				}
			} else if matches!(mode, Mode::Interactive) {
				interactive(
					&path,
					system.as_deref(),
					&images,
					mmproj.as_deref(),
					opts,
					n_gpu_layers,
					ctx,
					n_batch,
				)
			} else {
				generate(
					&path,
					&gen_prompt,
					chat_mode,
					system.as_deref(),
					&images,
					mmproj.as_deref(),
					opts,
					n_gpu_layers,
					ctx,
					n_batch,
				)
			}
		}
		Mode::ImageProbe => {
			// The mmproj is the point of this mode; fall back to the positional
			// path so `cli <mmproj.gguf> --image-probe --image x.png` works.
			let mm = mmproj.clone().unwrap_or_else(|| path.clone());
			chat::image_probe(&mm, &images, image_max_tokens, image_dump.as_deref())
		}
		Mode::Serve if npuchat::is_npu(&path) && flashchat::is_flash(&path) => {
			flashchat::serve(&path, &host, port, &cache_opts, image_max_tokens)
		}
		Mode::Serve if npuchat::is_npu(&path) => {
			npuchat::serve(&path, &host, port, &cache_opts, image_max_tokens)
		}
		Mode::Serve if ovchat::is_export(&path) => {
			if mmproj.is_some() || mtp > 0 || dflash.is_some() {
				eprintln!(
					"error: --mmproj / --mtp / --dflash are not available on the OpenVINO backend"
				);
				exit(2);
			}
			ovchat::serve(&path, &ov_opts, &host, port)
		}
		Mode::Serve => {
			server::serve(
				&path,
				mmproj.as_deref(),
				&host,
				port,
				n_gpu_layers,
				ctx,
				n_batch,
				image_max_tokens,
				&cache_opts,
				mtp,
				dflash.as_deref(),
			);
			Ok(())
		}
		Mode::Tokenize => tokenize_dump(&path, &tokenize_text),
		Mode::KldEval => kld::kld_eval(
			&path,
			&kld_file,
			kld_tokens,
			&sparse_sweep,
			n_gpu_layers,
			n_batch,
		),
		Mode::TuneSplit => tune_split(&path, &tune_seqs, tune_rounds.unwrap_or(3), n_gpu_layers),
		Mode::TuneProfile => {
			server::set_tune(server::TuneOpts {
				out: tune_out.clone().unwrap().into(),
				depths: tune_depths.clone(),
				tokens: tune_tokens,
				rounds: tune_rounds.unwrap_or(2),
				modes: tune_modes.clone(),
				corpus: tune_corpus.clone().map(Into::into),
				quick: tune_quick,
			});
			// as `--serve`, without a listener and without persisting
			// snapshots
			let mem_only = server::CacheOpts {
				mb: prompt_cache_mb,
				..Default::default()
			};
			if npuchat::is_npu(&path) && flashchat::is_flash(&path) {
				flashchat::serve(&path, &host, port, &mem_only, image_max_tokens)
			} else if npuchat::is_npu(&path) {
				npuchat::serve(&path, &host, port, &mem_only, image_max_tokens)
			} else if ovchat::is_export(&path) {
				ovchat::serve(&path, &ov_opts, &host, port)
			} else {
				server::serve(
					&path,
					mmproj.as_deref(),
					&host,
					port,
					n_gpu_layers,
					ctx,
					n_batch,
					image_max_tokens,
					&mem_only,
					mtp,
					dflash.as_deref(),
				);
				Ok(())
			}
		}
		Mode::EmbedTok => {
			let tokens: Vec<u32> = embed_ids
				.split(',')
				.filter(|s| !s.trim().is_empty())
				.map(|s| s.trim().parse::<u32>().expect("token id"))
				.collect();
			embed_check(&path, &tokens)
		}
	};

	if let Err(e) = result {
		eprintln!("error: {e}");
		exit(1);
	}
}
