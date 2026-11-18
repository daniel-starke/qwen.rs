//! `cli MODEL --tune-profile OUT`: measure this machine's best decode knobs
//! for agentic coding, per thinking mode and context depth, and write them
//! as a profile (`profile.rs`).
//!
//! It loads the model as `--serve` would and runs real requests through
//! the server's own path ([`generate`]: prompt template, tools, prefix
//! cache, drafts), no HTTP:
//!
//! - **Workload.** One agentic conversation: a coding-agent system prompt,
//!   six tools (Read, Edit, Write, Bash, Grep, Glob), a task, then Read
//!   calls whose results are source files (the embedded [`CORPUS`] of this
//!   repository's larger sources, or `--tune-corpus DIR`). It grows to each
//!   depth of `--tune-depths` (default: the base prompt, 16k, 64k and the
//!   context less the decode); at each depth the last tool result carries a
//!   user line asking for the change, so the reply is a short plan and an
//!   Edit call (non-thinking) or reasoning (thinking). Each depth's prompt
//!   extends the previous one, so the prefix cache prefills only what a
//!   depth adds, and every trial at a depth restores the same state.
//! - **Step costs.** On Flash-Next the MTP keeps the drafts that maximize
//!   expected tokens per ms by a table of step costs
//!   (`QWEN_FLASH_MTP_COST`): measured here at each depth by timing verify
//!   steps of 1-8 positions of real tokens plus their draft chains.
//! - **Knobs.** Coordinate descent from the code defaults (then from the
//!   previous depth's result): for each knob, every candidate value is run
//!   in `--tune-rounds` interleaved rounds (a round runs all values with
//!   the same seed, so the outputs agree but for near-ties); a value
//!   replaces the current one only when its mean rate relative to the
//!   current one's in the same round exceeds 1 + `MARGIN` (this host has
//!   ~35 % performance-state swings; interleaving is what makes a 2 %
//!   difference visible).
//!
//! Sampling is not tuned: it is the model's, not the machine's (the cards'
//! sets, or the `--profile` given). Earlier responses are cleared before
//! every trial so the cross-request context drafts cannot copy the
//! previous trial's answer.

use super::*;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// What `--tune-profile` was asked for.
#[derive(Clone, Debug)]
pub struct TuneOpts {
	pub out: PathBuf,
	/// Prompt depths in tokens; empty = the default ladder.
	pub depths: Vec<usize>,
	/// Tokens decoded per trial.
	pub tokens: usize,
	pub rounds: usize,
	/// Modes to tune (`true` = thinking).
	pub modes: Vec<bool>,
	/// Source files for the tool results instead of the embedded corpus.
	pub corpus: Option<PathBuf>,
	/// Fewer candidate values.
	pub quick: bool,
}

static TUNE: Mutex<Option<TuneOpts>> = Mutex::new(None);

/// Make the next `serve` / `serve_engine` run the tuner instead of
/// listening.
pub fn set_tune(o: TuneOpts) {
	*TUNE.lock().unwrap_or_else(|e| e.into_inner()) = Some(o);
}

pub(crate) fn take() -> Option<TuneOpts> {
	TUNE.lock().unwrap_or_else(|e| e.into_inner()).take()
}

/// A candidate's mean rate must beat the current value's by this much.
const MARGIN: f64 = 0.02;

/// Source files the tool results show, embedded so the tuner needs nothing
/// beside the binary: ~1.1 MB of Rust, ~330k tokens.
const CORPUS: &[(&str, &str)] = &[
	("crates/server/src/lib.rs", include_str!("lib.rs")),
	("crates/server/src/openai.rs", include_str!("openai.rs")),
	("crates/server/src/tools.rs", include_str!("tools.rs")),
	(
		"crates/prefixcache/src/lib.rs",
		include_str!("../../prefixcache/src/lib.rs"),
	),
	(
		"crates/model/src/forward.rs",
		include_str!("../../model/src/forward.rs"),
	),
	(
		"crates/model/src/mtp.rs",
		include_str!("../../model/src/mtp.rs"),
	),
	(
		"crates/model/src/dflash.rs",
		include_str!("../../model/src/dflash.rs"),
	),
	(
		"crates/model/src/suffix.rs",
		include_str!("../../model/src/suffix.rs"),
	),
	(
		"crates/tokenizer/src/lib.rs",
		include_str!("../../tokenizer/src/lib.rs"),
	),
	(
		"crates/quant/src/linear.rs",
		include_str!("../../quant/src/linear.rs"),
	),
	(
		"crates/quant/src/parallel.rs",
		include_str!("../../quant/src/parallel.rs"),
	),
	(
		"crates/offload/src/manager.rs",
		include_str!("../../offload/src/manager.rs"),
	),
	(
		"crates/offload/src/gpu/device.rs",
		include_str!("../../offload/src/gpu/device.rs"),
	),
	(
		"crates/offload/src/gpu/stream.rs",
		include_str!("../../offload/src/gpu/stream.rs"),
	),
	(
		"crates/offload/src/gpu/gemv.rs",
		include_str!("../../offload/src/gpu/gemv.rs"),
	),
	(
		"crates/offload/src/gpu/matmul.rs",
		include_str!("../../offload/src/gpu/matmul.rs"),
	),
	(
		"crates/offload/src/gpu/attn.rs",
		include_str!("../../offload/src/gpu/attn.rs"),
	),
	(
		"crates/npu/src/engine.rs",
		include_str!("../../npu/src/engine.rs"),
	),
	(
		"crates/npu/src/flash/engine.rs",
		include_str!("../../npu/src/flash/engine.rs"),
	),
	(
		"crates/npu/src/flash/gpudec.rs",
		include_str!("../../npu/src/flash/gpudec.rs"),
	),
	(
		"crates/ov/src/stateless.rs",
		include_str!("../../ov/src/stateless.rs"),
	),
];

const SYSTEM: &str = "You are a coding agent working in a user's repository. You \
help with software engineering tasks: reading and changing code, running commands, \
explaining what you find.\n\n\
# How to work\n\
- Read the code before you change it. Use the Read tool for files, Grep and Glob to \
find things, Bash to build and test.\n\
- Keep changes small and in the style of the surrounding code: same naming, same \
comment density, same idioms.\n\
- Prefer editing existing files with the Edit tool over writing new ones. An Edit's \
old_string must match the file exactly, including indentation.\n\
- Do not guess: if a fact can be checked in the code, check it.\n\
- After a change, build and run the tests that cover it, and report the outcome \
faithfully, including failures.\n\n\
# Output\n\
- Be concise. Lead with the result. Do not restate the request.\n\
- Reference code as path:line.\n\
- Before each tool call, say in one short sentence what you are doing.";

const TASK: &str = "The server takes its sampling defaults from a profile now \
(`server::profile`), but nothing reports which values a request actually used. \
Read the server's request path and the files it calls into, one file at a time with \
the Read tool, until you understand where the sampling parameters and the decode \
knobs are resolved. Then add a `sampling` object to the `usage` of every response \
(both APIs) that lists temperature, top_p, top_k, min_p and the penalties the \
request ran with. Keep the change minimal and match the code style.";

/// The user line on the last tool result of each depth.
const NUDGE: &str = "That is enough reading. Write the change now: a short plan, then \
the first Edit call.";

const TOOLS: &str = r#"[
{"name":"Read","description":"Read a file from the local filesystem. Returns its contents.","input_schema":{"type":"object","properties":{"file_path":{"type":"string","description":"Absolute or repository-relative path"},"offset":{"type":"integer","description":"Line to start at"},"limit":{"type":"integer","description":"Number of lines"}},"required":["file_path"]}},
{"name":"Edit","description":"Replace an exact string in a file. old_string must occur exactly once.","input_schema":{"type":"object","properties":{"file_path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["file_path","old_string","new_string"]}},
{"name":"Write","description":"Write a file, replacing it if it exists.","input_schema":{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}},"required":["file_path","content"]}},
{"name":"Bash","description":"Run a shell command and return its output.","input_schema":{"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"integer"}},"required":["command"]}},
{"name":"Grep","description":"Search file contents with a regular expression.","input_schema":{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"},"glob":{"type":"string"}},"required":["pattern"]}},
{"name":"Glob","description":"Find files by glob pattern.","input_schema":{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"}},"required":["pattern"]}}
]"#;

/// One knob and its candidate values.
struct Param {
	name: &'static str,
	values: Vec<String>,
}

fn param(name: &'static str, values: &[&str]) -> Param {
	Param {
		name,
		values: values.iter().map(|v| v.to_string()).collect(),
	}
}

/// What the loaded backend drafts with, which decides the knobs.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
	GgufMtp(usize),
	GgufDflash(usize),
	GgufPlain,
	/// Flash-Next's engine with its MTP head; the verify step's width.
	Flash(usize),
	/// Another engine that verifies drafts.
	EngineSpec,
	EnginePlain,
}

fn kind(l: &Loaded) -> Kind {
	match &l.backend {
		Backend::Gguf(m) if m.dflash_enabled() => Kind::GgufDflash(m.draft_cap()),
		Backend::Gguf(m) if m.mtp_enabled() => Kind::GgufMtp(m.draft_cap()),
		Backend::Gguf(_) => Kind::GgufPlain,
		Backend::Engine { eng, info } => {
			let k = eng.lock().unwrap_or_else(|e| e.into_inner()).spec_k();
			if k <= 1 {
				Kind::EnginePlain
			} else if info.model.contains("Flash") {
				Kind::Flash(k)
			} else {
				Kind::EngineSpec
			}
		}
	}
}

/// The code defaults (where descent starts) and the knobs to tune.
fn knobs(k: Kind, quick: bool) -> (Vec<(String, String)>, Vec<Param>) {
	let upto = |cap: usize, all: &[&'static str]| -> Vec<&'static str> {
		all.iter()
			.copied()
			.filter(|v| v.parse::<usize>().is_ok_and(|n| n <= cap))
			.collect()
	};
	let d = |v: &[(&str, &str)]| -> Vec<(String, String)> {
		v.iter()
			.map(|(a, b)| (a.to_string(), b.to_string()))
			.collect()
	};
	let q = |full: &[&'static str], fast: &[&'static str]| -> Vec<&'static str> {
		if quick { fast } else { full }.to_vec()
	};
	match k {
		Kind::GgufMtp(cap) => (
			d(&[
				("QWEN_MTP_N", &cap.to_string()),
				("QWEN_MTP_PMIN", "0.8"),
				("QWEN_SUFFIX_MIN", "12"),
			]),
			vec![
				param(
					"QWEN_MTP_N",
					&upto(
						cap,
						&q(&["1", "2", "3", "4", "5", "7"], &["1", "2", "3", "5"]),
					),
				),
				param(
					"QWEN_MTP_PMIN",
					&q(&["0.5", "0.65", "0.8", "0.9"], &["0.6", "0.8", "0.9"]),
				),
				param(
					"QWEN_SUFFIX_MIN",
					&q(&["6", "9", "12", "16"], &["8", "12", "16"]),
				),
			],
		),
		Kind::GgufDflash(cap) => (
			d(&[
				("QWEN_DFLASH_N", &cap.to_string()),
				("QWEN_DFLASH_PMIN", "0"),
				("QWEN_SUFFIX_MIN", "12"),
			]),
			vec![
				param(
					"QWEN_DFLASH_N",
					&upto(cap, &q(&["2", "3", "4", "5", "7"], &["3", "4", "5"])),
				),
				param(
					"QWEN_DFLASH_PMIN",
					&q(&["0", "0.2", "0.35", "0.5"], &["0", "0.3"]),
				),
				param(
					"QWEN_SUFFIX_MIN",
					&q(&["6", "9", "12", "16"], &["8", "12", "16"]),
				),
			],
		),
		Kind::GgufPlain => (
			d(&[("QWEN_SUFFIX_PLAIN_MIN", "3")]),
			vec![param(
				"QWEN_SUFFIX_PLAIN_MIN",
				&q(&["3", "4", "6", "8"], &["3", "5", "8"]),
			)],
		),
		Kind::Flash(width) => (
			d(&[
				("QWEN_FLASH_MTP_N", "3"),
				("QWEN_FLASH_MTP_PSCALE", "0.8"),
				("QWEN_FLASH_MTP_PMIN", "0.2"),
				("QWEN_SUFFIX_ENGINE_MIN", "6"),
			]),
			vec![
				param(
					"QWEN_FLASH_MTP_N",
					&upto(width - 1, &q(&["2", "3", "4", "5", "7"], &["2", "3", "5"])),
				),
				param(
					"QWEN_FLASH_MTP_PSCALE",
					&q(&["0.6", "0.7", "0.8", "0.9"], &["0.7", "0.8", "0.9"]),
				),
				param(
					"QWEN_FLASH_MTP_PMIN",
					&q(&["0.1", "0.2", "0.3"], &["0.1", "0.2", "0.3"]),
				),
				param(
					"QWEN_SUFFIX_ENGINE_MIN",
					&q(&["4", "6", "9", "12"], &["4", "6", "12"]),
				),
			],
		),
		Kind::EngineSpec => (
			d(&[("QWEN_SUFFIX_ENGINE_MIN", "6")]),
			vec![param(
				"QWEN_SUFFIX_ENGINE_MIN",
				&q(&["4", "6", "9", "12"], &["4", "6", "12"]),
			)],
		),
		Kind::EnginePlain => (Vec::new(), Vec::new()),
	}
}

/// The growing conversation: JSON message objects.
struct Conv {
	files: Vec<(String, String)>,
	/// Next file to read.
	next: usize,
	msgs: Vec<String>,
	/// Estimated prompt tokens of `msgs` (exact after [`Conv::count`]).
	est: usize,
	/// Times the corpus wrapped around.
	wraps: usize,
}

impl Conv {
	fn new(mut files: Vec<(String, String)>) -> Conv {
		// small files first: the base prompt reads one whole
		files.sort_by_key(|f| f.1.len());
		Conv {
			files,
			next: 0,
			msgs: vec![format!(r#"{{"role":"user","content":{}}}"#, jstr(TASK))],
			est: 0,
			wraps: 0,
		}
	}

	/// The request for the conversation so far.
	fn request(&self, think: bool, max_tokens: usize, seed: u64) -> String {
		let thinking = if think {
			// `medium`: no effort instruction in the system turn, so both
			// modes share every cached prefix
			r#""thinking":{"type":"enabled"},"output_config":{"effort":"medium"}"#
		} else {
			r#""thinking":{"type":"disabled"}"#
		};
		format!(
			r#"{{"model":"tune","max_tokens":{max_tokens},"seed":{seed},"stream":false,{thinking},"system":{},"tools":{TOOLS},"messages":[{}]}}"#,
			jstr(SYSTEM),
			self.msgs.join(",")
		)
	}

	/// Read calls until the prompt is about `target` tokens (0: one whole
	/// file); the last result carries [`NUDGE`].
	fn grow(&mut self, tok: &Tokenizer, target: usize) {
		loop {
			let left = target.saturating_sub(self.est);
			if self.msgs.len() > 1 && (target == 0 || left < 64) {
				break;
			}
			if self.next == self.files.len() {
				self.next = 0;
				self.wraps += 1;
			}
			let (path, text) = &self.files[self.next];
			self.next += 1;
			let mut text = text.clone();
			let mut n = tok.encode(&text).len() + 40;
			if target > 0 && n > left + 64 {
				// cut by lines to fit
				let keep = text.len() * left.saturating_sub(40) / n.max(1);
				let cut = text[..keep.min(text.len())].rfind('\n').unwrap_or(0);
				text.truncate(cut);
				n = tok.encode(&text).len() + 40;
			}
			let id = format!("toolu_{:04}", self.msgs.len());
			self.msgs.push(format!(
				r#"{{"role":"assistant","content":[{{"type":"text","text":{}}},{{"type":"tool_use","id":"{id}","name":"Read","input":{{"file_path":{}}}}}]}}"#,
				jstr(&format!("Reading {path}.")),
				jstr(path)
			));
			self.msgs.push(format!(
				r#"{{"role":"user","content":[{{"type":"tool_result","tool_use_id":"{id}","content":{}}}]}}"#,
				jstr(&text)
			));
			self.est += n;
		}
		// the nudge, on the last result
		let last = self.msgs.pop().unwrap();
		let last = last.strip_suffix("]}").unwrap();
		self.msgs.push(format!(
			r#"{last},{{"type":"text","text":{}}}]}}"#,
			jstr(NUDGE)
		));
	}

	/// The exact prompt length of the request (as `count_tokens`).
	fn count(&mut self, l: &Loaded, think: bool) -> Result<usize, String> {
		let req = Json::parse(self.request(think, 1, 0).as_bytes())?;
		let tools = tools::specs_from_request(&req);
		let tool_json: Vec<String> = tools.iter().map(|t| t.json.clone()).collect();
		let turns = turns_from_request(&req, &mut |_| Ok(String::new()))?;
		let th = thinking_from_request(&req)?.think;
		let prompt = Tokenizer::build_prompt_with(&turns, &tool_json, th, th.open());
		let n = encode_prompt(&l.tok, &prompt).len();
		self.est = n;
		Ok(n)
	}
}

/// The tool-result files: `dir`'s text sources (sorted, skipping build
/// output), else [`CORPUS`].
fn corpus(dir: Option<&Path>) -> Result<Vec<(String, String)>, String> {
	let Some(dir) = dir else {
		return Ok(CORPUS
			.iter()
			.map(|(p, t)| (p.to_string(), t.to_string()))
			.collect());
	};
	let exts = [
		"rs", "py", "c", "h", "cc", "cpp", "hpp", "js", "ts", "go", "java", "md", "toml", "glsl",
		"txt", "json", "sh",
	];
	let mut out = Vec::new();
	let mut stack = vec![dir.to_path_buf()];
	while let Some(d) = stack.pop() {
		let mut ents: Vec<_> = std::fs::read_dir(&d)
			.map_err(|e| format!("{}: {e}", d.display()))?
			.filter_map(|e| e.ok())
			.collect();
		ents.sort_by_key(|e| e.path());
		for e in ents {
			let p = e.path();
			let name = e.file_name().to_string_lossy().into_owned();
			if p.is_dir() {
				if !name.starts_with('.') && name != "target" && name != "node_modules" {
					stack.push(p);
				}
			} else if p
				.extension()
				.is_some_and(|x| exts.contains(&x.to_string_lossy().as_ref()))
			{
				let Ok(t) = std::fs::read_to_string(&p) else {
					continue;
				};
				if (512..400_000).contains(&t.len()) {
					let rel = p.strip_prefix(dir).unwrap_or(&p);
					out.push((rel.to_string_lossy().replace('\\', "/"), t));
				}
			}
		}
	}
	if out.is_empty() {
		return Err(format!("--tune-corpus {}: no source files", dir.display()));
	}
	Ok(out)
}

/// One request; its numbers.
fn trial(l: &Loaded, req: &str) -> Result<GenStats, String> {
	// no drafts from earlier trials' answers
	*l.responses.lock().unwrap_or_else(|e| e.into_inner()) = Arc::default();
	*l.last.lock().unwrap_or_else(|e| e.into_inner()) = None;
	let req = Json::parse(req.as_bytes())?;
	let mut sink: Vec<u8> = Vec::new();
	let mut wire = AnthropicWire {
		stream: false,
		id: new_id(),
		model: "tune".into(),
		open: None,
		index: 0,
	};
	generate(&mut sink, l, &req, &mut wire, 1).map_err(|e| e.to_string())?;
	l.last
		.lock()
		.unwrap_or_else(|e| e.into_inner())
		.take()
		.ok_or_else(|| {
			let body = String::from_utf8_lossy(&sink);
			format!(
				"request failed: {}",
				body.lines().last().unwrap_or_default()
			)
		})
}

fn rate(s: &GenStats) -> f64 {
	s.output_tokens as f64 / s.decode_s.max(1e-9)
}

/// Flash-Next's step costs at the engine's current depth: per width k, a
/// verify step of k real tokens plus a chain of k - 1 drafts (one head
/// pass that drafts nothing for k = 1), median of `reps`, in ms.
fn step_costs(l: &Loaded, width: usize, ids: &[u32], reps: usize) -> Result<Vec<f64>, String> {
	let Backend::Engine { eng, .. } = &l.backend else {
		return Err("step costs need an engine".into());
	};
	let mut e = eng.lock().unwrap_or_else(|e| e.into_inner());
	// a trial leaves its last verify step open
	e.accept(0, None)?;
	let mut t: Vec<Vec<f64>> = vec![Vec::new(); width];
	let mut off = 0usize;
	for _ in 0..reps {
		for k in 1..=width {
			let n = (k - 1).max(1).to_string();
			let pmin = if k == 1 { "2" } else { "0" };
			let _s = quant::knob::Scope::new(&[
				("QWEN_FLASH_MTP_N", n.as_str()),
				("QWEN_FLASH_MTP_PMIN", pmin),
				("QWEN_FLASH_MTP_EARLY", "0"),
				("QWEN_FLASH_MTP_COST", "0"),
				("QWEN_FLASH_MTP_CAL", "0"),
			]);
			off = (off + 37) % (ids.len() - width - 1);
			let t0 = Instant::now();
			e.verify(&ids[off..off + k])?;
			let d = e.accept(1, Some(ids[off + 1]))?;
			t[k - 1].push(t0.elapsed().as_secs_f64() * 1e3);
			if k > 1 && d.len() != k - 1 {
				eprintln!(
					"tune: warning: a chain of {} drafts asked, {} made",
					k - 1,
					d.len()
				);
			}
		}
	}
	// the drafts of the last accept were never verified: drop them
	e.accept(0, None)?;
	Ok(t.into_iter()
		.map(|mut v| {
			v.sort_by(|a, b| a.total_cmp(b));
			v[v.len() / 2]
		})
		.collect())
}

/// `cur` with `name` set to `v`.
fn with(cur: &[(String, String)], name: &str, v: &str) -> Vec<(String, String)> {
	let mut o: Vec<(String, String)> = cur.iter().filter(|(k, _)| k != name).cloned().collect();
	o.push((name.to_string(), v.to_string()));
	o
}

fn get<'a>(cur: &'a [(String, String)], name: &str) -> Option<&'a str> {
	cur.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
}

/// The result of one (depth, mode).
struct Tuned {
	knobs: Vec<(String, String)>,
	default_rate: f64,
	rate: f64,
	tokens_per_step: f64,
}

/// Coordinate descent over `params` from `start` (`fixed` held constant).
#[allow(clippy::too_many_arguments)]
fn descend(
	l: &Loaded,
	conv: &Conv,
	think: bool,
	o: &TuneOpts,
	fixed: &[(String, String)],
	start: &[(String, String)],
	defaults: &[(String, String)],
	params: &[Param],
	label: &str,
) -> Result<Tuned, String> {
	let run = |knobs: &[(String, String)], round: usize| -> Result<GenStats, String> {
		let mut all = fixed.to_vec();
		all.extend(knobs.iter().cloned());
		let _s = quant::knob::Scope::new(&all);
		trial(l, &conv.request(think, o.tokens, 7919 + round as u64))
	};
	let show = |knobs: &[(String, String)]| {
		knobs
			.iter()
			.map(|(k, v)| format!("{}={v}", k.trim_start_matches("QWEN_")))
			.collect::<Vec<_>>()
			.join(" ")
	};
	// the defaults, once per round, for the summary
	let mut def_rate = 0.0;
	for r in 0..o.rounds {
		let s = run(defaults, r)?;
		def_rate += rate(&s) / o.rounds as f64;
	}
	let mut cur = start.to_vec();
	let mut last = (0.0, 0.0);
	for p in params {
		let cv = get(&cur, p.name).unwrap_or_default().to_string();
		let mut vals = p.values.clone();
		if !vals.contains(&cv) {
			vals.push(cv.clone());
		}
		let mut ratio = vec![0.0f64; vals.len()];
		let mut base_sum = (0.0, 0.0);
		for r in 0..o.rounds {
			let base = run(&cur, r)?;
			let br = rate(&base);
			base_sum.0 += br / o.rounds as f64;
			base_sum.1 += base.output_tokens as f64 / base.steps.max(1) as f64 / o.rounds as f64;
			// rotate the order per round
			let n = vals.len();
			for i in 0..n {
				let j = (i + r) % n;
				let v = &vals[j];
				let x = if *v == cv {
					br
				} else {
					let s = run(&with(&cur, p.name, v), r)?;
					let x = rate(&s);
					eprintln!(
						"tune: {label} {}={v}: {x:.2} tok/s ({:.2} tok/step, {} tok) vs {br:.2}",
						p.name.trim_start_matches("QWEN_"),
						s.output_tokens as f64 / s.steps.max(1) as f64,
						s.output_tokens
					);
					x
				};
				ratio[j] += x / br.max(1e-9) / o.rounds as f64;
			}
		}
		let (bi, &best) = ratio
			.iter()
			.enumerate()
			.max_by(|a, b| a.1.total_cmp(b.1))
			.unwrap();
		if vals[bi] != cv && best > 1.0 + MARGIN {
			eprintln!(
				"tune: {label} {} {cv} -> {} (+{:.1} %)",
				p.name.trim_start_matches("QWEN_"),
				vals[bi],
				(best - 1.0) * 100.0
			);
			cur = with(&cur, p.name, &vals[bi]);
			last = (base_sum.0 * best, base_sum.1);
		} else {
			eprintln!(
				"tune: {label} {} stays {cv} (best other {:+.1} %)",
				p.name.trim_start_matches("QWEN_"),
				ratio
					.iter()
					.enumerate()
					.filter(|&(i, _)| vals[i] != cv)
					.map(|(_, &x)| (x - 1.0) * 100.0)
					.fold(f64::NEG_INFINITY, f64::max)
			);
			last = base_sum;
		}
	}
	eprintln!(
		"tune: {label} result {} ({:.2} tok/s, defaults {def_rate:.2})",
		show(&cur),
		last.0
	);
	Ok(Tuned {
		knobs: cur,
		default_rate: def_rate,
		rate: last.0,
		tokens_per_step: last.1,
	})
}

/// Today as YYYY-MM-DD (UTC).
fn today() -> String {
	let secs = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map_or(0, |d| d.as_secs()) as i64;
	// civil from days (H. Hinnant)
	let z = secs / 86400 + 719468;
	let era = z.div_euclid(146097);
	let doe = z - era * 146097;
	let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
	let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
	let mp = (5 * doy + 2) / 153;
	let d = doy - (153 * mp + 2) / 5 + 1;
	let m = if mp < 10 { mp + 3 } else { mp - 9 };
	let y = yoe + era * 400 + (m <= 2) as i64;
	format!("{y:04}-{m:02}-{d:02}")
}

/// Run the tuner on `l` (see the module docs).
pub(crate) fn run(l: &Loaded, o: TuneOpts) -> Result<(), String> {
	let t_all = Instant::now();
	// the trials' knobs must be the only ones: the profile keeps its
	// sampling, loses its knobs
	let base = current_profile();
	set_profile(base.sampling_only());
	let k = kind(l);
	let (defaults, params) = knobs(k, o.quick);
	eprintln!("tune: backend {k:?}, {} knobs", params.len());
	if !l.cache.enabled() {
		eprintln!("tune: warning: no prefix cache (--prompt-cache-mb 0?): every trial prefills its whole prompt");
	}
	let mut conv = Conv::new(corpus(o.corpus.as_deref())?);
	// the base prompt: system, tools, task and one file
	conv.grow(&l.tok, 0);
	let base_n = conv.count(l, false)?;
	let ctx = l.ctx();
	let top = ctx.saturating_sub(o.tokens + 1024);
	let mut depths: Vec<usize> = if o.depths.is_empty() {
		[base_n, 16384, 65536, top].to_vec()
	} else {
		o.depths.clone()
	};
	for d in depths.iter_mut() {
		*d = (*d).max(base_n);
	}
	depths.retain(|&d| d <= top);
	depths.sort_unstable();
	depths.dedup();
	if depths.is_empty() {
		return Err(format!(
			"no depth fits: base prompt {base_n} tokens, context {ctx} (--ctx)"
		));
	}
	eprintln!(
		"tune: depths {depths:?}, {} tokens per trial, {} rounds, context {ctx}",
		o.tokens, o.rounds
	);
	let ids_for_costs: Vec<u32> = l.tok.encode(&conv.files[0].1);

	let mut prof = profile::Profile::default();
	prof.load = base.load.clone();
	let mut summary: Vec<String> = Vec::new();
	let mut prev: Vec<Option<Vec<(String, String)>>> = vec![None; 2];
	let mut prev_cost: Option<String> = None;
	let mut written: Vec<Vec<(String, String)>> = vec![Vec::new(); 2];
	for (j, &d) in depths.iter().enumerate() {
		if j > 0 || d > conv.est {
			// (the per-file estimate runs ~0.1 % short of the exact count)
			conv.grow(&l.tok, d.saturating_sub(d / 200 + 256).max(conv.est + 1));
		}
		let n = conv.count(l, false)?;
		if n > top {
			return Err(format!(
				"depth {d}: the prompt came out at {n} tokens, past {top}"
			));
		}
		// the band this depth's values apply to: from the geometric middle
		// to the previous depth
		let from = if j == 0 {
			0
		} else {
			((depths[j - 1] as f64 * d as f64).sqrt() as usize) / 1024 * 1024
		};
		// prefill (cached for every trial below)
		let t = Instant::now();
		let s = trial(l, &conv.request(false, 1, 1))?;
		let pre = (s.input_tokens - s.reused) as f64 / s.prefill_s.max(1e-9);
		eprintln!(
			"tune: depth {n}: prefilled {} new tokens in {:.1} s ({pre:.1} tok/s), {:.1} s in all",
			s.input_tokens - s.reused,
			s.prefill_s,
			t.elapsed().as_secs_f64()
		);
		if conv.wraps > 0 {
			eprintln!("tune: warning: the corpus wrapped around {} times (pass a larger --tune-corpus): context drafts will find repeated files", conv.wraps);
		}
		let mut fixed = Vec::new();
		let mut band_common = Vec::new();
		if let Kind::Flash(width) = k {
			let c = step_costs(l, width, &ids_for_costs, 3)?;
			let cs = c
				.iter()
				.map(|x| format!("{}", x.round().max(1.0) as u64))
				.collect::<Vec<_>>()
				.join(",");
			eprintln!("tune: depth {n}: step costs (ms, 1..{width} positions + drafts) {cs}");
			fixed.push(("QWEN_FLASH_MTP_COST".to_string(), cs.clone()));
			if prev_cost.as_deref() != Some(cs.as_str()) {
				band_common.push(("QWEN_FLASH_MTP_COST".to_string(), cs.clone()));
				prev_cost = Some(cs);
			}
		}
		if !band_common.is_empty() {
			prof.push(None, from, band_common);
		}
		for &think in &o.modes {
			let mi = think as usize;
			let label = format!("depth {n} {}", if think { "think" } else { "nothink" });
			let start = prev[mi].clone().unwrap_or_else(|| defaults.clone());
			let r = descend(
				l, &conv, think, &o, &fixed, &start, &defaults, &params, &label,
			)?;
			summary.push(format!(
				"{label}: {:.2} tok/s ({:.2} tok/step) tuned, {:.2} with the defaults; prefill {pre:.0} tok/s",
				r.rate, r.tokens_per_step, r.default_rate
			));
			// a section holds what changed against the band before
			let mut vals: Vec<(String, String)> = Vec::new();
			if j == 0 {
				let p = base.resolve(think, 0).params;
				vals.extend([
					("temperature".into(), fmt(p.temperature)),
					("top_p".into(), fmt(p.top_p)),
					("top_k".into(), p.top_k.to_string()),
					("min_p".into(), fmt(p.min_p)),
					("presence_penalty".into(), fmt(p.presence_penalty)),
					("frequency_penalty".into(), fmt(p.frequency_penalty)),
					("repetition_penalty".into(), fmt(p.repeat_penalty)),
				]);
			}
			for (kk, v) in &r.knobs {
				if get(&written[mi], kk) != Some(v.as_str()) {
					vals.push((kk.clone(), v.clone()));
				}
			}
			written[mi] = r.knobs.clone();
			if !vals.is_empty() {
				prof.push(Some(think), from, vals);
			}
			prev[mi] = Some(r.knobs);
		}
		write(&o, &prof, l, k, &depths[..=j], &summary, t_all)?;
	}
	eprintln!(
		"tune: done in {:.1} min: {}",
		t_all.elapsed().as_secs_f64() / 60.0,
		o.out.display()
	);
	for s in &summary {
		eprintln!("tune:   {s}");
	}
	Ok(())
}

fn fmt(x: f32) -> String {
	let s = format!("{x}");
	if s.contains('.') || s.contains('e') {
		s
	} else {
		format!("{s}.0")
	}
}

fn write(
	o: &TuneOpts,
	prof: &profile::Profile,
	l: &Loaded,
	k: Kind,
	depths: &[usize],
	summary: &[String],
	t0: Instant,
) -> Result<(), String> {
	let mut h = vec![
		format!("qwen.rs profile, written by --tune-profile on {}", today()),
		format!(
			"model {} ({k:?}), context {}, {} logical CPUs, {}",
			l.model_id(),
			l.ctx(),
			std::thread::available_parallelism().map_or(0, |n| n.get()),
			std::env::consts::OS
		),
		format!(
			"{} tokens per trial, {} rounds, depths {depths:?}, {:.0} min",
			o.tokens,
			o.rounds,
			t0.elapsed().as_secs_f64() / 60.0
		),
		"".into(),
		"Measured (decode, agentic coding conversation at each depth):".into(),
	];
	h.extend(summary.iter().map(|s| format!("  {s}")));
	h.extend([
		"".into(),
		"Sampling is the model's (the cards' sets unless --profile gave others);".into(),
		"the QWEN_* knobs are this machine's: rerun --tune-profile on another one.".into(),
		"Format and precedence: crates/server/src/profile.rs.".into(),
	]);
	std::fs::write(&o.out, prof.to_text(&h)).map_err(|e| format!("{}: {e}", o.out.display()))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn knob_grids_hold_the_defaults_range() {
		for k in [
			Kind::GgufMtp(3),
			Kind::GgufMtp(7),
			Kind::GgufDflash(4),
			Kind::GgufPlain,
			Kind::Flash(8),
			Kind::EngineSpec,
		] {
			let (d, p) = knobs(k, false);
			for x in &p {
				assert!(!x.values.is_empty(), "{k:?} {}", x.name);
				assert!(get(&d, x.name).is_some(), "{k:?} {} has no default", x.name);
			}
		}
		let (_, p) = knobs(Kind::GgufMtp(3), false);
		assert_eq!(p[0].values, ["1", "2", "3"]);
	}

	#[test]
	fn today_is_a_date() {
		let t = today();
		assert_eq!(t.len(), 10);
		assert!(t.starts_with("20"));
	}
}
