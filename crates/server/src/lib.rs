//! `server`: standalone HTTP front-end for qwen.rs.
//!
//! Implements an Anthropic Messages API-compatible endpoint
//! (`POST /v1/messages` with SSE streaming), `GET /v1/models`, and
//! `GET /health`. It is intentionally **dependency-free** (only `std`): the
//! project targets a standalone Windows `.exe` and the only toolchain present
//! is a cross-compile `cargo.exe` whose access to crate registries is
//! uncertain, so we avoid pulling in `axum`/`tokio`/`serde`. A tiny HTTP/1.1
//! server, a minimal JSON reader, and a hand-rolled JSON writer cover exactly
//! the Anthropic schema we need.
//!
//! The model (and tokenizer) are loaded **once** at server startup and shared
//! across all request-handling threads behind an `Arc<Loaded>` (the model is
//! `Send + Sync`: it owns only parsed GGUF metadata plus the file `path`, and
//! reads each layer from disk on demand, so concurrent `forward` calls are
//! independent). This removes the per-request ~16 GB reload.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

pub mod images;
mod mcp;
mod openai;
pub mod profile;
mod snapdir;
mod tools;
mod tune;

pub use tune::{set_tune, TuneOpts};

use model::suffix::{
	best_draft_src, ctx_policy, suffix_enabled, suffix_ev, suffix_global_cap, suffix_min_engine,
	SuffixIndex, GRAM,
};
use model::{Checkpoint, Decoder, Model, PromptInput, RunState};
pub use prefixcache;
use prefixcache::{Cache, Class, SnapIn, Src};
pub use prefixcache::{Layout, Snap as EngineSnap};
use sampler::{sample, Rng, SampleParams};
use tokenizer::{Effort, StreamDecoder, Thinking, Tokenizer, Turn};
use vision::VisionModel;

/// `QWEN_FLASH_COUPLED=1` (a decode knob, default off; the shipped
/// Flash-Next profile turns it on for thinking requests):
/// coupled sampling on an engine that drafts, at temperature > 0. The
/// sampler draws each token by Gumbel-max with noise fixed by the request's
/// seed and the token's position ([`sampler::sample_coupled`]: the same
/// distribution as [`sampler::sample`]), and the engine's drafter picks
/// with the same noise ([`Engine::set_coupling`]), so a draft matches the
/// sample whenever the drafter's distribution is close to the model's, not
/// only when the model's most likely token is drawn.
pub fn coupled() -> bool {
	quant::knob::var("QWEN_FLASH_COUPLED").is_some_and(|v| v.trim() == "1")
}

/// An inference engine other than the GGUF [`Model`], served through the
/// same routes (`Backend::Engine`): the OpenVINO export
/// (`crates/cli/src/ovchat.rs`), the NPU engine (`npuchat.rs`). It holds one
/// conversation's state; the server resets it at the start of every request
/// and runs one request at a time on it. Text only, unless its
/// [`EngineInfo::image`] says it takes images ([`Engine::feed_images`]).
pub trait Engine: Send {
	/// Start a new conversation: forget every token fed so far.
	fn reset(&mut self) -> Result<(), String>;
	/// Feed `ids` after everything fed since the last reset and return the
	/// last position's logits, one per tokenizer id.
	fn forward(&mut self, ids: &[u32]) -> Result<Vec<f32>, String>;
	/// Feed `ids` when no logits are needed (prompt pieces before a prefix
	/// cache checkpoint).
	fn feed(&mut self, ids: &[u32]) -> Result<(), String> {
		self.forward(ids).map(|_| ())
	}
	/// A copy of the state after everything fed so far, for the prefix
	/// cache; `None` when the engine cannot copy its state (no cache).
	fn snapshot(&mut self) -> Option<EngineSnap> {
		None
	}
	/// Whether [`Engine::snapshot`] gives a state, when the engine knows
	/// without taking one (`None`: the server takes one to see; on
	/// Flash-Next that is ~113 MB of state copied out at startup).
	fn can_snapshot(&self) -> Option<bool> {
		None
	}
	/// The tokens of one prefill piece when a feed is cut into pieces
	/// that each pay a fixed cost whatever their length (Flash-Next: 4096,
	/// ~2.7 s a piece at 12k-17k depth); 0 = no such cost. The server then
	/// leaves out an interval checkpoint that would cost a piece more.
	fn piece_len(&self) -> usize {
		0
	}
	/// The cost of a [`Engine::verify`] step of 1, 2, ... positions with its
	/// drafts (any unit; the last repeats). With it the server picks context
	/// drafts by expected tokens per cost ([`model::suffix::CtxPolicy`]);
	/// `None`: by match length alone.
	fn step_costs(&self) -> Option<Vec<f32>> {
		None
	}
	/// Replace the state with one from [`Engine::snapshot`].
	fn restore(&mut self, _snap: &EngineSnap) -> Result<(), String> {
		Err("this engine cannot restore a snapshot".into())
	}
	/// Speculative decoding: the most tokens a [`Engine::verify`] step
	/// takes (the sampled token plus drafts); 0 when the engine does not
	/// draft.
	fn spec_k(&self) -> usize {
		0
	}
	/// Feed `ids` (at most `spec_k`) tentatively and return the logits of
	/// every position.
	fn verify(&mut self, _ids: &[u32]) -> Result<Vec<Vec<f32>>, String> {
		Err("this engine does not verify drafts".into())
	}
	/// Keep the first `n` tokens of the last `verify`; with `next` (the token
	/// sampled after them, not fed yet) return the drafts to follow it.
	fn accept(&mut self, _n: usize, _next: Option<u32>) -> Result<Vec<u32>, String> {
		Err("this engine does not verify drafts".into())
	}
	/// The sampler's shared noise for this request's drafts ([`coupled`]);
	/// an engine that ignores it drafts as before.
	fn set_coupling(&mut self, _c: Option<sampler::Coupling>) {}
	/// [`Engine::feed`] for tokens with images: `images` are (index of the
	/// first `<|image_pad|>` of its run in `ids`, pixels), in order; each
	/// run is as long as [`ImageGeom::grid`] says. The engine encodes them.
	fn feed_images(
		&mut self,
		_ids: &[u32],
		_images: &[(usize, &imgio::Image)],
	) -> Result<(), String> {
		Err("this engine takes no images".into())
	}
}

/// What the server needs to know of an engine's vision tower to lay an
/// image out in the prompt (the engine itself encodes it).
#[derive(Clone, Copy, Debug)]
pub struct ImageGeom {
	/// Pixels of a patch side, patches of a token's side.
	pub patch: usize,
	pub merge: usize,
	/// Cap on tokens per image.
	pub max_tokens: usize,
}

impl ImageGeom {
	/// The grid an image of `width x height` pixels is resized to.
	pub fn grid(&self, width: usize, height: usize) -> Result<vision::Grid, String> {
		vision::smart_resize(
			width,
			height,
			self.patch,
			self.merge,
			vision::DEFAULT_MIN_TOKENS,
			self.max_tokens.max(1),
		)
	}
}

/// An [`Engine`]'s state ([`EngineSnap`], `prefixcache::Snap`) is opaque
/// parts the engine alone interprets, plus a layout hint per part: a KV
/// part marked [`Layout::Rows`] is paged by the prefix cache and shared
/// between checkpoints. An engine may also hand the same chunk to several
/// snapshots (the Flash-Next KV cache's sealed segments); the cache counts
/// it once.
/// What the routes need to know about an [`Engine`] without taking its lock
/// (`/props` must answer while a generation runs).
#[derive(Clone, Debug)]
pub struct EngineInfo {
	/// Shown in the startup line.
	pub name: String,
	/// The model id the REST API reports (`/v1/models`, `/props`, the
	/// `model` of a response that names none).
	pub model: String,
	/// Context in tokens: prompt plus generation.
	pub ctx: usize,
	pub ctx_train: usize,
	pub n_vocab: usize,
	pub n_embd: usize,
	/// Tokens that end a turn.
	pub stop: Vec<u32>,
	/// Identifies the weights a persisted snapshot belongs to
	/// (`--prompt-cache-dir`).
	pub tag: String,
	/// The engine takes image blocks.
	pub image: Option<ImageGeom>,
}

/// What runs the forward pass.
#[allow(clippy::large_enum_variant)] // one per server
pub enum Backend {
	/// The GGUF model, with everything the server builds on it: prefix cache,
	/// context drafts, MTP / DFlash, images.
	Gguf(Model),
	/// Another engine: a prefix cache when the engine can snapshot its
	/// state (`Loaded::cache`), drafts and images when it has them.
	/// Each request holds the lock from its prefill to its last token.
	Engine {
		eng: Mutex<Box<dyn Engine>>,
		info: EngineInfo,
	},
}

/// Everything the server needs for inference, loaded once at startup.
pub struct Loaded {
	pub backend: Backend,
	pub tok: Tokenizer,
	/// Vision tower, present when the server was started with `--mmproj`.
	/// Without it an `image` content block is a 400, not a silent drop.
	pub vision: Option<VisionModel>,
	/// Cap on merged tokens per image.
	pub max_image_tokens: usize,
	/// States kept across requests, keyed on the prompt prefix they have
	/// consumed (`crates/prefixcache`; the GGUF model's through
	/// [`gguf_snap`] / [`gguf_restore`]). Disabled until `serve` sets it.
	pub cache: Cache,
	/// Earlier responses, for context drafts across requests
	/// (`model::suffix`). A request decodes against a snapshot; finished
	/// responses are added copy-on-write.
	pub responses: Mutex<Arc<SuffixIndex>>,
	/// The last request's numbers (`--tune-profile` reads them).
	pub(crate) last: Mutex<Option<GenStats>>,
}

/// What one request cost, for [`tune`].
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GenStats {
	pub input_tokens: usize,
	pub reused: usize,
	pub prefill_s: f64,
	pub output_tokens: usize,
	pub decode_s: f64,
	/// Verify / forward passes of the decode.
	pub steps: usize,
}

impl Loaded {
	/// Load the model weights and tokenizer from a GGUF file. `n_gpu_layers`
	/// (when > 0) initializes the Vulkan offload manager for the last N layers.
	pub fn load(
		path: &str,
		mmproj: Option<&str>,
		n_gpu_layers: usize,
		ctx: usize,
		n_batch: usize,
		max_image_tokens: usize,
		mtp: usize,
		dflash: Option<&str>,
	) -> Result<Loaded, String> {
		let mut model = Model::load(path).map_err(|e| format!("model load failed: {e}"))?;
		model.set_ctx(ctx);
		model.set_n_batch(n_batch);
		if mtp > 0 {
			model.enable_mtp(mtp)?;
			eprintln!("server: MTP speculative decoding, {mtp} draft token(s) per step");
		}
		if let Some(p) = dflash {
			model.enable_dflash(p, model::dflash_n_draft())?;
			eprintln!("server: DFlash2 speculative decoding");
		}
		model.warm();
		// Same condition as the CLI: with prefill streaming the manager has to
		// exist even at `--n-gpu-layers 0`, because a streamed weight needs no
		// layer to be resident.
		if n_gpu_layers > 0 || model::gpu_prefill_requested() {
			model.init_offload(n_gpu_layers);
		}
		let gguf = gguf::parse_file(path).map_err(|e| format!("gguf parse failed: {e}"))?;
		let tok = Tokenizer::from_gguf(&gguf).map_err(|e| format!("tokenizer: {e}"))?;
		let vision = match mmproj {
			Some(p) => Some(VisionModel::load(p, Some(model.hp.n_embd))?),
			None => None,
		};
		Ok(Loaded {
			backend: Backend::Gguf(model),
			tok,
			vision,
			max_image_tokens,
			cache: Cache::off(),
			responses: Mutex::default(),
			last: Mutex::default(),
		})
	}

	/// Serve `eng` instead of a GGUF model (see [`Engine`]).
	pub fn from_engine(eng: Box<dyn Engine>, info: EngineInfo, tok: Tokenizer) -> Loaded {
		Loaded {
			backend: Backend::Engine {
				eng: Mutex::new(eng),
				info,
			},
			tok,
			vision: None,
			max_image_tokens: 0,
			cache: Cache::off(),
			responses: Mutex::default(),
			last: Mutex::default(),
		}
	}

	/// Context in tokens: prompt plus generation.
	pub fn ctx(&self) -> usize {
		match &self.backend {
			Backend::Gguf(m) => m.ctx(),
			Backend::Engine { info, .. } => info.ctx,
		}
	}

	/// The model id the REST API reports.
	pub fn model_id(&self) -> &str {
		match &self.backend {
			Backend::Gguf(_) => DEFAULT_MODEL,
			Backend::Engine { info, .. } => &info.model,
		}
	}

	/// `(n_ctx_train, n_vocab, n_embd)`, for llama-server's model `meta`.
	pub fn dims(&self) -> (usize, usize, usize) {
		match &self.backend {
			Backend::Gguf(m) => (m.hp.context_length, m.hp.n_vocab, m.hp.n_embd),
			Backend::Engine { info, .. } => (info.ctx_train, info.n_vocab, info.n_embd),
		}
	}
}

// ===========================================================================
// Minimal JSON
// ===========================================================================

#[derive(Debug, Clone)]
pub enum Json {
	Null,
	Bool(bool),
	Num(f64),
	Str(String),
	Arr(Vec<Json>),
	Obj(Vec<(String, Json)>),
}

impl Json {
	pub fn parse(b: &[u8]) -> Result<Json, String> {
		Jp::new(b).parse()
	}

	pub fn as_str(&self) -> Option<&str> {
		if let Json::Str(s) = self {
			Some(s)
		} else {
			None
		}
	}
	pub fn as_f64(&self) -> Option<f64> {
		if let Json::Num(n) = self {
			Some(*n)
		} else {
			None
		}
	}
	pub fn as_bool(&self) -> Option<bool> {
		if let Json::Bool(b) = self {
			Some(*b)
		} else {
			None
		}
	}
	pub fn as_array(&self) -> Option<&[Json]> {
		if let Json::Arr(a) = self {
			Some(a)
		} else {
			None
		}
	}
	pub fn get(&self, key: &str) -> Option<&Json> {
		if let Json::Obj(o) = self {
			for (k, v) in o {
				if k == key {
					return Some(v);
				}
			}
		}
		None
	}

	/// Serialize back to JSON text.
	///
	/// Needed in two places tool use added: a tool definition is rendered into
	/// the prompt verbatim (the chat template's `tool | tojson`), and a parsed
	/// `tool_use.input` goes back out in the response. Object key order is
	/// preserved from the parse, which keeps the prompt byte-stable for a given
	/// request: worth having when the prefix is what the KV cache keys on.
	#[allow(clippy::inherent_to_string)] // no `Display`: `{}` would hide which form
	pub fn to_string(&self) -> String {
		let mut o = String::new();
		self.write(&mut o, false);
		o
	}

	/// Same, with `", "` / `": "` separators instead of bare commas and colons.
	///
	/// This is the one that renders tool definitions into the prompt. Python's
	/// `json.dumps` defaults to those separators and that is what `tojson`
	/// produces in the chat template the model was trained against, so the
	/// compact form would put every tool schema in front of the model in a
	/// spacing it never saw. Cheap to match; not obviously free to ignore.
	pub fn to_string_spaced(&self) -> String {
		let mut o = String::new();
		self.write(&mut o, true);
		o
	}

	fn write(&self, o: &mut String, sep: bool) {
		match self {
			Json::Null => o.push_str("null"),
			Json::Bool(b) => o.push_str(if *b { "true" } else { "false" }),
			Json::Num(n) => o.push_str(&fnum(*n)),
			Json::Str(s) => o.push_str(&jstr(s)),
			Json::Arr(a) => {
				o.push('[');
				for (i, v) in a.iter().enumerate() {
					if i > 0 {
						o.push_str(if sep { ", " } else { "," });
					}
					v.write(o, sep);
				}
				o.push(']');
			}
			Json::Obj(m) => {
				o.push('{');
				for (i, (k, v)) in m.iter().enumerate() {
					if i > 0 {
						o.push_str(if sep { ", " } else { "," });
					}
					o.push_str(&jstr(k));
					o.push_str(if sep { ": " } else { ":" });
					v.write(o, sep);
				}
				o.push('}');
			}
		}
	}
}

/// Render a number the way JSON expects: integral values without a `.0` tail,
/// which matters because these land in a *prompt*; a schema that said
/// `"maxLength": 100.0` would be describing something subtly different to the
/// model than `"maxLength": 100`. Non-finite values are not representable in
/// JSON and become `null`.
fn fnum(n: f64) -> String {
	if !n.is_finite() {
		return "null".to_string();
	}
	if n == n.trunc() && n.abs() < 1e15 {
		return format!("{}", n as i64);
	}
	format!("{n}")
}

struct Jp<'a> {
	b: &'a [u8],
	i: usize,
}

impl<'a> Jp<'a> {
	fn new(b: &'a [u8]) -> Self {
		Jp { b, i: 0 }
	}

	fn parse(&mut self) -> Result<Json, String> {
		self.value()
	}

	fn skip_ws(&mut self) {
		while self.i < self.b.len() {
			match self.b[self.i] {
				b' ' | b'\t' | b'\n' | b'\r' => self.i += 1,
				_ => break,
			}
		}
	}

	fn value(&mut self) -> Result<Json, String> {
		self.skip_ws();
		if self.i >= self.b.len() {
			return Err("unexpected end of input".into());
		}
		match self.b[self.i] {
			b'{' => self.object(),
			b'[' => self.array(),
			b'"' => Ok(Json::Str(self.string()?)),
			b't' | b'f' => self.boolean(),
			b'n' => self.null(),
			b'-' | b'0'..=b'9' => self.number(),
			_ => Err(format!("unexpected byte {}", self.b[self.i] as char)),
		}
	}

	fn object(&mut self) -> Result<Json, String> {
		self.i += 1; // '{'
		let mut out = Vec::new();
		self.skip_ws();
		if self.i < self.b.len() && self.b[self.i] == b'}' {
			self.i += 1;
			return Ok(Json::Obj(out));
		}
		loop {
			self.skip_ws();
			if self.i >= self.b.len() || self.b[self.i] != b'"' {
				return Err("expected object key".into());
			}
			let key = self.string()?;
			self.skip_ws();
			if self.i >= self.b.len() || self.b[self.i] != b':' {
				return Err("expected ':'".into());
			}
			self.i += 1;
			let val = self.value()?;
			out.push((key, val));
			self.skip_ws();
			if self.i >= self.b.len() {
				return Err("unexpected eof in object".into());
			}
			match self.b[self.i] {
				b',' => self.i += 1,
				b'}' => {
					self.i += 1;
					break;
				}
				_ => return Err("expected ',' or '}'".into()),
			}
		}
		Ok(Json::Obj(out))
	}

	fn array(&mut self) -> Result<Json, String> {
		self.i += 1; // '['
		let mut out = Vec::new();
		self.skip_ws();
		if self.i < self.b.len() && self.b[self.i] == b']' {
			self.i += 1;
			return Ok(Json::Arr(out));
		}
		loop {
			let v = self.value()?;
			out.push(v);
			self.skip_ws();
			if self.i >= self.b.len() {
				return Err("unexpected eof in array".into());
			}
			match self.b[self.i] {
				b',' => self.i += 1,
				b']' => {
					self.i += 1;
					break;
				}
				_ => return Err("expected ',' or ']'".into()),
			}
		}
		Ok(Json::Arr(out))
	}

	fn boolean(&mut self) -> Result<Json, String> {
		if self.b[self.i..].starts_with(b"true") {
			self.i += 4;
			Ok(Json::Bool(true))
		} else if self.b[self.i..].starts_with(b"false") {
			self.i += 5;
			Ok(Json::Bool(false))
		} else {
			Err("invalid literal".into())
		}
	}

	fn null(&mut self) -> Result<Json, String> {
		if self.b[self.i..].starts_with(b"null") {
			self.i += 4;
			Ok(Json::Null)
		} else {
			Err("invalid literal".into())
		}
	}

	fn number(&mut self) -> Result<Json, String> {
		let start = self.i;
		while self.i < self.b.len() {
			let c = self.b[self.i];
			if c == b'-' || c == b'+' || c == b'.' || c == b'e' || c == b'E' || c.is_ascii_digit() {
				self.i += 1;
			} else {
				break;
			}
		}
		let s =
			std::str::from_utf8(&self.b[start..self.i]).map_err(|_| "number utf8".to_string())?;
		s.parse::<f64>()
			.map(Json::Num)
			.map_err(|_| format!("invalid number {s}"))
	}

	fn string(&mut self) -> Result<String, String> {
		self.i += 1; // opening quote
		let mut out = String::new();
		while self.i < self.b.len() {
			let c = self.b[self.i];
			if c == b'"' {
				self.i += 1;
				return Ok(out);
			}
			if c == b'\\' {
				self.i += 1;
				if self.i >= self.b.len() {
					return Err("bad escape".into());
				}
				let e = self.b[self.i];
				self.i += 1;
				match e {
					b'"' => out.push('"'),
					b'\\' => out.push('\\'),
					b'/' => out.push('/'),
					b'b' => out.push('\u{0008}'),
					b'f' => out.push('\u{000C}'),
					b'n' => out.push('\n'),
					b'r' => out.push('\r'),
					b't' => out.push('\t'),
					b'u' => {
						let cp = self.hex4()?;
						if (0xD800..=0xDBFF).contains(&cp) {
							if self.i + 1 < self.b.len()
								&& self.b[self.i] == b'\\'
								&& self.b[self.i + 1] == b'u'
							{
								self.i += 2;
								let lo = self.hex4()?;
								if (0xDC00..=0xDFFF).contains(&lo) {
									let c = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
									match char::from_u32(c) {
										Some(ch) => out.push(ch),
										None => return Err("bad codepoint".into()),
									}
								} else {
									return Err("bad low surrogate".into());
								}
							} else {
								return Err("expected low surrogate".into());
							}
						} else if (0xDC00..=0xDFFF).contains(&cp) {
							return Err("unexpected low surrogate".into());
						} else {
							match char::from_u32(cp) {
								Some(ch) => out.push(ch),
								None => return Err("bad codepoint".into()),
							}
						}
					}
					_ => return Err("bad escape".into()),
				}
			} else {
				// The run up to the next quote or escape, validated once.
				// (Validating the rest of the body for every character made
				// a request quadratic in its size: 13 s for a 1 MB image.)
				let run = self.b[self.i..]
					.iter()
					.position(|&c| c == b'"' || c == b'\\')
					.unwrap_or(self.b.len() - self.i);
				out.push_str(
					std::str::from_utf8(&self.b[self.i..self.i + run])
						.map_err(|_| "utf8".to_string())?,
				);
				self.i += run;
			}
		}
		Err("unterminated string".into())
	}

	fn hex4(&mut self) -> Result<u32, String> {
		if self.i + 4 > self.b.len() {
			return Err("bad unicode escape".into());
		}
		let mut v = 0u32;
		for _ in 0..4 {
			let c = self.b[self.i];
			let d = match c {
				b'0'..=b'9' => c - b'0',
				b'a'..=b'f' => c - b'a' + 10,
				b'A'..=b'F' => c - b'A' + 10,
				_ => return Err("bad hex".into()),
			};
			v = v * 16 + d as u32;
			self.i += 1;
		}
		Ok(v)
	}
}

/// Render a Rust string as a JSON double-quoted, escaped string literal.
pub(crate) fn jstr(s: &str) -> String {
	let mut o = String::with_capacity(s.len() + 2);
	o.push('"');
	for c in s.chars() {
		match c {
			'"' => o.push_str("\\\""),
			'\\' => o.push_str("\\\\"),
			'\n' => o.push_str("\\n"),
			'\r' => o.push_str("\\r"),
			'\t' => o.push_str("\\t"),
			c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
			c => o.push(c),
		}
	}
	o.push('"');
	o
}

/// Extract plain text from an Anthropic `content` value (string or blocks).
fn text_of(v: Option<&Json>) -> Option<String> {
	let v = v?;
	match v {
		Json::Str(s) => Some(s.clone()),
		Json::Arr(a) => {
			let mut out = String::new();
			for item in a {
				if let Json::Obj(o) = item {
					if let Some(txt) = o
						.iter()
						.find(|(k, _)| k == "text")
						.and_then(|(_, vv)| vv.as_str())
					{
						out.push_str(txt);
					}
				}
			}
			if out.is_empty() {
				None
			} else {
				Some(out)
			}
		}
		_ => None,
	}
}

/// One piece of a message's content, in order.
enum Block {
	Text(String),
	/// Decoded pixels of an `image` block.
	Image(imgio::Image),
	/// An `assistant` turn's `tool_use` block, replayed back into the prompt.
	ToolUse {
		name: String,
		input: Json,
	},
	/// A `user` turn's `tool_result` block: its text and image parts, in
	/// order. The id is not rendered: the model's format has no slot for it
	/// and matches results to calls positionally.
	ToolResult(Vec<Block>),
	/// An `assistant` turn's `thinking` block, replayed into its `<think>`.
	/// (`redacted_thinking` has no text to replay and is skipped.)
	Thinking(String),
}

/// Decode base64, ignoring whitespace. Anthropic image blocks carry the bytes
/// as `source.data`, so this is the only place the server needs it.
fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
	let mut out = Vec::with_capacity(s.len() / 4 * 3);
	let mut acc = 0u32;
	let mut nbits = 0u32;
	for c in s.bytes() {
		let v = match c {
			b'A'..=b'Z' => c - b'A',
			b'a'..=b'z' => c - b'a' + 26,
			b'0'..=b'9' => c - b'0' + 52,
			b'+' => 62,
			b'/' => 63,
			b'=' => break,
			b' ' | b'\n' | b'\r' | b'\t' => continue,
			_ => return Err(format!("invalid base64 character {:?}", c as char)),
		} as u32;
		acc = (acc << 6) | v;
		nbits += 6;
		if nbits >= 8 {
			nbits -= 8;
			out.push((acc >> nbits) as u8);
		}
	}
	Ok(out)
}

/// Split an Anthropic `content` value into ordered text and image blocks.
///
/// Accepts `{"type":"image","source":{"type":"base64","data":"..."}}` and the
/// `data:image/png;base64,...` URL form, because clients send both.
fn blocks_of(v: Option<&Json>) -> Result<Vec<Block>, String> {
	let v = match v {
		Some(v) => v,
		None => return Ok(Vec::new()),
	};
	match v {
		Json::Str(s) => Ok(vec![Block::Text(s.clone())]),
		Json::Arr(a) => {
			let mut out = Vec::new();
			for item in a {
				let o = match item {
					Json::Obj(o) => o,
					_ => continue,
				};
				let ty = o
					.iter()
					.find(|(k, _)| k == "type")
					.and_then(|(_, v)| v.as_str());
				match ty {
					Some("image") => {
						let src = o
							.iter()
							.find(|(k, _)| k == "source")
							.map(|(_, v)| v)
							.ok_or("image block has no source")?;
						let data = src
							.get("data")
							.and_then(|v| v.as_str())
							.or_else(|| src.get("url").and_then(|v| v.as_str()))
							.ok_or("image source has neither data nor url")?;
						// Strip a data: URL prefix if present.
						let b64 = match data.find("base64,") {
							Some(i) => &data[i + 7..],
							None => data,
						};
						if b64.starts_with("http://") || b64.starts_with("https://") {
							return Err(
								"image source url must be a data: URL; this server does not fetch remote images"
									.to_string(),
							);
						}
						let bytes = base64_decode(b64)?;
						out.push(Block::Image(imgio::decode(&bytes)?));
					}
					Some("tool_use") => {
						let name = o
							.iter()
							.find(|(k, _)| k == "name")
							.and_then(|(_, v)| v.as_str())
							.ok_or("tool_use block has no name")?
							.to_string();
						let input = o
							.iter()
							.find(|(k, _)| k == "input")
							.map(|(_, v)| v.clone())
							.unwrap_or(Json::Obj(Vec::new()));
						out.push(Block::ToolUse { name, input });
					}
					Some("tool_result") => {
						// `content` is a string, or blocks of its own: text and
						// images (an agent reading a PNG), rendered in order
						// inside the `<tool_response>` as the template does.
						let c = o.iter().find(|(k, _)| k == "content").map(|(_, v)| v);
						let mut parts: Vec<Block> = blocks_of(c)?
							.into_iter()
							.filter(|b| matches!(b, Block::Text(_) | Block::Image(_)))
							.collect();
						let is_error = o
							.iter()
							.find(|(k, _)| k == "is_error")
							.and_then(|(_, v)| v.as_bool())
							.unwrap_or(false);
						// The model has no error channel in `<tool_response>`,
						// so say it in words. Without this a failed tool reads
						// exactly like a successful one whose output happened to
						// be a stack trace.
						if is_error {
							parts.insert(0, Block::Text("Error: ".to_string()));
						}
						out.push(Block::ToolResult(parts));
					}
					Some("thinking") => {
						if let Some(t) = o
							.iter()
							.find(|(k, _)| k == "thinking")
							.and_then(|(_, v)| v.as_str())
						{
							out.push(Block::Thinking(t.to_string()));
						}
					}
					Some("redacted_thinking") => {}
					_ => {
						if let Some(t) = o
							.iter()
							.find(|(k, _)| k == "text")
							.and_then(|(_, v)| v.as_str())
						{
							out.push(Block::Text(t.to_string()));
						}
					}
				}
			}
			Ok(out)
		}
		_ => Ok(Vec::new()),
	}
}

// ===========================================================================
// HTTP helpers
// ===========================================================================

static MSG_CTR: AtomicU64 = AtomicU64::new(0);

fn new_id() -> String {
	let c = MSG_CTR.fetch_add(1, Ordering::Relaxed);
	let t = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map(|d| d.as_nanos() as u64)
		.unwrap_or(0);
	format!("msg_{:016x}{:016x}", t ^ 0x9e37_79b9_7f4a_7c15, c)
}

fn respond_json(w: &mut dyn Write, code: u16, text: &str, body: &str) -> std::io::Result<()> {
	let resp = format!(
		"HTTP/1.1 {code} {text}\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
		body.len(),
		body
	);
	w.write_all(resp.as_bytes())?;
	w.flush()
}

fn write_error(w: &mut dyn Write, code: u16, msg: &str) -> std::io::Result<()> {
	let body = format!(
		"{{\"type\":\"error\",\"error\":{{\"type\":\"invalid_request_error\",\"message\":{}}}}}",
		jstr(msg)
	);
	let text = if code == 404 {
		"Not Found"
	} else {
		"Bad Request"
	};
	respond_json(w, code, text, &body)
}

/// The head of every SSE response. Like every other response it says
/// `Connection: close`: the server answers one request per connection, and
/// a stream's handler keeps the socket open after the last chunk (the
/// prefix cache stores the request's state on `Drop`). Advertised as
/// keep-alive, a pooling client (Zed) sent its next request (the MCP tool
/// call right after `message_stop`) into that socket and got a connection
/// reset, which ends Zed's MCP session for good ("server shut down").
pub(crate) const SSE_HEAD: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n";

fn sse_begin(w: &mut dyn Write) -> std::io::Result<()> {
	w.write_all(SSE_HEAD)?;
	w.flush()
}

fn sse_chunk(w: &mut dyn Write, text: &str) -> std::io::Result<()> {
	let chunk = format!("{:x}\r\n{}\r\n", text.len(), text);
	w.write_all(chunk.as_bytes())?;
	w.flush()
}

fn sse_end(w: &mut dyn Write) -> std::io::Result<()> {
	w.write_all(b"0\r\n\r\n")?;
	w.flush()
}

// ===========================================================================
// Request handling
// ===========================================================================

/// The GGUF model's id on the REST API; an engine names its own
/// (`EngineInfo::model`).
pub const DEFAULT_MODEL: &str = "Qwen3.8-27B";

/// Why generation ended, in neither API's vocabulary.
#[derive(Clone, Copy, PartialEq, Debug)]
#[allow(clippy::enum_variant_names)] // the APIs' stop reasons
pub(crate) enum Stop {
	EndTurn,
	MaxTokens,
	StopSequence,
	ToolUse,
}

/// Token counts of one request.
pub(crate) struct Usage {
	pub input: usize,
	pub output: usize,
	/// Prompt tokens the prefix cache supplied instead of a prefill.
	pub cached: usize,
}

/// The response side of one API: how the start of a generation, its visible
/// text as it streams, and its end are written. Everything between - prompt,
/// prefill, sampling, tool-call parsing - is the same for every API, so
/// `generate` owns it and a `Wire` only formats.
pub(crate) trait Wire {
	/// Before the first token; the prompt is known, nothing has been generated.
	fn start(&mut self, w: &mut dyn Write, input_tokens: usize) -> std::io::Result<()>;
	/// A run of visible text (never empty).
	fn text(&mut self, w: &mut dyn Write, text: &str) -> std::io::Result<()>;
	/// A run of the reply's reasoning (never empty; only when the request
	/// asked for thinking).
	fn thinking(&mut self, w: &mut dyn Write, text: &str) -> std::io::Result<()>;
	/// The end: `full` is all the visible text, `thinking` all the reasoning,
	/// `calls` the parsed tool calls.
	fn finish(
		&mut self,
		w: &mut dyn Write,
		full: &str,
		thinking: &str,
		calls: &[tools::ParsedCall],
		stop: Stop,
		usage: &Usage,
	) -> std::io::Result<()>;
}

/// What a `thinking` block's `signature` says. Anthropic's is an encrypted
/// copy of the reasoning that clients send back unchanged; this server reads
/// the plain `thinking` text it is sent back with instead, so the signature
/// only has to be present (clients and SDKs expect the field).
const THINKING_SIGNATURE: &str = "qwen.rs";

/// Anthropic Messages API (`POST /v1/messages`).
struct AnthropicWire {
	stream: bool,
	id: String,
	model: String,
	/// The streamed content block that is open (`"thinking"` or `"text"`),
	/// at index `index`. A block is opened on its first character rather than
	/// up front: a turn that is nothing but a tool call should not carry an
	/// empty text block ahead of it.
	open: Option<&'static str>,
	/// The open block's index, or the next block's when none is open.
	index: usize,
}

impl AnthropicWire {
	/// Close the open block, if any; a thinking block gets its signature
	/// first, as Anthropic's do.
	fn close(&mut self, w: &mut dyn Write) -> std::io::Result<()> {
		let Some(kind) = self.open.take() else {
			return Ok(());
		};
		let i = self.index;
		if kind == "thinking" {
			let d = format!(
				"{{\"type\":\"content_block_delta\",\"index\":{i},\"delta\":{{\"type\":\"signature_delta\",\"signature\":{}}}}}",
				jstr(THINKING_SIGNATURE)
			);
			sse_chunk(
				w,
				&format!("event: content_block_delta\r\ndata: {d}\r\n\r\n"),
			)?;
		}
		sse_chunk(
			w,
			&format!(
				"event: content_block_stop\r\ndata: {{\"type\":\"content_block_stop\",\"index\":{i}}}\r\n\r\n"
			),
		)?;
		self.index += 1;
		Ok(())
	}

	/// A delta of block kind `kind` (`"thinking"` / `"text"`), opening the
	/// block (and closing another kind's) when needed.
	fn delta(&mut self, w: &mut dyn Write, kind: &'static str, text: &str) -> std::io::Result<()> {
		if self.open != Some(kind) {
			self.close(w)?;
			let block = if kind == "thinking" {
				"{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}"
			} else {
				"{\"type\":\"text\",\"text\":\"\"}"
			};
			sse_chunk(
				w,
				&format!(
					"event: content_block_start\r\ndata: {{\"type\":\"content_block_start\",\"index\":{},\"content_block\":{block}}}\r\n\r\n",
					self.index
				),
			)?;
			self.open = Some(kind);
		}
		let (ty, field) = if kind == "thinking" {
			("thinking_delta", "thinking")
		} else {
			("text_delta", "text")
		};
		let delta = format!(
			"{{\"type\":\"content_block_delta\",\"index\":{},\"delta\":{{\"type\":\"{ty}\",\"{field}\":{}}}}}",
			self.index,
			jstr(text)
		);
		sse_chunk(
			w,
			&format!("event: content_block_delta\r\ndata: {delta}\r\n\r\n"),
		)
	}
}

impl Wire for AnthropicWire {
	fn start(&mut self, w: &mut dyn Write, input_tokens: usize) -> std::io::Result<()> {
		if !self.stream {
			return Ok(());
		}
		sse_begin(w)?;
		let msg = format!(
			"{{\"type\":\"message_start\",\"message\":{{\"id\":{id},\"type\":\"message\",\"role\":\"assistant\",\"model\":{mn},\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{{\"input_tokens\":{it},\"output_tokens\":0}}}}}}",
			id = jstr(&self.id),
			mn = jstr(&self.model),
			it = input_tokens
		);
		sse_chunk(w, &format!("event: message_start\r\ndata: {msg}\r\n\r\n"))
	}

	fn text(&mut self, w: &mut dyn Write, text: &str) -> std::io::Result<()> {
		if self.stream {
			self.delta(w, "text", text)?;
		}
		Ok(())
	}

	fn thinking(&mut self, w: &mut dyn Write, text: &str) -> std::io::Result<()> {
		if self.stream {
			self.delta(w, "thinking", text)?;
		}
		Ok(())
	}

	fn finish(
		&mut self,
		w: &mut dyn Write,
		full: &str,
		thinking: &str,
		calls: &[tools::ParsedCall],
		stop: Stop,
		usage: &Usage,
	) -> std::io::Result<()> {
		let stop_reason = match stop {
			Stop::EndTurn => "end_turn",
			Stop::MaxTokens => "max_tokens",
			Stop::StopSequence => "stop_sequence",
			Stop::ToolUse => "tool_use",
		};
		let output_tokens = usage.output;
		if self.stream {
			self.close(w)?;
			let mut index = self.index;
			for (n, c) in calls.iter().enumerate() {
				let tid = tools::new_tool_id(n);
				// `input` starts empty and arrives as `input_json_delta`. The whole
				// object goes in one delta: it is only known once the closing tag
				// has been parsed, so there is nothing to stream incrementally and
				// pretending otherwise would just be smaller lies.
				let start = format!(
					"{{\"type\":\"content_block_start\",\"index\":{index},\"content_block\":{{\"type\":\"tool_use\",\"id\":{id},\"name\":{name},\"input\":{{}}}}}}",
					id = jstr(&tid),
					name = jstr(&c.name),
				);
				sse_chunk(
					w,
					&format!("event: content_block_start\r\ndata: {start}\r\n\r\n"),
				)?;
				let delta = format!(
					"{{\"type\":\"content_block_delta\",\"index\":{index},\"delta\":{{\"type\":\"input_json_delta\",\"partial_json\":{pj}}}}}",
					pj = jstr(&c.input_json()),
				);
				sse_chunk(
					w,
					&format!("event: content_block_delta\r\ndata: {delta}\r\n\r\n"),
				)?;
				sse_chunk(
					w,
					&format!(
						"event: content_block_stop\r\ndata: {{\"type\":\"content_block_stop\",\"index\":{index}}}\r\n\r\n"
					),
				)?;
				index += 1;
			}
			let md = format!(
				"{{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":{sr},\"stop_sequence\":null}},\"usage\":{{\"output_tokens\":{ot}}}}}",
				sr = jstr(stop_reason),
				ot = output_tokens
			);
			sse_chunk(w, &format!("event: message_delta\r\ndata: {md}\r\n\r\n"))?;
			sse_chunk(
				w,
				&format!(
					"event: message_stop\r\ndata: {}\r\n\r\n",
					"{\"type\":\"message_stop\"}"
				),
			)?;
			sse_end(w)?;
		} else {
			let mut blocks: Vec<String> = Vec::new();
			if !thinking.is_empty() {
				blocks.push(format!(
					"{{\"type\":\"thinking\",\"thinking\":{},\"signature\":{}}}",
					jstr(thinking),
					jstr(THINKING_SIGNATURE)
				));
			}
			if !full.is_empty() || (calls.is_empty() && thinking.is_empty()) {
				// An empty `content` array is not a valid message, so a turn that
				// produced nothing at all still gets one empty text block.
				blocks.push(format!("{{\"type\":\"text\",\"text\":{}}}", jstr(full)));
			}
			for (n, c) in calls.iter().enumerate() {
				blocks.push(tools::tool_use_block(&tools::new_tool_id(n), c));
			}
			let body = format!(
				"{{\"id\":{id},\"type\":\"message\",\"role\":\"assistant\",\"model\":{mn},\"content\":[{ct}],\"stop_reason\":{sr},\"stop_sequence\":null,\"usage\":{{\"input_tokens\":{it},\"output_tokens\":{ot}}}}}",
				id = jstr(&self.id),
				mn = jstr(&self.model),
				ct = blocks.join(","),
				sr = jstr(stop_reason),
				it = usage.input,
				ot = output_tokens
			);
			respond_json(w, 200, "OK", &body)?;
		}
		Ok(())
	}
}

/// The thinking one request asks for.
pub(crate) struct ThinkReq {
	pub think: Thinking,
	/// `thinking.budget_tokens`: reasoning tokens after which the server
	/// closes the think block for the model.
	pub budget: Option<usize>,
}

/// The server's thinking for requests that do not say (`cli --serve --think
/// LEVEL`, `--no-preserve-thinking`): off by default, preserving.
static DEFAULT_THINKING: Mutex<Thinking> = Mutex::new(Thinking::OFF);

/// Whether [`set_default_thinking`] was called (`--think` /
/// `--no-preserve-thinking` given). Without it the llama-server dialect
/// (`openai.rs`) defaults thinking on, as llama-server does.
static DEFAULT_THINKING_SET: std::sync::atomic::AtomicBool =
	std::sync::atomic::AtomicBool::new(false);

/// Set the server's thinking default, before [`serve`].
pub fn set_default_thinking(t: Thinking) {
	*DEFAULT_THINKING.lock().unwrap_or_else(|e| e.into_inner()) = t;
	DEFAULT_THINKING_SET.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// See [`DEFAULT_THINKING_SET`].
pub(crate) fn default_thinking_set() -> bool {
	DEFAULT_THINKING_SET.load(std::sync::atomic::Ordering::Relaxed)
}

/// The profile requests take their defaults from (`profile.rs`).
static PROFILE: Mutex<Option<Arc<profile::Profile>>> = Mutex::new(None);

/// Set the server's profile, before [`serve`] (else the built-in one).
pub fn set_profile(p: profile::Profile) {
	*PROFILE.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(p));
}

/// The server's profile.
pub fn current_profile() -> Arc<profile::Profile> {
	PROFILE
		.lock()
		.unwrap_or_else(|e| e.into_inner())
		.get_or_insert_with(Default::default)
		.clone()
}

/// The thinking switches of a request (Messages shape; `openai.rs`
/// translates into it), on top of the server default.
///
/// Thinking is OFF by default, which is what this model's own template does
/// for `enable_thinking: false`. The reason is `max_tokens`: reasoning counts
/// against it, so with thinking on a caller who asks for 40 tokens spends all
/// 40 inside the think block and receives an empty message - measured, not
/// hypothesized. It also costs ~40 % more tokens per tool call (43 against
/// 27), which at this machine's decode rate is seconds per turn. (An earlier
/// revision defaulted this ON and blamed non-thinking mode for garbled tool
/// calls. That was wrong: the garbling was a hardcoded repetition penalty.)
///
/// * `thinking: {"type": "enabled" | "adaptive" | "disabled"}`, with
///   `budget_tokens` (enabled) enforced as a cap on the reasoning.
/// * The effort: `output_config.effort` (Anthropic: `low`, `medium`, `high`
///   / `xhigh` / `max` = the template's `xhigh`), or `reasoning_effort`
///   (top level or in `chat_template_kwargs`, the template's names). Only
///   with thinking on, as in the template; the template's default is
///   `xhigh`. An unknown level is an error, as in the template.
/// * `preserve_thinking` (top level or in `chat_template_kwargs`): replay
///   the reasoning of turns before the last user message (default on).
pub(crate) fn thinking_from_request(req: &Json) -> Result<ThinkReq, String> {
	let def = *DEFAULT_THINKING.lock().unwrap_or_else(|e| e.into_inner());
	let kw = req.get("chat_template_kwargs");
	let t = req.get("thinking");
	let enabled = match t.and_then(|t| t.get("type")).and_then(|v| v.as_str()) {
		Some("enabled" | "adaptive") => true,
		Some("disabled") => false,
		Some(o) => return Err(format!("unknown thinking type {o:?}")),
		None => def.enabled(),
	};
	let level = req
		.get("output_config")
		.and_then(|o| o.get("effort"))
		.or_else(|| req.get("reasoning_effort"))
		.or_else(|| kw.and_then(|k| k.get("reasoning_effort")))
		.and_then(|v| v.as_str());
	let effort =
		match level {
			Some("high" | "max") => Effort::Xhigh,
			Some(l) => Effort::parse(l).ok_or_else(|| {
				format!("unexpected reasoning effort {l:?}: supported are low, medium, high / xhigh / max")
			})?,
			None => def.effort.unwrap_or(Effort::Xhigh),
		};
	let preserve = req
		.get("preserve_thinking")
		.or_else(|| kw.and_then(|k| k.get("preserve_thinking")))
		.and_then(|v| v.as_bool())
		.unwrap_or(def.preserve);
	let budget = t
		.and_then(|t| t.get("budget_tokens"))
		.and_then(|v| v.as_f64())
		.map(|n| n.max(0.0) as usize);
	Ok(ThinkReq {
		think: Thinking {
			effort: enabled.then_some(effort),
			preserve,
		},
		budget,
	})
}

/// Answer one request through `wire`: everything between the parsed request
/// (Anthropic-shaped; other APIs translate into it) and the response format.
/// `default_max_tokens` applies when the request sets none; either way the
/// generation is capped at what is left of the context.
fn generate(
	w: &mut dyn Write,
	loaded: &Loaded,
	req: &Json,
	wire: &mut dyn Wire,
	default_max_tokens: usize,
) -> std::io::Result<()> {
	let max_tokens = req
		.get("max_tokens")
		.and_then(|v| v.as_f64())
		.map(|n| n as usize)
		.unwrap_or(default_max_tokens);
	let stop_seqs: Vec<String> = req
		.get("stop_sequences")
		.and_then(|v| v.as_array())
		.map(|a| {
			a.iter()
				.filter_map(|x| x.as_str().map(|s| s.to_string()))
				.collect()
		})
		.unwrap_or_default();

	let tok = &loaded.tok;

	// What the caller declared. `tool_choice: none` means "you may not call
	// these", which is the same prompt as having none at all.
	let tool_specs = tools::specs_from_request(req);
	let choice = tools::choice_from_request(req);
	let tool_json: Vec<String> = if choice == tools::Choice::None {
		Vec::new()
	} else {
		tool_specs.iter().map(|t| t.json.clone()).collect()
	};

	// Assemble conversation turns. Image blocks are replaced by their
	// placeholder run, sized from the image's grid; tool blocks become the
	// turns the chat template expects. The images are only *encoded* (the
	// tower: seconds each) once the prefix cache has said which of them it
	// already holds; an agent session resends every screenshot every turn.
	let mut images: Vec<(imgio::Image, vision::Grid)> = Vec::new();
	let turns = turns_from_request(req, &mut |img| {
		let grid =
			match (&loaded.vision, &loaded.backend) {
				(Some(vm), _) => vm.grid_for(img.width, img.height, loaded.max_image_tokens),
				(
					None,
					Backend::Engine {
						info: EngineInfo { image: Some(g), .. },
						..
					},
				) => g.grid(img.width, img.height),
				(None, Backend::Engine { .. }) => {
					return Err(
						"this engine has no vision tower, so it cannot accept image blocks".into(),
					)
				}
				(None, Backend::Gguf(_)) => return Err(
					"this server was started without --mmproj, so it cannot accept image blocks"
						.into(),
				),
			}
			.map_err(|e| format!("image encode failed: {e}"))?;
		let ph = vision::placeholder(grid.tokens());
		images.push((img, grid));
		Ok(ph)
	});
	let turns = match turns {
		Ok(t) => t,
		Err(e) => return write_error(w, 400, &e),
	};
	let has_system = turns
		.iter()
		.any(|t| matches!(t, Turn::System(s) if !s.trim().is_empty()));

	// A forced `tool_choice` is honored by opening the call for the model
	// rather than by asking it to comply; see `tools::forced_prefix`. The prefix
	// is part of the prompt, so the splitter has to see it too or the
	// `<function=` it already contains would never be recognized.
	let forced = tools::forced_prefix(&choice);
	let think = match thinking_from_request(req) {
		Ok(t) => t,
		Err(e) => return write_error(w, 400, &e),
	};
	// A forced call is the reply's first text, so it cannot follow an open
	// think block; Anthropic refuses the combination, this server thinks
	// less instead.
	let think = if forced.is_empty() {
		think
	} else {
		ThinkReq {
			think: Thinking {
				effort: None,
				..think.think
			},
			budget: None,
		}
	};
	let open = format!("{}{forced}", think.think.open());
	let prompt = Tokenizer::build_prompt_with(&turns, &tool_json, think.think, &open);
	let mut ids = encode_prompt(tok, &prompt);
	let input_tokens = ids.len();
	// Generation stops at the end of the context, whatever was asked for; a
	// prompt that does not leave room for one token is refused up front, as
	// llama.cpp does, rather than overrunning the KV reservation.
	let ctx = loaded.ctx();
	if input_tokens >= ctx {
		return write_error(
			w,
			400,
			&format!("the prompt is {input_tokens} tokens, which does not fit the context of {ctx} tokens (--ctx)"),
		);
	}
	let max_tokens = max_tokens.min(ctx - input_tokens);
	// `QWEN_SERVER_DEBUG=1`: the exact prompt and the raw completion, on stderr.
	// Everything about tool use is a text protocol between the template and the
	// parser, and when a call does not come back there is no way to tell which
	// side is wrong without seeing both.
	let debug = std::env::var("QWEN_SERVER_DEBUG").is_ok();
	if debug {
		eprintln!(
			"--- PROMPT ({input_tokens} tok) ---
{prompt}
--- END PROMPT ---"
		);
	}

	// Sampling: the profile's for this mode and prompt length (built in: the
	// model cards' thinking / non-thinking sets), under what the request
	// sets. The penalties apply to the generated tokens only. The Messages
	// API has no repetition-penalty knob, and one hardcoded at 1.1 was
	// actively harmful to tool use: a call is built from repeated tokens by
	// construction (`<parameter=` and `</parameter>` once per argument).
	// `repetition_penalty`, `presence_penalty`, `frequency_penalty` and
	// `min_p` in the request are non-standard escape hatches.
	let prof = current_profile().resolve(think.think.enabled(), input_tokens);
	let mut params = prof.params;
	for k in profile::SAMPLING_KEYS {
		if let Some(v) = req.get(k).and_then(|v| v.as_f64()) {
			let _ = profile::set_param(&mut params, k, &v.to_string());
		}
	}
	let asked = |k: &str| req.get(k).is_some();
	if asked("temperature") && params.temperature <= 0.0 {
		profile::greedy_penalties(&mut params, &asked);
	}
	// the profile's decode knobs (draft counts, cut-offs; `quant::knob`)
	// for everything this request runs on this thread
	let _knobs = quant::knob::Scope::new(&prof.knobs);
	if debug {
		eprintln!("--- SAMPLING {params:?} knobs {:?}", prof.knobs);
	}
	// Seed per request, not from a constant.
	//
	// This was `Rng::new(1234)`, which made every identical request return a
	// byte-identical reply at *any* temperature: `temperature: 1` asked twice
	// for a made-up coffee shop name returned the same name and the same
	// slogan. The sampling path was working; it was just being handed the same
	// stream every time, so nothing above temperature 0 meant anything across
	// requests. The Messages API has no `seed`, so one is derived from the
	// clock and a counter; `seed` in the request is a non-standard escape
	// hatch for reproducing a run.
	let mut rng = Rng::new(match req.get("seed").and_then(|v| v.as_f64()) {
		Some(n) => n as u64,
		None => {
			let t = std::time::SystemTime::now()
				.duration_since(std::time::UNIX_EPOCH)
				.map(|d| d.as_nanos() as u64)
				.unwrap_or(0);
			t ^ MSG_CTR
				.load(Ordering::Relaxed)
				.wrapping_mul(0x9e37_79b9_7f4a_7c15)
		}
	});
	let mut out = Output::new(tok, &forced, &stop_seqs, debug, &think);

	let model = match &loaded.backend {
		Backend::Gguf(m) => m,
		Backend::Engine { eng, info } => {
			let spans = match image_spans(tok, &ids, &images) {
				Ok(s) => s,
				Err(e) => return write_error(w, 400, &e),
			};
			let key = cache_key(&ids, &spans, &images);
			let images = spans
				.iter()
				.zip(images)
				.map(|(s, (img, _))| (s.0, s.1, img))
				.collect();
			let job = EngineJob {
				ids,
				key,
				images,
				input_tokens,
				max_tokens,
				params,
				rng,
				tool_specs,
				tool_json,
				has_system,
				think: think.think,
			};
			return generate_engine(w, loaded, eng, info, job, out, wire);
		}
	};

	// Incremental forward (M4): prefill the prompt once to build the run
	// state, then step one token at a time reusing the KV cache / SSM state.
	//
	// With the prefix cache, "the prompt" is only what the cached state has
	// not consumed yet, and the prefill is split at the checkpoint (the last
	// `<|im_start|>`, i.e. just before the assistant header: see
	// `prefix.rs` for why there). The cache is keyed on the token ids with
	// each image's placeholder run replaced by a hash of its pixels
	// (`cache_key`): the placeholder ids are the same for every image, so on
	// their own a match would say nothing about what the model saw.
	let spans = match image_spans(tok, &ids, &images) {
		Ok(s) => s,
		Err(e) => return write_error(w, 400, &e),
	};
	let key = cache_key(&ids, &spans, &images);
	let cache = &loaded.cache;
	let cacheable = cache.enabled() && prefix_cacheable();
	let diverging = if cacheable {
		cache.diverging(&key)
	} else {
		None
	};
	let mut reused = 0usize;
	let mut start_rs = None;
	if let Some(h) = if cacheable { cache.lookup(&key) } else { None } {
		// (a cached state ends at a turn boundary or a checkpoint outside
		// every image; if one ever did not, its embeddings could not be
		// spliced, so start cold)
		if !spans.iter().any(|&(s, n)| s < h.len && h.len < s + n) {
			match gguf_restore(model, &h.snap) {
				Ok(r) => {
					start_rs = Some(r);
					reused = h.len;
				}
				Err(e) => eprintln!("prefix cache: restore failed ({e}); prefilling from scratch"),
			}
		}
	}
	let mut slot = CacheSlot {
		cache,
		model,
		rs: Some(start_rs.unwrap_or_else(|| model.new_run_state())),
		tokens: Vec::new(),
		cp: None,
		res: None,
	};

	// Encode only the images the cached state has not consumed.
	let t_img = std::time::Instant::now();
	let todo: Vec<usize> = (0..spans.len()).filter(|&i| spans[i].0 >= reused).collect();
	let mut encoded: Vec<(usize, usize, Vec<f32>)> = Vec::with_capacity(todo.len());
	for &i in &todo {
		// `images` is non-empty only when the vision model is loaded.
		let vm = loaded.vision.as_ref().unwrap();
		match vm.encode(&images[i].0, loaded.max_image_tokens) {
			Ok((emb, grid)) => encoded.push((grid.mw, grid.mh, emb)),
			Err(e) => return write_error(w, 400, &format!("image encode failed: {e}")),
		}
	}
	let img_s = t_img.elapsed().as_secs_f64();
	let rs = slot.rs.as_mut().unwrap();
	// The new tokens' MRoPE positions continue from the state's clock, which
	// a cached image has moved away from `rs.pos`.
	let mm = if encoded.is_empty() {
		None
	} else {
		let start_pos = rs.mpos.iter().copied().max().unwrap_or(0);
		match vision::build_raw(tok, ids[reused..].to_vec(), encoded, start_pos) {
			Ok(m) => Some(m),
			Err(e) => return write_error(w, 400, &e),
		}
	};

	let mut output_tokens = 0usize;
	let mut stop = Stop::MaxTokens;
	wire.start(w, input_tokens)?;

	let t_prefill = std::time::Instant::now();
	let logits = {
		let start = t_prefill;
		let mut scratch = Vec::new();
		let input = match &mm {
			Some(m) => m.input(&mut scratch, model.hp.n_embd),
			None => PromptInput::text(&ids[reused..]),
		};
		// `ids[a..b]` as a prefill input.
		let part = |a: usize, b: usize| input.slice(a - reused, b - reused);
		let ck = if cacheable {
			Some(turn_cut(tok, &ids).max(reused)).filter(|&c| c > 0)
		} else {
			None
		};
		let mut from = reused;
		let bar = model::progress::Prefill::start(ids.len() - reused);
		bar.summarized();
		let mut stored: Vec<(usize, Class)> = Vec::new();
		// `ids[from..to]` through the model, timed for the cache's cost model
		let run = |rs: &mut RunState, a: usize, b: usize| {
			let t = std::time::Instant::now();
			let r = model.forward_prefill_mm(rs, part(a, b), None);
			if cacheable {
				cache.observe_prefill(a, b, t.elapsed().as_secs_f64());
			}
			r
		};
		if let Some(c) = ck {
			// Shared prefixes, a branch point and interval checkpoints on the
			// way to the turn's checkpoint, each snapshotted if nobody has.
			for (s, class) in plan_cuts(
				tok,
				cache,
				&tool_json,
				think.think,
				has_system,
				&ids,
				diverging.as_ref(),
				reused,
				c,
				&spans,
				0,
			) {
				run(rs, from, s);
				from = s;
				if !cache.contains(&key[..s]) {
					cache.insert(key[..s].to_vec(), gguf_snap(model, rs), class);
					stored.push((s, class));
				}
			}
			if c > from {
				run(rs, from, c);
				from = c;
			}
			if !cache.contains(&key[..c]) {
				slot.cp = Some(rs.checkpoint());
				slot.tokens = key[..c].to_vec();
			}
		}
		let logits = run(rs, from, ids.len()).0;
		bar.done();
		drop(bar);
		let imgs = if spans.is_empty() {
			String::new()
		} else {
			format!(
				", encoded {} of {} images in {img_s:.1} s",
				todo.len(),
				spans.len()
			)
		};
		if model::progress::enabled() {
			let cached = cacheable.then(|| cache.summary());
			model::progress::line(&prefill_line(
				input_tokens - reused,
				input_tokens,
				reused,
				cached,
				&stored,
				start.elapsed().as_secs_f64(),
				&imgs,
			));
		} else {
			for (s, class) in &stored {
				eprintln!(
					"prefix cache: {} state of {s} tokens stored",
					class_name(*class)
				);
			}
			if cacheable {
				eprintln!(
					"prefix cache: reused {reused} of {input_tokens} prompt tokens, prefilled {} in {:.1} s{imgs} ({})",
					input_tokens - reused,
					start.elapsed().as_secs_f64(),
					cache.summary(),
				);
			}
		}
		if cacheable {
			log_divergence(tok, &key, reused, diverging);
		}
		logits
	};
	let prefill_s = t_prefill.elapsed().as_secs_f64();

	// Drafts and verifies several positions per pass when the server runs
	// with `--mtp`; sampling is unchanged, one token at a time.
	let global = loaded
		.responses
		.lock()
		.unwrap_or_else(|e| e.into_inner())
		.clone();
	let mut dec = Decoder::with_context(logits, &ids).with_global(global);
	let t_decode = std::time::Instant::now();
	let mut bar = model::progress::Decode::start(model::progress::Show::Stderr);
	bar.summarized();
	'gen: for _ in 0..max_tokens {
		let next = match out.forced() {
			Some(t) => t,
			None => sample(dec.logits(), &ids[input_tokens..], &params, &mut rng),
		};
		if next == tok.eos() {
			stop = Stop::EndTurn;
			break;
		}
		ids.push(next);
		output_tokens += 1;
		bar.tokens(1);
		if out.token(w, wire, next)? {
			stop = Stop::StopSequence;
			break 'gen;
		}

		dec.feed(model, rs, next);
	}
	dec.finish(model, rs);
	dec.release_global();
	drop(bar);
	slot.reserve();
	{
		let mut g = loaded.responses.lock().unwrap_or_else(|e| e.into_inner());
		Arc::make_mut(&mut g).push_doc(&ids[input_tokens..], suffix_global_cap());
	}
	let ms = t_decode.elapsed().as_secs_f64() * 1e3;
	if model::progress::enabled() {
		let stats = if dec.stats.steps > 0 {
			format!("; {}", dec.stats.summary())
		} else {
			String::new()
		};
		model::progress::line(&decode_line(output_tokens, ms, &stats));
	} else {
		eprintln!(
			"decode: {output_tokens} tok in {ms:.0} ms ({:.2} tok/s)",
			output_tokens as f64 * 1e3 / ms.max(1e-9)
		);
		if dec.stats.steps > 0 {
			eprintln!("{}", dec.stats.summary());
		}
	}
	*loaded.last.lock().unwrap_or_else(|e| e.into_inner()) = Some(GenStats {
		input_tokens,
		reused,
		prefill_s,
		output_tokens,
		decode_s: ms / 1e3,
		steps: if dec.stats.steps > 0 {
			dec.stats.steps
		} else {
			output_tokens
		},
	});
	let usage = Usage {
		input: input_tokens,
		output: output_tokens,
		cached: reused,
	};
	out.finish(w, wire, &tool_specs, stop, &usage)
}

/// The generated tokens' way to the client, the same for every backend:
/// text decoding, the tool-call splitter, stop sequences.
struct Output<'a> {
	tok: &'a Tokenizer,
	/// Reassembles multi-byte characters across tokens; emitting each
	/// token's lossy decode would send U+FFFD for every byte of an emoji.
	sd: StreamDecoder,
	/// Separates prose from `<tool_call>` blocks as they stream. Seeded with
	/// the forced prefix so a `tool_choice`-opened call is parsed like any
	/// other.
	sp: tools::OutputSplitter,
	gate: LeadingSpace,
	stop_seqs: &'a [String],
	debug: bool,
	raw_out: String,
	/// All the visible text so far.
	full: String,
	/// Whether the request asked for thinking: the reply starts inside the
	/// prompt's open `<think>`, and its reasoning goes to the client. Without
	/// it a stray think block is dropped.
	think: bool,
	/// All the reasoning so far.
	full_think: String,
	/// `thinking.budget_tokens`: tokens inside the think block after which
	/// it is closed for the model ([`Output::forced`]).
	budget: Option<usize>,
	think_tokens: usize,
	/// Tokens fed in place of samples: the think block's closing, once the
	/// budget is spent.
	queue: std::collections::VecDeque<u32>,
	closed: bool,
}

impl<'a> Output<'a> {
	fn new(
		tok: &'a Tokenizer,
		forced: &str,
		stop_seqs: &'a [String],
		debug: bool,
		think: &ThinkReq,
	) -> Output<'a> {
		let on = think.think.enabled();
		let mut sp = if on {
			tools::OutputSplitter::thinking()
		} else {
			tools::OutputSplitter::new()
		};
		if !forced.is_empty() {
			let _ = sp.push(forced);
		}
		Output {
			tok,
			sd: StreamDecoder::new(),
			sp,
			gate: LeadingSpace::default(),
			stop_seqs,
			debug,
			raw_out: String::new(),
			full: String::new(),
			think: on,
			full_think: String::new(),
			budget: think.budget.filter(|_| on),
			think_tokens: 0,
			queue: Default::default(),
			closed: false,
		}
	}

	/// The next token to feed instead of sampling one: the think block's
	/// end (`\n</think>\n\n`, the template's spelling) once the reasoning
	/// has used up `budget_tokens`. The model then answers as it would after
	/// closing the block itself.
	fn forced(&mut self) -> Option<u32> {
		if !self.closed && self.sp.in_think() && self.budget.is_some_and(|b| self.think_tokens >= b)
		{
			self.closed = true;
			self.queue.extend(self.tok.encode("\n</think>\n\n"));
		}
		self.queue.pop_front()
	}

	/// One generated token; true when a stop sequence has matched.
	fn token(
		&mut self,
		w: &mut dyn Write,
		wire: &mut dyn Wire,
		next: u32,
	) -> std::io::Result<bool> {
		if self.sp.in_think() {
			self.think_tokens += 1;
		}
		let piece = self.sd.push(self.tok, next, true);
		if self.debug {
			self.raw_out.push_str(&piece);
		}
		let vis = self.gate.push(&self.sp.push(&piece));
		self.reasoning(w, wire)?;
		if !vis.is_empty() {
			self.full.push_str(&vis);
			wire.text(w, &vis)?;
		}
		// Stop sequences match the *visible* text: a client's sequence should
		// not be triggered by the markup of a tool call it never sees.
		Ok(self
			.stop_seqs
			.iter()
			.any(|st| self.full.ends_with(st.as_str())))
	}

	/// Flush the tail, parse the tool calls and write the response's end.
	fn finish(
		mut self,
		w: &mut dyn Write,
		wire: &mut dyn Wire,
		tool_specs: &[tools::ToolSpec],
		mut stop: Stop,
		usage: &Usage,
	) -> std::io::Result<()> {
		let flushed = self.sd.flush();
		if self.debug {
			self.raw_out.push_str(&flushed);
		}
		let tail = self.gate.push(&self.sp.push(&flushed));
		let tail = format!("{tail}{}", self.gate.push(&self.sp.finish()));
		self.reasoning(w, wire)?;
		if !tail.is_empty() {
			self.full.push_str(&tail);
			wire.text(w, &tail)?;
		}

		if self.debug {
			eprintln!(
				"--- RAW OUT ---
{}
--- END RAW ({} calls) ---",
				self.raw_out,
				self.sp.calls.len()
			);
			for (n, c) in self.sp.calls.iter().enumerate() {
				eprintln!(
					"--- CALL {n} ---
{c}
--- END CALL ---"
				);
			}
		}

		// Turn the captured blocks into `tool_use`. A block the parser cannot
		// make a call out of is dropped: the alternative is a `tool_use` with no
		// name, which the client would try to execute.
		let calls: Vec<tools::ParsedCall> = self
			.sp
			.calls
			.iter()
			.filter_map(|raw| tools::parse_call(raw, tool_specs))
			.collect();
		if !calls.is_empty() {
			stop = Stop::ToolUse;
		}
		wire.finish(w, &self.full, &self.full_think, &calls, stop, usage)
	}

	/// Pass on the reasoning the splitter has released, when asked for.
	fn reasoning(&mut self, w: &mut dyn Write, wire: &mut dyn Wire) -> std::io::Result<()> {
		let t = self.sp.take_think();
		if self.think && !t.is_empty() {
			self.full_think.push_str(&t);
			wire.thinking(w, &t)?;
		}
		Ok(())
	}
}

/// A request's inputs past the prompt, for [`generate_engine`].
struct EngineJob {
	ids: Vec<u32>,
	/// The prefix cache's key for `ids` (`cache_key`: an image's run is a
	/// hash of its pixels).
	key: Vec<u32>,
	/// (start, tokens, pixels) of each image's placeholder run in `ids`.
	images: Vec<(usize, usize, imgio::Image)>,
	input_tokens: usize,
	max_tokens: usize,
	params: SampleParams,
	rng: Rng,
	tool_specs: Vec<tools::ToolSpec>,
	/// For the shared-prefix cuts (`shared_cuts`).
	tool_json: Vec<String>,
	has_system: bool,
	think: Thinking,
}

/// `generate` on a `Backend::Engine`: the engine is reset, restored from the
/// longest cached prefix when it can snapshot its state (`crates/prefixcache`: the
/// same checkpoints and shared prefixes as the GGUF path, on
/// `Loaded::cache`), the rest of the prompt prefilled, then one token
/// per `forward` (or a verify step with the engine's drafts). Images are fed
/// where their runs fall ([`Engine::feed_images`]); a cached prefix's are not
/// encoded again.
fn generate_engine(
	w: &mut dyn Write,
	loaded: &Loaded,
	eng: &Mutex<Box<dyn Engine>>,
	info: &EngineInfo,
	job: EngineJob,
	mut out: Output,
	wire: &mut dyn Wire,
) -> std::io::Result<()> {
	let EngineJob {
		mut ids,
		key,
		images,
		input_tokens,
		max_tokens,
		params,
		mut rng,
		tool_specs,
		tool_json,
		has_system,
		think,
	} = job;
	// One conversation at a time: the engine's state is the conversation.
	let t_wait = std::time::Instant::now();
	let mut eng = eng.lock().unwrap_or_else(|e| e.into_inner());
	let waited = t_wait.elapsed().as_secs_f64();
	if waited > 0.5 {
		eprintln!(
			"{}: waited {waited:.1} s for the previous request",
			info.name
		);
	}
	// Prefill before the response starts, so a failure is still an HTTP error.
	let t = std::time::Instant::now();
	let m_prefill = quant::cputime::Meter::start();
	let _ = (quant::parallel::stats(), quant::parallel::stats_sites());
	let cache = &loaded.cache;
	let cacheable = cache.enabled();
	let diverging = if cacheable {
		cache.diverging(&key)
	} else {
		None
	};
	let mut reused = 0usize;
	// `--progress`: the request's log in one prefill and one decode line
	let compact = model::progress::enabled();
	let mut stored: Vec<(usize, Class)> = Vec::new();
	let spec_k = eng.spec_k();
	let spans: Vec<(usize, usize)> = images.iter().map(|i| (i.0, i.1)).collect();
	let mut n_encoded = 0usize;
	// tokens `a..b`, with the images whose runs they hold
	let mut feed = |eng: &mut Box<dyn Engine>, a: usize, b: usize| -> Result<(), String> {
		let imgs: Vec<(usize, &imgio::Image)> = images
			.iter()
			.filter(|i| i.0 >= a && i.0 + i.1 <= b)
			.map(|i| (i.0 - a, &i.2))
			.collect();
		if imgs.is_empty() {
			return eng.feed(&ids[a..b]);
		}
		n_encoded += imgs.len();
		eng.feed_images(&ids[a..b], &imgs)
	};
	// `QWEN_NPU_TIMING=1`: one line per request with the time of each step
	// of the prefill below (what the `prefilled ... tok/s` line sums)
	let phases = std::cell::RefCell::new(Vec::<(String, f64)>::new());
	let lap = std::cell::Cell::new(std::time::Instant::now());
	let mark = |what: String| {
		let now = std::time::Instant::now();
		phases
			.borrow_mut()
			.push((what, (now - lap.replace(now)).as_secs_f64()));
	};
	// The engine's state after the previous request (its prompt and reply),
	// when this prompt extends it: continued as is, no reset or restore,
	// and the reply is not prefilled again. Taken: a failure leaves none.
	let live = LIVE
		.lock()
		.unwrap_or_else(|e| e.into_inner())
		.take()
		.filter(|l| {
			live_on()
				&& (spec_k > 1 || cacheable)
				&& images.is_empty()
				&& l.len() < ids.len()
				&& ids.starts_with(l)
		});
	let r = (|| -> Result<Vec<Vec<f32>>, String> {
		if let Some(l) = &live {
			reused = l.len();
			mark(format!("live {reused}"));
		} else {
			eng.reset()?;
			mark("reset".into());
		}
		if let Some(h) = if cacheable && live.is_none() {
			cache.lookup(&key)
		} else {
			None
		} {
			// (a cached state ends outside every image)
			let split = spans.iter().any(|&(s, n)| s < h.len && h.len < s + n);
			match if split {
				Err("it ends inside an image".to_string())
			} else {
				eng.restore(&h.snap)
			} {
				Ok(()) => reused = h.len,
				Err(e) => {
					eprintln!(
						"{}: prefix cache restore failed ({e}); prefilling from scratch",
						info.name
					);
					eng.reset()?;
				}
			}
			mark(format!(
				"restore {reused}{}",
				if h.from_disk { " (disk)" } else { "" }
			));
		}
		let mut from = reused;
		// (timed from `t`, as the `prefilled ... tok/s` line: reset and restore
		// included)
		let bar = model::progress::Prefill::start_at(ids.len() - reused, t);
		bar.summarized();
		// `ids[a..b]` through the engine, timed for the cache's cost model
		let mut run = |eng: &mut Box<dyn Engine>, a: usize, b: usize| -> Result<(), String> {
			let t = std::time::Instant::now();
			feed(eng, a, b)?;
			if cacheable {
				cache.observe_prefill(a, b, t.elapsed().as_secs_f64());
			}
			mark(format!("feed {a}..{b}"));
			Ok(())
		};
		if cacheable {
			// Checkpoint at the last `<|im_start|>` (see `crates/prefixcache`),
			// shared prefixes, a branch point and interval checkpoints on the
			// way there.
			let c = turn_cut(&loaded.tok, &ids).max(reused);
			let piece = eng.piece_len();
			if c > 0 {
				for (s, class) in plan_cuts(
					&loaded.tok,
					cache,
					&tool_json,
					think,
					has_system,
					&ids,
					diverging.as_ref(),
					reused,
					c,
					&spans,
					piece,
				) {
					run(&mut eng, from, s)?;
					from = s;
					if !cache.contains(&key[..s]) {
						if let Some(snap) = eng.snapshot() {
							cache.insert(key[..s].to_vec(), snap.into(), class);
							stored.push((s, class));
						}
						mark(format!("snapshot {s} + cache"));
					}
				}
				if c > from {
					run(&mut eng, from, c)?;
					from = c;
				}
				// (a hit is not consumed: every checkpoint stays for branches
				// until the cache needs its room)
				if !cache.contains(&key[..c]) {
					if let Some(snap) = eng.snapshot() {
						mark(format!("snapshot {c}"));
						cache.insert(key[..c].to_vec(), snap.into(), Class::Turn);
						mark("cache put".into());
					}
				}
			}
		}
		let rows = if spec_k > 1 {
			// the last prompt token through a verify step: the drafts that
			// follow come from its accept
			feed(&mut eng, from, ids.len() - 1)?;
			mark(format!("feed {from}..{}", ids.len() - 1));
			let r = eng.verify(&ids[ids.len() - 1..]);
			mark("verify".into());
			r
		} else if images.iter().any(|i| i.0 >= from) {
			// (the last prompt token is the assistant header's: no image's)
			feed(&mut eng, from, ids.len() - 1)?;
			eng.forward(&ids[ids.len() - 1..]).map(|l| vec![l])
		} else {
			eng.forward(&ids[from..]).map(|l| vec![l])
		}?;
		bar.done();
		Ok(rows)
	})();
	if std::env::var_os("QWEN_NPU_TIMING").is_some() {
		let ph = phases.borrow();
		eprintln!(
			"{}: request prefill steps: {}",
			info.name,
			ph.iter()
				.map(|(w, t)| format!("{w} {:.0} ms", t * 1e3))
				.collect::<Vec<_>>()
				.join(", ")
		);
	}
	let mut rows = match r {
		Ok(l) => l,
		Err(e) => {
			eprintln!("{}: prefill failed: {e}", info.name);
			return write_error(w, 500, &format!("prefill failed: {e}"));
		}
	};
	let s = t.elapsed().as_secs_f64();
	let (_, prefill_cpu) = m_prefill.read();
	let prefill_pool = (quant::parallel::stats(), quant::parallel::stats_sites());
	let fed = input_tokens - reused;
	let imgs = if images.is_empty() {
		String::new()
	} else {
		format!(", encoded {n_encoded} of {} images", images.len())
	};
	if compact {
		let cached = cacheable.then(|| cache.summary());
		model::progress::line(&prefill_line(
			fed,
			input_tokens,
			reused,
			cached,
			&stored,
			s,
			&imgs,
		));
	} else {
		for (s, class) in &stored {
			eprintln!(
				"prefix cache: {} state of {s} tokens stored",
				class_name(*class)
			);
		}
		eprintln!(
			"{}: prefilled {fed} tok in {s:.1} s ({:.2} tok/s){imgs}",
			info.name,
			fed as f64 / s.max(1e-9)
		);
		if cacheable {
			eprintln!(
				"prefix cache: reused {reused} of {input_tokens} prompt tokens ({})",
				cache.summary()
			);
		}
	}
	if cacheable {
		log_divergence(&loaded.tok, &key, reused, diverging);
	}
	wire.start(w, input_tokens)?;

	let mut output_tokens = 0usize;
	let mut stop = Stop::MaxTokens;
	let t_decode = std::time::Instant::now();
	let m_decode = quant::cputime::Meter::start();
	// the tokens behind `rows`: row j's logits follow fed[j]; a sample that
	// equals fed[j + 1] (a draft) is accepted and row j + 1 continues
	let mut fed: Vec<u32> = ids[ids.len() - 1..].to_vec();
	let mut steps = 0usize;
	// context drafts (`model::suffix`) in place of the engine's own on a
	// long match: this conversation and earlier responses
	let global = loaded
		.responses
		.lock()
		.unwrap_or_else(|e| e.into_inner())
		.clone();
	let mut ix = (spec_k > 1 && suffix_enabled()).then(|| SuffixIndex::new(&ids));
	let mut ctx_steps = 0usize;
	// `QWEN_NPU_TIMING`: per draft source (engine, context): drafts verified,
	// accepted, steps with drafts, steps whose first draft was accepted
	let mut dstat = [[0usize; 4]; 2];
	let mut fed_src = 0usize;
	let (mut fed_m, mut fed_glob) = (0usize, false);
	let mut ev_costs: Option<Vec<f32>> = None;
	ctx_policy()
		.lock()
		.unwrap_or_else(|e| e.into_inner())
		.begin();
	let draft_log = std::env::var("QWEN_DRAFT_LOG").is_ok_and(|v| v == "1");
	// Mid-stream there is no status line left to fail with: drop the
	// connection, which the client sees as a truncated response.
	let fail = |e: String| {
		eprintln!("{}: decode failed: {e}", info.name);
		std::io::Error::other(e)
	};
	// coupled sampling: the drafts share the sampler's noise
	let coupling =
		(spec_k > 1 && params.temperature > 0.0 && coupled()).then(|| sampler::Coupling {
			seed: rng.next_u64(),
			params,
		});
	eng.set_coupling(coupling);
	let mut bar = model::progress::Decode::start(model::progress::Show::Stderr);
	bar.summarized();
	// `QWEN_NPU_TIMING`: decode's host time (sampling, the response
	// stream, context drafts) against the engine's (accept, verify)
	let timing = std::env::var_os("QWEN_NPU_TIMING").is_some();
	let (mut t_host, mut t_ctx, mut t_accept, mut t_verify) = (0f64, 0f64, 0f64, 0f64);
	let mut t_first: Option<f64> = None;
	let mut lap = std::time::Instant::now();
	let mut tick = |acc: &mut f64| {
		if timing {
			let now = std::time::Instant::now();
			*acc += (now - lap).as_secs_f64();
			lap = now;
		}
	};
	// the rows of the last step used and the tokens the engine then holds:
	// the state left for the next request ([`LIVE`])
	let (mut last_n, mut live_len) = (0usize, 0usize);
	'decode: while output_tokens < max_tokens {
		let mut next = 0;
		let mut n = 0;
		// (fed[0] is ids[base - 1]: the engine holds ids[..base - 1] and fed)
		let base = ids.len();
		for (j, logits) in rows.iter().enumerate() {
			(last_n, live_len) = (j + 1, base + j);
			next = match (out.forced(), &coupling) {
				(Some(t), _) => t,
				(None, Some(c)) => sampler::sample_coupled(
					logits,
					&ids[input_tokens..],
					&params,
					c,
					ids.len() as u64,
				),
				(None, None) => sample(logits, &ids[input_tokens..], &params, &mut rng),
			};
			n = j + 1;
			if info.stop.contains(&next) {
				stop = Stop::EndTurn;
				break 'decode;
			}
			ids.push(next);
			if let Some(ix) = ix.as_mut() {
				ix.push(next);
			}
			output_tokens += 1;
			bar.tokens(1);
			let stopped = out.token(w, wire, next)?;
			if t_first.is_none() {
				t_first = Some(t_decode.elapsed().as_secs_f64());
			}
			if stopped {
				stop = Stop::StopSequence;
				break 'decode;
			}
			if output_tokens == max_tokens {
				break 'decode;
			}
			if fed.get(j + 1) != Some(&next) {
				break;
			}
		}
		tick(&mut t_host);
		steps += 1;
		// the drafts of the step whose rows these were: rows 0..n were used,
		// so drafts 1..n were accepted
		if fed.len() > 1 && draft_log {
			eprintln!(
				"draftlog src={} m={fed_m} len={} acc={} pos={}",
				["mtp", "ctx", "glob"][fed_src + fed_glob as usize],
				fed.len() - 1,
				n - 1,
				ids.len()
			);
		}
		if let (Some(c), 0) = (ev_costs.as_deref(), fed_src) {
			ctx_policy()
				.lock()
				.unwrap_or_else(|e| e.into_inner())
				.model_step(fed.len() - 1, n - 1, c);
		}
		if fed.len() > 1 {
			let d = &mut dstat[fed_src];
			d[0] += fed.len() - 1;
			d[1] += n - 1;
			d[2] += 1;
			d[3] += (n > 1) as usize;
		}
		if spec_k > 1 {
			bar.step();
			let room = (spec_k - 1).min(info.ctx.saturating_sub(ids.len()));
			ev_costs = eng.step_costs().filter(|_| suffix_ev());
			let min = if ev_costs.is_some() {
				GRAM
			} else {
				suffix_min_engine()
			};
			let ctxd = ix
				.as_ref()
				.map(|ix| best_draft_src(ix, Some(&global), room))
				.filter(|(d, m, _)| *m >= min && !d.is_empty());
			// by expected tokens per step cost against the MTP's recent rate
			let ctxd = match (ev_costs.as_deref(), ix.as_ref()) {
				(Some(c), Some(ix)) => {
					let mut p = ctx_policy().lock().unwrap_or_else(|e| e.into_inner());
					p.observe(ix.tail(ix.len()));
					ctxd.and_then(|(d, m, g)| {
						p.choose(ix.len(), &d, m, c)
							.map(|k| (d[..k].to_vec(), m, g))
					})
				}
				_ => ctxd,
			};
			tick(&mut t_ctx);
			let drafts = match ctxd {
				Some((d, m, g)) => {
					(fed_m, fed_glob) = (m, g);
					ctx_steps += 1;
					fed_src = 1;
					eng.accept(n, None).map_err(fail)?;
					d
				}
				None => {
					(fed_m, fed_glob) = (0, false);
					fed_src = 0;
					eng.accept(n, Some(next)).map_err(fail)?
				}
			};
			tick(&mut t_accept);
			fed = std::iter::once(next).chain(drafts).take(spec_k).collect();
			rows = eng.verify(&fed).map_err(fail)?;
			tick(&mut t_verify);
		} else {
			fed = vec![next];
			rows = vec![eng.forward(&fed).map_err(fail)?];
			tick(&mut t_verify);
		}
	}
	tick(&mut t_host);
	drop(bar);
	let ms = t_decode.elapsed().as_secs_f64() * 1e3;
	let (_, decode_cpu) = m_decode.read();
	let decode_pool = (quant::parallel::stats(), quant::parallel::stats_sites());
	let t_post = std::time::Instant::now();
	// in place: `make_mut` would copy the whole index while `global` lives
	drop(global);
	if spec_k > 1 {
		let mut g = loaded.responses.lock().unwrap_or_else(|e| e.into_inner());
		Arc::make_mut(&mut g).push_doc(&ids[input_tokens..], suffix_global_cap());
	}
	let per_step = if spec_k > 1 {
		format!(
			", {:.2} tok/step, {ctx_steps} of {steps} steps context drafts",
			output_tokens as f64 / steps.max(1) as f64
		)
	} else {
		String::new()
	};
	if model::progress::enabled() {
		model::progress::line(&decode_line(output_tokens, ms, &per_step));
	} else {
		eprintln!(
			"decode: {output_tokens} tok in {ms:.0} ms ({:.2} tok/s{per_step})",
			output_tokens as f64 * 1e3 / ms.max(1e-9)
		);
	}
	*loaded.last.lock().unwrap_or_else(|e| e.into_inner()) = Some(GenStats {
		input_tokens,
		reused,
		prefill_s: s,
		output_tokens,
		decode_s: ms / 1e3,
		steps,
	});
	let usage = Usage {
		input: input_tokens,
		output: output_tokens,
		cached: reused,
	};
	let r = out.finish(w, wire, &tool_specs, stop, &usage);
	// keep the rows the last step used (the rest was fed tentatively): the
	// engine then holds ids[..live_len], which the next request continues
	// when its prompt extends them
	if live_on() && (spec_k > 1 || cacheable) && images.is_empty() && last_n > 0 && r.is_ok() {
		let ok = spec_k <= 1 || eng.accept(last_n, None).is_ok();
		if ok && live_len <= ids.len() {
			*LIVE.lock().unwrap_or_else(|e| e.into_inner()) = Some(ids[..live_len].to_vec());
		}
	}
	let done = std::time::Instant::now();
	let prev = LAST_DONE
		.lock()
		.unwrap_or_else(|e| e.into_inner())
		.replace(done);
	if timing {
		let msf = |d: std::time::Duration| d.as_secs_f64() * 1e3;
		let (read, parse) = REQ_CLOCK.with(|c| c.get()).map_or((0.0, 0.0), |(a, b)| {
			(
				msf(b.saturating_duration_since(a)),
				msf(t_wait.saturating_duration_since(b)),
			)
		});
		let gap = prev.map_or(String::new(), |p| {
			let start = REQ_CLOCK.with(|c| c.get()).map_or(t_wait, |c| c.0);
			format!(
				", idle since the previous request {:.0} ms",
				msf(start.saturating_duration_since(p))
			)
		});
		eprintln!(
			"{}: request timeline: body {read:.0} ms, parse + render + tokenize {parse:.0}, \
			 lock wait {:.0}, prefill {:.0}, first token after {:.0}; decode {ms:.0} = host \
			 {:.0} (sample, stream) + context drafts {:.0} + accept {:.0} + verify {:.0} over \
			 {steps} steps; after decode {:.0}{gap}",
			info.name,
			waited * 1e3,
			s * 1e3,
			(s + t_first.unwrap_or(0.0)) * 1e3,
			t_host * 1e3,
			t_ctx * 1e3,
			t_accept * 1e3,
			t_verify * 1e3,
			msf(done - t_post),
		);
		let ds = |d: [usize; 4]| {
			format!(
				"{} of {} accepted ({:.0} %), first draft {} of {} steps",
				d[1],
				d[0],
				100.0 * d[1] as f64 / d[0].max(1) as f64,
				d[3],
				d[2]
			)
		};
		eprintln!(
			"{}: drafts: engine {}; context {}",
			info.name,
			ds(dstat[0]),
			ds(dstat[1])
		);
		eprintln!(
			"{}: [eff] prefill {}  decode {}",
			info.name,
			quant::cputime::describe(s, prefill_cpu, input_tokens - reused),
			quant::cputime::describe(ms / 1e3, decode_cpu, output_tokens)
		);
		for (what, (st, sites)) in [("prefill", &prefill_pool), ("decode", &decode_pool)] {
			if st.3 == 0 {
				continue;
			}
			eprintln!(
				"{}: [pool] {what} work {:.2} s, spin {:.2} s, join {:.2} s, {} dispatches, {} parks",
				info.name, st.0, st.1, st.2, st.3, st.4
			);
			for (site, work, wall, n) in sites.iter().take(12) {
				eprintln!(
					"{}: [pool]   {work:8.2} s shard, {wall:7.2} s wall, {:5.1} x, {n:6} calls  {site}",
					info.name,
					work / wall.max(1e-9)
				);
			}
		}
	}
	r
}

/// `POST /v1/messages`.
fn handle_messages(w: &mut dyn Write, loaded: &Loaded, req: &Json) -> std::io::Result<()> {
	let mut wire = AnthropicWire {
		stream: req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false),
		id: new_id(),
		model: req
			.get("model")
			.and_then(|v| v.as_str())
			.unwrap_or(loaded.model_id())
			.to_string(),
		open: None,
		index: 0,
	};
	// `max_tokens` is required by the Messages API; 256 covers a client that
	// leaves it out anyway.
	generate(w, loaded, req, &mut wire, 256)
}

/// The run state a request is using, stored in the prefix cache at its
/// checkpoint when the request is over: on `Drop`, so a client that
/// disconnects mid-stream (an I/O error out of `generate`) still leaves its
/// prefix behind.
struct CacheSlot<'a> {
	cache: &'a Cache,
	model: &'a Model,
	rs: Option<RunState>,
	/// The tokens the checkpoint has consumed, i.e. the cache key.
	tokens: Vec<u32>,
	/// Where to rewind to before storing; `None` means do not store.
	cp: Option<Checkpoint>,
	/// Taken before the response's last bytes go out, so the client's next
	/// request waits for this state instead of missing it.
	res: Option<prefixcache::Reservation<'a>>,
}

impl CacheSlot<'_> {
	fn reserve(&mut self) {
		if self.cp.is_some() {
			self.res = self.cache.reserve(&self.tokens);
		}
	}
}

impl Drop for CacheSlot<'_> {
	fn drop(&mut self) {
		if let (Some(mut rs), Some(cp)) = (self.rs.take(), self.cp.take()) {
			if rs.rewind(cp) {
				self.cache.insert(
					std::mem::take(&mut self.tokens),
					gguf_snap(self.model, &rs),
					Class::Turn,
				);
			}
		}
		self.res = None;
	}
}

/// A GGUF run state as the prefix cache stores it: the recurrent part as one
/// opaque blob, each KV stream as rows (paged, shared between checkpoints;
/// only the pages no other state holds are copied).
pub fn gguf_snap<'a>(model: &Model, rs: &'a RunState) -> SnapIn<'a> {
	let (blob, rows) = model.state_parts(rs);
	let mut parts = vec![(Layout::Opaque, Src::Chunks(quant::snap::part(blob)))];
	for (r, n) in rows {
		// SAFETY: plain u16 data viewed as bytes
		let b = unsafe { std::slice::from_raw_parts(r.as_ptr() as *const u8, r.len() * 2) };
		parts.push((
			Layout::Rows {
				streams: 1,
				rows: n,
			},
			Src::Bytes(b),
		));
	}
	SnapIn { pos: rs.pos, parts }
}

/// A live run state from [`gguf_snap`]'s form.
pub fn gguf_restore(model: &Model, snap: &EngineSnap) -> Result<RunState, String> {
	if snap.parts.is_empty() {
		return Err("an empty state".into());
	}
	let blob = quant::snap::concat(&snap.parts[0]);
	let rows: Vec<usize> = (1..snap.parts.len())
		.map(|i| match snap.layout(i) {
			Layout::Rows { rows, .. } => rows,
			Layout::Opaque => 0,
		})
		.collect();
	model.state_from_parts(&blob, &rows, &mut |i, out| snap.copy_part(i + 1, out))
}

/// Where a prompt's own checkpoint goes: its last `<|im_start|>`, just
/// before the assistant header; everything before it re-renders
/// byte-identically next turn, the reply after it need not (tool calls
/// re-serialized, thinking stripped), and a recurrent state cannot be cut
/// back to a divergence.
fn turn_cut(tok: &Tokenizer, ids: &[u32]) -> usize {
	tok.token_id("<|im_start|>")
		.and_then(|t| ids.iter().rposition(|&x| x == t))
		.unwrap_or(0)
}

/// The checkpoints a prefill from `reused` to the turn checkpoint `c`
/// leaves on its way, ascending: the shared prefixes (tool preamble, system
/// turn), where the prompt parts from a cached one (`diverge_cut`; a shared
/// state when inside the conversation's opening, as a new session repeating
/// an earlier one's system prompt and first message), and every
/// `QWEN_CACHE_INTERVAL` tokens. Shared ones not past an image (no
/// conversation's common opening), none inside one.
#[allow(clippy::too_many_arguments)]
fn plan_cuts(
	tok: &Tokenizer,
	cache: &Cache,
	tool_json: &[String],
	think: Thinking,
	has_system: bool,
	ids: &[u32],
	diverging: Option<&(usize, usize, Vec<u32>)>,
	reused: usize,
	c: usize,
	spans: &[(usize, usize)],
	piece: usize,
) -> Vec<(usize, Class)> {
	let first_image = spans.first().map_or(usize::MAX, |s| s.0);
	// the conversation's opening: up to the first assistant turn
	let opening = tok
		.token_id("<|im_start|>")
		.and_then(|t| ids.iter().enumerate().filter(|&(_, &x)| x == t).nth(2))
		.map_or(ids.len(), |p| p.0);
	let mut cuts: Vec<(usize, Class)> = shared_cuts(tok, tool_json, think, has_system, ids)
		.into_iter()
		.filter(|&s| s <= first_image)
		.map(|s| (s, Class::Shared))
		.collect();
	if let Some(s) = diverge_cut(diverging, reused).filter(|&s| s <= first_image) {
		cuts.push((
			s,
			if s <= opening {
				Class::Shared
			} else {
				Class::Branch
			},
		));
	}
	cuts.extend(
		cache
			.interval_cuts(reused, c)
			.into_iter()
			.map(|s| (s, Class::Interval)),
	);
	cuts.retain(|&(s, _)| s > reused && s < c && !spans.iter().any(|&(a, n)| a < s && s < a + n));
	// (at one position the weightier class: Shared < Turn < Interval < Branch)
	cuts.sort_by_key(|&(s, k)| (s, k as u8));
	cuts.dedup_by_key(|x| x.0);
	if piece > 0 {
		drop_costly_intervals(&mut cuts, reused, c, piece);
	}
	cuts
}

/// Leave out each interval cut that makes the feed from the cut before it
/// (or `reused`) to the next non-interval cut (or `c`) take more pieces of
/// `piece` tokens than without it: a follow-up turn of ~2k tokens that
/// crosses a multiple of the interval was fed as two pieces, paying the
/// engine's per-piece cost twice (Flash-Next: ~2.7 s). Its turn checkpoint
/// at `c` is stored all the same; a cold prefill's intervals line up with
/// the pieces and stay.
fn drop_costly_intervals(cuts: &mut Vec<(usize, Class)>, reused: usize, c: usize, piece: usize) {
	let np = |a: usize, b: usize| (b - a).div_ceil(piece);
	let mut prev = reused;
	let mut i = 0;
	while i < cuts.len() {
		let (s, class) = cuts[i];
		if matches!(class, Class::Interval) {
			let next = cuts[i + 1..]
				.iter()
				.find(|x| !matches!(x.1, Class::Interval))
				.map_or(c, |x| x.0);
			if np(prev, s) + np(s, next) > np(prev, next) {
				cuts.remove(i);
				continue;
			}
		}
		prev = s;
		i += 1;
	}
}

fn class_name(c: Class) -> &'static str {
	match c {
		Class::Shared => "shared",
		Class::Turn => "turn",
		Class::Interval => "interval",
		Class::Branch => "branch",
	}
}

/// The fused GPU layer (`QWEN_GPU_LAYER`) keeps its own KV and SSM state in
/// VRAM, outside `RunState`, so a state restored from the cache would attend
/// over whichever conversation ran last. It is off by default; with it on the
/// cache stays out of the way.
/// Token positions, ascending, where `ids` ends a prefix other conversations
/// are likely to share (see `prefix.rs`): the tool preamble, and the whole
/// system turn. Only those long enough to be worth a snapshot.
fn shared_cuts(
	tok: &Tokenizer,
	tool_json: &[String],
	think: Thinking,
	has_system: bool,
	ids: &[u32],
) -> Vec<usize> {
	const MIN_TOKENS: usize = 1024;
	let mut cuts = Vec::new();
	if !tool_json.is_empty() {
		// The preamble rendered on its own and compared token by token: the
		// system text that follows it can change how its last characters
		// tokenize, so the common run is where the two states agree, not the
		// rendered length. Memoized, as a client sends the same tools every
		// turn and they are ~30k tokens to encode. The reasoning effort's
		// instruction opens it, so the effort is part of the key.
		type Memo = Option<(Vec<String>, &'static str, Vec<u32>)>;
		static PRE: std::sync::Mutex<Memo> = std::sync::Mutex::new(None);
		let instr = think.instruction();
		let mut g = PRE.lock().unwrap_or_else(|e| e.into_inner());
		if g.as_ref()
			.is_none_or(|(t, i, _)| t.as_slice() != tool_json || *i != instr)
		{
			let pre = encode_prompt(
				tok,
				&Tokenizer::build_prompt_with(&[], tool_json, think, ""),
			);
			*g = Some((tool_json.to_vec(), instr, pre));
		}
		let pre = &g.as_ref().unwrap().2;
		cuts.push(pre.iter().zip(ids).take_while(|(a, b)| a == b).count());
	}
	if has_system || !tool_json.is_empty() {
		// The system turn ends where the second turn's `<|im_start|>` begins.
		if let Some(t) = tok.token_id("<|im_start|>") {
			if let Some(p) = ids.iter().enumerate().filter(|&(_, &x)| x == t).nth(1) {
				cuts.push(p.0);
			}
		}
	}
	cuts.retain(|&c| c >= MIN_TOKENS);
	cuts.sort_unstable();
	cuts.dedup();
	cuts
}

/// Where a prompt that parts from a cached one at token `n` (`d`, from
/// [`Cache::diverging`]) should be snapshotted, when that saves at
/// least 1024 tokens of prefill over `reused`: a new session whose opening
/// (system prompt, an AGENTS.md / CLAUDE.md in its first message) repeats an
/// earlier session's, or a branch off a conversation at a point no
/// checkpoint marks (an edited message). The next prompt that shares the
/// run then restores it. A few tokens short of `n`, since the tokens just
/// before a difference may tokenize differently next time.
fn diverge_cut(d: Option<&(usize, usize, Vec<u32>)>, reused: usize) -> Option<usize> {
	const MIN_TOKENS: usize = 1024;
	const BACK: usize = 16;
	let s = d?.0.checked_sub(BACK)?;
	(s >= reused + MIN_TOKENS).then_some(s)
}

/// Log why a request reused less than a cached state holds: `d` is
/// `Cache::diverging` for its `key`, taken before the request's own
/// entries went in. A client that resends the conversation with anything
/// changed before its last turn (a timestamp in the system prompt, a
/// re-encoded image, a reworded message) pays for everything from there on,
/// and without this line that looks like a cache that does not work.
fn log_divergence(
	tok: &Tokenizer,
	key: &[u32],
	reused: usize,
	d: Option<(usize, usize, Vec<u32>)>,
) {
	let Some((n, len, old)) = d else { return };
	// (a few shared tokens are another conversation's opening, not a
	// changed one: every prompt starts with `<|im_start|>`)
	if n < reused + 64 {
		return;
	}
	// (an image's run in a key is a hash of its pixels, top bit set)
	let text = |k: &[u32]| {
		if k.first().is_some_and(|&x| x >= 0x8000_0000) {
			"an image".to_string()
		} else {
			format!(
				"{:?}",
				tok.decode(
					&k[..k.iter().position(|&x| x >= 0x8000_0000).unwrap_or(k.len())],
					false
				)
			)
		}
	};
	let before = &key[n.saturating_sub(16)..n];
	let before = &before[before
		.iter()
		.rposition(|&x| x >= 0x8000_0000)
		.map_or(0, |p| p + 1)..];
	let now = &key[n..key.len().min(n + 16)];
	eprintln!(
		"prefix cache: a cached state of {len} tokens matches this prompt only up to token {n} (a state cannot be cut back to there): after {:?} it has {}, the prompt {}",
		tok.decode(before, false),
		text(&old),
		text(now),
	);
}

fn prefix_cacheable() -> bool {
	std::env::var("QWEN_GPU_LAYER").is_err()
}

/// Encode a rendered prompt. `<|image_pad|>` is a special token only inside
/// an image's placeholder (`<|vision_start|>` + pads + `<|vision_end|>`);
/// anywhere else it is text that names the token (a rules file, a tool
/// description, a quoted error message) and is encoded as its characters.
/// Left special, such a mention counted as a run of its own and the request
/// was refused for having more runs than images.
fn encode_prompt(tok: &Tokenizer, prompt: &str) -> Vec<u32> {
	const PAD: &str = "<|image_pad|>";
	let ids = tok.encode(prompt);
	let Some(pad) = tok.token_id(PAD) else {
		return ids;
	};
	if !ids.contains(&pad) {
		return ids;
	}
	let (start, end) = (
		tok.token_id("<|vision_start|>"),
		tok.token_id("<|vision_end|>"),
	);
	let plain = tok.encode_plain(PAD);
	let mut out = Vec::with_capacity(ids.len());
	let mut i = 0;
	while i < ids.len() {
		if ids[i] != pad {
			out.push(ids[i]);
			i += 1;
			continue;
		}
		let s = i;
		while i < ids.len() && ids[i] == pad {
			i += 1;
		}
		let placeholder =
			s > 0 && Some(ids[s - 1]) == start && i < ids.len() && Some(ids[i]) == end;
		if placeholder {
			out.extend_from_slice(&ids[s..i]);
		} else {
			for _ in s..i {
				out.extend_from_slice(&plain);
			}
		}
	}
	out
}

/// `(start, len)` of each image's `<|image_pad|>` run in `ids`, in prompt
/// order, checked against the grids the placeholders were sized from.
fn image_spans(
	tok: &Tokenizer,
	ids: &[u32],
	images: &[(imgio::Image, vision::Grid)],
) -> Result<Vec<(usize, usize)>, String> {
	if images.is_empty() {
		return Ok(Vec::new());
	}
	let pad = tok
		.token_id("<|image_pad|>")
		.ok_or("tokenizer: this model has no <|image_pad|> token, so it cannot take images")?;
	let mut spans = Vec::with_capacity(images.len());
	let mut i = 0;
	while i < ids.len() {
		if ids[i] != pad {
			i += 1;
			continue;
		}
		let s = i;
		while i < ids.len() && ids[i] == pad {
			i += 1;
		}
		spans.push((s, i - s));
	}
	if spans.len() != images.len()
		|| spans
			.iter()
			.zip(images)
			.any(|(s, (_, g))| s.1 != g.tokens())
	{
		return Err(format!(
			"prompt: {} images do not match the {} <|image_pad|> runs the prompt encoded to",
			images.len(),
			spans.len()
		));
	}
	Ok(spans)
}

/// The prefix-cache key for `ids`: the ids themselves, with each image's
/// placeholder run overwritten by a hash of its pixels. The placeholder ids
/// are identical for every image, so without this two conversations that
/// differ only in a screenshot would share a cache entry. The hash is spread
/// over the run as 31-bit halves with the top bit set, which no vocabulary id
/// reaches, so a key never collides with plain text either.
fn cache_key(
	ids: &[u32],
	spans: &[(usize, usize)],
	images: &[(imgio::Image, vision::Grid)],
) -> Vec<u32> {
	let mut key = ids.to_vec();
	for (&(s, n), (img, _)) in spans.iter().zip(images) {
		let h = image_hash(img);
		let halves = [
			0x8000_0000 | (h >> 33) as u32,
			0x8000_0000 | (h as u32 & 0x7fff_ffff),
		];
		for (j, k) in key[s..s + n].iter_mut().enumerate() {
			*k = halves[j % 2];
		}
	}
	key
}

/// 64-bit hash of an image's size and pixels (word-wise multiply-rotate,
/// murmur3's finalizer). Not cryptographic: it only has to tell a client's
/// screenshots apart, and the text around them is part of the key too.
fn image_hash(img: &imgio::Image) -> u64 {
	const K: u64 = 0x9e37_79b9_7f4a_7c15;
	let mut h = 0xcbf2_9ce4_8422_2325u64 ^ ((img.width as u64) << 32 | img.height as u64);
	let mut words = img.rgb.chunks_exact(8);
	for w in &mut words {
		h = (h ^ u64::from_le_bytes(w.try_into().unwrap()))
			.wrapping_mul(K)
			.rotate_left(31);
	}
	for &b in words.remainder() {
		h = (h ^ b as u64).wrapping_mul(K).rotate_left(31);
	}
	h ^= img.rgb.len() as u64;
	h ^= h >> 33;
	h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
	h ^= h >> 33;
	h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
	h ^ (h >> 33)
}

/// Turn an Anthropic request's `system` + `messages` into chat template turns.
///
/// Split out of `handle_messages` because this is where a tool-use conversation
/// is most easily got subtly wrong (a result attached to the wrong turn, or an
/// assistant turn whose call is dropped, produces a prompt the model answers
/// confidently and incorrectly) and because it is the only part of the path
/// that can be tested without loading 16 GB of weights. `on_image` encodes an
/// image block and returns its placeholder text, so this stays independent of
/// the vision tower.
fn turns_from_request(
	req: &Json,
	on_image: &mut dyn FnMut(imgio::Image) -> Result<String, String>,
) -> Result<Vec<Turn>, String> {
	let mut turns: Vec<Turn> = Vec::new();
	if let Some(s) = text_of(req.get("system")) {
		turns.push(Turn::System(s));
	}
	let Some(arr) = req.get("messages").and_then(|v| v.as_array()) else {
		return Ok(turns);
	};
	for m in arr {
		let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("user");
		let blocks =
			blocks_of(m.get("content")).map_err(|e| format!("invalid content block: {e}"))?;
		let mut text = String::new();
		let mut reasoning = String::new();
		let mut calls: Vec<tokenizer::ToolCall> = Vec::new();
		let mut results: Vec<String> = Vec::new();
		// Tool results first: the template renders them ahead of the turn's
		// own text, and `on_image` must see the images in prompt order.
		let (res_blocks, blocks): (Vec<Block>, Vec<Block>) = blocks
			.into_iter()
			.partition(|b| matches!(b, Block::ToolResult(_)));
		for b in res_blocks {
			let Block::ToolResult(parts) = b else {
				unreachable!()
			};
			let mut r = String::new();
			for p in parts {
				match p {
					Block::Text(t) => r.push_str(&t),
					Block::Image(img) => r.push_str(&on_image(img)?),
					_ => {}
				}
			}
			results.push(r);
		}
		for b in blocks {
			match b {
				Block::Text(t) => text.push_str(&t),
				Block::Thinking(t) => {
					if !reasoning.is_empty() {
						reasoning.push_str("\n\n");
					}
					reasoning.push_str(t.trim());
				}
				Block::ToolUse { name, input } => calls.push(tokenizer::ToolCall {
					name,
					args: render_args(&input),
				}),
				Block::ToolResult(_) => unreachable!(),
				Block::Image(img) => text.push_str(&on_image(img)?),
			}
		}
		// Tool results come before any prose in the same message: the template
		// renders a run of results and then the user's own turn, and a client
		// that sends both means "here is the output, and also this" in that
		// order.
		if !results.is_empty() {
			turns.push(Turn::ToolResults(results));
		}
		if role == "assistant" {
			if !text.trim().is_empty() || !calls.is_empty() || !reasoning.is_empty() {
				turns.push(Turn::Assistant {
					text,
					reasoning,
					calls,
				});
			}
		} else if !text.is_empty() {
			turns.push(Turn::User(text));
		}
	}
	Ok(turns)
}

/// Holds back whitespace until real text follows it.
///
/// The template puts a blank line between the closing think tag and the call,
/// so a turn that is *only* a tool call still emits `"\n\n"` as visible text.
/// Sent as-is that becomes a whitespace-only text block in front of every
/// `tool_use`: harmless to read, but it is content the assistant did not
/// write, and in streaming it cannot be taken back once flushed.
#[derive(Default)]
struct LeadingSpace {
	/// Whitespace seen since the last real character.
	held: String,
	started: bool,
}

impl LeadingSpace {
	fn push(&mut self, text: &str) -> String {
		if self.started {
			return text.to_string();
		}
		match text.find(|c: char| !c.is_whitespace()) {
			Some(i) => {
				// The held run is leading whitespace by construction, so it is
				// dropped rather than flushed. Everything from the first real
				// character on is passed through untouched, interior whitespace
				// included.
				self.started = true;
				self.held.clear();
				text[i..].to_string()
			}
			None => {
				self.held.push_str(text);
				String::new()
			}
		}
	}
}

/// Render a `tool_use.input` object into the `(name, text)` pairs the chat
/// template writes.
///
/// String values go in verbatim and everything else as JSON: the same split
/// the template makes, and the one `tools::coerce` reverses when the call comes
/// back. A non-object `input` has no parameters to render.
fn render_args(input: &Json) -> Vec<(String, String)> {
	match input {
		Json::Obj(m) => m
			.iter()
			.map(|(k, v)| {
				let text = match v {
					Json::Str(s) => s.clone(),
					other => other.to_string(),
				};
				(k.clone(), text)
			})
			.collect(),
		_ => Vec::new(),
	}
}

fn handle_conn(stream: TcpStream, loaded: Option<Arc<Loaded>>) {
	let _ = stream.set_nodelay(true);
	let mut buf = BufReader::new(stream);
	let mut request_line = String::new();
	if buf.read_line(&mut request_line).is_err() {
		return;
	}
	let t_arrive = std::time::Instant::now();
	let mut it = request_line.split_whitespace();
	let method = it.next().unwrap_or("");
	let path = it.next().unwrap_or("/");

	let mut headers: Vec<(String, String)> = Vec::new();
	loop {
		let mut line = String::new();
		if buf.read_line(&mut line).is_err() {
			return;
		}
		let l = line.trim_end_matches(['\r', '\n']);
		if l.is_empty() {
			break;
		}
		if let Some(p) = l.find(':') {
			headers.push((
				l[..p].trim().to_ascii_lowercase(),
				l[p + 1..].trim().to_string(),
			));
		}
	}

	let cl = headers
		.iter()
		.find(|(k, _)| k == "content-length")
		.and_then(|(_, v)| v.parse::<usize>().ok())
		.unwrap_or(0);
	let mut body = vec![0u8; cl];
	if cl > 0 && buf.read_exact(&mut body).is_err() {
		return;
	}

	let ua = headers
		.iter()
		.find(|(k, _)| k == "user-agent")
		.map_or("", |(_, v)| v.as_str());
	record_request(method, path, &headers, &body);
	REQ_CLOCK.with(|c| c.set(Some((t_arrive, std::time::Instant::now()))));
	if let Some(b) = drop_client_fields(&client_ignore(), ua, &body) {
		body = b;
	}

	request_header(method, path, ua);
	let w = buf.get_mut();
	let _ = handle_request_h(method, path, &headers, &body, loaded.as_deref(), w);
}

/// `--progress`: a request's prefill in one line, in place of the bar's
/// final line and the separate prefill / prefix cache lines: `fed` of
/// `input` prompt tokens prefilled (`reused` from the prefix cache, which
/// holds `cached`; shared prefixes `stored` on the way) in `secs`.
fn prefill_line(
	fed: usize,
	input: usize,
	reused: usize,
	cached: Option<String>,
	stored: &[(usize, Class)],
	secs: f64,
	imgs: &str,
) -> String {
	let mut cache = Vec::new();
	if reused > 0 {
		cache.push(format!("{reused} reused"));
	}
	if let Some(c) = cached {
		cache.push(c);
	}
	if !stored.is_empty() {
		let s: Vec<String> = stored
			.iter()
			.map(|(s, c)| format!("{s} {}", class_name(*c)))
			.collect();
		cache.push(format!("stored {}", s.join(", ")));
	}
	let cache = if cache.is_empty() {
		String::new()
	} else {
		format!(" ({})", cache.join("; "))
	};
	let of = if fed == input {
		format!("{input}")
	} else {
		format!("{fed} of {input}")
	};
	format!(
		"prefill {of} tok{cache} in {}, {:.1} tok/s{imgs}",
		model::progress::fmt_secs(secs, 1),
		fed as f64 / secs.max(1e-9)
	)
}

/// `--progress`: a request's decode in one line, in place of the bar's
/// final line and the `decode:` line; `more` is appended (draft stats).
fn decode_line(tokens: usize, ms: f64, more: &str) -> String {
	format!(
		"decode {tokens} tok in {}, {:.1} tok/s{more}",
		model::progress::fmt_secs(ms / 1e3, 1),
		tokens as f64 * 1e3 / ms.max(1e-9)
	)
}

thread_local! {
	/// This connection's request: when its request line and its body had
	/// arrived (`QWEN_NPU_TIMING`'s request timeline).
	static REQ_CLOCK: std::cell::Cell<Option<(std::time::Instant, std::time::Instant)>> =
		const { std::cell::Cell::new(None) };
}

/// When the last engine request finished (the timeline's gap to the next).
static LAST_DONE: Mutex<Option<std::time::Instant>> = Mutex::new(None);

/// The tokens an engine's state holds after the last request (its prompt
/// and the reply's tokens fed): a request whose prompt extends them
/// continues that state instead of restoring a checkpoint and prefilling
/// the reply again. `None` after a failure or an image request.
static LIVE: Mutex<Option<Vec<u32>>> = Mutex::new(None);

/// `QWEN_ENGINE_LIVE` (default on; `0` restores a checkpoint for every
/// request, the A/B): see [`LIVE`].
fn live_on() -> bool {
	static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
	*V.get_or_init(|| std::env::var("QWEN_ENGINE_LIVE").map_or(true, |v| v != "0"))
}

static RECORD_DIR: Mutex<Option<std::path::PathBuf>> = Mutex::new(None);

/// `--record-requests DIR`: write every generation request to `DIR` as it
/// arrives (before `--client-ignore`), one file per request
/// (`examples/replay.rs` posts them again).
pub fn set_record_requests(dir: std::path::PathBuf) -> std::io::Result<()> {
	std::fs::create_dir_all(&dir)?;
	*RECORD_DIR.lock().unwrap_or_else(|e| e.into_inner()) = Some(dir);
	Ok(())
}

/// A recorded request: `METHOD PATH`, the headers as `name: value`, an
/// empty line, the body. Named `{unix ms}-{n}.req`, so the names sort in
/// arrival order and keep the original gaps.
fn record_request(method: &str, path: &str, headers: &[(String, String)], body: &[u8]) {
	static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
	let Some(dir) = RECORD_DIR.lock().unwrap_or_else(|e| e.into_inner()).clone() else {
		return;
	};
	if method != "POST" {
		return;
	}
	let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
	let ms = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map_or(0, |d| d.as_millis());
	let mut f = format!("{method} {path}\n").into_bytes();
	for (k, v) in headers {
		// (an API key is not recorded)
		if k == "authorization" || k == "x-api-key" {
			continue;
		}
		f.extend_from_slice(format!("{k}: {v}\n").as_bytes());
	}
	f.push(b'\n');
	f.extend_from_slice(body);
	let p = dir.join(format!("{ms}-{n:05}.req"));
	if let Err(e) = std::fs::write(&p, f) {
		eprintln!("--record-requests: {}: {e}", p.display());
	}
}

/// `--progress`: one line at the start of each generation request (chat or
/// image), so requests stand apart in the log: a running number, the route
/// and the client.
fn request_header(method: &str, path: &str, ua: &str) {
	static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
	let route = path.split('?').next().unwrap_or(path);
	let gen = matches!(
		route,
		"/v1/messages"
			| "/v1/chat/completions"
			| "/chat/completions"
			| "/v1/images/generations"
			| "/v1/images/edits"
	);
	if method != "POST" || !gen || !model::progress::enabled() {
		return;
	}
	let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
	let ua = if ua.is_empty() { "no user agent" } else { ua };
	model::progress::line(&format!("--- #{n} {route} ({ua})"));
}

/// One `--client-ignore` rule: requests whose `User-Agent` starts with
/// `prefix` (`*`: every client) lose these top-level fields.
#[derive(Clone, Debug, PartialEq)]
pub struct ClientIgnore {
	pub prefix: String,
	pub fields: Vec<String>,
}

impl ClientIgnore {
	/// `PREFIX=FIELD[,FIELD...]`.
	pub fn parse(spec: &str) -> Result<ClientIgnore, String> {
		let (p, f) = spec.split_once('=').ok_or_else(|| {
			format!("--client-ignore wants PREFIX=FIELD[,FIELD...], got {spec:?}")
		})?;
		let fields: Vec<String> = f
			.split(',')
			.map(|s| s.trim().to_string())
			.filter(|s| !s.is_empty())
			.collect();
		if p.is_empty() || fields.is_empty() {
			return Err(format!(
				"--client-ignore wants PREFIX=FIELD[,FIELD...], got {spec:?}"
			));
		}
		Ok(ClientIgnore {
			prefix: p.to_string(),
			fields,
		})
	}

	/// The default rules: every client's `temperature` is dropped, so the
	/// profile's for the request's mode and depth applies. Clients send their
	/// own fixed value whatever the mode: Zed's agent `temperature: 1.0` on
	/// every request, which overrode the model card's 0.7 for non-thinking
	/// requests.
	pub fn defaults() -> Vec<ClientIgnore> {
		vec![ClientIgnore {
			prefix: "*".into(),
			fields: vec!["temperature".into()],
		}]
	}
}

static CLIENT_IGNORE: Mutex<Option<Vec<ClientIgnore>>> = Mutex::new(None);

/// Set the `--client-ignore` rules, before [`serve`] (else
/// [`ClientIgnore::defaults`]; an empty list turns them off).
pub fn set_client_ignore(rules: Vec<ClientIgnore>) {
	*CLIENT_IGNORE.lock().unwrap_or_else(|e| e.into_inner()) = Some(rules);
}

fn client_ignore() -> Vec<ClientIgnore> {
	CLIENT_IGNORE
		.lock()
		.unwrap_or_else(|e| e.into_inner())
		.get_or_insert_with(ClientIgnore::defaults)
		.clone()
}

/// Remove the fields the matching rules name from a request body, so the
/// server's defaults (the profile's sampling for the request's mode and
/// depth, ...) apply instead of a client's fixed value. `None`: the body is
/// used as it is.
fn drop_client_fields(rules: &[ClientIgnore], user_agent: &str, body: &[u8]) -> Option<Vec<u8>> {
	let drop: Vec<&str> = rules
		.iter()
		.filter(|r| r.prefix == "*" || user_agent.starts_with(&r.prefix))
		.flat_map(|r| r.fields.iter().map(|f| f.as_str()))
		.collect();
	if drop.is_empty() {
		return None;
	}
	let Ok(Json::Obj(mut o)) = Json::parse(body) else {
		return None;
	};
	let n = o.len();
	o.retain(|(k, _)| !drop.contains(&k.as_str()));
	if std::env::var("QWEN_SERVER_DEBUG").is_ok() && o.len() != n {
		eprintln!(
			"--client-ignore: dropped {} field(s) of a request from {user_agent:?}",
			n - o.len()
		);
	}
	(o.len() != n).then(|| Json::Obj(o).to_string().into_bytes())
}

/// Route a single parsed request. Split out from `handle_conn` so it can be
/// exercised in tests with an in-memory writer. `loaded` is `None` only in
/// unit tests for the model-less routes; production always passes `Some`.
#[cfg(test)]
fn handle_request(
	method: &str,
	path: &str,
	body: &[u8],
	loaded: Option<&Loaded>,
	w: &mut dyn Write,
) -> std::io::Result<()> {
	handle_request_h(method, path, &[], body, loaded, w)
}

/// [`handle_request`] with the request's headers (lower-case names).
fn handle_request_h(
	method: &str,
	path: &str,
	headers: &[(String, String)],
	body: &[u8],
	loaded: Option<&Loaded>,
	w: &mut dyn Write,
) -> std::io::Result<()> {
	// Route on the path alone. Claude Code posts to `/v1/messages?beta=true`,
	// and matching the raw request target meant that 404'd, which the client
	// renders as "there's an issue with the selected model (…). It may not
	// exist or you may not have access to it", a message that sends you looking
	// at model names and auth for as long as you believe it. Stripped here
	// rather than at the call site so every caller, tests included, routes the
	// same way.
	let path = path.split('?').next().unwrap_or(path);
	// chat waits for a running image job (and an image job for chat)
	let _devices = matches!(
		path,
		"/v1/chat/completions" | "/chat/completions" | "/v1/messages"
	)
	.then(|| images::DEVICES.read().unwrap_or_else(|e| e.into_inner()));
	let ct = headers
		.iter()
		.find(|(k, _)| k == "content-type")
		.map_or("", |(_, v)| v.as_str());
	match (method, path) {
		("GET", "/health") => respond_json(w, 200, "OK", "{\"status\":\"ok\"}"),
		(_, "/mcp") => {
			if images::cfg().is_none() {
				return write_error(w, 404, "MCP image tools need --image-model");
			}
			mcp::handle(w, method, headers, body)
		}
		("POST", "/v1/images/generations" | "/images/generations") => {
			images::handle_generations(w, body)
		}
		("POST", "/v1/images/edits" | "/images/edits") => images::handle_edits(w, ct, body),
		// `/models` is llama-server's alias. The `meta` object is llama-server's
		// too (an OpenAI or Anthropic client ignores it); there must be no
		// `status` field, which is what tells Zed's llama.cpp provider it is
		// talking to a multi-model router rather than one loaded model.
		("GET", "/v1/models" | "/models") => {
			let meta = loaded.map_or(String::new(), |l| {
				format!(",\"meta\":{}", openai::model_meta_json(l))
			});
			let body = format!(
				"{{\"object\":\"list\",\"data\":[{{\"id\":{m},\"type\":\"model\",\"object\":\"model\",\"created\":0,\"owned_by\":\"qwen.rs\"{meta}}}]}}",
				m = jstr(loaded.map_or(DEFAULT_MODEL, |l| l.model_id()))
			);
			respond_json(w, 200, "OK", &body)
		}
		("GET", "/props") => match loaded {
			Some(l) => respond_json(w, 200, "OK", &openai::props_json(l)),
			None => write_error(w, 503, "model not loaded"),
		},
		// OpenAI Chat Completions, as llama-server serves it; Zed's llama.cpp
		// provider. See `openai.rs`.
		("POST", "/v1/chat/completions" | "/chat/completions") => match (Json::parse(body), loaded)
		{
			(Ok(req), Some(loaded)) => openai::handle_chat(w, loaded, &req),
			(Ok(_), None) => write_error(w, 500, "model not loaded"),
			(Err(e), _) => write_error(w, 400, &format!("invalid JSON: {e}")),
		},
		// Claude Code asks how big a request is before sending it, to decide
		// whether to compact the conversation. Answering 404 does not break it,
		// but it then has to guess.
		("POST", "/v1/messages/count_tokens") => match (Json::parse(body), loaded) {
			(Ok(req), Some(loaded)) => {
				let tools = tools::specs_from_request(&req);
				let tool_json: Vec<String> = tools.iter().map(|t| t.json.clone()).collect();
				let turns = match turns_from_request(&req, &mut |_| {
					// Counting must not run the vision tower: it is seconds of
					// work for a number the caller only wants an estimate of.
					// A blank placeholder undercounts an image, which is the
					// safe direction (the real request still enforces the cap).
					Ok(String::new())
				}) {
					Ok(t) => t,
					Err(e) => return write_error(w, 400, &e),
				};
				let think = match thinking_from_request(&req) {
					Ok(t) => t.think,
					Err(e) => return write_error(w, 400, &e),
				};
				let prompt = Tokenizer::build_prompt_with(&turns, &tool_json, think, think.open());
				let n = encode_prompt(&loaded.tok, &prompt).len();
				respond_json(w, 200, "OK", &format!("{{\"input_tokens\":{n}}}"))
			}
			(Err(e), _) => write_error(w, 400, &format!("invalid JSON: {e}")),
			(_, None) => write_error(w, 503, "model not loaded"),
		},
		("POST", "/v1/messages") => match (Json::parse(body), loaded) {
			(Ok(req), Some(loaded)) => handle_messages(w, loaded, &req),
			(Ok(_), None) => write_error(w, 500, "model not loaded"),
			(Err(e), _) => write_error(w, 400, &format!("invalid JSON: {e}")),
		},
		_ => write_error(w, 404, &format!("no route for {method} {path}")),
	}
}

/// Run the blocking HTTP server. Binds `host:port` and serves requests until
/// the process is terminated. The model and tokenizer are loaded **once** at
/// startup and shared (read-only) across all request threads via `Arc`.
#[allow(clippy::too_many_arguments)]
pub fn serve(
	model_path: &str,
	mmproj: Option<&str>,
	host: &str,
	port: u16,
	n_gpu_layers: usize,
	ctx: usize,
	n_batch: usize,
	max_image_tokens: usize,
	cache: &CacheOpts,
	mtp: usize,
	dflash: Option<&str>,
) {
	let mut loaded = match Loaded::load(
		model_path,
		mmproj,
		n_gpu_layers,
		ctx,
		n_batch,
		max_image_tokens,
		mtp,
		dflash,
	) {
		Ok(l) => l,
		Err(e) => {
			eprintln!("server: {e}");
			return;
		}
	};
	if cache.mb > 0 {
		// A snapshot is only valid for the model file that made it.
		let tag = std::fs::metadata(model_path).map_or_else(
			|_| model_path.to_string(),
			|m| {
				let name = std::path::Path::new(model_path)
					.file_name()
					.map(|n| n.to_string_lossy().into_owned());
				format!("{}:{}", name.unwrap_or_default(), m.len())
			},
		);
		loaded.cache = cache.build(120.0, &tag);
		let Backend::Gguf(model) = &loaded.backend else {
			unreachable!()
		};
		if let Some(dir) = &cache.dir {
			let known = |t: &[u32]| loaded.cache.contains(t);
			for (tokens, rs) in snapdir::import::<RunState>(std::path::Path::new(dir), &tag, &known)
			{
				loaded
					.cache
					.insert(tokens, gguf_snap(model, &rs), Class::Shared);
			}
		}
	}
	let what = format!(
		"model loaded once: {model_path}{}",
		match mmproj {
			Some(p) => format!("; vision: {p}"),
			None => String::new(),
		}
	);
	serve_loaded(loaded, host, port, &what);
}

/// The prefix cache's settings (`--prompt-cache-*`).
#[derive(Clone, Debug, Default)]
pub struct CacheOpts {
	/// RAM budget in MB; 0 disables the cache.
	pub mb: usize,
	/// `--prompt-cache-dir`: the disk tier (shared prefixes persist).
	pub dir: Option<String>,
	/// Its budget in MB (default: `mb`).
	pub disk_mb: Option<usize>,
	/// `--prompt-cache-spill`: states evicted from RAM go to disk too.
	pub spill: bool,
}

impl CacheOpts {
	/// The cache for a backend prefilling about `tok_s` tokens per second
	/// (the cost model's prior), its disk tier on `self.dir` for the model
	/// `tag`.
	pub fn build(&self, tok_s: f64, tag: &str) -> Cache {
		let c = Cache::new(prefixcache::Config::new(self.mb << 20, tok_s));
		let Some(dir) = &self.dir else { return c };
		let mut d = prefixcache::DiskCfg::new(
			std::path::Path::new(dir),
			tag.to_string(),
			(self.disk_mb.unwrap_or(self.mb) as u64) << 20,
		);
		d.spill = self.spill;
		match Cache::new(prefixcache::Config::new(self.mb << 20, tok_s)).with_disk(d) {
			Ok(c) => {
				eprintln!(
					"prompt cache dir: {dir}{} ({})",
					if self.spill {
						", spilling evicted states"
					} else {
						""
					},
					c.summary()
				);
				c
			}
			Err(e) => {
				eprintln!("prompt cache dir: {dir}: {e}; not persisting");
				c
			}
		}
	}
}

/// `serve` for a [`Backend::Engine`]: one request at a time. With a cache
/// budget and an engine that can snapshot its state, a prefix cache (and
/// its disk tier), as on the GGUF path; `tok_s` is about the engine's
/// prefill speed.
pub fn serve_engine(
	mut eng: Box<dyn Engine>,
	info: EngineInfo,
	tok: Tokenizer,
	host: &str,
	port: u16,
	cache_opts: &CacheOpts,
	tok_s: f64,
) {
	let cache = cache_opts.mb > 0
		&& eng
			.can_snapshot()
			.unwrap_or_else(|| eng.snapshot().is_some());
	let what = format!(
		"{}, context {} tokens; {}, one request at a time, {}",
		info.name,
		info.ctx,
		match &info.image {
			Some(g) => format!("images of up to {} tokens", g.max_tokens),
			None => "text only".into(),
		},
		if cache {
			format!("prefix cache {} MB", cache_opts.mb)
		} else {
			"no prefix cache".into()
		}
	);
	let tag = info.tag.clone();
	let _sp = quant::ltime::span(|| "server: setup".into());
	let mut loaded = Loaded::from_engine(eng, info, tok);
	if cache {
		loaded.cache = cache_opts.build(tok_s, &tag);
		if let Some(dir) = &cache_opts.dir {
			let known = |t: &[u32]| loaded.cache.contains(t);
			for (tokens, s) in
				snapdir::import::<EngineSnap>(std::path::Path::new(dir), &tag, &known)
			{
				loaded.cache.insert(tokens, s.into(), Class::Shared);
			}
		}
	}
	serve_loaded(loaded, host, port, &what);
}

/// Serve only the image routes (`/mcp`, `/v1/images/*`): no chat model.
pub fn serve_images(host: &str, port: u16) {
	let addr = format!("{host}:{port}");
	let listener = match TcpListener::bind(&addr) {
		Ok(l) => l,
		Err(e) => {
			eprintln!("server: bind {addr} failed: {e}");
			return;
		}
	};
	let dir = images::cfg().map_or(String::new(), |c| c.dir.display().to_string());
	eprintln!(
		"qwen.rs HTTP server listening on http://{addr}  (images only: {dir}, loaded on the first job; MCP at /mcp)"
	);
	for s in listener.incoming().flatten() {
		thread::spawn(move || handle_conn(s, None));
	}
}

fn serve_loaded(loaded: Loaded, host: &str, port: u16, what: &str) {
	if let Some(o) = tune::take() {
		eprintln!("qwen.rs tuning ({what})");
		if let Err(e) = tune::run(&loaded, o) {
			eprintln!("tune: {e}");
			std::process::exit(1);
		}
		return;
	}
	eprintln!("{}", current_profile().describe());
	let loaded = Arc::new(loaded);
	let addr = format!("{host}:{port}");
	let listener = match TcpListener::bind(&addr) {
		Ok(l) => l,
		Err(e) => {
			eprintln!("server: bind {addr} failed: {e}");
			return;
		}
	};
	eprintln!("qwen.rs HTTP server listening on http://{addr}  ({what})");
	if quant::ltime::on() {
		eprintln!("[load] {:6.2}  listening", quant::ltime::now());
	}
	for s in listener.incoming().flatten() {
		let l = loaded.clone();
		thread::spawn(move || handle_conn(s, Some(l)));
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn parse_anthropic_request() {
		let body = r#"{
            "model": "Qwen3.8-27B",
            "max_tokens": 128,
            "temperature": 0.7,
            "stream": true,
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
                {"role": "user", "content": [{"type": "text", "text": "How are you?"}]}
            ],
            "stop_sequences": ["STOP"]
        }"#;
		let j = Json::parse(body.as_bytes()).expect("parse");
		assert_eq!(j.get("model").and_then(|v| v.as_str()), Some("Qwen3.8-27B"));
		assert_eq!(j.get("max_tokens").and_then(|v| v.as_f64()), Some(128.0));
		assert_eq!(j.get("temperature").and_then(|v| v.as_f64()), Some(0.7));
		assert_eq!(j.get("stream").and_then(|v| v.as_bool()), Some(true));
		assert_eq!(
			text_of(j.get("system")),
			Some("You are helpful.".to_string())
		);
		let msgs = j.get("messages").and_then(|v| v.as_array()).unwrap();
		assert_eq!(text_of(msgs[0].get("content")), Some("Hello".to_string()));
		assert_eq!(text_of(msgs[1].get("content")), Some("Hi!".to_string()));
		assert_eq!(
			text_of(msgs[2].get("content")),
			Some("How are you?".to_string())
		);
		let stops = j.get("stop_sequences").and_then(|v| v.as_array()).unwrap();
		assert_eq!(stops[0].as_str(), Some("STOP"));
	}

	#[test]
	fn parse_unicode_escape() {
		let j = Json::parse(br#""caf\u00e9 \uD83D\uDE00""#).expect("parse");
		assert_eq!(j.as_str(), Some("café 😀"));
	}

	/// A full Claude-Code-shaped tool conversation must land in the template's
	/// turns in the right order and the right roles.
	///
	/// This is the part of the path that fails *quietly*: put a tool result in
	/// the wrong turn, drop an assistant's call, or render `input` as JSON where
	/// the template wants bare text, and the model still answers; just worse.
	/// Nothing downstream would notice.
	#[test]
	fn a_tool_conversation_becomes_the_right_turns() {
		let req = Json::parse(
			br#"{
              "system": "Be terse.",
              "messages": [
                {"role": "user", "content": "how big is main.rs?"},
                {"role": "assistant", "content": [
                   {"type": "text", "text": "Checking."},
                   {"type": "tool_use", "id": "toolu_1", "name": "Read",
                    "input": {"file_path": "/a/main.rs", "limit": 100}}]},
                {"role": "user", "content": [
                   {"type": "tool_result", "tool_use_id": "toolu_1", "content": "fn main() {}"}]},
                {"role": "user", "content": "thanks"}
              ]}"#,
		)
		.unwrap();
		let turns = turns_from_request(&req, &mut |_| Ok(String::new())).unwrap();
		let got = Tokenizer::build_prompt(&turns, &[], Tokenizer::ASSISTANT_OPEN_NOTHINK);
		assert_eq!(
			got,
			"<|im_start|>system\nBe terse.<|im_end|>\n\
             <|im_start|>user\nhow big is main.rs?<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\nChecking.\n\n\
             <tool_call>\n<function=Read>\n\
             <parameter=file_path>\n/a/main.rs\n</parameter>\n\
             <parameter=limit>\n100\n</parameter>\n\
             </function>\n</tool_call><|im_end|>\n\
             <|im_start|>user\n<tool_response>\nfn main() {}\n</tool_response><|im_end|>\n\
             <|im_start|>user\nthanks<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
		);
	}

	/// A string argument goes into the prompt as bare text and a non-string as
	/// JSON: the same split the template makes when it writes a call out, and
	/// the one the parser reverses. Quoting a string here would teach the model
	/// to quote them back.
	#[test]
	fn tool_use_input_renders_strings_bare_and_others_as_json() {
		let input = Json::parse(br#"{"s":"a b","n":3,"b":true,"o":{"k":[1,2]}}"#).unwrap();
		assert_eq!(
			render_args(&input),
			vec![
				("s".to_string(), "a b".to_string()),
				("n".to_string(), "3".to_string()),
				("b".to_string(), "true".to_string()),
				("o".to_string(), r#"{"k":[1,2]}"#.to_string()),
			]
		);
	}

	/// A failed tool must not read like a successful one. The model's
	/// `<tool_response>` has no error channel, so `is_error` has to become
	/// words or it is lost entirely.
	#[test]
	fn an_errored_tool_result_says_so() {
		let req = Json::parse(
			br#"{"messages":[{"role":"user","content":[
                 {"type":"tool_result","tool_use_id":"t","content":"no such file","is_error":true}]}]}"#,
		)
		.unwrap();
		let turns = turns_from_request(&req, &mut |_| Ok(String::new())).unwrap();
		let got = Tokenizer::build_prompt(&turns, &[], "");
		assert!(
			got.contains("<tool_response>\nError: no such file\n</tool_response>"),
			"{got}"
		);
	}

	/// An image in a tool result (an agent reading a PNG) is encoded and its
	/// placeholder rendered inside the `<tool_response>`, between the text
	/// parts; the results' images reach `on_image` before the turn's own,
	/// in the order the prompt renders them.
	#[test]
	fn tool_result_images_are_rendered_in_prompt_order() {
		let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
		let req = Json::parse(
			format!(
				r#"{{"messages":[{{"role":"user","content":[
                 {{"type":"text","text":"and this "}},
                 {{"type":"image","source":{{"type":"base64","data":"{png}"}}}},
                 {{"type":"tool_result","tool_use_id":"t","content":[
                   {{"type":"text","text":"a.png:"}},
                   {{"type":"image","source":{{"type":"base64","data":"data:image/png;base64,{png}"}}}},
                   {{"type":"text","text":"done"}}]}}]}}]}}"#
			)
			.as_bytes(),
		)
		.unwrap();
		let mut n = 0;
		let turns = turns_from_request(&req, &mut |img| {
			assert_eq!((img.width, img.height), (1, 1));
			n += 1;
			Ok(format!("[img{n}]"))
		})
		.unwrap();
		let got = Tokenizer::build_prompt(&turns, &[], "");
		assert_eq!(
			got,
			"<|im_start|>user\n<tool_response>\na.png:[img1]done\n</tool_response><|im_end|>\n\
             <|im_start|>user\nand this [img2]<|im_end|>\n"
		);
	}

	/// Parallel calls in one assistant turn stay in one turn, and their results
	/// come back in one user frame.
	#[test]
	fn parallel_calls_and_results_stay_grouped() {
		let req = Json::parse(
			br#"{"messages":[
                {"role":"assistant","content":[
                  {"type":"tool_use","id":"a","name":"A","input":{}},
                  {"type":"tool_use","id":"b","name":"B","input":{}}]},
                {"role":"user","content":[
                  {"type":"tool_result","tool_use_id":"a","content":"1"},
                  {"type":"tool_result","tool_use_id":"b","content":"2"}]}]}"#,
		)
		.unwrap();
		let turns = turns_from_request(&req, &mut |_| Ok(String::new())).unwrap();
		let got = Tokenizer::build_prompt(&turns, &[], "");
		assert_eq!(
			got,
			"<|im_start|>assistant\n<think>\n\n</think>\n\n\
             <tool_call>\n<function=A>\n</function>\n</tool_call>\n\
             <tool_call>\n<function=B>\n</function>\n</tool_call><|im_end|>\n\
             <|im_start|>user\n<tool_response>\n1\n</tool_response>\
             \n<tool_response>\n2\n</tool_response><|im_end|>\n"
		);
	}

	/// `tool_choice: none` means the tools must not be offered at all; leaving
	/// the preamble in would invite exactly the calls the caller forbade.
	#[test]
	fn tool_choice_none_drops_the_tool_preamble() {
		let req = Json::parse(
			br#"{"tool_choice":{"type":"none"},
                 "tools":[{"name":"X","input_schema":{"type":"object"}}]}"#,
		)
		.unwrap();
		let specs = tools::specs_from_request(&req);
		assert_eq!(specs.len(), 1);
		assert_eq!(tools::choice_from_request(&req), tools::Choice::None);
	}

	#[test]
	fn json_round_trips_through_the_writer() {
		for src in [
			r#"{"a":1,"b":[true,null,"x"],"c":{"d":-2.5}}"#,
			r#"[]"#,
			r#"{"n":100}"#,
		] {
			let v = Json::parse(src.as_bytes()).unwrap();
			assert_eq!(v.to_string(), src);
		}
		// Integral floats must not gain a `.0`: these land in a prompt, where
		// `"maxLength": 100.0` describes something different to the model than
		// `"maxLength": 100`.
		assert_eq!(Json::Num(100.0).to_string(), "100");
		assert_eq!(Json::Num(1.5).to_string(), "1.5");
	}

	/// Leading whitespace is dropped, interior whitespace is not, and the gate
	/// must behave the same however the stream is chunked: in SSE the text has
	/// already left the building by the time the tool call is recognized.
	#[test]
	fn leading_whitespace_is_gated_but_interior_is_kept() {
		let mut g = LeadingSpace::default();
		assert_eq!(
			g.push(
				"

"
			),
			""
		);
		assert_eq!(g.push("  "), "");
		assert_eq!(g.push("  hello"), "hello");
		assert_eq!(
			g.push(
				"

world"
			),
			"

world"
		);

		// A turn that is nothing but whitespace emits nothing at all.
		let mut g = LeadingSpace::default();
		assert_eq!(
			g.push(
				"

"
			),
			""
		);
		assert_eq!(g.push(" 	"), "");
	}

	/// A query string must not change the route. Claude Code posts to
	/// `/v1/messages?beta=true`; routing on the raw target 404'd it, and the
	/// client rendered that as a model-availability error, which is about as
	/// far from the cause as a message can get.
	#[test]
	fn routes_ignore_the_query_string() {
		let mut out: Vec<u8> = Vec::new();
		handle_request("GET", "/health?x=1", b"", None, &mut out).unwrap();
		let s = String::from_utf8(out).unwrap();
		assert!(s.starts_with("HTTP/1.1 200"), "{s}");

		// And an unknown path is still a 404.
		let mut out: Vec<u8> = Vec::new();
		handle_request("GET", "/nope?x=1", b"", None, &mut out).unwrap();
		assert!(String::from_utf8(out).unwrap().starts_with("HTTP/1.1 404"));
	}

	#[test]
	fn jstr_escapes() {
		assert_eq!(jstr("a\"b\\c\n"), "\"a\\\"b\\\\c\\n\"");
	}

	/// A string is taken in runs between quotes and escapes (it was
	/// validated from every character to the end of the body: a megabyte of
	/// base64 took 13 s).
	#[test]
	fn parse_strings_in_runs() {
		let j = Json::parse("\"h\u{e9}llo \\\"w\\\" \u{1F600}\\n\"".as_bytes()).unwrap();
		assert_eq!(j.as_str(), Some("h\u{e9}llo \"w\" \u{1F600}\n"));
		assert!(Json::parse(b"\"a\xffb\"").is_err());
		assert!(Json::parse(b"\"open").is_err());
		let big = format!("{{\"data\":\"{}\"}}", "A".repeat(8 << 20));
		let t = std::time::Instant::now();
		let j = Json::parse(big.as_bytes()).unwrap();
		assert_eq!(
			j.get("data").and_then(|v| v.as_str()).map(str::len),
			Some(8 << 20)
		);
		assert!(t.elapsed().as_secs() < 10, "{:?}", t.elapsed());
	}

	#[test]
	fn parse_invalid() {
		assert!(Json::parse(b"{not valid").is_err());
	}

	fn body_of(out: &[u8]) -> String {
		// strip HTTP headers, return everything after the first blank line
		let s = String::from_utf8_lossy(out);
		match s.find("\r\n\r\n") {
			Some(i) => s[i + 4..].to_string(),
			None => s.to_string(),
		}
	}

	#[test]
	fn route_health() {
		let mut out: Vec<u8> = Vec::new();
		handle_request("GET", "/health", b"", None, &mut out).unwrap();
		assert!(String::from_utf8_lossy(&out).starts_with("HTTP/1.1 200 OK"));
		assert_eq!(body_of(&out), "{\"status\":\"ok\"}");
	}

	#[test]
	fn route_models() {
		let mut out: Vec<u8> = Vec::new();
		handle_request("GET", "/v1/models", b"", None, &mut out).unwrap();
		let b = body_of(&out);
		assert!(b.contains("\"id\":\"Qwen3.8-27B\""));
		assert!(b.contains("\"type\":\"model\""));
	}

	#[test]
	fn route_not_found() {
		let mut out: Vec<u8> = Vec::new();
		handle_request("GET", "/nope", b"", None, &mut out).unwrap();
		assert!(String::from_utf8_lossy(&out).contains("404"));
	}

	#[test]
	fn route_bad_json() {
		let mut out: Vec<u8> = Vec::new();
		handle_request("POST", "/v1/messages", b"{bad", None, &mut out).unwrap();
		let b = body_of(&out);
		assert!(b.contains("\"type\":\"error\""));
		assert!(b.contains("invalid JSON"));
	}

	#[test]
	fn load_fails_on_missing_model() {
		// The server now loads the model once at startup, so the per-request
		// load-failure path is exercised through `Loaded::load` (which `serve`
		// calls). A bogus path must surface a clear error rather than panic.
		let err = Loaded::load("/no/such/model.gguf", None, 0, 0, 2048, 1024, 0, None)
			.err()
			.expect("should fail");
		assert!(err.contains("model load failed") || err.contains("parse"));
	}

	#[test]
	fn messages_without_loaded_model_errors() {
		// A /v1/messages request arriving before the model is loaded (e.g. in a
		// unit test) must yield a well-formed error, not a panic.
		let mut out: Vec<u8> = Vec::new();
		let req = br#"{"model":"m","max_tokens":4,"messages":[{"role":"user","content":"hi"}]}"#;
		handle_request("POST", "/v1/messages", req, None, &mut out).unwrap();
		let s = String::from_utf8_lossy(&out);
		assert!(s.contains("\"type\":\"error\""));
	}

	fn img(w: usize, h: usize, fill: u8) -> (imgio::Image, vision::Grid) {
		let grid = vision::Grid {
			width: 64,
			height: 64,
			gw: 4,
			gh: 4,
			mw: 2,
			mh: 2,
		};
		(
			imgio::Image {
				width: w,
				height: h,
				rgb: vec![fill; w * h * 3],
			},
			grid,
		)
	}

	#[test]
	fn cache_key_tells_images_apart_and_keeps_the_text() {
		let pad = 9;
		let ids = vec![1, 2, pad, pad, pad, pad, 3, pad, pad, pad, pad, 4];
		let spans = vec![(2, 4), (7, 4)];
		let a = [img(8, 8, 10), img(8, 8, 20)];
		let ka = cache_key(&ids, &spans, &a);
		// Text untouched, every placeholder replaced by a non-vocabulary id.
		assert_eq!((ka[0], ka[1], ka[6], ka[11]), (1, 2, 3, 4));
		assert!(spans
			.iter()
			.all(|&(s, n)| ka[s..s + n].iter().all(|&k| k >= 0x8000_0000)));
		// The same pixels give the same key; one changed image changes only
		// its own run, so the prefix before it still matches.
		assert_eq!(ka, cache_key(&ids, &spans, &[img(8, 8, 10), img(8, 8, 20)]));
		let kb = cache_key(&ids, &spans, &[img(8, 8, 10), img(8, 8, 21)]);
		assert_eq!(ka[..7], kb[..7]);
		assert_ne!(ka[7..11], kb[7..11]);
		// Same bytes, different shape.
		assert_ne!(image_hash(&img(8, 2, 0).0), image_hash(&img(2, 8, 0).0));
	}

	#[test]
	fn one_pixel_changes_the_image_hash() {
		let (mut a, _) = img(33, 7, 0);
		let h = image_hash(&a);
		let last = a.rgb.len() - 1;
		a.rgb[last] = 1; // in the tail the word loop does not cover
		assert_ne!(h, image_hash(&a));
		a.rgb[last] = 0;
		a.rgb[0] = 1;
		assert_ne!(h, image_hash(&a));
	}

	// --- Backend::Engine, on a scripted engine and a byte-level tokenizer ---

	/// A tokenizer with one token per byte (id = byte value) and the chat /
	/// tool specials, written as a Hugging Face `tokenizer.json`.
	fn byte_tokenizer() -> Tokenizer {
		let mut bs: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
		let mut map = vec![0u32; 256];
		for &b in &bs {
			map[b as usize] = b;
		}
		let mut n = 0;
		for b in 0u32..=255 {
			if !bs.contains(&b) {
				bs.push(b);
				map[b as usize] = 256 + n;
				n += 1;
			}
		}
		let vocab: Vec<String> = (0..256)
			.map(|b| format!("{}:{b}", jstr(&char::from_u32(map[b]).unwrap().to_string())))
			.collect();
		let added: Vec<String> = SPECIALS
			.iter()
			.enumerate()
			.map(|(i, (s, sp))| {
				format!(
					"{{\"id\":{},\"content\":{},\"special\":{sp}}}",
					256 + i,
					jstr(s)
				)
			})
			.collect();
		let json = format!(
			"{{\"added_tokens\":[{}],\"model\":{{\"type\":\"BPE\",\"vocab\":{{{}}},\"merges\":[]}}}}",
			added.join(","),
			vocab.join(",")
		);
		static N: AtomicU64 = AtomicU64::new(0);
		let dir = std::env::temp_dir().join(format!(
			"qwen-rs-server-test-{}-{}",
			std::process::id(),
			N.fetch_add(1, Ordering::Relaxed)
		));
		std::fs::create_dir_all(&dir).unwrap();
		std::fs::write(dir.join("tokenizer.json"), json).unwrap();
		std::fs::write(
			dir.join("tokenizer_config.json"),
			"{\"eos_token\":\"<|im_end|>\"}",
		)
		.unwrap();
		let tok = Tokenizer::from_hf_json(&dir.join("tokenizer.json")).unwrap();
		let _ = std::fs::remove_dir_all(&dir);
		tok
	}

	const SPECIALS: [(&str, bool); 12] = [
		("<|im_start|>", true),
		("<|im_end|>", true),
		("<|endoftext|>", true),
		("<|vision_start|>", true),
		("<|vision_end|>", true),
		("<|image_pad|>", true),
		("<tool_call>", false),
		("</tool_call>", false),
		("<think>", false),
		("</think>", false),
		("<tool_response>", false),
		("</tool_response>", false),
	];

	#[derive(Default)]
	struct Log {
		resets: usize,
		/// Length of each `forward` call.
		calls: Vec<usize>,
		/// Tokens fed since the last reset.
		fed: usize,
		/// `restore` calls, with the position restored to.
		restores: Vec<usize>,
		/// Images fed: (position of the run's first token, pixels).
		images: Vec<(usize, usize)>,
	}

	/// Answers every conversation with `reply` (token ids), one-hot logits.
	struct Scripted {
		reply: Vec<u32>,
		nv: usize,
		log: Arc<Mutex<Log>>,
		/// `forward` calls since the last reset.
		k: usize,
		/// Whether `snapshot` / `restore` work (a prefix cache).
		snaps: bool,
	}

	impl Engine for Scripted {
		fn reset(&mut self) -> Result<(), String> {
			let mut l = self.log.lock().unwrap();
			l.resets += 1;
			l.fed = 0;
			self.k = 0;
			Ok(())
		}

		fn forward(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
			let mut l = self.log.lock().unwrap();
			l.calls.push(ids.len());
			l.fed += ids.len();
			let mut v = vec![0.0; self.nv];
			v[self.reply[self.k.min(self.reply.len() - 1)] as usize] = 1.0;
			self.k += 1;
			Ok(v)
		}

		fn feed(&mut self, ids: &[u32]) -> Result<(), String> {
			let mut l = self.log.lock().unwrap();
			l.calls.push(ids.len());
			l.fed += ids.len();
			Ok(())
		}

		fn snapshot(&mut self) -> Option<EngineSnap> {
			let fed = self.log.lock().unwrap().fed;
			self.snaps.then(|| EngineSnap {
				pos: fed,
				parts: vec![quant::snap::part(vec![fed as u8; 3])],
				layout: Vec::new(),
			})
		}

		fn restore(&mut self, snap: &EngineSnap) -> Result<(), String> {
			assert_eq!(snap.parts, vec![quant::snap::part(vec![snap.pos as u8; 3])]);
			let mut l = self.log.lock().unwrap();
			l.fed = snap.pos;
			l.restores.push(snap.pos);
			Ok(())
		}

		fn feed_images(
			&mut self,
			ids: &[u32],
			images: &[(usize, &imgio::Image)],
		) -> Result<(), String> {
			let mut l = self.log.lock().unwrap();
			for (start, img) in images {
				let at = l.fed + start;
				l.images.push((at, img.width * img.height));
			}
			l.calls.push(ids.len());
			l.fed += ids.len();
			Ok(())
		}
	}

	fn engine_server(reply: &str, ctx: usize) -> (Loaded, Arc<Mutex<Log>>) {
		engine_server_with(reply, ctx, false)
	}

	fn engine_server_with(reply: &str, ctx: usize, snaps: bool) -> (Loaded, Arc<Mutex<Log>>) {
		let tok = byte_tokenizer();
		let log = Arc::new(Mutex::new(Log::default()));
		let reply = tok.encode(reply);
		let nv = tok.vocab_size();
		let info = EngineInfo {
			name: "scripted".into(),
			model: DEFAULT_MODEL.into(),
			ctx,
			ctx_train: 4096,
			n_vocab: nv,
			n_embd: 8,
			stop: vec![tok.eos(), tok.token_id("<|endoftext|>").unwrap()],
			tag: "scripted".into(),
			image: None,
		};
		let eng = Scripted {
			reply,
			nv,
			log: log.clone(),
			k: 0,
			snaps,
		};
		(Loaded::from_engine(Box::new(eng), info, tok), log)
	}

	#[test]
	fn client_ignore_drops_the_named_fields() {
		let body = br#"{"model":"m","temperature":1.0,"top_p":0.9,"max_tokens":8,"messages":[]}"#;
		let zed = "Zed/1.27.0+stable (windows; x86_64)";
		// The default drops every client's temperature, nothing else.
		let def = ClientIgnore::defaults();
		for ua in [zed, "claude-cli/2.0", ""] {
			let out = drop_client_fields(&def, ua, body).expect("rewritten");
			let j = Json::parse(&out).unwrap();
			assert!(j.get("temperature").is_none());
			assert_eq!(j.get("top_p").and_then(|v| v.as_f64()), Some(0.9));
			assert_eq!(j.get("max_tokens").and_then(|v| v.as_f64()), Some(8.0));
		}
		// A request without the fields, and a rule for another client, leave
		// the body alone.
		assert!(drop_client_fields(&def, zed, br#"{"messages":[]}"#).is_none());
		let zed_only = vec![ClientIgnore::parse("Zed/=temperature").unwrap()];
		assert!(drop_client_fields(&zed_only, "claude-cli/2.0", body).is_none());
		assert!(drop_client_fields(&zed_only, zed, body).is_some());
		assert!(drop_client_fields(&[], zed, body).is_none());
		// `*` matches every client; several fields per rule.
		let any = vec![ClientIgnore::parse("*=temperature, top_p").unwrap()];
		let j = Json::parse(&drop_client_fields(&any, "curl/8", body).unwrap()).unwrap();
		assert!(j.get("temperature").is_none() && j.get("top_p").is_none());
		assert!(ClientIgnore::parse("Zed/").is_err());
		assert!(ClientIgnore::parse("=temperature").is_err());
	}

	#[test]
	fn engine_prefix_cache_reuses_the_conversation() {
		let (mut l, log) = engine_server_with("Paris.<|im_end|>", 4096, true);
		l.cache = Cache::new(prefixcache::Config::new(1 << 20, 100.0));
		let first = r#"{"max_tokens":32,"temperature":0,"system":"Be terse.","messages":[{"role":"user","content":"Capital of France?"}]}"#;
		// The llama-server dialect defaults thinking on; this one says off so
		// both requests render the same prompt.
		let second = r#"{"max_tokens":32,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"system","content":"Be terse."},{"role":"user","content":"Capital of France?"},{"role":"assistant","content":"Paris."},{"role":"user","content":"And Spain?"}]}"#;
		let a = post(&l, "/v1/messages", first);
		assert!(a.contains("\"text\":\"Paris.\""), "{a}");
		let b = post(&l, "/v1/chat/completions", second);
		assert!(b.contains("Paris."), "{b}");
		let render = |turns: &[Turn]| {
			l.tok.encode(&Tokenizer::build_prompt(
				turns,
				&[],
				Tokenizer::ASSISTANT_OPEN_NOTHINK,
			))
		};
		let p1 = render(&[
			Turn::System("Be terse.".into()),
			Turn::User("Capital of France?".into()),
		]);
		let im = l.tok.token_id("<|im_start|>").unwrap();
		let c1 = p1.iter().rposition(|&x| x == im).unwrap();
		let g = log.lock().unwrap();
		// The second request starts from the first one's checkpoint ...
		assert_eq!(g.restores, vec![c1]);
		assert!(b.contains(&format!("\"cached_tokens\":{c1}")), "{b}");
		// ... and the first request split its prompt there: feed(..c1), then
		// forward(c1..), then one call per generated token but the last.
		assert_eq!(&g.calls[..2], &[c1, p1.len() - c1]);
		drop(g);
		// A third request on another conversation misses and prefills it all.
		let c = post(
			&l,
			"/v1/messages",
			r#"{"max_tokens":32,"temperature":0,"messages":[{"role":"user","content":"hi"}]}"#,
		);
		assert!(c.contains("Paris."), "{c}");
		assert_eq!(log.lock().unwrap().restores.len(), 1);
		// The cache holds every checkpoint: a hit does not consume one.
		assert_eq!(l.cache.len(), 3);
		// So a branch off the first turn (another follow-up than the second
		// request's) still restores the first checkpoint, which the second
		// request used to consume.
		let branch = r#"{"max_tokens":32,"temperature":0,"system":"Be terse.","messages":[{"role":"user","content":"Capital of France?"},{"role":"assistant","content":"Paris."},{"role":"user","content":"And Italy?"}]}"#;
		let d = post(&l, "/v1/messages", branch);
		assert!(d.contains("Paris."), "{d}");
		assert_eq!(log.lock().unwrap().restores, vec![c1, c1]);
	}

	#[test]
	fn interval_cuts_that_cost_a_piece_are_dropped() {
		let iv = |v: &[usize]| v.iter().map(|&s| (s, Class::Interval)).collect::<Vec<_>>();
		// a follow-up turn across 12288: one piece, not two
		let mut c = iv(&[12_288]);
		drop_costly_intervals(&mut c, 11_366, 13_394, 4096);
		assert!(c.is_empty());
		// a cold prefill: the intervals are the pieces' ends
		let mut c = iv(&[4096, 8192, 12_288]);
		drop_costly_intervals(&mut c, 0, 13_000, 4096);
		assert_eq!(c, iv(&[4096, 8192, 12_288]));
		// a long follow-up: the cut costs no piece (922 + 4096 + 3616)
		let mut c = iv(&[12_288, 16_384]);
		drop_costly_intervals(&mut c, 11_366, 20_000, 4096);
		assert_eq!(c, iv(&[12_288, 16_384]));
		// other classes stay, and bound the intervals' pieces
		let mut c = vec![(5000, Class::Shared), (8192, Class::Interval)];
		drop_costly_intervals(&mut c, 4500, 9000, 4096);
		assert_eq!(c, vec![(5000, Class::Shared)]);
	}

	#[test]
	fn interval_checkpoints_serve_an_edited_message() {
		let (mut l, log) = engine_server_with("Ok.<|im_end|>", 4096, true);
		let mut cfg = prefixcache::Config::new(1 << 20, 100.0);
		cfg.interval = 64;
		l.cache = Cache::new(cfg);
		let long = "a".repeat(150) + &"b".repeat(150);
		let edited = "a".repeat(150) + &"c".repeat(150);
		let req = |text: &str| {
			format!(
				r#"{{"max_tokens":8,"temperature":0,"messages":[{{"role":"user","content":"{text}"}}]}}"#
			)
		};
		let a = post(&l, "/v1/messages", &req(&long));
		assert!(a.contains("Ok."), "{a}");
		// every 64 tokens of the prompt left a state (and the turn one)
		assert!(l.cache.len() >= 4, "{}", l.cache.len());
		// the same message edited after its first 150 characters restores
		// the last interval state before the edit, not nothing
		let b = post(&l, "/v1/messages", &req(&edited));
		assert!(b.contains("Ok."), "{b}");
		let g = log.lock().unwrap();
		assert_eq!(g.restores.len(), 1);
		let r = g.restores[0];
		assert!(r % 64 == 0 && r > 64 && r <= 150 + 16, "{r}");
	}

	#[test]
	fn diverge_cut_marks_a_long_shared_run_only() {
		// a cached prompt shares 5000 tokens, the best hit holds 1000
		assert_eq!(diverge_cut(Some(&(5000, 9000, vec![])), 1000), Some(4984));
		// not worth a snapshot: under 1024 tokens beyond the hit
		assert_eq!(diverge_cut(Some(&(2000, 9000, vec![])), 1000), None);
		assert_eq!(diverge_cut(None, 0), None);
	}

	/// An engine with a vision tower: an image's run of placeholders is
	/// sized from the engine's geometry and fed through `feed_images`, the
	/// prefix cache tells images apart by their pixels, and an image inside
	/// a restored prefix is not fed again.
	/// Text that names `<|image_pad|>` is not an image's placeholder run.
	#[test]
	fn image_pad_in_text_is_text() {
		let tok = byte_tokenizer();
		let pad = tok.token_id("<|image_pad|>").unwrap();
		let ph = vision::placeholder(4);
		let ids = encode_prompt(
			&tok,
			&format!("the `<|image_pad|>` token{ph}and <|image_pad|><|image_pad|> again"),
		);
		let runs: Vec<usize> = ids
			.iter()
			.enumerate()
			.filter(|&(i, &x)| x == pad && (i == 0 || ids[i - 1] != pad))
			.map(|(i, _)| i)
			.collect();
		assert_eq!(runs.len(), 1);
		assert_eq!(ids.iter().filter(|&&x| x == pad).count(), 4);
		assert_eq!(
			tok.decode(&ids, false),
			format!("the `<|image_pad|>` token{ph}and <|image_pad|><|image_pad|> again")
		);
		// a prompt without the token is untouched
		assert_eq!(
			encode_prompt(&tok, "plain <|im_end|>"),
			tok.encode("plain <|im_end|>")
		);
	}

	#[test]
	fn engine_takes_images() {
		let (mut l, log) = engine_server_with("Red.<|im_end|>", 4096, true);
		l.cache = Cache::new(prefixcache::Config::new(1 << 20, 100.0));
		// 1 x 1 px at 4 tokens: resized to 64 x 64 px, 2 x 2 tokens
		let geom = ImageGeom {
			patch: 16,
			merge: 2,
			max_tokens: 4,
		};
		assert_eq!(geom.grid(1, 1).unwrap().tokens(), 4);
		if let Backend::Engine { info, .. } = &mut l.backend {
			info.image = Some(geom);
		}
		let (red, blue) = ("UDYKMSAxCjI1NQr/AAA=", "UDYKMSAxCjI1NQoAAP8=");
		let turn = |data: &str| {
			format!(
				r#"{{"role":"user","content":[{{"type":"image","source":{{"type":"base64","media_type":"image/x-portable-pixmap","data":"{data}"}}}},{{"type":"text","text":"Which color?"}}]}}"#
			)
		};
		let a = post(
			&l,
			"/v1/messages",
			&format!(
				r#"{{"max_tokens":16,"temperature":0,"messages":[{}]}}"#,
				turn(red)
			),
		);
		assert!(a.contains("\"text\":\"Red.\""), "{a}");
		let pad = l.tok.token_id("<|image_pad|>").unwrap();
		let p1 = l.tok.encode(&Tokenizer::build_prompt(
			&[Turn::User(format!(
				"{}Which color?",
				vision::placeholder(4)
			))],
			&[],
			Tokenizer::ASSISTANT_OPEN_NOTHINK,
		));
		let at = p1.iter().position(|&x| x == pad).unwrap();
		assert_eq!(p1.iter().filter(|&&x| x == pad).count(), 4);
		assert_eq!(log.lock().unwrap().images, vec![(at, 1)]);
		// the next turn of that conversation: restored past the image
		let b = post(
			&l,
			"/v1/messages",
			&format!(
				r#"{{"max_tokens":16,"temperature":0,"messages":[{},{{"role":"assistant","content":"Red."}},{{"role":"user","content":"Sure?"}}]}}"#,
				turn(red)
			),
		);
		assert!(b.contains("\"text\":\"Red.\""), "{b}");
		{
			let g = log.lock().unwrap();
			assert_eq!(g.restores.len(), 1);
			assert_eq!(g.images.len(), 1);
		}
		// the same text around other pixels is another conversation
		let c = post(
			&l,
			"/v1/messages",
			&format!(
				r#"{{"max_tokens":16,"temperature":0,"messages":[{},{{"role":"assistant","content":"Red."}},{{"role":"user","content":"Sure?"}}]}}"#,
				turn(blue)
			),
		);
		assert!(c.contains("\"text\":\"Red.\""), "{c}");
		let g = log.lock().unwrap();
		assert_eq!(g.restores.len(), 1);
		assert_eq!(g.images, vec![(at, 1), (at, 1)]);
		// `/props` says so
		drop(g);
		let mut out = Vec::new();
		handle_request("GET", "/props", b"", Some(&l), &mut out).unwrap();
		assert!(String::from_utf8_lossy(&out).contains("\"vision\":true"));
	}

	/// Answers with `reply` through speculative verify steps of 4: it drafts
	/// the reply itself, with the second draft of every other step wrong.
	struct SpecScripted {
		reply: Vec<u32>,
		nv: usize,
		/// positions fed (committed), the position whose logits give reply[0]
		pos: usize,
		base: Option<usize>,
		pending: usize,
		verifies: Arc<Mutex<Vec<usize>>>,
	}

	impl SpecScripted {
		fn row(&self, q: usize) -> Vec<f32> {
			let mut v = vec![0.0; self.nv];
			v[self.reply[(q - self.base.unwrap()).min(self.reply.len() - 1)] as usize] = 1.0;
			v
		}
	}

	impl Engine for SpecScripted {
		fn reset(&mut self) -> Result<(), String> {
			(self.pos, self.base, self.pending) = (0, None, 0);
			Ok(())
		}

		fn forward(&mut self, _ids: &[u32]) -> Result<Vec<f32>, String> {
			Err("a drafting engine is driven through verify".into())
		}

		fn feed(&mut self, ids: &[u32]) -> Result<(), String> {
			assert_eq!(self.pending, 0, "feed with a verify step pending");
			self.pos += ids.len();
			Ok(())
		}

		fn spec_k(&self) -> usize {
			4
		}

		fn verify(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>, String> {
			assert!(!ids.is_empty() && ids.len() <= 4 && self.pending == 0);
			self.base.get_or_insert(self.pos);
			self.pending = ids.len();
			self.verifies.lock().unwrap().push(ids.len());
			Ok((0..ids.len()).map(|j| self.row(self.pos + j)).collect())
		}

		fn accept(&mut self, n: usize, next: Option<u32>) -> Result<Vec<u32>, String> {
			assert!(n >= 1 && n <= self.pending);
			self.pending = 0;
			self.pos += n;
			let g = self.pos - self.base.unwrap();
			assert_eq!(
				next.map(|t| t == self.reply[g - 1]),
				next.map(|_| true),
				"next is the reply token sampled last"
			);
			let mut d: Vec<u32> = self.reply[g.min(self.reply.len())..]
				.iter()
				.copied()
				.take(3)
				.collect();
			if self.verifies.lock().unwrap().len().is_multiple_of(2) && d.len() > 1 {
				d[1] = 0;
			}
			Ok(if next.is_some() { d } else { Vec::new() })
		}
	}

	#[test]
	fn engine_speculative_decoding_matches_the_reply() {
		let tok = byte_tokenizer();
		let reply = tok.encode("Hello speculative world.<|im_end|>");
		let nv = tok.vocab_size();
		let verifies = Arc::new(Mutex::new(Vec::new()));
		let info = EngineInfo {
			name: "spec".into(),
			model: DEFAULT_MODEL.into(),
			ctx: 4096,
			ctx_train: 4096,
			n_vocab: nv,
			n_embd: 8,
			stop: vec![tok.eos(), tok.token_id("<|endoftext|>").unwrap()],
			tag: "spec".into(),
			image: None,
		};
		let eng = SpecScripted {
			reply: reply.clone(),
			nv,
			pos: 0,
			base: None,
			pending: 0,
			verifies: verifies.clone(),
		};
		let l = Loaded::from_engine(Box::new(eng), info, tok);
		let a = post(
			&l,
			"/v1/messages",
			r#"{"max_tokens":64,"temperature":0,"messages":[{"role":"user","content":"hi"}]}"#,
		);
		assert!(a.contains("\"text\":\"Hello speculative world.\""), "{a}");
		assert!(a.contains("\"stop_reason\":\"end_turn\""), "{a}");
		// 24 reply tokens in far fewer verify steps (drafts rejected half the time)
		let v = verifies.lock().unwrap().clone();
		assert!(v.len() < 14 && v[0] == 1, "{v:?}");
		// max_tokens and stop sequences end it mid-step
		let r = post(
			&l,
			"/v1/messages",
			r#"{"max_tokens":7,"temperature":0,"messages":[{"role":"user","content":"hi"}]}"#,
		);
		assert!(
			r.contains("\"text\":\"Hello s\"") && r.contains("\"stop_reason\":\"max_tokens\""),
			"{r}"
		);
		let r = post(
			&l,
			"/v1/messages",
			r#"{"max_tokens":64,"temperature":0,"stop_sequences":["ive"],"messages":[{"role":"user","content":"hi"}]}"#,
		);
		assert!(
			r.contains("\"text\":\"Hello speculative\"")
				&& r.contains("\"stop_reason\":\"stop_sequence\""),
			"{r}"
		);
	}

	fn post(l: &Loaded, path: &str, body: &str) -> String {
		let mut out: Vec<u8> = Vec::new();
		handle_request("POST", path, body.as_bytes(), Some(l), &mut out).unwrap();
		String::from_utf8_lossy(&out).into_owned()
	}

	#[test]
	fn engine_resets_for_every_request() {
		let (l, log) = engine_server("Paris.<|im_end|>", 4096);
		let req = r#"{"max_tokens":32,"temperature":0,"messages":[{"role":"user","content":"Capital of France?"}]}"#;
		let a = post(&l, "/v1/messages", req);
		let b = post(&l, "/v1/messages", req);
		assert!(a.contains("\"text\":\"Paris.\""), "{a}");
		assert!(a.contains("\"stop_reason\":\"end_turn\""), "{a}");
		assert!(b.contains("\"text\":\"Paris.\""), "{b}");
		let g = log.lock().unwrap();
		assert_eq!(g.resets, 2);
		// Each request: the whole prompt in one call, then one call per
		// generated token but the last (`P`, `a`, ..., `.`: 6 tokens, EOS ends).
		let n = l.tok.encode(&Tokenizer::build_prompt(
			&[Turn::User("Capital of France?".into())],
			&[],
			Tokenizer::ASSISTANT_OPEN_NOTHINK,
		));
		assert_eq!(
			g.calls,
			[vec![n.len()], vec![1; 6], vec![n.len()], vec![1; 6]].concat()
		);
		assert!(a.contains(&format!("\"input_tokens\":{}", n.len())), "{a}");
	}

	#[test]
	fn engine_streams_and_stops() {
		let (l, log) = engine_server("Hello world<|endoftext|>", 4096);
		let req = r#"{"max_tokens":32,"temperature":0,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
		let s = post(&l, "/v1/messages", req);
		assert!(s.contains("event: message_start"), "{s}");
		assert!(s.contains("\"text_delta\",\"text\":\"H\""), "{s}");
		assert!(s.contains("\"stop_reason\":\"end_turn\""), "{s}");
		assert!(s.contains("event: message_stop"), "{s}");
		// A stop sequence ends it early; `max_tokens` too, without feeding
		// the last token.
		let r = post(
			&l,
			"/v1/messages",
			r#"{"max_tokens":32,"temperature":0,"stop_sequences":["lo"],"messages":[{"role":"user","content":"hi"}]}"#,
		);
		assert!(
			r.contains("\"text\":\"Hello\"") && r.contains("\"stop_reason\":\"stop_sequence\""),
			"{r}"
		);
		let r = post(
			&l,
			"/v1/messages",
			r#"{"max_tokens":3,"temperature":0,"messages":[{"role":"user","content":"hi"}]}"#,
		);
		assert!(
			r.contains("\"text\":\"Hel\"") && r.contains("\"stop_reason\":\"max_tokens\""),
			"{r}"
		);
		let n = l
			.tok
			.encode(&Tokenizer::build_prompt(
				&[Turn::User("hi".into())],
				&[],
				Tokenizer::ASSISTANT_OPEN_NOTHINK,
			))
			.len();
		assert_eq!(log.lock().unwrap().fed, n + 2);
		// OpenAI, streamed.
		let o = post(
			&l,
			"/v1/chat/completions",
			r#"{"stream":true,"temperature":0,"messages":[{"role":"user","content":"hi"}]}"#,
		);
		assert!(
			o.contains("\"finish_reason\":\"stop\"") && o.contains("data: [DONE]"),
			"{o}"
		);
	}

	#[test]
	fn engine_tool_call_and_refusals() {
		let call = "<tool_call>\n<function=Bash>\n<parameter=command>\nls /tmp\n</parameter>\n</function>\n</tool_call><|im_end|>";
		let (l, _) = engine_server(call, 4096);
		let req = r#"{"max_tokens":200,"temperature":0,"tools":[{"name":"Bash","description":"run","input_schema":{"type":"object","properties":{"command":{"type":"string"}}}}],"messages":[{"role":"user","content":"list /tmp"}]}"#;
		let r = post(&l, "/v1/messages", req);
		assert!(
			r.contains("\"type\":\"tool_use\"") && r.contains("\"name\":\"Bash\""),
			"{r}"
		);
		assert!(
			r.contains("\"command\":\"ls /tmp\"") && r.contains("\"stop_reason\":\"tool_use\""),
			"{r}"
		);
		// No vision tower: an image block is a 400.
		let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
		let img = format!(
			r#"{{"max_tokens":8,"messages":[{{"role":"user","content":[{{"type":"image","source":{{"type":"base64","media_type":"image/png","data":"{png}"}}}}]}}]}}"#
		);
		let r = post(&l, "/v1/messages", &img);
		assert!(
			r.starts_with("HTTP/1.1 400") && r.contains("no vision tower"),
			"{r}"
		);
		// A prompt that does not fit the context is a 400 before the engine runs.
		let (small, log) = engine_server("x", 8);
		let r = post(
			&small,
			"/v1/messages",
			r#"{"max_tokens":8,"messages":[{"role":"user","content":"a long enough prompt"}]}"#,
		);
		assert!(r.starts_with("HTTP/1.1 400") && r.contains("--ctx"), "{r}");
		assert_eq!(log.lock().unwrap().resets, 0);
		// `/props` and the model `meta` come from the engine's info.
		let mut out = Vec::new();
		handle_request("GET", "/props", b"", Some(&small), &mut out).unwrap();
		assert!(String::from_utf8_lossy(&out).contains("\"n_ctx\":8"));
	}
}
