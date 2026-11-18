//! GGUF tokenizer for the qwen.rs project: byte-level BPE (GPT-2 style) with
//! the `qwen35` pre-tokenizer, plus special-token handling and a minimal chat
//! prompt builder for Qwen3.8.
//!
//! Reference: llama.cpp `src/llama-vocab.cpp`
//! (`LLAMA_VOCAB_PRE_TYPE_QWEN35` regex + `LLAMA_VOCAB_TYPE_BPE` byte-level
//! encoding) and `convert_hf_to_gguf.py` for the GGUF `tokenizer.ggml.*` keys.

use std::collections::HashMap;

use gguf::{GgufFile, MetadataValue};

mod hfjson;

/// GPT-2 byte-level translation: maps each of the 256 byte values to a unique
/// printable Unicode code point so that BPE can operate on single "characters"
/// losslessly. Space (0x20) maps to U+0120 (`Ġ`).
fn bytes_to_unicode() -> (Vec<char>, HashMap<char, u8>) {
	let mut bs: Vec<u8> = (33..=126).collect();
	bs.extend(161..=172);
	bs.extend(174..=255);

	let mut cs: Vec<u32> = bs.iter().map(|b| *b as u32).collect();
	let mut n = 0u32;
	for b in 0u32..=255 {
		if !bs.contains(&(b as u8)) {
			bs.push(b as u8);
			cs.push(256 + n);
			n += 1;
		}
	}

	let mut encoder = vec!['\0'; 256];
	let mut decoder = HashMap::new();
	for (&b, &c) in bs.iter().zip(cs.iter()) {
		let ch = char::from_u32(c).expect("byte->unicode code point is valid");
		encoder[b as usize] = ch;
		decoder.insert(ch, b);
	}
	(encoder, decoder)
}

fn meta_u32(g: &GgufFile, key: &str) -> Option<u32> {
	g.metadata
		.iter()
		.find(|e| e.key == key)
		.and_then(|e| match &e.value {
			MetadataValue::U32(v) => Some(*v),
			MetadataValue::I32(v) => Some(*v as u32),
			_ => None,
		})
}

fn meta_bool(g: &GgufFile, key: &str) -> Option<bool> {
	g.metadata
		.iter()
		.find(|e| e.key == key)
		.and_then(|e| match &e.value {
			MetadataValue::Bool(b) => Some(*b),
			_ => None,
		})
}

fn array_i32(g: &GgufFile, key: &str) -> Option<Vec<i32>> {
	g.metadata
		.iter()
		.find(|e| e.key == key)
		.and_then(|e| match &e.value {
			MetadataValue::Array { values, .. } => {
				let mut out = Vec::with_capacity(values.len());
				for v in values {
					match v {
						MetadataValue::I32(n) => out.push(*n),
						MetadataValue::U32(n) => out.push(*n as i32),
						_ => return None,
					}
				}
				Some(out)
			}
			_ => None,
		})
}

fn array_strings(g: &GgufFile, key: &str) -> Option<Vec<String>> {
	g.metadata
		.iter()
		.find(|e| e.key == key)
		.and_then(|e| match &e.value {
			MetadataValue::Array { values, .. } => {
				let mut out = Vec::with_capacity(values.len());
				for v in values {
					match v {
						MetadataValue::String(s) => out.push(s.clone()),
						_ => return None,
					}
				}
				Some(out)
			}
			_ => None,
		})
}

pub struct Tokenizer {
	vocab: Vec<String>,
	token_to_id: HashMap<String, u32>,
	merges: HashMap<(String, String), usize>,
	byte_encoder: Vec<char>,
	byte_decoder: HashMap<char, u8>,
	special: Vec<String>, // exact special-token strings, longest first
	special_ids: std::collections::HashSet<u32>,
	eos: u32,
	bos: u32,
	add_bos: bool,
	unk: Option<u32>,
	pretok: fancy_regex::Regex,
}

enum Segment {
	Text(String),
	Special(u32), // token id
}

impl Tokenizer {
	pub fn from_gguf(g: &GgufFile) -> Result<Tokenizer, String> {
		let vocab = array_strings(g, "tokenizer.ggml.tokens")
			.ok_or_else(|| "missing tokenizer.ggml.tokens".to_string())?;
		let merges = array_strings(g, "tokenizer.ggml.merges").unwrap_or_default();
		let eos = meta_u32(g, "tokenizer.ggml.eos_token_id")
			.ok_or_else(|| "missing eos_token_id".to_string())?;
		let bos = meta_u32(g, "tokenizer.ggml.bos_token_id").unwrap_or(eos);
		let add_bos = meta_bool(g, "tokenizer.ggml.add_bos_token").unwrap_or(false);
		let unk = meta_u32(g, "tokenizer.ggml.unknown_token_id");
		let types = array_i32(g, "tokenizer.ggml.token_type");
		Self::from_parts(vocab, merges, types, eos, bos, add_bos, unk)
	}

	/// The tokenizer of a Hugging Face `tokenizer.json` (and the
	/// `tokenizer_config.json` beside it, for the EOS token), as shipped with
	/// an OpenVINO export. Same byte-level BPE and pre-tokenizer as the GGUF
	/// one; the token types come from `added_tokens`: `special: true` is
	/// CONTROL (hidden on decode), `special: false` (`<tool_call>`,
	/// `<think>`, …) USER_DEFINED.
	pub fn from_hf_json(path: &std::path::Path) -> Result<Tokenizer, String> {
		let hf = hfjson::load(path)?;
		let eos_name = std::fs::read_to_string(path.with_file_name("tokenizer_config.json"))
			.ok()
			.and_then(|s| hfjson::string_field(&s, "eos_token"))
			.unwrap_or_else(|| "<|im_end|>".to_string());
		let n = hf.vocab.len();
		let id_of = |s: &str| hf.vocab.iter().position(|v| v == s).map(|i| i as u32);
		let eos = id_of(&eos_name).ok_or_else(|| format!("eos token {eos_name} not in vocab"))?;
		debug_assert_eq!(hf.types.len(), n);
		Self::from_parts(hf.vocab, hf.merges, Some(hf.types), eos, eos, false, None)
	}

	#[allow(clippy::too_many_arguments)]
	fn from_parts(
		vocab: Vec<String>,
		merges: Vec<String>,
		types: Option<Vec<i32>>,
		eos: u32,
		bos: u32,
		add_bos: bool,
		unk: Option<u32>,
	) -> Result<Tokenizer, String> {
		let mut token_to_id = HashMap::new();
		for (i, t) in vocab.iter().enumerate() {
			token_to_id.entry(t.clone()).or_insert(i as u32);
		}

		let mut merge_map = HashMap::new();
		for (rank, m) in merges.iter().enumerate() {
			// format: "LEFT RIGHT" (single space separator; byte-level tokens
			// never contain a literal space, only the U+0120 glyph).
			if let Some((a, b)) = m.split_once(' ') {
				merge_map.insert((a.to_string(), b.to_string()), rank);
			}
		}

		let (byte_encoder, byte_decoder) = bytes_to_unicode();

		// Tokens that must be matched whole and bypass BPE.
		//
		// Matching "vocab entries shaped like `<|name|>`" finds only the
		// 27 CONTROL tokens of this vocab and **misses all six USER_DEFINED
		// ones**: `<tool_call>`, `</tool_call>`, `<tool_response>`,
		// `</tool_response>`, `<think>` and `</think>`. Every one of those is a
		// single id the model emits and expects to read back, and BPE splits
		// each into three or four pieces: `<think>` became `<th` + `ink` + `>`.
		//
		// The failure mode was not a wrong answer, it was *no* answer: a prompt
		// ending in the template's own `<think>\n\n</think>\n\n` opener put the
		// model so far outside its distribution that it emitted EOS as the very
		// first token, every time. The same split silently degraded every
		// replayed tool call and tool result, where it produced no visible
		// symptom at all.
		//
		// GGUF token types (llama.cpp's `llama_token_attr`): 1 NORMAL,
		// 2 UNKNOWN, 3 CONTROL, 4 USER_DEFINED, 5 UNUSED, 6 BYTE. CONTROL and
		// USER_DEFINED are the two that are matched literally; UNUSED is the
		// `[PAD…]` filler and must not be. The name-shape rule stays as the
		// fallback for a GGUF that ships no `token_type` array.
		let by_type = |keep: &dyn Fn(i32) -> bool| -> Vec<String> {
			match &types {
				Some(ty) if ty.len() == vocab.len() => vocab
					.iter()
					.zip(ty.iter())
					.filter(|(_, t)| keep(**t))
					.map(|(v, _)| v.clone())
					.collect(),
				_ => vocab
					.iter()
					.filter(|t| t.starts_with("<|") && t.ends_with("|>"))
					.cloned()
					.collect(),
			}
		};
		let mut special: Vec<String> = by_type(&|t| t == 3 || t == 4);
		// longest first
		special.sort_by_key(|s| std::cmp::Reverse(s.len()));
		// **Only CONTROL tokens are hidden on decode.** The two sets are not the
		// same set and conflating them breaks tool use in the opposite
		// direction: `<tool_call>` and `</think>` are USER_DEFINED, so they must
		// be matched whole when encoding *and* still appear in the decoded text,
		// because that text is what the tool-call parser reads. Skipping them
		// would delete the very markers it looks for and every call would come
		// back as prose.
		let special_ids: std::collections::HashSet<u32> = by_type(&|t| t == 3)
			.iter()
			.filter_map(|s| token_to_id.get(s).copied())
			.collect();

		let pretok = fancy_regex::Regex::new(
			r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
		)
		.map_err(|e| format!("pretokenizer regex: {e}"))?;

		Ok(Tokenizer {
			vocab,
			token_to_id,
			merges: merge_map,
			byte_encoder,
			byte_decoder,
			special,
			special_ids,
			eos,
			bos,
			add_bos,
			unk,
			pretok,
		})
	}

	pub fn eos(&self) -> u32 {
		self.eos
	}
	pub fn bos(&self) -> u32 {
		self.bos
	}
	pub fn vocab_size(&self) -> usize {
		self.vocab.len()
	}

	/// Id of an exact vocabulary entry, e.g. a control token like
	/// `"<|image_pad|>"`.
	pub fn token_id(&self, s: &str) -> Option<u32> {
		self.token_to_id.get(s).copied()
	}

	fn is_special_id(&self, id: u32) -> bool {
		self.special_ids.contains(&id)
	}

	/// Split `text` into text and special-token segments. Specials are matched
	/// as exact (ASCII) substrings so they bypass BPE entirely.
	fn split_specials(&self, text: &str) -> Vec<Segment> {
		if self.special.is_empty() {
			return vec![Segment::Text(text.to_string())];
		}
		let mut out = Vec::new();
		let mut seg_start = 0usize;
		let mut idx = 0usize;
		while idx < text.len() {
			let mut matched: Option<&String> = None;
			for sp in &self.special {
				if text[idx..].starts_with(sp.as_str()) {
					matched = Some(sp);
					break;
				}
			}
			if let Some(sp) = matched {
				if idx > seg_start {
					out.push(Segment::Text(text[seg_start..idx].to_string()));
				}
				if let Some(&id) = self.token_to_id.get(sp) {
					out.push(Segment::Special(id));
				} else {
					out.push(Segment::Text(sp.clone()));
				}
				idx += sp.len();
				seg_start = idx;
			} else {
				// advance by one UTF-8 character
				let step = text[idx..]
					.chars()
					.next()
					.map(|c| c.len_utf8())
					.unwrap_or(1);
				idx += step;
			}
		}
		if seg_start < text.len() {
			out.push(Segment::Text(text[seg_start..].to_string()));
		}
		out
	}

	/// Pre-tokenize one normal-text segment into pieces (the qwen35 regex).
	fn pretokenize(&self, text: &str) -> Vec<String> {
		let mut out = Vec::new();
		for m in self.pretok.find_iter(text).flatten() {
			out.push(text[m.start()..m.end()].to_string());
		}
		out
	}

	/// Byte-level BPE on a single pre-token (already a `&str` of byte-level
	/// glyphs). Returns the merged symbols in order.
	fn bpe(&self, piece: &str) -> Vec<String> {
		let mut word: Vec<String> = piece.chars().map(|c| c.to_string()).collect();
		if word.len() <= 1 {
			return word;
		}
		loop {
			let mut best_rank = usize::MAX;
			let mut best_i = None;
			for i in 0..word.len() - 1 {
				let rank = self
					.merges
					.get(&(word[i].clone(), word[i + 1].clone()))
					.copied()
					.unwrap_or(usize::MAX);
				if rank < best_rank {
					best_rank = rank;
					best_i = Some(i);
				}
			}
			let Some(i) = best_i else { break };
			word[i] = word[i].clone() + &word[i + 1];
			word.remove(i + 1);
			if word.len() <= 1 {
				break;
			}
		}
		word
	}

	/// Encode `text` to a sequence of token ids.
	pub fn encode(&self, text: &str) -> Vec<u32> {
		let mut ids = Vec::new();
		if self.add_bos {
			ids.push(self.bos);
		}
		for seg in self.split_specials(text) {
			match seg {
				Segment::Special(id) => ids.push(id),
				Segment::Text(s) => self.encode_plain_into(&s, &mut ids),
			}
		}
		ids
	}

	/// Encode `text` as ordinary text: no special token is matched (a literal
	/// `<|image_pad|>` becomes the BPE pieces of its characters) and no BOS is
	/// added. For text that names a special token rather than using it.
	pub fn encode_plain(&self, text: &str) -> Vec<u32> {
		let mut ids = Vec::new();
		self.encode_plain_into(text, &mut ids);
		ids
	}

	fn encode_plain_into(&self, text: &str, ids: &mut Vec<u32>) {
		for piece in self.pretokenize(text) {
			// byte-level encode the piece
			let mut bl = String::new();
			for &b in piece.as_bytes() {
				bl.push(self.byte_encoder[b as usize]);
			}
			for sym in self.bpe(&bl) {
				match self.token_to_id.get(&sym) {
					Some(&id) => ids.push(id),
					None => {
						if let Some(unk) = self.unk {
							ids.push(unk);
						}
					}
				}
			}
		}
	}

	/// Decode token ids back to a string. When `skip_specials` is true, control
	/// tokens (`<|...|>`) are omitted.
	///
	/// Lossy: a token whose bytes end mid-character becomes U+FFFD. For
	/// streaming output use [`StreamDecoder`], which carries the partial
	/// character across steps instead.
	pub fn decode(&self, ids: &[u32], skip_specials: bool) -> String {
		String::from_utf8_lossy(&self.decode_bytes(ids, skip_specials)).into_owned()
	}

	/// Decode to raw bytes, without the lossy UTF-8 conversion.
	pub fn decode_bytes(&self, ids: &[u32], skip_specials: bool) -> Vec<u8> {
		let mut bytes: Vec<u8> = Vec::new();
		for &id in ids {
			let Some(tok) = self.vocab.get(id as usize) else {
				continue;
			};
			if skip_specials && self.is_special_id(id) {
				continue;
			}
			let mut all_byte = true;
			for ch in tok.chars() {
				if let Some(&b) = self.byte_decoder.get(&ch) {
					bytes.push(b);
				} else {
					all_byte = false;
					break;
				}
			}
			if !all_byte {
				// control / literal token: keep its UTF-8 bytes verbatim
				bytes.extend_from_slice(tok.as_bytes());
			}
		}
		bytes
	}

	/// Build a Qwen3 chat prompt from `messages` (`(role, content)` pairs).
	/// Renders the standard `<|im_start|>role\ncontent<|im_end|>\n` framing and
	/// leaves the final `<|im_start|>assistant\n` open for generation.
	pub fn build_chat_prompt(&self, messages: &[(&str, &str)]) -> String {
		let mut s = String::new();
		for (role, content) in messages {
			s.push_str(&Self::chat_turn(role, content));
		}
		s.push_str(Self::ASSISTANT_OPEN);
		s
	}

	/// One rendered `<|im_start|>role\ncontent<|im_end|>\n` turn.
	///
	/// Split out of [`Tokenizer::build_chat_prompt`] so a multi-turn session
	/// can append the turns it has not fed yet, instead of re-rendering (and
	/// re-prefilling) the whole conversation every round.
	pub fn chat_turn(role: &str, content: &str) -> String {
		format!("<|im_start|>{role}\n{content}<|im_end|>\n")
	}

	/// The open assistant header a prompt ends on, ready for generation.
	pub const ASSISTANT_OPEN: &'static str = "<|im_start|>assistant\n";

	/// The assistant header with an **empty** thinking block already closed.
	///
	/// This model's template opens `<think>\n` in its generation prompt unless
	/// `enable_thinking` is false, in which case it emits exactly this. Leaving
	/// the block open makes the model reason first, which is the right default
	/// for a chat UI and the wrong one for an API caller that has to parse the
	/// reply - and at this machine's decode rate a few hundred tokens of
	/// reasoning per turn is most of the wall clock.
	pub const ASSISTANT_OPEN_NOTHINK: &'static str =
		"<|im_start|>assistant\n<think>\n\n</think>\n\n";

	/// A plain system turn as the template renders it under `think`: the
	/// reasoning-effort instruction first, then the caller's text, a blank line
	/// between; with no text, the instruction alone; with neither, nothing.
	pub fn system_turn(sys: Option<&str>, think: Thinking) -> String {
		let sys = sys.map(str::trim).filter(|t| !t.is_empty());
		let instr = think.instruction();
		match (sys, instr.is_empty()) {
			(Some(t), true) => Self::chat_turn("system", t),
			(Some(t), false) => Self::chat_turn("system", &format!("{instr}\n\n{t}")),
			(None, false) => Self::chat_turn("system", instr),
			(None, true) => String::new(),
		}
	}

	/// Build a prompt from `turns`, optionally with tool definitions.
	///
	/// Mirrors `tokenizer.chat_template` in the GGUF (extracted and read, not
	/// guessed). The parts that matter and are easy to get subtly wrong:
	///
	/// * **Tools live in the system turn**, not a separate one, and the system
	///   text the caller supplied is appended *after* the tool preamble with a
	///   blank line between. A tools request therefore always has a system turn
	///   even when the caller sent no system prompt.
	/// * **Tool results are `user` turns**, not a `tool` role in the rendered
	///   text: a run of consecutive results shares one `<|im_start|>user` frame,
	///   each wrapped in `<tool_response>`.
	/// * **The call format is XML-ish, not JSON.** Qwen3.8 wants
	///   `<function=name><parameter=key>` + newline + value inside
	///   `<tool_call>`; the `{"name":...,"arguments":...}` JSON body older Qwen
	///   releases used is not what this template describes, and a model
	///   prompted in one format will not reliably answer in the other.
	///
	/// Thinking off (`enable_thinking: false`) and `preserve_thinking` on; see
	/// [`Tokenizer::build_prompt_with`]. `open` is the text after the turns
	/// (the generation prompt, or nothing).
	pub fn build_prompt(turns: &[Turn], tools: &[String], open: &str) -> String {
		Self::build_prompt_with(turns, tools, Thinking::OFF, open)
	}

	/// [`Tokenizer::build_prompt`] with the template's thinking switches.
	///
	/// * With thinking on, the effort's instruction ([`Effort::instruction`])
	///   opens the system turn - ahead of the tool preamble, or of the caller's
	///   system text - and makes a system turn of its own when there is none.
	/// * A past assistant turn carries `<think>\n{reasoning}\n</think>\n\n`
	///   (empty when there is none) when `preserve` is on or it follows the
	///   last real user query (an agent's tool loop in progress); otherwise no
	///   thinking block at all.
	///
	/// The generation prompt is `open`: [`Thinking::open`] plus whatever the
	/// caller seeds the reply with.
	pub fn build_prompt_with(
		turns: &[Turn],
		tools: &[String],
		think: Thinking,
		open: &str,
	) -> String {
		let mut s = String::new();
		// System turn. With tools it carries the tool preamble; the caller's own
		// system text follows it.
		let sys = turns.iter().find_map(|t| match t {
			Turn::System(text) => Some(text.trim()),
			_ => None,
		});
		if !tools.is_empty() {
			s.push_str("<|im_start|>system\n");
			let instr = think.instruction();
			if !instr.is_empty() {
				s.push_str(instr);
				s.push_str("\n\n");
			}
			s.push_str(TOOL_PREAMBLE_HEAD);
			for t in tools {
				s.push('\n');
				s.push_str(t);
			}
			s.push_str(TOOL_PREAMBLE_TAIL);
			if let Some(sys) = sys.filter(|t| !t.is_empty()) {
				s.push_str("\n\n");
				s.push_str(sys);
			}
			s.push_str("<|im_end|>\n");
		} else {
			s.push_str(&Self::system_turn(sys, think));
		}

		// The template's `last_query_index`: the last user turn that is not
		// just tool results. Assistant turns after it keep their reasoning
		// whatever `preserve` says.
		let last_query = turns.iter().rposition(|t| matches!(t, Turn::User(_)));
		let mut i = 0;
		while i < turns.len() {
			match &turns[i] {
				Turn::System(_) => {}
				Turn::User(text) => s.push_str(&Self::chat_turn("user", text.trim())),
				Turn::Assistant {
					text,
					reasoning,
					calls,
				} => {
					// The thinking block is not decoration: the template emits
					// one for every past assistant turn (`preserve_thinking`
					// defaults to true), empty when there is no reasoning to
					// replay, and that is also exactly what this prompt's own
					// generation opening looks like with thinking off. Leaving
					// it out would make replayed turns a different shape from
					// generated ones.
					s.push_str("<|im_start|>assistant\n");
					if think.preserve || last_query.is_none_or(|q| i > q) {
						s.push_str("<think>\n");
						s.push_str(reasoning.trim());
						s.push_str("\n</think>\n\n");
					}
					let text = text.trim();
					s.push_str(text);
					for (n, c) in calls.iter().enumerate() {
						// The template puts a blank line between prose and the
						// first call, and a single newline between calls.
						if n == 0 {
							if !text.is_empty() {
								s.push_str("\n\n");
							}
						} else {
							s.push('\n');
						}
						s.push_str(&c.render());
					}
					s.push_str("<|im_end|>\n");
				}
				Turn::ToolResults(_) => {
					// Consume the whole run: they share one user frame.
					s.push_str("<|im_start|>user");
					while let Some(Turn::ToolResults(results)) = turns.get(i) {
						for r in results {
							s.push_str("\n<tool_response>\n");
							s.push_str(r.trim());
							s.push_str("\n</tool_response>");
						}
						i += 1;
					}
					s.push_str("<|im_end|>\n");
					continue;
				}
			}
			i += 1;
		}
		s.push_str(open);
		s
	}
}

/// The template's `reasoning_effort` levels (Qwen3.8; the 27B GGUF and
/// Flash-Next ship the same template). The template's default is `Xhigh`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Effort {
	Low,
	Medium,
	Xhigh,
}

impl Effort {
	/// `low`, `medium`, `xhigh`: the template's own names, nothing else.
	pub fn parse(s: &str) -> Option<Effort> {
		match s {
			"low" => Some(Effort::Low),
			"medium" => Some(Effort::Medium),
			"xhigh" => Some(Effort::Xhigh),
			_ => None,
		}
	}

	pub fn name(self) -> &'static str {
		match self {
			Effort::Low => "low",
			Effort::Medium => "medium",
			Effort::Xhigh => "xhigh",
		}
	}

	/// What the template puts at the top of the system turn, verbatim. Medium
	/// has none: it is the model's unprompted thinking.
	pub fn instruction(self) -> &'static str {
		match self {
			Effort::Xhigh => "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
			Effort::Medium => "",
			Effort::Low => "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
		}
	}
}

/// The template's thinking switches for one prompt: `enable_thinking` +
/// `reasoning_effort` (`effort`, `None` = thinking off) and
/// `preserve_thinking`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Thinking {
	pub effort: Option<Effort>,
	pub preserve: bool,
}

impl Thinking {
	/// `enable_thinking: false`.
	pub const OFF: Thinking = Thinking {
		effort: None,
		preserve: true,
	};

	pub fn on(effort: Effort) -> Thinking {
		Thinking {
			effort: Some(effort),
			preserve: true,
		}
	}

	/// `off`, or an [`Effort`] name.
	pub fn parse(s: &str) -> Option<Thinking> {
		match s {
			"off" | "none" => Some(Thinking::OFF),
			_ => Effort::parse(s).map(Thinking::on),
		}
	}

	pub fn enabled(self) -> bool {
		self.effort.is_some()
	}

	/// The system turn's instruction (empty when off, or at medium).
	pub fn instruction(self) -> &'static str {
		self.effort.map_or("", Effort::instruction)
	}

	/// The generation prompt: the assistant header with `<think>\n` left open
	/// for the model, or with an empty block already closed when off.
	pub fn open(self) -> &'static str {
		if self.enabled() {
			"<|im_start|>assistant\n<think>\n"
		} else {
			Tokenizer::ASSISTANT_OPEN_NOTHINK
		}
	}
}

/// A rendered conversation turn, in the shape the template consumes.
pub enum Turn {
	/// Only the first is used, and only its text; the template refuses a system
	/// message anywhere but the front.
	System(String),
	User(String),
	Assistant {
		text: String,
		/// The turn's thinking, replayed inside its `<think>` block (empty:
		/// none, or the client did not send it back).
		reasoning: String,
		calls: Vec<ToolCall>,
	},
	/// One or more tool results. Consecutive `ToolResults` turns are merged into
	/// a single user frame, as the template does.
	ToolResults(Vec<String>),
}

/// One function call, with its arguments already rendered to the text the
/// template writes for them (strings verbatim, everything else as JSON).
pub struct ToolCall {
	pub name: String,
	pub args: Vec<(String, String)>,
}

impl ToolCall {
	/// The `<tool_call>` / `<function=>` / `<parameter=>` nest the template
	/// specifies, with a newline around every value.
	pub fn render(&self) -> String {
		let mut s = String::new();
		s.push_str("<tool_call>\n<function=");
		s.push_str(&self.name);
		s.push_str(">\n");
		for (k, v) in &self.args {
			s.push_str("<parameter=");
			s.push_str(k);
			s.push_str(">\n");
			s.push_str(v);
			s.push_str("\n</parameter>\n");
		}
		s.push_str("</function>\n</tool_call>");
		s
	}
}

/// Everything in the tools system turn before the tool JSON list.
const TOOL_PREAMBLE_HEAD: &str =
	"# Tools\n\nYou have access to the following functions:\n\n<tools>";

/// Everything after it, verbatim from the GGUF's `tokenizer.chat_template`.
/// Copied exactly: the model was trained against this wording, and paraphrasing
/// a format description is the kind of edit that degrades tool-call reliability
/// without failing anything.
const TOOL_PREAMBLE_TAIL: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// Incremental decoder for token-at-a-time output.
///
/// Byte-level BPE splits multi-byte characters across tokens: an emoji is
/// routinely four tokens of one byte each. Decoding each token on its own and
/// printing it turns every one of those into `U+FFFD`, which is what the
/// streamed output actually showed. This holds the incomplete tail back until
/// the bytes that finish the character arrive.
#[derive(Default)]
pub struct StreamDecoder {
	pending: Vec<u8>,
}

impl StreamDecoder {
	pub fn new() -> StreamDecoder {
		StreamDecoder {
			pending: Vec::new(),
		}
	}

	/// Feed one token; returns whatever text is now complete (possibly empty).
	pub fn push(&mut self, tok: &Tokenizer, id: u32, skip_specials: bool) -> String {
		self.pending
			.extend_from_slice(&tok.decode_bytes(&[id], skip_specials));
		match std::str::from_utf8(&self.pending) {
			Ok(s) => {
				let out = s.to_string();
				self.pending.clear();
				out
			}
			Err(e) => {
				let good = e.valid_up_to();
				// An error *with* an error_len is a genuinely invalid sequence,
				// not a truncated one; emit it lossily so a bad token cannot
				// wedge the stream forever.
				let take = match e.error_len() {
					Some(l) => good + l,
					None => good,
				};
				let out = String::from_utf8_lossy(&self.pending[..take]).into_owned();
				self.pending.drain(..take);
				out
			}
		}
	}

	/// Anything still buffered at the end of a generation, rendered lossily.
	pub fn flush(&mut self) -> String {
		let out = String::from_utf8_lossy(&self.pending).into_owned();
		self.pending.clear();
		out
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Build a tiny standalone BPE vocab to exercise encode/decode without a
	/// real GGUF. We use the real GPT-2 byte mapping so spaces become 'Ġ'.
	fn tiny_tok() -> Tokenizer {
		// bytes_to_unicode for space -> U+0120 ('Ġ')
		let (enc, dec) = bytes_to_unicode();
		assert_eq!(enc[b' ' as usize], '\u{0120}');

		// vocab: single chars + a couple of merges
		let mut vocab = Vec::new();
		// include byte-level chars we need
		for ch in [
			'\u{0120}', 'h', 'e', 'l', 'o', 'w', 'r', 'd', 't', 'n', 'a', 'i', 's', '\'', '!',
		] {
			vocab.push(ch.to_string());
		}
		// merged tokens
		vocab.push("\u{0120}hello".to_string()); // " hello"
		vocab.push("world".to_string());

		let mut token_to_id = HashMap::new();
		for (i, t) in vocab.iter().enumerate() {
			token_to_id.entry(t.clone()).or_insert(i as u32);
		}
		let mut merges = HashMap::new();
		merges.insert(("\u{0120}".to_string(), "hello".to_string()), 0usize);
		merges.insert(("w".to_string(), "orld".to_string()), 1usize);

		Tokenizer {
			vocab,
			token_to_id,
			merges,
			byte_encoder: enc,
			byte_decoder: dec,
			special: vec![],
			special_ids: std::collections::HashSet::new(),
			eos: 0,
			bos: 0,
			add_bos: false,
			unk: None,
			pretok: fancy_regex::Regex::new(
				r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
			)
			.unwrap(),
		}
	}

	#[test]
	fn byte_level_space() {
		let (enc, _dec) = bytes_to_unicode();
		assert_eq!(enc[b' ' as usize], '\u{0120}');
		assert_eq!(enc[b'h' as usize], 'h');
	}

	/// Byte-level BPE emits a multi-byte character as several single-byte
	/// tokens, so a streaming decoder that renders each token on its own turns
	/// every one of them into U+FFFD. This is that case.
	#[test]
	fn stream_decoder_joins_a_split_multibyte_character() {
		let (enc, dec) = bytes_to_unicode();
		// U+00E9 is the two bytes 0xC3 0xA9; both map to themselves in the
		// GPT-2 byte alphabet, so each is its own vocab entry here.
		let vocab = vec!["\u{00c3}".to_string(), "\u{00a9}".to_string()];
		let mut token_to_id = HashMap::new();
		for (i, v) in vocab.iter().enumerate() {
			token_to_id.insert(v.clone(), i as u32);
		}
		let t = Tokenizer {
			vocab,
			token_to_id,
			merges: HashMap::new(),
			byte_encoder: enc,
			byte_decoder: dec,
			special: vec![],
			special_ids: std::collections::HashSet::new(),
			eos: 0,
			bos: 0,
			add_bos: false,
			unk: None,
			pretok: fancy_regex::Regex::new(r"\p{L}+|\s+").unwrap(),
		};
		let mut sd = StreamDecoder::new();
		assert_eq!(
			sd.push(&t, 0, true),
			"",
			"the lead byte alone is not a character yet"
		);
		assert_eq!(sd.push(&t, 1, true), "\u{00e9}");
		assert_eq!(sd.flush(), "");
		// Streaming and one-shot decoding must agree.
		assert_eq!(t.decode(&[0, 1], true), "\u{00e9}");
	}

	/// A byte that can never start a character must not wedge the stream.
	#[test]
	fn stream_decoder_does_not_stall_on_an_invalid_byte() {
		let (enc, dec) = bytes_to_unicode();
		let vocab = vec!["\u{00a9}".to_string()]; // 0xA9: a lone continuation byte
		let mut token_to_id = HashMap::new();
		token_to_id.insert(vocab[0].clone(), 0u32);
		let t = Tokenizer {
			vocab,
			token_to_id,
			merges: HashMap::new(),
			byte_encoder: enc,
			byte_decoder: dec,
			special: vec![],
			special_ids: std::collections::HashSet::new(),
			eos: 0,
			bos: 0,
			add_bos: false,
			unk: None,
			pretok: fancy_regex::Regex::new(r"\p{L}+|\s+").unwrap(),
		};
		let mut sd = StreamDecoder::new();
		assert_eq!(sd.push(&t, 0, true), "\u{fffd}");
		assert_eq!(sd.flush(), "");
	}

	#[test]
	fn encode_decode_roundtrip() {
		let t = tiny_tok();
		let text = "hello world";
		let ids = t.encode(text);
		assert!(!ids.is_empty());
		// without an explicit merge chain, each byte-level char is its own token,
		// but decoding must reconstruct the original text exactly.
		let back = t.decode(&ids, true);
		assert_eq!(back, text);
	}

	#[test]
	fn bpe_merges_chars_into_token() {
		// vocab: 'a', 'b', and the merged token "ab"; merge ('a','b')
		let (enc, dec) = bytes_to_unicode();
		let vocab = vec!["a".to_string(), "b".to_string(), "ab".to_string()];
		let mut token_to_id = HashMap::new();
		for (i, v) in vocab.iter().enumerate() {
			token_to_id.insert(v.clone(), i as u32);
		}
		let mut merges = HashMap::new();
		merges.insert(("a".to_string(), "b".to_string()), 0usize);
		let t = Tokenizer {
			vocab,
			token_to_id,
			merges,
			byte_encoder: enc,
			byte_decoder: dec,
			special: vec![],
			special_ids: std::collections::HashSet::new(),
			eos: 0,
			bos: 0,
			add_bos: false,
			unk: None,
			pretok: fancy_regex::Regex::new(r"\p{L}+|\s+").unwrap(),
		};
		let ids = t.encode("ab");
		assert_eq!(ids, vec![2]); // the merged "ab" token
		assert_eq!(t.decode(&ids, true), "ab");
	}

	#[test]
	fn pretoken_split() {
		let t = tiny_tok();
		// leading space attaches to the word
		let pieces = t.pretokenize(" hello");
		assert_eq!(pieces, vec![" hello".to_string()]);
	}

	#[test]
	fn special_token_split() {
		let (enc, dec) = bytes_to_unicode();
		let vocab = vec![
			"<|endoftext|>".to_string(),
			"\u{0120}hi".to_string(),
			"there".to_string(),
		];
		let mut token_to_id = HashMap::new();
		for (i, v) in vocab.iter().enumerate() {
			token_to_id.insert(v.clone(), i as u32);
		}
		let special = vec!["<|endoftext|>".to_string()];
		let special_ids: std::collections::HashSet<u32> =
			special.iter().map(|s| token_to_id[s]).collect();
		let t = Tokenizer {
			vocab,
			token_to_id,
			merges: HashMap::new(),
			byte_encoder: enc,
			byte_decoder: dec,
			special,
			special_ids,
			eos: 0,
			bos: 0,
			add_bos: false,
			unk: None,
			pretok: fancy_regex::Regex::new(r"\p{L}+|\s+").unwrap(),
		};
		let segs = t.split_specials("hi<|endoftext|>there");
		assert_eq!(segs.len(), 3);
		assert!(matches!(segs[0], Segment::Text(_)));
		assert!(matches!(segs[1], Segment::Special(_)));
		assert!(matches!(segs[2], Segment::Text(_)));
	}
}

#[cfg(test)]
mod chat_tests {
	use super::*;

	/// The rendered prompt must match the model's **own** `tokenizer.chat_template`
	/// byte for byte.
	///
	/// `tests/data/tools_prompt.txt` is that template's output for the
	/// conversation below, produced by running the Jinja from the GGUF through
	/// Jinja2 (`tests/render_reference.py`, which reads
	/// `tests/data/chat_template.jinja`: the template as extracted from
	/// `Qwen3.8-27B-Q4_K_M.gguf`). Nothing here was written from memory of what
	/// Qwen tool calls look like: the format is XML-ish rather than the
	/// `{"name":...,"arguments":...}` JSON older Qwen releases used, tool
	/// definitions land in the *system* turn, tool results come back as `user`
	/// turns, and past assistant turns carry an empty `<think>` block. Every one
	/// of those is a silent-wrongness bug if guessed: the model would still
	/// answer, just worse, and no assertion anywhere would fire.
	#[test]
	fn tools_prompt_matches_the_models_chat_template() {
		let tools = [
			r#"{"type": "function", "function": {"name": "Read", "description": "Read a file from disk.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string", "description": "Absolute path."}, "limit": {"type": "integer", "description": "Max lines."}}, "required": ["file_path"]}}}"#.to_string(),
			r#"{"type": "function", "function": {"name": "Bash", "description": "Run a shell command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}"#.to_string(),
		];
		let turns = vec![
			Turn::System("You are a terse assistant.".to_string()),
			Turn::User("How many lines are in src/main.rs?".to_string()),
			Turn::Assistant {
				text: "I will look.".to_string(),
				reasoning: String::new(),
				calls: vec![ToolCall {
					name: "Read".to_string(),
					args: vec![
						("file_path".to_string(), "/repo/src/main.rs".to_string()),
						("limit".to_string(), "100".to_string()),
					],
				}],
			},
			Turn::ToolResults(vec!["fn main() {}\n".to_string()]),
			Turn::User("And now run the tests.".to_string()),
		];
		let got = Tokenizer::build_prompt(&turns, &tools, Tokenizer::ASSISTANT_OPEN_NOTHINK);
		let want = include_str!("../tests/data/tools_prompt.txt");
		assert_eq!(got, want);
	}

	/// Thinking on, byte for byte against the template (the `think_*` files of
	/// `tests/render_reference.py`): the effort's instruction ahead of the tool
	/// preamble / the system text / alone, past reasoning replayed (trimmed),
	/// and with `preserve_thinking: false` no block on turns before the last
	/// query while the tool loop after it keeps its own.
	#[test]
	fn thinking_prompts_match_the_template() {
		let tools = [
			r#"{"type": "function", "function": {"name": "Read", "description": "Read a file from disk.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string", "description": "Absolute path."}, "limit": {"type": "integer", "description": "Max lines."}}, "required": ["file_path"]}}}"#.to_string(),
			r#"{"type": "function", "function": {"name": "Bash", "description": "Run a shell command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}"#.to_string(),
		];
		let turns = || {
			vec![
				Turn::System("You are a terse assistant.".to_string()),
				Turn::User("What is 2+2?".to_string()),
				Turn::Assistant {
					text: "4.".to_string(),
					reasoning: "Simple arithmetic.".to_string(),
					calls: vec![],
				},
				Turn::User("How many lines are in src/main.rs?".to_string()),
				Turn::Assistant {
					text: "I will look.".to_string(),
					reasoning: "\nRead the file first.\n".to_string(),
					calls: vec![ToolCall {
						name: "Read".to_string(),
						args: vec![("file_path".to_string(), "/repo/src/main.rs".to_string())],
					}],
				},
				Turn::ToolResults(vec!["fn main() {}\n".to_string()]),
			]
		};
		let x = Thinking::on(Effort::Xhigh);
		let got = Tokenizer::build_prompt_with(&turns(), &tools, x, x.open());
		assert_eq!(got, include_str!("../tests/data/think_xhigh_prompt.txt"));

		let low = Thinking {
			effort: Some(Effort::Low),
			preserve: false,
		};
		let got = Tokenizer::build_prompt_with(&turns(), &[], low, low.open());
		assert_eq!(
			got,
			include_str!("../tests/data/think_low_nopreserve_prompt.txt")
		);

		let one = [Turn::User("What is 2+2?".to_string())];
		let m = Thinking::on(Effort::Medium);
		let got = Tokenizer::build_prompt_with(&one, &[], m, m.open());
		assert_eq!(
			got,
			include_str!("../tests/data/think_medium_nosystem_prompt.txt")
		);
		let got = Tokenizer::build_prompt_with(&one, &[], x, x.open());
		assert_eq!(
			got,
			include_str!("../tests/data/think_xhigh_nosystem_prompt.txt")
		);
	}

	/// The six USER_DEFINED tokens must survive a round trip **as single ids**,
	/// and must still be visible after a `skip_specials` decode.
	///
	/// Both halves are load-bearing and they pull in opposite directions. If
	/// they are not matched on encode, `<think>` becomes `<th`+`ink`+`>` and the
	/// model is handed a sequence it has never seen: the prompt that exposed
	/// this returned EOS as its first token, every time. If they are *also*
	/// hidden on decode, the tool-call parser never sees the `<tool_call>` it
	/// looks for and every call comes back as prose. Only CONTROL tokens are
	/// hidden.
	///
	/// Needs the real GGUF (`QWEN_TEST_GGUF`), so it is skipped without it.
	#[test]
	fn user_defined_tokens_are_whole_on_encode_and_visible_on_decode() {
		let model = std::env::var("QWEN_TEST_GGUF").unwrap_or_default();
		let Ok(g) = gguf::parse_file(&model) else {
			eprintln!("user_defined_tokens…: skipped (QWEN_TEST_GGUF = {model:?} not a model)");
			return;
		};
		let tk = Tokenizer::from_gguf(&g).unwrap();
		for t in [
			"<think>",
			"</think>",
			"<tool_call>",
			"</tool_call>",
			"<tool_response>",
			"</tool_response>",
		] {
			let ids = tk.encode(t);
			assert_eq!(ids.len(), 1, "{t} should be one token, got {ids:?}");
			assert_eq!(tk.decode(&ids, true), t, "{t} must survive skip_specials");
		}
		// CONTROL tokens are the ones that get hidden.
		let im = tk.encode("<|im_start|>");
		assert_eq!(im.len(), 1);
		assert_eq!(tk.decode(&im, true), "");
		assert_eq!(tk.decode(&im, false), "<|im_start|>");
	}

	/// A Hugging Face `tokenizer.json` must tokenize exactly as the GGUF's
	/// vocab does: same ids, same special-token handling, same decode.
	/// `QWEN_TEST_HF_TOKENIZER`: the BF16 checkpoint's `tokenizer.json` (what
	/// the NPU engine's build copies) or an OpenVINO export's. Needs it and
	/// `QWEN_TEST_GGUF`, so it is skipped when either is absent.
	#[test]
	fn hf_json_tokenizer_matches_gguf() {
		let model = std::env::var("QWEN_TEST_GGUF").unwrap_or_default();
		let hf = std::env::var("QWEN_TEST_HF_TOKENIZER").unwrap_or_default();
		let (Ok(g), true) = (
			gguf::parse_file(&model),
			std::path::Path::new(&hf).is_file(),
		) else {
			eprintln!(
				"hf_json_tokenizer_matches_gguf: skipped (needs QWEN_TEST_GGUF and QWEN_TEST_HF_TOKENIZER)"
			);
			return;
		};
		let a = Tokenizer::from_gguf(&g).unwrap();
		let b = Tokenizer::from_hf_json(std::path::Path::new(&hf)).unwrap();
		assert_eq!(a.eos(), b.eos());
		let text = "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHow's the café?  \u{1F600} \
                    fn main() { println!(\"{}\", 12345); }\n\tÜber naïve 東京 <tool_call>\n<function=Bash>\n\
                    </function>\n</tool_call><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
		let (ia, ib) = (a.encode(text), b.encode(text));
		assert_eq!(ia, ib);
		assert_eq!(a.decode(&ia, true), b.decode(&ib, true));
		assert_eq!(a.decode(&ia, false), text);
		assert_eq!(b.decode(&ib, false), text);
	}

	/// Without tools the system turn is a plain one and nothing tool-shaped
	/// appears: the tool preamble is ~1.4 kB of prompt and must not be charged
	/// to callers who sent no tools.
	#[test]
	fn no_tools_means_no_tool_preamble() {
		let turns = vec![
			Turn::System("Be brief.".to_string()),
			Turn::User("Hi".to_string()),
		];
		let got = Tokenizer::build_prompt(&turns, &[], Tokenizer::ASSISTANT_OPEN_NOTHINK);
		assert_eq!(
			got,
			"<|im_start|>system\nBe brief.<|im_end|>\n\
             <|im_start|>user\nHi<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
		);
	}

	/// Consecutive tool results share one `user` frame, which is what the
	/// template does for a run of `tool` messages: parallel tool calls come
	/// back as one turn, not one turn each.
	#[test]
	fn consecutive_tool_results_share_one_user_turn() {
		let turns = vec![
			Turn::User("go".to_string()),
			Turn::ToolResults(vec!["a".to_string()]),
			Turn::ToolResults(vec!["b".to_string(), "c".to_string()]),
			Turn::User("thanks".to_string()),
		];
		let got = Tokenizer::build_prompt(&turns, &[], "");
		assert_eq!(
			got,
			"<|im_start|>user\ngo<|im_end|>\n\
             <|im_start|>user\n<tool_response>\na\n</tool_response>\
             \n<tool_response>\nb\n</tool_response>\
             \n<tool_response>\nc\n</tool_response><|im_end|>\n\
             <|im_start|>user\nthanks<|im_end|>\n"
		);
	}

	/// A call with no prose before it starts at the top of the turn; with prose
	/// it is separated by a blank line. Both spellings are in the template and
	/// they are not interchangeable.
	#[test]
	fn tool_call_spacing_depends_on_whether_there_is_prose() {
		let bare = Tokenizer::build_prompt(
			&[Turn::Assistant {
				text: String::new(),
				reasoning: String::new(),
				calls: vec![ToolCall {
					name: "f".to_string(),
					args: vec![],
				}],
			}],
			&[],
			"",
		);
		assert_eq!(
			bare,
			"<|im_start|>assistant\n<think>\n\n</think>\n\n\
             <tool_call>\n<function=f>\n</function>\n</tool_call><|im_end|>\n"
		);
		let with_prose = Tokenizer::build_prompt(
			&[Turn::Assistant {
				text: "ok".to_string(),
				reasoning: String::new(),
				calls: vec![ToolCall {
					name: "f".to_string(),
					args: vec![],
				}],
			}],
			&[],
			"",
		);
		assert!(with_prose.contains("ok\n\n<tool_call>"), "{with_prose}");
	}
}
