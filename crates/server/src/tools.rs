//! Tool use: the Anthropic request/response side, and the model's own format.
//!
//! Two formats meet here and neither is negotiable:
//!
//! * **Anthropic Messages API**: tools are `{name, description, input_schema}`,
//!   calls come back as `tool_use` blocks with a JSON `input` object, results
//!   arrive as `tool_result` blocks.
//! * **This model's chat template**: tools are rendered into the system turn
//!   as OpenAI-shaped `{"type":"function","function":{...}}` JSON, and a call is
//!   *XML-ish text*: `<tool_call><function=name><parameter=k>` with the value on
//!   its own lines. Not the `{"name":…,"arguments":…}` JSON body older Qwen
//!   releases used. See `Tokenizer::build_prompt`, which is pinned against the
//!   template extracted from the GGUF.
//!
//! So a call round-trips through text, and the interesting work is turning that
//! text back into typed JSON without a schema-aware parser: the wire format
//! carries no types at all. [`ToolSpec::params`] is what supplies them.

use crate::{jstr, Json};

/// One tool the caller declared, ready for both directions.
pub struct ToolSpec {
	pub name: String,
	/// The tool rendered as the template's `tool | tojson`, cached because it
	/// goes into the prompt on every request of the conversation.
	pub json: String,
	/// `(parameter, JSON Schema "type")` for each declared property. This is
	/// the *only* thing that can tell `"123"` (a string parameter) from `123`
	/// (a number one) when parsing a call back out of the model's text.
	pub params: Vec<(String, String)>,
}

/// What the caller asked for in `tool_choice`.
#[derive(PartialEq, Debug)]
pub enum Choice {
	/// The model decides (the default).
	Auto,
	/// It must call something.
	Any,
	/// It must call this one.
	Named(String),
	/// Tools are declared but must not be called.
	None,
}

/// Read `tools` out of a request.
///
/// Anthropic's `input_schema` becomes the template's `function.parameters`; the
/// wrapper is what the model was trained to read. A tool without a name is
/// dropped rather than rendered as `"name": null`, which the model would
/// cheerfully try to call.
pub fn specs_from_request(req: &Json) -> Vec<ToolSpec> {
	let Some(arr) = req.get("tools").and_then(|v| v.as_array()) else {
		return Vec::new();
	};
	let mut out = Vec::new();
	for t in arr {
		let Some(name) = t.get("name").and_then(|v| v.as_str()) else {
			continue;
		};
		// Anthropic ships server-side tool types (web_search and friends) in the
		// same array, marked by `type` and carrying no input_schema. We cannot
		// run those, and offering them to the model would produce calls nothing
		// can answer.
		let schema = t.get("input_schema");
		if schema.is_none() && t.get("type").is_some() {
			continue;
		}
		let mut func: Vec<(String, Json)> = vec![("name".to_string(), Json::Str(name.to_string()))];
		if let Some(d) = t.get("description").and_then(|v| v.as_str()) {
			func.push(("description".to_string(), Json::Str(d.to_string())));
		}
		let mut params = Vec::new();
		if let Some(schema) = schema {
			if let Some(Json::Obj(props)) = schema.get("properties") {
				for (k, v) in props {
					let ty = v
						.get("type")
						.and_then(|t| t.as_str())
						.map(|s| s.to_string())
						// A property with no `type` is legal JSON Schema
						// (`enum`, `anyOf`, `$ref`). "" means "guess from the
						// text" in `coerce`.
						.unwrap_or_default();
					params.push((k.clone(), ty));
				}
			}
			func.push(("parameters".to_string(), schema.clone()));
		}
		let wrapped = Json::Obj(vec![
			("type".to_string(), Json::Str("function".to_string())),
			("function".to_string(), Json::Obj(func)),
		]);
		out.push(ToolSpec {
			name: name.to_string(),
			json: wrapped.to_string_spaced(),
			params,
		});
	}
	out
}

/// Read `tool_choice`. Anything unrecognized is `Auto`, which is the API default
/// and the only safe way to be wrong.
pub fn choice_from_request(req: &Json) -> Choice {
	let Some(tc) = req.get("tool_choice") else {
		return Choice::Auto;
	};
	match tc.get("type").and_then(|v| v.as_str()) {
		Some("any") => Choice::Any,
		Some("none") => Choice::None,
		Some("tool") => match tc.get("name").and_then(|v| v.as_str()) {
			Some(n) => Choice::Named(n.to_string()),
			None => Choice::Auto,
		},
		_ => Choice::Auto,
	}
}

/// Text to seed the assistant turn with so a forced choice is actually forced.
///
/// `tool_choice` cannot be honored by asking nicely: a model told "you must
/// call a function" still sometimes answers in prose. Opening the call for it
/// makes the constraint structural: with `<tool_call>\n` already in the context
/// the only continuation the format allows is a function name. `Named` goes one
/// token further and writes the name too, so "call *this*" cannot be missed.
pub fn forced_prefix(choice: &Choice) -> String {
	match choice {
		Choice::Any => "<tool_call>\n".to_string(),
		Choice::Named(n) => format!("<tool_call>\n<function={n}>\n"),
		_ => String::new(),
	}
}

// ===========================================================================
// Parsing the model's output
// ===========================================================================

/// A call recovered from the model's text.
pub struct ParsedCall {
	pub name: String,
	/// Always a `Json::Obj`.
	pub input: Json,
}

impl ParsedCall {
	pub fn input_json(&self) -> String {
		self.input.to_string()
	}
}

/// Parse the inside of one `<tool_call>…</tool_call>` block.
///
/// Lenient about the closing tags and strict about the opening ones: a call
/// truncated by `max_tokens` still names a function and may carry complete
/// parameters, and dropping all of that on a missing `</function>` turns a
/// recoverable turn into a confusing empty one. A *missing* `<function=` is
/// different: there is nothing to call.
pub fn parse_call(raw: &str, tools: &[ToolSpec]) -> Option<ParsedCall> {
	let i = raw.find("<function=")?;
	let rest = &raw[i + "<function=".len()..];
	let j = rest.find('>')?;
	let name = rest[..j].trim().to_string();
	if name.is_empty() {
		return None;
	}
	let body = &rest[j + 1..];
	let spec = tools.iter().find(|t| t.name == name);

	let mut fields: Vec<(String, Json)> = Vec::new();
	let mut cur = body;
	while let Some(p) = cur.find("<parameter=") {
		// A `<parameter=` after `</function>` belongs to a later call, not this
		// one; stop at the boundary when there is one.
		if let Some(e) = cur.find("</function>") {
			if e < p {
				break;
			}
		}
		let after = &cur[p + "<parameter=".len()..];
		let Some(q) = after.find('>') else { break };
		let key = after[..q].trim().to_string();
		let val_start = &after[q + 1..];
		// An unterminated value is the truncation case: take the rest.
		let (val, next) = match val_start.find("</parameter>") {
			Some(e) => (&val_start[..e], &val_start[e + "</parameter>".len()..]),
			None => (val_start, ""),
		};
		// The template writes a newline after the tag and before the closer;
		// they are framing, not part of the value.
		let val = val.strip_prefix('\n').unwrap_or(val);
		let val = val.strip_suffix('\n').unwrap_or(val);
		let ty = spec
			.and_then(|s| s.params.iter().find(|(k, _)| *k == key))
			.map(|(_, t)| t.as_str())
			.unwrap_or("");
		fields.push((key, coerce(val, ty)));
		if next.is_empty() {
			break;
		}
		cur = next;
	}
	Some(ParsedCall {
		name,
		input: Json::Obj(fields),
	})
}

/// Turn a parameter's raw text back into a typed JSON value.
///
/// The declared type wins whenever there is one, and that direction matters
/// more than it looks: a `"string"` parameter whose value happens to be `123`
/// or `true` must stay text, or a shell command like `0` becomes the number
/// zero and the tool receives the wrong type. With no declared type the text is
/// tried as JSON and kept as a string if that fails, which is the same rule the
/// template used when it wrote the value out.
fn coerce(text: &str, ty: &str) -> Json {
	match ty {
		"string" => Json::Str(text.to_string()),
		"" => Json::parse(text.trim().as_bytes()).unwrap_or_else(|_| Json::Str(text.to_string())),
		_ => match Json::parse(text.trim().as_bytes()) {
			// A malformed number or object is still better delivered as the
			// text the model wrote than dropped: the tool can say so.
			Ok(v) => v,
			Err(_) => Json::Str(text.to_string()),
		},
	}
}

// ===========================================================================
// Splitting the output stream
// ===========================================================================

#[derive(PartialEq)]
enum Mode {
	Text,
	Think,
	Call,
}

/// Separates visible text from `<tool_call>` blocks (and stray `<think>` ones)
/// as tokens arrive.
///
/// The hard part is streaming: `<tool_call>` arrives one token at a time, and by
/// the time the tag is recognizable its first characters have already been
/// eligible to send. So text is emitted only up to the longest suffix that could
/// still be the start of a marker: `"<to"` is held, `"<x"` is not. Without that
/// the client sees the literal `<tool_call>` XML in the assistant's prose before
/// the tool block replaces it.
pub struct OutputSplitter {
	buf: String,
	mode: Mode,
	/// Raw bodies of the tool-call blocks seen so far.
	pub calls: Vec<String>,
	/// Reasoning that became safe to send since the last [`Self::take_think`]:
	/// a think block's body without the whitespace that touches its tags.
	think: String,
	/// Whether the current think block has produced a non-space yet.
	think_started: bool,
}

/// Markers that end plain text. With thinking off `<think>` is a safety net:
/// the server closes the thinking block in the prompt, so the model should
/// never open another, but a leaked `<think>` rendered as assistant prose is a
/// bad failure mode for a caller that parses the reply.
const OPENERS: [&str; 2] = ["<tool_call>", "<think>"];

impl Default for OutputSplitter {
	fn default() -> Self {
		OutputSplitter {
			buf: String::new(),
			mode: Mode::Text,
			calls: Vec::new(),
			think: String::new(),
			think_started: false,
		}
	}
}

impl OutputSplitter {
	pub fn new() -> OutputSplitter {
		Self::default()
	}

	/// A splitter for a reply that starts inside a think block: the prompt
	/// ended on the template's open `<think>\n`.
	pub fn thinking() -> OutputSplitter {
		OutputSplitter {
			mode: Mode::Think,
			..Self::default()
		}
	}

	/// Whether the stream is inside a think block.
	pub fn in_think(&self) -> bool {
		self.mode == Mode::Think
	}

	/// The reasoning that is safe to send since the last call.
	pub fn take_think(&mut self) -> String {
		std::mem::take(&mut self.think)
	}

	/// Pass on `body` (think-block text) as reasoning: leading whitespace of
	/// the block dropped, trailing whitespace only once more text follows it.
	fn think_text(&mut self, body: &str, last: bool) {
		let body = if self.think_started {
			body
		} else {
			body.trim_start()
		};
		let body = if last { body.trim_end() } else { body };
		if !body.is_empty() {
			self.think_started = true;
			self.think.push_str(body);
		}
	}

	/// Feed decoded text; returns whatever is now safe to send to the client.
	pub fn push(&mut self, piece: &str) -> String {
		self.buf.push_str(piece);
		let mut out = String::new();
		loop {
			match self.mode {
				Mode::Text => {
					if let Some((i, m)) = earliest(&self.buf, &OPENERS) {
						out.push_str(&self.buf[..i]);
						self.buf.drain(..i + m.len());
						self.mode = if m == "<tool_call>" {
							Mode::Call
						} else {
							Mode::Think
						};
						continue;
					}
					let keep = held_back(&self.buf, &OPENERS);
					let n = self.buf.len() - keep;
					out.push_str(&self.buf[..n]);
					self.buf.drain(..n);
					break;
				}
				Mode::Call => match self.buf.find("</tool_call>") {
					Some(i) => {
						self.calls.push(self.buf[..i].to_string());
						self.buf.drain(..i + "</tool_call>".len());
						self.mode = Mode::Text;
					}
					None => break,
				},
				Mode::Think => match self.buf.find("</think>") {
					Some(i) => {
						let body: String = self.buf.drain(..i + "</think>".len()).collect();
						self.think_text(&body[..i], true);
						self.think_started = false;
						self.mode = Mode::Text;
					}
					None => {
						// Up to a possible partial `</think>`, and short of the
						// trailing whitespace, which may be the block's last.
						let keep = held_back(&self.buf, &["</think>"]);
						let n = self.buf[..self.buf.len() - keep].trim_end().len();
						let body: String = self.buf.drain(..n).collect();
						self.think_text(&body, false);
						break;
					}
				},
			}
		}
		out
	}

	/// End of generation: flush what is left.
	///
	/// A held-back partial marker was never a marker after all, so it is text. A
	/// tool call cut off by `max_tokens` is kept ([`parse_call`] recovers what
	/// it can) and an unterminated `<think>` (the reply ran out inside its
	/// reasoning) ends as reasoning: [`Self::take_think`] has the rest.
	pub fn finish(&mut self) -> String {
		let tail = std::mem::take(&mut self.buf);
		match self.mode {
			Mode::Text => tail,
			Mode::Call => {
				if !tail.trim().is_empty() {
					self.calls.push(tail);
				}
				String::new()
			}
			Mode::Think => {
				self.think_text(&tail, true);
				String::new()
			}
		}
	}
}

/// Earliest occurrence of any marker, and which one.
fn earliest<'a>(hay: &str, marks: &[&'a str]) -> Option<(usize, &'a str)> {
	marks
		.iter()
		.filter_map(|m| hay.find(m).map(|i| (i, *m)))
		.min_by_key(|(i, _)| *i)
}

/// Length of the trailing run that could still grow into one of `marks`.
fn held_back(hay: &str, marks: &[&str]) -> usize {
	let max = marks.iter().map(|m| m.len()).max().unwrap_or(0);
	let start = hay.len().saturating_sub(max - 1);
	for i in start..hay.len() {
		// Marker text is ASCII, so any boundary a marker could start on is a
		// char boundary; skip the rest rather than panicking on a slice.
		if !hay.is_char_boundary(i) {
			continue;
		}
		let tail = &hay[i..];
		if marks.iter().any(|m| m.starts_with(tail)) {
			return hay.len() - i;
		}
	}
	0
}

/// `tool_use` block id. Anthropic's are `toolu_`-prefixed and opaque; the client
/// only has to match the one it echoes back in `tool_result`.
pub fn new_tool_id(seq: usize) -> String {
	let t = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map(|d| d.as_nanos() as u64)
		.unwrap_or(0);
	format!(
		"toolu_{:016x}{:04x}",
		t ^ 0x9e37_79b9_7f4a_7c15,
		seq & 0xffff
	)
}

/// Render one `tool_use` content block.
pub fn tool_use_block(id: &str, call: &ParsedCall) -> String {
	format!(
		"{{\"type\":\"tool_use\",\"id\":{id},\"name\":{name},\"input\":{input}}}",
		id = jstr(id),
		name = jstr(&call.name),
		input = call.input_json(),
	)
}

#[cfg(test)]
mod tests {
	use super::*;

	fn tools() -> Vec<ToolSpec> {
		vec![ToolSpec {
			name: "Bash".to_string(),
			json: String::new(),
			params: vec![
				("command".to_string(), "string".to_string()),
				("timeout".to_string(), "integer".to_string()),
				("bg".to_string(), "boolean".to_string()),
			],
		}]
	}

	#[test]
	fn parses_a_call_and_types_its_arguments_from_the_schema() {
		let raw = "\n<function=Bash>\n<parameter=command>\nls -la\n</parameter>\n\
                   <parameter=timeout>\n5000\n</parameter>\n\
                   <parameter=bg>\ntrue\n</parameter>\n</function>\n";
		let c = parse_call(raw, &tools()).unwrap();
		assert_eq!(c.name, "Bash");
		assert_eq!(
			c.input_json(),
			r#"{"command":"ls -la","timeout":5000,"bg":true}"#
		);
	}

	/// The case the schema exists to settle: a string parameter whose text is a
	/// valid number or boolean must not become one. `git checkout 123` and
	/// `git checkout "123"` are different commands.
	#[test]
	fn a_string_parameter_stays_a_string() {
		let raw = "<function=Bash>\n<parameter=command>\n123\n</parameter>\n</function>";
		let c = parse_call(raw, &tools()).unwrap();
		assert_eq!(c.input_json(), r#"{"command":"123"}"#);
	}

	/// Values may span lines; only the template's own framing newlines come off.
	#[test]
	fn multi_line_values_keep_their_interior_newlines() {
		let raw = "<function=Bash>\n<parameter=command>\nline one\n\nline three\n</parameter>\n</function>";
		let c = parse_call(raw, &tools()).unwrap();
		assert_eq!(c.input_json(), r#"{"command":"line one\n\nline three"}"#);
	}

	/// An unknown tool has no schema, so values are read as JSON when they look
	/// like it and as text otherwise.
	#[test]
	fn an_undeclared_tool_falls_back_to_guessing() {
		let raw = "<function=Other>\n<parameter=n>\n7\n</parameter>\n\
                   <parameter=s>\nhello\n</parameter>\n</function>";
		let c = parse_call(raw, &tools()).unwrap();
		assert_eq!(c.name, "Other");
		assert_eq!(c.input_json(), r#"{"n":7,"s":"hello"}"#);
	}

	/// `max_tokens` in the middle of a call: keep the name and whatever
	/// parameters completed, rather than returning nothing.
	#[test]
	fn a_truncated_call_keeps_what_it_has() {
		let raw = "<function=Bash>\n<parameter=command>\nls\n</parameter>\n<parameter=timeout>\n50";
		let c = parse_call(raw, &tools()).unwrap();
		assert_eq!(c.input_json(), r#"{"command":"ls","timeout":50}"#);
	}

	#[test]
	fn a_block_with_no_function_is_not_a_call() {
		assert!(parse_call("\nnonsense\n", &tools()).is_none());
	}

	#[test]
	fn splitter_separates_prose_from_a_call() {
		let mut s = OutputSplitter::new();
		let mut out = String::new();
		out.push_str(&s.push("I will run it.\n\n<tool_call>\n<function=Bash>\n"));
		out.push_str(&s.push("<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"));
		out.push_str(&s.finish());
		assert_eq!(out, "I will run it.\n\n");
		assert_eq!(s.calls.len(), 1);
		assert_eq!(
			parse_call(&s.calls[0], &tools()).unwrap().input_json(),
			r#"{"command":"ls"}"#
		);
	}

	/// The streaming hazard: a marker split across token boundaries must never
	/// be forwarded as text, however finely it is cut.
	#[test]
	fn splitter_never_leaks_a_marker_split_across_pieces() {
		for chunk in 1..=4 {
			let src = "ok<tool_call>\n<function=Bash>\n</function>\n</tool_call>tail";
			let mut s = OutputSplitter::new();
			let mut out = String::new();
			let bytes: Vec<char> = src.chars().collect();
			for piece in bytes.chunks(chunk) {
				out.push_str(&s.push(&piece.iter().collect::<String>()));
			}
			out.push_str(&s.finish());
			assert_eq!(out, "oktail", "chunk size {chunk}");
			assert_eq!(s.calls.len(), 1, "chunk size {chunk}");
		}
	}

	/// A `<` that never becomes a marker is ordinary text and must still come
	/// out: held back briefly, not swallowed.
	#[test]
	fn splitter_releases_a_false_start() {
		let mut s = OutputSplitter::new();
		let mut out = String::new();
		out.push_str(&s.push("a < b and <to"));
		out.push_str(&s.push("m is 3"));
		out.push_str(&s.finish());
		assert_eq!(out, "a < b and <tom is 3");
		assert!(s.calls.is_empty());
	}

	#[test]
	fn splitter_drops_a_thinking_block() {
		let mut s = OutputSplitter::new();
		let mut out = String::new();
		out.push_str(&s.push("<think>\nhmm\n</think>\n\nthe answer"));
		out.push_str(&s.finish());
		assert_eq!(out, "\n\nthe answer");
	}

	/// A reply that starts inside the prompt's open `<think>\n`: the reasoning
	/// comes out through `take_think` without the whitespace around it, at any
	/// chunking, and a reply cut off inside it still yields what it had.
	#[test]
	fn splitter_streams_reasoning() {
		let raw = "First, add.\n\n2 + 2 = 4\n</think>\n\nIt is 4.<tool_call>\n<function=A>\n</function>\n</tool_call>";
		for chunk in [1, 2, 3, 7, raw.len()] {
			let mut s = OutputSplitter::thinking();
			let (mut think, mut out) = (String::new(), String::new());
			for c in raw.as_bytes().chunks(chunk) {
				out.push_str(&s.push(std::str::from_utf8(c).unwrap()));
				think.push_str(&s.take_think());
			}
			out.push_str(&s.finish());
			think.push_str(&s.take_think());
			assert_eq!(think, "First, add.\n\n2 + 2 = 4", "chunk size {chunk}");
			assert_eq!(out, "\n\nIt is 4.", "chunk size {chunk}");
			assert_eq!(s.calls.len(), 1, "chunk size {chunk}");
		}
		let mut s = OutputSplitter::thinking();
		assert_eq!(s.push("\nhalf a thought \n"), "");
		assert!(s.in_think());
		s.finish();
		assert_eq!(s.take_think(), "half a thought");
	}

	#[test]
	fn splitter_handles_two_calls_in_one_turn() {
		let mut s = OutputSplitter::new();
		s.push("<tool_call>\n<function=A>\n</function>\n</tool_call>\n");
		s.push("<tool_call>\n<function=B>\n</function>\n</tool_call>");
		s.finish();
		assert_eq!(s.calls.len(), 2);
		assert_eq!(parse_call(&s.calls[1], &[]).unwrap().name, "B");
	}

	#[test]
	fn specs_convert_anthropic_tools_to_the_template_shape() {
		let req = Json::parse(
			br#"{"tools":[{"name":"Read","description":"d","input_schema":{"type":"object","properties":{"p":{"type":"string"}}}}]}"#,
		)
		.unwrap();
		let s = specs_from_request(&req);
		assert_eq!(s.len(), 1);
		assert_eq!(s[0].name, "Read");
		assert_eq!(s[0].params, vec![("p".to_string(), "string".to_string())]);
		assert_eq!(
			s[0].json,
			r#"{"type": "function", "function": {"name": "Read", "description": "d", "parameters": {"type": "object", "properties": {"p": {"type": "string"}}}}}"#
		);
	}

	/// Server-side tool types (`web_search` and friends) carry no schema and
	/// nothing here can run them; offering them would produce calls the client
	/// never asked to handle.
	#[test]
	fn specs_skip_server_side_tools() {
		let req = Json::parse(br#"{"tools":[{"type":"web_search_20250305","name":"web_search"}]}"#)
			.unwrap();
		assert!(specs_from_request(&req).is_empty());
	}

	#[test]
	fn tool_choice_is_read_and_forced_structurally() {
		let any = Json::parse(br#"{"tool_choice":{"type":"any"}}"#).unwrap();
		assert_eq!(choice_from_request(&any), Choice::Any);
		assert_eq!(forced_prefix(&Choice::Any), "<tool_call>\n");
		let named = Json::parse(br#"{"tool_choice":{"type":"tool","name":"Bash"}}"#).unwrap();
		assert_eq!(
			choice_from_request(&named),
			Choice::Named("Bash".to_string())
		);
		assert_eq!(
			forced_prefix(&Choice::Named("Bash".into())),
			"<tool_call>\n<function=Bash>\n"
		);
		assert_eq!(choice_from_request(&Json::Obj(vec![])), Choice::Auto);
		assert_eq!(forced_prefix(&Choice::Auto), "");
	}
}
