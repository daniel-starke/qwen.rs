//! OpenAI Chat Completions (`POST /v1/chat/completions`) and the llama.cpp
//! `llama-server` discovery routes (`GET /props`, the `meta` of `/v1/models`).
//!
//! This is what Zed's **llama.cpp** provider speaks: it lists `/v1/models`,
//! reads the context window, vision and tool support from `/props`, and streams
//! `/v1/chat/completions` with `stream_options.include_usage`. Zed's own client
//! (`crates/llama_cpp` and `language_models/src/provider/llama_cpp.rs` in the
//! Zed tree) is the reference for which fields matter.
//!
//! A request is translated into the Anthropic-shaped request `generate` already
//! takes, so the prompt, tool-call format and prefix cache are shared with
//! `/v1/messages` and the two APIs cannot drift apart. Only the response is
//! formatted here ([`OpenAiWire`]).

use std::io::Write;

use crate::tools::{self, ParsedCall};
use crate::{jstr, respond_json, sse_begin, sse_chunk, sse_end, Json, Loaded, Stop, Usage, Wire};

fn obj(kv: Vec<(&str, Json)>) -> Json {
	Json::Obj(kv.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn s(v: &str) -> Json {
	Json::Str(v.to_string())
}

/// Translate a Chat Completions request into the Messages API shape.
///
/// * `system` / `developer` messages, wherever they are, become the top-level
///   `system` (the chat template has one system turn, at the start).
/// * `image_url` parts become `image` blocks; `blocks_of` already accepts a
///   `data:` URL as the source and refuses remote ones.
/// * An assistant's `tool_calls` become `tool_use` blocks, with `arguments`
///   (a JSON *string* here) parsed back into the object `input` is.
/// * A `tool` message becomes a user `tool_result`; runs of them merge into one
///   `<tool_response>` run in the template, as Anthropic's do.
/// * `tools`' `function.parameters` becomes `input_schema`, `tool_choice`
///   `required` becomes `any`, and `stop` / `max_completion_tokens` /
///   `repeat_penalty` map onto their Messages names.
pub fn to_messages_request(req: &Json) -> Result<Json, String> {
	let mut out: Vec<(&str, Json)> = Vec::new();
	let mut system: Vec<String> = Vec::new();
	let mut messages: Vec<Json> = Vec::new();

	let msgs = req
		.get("messages")
		.and_then(|v| v.as_array())
		.ok_or("`messages` must be an array")?;
	for m in msgs {
		let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("user");
		let content = m.get("content");
		match role {
			"system" | "developer" => {
				if let Some(t) = crate::text_of(content) {
					system.push(t);
				}
			}
			"tool" => {
				let mut r = vec![
					("type", s("tool_result")),
					("content", tool_content(content)?),
				];
				if let Some(id) = m.get("tool_call_id").and_then(|v| v.as_str()) {
					r.push(("tool_use_id", s(id)));
				}
				messages.push(obj(vec![
					("role", s("user")),
					("content", Json::Arr(vec![obj(r)])),
				]));
			}
			"assistant" => {
				let mut blocks = Vec::new();
				// llama-server's and vLLM's names for a turn's reasoning; it is
				// replayed in the turn's `<think>` (`preserve_thinking`).
				if let Some(r) = m
					.get("reasoning_content")
					.or_else(|| m.get("reasoning"))
					.and_then(|v| v.as_str())
					.filter(|r| !r.trim().is_empty())
				{
					blocks.push(obj(vec![("type", s("thinking")), ("thinking", s(r))]));
				}
				blocks.extend(parts(content)?);
				for c in m
					.get("tool_calls")
					.and_then(|v| v.as_array())
					.unwrap_or(&[])
				{
					let f = c.get("function").ok_or("tool call has no `function`")?;
					let name = f
						.get("name")
						.and_then(|v| v.as_str())
						.ok_or("tool call has no name")?;
					// `arguments` is a string of JSON. One that does not parse
					// (a client replaying a call the model cut off) is replayed
					// with no arguments rather than failing the conversation.
					let input = match f.get("arguments") {
						Some(Json::Str(a)) if !a.trim().is_empty() => {
							Json::parse(a.as_bytes()).unwrap_or(Json::Obj(Vec::new()))
						}
						Some(o @ Json::Obj(_)) => o.clone(),
						_ => Json::Obj(Vec::new()),
					};
					let mut b = vec![("type", s("tool_use")), ("name", s(name)), ("input", input)];
					if let Some(id) = c.get("id").and_then(|v| v.as_str()) {
						b.push(("id", s(id)));
					}
					blocks.push(obj(b));
				}
				messages.push(obj(vec![
					("role", s("assistant")),
					("content", Json::Arr(blocks)),
				]));
			}
			_ => messages.push(obj(vec![
				("role", s("user")),
				("content", Json::Arr(parts(content)?)),
			])),
		}
	}
	if !system.is_empty() {
		out.push(("system", s(&system.join("\n\n"))));
	}
	out.push(("messages", Json::Arr(messages)));

	if let Some(tools) = req.get("tools").and_then(|v| v.as_array()) {
		let mut ts = Vec::new();
		for t in tools {
			// `{"type":"function","function":{...}}`; anything else is not a
			// tool this server can offer the model.
			let Some(f) = t.get("function") else { continue };
			let Some(name) = f.get("name").and_then(|v| v.as_str()) else {
				continue;
			};
			let mut d = vec![("name", s(name))];
			if let Some(desc) = f.get("description").and_then(|v| v.as_str()) {
				d.push(("description", s(desc)));
			}
			d.push((
				"input_schema",
				f.get("parameters")
					.cloned()
					.unwrap_or_else(|| obj(vec![("type", s("object"))])),
			));
			ts.push(obj(d));
		}
		out.push(("tools", Json::Arr(ts)));
	}
	match req.get("tool_choice") {
		Some(Json::Str(c)) => {
			let ty = match c.as_str() {
				"required" => "any",
				"none" => "none",
				_ => "auto",
			};
			out.push(("tool_choice", obj(vec![("type", s(ty))])));
		}
		Some(c) => {
			if let Some(n) = c
				.get("function")
				.and_then(|f| f.get("name"))
				.and_then(|v| v.as_str())
			{
				out.push((
					"tool_choice",
					obj(vec![("type", s("tool")), ("name", s(n))]),
				));
			}
		}
		None => {}
	}

	let max = req
		.get("max_completion_tokens")
		.or_else(|| req.get("max_tokens"));
	if let Some(n) = max.and_then(|v| v.as_f64()).filter(|&n| n > 0.0) {
		out.push(("max_tokens", Json::Num(n)));
	}
	match req.get("stop") {
		Some(Json::Str(x)) => out.push(("stop_sequences", Json::Arr(vec![s(x)]))),
		Some(a @ Json::Arr(_)) => out.push(("stop_sequences", a.clone())),
		_ => {}
	}
	for (from, to) in [
		("temperature", "temperature"),
		("top_p", "top_p"),
		("top_k", "top_k"),
		("min_p", "min_p"),
		("presence_penalty", "presence_penalty"),
		("frequency_penalty", "frequency_penalty"),
		("seed", "seed"),
		("repetition_penalty", "repetition_penalty"),
		("repeat_penalty", "repetition_penalty"),
	] {
		if let Some(v @ Json::Num(_)) = req.get(from) {
			out.push((to, v.clone()));
		}
	}
	// The model card's switches: `chat_template_kwargs.enable_thinking`
	// (Qwen Cloud: top level), OpenAI's `reasoning_effort` (any level but
	// `none` turns thinking on; `minimal` is the template's `low`), and
	// `preserve_thinking`. Unset, thinking is on at the template's default
	// effort, as llama-server renders it (Zed's llama.cpp provider sends no
	// switch at all and relies on that default) unless the server was
	// started with `--think` / `--no-preserve-thinking`, whose default then
	// applies (`thinking_from_request`).
	let kw = req.get("chat_template_kwargs");
	let enable = kw
		.and_then(|k| k.get("enable_thinking"))
		.or_else(|| req.get("enable_thinking"))
		.and_then(|v| v.as_bool());
	let effort = req
		.get("reasoning_effort")
		.or_else(|| kw.and_then(|k| k.get("reasoning_effort")))
		.and_then(|v| v.as_str());
	let on = match (enable, effort) {
		(Some(false), _) | (_, Some("none")) => Some(false),
		(Some(true), _) | (_, Some(_)) => Some(true),
		(None, None) => (!crate::default_thinking_set()).then_some(true),
	};
	if let Some(on) = on {
		let ty = if on { "enabled" } else { "disabled" };
		out.push(("thinking", obj(vec![("type", s(ty))])));
	}
	if let Some(e) = effort.filter(|&e| e != "none") {
		let e = if e == "minimal" { "low" } else { e };
		out.push(("reasoning_effort", s(e)));
	}
	if let Some(p) = req
		.get("preserve_thinking")
		.or_else(|| kw.and_then(|k| k.get("preserve_thinking")))
		.and_then(|v| v.as_bool())
	{
		out.push(("preserve_thinking", Json::Bool(p)));
	}
	Ok(obj(out))
}

/// A message's `content` (string, parts, or null) as Messages API blocks.
/// A `tool` message's `content` as a `tool_result`'s: a string as is, the
/// `image_url` parts of an array as `image` blocks (an agent's `read_file`
/// on a picture), every other part as it came.
fn tool_content(content: Option<&Json>) -> Result<Json, String> {
	match content {
		Some(Json::Arr(a)) => a
			.iter()
			.map(|p| match p.get("type").and_then(|v| v.as_str()) {
				Some("image_url") => Ok(parts(Some(&Json::Arr(vec![p.clone()])))?.remove(0)),
				_ => Ok(p.clone()),
			})
			.collect::<Result<Vec<_>, String>>()
			.map(Json::Arr),
		Some(c) => Ok(c.clone()),
		None => Ok(Json::Str(String::new())),
	}
}

fn parts(content: Option<&Json>) -> Result<Vec<Json>, String> {
	match content {
		None | Some(Json::Null) => Ok(Vec::new()),
		Some(Json::Str(t)) => Ok(vec![obj(vec![("type", s("text")), ("text", s(t))])]),
		Some(Json::Arr(a)) => {
			let mut out = Vec::new();
			for p in a {
				match p.get("type").and_then(|v| v.as_str()) {
					Some("text") => {
						if let Some(t) = p.get("text").and_then(|v| v.as_str()) {
							out.push(obj(vec![("type", s("text")), ("text", s(t))]));
						}
					}
					Some("image_url") => {
						// `{"url": ...}`, or the bare string some clients send.
						let iu = p
							.get("image_url")
							.ok_or("image_url part has no image_url")?;
						let url = iu
							.as_str()
							.or_else(|| iu.get("url").and_then(|v| v.as_str()))
							.ok_or("image_url has no url")?;
						out.push(obj(vec![
							("type", s("image")),
							("source", obj(vec![("type", s("base64")), ("data", s(url))])),
						]));
					}
					other => return Err(format!("unsupported content part type {other:?}")),
				}
			}
			Ok(out)
		}
		Some(_) => Err("`content` must be a string, an array of parts, or null".to_string()),
	}
}

/// Chat Completions response, streamed (`chat.completion.chunk` events ending
/// in `data: [DONE]`) or whole (`chat.completion`).
pub struct OpenAiWire {
	pub stream: bool,
	/// `stream_options.include_usage`: a final chunk with `usage` and no
	/// choices. Zed asks for it and shows the numbers.
	pub include_usage: bool,
	pub id: String,
	pub model: String,
	pub created: u64,
}

impl OpenAiWire {
	/// `model`: the id reported when the request names none.
	pub fn new(req: &Json, model: &str) -> OpenAiWire {
		let c = crate::new_id();
		OpenAiWire {
			stream: req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false),
			include_usage: req
				.get("stream_options")
				.and_then(|o| o.get("include_usage"))
				.and_then(|v| v.as_bool())
				.unwrap_or(false),
			id: format!("chatcmpl-{}", c.trim_start_matches("msg_")),
			model: req
				.get("model")
				.and_then(|v| v.as_str())
				.unwrap_or(model)
				.to_string(),
			created: std::time::SystemTime::now()
				.duration_since(std::time::UNIX_EPOCH)
				.map(|d| d.as_secs())
				.unwrap_or(0),
		}
	}

	/// One `chat.completion.chunk` with a single choice.
	fn chunk(&self, w: &mut dyn Write, delta: &str, finish: Option<&str>) -> std::io::Result<()> {
		let fr = finish.map_or("null".to_string(), jstr);
		let ev = format!(
			"{{\"id\":{id},\"object\":\"chat.completion.chunk\",\"created\":{cr},\"model\":{m},\"choices\":[{{\"index\":0,\"delta\":{delta},\"finish_reason\":{fr}}}]}}",
			id = jstr(&self.id),
			cr = self.created,
			m = jstr(&self.model),
		);
		sse_chunk(w, &format!("data: {ev}\n\n"))
	}
}

fn finish_reason(stop: Stop) -> &'static str {
	match stop {
		Stop::EndTurn | Stop::StopSequence => "stop",
		Stop::MaxTokens => "length",
		Stop::ToolUse => "tool_calls",
	}
}

fn usage_json(u: &Usage) -> String {
	format!(
		"{{\"prompt_tokens\":{i},\"completion_tokens\":{o},\"total_tokens\":{t},\"prompt_tokens_details\":{{\"cached_tokens\":{c}}}}}",
		i = u.input,
		o = u.output,
		t = u.input + u.output,
		c = u.cached,
	)
}

/// One entry of `tool_calls`; `index` is only part of the streamed form.
fn tool_call_json(n: usize, c: &ParsedCall, with_index: bool) -> String {
	format!(
		"{{{idx}\"id\":{id},\"type\":\"function\",\"function\":{{\"name\":{name},\"arguments\":{args}}}}}",
		idx = if with_index { format!("\"index\":{n},") } else { String::new() },
		id = jstr(&tools::new_tool_id(n).replacen("toolu_", "call_", 1)),
		name = jstr(&c.name),
		args = jstr(&c.input_json()),
	)
}

impl Wire for OpenAiWire {
	fn start(&mut self, w: &mut dyn Write, _input_tokens: usize) -> std::io::Result<()> {
		if !self.stream {
			return Ok(());
		}
		sse_begin(w)?;
		self.chunk(w, "{\"role\":\"assistant\",\"content\":\"\"}", None)
	}

	fn text(&mut self, w: &mut dyn Write, text: &str) -> std::io::Result<()> {
		if !self.stream {
			return Ok(());
		}
		self.chunk(w, &format!("{{\"content\":{}}}", jstr(text)), None)
	}

	/// As llama-server streams it (`reasoning_format: deepseek`), which is
	/// what Zed and the model card's client read.
	fn thinking(&mut self, w: &mut dyn Write, text: &str) -> std::io::Result<()> {
		if !self.stream {
			return Ok(());
		}
		self.chunk(
			w,
			&format!("{{\"reasoning_content\":{}}}", jstr(text)),
			None,
		)
	}

	fn finish(
		&mut self,
		w: &mut dyn Write,
		full: &str,
		thinking: &str,
		calls: &[ParsedCall],
		stop: Stop,
		usage: &Usage,
	) -> std::io::Result<()> {
		let fr = finish_reason(stop);
		if self.stream {
			// A call is only known once its closing tag has parsed, so each
			// arrives whole, in one chunk.
			for (n, c) in calls.iter().enumerate() {
				self.chunk(
					w,
					&format!("{{\"tool_calls\":[{}]}}", tool_call_json(n, c, true)),
					None,
				)?;
			}
			self.chunk(w, "{}", Some(fr))?;
			if self.include_usage {
				let ev = format!(
					"{{\"id\":{id},\"object\":\"chat.completion.chunk\",\"created\":{cr},\"model\":{m},\"choices\":[],\"usage\":{u}}}",
					id = jstr(&self.id),
					cr = self.created,
					m = jstr(&self.model),
					u = usage_json(usage),
				);
				sse_chunk(w, &format!("data: {ev}\n\n"))?;
			}
			sse_chunk(w, "data: [DONE]\n\n")?;
			return sse_end(w);
		}
		let content = if full.is_empty() && !calls.is_empty() {
			"null".to_string()
		} else {
			jstr(full)
		};
		let tc = if calls.is_empty() {
			String::new()
		} else {
			let v: Vec<String> = calls
				.iter()
				.enumerate()
				.map(|(n, c)| tool_call_json(n, c, false))
				.collect();
			format!(",\"tool_calls\":[{}]", v.join(","))
		};
		let rc = if thinking.is_empty() {
			String::new()
		} else {
			format!(",\"reasoning_content\":{}", jstr(thinking))
		};
		let body = format!(
			"{{\"id\":{id},\"object\":\"chat.completion\",\"created\":{cr},\"model\":{m},\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":{content}{rc}{tc}}},\"finish_reason\":{fr}}}],\"usage\":{u}}}",
			id = jstr(&self.id),
			cr = self.created,
			m = jstr(&self.model),
			fr = jstr(fr),
			u = usage_json(usage),
		);
		respond_json(w, 200, "OK", &body)
	}
}

/// `POST /v1/chat/completions`.
pub fn handle_chat(w: &mut dyn Write, loaded: &Loaded, req: &Json) -> std::io::Result<()> {
	let mreq = match to_messages_request(req) {
		Ok(r) => r,
		Err(e) => return crate::write_error(w, 400, &e),
	};
	let mut wire = OpenAiWire::new(req, loaded.model_id());
	// No `max_tokens` means "until the model stops", as in llama-server;
	// `generate` caps it at the end of the context.
	crate::generate(w, loaded, &mreq, &mut wire, usize::MAX)
}

/// `GET /props`, the part of llama-server's answer clients read: the context
/// size, vision, and what the chat template supports. Zed takes its context
/// window and its tool / image switches for the model from here.
///
/// `supports_preserve_reasoning` is true: an assistant message's
/// `reasoning_content` is replayed into its `<think>` block, as the template
/// does (`preserve_thinking`).
pub fn props_json(loaded: &Loaded) -> String {
	format!(
		"{{\"default_generation_settings\":{{\"id\":0,\"n_ctx\":{ctx},\"params\":{{}}}},\"total_slots\":1,\"model_alias\":{name},\"model_path\":{name},\"modalities\":{{\"vision\":{vision},\"audio\":false}},\"chat_template_caps\":{{\"supports_tools\":true,\"supports_tool_calls\":true,\"supports_system_role\":true,\"supports_parallel_tool_calls\":true,\"supports_preserve_reasoning\":true}},\"build_info\":\"qwen.rs\"}}",
		ctx = loaded.ctx(),
		name = jstr(loaded.model_id()),
		vision = loaded.vision.is_some() || matches!(&loaded.backend, crate::Backend::Engine { info, .. } if info.image.is_some()),
	)
}

/// The `meta` object llama-server puts on its `/v1/models` entry. Zed falls back
/// to `n_ctx` from it when `/props` is unavailable.
pub fn model_meta_json(loaded: &Loaded) -> String {
	let (ctx_train, n_vocab, n_embd) = loaded.dims();
	format!(
		"{{\"n_ctx\":{},\"n_ctx_train\":{ctx_train},\"n_vocab\":{n_vocab},\"n_embd\":{n_embd}}}",
		loaded.ctx()
	)
}

#[cfg(test)]
mod tests {
	use super::*;

	fn tr(j: &str) -> Json {
		to_messages_request(&Json::parse(j.as_bytes()).unwrap()).unwrap()
	}

	#[test]
	fn system_messages_are_hoisted_and_joined() {
		let r = tr(
			r#"{"messages":[{"role":"system","content":"A"},{"role":"user","content":"hi"},{"role":"system","content":[{"type":"text","text":"B"}]}]}"#,
		);
		assert_eq!(r.get("system").and_then(|v| v.as_str()), Some("A\n\nB"));
		assert_eq!(r.get("messages").unwrap().as_array().unwrap().len(), 1);
	}

	#[test]
	fn tool_round_trip_becomes_tool_use_and_tool_result() {
		let r = tr(r#"{"messages":[
            {"role":"user","content":"list"},
            {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"ls","arguments":"{\"path\":\"/tmp\",\"n\":3}"}}]},
            {"role":"tool","tool_call_id":"call_1","content":"a.txt"}
        ],
        "tools":[{"type":"function","function":{"name":"ls","description":"List","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}],
        "tool_choice":"required"}"#);
		let msgs = r.get("messages").unwrap().as_array().unwrap();
		let call = &msgs[1].get("content").unwrap().as_array().unwrap()[0];
		assert_eq!(call.get("type").unwrap().as_str(), Some("tool_use"));
		assert_eq!(
			call.get("input").unwrap().to_string(),
			r#"{"path":"/tmp","n":3}"#
		);
		let res = &msgs[2].get("content").unwrap().as_array().unwrap()[0];
		assert_eq!(res.get("type").unwrap().as_str(), Some("tool_result"));
		assert_eq!(msgs[2].get("role").unwrap().as_str(), Some("user"));

		let specs = tools::specs_from_request(&r);
		assert_eq!(specs.len(), 1);
		assert_eq!(
			specs[0].params,
			vec![("path".to_string(), "string".to_string())]
		);
		assert_eq!(tools::choice_from_request(&r), tools::Choice::Any);

		// And the turns the template sees.
		let turns = crate::turns_from_request(&r, &mut |_| unreachable!()).unwrap();
		assert!(matches!(&turns[1], tokenizer::Turn::Assistant { calls, .. } if calls.len() == 1));
		assert!(
			matches!(&turns[2], tokenizer::Turn::ToolResults(v) if v == &vec!["a.txt".to_string()])
		);
	}

	/// Zed's `read_file` on a picture: a `tool` message whose content holds an
	/// `image_url` part. It reaches the tool result as an image block.
	#[test]
	fn tool_message_image_url_becomes_an_image_block() {
		let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
		let r = tr(&format!(
			r#"{{"messages":[
            {{"role":"user","content":"look"}},
            {{"role":"assistant","content":null,"tool_calls":[{{"id":"c","type":"function","function":{{"name":"read_file","arguments":"{{}}"}}}}]}},
            {{"role":"tool","tool_call_id":"c","content":[{{"type":"image_url","image_url":{{"url":"data:image/png;base64,{png}"}}}}]}}
        ]}}"#
		));
		let mut n = 0;
		let turns = crate::turns_from_request(&r, &mut |_| {
			n += 1;
			Ok("[img]".to_string())
		})
		.unwrap();
		assert_eq!(n, 1);
		assert!(
			matches!(&turns[2], tokenizer::Turn::ToolResults(v) if v == &vec!["[img]".to_string()])
		);
	}

	#[test]
	fn named_tool_choice_stop_and_limits() {
		let r = tr(
			r#"{"messages":[{"role":"user","content":"x"}],"tool_choice":{"type":"function","function":{"name":"f"}},"stop":"END","max_completion_tokens":7,"repeat_penalty":1.1,"chat_template_kwargs":{"enable_thinking":true}}"#,
		);
		assert_eq!(
			tools::choice_from_request(&r),
			tools::Choice::Named("f".to_string())
		);
		assert_eq!(r.get("stop_sequences").unwrap().to_string(), r#"["END"]"#);
		assert_eq!(r.get("max_tokens").and_then(|v| v.as_f64()), Some(7.0));
		assert_eq!(
			r.get("repetition_penalty").and_then(|v| v.as_f64()),
			Some(1.1)
		);
		assert_eq!(
			r.get("thinking").unwrap().to_string(),
			r#"{"type":"enabled"}"#
		);
	}

	#[test]
	fn thinking_defaults_on_as_in_llama_server() {
		// Zed's llama.cpp provider sends no thinking switch at all: llama-server
		// then renders the template's default, thinking on.
		let r = tr(
			r#"{"model":"m","messages":[{"role":"user","content":"x"}],"stream":true,"stream_options":{"include_usage":true}}"#,
		);
		assert_eq!(
			r.get("thinking").unwrap().to_string(),
			r#"{"type":"enabled"}"#
		);
		let t = crate::thinking_from_request(&r).unwrap();
		assert!(t.think.enabled());
		// An explicit off still wins.
		let r = tr(r#"{"messages":[{"role":"user","content":"x"}],"reasoning_effort":"none"}"#);
		assert_eq!(
			r.get("thinking").unwrap().to_string(),
			r#"{"type":"disabled"}"#
		);
	}

	#[test]
	fn image_url_part_becomes_an_image_block() {
		let r = tr(
			r#"{"messages":[{"role":"user","content":[{"type":"text","text":"what"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}"#,
		);
		let blocks = r.get("messages").unwrap().as_array().unwrap()[0]
			.get("content")
			.unwrap()
			.as_array()
			.unwrap()
			.to_vec();
		assert_eq!(blocks[1].get("type").unwrap().as_str(), Some("image"));
		assert_eq!(
			blocks[1]
				.get("source")
				.unwrap()
				.get("data")
				.unwrap()
				.as_str(),
			Some("data:image/png;base64,AAAA")
		);
	}

	fn call() -> ParsedCall {
		ParsedCall {
			name: "ls".to_string(),
			input: Json::parse(br#"{"path":"/tmp"}"#).unwrap(),
		}
	}

	#[test]
	fn streamed_response_shape() {
		let mut wire = OpenAiWire {
			stream: true,
			include_usage: true,
			id: "chatcmpl-1".to_string(),
			model: "m".to_string(),
			created: 0,
		};
		let mut out = Vec::new();
		wire.start(&mut out, 5).unwrap();
		wire.text(&mut out, "hi \"x\"").unwrap();
		let u = Usage {
			input: 5,
			output: 3,
			cached: 2,
		};
		wire.finish(&mut out, "hi", "", &[call()], Stop::ToolUse, &u)
			.unwrap();
		let text = String::from_utf8(out).unwrap();
		// Every event a client parses is one `data:` line of JSON.
		let events: Vec<&str> = text
			.lines()
			.filter_map(|l| l.strip_prefix("data: "))
			.collect();
		assert_eq!(events.last(), Some(&"[DONE]"));
		for e in &events[..events.len() - 1] {
			Json::parse(e.as_bytes()).unwrap_or_else(|err| panic!("{err}: {e}"));
		}
		let tc = Json::parse(events[2].as_bytes()).unwrap();
		let f = &tc.get("choices").unwrap().as_array().unwrap()[0]
			.get("delta")
			.unwrap()
			.get("tool_calls")
			.unwrap()
			.as_array()
			.unwrap()[0];
		assert_eq!(f.get("index").and_then(|v| v.as_f64()), Some(0.0));
		assert_eq!(
			f.get("function")
				.unwrap()
				.get("arguments")
				.unwrap()
				.as_str(),
			Some(r#"{"path":"/tmp"}"#)
		);
		assert!(events[3].contains(r#""finish_reason":"tool_calls""#));
		assert!(
			events[4].contains(r#""choices":[]"#) && events[4].contains(r#""cached_tokens":2"#)
		);
	}

	#[test]
	fn whole_response_shape() {
		let mut wire = OpenAiWire {
			stream: false,
			include_usage: false,
			id: "c".into(),
			model: "m".into(),
			created: 0,
		};
		let mut out = Vec::new();
		wire.finish(
			&mut out,
			"",
			"",
			&[call()],
			Stop::ToolUse,
			&Usage {
				input: 1,
				output: 2,
				cached: 0,
			},
		)
		.unwrap();
		let text = String::from_utf8(out).unwrap();
		let body = Json::parse(text.split("\r\n\r\n").nth(1).unwrap().as_bytes()).unwrap();
		let ch = &body.get("choices").unwrap().as_array().unwrap()[0];
		assert_eq!(
			ch.get("finish_reason").unwrap().as_str(),
			Some("tool_calls")
		);
		let msg = ch.get("message").unwrap();
		assert!(matches!(msg.get("content"), Some(Json::Null)));
		assert_eq!(msg.get("tool_calls").unwrap().as_array().unwrap().len(), 1);
		assert_eq!(
			body.get("usage")
				.unwrap()
				.get("total_tokens")
				.and_then(|v| v.as_f64()),
			Some(3.0)
		);
	}
}
