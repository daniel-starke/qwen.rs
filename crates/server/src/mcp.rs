//! MCP server over Streamable HTTP (`POST /mcp`, protocol 2025-06-18):
//! `initialize`, `ping`, `tools/list`, `tools/call` (`generate_image`,
//! `edit_image`), `notifications/cancelled`, and `roots/list` asked of the
//! client. A `tools/call` answers as an SSE stream when the client accepts
//! one: `notifications/progress` per stage and denoising step (with the
//! request's `progressToken`), a `roots/list` request when a file is to be
//! written and the client has roots, then the result. The client's answer
//! to `roots/list` arrives as a separate `POST /mcp` (202).
//!
//! Write paths (`docs/image.md` "Interface"): a file goes only inside one
//! of the client's roots or an `--image-out-dir`.

use crate::images::{self, Done, Job};
use crate::{jstr, Json};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const PROTOCOLS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

#[derive(Default)]
struct Session {
	/// The client declared the `roots` capability.
	roots_cap: bool,
	/// Its roots, once asked (cleared by `notifications/roots/list_changed`).
	roots: Option<Vec<PathBuf>>,
}

static SESSIONS: Mutex<Option<HashMap<String, Session>>> = Mutex::new(None);
/// Our requests to clients awaiting their answer, by JSON-RPC id.
static PENDING: Mutex<Option<HashMap<String, mpsc::Sender<Json>>>> = Mutex::new(None);
/// Running `tools/call`s: `(session, request id)` -> cancel flag.
static RUNNING: Mutex<Option<HashMap<(String, String), Arc<AtomicBool>>>> = Mutex::new(None);
static CTR: AtomicU64 = AtomicU64::new(1);

fn with<T, R>(
	m: &Mutex<Option<HashMap<String, T>>>,
	f: impl FnOnce(&mut HashMap<String, T>) -> R,
) -> R {
	let mut g = m.lock().unwrap_or_else(|e| e.into_inner());
	f(g.get_or_insert_with(HashMap::new))
}

fn header<'a>(h: &'a [(String, String)], k: &str) -> Option<&'a str> {
	h.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str())
}

fn id_key(id: &Json) -> String {
	id.to_string()
}

fn rpc_result(id: &Json, result: &str) -> String {
	format!(
		"{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{result}}}",
		id.to_string()
	)
}

fn rpc_error(id: &Json, code: i32, msg: &str) -> String {
	format!(
		"{{\"jsonrpc\":\"2.0\",\"id\":{},\"error\":{{\"code\":{code},\"message\":{}}}}}",
		id.to_string(),
		jstr(msg)
	)
}

fn reply(
	w: &mut dyn Write,
	code: u16,
	text: &str,
	session: Option<&str>,
	body: &str,
) -> std::io::Result<()> {
	let sid = session.map_or(String::new(), |s| format!("Mcp-Session-Id: {s}\r\n"));
	let ct = if body.is_empty() {
		String::new()
	} else {
		"Content-Type: application/json\r\n".into()
	};
	let r = format!(
		"HTTP/1.1 {code} {text}\r\n{ct}{sid}Access-Control-Allow-Origin: *\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
		body.len()
	);
	w.write_all(r.as_bytes())?;
	w.flush()
}

/// The response of a `tools/call`: an SSE stream of JSON-RPC messages when
/// the client accepts one (`stream`), else one JSON body.
struct Sse<'a> {
	w: &'a mut dyn Write,
	stream: bool,
	open: bool,
}

impl Sse<'_> {
	/// Send the stream's head (once). A `tools/call` opens it right away:
	/// Zed's transport sends its messages one at a time and waits for each
	/// response's head, so a head held back for the minutes of a job would
	/// hold back everything else the client has to say (another call,
	/// `notifications/cancelled`).
	fn open(&mut self) -> std::io::Result<()> {
		if self.stream && !self.open {
			self.w.write_all(crate::SSE_HEAD)?;
			self.w.flush()?;
			self.open = true;
		}
		Ok(())
	}

	/// A message before the result (a notification or our request);
	/// dropped without a stream.
	fn send(&mut self, msg: &str) -> std::io::Result<()> {
		if !self.stream {
			return Ok(());
		}
		self.open()?;
		let ev = format!("event: message\ndata: {msg}\n\n");
		self.w
			.write_all(format!("{:x}\r\n{ev}\r\n", ev.len()).as_bytes())?;
		self.w.flush()
	}

	/// End a stream without a result.
	fn end(self) -> std::io::Result<()> {
		if !self.open {
			// nothing was sent: an empty answer
			return reply(self.w, 202, "Accepted", None, "");
		}
		self.w.write_all(b"0\r\n\r\n")?;
		self.w.flush()
	}

	/// The result, ending the response.
	fn finish(mut self, msg: &str) -> std::io::Result<()> {
		if !self.stream {
			return reply(self.w, 200, "OK", None, msg);
		}
		self.send(msg)?;
		self.w.write_all(b"0\r\n\r\n")?;
		self.w.flush()
	}
}

/// `POST /mcp` (and `GET` / `DELETE`).
pub(crate) fn handle(
	w: &mut dyn Write,
	method: &str,
	headers: &[(String, String)],
	body: &[u8],
) -> std::io::Result<()> {
	let sid = header(headers, "mcp-session-id").map(str::to_string);
	match method {
		"GET" => return reply(w, 405, "Method Not Allowed", None, ""),
		"DELETE" => {
			if let Some(s) = &sid {
				with(&SESSIONS, |m| m.remove(s));
			}
			return reply(w, 200, "OK", None, "");
		}
		"OPTIONS" => {
			return w.write_all(b"HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Expose-Headers: Mcp-Session-Id\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
		}
		_ => {}
	}
	let msg = match Json::parse(body) {
		Ok(m) => m,
		Err(e) => {
			return reply(
				w,
				400,
				"Bad Request",
				None,
				&rpc_error(&Json::Null, -32700, &format!("parse error: {e}")),
			)
		}
	};
	if let Json::Arr(_) = msg {
		return reply(
			w,
			400,
			"Bad Request",
			None,
			&rpc_error(&Json::Null, -32600, "batches are not supported"),
		);
	}
	let id = msg.get("id").cloned();
	let m = msg.get("method").and_then(Json::as_str).unwrap_or("");
	// a response to one of our requests (roots/list)
	if m.is_empty() {
		if let Some(id) = &id {
			if let Some(tx) = with(&PENDING, |p| p.remove(&id_key(id))) {
				let _ = tx.send(msg.clone());
			}
		}
		return reply(w, 202, "Accepted", None, "");
	}
	let params = msg.get("params").cloned().unwrap_or(Json::Null);
	let Some(id) = id else {
		// notifications
		match m {
			"notifications/cancelled" => {
				if let Some(r) = params.get("requestId") {
					let k = (sid.clone().unwrap_or_default(), id_key(r));
					let mut g = RUNNING.lock().unwrap_or_else(|e| e.into_inner());
					if let Some(c) = g.get_or_insert_with(HashMap::new).get(&k) {
						c.store(true, Ordering::Relaxed);
						eprintln!("mcp: request {} canceled", k.1);
					}
				}
			}
			"notifications/roots/list_changed" => {
				if let Some(s) = &sid {
					with(&SESSIONS, |m| {
						if let Some(x) = m.get_mut(s) {
							x.roots = None;
						}
					});
				}
			}
			_ => {}
		}
		return reply(w, 202, "Accepted", None, "");
	};
	match m {
		"initialize" => {
			let want = params
				.get("protocolVersion")
				.and_then(Json::as_str)
				.unwrap_or(PROTOCOLS[0]);
			let ver = if PROTOCOLS.contains(&want) {
				want
			} else {
				PROTOCOLS[0]
			};
			let s = format!(
				"{:016x}{:016x}",
				std::time::SystemTime::now()
					.duration_since(std::time::UNIX_EPOCH)
					.map_or(0, |d| d.as_nanos() as u64),
				CTR.fetch_add(1, Ordering::Relaxed)
			);
			let roots_cap = params
				.get("capabilities")
				.and_then(|c| c.get("roots"))
				.is_some();
			let client = params
				.get("clientInfo")
				.and_then(|c| c.get("name"))
				.and_then(Json::as_str)
				.unwrap_or("?")
				.to_string();
			eprintln!("mcp: initialize from {client} (protocol {ver}, roots {roots_cap})");
			with(&SESSIONS, |m| {
				m.insert(
					s.clone(),
					Session {
						roots_cap,
						roots: None,
					},
				)
			});
			let result = format!(
				"{{\"protocolVersion\":{},\"capabilities\":{{\"tools\":{{\"listChanged\":false}}}},\"serverInfo\":{{\"name\":\"qwen.rs\",\"title\":\"qwen.rs image generation\",\"version\":{}}},\"instructions\":{}}}",
				jstr(ver),
				jstr(env!("CARGO_PKG_VERSION")),
				jstr(instructions())
			);
			reply(w, 200, "OK", Some(&s), &rpc_result(&id, &result))
		}
		"ping" => reply(w, 200, "OK", None, &rpc_result(&id, "{}")),
		"tools/list" => reply(
			w,
			200,
			"OK",
			None,
			&rpc_result(&id, &format!("{{\"tools\":{}}}", tools_json())),
		),
		"tools/call" => {
			let sse = header(headers, "accept").is_some_and(|a| a.contains("text/event-stream"));
			call(w, sid, &id, &params, sse)
		}
		_ => reply(
			w,
			200,
			"OK",
			None,
			&rpc_error(&id, -32601, &format!("method not found: {m}")),
		),
	}
}

const INSTRUCTIONS: &str = "Local image generation with Qwen-Image-2.1 on this machine. generate_image makes an image from a detailed prompt (describe subject, composition, style, lighting, any text to render in quotes); edit_image changes or combines existing images (1-10, absolute paths or data: URLs) as the prompt says. Give output_path (inside the project) to save the PNG there. An image takes a few minutes on the CPU; progress is reported per step.";

const INSTRUCTIONS_H3: &str = "Local image generation with MiniMax-H3-Image on this machine. generate_image makes an image from a long, precise English description (subject, composition, camera, style, lighting, colors, any text to render in quotes); the model follows detailed descriptions, so write one rather than a short phrase. edit_image changes or combines 1-16 existing images (absolute paths or data: URLs); the prompt refers to them as <Picture 1>, <Picture 2>, ... in the order given. Images are generated at 4 megapixels or more (2048x2048 by default; smaller sizes are generated at 4 MP and scaled down) and look best toward 16 MP. Give output_path (inside the project) to save the PNG there. This is a large model: a 4 MP image takes about 12 minutes on the CPU (16 MP several times that), and every reference image makes each of the 6 steps longer (a 1 MP reference adds about a quarter to a 4 MP image, a 4 MP reference doubles it); progress is reported per step.";

fn instructions() -> &'static str {
	match images::info().steps {
		true => INSTRUCTIONS,
		false => INSTRUCTIONS_H3,
	}
}

fn tools_json() -> String {
	if !images::info().steps {
		return tools_json_h3();
	}
	let common = r#""prompt":{"type":"string","description":"What to draw: subject, composition, style, lighting, colors; text to render in quotes."},"aspect":{"type":"string","enum":["1:1","4:3","3:4","3:2","2:3","16:9","9:16"],"description":"Aspect ratio at about 1 megapixel (default 1:1, or the last input image's for an edit)."},"width":{"type":"integer","description":"Width in pixels (multiple of 32, 256-2048); with height, overrides aspect."},"height":{"type":"integer","description":"Height in pixels (multiple of 32, 256-2048)."},"seed":{"type":"integer","description":"Seed for a reproducible image (random when absent; returned)."},"steps":{"type":"integer","minimum":1,"maximum":50,"description":"Denoising steps (default 9: the model's 8 distilled steps plus one refinement step for fine texture). 8 = the model's schedule as shipped (~11 % faster, softer detail). Other counts stretch the schedule: slower, change the composition, and look over-sharpened from about 12 on; rarely useful."},"transparent":{"type":"boolean","description":"Keep the alpha channel (RGBA PNG) for a transparent background; ask for it in the prompt too."},"output_path":{"type":"string","description":OUTPUT_PATH_DOC},"return":{"type":"string","enum":["path","inline","both"],"description":"path: only save the file; inline: only return the image; both (default when output_path is given, else inline)."}"#;
	let dirs = images::cfg().map_or(&[][..], |c| &c.out_dirs[..]);
	let common = common.replace("OUTPUT_PATH_DOC", &jstr(&output_path_doc(dirs)));
	format!(
		r#"[{{"name":"generate_image","title":"Generate image","description":"Generate an image from a text prompt with Qwen-Image-2.1 (local). Returns the image inline (a preview of at most 1024 px) and/or saves the PNG to output_path.","inputSchema":{{"type":"object","properties":{{{common}}},"required":["prompt"]}}}},{{"name":"edit_image","title":"Edit image","description":"Edit or combine images with Qwen-Image-2.1 (local) as the prompt instructs (e.g. 'replace the sky with a sunset', 'put the cat from image 1 on the sofa of image 2'). Returns the result inline and/or saves it to output_path.","inputSchema":{{"type":"object","properties":{{"images":{{"type":"array","items":{{"type":"string"}},"minItems":1,"maxItems":10,"description":"Input images: absolute file paths or data: URLs."}},{common}}},"required":["prompt","images"]}}}}]"#
	)
}

/// The tools on MiniMax-H3-Image: 4 MP and up, up to 16 references, a fixed
/// schedule (no steps), no alpha.
fn tools_json_h3() -> String {
	let common = r#""prompt":{"type":"string","description":"A long, precise English description of the image: subject, composition, camera and lens, style, lighting, colors; text to render in quotes. For an edit, refer to the inputs as <Picture 1>, <Picture 2>, ..."},"aspect":{"type":"string","enum":["1:1","4:3","3:4","3:2","2:3","16:9","9:16","21:9","9:21"],"description":"Aspect ratio at 4 megapixels (default 1:1, or the last input image's for an edit)."},"width":{"type":"integer","description":"Width in pixels (multiple of 32, 256-8192, at most 16 megapixels with height); with height, overrides aspect. Below 4 MP the image is generated at 4 MP and scaled down."},"height":{"type":"integer","description":"Height in pixels (multiple of 32, 256-8192)."},"seed":{"type":"integer","description":"Seed for a reproducible image (random when absent; returned)."},"output_path":{"type":"string","description":OUTPUT_PATH_DOC},"return":{"type":"string","enum":["path","inline","both"],"description":"path: only save the file; inline: only return the image; both (default when output_path is given, else inline)."}"#;
	let dirs = images::cfg().map_or(&[][..], |c| &c.out_dirs[..]);
	let common = common.replace("OUTPUT_PATH_DOC", &jstr(&output_path_doc(dirs)));
	format!(
		r#"[{{"name":"generate_image","title":"Generate image","description":"Generate an image from a detailed text description with MiniMax-H3-Image (local, 4-16 megapixels, about 12 minutes for 4 MP). Returns the image inline (a preview of at most 1024 px) and/or saves the PNG to output_path.","inputSchema":{{"type":"object","properties":{{{common}}},"required":["prompt"]}}}},{{"name":"edit_image","title":"Edit image","description":"Edit or combine 1-16 images with MiniMax-H3-Image (local) as the prompt instructs, referring to them as <Picture 1>, <Picture 2>, ... (e.g. 'Put the cat from <Picture 1> on the sofa of <Picture 2>'). Each reference makes the job slower: a 1 MP reference adds about a quarter to a 4 MP image, a 4 MP reference doubles it. Returns the result inline and/or saves it to output_path.","inputSchema":{{"type":"object","properties":{{"images":{{"type":"array","items":{{"type":"string"}},"minItems":1,"maxItems":16,"description":"Input images in order (<Picture 1> first): absolute file paths or data: URLs."}},{common}}},"required":["prompt","images"]}}}}]"#
	)
}

/// A tool error: a result with `isError`, as MCP wants for failures the
/// model should see.
fn tool_error(id: &Json, msg: &str) -> String {
	rpc_result(
		id,
		&format!(
			"{{\"content\":[{{\"type\":\"text\",\"text\":{}}}],\"isError\":true}}",
			jstr(msg)
		),
	)
}

fn job_of(name: &str, a: &Json) -> Result<Job, String> {
	let prompt = a
		.get("prompt")
		.and_then(Json::as_str)
		.filter(|p| !p.trim().is_empty())
		.ok_or("prompt is required")?
		.to_string();
	let num = |k: &str| a.get(k).and_then(Json::as_f64).map(|v| v as i64);
	let (width, height) = match (num("width"), num("height")) {
		(Some(w), Some(h)) if w > 0 && h > 0 => images::check_size(w as usize, h as usize)?,
		(None, None) => match a.get("aspect").and_then(Json::as_str) {
			Some(s) => images::aspect_size(s)?,
			None => (0, 0),
		},
		_ => return Err("give both width and height, or aspect".into()),
	};
	let mut job = Job {
		prompt,
		width,
		height,
		seed: num("seed").filter(|&s| s >= 0).map(|s| s as u64),
		steps: match num("steps") {
			Some(n) => images::steps_of(n)?,
			None => 0,
		},
		images: Vec::new(),
		transparent: a
			.get("transparent")
			.and_then(Json::as_bool)
			.unwrap_or(false),
	};
	if name == "edit_image" {
		let list = a
			.get("images")
			.and_then(Json::as_array)
			.ok_or("images is required")?;
		let max = images::info().max_images;
		if list.is_empty() || list.len() > max {
			return Err(format!("edit_image takes 1..{max} images"));
		}
		for v in list {
			let s = v.as_str().ok_or("images: want strings")?;
			if !s.starts_with("data:")
				&& !std::path::Path::new(s.strip_prefix("file://").unwrap_or(s)).is_absolute()
			{
				return Err(format!("image {s}: give an absolute path or a data: URL"));
			}
			job.images.push(images::image_arg(s)?);
		}
	}
	Ok(job)
}

/// A `file://` URI as a path.
fn uri_path(u: &str) -> Option<PathBuf> {
	let r = u.strip_prefix("file://")?;
	let r = r.strip_prefix("localhost").unwrap_or(r);
	let mut b = Vec::new();
	let s = r.as_bytes();
	let mut i = 0;
	while i < s.len() {
		if s[i] == b'%' && i + 2 < s.len() {
			if let Ok(v) = u8::from_str_radix(&r[i + 1..i + 3], 16) {
				b.push(v);
				i += 3;
				continue;
			}
		}
		b.push(s[i]);
		i += 1;
	}
	let p = String::from_utf8(b).ok()?;
	// file:///C:/x -> C:/x on Windows
	let p = if cfg!(windows) && p.len() > 2 && p.as_bytes()[0] == b'/' && p.as_bytes()[2] == b':' {
		p[1..].to_string()
	} else {
		p
	};
	Some(PathBuf::from(p))
}

/// The client's roots: cached per session, else asked over `sse` (the
/// answer comes in another POST), waiting up to 15 s.
fn roots(sid: Option<&str>, sse: &mut Sse) -> Vec<PathBuf> {
	let Some(s) = sid else { return Vec::new() };
	let (cap, cached) = with(&SESSIONS, |m| {
		m.get(s)
			.map_or((false, None), |x| (x.roots_cap, x.roots.clone()))
	});
	if let Some(r) = cached {
		return r;
	}
	if !cap || !sse.stream {
		return Vec::new();
	}
	let rid = format!("qwen-roots-{}", CTR.fetch_add(1, Ordering::Relaxed));
	let (tx, rx) = mpsc::channel();
	with(&PENDING, |p| p.insert(jstr(&rid), tx));
	if sse
		.send(&format!(
			"{{\"jsonrpc\":\"2.0\",\"id\":{},\"method\":\"roots/list\"}}",
			jstr(&rid)
		))
		.is_err()
	{
		return Vec::new();
	}
	let r = match rx.recv_timeout(Duration::from_secs(15)) {
		Ok(m) => m
			.get("result")
			.and_then(|r| r.get("roots"))
			.and_then(Json::as_array)
			.map(|a| {
				a.iter()
					.filter_map(|x| x.get("uri").and_then(Json::as_str).and_then(uri_path))
					.collect()
			})
			.unwrap_or_default(),
		Err(_) => {
			eprintln!("mcp: no answer to roots/list");
			with(&PENDING, |p| p.remove(&jstr(&rid)));
			Vec::new()
		}
	};
	with(&SESSIONS, |m| {
		if let Some(x) = m.get_mut(s) {
			x.roots = Some(r.clone());
		}
	});
	r
}

/// A refused `output_path`, with what to do about it when nothing at all
/// is writable: the agent relays this to the user, and "not inside the
/// client's roots" alone sent it guessing at path spellings.
fn write_error(e: String, sid: Option<&str>, allowed: &[PathBuf]) -> String {
	if !allowed.is_empty() {
		return e;
	}
	let cap = sid.is_some_and(|s| with(&SESSIONS, |m| m.get(s).is_some_and(|x| x.roots_cap)));
	let why = if cap {
		"the client listed no roots"
	} else {
		"this client does not support MCP roots (Zed does not)"
	};
	format!(
		"{e}. Nothing is writable: {why} and the server was started without --image-out-dir. \
		 The user has to restart the server with --image-out-dir DIR (a directory containing \
		 the project); retrying with another path spelling will not help. Without output_path \
		 the image is returned inline."
	)
}

/// `output_path`'s description, naming the server's `--image-out-dir`s so
/// an agent picks a path that will be accepted.
fn output_path_doc(dirs: &[PathBuf]) -> String {
	let mut d = String::from(
		"Where to save the PNG: absolute, or relative to the first workspace root; must lie inside the client's roots or the server's --image-out-dir",
	);
	if dirs.is_empty() {
		d.push_str(" (none configured)");
	} else {
		let l: Vec<String> = dirs.iter().map(|p| p.display().to_string()).collect();
		d.push_str(&format!(" ({})", l.join(", ")));
	}
	d.push('.');
	d
}

fn call(
	w: &mut dyn Write,
	sid: Option<String>,
	id: &Json,
	params: &Json,
	use_sse: bool,
) -> std::io::Result<()> {
	let name = params.get("name").and_then(Json::as_str).unwrap_or("");
	if name != "generate_image" && name != "edit_image" {
		return reply(
			w,
			200,
			"OK",
			None,
			&rpc_error(id, -32602, &format!("unknown tool {name:?}")),
		);
	}
	let empty = Json::Obj(Vec::new());
	let args = params.get("arguments").unwrap_or(&empty);
	let token = params
		.get("_meta")
		.and_then(|m| m.get("progressToken"))
		.cloned();
	let job = match job_of(name, args) {
		Ok(j) => j,
		Err(e) => return reply(w, 200, "OK", None, &tool_error(id, &e)),
	};
	let want = args
		.get("output_path")
		.and_then(Json::as_str)
		.map(str::to_string);
	let ret = args.get("return").and_then(Json::as_str).unwrap_or("");
	let transparent = job.transparent;

	let mut sse = Sse {
		w,
		stream: use_sse,
		open: false,
	};
	sse.open()?;
	// the allowed directories, before the minutes of work
	let mut allowed: Vec<PathBuf> = Vec::new();
	let need_dirs = want.is_some();
	if need_dirs {
		allowed = roots(sid.as_deref(), &mut sse);
		if let Some(c) = images::cfg() {
			allowed.extend(c.out_dirs.iter().cloned());
		}
		if let Err(e) = images::out_path(want.as_deref(), &allowed, 0) {
			return sse.finish(&tool_error(id, &write_error(e, sid.as_deref(), &allowed)));
		}
	}

	let cancel = Arc::new(AtomicBool::new(false));
	let rkey = (sid.clone().unwrap_or_default(), id_key(id));
	RUNNING
		.lock()
		.unwrap_or_else(|e| e.into_inner())
		.get_or_insert_with(HashMap::new)
		.insert(rkey.clone(), cancel.clone());
	let mut last = -1f64;
	let total = 3.0 + 8.0;
	let t0 = std::time::Instant::now();
	let r = {
		let cancel2 = cancel.clone();
		let mut progress = |stage: &str, i: usize, n: usize| {
			let at = match stage {
				s if s.starts_with("load") => {
					0.5 * i as f64 / n.max(1) as f64 + if s.contains("text") { 0.0 } else { 0.5 }
				}
				"text encoder" => 1.0,
				"vae encoder" => 1.5,
				"denoise" => 1.0 + i as f64,
				_ => 2.0 + n.max(8) as f64,
			};
			if at <= last {
				return;
			}
			last = at;
			if stage.starts_with("load") && !i.is_multiple_of(4) && i != n {
				return;
			}
			let msg = if n > 0 {
				format!("{stage} {i}/{n} ({:.0} s)", t0.elapsed().as_secs_f64())
			} else {
				format!("{stage} ({:.0} s)", t0.elapsed().as_secs_f64())
			};
			if let Some(t) = &token {
				let n = format!(
					"{{\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{{\"progressToken\":{},\"progress\":{at},\"total\":{total},\"message\":{}}}}}",
					t.to_string(),
					jstr(&msg)
				);
				if sse.send(&n).is_err() {
					// the client went away
					cancel2.store(true, Ordering::Relaxed);
				}
			}
		};
		images::run(job, cancel.clone(), &mut progress)
	};
	RUNNING
		.lock()
		.unwrap_or_else(|e| e.into_inner())
		.get_or_insert_with(HashMap::new)
		.remove(&rkey);
	let msg = match r {
		Ok(d) => match finish(&d, want.as_deref(), &allowed, ret, transparent) {
			Ok(result) => rpc_result(id, &result),
			Err(e) => tool_error(id, &e),
		},
		// a canceled request gets no response (the spec's SHOULD); a
		// stream just ends
		Err(_) if cancel.load(Ordering::Relaxed) && sse.stream => return sse.end(),
		Err(e) => tool_error(id, &e),
	};
	sse.finish(&msg)
}

/// Write the file and build the tool result.
fn finish(
	d: &Done,
	want: Option<&str>,
	allowed: &[PathBuf],
	ret: &str,
	alpha: bool,
) -> Result<String, String> {
	let path = images::out_path(want, allowed, d.seed)?;
	if let Some(p) = &path {
		if let Some(dir) = p.parent() {
			std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
		}
		std::fs::write(p, &d.png).map_err(|e| format!("{}: {e}", p.display()))?;
	}
	let inline = match ret {
		"path" if path.is_some() => false,
		"path" => true,
		_ => true,
	};
	let mut text = match &path {
		Some(p) => format!("Saved {}: ", p.display()),
		None => String::new(),
	};
	text.push_str(&format!(
		"{}x{} PNG, seed {}, {:.1} s",
		d.width, d.height, d.seed, d.secs
	));
	if let Some(l) = d.load_secs {
		text.push_str(&format!(" (model load {l:.1} s)"));
	}
	let mut content = vec![format!("{{\"type\":\"text\",\"text\":{}}}", jstr(&text))];
	if inline {
		let png = images::preview_png(d, 1024, alpha)?;
		content.push(format!(
			"{{\"type\":\"image\",\"data\":{},\"mimeType\":\"image/png\"}}",
			jstr(&images::base64_encode(&png))
		));
	}
	Ok(format!(
		"{{\"content\":[{}],\"isError\":false}}",
		content.join(",")
	))
}

#[cfg(test)]
mod tests {
	use super::*;

	fn post(body: &str, headers: &[(&str, &str)]) -> String {
		let h: Vec<(String, String)> = headers
			.iter()
			.map(|(a, b)| (a.to_string(), b.to_string()))
			.collect();
		let mut out = Vec::new();
		handle(&mut out, "POST", &h, body.as_bytes()).unwrap();
		String::from_utf8(out).unwrap()
	}

	#[test]
	fn initialize_and_list() {
		let r = post(
			r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{"roots":{}},"clientInfo":{"name":"t"}}}"#,
			&[],
		);
		assert!(r.contains("Mcp-Session-Id: "), "{r}");
		assert!(r.contains("\"protocolVersion\":\"2025-06-18\""));
		let r = post(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#, &[]);
		assert!(r.contains("generate_image") && r.contains("edit_image"));
		let body = &r[r.find("\r\n\r\n").unwrap() + 4..];
		let j = Json::parse(body.as_bytes()).unwrap();
		assert_eq!(
			j.get("result")
				.unwrap()
				.get("tools")
				.unwrap()
				.as_array()
				.unwrap()
				.len(),
			2
		);
		let r = post(
			r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
			&[],
		);
		assert!(r.starts_with("HTTP/1.1 202"));
		let r = post(r#"{"jsonrpc":"2.0","id":3,"method":"nope"}"#, &[]);
		assert!(r.contains("-32601"));
	}

	#[test]
	fn bad_arguments_are_tool_errors() {
		let r = post(
			r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"generate_image","arguments":{"prompt":"x","width":1000,"height":512}}}"#,
			&[],
		);
		assert!(
			r.contains("\"isError\":true") && r.contains("multiples of 32"),
			"{r}"
		);
		let r = post(
			r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"edit_image","arguments":{"prompt":"x","images":["rel.png"]}}}"#,
			&[],
		);
		assert!(r.contains("absolute path"), "{r}");
	}

	/// A streamed call opens its stream at once, closes the connection (a
	/// pooling client must not reuse it), and a refused path with nothing
	/// writable says what to do instead of only "not inside the roots".
	#[test]
	fn streamed_call_closes_and_explains_refused_path() {
		let r = post(
			r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"generate_image","arguments":{"prompt":"x","width":512,"height":512,"output_path":"images/a.png"}}}"#,
			&[("accept", "application/json, text/event-stream")],
		);
		assert!(
			r.starts_with("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream"),
			"{r}"
		);
		let head = &r[..r.find("\r\n\r\n").unwrap()];
		assert!(
			head.contains("Connection: close") && !head.contains("keep-alive"),
			"{head}"
		);
		assert!(
			r.contains("\"isError\":true") && r.contains("--image-out-dir DIR"),
			"{r}"
		);
		assert!(r.contains("does not support MCP roots"), "{r}");
		assert!(r.ends_with("0\r\n\r\n"), "{r}");
		assert!(!crate::SSE_HEAD.windows(10).any(|w| w == b"keep-alive"));
	}

	#[test]
	fn file_uris() {
		if cfg!(windows) {
			assert_eq!(
				uri_path("file:///C:/a%20b/c").unwrap(),
				PathBuf::from("C:/a b/c")
			);
		} else {
			assert_eq!(
				uri_path("file:///a%20b/c").unwrap(),
				PathBuf::from("/a b/c")
			);
		}
	}
}
