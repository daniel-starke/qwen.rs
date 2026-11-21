//! Image generation on the server (`docs/image.md` "Server behavior"):
//! the model named by `--image-model` is loaded on the first image job and
//! dropped after `--image-idle` seconds without one; one job runs at a time
//! and holds the devices, so a chat request waits for a running image job
//! and an image job for running chat requests ([`DEVICES`]). The OpenAI
//! Images routes live here; MCP (`mcp.rs`) runs the same jobs.

use crate::{jstr, respond_json, write_error, Json};
use imagegen::{ImageSrc, Model, Options, WFmt};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

/// The devices (cores, NPU, iGPU) as one lock: chat requests share it
/// (read), an image job holds it alone (write).
pub(crate) static DEVICES: RwLock<()> = RwLock::new(());

/// What `--image-model` / `--image-out-dir` / `--image-idle` configure.
#[derive(Clone, Debug)]
pub struct ImageCfg {
	/// The checkpoint directory (`Qwen-Image-2.1-Turbo`, diffusers layout).
	pub dir: PathBuf,
	/// Directories an MCP client may write into besides its roots; the first
	/// is where an unnamed file goes.
	pub out_dirs: Vec<PathBuf>,
	/// Seconds without a job after which the model is dropped.
	pub idle_secs: u64,
	/// Weight format of the CPU backend.
	pub fmt: WFmt,
}

static CFG: OnceLock<ImageCfg> = OnceLock::new();

struct Slot {
	m: Arc<Model>,
	last: Instant,
}

static MODEL: Mutex<Option<Slot>> = Mutex::new(None);

/// What the chat engine frees before an image job ([`on_image_job`]).
type MakeRoom = Box<dyn Fn() + Send + Sync>;
static MAKE_ROOM: OnceLock<MakeRoom> = OnceLock::new();

/// Set what runs before each image job, once no chat request runs (the
/// server: the chat engine's `Engine::release_devices`).
pub(crate) fn on_image_job(f: impl Fn() + Send + Sync + 'static) {
	let _ = MAKE_ROOM.set(Box::new(f));
}

/// Drop the loaded model now (and the allocator's cached blocks of it).
fn unload(why: &str) {
	let mut g = MODEL.lock().unwrap_or_else(|e| e.into_inner());
	if g.take().is_some() {
		drop(g);
		let t = quant::alloc::trim();
		eprintln!(
			"image: model unloaded {why} ({:.1} GB of cached blocks released)",
			t as f64 / 1e9
		);
	}
}

/// Enable the image routes (call once, before serving). Starts the idle
/// unloader.
pub fn configure(cfg: ImageCfg) {
	let idle = Duration::from_secs(cfg.idle_secs.max(1));
	if cfg.out_dirs.is_empty() {
		eprintln!(
			"image: no --image-out-dir: MCP clients without roots (Zed declares none) cannot save files (output_path), only get images inline"
		);
	}
	if CFG.set(cfg).is_err() {
		return;
	}
	std::thread::spawn(move || loop {
		std::thread::sleep(idle.min(Duration::from_secs(15)));
		let mut g = MODEL.lock().unwrap_or_else(|e| e.into_inner());
		if let Some(s) = g.as_ref() {
			if s.last.elapsed() >= idle && Arc::strong_count(&s.m) == 1 {
				*g = None;
				// the freed weights would otherwise sit in the allocator's cache
				let t = quant::alloc::trim();
				eprintln!(
					"image: model unloaded after {} s idle ({:.1} GB of cached blocks released)",
					idle.as_secs(),
					t as f64 / 1e9
				);
			}
		}
	});
}

pub fn cfg() -> Option<&'static ImageCfg> {
	CFG.get()
}

/// The model id the image routes report.
pub const IMAGE_MODEL_ID: &str = "qwen-image-2.1-turbo";

/// One image job.
#[derive(Default)]
pub(crate) struct Job {
	pub prompt: String,
	/// 0 x 0: the default (1 MP; an edit takes its last image's aspect).
	pub width: usize,
	pub height: usize,
	pub seed: Option<u64>,
	/// Denoising steps, 0 = the model's default (`imagegen::Options::steps`).
	pub steps: usize,
	pub images: Vec<ImageSrc>,
	/// Keep the alpha channel (else the image is composited over white and
	/// written as RGB).
	pub transparent: bool,
}

/// What a job made.
pub(crate) struct Done {
	pub width: usize,
	pub height: usize,
	/// The final pixels (RGBA, alpha 255 unless `transparent`).
	pub rgba: Vec<u8>,
	pub png: Vec<u8>,
	pub seed: u64,
	/// Seconds from the start of the job (load included) to the PNG.
	pub secs: f64,
	/// Seconds the model load took when this job loaded it.
	pub load_secs: Option<f64>,
}

fn random_seed() -> u64 {
	let t = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map_or(0, |d| d.as_nanos() as u64);
	// splitmix64 of the clock
	let mut z = t.wrapping_add(0x9e37_79b9_7f4a_7c15);
	z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
	z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
	(z ^ (z >> 31)) % 4_294_967_296
}

/// Sides for a requested aspect at 1 MP (multiples of 32).
pub(crate) fn aspect_size(aspect: &str) -> Result<(usize, usize), String> {
	let (a, b) = aspect
		.split_once(':')
		.and_then(|(a, b)| Some((a.trim().parse::<f64>().ok()?, b.trim().parse::<f64>().ok()?)))
		.filter(|(a, b)| *a > 0.0 && *b > 0.0 && a / b <= 4.0 && b / a <= 4.0)
		.ok_or_else(|| format!("aspect {aspect:?}: want W:H, e.g. 16:9 (at most 4:1)"))?;
	Ok(imagegen::cond::dims(1024.0 * 1024.0, a / b))
}

/// Check explicit sides: multiples of 32, 256..2048.
pub(crate) fn check_size(w: usize, h: usize) -> Result<(usize, usize), String> {
	if !w.is_multiple_of(32)
		|| !h.is_multiple_of(32)
		|| !(256..=2048).contains(&w)
		|| !(256..=2048).contains(&h)
	{
		return Err(format!(
			"size {w}x{h}: sides must be multiples of 32 between 256 and 2048"
		));
	}
	Ok((w, h))
}

/// Run `job` on the loaded (or now loaded) model. `progress(stage, step,
/// steps)` as `imagegen` reports it (`load ...` stages while loading).
pub(crate) fn run(
	job: Job,
	cancel: Arc<AtomicBool>,
	progress: &mut dyn FnMut(&str, usize, usize),
) -> Result<Done, String> {
	let cfg =
		cfg().ok_or("image generation is not enabled (start the server with --image-model DIR)")?;
	let t0 = Instant::now();
	let waited = Instant::now();
	let _devices = DEVICES.write().unwrap_or_else(|e| e.into_inner());
	if waited.elapsed().as_secs_f64() > 0.5 {
		eprintln!(
			"image: waited {:.1} s for running requests",
			waited.elapsed().as_secs_f64()
		);
	}
	if cancel.load(Ordering::Relaxed) {
		return Err("canceled".into());
	}
	// the chat engine's NPU graphs share the driver's memory with this
	// model's, and a job does not fit beside Flash-Next's (0x70000002)
	if let Some(f) = MAKE_ROOM.get() {
		f();
	}
	// `--progress`: a stderr line per stage and step next to the caller's reports
	let bar = model::progress::Stages::start("image");
	let progress = &mut |stage: &str, i: usize, n: usize| {
		bar.update(stage, imagegen::steps_done(stage, i), n);
		progress(stage, i, n)
	};
	let (m, load_secs) = {
		let mut g = MODEL.lock().unwrap_or_else(|e| e.into_inner());
		match g.as_ref() {
			Some(s) => (s.m.clone(), None),
			None => {
				eprintln!("image: loading {} ({:?})", cfg.dir.display(), cfg.fmt);
				let m = Arc::new(Model::load(&cfg.dir, cfg.fmt, &mut *progress)?);
				let (te, dit) = m.resident_bytes();
				// (the bar may stand on the line: erased first)
				model::progress::line(&format!(
					"image: loaded in {:.1} s, transformer on {} (text encoder {:.2} GB, transformer {:.2} GB)",
					m.load_secs,
					m.backend(),
					te as f64 / 1e9,
					dit as f64 / 1e9
				));
				*g = Some(Slot {
					m: m.clone(),
					last: Instant::now(),
				});
				(m.clone(), Some(m.load_secs))
			}
		}
	};
	let seed = job.seed.unwrap_or_else(random_seed);
	let o = Options {
		prompt: job.prompt,
		width: job.width,
		height: job.height,
		seed,
		steps: job.steps,
		images: job.images,
		cancel: Some(cancel),
		..Options::default()
	};
	let r = m.generate(&o, &mut *progress);
	// (the summary below replaces the bar)
	drop(bar);
	if let Some(s) = MODEL.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
		s.last = Instant::now();
	}
	// (before the devices lock is released: a chat request may follow)
	m.release_devices();
	drop(m);
	let out = match r {
		Ok(o) => o,
		Err(e) => {
			// a device error leaves the model's graphs holding what the next
			// chat request or job may need; loaded again by the next job
			if e != "canceled" {
				unload("after a failed job");
			}
			return Err(e);
		}
	};
	let mut rgba = out.rgba;
	if !job.transparent {
		over_white(&mut rgba);
	}
	let png = imgio::encode_png(out.width, out.height, &rgba, job.transparent)?;
	let secs = t0.elapsed().as_secs_f64();
	eprintln!(
		"image: {}x{} seed {seed} in {secs:.1} s (text encoder {:.1} s, vae encoder {:.1} s, transformer {:.1} s, vae {:.1} s)",
		out.width, out.height, out.secs[0], out.secs[1], out.secs[2], out.secs[3]
	);
	Ok(Done {
		width: out.width,
		height: out.height,
		rgba,
		png,
		seed,
		secs,
		load_secs,
	})
}

/// Composite RGBA over white in place (alpha becomes 255).
fn over_white(rgba: &mut [u8]) {
	for p in rgba.chunks_exact_mut(4) {
		let a = p[3] as u32;
		if a == 255 {
			continue;
		}
		for c in &mut p[..3] {
			*c = ((*c as u32 * a + 255 * (255 - a) + 127) / 255) as u8;
		}
		p[3] = 255;
	}
}

/// A PNG of `d` scaled so its long side is at most `max` (the preview an
/// MCP result carries inline).
pub(crate) fn preview_png(d: &Done, max: usize, alpha: bool) -> Result<Vec<u8>, String> {
	let long = d.width.max(d.height);
	if long <= max {
		return Ok(d.png.clone());
	}
	let (w, h) = (
		(d.width * max / long).max(1),
		(d.height * max / long).max(1),
	);
	let px = imgio::pil::resize_rgba(
		&d.rgba,
		d.width,
		d.height,
		w,
		h,
		imgio::pil::Filter::Lanczos,
	);
	imgio::encode_png(w, h, &px, alpha)
}

pub(crate) fn base64_encode(b: &[u8]) -> String {
	const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	let mut o = String::with_capacity(b.len().div_ceil(3) * 4);
	for c in b.chunks(3) {
		let n = (c[0] as u32) << 16
			| (*c.get(1).unwrap_or(&0) as u32) << 8
			| *c.get(2).unwrap_or(&0) as u32;
		o.push(T[(n >> 18) as usize & 63] as char);
		o.push(T[(n >> 12) as usize & 63] as char);
		o.push(if c.len() > 1 {
			T[(n >> 6) as usize & 63] as char
		} else {
			'='
		});
		o.push(if c.len() > 2 {
			T[n as usize & 63] as char
		} else {
			'='
		});
	}
	o
}

/// An image argument: a `data:` URL (base64) or a file path, decoded.
pub(crate) fn image_arg(s: &str) -> Result<ImageSrc, String> {
	let bytes = if let Some(rest) = s.strip_prefix("data:") {
		let (_, b64) = rest
			.split_once(";base64,")
			.ok_or("image: a data: URL must be base64")?;
		crate::base64_decode(b64)?
	} else if s.starts_with("http://") || s.starts_with("https://") {
		return Err("image: remote URLs are not fetched; pass a path or a data: URL".into());
	} else {
		let p = s.strip_prefix("file://").unwrap_or(s);
		std::fs::read(p).map_err(|e| format!("image {p}: {e}"))?
	};
	image_bytes(&bytes)
}

pub(crate) fn image_bytes(b: &[u8]) -> Result<ImageSrc, String> {
	let (w, h, px) = imgio::decode_rgba(b)?;
	Ok(ImageSrc::Rgba(w, h, px))
}

/// A request's `steps`: 0 = the model's default.
pub(crate) fn steps_of(n: i64) -> Result<usize, String> {
	if (0..=imagegen::MAX_STEPS as i64).contains(&n) {
		Ok(n as usize)
	} else {
		Err(format!("steps {n}: want 0..={}", imagegen::MAX_STEPS))
	}
}

// ---------------------------------------------------------------------------
// OpenAI Images API
// ---------------------------------------------------------------------------

/// The fields both OpenAI routes share; `n` images.
fn openai_job(get: &dyn Fn(&str) -> Option<String>) -> Result<(Job, usize), String> {
	let prompt = get("prompt")
		.filter(|p| !p.trim().is_empty())
		.ok_or("prompt is required")?;
	let n = match get("n") {
		Some(v) => v.parse::<usize>().map_err(|_| format!("n {v:?}"))?,
		None => 1,
	};
	if !(1..=4).contains(&n) {
		return Err("n must be 1..4".into());
	}
	let (width, height) = match get("size").as_deref() {
		None | Some("auto") | Some("") => (0, 0),
		Some(s) => {
			let p: Vec<usize> = s.split(['x', 'X']).filter_map(|v| v.parse().ok()).collect();
			match p[..] {
				[w, h] => check_size(w, h)?,
				_ => return Err(format!("size {s:?}: want WxH or auto")),
			}
		}
	};
	match get("response_format").as_deref() {
		None | Some("b64_json") => {}
		Some(f) => {
			return Err(format!(
				"response_format {f:?}: only b64_json (the server stores nothing)"
			))
		}
	}
	let transparent = match get("background").as_deref() {
		None | Some("auto") | Some("opaque") => false,
		Some("transparent") => true,
		Some(b) => return Err(format!("background {b:?}")),
	};
	let seed = match get("seed") {
		Some(v) => Some(v.parse::<u64>().map_err(|_| format!("seed {v:?}"))?),
		None => None,
	};
	let steps = match get("steps") {
		Some(v) => steps_of(v.parse::<i64>().map_err(|_| format!("steps {v:?}"))?)?,
		None => 0,
	};
	Ok((
		Job {
			prompt,
			width,
			height,
			seed,
			steps,
			images: Vec::new(),
			transparent,
		},
		n,
	))
}

fn json_field(req: &Json, k: &str) -> Option<String> {
	match req.get(k)? {
		Json::Str(s) => Some(s.clone()),
		Json::Num(n) => Some(format!("{}", *n as i64)),
		Json::Bool(b) => Some(b.to_string()),
		_ => None,
	}
}

/// Run `n` jobs (seeds `seed, seed + 1, ...`) and answer with their PNGs.
fn openai_run(w: &mut dyn Write, job: Job, n: usize) -> std::io::Result<()> {
	let seed = job.seed.unwrap_or_else(random_seed);
	let mut data = Vec::new();
	for i in 0..n {
		let j = Job {
			prompt: job.prompt.clone(),
			images: job.images.clone(),
			seed: Some(seed + i as u64),
			..job
		};
		match run(j, Arc::new(AtomicBool::new(false)), &mut |_, _, _| {}) {
			Ok(d) => data.push(format!(
				"{{\"b64_json\":{},\"seed\":{}}}",
				jstr(&base64_encode(&d.png)),
				d.seed
			)),
			Err(e) => return write_error_500(w, &e),
		}
	}
	let created = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map_or(0, |d| d.as_secs());
	respond_json(
		w,
		200,
		"OK",
		&format!("{{\"created\":{created},\"data\":[{}]}}", data.join(",")),
	)
}

fn write_error_500(w: &mut dyn Write, msg: &str) -> std::io::Result<()> {
	let body = format!(
		"{{\"error\":{{\"type\":\"server_error\",\"message\":{}}}}}",
		jstr(msg)
	);
	respond_json(w, 500, "Internal Server Error", &body)
}

/// `POST /v1/images/generations`.
pub(crate) fn handle_generations(w: &mut dyn Write, body: &[u8]) -> std::io::Result<()> {
	if cfg().is_none() {
		return write_error(w, 404, "image generation is not enabled (--image-model)");
	}
	let req = match Json::parse(body) {
		Ok(r) => r,
		Err(e) => return write_error(w, 400, &format!("invalid JSON: {e}")),
	};
	match openai_job(&|k| json_field(&req, k)) {
		Ok((job, n)) => openai_run(w, job, n),
		Err(e) => write_error(w, 400, &e),
	}
}

/// `POST /v1/images/edits`: multipart (`image` / `image[]` files and the
/// fields), or JSON with `images` as `data:` URLs (strings or `{image_url}`).
pub(crate) fn handle_edits(
	w: &mut dyn Write,
	content_type: &str,
	body: &[u8],
) -> std::io::Result<()> {
	if cfg().is_none() {
		return write_error(w, 404, "image generation is not enabled (--image-model)");
	}
	let r = if content_type.starts_with("multipart/form-data") {
		edits_multipart(content_type, body)
	} else {
		edits_json(body)
	};
	match r {
		Ok((job, n)) => openai_run(w, job, n),
		Err(e) => write_error(w, 400, &e),
	}
}

fn edits_json(body: &[u8]) -> Result<(Job, usize), String> {
	let req = Json::parse(body).map_err(|e| format!("invalid JSON: {e}"))?;
	let (mut job, n) = openai_job(&|k| json_field(&req, k))?;
	let list: Vec<&Json> = match (req.get("images"), req.get("image")) {
		(Some(Json::Arr(a)), _) => a.iter().collect(),
		(_, Some(Json::Arr(a))) => a.iter().collect(),
		(_, Some(v)) => vec![v],
		_ => Vec::new(),
	};
	for v in list {
		let s = match v {
			Json::Str(s) => s.as_str(),
			v => v
				.get("image_url")
				.and_then(|u| u.as_str().or_else(|| u.get("url").and_then(Json::as_str)))
				.ok_or("images: want data: URLs")?,
		};
		if !s.starts_with("data:") {
			return Err("images: only data: URLs over HTTP".into());
		}
		job.images.push(image_arg(s)?);
	}
	if job.images.is_empty() || job.images.len() > imagegen::MAX_IMAGES {
		return Err(format!("an edit takes 1..{} images", imagegen::MAX_IMAGES));
	}
	Ok((job, n))
}

/// One part of a multipart body: its name, whether it is a file, its bytes.
pub(crate) fn multipart(
	content_type: &str,
	body: &[u8],
) -> Result<Vec<(String, bool, Vec<u8>)>, String> {
	let b = content_type
		.split(';')
		.filter_map(|p| p.trim().strip_prefix("boundary="))
		.next()
		.ok_or("multipart: no boundary")?
		.trim_matches('"');
	let delim = format!("--{b}").into_bytes();
	let find = |h: &[u8], n: &[u8], from: usize| -> Option<usize> {
		(from..=h.len().checked_sub(n.len())?).find(|&i| &h[i..i + n.len()] == n)
	};
	let mut out = Vec::new();
	let mut pos = find(body, &delim, 0).ok_or("multipart: no first boundary")? + delim.len();
	loop {
		if body[pos..].starts_with(b"--") {
			break;
		}
		let hstart = pos + 2; // CRLF
		let hend = find(body, b"\r\n\r\n", hstart).ok_or("multipart: unterminated headers")?;
		let head = String::from_utf8_lossy(&body[hstart..hend]).to_string();
		let dstart = hend + 4;
		let next = find(body, &delim, dstart).ok_or("multipart: unterminated part")?;
		let dend = next.saturating_sub(2).max(dstart); // CRLF before the delimiter
		let disp = head
			.lines()
			.find(|l| l.to_ascii_lowercase().starts_with("content-disposition"))
			.unwrap_or("");
		let attr = |k: &str| {
			disp.split(';')
				.filter_map(|p| p.trim().strip_prefix(&format!("{k}=")))
				.next()
				.map(|v| v.trim_matches('"').to_string())
		};
		out.push((
			attr("name").unwrap_or_default(),
			attr("filename").is_some(),
			body[dstart..dend].to_vec(),
		));
		pos = next + delim.len();
		if pos >= body.len() {
			break;
		}
	}
	Ok(out)
}

fn edits_multipart(content_type: &str, body: &[u8]) -> Result<(Job, usize), String> {
	let parts = multipart(content_type, body)?;
	let field = |k: &str| {
		parts
			.iter()
			.find(|(n, f, _)| n == k && !f)
			.map(|(_, _, v)| String::from_utf8_lossy(v).trim().to_string())
	};
	let (mut job, n) = openai_job(&field)?;
	for (name, file, v) in &parts {
		if (name == "image" || name == "image[]") && (*file || !v.is_empty()) {
			job.images.push(image_bytes(v)?);
		}
	}
	if job.images.is_empty() || job.images.len() > imagegen::MAX_IMAGES {
		return Err(format!("an edit takes 1..{} images", imagegen::MAX_IMAGES));
	}
	Ok((job, n))
}

// ---------------------------------------------------------------------------
// Write paths (MCP)
// ---------------------------------------------------------------------------

/// `p` made absolute and normalized (`.` / `..` removed), without touching
/// the file system.
pub(crate) fn normalize(p: &Path) -> PathBuf {
	use std::path::Component;
	let mut out = PathBuf::new();
	for c in p.components() {
		match c {
			Component::ParentDir => {
				out.pop();
			}
			Component::CurDir => {}
			c => out.push(c.as_os_str()),
		}
	}
	out
}

fn key(p: &Path) -> String {
	let s = normalize(p).to_string_lossy().replace('\\', "/");
	let s = s.trim_end_matches('/').to_string();
	if cfg!(windows) {
		s.to_lowercase()
	} else {
		s
	}
}

/// True when `p` lies inside `dir` (both absolute).
pub(crate) fn inside(p: &Path, dir: &Path) -> bool {
	let (p, d) = (key(p), key(dir));
	p.len() > d.len() && p.starts_with(&d) && p.as_bytes()[d.len()] == b'/'
}

/// Where to write: `want` (absolute, or relative to the first allowed
/// directory) if it lies inside one of `allowed`; with no `want`, a new
/// name in the first `--image-out-dir`. `None`: no file.
pub(crate) fn out_path(
	want: Option<&str>,
	allowed: &[PathBuf],
	seed: u64,
) -> Result<Option<PathBuf>, String> {
	let Some(w) = want.filter(|w| !w.is_empty()) else {
		let Some(d) = cfg().and_then(|c| c.out_dirs.first()) else {
			return Ok(None);
		};
		let t = std::time::SystemTime::now()
			.duration_since(std::time::UNIX_EPOCH)
			.map_or(0, |d| d.as_secs());
		return Ok(Some(d.join(format!("qwen-image-{t}-{seed}.png"))));
	};
	let p = Path::new(w);
	let p = if p.is_absolute() {
		normalize(p)
	} else {
		let base = allowed
			.first()
			.ok_or("output_path: no MCP roots and no --image-out-dir to resolve it against")?;
		normalize(&base.join(p))
	};
	if !allowed.iter().any(|d| inside(&p, d)) {
		return Err(format!(
			"output_path {}: not inside the client's roots or an --image-out-dir ({})",
			p.display(),
			if allowed.is_empty() {
				"none".to_string()
			} else {
				allowed
					.iter()
					.map(|d| d.display().to_string())
					.collect::<Vec<_>>()
					.join(", ")
			}
		));
	}
	let p = if p.extension().is_none() {
		p.with_extension("png")
	} else {
		p
	};
	Ok(Some(p))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn base64_round_trip() {
		for n in 0..7 {
			let b: Vec<u8> = (0..n as u8).map(|x| x.wrapping_mul(37)).collect();
			assert_eq!(crate::base64_decode(&base64_encode(&b)).unwrap(), b);
		}
		assert_eq!(base64_encode(b"Man"), "TWFu");
		assert_eq!(base64_encode(b"Ma"), "TWE=");
	}

	#[test]
	fn sizes() {
		assert_eq!(aspect_size("1:1").unwrap(), (1024, 1024));
		let (w, h) = aspect_size("16:9").unwrap();
		assert!(w % 32 == 0 && h % 32 == 0 && w > h);
		assert!(aspect_size("9:0").is_err());
		assert!(check_size(1000, 1024).is_err());
		assert!(check_size(1024, 768).is_ok());
	}

	#[test]
	fn write_paths() {
		let root = if cfg!(windows) {
			PathBuf::from("C:\\proj")
		} else {
			PathBuf::from("/proj")
		};
		let roots = vec![root.clone()];
		let p = out_path(Some("img/a.png"), &roots, 1).unwrap().unwrap();
		assert!(inside(&p, &root));
		assert!(out_path(Some("../x.png"), &roots, 1).is_err());
		let other = if cfg!(windows) {
			"C:\\projx\\a.png"
		} else {
			"/projx/a.png"
		};
		assert!(out_path(Some(other), &roots, 1).is_err());
		let q = out_path(Some("b"), &roots, 1).unwrap().unwrap();
		assert_eq!(q.extension().unwrap(), "png");
	}

	#[test]
	fn multipart_parts() {
		let ct = "multipart/form-data; boundary=XyZ";
		let body = b"--XyZ\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nmake it blue\r\n--XyZ\r\nContent-Disposition: form-data; name=\"image[]\"; filename=\"a.png\"\r\nContent-Type: image/png\r\n\r\n\x89PNG\r\n\x00\r\n--XyZ--\r\n";
		let p = multipart(ct, body).unwrap();
		assert_eq!(p.len(), 2);
		assert_eq!(p[0].0, "prompt");
		assert_eq!(p[0].2, b"make it blue");
		assert!(p[1].1);
		assert_eq!(p[1].2, b"\x89PNG\r\n\x00");
	}
}
