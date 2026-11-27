//! Everything the NPU engine (`cli MODEL.npu`) loads, built from the BF16
//! checkpoint (`cli BF16_DIR --npu-build OUT`): see [`build`].
//!
//! The steps, each skipped when its output exists:
//!
//! | step | output | needs |
//! |---|---|---|
//! | `tokenizer` | `OUT/tokenizer/` | |
//! | `emb` | `OUT/emb_table.f16` | |
//! | `weights` | `OUT/cal/L{L}.cal`, `OUT/w/lean{L}.*.in` | CPU (+ Vulkan) |
//! | `graphs` | `OUT/graphs/**.xml` | |
//! | `compile` | `OUT/graphs/**.blob` | the NPU compiler, the NPU driver |
//! | `sets` | `OUT/sets/*`, `OUT/*.npu` | |
//!
//! The graphs include the vision tower's ([`vis`]: `graphs/vis`, `sets/vis`)
//! when the checkpoint has one.

pub mod dec;
pub mod flash;
pub mod g;
pub mod gptq;
#[cfg(feature = "vulkan")]
pub mod gpufac;
pub mod h3img;
pub mod h3te;
pub mod la;
pub mod lean;
pub mod qimage;
pub mod qvae;
pub mod refm;
pub mod sets;
pub mod st;
pub mod vis;
pub mod weights;

use std::path::{Path, PathBuf};

/// The 27B's calibration text without `--npu-calib`: a stretch of source code
/// and one of prose. The texts under `calib/` are frozen copies, never edited
/// with the code (see `calib/README.md`): a changed byte changes the
/// calibration, and with it every weight the build derives from it.
pub fn calib() -> &'static str {
	quant::packed_str!("base.txt")
}

/// Flash-Next's calibration texts without `--npu-calib`: [`calib`], agentic
/// server code and public-domain poetry (English, German, French).
pub fn flash_calib() -> [&'static str; 3] {
	[
		calib(),
		quant::packed_str!("code.txt"),
		quant::packed_str!("poetry.txt"),
	]
}

pub struct Options {
	/// The BF16 checkpoint directory.
	pub src: PathBuf,
	pub out: PathBuf,
	/// Calibration text files (empty: the build's built-in texts, [`calib`]
	/// or [`flash_calib`]).
	pub calib: Vec<PathBuf>,
	/// Calibration chunks of [`weights::CHUNK`] tokens at most.
	pub max_chunks: usize,
	/// Steps to run (empty: all).
	pub steps: Vec<String>,
	/// The directory of our build of the NPU compiler (else `NPUC_BIN`).
	pub compiler: Option<PathBuf>,
	/// Compiles at once.
	pub jobs: usize,
	/// Use a Vulkan device for GPTQ's largest factorization (a `vulkan`
	/// build; else the CPU does everything).
	pub gpu: bool,
}

impl Options {
	pub fn new(src: &str, out: &str) -> Options {
		Options {
			src: src.into(),
			out: out.into(),
			calib: Vec::new(),
			max_chunks: 16,
			steps: Vec::new(),
			compiler: None,
			jobs: 6,
			gpu: true,
		}
	}

	pub(crate) fn wants(&self, step: &str) -> bool {
		self.steps.is_empty() || self.steps.iter().any(|s| s == step)
	}
}

/// True when `path` is the BF16 checkpoint directory.
pub fn is_checkpoint(path: &str) -> bool {
	st::Checkpoint::is_checkpoint(Path::new(path))
		|| qimage::is_qwen_image(Path::new(path))
		|| h3img::is_h3(Path::new(path))
}

pub(crate) fn say(m: &str) {
	eprintln!("npu-build: {m}");
}

pub(crate) fn mkdir(d: &Path) -> Result<(), String> {
	std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))
}

/// `OUT/tokenizer`: the checkpoint's `tokenizer.json` and
/// `tokenizer_config.json` (what `tokenizer = DIR` of a `.npu` file reads).
pub(crate) fn step_tokenizer(o: &Options) -> Result<tokenizer::Tokenizer, String> {
	let d = o.out.join("tokenizer");
	mkdir(&d)?;
	for f in ["tokenizer.json", "tokenizer_config.json"] {
		let (a, b) = (o.src.join(f), d.join(f));
		if !b.exists() {
			std::fs::copy(&a, &b)
				.map_err(|e| format!("{} -> {}: {e}", a.display(), b.display()))?;
		}
	}
	tokenizer::Tokenizer::from_hf_json(&d.join("tokenizer.json"))
		.map_err(|e| format!("tokenizer: {e}"))
}

/// `OUT/emb_table.f16`: `[vocab][width]` f16, row `i` the embedding of token `i`.
pub(crate) fn step_emb(o: &Options, ck: &st::Checkpoint) -> Result<(), String> {
	let p = o.out.join("emb_table.f16");
	let name = "model.language_model.embed_tokens.weight";
	let sh = ck.shape(name)?;
	let (v, w) = (sh[0], sh[1]);
	if p.metadata()
		.map(|m| m.len() == (v * w * 2) as u64)
		.unwrap_or(false)
	{
		return Ok(());
	}
	say("embedding table");
	use std::io::Write;
	let tmp = p.with_extension("tmp");
	let mut f = std::io::BufWriter::new(
		std::fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?,
	);
	for r0 in (0..v).step_by(4096) {
		let rows = ck.rows(name, r0, (r0 + 4096).min(v))?;
		let b: Vec<u8> = rows
			.iter()
			.flat_map(|&x| g::f16_bits(x as f64).to_le_bytes())
			.collect();
		f.write_all(&b).map_err(|e| e.to_string())?;
	}
	f.flush().map_err(|e| e.to_string())?;
	drop(f);
	std::fs::rename(&tmp, &p).map_err(|e| format!("{}: {e}", p.display()))
}

/// The calibration chunks: the texts (`o.calib`, else `builtin`) tokenized,
/// cut into [`weights::CHUNK`]-token pieces and taken from each text in turn,
/// so every text contributes whatever `o.max_chunks` is.
pub(crate) fn calib_chunks(
	o: &Options,
	tok: &tokenizer::Tokenizer,
	builtin: &[&str],
) -> Result<Vec<Vec<u32>>, String> {
	let mut texts = Vec::new();
	for f in &o.calib {
		texts.push(std::fs::read_to_string(f).map_err(|e| format!("{}: {e}", f.display()))?);
	}
	if texts.is_empty() {
		texts.extend(builtin.iter().map(|t| t.to_string()));
	}
	let per: Vec<Vec<Vec<u32>>> = texts
		.iter()
		.map(|t| {
			tok.encode(t)
				.chunks_exact(weights::CHUNK)
				.map(|c| c.to_vec())
				.collect()
		})
		.collect();
	let mut chunks = Vec::new();
	for i in 0..per.iter().map(Vec::len).max().unwrap_or(0) {
		chunks.extend(per.iter().filter_map(|p| p.get(i).cloned()));
	}
	chunks.truncate(o.max_chunks);
	if chunks.len() < 2 {
		return Err(format!(
			"calibration: {} chunks of {} tokens; give more text (--npu-calib FILE)",
			chunks.len(),
			weights::CHUNK
		));
	}
	Ok(chunks)
}

/// Build what `o` asks for. Progress goes to stderr.
pub fn build(o: &Options) -> Result<(), String> {
	if qimage::is_qwen_image(&o.src) {
		return qimage::build(o, None);
	}
	if h3img::is_h3(&o.src) {
		return h3img::build(o, None, None);
	}
	let ck = st::Checkpoint::open(&o.src)?;
	mkdir(&o.out)?;
	if flash::is_flash(&ck) {
		return flash::build(o, &ck);
	}
	let tok = step_tokenizer(o)?;
	if o.wants("emb") {
		step_emb(o, &ck)?;
	}
	if o.wants("weights") {
		let chunks = calib_chunks(o, &tok, &[calib()])?;
		say(&format!(
			"calibration on {} chunks of {} tokens",
			chunks.len(),
			weights::CHUNK
		));
		let (cal, w) = (o.out.join("cal"), o.out.join("w"));
		let done = |f: &str| cal.join(f).exists();
		if !(done("mtp.cal") && done(&format!("L{}.cal", refm::NL - 1))) {
			let (accel, what) = weights::Accel::new(o.gpu);
			say(&format!("GPTQ on: {what}"));
			weights::calibrate(&ck, &chunks, &cal, &w, &accel, &mut |m| say(m))?;
			weights::calibrate_mtp(&ck, &chunks, &cal, &w, &accel, &mut |m| say(m))?;
		}
		weights::quantize_head(&ck, &cal, &w, &mut |m| say(m))?;
	}
	if o.wants("graphs") {
		let n = sets::graphs(&ck, &o.out)?;
		say(&format!("{n} graphs in {}", o.out.join("graphs").display()));
	}
	if o.wants("compile") {
		let tools = sets::Tools::find(o.compiler.as_deref(), o.jobs);
		// (without the compiler only the graphs that are compiled already pass)
		sets::compile(&o.out, tools.as_ref().ok(), &|m| say(m)).map_err(|e| match &tools {
			Err(t) if e.contains("NPU compiler") => t.clone(),
			_ => e,
		})?;
	}
	if o.wants("sets") {
		sets::assemble(&ck, &o.out, &|m| say(m))?;
		say(&format!("done: {}", o.out.join("model.npu").display()));
	}
	Ok(())
}

/// KL(p || q) and whether the arg max agrees, of two logit rows.
pub fn kl_row(p: &[f32], q: &[f32]) -> (f64, bool) {
	let lsm = |v: &[f32]| -> Vec<f64> {
		let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
		let z = v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln() + m;
		v.iter().map(|&x| x as f64 - z).collect()
	};
	let (a, b) = (lsm(p), lsm(q));
	let kl = a.iter().zip(&b).map(|(x, y)| x.exp() * (x - y)).sum();
	let am = |v: &[f32]| {
		v.iter()
			.enumerate()
			.max_by(|x, y| x.1.total_cmp(y.1))
			.map(|x| x.0)
	};
	(kl, am(p) == am(q))
}

#[cfg(test)]
mod tests;
