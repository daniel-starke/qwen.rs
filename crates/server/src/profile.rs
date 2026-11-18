//! Generation profiles: the sampling and decode settings a request gets
//! when it does not name its own, per thinking mode and context depth.
//!
//! A profile is a text file (`cli MODEL --profile FILE`, `QWEN_PROFILE`, or
//! `<model>.profile` beside the model file) of `key = value` lines in
//! sections; `#` starts a comment:
//!
//! ```text
//! [load]              QWEN_* variables set before the model loads
//! [all]               both modes, from the first prompt token
//! [think]             thinking mode, from the first prompt token
//! [nothink]           thinking off
//! [@32768]            both modes, from a 32768-token prompt on
//! [think@131072]      thinking mode, from 131072 tokens on
//! ```
//!
//! A mode section holds the sampling defaults (`temperature`, `top_p`,
//! `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`,
//! `repetition_penalty`) and any `QWEN_*` decode knob read through
//! `quant::knob` (draft counts, cut-offs, context-draft match lengths,
//! Flash-Next's MTP cost model; `crates/cli/README.md`). For a request the sections
//! that apply are layered by depth, a mode's section over `[@N]` at the same
//! depth; values the request itself sets (`temperature`, ...) win over all
//! of them, and a `QWEN_*` variable set in the shell wins over the file.
//!
//! `cli MODEL --tune-profile OUT` measures the decode knobs on this machine
//! and writes such a file (`tune.rs`).
//!
//! **Built-in defaults** (no file) are the model cards' sampling sets, which
//! are the same for Qwen3.8-27B and Qwen3.8-Flash-Next: thinking
//! temperature 1.0 / top-p 0.95 / top-k 20, no penalties; non-thinking
//! 0.7 / 0.8 / 20 with `presence_penalty` 1.5. The penalty applies to the
//! generated tokens only, and was checked against what agentic coding has
//! to repeat (identifiers, call markup): harmless on both models
//! (`SampleParams::instruct`). A request for temperature 0 keeps only the
//! penalties it sets itself ([`greedy_penalties`]).

use sampler::SampleParams;
use std::path::{Path, PathBuf};

/// The sampling keys a mode section may set, in file order.
pub const SAMPLING_KEYS: [&str; 7] = [
	"temperature",
	"top_p",
	"top_k",
	"min_p",
	"presence_penalty",
	"frequency_penalty",
	"repetition_penalty",
];

/// The decode knobs read per request through `quant::knob` (a profile
/// may set others in `[load]` only).
pub const KNOWN: [&str; 22] = [
	"QWEN_MTP_N",
	"QWEN_MTP_PMIN",
	"QWEN_DFLASH_N",
	"QWEN_DFLASH_PMIN",
	"QWEN_SUFFIX_MIN",
	"QWEN_SUFFIX_ENGINE_MIN",
	"QWEN_SUFFIX_PLAIN_MIN",
	"QWEN_SUFFIX_EV",
	"QWEN_NPU_MTP_CTX",
	"QWEN_FLASH_MTP_N",
	"QWEN_FLASH_MTP_PMIN",
	"QWEN_FLASH_MTP_PSCALE",
	"QWEN_FLASH_MTP_COST",
	"QWEN_FLASH_MTP_AUTO",
	"QWEN_FLASH_MTP_CAL",
	"QWEN_FLASH_MTP_EARLY",
	"QWEN_FLASH_MTP_CTX",
	"QWEN_FLASH_COUPLED",
	"QWEN_FLASH_COUPLED_TAU",
	"QWEN_FLASH_WAVES",
	"QWEN_FLASH_WAVES_MIN",
	"QWEN_FLASH_WAVE_PIPE",
];

#[derive(Clone, Debug, PartialEq)]
struct Section {
	/// `Some(true)` thinking, `Some(false)` not, `None` both.
	think: Option<bool>,
	/// First prompt length (tokens) it applies to.
	from: usize,
	vals: Vec<(String, String)>,
}

/// A parsed profile (see the module docs).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Profile {
	/// `[load]`.
	pub load: Vec<(String, String)>,
	sections: Vec<Section>,
	/// Where it was read from, for the startup line.
	pub source: Option<PathBuf>,
}

/// What a request gets from the profile.
#[derive(Clone, Debug, PartialEq)]
pub struct Resolved {
	pub params: SampleParams,
	/// `QWEN_*` values for the request's `quant::knob::Scope`.
	pub knobs: Vec<(String, String)>,
}

/// The built-in sampling of a mode (the model cards').
pub fn builtin_params(think: bool) -> SampleParams {
	if think {
		SampleParams::thinking()
	} else {
		SampleParams::instruct()
	}
}

/// `params` with the sampling key `k` set to `v`.
pub fn set_param(params: &mut SampleParams, k: &str, v: &str) -> Result<(), String> {
	let f = || {
		v.trim()
			.parse::<f32>()
			.map_err(|e| format!("{k} = {v}: {e}"))
	};
	match k {
		"temperature" => params.temperature = f()?,
		"top_p" => params.top_p = f()?,
		"top_k" => params.top_k = f()?.max(0.0) as usize,
		"min_p" => params.min_p = f()?,
		"presence_penalty" => params.presence_penalty = f()?,
		"frequency_penalty" => params.frequency_penalty = f()?,
		"repetition_penalty" => params.repeat_penalty = f()?,
		_ => return Err(format!("unknown sampling key {k}")),
	}
	Ok(())
}

/// A greedy request (temperature 0, asked for) is a request for the
/// model's argmax: it keeps only the penalties it set itself (`asked`),
/// none of the profile's.
pub fn greedy_penalties(params: &mut SampleParams, asked: &dyn Fn(&str) -> bool) {
	if !asked("presence_penalty") {
		params.presence_penalty = 0.0;
	}
	if !asked("frequency_penalty") {
		params.frequency_penalty = 0.0;
	}
	if !asked("repetition_penalty") {
		params.repeat_penalty = 1.0;
	}
}

impl Profile {
	pub fn parse(text: &str) -> Result<Profile, String> {
		let mut p = Profile::default();
		// `None` = `[load]`
		let mut cur: Option<Section> = Some(Section {
			think: None,
			from: 0,
			vals: Vec::new(),
		});
		let mut in_load = false;
		let flush = |p: &mut Profile, s: Option<Section>| {
			if let Some(s) = s.filter(|s| !s.vals.is_empty()) {
				p.sections.push(s);
			}
		};
		for (no, line) in text.lines().enumerate() {
			let line = line.split('#').next().unwrap_or("").trim();
			if line.is_empty() {
				continue;
			}
			let err = |m: String| format!("line {}: {m}", no + 1);
			if let Some(h) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
				flush(&mut p, cur.take());
				let h = h.trim();
				in_load = h == "load";
				if in_load {
					continue;
				}
				let (mode, from) = match h.split_once('@') {
					Some((m, f)) => (
						m.trim(),
						f.trim()
							.parse::<usize>()
							.map_err(|e| err(format!("[{h}]: {e}")))?,
					),
					None => (h, 0),
				};
				let think = match mode {
					"think" => Some(true),
					"nothink" => Some(false),
					"" | "all" => None,
					_ => return Err(err(format!("unknown section [{h}]"))),
				};
				cur = Some(Section {
					think,
					from,
					vals: Vec::new(),
				});
				continue;
			}
			let Some((k, v)) = line.split_once('=') else {
				return Err(err(format!("expected key = value, got {line:?}")));
			};
			let (k, v) = (k.trim().to_string(), v.trim().to_string());
			if in_load {
				if !k.starts_with("QWEN_") {
					return Err(err(format!("[load] takes QWEN_* variables, got {k}")));
				}
				p.load.push((k, v));
				continue;
			}
			if !k.starts_with("QWEN_") {
				set_param(&mut SampleParams::default(), &k, &v).map_err(err)?;
			} else if !KNOWN.contains(&k.as_str()) {
				return Err(err(format!(
					"{k} is not read per request (put load-time variables in [load])"
				)));
			}
			cur.as_mut().unwrap().vals.push((k, v));
		}
		flush(&mut p, cur);
		Ok(p)
	}

	pub fn read(path: &Path) -> Result<Profile, String> {
		let text = std::fs::read_to_string(path)
			.map_err(|e| format!("profile {}: {e}", path.display()))?;
		let mut p =
			Profile::parse(&text).map_err(|e| format!("profile {}: {e}", path.display()))?;
		p.source = Some(path.to_path_buf());
		Ok(p)
	}

	/// The profile file beside `model` (`X.gguf` / `X.npu` -> `X.profile`;
	/// an OpenVINO export directory -> `DIR/qwen-rs.profile`), if it exists.
	pub fn beside(model: &str) -> Option<PathBuf> {
		let m = Path::new(model);
		let p = if m.is_dir() {
			m.join("qwen-rs.profile")
		} else {
			m.with_extension("profile")
		};
		p.is_file().then_some(p)
	}

	/// The settings for a request in mode `think` whose prompt is `depth`
	/// tokens: the built-in sampling, then every section that applies, by
	/// depth, a mode's over `[@N]` of the same depth.
	pub fn resolve(&self, think: bool, depth: usize) -> Resolved {
		let mut params = builtin_params(think);
		let mut knobs: Vec<(String, String)> = Vec::new();
		let mut secs: Vec<&Section> = self
			.sections
			.iter()
			.filter(|s| s.from <= depth && s.think.is_none_or(|t| t == think))
			.collect();
		secs.sort_by_key(|s| (s.from, s.think.is_some()));
		for s in secs {
			for (k, v) in &s.vals {
				if k.starts_with("QWEN_") {
					knobs.retain(|(x, _)| x != k);
					knobs.push((k.clone(), v.clone()));
				} else {
					// checked by `parse`
					let _ = set_param(&mut params, k, v);
				}
			}
		}
		Resolved { params, knobs }
	}

	/// Set the `[load]` variables the user has not set (`quant::knob`'s
	/// [`freeze_user_env`](quant::knob::freeze_user_env) must have run).
	pub fn apply_load(&self) {
		for (k, v) in &self.load {
			if !quant::knob::user_set(k) {
				std::env::set_var(k, v);
			}
		}
	}

	/// This profile without its decode knobs (what `--tune-profile` runs
	/// its trials under: the trial's own values must be the only ones).
	pub fn sampling_only(&self) -> Profile {
		let mut p = self.clone();
		for s in &mut p.sections {
			s.vals.retain(|(k, _)| !k.starts_with("QWEN_"));
		}
		p.sections.retain(|s| !s.vals.is_empty());
		p
	}

	/// The largest value any section gives the knob `name` (the `--mtp`
	/// capacity a profile's `QWEN_MTP_N` needs).
	pub fn max_knob(&self, name: &str) -> Option<usize> {
		self.sections
			.iter()
			.flat_map(|s| &s.vals)
			.filter(|(k, _)| k == name)
			.filter_map(|(_, v)| v.parse().ok())
			.max()
	}

	/// Append a section (for the tuner's writer).
	pub fn push(&mut self, think: Option<bool>, from: usize, vals: Vec<(String, String)>) {
		self.sections.push(Section { think, from, vals });
	}

	/// The file form of this profile, `header` lines as comments first.
	pub fn to_text(&self, header: &[String]) -> String {
		let mut s = String::new();
		for h in header {
			s.push_str(&format!("# {h}\n").replace("# \n", "#\n"));
		}
		if !self.load.is_empty() {
			s.push_str("\n[load]\n");
			for (k, v) in &self.load {
				s.push_str(&format!("{k} = {v}\n"));
			}
		}
		for sec in &self.sections {
			let mode = match sec.think {
				Some(true) => "think",
				Some(false) => "nothink",
				None if sec.from == 0 => "all",
				None => "",
			};
			if sec.from > 0 {
				s.push_str(&format!("\n[{mode}@{}]\n", sec.from));
			} else {
				s.push_str(&format!("\n[{mode}]\n"));
			}
			for (k, v) in &sec.vals {
				s.push_str(&format!("{k} = {v}\n"));
			}
		}
		s
	}

	/// One line for the startup log.
	pub fn describe(&self) -> String {
		let src = match &self.source {
			Some(p) => p.display().to_string(),
			None => "built-in".into(),
		};
		let line = |t: bool| {
			let r = self.resolve(t, 0);
			let p = r.params;
			format!(
				"{} temp {} top-p {} top-k {}{}{}",
				if t { "think" } else { "nothink" },
				p.temperature,
				p.top_p,
				p.top_k,
				if p.presence_penalty != 0.0 {
					format!(" presence {}", p.presence_penalty)
				} else {
					String::new()
				},
				if r.knobs.is_empty() {
					String::new()
				} else {
					format!(" + {} knobs", r.knobs.len())
				}
			)
		};
		format!("profile {src}: {}; {}", line(true), line(false))
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	const TEXT: &str = "# test\n\
		[load]\n\
		QWEN_FLASH_HEAD_SPLIT = 12\n\
		[think]\n\
		temperature = 0.9\n\
		QWEN_FLASH_MTP_N = 4\n\
		[nothink]\n\
		presence_penalty = 1.5 # the card's\n\
		[@32768]\n\
		QWEN_FLASH_MTP_N = 5\n\
		QWEN_SUFFIX_ENGINE_MIN = 9\n\
		[think@32768]\n\
		QWEN_FLASH_MTP_N = 6\n";

	#[test]
	fn layers_by_mode_and_depth() {
		let p = Profile::parse(TEXT).unwrap();
		assert_eq!(p.load, vec![("QWEN_FLASH_HEAD_SPLIT".into(), "12".into())]);
		let knob =
			|r: &Resolved, k: &str| r.knobs.iter().find(|(x, _)| x == k).map(|(_, v)| v.clone());
		let t0 = p.resolve(true, 100);
		assert_eq!(t0.params.temperature, 0.9);
		assert_eq!(t0.params.top_p, 0.95);
		assert_eq!(knob(&t0, "QWEN_FLASH_MTP_N").as_deref(), Some("4"));
		let n0 = p.resolve(false, 100);
		assert_eq!(
			n0.params,
			SampleParams {
				presence_penalty: 1.5,
				..SampleParams::instruct()
			}
		);
		assert!(n0.knobs.is_empty());
		// a mode's section beats [@N] at the same depth, both beat shallower
		let t1 = p.resolve(true, 40000);
		assert_eq!(knob(&t1, "QWEN_FLASH_MTP_N").as_deref(), Some("6"));
		assert_eq!(knob(&t1, "QWEN_SUFFIX_ENGINE_MIN").as_deref(), Some("9"));
		let n1 = p.resolve(false, 40000);
		assert_eq!(knob(&n1, "QWEN_FLASH_MTP_N").as_deref(), Some("5"));
	}

	/// The shipped profiles parse and set only known knobs.
	#[test]
	fn shipped_profiles_parse() {
		let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../profiles");
		let mut n = 0;
		for e in std::fs::read_dir(&dir).unwrap() {
			let p = e.unwrap().path();
			if p.extension().is_some_and(|x| x == "profile") {
				let prof = Profile::read(&p).unwrap();
				for t in [false, true] {
					for (k, _) in prof.resolve(t, 1 << 20).knobs {
						assert!(KNOWN.contains(&k.as_str()), "{}: {k}", p.display());
					}
				}
				n += 1;
			}
		}
		assert!(n > 0);
	}

	#[test]
	fn round_trips_and_rejects_typos() {
		let p = Profile::parse(TEXT).unwrap();
		let q = Profile::parse(&p.to_text(&["header".into()])).unwrap();
		assert_eq!(p, q);
		assert!(Profile::parse("[think]\ntemprature = 1\n").is_err());
		assert!(Profile::parse("[thinking]\n").is_err());
		assert!(Profile::parse("[load]\ntemperature = 1\n").is_err());
		assert_eq!(
			Profile::default().resolve(false, 0).params,
			SampleParams::instruct()
		);
	}
}
