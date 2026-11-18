//! Work-split calibration that follows the kernels instead of being retuned
//! by hand: a split's cost model keeps its *shape* in code (which work costs
//! what relative to what) and learns one *scale factor* per device from the
//! run's own timings, `sum(measured) / sum(estimated)` over many samples.
//!
//! **Learn, then freeze.** Until an entry has [`Fit::need`] samples the
//! split uses the code's defaults (the behavior without this module); then
//! the factors are fitted once, printed, stored and **frozen**: the split no
//! longer moves with timing noise, so output stays reproducible from there
//! on. Unfinished sums are stored too, so the samples of short runs add up.
//!
//! **Keyed by what makes the numbers stale.** An entry belongs to a
//! fingerprint: this executable (size + modification time: every rebuild,
//! i.e. every kernel change, starts a new fit) plus what the caller adds
//! (devices, the NPU graph blobs). A stale entry is never applied.
//!
//! **Drift check.** Frozen entries keep measuring; a window of `need`
//! samples whose factor is more than 15 % off the frozen one prints a
//! warning once (something outside the fingerprint changed: driver, power
//! plan, another process). `QWEN_CALIB=reset` refits.
//!
//! `QWEN_CALIB=0`: code defaults only, nothing measured or stored.
//! `QWEN_CALIB_DIR`: where the entries live (default
//! `%LOCALAPPDATA%\qwen-rs\calib`, `$XDG_CACHE_HOME/qwen-rs/calib` or
//! `~/.cache/qwen-rs/calib`), one `NAME.txt` per fit, a line per
//! fingerprint: `FP N FROZEN M0 E0 M1 E1 ...`.

use std::path::PathBuf;
use std::sync::OnceLock;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
	Off,
	On,
	Reset,
}

fn mode() -> Mode {
	static M: OnceLock<Mode> = OnceLock::new();
	*M.get_or_init(
		|| match std::env::var("QWEN_CALIB").as_deref().map(str::trim) {
			Ok("0") | Ok("off") => Mode::Off,
			Ok("reset") => Mode::Reset,
			_ => Mode::On,
		},
	)
}

fn dir() -> Option<PathBuf> {
	if let Some(d) = std::env::var_os("QWEN_CALIB_DIR") {
		return Some(PathBuf::from(d));
	}
	let base = std::env::var_os("LOCALAPPDATA")
		.or_else(|| std::env::var_os("XDG_CACHE_HOME"))
		.map(PathBuf::from)
		.or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
	Some(base.join("qwen-rs").join("calib"))
}

/// FNV-1a, 64 bit.
fn fnv(h: &mut u64, bytes: &[u8]) {
	for &b in bytes {
		*h ^= b as u64;
		*h = h.wrapping_mul(0x100_0000_01b3);
	}
}

/// `(size, modification time)` of a file, as text ("-" when unreadable).
pub fn file_id(p: &std::path::Path) -> String {
	std::fs::metadata(p)
		.ok()
		.map(|m| {
			let t = m
				.modified()
				.ok()
				.and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
				.map_or(0, |d| d.as_nanos());
			format!("{}:{t}", m.len())
		})
		.unwrap_or_else(|| "-".into())
}

/// The fingerprint of this executable plus `parts`.
pub fn fingerprint(parts: &[&str]) -> u64 {
	static EXE: OnceLock<String> = OnceLock::new();
	let exe = EXE.get_or_init(|| {
		std::env::current_exe()
			.map(|p| file_id(&p))
			.unwrap_or_default()
	});
	let mut h = 0xcbf2_9ce4_8422_2325u64;
	fnv(&mut h, exe.as_bytes());
	for p in parts {
		fnv(&mut h, &[0]);
		fnv(&mut h, p.as_bytes());
	}
	h
}

/// One learned set of scale factors (see the module docs).
pub struct Fit {
	name: &'static str,
	fp: u64,
	dflt: Vec<f64>,
	/// per factor (measured, estimated) since the entry was started
	sums: Vec<(f64, f64)>,
	n: u64,
	need: u64,
	/// the factors in use once frozen
	frozen: Option<Vec<f64>>,
	/// after freezing: the current drift window
	win: Vec<(f64, f64)>,
	wn: u64,
	warned: bool,
}

impl Fit {
	/// The fit `name` under fingerprint `fp`, with `dflt` (the code's
	/// factors) until `need` samples were taken.
	pub fn open(name: &'static str, fp: u64, dflt: &[f64], need: u64) -> Fit {
		let mut f = Fit {
			name,
			fp,
			dflt: dflt.to_vec(),
			sums: vec![(0.0, 0.0); dflt.len()],
			n: 0,
			need: need.max(1),
			frozen: None,
			win: vec![(0.0, 0.0); dflt.len()],
			wn: 0,
			warned: false,
		};
		if mode() == Mode::On {
			if let Some((n, frozen, sums)) = f.read() {
				f.n = n;
				f.sums = sums;
				if frozen {
					f.frozen = Some(f.ratios());
					eprintln!(
						"calib: {name}: frozen factors {} ({n} samples)",
						fmt(f.frozen.as_ref().unwrap())
					);
				}
			}
		}
		f
	}

	/// Whether samples are taken at all (`QWEN_CALIB` not `0`).
	pub fn active(&self) -> bool {
		mode() != Mode::Off
	}

	/// Factor `i` in use: the frozen fit, else the code's default.
	pub fn get(&self, i: usize) -> f64 {
		self.frozen.as_ref().map_or(self.dflt[i], |v| v[i])
	}

	/// One sample: per factor `(measured, estimated)` (estimated with the
	/// code's cost shape and a factor of 1), the same units.
	pub fn add(&mut self, sample: &[(f64, f64)]) {
		if !self.active() || sample.len() != self.dflt.len() {
			return;
		}
		if sample
			.iter()
			.any(|&(m, e)| !(m.is_finite() && e.is_finite() && m > 0.0 && e > 0.0))
		{
			return;
		}
		if let Some(fz) = &self.frozen {
			for (w, &(m, e)) in self.win.iter_mut().zip(sample) {
				w.0 += m;
				w.1 += e;
			}
			self.wn += 1;
			if self.wn >= self.need {
				if !self.warned {
					for (i, (w, &z)) in self.win.iter().zip(fz).enumerate() {
						let r = w.0 / w.1;
						if (r / z - 1.0).abs() > 0.15 {
							eprintln!(
								"calib: {}: factor {i} measures {r:.3}, frozen {z:.3} ({:+.0} %): \
								 set QWEN_CALIB=reset once to refit",
								self.name,
								100.0 * (r / z - 1.0)
							);
							self.warned = true;
						}
					}
				}
				self.win.iter_mut().for_each(|w| *w = (0.0, 0.0));
				self.wn = 0;
			}
			return;
		}
		for (s, &(m, e)) in self.sums.iter_mut().zip(sample) {
			s.0 += m;
			s.1 += e;
		}
		self.n += 1;
		if self.n >= self.need {
			let r = self.ratios();
			eprintln!(
				"calib: {}: fitted factors {} (defaults {}, {} samples), frozen",
				self.name,
				fmt(&r),
				fmt(&self.dflt),
				self.n
			);
			self.frozen = Some(r);
			self.write();
		} else if self.n.is_multiple_of((self.need / 4).max(1)) {
			// progress survives a short run
			self.write();
		}
	}

	/// Store the unfinished sums now (e.g. when the engine is dropped).
	pub fn flush(&self) {
		if self.active() && self.frozen.is_none() && self.n > 0 {
			self.write();
		}
	}

	fn ratios(&self) -> Vec<f64> {
		self.sums
			.iter()
			.zip(&self.dflt)
			.map(|(&(m, e), &d)| if e > 0.0 { m / e } else { d })
			.collect()
	}

	fn path(&self) -> Option<PathBuf> {
		dir().map(|d| d.join(format!("{}.txt", self.name)))
	}

	fn read(&self) -> Option<(u64, bool, Vec<(f64, f64)>)> {
		let s = std::fs::read_to_string(self.path()?).ok()?;
		let fp = format!("{:016x}", self.fp);
		let line = s
			.lines()
			.find(|l| l.split_whitespace().next() == Some(&fp))?;
		let v: Vec<&str> = line.split_whitespace().collect();
		let n: u64 = v.get(1)?.parse().ok()?;
		let frozen = *v.get(2)? == "1";
		let nums: Vec<f64> = v[3..].iter().filter_map(|x| x.parse().ok()).collect();
		if nums.len() != 2 * self.dflt.len() {
			return None;
		}
		Some((n, frozen, nums.chunks(2).map(|c| (c[0], c[1])).collect()))
	}

	/// Replace this fingerprint's line, keep the 15 most recent others.
	fn write(&self) {
		let Some(p) = self.path() else { return };
		let fp = format!("{:016x}", self.fp);
		let old = std::fs::read_to_string(&p).unwrap_or_default();
		let mut lines: Vec<String> = old
			.lines()
			.filter(|l| !l.trim().is_empty() && l.split_whitespace().next() != Some(&fp))
			.map(str::to_string)
			.collect();
		let keep = lines.len().saturating_sub(15);
		lines.drain(..keep);
		let mut l = format!(
			"{fp} {} {}",
			self.n,
			if self.frozen.is_some() { 1 } else { 0 }
		);
		for &(m, e) in &self.sums {
			l += &format!(" {m:e} {e:e}");
		}
		lines.push(l);
		let _ = std::fs::create_dir_all(p.parent().unwrap());
		let tmp = p.with_extension(format!("tmp{}", std::process::id()));
		if std::fs::write(&tmp, lines.join("\n") + "\n").is_ok()
			&& std::fs::rename(&tmp, &p).is_err()
		{
			let _ = std::fs::remove_file(&tmp);
		}
	}
}

impl Drop for Fit {
	fn drop(&mut self) {
		self.flush();
	}
}

fn fmt(v: &[f64]) -> String {
	v.iter()
		.map(|x| format!("{x:.3}"))
		.collect::<Vec<_>>()
		.join(" / ")
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn fit_learns_freezes_and_reloads() {
		let d = std::env::temp_dir().join(format!("qwen-calib-test-{}", std::process::id()));
		// SAFETY: this test is the only user of the variable
		unsafe { std::env::set_var("QWEN_CALIB_DIR", &d) };
		let fp = fingerprint(&["test"]);
		{
			let mut f = Fit::open("unit", fp, &[1.0, 1.5], 4);
			assert_eq!(f.get(1), 1.5);
			for _ in 0..3 {
				f.add(&[(2.0, 1.0), (1.0, 1.0)]);
			}
			assert_eq!(f.get(0), 1.0, "not frozen before `need` samples");
		}
		// the unfinished sums were stored on drop: one more sample freezes
		let mut f = Fit::open("unit", fp, &[1.0, 1.5], 4);
		f.add(&[(2.0, 1.0), (1.0, 1.0)]);
		assert!((f.get(0) - 2.0).abs() < 1e-12 && (f.get(1) - 1.0).abs() < 1e-12);
		// frozen: later samples do not move it
		f.add(&[(9.0, 1.0), (9.0, 1.0)]);
		assert!((f.get(0) - 2.0).abs() < 1e-12);
		drop(f);
		let f = Fit::open("unit", fp, &[1.0, 1.5], 4);
		assert!((f.get(0) - 2.0).abs() < 1e-12, "frozen entry reloaded");
		// another fingerprint starts from the defaults
		let g = Fit::open("unit", fingerprint(&["other"]), &[1.0, 1.5], 4);
		assert_eq!(g.get(0), 1.0);
		let _ = std::fs::remove_dir_all(&d);
	}
}
