//! Accuracy-debugging hooks of the OpenVINO engines, all off unless their
//! variable is set. They exist to compare devices position by position:
//!
//! - `QWEN_OV_TF_DUMP=FILE`: every position's logits that a forward pass
//!   computes (all rows of a prefill chunk, not just the last), raw f32, one
//!   row of the model's full (padded) vocab after another. Fed the same
//!   prompt, two devices' dumps give the teacher-forced error per position
//!   (`scripts/kl.py`).
//! - `QWEN_OV_STATE_SAVE=DIR@POS` (NPU engine, any static device): after the
//!   step that brings the conversation to `POS` tokens, write every state
//!   (KV past buffer, conv, SSM; f16) to `DIR/sNNN.f16`, and print the
//!   magnitude of each recurrent state.
//! - `QWEN_OV_STATE_LOAD=DIR@POS`: before the step that starts at `POS`,
//!   replace the states with those files. `QWEN_OV_STATE_LOAD_ONLY=kv,ssm,
//!   conv` (substrings of the state names; `key` / `value` for one KV half,
//!   `ssm.3` for one layer: matched as `.ssm.3` exactly) restricts it.
//!   With a dump saved on another device, this separates "error carried in
//!   the state" from "error of the remaining steps". `QWEN_OV_STATE_LOAD_SKIP=1`
//!   does not run the first `POS` positions of the prompt at all (the
//!   loaded states stand in for them), so a test of the last few positions
//!   costs a few decode steps.

use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

static TF: Mutex<Option<std::fs::File>> = Mutex::new(None);
static TF_INIT: std::sync::Once = std::sync::Once::new();

/// Append `rows` (whole logits rows) to the `QWEN_OV_TF_DUMP` file.
pub fn tf_dump(rows: &[f32]) {
	TF_INIT.call_once(|| {
		if let Some(f) = std::env::var_os("QWEN_OV_TF_DUMP") {
			match std::fs::File::create(&f) {
				Ok(h) => *TF.lock().unwrap() = Some(h),
				Err(e) => eprintln!("QWEN_OV_TF_DUMP: {e}"),
			}
		}
	});
	if let Some(f) = TF.lock().unwrap().as_mut() {
		let b: &[u8] = unsafe {
			std::slice::from_raw_parts(rows.as_ptr() as *const u8, std::mem::size_of_val(rows))
		};
		if let Err(e) = f.write_all(b) {
			eprintln!("QWEN_OV_TF_DUMP: {e}");
		}
	}
}

pub fn tf_enabled() -> bool {
	std::env::var_os("QWEN_OV_TF_DUMP").is_some()
}

/// `DIR@POS` from `var`.
pub fn dir_at(var: &str) -> Option<(PathBuf, usize)> {
	let v = std::env::var(var).ok()?;
	let (d, p) = v.rsplit_once('@')?;
	Some((PathBuf::from(d), p.parse().ok()?))
}

/// Does state `name` pass the `QWEN_OV_STATE_LOAD_ONLY` filter?
pub fn load_selected(name: &str) -> bool {
	let Ok(f) = std::env::var("QWEN_OV_STATE_LOAD_ONLY") else {
		return true;
	};
	f.split(',')
		.map(str::trim)
		.filter(|s| !s.is_empty())
		.any(|s| {
			if s == "kv" {
				name.contains(".key.") || name.contains(".value.")
			} else if s.contains('.') {
				// `ssm.3`: that layer only (not `ssm.30`).
				name.ends_with(&format!(".{s}"))
			} else {
				name.contains(&format!(".{s}."))
			}
		})
}

pub fn f16_to_f32(h: u16) -> f32 {
	let s = ((h >> 15) as u32) << 31;
	let e = ((h >> 10) & 0x1f) as u32;
	let m = (h & 0x3ff) as u32;
	let bits = if e == 0 {
		if m == 0 {
			s
		} else {
			// Subnormal: normalize.
			let mut e2 = 127 - 15 + 1;
			let mut m2 = m;
			while m2 & 0x400 == 0 {
				m2 <<= 1;
				e2 -= 1;
			}
			s | (e2 << 23) | ((m2 & 0x3ff) << 13)
		}
	} else if e == 31 {
		s | 0x7f80_0000 | (m << 13)
	} else {
		s | ((e + 127 - 15) << 23) | (m << 13)
	};
	f32::from_bits(bits)
}

/// `(rms, max |x|, zeros, subnormals, non-finite)` of an f16 buffer.
pub fn f16_stats(x: &[u16]) -> (f64, f32, usize, usize, usize) {
	let (mut ss, mut mx, mut z, mut sub, mut nf) = (0f64, 0f32, 0, 0, 0);
	for &h in x {
		let e = (h >> 10) & 0x1f;
		if h & 0x7fff == 0 {
			z += 1;
		} else if e == 0 {
			sub += 1;
		} else if e == 31 {
			nf += 1;
			continue;
		}
		let v = f16_to_f32(h);
		ss += (v as f64) * (v as f64);
		mx = mx.max(v.abs());
	}
	((ss / x.len().max(1) as f64).sqrt(), mx, z, sub, nf)
}

static PROBE: Mutex<Option<std::fs::File>> = Mutex::new(None);
static PROBE_INIT: std::sync::Once = std::sync::Once::new();

/// Append one probe output to the `QWEN_OV_PROBE_DUMP` file.
pub fn probe_dump(x: &[f32]) {
	PROBE_INIT.call_once(|| {
		if let Some(f) = std::env::var_os("QWEN_OV_PROBE_DUMP") {
			match std::fs::File::create(&f) {
				Ok(h) => *PROBE.lock().unwrap() = Some(h),
				Err(e) => eprintln!("QWEN_OV_PROBE_DUMP: {e}"),
			}
		}
	});
	if let Some(f) = PROBE.lock().unwrap().as_mut() {
		let b: &[u8] = unsafe {
			std::slice::from_raw_parts(x.as_ptr() as *const u8, std::mem::size_of_val(x))
		};
		if let Err(e) = f.write_all(b) {
			eprintln!("QWEN_OV_PROBE_DUMP: {e}");
		}
	}
}
