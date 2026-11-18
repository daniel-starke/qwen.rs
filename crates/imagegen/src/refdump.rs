//! Reader for the reference dumps of `scripts/refdump.py` (`NAME.f32` raw
//! little-endian f32 + `manifest.json`), and the error measure of the gates.

use npubuild::st::{parse_json, Json};
use std::path::{Path, PathBuf};

pub struct Dumps {
	pub dir: PathBuf,
	pub manifest: Json,
}

impl Dumps {
	pub fn open(dir: &Path) -> Result<Dumps, String> {
		let p = dir.join("manifest.json");
		let b = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
		Ok(Dumps {
			dir: dir.to_path_buf(),
			manifest: parse_json(&b)?,
		})
	}

	pub fn shape(&self, name: &str) -> Option<Vec<usize>> {
		let t = self.manifest.get("tensors")?.get(name)?;
		Some(
			t.arr()
				.iter()
				.filter_map(Json::num)
				.map(|n| n as usize)
				.collect(),
		)
	}

	pub fn get(&self, name: &str) -> Result<Vec<f32>, String> {
		let p = self.dir.join(format!("{name}.f32"));
		let b = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
		Ok(b.chunks_exact(4)
			.map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
			.collect())
	}

	pub fn num(&self, key: &str) -> Option<f64> {
		self.manifest.get(key).and_then(Json::num)
	}

	pub fn str(&self, key: &str) -> Option<&str> {
		self.manifest.get(key).and_then(Json::str)
	}
}

/// `|a - b| / |b|` over the whole tensor (L2), and the max abs diff.
pub fn rel_err(a: &[f32], b: &[f32]) -> (f64, f64) {
	assert_eq!(a.len(), b.len(), "length mismatch");
	let (mut num, mut den, mut mx) = (0f64, 0f64, 0f64);
	for (&x, &y) in a.iter().zip(b) {
		let d = x as f64 - y as f64;
		num += d * d;
		den += y as f64 * y as f64;
		mx = mx.max(d.abs());
	}
	((num / den.max(1e-30)).sqrt(), mx)
}
