//! Weights converted once and kept on disk (`--image-cache DIR`,
//! `QWEN_IMAGE_CACHE_DIR`): a model's quantized matmuls and the small f32
//! tensors next to them, read back in the order they were written. A file
//! starts with a key naming what it was made from (the checkpoint shards'
//! names, sizes and modification times, the weight format, the rotation
//! switch); a file whose key differs is made again. A file is written under
//! a temporary name and renamed when complete, so a stopped write leaves
//! nothing that would be read.

use crate::ops::Lin;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"QWIMGW01";

static DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// Set the cache directory (`--image-cache`), ahead of the variable.
pub fn set_dir(p: PathBuf) {
	let _ = DIR.set(p);
}

/// The cache directory: [`set_dir`], else `QWEN_IMAGE_CACHE_DIR`; `None` =
/// nothing is cached.
pub fn dir() -> Option<PathBuf> {
	DIR.get().cloned().or_else(|| {
		std::env::var_os("QWEN_IMAGE_CACHE_DIR")
			.filter(|v| !v.is_empty())
			.map(PathBuf::from)
	})
}

/// A key for the `*.safetensors` files of `folder` plus `extra`.
pub fn key(folder: &Path, extra: &str) -> Result<String, String> {
	let mut files: Vec<_> = std::fs::read_dir(folder)
		.map_err(|e| format!("{}: {e}", folder.display()))?
		.filter_map(Result::ok)
		.filter(|e| e.file_name().to_string_lossy().ends_with(".safetensors"))
		.collect();
	files.sort_by_key(|e| e.file_name());
	let mut k = String::from(extra);
	for f in files {
		let m = f
			.metadata()
			.map_err(|e| format!("{}: {e}", f.path().display()))?;
		let t = m
			.modified()
			.ok()
			.and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
			.map_or(0, |d| d.as_secs());
		k.push_str(&format!(
			";{}:{}:{t}",
			f.file_name().to_string_lossy(),
			m.len()
		));
	}
	Ok(k)
}

pub struct Writer {
	f: BufWriter<File>,
	tmp: PathBuf,
	path: PathBuf,
}

impl Writer {
	pub fn create(path: &Path, key: &str) -> Result<Writer, String> {
		if let Some(p) = path.parent() {
			std::fs::create_dir_all(p).map_err(|e| format!("{}: {e}", p.display()))?;
		}
		let tmp = path.with_extension("tmp");
		let f = File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
		let mut w = Writer {
			f: BufWriter::with_capacity(8 << 20, f),
			tmp,
			path: path.to_path_buf(),
		};
		w.bytes(MAGIC)?;
		w.u64(key.len() as u64)?;
		w.bytes(key.as_bytes())?;
		Ok(w)
	}

	fn bytes(&mut self, b: &[u8]) -> Result<(), String> {
		self.f
			.write_all(b)
			.map_err(|e| format!("{}: {e}", self.tmp.display()))
	}

	fn u64(&mut self, v: u64) -> Result<(), String> {
		self.bytes(&v.to_le_bytes())
	}

	pub fn f32s(&mut self, v: &[f32]) -> Result<(), String> {
		self.u64(v.len() as u64)?;
		// SAFETY: f32 has no padding; the file is little-endian like the host
		self.bytes(unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 4) })
	}

	pub fn lin(&mut self, l: &Lin) -> Result<(), String> {
		match l {
			Lin::F32 { w, n, k } => {
				self.bytes(&[0, 0])?;
				self.u64(*n as u64)?;
				self.u64(*k as u64)?;
				self.f32s(w)
			}
			Lin::Q { q, rot } => {
				let (tag, raw, n, k) = match q {
					quant::LinearWeight::Q4K {
						raw,
						out_dim,
						in_dim,
					} => (1u8, raw, out_dim, in_dim),
					quant::LinearWeight::Q6K {
						raw,
						out_dim,
						in_dim,
					} => (2u8, raw, out_dim, in_dim),
					_ => return Err("weight cache: only K-quant or f32 weights".into()),
				};
				self.bytes(&[tag, *rot as u8])?;
				self.u64(*n as u64)?;
				self.u64(*k as u64)?;
				self.u64(raw.len() as u64)?;
				self.bytes(raw)
			}
		}
	}

	/// Flush and put the file in place.
	pub fn finish(mut self) -> Result<(), String> {
		self.f
			.flush()
			.map_err(|e| format!("{}: {e}", self.tmp.display()))?;
		drop(self.f);
		std::fs::rename(&self.tmp, &self.path)
			.map_err(|e| format!("{} -> {}: {e}", self.tmp.display(), self.path.display()))
	}
}

pub struct Reader {
	f: BufReader<File>,
	path: PathBuf,
}

impl Reader {
	/// `path` when it exists and was made for `key`.
	pub fn open(path: &Path, key: &str) -> Option<Reader> {
		let f = File::open(path).ok()?;
		let mut r = Reader {
			f: BufReader::with_capacity(8 << 20, f),
			path: path.to_path_buf(),
		};
		let mut m = [0u8; 8];
		r.f.read_exact(&mut m).ok()?;
		if &m != MAGIC {
			return None;
		}
		let n = r.u64().ok()? as usize;
		if n != key.len() {
			return None;
		}
		let k = r.vec(n).ok()?;
		(k == key.as_bytes()).then_some(r)
	}

	fn err(&self, e: impl std::fmt::Display) -> String {
		format!("{}: {e}", self.path.display())
	}

	fn u64(&mut self) -> Result<u64, String> {
		let mut b = [0u8; 8];
		self.f.read_exact(&mut b).map_err(|e| self.err(e))?;
		Ok(u64::from_le_bytes(b))
	}

	fn vec(&mut self, n: usize) -> Result<Vec<u8>, String> {
		let mut v = Vec::with_capacity(n);
		(&mut self.f)
			.take(n as u64)
			.read_to_end(&mut v)
			.map_err(|e| format!("{}: {e}", self.path.display()))?;
		if v.len() != n {
			return Err(self.err("truncated"));
		}
		Ok(v)
	}

	pub fn f32s(&mut self) -> Result<Vec<f32>, String> {
		let n = self.u64()? as usize;
		let mut v = quant::uninit_vec(n);
		// SAFETY: the bytes of n f32s, every one written by read_exact
		let b = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), n * 4) };
		self.f.read_exact(b).map_err(|e| self.err(e))?;
		Ok(v)
	}

	pub fn lin(&mut self) -> Result<Lin, String> {
		let mut t = [0u8; 2];
		self.f.read_exact(&mut t).map_err(|e| self.err(e))?;
		let n = self.u64()? as usize;
		let k = self.u64()? as usize;
		if t[0] == 0 {
			let w = self.f32s()?;
			if w.len() != n * k {
				return Err(self.err("f32 weight of the wrong size"));
			}
			return Ok(Lin::F32 { w, n, k });
		}
		let len = self.u64()? as usize;
		let raw = self.vec(len)?;
		let q = match t[0] {
			1 => quant::LinearWeight::Q4K {
				raw,
				out_dim: n,
				in_dim: k,
			},
			2 => quant::LinearWeight::Q6K {
				raw,
				out_dim: n,
				in_dim: k,
			},
			x => return Err(self.err(format!("weight tag {x}"))),
		};
		Ok(Lin::Q { q, rot: t[1] != 0 })
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::ops::WFmt;

	#[test]
	fn roundtrip() {
		let dir = std::env::temp_dir().join(format!("qwimgw-{}", std::process::id()));
		let p = dir.join("t.bin");
		let w: Vec<f32> = (0..2 * 256).map(|i| (i as f32 * 0.37).sin()).collect();
		let q = Lin::new(w.clone(), 2, 256, WFmt::Q6K);
		let f = Lin::new(w[..2 * 100].to_vec(), 2, 100, WFmt::Q6K);
		let mut wr = Writer::create(&p, "k1").unwrap();
		wr.lin(&q).unwrap();
		wr.lin(&f).unwrap();
		wr.f32s(&[1.0, 2.0]).unwrap();
		wr.finish().unwrap();
		assert!(Reader::open(&p, "k2").is_none());
		let mut r = Reader::open(&p, "k1").unwrap();
		let (q2, f2) = (r.lin().unwrap(), r.lin().unwrap());
		assert_eq!(r.f32s().unwrap(), vec![1.0, 2.0]);
		let x: Vec<f32> = (0..256).map(|i| i as f32 / 256.0).collect();
		assert_eq!(q.apply(&x, 1), q2.apply(&x, 1));
		assert_eq!(f.apply(&x[..100], 1), f2.apply(&x[..100], 1));
		let _ = std::fs::remove_dir_all(dir);
	}
}
