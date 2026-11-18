//! The files `--prompt-cache-dir` held before the disk tier of
//! `crates/prefixcache`: one state per file (`*.qwrs`: a GGUF `RunState`,
//! `*.qwes`: an engine's snapshot), each recording the model it was made
//! with (`tag`) and its token run. Read once more at startup, so a restart
//! after the upgrade does not pay their prefill again: [`import`] hands the
//! states to the cache, which writes them as packs, and deletes a file once
//! the cache's disk tier has its state.

use std::fs;
#[cfg(test)]
use std::io::Write;
use std::io::{BufReader, Read};
use std::path::Path;

use model::RunState;

use crate::EngineSnap;

/// A state the directory can hold: the GGUF `RunState`, or an engine's
/// `EngineSnap`. Each kind has its own file extension, so one directory can
/// serve both without either deleting the other's files as unreadable.
pub trait Persist: Send + Sync + Sized + 'static {
	const EXT: &'static str;
	/// (only the tests still write the old format)
	#[cfg(test)]
	fn write_to(&self, w: &mut impl Write, tag: &str, tokens: &[u32]) -> std::io::Result<()>;
	fn read_header(r: &mut impl Read) -> std::io::Result<(String, Vec<u32>)>;
	fn read_body(r: &mut impl Read) -> std::io::Result<Self>;
}

impl Persist for RunState {
	const EXT: &'static str = "qwrs";
	#[cfg(test)]
	fn write_to(&self, w: &mut impl Write, tag: &str, tokens: &[u32]) -> std::io::Result<()> {
		RunState::write_to(self, w, tag, tokens)
	}
	fn read_header(r: &mut impl Read) -> std::io::Result<(String, Vec<u32>)> {
		RunState::read_header(r)
	}
	fn read_body(r: &mut impl Read) -> std::io::Result<Self> {
		RunState::read_body(r)
	}
}

const ENGINE_MAGIC: &[u8; 8] = b"QWENES01";

#[cfg(test)]
fn put_u64(w: &mut impl Write, v: u64) -> std::io::Result<()> {
	w.write_all(&v.to_le_bytes())
}

fn get_u64(r: &mut impl Read) -> std::io::Result<u64> {
	let mut b = [0u8; 8];
	r.read_exact(&mut b)?;
	Ok(u64::from_le_bytes(b))
}

fn get_len(r: &mut impl Read, max: u64) -> std::io::Result<usize> {
	let n = get_u64(r)?;
	if n > max {
		return Err(std::io::Error::new(
			std::io::ErrorKind::InvalidData,
			format!("length {n} out of range"),
		));
	}
	Ok(n as usize)
}

impl Persist for EngineSnap {
	const EXT: &'static str = "qwes";
	#[cfg(test)]
	fn write_to(&self, w: &mut impl Write, tag: &str, tokens: &[u32]) -> std::io::Result<()> {
		w.write_all(ENGINE_MAGIC)?;
		put_u64(w, tag.len() as u64)?;
		w.write_all(tag.as_bytes())?;
		put_u64(w, tokens.len() as u64)?;
		for t in tokens {
			w.write_all(&t.to_le_bytes())?;
		}
		put_u64(w, self.pos as u64)?;
		put_u64(w, self.parts.len() as u64)?;
		for p in &self.parts {
			put_u64(w, p.iter().map(|c| c.len()).sum::<usize>() as u64)?;
			for c in p {
				w.write_all(c.bytes())?;
			}
		}
		Ok(())
	}
	fn read_header(r: &mut impl Read) -> std::io::Result<(String, Vec<u32>)> {
		let mut magic = [0u8; 8];
		r.read_exact(&mut magic)?;
		if &magic != ENGINE_MAGIC {
			return Err(std::io::Error::new(
				std::io::ErrorKind::InvalidData,
				"magic",
			));
		}
		let mut tag = vec![0u8; get_len(r, 1 << 16)?];
		r.read_exact(&mut tag)?;
		let mut tb = vec![0u8; get_len(r, 1 << 24)? * 4];
		r.read_exact(&mut tb)?;
		let tokens = tb
			.chunks_exact(4)
			.map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
			.collect();
		Ok((String::from_utf8_lossy(&tag).into_owned(), tokens))
	}
	fn read_body(r: &mut impl Read) -> std::io::Result<Self> {
		let pos = get_u64(r)? as usize;
		let n = get_len(r, 1 << 16)?;
		let mut parts = Vec::with_capacity(n);
		for _ in 0..n {
			let mut p = vec![0u8; get_len(r, 1 << 34)?];
			r.read_exact(&mut p)?;
			parts.push(quant::snap::part(p));
		}
		Ok(EngineSnap {
			pos,
			parts,
			layout: Vec::new(),
		})
	}
}

/// Every legacy state in `dir` made for the model `tag`, except those
/// `known` already holds (their files are deleted: the disk tier has them).
/// Unreadable files are reported and skipped.
pub fn import<S: Persist>(
	dir: &Path,
	tag: &str,
	known: &dyn Fn(&[u32]) -> bool,
) -> Vec<(Vec<u32>, S)> {
	let mut out = Vec::new();
	let Ok(rd) = fs::read_dir(dir) else {
		return out;
	};
	for e in rd.flatten() {
		let path = e.path();
		if path.extension().is_none_or(|x| x != S::EXT) {
			continue;
		}
		let r = fs::File::open(&path).and_then(|f| {
			let mut r = BufReader::with_capacity(1 << 20, f);
			let (t, tokens) = S::read_header(&mut r)?;
			if t != tag {
				return Ok(None);
			}
			if known(&tokens) {
				return Ok(Some(None));
			}
			Ok(Some(Some((tokens, S::read_body(&mut r)?))))
		});
		match r {
			Ok(Some(Some(e))) => out.push(e),
			Ok(Some(None)) => {
				if fs::remove_file(&path).is_ok() {
					eprintln!(
						"prompt cache dir: {} moved to the new store, removed",
						path.display()
					);
				}
			}
			Ok(None) => {}
			Err(err) => eprintln!(
				"prompt cache dir: {} unreadable ({err}), skipped",
				path.display()
			),
		}
	}
	if !out.is_empty() {
		eprintln!(
			"prompt cache dir: importing {} state(s) of the old format",
			out.len()
		);
	}
	out
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn engine_snapshots_import_and_keep_to_their_tag() {
		let dir = std::env::temp_dir().join(format!("qwen-rs-snapdir-test-{}", std::process::id()));
		let _ = fs::remove_dir_all(&dir);
		fs::create_dir_all(&dir).unwrap();
		use quant::snap::part;
		let snap = EngineSnap {
			pos: 5,
			parts: vec![part(vec![1, 2, 3]), part(vec![]), part(vec![9; 100])],
			layout: vec![],
		};
		let write = |name: &str, tag: &str, tokens: &[u32]| {
			let mut f = fs::File::create(dir.join(name)).unwrap();
			snap.write_to(&mut f, tag, tokens).unwrap();
		};
		write("a.qwes", "a", &[7, 8, 9]);
		write("b.qwes", "b", &[1]);
		let got = import::<EngineSnap>(&dir, "a", &|_| false);
		assert_eq!(got, vec![(vec![7, 8, 9], snap.clone())]);
		// a RunState import does not see the engine files
		assert!(import::<RunState>(&dir, "a", &|_| false).is_empty());
		// once the new store has it, the old file goes; another model's stays
		assert!(import::<EngineSnap>(&dir, "a", &|t| t == [7, 8, 9]).is_empty());
		assert!(!dir.join("a.qwes").exists() && dir.join("b.qwes").exists());
		let _ = fs::remove_dir_all(&dir);
	}
}
