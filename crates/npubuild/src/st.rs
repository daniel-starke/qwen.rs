//! The BF16 checkpoint: a directory of safetensors shards with
//! `model.safetensors.index.json` (Hugging Face `Qwen/Qwen3.8-27B`), read
//! tensor by tensor.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Just enough JSON for a safetensors header, the shard index and
/// `config.json`.
#[derive(Debug, Clone)]
pub enum Json {
	Null,
	Bool(bool),
	Num(f64),
	Str(String),
	Arr(Vec<Json>),
	Obj(Vec<(String, Json)>),
}

impl Json {
	pub fn get(&self, k: &str) -> Option<&Json> {
		match self {
			Json::Obj(v) => v.iter().find(|(n, _)| n == k).map(|(_, v)| v),
			_ => None,
		}
	}
	pub fn num(&self) -> Option<f64> {
		match self {
			Json::Num(n) => Some(*n),
			_ => None,
		}
	}
	pub fn str(&self) -> Option<&str> {
		match self {
			Json::Str(s) => Some(s),
			_ => None,
		}
	}
	pub fn arr(&self) -> &[Json] {
		match self {
			Json::Arr(v) => v,
			_ => &[],
		}
	}
	pub fn obj(&self) -> &[(String, Json)] {
		match self {
			Json::Obj(v) => v,
			_ => &[],
		}
	}
}

pub fn parse_json(s: &[u8]) -> Result<Json, String> {
	let mut p = 0;
	let v = value(s, &mut p)?;
	Ok(v)
}

fn ws(s: &[u8], p: &mut usize) {
	while *p < s.len() && s[*p].is_ascii_whitespace() {
		*p += 1;
	}
}

fn value(s: &[u8], p: &mut usize) -> Result<Json, String> {
	ws(s, p);
	let err = |p: usize| format!("json: unexpected input at byte {p}");
	match s.get(*p).copied().ok_or_else(|| err(*p))? {
		b'{' => {
			*p += 1;
			let mut v = Vec::new();
			loop {
				ws(s, p);
				if s.get(*p) == Some(&b'}') {
					*p += 1;
					return Ok(Json::Obj(v));
				}
				let k = match value(s, p)? {
					Json::Str(k) => k,
					_ => return Err(err(*p)),
				};
				ws(s, p);
				if s.get(*p) != Some(&b':') {
					return Err(err(*p));
				}
				*p += 1;
				v.push((k, value(s, p)?));
				ws(s, p);
				if s.get(*p) == Some(&b',') {
					*p += 1;
				}
			}
		}
		b'[' => {
			*p += 1;
			let mut v = Vec::new();
			loop {
				ws(s, p);
				if s.get(*p) == Some(&b']') {
					*p += 1;
					return Ok(Json::Arr(v));
				}
				v.push(value(s, p)?);
				ws(s, p);
				if s.get(*p) == Some(&b',') {
					*p += 1;
				}
			}
		}
		b'"' => {
			*p += 1;
			let mut out = Vec::new();
			while *p < s.len() && s[*p] != b'"' {
				if s[*p] == b'\\' && *p + 1 < s.len() {
					*p += 1;
					match s[*p] {
						b'n' => out.push(b'\n'),
						b't' => out.push(b'\t'),
						b'r' => out.push(b'\r'),
						b'u' if *p + 4 < s.len() => {
							let c = std::str::from_utf8(&s[*p + 1..*p + 5])
								.ok()
								.and_then(|h| u32::from_str_radix(h, 16).ok())
								.and_then(char::from_u32)
								.unwrap_or('?');
							out.extend_from_slice(c.to_string().as_bytes());
							*p += 4;
						}
						c => out.push(c),
					}
				} else {
					out.push(s[*p]);
				}
				*p += 1;
			}
			*p += 1;
			Ok(Json::Str(String::from_utf8_lossy(&out).into_owned()))
		}
		b't' if s[*p..].starts_with(b"true") => {
			*p += 4;
			Ok(Json::Bool(true))
		}
		b'f' if s[*p..].starts_with(b"false") => {
			*p += 5;
			Ok(Json::Bool(false))
		}
		b'n' if s[*p..].starts_with(b"null") => {
			*p += 4;
			Ok(Json::Null)
		}
		_ => {
			let b = *p;
			while *p < s.len() && matches!(s[*p], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') {
				*p += 1;
			}
			std::str::from_utf8(&s[b..*p])
				.ok()
				.and_then(|t| t.parse().ok())
				.map(Json::Num)
				.ok_or_else(|| err(b))
		}
	}
}

struct Entry {
	shard: usize,
	dtype: String,
	shape: Vec<usize>,
	off: u64,
	len: u64,
}

/// The checkpoint directory.
pub struct Checkpoint {
	pub dir: PathBuf,
	shards: Vec<PathBuf>,
	tensors: HashMap<String, Entry>,
	pub config: Json,
}

impl Checkpoint {
	pub fn open(dir: &Path) -> Result<Checkpoint, String> {
		let rd = |f: &str| {
			std::fs::read(dir.join(f)).map_err(|e| format!("{}: {e}", dir.join(f).display()))
		};
		let index = parse_json(&rd("model.safetensors.index.json")?)?;
		let config = parse_json(&rd("config.json")?)?;
		let map = index
			.get("weight_map")
			.ok_or("model.safetensors.index.json: no weight_map")?;
		let mut names: Vec<String> = map
			.obj()
			.iter()
			.filter_map(|(_, v)| v.str().map(str::to_string))
			.collect();
		names.sort_unstable();
		names.dedup();
		Self::open_shards(dir, &names, config)
	}

	/// A diffusers-style component directory (`transformer/`, `vae/`,
	/// `text_encoder/`): the shards named by its `*.safetensors.index.json`,
	/// else its one `*.safetensors` file; `config.json` when present.
	pub fn open_dir(dir: &Path) -> Result<Checkpoint, String> {
		let e = |x: std::io::Error| format!("{}: {x}", dir.display());
		let mut files: Vec<String> = std::fs::read_dir(dir)
			.map_err(e)?
			.filter_map(|d| d.ok()?.file_name().into_string().ok())
			.collect();
		files.sort_unstable();
		let config = match std::fs::read(dir.join("config.json")) {
			Ok(b) => parse_json(&b)?,
			Err(_) => Json::Null,
		};
		if let Some(ix) = files
			.iter()
			.find(|f| f.ends_with(".safetensors.index.json"))
		{
			let index = parse_json(&std::fs::read(dir.join(ix)).map_err(e)?)?;
			let map = index
				.get("weight_map")
				.ok_or(format!("{ix}: no weight_map"))?;
			let mut names: Vec<String> = map
				.obj()
				.iter()
				.filter_map(|(_, v)| v.str().map(str::to_string))
				.collect();
			names.sort_unstable();
			names.dedup();
			return Self::open_shards(dir, &names, config);
		}
		let st: Vec<String> = files
			.into_iter()
			.filter(|f| f.ends_with(".safetensors"))
			.collect();
		if st.len() != 1 {
			return Err(format!(
				"{}: {} safetensors files and no index",
				dir.display(),
				st.len()
			));
		}
		Self::open_shards(dir, &st, config)
	}

	fn open_shards(dir: &Path, names: &[String], config: Json) -> Result<Checkpoint, String> {
		let mut shards = Vec::new();
		let mut tensors = HashMap::new();
		for (si, n) in names.iter().enumerate() {
			let path = dir.join(n);
			let e = |x: std::io::Error| format!("{}: {x}", path.display());
			let mut f = std::fs::File::open(&path).map_err(e)?;
			let mut b = [0u8; 8];
			f.read_exact(&mut b).map_err(e)?;
			let hn = u64::from_le_bytes(b);
			if hn > 1 << 27 {
				return Err(format!("{}: not a safetensors file", path.display()));
			}
			let mut h = vec![0u8; hn as usize];
			f.read_exact(&mut h).map_err(e)?;
			for (name, info) in parse_json(&h)?.obj() {
				if name == "__metadata__" {
					continue;
				}
				let off: Vec<u64> = info
					.get("data_offsets")
					.map_or(&[][..], Json::arr)
					.iter()
					.filter_map(Json::num)
					.map(|n| n as u64)
					.collect();
				if off.len() != 2 {
					return Err(format!("{name}: bad data_offsets"));
				}
				tensors.insert(
					name.clone(),
					Entry {
						shard: si,
						dtype: info
							.get("dtype")
							.and_then(Json::str)
							.unwrap_or("")
							.to_string(),
						shape: info
							.get("shape")
							.map_or(&[][..], Json::arr)
							.iter()
							.filter_map(Json::num)
							.map(|n| n as usize)
							.collect(),
						off: 8 + hn + off[0],
						len: off[1] - off[0],
					},
				);
			}
			shards.push(path);
		}
		Ok(Checkpoint {
			dir: dir.to_path_buf(),
			shards,
			tensors,
			config,
		})
	}

	/// True when `dir` looks like the checkpoint (not an OpenVINO export).
	pub fn is_checkpoint(dir: &Path) -> bool {
		dir.join("model.safetensors.index.json").is_file()
	}

	pub fn has(&self, name: &str) -> bool {
		self.tensors.contains_key(name)
	}

	/// Every tensor's name.
	pub fn names(&self) -> impl Iterator<Item = &str> {
		self.tensors.keys().map(String::as_str)
	}

	pub fn shape(&self, name: &str) -> Result<&[usize], String> {
		self.tensors
			.get(name)
			.map(|e| &e.shape[..])
			.ok_or_else(|| format!("checkpoint: no tensor {name}"))
	}

	fn raw(&self, name: &str, rows: Option<(usize, usize)>) -> Result<(Vec<u8>, &Entry), String> {
		let e = self
			.tensors
			.get(name)
			.ok_or_else(|| format!("checkpoint: no tensor {name}"))?;
		let (mut off, mut len) = (e.off, e.len);
		if let Some((r0, r1)) = rows {
			let row = e.len / e.shape[0] as u64;
			off += r0 as u64 * row;
			len = (r1 - r0) as u64 * row;
		}
		let path = &self.shards[e.shard];
		let er = |x: std::io::Error| format!("{}: {x}", path.display());
		let mut f = std::fs::File::open(path).map_err(er)?;
		f.seek(SeekFrom::Start(off)).map_err(er)?;
		// read into the spare capacity (File reads uninitialized memory
		// natively): no zero fill of a buffer the read overwrites
		let mut raw = Vec::with_capacity(len as usize);
		(&mut f).take(len).read_to_end(&mut raw).map_err(er)?;
		if raw.len() as u64 != len {
			return Err(format!("{}: {name}: short read", path.display()));
		}
		Ok((raw, e))
	}

	fn widen(name: &str, raw: &[u8], dtype: &str) -> Result<Vec<f32>, String> {
		if dtype == "BF16" && raw.len() >= 1 << 22 {
			// large matrices (the image model's per-step reloads): on the pool
			let n = raw.len() / 2;
			let mut out = quant::uninit_vec(n);
			const C: usize = 1 << 18;
			let (src, dst) = (raw.as_ptr() as usize, out.as_mut_ptr() as usize);
			quant::par_for(n.div_ceil(C), move |c| {
				let (i0, i1) = (c * C, ((c + 1) * C).min(n));
				for i in i0..i1 {
					// SAFETY: chunk c owns out[i0..i1]; raw holds 2 n bytes
					unsafe {
						let b = *(src as *const [u8; 2]).add(i);
						*(dst as *mut f32).add(i) =
							f32::from_bits((u16::from_le_bytes(b) as u32) << 16);
					}
				}
			});
			return Ok(out);
		}
		Ok(match dtype {
			"BF16" => raw
				.chunks_exact(2)
				.map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
				.collect(),
			"F16" => raw
				.chunks_exact(2)
				.map(|b| quant::fp16_to_f32(u16::from_le_bytes([b[0], b[1]])))
				.collect(),
			"F32" => raw
				.chunks_exact(4)
				.map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
				.collect(),
			d => return Err(format!("{name}: unsupported dtype {d}")),
		})
	}

	/// Tensor `name` as f32 (BF16 / F16 / F32).
	pub fn f32(&self, name: &str) -> Result<Vec<f32>, String> {
		let (raw, e) = self.raw(name, None)?;
		Self::widen(name, &raw, &e.dtype)
	}

	/// Tensor `name`'s raw bytes, streamed into `w` in pieces (for tensors
	/// too large to hold).
	pub fn copy_raw(&self, name: &str, w: &mut dyn std::io::Write) -> Result<u64, String> {
		let e = self
			.tensors
			.get(name)
			.ok_or_else(|| format!("checkpoint: no tensor {name}"))?;
		let path = &self.shards[e.shard];
		let er = |x: std::io::Error| format!("{}: {x}", path.display());
		let mut f = std::fs::File::open(path).map_err(er)?;
		f.seek(SeekFrom::Start(e.off)).map_err(er)?;
		let mut left = e.len;
		let mut buf = vec![0u8; 64 << 20];
		while left > 0 {
			let n = left.min(buf.len() as u64) as usize;
			f.read_exact(&mut buf[..n]).map_err(er)?;
			w.write_all(&buf[..n]).map_err(|x| x.to_string())?;
			left -= n as u64;
		}
		Ok(e.len)
	}

	/// The dtype of tensor `name` (`"BF16"`, ...).
	pub fn dtype(&self, name: &str) -> Result<&str, String> {
		self.tensors
			.get(name)
			.map(|e| e.dtype.as_str())
			.ok_or_else(|| format!("checkpoint: no tensor {name}"))
	}

	/// An `I64` tensor (small index buffers).
	pub fn i64s(&self, name: &str) -> Result<Vec<i64>, String> {
		let (raw, e) = self.raw(name, None)?;
		if e.dtype != "I64" {
			return Err(format!("{name}: {} is not I64", e.dtype));
		}
		Ok(raw
			.chunks_exact(8)
			.map(|b| i64::from_le_bytes(b.try_into().unwrap()))
			.collect())
	}

	/// Rows `[r0, r1)` of the first axis of `name`, as f32.
	pub fn rows(&self, name: &str, r0: usize, r1: usize) -> Result<Vec<f32>, String> {
		let (raw, e) = self.raw(name, Some((r0, r1)))?;
		Self::widen(name, &raw, &e.dtype)
	}

	/// A matrix `[n, k]` (any trailing axes flattened into `k`), checked.
	pub fn mat(&self, name: &str, n: usize, k: usize) -> Result<Vec<f32>, String> {
		let sh = self.shape(name)?;
		if sh.first() != Some(&n) || sh.iter().product::<usize>() != n * k {
			return Err(format!("{name}: shape {sh:?}, expected [{n}, {k}]"));
		}
		self.f32(name)
	}

	pub fn vec(&self, name: &str, n: usize) -> Result<Vec<f32>, String> {
		let v = self.f32(name)?;
		if v.len() != n {
			return Err(format!("{name}: {} values, expected {n}", v.len()));
		}
		Ok(v)
	}
}
