//! Just enough JSON to read a Hugging Face `tokenizer.json`: the BPE vocab,
//! the merges and `added_tokens`. A full parser into a value tree; the file is
//! ~20 MB and read once.

use std::collections::HashMap;

pub enum J {
	Null,
	Bool(bool),
	Num(f64),
	Str(String),
	Arr(Vec<J>),
	Obj(Vec<(String, J)>),
}

impl J {
	pub fn get(&self, k: &str) -> Option<&J> {
		match self {
			J::Obj(m) => m.iter().find(|(a, _)| a == k).map(|(_, v)| v),
			_ => None,
		}
	}
	fn str(&self) -> Option<&str> {
		match self {
			J::Str(s) => Some(s),
			_ => None,
		}
	}
	fn num(&self) -> Option<f64> {
		match self {
			J::Num(n) => Some(*n),
			_ => None,
		}
	}
}

struct P<'a> {
	b: &'a [u8],
	i: usize,
}

impl P<'_> {
	fn ws(&mut self) {
		while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
			self.i += 1;
		}
	}
	fn err(&self, m: &str) -> String {
		format!("tokenizer.json: {m} at byte {}", self.i)
	}
	fn lit(&mut self, s: &str, v: J) -> Result<J, String> {
		if self.b[self.i..].starts_with(s.as_bytes()) {
			self.i += s.len();
			Ok(v)
		} else {
			Err(self.err("bad literal"))
		}
	}
	fn value(&mut self) -> Result<J, String> {
		self.ws();
		match self.b.get(self.i) {
			Some(b'{') => {
				self.i += 1;
				let mut m = Vec::new();
				self.ws();
				if self.b.get(self.i) == Some(&b'}') {
					self.i += 1;
					return Ok(J::Obj(m));
				}
				loop {
					self.ws();
					let k = self.string()?;
					self.ws();
					if self.b.get(self.i) != Some(&b':') {
						return Err(self.err("expected ':'"));
					}
					self.i += 1;
					m.push((k, self.value()?));
					self.ws();
					match self.b.get(self.i) {
						Some(b',') => self.i += 1,
						Some(b'}') => {
							self.i += 1;
							return Ok(J::Obj(m));
						}
						_ => return Err(self.err("expected ',' or '}'")),
					}
				}
			}
			Some(b'[') => {
				self.i += 1;
				let mut a = Vec::new();
				self.ws();
				if self.b.get(self.i) == Some(&b']') {
					self.i += 1;
					return Ok(J::Arr(a));
				}
				loop {
					a.push(self.value()?);
					self.ws();
					match self.b.get(self.i) {
						Some(b',') => self.i += 1,
						Some(b']') => {
							self.i += 1;
							return Ok(J::Arr(a));
						}
						_ => return Err(self.err("expected ',' or ']'")),
					}
				}
			}
			Some(b'"') => Ok(J::Str(self.string()?)),
			Some(b't') => self.lit("true", J::Bool(true)),
			Some(b'f') => self.lit("false", J::Bool(false)),
			Some(b'n') => self.lit("null", J::Null),
			Some(_) => {
				let s = self.i;
				while self.i < self.b.len()
					&& matches!(
						self.b[self.i],
						b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9'
					) {
					self.i += 1;
				}
				std::str::from_utf8(&self.b[s..self.i])
					.ok()
					.and_then(|t| t.parse().ok())
					.map(J::Num)
					.ok_or_else(|| self.err("bad number"))
			}
			None => Err(self.err("unexpected end")),
		}
	}
	fn hex4(&mut self) -> Result<u32, String> {
		let h = self
			.b
			.get(self.i..self.i + 4)
			.ok_or_else(|| self.err("short \\u"))?;
		let v = u32::from_str_radix(std::str::from_utf8(h).unwrap_or("x"), 16)
			.map_err(|_| self.err("bad \\u"))?;
		self.i += 4;
		Ok(v)
	}
	fn string(&mut self) -> Result<String, String> {
		if self.b.get(self.i) != Some(&b'"') {
			return Err(self.err("expected string"));
		}
		self.i += 1;
		let mut out: Vec<u8> = Vec::new();
		loop {
			let c = *self
				.b
				.get(self.i)
				.ok_or_else(|| self.err("unterminated string"))?;
			self.i += 1;
			match c {
				b'"' => break,
				b'\\' => {
					let e = *self.b.get(self.i).ok_or_else(|| self.err("bad escape"))?;
					self.i += 1;
					let ch = match e {
						b'"' => '"',
						b'\\' => '\\',
						b'/' => '/',
						b'b' => '\u{8}',
						b'f' => '\u{c}',
						b'n' => '\n',
						b'r' => '\r',
						b't' => '\t',
						b'u' => {
							let hi = self.hex4()?;
							let cp = if (0xD800..0xDC00).contains(&hi)
								&& self.b[self.i..].starts_with(b"\\u")
							{
								self.i += 2;
								let lo = self.hex4()?;
								0x10000 + ((hi - 0xD800) << 10) + (lo.wrapping_sub(0xDC00) & 0x3FF)
							} else {
								hi
							};
							char::from_u32(cp).unwrap_or('\u{FFFD}')
						}
						_ => return Err(self.err("bad escape")),
					};
					let mut buf = [0u8; 4];
					out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
				}
				_ => out.push(c),
			}
		}
		String::from_utf8(out).map_err(|_| self.err("invalid UTF-8"))
	}
}

pub fn parse(text: &str) -> Result<J, String> {
	P {
		b: text.as_bytes(),
		i: 0,
	}
	.value()
}

/// A top-level string field of a small JSON object (`tokenizer_config.json`).
pub fn string_field(text: &str, key: &str) -> Option<String> {
	parse(text).ok()?.get(key)?.str().map(str::to_string)
}

pub struct Hf {
	/// Indexed by id; ids no entry names are `[PAD<id>]` fillers.
	pub vocab: Vec<String>,
	/// `"LEFT RIGHT"`, rank order.
	pub merges: Vec<String>,
	/// GGUF token types: 1 NORMAL, 3 CONTROL, 4 USER_DEFINED, 5 UNUSED.
	pub types: Vec<i32>,
}

pub fn load(path: &std::path::Path) -> Result<Hf, String> {
	let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
	let j = parse(&text)?;
	let model = j.get("model").ok_or("tokenizer.json: no model")?;
	if model.get("type").and_then(J::str) != Some("BPE") {
		return Err("tokenizer.json: only BPE models are supported".into());
	}
	let mut by_id: HashMap<u32, (String, i32)> = HashMap::new();
	if let Some(J::Obj(v)) = model.get("vocab") {
		for (k, id) in v {
			let id = id.num().ok_or("tokenizer.json: non-numeric vocab id")? as u32;
			by_id.insert(id, (k.clone(), 1));
		}
	} else {
		return Err("tokenizer.json: no model.vocab".into());
	}
	if let Some(J::Arr(a)) = j.get("added_tokens") {
		for t in a {
			let (Some(id), Some(c)) = (
				t.get("id").and_then(J::num),
				t.get("content").and_then(J::str),
			) else {
				continue;
			};
			let special = matches!(t.get("special"), Some(J::Bool(true)));
			by_id.insert(id as u32, (c.to_string(), if special { 3 } else { 4 }));
		}
	}
	let n = by_id.keys().max().map_or(0, |&m| m as usize + 1);
	let mut vocab = Vec::with_capacity(n);
	let mut types = Vec::with_capacity(n);
	for id in 0..n as u32 {
		match by_id.remove(&id) {
			Some((s, t)) => {
				vocab.push(s);
				types.push(t);
			}
			None => {
				vocab.push(format!("[PAD{id}]"));
				types.push(5);
			}
		}
	}
	let mut merges = Vec::new();
	if let Some(J::Arr(a)) = model.get("merges") {
		for m in a {
			match m {
				J::Str(s) => merges.push(s.clone()),
				J::Arr(p) if p.len() == 2 => {
					if let (Some(l), Some(r)) = (p[0].str(), p[1].str()) {
						merges.push(format!("{l} {r}"));
					}
				}
				_ => return Err("tokenizer.json: bad merge entry".into()),
			}
		}
	}
	Ok(Hf {
		vocab,
		merges,
		types,
	})
}
