//! GGUF (GPT-Generated Unified Format) v3 parser.
#![allow(non_camel_case_types)] // ggml/GGUF type names like Q4_K_M are idiomatic
//!
//! Reference: llama.cpp `src/gguf.cpp`, `ggml/include/ggml.h` (`GGUF_*` types).
//!
//! This crate only parses the *header + KV metadata + tensor index* region of a
//! GGUF file. It never reads the (potentially multi-GB) weight data, so the
//! whole model file does not need to be materialized in memory. Parsing is
//! driven by a streaming [`std::io::Read`] so only the leading portion of the
//! file is consumed.

use std::io::{Cursor, Read};
use std::path::Path;

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF"

/// Elements per K-quant super-block (matches `ggml.h` `QK_K`).
pub const QK_K: usize = 256;

/// GGUF version we support (and the only one that carries the `qwen35` hybrid
/// arch metadata this project targets).
pub const GGUF_VERSION: u32 = 3;

/// Metadata value type table (GGUF spec / `ggml/include/gguf.h`).
///
/// Authoritative mapping (do NOT reorder):
/// 0 U8, 1 I8, 2 U16, 3 I16, 4 U32, 5 I32, 6 F32, 7 BOOL, 8 STRING, 9 ARRAY,
/// 10 U64, 11 I64, 12 F64.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MetadataValueType {
	U8 = 0,
	I8 = 1,
	U16 = 2,
	I16 = 3,
	U32 = 4,
	I32 = 5,
	F32 = 6,
	Bool = 7,
	String = 8,
	Array = 9,
	U64 = 10,
	I64 = 11,
	F64 = 12,
	// Anything outside 0..=12 is not a valid GGUF metadata value type.
	Unknown(u32),
}

impl MetadataValueType {
	fn from_u32(v: u32) -> Self {
		match v {
			0 => Self::U8,
			1 => Self::I8,
			2 => Self::U16,
			3 => Self::I16,
			4 => Self::U32,
			5 => Self::I32,
			6 => Self::F32,
			7 => Self::Bool,
			8 => Self::String,
			9 => Self::Array,
			10 => Self::U64,
			11 => Self::I64,
			12 => Self::F64,
			other => Self::Unknown(other),
		}
	}

	/// Discriminant code as stored in the GGUF file.
	pub fn code(self) -> u32 {
		match self {
			Self::U8 => 0,
			Self::I8 => 1,
			Self::U16 => 2,
			Self::I16 => 3,
			Self::U32 => 4,
			Self::I32 => 5,
			Self::F32 => 6,
			Self::Bool => 7,
			Self::String => 8,
			Self::Array => 9,
			Self::U64 => 10,
			Self::I64 => 11,
			Self::F64 => 12,
			Self::Unknown(v) => v,
		}
	}
}

/// ggml quantization type ids (`ggml.h`). The `-M`/`-S` variants are encoded
/// with the same id as their base K-quant type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
	F32 = 0,
	F16 = 1,
	Q4_0 = 2,
	Q4_1 = 3,
	Q5_0 = 6,
	Q5_1 = 7,
	Q8_0 = 8,
	Q8_1 = 9,
	Q2_K = 10,
	Q3_K = 11,
	Q4_K = 12,
	Q5_K = 13,
	Q6_K = 14,
	Q8_K = 15,
	// The remaining `ggml.h` ids, in their real numbering: an off-by-one from
	// `IQ1_S` onward parses the mmproj's BF16 tensors (id 30) as `Unknown(30)`.
	IQ2_XXS = 16,
	IQ2_XS = 17,
	IQ3_XXS = 18,
	IQ1_S = 19,
	IQ4_NL = 20,
	IQ3_S = 21,
	IQ2_S = 22,
	IQ4_XS = 23,
	I8 = 24,
	I16 = 25,
	I32 = 26,
	I64 = 27,
	F64 = 28,
	IQ1_M = 29,
	BF16 = 30,
	TQ1_0 = 34,
	TQ2_0 = 35,
	MXFP4 = 39,
	/// Internal alias for a `Q4_K` tensor known to come from a `_M` mix. It is
	/// never produced by [`GgmlType::from_u32`] (GGUF stores both under id 12)
	/// so its Rust discriminant sits outside the ggml range while
	/// [`GgmlType::code`] still reports 12.
	Q4_K_M = 1000,
	Unknown(u32),
}

impl GgmlType {
	pub fn from_u32(v: u32) -> Self {
		match v {
			0 => Self::F32,
			1 => Self::F16,
			2 => Self::Q4_0,
			3 => Self::Q4_1,
			6 => Self::Q5_0,
			7 => Self::Q5_1,
			8 => Self::Q8_0,
			9 => Self::Q8_1,
			10 => Self::Q2_K,
			11 => Self::Q3_K,
			12 => Self::Q4_K,
			13 => Self::Q5_K,
			14 => Self::Q6_K,
			15 => Self::Q8_K,
			16 => Self::IQ2_XXS,
			17 => Self::IQ2_XS,
			18 => Self::IQ3_XXS,
			19 => Self::IQ1_S,
			20 => Self::IQ4_NL,
			21 => Self::IQ3_S,
			22 => Self::IQ2_S,
			23 => Self::IQ4_XS,
			24 => Self::I8,
			25 => Self::I16,
			26 => Self::I32,
			27 => Self::I64,
			28 => Self::F64,
			29 => Self::IQ1_M,
			30 => Self::BF16,
			34 => Self::TQ1_0,
			35 => Self::TQ2_0,
			39 => Self::MXFP4,
			other => Self::Unknown(other),
		}
	}

	/// Discriminant code as stored in the GGUF file.
	pub fn code(self) -> u32 {
		match self {
			Self::F32 => 0,
			Self::F16 => 1,
			Self::Q4_0 => 2,
			Self::Q4_1 => 3,
			Self::Q5_0 => 6,
			Self::Q5_1 => 7,
			Self::Q8_0 => 8,
			Self::Q8_1 => 9,
			Self::Q2_K => 10,
			Self::Q3_K => 11,
			Self::Q4_K => 12,
			Self::Q5_K => 13,
			Self::Q6_K => 14,
			Self::Q8_K => 15,
			Self::IQ2_XXS => 16,
			Self::IQ2_XS => 17,
			Self::IQ3_XXS => 18,
			Self::IQ1_S => 19,
			Self::IQ4_NL => 20,
			Self::IQ3_S => 21,
			Self::IQ2_S => 22,
			Self::IQ4_XS => 23,
			Self::I8 => 24,
			Self::I16 => 25,
			Self::I32 => 26,
			Self::I64 => 27,
			Self::F64 => 28,
			Self::IQ1_M => 29,
			Self::BF16 => 30,
			Self::TQ1_0 => 34,
			Self::TQ2_0 => 35,
			Self::MXFP4 => 39,
			// GGUF stores a `Q4_K_M` mix under the plain `Q4_K` id.
			Self::Q4_K_M => 12,
			Self::Unknown(v) => v,
		}
	}

	/// Number of elements per quantization *block* for this type (1 for
	/// non-block types). Used to compute the byte size of a tensor.
	pub fn block_elements(self) -> usize {
		match self {
			Self::Q4_0 | Self::Q4_1 | Self::Q8_0 | Self::Q8_1 => 32,
			Self::Q2_K
			| Self::Q3_K
			| Self::Q4_K
			| Self::Q4_K_M
			| Self::Q5_K
			| Self::Q6_K
			| Self::Q8_K => QK_K,
			_ => 1,
		}
	}

	/// Byte length of a single quantization block (or of one element for
	/// non-block types).
	pub fn block_bytes(self) -> usize {
		match self {
			Self::F32 | Self::I32 => 4,
			Self::F16 | Self::BF16 | Self::I16 => 2,
			Self::I8 => 1,
			Self::I64 | Self::F64 => 8,
			Self::Q4_0 => 2 + 16,
			Self::Q4_1 => 2 + 2 + 16,
			Self::Q5_0 => 2 + QK_K / 8 + QK_K / 2,
			Self::Q5_1 => 2 + 2 + QK_K / 8 + QK_K / 2,
			Self::Q8_0 => 2 + 32,
			Self::Q8_1 => 2 + 2 + 32,
			Self::Q2_K => 2 + 2 + QK_K / 16 + QK_K / 4,
			Self::Q3_K => 2 + QK_K / 4 + QK_K / 8 + 12,
			Self::Q4_K | Self::Q4_K_M => 2 + 2 + 12 + QK_K / 2,
			Self::Q5_K => 2 + 2 + 12 + QK_K / 2 + QK_K / 8,
			Self::Q6_K => 2 + QK_K / 2 + QK_K / 4 + QK_K / 16,
			Self::Q8_K => 2 + 4 + QK_K + QK_K / 8,
			_ => 4,
		}
	}

	/// Canonical short name used when printing tensor lists.
	pub fn name(self) -> String {
		match self {
			Self::F32 => "F32".into(),
			Self::F16 => "F16".into(),
			Self::Q4_0 => "Q4_0".into(),
			Self::Q4_1 => "Q4_1".into(),
			Self::Q5_0 => "Q5_0".into(),
			Self::Q5_1 => "Q5_1".into(),
			Self::Q8_0 => "Q8_0".into(),
			Self::Q8_1 => "Q8_1".into(),
			Self::Q2_K => "Q2_K".into(),
			Self::Q3_K => "Q3_K".into(),
			Self::Q4_K => "Q4_K".into(),
			Self::Q5_K => "Q5_K".into(),
			Self::Q6_K => "Q6_K".into(),
			Self::Q8_K => "Q8_K".into(),
			Self::IQ2_XXS => "IQ2_XXS".into(),
			Self::IQ2_XS => "IQ2_XS".into(),
			Self::IQ3_XXS => "IQ3_XXS".into(),
			Self::IQ1_S => "IQ1_S".into(),
			Self::IQ4_NL => "IQ4_NL".into(),
			Self::IQ3_S => "IQ3_S".into(),
			Self::IQ2_S => "IQ2_S".into(),
			Self::IQ4_XS => "IQ4_XS".into(),
			Self::I8 => "I8".into(),
			Self::I16 => "I16".into(),
			Self::I32 => "I32".into(),
			Self::I64 => "I64".into(),
			Self::F64 => "F64".into(),
			Self::IQ1_M => "IQ1_M".into(),
			Self::BF16 => "BF16".into(),
			Self::TQ1_0 => "TQ1_0".into(),
			Self::TQ2_0 => "TQ2_0".into(),
			Self::MXFP4 => "MXFP4".into(),
			// Q4_K_M etc. are stored under the base id; report the richer name
			// when the id is the K-quant base for clarity.
			Self::Q4_K_M => "Q4_K_M".into(),
			Self::Unknown(v) => format!("UNKNOWN({v})"),
		}
	}
}

/// A single GGUF KV metadata entry.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
	U8(u8),
	I8(i8),
	U16(u16),
	I16(i16),
	U32(u32),
	I32(i32),
	U64(u64),
	I64(i64),
	F32(f32),
	F64(f64),
	Bool(bool),
	String(String),
	Array {
		ty: MetadataValueType,
		values: Vec<MetadataValue>,
	},
}

impl MetadataValue {
	/// Numeric value widened to `u64`, for the many `*.count` / `*.length`
	/// keys that GGUF writers emit as U32 but are free to widen.
	pub fn as_u64(&self) -> Option<u64> {
		Some(match self {
			Self::U8(v) => *v as u64,
			Self::U16(v) => *v as u64,
			Self::U32(v) => *v as u64,
			Self::U64(v) => *v,
			Self::I8(v) if *v >= 0 => *v as u64,
			Self::I16(v) if *v >= 0 => *v as u64,
			Self::I32(v) if *v >= 0 => *v as u64,
			Self::I64(v) if *v >= 0 => *v as u64,
			Self::Bool(v) => *v as u64,
			_ => return None,
		})
	}

	/// Numeric value widened to `f64` (integers included).
	pub fn as_f64(&self) -> Option<f64> {
		Some(match self {
			Self::F32(v) => *v as f64,
			Self::F64(v) => *v,
			Self::I8(v) => *v as f64,
			Self::I16(v) => *v as f64,
			Self::I32(v) => *v as f64,
			Self::I64(v) => *v as f64,
			Self::U8(v) => *v as f64,
			Self::U16(v) => *v as f64,
			Self::U32(v) => *v as f64,
			Self::U64(v) => *v as f64,
			_ => return None,
		})
	}

	pub fn as_bool(&self) -> Option<bool> {
		match self {
			Self::Bool(v) => Some(*v),
			_ => None,
		}
	}

	pub fn as_str(&self) -> Option<&str> {
		match self {
			Self::String(s) => Some(s.as_str()),
			_ => None,
		}
	}

	/// Render a compact, human-readable form for dumping.
	pub fn display(&self) -> String {
		match self {
			Self::String(s) => s.clone(),
			Self::Array { ty, values } => {
				let items: Vec<String> = values.iter().map(|v| v.display()).collect();
				format!("[{ty:?}; {}] {}", values.len(), items.join(", "))
			}
			other => format!("{other:?}"),
		}
	}
}

#[derive(Debug, Clone)]
pub struct MetadataEntry {
	pub key: String,
	pub value: MetadataValue,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
	pub name: String,
	pub dims: Vec<u64>,
	pub ggml_type: GgmlType,
	pub offset: u64,
}

impl TensorInfo {
	/// Total element count (`product(dims)`).
	pub fn num_elements(&self) -> usize {
		self.dims.iter().map(|d| *d as usize).product()
	}

	/// Byte length of this tensor's data region within the file.
	pub fn byte_length(&self) -> usize {
		let n = self.num_elements();
		let nblocks = n / self.ggml_type.block_elements().max(1);
		nblocks * self.ggml_type.block_bytes()
	}
}

#[derive(Debug, Clone)]
pub struct GgufFile {
	pub version: u32,
	pub n_tensors: u64,
	pub n_kv: u64,
	/// Offset (from start of file) where the tensor data section begins.
	pub data_offset: u64,
	pub metadata: Vec<MetadataEntry>,
	pub tensors: Vec<TensorInfo>,
}

/// Small endian-reading helper wrapping a `Read`, tracking byte position so we
/// can compute the aligned weight-data offset after the tensor index.
struct Reader<R: Read> {
	inner: R,
	pos: u64,
}

impl<R: Read> Reader<R> {
	fn new(inner: R) -> Self {
		Self { inner, pos: 0 }
	}

	fn u8(&mut self) -> std::io::Result<u8> {
		let mut b = [0u8; 1];
		self.inner.read_exact(&mut b)?;
		self.pos += 1;
		Ok(b[0])
	}

	fn u16(&mut self) -> std::io::Result<u16> {
		let mut b = [0u8; 2];
		self.inner.read_exact(&mut b)?;
		self.pos += 2;
		Ok(u16::from_le_bytes(b))
	}

	fn u32(&mut self) -> std::io::Result<u32> {
		let mut b = [0u8; 4];
		self.inner.read_exact(&mut b)?;
		self.pos += 4;
		Ok(u32::from_le_bytes(b))
	}

	fn u64(&mut self) -> std::io::Result<u64> {
		let mut b = [0u8; 8];
		self.inner.read_exact(&mut b)?;
		self.pos += 8;
		Ok(u64::from_le_bytes(b))
	}

	fn i8(&mut self) -> std::io::Result<i8> {
		Ok(self.u8()? as i8)
	}
	fn i16(&mut self) -> std::io::Result<i16> {
		Ok(self.u16()? as i16)
	}
	fn i32(&mut self) -> std::io::Result<i32> {
		Ok(self.u32()? as i32)
	}
	fn i64(&mut self) -> std::io::Result<i64> {
		Ok(self.u64()? as i64)
	}
	fn f32(&mut self) -> std::io::Result<f32> {
		Ok(f32::from_bits(self.u32()?))
	}
	fn f64(&mut self) -> std::io::Result<f64> {
		Ok(f64::from_bits(self.u64()?))
	}

	/// Read a GGUF string: u64 length followed by UTF-8 bytes.
	fn gguf_string(&mut self) -> std::io::Result<String> {
		let len = self.u64()? as usize;
		let mut buf = vec![0u8; len];
		self.inner.read_exact(&mut buf)?;
		self.pos += len as u64;
		Ok(String::from_utf8_lossy(&buf).into_owned())
	}
}

/// Read a single KV value of the given metadata type, advancing the reader by
/// exactly the number of bytes it occupies. Arrays recurse so nested arrays and
/// variable-length string elements are handled correctly (the GGUF spec
/// permits nested arrays).
fn read_value(r: &mut Reader<impl Read>, ty: MetadataValueType) -> std::io::Result<MetadataValue> {
	Ok(match ty {
		MetadataValueType::U8 => MetadataValue::U8(r.u8()?),
		MetadataValueType::I8 => MetadataValue::I8(r.i8()?),
		MetadataValueType::U16 => MetadataValue::U16(r.u16()?),
		MetadataValueType::I16 => MetadataValue::I16(r.i16()?),
		MetadataValueType::U32 => MetadataValue::U32(r.u32()?),
		MetadataValueType::I32 => MetadataValue::I32(r.i32()?),
		MetadataValueType::U64 => MetadataValue::U64(r.u64()?),
		MetadataValueType::I64 => MetadataValue::I64(r.i64()?),
		MetadataValueType::F32 => MetadataValue::F32(r.f32()?),
		MetadataValueType::F64 => MetadataValue::F64(r.f64()?),
		MetadataValueType::Bool => MetadataValue::Bool(r.u8()? != 0),
		MetadataValueType::String => MetadataValue::String(r.gguf_string()?),
		MetadataValueType::Array => {
			let elem_ty = MetadataValueType::from_u32(r.u32()?);
			let count = r.u64()? as usize;
			let mut values = Vec::with_capacity(count);
			for _ in 0..count {
				values.push(read_value(r, elem_ty)?);
			}
			MetadataValue::Array {
				ty: elem_ty,
				values,
			}
		}
		MetadataValueType::Unknown(v) => {
			return Err(std::io::Error::new(
				std::io::ErrorKind::InvalidData,
				format!("unsupported metadata value type {v}"),
			))
		}
	})
}

/// Parse a GGUF file from any reader. Stops after the tensor index (does not
/// read weight data).
pub fn parse_reader<R: Read>(mut r: R) -> std::io::Result<GgufFile> {
	let mut reader = Reader::new(&mut r);

	let magic = reader.u32()?;
	if magic != GGUF_MAGIC {
		return Err(std::io::Error::new(
			std::io::ErrorKind::InvalidData,
			format!("bad GGUF magic: {magic:#x}"),
		));
	}
	let version = reader.u32()?;
	if version != GGUF_VERSION {
		return Err(std::io::Error::new(
			std::io::ErrorKind::InvalidData,
			format!("unsupported GGUF version {version} (expected {GGUF_VERSION})"),
		));
	}
	let n_tensors = reader.u64()?;
	let n_kv = reader.u64()?;

	let mut metadata = Vec::with_capacity(n_kv as usize);
	for _ in 0..n_kv {
		let key = reader.gguf_string()?;
		let ty = MetadataValueType::from_u32(reader.u32()?);
		let value = read_value(&mut reader, ty)?;
		metadata.push(MetadataEntry { key, value });
	}

	// Determine alignment (default 32) from metadata if present.
	let alignment = metadata
		.iter()
		.find(|e| e.key == "general.alignment")
		.and_then(|e| match &e.value {
			MetadataValue::U32(v) => Some(*v as u64),
			MetadataValue::U64(v) => Some(*v),
			_ => None,
		})
		.unwrap_or(32);

	let mut tensors = Vec::with_capacity(n_tensors as usize);
	for _ in 0..n_tensors {
		let name = reader.gguf_string()?;
		let n_dims = reader.u32()? as usize;
		let mut dims = Vec::with_capacity(n_dims);
		for _ in 0..n_dims {
			dims.push(reader.u64()?);
		}
		let ggml_type = GgmlType::from_u32(reader.u32()?);
		let offset = reader.u64()?;
		tensors.push(TensorInfo {
			name,
			dims,
			ggml_type,
			offset,
		});
	}

	// Weight data begins immediately after the tensor index, aligned up to
	// `alignment`.
	let index_end = reader.pos;
	let data_offset = index_end.div_ceil(alignment) * alignment;

	Ok(GgufFile {
		version,
		n_tensors,
		n_kv,
		data_offset,
		metadata,
		tensors,
	})
}

/// Convenience loader that memory-maps is unnecessary; we just open and stream.
pub fn parse_file<P: AsRef<Path>>(path: P) -> std::io::Result<GgufFile> {
	let file = std::fs::File::open(path)?;
	let buf = std::io::BufReader::new(file);
	parse_reader(buf)
}

impl GgufFile {
	/// Read the raw (still-quantized) bytes of a tensor from the file at
	/// `path`. The caller is expected to `dequantize` them afterwards.
	pub fn read_tensor_bytes<P: AsRef<Path>>(
		&self,
		path: P,
		t: &TensorInfo,
	) -> std::io::Result<Vec<u8>> {
		use std::io::{Read, Seek, SeekFrom};
		let mut file = std::fs::File::open(path)?;
		let mut abs = self.data_offset + t.offset;
		if t.name == "token_embd.weight" {
			if let Ok(o) = std::env::var("QWEN_OFFSET") {
				abs = o.parse::<u64>().unwrap_or(abs);
			}
			if std::env::var("QWEN_DEBUG").is_ok() {
				eprintln!(
					"DBG read_tensor_bytes name={} data_offset={} offset={} abs={}",
					t.name, self.data_offset, t.offset, abs
				);
			}
		}
		file.seek(SeekFrom::Start(abs))?;
		let len = t.byte_length();
		let mut buf = vec![0u8; len];
		file.read_exact(&mut buf)?;
		Ok(buf)
	}
}

/// Parse from an in-memory buffer (used by tests and small files).
pub fn parse_bytes(buf: &[u8]) -> std::io::Result<GgufFile> {
	let cur = Cursor::new(buf);
	parse_reader(cur)
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Build a minimal but valid GGUF v3 byte stream in memory.
	fn build_sample() -> Vec<u8> {
		let mut b = Vec::new();
		b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
		b.extend_from_slice(&GGUF_VERSION.to_le_bytes());
		b.extend_from_slice(&2u64.to_le_bytes()); // n_tensors
		b.extend_from_slice(&4u64.to_le_bytes()); // n_kv

		// helper to push a GGUF string
		let push_str = |b: &mut Vec<u8>, s: &str| {
			b.extend_from_slice(&(s.len() as u64).to_le_bytes());
			b.extend_from_slice(s.as_bytes());
		};

		// KV 1: general.alignment = u32 32
		push_str(&mut b, "general.alignment");
		b.extend_from_slice(&(MetadataValueType::U32.code()).to_le_bytes());
		b.extend_from_slice(&32u32.to_le_bytes());

		// KV 2: qwen35.block_count = u32 2
		push_str(&mut b, "qwen35.block_count");
		b.extend_from_slice(&(MetadataValueType::U32.code()).to_le_bytes());
		b.extend_from_slice(&2u32.to_le_bytes());

		// KV 3: tokenizer.ggml.model = string "llama"
		push_str(&mut b, "tokenizer.ggml.model");
		b.extend_from_slice(&(MetadataValueType::String.code()).to_le_bytes());
		push_str(&mut b, "llama");

		// KV 4: an array of u32 [1,2,3]
		push_str(&mut b, "test.array");
		b.extend_from_slice(&(MetadataValueType::Array.code()).to_le_bytes());
		b.extend_from_slice(&(MetadataValueType::U32.code()).to_le_bytes()); // elem type
		b.extend_from_slice(&3u64.to_le_bytes()); // count
		for v in [1u32, 2, 3] {
			b.extend_from_slice(&v.to_le_bytes());
		}

		// Tensor 1
		push_str(&mut b, "tok_embeddings.weight");
		b.extend_from_slice(&2u32.to_le_bytes()); // n_dims
		b.extend_from_slice(&32000u64.to_le_bytes());
		b.extend_from_slice(&4096u64.to_le_bytes());
		b.extend_from_slice(&(GgmlType::Q8_0.code()).to_le_bytes());
		b.extend_from_slice(&0u64.to_le_bytes());

		// Tensor 2
		push_str(&mut b, "blk.0.attn.weight");
		b.extend_from_slice(&1u32.to_le_bytes());
		b.extend_from_slice(&10u64.to_le_bytes());
		b.extend_from_slice(&(GgmlType::F32.code()).to_le_bytes());
		b.extend_from_slice(&123456u64.to_le_bytes());

		b
	}

	#[test]
	fn parse_sample_metadata() {
		let f = parse_bytes(&build_sample()).unwrap();
		assert_eq!(f.version, 3);
		assert_eq!(f.n_tensors, 2);
		assert_eq!(f.n_kv, 4);

		let align = f
			.metadata
			.iter()
			.find(|e| e.key == "general.alignment")
			.unwrap();
		assert_eq!(align.value, MetadataValue::U32(32));

		let bc = f
			.metadata
			.iter()
			.find(|e| e.key == "qwen35.block_count")
			.unwrap();
		assert_eq!(bc.value, MetadataValue::U32(2));

		let model = f
			.metadata
			.iter()
			.find(|e| e.key == "tokenizer.ggml.model")
			.unwrap();
		assert_eq!(model.value, MetadataValue::String("llama".into()));

		let arr = f.metadata.iter().find(|e| e.key == "test.array").unwrap();
		match &arr.value {
			MetadataValue::Array { ty, values } => {
				assert_eq!(*ty, MetadataValueType::U32);
				assert_eq!(values.len(), 3);
				assert_eq!(values[0], MetadataValue::U32(1));
			}
			_ => panic!("expected array"),
		}
	}

	#[test]
	fn parse_sample_tensors() {
		let f = parse_bytes(&build_sample()).unwrap();
		assert_eq!(f.tensors[0].name, "tok_embeddings.weight");
		assert_eq!(f.tensors[0].dims, vec![32000, 4096]);
		assert_eq!(f.tensors[0].ggml_type, GgmlType::Q8_0);
		assert_eq!(f.tensors[1].name, "blk.0.attn.weight");
		assert_eq!(f.tensors[1].ggml_type, GgmlType::F32);
		assert_eq!(f.tensors[1].offset, 123456);
		assert_eq!(f.data_offset % 32, 0);
	}

	#[test]
	fn bad_magic_rejected() {
		let mut b = build_sample();
		b[0] = 0x00;
		assert!(parse_bytes(&b).is_err());
	}
}
