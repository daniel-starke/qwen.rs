//! Pieces of an engine's state snapshot (the server's prefix cache).
//!
//! A snapshot is a list of parts, each a list of [`Chunk`]s whose bytes
//! concatenate to the part. A chunk is reference counted, so an engine
//! whose cache grows by appending (the Flash-Next KV) hands the same sealed
//! segments to every snapshot of a conversation and keeps using them
//! itself: a checkpoint costs the rows since the last segment plus the
//! recurrent state, not another copy of the whole cache. The server counts
//! a chunk once however many cached snapshots hold it ([`Chunk`] pointer
//! identity).

use std::sync::Arc;

/// The data of a chunk, in the element type its engine keeps it in (so an
/// engine can adopt a chunk as its own storage again without a copy).
#[derive(Debug, PartialEq)]
pub enum Data {
	U8(Vec<u8>),
	U16(Vec<u16>),
	F32(Vec<f32>),
}

pub type Chunk = Arc<Data>;

impl Data {
	pub fn bytes(&self) -> &[u8] {
		// SAFETY: plain data reinterpreted as bytes
		unsafe {
			match self {
				Data::U8(v) => v,
				Data::U16(v) => std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2),
				Data::F32(v) => std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4),
			}
		}
	}

	pub fn len(&self) -> usize {
		self.bytes().len()
	}

	pub fn is_empty(&self) -> bool {
		self.len() == 0
	}
}

/// Element types a chunk can hold.
pub trait Elem: Copy + Default + Send + Sync + 'static {
	/// The chunk's values when it holds this type.
	fn of(d: &Data) -> Option<&[Self]>;
	fn wrap(v: Vec<Self>) -> Data;
	/// One value from its little-endian bytes (a chunk read back from disk).
	fn from_le(b: &[u8]) -> Self;
	const SIZE: usize;
}

impl Elem for u16 {
	fn of(d: &Data) -> Option<&[u16]> {
		match d {
			Data::U16(v) => Some(v),
			_ => None,
		}
	}
	fn wrap(v: Vec<u16>) -> Data {
		Data::U16(v)
	}
	fn from_le(b: &[u8]) -> u16 {
		u16::from_le_bytes([b[0], b[1]])
	}
	const SIZE: usize = 2;
}

impl Elem for f32 {
	fn of(d: &Data) -> Option<&[f32]> {
		match d {
			Data::F32(v) => Some(v),
			_ => None,
		}
	}
	fn wrap(v: Vec<f32>) -> Data {
		Data::F32(v)
	}
	fn from_le(b: &[u8]) -> f32 {
		f32::from_le_bytes([b[0], b[1], b[2], b[3]])
	}
	const SIZE: usize = 4;
}

/// A part of one chunk, from bytes.
pub fn part(b: Vec<u8>) -> Vec<Chunk> {
	vec![Arc::new(Data::U8(b))]
}

/// A part's bytes, concatenated (borrowed when it is one chunk).
pub fn concat(p: &[Chunk]) -> std::borrow::Cow<'_, [u8]> {
	match p {
		[c] => std::borrow::Cow::Borrowed(c.bytes()),
		_ => std::borrow::Cow::Owned(p.iter().flat_map(|c| c.bytes().iter().copied()).collect()),
	}
}

/// A `Vec` of `n` copies of `x`, or an error instead of an abort when the
/// memory is not there (a long context's state is gigabytes).
pub fn try_vec<T: Clone>(n: usize, x: T) -> Result<Vec<T>, String> {
	let mut v = Vec::new();
	v.try_reserve_exact(n).map_err(|_| {
		format!(
			"out of memory for {} MB",
			(n * std::mem::size_of::<T>()) >> 20
		)
	})?;
	v.resize(n, x);
	Ok(v)
}

/// `s` copied, or an error when the memory is not there.
pub fn try_copy<T: Copy>(s: &[T]) -> Result<Vec<T>, String> {
	let mut v = Vec::new();
	v.try_reserve_exact(s.len())
		.map_err(|_| format!("out of memory for {} MB", std::mem::size_of_val(s) >> 20))?;
	v.extend_from_slice(s);
	Ok(v)
}
