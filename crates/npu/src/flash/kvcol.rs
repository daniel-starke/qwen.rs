//! The attention layers' host cache as rows in sealed segments.
//!
//! The cache only grows by appending (a verify rollback cuts a few rows),
//! so every [`SEG`] rows are sealed into a reference-counted chunk
//! ([`quant::snap::Chunk`]) and never written again. A snapshot (the
//! server's prefix cache) takes the sealed chunks by reference and copies
//! only the open tail, and a restore adopts them back: the live cache and
//! every checkpoint of a conversation hold one copy of its rows. A
//! contiguous `Vec` held a second copy per checkpoint and doubled its
//! capacity on growth, which ran a 140k-token conversation out of memory
//! (an abort in `snapshot`). Every allocation here is fallible.

use quant::snap::{Chunk, Elem};
use std::sync::Arc;

/// Rows per sealed segment (a power of two: [`View::row`] shifts).
pub(super) const SEG: usize = 4096;
const SEG_SHIFT: u32 = SEG.trailing_zeros();

/// Rows of `w` values: sealed segments of [`SEG`] rows, then the open tail.
#[derive(Clone)]
pub(super) struct Col<T: Elem> {
	w: usize,
	segs: Vec<Chunk>,
	tail: Vec<T>,
}

/// A [`Col`]'s rows by index: the segments' slices, then the tail's.
pub(super) struct View<'a, T> {
	w: usize,
	s: Vec<&'a [T]>,
}

impl<T> View<'_, T> {
	#[inline]
	pub(super) fn row(&self, i: usize) -> &[T] {
		&self.s[i >> SEG_SHIFT][(i & (SEG - 1)) * self.w..][..self.w]
	}
}

fn oom(what: &str, n: usize, size: usize) -> String {
	format!(
		"kv cache: out of memory for {what} ({} MB)",
		(n * size) >> 20
	)
}

impl<T: Elem> Col<T> {
	pub(super) fn new(w: usize) -> Self {
		Col {
			w,
			segs: Vec::new(),
			tail: Vec::new(),
		}
	}

	pub(super) fn rows(&self) -> usize {
		self.segs.len() * SEG + self.tail.len() / self.w
	}

	fn seg(&self, s: usize) -> &[T] {
		T::of(&self.segs[s]).expect("kv segment of another type")
	}

	pub(super) fn view(&self) -> View<'_, T> {
		let mut s: Vec<&[T]> = (0..self.segs.len()).map(|i| self.seg(i)).collect();
		s.push(&self.tail);
		View { w: self.w, s }
	}

	/// The rows as contiguous slices in order (segments, then the tail).
	pub(super) fn slices(&self) -> Vec<&[T]> {
		let mut s = self.view().s;
		if s.last().is_some_and(|t| t.is_empty()) {
			s.pop();
		}
		s
	}

	/// Rows `[a, b)` into `out` (`(b - a) w` values).
	pub(super) fn copy_rows(&self, a: usize, b: usize, out: &mut [T]) {
		let w = self.w;
		let v = self.view();
		let mut r = a;
		while r < b {
			let s = r >> SEG_SHIFT;
			let o = r & (SEG - 1);
			let n = (SEG - o).min(b - r);
			out[(r - a) * w..(r - a + n) * w].copy_from_slice(&v.s[s][o * w..(o + n) * w]);
			r += n;
		}
	}

	/// Append rows (`vals.len()` a multiple of `w`).
	pub(super) fn push(&mut self, vals: &[T]) -> Result<(), String> {
		debug_assert!(vals.len().is_multiple_of(self.w));
		self.push_unaligned(vals)
	}

	/// Keep the first `rows` rows.
	pub(super) fn truncate(&mut self, rows: usize) -> Result<(), String> {
		let sealed = self.segs.len() * SEG;
		if rows >= sealed {
			self.tail.truncate((rows - sealed) * self.w);
			return Ok(());
		}
		// the cut is inside a sealed segment, which snapshots may share:
		// its kept rows become the open tail
		let s = rows / SEG;
		let keep = (rows - s * SEG) * self.w;
		let full = SEG * self.w;
		let mut t = Vec::new();
		t.try_reserve_exact(full)
			.map_err(|_| oom("a segment", full, T::SIZE))?;
		t.extend_from_slice(&self.seg(s)[..keep]);
		self.segs.truncate(s);
		self.tail = t;
		Ok(())
	}

	/// The rows as snapshot chunks: the sealed segments shared, the tail
	/// copied.
	pub(super) fn snapshot(&self) -> Result<Vec<Chunk>, String> {
		let mut c = self.segs.clone();
		if !self.tail.is_empty() {
			c.push(Arc::new(T::wrap(quant::snap::try_copy(&self.tail)?)));
		}
		Ok(c)
	}

	/// A column of `rows` rows of `w` values from [`Col::snapshot`]'s
	/// chunks, or from any chunking of the same bytes (a snapshot read back
	/// from disk): full segments of this type are adopted, the rest copied.
	pub(super) fn restore(w: usize, chunks: &[Chunk], rows: usize) -> Result<Self, String> {
		let bytes: usize = chunks.iter().map(|c| c.len()).sum();
		if bytes != rows * w * T::SIZE {
			return Err(format!("kv cache: {bytes} bytes for {rows} rows of {w}"));
		}
		let mut c = Col::new(w);
		for ch in chunks {
			match T::of(ch) {
				Some(v) if c.tail.is_empty() && v.len() == SEG * w => c.segs.push(ch.clone()),
				Some(v) => c.push(v)?,
				None => {
					let b = ch.bytes();
					if b.len() % T::SIZE != 0 {
						return Err("kv cache: a partial value".into());
					}
					// in pieces, so the conversion needs no second copy
					let mut tmp = Vec::new();
					for p in b.chunks(T::SIZE << 16) {
						tmp.clear();
						tmp.extend(p.chunks_exact(T::SIZE).map(T::from_le));
						c.push_unaligned(&tmp)?;
					}
				}
			}
		}
		if c.tail.len() % w != 0 {
			return Err("kv cache: a partial row".into());
		}
		Ok(c)
	}

	/// [`Col::push`] for values that need not end on a row (restore's
	/// conversion in pieces).
	fn push_unaligned(&mut self, mut vals: &[T]) -> Result<(), String> {
		let full = SEG * self.w;
		while !vals.is_empty() {
			if self.tail.capacity() < full {
				let n = full - self.tail.len();
				self.tail
					.try_reserve_exact(n)
					.map_err(|_| oom("a segment", full, T::SIZE))?;
			}
			let take = (full - self.tail.len()).min(vals.len());
			self.tail.extend_from_slice(&vals[..take]);
			vals = &vals[take..];
			if self.tail.len() == full {
				let t = std::mem::take(&mut self.tail);
				self.segs.push(Arc::new(T::wrap(t)));
			}
		}
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn col(rows: usize, w: usize) -> Col<u16> {
		let mut c = Col::new(w);
		let v: Vec<u16> = (0..rows * w).map(|i| i as u16).collect();
		c.push(&v[..w * 3]).unwrap();
		c.push(&v[w * 3..]).unwrap();
		c
	}

	#[test]
	fn rows_survive_segments_truncate_and_restore() {
		let w = 3;
		let rows = 2 * SEG + 17;
		let c = col(rows, w);
		assert_eq!(c.rows(), rows);
		assert_eq!(c.segs.len(), 2);
		let v = c.view();
		for i in [0, SEG - 1, SEG, 2 * SEG + 16] {
			assert_eq!(v.row(i)[0], (i * w) as u16);
		}
		let mut out = vec![0u16; 10 * w];
		c.copy_rows(SEG - 5, SEG + 5, &mut out);
		assert_eq!(out[0], ((SEG - 5) * w) as u16);
		assert_eq!(out[10 * w - 1], ((SEG + 5) * w - 1) as u16);
		// a snapshot shares the sealed segments
		let s = c.snapshot().unwrap();
		assert!(Arc::ptr_eq(&s[0], &c.segs[0]));
		let r = Col::<u16>::restore(w, &s, rows).unwrap();
		assert!(Arc::ptr_eq(&r.segs[1], &c.segs[1]));
		assert_eq!(r.view().row(rows - 1), c.view().row(rows - 1));
		// from bytes (a snapshot read back from disk)
		let b: Vec<u8> = s.iter().flat_map(|c| c.bytes().to_vec()).collect();
		let d = Col::<u16>::restore(w, &quant::snap::part(b), rows).unwrap();
		assert_eq!(d.segs.len(), 2);
		assert_eq!(d.view().row(SEG + 1), c.view().row(SEG + 1));
		// a cut inside a shared segment leaves the snapshot alone
		let mut t = r;
		t.truncate(SEG + 2).unwrap();
		assert_eq!(t.rows(), SEG + 2);
		assert_eq!(t.view().row(SEG + 1), c.view().row(SEG + 1));
		t.push(&[7, 7, 7]).unwrap();
		assert_eq!(t.view().row(SEG + 2), &[7, 7, 7]);
		assert_eq!(c.view().row(SEG + 2)[0], ((SEG + 2) * w) as u16);
		assert!(Col::<u16>::restore(w, &s, rows + 1).is_err());
	}
}
