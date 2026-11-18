//! A state as the cache stores it: row parts cut into pages of [`PAGE`]
//! rows that checkpoints share.
//!
//! A row part is `streams` blocks of `rows` rows of `w` bytes each,
//! stream-major (one block per head or per layer, as an engine lays its KV
//! out). Page `j` holds rows `[j PAGE, (j + 1) PAGE)` of every stream,
//! stream-major within the page. A full page whose bytes equal the donor's
//! page `j` (the cached state sharing the longest token prefix) is the
//! donor's chunk: two checkpoints of one conversation hold its rows once.
//! The compare is on bytes, not on tokens, so sharing is exact by
//! construction: a state prefilled in other chunks, whose rows differ in
//! the last bits, simply shares nothing.

use crate::{Chunk, Data, Layout, Snap, SnapIn, Src, PAGE};
use std::borrow::Cow;
use std::sync::Arc;

pub(crate) struct Part {
	pub layout: Layout,
	/// Bytes per row of a row part (0 for an opaque one).
	pub w: usize,
	/// An opaque part's chunks as given; a row part's pages.
	pub chunks: Vec<Chunk>,
}

pub(crate) struct Stored {
	pub pos: usize,
	pub parts: Vec<Part>,
}

impl Stored {
	/// Every chunk with its identity, for the shared-bytes count.
	pub fn chunks(&self) -> impl Iterator<Item = &Chunk> {
		self.parts.iter().flat_map(|p| p.chunks.iter())
	}
}

fn rows_of(page: usize, rows: usize) -> usize {
	(rows - page * PAGE).min(PAGE)
}

fn check_rows(bytes: usize, streams: usize, rows: usize) -> Result<usize, String> {
	if streams == 0 || rows == 0 {
		return if bytes == 0 {
			Ok(0)
		} else {
			Err(format!("prefix cache: {bytes} bytes for no rows"))
		};
	}
	if !bytes.is_multiple_of(streams * rows) {
		return Err(format!(
			"prefix cache: {bytes} bytes are not {streams} streams of {rows} rows"
		));
	}
	Ok(bytes / (streams * rows))
}

/// `snap` in the stored form, sharing pages with `donor` where the bytes
/// agree.
pub(crate) fn store(snap: SnapIn<'_>, donor: Option<&Stored>) -> Result<Stored, String> {
	let mut parts = Vec::with_capacity(snap.parts.len());
	for (i, (layout, src)) in snap.parts.into_iter().enumerate() {
		let Layout::Rows { streams, rows } = layout else {
			let chunks = match src {
				Src::Chunks(c) => c,
				Src::Bytes(b) => vec![Arc::new(Data::U8(quant::snap::try_copy(b)?))],
			};
			parts.push(Part {
				layout,
				w: 0,
				chunks,
			});
			continue;
		};
		let flat: Cow<[u8]> = match &src {
			Src::Chunks(c) => quant::snap::concat(c),
			Src::Bytes(b) => Cow::Borrowed(b),
		};
		let w = check_rows(flat.len(), streams, rows)?;
		let d = donor.and_then(|d| d.parts.get(i)).filter(|p| {
			matches!(p.layout, Layout::Rows { streams: s, .. } if s == streams) && p.w == w
		});
		let d_rows = match d.map(|p| p.layout) {
			Some(Layout::Rows { rows, .. }) => rows,
			_ => 0,
		};
		let n_pages = if w == 0 { 0 } else { rows.div_ceil(PAGE) };
		let mut pages = Vec::with_capacity(n_pages);
		for j in 0..n_pages {
			let n = rows_of(j, rows);
			let at = |s: usize| s * rows * w + j * PAGE * w;
			if n == PAGE && d_rows >= (j + 1) * PAGE {
				let dp = &d.unwrap().chunks[j];
				let db = dp.bytes();
				if (0..streams)
					.all(|s| flat[at(s)..at(s) + PAGE * w] == db[s * PAGE * w..(s + 1) * PAGE * w])
				{
					pages.push(dp.clone());
					continue;
				}
			}
			let mut v = Vec::new();
			v.try_reserve_exact(streams * n * w).map_err(|_| {
				format!(
					"prefix cache: out of memory for {} KB",
					(streams * n * w) >> 10
				)
			})?;
			for s in 0..streams {
				v.extend_from_slice(&flat[at(s)..at(s) + n * w]);
			}
			pages.push(Arc::new(Data::U8(v)));
		}
		parts.push(Part {
			layout,
			w,
			chunks: pages,
		});
	}
	Ok(Stored {
		pos: snap.pos,
		parts,
	})
}

/// `st` (read back from disk: every chunk its own) with each chunk that
/// equals the donor's chunk at the same place replaced by the donor's, so
/// a state reloaded next to its conversation's other checkpoints shares
/// their pages (and Flash-Next's segments) again.
pub(crate) fn reshare(mut st: Stored, donor: &Stored) -> Stored {
	for (p, d) in st.parts.iter_mut().zip(&donor.parts) {
		if p.layout != d.layout
			&& !matches!((p.layout, d.layout), (Layout::Rows { streams: a, .. }, Layout::Rows { streams: b, .. }) if a == b && p.w == d.w)
		{
			continue;
		}
		for (c, dc) in p.chunks.iter_mut().zip(&d.chunks) {
			if c.len() == dc.len() && c.bytes() == dc.bytes() {
				*c = dc.clone();
			}
		}
	}
	st
}

/// The state as its engine made it: each part's chunks concatenate to the
/// part's bytes. A row part of one stream hands out its pages; one of
/// several streams is put back in stream-major order (one copy).
pub(crate) fn hand_out(st: &Stored) -> Result<Snap, String> {
	let mut parts = Vec::with_capacity(st.parts.len());
	let mut layout = Vec::with_capacity(st.parts.len());
	for p in &st.parts {
		layout.push(p.layout);
		match p.layout {
			Layout::Rows { streams, rows } if streams > 1 && rows > 0 => {
				let w = p.w;
				let mut v = quant::snap::try_vec(streams * rows * w, 0u8)?;
				for (j, pg) in p.chunks.iter().enumerate() {
					let n = rows_of(j, rows);
					let b = pg.bytes();
					for s in 0..streams {
						let to = s * rows * w + j * PAGE * w;
						v[to..to + n * w].copy_from_slice(&b[s * n * w..(s + 1) * n * w]);
					}
				}
				parts.push(vec![Arc::new(Data::U8(v))]);
			}
			_ => parts.push(p.chunks.clone()),
		}
	}
	Ok(Snap {
		pos: st.pos,
		parts,
		layout,
	})
}

/// `st` cut back to its first `m` positions, for a model whose state is
/// rows alone ([`crate::Config::rewindable`]): every row part keeps rows
/// `[0, m)` (whole pages shared, the partial one copied).
pub(crate) fn truncate(st: &Stored, m: usize) -> Stored {
	let parts = st
		.parts
		.iter()
		.map(|p| match p.layout {
			Layout::Rows { streams, rows } if m < rows => {
				let w = p.w;
				let full = m / PAGE;
				let mut chunks: Vec<Chunk> = p.chunks[..full].to_vec();
				let n = m - full * PAGE;
				if n > 0 {
					let old = rows_of(full, rows);
					let b = p.chunks[full].bytes();
					let mut v = Vec::with_capacity(streams * n * w);
					for s in 0..streams {
						v.extend_from_slice(&b[s * old * w..s * old * w + n * w]);
					}
					chunks.push(Arc::new(Data::U8(v)));
				}
				Part {
					layout: Layout::Rows { streams, rows: m },
					w,
					chunks,
				}
			}
			_ => Part {
				layout: p.layout,
				w: p.w,
				chunks: p.chunks.clone(),
			},
		})
		.collect();
	Stored { pos: m, parts }
}

#[cfg(test)]
mod tests {
	use super::*;

	/// `streams` streams of `rows` rows of 2 bytes, values from `f(s, r)`.
	fn rows(streams: usize, rows: usize, f: impl Fn(usize, usize) -> u8) -> Vec<u8> {
		let mut v = Vec::new();
		for s in 0..streams {
			for r in 0..rows {
				v.extend_from_slice(&[f(s, r), f(s, r) ^ 0x55]);
			}
		}
		v
	}

	fn snap(pos: usize, streams: usize, b: &[u8]) -> SnapIn<'_> {
		SnapIn {
			pos,
			parts: vec![
				(Layout::Opaque, Src::Bytes(b"meta")),
				(Layout::Rows { streams, rows: pos }, Src::Bytes(b)),
			],
		}
	}

	#[test]
	fn pages_round_trip_and_share_equal_bytes_only() {
		let f = |s: usize, r: usize| (s * 31 + r) as u8;
		let a = rows(3, PAGE * 2 + 10, f);
		let sa = store(snap(PAGE * 2 + 10, 3, &a), None).unwrap();
		assert_eq!(sa.parts[1].chunks.len(), 3);
		let out = hand_out(&sa).unwrap();
		assert_eq!(&quant::snap::concat(&out.parts[1])[..], &a[..]);
		assert_eq!(&quant::snap::concat(&out.parts[0])[..], b"meta");
		// a longer state with the same first rows shares its full pages
		let b = rows(3, PAGE * 3, f);
		let sb = store(snap(PAGE * 3, 3, &b), Some(&sa)).unwrap();
		let same = |x: &Chunk, y: &Chunk| Arc::ptr_eq(x, y);
		assert!(same(&sa.parts[1].chunks[0], &sb.parts[1].chunks[0]));
		assert!(same(&sa.parts[1].chunks[1], &sb.parts[1].chunks[1]));
		// (the donor's third page is partial: copied)
		assert!(!same(&sa.parts[1].chunks[2], &sb.parts[1].chunks[2]));
		assert_eq!(
			&quant::snap::concat(&hand_out(&sb).unwrap().parts[1])[..],
			&b[..]
		);
		// a row that differs in one stream breaks the share of that page
		let c = rows(3, PAGE * 2, |s, r| {
			if s == 2 && r == PAGE + 3 {
				0
			} else {
				f(s, r)
			}
		});
		let sc = store(snap(PAGE * 2, 3, &c), Some(&sb)).unwrap();
		assert!(same(&sb.parts[1].chunks[0], &sc.parts[1].chunks[0]));
		assert!(!same(&sb.parts[1].chunks[1], &sc.parts[1].chunks[1]));
		assert_eq!(
			&quant::snap::concat(&hand_out(&sc).unwrap().parts[1])[..],
			&c[..]
		);
	}

	#[test]
	fn truncation_keeps_the_first_rows_of_every_stream() {
		let f = |s: usize, r: usize| (s * 7 + r * 3) as u8;
		let a = rows(2, PAGE + 40, f);
		let st = store(snap(PAGE + 40, 2, &a), None).unwrap();
		for m in [5, PAGE, PAGE + 1] {
			let t = truncate(&st, m);
			assert_eq!(t.pos, m);
			let want = rows(2, m, f);
			assert_eq!(
				&quant::snap::concat(&hand_out(&t).unwrap().parts[1])[..],
				&want[..]
			);
		}
	}

	#[test]
	fn bad_row_shapes_are_errors() {
		let b = [0u8; 10];
		let s = SnapIn {
			pos: 3,
			parts: vec![(
				Layout::Rows {
					streams: 1,
					rows: 3,
				},
				Src::Bytes(&b),
			)],
		};
		assert!(store(s, None).is_err());
	}
}
