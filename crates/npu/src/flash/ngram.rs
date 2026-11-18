//! The PLE layer's n-gram embedding: 8 hash heads each for the bigram and
//! the trigram ending at a token, 16 rows of 160 BF16 from one 320M-row table
//! (95 GB), concatenated to the token's 2560-wide embedding.
//!
//! The ids follow `Qwen4ExpTextNGramEmbedding`: the n-gram's tokens times
//! per-position odd multipliers, XORed, modulo a distinct prime per head,
//! plus the head's offset. Tokens before the start or across an `eos` read
//! as `eos`. The multipliers and primes are derived from the configuration
//! (`seed` 1234, layer 0 of the PLE layers) exactly as the reference does;
//! the checkpoint stores them too (`layer_multipliers`,
//! `ngram_heads_vocab_sizes`, `ngram_heads_offsets`), which the builder
//! checks against.
//!
//! The table is one row-major BF16 file (`flash_ngram.py`), read per token
//! with positional reads: 16 x 320 B per token.

use crate::ReadAt;

pub const N: usize = 3;
pub const HEADS_PER: usize = 8;
pub const HEADS: usize = (N - 1) * HEADS_PER;
pub const DIM: usize = 160;
const VOCAB_BASE: u64 = 20_000_000;
const SEED: u64 = 1234;

/// Multipliers, head sizes and offsets of one PLE layer.
#[derive(Debug, Clone, PartialEq)]
pub struct Hash {
	pub mult: [i64; N],
	pub sizes: [i64; HEADS],
	pub offs: [i64; HEADS],
}

fn splitmix64(v: u64) -> u64 {
	let v = v.wrapping_add(0x9E37_79B9_7F4A_7C15);
	let v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
	let v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
	v ^ (v >> 31)
}

fn is_prime(v: u64) -> bool {
	if v < 2 {
		return false;
	}
	if v.is_multiple_of(2) {
		return v == 2;
	}
	let mut d = 3;
	while d * d <= v {
		if v.is_multiple_of(d) {
			return false;
		}
		d += 2;
	}
	true
}

impl Hash {
	/// The hash of PLE layer `ple_index` (0 for the model's one PLE layer)
	/// for a token vocabulary of `vocab`.
	pub fn new(vocab: u64, ple_index: u64) -> Hash {
		let half = ((i64::MAX as u64) / vocab.max(1) / 2).max(1);
		let base = SEED.wrapping_add(10007u64.wrapping_mul(ple_index));
		let mut mult = [0i64; N];
		for (i, m) in mult.iter_mut().enumerate() {
			let v = base.wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i as u64 + 1));
			*m = (2 * (splitmix64(v) % half) + 1) as i64;
		}
		let mut sizes = [0i64; HEADS];
		let mut offs = [0i64; HEADS];
		let mut p = VOCAB_BASE - 1;
		// global head g takes the (g + 1)-th prime after VOCAB_BASE - 1
		for _ in 0..ple_index as usize * HEADS {
			p += 1;
			while !is_prime(p) {
				p += 1;
			}
		}
		let mut total = 0i64;
		for h in 0..HEADS {
			p += 1;
			while !is_prime(p) {
				p += 1;
			}
			sizes[h] = p as i64;
			offs[h] = total;
			total += p as i64;
		}
		Hash { mult, sizes, offs }
	}

	/// Rows of the table (incl. padding to 128) a layer's heads span.
	pub fn rows(&self) -> u64 {
		let t = (self.offs[HEADS - 1] + self.sizes[HEADS - 1]) as u64;
		t.div_ceil(128) * 128
	}

	/// The 16 row ids of the token whose n-gram tokens are `s[0]` (itself),
	/// `s[1]` (previous), `s[2]`.
	pub fn ids(&self, s: [u32; N]) -> [u64; HEADS] {
		let m = |i: usize| (s[i] as i64).wrapping_mul(self.mult[i]);
		let bi = m(0) ^ m(1);
		let tri = bi ^ m(2);
		let mut out = [0u64; HEADS];
		for h in 0..HEADS {
			let x = if h < HEADS_PER { bi } else { tri };
			out[h] = (x.rem_euclid(self.sizes[h]) + self.offs[h]) as u64;
		}
		out
	}
}

/// The n-gram context carried from token to token (part of the run state):
/// the two previous tokens and how many tokens since the last `eos` (capped
/// at 2), so a prefix cache snapshot restores it exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ctx {
	pub prev: [u32; N - 1],
	pub seg: u8,
}

impl Ctx {
	pub fn new(eos: u32) -> Ctx {
		Ctx {
			prev: [eos; N - 1],
			seg: 0,
		}
	}

	/// The n-gram tokens of `tok` (itself first), then advance past it.
	pub fn push(&mut self, tok: u32, eos: u32) -> [u32; N] {
		let s = [
			tok,
			if self.seg >= 1 { self.prev[0] } else { eos },
			if self.seg >= 2 { self.prev[1] } else { eos },
		];
		self.prev = [tok, self.prev[0]];
		self.seg = if tok == eos { 0 } else { (self.seg + 1).min(2) };
		s
	}
}

/// The table file: rows of `DIM` BF16.
pub struct Table {
	f: std::fs::File,
	path: std::path::PathBuf,
	pub rows: u64,
	/// [`Table::gather`]'s reader threads, started on first use
	readers: std::sync::Mutex<Readers>,
}

/// A part of a [`Table::gather`]: rows from `at` (in rows of the gather).
struct Job {
	rows: Vec<u64>,
	at: usize,
	done: std::sync::mpsc::Sender<(usize, Result<Vec<u16>, String>)>,
}

/// Persistent reader threads, each with its own handle of the table (one
/// a decode step started 16 threads that each opened the file: ~7800
/// short-lived threads in a few minutes of a Zed loop). They exit when the
/// table is dropped (the queue's sender with it).
struct Readers {
	tx: std::sync::mpsc::Sender<Job>,
	rx: std::sync::Arc<std::sync::Mutex<std::sync::mpsc::Receiver<Job>>>,
	n: usize,
}

/// A [`Table::gather`] in flight ([`Table::gather_async`]).
pub struct Pending {
	rx: std::sync::mpsc::Receiver<(usize, Result<Vec<u16>, String>)>,
	parts: usize,
	len: usize,
}

impl Pending {
	/// Wait for every part; the embeddings `[t][HEADS * DIM]`.
	pub fn wait(self) -> Result<Vec<u16>, String> {
		let mut out = vec![0u16; self.len * DIM];
		let mut err = Ok(());
		for _ in 0..self.parts {
			let (at, r) = self
				.rx
				.recv()
				.map_err(|_| "n-gram reader panicked".to_string())?;
			match r {
				Ok(v) => out[at * DIM..][..v.len()].copy_from_slice(&v),
				Err(e) => err = Err(e),
			}
		}
		err.map(|_| out)
	}
}

impl Table {
	pub fn open(path: &std::path::Path) -> Result<Table, String> {
		let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
		let len = f.metadata().map_err(|e| e.to_string())?.len();
		if len % (DIM as u64 * 2) != 0 {
			return Err(format!("{}: not rows of {DIM} BF16", path.display()));
		}
		let (tx, rx) = std::sync::mpsc::channel();
		Ok(Table {
			f,
			path: path.to_path_buf(),
			rows: len / (DIM as u64 * 2),
			readers: std::sync::Mutex::new(Readers {
				tx,
				rx: std::sync::Arc::new(std::sync::Mutex::new(rx)),
				n: 0,
			}),
		})
	}

	/// Row `r` as BF16 bits.
	pub fn row(&self, r: u64, out: &mut [u16; DIM]) -> Result<(), String> {
		Self::row_from(&self.f, self.rows, r, out)
	}

	fn row_from(f: &std::fs::File, rows: u64, r: u64, out: &mut [u16; DIM]) -> Result<(), String> {
		if r >= rows {
			return Err(format!("n-gram row {r} past the table's {rows}"));
		}
		let mut b = [0u8; DIM * 2];
		let mut done = 0;
		while done < b.len() {
			let n = f
				.seek_read(&mut b[done..], r * (DIM as u64 * 2) + done as u64)
				.map_err(|e| format!("n-gram table: {e}"))?;
			if n == 0 {
				return Err("n-gram table: short read".into());
			}
			done += n;
		}
		for (o, c) in out.iter_mut().zip(b.chunks_exact(2)) {
			*o = u16::from_le_bytes([c[0], c[1]]);
		}
		Ok(())
	}

	/// The embeddings `[t][HEADS * DIM]` (BF16 bits) of `ids` `[t][HEADS]`,
	/// read in `threads` parts (NVMe reads overlap), each by a reader with
	/// its own handle: Windows runs synchronous reads of one handle one at a
	/// time, so through a shared one the threads overlapped nothing (65536
	/// cold rows 6.5 s).
	pub fn gather(&self, ids: &[[u64; HEADS]], threads: usize) -> Result<Vec<u16>, String> {
		self.gather_async(ids, threads)?.wait()
	}

	/// [`Self::gather`], the reads queued to the readers; wait with
	/// [`Pending::wait`].
	pub fn gather_async(&self, ids: &[[u64; HEADS]], threads: usize) -> Result<Pending, String> {
		let threads = threads.max(1);
		let rows: Vec<u64> = ids.iter().flatten().copied().collect();
		let per = rows.len().div_ceil(threads).max(1);
		let (done, rx) = std::sync::mpsc::channel();
		let mut parts = 0;
		let mut rd = self.readers.lock().unwrap();
		while rd.n < threads {
			let (q, path, nrows) = (rd.rx.clone(), self.path.clone(), self.rows);
			std::thread::Builder::new()
				.name("ngram-read".into())
				.spawn(move || Self::reader(q, path, nrows))
				.map_err(|e| format!("n-gram reader: {e}"))?;
			rd.n += 1;
		}
		for (i, part) in rows.chunks(per).enumerate() {
			rd.tx
				.send(Job {
					rows: part.to_vec(),
					at: i * per,
					done: done.clone(),
				})
				.map_err(|_| "n-gram readers stopped".to_string())?;
			parts += 1;
		}
		Ok(Pending {
			rx,
			parts,
			len: rows.len(),
		})
	}

	/// One reader: takes jobs from the queue until the table is dropped.
	fn reader(
		q: std::sync::Arc<std::sync::Mutex<std::sync::mpsc::Receiver<Job>>>,
		path: std::path::PathBuf,
		nrows: u64,
	) {
		let f = std::fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()));
		loop {
			let Ok(job) = q.lock().unwrap().recv() else {
				return;
			};
			let r = f.as_ref().map_err(|e| e.clone()).and_then(|f| {
				let mut out = vec![0u16; job.rows.len() * DIM];
				let mut row = [0u16; DIM];
				for (&r, o) in job.rows.iter().zip(out.chunks_exact_mut(DIM)) {
					Self::row_from(f, nrows, r, &mut row)?;
					o.copy_from_slice(&row);
				}
				Ok(out)
			});
			// the gather may have been dropped (its result unused)
			let _ = job.done.send((job.at, r));
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn ctx_matches_the_reference_shift() {
		// shifts of [eos, eos] ++ ids, eos resetting the segment
		let eos = 9;
		let ids = [1u32, 2, 9, 3, 4, 5];
		let mut c = Ctx::new(eos);
		let got: Vec<[u32; 3]> = ids.iter().map(|&t| c.push(t, eos)).collect();
		assert_eq!(
			got,
			vec![
				[1, 9, 9],
				[2, 1, 9],
				[9, 2, 1],
				[3, 9, 9],
				[4, 3, 9],
				[5, 4, 3]
			]
		);
	}

	#[test]
	fn hash_of_layer_zero_is_stable() {
		let h = Hash::new(248320, 0);
		// the first heads take the primes after 20M
		assert_eq!(h.sizes[0], 20_000_003);
		assert!(h.mult.iter().all(|m| m % 2 == 1 && *m > 0));
		assert_eq!(h.offs[1], h.sizes[0]);
	}

	/// The ids of `flash_ref.py --dump` (`QWEN_TEST_NGRAM_IDS`, its
	/// `ngram_ids.i64`) for its prompt (`QWEN_TEST_IDS`, the `ids.txt` it
	/// read): `cargo test --release -p npu flash::ngram -- --ignored`.
	#[test]
	#[ignore]
	fn ids_match_the_torch_reference() {
		let file = |var: &str| std::env::var(var).unwrap_or_else(|_| panic!("set {var}"));
		let ids: Vec<u32> = std::fs::read_to_string(file("QWEN_TEST_IDS"))
			.unwrap()
			.split_whitespace()
			.map(|s| s.parse().unwrap())
			.collect();
		let want: Vec<u64> = std::fs::read(file("QWEN_TEST_NGRAM_IDS"))
			.unwrap()
			.chunks_exact(8)
			.map(|b| u64::from_le_bytes(b.try_into().unwrap()))
			.collect();
		let h = Hash::new(248320, 0);
		let mut c = Ctx::new(super::super::EOS);
		for (t, &tok) in ids.iter().enumerate() {
			let got = h.ids(c.push(tok, super::super::EOS));
			assert_eq!(&got[..], &want[t * HEADS..(t + 1) * HEADS], "token {t}");
		}
	}
}
