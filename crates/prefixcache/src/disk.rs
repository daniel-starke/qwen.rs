//! The disk tier: states in a directory, written so as to wear the SSD as
//! little as the savings allow.
//!
//! **Packs.** One file per state (`<id>.qpc`), written once, sequentially,
//! under a temporary name and renamed, and never modified. A pack holds
//! only the chunks no earlier pack has (content hash + length); the rest it
//! references in the packs that hold them. So a conversation's checkpoints
//! on disk cost, like in RAM, their recurrent state plus the pages since
//! the previous one. A pack whose state is evicted while later packs still
//! reference its chunks is renamed to `<id>.qpd` (a metadata write) and
//! deleted when the last reference goes.
//!
//! **Wear.** A write must pay for itself: the prefill seconds a state is
//! expected to save (the caller's score without recency) must reach
//! [`DiskCfg::wear_s_per_gb`] per GB of *new* bytes, and the writes of a
//! day stay within [`DiskCfg::day_bytes`] (counted across restarts in
//! `wear.txt`). Nothing is written through: a conversation's state reaches
//! the disk only when the RAM tier evicts it (`spill`), and a shared
//! prefix once. Reads are free, deletes are cheap (TRIM).
//!
//! **Format.** Chunk bytes, then the header (magic, model tag, class,
//! position, tokens, per part its layout and chunk references), then the
//! header's offset and an end magic. A file without a valid end is a crash
//! mid-write and is deleted at startup.

use crate::key::Key;
use crate::page::{Part, Stored};
use crate::{hash, Class, Data, Layout};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Condvar, Mutex, Weak};

/// The disk tier's settings.
#[derive(Clone, Debug)]
pub struct DiskCfg {
	pub dir: PathBuf,
	/// Identifies the model: a pack of another model is left alone.
	pub tag: String,
	/// Bytes the packs may take.
	pub budget: u64,
	/// Evicted states go to disk (else only shared prefixes are written).
	pub spill: bool,
	/// Prefill seconds a write must save per GB of new bytes.
	pub wear_s_per_gb: f64,
	/// Bytes written per day at most.
	pub day_bytes: u64,
}

impl DiskCfg {
	/// Defaults (spilling off), with `QWEN_CACHE_WEAR_S_PER_GB` and
	/// `QWEN_CACHE_WRITE_GB_DAY` overrides.
	pub fn new(dir: &Path, tag: String, budget: u64) -> DiskCfg {
		let f = |n: &str| {
			std::env::var(n)
				.ok()
				.and_then(|v| v.trim().parse::<f64>().ok())
		};
		DiskCfg {
			dir: dir.to_path_buf(),
			tag,
			budget,
			spill: false,
			wear_s_per_gb: f("QWEN_CACHE_WEAR_S_PER_GB").unwrap_or(30.0),
			day_bytes: (f("QWEN_CACHE_WRITE_GB_DAY").unwrap_or(64.0) * 1e9) as u64,
		}
	}
}

const MAGIC: &[u8; 8] = b"QWPC0001";
const END: &[u8; 8] = b"QWPCEND1";
const LIVE: &str = "qpc";
const DEAD: &str = "qpd";

#[derive(Clone, Copy)]
struct CRef {
	pack: u64,
	off: u64,
	len: u64,
	h: (u64, u64),
}

struct Pack {
	/// `None`: dead, kept for the chunks other packs reference.
	key: Option<Key>,
	class: Class,
	pos: usize,
	parts: Vec<(Layout, usize, Vec<CRef>)>,
	file_len: u64,
	/// Unix seconds of the last write or read.
	used: u64,
	/// Other packs whose chunks this one references.
	deps: HashSet<u64>,
	/// Packs referencing this one's chunks.
	refs_in: usize,
}

impl Pack {
	fn own_bytes(&self, id: u64) -> u64 {
		self.parts
			.iter()
			.flat_map(|p| &p.2)
			.filter(|c| c.pack == id)
			.map(|c| c.len)
			.sum()
	}

	fn bytes(&self) -> usize {
		self.parts
			.iter()
			.flat_map(|p| &p.2)
			.map(|c| c.len as usize)
			.sum()
	}
}

struct State {
	packs: HashMap<u64, Pack>,
	/// (hash, len) -> (pack, offset)
	chunks: HashMap<(u64, u64, u64), (u64, u64)>,
	total: u64,
	day: u64,
	day_written: u64,
	pending: usize,
}

struct Job {
	key: Key,
	st: Arc<Stored>,
	class: Class,
	value: f64,
}

pub(crate) struct Disk {
	cfg: DiskCfg,
	st: Mutex<State>,
	tx: Mutex<mpsc::Sender<Job>>,
	idle: Condvar,
}

fn now() -> u64 {
	std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map_or(0, |d| d.as_secs())
}

fn bad(what: &str) -> std::io::Error {
	std::io::Error::new(
		std::io::ErrorKind::InvalidData,
		format!("prefix cache pack: {what}"),
	)
}

fn put(w: &mut impl Write, x: u64) -> std::io::Result<()> {
	w.write_all(&x.to_le_bytes())
}

fn get(r: &mut impl Read) -> std::io::Result<u64> {
	let mut b = [0u8; 8];
	r.read_exact(&mut b)?;
	Ok(u64::from_le_bytes(b))
}

fn get_n(r: &mut impl Read, max: u64) -> std::io::Result<usize> {
	let n = get(r)?;
	if n > max {
		return Err(bad("length out of range"));
	}
	Ok(n as usize)
}

/// A pack's header: (tag, class, pos, tokens, parts).
type Header = (
	String,
	Class,
	usize,
	Vec<u32>,
	Vec<(Layout, usize, Vec<CRef>)>,
);

fn read_header(path: &Path) -> std::io::Result<Header> {
	let mut f = fs::File::open(path)?;
	let len = f.metadata()?.len();
	if len < 16 {
		return Err(bad("truncated"));
	}
	f.seek(SeekFrom::End(-16))?;
	let at = get(&mut f)?;
	let mut end = [0u8; 8];
	f.read_exact(&mut end)?;
	if &end != END || at >= len - 16 {
		return Err(bad("no end (an interrupted write)"));
	}
	f.seek(SeekFrom::Start(at))?;
	let mut r = BufReader::new(f.take(len - 16 - at));
	let mut m = [0u8; 8];
	r.read_exact(&mut m)?;
	if &m != MAGIC {
		return Err(bad("magic"));
	}
	let mut tag = vec![0u8; get_n(&mut r, 1 << 16)?];
	r.read_exact(&mut tag)?;
	let class = Class::from_code(get(&mut r)?);
	let pos = get(&mut r)? as usize;
	let mut tb = vec![0u8; get_n(&mut r, 1 << 28)? * 4];
	r.read_exact(&mut tb)?;
	let tokens = tb
		.chunks_exact(4)
		.map(|c| u32::from_le_bytes(c.try_into().unwrap()))
		.collect();
	let np = get_n(&mut r, 1 << 20)?;
	let mut parts = Vec::with_capacity(np);
	for _ in 0..np {
		let kind = get(&mut r)?;
		let streams = get(&mut r)? as usize;
		let rows = get(&mut r)? as usize;
		let w = get(&mut r)? as usize;
		let layout = if kind == 0 {
			Layout::Opaque
		} else {
			Layout::Rows { streams, rows }
		};
		let nc = get_n(&mut r, 1 << 24)?;
		let mut cs = Vec::with_capacity(nc);
		for _ in 0..nc {
			cs.push(CRef {
				pack: get(&mut r)?,
				off: get(&mut r)?,
				len: get(&mut r)?,
				h: (get(&mut r)?, get(&mut r)?),
			});
		}
		parts.push((layout, w, cs));
	}
	Ok((
		String::from_utf8_lossy(&tag).into_owned(),
		class,
		pos,
		tokens,
		parts,
	))
}

impl Disk {
	pub fn open(cfg: DiskCfg) -> std::io::Result<Arc<Disk>> {
		fs::create_dir_all(&cfg.dir)?;
		let mut packs: HashMap<u64, Pack> = HashMap::new();
		for e in fs::read_dir(&cfg.dir)?.flatten() {
			let p = e.path();
			let ext = p.extension().and_then(|x| x.to_str()).unwrap_or("");
			if p.to_string_lossy().ends_with(".qpc.tmp") {
				let _ = fs::remove_file(&p);
				continue;
			}
			if ext != LIVE && ext != DEAD {
				continue;
			}
			let Some(id) = p
				.file_stem()
				.and_then(|s| s.to_str())
				.and_then(|s| u64::from_str_radix(s, 16).ok())
			else {
				continue;
			};
			match read_header(&p) {
				Ok((tag, class, pos, tokens, parts)) => {
					if tag != cfg.tag {
						continue;
					}
					let meta = e.metadata()?;
					let used = meta
						.modified()
						.ok()
						.and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
						.map_or(0, |d| d.as_secs());
					let deps = parts
						.iter()
						.flat_map(|p| &p.2)
						.map(|c| c.pack)
						.filter(|&q| q != id)
						.collect();
					packs.insert(
						id,
						Pack {
							key: (ext == LIVE).then(|| Key::new(tokens)),
							class,
							pos,
							parts,
							file_len: meta.len(),
							used,
							deps,
							refs_in: 0,
						},
					);
				}
				Err(err) => {
					let gone = fs::remove_file(&p).is_ok();
					eprintln!(
						"prefix cache dir: {} unreadable ({err}), {}",
						p.display(),
						if gone { "removed" } else { "skipped" }
					);
				}
			}
		}
		// a pack whose referenced pack is gone cannot be read: drop it (and
		// so on, until every reference resolves)
		loop {
			let broken: Vec<u64> = packs
				.iter()
				.filter(|(_, p)| p.deps.iter().any(|d| !packs.contains_key(d)))
				.map(|(&id, _)| id)
				.collect();
			if broken.is_empty() {
				break;
			}
			for id in broken {
				let p = packs.remove(&id).unwrap();
				let _ = fs::remove_file(path_of(&cfg.dir, id, p.key.is_some()));
			}
		}
		let deps: Vec<u64> = packs
			.values()
			.flat_map(|p| p.deps.iter().copied())
			.collect();
		for d in deps {
			packs.get_mut(&d).unwrap().refs_in += 1;
		}
		let mut st = State {
			packs,
			chunks: HashMap::new(),
			total: 0,
			day: now() / 86400,
			day_written: 0,
			pending: 0,
		};
		let dead: Vec<u64> = st
			.packs
			.iter()
			.filter(|(_, p)| p.key.is_none() && p.refs_in == 0)
			.map(|(&id, _)| id)
			.collect();
		for id in dead {
			delete(&cfg.dir, &mut st, id);
		}
		for (&id, p) in &st.packs {
			for c in p.parts.iter().flat_map(|p| &p.2).filter(|c| c.pack == id) {
				st.chunks.insert((c.h.0, c.h.1, c.len), (id, c.off));
			}
		}
		st.total = st.packs.values().map(|p| p.file_len).sum();
		if let Ok(s) = fs::read_to_string(cfg.dir.join("wear.txt")) {
			let v: Vec<u64> = s
				.split_whitespace()
				.filter_map(|x| x.parse().ok())
				.collect();
			if let [day, bytes] = v[..] {
				if day == st.day {
					st.day_written = bytes;
				}
			}
		}
		let (tx, rx) = mpsc::channel::<Job>();
		let disk = Arc::new(Disk {
			cfg,
			st: Mutex::new(st),
			tx: Mutex::new(tx),
			idle: Condvar::new(),
		});
		let weak: Weak<Disk> = Arc::downgrade(&disk);
		std::thread::Builder::new()
			.name("prefix-cache-disk".into())
			.spawn(move || {
				while let Ok(job) = rx.recv() {
					let Some(d) = weak.upgrade() else { break };
					if let Err(e) = d.write(job) {
						eprintln!("prefix cache dir: write failed: {e}");
					}
					let mut g = d.lock();
					g.pending -= 1;
					d.idle.notify_all();
				}
			})?;
		Ok(disk)
	}

	fn lock(&self) -> std::sync::MutexGuard<'_, State> {
		self.st.lock().unwrap_or_else(|e| e.into_inner())
	}

	pub fn spill(&self) -> bool {
		self.cfg.spill
	}

	/// The longest live pack that is a proper prefix of `ids`: (its
	/// length, id, bytes to read).
	pub fn best(&self, ids: &[u32]) -> Option<(usize, u64, usize)> {
		let g = self.lock();
		g.packs
			.iter()
			.filter_map(|(&id, p)| {
				p.key
					.as_ref()
					.filter(|k| k.is_prefix_exact(ids))
					.map(|k| (k.len(), id, p.bytes()))
			})
			.max_by_key(|x| x.0)
	}

	pub fn has(&self, tokens: &[u32]) -> bool {
		self.lock()
			.packs
			.values()
			.any(|p| p.key.as_ref().is_some_and(|k| k.tokens == tokens))
	}

	/// Length of the longest live pack that is a proper prefix of `key`.
	pub fn ancestor_len(&self, key: &Key) -> usize {
		self.lock()
			.packs
			.values()
			.filter_map(|p| p.key.as_ref())
			.filter(|k| k.is_proper_prefix_of(key))
			.map(|k| k.len())
			.max()
			.unwrap_or(0)
	}

	/// As [`crate::Cache::diverging`], over the packs.
	pub fn diverging(&self, q: &Key) -> Option<(usize, usize, Vec<u32>)> {
		self.lock()
			.packs
			.values()
			.filter_map(|p| p.key.as_ref())
			.filter_map(|k| {
				let n = k.common(q);
				(n < k.len() && n < q.len())
					.then(|| (n, k.len(), k.tokens[n..k.len().min(n + 16)].to_vec()))
			})
			.max_by_key(|d| d.0)
	}

	/// (live packs, bytes on disk)
	pub fn stats(&self) -> (usize, u64) {
		let g = self.lock();
		(
			g.packs.values().filter(|p| p.key.is_some()).count(),
			g.total,
		)
	}

	/// A pack's state: (tokens, the stored state, class).
	pub fn load(&self, id: u64) -> Result<(Vec<u32>, Stored, Class), String> {
		let (tokens, class, pos, parts, files) = {
			let mut g = self.lock();
			let p = g.packs.get_mut(&id).ok_or("pack gone")?;
			p.used = now();
			let tokens = p.key.as_ref().ok_or("pack evicted")?.tokens.clone();
			let files: HashMap<u64, PathBuf> = p
				.parts
				.iter()
				.flat_map(|p| &p.2)
				.map(|c| c.pack)
				.collect::<HashSet<_>>()
				.into_iter()
				.map(|q| (q, path_of(&self.cfg.dir, q, g.packs[&q].key.is_some())))
				.collect();
			let p = &g.packs[&id];
			(tokens, p.class, p.pos, p.parts.clone(), files)
		};
		let mut open: HashMap<u64, fs::File> = HashMap::new();
		let mut out = Vec::with_capacity(parts.len());
		for (layout, w, cs) in parts {
			let mut chunks = Vec::with_capacity(cs.len());
			for c in cs {
				let f = match open.entry(c.pack) {
					std::collections::hash_map::Entry::Occupied(o) => o.into_mut(),
					std::collections::hash_map::Entry::Vacant(v) => {
						v.insert(fs::File::open(&files[&c.pack]).map_err(|e| e.to_string())?)
					}
				};
				let mut b = quant::snap::try_vec(c.len as usize, 0u8)?;
				f.seek(SeekFrom::Start(c.off))
					.and_then(|_| f.read_exact(&mut b))
					.map_err(|e| e.to_string())?;
				chunks.push(Arc::new(Data::U8(b)));
			}
			out.push(Part { layout, w, chunks });
		}
		Ok((tokens, Stored { pos, parts: out }, class))
	}

	/// Queue `st` for writing, if it pays for its wear (decided in the
	/// writer, which hashes the chunks). At most two jobs wait: past that
	/// the states are dropped as a cache without a disk would.
	pub fn offer(&self, key: Key, st: Arc<Stored>, class: Class, value: f64) {
		{
			let mut g = self.lock();
			if g.pending >= 2 {
				return;
			}
			g.pending += 1;
		}
		let job = Job {
			key,
			st,
			class,
			value,
		};
		if self
			.tx
			.lock()
			.unwrap_or_else(|e| e.into_inner())
			.send(job)
			.is_err()
		{
			self.lock().pending -= 1;
		}
	}

	/// Wait until every queued write is done.
	pub fn flush(&self) {
		let mut g = self.lock();
		while g.pending > 0 {
			g = self.idle.wait(g).unwrap_or_else(|e| e.into_inner());
		}
	}

	fn write(&self, job: Job) -> std::io::Result<()> {
		if self.has(&job.key.tokens) {
			return Ok(());
		}
		// what is not on disk yet
		let hashes: Vec<Vec<(u64, u64)>> = job
			.st
			.parts
			.iter()
			.map(|p| p.chunks.iter().map(|c| hash::h128(c.bytes())).collect())
			.collect();
		let new_bytes: u64 = {
			let g = self.lock();
			let mut seen = HashSet::new();
			job.st
				.parts
				.iter()
				.zip(&hashes)
				.flat_map(|(p, h)| p.chunks.iter().zip(h))
				.filter(|(c, h)| {
					let k = (h.0, h.1, c.len() as u64);
					!g.chunks.contains_key(&k) && seen.insert(k)
				})
				.map(|(c, _)| c.len() as u64)
				.sum()
		};
		let gb = new_bytes as f64 / 1e9;
		{
			let mut g = self.lock();
			let today = now() / 86400;
			if g.day != today {
				g.day = today;
				g.day_written = 0;
			}
			if job.value < self.cfg.wear_s_per_gb * gb {
				eprintln!(
					"prefix cache dir: {} tokens not written: saves ~{:.0} s for {gb:.2} GB (needs {:.0} s/GB)",
					job.key.len(),
					job.value,
					self.cfg.wear_s_per_gb
				);
				return Ok(());
			}
			if new_bytes > self.cfg.budget {
				return Ok(());
			}
			if g.day_written + new_bytes > self.cfg.day_bytes {
				eprintln!(
					"prefix cache dir: {} tokens not written: today's write budget ({:.0} GB) is spent",
					job.key.len(),
					self.cfg.day_bytes as f64 / 1e9
				);
				return Ok(());
			}
		}
		let id = {
			let g = self.lock();
			let mut id = now() << 20 ^ (std::process::id() as u64) << 4 ^ job.key.len() as u64;
			while g.packs.contains_key(&id) || path_of(&self.cfg.dir, id, true).exists() {
				id = id.wrapping_add(0x9e37_79b9);
			}
			id
		};
		let path = path_of(&self.cfg.dir, id, true);
		let tmp = path.with_extension("qpc.tmp");
		let t = std::time::Instant::now();
		let mut refs: Vec<(Layout, usize, Vec<CRef>)> = Vec::with_capacity(job.st.parts.len());
		let file_len = {
			let mut w = BufWriter::with_capacity(1 << 20, fs::File::create(&tmp)?);
			let mut off = 0u64;
			let mut mine: HashMap<(u64, u64, u64), u64> = HashMap::new();
			let known = self.lock().chunks.clone();
			for (p, hs) in job.st.parts.iter().zip(&hashes) {
				let mut cs = Vec::with_capacity(p.chunks.len());
				for (c, h) in p.chunks.iter().zip(hs) {
					let k = (h.0, h.1, c.len() as u64);
					let (pack, at) = if let Some(&(q, o)) = known.get(&k) {
						(q, o)
					} else if let Some(&o) = mine.get(&k) {
						(id, o)
					} else {
						w.write_all(c.bytes())?;
						mine.insert(k, off);
						off += c.len() as u64;
						(id, off - c.len() as u64)
					};
					cs.push(CRef {
						pack,
						off: at,
						len: c.len() as u64,
						h: *h,
					});
				}
				refs.push((p.layout, p.w, cs));
			}
			let at = off;
			w.write_all(MAGIC)?;
			put(&mut w, self.cfg.tag.len() as u64)?;
			w.write_all(self.cfg.tag.as_bytes())?;
			put(&mut w, job.class.code())?;
			put(&mut w, job.st.pos as u64)?;
			put(&mut w, job.key.len() as u64)?;
			for &x in &job.key.tokens {
				w.write_all(&x.to_le_bytes())?;
			}
			put(&mut w, refs.len() as u64)?;
			for (layout, pw, cs) in &refs {
				let (kind, s, r) = match *layout {
					Layout::Opaque => (0, 0, 0),
					Layout::Rows { streams, rows } => (1, streams, rows),
				};
				put(&mut w, kind)?;
				put(&mut w, s as u64)?;
				put(&mut w, r as u64)?;
				put(&mut w, *pw as u64)?;
				put(&mut w, cs.len() as u64)?;
				for c in cs {
					for x in [c.pack, c.off, c.len, c.h.0, c.h.1] {
						put(&mut w, x)?;
					}
				}
			}
			put(&mut w, at)?;
			w.write_all(END)?;
			w.flush()?;
			w.get_ref().metadata()?.len()
		};
		fs::rename(&tmp, &path)?;
		let deps: HashSet<u64> = refs
			.iter()
			.flat_map(|p| &p.2)
			.map(|c| c.pack)
			.filter(|&q| q != id)
			.collect();
		let mut g = self.lock();
		for d in &deps {
			if let Some(p) = g.packs.get_mut(d) {
				p.refs_in += 1;
			}
		}
		for c in refs.iter().flat_map(|p| &p.2).filter(|c| c.pack == id) {
			g.chunks.insert((c.h.0, c.h.1, c.len), (id, c.off));
		}
		g.packs.insert(
			id,
			Pack {
				key: Some(job.key),
				class: job.class,
				pos: job.st.pos,
				parts: refs,
				file_len,
				used: now(),
				deps,
				refs_in: 0,
			},
		);
		g.total += file_len;
		g.day_written += file_len;
		let _ = fs::write(
			self.cfg.dir.join("wear.txt"),
			format!("{} {}\n", g.day, g.day_written),
		);
		eprintln!(
			"prefix cache dir: wrote {} tokens ({:.2} GB new) in {:.1} s",
			g.packs[&id].key.as_ref().unwrap().len(),
			file_len as f64 / 1e9,
			t.elapsed().as_secs_f64()
		);
		self.enforce(&mut g, id);
		Ok(())
	}

	/// Evict packs until the directory fits its budget: lowest
	/// `tokens over the nearest pack before it x weight x recency / own
	/// bytes` first (a week's half-life), packs others reference (which
	/// free nothing) after those; never `keep`, the pack just written.
	fn enforce(&self, g: &mut State, keep: u64) {
		while g.total > self.cfg.budget {
			let t = now();
			let live: Vec<(u64, &Key)> = g
				.packs
				.iter()
				.filter_map(|(&id, p)| p.key.as_ref().map(|k| (id, k)))
				.collect();
			let score = |id: u64, k: &Key| {
				let p = &g.packs[&id];
				let a = live
					.iter()
					.filter(|(_, o)| o.is_proper_prefix_of(k))
					.map(|(_, o)| o.len())
					.max()
					.unwrap_or(0);
				let w = if p.class == Class::Shared { 2.0 } else { 1.0 };
				let r = 0.5f64.powf(t.saturating_sub(p.used) as f64 / (7.0 * 86400.0));
				let s = (k.len() - a) as f64 * w * r / p.own_bytes(id).max(1 << 20) as f64;
				// a pack others read frees nothing when evicted: last
				s + if p.refs_in > 0 { 1e20 } else { 0.0 }
			};
			let Some(victim) = live
				.iter()
				.filter(|&&(id, _)| id != keep)
				.map(|&(id, k)| (score(id, k), id))
				.min_by(|a, b| a.0.total_cmp(&b.0))
				.map(|x| x.1)
			else {
				break;
			};
			let p = g.packs.get_mut(&victim).unwrap();
			if p.refs_in == 0 {
				delete(&self.cfg.dir, g, victim);
			} else {
				p.key = None;
				let _ = fs::rename(
					path_of(&self.cfg.dir, victim, true),
					path_of(&self.cfg.dir, victim, false),
				);
			}
		}
	}
}

fn path_of(dir: &Path, id: u64, live: bool) -> PathBuf {
	dir.join(format!("{id:016x}.{}", if live { LIVE } else { DEAD }))
}

/// Delete a pack nothing references, and the dead packs only it kept.
fn delete(dir: &Path, g: &mut State, id: u64) {
	let Some(p) = g.packs.remove(&id) else { return };
	let _ = fs::remove_file(path_of(dir, id, p.key.is_some()));
	g.total = g.total.saturating_sub(p.file_len);
	g.chunks.retain(|_, v| v.0 != id);
	for d in p.deps {
		let free = match g.packs.get_mut(&d) {
			Some(q) => {
				q.refs_in -= 1;
				q.refs_in == 0 && q.key.is_none()
			}
			None => false,
		};
		if free {
			delete(dir, g, d);
		}
	}
}
