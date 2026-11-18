//! A prefix cache for any model's state, under a RAM budget, with an
//! optional disk tier.
//!
//! A chat client resends the whole conversation every request; the cache
//! keeps the model's state after prompt prefixes so a request prefills only
//! what no kept state has consumed. Which states to keep, and where, is the
//! whole problem once memory is short, and this crate decides it the same
//! way for every backend (the GGUF model, the NPU engines, Flash-Next, and
//! models to come): a backend hands in its state as **parts** and gets the
//! same parts back.
//!
//! # A state as parts ([`Snap`], [`Layout`])
//!
//! - An **opaque** part (recurrent state, conv windows, counters) is kept as
//!   given. Its chunks are reference counted: an engine that hands the same
//!   chunk to several snapshots (Flash-Next's sealed KV segments) has it
//!   stored and counted once.
//! - A **row** part (a KV cache) is `streams` blocks of `rows` rows, one
//!   row per position. The cache cuts it into pages of [`PAGE`] rows and a
//!   page whose bytes equal the corresponding page of the state sharing the
//!   longest token prefix is that state's page. So every checkpoint of a
//!   conversation costs its recurrent state plus the rows since the
//!   previous one, not another copy of the whole KV (`page.rs`).
//!
//! # What to drop when the budget is full (the eviction score)
//!
//! LRU over whole states is wrong for hybrid models: a recurrent state
//! cannot be cut back, so a hit needs a state at exactly a prefix of the
//! prompt, and the value of keeping a state is the prefill it saves *over
//! the next state behind it*, not its length (Marconi, MLSys 2025; SegLen's
//! replay distance). Each entry is scored
//!
//! ```text
//! score = L * replay(a, len) / freed
//! L     = 0.5^(age / half_life) * (1 + ln(1 + hits)) * class weight
//!         * (1 if no kept state extends it, else beta)
//! ```
//!
//! with `a` the longest kept state that is a prefix of it, `replay` the
//! learned prefill time ([`CostModel`]: seconds per token growing with the
//! context, fitted from the prefills the cache sees) and `freed` the bytes
//! only this entry holds. The lowest score goes first. A conversation's
//! tip is worth one turn of replay and is used next; its earlier
//! checkpoints are worth something only for a branch (an edited message, a
//! retried turn) and weigh `beta`. Under pressure this thins a
//! conversation's chain of checkpoints from the dense end (the gap a
//! dropped state leaves is what the next one in line is charged) which
//! converges on checkpoints spaced ever wider towards the start, the shape
//! binomial checkpointing proves optimal for a fixed number of states.
//!
//! # Interval checkpoints
//!
//! A long prefill leaves a state every [`Config::interval`] tokens
//! ([`Cache::interval_cuts`]), so a later prompt that differs anywhere in a
//! long run replays at most to the next interval, not from the last turn
//! boundary. With paged rows each costs one recurrent state.
//!
//! # Disk tier, and flash wear (`disk.rs`)
//!
//! With a directory configured, shared prefixes (tool preambles, system
//! prompts) persist across restarts, and with spilling on, evicted states
//! go to disk instead of being dropped. Writes wear an SSD, so a write must
//! pay for itself: only chunks not on disk yet are written (content
//! hashed), each state is written once and never rewritten, and a state is
//! written only when the prefill seconds it is expected to save reach
//! [`DiskCfg::wear_s_per_gb`] per GB of new bytes, within a daily write
//! budget. Spilling is off by default.

mod cost;
mod disk;
mod hash;
mod key;
mod page;

pub use cost::CostModel;
pub use disk::DiskCfg;
pub use quant::snap::{Chunk, Data};

use key::Key;
use page::Stored;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Rows per page of a row part.
pub const PAGE: usize = 256;

/// How a part's bytes are laid out.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Layout {
	/// Kept as given (chunks shared by identity).
	#[default]
	Opaque,
	/// `streams` blocks of `rows` rows each, stream-major; row `i` belongs
	/// to position `i` (paged, shared by content).
	Rows { streams: usize, rows: usize },
}

/// A state after `pos` positions, as an engine makes and takes it: each
/// part's chunks concatenate to the part's bytes. `layout` has one entry
/// per part, or is empty (every part opaque).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Snap {
	pub pos: usize,
	pub parts: Vec<Vec<Chunk>>,
	pub layout: Vec<Layout>,
}

impl Snap {
	/// Every byte, shared chunks included.
	pub fn bytes(&self) -> usize {
		self.parts.iter().flatten().map(|c| c.len()).sum()
	}

	pub fn layout(&self, i: usize) -> Layout {
		self.layout.get(i).copied().unwrap_or_default()
	}

	/// Part `i` copied into `out`, which must be exactly its size.
	pub fn copy_part(&self, i: usize, out: &mut [u8]) -> Result<(), String> {
		let p = &self.parts[i];
		let n: usize = p.iter().map(|c| c.len()).sum();
		if n != out.len() {
			return Err(format!(
				"prefix cache: part {i} has {n} bytes, {} wanted",
				out.len()
			));
		}
		let mut at = 0;
		for c in p {
			out[at..at + c.len()].copy_from_slice(c.bytes());
			at += c.len();
		}
		Ok(())
	}
}

/// A state to store: its parts by value (chunks) or borrowed from the live
/// state (the cache copies only the pages it does not share).
pub struct SnapIn<'a> {
	pub pos: usize,
	pub parts: Vec<(Layout, Src<'a>)>,
}

pub enum Src<'a> {
	Chunks(Vec<Chunk>),
	Bytes(&'a [u8]),
}

impl From<Snap> for SnapIn<'static> {
	fn from(s: Snap) -> Self {
		let layout: Vec<Layout> = (0..s.parts.len()).map(|i| s.layout(i)).collect();
		SnapIn {
			pos: s.pos,
			parts: layout
				.into_iter()
				.zip(s.parts)
				.map(|(l, p)| (l, Src::Chunks(p)))
				.collect(),
		}
	}
}

/// Why a state was stored; sets its weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
	/// A prefix many conversations begin with (tool preamble, system turn).
	Shared,
	/// A conversation's checkpoint at a turn boundary.
	Turn,
	/// Every [`Config::interval`] tokens of a long prefill.
	Interval,
	/// Where a prompt parts from a cached one.
	Branch,
}

impl Class {
	fn code(self) -> u64 {
		self as u64
	}

	fn from_code(c: u64) -> Class {
		match c {
			0 => Class::Shared,
			1 => Class::Turn,
			2 => Class::Interval,
			_ => Class::Branch,
		}
	}
}

/// The cache's policy knobs. [`Config::new`] reads the `QWEN_CACHE_*`
/// overrides.
#[derive(Clone, Debug)]
pub struct Config {
	/// RAM budget in bytes; 0 disables the cache.
	pub budget: usize,
	/// Weight of a state some kept state extends (it serves branches only).
	pub beta: f64,
	/// Cache operations (lookups and stores) for a state's recency weight to
	/// halve.
	pub half_life: f64,
	/// Weight of [`Class::Shared`] states.
	pub shared_weight: f64,
	/// Tokens between interval checkpoints; 0 = none.
	pub interval: usize,
	/// The state is row parts alone (attention without recurrent layers):
	/// any common prefix with a cached state is a hit, the rows cut back.
	pub rewindable: bool,
	/// The cost model's priors until it has measured prefills.
	pub prefill_tok_s: f64,
}

fn env_f(name: &str) -> Option<f64> {
	std::env::var(name).ok()?.trim().parse().ok()
}

impl Config {
	/// Defaults for a backend prefilling about `prefill_tok_s` tokens per
	/// second, with `QWEN_CACHE_BETA`, `QWEN_CACHE_HALF_LIFE`,
	/// `QWEN_CACHE_SHARED_WEIGHT` and `QWEN_CACHE_INTERVAL` overrides.
	pub fn new(budget: usize, prefill_tok_s: f64) -> Config {
		Config {
			budget,
			beta: env_f("QWEN_CACHE_BETA").unwrap_or(0.3),
			half_life: env_f("QWEN_CACHE_HALF_LIFE").unwrap_or(64.0),
			shared_weight: env_f("QWEN_CACHE_SHARED_WEIGHT").unwrap_or(2.0),
			interval: env_f("QWEN_CACHE_INTERVAL").map_or(4096, |v| v as usize),
			rewindable: false,
			prefill_tok_s,
		}
	}
}

/// What [`Cache::lookup`] found: the longest usable state, `len` tokens of
/// the prompt.
pub struct Hit {
	pub len: usize,
	pub snap: Snap,
	/// Read back from the disk tier.
	pub from_disk: bool,
}

struct Entry {
	key: Key,
	st: Arc<Stored>,
	class: Class,
	used: u64,
	hits: u32,
	/// A copy is on disk (never written again).
	on_disk: bool,
}

struct Inner {
	entries: Vec<Entry>,
	tick: u64,
	/// States being stored ([`Cache::reserve`]).
	pending: Vec<(u64, Key)>,
	next_pending: u64,
}

/// A state about to be stored: [`Cache::lookup`]s that it would serve
/// better than what is kept wait for it (bounded). Released on drop.
pub struct Reservation<'a> {
	cache: &'a Cache,
	id: u64,
}

impl Drop for Reservation<'_> {
	fn drop(&mut self) {
		let mut g = self.cache.lock();
		g.pending.retain(|p| p.0 != self.id);
		self.cache.stored.notify_all();
	}
}

pub struct Cache {
	cfg: Config,
	inner: Mutex<Inner>,
	/// Signaled when a reservation ends.
	stored: std::sync::Condvar,
	cost: Mutex<CostModel>,
	disk: Option<Arc<disk::Disk>>,
}

/// An evicted state offered to the disk tier.
struct Spill {
	key: Key,
	st: Arc<Stored>,
	class: Class,
	weight: f64,
}

impl Cache {
	pub fn new(cfg: Config) -> Cache {
		let cost = CostModel::new(cfg.prefill_tok_s, 200_000.0);
		Cache {
			cfg,
			inner: Mutex::new(Inner {
				entries: Vec::new(),
				tick: 0,
				pending: Vec::new(),
				next_pending: 0,
			}),
			stored: std::sync::Condvar::new(),
			cost: Mutex::new(cost),
			disk: None,
		}
	}

	/// A disabled cache.
	pub fn off() -> Cache {
		Cache::new(Config::new(0, 100.0))
	}

	/// Add the disk tier (see `disk.rs`).
	pub fn with_disk(mut self, cfg: DiskCfg) -> std::io::Result<Cache> {
		self.disk = Some(disk::Disk::open(cfg)?);
		Ok(self)
	}

	pub fn enabled(&self) -> bool {
		self.cfg.budget > 0
	}

	pub fn config(&self) -> &Config {
		&self.cfg
	}

	fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
		self.inner.lock().unwrap_or_else(|e| e.into_inner())
	}

	fn cost(&self) -> CostModel {
		self.cost.lock().unwrap_or_else(|e| e.into_inner()).clone()
	}

	/// A prefill of positions `[a, p)` took `secs`: the cost model learns
	/// this machine's prefill speed from it.
	pub fn observe_prefill(&self, a: usize, p: usize, secs: f64) {
		self.cost
			.lock()
			.unwrap_or_else(|e| e.into_inner())
			.observe(a, p, secs);
	}

	/// Announce that the state after `tokens` is about to be [`insert`]ed
	/// (a request that stores its checkpoint after its response is out: the
	/// client's next request must not miss it). Hold the reservation until
	/// the insert is done.
	///
	/// [`insert`]: Cache::insert
	pub fn reserve(&self, tokens: &[u32]) -> Option<Reservation<'_>> {
		if !self.enabled() || tokens.is_empty() {
			return None;
		}
		let mut g = self.lock();
		g.next_pending += 1;
		let id = g.next_pending;
		g.pending.push((id, Key::new(tokens.to_vec())));
		Some(Reservation { cache: self, id })
	}

	/// Wait (at most `QWEN_CACHE_WAIT` s, default 30) while a reserved state
	/// is a proper prefix of `ids` longer than every kept one.
	fn await_pending(&self, ids: &[u32]) {
		let limit = std::time::Duration::from_secs_f64(env_f("QWEN_CACHE_WAIT").unwrap_or(30.0));
		let t = std::time::Instant::now();
		let mut g = self.lock();
		loop {
			let kept = g
				.entries
				.iter()
				.filter(|e| e.key.is_prefix_exact(ids))
				.map(|e| e.key.len())
				.max()
				.unwrap_or(0);
			let wait = g
				.pending
				.iter()
				.any(|(_, k)| k.len() > kept && k.is_prefix_exact(ids));
			let left = limit.saturating_sub(t.elapsed());
			if !wait || left.is_zero() {
				return;
			}
			g = self
				.stored
				.wait_timeout(g, left)
				.unwrap_or_else(|e| e.into_inner())
				.0;
		}
	}

	/// The longest state that is a **proper** prefix of `ids` (one token at
	/// least must be left to prefill, for logits), from RAM or, when that
	/// saves more than the read costs, from disk. A hit never consumes the
	/// state: it is shared, and restoring copies from it.
	pub fn lookup(&self, ids: &[u32]) -> Option<Hit> {
		if !self.enabled() {
			return None;
		}
		self.await_pending(ids);
		let cost = self.cost();
		let (ram_len, ram) = {
			let mut g = self.lock();
			g.tick += 1;
			let tick = g.tick;
			let best = g
				.entries
				.iter()
				.enumerate()
				.filter(|(_, e)| e.key.is_prefix_exact(ids))
				.max_by_key(|(_, e)| e.key.len())
				.map(|(i, _)| i);
			// a model without recurrent state can also cut a longer one back
			let rew = if self.cfg.rewindable {
				let q = Key::new(ids.to_vec());
				g.entries
					.iter()
					.enumerate()
					.map(|(i, e)| (i, e.key.common(&q).min(ids.len() - 1)))
					.filter(|&(i, m)| {
						m > 0
							&& best.is_none_or(|b| m > g.entries[b].key.len())
							&& g.entries[i].key.tokens[..m] == ids[..m]
					})
					.max_by_key(|&(_, m)| m)
			} else {
				None
			};
			match (best, rew) {
				(_, Some((i, m))) => {
					let e = &mut g.entries[i];
					e.used = tick;
					e.hits += 1;
					(m, Some(Arc::new(page::truncate(&e.st, m))))
				}
				(Some(i), None) => {
					let e = &mut g.entries[i];
					e.used = tick;
					e.hits += 1;
					(e.key.len(), Some(e.st.clone()))
				}
				(None, None) => (0, None),
			}
		};
		if let Some(d) = &self.disk {
			if let Some((len, id, bytes)) = d.best(ids) {
				if len > ram_len && cost.replay(ram_len, len) > 1.2 * cost.read(bytes) {
					let t = std::time::Instant::now();
					match d.load(id) {
						Ok((tokens, st, class)) => {
							self.cost
								.lock()
								.unwrap_or_else(|e| e.into_inner())
								.observe_read(bytes, t.elapsed().as_secs_f64());
							let key = Key::new(tokens);
							let st = match self.donor(&key) {
								Some(d) => page::reshare(st, &d),
								None => st,
							};
							let st = Arc::new(st);
							let snap = page::hand_out(&st).ok()?;
							self.put(key, st, class, true);
							return Some(Hit {
								len,
								snap,
								from_disk: true,
							});
						}
						Err(e) => eprintln!("prefix cache: disk read failed: {e}"),
					}
				}
			}
		}
		let st = ram?;
		match page::hand_out(&st) {
			Ok(snap) => Some(Hit {
				len: ram_len,
				snap,
				from_disk: false,
			}),
			Err(e) => {
				eprintln!("{e}");
				None
			}
		}
	}

	/// Store `snap`, the state after exactly `tokens`. Replaces a state for
	/// the same tokens, then evicts by score until the budget holds (see the
	/// crate docs); a state can lose to the ones already there. Pages equal
	/// to those of the state sharing the longest prefix are shared.
	pub fn insert(&self, tokens: Vec<u32>, snap: SnapIn<'_>, class: Class) {
		if !self.enabled() || tokens.is_empty() {
			return;
		}
		let key = Key::new(tokens);
		let donor = self.donor(&key);
		let st = match page::store(snap, donor.as_deref()) {
			Ok(s) => Arc::new(s),
			Err(e) => {
				eprintln!("{e}; not cached");
				return;
			}
		};
		if class == Class::Shared {
			if let Some(d) = self.disk.as_ref().filter(|d| !d.has(&key.tokens)) {
				let a = d.ancestor_len(&key);
				let value = self.cost().replay(a, key.len()) * self.cfg.shared_weight;
				d.offer(key.clone(), st.clone(), class, value);
			}
		}
		self.put(key, st, class, false);
	}

	/// The kept state sharing the longest token prefix with `key`: the one
	/// whose pages a new state most likely repeats.
	fn donor(&self, key: &Key) -> Option<Arc<Stored>> {
		let g = self.lock();
		g.entries
			.iter()
			.map(|e| (e.key.common(key), e))
			.filter(|(n, _)| *n > 0)
			.max_by_key(|(n, _)| *n)
			.map(|(_, e)| e.st.clone())
	}

	fn put(&self, key: Key, st: Arc<Stored>, class: Class, on_disk: bool) {
		let spills = {
			let mut g = self.lock();
			g.entries.retain(|e| e.key.tokens != key.tokens);
			g.tick += 1;
			let used = g.tick;
			let on_disk = on_disk || self.disk.as_ref().is_some_and(|d| d.has(&key.tokens));
			g.entries.push(Entry {
				key,
				st,
				class,
				used,
				hits: 0,
				on_disk,
			});
			self.evict(&mut g)
		};
		if let Some(d) = self.disk.as_ref().filter(|d| d.spill()) {
			let cost = self.cost();
			for s in spills {
				let a = d.ancestor_len(&s.key);
				let value = cost.replay(a, s.key.len()) * s.weight;
				d.offer(s.key, s.st, s.class, value);
			}
		}
	}

	/// The class weight of an entry, times its use count (no recency).
	fn weight(&self, e: &Entry, tip: bool) -> f64 {
		let w = if e.class == Class::Shared {
			self.cfg.shared_weight
		} else {
			1.0
		};
		w * (1.0 + (1.0 + e.hits as f64).ln()) * if tip { 1.0 } else { self.cfg.beta }
	}

	/// Drop entries by score until the budget holds; the dropped ones that
	/// are not on disk yet, for the disk tier.
	fn evict(&self, g: &mut Inner) -> Vec<Spill> {
		// references per chunk across entries
		let mut rc: HashMap<usize, (u32, usize)> = HashMap::new();
		for e in &g.entries {
			for c in e.st.chunks() {
				rc.entry(Arc::as_ptr(c) as *const u8 as usize)
					.or_insert((0, c.len()))
					.0 += 1;
			}
		}
		let mut held: usize = rc.values().map(|v| v.1).sum();
		if held <= self.cfg.budget {
			return Vec::new();
		}
		let n = g.entries.len();
		// anc[i]: entries that are proper prefixes of i, longest first
		let mut anc: Vec<Vec<usize>> = vec![Vec::new(); n];
		for i in 0..n {
			for j in 0..n {
				if g.entries[j].key.is_proper_prefix_of(&g.entries[i].key) {
					anc[i].push(j);
				}
			}
			anc[i].sort_by_key(|&j| std::cmp::Reverse(g.entries[j].key.len()));
		}
		// desc[i]: entries that i is a proper prefix of
		let mut desc: Vec<Vec<usize>> = vec![Vec::new(); n];
		for (k, a) in anc.iter().enumerate() {
			for &i in a {
				desc[i].push(k);
			}
		}
		let cost = self.cost();
		let mut live = vec![true; n];
		let mut out = Vec::new();
		let tick = g.tick;
		while held > self.cfg.budget {
			let mut best: Option<(f64, usize)> = None;
			for i in (0..n).filter(|&i| live[i]) {
				let e = &g.entries[i];
				let a = anc[i]
					.iter()
					.find(|&&j| live[j])
					.map_or(0, |&j| g.entries[j].key.len());
				let tip = !desc[i].iter().any(|&k| live[k]);
				let age = (tick - e.used) as f64;
				let r = 0.5f64.powf(age / self.cfg.half_life.max(1e-9));
				let l = r * self.weight(e, tip);
				let freed: usize =
					e.st.chunks()
						.filter_map(|c| rc.get(&(Arc::as_ptr(c) as *const u8 as usize)))
						.filter(|v| v.0 == 1)
						.map(|v| v.1)
						.sum();
				let s = l * cost.replay(a, e.key.len()) / freed.max(1 << 20) as f64;
				if best.is_none_or(|b| s < b.0) {
					best = Some((s, i));
				}
			}
			let Some((_, i)) = best else { break };
			live[i] = false;
			for c in g.entries[i].st.chunks() {
				if let Some(v) = rc.get_mut(&(Arc::as_ptr(c) as *const u8 as usize)) {
					v.0 -= 1;
					if v.0 == 0 {
						held -= v.1;
					}
				}
			}
			let e = &g.entries[i];
			if !e.on_disk {
				let tip = !desc[i].iter().any(|&k| live[k]);
				out.push(Spill {
					key: e.key.clone(),
					st: e.st.clone(),
					class: e.class,
					weight: self.weight(e, tip),
				});
			}
		}
		let mut k = 0;
		g.entries.retain(|_| {
			k += 1;
			live[k - 1]
		});
		out
	}

	/// Whether a state for exactly `tokens` is kept, in RAM or on disk.
	pub fn contains(&self, tokens: &[u32]) -> bool {
		self.lock().entries.iter().any(|e| e.key.tokens == tokens)
			|| self.disk.as_ref().is_some_and(|d| d.has(tokens))
	}

	/// The kept state that shares the longest run with `ids` and then
	/// differs from it: one no lookup can use: (shared tokens, its length,
	/// up to 16 of its tokens from the difference on). For the log line that
	/// says why a request missed, and for the branch checkpoint.
	pub fn diverging(&self, ids: &[u32]) -> Option<(usize, usize, Vec<u32>)> {
		let q = Key::new(ids.to_vec());
		let pick = |k: &Key| {
			let n = k.common(&q);
			(n < k.len() && n < ids.len())
				.then(|| (n, k.len(), k.tokens[n..k.len().min(n + 16)].to_vec()))
		};
		let ram = self
			.lock()
			.entries
			.iter()
			.filter_map(|e| pick(&e.key))
			.max_by_key(|d| d.0);
		let disk = self.disk.as_ref().and_then(|d| d.diverging(&q));
		[ram, disk].into_iter().flatten().max_by_key(|d| d.0)
	}

	/// Interval checkpoint positions in `(from, to)`: multiples of
	/// [`Config::interval`].
	pub fn interval_cuts(&self, from: usize, to: usize) -> Vec<usize> {
		let k = self.cfg.interval;
		if k == 0 || !self.enabled() {
			return Vec::new();
		}
		(from / k + 1..)
			.map(|i| i * k)
			.take_while(|&c| c < to)
			.collect()
	}

	/// States in RAM.
	pub fn len(&self) -> usize {
		self.lock().entries.len()
	}

	pub fn is_empty(&self) -> bool {
		self.len() == 0
	}

	/// Bytes the RAM states hold, shared chunks once.
	pub fn bytes(&self) -> usize {
		let g = self.lock();
		let mut seen = std::collections::HashSet::new();
		g.entries
			.iter()
			.flat_map(|e| e.st.chunks())
			.filter(|c| seen.insert(Arc::as_ptr(*c) as *const u8 as usize))
			.map(|c| c.len())
			.sum()
	}

	/// `N states, X GiB` (and the disk tier's), for the logs.
	pub fn summary(&self) -> String {
		let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
		let mut s = format!("{} states, {:.2} GiB", self.len(), gib(self.bytes() as u64));
		if let Some(d) = &self.disk {
			let (n, b) = d.stats();
			s += &format!("; disk {n}, {:.2} GiB", gib(b));
		}
		s
	}

	/// Wait for the disk tier's pending writes (tests, shutdown).
	pub fn flush(&self) {
		if let Some(d) = &self.disk {
			d.flush();
		}
	}
}

#[cfg(test)]
mod tests;
