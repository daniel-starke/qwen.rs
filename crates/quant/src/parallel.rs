//! Persistent thread-pool helpers for parallelizing GEMV / GEMM output rows.
//!
//! A process-global pool of worker threads is created **once** (not per matmul)
//! and every helper hands it a *shard function* that each worker calls with its
//! own shard index. The helper blocks until every shard has finished, so any
//! borrowed `y` / `f` outlives the jobs.
//!
//! ## Why the barrier is a spin-then-park loop, not a channel
//!
//! Decode issues ~575 dispatches per token (~9 GEMVs x 64 layers). The earlier
//! design sent one boxed job per worker down a per-worker `mpsc::channel` and
//! collected one ack per job, so every dispatch paid ~2*N condvar wake-ups on a
//! pool whose workers had been parked microseconds earlier; tens of
//! microseconds each, i.e. **milliseconds per token** on the critical path of a
//! kernel that is otherwise DRAM-bound.
//!
//! Workers now spin on a generation counter for a short window before parking,
//! so back-to-back dispatches (the decode case) never touch the futex. The
//! calling thread runs the last shard itself, which removes one wake-up and
//! keeps the pool at exactly `available_parallelism()` runnable threads instead
//! of one more than that.
//!
//! **Only one dispatch may be in flight at a time** (a mutex enforces it), so a
//! pool helper must never be called from inside another one.
//!
//! ## Width: fewer shards where more cores add nothing
//!
//! A dispatch runs on `width` shards: `n` unless a [`Width`] guard caps it.
//! Decode is DRAM-bound: on the target host 16 of the 24 threads decode as
//! fast as all 24 (5.19-5.23 vs 5.15-5.22 tok/s) on two thirds of the CPU
//! time, because a worker stalled on DRAM still counts as busy. Workers outside
//! the width are not woken: each parks on its own handle and the dispatcher
//! unparks only the workers it includes, so a narrowed dispatch costs the
//! excluded cores nothing, not even the spin window.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread::{self, Thread};

/// The shard function of the dispatch currently in flight. Published as a raw
/// fat pointer (data, vtable) through the release/acquire pair on `word`.
type ShardFn<'a> = &'a (dyn Fn(usize) + Sync);

/// How long a worker spins before parking. Decode dispatches land a few hundred
/// microseconds apart, so this window is what keeps the futex out of the hot
/// path; it costs idle CPU only for that window after the last dispatch.
/// Measured on the target host (Core Ultra 285K): long spins lose ~7 % of
/// decode because a worker that finishes early burns its core's power budget
/// while the others are still streaming; parking immediately loses the
/// back-to-back dispatch. ~2 k pause rounds (~20 us) is the plateau.
const SPIN_ROUNDS: u32 = 2_000;

/// The spin window workers use now ([`spin_window`]).
static SPIN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(SPIN_ROUNDS);

/// Chunks handed out per thread per dispatch. Enough slack to absorb the
/// P-core/E-core speed spread without making the cursor a contention point;
/// swept 4/8/16/32 on the target host, flat from 8 upward.
const CHUNKS_PER_THREAD: usize = 8;

/// Low bits of [`Pool::word`] that hold the dispatch width.
const WIDTH_BITS: u32 = 8;
const WIDTH_MASK: usize = (1 << WIDTH_BITS) - 1;

/// `QWEN_POOL_STATS=1`: account where the pool's threads spend their time
/// (see [`stats`]). Off by default; when off every probe is one relaxed load.
pub fn stats_on() -> bool {
	static S: OnceLock<bool> = OnceLock::new();
	*S.get_or_init(|| std::env::var("QWEN_POOL_STATS").is_ok_and(|v| v == "1"))
}

/// Nanosecond counters of [`stats`]: shard work, worker spin before a dispatch
/// arrived or the worker parked, the dispatcher's join spin, dispatches, parks.
static ST_WORK: AtomicU64 = AtomicU64::new(0);
static ST_SPIN: AtomicU64 = AtomicU64::new(0);
static ST_JOIN: AtomicU64 = AtomicU64::new(0);
static ST_DISP: AtomicU64 = AtomicU64::new(0);
static ST_PARK: AtomicU64 = AtomicU64::new(0);
/// Shard time of the dispatch in flight (dispatches are serialized).
static ST_CUR: AtomicU64 = AtomicU64::new(0);

/// Per call site of a pool helper: (shard ns, wall ns, dispatches).
type Sites = std::collections::HashMap<&'static std::panic::Location<'static>, (u64, u64, u64)>;
fn sites() -> &'static Mutex<Sites> {
	static M: OnceLock<Mutex<Sites>> = OnceLock::new();
	M.get_or_init(|| Mutex::new(Sites::new()))
}

/// Per call site since the last call, heaviest first: `(file:line, shard s,
/// wall s, dispatches)`. Empty unless `QWEN_POOL_STATS=1`.
pub fn stats_sites() -> Vec<(String, f64, f64, u64)> {
	let m = std::mem::take(&mut *sites().lock().unwrap());
	let mut v: Vec<_> = m
		.into_iter()
		.map(|(l, (w, t, n))| {
			(
				format!("{}:{}", l.file(), l.line()),
				w as f64 * 1e-9,
				t as f64 * 1e-9,
				n,
			)
		})
		.collect();
	v.sort_by(|a, b| b.1.total_cmp(&a.1));
	v
}

/// `(work_s, spin_s, join_s, dispatches, parks)` since the last call, summed
/// over every pool thread; all zero unless `QWEN_POOL_STATS=1`. Work is the
/// time inside shard functions (useful or not), spin the time a worker burned
/// waiting for the next dispatch before it parked, join the time the
/// dispatching thread spun waiting for the other shards.
pub fn stats() -> (f64, f64, f64, u64, u64) {
	let t = |a: &AtomicU64| a.swap(0, Ordering::Relaxed);
	(
		t(&ST_WORK) as f64 * 1e-9,
		t(&ST_SPIN) as f64 * 1e-9,
		t(&ST_JOIN) as f64 * 1e-9,
		t(&ST_DISP),
		t(&ST_PARK),
	)
}

#[inline]
fn now_if(on: bool) -> Option<std::time::Instant> {
	on.then(std::time::Instant::now)
}

#[inline]
fn add_since(c: &AtomicU64, t: Option<std::time::Instant>) {
	if let Some(t) = t {
		c.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
	}
}

#[inline]
fn add_work(t: Option<std::time::Instant>) {
	if let Some(t) = t {
		let ns = t.elapsed().as_nanos() as u64;
		ST_WORK.fetch_add(ns, Ordering::Relaxed);
		ST_CUR.fetch_add(ns, Ordering::Relaxed);
	}
}

/// `QWEN_UMWAIT`: how the pool's threads wait inside their spin window.
/// Default `0`: a `pause` loop. `2`: `UMONITOR` on the watched word +
/// `UMWAIT` in the C0.2 state (WAITPKG; the core idles at low power and
/// wakes when the line is written, ~0.25 us); `1` the C0.1 state. Opt-in
/// only: Flash-Next decode measured 1.5 % (C0.2) / 2.4 % (C0.1) slower
/// than the `pause` loop, same text. A CPU without WAITPKG (CPUID leaf 7,
/// ECX bit 5) or a target other than x86-64 always gets the `pause` loop.
fn umwait_mode() -> u32 {
	static M: OnceLock<u32> = OnceLock::new();
	*M.get_or_init(|| {
		#[cfg(target_arch = "x86_64")]
		let waitpkg = core::arch::x86_64::__cpuid_count(7, 0).ecx >> 5 & 1 == 1;
		#[cfg(not(target_arch = "x86_64"))]
		let waitpkg = false;
		let m = std::env::var("QWEN_UMWAIT")
			.ok()
			.and_then(|v| v.trim().parse().ok())
			.unwrap_or(0u32)
			.min(2);
		if waitpkg {
			m
		} else {
			0
		}
	})
}

/// Wait while `a` holds `v`, for up to `rounds` pause rounds; returns the
/// last value read (`v` when the window ran out).
#[inline]
fn wait_change(a: &AtomicUsize, v: usize, rounds: u32) -> usize {
	wait_cond(a, rounds, |w| w != v).unwrap_or(v)
}

/// Wait until `a` reads 0, for up to `rounds` pause rounds.
#[inline]
fn wait_until_zero(a: &AtomicUsize, rounds: u32) {
	let _ = wait_cond(a, rounds, |w| w == 0);
}

/// Wait until `done(a)` for up to `rounds` pause rounds ([`umwait_mode`]);
/// the value that satisfied it, or `None` when the window ran out.
#[inline]
fn wait_cond(a: &AtomicUsize, rounds: u32, done: impl Fn(usize) -> bool) -> Option<usize> {
	#[cfg(target_arch = "x86_64")]
	{
		let mode = umwait_mode();
		if mode != 0 {
			return umwait::wait(a, rounds, mode == 1, done);
		}
	}
	for _ in 0..rounds {
		let w = a.load(Ordering::Acquire);
		if done(w) {
			return Some(w);
		}
		std::hint::spin_loop();
	}
	None
}

#[cfg(target_arch = "x86_64")]
mod umwait {
	use core::arch::x86_64::_rdtsc;
	use std::sync::atomic::{AtomicUsize, Ordering};
	use std::sync::OnceLock;

	/// TSC ticks per pause round of the spin window (~10 ns), measured once.
	fn tsc_per_round() -> u64 {
		static T: OnceLock<u64> = OnceLock::new();
		*T.get_or_init(|| {
			let t0 = std::time::Instant::now();
			// SAFETY: RDTSC exists on every x86-64 CPU
			let c0 = unsafe { _rdtsc() };
			while t0.elapsed() < std::time::Duration::from_millis(2) {}
			let c1 = unsafe { _rdtsc() };
			let ns = t0.elapsed().as_nanos().max(1) as u64;
			((c1 - c0) * 10 / ns).max(1)
		})
	}

	/// [`super::wait_cond`] with `UMONITOR` / `UMWAIT` (C0.1 if `c01`, else
	/// C0.2); only called once [`super::umwait_mode`] saw WAITPKG.
	#[inline]
	pub(super) fn wait(
		a: &AtomicUsize,
		rounds: u32,
		c01: bool,
		done: impl Fn(usize) -> bool,
	) -> Option<usize> {
		// SAFETY: RDTSC exists on every x86-64 CPU
		let deadline = unsafe { _rdtsc() } + rounds as u64 * tsc_per_round();
		let ctrl = c01 as u32;
		loop {
			let w = a.load(Ordering::Acquire);
			if done(w) {
				return Some(w);
			}
			// SAFETY: as above
			if unsafe { _rdtsc() } >= deadline {
				return None;
			}
			// SAFETY: WAITPKG checked by the caller; UMONITOR only arms the
			// monitor on the line holding `a`, UMWAIT returns on a write to
			// it, the deadline, an interrupt or the OS's time limit
			unsafe {
				core::arch::asm!("umonitor {0}", in(reg) a.as_ptr(), options(nostack, preserves_flags));
				if !done(a.load(Ordering::Acquire)) {
					core::arch::asm!(
						"umwait {0:e}",
						in(reg) ctrl,
						in("eax") deadline as u32,
						in("edx") (deadline >> 32) as u32,
						options(nostack),
					);
				}
			}
		}
	}
}

struct Pool {
	/// Threads in the pool = worker threads + the calling thread.
	n: usize,
	/// `generation << WIDTH_BITS | width` of the latest dispatch. Bumped once
	/// per dispatch; a worker runs a shard when it is inside the width.
	word: AtomicUsize,
	/// Worker shards still running in the current dispatch.
	remaining: AtomicUsize,
	/// Fat pointer to the current [`ShardFn`], split into (data, vtable).
	task: [AtomicUsize; 2],
	/// Next unclaimed chunk of the current dispatch (see [`par_chunks`]).
	cursor: AtomicUsize,
	/// Per worker: set while it is parked or about to park.
	sleeping: Vec<AtomicBool>,
	/// Per worker, for `unpark`. Set before the pool is handed out.
	handles: OnceLock<Vec<Thread>>,
	/// Width cap of the next dispatches ([`Width`]); 0 = `n`.
	limit: AtomicUsize,
	/// Serializes dispatches (the pool holds exactly one task at a time).
	serial: Mutex<()>,
}

// `task` is only dereferenced between the `word` bump and the `remaining == 0`
// observation (a window the dispatcher blocks through) so the referent
// always outlives every read.
unsafe impl Sync for Pool {}

fn pool() -> &'static Pool {
	static P: OnceLock<&'static Pool> = OnceLock::new();
	P.get_or_init(|| {
		let n = std::env::var("QWEN_THREADS")
			.ok()
			.and_then(|s| s.parse::<usize>().ok())
			.unwrap_or_else(|| {
				std::thread::available_parallelism()
					.map(|n| n.get())
					.unwrap_or(1)
			})
			.clamp(1, 32);
		let p: &'static Pool = Box::leak(Box::new(Pool {
			n,
			word: AtomicUsize::new(0),
			remaining: AtomicUsize::new(0),
			task: [AtomicUsize::new(0), AtomicUsize::new(0)],
			cursor: AtomicUsize::new(0),
			sleeping: (0..n).map(|_| AtomicBool::new(false)).collect(),
			handles: OnceLock::new(),
			limit: AtomicUsize::new(0),
			serial: Mutex::new(()),
		}));
		// `n - 1` workers; the dispatching thread runs the remaining shard.
		let hs = (0..n - 1)
			.map(|id| thread::spawn(move || p.worker(id)).thread().clone())
			.collect();
		let _ = p.handles.set(hs);
		p
	})
}

impl Pool {
	fn worker(&'static self, id: usize) {
		let mut seen = 0usize;
		let mut spin = true;
		loop {
			seen = self.wait_for(id, seen, spin);
			// Worker `id` is in a dispatch of width `w` iff `id < w - 1`.
			if id + 1 < seen & WIDTH_MASK {
				let t = now_if(stats_on());
				self.run_shard(id);
				add_work(t);
				self.remaining.fetch_sub(1, Ordering::Release);
				spin = true;
			} else {
				// Left out: the next dispatch is likely narrow too, so do not
				// burn the spin window waiting for it.
				spin = false;
			}
		}
	}

	/// Block until `word` moves past `seen`; returns the new word. Parking is
	/// a Dekker pair with `dispatch`: the worker publishes `sleeping` then
	/// re-reads `word`, the dispatcher publishes `word` then reads `sleeping`
	/// (all `SeqCst`), so one of them always sees the other.
	fn wait_for(&self, id: usize, seen: usize, spin: bool) -> usize {
		if spin {
			let t = now_if(stats_on());
			let w = wait_change(&self.word, seen, SPIN.load(Ordering::Relaxed));
			add_since(&ST_SPIN, t);
			if w != seen {
				return w;
			}
		}
		if stats_on() {
			ST_PARK.fetch_add(1, Ordering::Relaxed);
		}
		loop {
			self.sleeping[id].store(true, Ordering::SeqCst);
			let w = self.word.load(Ordering::SeqCst);
			if w != seen {
				self.sleeping[id].store(false, Ordering::Relaxed);
				return w;
			}
			// An `unpark` that lands before this makes it return at once.
			thread::park();
		}
	}

	#[inline]
	fn run_shard(&self, id: usize) {
		let data = self.task[0].load(Ordering::Acquire);
		let vtable = self.task[1].load(Ordering::Acquire);
		// Reassemble the `&dyn Fn(usize) + Sync` published by `dispatch`.
		let f: ShardFn<'_> = unsafe { std::mem::transmute((data, vtable)) };
		f(id);
	}

	/// Shards the next dispatch gets.
	fn width(&self) -> usize {
		match self.limit.load(Ordering::Relaxed) {
			0 => self.n,
			l => l.min(self.n),
		}
	}

	/// Run shards `0..width` across the pool (shard `width - 1` on this
	/// thread) and return once every one has completed. `f` may borrow freely:
	/// this blocks. Helpers read `width` once and pass it in, so a [`Width`]
	/// change mid-helper cannot disagree with their chunking.
	#[track_caller]
	fn dispatch<F: Fn(usize) + Sync>(&'static self, width: usize, f: F) {
		let site = std::panic::Location::caller();
		let width = width.clamp(1, self.n);
		let _serial = self.serial.lock().unwrap();
		let t0 = now_if(stats_on());
		if t0.is_some() {
			ST_CUR.store(0, Ordering::Relaxed);
		}
		if width == 1 {
			// The cursor has to be rewound here too. Every helper above pulls
			// its work from it, so a one-thread pool that skipped this reset
			// ran the *first* dispatch and then silently did nothing for the
			// rest of the process: `QWEN_THREADS=1` produced wrong numbers
			// very fast, which reads like a speed-up rather than a bug.
			// Under the lock: a width cap makes this path reachable while
			// another thread's dispatch is pulling from the same cursor.
			self.cursor.store(0, Ordering::Release);
			f(0);
			return;
		}
		// Reset the work cursor *inside* the lock: `cargo test` (and the HTTP
		// server) can call two helpers concurrently, and a reset outside it
		// would rewind a dispatch that is already handing out chunks.
		self.cursor.store(0, Ordering::Release);
		let dynf: &(dyn Fn(usize) + Sync) = &f;
		// Erase the lifetime: `dispatch` does not return until `remaining`
		// reaches 0, so no worker can observe `f` after it dies.
		let (data, vtable): (usize, usize) = unsafe { std::mem::transmute(dynf) };
		self.task[0].store(data, Ordering::Release);
		self.task[1].store(vtable, Ordering::Release);
		self.remaining.store(width - 1, Ordering::Release);
		let generation = (self.word.load(Ordering::Relaxed) >> WIDTH_BITS) + 1;
		self.word
			.store(generation << WIDTH_BITS | width, Ordering::SeqCst);
		let hs = self.handles.get().unwrap();
		for (id, h) in hs.iter().enumerate().take(width - 1) {
			if self.sleeping[id].load(Ordering::SeqCst) {
				h.unpark();
			}
		}

		// The dispatching thread is a worker too: it takes the last shard.
		let on = stats_on();
		let t = now_if(on);
		f(width - 1);
		add_work(t);
		let t = now_if(on);
		if on {
			ST_DISP.fetch_add(1, Ordering::Relaxed);
		}

		while self.remaining.load(Ordering::Acquire) != 0 {
			wait_until_zero(&self.remaining, 1 << 16);
			if self.remaining.load(Ordering::Acquire) != 0 {
				std::thread::yield_now();
			}
		}
		add_since(&ST_JOIN, t);
		if let Some(t0) = t0 {
			let work = ST_CUR.swap(0, Ordering::Relaxed);
			let wall = t0.elapsed().as_nanos() as u64;
			let mut m = sites().lock().unwrap();
			let e = m.entry(site).or_insert((0, 0, 0));
			*e = (e.0 + work, e.1 + wall, e.2 + 1);
		}
	}
}

/// Shards the next dispatch is split into (pool threads including the caller,
/// capped by a live [`Width`]).
#[inline]
pub fn threads() -> usize {
	pool().width()
}

/// Caps every dispatch made while it lives at `n` shards; dropping it restores
/// the previous cap. Process-wide, not per thread: dispatches are serialized
/// anyway, and the cap belongs to the phase being run (see the module docs),
/// not to the thread asking.
pub struct Width {
	prev: usize,
}

/// Sets the workers' spin window to `rounds` pause rounds (~10 ns each)
/// while the guard lives, for a phase whose dispatches are further apart
/// than [`SPIN_ROUNDS`] but too close for a park: a worker woken from a
/// park joins a dispatch 40-60 us late at 24 shards (4 us spinning).
pub fn spin_window(rounds: u32) -> Spin {
	Spin {
		prev: SPIN.swap(rounds, Ordering::Relaxed),
	}
}

/// Restores the previous spin window on drop ([`spin_window`]).
pub struct Spin {
	prev: u32,
}

impl Drop for Spin {
	fn drop(&mut self) {
		SPIN.store(self.prev, Ordering::Relaxed);
	}
}

/// See [`Width`]. `n = 0` lifts the cap.
pub fn limit_width(n: usize) -> Width {
	Width {
		prev: pool().limit.swap(n, Ordering::Relaxed),
	}
}

impl Drop for Width {
	fn drop(&mut self) {
		pool().limit.store(self.prev, Ordering::Relaxed);
	}
}

/// Decode's shard count: `QWEN_DECODE_THREADS`, else two thirds of the pool
/// (16 of 24 here, the measured plateau; 12 lost ~3 %).
pub fn decode_threads() -> usize {
	static D: OnceLock<usize> = OnceLock::new();
	*D.get_or_init(|| {
		let n = pool().n;
		std::env::var("QWEN_DECODE_THREADS")
			.ok()
			.and_then(|s| s.parse::<usize>().ok())
			.filter(|&d| d > 0)
			.unwrap_or((2 * n).div_ceil(3))
			.min(n)
	})
}

/// Split `0..n` into work-stolen chunks and call `f(start, end)` on each. The
/// callee does its own writing, so unlike [`par_rows`] the range need not map
/// to one contiguous output buffer: that is what lets several matmuls sharing
/// an activation be covered by a **single** fork-join.
///
/// `unit_bytes` is the memory traffic of one unit, used only to decide whether
/// the dispatch is worth its ~7 us barrier.
#[track_caller]
pub(crate) fn par_range<F: Fn(usize, usize) + Sync + Send>(n: usize, unit_bytes: usize, f: F) {
	par_range_pre(n, unit_bytes, || {}, f)
}

/// [`par_range`] that first runs `pre` on the calling thread, **after** the
/// workers have been woken and are already pulling chunks. For host work that
/// must overlap the dispatch rather than precede it: the decode row split
/// submits its GPU share here, a ~15 us kernel transition that would otherwise
/// sit in front of the whole CPU share. The shared cursor lets the other
/// workers absorb the caller's late start. `pre` runs exactly once, even when
/// the range is too small to dispatch.
#[track_caller]
pub(crate) fn par_range_pre<P, F>(n: usize, unit_bytes: usize, pre: P, f: F)
where
	P: FnOnce() + Send,
	F: Fn(usize, usize) + Sync + Send,
{
	let p = pool();
	let w = p.width();
	if n == 0 || n * unit_bytes < 256 * 1024 || w == 1 {
		pre();
		if n > 0 {
			f(0, n);
		}
		return;
	}
	let chunk = n.div_ceil(CHUNKS_PER_THREAD * w).max(1);
	let pre = Mutex::new(Some(pre));
	p.dispatch(w, |shard| {
		// Shard `w - 1` is the dispatching thread (see `Pool::dispatch`).
		if shard == w - 1 {
			if let Some(pre) = pre.lock().unwrap().take() {
				pre();
			}
		}
		loop {
			let c = p.cursor.fetch_add(1, Ordering::Relaxed);
			let rs = c * chunk;
			if rs >= n {
				return;
			}
			f(rs, (rs + chunk).min(n));
		}
	});
}

/// Parallel row helper. `f(rs, re, y_slice)` writes its output rows directly
/// into the disjoint `y_slice` sub-slice, so there is **no per-matmul `Vec`
/// allocation or result copy**. Blocks until every chunk has been written, so
/// the borrowed `y` and `f` outlive the dispatch.
///
/// Rows are handed out in chunks from a shared cursor rather than pre-cut into
/// one equal stripe per thread. This host is 8 P-cores + 16 E-cores: with equal
/// stripes every GEMV finishes at E-core speed, because the fork-join waits on
/// the slowest shard. Pulling chunks lets a P-core take more of them.
#[track_caller]
pub(crate) fn par_rows<F: Fn(usize, usize, &mut [f32]) + Sync + Send>(
	out_rows: usize,
	row_bytes: usize,
	y: &mut [f32],
	f: F,
) {
	let p = pool();
	// A dispatch costs ~15 us of wake-up. Below roughly one L2's worth of
	// weight traffic the GEMV is cheaper to just run here: `out_rows` alone
	// is the wrong test, since the 32-row f32 SSM gates still stream 640 KB.
	if out_rows * row_bytes < 256 * 1024 {
		f(0, out_rows, y);
		return;
	}
	let w = p.width();
	let chunk = out_rows.div_ceil(CHUNKS_PER_THREAD * w).max(1);
	let base = y.as_mut_ptr() as usize;
	p.dispatch(w, |_shard| loop {
		let c = p.cursor.fetch_add(1, Ordering::Relaxed);
		let rs = c * chunk;
		if rs >= out_rows {
			return;
		}
		let re = (rs + chunk).min(out_rows);
		// Disjoint per chunk; the dispatcher joins before `y` is read again.
		let slice = unsafe { std::slice::from_raw_parts_mut((base as *mut f32).add(rs), re - rs) };
		f(rs, re, slice);
	});
}

/// Pooled parallel helper for **batched** (seq_len > 1) GEMM output.
///
/// Like `par_rows` it pulls disjoint output-row chunks from a shared cursor
/// (same P/E-core balancing reason), but each chunk writes a
/// `[(re-rs) * seq_len]` buffer (row `o`'s positions at `buf[(o-rs)*seq + t]`)
/// and transposes it into the `[t * out_rows + o]` `out` layout.
#[track_caller]
pub(crate) fn par_gemm_collect<F>(
	out_rows: usize,
	seq_len: usize,
	out: &mut [f32],
	ld: usize,
	col0: usize,
	min_chunk: usize,
	f: F,
) where
	F: Fn(usize, usize, &mut [f32]) + Send + Sync,
{
	// Output element (row o, position t) lands at `out[t * ld + col0 + o]`:
	// `ld = out_rows, col0 = 0` is the packed `[seq x out_rows]` layout, and a
	// wider `ld` writes a column block of a larger matrix in place.
	assert!(col0 + out_rows <= ld);
	assert!(seq_len == 0 || out.len() >= (seq_len - 1) * ld + col0 + out_rows);
	let p = pool();
	let w = p.width();
	// `min_chunk`: the row kernels run rows in groups, which a smaller chunk
	// would cut short.
	let chunk = out_rows
		.div_ceil(CHUNKS_PER_THREAD * w)
		.max(min_chunk.max(1));
	let out_addr = out.as_mut_ptr() as usize;
	p.dispatch(w, |_shard| loop {
		let c = p.cursor.fetch_add(1, Ordering::Relaxed);
		let rs = c * chunk;
		if rs >= out_rows {
			return;
		}
		let re = (rs + chunk).min(out_rows);
		// `f` writes every (row, position) of the chunk.
		let mut buf = super::uninit_vec((re - rs) * seq_len);
		f(rs, re, &mut buf);
		// Position-outer, so each `t` writes the chunk's rows as one
		// contiguous run of `out` instead of one strided store per (row,
		// position): the column of `buf` it reads stays in L1 across the
		// following positions (a line holds 16 of them).
		let rows = re - rs;
		for t in 0..seq_len {
			let dst = unsafe {
				std::slice::from_raw_parts_mut((out_addr as *mut f32).add(t * ld + col0 + rs), rows)
			};
			for (i, d) in dst.iter_mut().enumerate() {
				*d = buf[i * seq_len + t];
			}
		}
	});
}

/// Parallelize a `0..n` index range across the pool, one index per claim. Each
/// index runs the same closure with a disjoint sub-slice, so the result is
/// bit-identical to a serial `for` loop regardless of who claims what.
#[track_caller]
pub fn par_for<F: Fn(usize) + Sync + Send>(n: usize, f: F) {
	if n == 0 {
		return;
	}
	let p = pool();
	p.dispatch(p.width(), |_shard| loop {
		let i = p.cursor.fetch_add(1, Ordering::Relaxed);
		if i >= n {
			return;
		}
		f(i);
	});
}

#[cfg(test)]
mod wake_tests {
	/// Cost of a tiny dispatch after the pool has parked (a decode step's
	/// GPU wait between two of them): `cargo test --release -p quant
	/// wake_tests -- --ignored --nocapture`.
	#[test]
	#[ignore]
	fn wake() {
		for (w, idle_us) in [
			(24usize, 0u64),
			(24, 400),
			(16, 400),
			(8, 400),
			(4, 400),
			(2, 400),
		] {
			let _l = super::limit_width(w);
			let mut ts = Vec::new();
			for _ in 0..300 {
				let t = std::time::Instant::now();
				while t.elapsed().as_micros() < idle_us as u128 {
					std::thread::sleep(std::time::Duration::from_micros(50));
				}
				let t = std::time::Instant::now();
				super::par_for(48, |i| {
					std::hint::black_box(i);
				});
				ts.push(t.elapsed().as_secs_f64() * 1e6);
			}
			ts.sort_by(|a, b| a.total_cmp(b));
			eprintln!(
				"width {w}, idle {idle_us} us: median {:.1} us, p90 {:.1}",
				ts[150], ts[270]
			);
		}
	}
}
