//! A global allocator that recycles **large** blocks.
//!
//! Prefill allocates the same set of activation buffers for every layer
//! (`seq x n_embd`, `seq x n_ff`, the SSM's q/k/v) about 1 GB of them per layer
//! at `seq = 2048`. The Windows heap hands any block past ~1 MB straight to
//! `VirtualAlloc` and returns it to the OS on free, so every layer's buffers
//! were fresh pages and the first write to each 4 KB page was a demand-zero
//! fault. Those faults do not scale across cores and landed inside whatever
//! phase touched the buffer first (mostly the GEMMs and the elementwise passes).
//!
//! This keeps freed blocks of at least [`MIN_BLOCK`] bytes, keyed by exact
//! layout, and hands them back to the next allocation of the same layout.
//! Shapes repeat exactly from layer to layer, so nearly every large allocation
//! after the first layer is a hit. The cache is bounded (`QWEN_ALLOC_CACHE_MB`,
//! default 4096; `0` disables it) and evicts the oldest block first; a block
//! larger than the whole bound is never kept.
//!
//! Every cached block came from [`System`] with the same layout it is handed
//! out for, so `realloc` / `dealloc` on it stay plain `System` calls.
//!
//! **Release hook.** A large block can be registered with the GPU as imported
//! host memory (`VK_EXT_external_memory_host`, the prefill split writing its
//! rows straight into the output matrix). The import pins the pages and must
//! be destroyed before they go back to the OS, so every large block passes
//! through [`set_release_hook`]'s function right before `System` frees or
//! moves it. A block returned to the cache keeps its import, which is the
//! point: the next layer gets the same block back already imported.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Smallest block worth keeping. Below this the heap reuses memory itself.
pub const MIN_BLOCK: usize = 1 << 20;
const SLOTS: usize = 64;

#[derive(Clone, Copy)]
struct Slot {
	ptr: usize,
	size: usize,
	align: usize,
	age: u64,
}

const EMPTY: Slot = Slot {
	ptr: 0,
	size: 0,
	align: 0,
	age: 0,
};

/// `fn(ptr, size)` as `usize`; 0 = none.
static RELEASE_HOOK: AtomicUsize = AtomicUsize::new(0);
/// Set by the first allocation through a [`RecyclingAlloc`], i.e. when it is
/// the global allocator and so sees every free.
static ACTIVE: AtomicBool = AtomicBool::new(false);

/// Called with `(ptr, size)` before any block of at least [`MIN_BLOCK`]
/// bytes is freed to (or reallocated by) `System`. It runs inside the
/// allocator: it must not free a block of [`MIN_BLOCK`] or more itself, nor
/// take a lock held by a thread that does.
pub fn set_release_hook(f: fn(usize, usize)) {
	RELEASE_HOOK.store(f as usize, Ordering::Release);
}

/// Whether the release hook sees every large free: a [`RecyclingAlloc`] is
/// the global allocator (it has served an allocation) and a hook is set.
/// Without it nothing may hold on to a heap block past its owner.
pub fn release_tracked() -> bool {
	ACTIVE.load(Ordering::Acquire) && RELEASE_HOOK.load(Ordering::Acquire) != 0
}

fn release(ptr: *mut u8, size: usize) {
	let h = RELEASE_HOOK.load(Ordering::Acquire);
	if h != 0 {
		// SAFETY: only ever stored from a `fn(usize, usize)`.
		let f: fn(usize, usize) = unsafe { std::mem::transmute(h) };
		f(ptr as usize, size);
	}
}

/// See the module docs. Install with `#[global_allocator]` in the binary.
pub struct RecyclingAlloc {
	lock: AtomicBool,
	slots: std::cell::UnsafeCell<[Slot; SLOTS]>,
	held: AtomicUsize,
	clock: AtomicUsize,
	/// Byte bound, resolved on first use (`usize::MAX` = not yet read).
	cap: AtomicUsize,
}

// SAFETY: `slots` is only touched while `lock` is held.
unsafe impl Sync for RecyclingAlloc {}

impl RecyclingAlloc {
	pub const fn new() -> Self {
		Self {
			lock: AtomicBool::new(false),
			slots: std::cell::UnsafeCell::new([EMPTY; SLOTS]),
			held: AtomicUsize::new(0),
			clock: AtomicUsize::new(0),
			cap: AtomicUsize::new(usize::MAX),
		}
	}

	fn cap(&self) -> usize {
		let c = self.cap.load(Ordering::Relaxed);
		if c != usize::MAX {
			return c;
		}
		// Reading the environment allocates, but only small blocks, which
		// never reach this path.
		let mb = std::env::var("QWEN_ALLOC_CACHE_MB")
			.ok()
			.and_then(|v| v.parse::<usize>().ok())
			.unwrap_or(4096);
		let c = mb.saturating_mul(1 << 20);
		self.cap.store(c, Ordering::Relaxed);
		c
	}

	fn with_slots<R>(&self, f: impl FnOnce(&mut [Slot; SLOTS]) -> R) -> R {
		while self
			.lock
			.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
			.is_err()
		{
			std::hint::spin_loop();
		}
		// SAFETY: exclusive while the lock is held.
		let r = f(unsafe { &mut *self.slots.get() });
		self.lock.store(false, Ordering::Release);
		r
	}

	/// A cached block of exactly `layout`, if one is held.
	fn take(&self, layout: Layout) -> Option<*mut u8> {
		let p = self.with_slots(|slots| {
			let s = slots
				.iter_mut()
				.find(|s| s.ptr != 0 && s.size == layout.size() && s.align == layout.align())?;
			let p = s.ptr;
			*s = EMPTY;
			Some(p)
		})?;
		self.held.fetch_sub(layout.size(), Ordering::Relaxed);
		Some(p as *mut u8)
	}

	/// Keep `ptr`, evicting the oldest blocks to stay under the bound. Returns
	/// `false` if it was not kept (the caller frees it).
	fn keep(&self, ptr: *mut u8, layout: Layout, cap: usize) -> bool {
		if layout.size() > cap {
			return false;
		}
		let age = self.clock.fetch_add(1, Ordering::Relaxed) as u64;
		let mut evicted = [EMPTY; SLOTS];
		let mut n_ev = 0;
		let kept = self.with_slots(|slots| {
			let mut held = self.held.load(Ordering::Relaxed);
			let kept = loop {
				let free = slots.iter().position(|s| s.ptr == 0);
				if let (Some(i), true) = (free, held + layout.size() <= cap) {
					slots[i] = Slot {
						ptr: ptr as usize,
						size: layout.size(),
						align: layout.align(),
						age,
					};
					held += layout.size();
					break true;
				}
				let oldest = slots
					.iter()
					.enumerate()
					.filter(|(_, s)| s.ptr != 0)
					.min_by_key(|(_, s)| s.age)
					.map(|(i, _)| i);
				let Some(old) = oldest else { break false };
				held -= slots[old].size;
				evicted[n_ev] = slots[old];
				n_ev += 1;
				slots[old] = EMPTY;
			};
			self.held.store(held, Ordering::Relaxed);
			kept
		});
		for s in &evicted[..n_ev] {
			release(s.ptr as *mut u8, s.size);
			// SAFETY: every cached block came from `System` with this layout.
			unsafe {
				System.dealloc(
					s.ptr as *mut u8,
					Layout::from_size_align_unchecked(s.size, s.align),
				)
			};
		}
		kept
	}
}

impl Default for RecyclingAlloc {
	fn default() -> Self {
		Self::new()
	}
}

/// The global instance, once it has allocated (for [`trim`]).
static INSTANCE: std::sync::atomic::AtomicPtr<RecyclingAlloc> =
	std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// Return every cached block to the OS (after dropping a model: the cache
/// would otherwise hold up to its bound of idle memory). Returns the bytes
/// freed; 0 when the recycling allocator is not the global allocator.
pub fn trim() -> usize {
	let p = INSTANCE.load(Ordering::Acquire);
	if p.is_null() {
		return 0;
	}
	// SAFETY: set from `&self` of the global allocator, a static.
	let a = unsafe { &*p };
	let mut out = [EMPTY; SLOTS];
	a.with_slots(|slots| {
		for (o, s) in out.iter_mut().zip(slots.iter_mut()) {
			*o = *s;
			*s = EMPTY;
		}
	});
	let mut n = 0;
	for s in out.iter().filter(|s| s.ptr != 0) {
		a.held.fetch_sub(s.size, Ordering::Relaxed);
		n += s.size;
		release(s.ptr as *mut u8, s.size);
		// SAFETY: every cached block came from `System` with this layout.
		unsafe {
			System.dealloc(
				s.ptr as *mut u8,
				Layout::from_size_align_unchecked(s.size, s.align),
			)
		};
	}
	n
}

unsafe impl GlobalAlloc for RecyclingAlloc {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		if !ACTIVE.load(Ordering::Relaxed) {
			ACTIVE.store(true, Ordering::Release);
			INSTANCE.store(self as *const _ as *mut _, Ordering::Release);
		}
		if layout.size() >= MIN_BLOCK && self.cap() > 0 {
			if let Some(p) = self.take(layout) {
				return p;
			}
		}
		System.alloc(layout)
	}

	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		if !ACTIVE.load(Ordering::Relaxed) {
			ACTIVE.store(true, Ordering::Release);
		}
		if layout.size() >= MIN_BLOCK && self.cap() > 0 {
			if let Some(p) = self.take(layout) {
				std::ptr::write_bytes(p, 0, layout.size());
				return p;
			}
		}
		System.alloc_zeroed(layout)
	}

	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		if layout.size() >= MIN_BLOCK {
			let cap = self.cap();
			if cap > 0 && self.keep(ptr, layout, cap) {
				return;
			}
			release(ptr, layout.size());
		}
		System.dealloc(ptr, layout)
	}

	unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
		if layout.size() >= MIN_BLOCK {
			release(ptr, layout.size());
		}
		System.realloc(ptr, layout, new_size)
	}
}
