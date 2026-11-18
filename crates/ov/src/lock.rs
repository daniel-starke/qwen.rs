//! Machine-wide locks around the NPU, as named mutexes.
//!
//! - [`NpuLock`] (`Global\qwen-rs-npu`): the device itself. Two processes
//!   running on the NPU at once make every timing meaningless, so the NPU
//!   engine holds it from its first infer request to its end.
//! - [`CompileSlot`] (`Global\qwen-rs-npu-compile-0` / `-1`): compiling for
//!   the NPU runs on the host CPU and peaks at 30-47 GB, so at most two
//!   compiles run at once; a compile does not need the device.
//!
//! The kernel releases a mutex when its owner dies (an abandoned mutex is
//! acquired normally), so a killed run cannot leave one stuck, which is why
//! the compile slots are two mutexes and not a semaphore. A mutex belongs to
//! the thread that acquired it; both are taken and dropped on the loading
//! thread (or released by process exit). `crates/ov/scripts/npu_lock.ps1`
//! takes the NPU lock around any other command (the Python experiments).
//! `QWEN_OV_NPU_LOCK=0` skips both.

#[cfg(windows)]
type Handle = *mut std::ffi::c_void;

/// Owns acquired mutexes; releases them on drop.
struct Held {
	#[cfg(windows)]
	h: Vec<Handle>,
}

/// The device lock; the flag: counted in `HELD_HERE`.
pub struct NpuLock(#[allow(dead_code)] Held, bool);

/// Device locks this process holds ([`NpuLock::acquire`]).
static HELD_HERE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// [`NpuLock::shared`]: the device lock on a thread of its own, or nothing
/// when this process holds it already.
pub struct SharedNpuLock(Option<std::sync::mpsc::Sender<()>>);

impl Drop for SharedNpuLock {
	fn drop(&mut self) {
		if let Some(tx) = self.0.take() {
			let _ = tx.send(());
		}
	}
}
pub struct CompileSlot(#[allow(dead_code)] Held);

// Only the handles move; see the module comment on thread ownership.
unsafe impl Send for Held {}
unsafe impl Sync for Held {}

pub const NAME: &str = "Global\\qwen-rs-npu";
const COMPILE_NAMES: [&str; 2] = [
	"Global\\qwen-rs-npu-compile-0",
	"Global\\qwen-rs-npu-compile-1",
];

#[cfg(windows)]
mod sys {
	use std::ffi::c_void;
	#[link(name = "kernel32")]
	extern "system" {
		pub fn CreateMutexW(attr: *mut c_void, initial: i32, name: *const u16) -> *mut c_void;
		pub fn WaitForMultipleObjects(n: u32, h: *const *mut c_void, all: i32, ms: u32) -> u32;
		pub fn ReleaseMutex(h: *mut c_void) -> i32;
		pub fn CloseHandle(h: *mut c_void) -> i32;
		pub fn GetLastError() -> u32;
	}
	pub const WAIT_OBJECT_0: u32 = 0;
	pub const WAIT_ABANDONED_0: u32 = 0x80;
	pub const WAIT_TIMEOUT: u32 = 0x102;
	pub const INFINITE: u32 = 0xFFFF_FFFF;
}

fn disabled() -> bool {
	std::env::var("QWEN_OV_NPU_LOCK").is_ok_and(|v| v == "0")
}

impl Held {
	fn none() -> Held {
		Held {
			#[cfg(windows)]
			h: Vec::new(),
		}
	}

	/// Block until one of `names` is free and own it; `what` names it in the
	/// waiting message.
	#[cfg(windows)]
	fn any_of(names: &[&str], what: &str) -> Result<Held, String> {
		if disabled() {
			return Ok(Held::none());
		}
		let mut hs = Vec::new();
		// SAFETY: plain Win32 calls on handles this function owns.
		unsafe {
			for n in names {
				let w: Vec<u16> = n.encode_utf16().chain(Some(0)).collect();
				let h = sys::CreateMutexW(std::ptr::null_mut(), 0, w.as_ptr());
				if h.is_null() {
					let e = sys::GetLastError();
					hs.into_iter().for_each(|h| {
						sys::CloseHandle(h);
					});
					return Err(format!("CreateMutexW({n}) failed: error {e}"));
				}
				hs.push(h);
			}
			let wait = |ms| sys::WaitForMultipleObjects(hs.len() as u32, hs.as_ptr(), 0, ms);
			let mut r = wait(0);
			if r == sys::WAIT_TIMEOUT {
				eprintln!("  npu: waiting for {what} ({})", names.join(", "));
				let t = std::time::Instant::now();
				r = wait(sys::INFINITE);
				eprintln!("  npu: got {what} after {:.0} s", t.elapsed().as_secs_f64());
			}
			let n = hs.len() as u32;
			let i = if (sys::WAIT_OBJECT_0..sys::WAIT_OBJECT_0 + n).contains(&r) {
				r - sys::WAIT_OBJECT_0
			} else if (sys::WAIT_ABANDONED_0..sys::WAIT_ABANDONED_0 + n).contains(&r) {
				r - sys::WAIT_ABANDONED_0
			} else {
				hs.into_iter().for_each(|h| {
					sys::CloseHandle(h);
				});
				return Err(format!("waiting on {} failed: {r:#x}", names.join(", ")));
			} as usize;
			let h = hs.swap_remove(i);
			hs.into_iter().for_each(|h| {
				sys::CloseHandle(h);
			});
			Ok(Held { h: vec![h] })
		}
	}

	#[cfg(not(windows))]
	fn any_of(_: &[&str], _: &str) -> Result<Held, String> {
		Ok(Held::none())
	}
}

impl Drop for Held {
	fn drop(&mut self) {
		#[cfg(windows)]
		for &h in &self.h {
			// SAFETY: `h` is owned and was acquired by `any_of`.
			unsafe {
				sys::ReleaseMutex(h);
				sys::CloseHandle(h);
			}
		}
	}
}

impl NpuLock {
	/// Block until no other process holds the NPU.
	pub fn acquire() -> Result<NpuLock, String> {
		let h = Held::any_of(&[NAME], "another process to release the NPU")?;
		HELD_HERE.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
		Ok(NpuLock(h, true))
	}

	/// The device lock for a model another thread may drop (a mutex
	/// belongs to the thread that took it): taken and released on a thread
	/// of its own. When this process holds the lock already (an NPU chat
	/// engine next to the image model), nothing is taken: waiting for it
	/// would wait for ourselves.
	pub fn shared() -> Result<SharedNpuLock, String> {
		if HELD_HERE.load(std::sync::atomic::Ordering::SeqCst) > 0 {
			return Ok(SharedNpuLock(None));
		}
		let (tx, rx) = std::sync::mpsc::channel::<Result<(), String>>();
		let (rel_tx, rel_rx) = std::sync::mpsc::channel::<()>();
		std::thread::spawn(move || match NpuLock::acquire() {
			Ok(l) => {
				let _ = tx.send(Ok(()));
				let _ = rel_rx.recv();
				drop(l);
			}
			Err(e) => {
				let _ = tx.send(Err(e));
			}
		});
		rx.recv().map_err(|e| e.to_string())??;
		Ok(SharedNpuLock(Some(rel_tx)))
	}

	/// A lock that holds nothing (another device than the NPU).
	pub fn none() -> NpuLock {
		NpuLock(Held::none(), false)
	}
}

impl Drop for NpuLock {
	fn drop(&mut self) {
		if self.1 {
			HELD_HERE.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
		}
	}
}

impl CompileSlot {
	/// Block until fewer than two NPU compiles run machine-wide.
	pub fn acquire() -> Result<CompileSlot, String> {
		Held::any_of(
			&COMPILE_NAMES,
			"a compile slot (two NPU compiles already running)",
		)
		.map(CompileSlot)
	}
}

#[cfg(all(test, windows))]
mod tests {
	use super::*;
	use std::sync::mpsc;
	use std::time::Duration;

	/// Takes the real machine-wide slots, so it would wait behind running
	/// compiles: `cargo test -p ov -- --ignored compile_slots`.
	#[test]
	#[ignore]
	fn third_compile_waits_for_a_free_slot() {
		let (rel_tx, rel_rx) = mpsc::channel::<()>();
		let (got_tx, got_rx) = mpsc::channel::<()>();
		// Two holders, each on its own thread (a mutex is re-entrant within
		// its owning thread).
		let mut holders = Vec::new();
		let rel_rx = std::sync::Arc::new(std::sync::Mutex::new(rel_rx));
		for _ in 0..2 {
			let (rel_rx, got_tx) = (rel_rx.clone(), got_tx.clone());
			holders.push(std::thread::spawn(move || {
				let s = CompileSlot::acquire().unwrap();
				got_tx.send(()).unwrap();
				rel_rx.lock().unwrap().recv().unwrap();
				drop(s);
			}));
		}
		got_rx.recv().unwrap();
		got_rx.recv().unwrap();
		let (third_tx, third_rx) = mpsc::channel();
		let third = std::thread::spawn(move || {
			let _s = CompileSlot::acquire().unwrap();
			third_tx.send(()).unwrap();
		});
		assert!(
			third_rx.recv_timeout(Duration::from_millis(500)).is_err(),
			"third compile did not wait"
		);
		rel_tx.send(()).unwrap();
		third_rx
			.recv_timeout(Duration::from_secs(5))
			.expect("third compile never got a slot");
		rel_tx.send(()).unwrap();
		holders
			.into_iter()
			.chain(Some(third))
			.for_each(|h| h.join().unwrap());
	}
}
