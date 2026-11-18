//! Energy counters for the `[eff]` line: tokens per joule is what
//! efficiency work is judged on, and CPU-seconds cannot tell a spinning core
//! (full power) from a busy one, nor see the GPU or the NPU at all.
//!
//! - **SoC package** (`pkg`: cores, iGPU, NPU, uncore), **cores** (`PP0`) and
//!   **iGPU** (`PP1`): the RAPL domains Windows exposes as the `Energy Meter`
//!   performance counters (cumulative, picowatt-hours; read through PDH, no
//!   driver or privilege needed). The NPU has no domain of its own: it is in
//!   `pkg - cores - igpu`. `RAPL_Package0_DRAM` reads 0 on this platform, so
//!   DRAM energy is not reported. Linux: `/sys/class/powercap/intel-rapl`.
//! - **GTX 1060** (`gpu`): NVML, loaded at run time. Pascal has no energy
//!   counter (`nvmlDeviceGetTotalEnergyConsumption` is Volta+), so a
//!   background thread integrates `nvmlDeviceGetPowerUsage` every 100 ms
//!   (the board reports a ~1 s average anyway): good for phases of a second
//!   or more, noise below that.
//!
//! RAPL counters update every ~1 ms but PDH may refresh them more slowly;
//! like the GPU figure, a phase under ~100 ms is not meaningful.
//! `QWEN_ENERGY=0` turns all of it off (no PDH query, no NVML thread).

use std::sync::OnceLock;

/// Cumulative joules per source at one instant; `None` = not available.
#[derive(Clone, Copy, Debug, Default)]
pub struct Joules {
	pub pkg: Option<f64>,
	pub cores: Option<f64>,
	pub igpu: Option<f64>,
	pub gpu: Option<f64>,
}

impl Joules {
	/// `self - earlier`, per source present in both.
	pub fn since(&self, earlier: &Joules) -> Joules {
		let d = |a: Option<f64>, b: Option<f64>| a.zip(b).map(|(a, b)| (a - b).max(0.0));
		Joules {
			pkg: d(self.pkg, earlier.pkg),
			cores: d(self.cores, earlier.cores),
			igpu: d(self.igpu, earlier.igpu),
			gpu: d(self.gpu, earlier.gpu),
		}
	}

	/// Package + discrete GPU, the system's compute energy (DRAM, board and
	/// PSU losses excluded); `None` when neither is available.
	pub fn total(&self) -> Option<f64> {
		match (self.pkg, self.gpu) {
			(None, None) => None,
			(a, b) => Some(a.unwrap_or(0.0) + b.unwrap_or(0.0)),
		}
	}
}

fn enabled() -> bool {
	static E: OnceLock<bool> = OnceLock::new();
	*E.get_or_init(|| std::env::var("QWEN_ENERGY").map_or(true, |v| v.trim() != "0"))
}

/// The counters now. The first call opens the counters (PDH query, NVML,
/// the GPU sampler thread), a few tens of ms; later calls are cheap.
pub fn now() -> Joules {
	if !enabled() {
		return Joules::default();
	}
	let (pkg, cores, igpu) = rapl::read();
	Joules {
		pkg,
		cores,
		igpu,
		gpu: gpu::read(),
	}
}

#[cfg(windows)]
mod rapl {
	use std::sync::{Mutex, OnceLock};

	#[link(name = "pdh")]
	extern "system" {
		fn PdhOpenQueryW(src: *const u16, user: usize, query: *mut isize) -> u32;
		fn PdhAddEnglishCounterW(q: isize, path: *const u16, user: usize, c: *mut isize) -> u32;
		fn PdhCollectQueryData(q: isize) -> u32;
		fn PdhGetRawCounterValue(c: isize, ty: *mut u32, v: *mut RawCounter) -> u32;
	}

	/// `PDH_RAW_COUNTER`: `FirstValue` at offset 16 (FILETIME is 4-aligned).
	#[repr(C)]
	struct RawCounter {
		status: u32,
		time: [u32; 2],
		first: i64,
		second: i64,
		multi: u32,
	}

	/// One picowatt-hour in joules: the `Energy Meter` counters' unit
	/// (checked against the set's `Power` counter, mW).
	const PWH: f64 = 3.6e-9;

	struct Pdh {
		query: isize,
		/// PKG, PP0 (cores), PP1 (iGPU)
		ctr: [Option<isize>; 3],
	}

	// SAFETY: PDH handles may be used from any thread; access is serialized
	// by the mutex below.
	unsafe impl Send for Pdh {}

	fn open() -> Option<Pdh> {
		unsafe {
			let mut q = 0isize;
			if PdhOpenQueryW(std::ptr::null(), 0, &mut q) != 0 {
				return None;
			}
			let mut ctr = [None; 3];
			for (i, inst) in ["PKG", "PP0", "PP1"].iter().enumerate() {
				let path: Vec<u16> = format!("\\Energy Meter(RAPL_Package0_{inst})\\Energy\0")
					.encode_utf16()
					.collect();
				let mut c = 0isize;
				if PdhAddEnglishCounterW(q, path.as_ptr(), 0, &mut c) == 0 {
					ctr[i] = Some(c);
				}
			}
			ctr[0].is_some().then_some(Pdh { query: q, ctr })
		}
	}

	pub fn read() -> (Option<f64>, Option<f64>, Option<f64>) {
		static P: OnceLock<Option<Mutex<Pdh>>> = OnceLock::new();
		let Some(p) = P.get_or_init(|| open().map(Mutex::new)) else {
			return (None, None, None);
		};
		let p = p.lock().unwrap();
		unsafe {
			if PdhCollectQueryData(p.query) != 0 {
				return (None, None, None);
			}
			let get = |c: Option<isize>| {
				let mut ty = 0u32;
				let mut v: RawCounter = std::mem::zeroed();
				let c = c?;
				(PdhGetRawCounterValue(c, &mut ty, &mut v) == 0 && v.status == 0)
					.then_some(v.first as f64 * PWH)
			};
			(get(p.ctr[0]), get(p.ctr[1]), get(p.ctr[2]))
		}
	}
}

#[cfg(not(windows))]
mod rapl {
	fn uj(path: &str) -> Option<f64> {
		let s = std::fs::read_to_string(path).ok()?;
		Some(s.trim().parse::<f64>().ok()? * 1e-6)
	}

	/// `intel-rapl:0` is the package; its subzones are named `core` and
	/// `uncore` (the iGPU). Wrap-around (`max_energy_range_uj`) is ignored:
	/// at ~100 W the counter wraps after ~45 min.
	pub fn read() -> (Option<f64>, Option<f64>, Option<f64>) {
		let base = "/sys/class/powercap/intel-rapl:0";
		let pkg = uj(&format!("{base}/energy_uj"));
		let (mut cores, mut igpu) = (None, None);
		for i in 0..4 {
			let z = format!("{base}/intel-rapl:0:{i}");
			match std::fs::read_to_string(format!("{z}/name"))
				.ok()
				.as_deref()
				.map(str::trim)
			{
				Some("core") => cores = uj(&format!("{z}/energy_uj")),
				Some("uncore") => igpu = uj(&format!("{z}/energy_uj")),
				_ => {}
			}
		}
		(pkg, cores, igpu)
	}
}

#[cfg(windows)]
mod gpu {
	use std::sync::atomic::{AtomicU64, Ordering};
	use std::sync::OnceLock;

	#[link(name = "kernel32")]
	extern "system" {
		fn LoadLibraryW(name: *const u16) -> isize;
		fn GetProcAddress(m: isize, name: *const u8) -> *const ();
	}

	type Dev = *mut ();
	type Init = unsafe extern "C" fn() -> i32;
	type Handle = unsafe extern "C" fn(u32, *mut Dev) -> i32;
	type Power = unsafe extern "C" fn(Dev, *mut u32) -> i32;
	type Energy = unsafe extern "C" fn(Dev, *mut u64) -> i32;

	enum Src {
		/// Volta+: the board's own millijoule counter.
		Counter(Energy, usize),
		/// Pascal: microjoules integrated by the sampler thread.
		Sampled(&'static AtomicU64),
	}

	fn wide(s: &str) -> Vec<u16> {
		s.encode_utf16().chain([0]).collect()
	}

	fn open() -> Option<Src> {
		unsafe {
			let mut m = LoadLibraryW(wide("nvml.dll").as_ptr());
			if m == 0 {
				m = LoadLibraryW(
					wide("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll").as_ptr(),
				);
			}
			if m == 0 {
				return None;
			}
			let sym = |n: &[u8]| {
				let p = GetProcAddress(m, n.as_ptr());
				(!p.is_null()).then_some(p)
			};
			let init: Init = std::mem::transmute(sym(b"nvmlInit_v2\0")?);
			let handle: Handle = std::mem::transmute(sym(b"nvmlDeviceGetHandleByIndex_v2\0")?);
			let power: Power = std::mem::transmute(sym(b"nvmlDeviceGetPowerUsage\0")?);
			if init() != 0 {
				return None;
			}
			let mut dev: Dev = std::ptr::null_mut();
			if handle(0, &mut dev) != 0 {
				return None;
			}
			if let Some(f) = sym(b"nvmlDeviceGetTotalEnergyConsumption\0") {
				let f: Energy = std::mem::transmute(f);
				let mut mj = 0u64;
				if f(dev, &mut mj) == 0 {
					return Some(Src::Counter(f, dev as usize));
				}
			}
			let mut mw = 0u32;
			if power(dev, &mut mw) != 0 {
				return None;
			}
			static UJ: AtomicU64 = AtomicU64::new(0);
			let d = dev as usize;
			std::thread::Builder::new()
				.name("nvml-power".into())
				.spawn(move || {
					let mut t = std::time::Instant::now();
					loop {
						std::thread::sleep(std::time::Duration::from_millis(100));
						let mut mw = 0u32;
						let ok = power(d as Dev, &mut mw) == 0;
						let n = std::time::Instant::now();
						if ok {
							// mW x us = nJ; / 1000 = uJ
							let us = (n - t).as_micros() as u64;
							UJ.fetch_add(mw as u64 * us / 1000, Ordering::Relaxed);
						}
						t = n;
					}
				})
				.ok()?;
			Some(Src::Sampled(&UJ))
		}
	}

	pub fn read() -> Option<f64> {
		static S: OnceLock<Option<Src>> = OnceLock::new();
		match S.get_or_init(open).as_ref()? {
			Src::Counter(f, d) => {
				let mut mj = 0u64;
				let ok = unsafe { f(*d as Dev, &mut mj) } == 0;
				ok.then_some(mj as f64 * 1e-3)
			}
			Src::Sampled(uj) => Some(uj.load(Ordering::Relaxed) as f64 * 1e-6),
		}
	}
}

#[cfg(not(windows))]
mod gpu {
	pub fn read() -> Option<f64> {
		None
	}
}
