//! Host memory/compute probe: the ceilings every CPU-side measurement should
//! be read against.
//!
//!     cargo run --release -p quant --example bwbench -- bw [GiB]   # DRAM read GB/s by core set
//!     cargo run --release -p quant --example bwbench -- burn [SECS] # FMA GFLOP/s, P vs E, per second
//!     cargo run --release -p quant --example bwbench -- fault [GiB] # fresh-page (demand-zero) fault rate
//!     cargo run --release -p quant --example bwbench -- topo        # logical CPU -> efficiency class
//!
//! `bw` reads one buffer with AVX2 loads from different thread sets, each pinned
//! to a P- or E-core (Windows `GetSystemCpuSetInformation` efficiency class), in
//! two modes: `static` (one fixed stripe per thread; the fork-join waits on the
//! slowest) and `steal` (1 MiB chunks from a shared cursor, as `quant::parallel`
//! does). Measured on the 285K with 4x32 GB DDR5 at 4400 MT/s (2 DIMMs
//! per channel; 70.4 GB/s theoretical): **8 P-cores 60.5-61 GB/s, all 24 threads
//! ~55-57**; adding the E-cores *lowers* aggregate read bandwidth. Decode
//! reads ~11.1 GB/token in ~214 ms (~52 GB/s over the whole token), so it is
//! within ~7 % of what 24 threads can pull.
//!
//! Pinning and the P/E split are Windows-only; elsewhere every CPU is one class
//! and threads float.

use std::arch::x86_64::*;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Instant;

#[cfg(windows)]
mod os {
	#[repr(C)]
	struct CpuSet {
		size: u32,
		ty: u32,
		id: u32,
		group: u16,
		lp: u8,
		core: u8,
		llc: u8,
		numa: u8,
		eff: u8,
		flags: u8,
		sched: u32,
		tag: u64,
	}

	#[link(name = "kernel32")]
	extern "system" {
		fn GetSystemCpuSetInformation(
			info: *mut u8,
			len: u32,
			ret: *mut u32,
			proc: isize,
			flags: u32,
		) -> i32;
		fn GetCurrentThread() -> isize;
		fn SetThreadAffinityMask(h: isize, mask: usize) -> usize;
	}

	/// (logical CPU, efficiency class) per CPU; a higher class is a faster core.
	pub fn topo() -> Vec<(usize, u8)> {
		let mut len = 0u32;
		unsafe { GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut len, -1, 0) };
		let mut buf = vec![0u8; len as usize];
		assert!(unsafe { GetSystemCpuSetInformation(buf.as_mut_ptr(), len, &mut len, -1, 0) } != 0);
		let mut out = vec![];
		let mut off = 0usize;
		while off < len as usize {
			let s = unsafe { &*(buf.as_ptr().add(off) as *const CpuSet) };
			out.push((s.lp as usize, s.eff));
			off += s.size as usize;
		}
		out
	}

	pub fn pin(cpu: usize) {
		unsafe { SetThreadAffinityMask(GetCurrentThread(), 1usize << cpu) };
	}
}

#[cfg(not(windows))]
mod os {
	pub fn topo() -> Vec<(usize, u8)> {
		let n = std::thread::available_parallelism()
			.map(|n| n.get())
			.unwrap_or(1);
		(0..n).map(|c| (c, 0)).collect()
	}
	pub fn pin(_cpu: usize) {}
}

#[target_feature(enable = "avx2")]
unsafe fn read_sum(p: *const u8, n: usize) -> u64 {
	let mut a0 = _mm256_setzero_si256();
	let (mut a1, mut a2, mut a3) = (a0, a0, a0);
	let mut i = 0;
	while i + 128 <= n {
		a0 = _mm256_xor_si256(a0, _mm256_load_si256(p.add(i) as *const __m256i));
		a1 = _mm256_xor_si256(a1, _mm256_load_si256(p.add(i + 32) as *const __m256i));
		a2 = _mm256_xor_si256(a2, _mm256_load_si256(p.add(i + 64) as *const __m256i));
		a3 = _mm256_xor_si256(a3, _mm256_load_si256(p.add(i + 96) as *const __m256i));
		i += 128;
	}
	let a = _mm256_xor_si256(_mm256_xor_si256(a0, a1), _mm256_xor_si256(a2, a3));
	_mm256_extract_epi64(a, 0) as u64 ^ _mm256_extract_epi64(a, 3) as u64
}

/// 8 independent FMA chains: enough to cover the 4-cycle latency on both ports.
/// `iters` must come through `black_box`, with a constant the call is pure and
/// loop-invariant, and LLVM hoists it out of the caller's loop.
#[target_feature(enable = "avx2,fma")]
unsafe fn fma_burn(iters: u32) -> f32 {
	let m = _mm256_set1_ps(0.999_999);
	let a = _mm256_set1_ps(1e-7);
	let mut r = [_mm256_set1_ps(1.0); 8];
	for _ in 0..iters {
		for x in r.iter_mut() {
			*x = _mm256_fmadd_ps(*x, m, a);
		}
	}
	let mut s = _mm256_setzero_ps();
	for x in r {
		s = _mm256_add_ps(s, x);
	}
	_mm256_cvtss_f32(s)
}
const BURN_ITERS: u32 = 1 << 16;
const BURN_FLOP: f64 = BURN_ITERS as f64 * 8.0 * 8.0 * 2.0;

struct Buf(*mut u8, usize);
unsafe impl Send for Buf {}
unsafe impl Sync for Buf {}

fn alloc_touched(bytes: usize) -> Arc<Buf> {
	let layout = std::alloc::Layout::from_size_align(bytes, 4096).unwrap();
	let b = Arc::new(Buf(unsafe { std::alloc::alloc(layout) }, bytes));
	let th = 24;
	let chunk = bytes / th;
	std::thread::scope(|s| {
		for t in 0..th {
			let b = b.clone();
			s.spawn(move || unsafe {
				std::ptr::write_bytes(b.0.add(t * chunk), (t as u8) | 1, chunk)
			});
		}
	});
	b
}

/// GB/s per pass. Each thread is pinned to `cpus[i]` (`None` = unpinned).
fn read_bw(buf: &Arc<Buf>, cpus: &[Option<usize>], passes: usize, steal: bool) -> Vec<f64> {
	const CH: usize = 1 << 20;
	let n = cpus.len();
	let nch = buf.1 / CH;
	let stripe = (buf.1 / n) & !127;
	let cursor = Arc::new(AtomicUsize::new(0));
	let go = Arc::new(Barrier::new(n + 1));
	let done = Arc::new(Barrier::new(n + 1));
	let hs: Vec<_> = cpus
		.iter()
		.enumerate()
		.map(|(i, &c)| {
			let (b, go, done, cur) = (buf.clone(), go.clone(), done.clone(), cursor.clone());
			std::thread::spawn(move || {
				if let Some(c) = c {
					os::pin(c);
				}
				let mut x = 0u64;
				for _ in 0..passes {
					go.wait();
					if steal {
						loop {
							let k = cur.fetch_add(1, Ordering::Relaxed);
							if k >= nch {
								break;
							}
							x ^= unsafe { read_sum(b.0.add(k * CH), CH) };
						}
					} else {
						x ^= unsafe { read_sum(b.0.add(i * stripe), stripe) };
					}
					done.wait();
				}
				x
			})
		})
		.collect();
	let bytes = if steal { nch * CH } else { stripe * n };
	let mut r = vec![];
	for _ in 0..passes {
		cursor.store(0, Ordering::Relaxed);
		go.wait();
		let t = Instant::now();
		done.wait();
		r.push(bytes as f64 / t.elapsed().as_secs_f64() / 1e9);
	}
	let x = hs.into_iter().fold(0, |a, h| a ^ h.join().unwrap());
	std::hint::black_box(x);
	r
}

fn stats(v: &[f64]) -> String {
	let mut s = v.to_vec();
	s.sort_by(|a, b| a.partial_cmp(b).unwrap());
	format!(
		"best {:6.1}  med {:6.1}  min {:6.1}",
		s[s.len() - 1],
		s[s.len() / 2],
		s[0]
	)
}

fn main() {
	let args: Vec<String> = std::env::args().skip(1).collect();
	let mode = args.first().map(String::as_str).unwrap_or("bw");
	let num = |d: f64| args.get(1).and_then(|s| s.parse::<f64>().ok()).unwrap_or(d);
	let t = os::topo();
	let fast = t.iter().map(|x| x.1).max().unwrap_or(0);
	let p: Vec<usize> = t.iter().filter(|x| x.1 == fast).map(|x| x.0).collect();
	let e: Vec<usize> = t.iter().filter(|x| x.1 != fast).map(|x| x.0).collect();
	println!("P cores {p:?}\nE cores {e:?}");
	let pin = |v: &[usize]| v.iter().map(|&c| Some(c)).collect::<Vec<_>>();

	match mode {
		"topo" => {
			for (lp, eff) in &t {
				println!("cpu {lp:2} efficiency class {eff}");
			}
		}
		"bw" => {
			let buf = alloc_touched((num(8.0) as usize) << 30);
			let mut cfgs: Vec<(String, Vec<Option<usize>>)> = vec![];
			for k in [t.len(), 16, 8] {
				cfgs.push((format!("unpinned x{k}"), vec![None; k.min(t.len())]));
			}
			for k in [1, 2, 4, 8] {
				if k <= p.len() {
					cfgs.push((format!("P x{k}"), pin(&p[..k])));
				}
			}
			for k in [4, 8, 16] {
				if k <= e.len() {
					cfgs.push((format!("E x{k}"), pin(&e[..k])));
				}
			}
			if !e.is_empty() {
				let all: Vec<usize> = p.iter().chain(&e).copied().collect();
				cfgs.push(("P+E all".into(), pin(&all)));
				let half: Vec<usize> = p
					.iter()
					.chain(e.iter().take(e.len() / 2))
					.copied()
					.collect();
				cfgs.push(("P+E half".into(), pin(&half)));
			}
			for (name, c) in &cfgs {
				read_bw(&buf, c, 1, false);
				println!(
					"{name:14} static {} GB/s",
					stats(&read_bw(&buf, c, 7, false))
				);
				if c.len() > 1 {
					println!(
						"{name:14} steal  {} GB/s",
						stats(&read_bw(&buf, c, 7, true))
					);
				}
			}
		}
		"soak" => {
			// `soak SECS THREADS [p|e]`: sustained DRAM reads from THREADS
			// pinned cores, GB/s per 2 s window; a load to run next to
			// another reader (the NPU) to see how the two share DRAM.
			let secs = num(30.0);
			let k = args
				.get(2)
				.and_then(|s| s.parse::<usize>().ok())
				.unwrap_or(8);
			let set = if args.get(3).map(String::as_str) == Some("e") {
				&e
			} else {
				&p
			};
			let cpus: Vec<usize> = set.iter().cycle().take(k).copied().collect();
			let buf = alloc_touched(4 << 30);
			let stripe = (buf.1 / k) & !((1 << 20) - 1);
			let ctr = Arc::new(AtomicU64::new(0));
			let stop = Arc::new(AtomicBool::new(false));
			let hs: Vec<_> = cpus
				.iter()
				.enumerate()
				.map(|(i, &c)| {
					let (b, ctr, stop) = (buf.clone(), ctr.clone(), stop.clone());
					std::thread::spawn(move || {
						os::pin(c);
						let mut x = 0u64;
						let mut off = 0usize;
						while !stop.load(Ordering::Relaxed) {
							x ^= unsafe { read_sum(b.0.add(i * stripe + off), 1 << 20) };
							ctr.fetch_add(1 << 20, Ordering::Relaxed);
							off = (off + (1 << 20)) % stripe;
						}
						x
					})
				})
				.collect();
			println!("soak: {k} threads on {cpus:?}, {secs} s");
			let t0 = Instant::now();
			let (mut tl, mut last) = (Instant::now(), 0u64);
			while t0.elapsed().as_secs_f64() < secs {
				std::thread::sleep(std::time::Duration::from_secs(2));
				let v = ctr.load(Ordering::Relaxed);
				println!(
					"  t {:6.1} s  {:6.1} GB/s",
					t0.elapsed().as_secs_f64(),
					(v - last) as f64 / tl.elapsed().as_secs_f64() / 1e9
				);
				tl = Instant::now();
				last = v;
			}
			stop.store(true, Ordering::Relaxed);
			let x = hs.into_iter().fold(0, |a, h| a ^ h.join().unwrap());
			std::hint::black_box(x);
			println!(
				"soak: mean {:.1} GB/s",
				ctr.load(Ordering::Relaxed) as f64 / t0.elapsed().as_secs_f64() / 1e9
			);
		}
		"burn" => {
			// One pinned FMA thread per CPU; a power-limit step (PL2 -> PL1)
			// or a frequency change shows as a step in the per-second rate.
			let secs = num(30.0);
			let n = t.len();
			let ctr: Arc<Vec<AtomicU64>> = Arc::new((0..n).map(|_| AtomicU64::new(0)).collect());
			let stop = Arc::new(AtomicBool::new(false));
			let hs: Vec<_> = (0..n)
				.map(|i| {
					let (ctr, stop, cpu) = (ctr.clone(), stop.clone(), t[i].0);
					std::thread::spawn(move || {
						os::pin(cpu);
						while !stop.load(Ordering::Relaxed) {
							std::hint::black_box(unsafe {
								fma_burn(std::hint::black_box(BURN_ITERS))
							});
							ctr[i].fetch_add(1, Ordering::Relaxed);
						}
					})
				})
				.collect();
			let t0 = Instant::now();
			let mut last = vec![0u64; n];
			let mut tl = Instant::now();
			while t0.elapsed().as_secs_f64() < secs {
				std::thread::sleep(std::time::Duration::from_secs(1));
				let dt = tl.elapsed().as_secs_f64();
				tl = Instant::now();
				let (mut ps, mut es) = (0.0, 0.0);
				for i in 0..n {
					let v = ctr[i].load(Ordering::Relaxed);
					let g = (v - last[i]) as f64 * BURN_FLOP / dt / 1e9;
					last[i] = v;
					if t[i].1 == fast {
						ps += g
					} else {
						es += g
					}
				}
				println!(
					"{:5.1}s  P {:7.1}  E {:7.1}  total {:7.1} GFLOP/s",
					t0.elapsed().as_secs_f64(),
					ps,
					es,
					ps + es
				);
			}
			stop.store(true, Ordering::Relaxed);
			hs.into_iter().for_each(|h| h.join().unwrap());
		}
		"fault" => {
			// Fresh pages: allocate, touch one byte per 4 KiB page, free. What
			// the first use of every buffer costs before `quant::alloc` recycles it.
			let bytes = (num(2.0) as usize) << 30;
			for rep in 0..4 {
				for th in [1usize, t.len()] {
					let layout = std::alloc::Layout::from_size_align(bytes, 4096).unwrap();
					let tm = Instant::now();
					let base = unsafe { std::alloc::alloc(layout) } as usize;
					let chunk = bytes / th;
					std::thread::scope(|s| {
						for k in 0..th {
							s.spawn(move || {
								let mut o = k * chunk;
								while o < (k + 1) * chunk {
									unsafe { *((base + o) as *mut u8) = 1 };
									o += 4096;
								}
							});
						}
					});
					let dt = tm.elapsed().as_secs_f64();
					unsafe { std::alloc::dealloc(base as *mut u8, layout) };
					println!(
						"rep {rep} threads {th:2}: {:6.1} GB/s of fresh pages ({:.0} ns/page)",
						bytes as f64 / dt / 1e9,
						dt * 1e9 / (bytes / 4096) as f64
					);
				}
			}
		}
		other => {
			eprintln!("unknown mode {other}; use bw | burn | fault | topo");
			std::process::exit(2);
		}
	}
}
