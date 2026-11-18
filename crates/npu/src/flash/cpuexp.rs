//! A decode step's routed experts on the CPU cores: the `experts` graph's
//! math (`npubuild::flash::graphs::experts`) on the layer's expert pools as
//! the NPU reads them (int4 codes, f16 row scales), without a job per layer.
//! At one position a layer's 10 experts are 24.5 MB of int4: the cores read
//! that in well under a millisecond, where an NPU job costs ~1.7 ms of
//! binding, recording and fixed overhead.
//!
//! Per expert: `xq = round(moe_q)` (already prescaled and rotated, int8),
//! `h = silu(g) * u` of `gu = xq W^T 2^-8 s`, `hq = round(rotate(h *
//! d.pre))` in Hadamard blocks of 128, `y = hq D^T 2^-8 s`. A code byte
//! holds the even element in its low nibble (`gptq::pack4`).

use core::arch::x86_64::*;

use super::W;
use quant::parallel::par_for;

const FE: usize = 640;

/// Nanoseconds spent in [`run_n`]'s phases (setup: the input's int8 form
/// and the expert list; the experts' dispatch), for the step timing line.
pub static T_PHASE: [std::sync::atomic::AtomicU64; 2] =
	[const { std::sync::atomic::AtomicU64::new(0) }; 2];

fn tick(i: usize, t: &mut std::time::Instant) {
	let n = std::time::Instant::now();
	T_PHASE[i].fetch_add(
		(n - *t).as_nanos() as u64,
		std::sync::atomic::Ordering::Relaxed,
	);
	*t = n;
}
const RB: usize = 128;

/// An int8 vector, even and odd elements apart, and 8 x its sum (the bias
/// of the nibbles stored + 8).
struct Q {
	even: Vec<i8>,
	odd: Vec<i8>,
	bias: i32,
}

fn quant(v: &[f32]) -> Q {
	let r: Vec<i8> = v
		.iter()
		.map(|&x| x.round().clamp(-128.0, 127.0) as i8)
		.collect();
	Q {
		even: r.iter().step_by(2).copied().collect(),
		odd: r.iter().skip(1).step_by(2).copied().collect(),
		bias: 8 * r.iter().map(|&x| x as i32).sum::<i32>(),
	}
}

/// `sum_i code_i x_i` over a row of `x.even.len()` bytes.
///
/// # Safety
/// AVX2; the row has that many bytes, a multiple of 32.
#[target_feature(enable = "avx2")]
unsafe fn dot(row: *const u8, x: &Q) -> i32 {
	let n = x.even.len();
	let m = _mm256_set1_epi8(0x0f);
	let flip = _mm256_set1_epi8(0x88u8 as i8);
	let ones = _mm256_set1_epi16(1);
	let mut acc = _mm256_setzero_si256();
	let mut i = 0;
	while i < n {
		let b = _mm256_xor_si256(_mm256_loadu_si256(row.add(i) as *const __m256i), flip);
		let lo = _mm256_and_si256(b, m);
		let hi = _mm256_and_si256(_mm256_srli_epi16(b, 4), m);
		let e = _mm256_loadu_si256(x.even.as_ptr().add(i) as *const __m256i);
		let o = _mm256_loadu_si256(x.odd.as_ptr().add(i) as *const __m256i);
		let p = _mm256_add_epi16(_mm256_maddubs_epi16(lo, e), _mm256_maddubs_epi16(hi, o));
		acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, ones));
		i += 32;
	}
	let s = _mm_add_epi32(
		_mm256_castsi256_si128(acc),
		_mm256_extracti128_si256(acc, 1),
	);
	let s = _mm_hadd_epi32(s, s);
	let s = _mm_hadd_epi32(s, s);
	_mm_cvtsi128_si32(s) - x.bias
}

/// The Sylvester Hadamard / sqrt(n) in place on blocks of `n`.
pub(crate) fn fwht(v: &mut [f32], n: usize) {
	let s = 1.0 / (n as f32).sqrt();
	for b in v.chunks_exact_mut(n) {
		let mut h = 1;
		while h < n {
			for i in (0..n).step_by(2 * h) {
				for j in i..i + h {
					let (a, c) = (b[j], b[j + h]);
					b[j] = a + c;
					b[j + h] = a - c;
				}
			}
			h *= 2;
		}
		b.iter_mut().for_each(|x| *x *= s);
	}
}

fn h(x: u16) -> f32 {
	crate::ir::f16_to_f32(x)
}

/// An expert's down input from its gate / up outputs `g` (`[2 FE]`):
/// `round(rotate(silu(gate) * up * d.pre))`. The SwiGLU is
/// `quant::act::silu_mul` (AVX2 `v_expf`), not a scalar `f32::exp` per
/// element: 6400 of them per decode layer on the thread that finishes an
/// expert, while the others wait on it.
fn down_input(g: &[f32], pre: &[u16]) -> Q {
	let mut v = [0f32; FE];
	quant::act::silu_mul(&g[..FE], &g[FE..2 * FE], &mut v);
	for (x, &p) in v.iter_mut().zip(&pre[..FE]) {
		*x *= h(p);
	}
	fwht(&mut v, RB);
	quant(&v)
}

/// An expert pool of one layer as the NPU binds it.
pub struct Pool<'a> {
	/// `[NE][2 FE][W / 2]`, `[NE][2 FE]` f16, `[NE][W][FE / 2]`, `[NE][W]` f16
	pub gu_w: &'a [u8],
	pub gu_s: &'a [u16],
	pub d_w: &'a [u8],
	pub d_s: &'a [u16],
	/// `[NE][FE]` f16: each expert's down input signs and range
	pub d_pre: &'a [u16],
}

/// `sum_e w_e expert_e(x)` for one position: `x` the MoE input as `post`
/// leaves it (`moe_q`, f16 values), `sel` the experts and their weights.
pub fn run(p: &Pool<'_>, x: &[u16], sel: &[(usize, f32)]) -> Vec<f32> {
	let xf: Vec<f32> = x[..W].iter().map(|&v| h(v)).collect();
	run_f32(p, &xf, sel)
}

/// [`run`] on an f32 MoE input.
pub fn run_f32(p: &Pool<'_>, xf: &[f32], sel: &[(usize, f32)]) -> Vec<f32> {
	run_n(p, &[xf], &[sel]).remove(0)
}

/// [`run_f32`] of several positions (`xs[t]`, `sels[t]`): an expert's rows
/// are read once for every position routed to it. Each position's result
/// is [`run_f32`]'s, bit for bit.
///
/// One dispatch: the gate / up blocks of every expert hit, then its down
/// blocks. The thread that finishes an expert's last gate / up block forms
/// its down input (SwiGLU, rotation, int8), so that runs in parallel with
/// the other experts' reads instead of between two dispatches; a down task
/// waits for its expert's input (every gate / up task is claimed before any
/// down task, so it waits only on running ones) and writes per-slot
/// partials, which the last down task of a row block sums in slot order:
/// the phased form's sums ([`run_n_phased`]), bit for bit.
pub fn run_n(p: &Pool<'_>, xs: &[&[f32]], sels: &[&[(usize, f32)]]) -> Vec<Vec<f32>> {
	use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
	assert!(std::is_x86_feature_detected!("avx2"));
	let mut tt = std::time::Instant::now();
	let np = xs.len();
	let xq: Vec<Q> = xs.iter().map(|x| quant(&x[..W])).collect();
	// slot (t, j) is number so[t] + j; its gate / up outputs at slot 2 FE
	let mut so = vec![0usize; np + 1];
	for t in 0..np {
		so[t + 1] = so[t] + sels[t].len();
	}
	let ns = so[np];
	let mut users: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
	for (t, sel) in sels.iter().enumerate() {
		for (j, &(e, _)) in sel.iter().enumerate() {
			match users.iter_mut().find(|u| u.0 == e) {
				Some(u) => u.1.push((t, j)),
				None => users.push((e, vec![(t, j)])),
			}
		}
	}
	let nu = users.len();
	const GB: usize = 64;
	const DB: usize = 256;
	let (nb, ndb) = (2 * FE / GB, W / DB);
	let mut gu = vec![0f32; ns * 2 * FE];
	let mut hq: Vec<Q> = (0..ns)
		.map(|_| Q {
			even: Vec::new(),
			odd: Vec::new(),
			bias: 0,
		})
		.collect();
	let mut part = vec![0f32; ns * W];
	let mut y = vec![0f32; np * W];
	let (gp, hp, pp, yp) = (
		gu.as_mut_ptr() as usize,
		hq.as_mut_ptr() as usize,
		part.as_mut_ptr() as usize,
		y.as_mut_ptr() as usize,
	);
	let gdone: Vec<AtomicUsize> = (0..nu).map(|_| AtomicUsize::new(0)).collect();
	let ready: Vec<AtomicBool> = (0..nu).map(|_| AtomicBool::new(false)).collect();
	let bdone: Vec<AtomicUsize> = (0..ndb).map(|_| AtomicUsize::new(0)).collect();
	tick(0, &mut tt);
	par_for(nu * (nb + ndb), |task| {
		if task < nu * nb {
			let (u, b) = (task / nb, task % nb);
			let (e, us) = (users[u].0, &users[u].1);
			for r in b * GB..(b + 1) * GB {
				let er = e * 2 * FE + r;
				for &(t, j) in us {
					// SAFETY: AVX2 checked; the row lies in the pool; task
					// (u, b) owns rows [b GB, (b + 1) GB) of its users' outputs.
					unsafe {
						let d = dot(p.gu_w.as_ptr().add(er * (W / 2)), &xq[t]);
						*(gp as *mut f32).add((so[t] + j) * 2 * FE + r) =
							d as f32 * (1.0 / 256.0) * h(p.gu_s[er]);
					}
				}
			}
			if gdone[u].fetch_add(1, Ordering::AcqRel) + 1 == nb {
				for &(t, j) in us {
					let s = so[t] + j;
					// SAFETY: every block of slot s is written (the counter's
					// acquire); only this task writes hq[s]
					let g = unsafe {
						std::slice::from_raw_parts((gp as *const f32).add(s * 2 * FE), 2 * FE)
					};
					let q = down_input(g, &p.d_pre[e * FE..]);
					unsafe { *(hp as *mut Q).add(s) = q };
				}
				ready[u].store(true, Ordering::Release);
			}
			return;
		}
		let task = task - nu * nb;
		let (u, b) = (task / ndb, task % ndb);
		let (e, us) = (users[u].0, &users[u].1);
		while !ready[u].load(Ordering::Acquire) {
			std::hint::spin_loop();
		}
		for r in b * DB..(b + 1) * DB {
			let er = e * W + r;
			for &(t, j) in us {
				let s = so[t] + j;
				// SAFETY: hq[s] is complete (ready's acquire); task (u, b)
				// owns rows [b DB, (b + 1) DB) of its slots' partials
				unsafe {
					let d = dot(p.d_w.as_ptr().add(er * (FE / 2)), &*(hp as *const Q).add(s));
					*(pp as *mut f32).add(s * W + r) =
						sels[t][j].1 * d as f32 * (1.0 / 256.0) * h(p.d_s[er]);
				}
			}
		}
		if bdone[b].fetch_add(1, Ordering::AcqRel) + 1 == nu {
			for t in 0..np {
				for r in b * DB..(b + 1) * DB {
					let mut a = 0f32;
					for s in so[t]..so[t + 1] {
						// SAFETY: every partial of block b is written
						a += unsafe { *(pp as *const f32).add(s * W + r) };
					}
					// SAFETY: only the last task of block b writes its rows
					unsafe { *(yp as *mut f32).add(t * W + r) = a };
				}
			}
		}
	});
	tick(1, &mut tt);
	y.chunks_exact(W).map(|c| c.to_vec()).collect()
}

/// [`run_n`] in three phases (gate / up, SwiGLU on one thread, down): the
/// reference its tests compare with.
#[cfg(test)]
fn run_n_phased(p: &Pool<'_>, xs: &[&[f32]], sels: &[&[(usize, f32)]]) -> Vec<Vec<f32>> {
	assert!(std::is_x86_feature_detected!("avx2"));
	let np = xs.len();
	let xq: Vec<Q> = xs.iter().map(|x| quant(&x[..W])).collect();
	// gate / up outputs of (position, slot) at off[t] + j 2 FE
	let mut off = vec![0usize; np + 1];
	for t in 0..np {
		off[t + 1] = off[t] + sels[t].len() * 2 * FE;
	}
	// the experts hit and their (position, slot) users
	let mut users: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
	for (t, sel) in sels.iter().enumerate() {
		for (j, &(e, _)) in sel.iter().enumerate() {
			match users.iter_mut().find(|u| u.0 == e) {
				Some(u) => u.1.push((t, j)),
				None => users.push((e, vec![(t, j)])),
			}
		}
	}
	// gate / up rows of every expert hit, blocks of 64 rows
	let mut gu = vec![0f32; off[np]];
	let gp = gu.as_mut_ptr() as usize;
	const GB: usize = 64;
	let nb = 2 * FE / GB;
	par_for(users.len() * nb, |task| {
		let (u, b) = (task / nb, task % nb);
		let (e, us) = (&users[u].0, &users[u].1);
		for r in b * GB..(b + 1) * GB {
			let er = e * 2 * FE + r;
			for &(t, j) in us {
				// SAFETY: AVX2 checked; the row lies in the pool; task
				// (u, b) owns rows [b GB, (b + 1) GB) of its users' outputs.
				unsafe {
					let d = dot(p.gu_w.as_ptr().add(er * (W / 2)), &xq[t]);
					*(gp as *mut f32).add(off[t] + j * 2 * FE + r) =
						d as f32 * (1.0 / 256.0) * h(p.gu_s[er]);
				}
			}
		}
	});
	let hq: Vec<Vec<Q>> = (0..np)
		.map(|t| {
			(0..sels[t].len())
				.map(|j| {
					let e = sels[t][j].0;
					let g = &gu[off[t] + j * 2 * FE..][..2 * FE];
					down_input(g, &p.d_pre[e * FE..])
				})
				.collect()
		})
		.collect();
	let mut y = vec![0f32; np * W];
	let yp = y.as_mut_ptr() as usize;
	const DB: usize = 64;
	par_for(W / DB, |b| {
		for r in b * DB..(b + 1) * DB {
			for t in 0..np {
				let mut a = 0f32;
				for (j, &(e, w)) in sels[t].iter().enumerate() {
					let er = e * W + r;
					// SAFETY: as above
					let d = unsafe { dot(p.d_w.as_ptr().add(er * (FE / 2)), &hq[t][j]) };
					a += w * d as f32 * (1.0 / 256.0) * h(p.d_s[er]);
				}
				// SAFETY: block b owns rows [b DB, (b + 1) DB) of every output
				unsafe { *(yp as *mut f32).add(t * W + r) = a };
			}
		}
	});
	y.chunks_exact(W).map(|c| c.to_vec()).collect()
}

#[cfg(test)]
mod tests {
	use super::*;

	/// `cargo test --release -p npu flash::cpuexp::tests::bench -- --ignored
	/// --nocapture`: one decode layer's routed experts (10 of 512) on
	/// random pools of 6 layers (7.4 GB, so the hits come from DRAM), per
	/// thread count; ms per layer and the GB/s of the codes read.
	#[test]
	#[ignore]
	fn bench() {
		const NE: usize = 512;
		const NL: usize = 6;
		let mut seed = 7u64;
		let mut rnd = move || {
			seed = seed
				.wrapping_mul(6364136223846793005)
				.wrapping_add(1442695040888963407);
			(seed >> 33) as u32
		};
		let fill = |n: usize, r: &mut dyn FnMut() -> u32| -> Vec<u8> {
			(0..n).map(|_| r() as u8).collect()
		};
		let scale = |n: usize, r: &mut dyn FnMut() -> u32| -> Vec<u16> {
			(0..n).map(|_| 0x2000 | (r() & 0x3ff) as u16).collect()
		};
		let pools: Vec<[Vec<u8>; 2]> = (0..NL)
			.map(|_| {
				[
					fill(NE * 2 * FE * W / 2, &mut rnd),
					fill(NE * W * FE / 2, &mut rnd),
				]
			})
			.collect();
		let (gs, ds, pre) = (
			scale(NE * 2 * FE, &mut rnd),
			scale(NE * W, &mut rnd),
			scale(NE * FE, &mut rnd),
		);
		let x: Vec<f32> = (0..W).map(|_| (rnd() % 255) as f32 - 127.0).collect();
		let bytes = 10.0 * (2 * FE * W / 2 + W * FE / 2) as f64;
		let mut sels = Vec::new();
		for _ in 0..2400 {
			let mut sel: Vec<(usize, f32)> = Vec::new();
			while sel.len() < 10 {
				let e = rnd() as usize % NE;
				if sel.iter().all(|s| s.0 != e) {
					sel.push((e, (rnd() % 1000) as f32 / 1000.0));
				}
			}
			sels.push(sel);
		}
		let pool = |i: usize| Pool {
			gu_w: &pools[i % NL][0],
			gu_s: &gs,
			d_w: &pools[i % NL][1],
			d_s: &ds,
			d_pre: &pre,
		};
		// bit for bit against the phased form, at 1 and 3 positions (shared hits)
		let x2: Vec<f32> = x.iter().map(|v| -v * 0.5).collect();
		for i in 0..8 {
			let a = run_n(&pool(i), &[&x], &[&sels[i]]);
			let b = run_n_phased(&pool(i), &[&x], &[&sels[i]]);
			assert!(a == b, "1 position, case {i}");
			let s3: [&[(usize, f32)]; 3] = [&sels[i], &sels[i + 1][..7], &sels[i]];
			let a = run_n(&pool(i), &[&x, &x2, &x2], &s3);
			let b = run_n_phased(&pool(i), &[&x, &x2, &x2], &s3);
			assert!(a == b, "3 positions, case {i}");
		}
		for th in [24usize, 16] {
			let _w = quant::parallel::limit_width(th);
			for round in 0..4 {
				for (name, f) in [
					(
						"phased",
						run_n_phased
							as fn(&Pool<'_>, &[&[f32]], &[&[(usize, f32)]]) -> Vec<Vec<f32>>,
					),
					("fused", run_n),
				] {
					const R: usize = 240;
					let t = std::time::Instant::now();
					for i in 0..R {
						std::hint::black_box(f(
							&pool(i),
							&[&x],
							&[&sels[(i + round * R) % sels.len()]],
						));
					}
					let ms = t.elapsed().as_secs_f64() * 1e3 / R as f64;
					if round > 0 {
						eprintln!(
							"{th} threads {name}: {ms:.3} ms per layer ({:.1} GB/s)",
							bytes / ms * 1e-6
						);
					}
				}
			}
		}
	}
}

#[cfg(test)]
mod bwtests {
	use core::arch::x86_64::*;

	/// The read ceiling of [`super::tests::bench`]'s pattern: 10 random
	/// 2.46 MB regions of a 7.4 GB buffer summed in 80 KB tasks.
	#[test]
	#[ignore]
	fn readbw() {
		let n = 6 * 512 * 2_457_600usize;
		let buf: Vec<u8> = (0..n).map(|i| (i * 31) as u8).collect();
		let mut seed = 9u64;
		for (th, task) in [
			(24usize, 81920usize),
			(16, 81920),
			(24, 16384),
			(24, 327680),
		] {
			let _w = quant::parallel::limit_width(th);
			let t = std::time::Instant::now();
			const R: usize = 300;
			for _ in 0..R {
				let regs: Vec<usize> = (0..10)
					.map(|_| {
						seed = seed
							.wrapping_mul(6364136223846793005)
							.wrapping_add(1442695040888963407);
						((seed >> 33) as usize % (n / 2_457_600)) * 2_457_600
					})
					.collect();
				let per = 2_457_600 / task;
				let bp = buf.as_ptr() as usize;
				quant::parallel::par_for(10 * per, |i| unsafe {
					let p = (bp + regs[i / per] + (i % per) * task) as *const __m256i;
					let mut a = _mm256_setzero_si256();
					for j in 0..task / 32 {
						a = _mm256_add_epi8(a, _mm256_loadu_si256(p.add(j)));
					}
					std::hint::black_box(a);
				});
			}
			let s = t.elapsed().as_secs_f64() / R as f64;
			eprintln!(
				"{th} threads, {task} B tasks: {:.3} ms ({:.1} GB/s)",
				s * 1e3,
				24.576e6 / s * 1e-9
			);
		}
	}

	/// Sustained sequential read of 1 GB spans over all threads: the DRAM
	/// ceiling [`readbw`]'s pattern is compared with.
	#[test]
	#[ignore]
	fn seqbw() {
		let n = 6 * 512 * 2_457_600usize;
		let buf: Vec<u8> = (0..n).map(|i| (i * 31) as u8).collect();
		for (th, task) in [
			(24usize, 1usize << 20),
			(16, 1 << 20),
			(8, 1 << 20),
			(24, 81920),
		] {
			let _w = quant::parallel::limit_width(th);
			let span = 1usize << 30;
			let t = std::time::Instant::now();
			const R: usize = 20;
			for r in 0..R {
				let off = (r * span) % (n - span);
				let bp = buf.as_ptr() as usize + off;
				quant::parallel::par_for(span / task, |i| unsafe {
					let p = (bp + i * task) as *const __m256i;
					let mut a = _mm256_setzero_si256();
					for j in 0..task / 32 {
						a = _mm256_add_epi8(a, _mm256_loadu_si256(p.add(j)));
					}
					std::hint::black_box(a);
				});
			}
			let s = t.elapsed().as_secs_f64() / R as f64;
			eprintln!(
				"seq {th} threads, {task} B tasks: {:.1} GB/s",
				span as f64 / s * 1e-9
			);
		}
	}
}

/// Large pages for the expert pools (plan item 4a): is the decode expert
/// pass faster when the pools sit in 2 MB pages?
#[cfg(all(test, windows))]
mod lptests {
	use super::*;

	#[link(name = "kernel32")]
	extern "system" {
		fn GetCurrentProcess() -> isize;
		fn GetLargePageMinimum() -> usize;
		fn VirtualAlloc(addr: *mut u8, size: usize, kind: u32, protect: u32) -> *mut u8;
	}
	#[link(name = "advapi32")]
	extern "system" {
		fn OpenProcessToken(p: isize, access: u32, token: *mut isize) -> i32;
		fn LookupPrivilegeValueW(sys: *const u16, name: *const u16, luid: *mut u64) -> i32;
		fn AdjustTokenPrivileges(
			t: isize,
			disable: i32,
			new: *const u8,
			len: u32,
			prev: *mut u8,
			ret: *mut u32,
		) -> i32;
	}

	/// Enable `SeLockMemoryPrivilege` in this process's token (it must be
	/// assigned to the account).
	fn enable_lock_pages() -> bool {
		unsafe {
			let mut tok = 0isize;
			// TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY
			if OpenProcessToken(GetCurrentProcess(), 0x20 | 0x8, &mut tok) == 0 {
				return false;
			}
			let name: Vec<u16> = "SeLockMemoryPrivilege\0".encode_utf16().collect();
			let mut luid = 0u64;
			if LookupPrivilegeValueW(std::ptr::null(), name.as_ptr(), &mut luid) == 0 {
				return false;
			}
			// TOKEN_PRIVILEGES { count 1, LUID, SE_PRIVILEGE_ENABLED }
			let mut tp = [0u8; 16];
			tp[..4].copy_from_slice(&1u32.to_le_bytes());
			tp[4..12].copy_from_slice(&luid.to_le_bytes());
			tp[12..].copy_from_slice(&2u32.to_le_bytes());
			AdjustTokenPrivileges(
				tok,
				0,
				tp.as_ptr(),
				16,
				std::ptr::null_mut(),
				std::ptr::null_mut(),
			) != 0 && std::io::Error::last_os_error().raw_os_error() == Some(0)
		}
	}

	/// A copy of `src` in large pages (leaked: a test).
	fn large_copy(src: &[u8]) -> Option<&'static [u8]> {
		unsafe {
			let lp = GetLargePageMinimum();
			let n = src.len().div_ceil(lp) * lp;
			// MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE
			let p = VirtualAlloc(std::ptr::null_mut(), n, 0x1000 | 0x2000 | 0x2000_0000, 4);
			if p.is_null() {
				return None;
			}
			std::ptr::copy_nonoverlapping(src.as_ptr(), p, src.len());
			Some(std::slice::from_raw_parts(p, src.len()))
		}
	}

	/// `cargo test --release -p npu flash::cpuexp::lptests -- --ignored
	/// --nocapture`: [`super::tests::bench`]'s decode layer (10 of 512
	/// experts, 6 layers of pools so the hits come from DRAM, 16 threads) on
	/// 4 KiB pages and on a copy in 2 MB pages, interleaved rounds.
	#[test]
	#[ignore]
	fn bench_large_pages() {
		const NE: usize = 512;
		const NL: usize = 6;
		assert!(
			enable_lock_pages(),
			"SeLockMemoryPrivilege is not assigned / enabled"
		);
		let mut seed = 7u64;
		let mut rnd = move || {
			seed = seed
				.wrapping_mul(6364136223846793005)
				.wrapping_add(1442695040888963407);
			(seed >> 33) as u32
		};
		let fill = |n: usize, r: &mut dyn FnMut() -> u32| -> Vec<u8> {
			(0..n).map(|_| r() as u8).collect()
		};
		let scale = |n: usize, r: &mut dyn FnMut() -> u32| -> Vec<u16> {
			(0..n).map(|_| 0x2000 | (r() & 0x3ff) as u16).collect()
		};
		let small: Vec<[Vec<u8>; 2]> = (0..NL)
			.map(|_| {
				[
					fill(NE * 2 * FE * W / 2, &mut rnd),
					fill(NE * W * FE / 2, &mut rnd),
				]
			})
			.collect();
		let large: Vec<[&[u8]; 2]> = small
			.iter()
			.map(|p| {
				[
					large_copy(&p[0]).expect("large-page allocation failed"),
					large_copy(&p[1]).expect("large-page allocation failed"),
				]
			})
			.collect();
		let (gs, ds, pre) = (
			scale(NE * 2 * FE, &mut rnd),
			scale(NE * W, &mut rnd),
			scale(NE * FE, &mut rnd),
		);
		let x: Vec<f32> = (0..W).map(|_| (rnd() % 255) as f32 - 127.0).collect();
		let bytes = 10.0 * (2 * FE * W / 2 + W * FE / 2) as f64;
		let sels: Vec<Vec<(usize, f32)>> = (0..2400)
			.map(|_| {
				let mut sel: Vec<(usize, f32)> = Vec::new();
				while sel.len() < 10 {
					let e = rnd() as usize % NE;
					if sel.iter().all(|s| s.0 != e) {
						sel.push((e, (rnd() % 1000) as f32 / 1000.0));
					}
				}
				sel
			})
			.collect();
		fn mk<'a>(w: [&'a [u8]; 2], gs: &'a [u16], ds: &'a [u16], pre: &'a [u16]) -> Pool<'a> {
			Pool {
				gu_w: w[0],
				gu_s: gs,
				d_w: w[1],
				d_s: ds,
				d_pre: pre,
			}
		}
		let pool = |w| mk(w, &gs, &ds, &pre);
		// same results from both copies
		let a = run_n(&pool([&small[0][0], &small[0][1]]), &[&x], &[&sels[0]]);
		let b = run_n(&pool(large[0]), &[&x], &[&sels[0]]);
		assert!(a == b);
		let _w = quant::parallel::limit_width(16);
		let (mut ts, mut tl) = (0f64, 0f64);
		const R: usize = 240;
		for round in 0..8 {
			for lp in [false, true] {
				let t = std::time::Instant::now();
				for i in 0..R {
					let w = if lp {
						large[i % NL]
					} else {
						[&small[i % NL][0][..], &small[i % NL][1][..]]
					};
					std::hint::black_box(run_n(
						&pool(w),
						&[&x],
						&[&sels[(i + round * R) % sels.len()]],
					));
				}
				let ms = t.elapsed().as_secs_f64() * 1e3 / R as f64;
				if round > 0 {
					*(if lp { &mut tl } else { &mut ts }) += ms;
				}
				eprintln!(
					"round {round} {}: {ms:.3} ms per layer ({:.1} GB/s)",
					if lp { "2 MB pages" } else { "4 KiB pages" },
					bytes / ms * 1e-6
				);
			}
		}
		eprintln!(
			"mean 4 KiB {:.3} ms, 2 MB {:.3} ms ({:+.1} %)",
			ts / 7.0,
			tl / 7.0,
			100.0 * (tl / ts - 1.0)
		);
	}
}
