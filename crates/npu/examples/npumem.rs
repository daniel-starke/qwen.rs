//! How much host memory one process can hand the NPU: `npumem BLOB GB`.
//!
//! Allocates copies of the graph's whole argument set until `GB` gigabytes
//! are allocated, fills the even copies with one set of bytes and the odd
//! ones with another, then binds each copy in turn, runs the graph and
//! compares its outputs bit for bit with copy 0's (even) or copy 1's (odd),
//! which must differ: so every run read the buffers it was bound to.
//! Reports where an allocation, a bind, a record or a run fails, and the time
//! per run. Built for the Flash-Next question (60 GB of int4 experts against
//! the ~25 GB of bound inputs a process was seen to get).

use npu::ze::{HostBuf, Npu};
use std::time::Instant;

fn main() {
	if let Err(e) = run() {
		eprintln!("npumem: {e}");
		std::process::exit(1);
	}
}

fn run() -> Result<(), String> {
	let a: Vec<String> = std::env::args().collect();
	if a.len() < 3 {
		return Err("usage: npumem BLOB GB".into());
	}
	let blob = std::fs::read(&a[1]).map_err(|e| format!("{}: {e}", a[1]))?;
	let gb: f64 = a[2].parse().map_err(|_| "GB: not a number")?;
	let _lock = npu::lock::NpuLock::acquire()?;
	let n = Npu::open()?;
	let mut g = n.load(&blob, 0)?;
	let set_bytes: usize = g.args.iter().map(|x| x.bytes.max(1)).sum();
	let inb: usize = g
		.args
		.iter()
		.filter(|x| !x.is_output)
		.map(|x| x.bytes)
		.sum();
	println!(
		"{} args, {:.1} MB per set ({:.1} MB inputs)",
		g.args.len(),
		set_bytes as f64 / 1e6,
		inb as f64 / 1e6
	);
	let copies = ((gb * 1e9) / set_bytes as f64).ceil().max(1.0) as usize;
	let mut sets: Vec<Vec<HostBuf<'_>>> = Vec::with_capacity(copies);
	let t = Instant::now();
	'alloc: for k in 0..copies {
		let mut s = Vec::with_capacity(g.args.len());
		for x in &g.args {
			match n.alloc(x.bytes.max(1)) {
				Ok(b) => s.push(b),
				Err(e) => {
					println!(
						"alloc failed at copy {k} ({:.2} GB allocated): {e}",
						(k * set_bytes) as f64 / 1e9
					);
					break 'alloc;
				}
			}
		}
		sets.push(s);
	}
	let total = sets.len() * set_bytes;
	println!(
		"{} copies, {:.2} GB allocated in {:.1} s",
		sets.len(),
		total as f64 / 1e9,
		t.elapsed().as_secs_f64()
	);
	// Two templates per input (even / odd copies): an LCG's bytes.
	let mut seed = 0x9E37_79B9_7F4A_7C15u64;
	let mut tmpl = || -> Vec<Vec<u8>> {
		g.args
			.iter()
			.map(|x| {
				(0..x.bytes)
					.map(|_| {
						seed = seed
							.wrapping_mul(6364136223846793005)
							.wrapping_add(1442695040888963407);
						// f16 bytes below 0x3c keep values finite and small
						((seed >> 33) as u8) & 0x3b
					})
					.collect()
			})
			.collect()
	};
	let tmpl = [tmpl(), tmpl()];
	let t = Instant::now();
	for (k, s) in sets.iter_mut().enumerate() {
		for (i, b) in s.iter_mut().enumerate() {
			if !g.args[i].is_output {
				b.as_mut_slice()[..tmpl[k % 2][i].len()].copy_from_slice(&tmpl[k % 2][i]);
			}
		}
	}
	println!("filled in {:.1} s", t.elapsed().as_secs_f64());
	let mut want: Vec<Vec<Vec<u8>>> = Vec::new();
	let mut worst = 0f64;
	let mut sum = 0f64;
	for (k, s) in sets.iter().enumerate() {
		for (i, b) in s.iter().enumerate() {
			g.bind(i, b)
				.map_err(|e| format!("copy {k}: bind {i}: {e}"))?;
		}
		let mut bt = n.batch(&mut [&mut g]).map_err(|e| {
			format!(
				"copy {k} ({:.2} GB): record: {e}",
				(k * set_bytes) as f64 / 1e9
			)
		})?;
		let ms = 1e3
			* bt.run().map_err(|e| {
				format!(
					"copy {k} ({:.2} GB): run: {e}",
					(k * set_bytes) as f64 / 1e9
				)
			})?;
		worst = worst.max(ms);
		sum += ms;
		let outs: Vec<Vec<u8>> = s
			.iter()
			.enumerate()
			.filter(|(i, _)| g.args[*i].is_output)
			.map(|(i, b)| b.as_slice()[..g.args[i].bytes].to_vec())
			.collect();
		if k < 2 {
			let nz: usize = outs
				.iter()
				.map(|o| o.iter().filter(|&&b| b != 0).count())
				.sum();
			println!("copy {k}: {nz} nonzero output bytes");
			want.push(outs);
			if k == 1 && want[0] == want[1] {
				return Err("copies 0 and 1 have equal outputs from different inputs".into());
			}
		} else if outs != want[k % 2] {
			return Err(format!(
				"copy {k} ({:.2} GB): outputs differ from copy {}",
				(k * set_bytes) as f64 / 1e9,
				k % 2
			));
		}
		if k % 50 == 0 || k + 1 == sets.len() {
			println!(
				"copy {k:5} at {:6.2} GB: ok, {ms:.2} ms",
				(k * set_bytes) as f64 / 1e9
			);
		}
	}
	println!(
		"all {} copies ok: {:.2} ms mean, {worst:.2} ms worst",
		sets.len(),
		sum / sets.len() as f64
	);
	Ok(())
}
