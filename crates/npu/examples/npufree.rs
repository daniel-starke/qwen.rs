//! What a pinned NPU host buffer costs to give back: `npufree GB [CHUNK_MB]
//! [ALIGN_KB] [FLAGS] [free|exit]`.
//!
//! Allocates `GB` gigabytes with `zeMemAllocHost` in `CHUNK_MB` pieces at
//! `ALIGN_KB` alignment and `FLAGS` (`ze_host_mem_alloc_flags`), touches
//! every page, reports the page size Windows backs the first piece with
//! (`QueryWorkingSetEx`'s large-page bit), then times `zeMemFree` of all
//! pieces (`free`, default) or leaves them to process exit (`exit`; time it
//! from outside). Built for the Flash-Next teardown question.

use npu::ze::{HostBuf, Npu};
use std::time::Instant;

#[cfg(windows)]
fn large_page(p: *const u8) -> Option<bool> {
	#[repr(C)]
	struct Info {
		va: *const u8,
		attr: usize,
	}
	#[link(name = "kernel32")]
	extern "system" {
		fn GetCurrentProcess() -> isize;
		fn K32QueryWorkingSetEx(p: isize, info: *mut Info, len: u32) -> i32;
	}
	let mut i = Info { va: p, attr: 0 };
	// SAFETY: one PSAPI_WORKING_SET_EX_INFORMATION entry.
	let ok = unsafe { K32QueryWorkingSetEx(GetCurrentProcess(), &mut i, 16) };
	(ok != 0 && i.attr & 1 != 0).then_some(i.attr >> 22 & 1 != 0)
}
#[cfg(not(windows))]
fn large_page(_: *const u8) -> Option<bool> {
	None
}

fn main() {
	if let Err(e) = run() {
		eprintln!("npufree: {e}");
		std::process::exit(1);
	}
}

fn run() -> Result<(), String> {
	let a: Vec<String> = std::env::args().collect();
	let arg = |i: usize, d: &str| a.get(i).cloned().unwrap_or_else(|| d.into());
	let gb: f64 = arg(1, "8").parse().map_err(|_| "GB: not a number")?;
	let chunk: usize = arg(2, "1024").parse::<usize>().map_err(|_| "CHUNK_MB")? << 20;
	let align: usize = arg(3, "4").parse::<usize>().map_err(|_| "ALIGN_KB")? << 10;
	let flags: u32 = arg(4, "0").parse().map_err(|_| "FLAGS")?;
	let exit = arg(5, "free") == "exit";
	let _lock = npu::lock::NpuLock::acquire()?;
	let n = Npu::open()?;
	let total = (gb * 1e9) as usize;
	let t = Instant::now();
	let mut bufs: Vec<HostBuf<'_>> = Vec::new();
	let mut got = 0;
	while got < total {
		let mut b = n.alloc_with(chunk.min(total - got), align, flags)?;
		got += b.len();
		for x in b.as_mut_slice().iter_mut().step_by(4096) {
			*x = 1;
		}
		bufs.push(b);
	}
	println!(
		"{} pieces, {:.2} GB: alloc + touch {:.2} s, first piece large pages: {:?}",
		bufs.len(),
		got as f64 / 1e9,
		t.elapsed().as_secs_f64(),
		large_page(bufs[0].as_slice().as_ptr())
	);
	if exit {
		println!("exiting with the buffers held");
		std::mem::forget(bufs);
		std::process::exit(0);
	}
	let t = Instant::now();
	drop(bufs);
	println!("zeMemFree: {:.2} s", t.elapsed().as_secs_f64());
	Ok(())
}
