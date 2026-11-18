//! Direct Intel NPU access (`NPU.md`): Level Zero and the driver's graph extension, no
//! OpenVINO runtime and no NPUW.
//!
//! - [`ze`]: open the NPU, compile an OpenVINO-format IR with the installed
//!   driver's compiler (or load a native blob), bind host buffers, run, and
//!   read per-task / per-layer profiling.
//! - [`engine`]: the NPU-only model behind `reset` / `forward` (prefill
//!   through the lean graph sets, decode through the s = 1 decode set).
//! - [`ir`]: a small in-memory IR writer for microbenchmark graphs.
//! - [`shave`]: kernel compiler, assembler and linker for ACT-SHAVE kernels.
//! - [`flash`]: Qwen3.8-Flash-Next's host-side pieces (n-gram hash and table).

pub mod engine;
pub mod farcpu;
pub mod flash;
pub mod ir;
pub mod mlp;
pub mod shave;
pub mod vis;
pub mod ze;

pub use ov::lock;

/// Positional file reads on every host: `seek_read` on Windows, `read_at`
/// elsewhere (the engine only runs on Windows, but the crate builds and its
/// tests run on Linux too).
pub trait ReadAt {
	fn seek_read(&self, buf: &mut [u8], off: u64) -> std::io::Result<usize>;
}

impl ReadAt for std::fs::File {
	#[cfg(windows)]
	fn seek_read(&self, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
		std::os::windows::fs::FileExt::seek_read(self, buf, off)
	}

	#[cfg(unix)]
	fn seek_read(&self, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
		std::os::unix::fs::FileExt::read_at(self, buf, off)
	}
}

/// Deterministic pseudo-random bytes (xorshift64*), for weights whose values
/// do not matter to a timing.
pub fn noise(n: usize, seed: u64) -> Vec<u8> {
	let mut s = seed | 1;
	let mut v = Vec::with_capacity(n + 8);
	while v.len() < n {
		s ^= s >> 12;
		s ^= s << 25;
		s ^= s >> 27;
		v.extend_from_slice(&s.wrapping_mul(0x2545_f491_4f6c_dd1d).to_le_bytes());
	}
	v.truncate(n);
	v
}
