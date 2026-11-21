//! Qwen3.8-Flash-Next on the NPU, v1: every
//! position goes through the prefill graphs in chunks of [`S`], or, where
//! the set has them (`*_d1.blob`), a decode step and a short tail through
//! the one-position graphs ([`D1_MAX`]). Per layer and chunk: the PLE graph (layer
//! 1), `delta_pre` / `attn_pre`, the recurrence on the scan device (the
//! iGPU; without one on the host) or the attention (host), `post_*`, the
//! routed experts as jobs of [`G`] experts x [`C`] tokens, and the combine
//! on the host. The final mixer and the lm_head run on the host.
//!
//! Each layer's dense graphs are recorded once at load with that layer's
//! weight buffers and the engine's fixed runtime buffers; the expert jobs
//! are bound per job (their weights live in one buffer per layer).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use super::kvcol::Col;
use super::ngram;
use super::{EOS, HC, NL, PLE_LAYER, W};
use crate::engine::Scan;
use crate::ze::{self, Batch, Graph, HostBuf, Npu};
use crate::ReadAt;
use quant::parallel::par_for;
use quant::snap::{self, Chunk};
use std::sync::Arc;

/// Positions per chunk (the graphs' static shape).
pub const S: usize = 256;
/// Sub-chunks of [`S`] positions per expert pass in prefill
/// (`QWEN_FLASH_KSUP`, 1 = one pass per chunk). 16 (4096 tokens): more
/// tokens per expert, so fewer DMA-bound narrow slots per token; prefill
/// 131.9 -> 146.5 tok/s at 3.2k, 128.6 -> 137.2 at 9.7k against 8 (32: no
/// better).
pub const KSUP: usize = 16;
/// Experts per expert job and token slots per expert.
pub const G: usize = 32;
pub const C: usize = 32;
/// The wide-slot expert jobs (`experts_c64`): experts per job, slots per
/// expert (an expert with 33-64 tokens: one wide slot at ~0.24 ms instead
/// of two of [`C`] at ~0.2 each).
pub const C2: (usize, usize) = (16, 64);
/// `experts_w`: wide slots as [`C2`], every op per expert and an output
/// per expert (5.0-5.3 ms per job against 3.1-3.4 for each of two
/// `experts_c64` jobs)
pub const C2W: (usize, usize) = (32, 64);
/// The decode set's experts per job (a token's routed experts).
const G1: usize = 10;
/// Tails of at most this many tokens go through the one-position graphs
/// one token at a time instead of a padded 256-position chunk.
pub const D1_MAX: usize = 8;

/// With the dense GPU, prefill pieces of at most this many tokens run as
/// GPU steps of up to 8 positions instead of a prefill chunk
/// (`QWEN_FLASH_STEP_MAX`; 0 = only tails of [`D1_MAX`]): a chunk costs
/// ~1.6 s however short (every 256-position graph and an expert pass per
/// layer), a two-wave step of 8 ~23 ms per position. Default 96: a piece
/// of ~70 tokens 2.46 -> 1.58 s; 112 tokens even (2.71 / 2.76 s), 121
/// 2.81 -> 2.99, 171 2.81 -> 3.87.
fn step_max() -> usize {
	std::env::var("QWEN_FLASH_STEP_MAX")
		.ok()
		.and_then(|v| v.parse().ok())
		.unwrap_or(96)
		.max(D1_MAX)
}
const HCW: usize = HC * W;
const H: usize = 48;
const HK: usize = 16;
const D: usize = 128;
const NQKV: usize = 2 * HK * D + H * D;
const HQ: usize = 24;
const HKV: usize = 2;
const DA: usize = 256;
const RD: usize = 64;
const IH: usize = 4;
const ID: usize = 128;
const NE: usize = 512;
const TOPK: usize = 10;
const FE: usize = 640;
/// Scale of `o` into a one-position set's `post` (prefill sets:
/// [`super::OSCALE_P`], [`oscale`]).
const OSCALE: f32 = 16.0;

/// Scale of the recurrence's output into `post` of a set of `s` positions.
fn oscale(s: usize) -> f32 {
	if s > 1 {
		super::OSCALE_P as f32
	} else {
		OSCALE
	}
}
const BACK: usize = 3 * ngram::N;

/// [`FlashEngine::ple_all`] (`QWEN_FLASH_PLE_FIRST=0`: the PLE graph in
/// each sub-chunk's `pre_prep`, the A/B).
fn ple_first() -> bool {
	std::env::var("QWEN_FLASH_PLE_FIRST").map_or(true, |v| v != "0")
}

/// Expert passes of at most this many tokens also use the slot kinds
/// narrower than the set's ([`Set::big`]).
const NS_MAX_TOKENS: usize = 1024;

fn is_attn(l: usize) -> bool {
	l % 4 == 3
}

/// A prefill chunk's attention on a device (the scan device's adapter,
/// `offload::FlashAttn`), keeping the cores free.
pub trait PrefillAttn {
	/// Attention layer `a` (`l / 4`), queries at `[p0, p0 + n)`: `k` / `v`
	/// the layer's whole cache `[p0 + n][2][256]` f16, as contiguous runs of
	/// rows in order (the device mirrors the rows it lacks), `q` `[n][24][256]` f16, `sel` `[n][513]`: per query
	/// `u32::MAX` (every position from `kmin`) or the number of listed
	/// blocks (at most 512) and the blocks, ascending, then the tail ->
	/// `out` `[n][24][256]` f16.
	#[allow(clippy::too_many_arguments)]
	fn attend(
		&mut self,
		a: usize,
		k: &[&[u16]],
		v: &[&[u16]],
		p0: usize,
		n: usize,
		q: &[u16],
		sel: &[u32],
		out: &mut [u16],
	) -> Result<(), String> {
		self.attend_begin(a, k, v, p0, n, q, sel, 0)?;
		self.attend_end(out, None)
	}
	/// [`PrefillAttn::attend`] submitted and left running (one at a time).
	#[allow(clippy::too_many_arguments)]
	fn attend_begin(
		&mut self,
		a: usize,
		k: &[&[u16]],
		v: &[&[u16]],
		p0: usize,
		n: usize,
		q: &[u16],
		sel: &[u32],
		kmin: usize,
	) -> Result<(), String>;
	/// Wait for [`PrefillAttn::attend_begin`]'s work: its `out` and, if
	/// asked, `lse` `[n][24]` (max + ln sum of the scaled scores: the
	/// merge with [`super::far::NpuFar`]'s blocks).
	fn attend_end(&mut self, out: &mut [u16], lse: Option<&mut [f32]>) -> Result<(), String>;
	/// The engine's cache was reset: forget the mirrored rows.
	fn invalidate(&mut self);
	/// Host seconds and bytes spent mirroring rows since the last call
	/// (`QWEN_NPU_TIMING`).
	fn take_fill(&mut self) -> (f64, u64) {
		(0.0, 0)
	}
	/// The indexer's pooled block keys the device mirrors for layer `a`;
	/// `None` when it does not score ([`PrefillAttn::score_begin`]).
	fn idx_len(&self, _a: usize) -> Option<usize> {
		None
	}
	/// Set layer `a`'s pooled keys from block `at` (at most
	/// [`PrefillAttn::idx_len`]) on: `rows` f32 `[..][128]`.
	fn idx_append(&mut self, _a: usize, _at: usize, _rows: &[f32]) -> Result<(), String> {
		Err("prefill attention: no indexer".into())
	}
	/// Queue the indexer's scores of layer `a`'s first `nb` pooled blocks
	/// against `n` queries `iq` `[n][4][128]` f16 (beside one
	/// [`PrefillAttn::attend_begin`]).
	fn score_begin(&mut self, _a: usize, _iq: &[u16], _n: usize, _nb: usize) -> Result<(), String> {
		Err("prefill attention: no indexer".into())
	}
	/// Wait for [`PrefillAttn::score_begin`]'s scores: `(layer, queries,
	/// blocks, row stride, scores)`, each score's f32 bits (>= +0).
	#[allow(clippy::type_complexity)]
	fn score_end(&mut self) -> Result<Option<(usize, usize, usize, usize, &[u32])>, String> {
		Ok(None)
	}
}

/// A delta layer's attention hyper-connection in prefill on a device (the
/// scan device's adapter, `offload::HcPre`), for the `delta_pre_m` graphs.
pub trait PrefillHc {
	/// Delta layer `di` on the first `n` positions of the streams `x`
	/// `[..][10240]` f16, submitted and left running (one at a time: a
	/// second one waits for the first). `x` may be read in place until
	/// [`PrefillHc::end`]: an NPU host buffer (fixed address, alive as long
	/// as the engine), not written before then.
	fn begin(&mut self, di: usize, x: &[u16], n: usize) -> Result<(), String>;
	/// Wait for [`PrefillHc::begin`]'s work: the mixed input `mixed`
	/// `[n][2560]` and the injection weights `inj` `[n][4]`, f16.
	fn end(&mut self, mixed: &mut [u16], inj: &mut [u16]) -> Result<(), String>;
}

/// Makes a [`PrefillExp`] ([`FlashEngine::load_with`]: called on a loader
/// thread).
pub type PrefillExpNew = Box<dyn FnOnce() -> Result<Box<dyn PrefillExp + Send>, String> + Send>;

/// A share of a prefill's routed experts on a device (the scan device's
/// adapter, `offload::Experts`) while the NPU runs the rest.
pub trait PrefillExp {
	/// Layer `l`'s expert pools as the NPU binds them (`cpuexp::Pool`);
	/// the memory outlives the engine's use of `self`.
	fn add_layer(
		&mut self,
		l: usize,
		gu_w: &[u8],
		gu_s: &[u16],
		d_w: &[u8],
		d_s: &[u16],
		d_pre: &[u16],
	) -> Result<(), String>;
	/// Layer `l`'s experts on the MoE input rows `x` (`[tokens][2560]` f16
	/// as bytes) for `slots` (expert, its tokens: rows of `x`, at most 64),
	/// submitted and left running.
	fn begin(&mut self, l: usize, x: &[u8], slots: &[(usize, &[u32])]) -> Result<(), String>;
	/// Wait for [`PrefillExp::begin`]'s work: each slot row's output
	/// `[rows][2560]` f16 (slot order, then token order), the row scale
	/// applied, not the routing weight.
	fn end(&mut self) -> Result<&[u16], String>;
	/// The device time of the work the last [`PrefillExp::end`] waited for,
	/// ms (for the split's calibration; `None`: not measured).
	fn last_ms(&self) -> Option<f64> {
		None
	}
}

/// `QWEN_FLASH_EXP_GPU=F`: give [`PrefillExp`] a fixed share `F` of each
/// slot kind's slots (in whole NPU jobs; `0` = all on the NPU, the A/B)
/// instead of the default balance by estimated time
/// ([`npu_job_ms`], [`exp_gpu_slot_ms`]); of the experts below
/// [`exp_gpu_ne`] only. Before the balance the default was 0.35.
fn exp_gpu_frac() -> Option<f64> {
	std::env::var("QWEN_FLASH_EXP_GPU")
		.ok()
		.and_then(|v| v.parse::<f64>().ok())
		.map(|f| f.clamp(0.0, 1.0))
}

/// The NPU's time for one expert job of slots of `c` tokens (32 experts),
/// ms: from the `QWEN_FLASH_MOE_LOG` lines (run time against jobs per
/// layer at 432 and 2274 tokens; the 64-token kind computes twice the
/// rows; the narrow kinds cost about the same whatever their width). The
/// cost *shape*: the balance multiplies it by the calibrated NPU factor
/// ([`EXP_CAL`], default 1).
fn npu_job_ms(c: usize) -> f64 {
	if c > 32 {
		6.2
	} else {
		3.3
	}
}

/// The device's time for one slot of `n` tokens alone, ms: `expbench` with
/// `EXPBENCH_CHUNK`, per kernel size (a slot computes all the tokens of
/// its size's workgroup). The balance multiplies it by the calibrated
/// device factor ([`EXP_CAL`], default 1.5: the NPU's expert jobs read
/// DRAM at the same time; at 2274 tokens the engine's calls took ~1.5x
/// this); per call [`EXP_GPU_CALL_MS`] (the int8 input, the submission).
fn exp_gpu_slot_ms(n: usize) -> f64 {
	match n {
		0..=8 => 0.058,
		9..=16 => 0.11,
		17..=32 => 0.155,
		_ => 0.26,
	}
}
const EXP_GPU_CALL_MS: f64 = 1.0;

/// The expert split's calibration (`quant::calib`): the NPU factor on
/// [`npu_job_ms`] and the device factor on [`exp_gpu_slot_ms`], learned
/// from each layer's NPU run time and the device's GPU timestamps over
/// [`EXP_CAL_NEED`] layers (two prefill pieces), then frozen until the
/// executable or the expert graphs change.
const EXP_CAL: (&str, [f64; 2]) = ("flash_experts", [1.0, 1.5]);
const EXP_CAL_NEED: u64 = 96;

/// `QWEN_FLASH_EXP_GPU_NE=N`: the experts of each layer [`PrefillExp`]
/// gets (`[0, N)`, default 224 of 512). Only their pools are imported,
/// 0.110 GiB an expert over the 48 layers.
///
/// What the NPU, the iGPU and the GTX 1060 keep resident in system memory
/// counts against one WDDM limit for all adapters together, VidMm's system
/// partition (dxdiag's "Shared Memory": by default 80 % of RAM, 101.98 GiB
/// with 128 GiB; `GraphicsDrivers\MemoryManager`'s
/// `SystemPartitionCommitLimitPercentageMax` = 90 raises it to RAM - 16
/// GB, 111.5 GiB, where 360 and 400 run; the Intel driver's "Shared GPU/NPU
/// Memory Override" sets only the minimum and leaves it): the NPU holds 60.7
/// GiB (the pools and its graphs), the iGPU its imports plus ~4 GiB, and
/// the prefill attention's mirrors on the iGPU grow 24.4 KiB a position
/// (6.1 GiB at the 262144-position context). At 320 the load held 98.3
/// GiB and a 117k-token conversation 101.2: the GTX 1060's VRAM was
/// demoted, its device lost when built again, then the NPU's
/// (`ZE_RESULT_ERROR_DEVICE_LOST`, every later request failed). At 224
/// the load holds 87.6 GiB, the full context ~94, prefill speed the same
/// within 1 % (2 interleaved rounds, 19k tokens 248.5 / 248.0 -> 247.7 /
/// 245.5 tok/s, 81k 238.0 / 238.4 -> 237.7 / 238.1).
/// The whole 60 GB on the iGPU (46 GB too) lost its device at the first
/// expert call.
fn exp_gpu_ne() -> usize {
	std::env::var("QWEN_FLASH_EXP_GPU_NE")
		.ok()
		.and_then(|v| v.parse().ok())
		.unwrap_or(224usize)
		.min(NE)
}

/// The top-[`TOPK`] experts of one position's router logits with their
/// renormalized softmax weights.
/// `QWEN_FLASH_SEL_DUMP=FILE`: every routing decision appended, one record
/// per layer pass (the positions whose experts one pass reads together):
/// u16 `layer | 0x8000 if prefill`, u16 position count, then per position
/// its [`TOPK`] expert ids as u16 (a study of expert reuse).
fn sel_dump<'a>(l: usize, prefill: bool, sels: impl Iterator<Item = &'a [(usize, f32)]>) {
	use std::io::Write;
	static F: std::sync::OnceLock<Option<std::sync::Mutex<std::fs::File>>> =
		std::sync::OnceLock::new();
	let f = F.get_or_init(|| {
		let p = std::env::var_os("QWEN_FLASH_SEL_DUMP")?;
		Some(std::sync::Mutex::new(std::fs::File::create(p).ok()?))
	});
	if let Some(f) = f {
		let mut b = Vec::new();
		let mut n = 0u16;
		for sel in sels.filter(|s| !s.is_empty()) {
			for &(e, _) in sel {
				b.extend((e as u16).to_le_bytes());
			}
			n += 1;
		}
		let tag = l as u16 | if prefill { 0x8000 } else { 0 };
		let mut r = tag.to_le_bytes().to_vec();
		r.extend(n.to_le_bytes());
		r.extend(b);
		let _ = f.lock().unwrap().write_all(&r);
	}
}

/// `a . b` (the lengths are multiples of 8), AVX2 + FMA where present.
fn dot8(a: &[f32], b: &[f32]) -> f32 {
	if std::is_x86_feature_detected!("fma") {
		// SAFETY: AVX2 / FMA checked
		return unsafe { dot8_fma(a, b) };
	}
	let mut acc = [0f32; 8];
	for (x, y) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
		for i in 0..8 {
			acc[i] += x[i] * y[i];
		}
	}
	acc.iter().sum()
}

/// # Safety
/// AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
unsafe fn dot8_fma(a: &[f32], b: &[f32]) -> f32 {
	use core::arch::x86_64::*;
	let n = a.len().min(b.len());
	let (mut s0, mut s1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
	let mut i = 0;
	while i + 16 <= n {
		s0 = _mm256_fmadd_ps(
			_mm256_loadu_ps(a.as_ptr().add(i)),
			_mm256_loadu_ps(b.as_ptr().add(i)),
			s0,
		);
		s1 = _mm256_fmadd_ps(
			_mm256_loadu_ps(a.as_ptr().add(i + 8)),
			_mm256_loadu_ps(b.as_ptr().add(i + 8)),
			s1,
		);
		i += 16;
	}
	if i < n {
		s0 = _mm256_fmadd_ps(
			_mm256_loadu_ps(a.as_ptr().add(i)),
			_mm256_loadu_ps(b.as_ptr().add(i)),
			s0,
		);
	}
	let s = _mm256_add_ps(s0, s1);
	let h = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
	let h = _mm_add_ps(h, _mm_movehl_ps(h, h));
	let h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
	_mm_cvtss_f32(h)
}

/// `acc += w b` over [`DA`] values, AVX2 + FMA where present.
fn axpy(acc: &mut [f32; DA], w: f32, b: &[f32]) {
	if std::is_x86_feature_detected!("fma") {
		// SAFETY: AVX2 / FMA checked; b has DA values
		unsafe { axpy_fma(acc, w, &b[..DA]) };
		return;
	}
	for (a, b) in acc.iter_mut().zip(b) {
		*a += w * b;
	}
}

/// # Safety
/// AVX2 and FMA; `b` has [`DA`] values.
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_fma(acc: &mut [f32; DA], w: f32, b: &[f32]) {
	use core::arch::x86_64::*;
	let wv = _mm256_set1_ps(w);
	for i in (0..DA).step_by(8) {
		let p = acc.as_mut_ptr().add(i);
		_mm256_storeu_ps(
			p,
			_mm256_fmadd_ps(wv, _mm256_loadu_ps(b.as_ptr().add(i)), _mm256_loadu_ps(p)),
		);
	}
}

/// [`dot8`] against an f16 row (the attention cache).
fn dot_h(a: &[f32], b: &[u16]) -> f32 {
	if std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("f16c") {
		// SAFETY: AVX2 / FMA / F16C checked
		return unsafe { dot_h_fma(a, b) };
	}
	a.iter().zip(b).map(|(&x, &y)| x * unf16(y)).sum()
}

/// # Safety
/// AVX2, FMA and F16C.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_h_fma(a: &[f32], b: &[u16]) -> f32 {
	use core::arch::x86_64::*;
	let n = a.len().min(b.len());
	let (mut s0, mut s1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
	let ld = |i: usize| _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i) as *const __m128i));
	let mut i = 0;
	while i + 16 <= n {
		s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i)), ld(i), s0);
		s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 8)), ld(i + 8), s1);
		i += 16;
	}
	if i < n {
		s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i)), ld(i), s0);
	}
	// (dot8_fma's reduction order: the same sums as on the f32 cache)
	let s = _mm256_add_ps(s0, s1);
	let h = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
	let h = _mm_add_ps(h, _mm_movehl_ps(h, h));
	let h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
	_mm_cvtss_f32(h)
}

/// [`axpy`] of an f16 row (the attention cache).
fn axpy_h(acc: &mut [f32; DA], w: f32, b: &[u16]) {
	if std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("f16c") {
		// SAFETY: AVX2 / FMA / F16C checked; b has DA values
		unsafe { axpy_h_fma(acc, w, &b[..DA]) };
		return;
	}
	for (a, &b) in acc.iter_mut().zip(b) {
		*a += w * unf16(b);
	}
}

/// # Safety
/// AVX2, FMA and F16C; `b` has [`DA`] values.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn axpy_h_fma(acc: &mut [f32; DA], w: f32, b: &[u16]) {
	use core::arch::x86_64::*;
	let wv = _mm256_set1_ps(w);
	for i in (0..DA).step_by(8) {
		let p = acc.as_mut_ptr().add(i);
		let bv = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i) as *const __m128i));
		_mm256_storeu_ps(p, _mm256_fmadd_ps(wv, bv, _mm256_loadu_ps(p)));
	}
}

/// `QWEN_FLASH_MTP_CTX`: positions of a prefill the MTP head catches up
/// over, and the most rows it keeps pending (default 64).
fn mtp_ctx() -> usize {
	quant::knob::var("QWEN_FLASH_MTP_CTX")
		.and_then(|v| v.parse().ok())
		.unwrap_or(64)
}

/// `QWEN_FLASH_SPIN`: the pool's spin window during a GPU decode step,
/// in pause rounds (default 200000, ~2 ms: longer than any of the step's
/// GPU waits); `0` keeps the default window (the A/B).
fn flash_spin() -> u32 {
	std::env::var("QWEN_FLASH_SPIN")
		.ok()
		.and_then(|v| v.parse().ok())
		.unwrap_or(200_000)
}

/// `QWEN_FLASH_WAVE_PIPE` (default on): two waves' runs queued behind each
/// other ([`FlashEngine::step_waves_pipe`]); `0` = one run in flight at a
/// time, the host between them (the A/B).
fn wave_pipe() -> bool {
	quant::knob::var("QWEN_FLASH_WAVE_PIPE").is_none_or(|v| v.trim() != "0")
}

/// Where [`FlashEngine::pipe_layers`] calls its hook.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Hook {
	/// wave A's first run of a layer queued
	Begun,
	/// wave A's experts sent to the cores (the GPU idle unless B runs)
	Experts,
	/// a layer of A alone done, nothing in flight: `true` stops
	LayerDone,
}

/// A GPU step's n-gram rows, read when first needed (at the PLE layer).
type Rows<'a> = Box<dyn FnOnce() -> Result<Vec<Vec<f32>>, String> + 'a>;

/// A GPU step's n-gram rows ([`FlashEngine::step_inputs_at`]) as f32, one
/// row of `W` per position.
fn join_rows(p: ngram::Pending) -> Result<Vec<Vec<f32>>, String> {
	Ok(p.wait()?
		.chunks_exact(W)
		.map(|r| {
			r.iter()
				.map(|&v| f32::from_bits((v as u32) << 16))
				.collect()
		})
		.collect())
}

/// `QWEN_NPU_TIMING`'s split of a pipelined step's driver work.
fn print_waves(n: usize, tw: &[f64; 9]) {
	let [a, b, c, d, e, f, g, h, i] = tw.map(|v| v * 1e3);
	eprintln!(
		"[flash]   waves of {n}: post wait {a:.1}, post read {b:.1}, route + job {c:.1}, expert wait {d:.1} (helper wake-up {h:.1}, result back {i:.1}), combine {e:.1}, run prep + submit {f:.1}, attention {g:.1}"
	);
}

/// `QWEN_FLASH_WAVES` (default 2): a GPU step of at least
/// `QWEN_FLASH_WAVES_MIN` positions (default 2: a step of 2 as two waves
/// of 1) runs as two waves ([`FlashEngine::step_waves`]); `1` as one (the
/// A/B). Returns the least positions, `usize::MAX` = off.
fn waves_min() -> usize {
	if quant::knob::var("QWEN_FLASH_WAVES").is_some_and(|v| v.trim() == "1") {
		return usize::MAX;
	}
	quant::knob::var("QWEN_FLASH_WAVES_MIN")
		.and_then(|v| v.parse().ok())
		.unwrap_or(2)
		.max(2)
}

/// `QWEN_FLASH_MTP_N`: MTP drafts per step (default 3).
fn mtp_n() -> usize {
	quant::knob::var("QWEN_FLASH_MTP_N")
		.and_then(|v| v.parse().ok())
		.unwrap_or(3)
}

/// `QWEN_FLASH_MTP_PMIN`: a chain stops before the draft at which the
/// head's probability of all its drafts falls below this (default 0.2).
fn mtp_pmin() -> f32 {
	quant::knob::var("QWEN_FLASH_MTP_PMIN")
		.and_then(|v| v.parse().ok())
		.unwrap_or(0.2)
}

/// `QWEN_FLASH_MTP_EARLY`: stop a chain once the cost model would drop
/// every longer one (default on; `0` runs the whole chain, the A/B).
fn mtp_early() -> bool {
	quant::knob::var("QWEN_FLASH_MTP_EARLY").is_none_or(|v| v != "0")
}

/// `QWEN_FLASH_MTP_PSCALE`: the factor on each MTP draft's probability
/// (default 0.8): the head is overconfident; on prose its drafts of
/// probability 0.9-1.0 were accepted 75 % of the time, 0.8-0.9 62 %.
fn mtp_pscale() -> f32 {
	quant::knob::var("QWEN_FLASH_MTP_PSCALE")
		.and_then(|v| v.parse().ok())
		.unwrap_or(0.8)
}

/// `QWEN_FLASH_COUPLED_TAU` (default 1): with coupled sampling the MTP
/// head's draw is taken at the sampler's temperature times this; above 1
/// it flattens a head that is sharper than the model, so its coupled draw
/// follows the model's more often. Drafts only: lossless.
fn coupled_tau() -> f32 {
	quant::knob::var("QWEN_FLASH_COUPLED_TAU")
		.and_then(|v| v.trim().parse().ok())
		.filter(|&v: &f32| v > 0.0)
		.unwrap_or(1.0)
}

/// `QWEN_FLASH_MTP_CAL=a,b`: the probability that an MTP draft is
/// accepted given the ones before it, from the head's top probability `p`:
/// `sigmoid(a + b ln(1 - p))`, `1 - p` clamped at 1e-6, in place of
/// `QWEN_FLASH_MTP_PSCALE` x `p` (default off). `-1.2,-0.374` is a
/// logistic fit over 991 drafts of four 256-token answers, code and prose
/// (the depth in the chain added nothing): the head's `p` is not a
/// probability of acceptance, drafts at 0.5-0.9 were accepted ~33 % of the
/// time, at 0.99 ~63 %, past 0.999 ~85 %. With the measured step costs
/// (`62,90,112,...`) it decoded as fast as the default over six prompts
/// (17.72 vs 17.77 tok/s, session 48): the old cost vector's longer
/// steps had made up for the overconfidence.
fn mtp_cal() -> Option<(f32, f32)> {
	let v = quant::knob::var("QWEN_FLASH_MTP_CAL").unwrap_or_else(|| "0".into());
	let c: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
	(c.len() == 2).then(|| (c[0], c[1]))
}

/// The draft's acceptance probability by [`mtp_cal`] (or `pscale` x `p`)
/// from the head's softmax sum relative to its top logit (`p = 1 / sum`),
/// and the most it can be (the bound [`mtp_may_grow`] needs).
fn mtp_accept(cal: Option<(f32, f32)>, pscale: f32, sum: f32) -> (f32, f32) {
	match cal {
		Some((a, b)) => {
			let f = |q: f32| 1.0 / (1.0 + (-(a + b * q.max(1e-6).ln())).exp());
			(f((sum - 1.0) / sum), f(0.0))
		}
		None => (pscale / sum, pscale),
	}
}

/// `QWEN_FLASH_MTP_COST`: the cost of a step of 1, 2, ... positions plus
/// its drafts (ms, comma-separated; the last one repeats). Of a chain the
/// engine keeps the drafts that maximize expected tokens / cost (by the
/// head's probabilities). Default measured on the GTX 1060 decode path
///; `0`: keep the whole chain.
fn mtp_cost() -> Option<Vec<f32>> {
	let v = quant::knob::var("QWEN_FLASH_MTP_COST")
		.unwrap_or_else(|| "89,143,190,235,280,325,370,415".into());
	let c: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
	(c.len() > 1 && c.iter().all(|&x| x > 0.0)).then_some(c)
}

/// `QWEN_FLASH_MTP_AUTO=1`: the step costs measured at run time
/// ([`CostMeter`]) in place of [`mtp_cost`]'s table, which stays the prior
/// for widths not measured yet (scaled by the measured ones). `0` (the
/// default) keeps the table.
fn mtp_auto() -> bool {
	quant::knob::var("QWEN_FLASH_MTP_AUTO").is_some_and(|v| v != "0")
}

/// A running mean that turns into an exponential one (weight 1 / 8) after
/// eight samples; a sample over three times the mean (a host stall) counts
/// as three times the mean.
#[derive(Clone, Copy, Default, Debug)]
struct Ema {
	v: f64,
	n: u32,
}

impl Ema {
	fn add(&mut self, x: f64) {
		let x = if self.n >= 3 { x.min(3.0 * self.v) } else { x };
		self.n += 1;
		let a = (1.0 / self.n as f64).max(1.0 / 8.0);
		self.v += a * (x - self.v);
	}
	fn get(&self, min: u32) -> Option<f64> {
		(self.n >= min).then_some(self.v)
	}
}

/// What the cost table of [`mtp_keep`] measures, measured on the running
/// engine so it follows depth and text: per width k the wall time of a
/// [`FlashEngine::verify`] of k positions and the [`FlashEngine::commit`]
/// after it, and of a [`FlashEngine::drafts`] call its catch-up with the
/// first head pass and each further draft (MTP layer + head). The cost of
/// width k is the verify of k plus the chain of k - 1 drafts that makes
/// its drafts (one head pass for k = 1), as `--tune-profile` times it.
#[derive(Default, Debug)]
struct CostMeter {
	verify: [Ema; 8],
	first: Ema,
	pass: Ema,
	/// the last verify's width and ms, until its commit (or the next
	/// verify) closes it
	open: Option<(usize, f64)>,
	/// verify steps since the last log line
	since_log: u32,
}

impl CostMeter {
	fn close(&mut self, extra_ms: f64) {
		if let Some((k, ms)) = self.open.take() {
			self.verify[k - 1].add(ms + extra_ms);
		}
	}

	/// The measured cost of width `k` (1-based), when all its parts have
	/// at least `min` samples.
	fn measured(&self, k: usize, min: u32) -> Option<f64> {
		let v = self.verify[k - 1].get(min)?;
		let f = self.first.get(min)?;
		let d = if k >= 3 {
			(k - 2) as f64 * self.pass.get(min)?
		} else {
			0.0
		};
		Some(v + f + d)
	}

	/// `prior` with each width measured often enough replaced, the rest
	/// scaled by the measured widths' total over their prior total.
	fn costs(&self, prior: &[f32]) -> Vec<f32> {
		let n = prior.len().max(self.verify.len());
		let pri = |k: usize| prior[(k - 1).min(prior.len() - 1)] as f64;
		let m: Vec<Option<f64>> = (1..=n)
			.map(|k| {
				(k <= self.verify.len())
					.then(|| self.measured(k, 4))
					.flatten()
			})
			.collect();
		let (sm, sp) = m
			.iter()
			.enumerate()
			.filter_map(|(i, v)| v.map(|v| (v, pri(i + 1))))
			.fold((0.0, 0.0), |a, (v, p)| (a.0 + v, a.1 + p));
		let f = if sp > 0.0 { sm / sp } else { 1.0 };
		m.iter()
			.enumerate()
			.map(|(i, v)| v.unwrap_or(pri(i + 1) * f) as f32)
			.collect()
	}
}

/// Of a chain with the probabilities `ps` (each that every draft up to it
/// is accepted), the number of drafts with the most expected tokens per
/// unit of step cost `c` (the first strict maximum).
fn mtp_keep(ps: &[f32], c: &[f32]) -> usize {
	let mut best = (0, 1.0 / c[0]);
	let mut e = 1f32;
	for (i, p) in ps.iter().enumerate() {
		e += p;
		let r = e / c[(i + 1).min(c.len() - 1)];
		if r > best.1 {
			best = (i + 1, r);
		}
	}
	best.0
}

/// False when no chain grown from `ps` by up to `room` drafts can change
/// [`mtp_keep`]'s choice: every further draft's probability is at most the
/// last one's times `qmax` ([`mtp_accept`]'s bound: `pscale` over a
/// softmax sum of at least 1, or the calibration at `p = 1`), so the
/// expected tokens of a longer prefix are bounded. Lets the chain stop
/// before running drafts the cost model would drop.
fn mtp_may_grow(ps: &[f32], c: &[f32], room: usize, qmax: f32) -> bool {
	let Some(&last) = ps.last() else {
		return room > 0;
	};
	let k = mtp_keep(ps, c);
	let best = (1.0 + ps[..k].iter().sum::<f32>()) / c[k.min(c.len() - 1)];
	let (mut e, mut q) = (1.0 + ps.iter().sum::<f32>(), last);
	for m in 1..=room {
		q *= qmax;
		e += q;
		// a margin over the rounding of `mtp_keep`'s sums
		if e * 1.001 / c[(ps.len() + m).min(c.len() - 1)] > best {
			return true;
		}
	}
	false
}

pub(super) fn route(lr: &[f32]) -> Vec<(usize, f32)> {
	let m = lr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
	let p: Vec<f32> = lr.iter().map(|v| (v - m).exp()).collect();
	let mut idx: Vec<usize> = (0..NE).collect();
	idx.select_nth_unstable_by(TOPK - 1, |&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
	let sum: f32 = idx[..TOPK].iter().map(|&i| p[i]).sum();
	idx[..TOPK].iter().map(|&i| (i, p[i] / sum)).collect()
}

/// A Flash-Next `.npu` file (`arch = flash`): `graphs` (the compiled
/// blobs), `weights` (the `fl{L}.*.in` files), `ngram`, `emb`, `head` (the
/// lm_head as f16), `tokenizer`, optional `scan`.
#[derive(Default, Debug, Clone)]
pub struct Config {
	pub graphs: PathBuf,
	pub weights: PathBuf,
	pub ngram: PathBuf,
	pub emb: PathBuf,
	pub head: PathBuf,
	pub tokenizer: PathBuf,
	pub scan: Option<String>,
	/// `dense = ADAPTER`: a decode step's dense layers on that GPU
	/// ([`super::gpudec`]) instead of the one-position graphs
	pub dense: Option<String>,
	/// `vision = DIR`: the vision tower's set (`crate::vis`, out width
	/// [`W`]); with it the engine takes images ([`FlashEngine::feed_mm`]).
	pub vision: Option<PathBuf>,
	/// `mtp = DIR`: the MTP head's files (`npubuild`'s `mtp` step); with
	/// `dense` it drafts for [`FlashEngine::verify`] ([`super::mtp`]).
	pub mtp: Option<PathBuf>,
}

impl Config {
	/// True when the `.npu` file at `path` describes Flash-Next.
	pub fn is_flash(path: &str) -> bool {
		std::fs::read_to_string(path)
			.map(|t| {
				t.lines().any(|l| {
					l.split_once('=')
						.is_some_and(|(k, v)| k.trim() == "arch" && v.trim() == "flash")
				})
			})
			.unwrap_or(false)
	}

	pub fn parse(path: &str) -> Result<Config, String> {
		let t = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
		let dir = Path::new(path).parent().unwrap_or(Path::new("."));
		let mut c = Config::default();
		for line in t.lines() {
			let line = line.trim();
			if line.is_empty() || line.starts_with('#') {
				continue;
			}
			let (k, v) = line
				.split_once('=')
				.ok_or(format!("{path}: bad line {line}"))?;
			let (k, v) = (k.trim(), v.trim());
			let p = if Path::new(v).is_relative() {
				dir.join(v)
			} else {
				PathBuf::from(v)
			};
			match k {
				"arch" => {}
				"graphs" => c.graphs = p,
				"weights" => c.weights = p,
				"ngram" => c.ngram = p,
				"emb" => c.emb = p,
				"head" => c.head = p,
				"tokenizer" => c.tokenizer = p,
				"scan" => c.scan = Some(v.to_string()),
				"dense" => c.dense = Some(v.to_string()),
				"vision" => c.vision = Some(p),
				"mtp" => c.mtp = Some(p),
				k => return Err(format!("{path}: unknown key {k}")),
			}
		}
		Ok(c)
	}
}

fn f16(x: f32) -> u16 {
	crate::ir::f16_bits(x)
}

fn unf16(h: u16) -> f32 {
	crate::ir::f16_to_f32(h)
}

/// [`unf16`] of a slice (F16C: the same bits).
fn unf16v(h: &[u16]) -> Vec<f32> {
	let mut v = vec![0f32; h.len()];
	crate::ir::f16s_to_f32(u8s(h), &mut v, 1.0);
	v
}

/// [`f16`] of `src` into the front of `dst` (F16C: the same bits).
fn f16_into(dst: &mut [u16], src: &[f32]) {
	let n = src.len().min(dst.len());
	crate::ir::f32s_to_f16(&src[..n], u8sm(&mut dst[..n]), 1.0);
}

/// Delta layer `l`'s gated-norm factors as `post_delta_a`'s kernel takes
/// them (`graphs::post_x1` with `GN_FOLD`): `[2][6144]`, per channel
/// `gnw * out.pre` (rounded to f16, as the graph's Multiply), then `z.s`.
fn gn_factors(w: &Path, l: usize) -> Result<Vec<f32>, String> {
	let rd = |n: &str, len: usize| -> Result<Vec<f32>, String> {
		let p = w.join(format!("fl{l}.{n}.in"));
		let v: Vec<f32> = read(&p)?
			.chunks_exact(2)
			.map(|c| unf16(u16::from_le_bytes([c[0], c[1]])))
			.collect();
		if v.len() != len {
			return Err(format!("{}: {} values, wants {len}", p.display(), v.len()));
		}
		Ok(v)
	};
	let (gw, op, zs) = (rd("gnw", D)?, rd("out.pre", H * D)?, rd("z.s", H * D)?);
	let mut f: Vec<f32> = (0..H * D).map(|i| unf16(f16(op[i] * gw[i % D]))).collect();
	f.extend_from_slice(&zs);
	Ok(f)
}

/// Delta layer `l`'s conv for the scan device ([`Scan::set_conv`]): the
/// taps `[10240][4]`, then `qkv`'s row scale `[10240]` (the conv runs on
/// `qkv * qs`, as the window holds it).
fn conv_taps(w: &Path, l: usize) -> Result<Vec<f32>, String> {
	let rd = |n: &str, len: usize| -> Result<Vec<f32>, String> {
		let p = w.join(format!("fl{l}.{n}.in"));
		let v: Vec<f32> = read(&p)?
			.chunks_exact(2)
			.map(|c| unf16(u16::from_le_bytes([c[0], c[1]])))
			.collect();
		if v.len() != len {
			return Err(format!("{}: {} values, wants {len}", p.display(), v.len()));
		}
		Ok(v)
	};
	let (cw, qs) = (rd("conv", NQKV * 4)?, rd("qkv.s", NQKV)?);
	let mut f = cw;
	f.extend_from_slice(&qs);
	Ok(f)
}

/// The scan device's conv ([`Scan::conv_next`], `conv_npu.glsl`) on the
/// host, for `n` positions: from `cpast` [3][10240] and the raw `qkv`
/// [n][10240] into `qn`, `kn` [n][2048] and `v` [n][6144] (`w` from
/// [`conv_taps`]).
fn conv_host(
	w: &[f32],
	cpast: &[u16],
	qkv: &[u16],
	n: usize,
	qn: &mut [u16],
	kn: &mut [u16],
	v: &mut [u16],
) {
	let qs = &w[4 * NQKV..];
	let x = |r: usize, c: usize| -> f32 {
		if r < 3 {
			unf16(cpast[r * NQKV + c])
		} else {
			unf16(qkv[(r - 3) * NQKV + c]) * qs[c]
		}
	};
	let mut y = vec![0f32; NQKV];
	let qk = HK * D;
	for t in 0..n {
		for (c, yc) in y.iter_mut().enumerate() {
			let a: f32 = (0..4).map(|k| w[4 * c + k] * x(t + k, c)).sum();
			*yc = a / (1.0 + (-a).exp());
		}
		for hr in 0..2 * HK {
			let r = &mut y[hr * D..][..D];
			let ss: f32 = r.iter().map(|a| a * a).sum();
			let inv = 1.0 / (ss + 1e-6).sqrt();
			r.iter_mut().for_each(|a| *a *= inv);
		}
		f16_into(&mut qn[t * qk..][..qk], &y[..qk]);
		f16_into(&mut kn[t * qk..][..qk], &y[qk..2 * qk]);
		f16_into(&mut v[t * H * D..][..H * D], &y[2 * qk..]);
	}
}

/// [`Scan::scan_begin_gn`]'s gated norm on the host, in place of `o` (`n`
/// positions; a sub-chunk the scan device does not take).
fn gnorm_rot(o: &mut [u16], z: &[u16], w: &[f32], n: usize, oscale: f32) {
	let eps = 1e-6 * oscale * oscale;
	let hd = H * D;
	let mut y = vec![0f32; hd];
	for t in 0..n {
		let (ot, zt) = (&mut o[t * hd..][..hd], &z[t * hd..][..hd]);
		for h in 0..H {
			let r = &ot[h * D..][..D];
			let ss: f32 = r.iter().map(|&x| unf16(x) * unf16(x)).sum();
			let inv = 1.0 / (ss / D as f32 + eps).sqrt();
			for d in 0..D {
				let i = h * D + d;
				let g = 1.0 / (1.0 + (-(unf16(zt[i]) * w[hd + i])).exp());
				y[i] = unf16(ot[i]) * inv * w[i] * g;
			}
		}
		for b in y.chunks_exact_mut(256) {
			fwht(b);
		}
		for (a, &b) in ot.iter_mut().zip(&y) {
			*a = f16(b / 16.0);
		}
	}
}

/// In-place Walsh-Hadamard transform (Sylvester order, unnormalized):
/// `graphs::rotate`'s block times `sqrt(len)`.
fn fwht(x: &mut [f32]) {
	let mut h = 1;
	while h < x.len() {
		for i in (0..x.len()).step_by(2 * h) {
			for j in i..i + h {
				let (a, b) = (x[j], x[j + h]);
				x[j] = a + b;
				x[j + h] = a - b;
			}
		}
		h *= 2;
	}
}

fn read(p: &Path) -> Result<Vec<u8>, String> {
	std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))
}

/// A host buffer's contents as f16 / i32 slices.
fn h16<'a>(b: &'a HostBuf<'_>) -> &'a [u16] {
	let s = b.as_slice();
	// SAFETY: host allocations are 4 KiB aligned; f16 is plain data.
	unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u16, s.len() / 2) }
}

/// Per layer the expert pools' host buffers, shared with the expert
/// thread ([`ExpThread`]) for a decode step.
struct ExpBufs<const K: usize>(Vec<[HostBuf<'static>; K]>);

// SAFETY: host allocations of the engine's `'static` device, only read
// while shared (written by the load, before the engine exists).
unsafe impl<const K: usize> Send for ExpBufs<K> {}
unsafe impl<const K: usize> Sync for ExpBufs<K> {}

impl<const K: usize> std::ops::Deref for ExpBufs<K> {
	type Target = [[HostBuf<'static>; K]];
	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

/// The cores' view of every layer's expert pools.
fn pool_views<'a>(pools: &'a ExpBufs<4>, epre: &'a ExpBufs<2>) -> Vec<super::cpuexp::Pool<'a>> {
	pools
		.iter()
		.zip(epre.iter())
		.map(|(pl, pre)| super::cpuexp::Pool {
			gu_w: pl[0].as_slice(),
			gu_s: h16(&pl[1]),
			d_w: pl[2].as_slice(),
			d_s: h16(&pl[3]),
			d_pre: h16(&pre[1]),
		})
		.collect()
}

/// The thread that drives the pool through a GPU step's routed experts
/// while the engine's thread drives the GPU: one for the engine's life (a
/// thread per step before, with the n-gram readers ~18 short-lived threads
/// per step). Exits when the engine is dropped.
struct ExpThread {
	tx: std::sync::mpsc::Sender<Box<dyn FnOnce() + Send>>,
}

impl ExpThread {
	fn new() -> Result<ExpThread, String> {
		let (tx, rx) = std::sync::mpsc::channel::<Box<dyn FnOnce() + Send>>();
		std::thread::Builder::new()
			.name("flash-experts".into())
			.spawn(move || {
				for f in rx {
					// a panic drops the job's channels: the step sees an error
					let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
				}
			})
			.map_err(|e| format!("flash: expert thread: {e}"))?;
		Ok(ExpThread { tx })
	}

	fn run(&self, f: Box<dyn FnOnce() + Send>) -> Result<(), String> {
		self.tx
			.send(f)
			.map_err(|_| "flash: the expert thread stopped".to_string())
	}
}

fn h16m<'a>(b: &'a mut HostBuf<'_>) -> &'a mut [u16] {
	let s = b.as_mut_slice();
	// SAFETY: as `h16`.
	unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut u16, s.len() / 2) }
}

/// `sum_i h_i m_i` of an f16 row and an f32 vector (lengths a multiple of 8).
///
/// # Safety
/// AVX2, F16C and FMA.
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn dot_f16(h: &[u16], m: &[f32]) -> f32 {
	use core::arch::x86_64::*;
	let (mut a, mut b) = (_mm256_setzero_ps(), _mm256_setzero_ps());
	let mut i = 0;
	while i + 16 <= h.len() {
		let x = _mm256_cvtph_ps(_mm_loadu_si128(h.as_ptr().add(i) as *const __m128i));
		let y = _mm256_cvtph_ps(_mm_loadu_si128(h.as_ptr().add(i + 8) as *const __m128i));
		a = _mm256_fmadd_ps(x, _mm256_loadu_ps(m.as_ptr().add(i)), a);
		b = _mm256_fmadd_ps(y, _mm256_loadu_ps(m.as_ptr().add(i + 8)), b);
		i += 16;
	}
	let mut t = [0f32; 8];
	_mm256_storeu_ps(t.as_mut_ptr(), _mm256_add_ps(a, b));
	t.iter().sum()
}

/// `sum_i w_i m_i` of an int8 row and an f32 vector (lengths a multiple of
/// 16).
///
/// # Safety
/// AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn dot_i8(w: &[i8], m: &[f32]) -> f32 {
	use core::arch::x86_64::*;
	let (mut a, mut b) = (_mm256_setzero_ps(), _mm256_setzero_ps());
	let mut i = 0;
	while i + 16 <= w.len() {
		let q = _mm_loadu_si128(w.as_ptr().add(i) as *const __m128i);
		let x = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q));
		let y = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q, 8)));
		a = _mm256_fmadd_ps(x, _mm256_loadu_ps(m.as_ptr().add(i)), a);
		b = _mm256_fmadd_ps(y, _mm256_loadu_ps(m.as_ptr().add(i + 8)), b);
		i += 16;
	}
	let mut t = [0f32; 8];
	_mm256_storeu_ps(t.as_mut_ptr(), _mm256_add_ps(a, b));
	t.iter().sum()
}

/// [`dot_i8`] of `w` with each of the `NP` vectors `ms` into `out`, each
/// int8 converted once: per vector the same operations in the same order.
///
/// # Safety
/// AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_i8_np<const NP: usize>(w: &[i8], ms: &[Vec<f32>], out: &mut [f32]) {
	use core::arch::x86_64::*;
	let mut a = [_mm256_setzero_ps(); NP];
	let mut b = [_mm256_setzero_ps(); NP];
	let mut i = 0;
	while i + 16 <= w.len() {
		let q = _mm_loadu_si128(w.as_ptr().add(i) as *const __m128i);
		let x = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q));
		let y = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q, 8)));
		for p in 0..NP {
			let m = ms[p].as_ptr();
			a[p] = _mm256_fmadd_ps(x, _mm256_loadu_ps(m.add(i)), a[p]);
			b[p] = _mm256_fmadd_ps(y, _mm256_loadu_ps(m.add(i + 8)), b[p]);
		}
		i += 16;
	}
	for p in 0..NP {
		let mut t = [0f32; 8];
		_mm256_storeu_ps(t.as_mut_ptr(), _mm256_add_ps(a[p], b[p]));
		out[p] = t.iter().sum();
	}
}

/// `sum_i a_i b_i` of two f32 vectors (lengths a multiple of 16).
///
/// # Safety
/// AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32(x: &[f32], m: &[f32]) -> f32 {
	use core::arch::x86_64::*;
	let (mut a, mut b) = (_mm256_setzero_ps(), _mm256_setzero_ps());
	let mut i = 0;
	while i + 16 <= x.len() {
		a = _mm256_fmadd_ps(
			_mm256_loadu_ps(x.as_ptr().add(i)),
			_mm256_loadu_ps(m.as_ptr().add(i)),
			a,
		);
		b = _mm256_fmadd_ps(
			_mm256_loadu_ps(x.as_ptr().add(i + 8)),
			_mm256_loadu_ps(m.as_ptr().add(i + 8)),
			b,
		);
		i += 16;
	}
	let mut t = [0f32; 8];
	_mm256_storeu_ps(t.as_mut_ptr(), _mm256_add_ps(a, b));
	t.iter().sum()
}

/// The dense part on the GPU `new` builds: its matrices, the int8 lm_head
/// with `head_gpu`, the MTP head's dense part from `mtp`.
fn dense_open(
	new: &super::gpudec::DenseGpuNew,
	need: u64,
	weights: &Path,
	head: &Head,
	head_gpu: bool,
	mtp: Option<&Path>,
) -> Result<super::gpudec::GpuDec, String> {
	let mut g = super::gpudec::GpuDec::load(weights, new(need)?)?;
	if let (Head::I8(c, s), true) = (head, head_gpu) {
		g.add_head(c, s)?;
	}
	if let Some(d) = mtp {
		g.add_mtp(d)?;
	}
	Ok(g)
}

/// `QWEN_GPU_RECOVER_SECS`: seconds between attempts to build the dense
/// GPU again (default 15, 0 = never).
fn dense_retry_secs() -> u64 {
	std::env::var("QWEN_GPU_RECOVER_SECS")
		.ok()
		.and_then(|v| v.parse().ok())
		.unwrap_or(15)
}

/// The next attempt, `QWEN_GPU_RECOVER_SECS` x `backoff` from now.
fn dense_retry_at(backoff: u64) -> Option<std::time::Instant> {
	let secs = dense_retry_secs() * backoff;
	(secs > 0).then(|| std::time::Instant::now() + std::time::Duration::from_secs(secs))
}

fn retry_note(backoff: u64) -> String {
	match dense_retry_secs() * backoff {
		0 => "; decode on the one-position graphs".into(),
		s => format!("; decode on the one-position graphs, next attempt in {s} s"),
	}
}

/// The lm_head's row blocks (of 16) a GPU step's head runs on the dense
/// GPU, the cores the rest meanwhile (`QWEN_FLASH_HEAD_SPLIT`; 16 = all on
/// the GPU, the A/B).
///
/// The default depends on the step's positions `np`: the cores read
/// each row once for all positions ([`head_rows_host`]), the GPU's
/// blocks cost more per position, so wider steps give the cores more
/// (`QWEN_NPU_TIMING` head medians on `ids.txt`, `QWEN_FLASH_D1=N`:
/// 1 / 2 positions best at 12, 3 / 4 at 10 (6.2 vs 7.3 ms at 4), 6 / 8
/// at 8 (10.8 vs 15.2 ms at 8)). A number sets every step's split.
fn head_split(np: usize) -> usize {
	static V: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
	let v = V.get_or_init(|| {
		std::env::var("QWEN_FLASH_HEAD_SPLIT")
			.ok()
			.and_then(|v| v.parse().ok())
	});
	v.unwrap_or(match np {
		0..=2 => 12,
		3 | 4 => 10,
		5 => 9,
		_ => 8,
	})
}

/// `QWEN_FLASH_MTP_VBLK`: the MTP drafts' vocabulary, as row blocks of
/// the GPU lm_head (`gpudec::HEAD_ROWS` = 16384 ids each) from id 0, plus
/// the special tokens ([`DRAFT_SPECIAL0`] on) on the cores; default 6
/// (98304 ids: 99.8-99.97 % of the tokens of English prose and code); 16
/// = the whole head (the A/B). Lossless: only a draft's chance of
/// acceptance depends on it, every token is sampled from the main model.
fn mtp_vblk() -> usize {
	static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
	*V.get_or_init(|| {
		std::env::var("QWEN_FLASH_MTP_VBLK")
			.ok()
			.and_then(|v| v.parse().ok())
			.unwrap_or(6)
	})
}

/// The first special token id: the tokenizer's added tokens follow its
/// 248044 BPE ids.
const DRAFT_SPECIAL0: usize = 248044;

/// The lm_head on the host: the f16 file as is, or int8 per row (half the
/// bytes a decode step reads; `QWEN_FLASH_HEAD=f16` is the A/B).
enum Head {
	F16(Vec<u16>),
	/// codes `[VOCAB][W]`, row scales
	I8(Vec<i8>, Vec<f32>),
}

/// Token `id`'s embedding row (f32) from the f16 table `emb`.
fn emb_row_from(emb: &std::fs::File, id: u32) -> Result<Vec<f32>, String> {
	let mut row = vec![0u8; W * 2];
	let mut done = 0;
	while done < row.len() {
		let r = emb
			.seek_read(&mut row[done..], id as u64 * (W as u64 * 2) + done as u64)
			.map_err(|e| format!("embedding table: {e}"))?;
		if r == 0 {
			return Err("embedding table: short read".into());
		}
		done += r;
	}
	Ok(row
		.chunks_exact(2)
		.map(|c| unf16(u16::from_le_bytes([c[0], c[1]])))
		.collect())
}

/// The host lm_head's logits of final mixer outputs `ms`, rows from `r0`
/// on (the rest left zero). Each row is read (and converted) once for up
/// to [`MIX_G`] positions, written straight into the returned rows.
fn head_rows_host(head: &Head, ms: &[Vec<f32>], r0: usize) -> Vec<Vec<f32>> {
	let mut out: Vec<Vec<f32>> = ms.iter().map(|_| vec![0f32; super::VOCAB]).collect();
	#[derive(Clone, Copy)]
	struct P(*mut f32);
	unsafe impl Send for P {}
	unsafe impl Sync for P {}
	let ps: Vec<P> = out.iter_mut().map(|o| P(o.as_mut_ptr())).collect();
	const R: usize = 256;
	par_for((super::VOCAB - r0).div_ceil(R), |b| {
		let ps = &ps;
		let mut t = [0f32; MIX_G];
		for r in r0 + b * R..(r0 + (b + 1) * R).min(super::VOCAB) {
			for (g, mg) in ms.chunks(MIX_G).enumerate() {
				// SAFETY: F16C / FMA checked at load; block b owns rows
				// [b R, (b + 1) R) of every output
				unsafe {
					head.dot_n(r, mg, &mut t[..mg.len()]);
					for (j, &v) in t[..mg.len()].iter().enumerate() {
						*ps[g * MIX_G + j].0.add(r) = v;
					}
				}
			}
		}
	});
	out
}

/// Positions per pass of the final mixer and the host lm_head.
const MIX_G: usize = 8;

impl Head {
	/// Row `r` dotted with `m`.
	///
	/// # Safety
	/// AVX2, F16C and FMA.
	unsafe fn dot(&self, r: usize, m: &[f32]) -> f32 {
		match self {
			Head::F16(h) => dot_f16(&h[r * W..][..W], m),
			Head::I8(c, s) => dot_i8(&c[r * W..][..W], m) * s[r],
		}
	}

	/// Row `r` dotted with each of `ms` (at most [`MIX_G`]) into `out`:
	/// per position the value of [`Self::dot`], an int8 row converted once.
	///
	/// # Safety
	/// AVX2, F16C and FMA.
	unsafe fn dot_n(&self, r: usize, ms: &[Vec<f32>], out: &mut [f32]) {
		match self {
			Head::F16(h) => {
				for (o, m) in out.iter_mut().zip(ms) {
					*o = dot_f16(&h[r * W..][..W], m);
				}
			}
			Head::I8(c, s) => {
				let w = &c[r * W..][..W];
				match ms.len() {
					1 => dot_i8_np::<1>(w, ms, out),
					2 => dot_i8_np::<2>(w, ms, out),
					3 => dot_i8_np::<3>(w, ms, out),
					4 => dot_i8_np::<4>(w, ms, out),
					5 => dot_i8_np::<5>(w, ms, out),
					6 => dot_i8_np::<6>(w, ms, out),
					7 => dot_i8_np::<7>(w, ms, out),
					_ => dot_i8_np::<8>(w, ms, out),
				}
				for o in out.iter_mut() {
					*o *= s[r];
				}
			}
		}
	}

	/// Symmetric int8 per row (max-abs scale) of the f16 rows `h`.
	fn quantize(h: &[u16]) -> Head {
		let nr = h.len() / W;
		let mut c = vec![0i8; nr * W];
		let mut s = vec![0f32; nr];
		#[derive(Clone, Copy)]
		struct P(*mut i8, *mut f32);
		unsafe impl Send for P {}
		unsafe impl Sync for P {}
		let p = P(c.as_mut_ptr(), s.as_mut_ptr());
		const R: usize = 1024;
		par_for(nr.div_ceil(R), |b| {
			let p = p;
			let mut row = [0f32; W];
			for r in b * R..((b + 1) * R).min(nr) {
				for (o, &v) in row.iter_mut().zip(&h[r * W..][..W]) {
					*o = unf16(v);
				}
				let m = row.iter().fold(0f32, |a, v| a.max(v.abs()));
				let sc = if m > 0.0 { m / 127.0 } else { 1.0 };
				// SAFETY: block b owns rows [b R, (b + 1) R)
				unsafe {
					*p.1.add(r) = sc;
					for (i, &v) in row.iter().enumerate() {
						*p.0.add(r * W + i) = (v / sc).round().clamp(-127.0, 127.0) as i8;
					}
				}
			}
		});
		Head::I8(c, s)
	}
}

/// The engine's runtime buffers (fixed addresses: the recorded lists and
/// the scan device's imports refer to them).
#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
enum B {
	x,
	cpast,
	valid,
	keep3,
	qn,
	kn,
	v,
	gb,
	z,
	conv_out,
	inj,
	cos,
	sin,
	q,
	gate,
	k,
	va,
	iq,
	ik,
	o,
	x1,
	moe_q,
	inj_m,
	logits,
	shared,
	e,
	cstate,
	keep9,
	xo,
	cstate_out,
	idx,
	y,
	/// `delta_pre_m`'s inputs: the hyper-connection's outputs
	mixed,
	hinj,
	/// `post_delta_b`'s input: the MoE hyper-connection's mixed input
	/// (`phm`)
	mm,
	/// `delta_pre_c`'s output: the raw qkv projection, the scan device's
	/// conv input ([`Set::conv_c`])
	qkv,
	N,
}

impl B {
	/// Bytes in a set of `s` positions, `g` experts x `c` slots per job.
	fn bytes(self, sn: usize, g: usize, c: usize) -> usize {
		use B::*;
		match self {
			x | x1 | xo => sn * HCW * 2,
			cpast | conv_out => 3 * NQKV * 2,
			valid => sn * 2,
			keep3 => 3 * 4,
			qn | kn => sn * HK * D * 2,
			v | z | o | gate => sn * H * D * 2,
			gb => sn * 2 * H * 2,
			inj | inj_m | hinj => sn * HC * 2,
			cos | sin => sn * RD * 2,
			q => sn * HQ * DA * 2,
			k | va => sn * HKV * DA * 2,
			iq => sn * IH * ID * 2,
			ik => sn * ID * 2,
			moe_q | shared | e | mixed | mm => sn * W * 2,
			qkv => sn * NQKV * 2,
			logits => sn * NE * 2,
			cstate | cstate_out => BACK * HCW * 2,
			keep9 => BACK * 4,
			idx => g * c * 4,
			y => g * c * W * 2,
			N => 0,
		}
	}
}

const ALL: [B; B::N as usize] = {
	use B::*;
	[
		x, cpast, valid, keep3, qn, kn, v, gb, z, conv_out, inj, cos, sin, q, gate, k, va, iq, ik,
		o, x1, moe_q, inj_m, logits, shared, e, cstate, keep9, xo, cstate_out, idx, y, mixed, hinj,
		mm, qkv,
	]
};

/// Which runtime buffer a graph argument is, by kind and argument name.
fn runtime_arg(kind: &str, name: &str) -> Option<B> {
	use B::*;
	let kind = kind.strip_suffix("_d1").unwrap_or(kind);
	Some(match (kind, name) {
		(_, "x") => x,
		("delta_pre", "cpast") => cpast,
		("delta_pre", "valid") => valid,
		("delta_pre", "keep") => keep3,
		("delta_pre", "qn") => qn,
		("delta_pre", "kn") => kn,
		("delta_pre", "v") => v,
		("delta_pre", "gb") => gb,
		("delta_pre", "z") => z,
		("delta_pre", "conv_out") => conv_out,
		("delta_pre", "mixed") => mixed,
		("delta_pre", "hinj") => hinj,
		("delta_pre", "qkv") => qkv,
		(_, "inj") => inj,
		("attn_pre", "cos") => cos,
		("attn_pre", "sin") => sin,
		("attn_pre", "q") => q,
		("attn_pre", "gate") => gate,
		("attn_pre", "k") => k,
		("attn_pre", "v") => va,
		("attn_pre", "iq") => iq,
		("attn_pre", "ik") => ik,
		(_, "o") => o,
		("post_delta" | "post_delta_a", "zg") => z,
		("post_delta_b", "m") => mm,
		("post_attn", "zg") => gate,
		(_, "x1") => x1,
		(_, "moe_q") => moe_q,
		(_, "inj_m") => inj_m,
		(_, "logits") => logits,
		(_, "shared") => shared,
		("ple", "e") => e,
		("ple", "cstate") => cstate,
		("ple", "keep") => keep9,
		("ple", "xo") => xo,
		("ple", "cstate_out") => cstate_out,
		("experts", "y") => y,
		_ => return None,
	})
}

/// The output arguments' names the compiler gives results (`Reshape_53`...):
/// matched to the graph's result names by order (`npubuild`'s tests check
/// them against the builders).
pub const OUTS: &[(&str, &[&str])] = &[
	(
		"delta_pre",
		&["qn", "kn", "v", "gb", "z", "conv_out", "inj"],
	),
	// the prefill form without the conv ([`Set::conv_c`])
	("delta_pre_c", &["qkv", "gb", "z", "inj"]),
	("attn_pre", &["q", "gate", "k", "v", "iq", "ik", "inj"]),
	("post_delta", &["x1", "moe_q", "inj_m", "logits", "shared"]),
	("post_attn", &["x1", "moe_q", "inj_m", "logits", "shared"]),
	("post_delta_a", &["x1"]),
	("post_delta_t", &["x1"]),
	("post_delta_b", &["moe_q", "logits", "shared"]),
	("ple", &["xo", "cstate_out"]),
	("experts", &["y"]),
	// `far::NpuFar` binds them by size (o is the larger)
	("attn_far", &["o", "lse"]),
	("attn_farm", &["o", "lse"]),
	("idx_score", &["sc"]),
];

/// An attention layer's cache (host): keys / values `[pos][2][256]` f16
/// (as the graphs produce them: f32 here held a second, lossless copy's
/// worth of memory), the indexer's raw keys `[pos][128]` f16 and its roped
/// block keys `[blocks][128]` f32; in sealed segments the prefix cache's
/// snapshots share ([`super::kvcol`]).
#[derive(Clone)]
pub(super) struct Kv {
	pub(super) k: Col<u16>,
	pub(super) v: Col<u16>,
	ik: Col<u16>,
	bk: Col<f32>,
}

impl Default for Kv {
	fn default() -> Self {
		Kv {
			k: Col::new(HKV * DA),
			v: Col::new(HKV * DA),
			ik: Col::new(ID),
			bk: Col::new(ID),
		}
	}
}

/// A sub-chunk of a chunk: its index `j` in the chunk, its streams and
/// its other runtime inputs ([`SUB_IN`]). The streams `x` are carried from
/// layer to layer in an NPU buffer of their own, which the pre and post
/// graphs read in place; a layer's dense outputs ([`CO`]) are written at
/// the sub-chunk's slot of the chunk's [`Co`] buffers ([`FlashEngine::bind_io`]).
struct Sub {
	j: usize,
	n: usize,
	pos: usize,
	x: HostBuf<'static>,
	ins: Vec<Vec<u8>>,
}

/// A prefill piece's n-gram embeddings, gathered on a background thread
/// while the piece before it runs: from the table's file a new piece's rows
/// are cold, and read in [`FlashEngine::inputs`] they took 3-4.5 s per
/// 4096 tokens with the NPU idle (the gaps between pieces). Keyed on the
/// rows, so a stale prefetch is never used.
struct Prefetch {
	rows: Vec<[u64; ngram::HEADS]>,
	e: PfData,
}

enum PfData {
	Run(ngram::Pending),
	Done(Vec<u16>),
	Failed,
}

const SUB_IN: [B; 6] = [B::e, B::valid, B::keep3, B::keep9, B::cos, B::sin];
/// The dense part's outputs a chunk keeps per sub-chunk, in [`Co`] order.
const CO: [B; 5] = [B::x1, B::inj_m, B::logits, B::shared, B::moe_q];
const CO_X1: usize = 0;
const CO_INJ: usize = 1;
const CO_LOGITS: usize = 2;
const CO_SHARED: usize = 3;
const CO_MOEQ: usize = 4;

/// A chunk's dense outputs ([`CO`]): one NPU buffer per kind, sub-chunk
/// `j`'s at `j * stride`, where the post graphs write them (their lists'
/// arguments re-pointed per sub-chunk), read in place by the MoE
/// hyper-connection, the routing, the expert pass and the combine. A
/// slot's outputs stay until the next layer's post of that sub-chunk,
/// which runs after the combine read them. `moe_q`'s slots are adjacent
/// (its stride is its size): the expert pass reads all of a chunk's rows.
struct Co {
	bufs: Vec<HostBuf<'static>>,
	sn: usize,
}

impl Co {
	fn stride(i: usize, sn: usize) -> usize {
		CO[i].bytes(sn, 0, 0).next_multiple_of(64)
	}

	/// Kind `i`'s slot of sub-chunk `j`.
	fn at(&self, i: usize, j: usize) -> &[u8] {
		let d = Self::stride(i, self.sn);
		&self.bufs[i].as_slice()[j * d..][..CO[i].bytes(self.sn, 0, 0)]
	}

	fn at_mut(&mut self, i: usize, j: usize) -> &mut [u8] {
		let d = Self::stride(i, self.sn);
		&mut self.bufs[i].as_mut_slice()[j * d..][..CO[i].bytes(self.sn, 0, 0)]
	}

	/// Kind `i`'s buffer and sub-chunk `j`'s offset in it.
	fn slot(&self, i: usize, j: usize) -> (&HostBuf<'static>, usize) {
		(&self.bufs[i], j * Self::stride(i, self.sn))
	}
}

fn u16s(b: &[u8]) -> &[u16] {
	// SAFETY: host buffers' copies, 2-byte aligned (Vec<u8> from the
	// allocator, at least 8-byte aligned)
	unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u16, b.len() / 2) }
}

fn u16sm(b: &mut [u8]) -> &mut [u16] {
	// SAFETY: as `u16s` (NPU buffers' slots at even offsets)
	unsafe { std::slice::from_raw_parts_mut(b.as_mut_ptr() as *mut u16, b.len() / 2) }
}

pub(super) fn u8s(h: &[u16]) -> &[u8] {
	// SAFETY: plain data reinterpreted as bytes
	unsafe { std::slice::from_raw_parts(h.as_ptr() as *const u8, h.len() * 2) }
}

pub(super) fn u8sm(h: &mut [u16]) -> &mut [u8] {
	// SAFETY: plain data reinterpreted as bytes
	unsafe { std::slice::from_raw_parts_mut(h.as_mut_ptr() as *mut u8, h.len() * 2) }
}

/// The streams after a layer: `x1 + inj (shared + y)` per stream; padding
/// rows zero.
fn combine(sb: &mut Sub, co: &Co, sn: usize, y: &[f32]) {
	let n = sb.n;
	let (x1, inj, sh) = (
		u16s(co.at(CO_X1, sb.j)),
		u16s(co.at(CO_INJ, sb.j)),
		u16s(co.at(CO_SHARED, sb.j)),
	);
	let x = h16m(&mut sb.x);
	#[derive(Clone, Copy)]
	struct P(*mut u16);
	unsafe impl Send for P {}
	unsafe impl Sync for P {}
	let xp = P(x.as_mut_ptr());
	par_for(sn, |t| {
		let xp = &xp;
		// SAFETY: task t owns x[t][..]
		let xt = unsafe { std::slice::from_raw_parts_mut(xp.0.add(t * HCW), HCW) };
		if t >= n {
			xt.fill(0);
			return;
		}
		for c in 0..HC {
			let j = unf16(inj[t * HC + c]);
			let o = t * HCW + c * W;
			for i in 0..W {
				let v = unf16(x1[o + i]) + j * (unf16(sh[t * W + i]) + y[t * W + i]);
				xt[c * W + i] = f16(v);
			}
		}
	});
}

/// `QWEN_FLASH_MOE_DEFER`: how much of a layer's expert pass the host
/// leaves to the next layer, done per sub-chunk just before that layer
/// first reads its streams (while the devices run the sub-chunks before
/// it): `0` none (the combine of every token right after the pass, the
/// A/B), `1` the combine, `2` (default) the combine and the gather of every
/// list. Bit-identical: per token the same sums in the same order.
fn moe_defer() -> u8 {
	std::env::var("QWEN_FLASH_MOE_DEFER")
		.ok()
		.and_then(|v| v.parse::<u8>().ok())
		.map_or(2, |v| v.min(2))
}

/// An expert pass's combine still owed to the streams of sub-chunks
/// `done..k` ([`FlashEngine::comb_to`]): per token `y` (the lists gathered
/// so far, or none) plus the sources of the lists `src`, plus the device's
/// rows, then `x1 + inj (shared + y)` per stream. The pointers are job
/// output buffers, the device's output and the layer's row scales, all
/// left alone until the next expert pass, which combines what is left
/// first.
struct Comb {
	pos0: usize,
	sn: usize,
	k: usize,
	done: usize,
	/// per token the lists gathered during the pass (`n * W`), or empty
	y: Vec<f32>,
	/// per list still to add, per token its (job, row, weight, expert
	/// whose `d.s` the row lacks or `NE`)
	src: Vec<Vec<Vec<(usize, usize, f32, usize)>>>,
	/// per token its device rows (row, weight)
	gsrc: Vec<Vec<(usize, f32)>>,
	gy: *const u16,
	ys: Vec<*const u8>,
	dsf: *const f32,
}

// SAFETY: plain pointers into buffers the engine owns, read only by the
// engine's own calls
unsafe impl Send for Comb {}
unsafe impl Sync for Comb {}

/// `y[t] += w y_k[r]` over the sources `srcs` of a token, `ys[k]` job
/// `k`'s outputs (f16 rows of `W`), `dsf` the row scales by expert.
fn moe_acc(yt: &mut [f32], srcs: &[(usize, usize, f32, usize)], ys: &[*const u8], dsf: *const f32) {
	for &(k, r, w, e) in srcs {
		// SAFETY: job k's output buffer, written by a finished list
		let yk = unsafe { std::slice::from_raw_parts(ys[k].add(r * W * 2), W * 2) };
		let mut b = [0f32; W];
		crate::ir::f16s_to_f32(yk, &mut b, 1.0);
		if e < NE {
			// SAFETY: `NE * W` row scales
			let s = unsafe { std::slice::from_raw_parts(dsf.add(e * W), W) };
			for ((a, &b), &s) in yt.iter_mut().zip(&b).zip(s) {
				*a += w * (b * s);
			}
		} else {
			for (a, &b) in yt.iter_mut().zip(&b) {
				*a += w * b;
			}
		}
	}
}

/// [`Comb`]'s work for sub-chunks `j0..j1` of `st` (in parallel over
/// their tokens).
fn comb_run(c: &Comb, st: &mut [Sub], co: &Co, j0: usize, j1: usize) {
	struct S(*mut u16, *const u16, *const u16, *const u16, usize);
	unsafe impl Send for S {}
	unsafe impl Sync for S {}
	let sn = c.sn;
	let subs: Vec<S> = st[j0..j1]
		.iter_mut()
		.map(|sb| {
			let (x1, inj, sh) = (
				u16s(co.at(CO_X1, sb.j)).as_ptr(),
				u16s(co.at(CO_INJ, sb.j)).as_ptr(),
				u16s(co.at(CO_SHARED, sb.j)).as_ptr(),
			);
			S(h16m(&mut sb.x).as_mut_ptr(), x1, inj, sh, sb.n)
		})
		.collect();
	#[derive(Clone, Copy)]
	struct PF(*const f32);
	unsafe impl Send for PF {}
	unsafe impl Sync for PF {}
	let yp = PF(c.y.as_ptr());
	let has_y = !c.y.is_empty();
	par_for((j1 - j0) * sn, |i| {
		let (subs, yp) = (&subs, &yp);
		let t = j0 * sn + i;
		let (s, tt) = (&subs[i / sn], i % sn);
		// SAFETY: task i owns row tt of its sub-chunk's streams; the
		// outputs are read only
		let xt = unsafe { std::slice::from_raw_parts_mut(s.0.add(tt * HCW), HCW) };
		if tt >= s.4 {
			xt.fill(0);
			return;
		}
		let mut yt = [0f32; W];
		if has_y {
			// SAFETY: row t of `y`, `n * W` long
			yt.copy_from_slice(unsafe { std::slice::from_raw_parts(yp.0.add(t * W), W) });
		}
		for sr in &c.src {
			moe_acc(&mut yt, &sr[t], &c.ys, c.dsf);
		}
		if let Some(gs) = c.gsrc.get(t) {
			let mut b = [0f32; W];
			for &(r, w) in gs {
				// SAFETY: the device's rows, one per (slot, token)
				let g = unsafe { std::slice::from_raw_parts(c.gy.add(r * W), W) };
				crate::ir::f16s_to_f32(u8s(g), &mut b, 1.0);
				for (a, &b) in yt.iter_mut().zip(&b) {
					*a += w * b;
				}
			}
		}
		// SAFETY: as above
		let (x1, inj, sh) = unsafe {
			(
				std::slice::from_raw_parts(s.1.add(tt * HCW), HCW),
				std::slice::from_raw_parts(s.2.add(tt * HC), HC),
				std::slice::from_raw_parts(s.3.add(tt * W), W),
			)
		};
		// F16C conversions (the bits of `unf16` / `f16`)
		let mut shy = [0f32; W];
		crate::ir::f16s_to_f32(u8s(sh), &mut shy, 1.0);
		for (a, &b) in shy.iter_mut().zip(yt.iter()) {
			*a += b;
		}
		let mut v = [0f32; W];
		for c in 0..HC {
			let j = unf16(inj[c]);
			crate::ir::f16s_to_f32(u8s(&x1[c * W..][..W]), &mut v, 1.0);
			for (a, &b) in v.iter_mut().zip(&shy) {
				*a += j * b;
			}
			crate::ir::f32s_to_f16(&v, u8sm(&mut xt[c * W..][..W]), 1.0);
		}
	});
}

/// Per-layer small vectors the host needs (the indexer's key norm).
struct HostLayer {
	ikn: Vec<f32>,
}

/// The graphs of one static size: `s` positions, expert jobs of `g`
/// experts x `c` slots, with their runtime buffers.
struct Set {
	s: usize,
	g: usize,
	c: usize,
	bufs: Vec<HostBuf<'static>>,
	/// the blobs of [`KINDS`], kept to load the graphs again
	blobs: Vec<Vec<u8>>,
	/// the graphs of [`KINDS`]: loaded only while the set is the active
	/// one (see [`Set::deactivate`])
	graphs: Vec<Graph<'static>>,
	/// the other slot widths' expert graphs: the wide ones (`experts_ws`
	/// / `experts_w` / `experts_c64`) for experts with more than `c`
	/// tokens, the narrow ones (`experts_ns8` / `experts_ns16`) for
	/// experts with at most 8 / 16; `g2` experts x `c2` slots, their
	/// blobs, whether their outputs lack the row scale `d.s` (`experts_ws`,
	/// `experts_ns*`: the gather applies it), and the graphs while active
	big: Vec<(Vec<u8>, usize, usize, bool)>,
	/// the narrow expert graph is `experts_ns` (outputs without `d.s`)
	nsc: bool,
	/// the delta layers' post in two graphs around the MoE hyper-connection
	/// (`post_delta_a` / `post_delta_b`: [`FlashEngine::delta_split`]),
	/// their blobs and the graphs while active
	split: bool,
	/// with `split`: `post_delta_t` in place of `post_delta_a` (the gated
	/// norm on the scan device, [`Scan::scan_begin_gn`])
	gn: bool,
	/// the delta layers' pre graph is `delta_pre_c`: the conv, Swish and
	/// q / k norms run on the scan device ahead of the recurrence
	/// ([`Scan::conv_next`]), the pre graph leaves the raw `qkv`
	conv_c: bool,
	sblobs: Vec<Vec<u8>>,
	/// per graph kind as [`Set::record`] names it: the weight inputs its
	/// blob reads swizzled (`{blob}.swz`, made by the compile step), which
	/// get their own buffer, swizzled after the read ([`ze::swizzle`])
	swz: HashMap<&'static str, HashSet<String>>,
	gsplit: Vec<Graph<'static>>,
	gbig: Vec<Graph<'static>>,
	/// recorded dense lists: per layer (pre, post, with `split` a delta
	/// layer's `post_delta_b` after `post_delta_a` in place of post), and
	/// layer 1's PLE
	lists: Vec<Lists>,
	/// per layer and list (pre, post, `post_delta_b`) the arguments
	/// re-pointed per sub-chunk ([`FlashEngine::bind_io`]): the streams `x`
	/// and the [`CO`] outputs, by buffer and argument index (those lists
	/// are mutable)
	ri: Vec<[Vec<(B, usize)>; 3]>,
	ple: Option<Batch<'static>>,
	/// the other buffer banks (`bufs`, `lists`, `ple`), by bank; the
	/// active bank's slot (`bk`) is `None`. Empty for one bank. A chunk's
	/// sub-chunk `j` runs on bank `j % banks` (see [`FlashEngine::chunk`]).
	alt: Vec<Option<Bank>>,
	bk: usize,
	/// per expert job of a layer: its tokens' inputs and its outputs
	/// (grown on demand; the jobs of a layer run as one list)
	jx: Vec<HostBuf<'static>>,
	jy: Vec<HostBuf<'static>>,
	/// sub-chunks of `s` positions per expert pass
	k: usize,
}

type Lists = (Batch<'static>, Batch<'static>, Option<Batch<'static>>);

type Bank = (Vec<HostBuf<'static>>, Vec<Lists>, Option<Batch<'static>>);

/// A set's graph kinds, in [`Set::graphs`] order.
const KINDS: [&str; 6] = [
	"delta_pre",
	"attn_pre",
	"post_delta",
	"post_attn",
	"ple",
	"experts",
];

/// The layers' weight buffers by (layer, file name, read swizzled).
type WBufs = HashMap<(usize, String, bool), HostBuf<'static>>;

/// The weight inputs the graph `blob` reads swizzled: the names in the
/// compile step's `.swz` beside it (none without one).
fn swz_names(blob: &Path) -> HashSet<String> {
	std::fs::read_to_string(blob.with_extension("swz"))
		.map(|t| t.split_whitespace().map(str::to_string).collect())
		.unwrap_or_default()
}

impl Set {
	/// The set of graphs `{kind}{sfx}.blob`; the layers' weight buffers
	/// are shared between sets through `wbufs` (one buffer per file). The
	/// lists are recorded (which also reads the weights and checks them).
	/// `hcm`: `delta_pre_m` in place of `delta_pre` ([`PrefillHc`]);
	/// `split`: the delta layers' post as `post_delta_a` / `post_delta_b`,
	/// with `gn` `post_delta_t` in place of `post_delta_a`.
	#[allow(clippy::too_many_arguments)]
	fn load(
		npu: &'static Npu,
		cfg: &Config,
		sfx: &str,
		(s, g, c): (usize, usize, usize),
		wbufs: &mut WBufs,
		hcm: bool,
		split: bool,
		gn: bool,
	) -> Result<Set, String> {
		let _sp = quant::ltime::span(|| format!("flash: set '{sfx}' (s = {s})"));
		let alloc = || -> Result<Vec<HostBuf<'static>>, String> {
			ALL.iter().map(|b| npu.alloc(b.bytes(s, g, c))).collect()
		};
		let bufs = alloc()?;
		// `QWEN_FLASH_BANKS=1`: one bank, sub-chunks one after the other
		// (the A/B); 2 lets the NPU run post(j - 1) and pre(j + 1) while
		// the host / iGPU run sub-chunk j's mixer (3 banks' lists overflow
		// the driver's descriptor heap: `0x70000002` at append)
		let nb = if s > 1 {
			std::env::var("QWEN_FLASH_BANKS")
				.ok()
				.and_then(|v| v.parse().ok())
				.unwrap_or(2usize)
		} else {
			1
		};
		let nb = nb.clamp(1, 2);
		let mut alt = Vec::new();
		if nb > 1 {
			alt.push(None);
			for _ in 1..nb {
				alt.push(Some((alloc()?, Vec::new(), None)));
			}
		}
		let on = |v: &str| std::env::var(v).map_or(true, |v| v != "0");
		// `QWEN_FLASH_WS=0`: `experts` / `experts_w` instead of
		// `experts_ns` / `experts_ws` (the row scale `d.s` on the NPU)
		let ns = cfg.graphs.join(format!("experts_ns{sfx}.blob"));
		let nsc = s > 1 && ns.exists() && on("QWEN_FLASH_WS");
		// `QWEN_FLASH_CONV_GPU=0`: `delta_pre_m` with its conv, Swish and
		// q / k norms on the NPU (the A/B); else `delta_pre_c` where its
		// blob exists, those on the scan device
		let pc = cfg.graphs.join(format!("delta_pre_c{sfx}.blob"));
		let conv_c = hcm && pc.exists() && on("QWEN_FLASH_CONV_GPU");
		let mut swz = HashMap::new();
		let blobs = KINDS
			.iter()
			.map(|&k| {
				let p = if k == "experts" && nsc {
					ns.clone()
				} else if k == "delta_pre" && conv_c {
					pc.clone()
				} else if k == "delta_pre" && hcm {
					cfg.graphs.join(format!("delta_pre_m{sfx}.blob"))
				} else {
					cfg.graphs.join(format!("{k}{sfx}.blob"))
				};
				swz.insert(k, swz_names(&p));
				read(&p)
			})
			.collect::<Result<Vec<_>, _>>()?;
		// `QWEN_FLASH_C64=0`: every expert in slots of `c` (the A/B);
		// `QWEN_FLASH_PE=0`: `experts_c64` instead of `experts_w`
		let bp = cfg.graphs.join(format!("experts_c64{sfx}.blob"));
		let bw = cfg.graphs.join(format!("experts_w{sfx}.blob"));
		let mut big = Vec::new();
		if s > 1 && on("QWEN_FLASH_C64") {
			let bs = cfg.graphs.join(format!("experts_ws{sfx}.blob"));
			if bs.exists() && on("QWEN_FLASH_PE") && on("QWEN_FLASH_WS") {
				big.push((read(&bs)?, C2W.0, C2W.1, true));
			} else if bw.exists() && on("QWEN_FLASH_PE") {
				big.push((read(&bw)?, C2W.0, C2W.1, false));
			} else if bp.exists() {
				big.push((read(&bp)?, C2.0, C2.1, false));
			}
		}
		// `QWEN_FLASH_NS8=0`: no slot kinds narrower than `c` (the A/B).
		// An expert with a few tokens (short chunks: most of them) costs
		// a narrow job's place either way, and the job's cost falls with
		// its width: 32 x 8 / 16 / 32 slots 2.8 / 3.1 / 4.5 ms
		if nsc && on("QWEN_FLASH_NS8") {
			for cw in [8, 16] {
				let p = cfg.graphs.join(format!("experts_ns{cw}{sfx}.blob"));
				if cw < c && p.exists() {
					big.push((read(&p)?, g, cw, true));
				}
			}
		}
		let sblobs = if split {
			[
				if gn { "post_delta_t" } else { "post_delta_a" },
				"post_delta_b",
			]
			.iter()
			.map(|&k| {
				let p = cfg.graphs.join(format!("{k}{sfx}.blob"));
				swz.insert(k, swz_names(&p));
				read(&p)
			})
			.collect::<Result<Vec<_>, _>>()?
		} else {
			Vec::new()
		};
		let mut set = Set {
			s,
			g,
			c,
			bufs,
			blobs,
			graphs: Vec::new(),
			big,
			nsc,
			split,
			gn: split && gn,
			conv_c,
			sblobs,
			swz,
			gsplit: Vec::new(),
			gbig: Vec::new(),
			lists: Vec::new(),
			ri: Vec::new(),
			ple: None,
			alt,
			bk: 0,
			jx: Vec::new(),
			jy: Vec::new(),
			k: if s > 1 {
				std::env::var("QWEN_FLASH_KSUP")
					.ok()
					.and_then(|v| v.parse().ok())
					.unwrap_or(KSUP)
					.max(1)
			} else {
				1
			},
		};
		set.activate(npu, &cfg.weights, sfx, wbufs)?;
		Ok(set)
	}

	/// Unload the graphs (lists first): with both sets' graphs loaded the
	/// NPU lost the device (`0x70000001`) at the first one-position chunk
	/// in about half the runs of the scan config.
	fn deactivate(&mut self) {
		for b in self.alt.iter_mut().flatten() {
			b.1.clear();
			b.2 = None;
		}
		self.lists.clear();
		self.ple = None;
		self.gbig.clear();
		self.gsplit.clear();
		self.graphs.clear();
	}

	/// Make bank `t` the active one (`bufs`, `lists`, `ple`).
	fn use_bank(&mut self, t: usize) {
		if self.alt.is_empty() || t == self.bk {
			return;
		}
		let mut o = self.alt[t].take().unwrap();
		std::mem::swap(&mut self.bufs, &mut o.0);
		std::mem::swap(&mut self.lists, &mut o.1);
		std::mem::swap(&mut self.ple, &mut o.2);
		self.alt[self.bk] = Some(o);
		self.bk = t;
	}

	/// Bank `t`'s lists, active or not.
	fn lists_at(&mut self, t: usize) -> &mut Vec<Lists> {
		if t == self.bk {
			&mut self.lists
		} else {
			&mut self.alt[t].as_mut().unwrap().1
		}
	}

	fn banks(&self) -> usize {
		self.alt.len().max(1)
	}

	/// Load the graphs and record the lists (no-op when loaded).
	fn activate(
		&mut self,
		npu: &'static Npu,
		weights: &Path,
		sfx: &str,
		wbufs: &mut WBufs,
	) -> Result<(), String> {
		if self.ple.is_some() {
			return Ok(());
		}
		let sp = quant::ltime::span(|| format!("flash: set '{sfx}' graphs loaded"));
		for (k, b) in KINDS.iter().zip(&self.blobs) {
			let mut g = npu.load(b, 0)?;
			g.initialize()?;
			if std::env::var_os("QWEN_FLASH_ARGS").is_some() {
				for a in g.args.iter().filter(|a| a.is_output) {
					eprintln!("{k}{sfx}: out {} {} B", a.name, a.bytes);
				}
			}
			self.graphs.push(g);
		}
		for (b, ..) in &self.big {
			let mut g = npu.load(b, 0)?;
			g.initialize()?;
			self.gbig.push(g);
		}
		for b in &self.sblobs {
			let mut g = npu.load(b, 0)?;
			g.initialize()?;
			self.gsplit.push(g);
		}
		sp.end();
		let _sp = quant::ltime::span(|| format!("flash: set '{sfx}' lists recorded"));
		for t in 0..self.banks() {
			self.use_bank(t);
			self.record(npu, weights, sfx, wbufs)?;
		}
		self.use_bank(0);
		Ok(())
	}

	/// Record the active bank's lists.
	fn record(
		&mut self,
		npu: &'static Npu,
		weights: &Path,
		sfx: &str,
		wbufs: &mut WBufs,
	) -> Result<(), String> {
		let bufs = &self.bufs;
		let conv_c = self.conv_c;
		let swz = &self.swz;
		let mut tt = [0f64; 4]; // weights read, alloc, -, batch
						  // the streams `x` and the outputs kept per sub-chunk (`CO`) of a
						  // pre / post graph are bound per sub-chunk (`FlashEngine::bind_io`):
						  // those lists are mutable, recorded with the bank's buffers until
						  // then; the PLE graph keeps the bank's `x`
		let mut record = |g: &mut Graph<'static>,
		                  kind: &str,
		                  l: usize|
		 -> Result<(Batch<'static>, Vec<(B, usize)>), String> {
			let mut ri = Vec::new();
			let okind = if kind == "delta_pre" && conv_c {
				"delta_pre_c"
			} else {
				kind
			};
			let outs = OUTS
				.iter()
				.find(|o| o.0 == okind)
				.map(|o| o.1)
				.unwrap_or(&[]);
			let mut oi = 0;
			for i in 0..g.args.len() {
				let a = g.args[i].clone();
				let name = if a.is_output {
					oi += 1;
					outs.get(oi - 1)
						.copied()
						.ok_or(format!("{kind}: extra output"))?
				} else {
					a.name.as_str()
				};
				let name = name.to_string();
				if let Some(b) = runtime_arg(kind, &name) {
					if kind != "ple"
						&& matches!(
							b,
							B::x | B::x1 | B::inj_m | B::logits | B::shared | B::moe_q
						) {
						ri.push((b, i));
					}
					let have = bufs[b as usize].as_slice().len();
					if have < a.bytes {
						return Err(format!(
							"{kind}{sfx}: runtime buffer {name} has {have} bytes, wants {}",
							a.bytes
						));
					}
					g.bind(i, &bufs[b as usize])?;
				} else {
					let sw = swz.get(kind).is_some_and(|n| n.contains(&name));
					let key = (l, name.clone(), sw);
					if !wbufs.contains_key(&key) {
						// straight into the NPU's buffer (a `Vec` and a
						// copy were 2.8 s of the set's 3.4 at startup)
						use std::io::Read;
						let p = weights.join(format!("fl{l}.{name}.in"));
						let e = |e: std::io::Error| format!("{}: {e}", p.display());
						let t = std::time::Instant::now();
						let mut f = std::fs::File::open(&p).map_err(e)?;
						let n = f.metadata().map_err(e)?.len() as usize;
						let mut hb = npu.alloc(n)?;
						tt[1] += t.elapsed().as_secs_f64();
						let t = std::time::Instant::now();
						f.read_exact(hb.as_mut_slice()).map_err(e)?;
						if sw {
							ze::swizzle(hb.as_mut_slice(), ze::SWZ_KEY)
								.map_err(|e| format!("{}: {e}", p.display()))?;
						}
						tt[0] += t.elapsed().as_secs_f64();
						wbufs.insert(key.clone(), hb);
					}
					let hb = &wbufs[&key];
					if hb.as_slice().len() != a.bytes {
						return Err(format!(
							"{kind}{sfx} layer {l}: {name} has {} bytes, wants {}",
							hb.as_slice().len(),
							a.bytes
						));
					}
					g.bind(i, hb)?;
				}
			}
			let t = std::time::Instant::now();
			let b = if !ri.is_empty() {
				npu.batch_plan_mutable(std::slice::from_mut(g), &[(0, Vec::new())])?
			} else {
				npu.batch(&mut [g])?
			};
			tt[3] += t.elapsed().as_secs_f64();
			Ok((b, ri))
		};
		let [gdp, gap, gpd, gpa, gple, _] = &mut self.graphs[..] else {
			unreachable!()
		};
		let ka = if self.gn {
			"post_delta_t"
		} else {
			"post_delta_a"
		};
		let mut gs = match &mut self.gsplit[..] {
			[a, b] => Some((a, b)),
			_ => None,
		};
		let mut lists = Vec::with_capacity(NL);
		let mut ris = Vec::with_capacity(NL);
		for l in 0..NL {
			let ((pre, rp), (post, rq), b) = if is_attn(l) {
				(
					record(gap, "attn_pre", l)?,
					record(gpa, "post_attn", l)?,
					None,
				)
			} else if let Some((a, b)) = gs.as_mut() {
				(
					record(gdp, "delta_pre", l)?,
					record(a, ka, l)?,
					Some(record(b, "post_delta_b", l)?),
				)
			} else {
				(
					record(gdp, "delta_pre", l)?,
					record(gpd, "post_delta", l)?,
					None,
				)
			};
			let (b, rb) = match b {
				Some((b, r)) => (Some(b), r),
				None => (None, Vec::new()),
			};
			lists.push((pre, post, b));
			ris.push([rp, rq, rb]);
		}
		let ple = record(gple, "ple", PLE_LAYER)?.0;
		if quant::ltime::on() {
			eprintln!(
				"[load] set '{sfx}' lists: weights read {:.2} s, alloc {:.2} s, batch {:.2} s",
				tt[0], tt[1], tt[3]
			);
		}
		self.lists = lists;
		self.ri = ris;
		self.ple = Some(ple);
		Ok(())
	}
}

/// What [`FlashEngine::commit`] needs to drop a tail of a multi-position
/// step: the state before it, and per position what it changed.
struct Spec {
	pos0: usize,
	ids: Vec<u32>,
	nctx: ngram::Ctx,
	/// the PLE conv window before the step and after each position
	csts: Vec<Vec<u16>>,
	/// per delta layer: its conv window before the step and after each
	/// position
	convs: Vec<Vec<Vec<u16>>>,
	/// per delta layer: each position's recurrence inputs (the state
	/// before the step is in `FlashEngine::bak`)
	#[allow(clippy::type_complexity)]
	rin: Vec<Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>>,
}

/// An image fed so far: its run of `gw * gh` positions from `start`, row
/// by row. Its MRoPE coordinates follow from the images before it
/// ([`FlashEngine::mrope`]).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Img {
	start: usize,
	gw: usize,
	gh: usize,
}

/// The MRoPE coordinates (t, h, w) of sequence position `p` after the
/// images `imgs`: text advances all three together; an image's positions
/// share one `t`, `h` / `w` are that plus the token's row / column, and the
/// text after it resumes one past the largest coordinate it used (as
/// `crate::engine`'s `mm_positions`).
fn mrope(imgs: &[Img], p: usize) -> [f64; 3] {
	let mut shift = 0;
	for im in imgs {
		if p < im.start {
			break;
		}
		let n = im.gw * im.gh;
		if p < im.start + n {
			let (b, i) = ((im.start - shift) as f64, p - im.start);
			return [b, b + (i / im.gw) as f64, b + (i % im.gw) as f64];
		}
		shift += n - im.gw.max(im.gh);
	}
	[(p - shift) as f64; 3]
}

/// (sin, cos) of the rotary pairs of position `p`. The MRoPE is the
/// interleaved one (sections 11 / 11 / 10 cover all 32 pairs): pair `i`
/// turns by coordinate `i % 3`. For text the three are equal.
fn rope_sc(imgs: &[Img], p: usize) -> [(f64, f64); RD / 2] {
	let c = mrope(imgs, p);
	let mut o = [(0f64, 0f64); RD / 2];
	for (i, v) in o.iter_mut().enumerate() {
		let w = 1e7f64.powf(-(i as f64) / (RD / 2) as f64);
		*v = (c[i % 3] * w).sin_cos();
	}
	o
}

pub struct FlashEngine {
	npu: &'static Npu,
	_lock: crate::lock::NpuLock,
	/// the 256-position set, then the one-position set if present
	sets: Vec<Set>,
	/// the set the current chunk runs on (the one with recorded lists)
	cur: usize,
	/// the layers' weight buffers, shared by the sets' lists
	wb: &'static mut WBufs,
	weights: PathBuf,
	/// per layer: the expert pool `egu.w`, `egu.s`, `ed.w`, `ed.s`
	pools: std::sync::Arc<ExpBufs<4>>,
	/// per layer: `moe.pre`, `ed.pre` (`[NE][FE]`: a range per expert)
	epre: std::sync::Arc<ExpBufs<2>>,
	exp_thread: ExpThread,
	/// per layer: `experts_ws`'s `gu.s2` (`[NE][2 FE]`: `egu.s`, its up
	/// rows times `ed.pre`) and `ed.s` as f32 (the gather's row scale)
	ews: Vec<(HostBuf<'static>, Vec<f32>)>,
	host: Vec<HostLayer>,
	/// per delta layer: the conv window `[3][10240]` f16
	conv: Vec<Vec<u16>>,
	kv: Vec<Kv>,
	cstate: Vec<u16>,
	nctx: ngram::Ctx,
	hash: ngram::Hash,
	table: std::sync::Arc<ngram::Table>,
	emb: std::fs::File,
	/// the next prefill piece's n-gram rows, read in the background
	/// ([`Self::prefetch`])
	pf: Vec<Prefetch>,
	/// [`Self::inputs`]' seconds: embeddings, n-gram rows (`QWEN_NPU_TIMING`)
	t_inp: [f64; 2],
	/// final mixer (f32): norm `[10240]`, down `[320][10240]`, up `[10240][320]`
	hcf: [Vec<f32>; 3],
	head: Arc<Head>,
	scan: Option<Box<dyn Scan>>,
	/// with [`Set::gn`]: per delta layer the gated norm's factors
	/// ([`Scan::set_gnorm`], [`gn_factors`])
	gnw: Vec<Vec<f32>>,
	/// with [`Set::conv_c`]: per delta layer the conv's taps `[10240][4]`,
	/// then `qkv`'s row scale `[10240]`
	/// ([`Scan::set_conv`], [`conv_taps`])
	cvw: Vec<Vec<f32>>,
	/// host recurrence states (no scan device): `[36][48][128][128]`
	hstate: Vec<Vec<f32>>,
	pos: usize,
	pub timing: bool,
	/// The sampler's shared noise ([`Self::set_coupling`]): MTP drafts are
	/// the head's coupled draws instead of its argmax.
	coupling: Option<sampler::Coupling>,
	/// step costs measured at run time ([`mtp_auto`])
	meter: CostMeter,
	/// `QWEN_FLASH_POKE=0`: no `Scan::poke` in attention layers (the A/B)
	poke: bool,
	/// last scan or poke
	poked: std::time::Instant,
	/// seconds in [`Self::attention`] since the last step line
	t_att: f64,
	/// a chunk's expert jobs: record, NPU run, host gather (seconds)
	t_moe: [f64; 5],
	/// the last chunk's streams per sub-chunk: (positions, `x`)
	xfin: Vec<(usize, Vec<u8>)>,
	/// sub-chunks' stream buffers between chunks ([`Sub::x`])
	xpool: Vec<HostBuf<'static>>,
	/// `QWEN_FLASH_D1=N`: every position through steps of N, the decode
	/// path (1: the one-position graphs or GPU steps of one; a check of the
	/// decode path against the prefill's, and of N against 1).
	d1_all: usize,
	/// A one-position step's routed experts on the CPU cores
	/// ([`super::cpuexp`]) instead of an NPU job; `QWEN_FLASH_CPU_EXP=0` is
	/// the A/B.
	cpu_exp: bool,
	/// The delta states are in `hstate`, not on the scan device.
	states_on_host: bool,
	/// The delta states and conv windows are on the dense GPU
	/// (`GpuDec::mix_gpu`), not in `hstate` / `conv`.
	on_gtx: bool,
	/// A one-position step's dense layers on a GPU (`dense = ADAPTER`).
	gpu: Option<super::gpudec::GpuDec>,
	/// What builds `gpu` (`dense = ADAPTER`), the VRAM it needs, and the
	/// lm_head goes with it. A GPU that fails, stalls or stays slow
	/// (another application took its VRAM: WDDM pages instead of failing)
	/// is dropped and decode runs on the one-position graphs; a new one is
	/// built every `QWEN_GPU_RECOVER_SECS` ([`Self::dense_heal`]).
	dense_new: Option<super::gpudec::DenseGpuNew>,
	dense_need: u64,
	head_gpu: bool,
	/// the MTP head's files when its dense part goes on the dense GPU
	/// (`QWEN_FLASH_MTP_GPU`)
	mtp_dir: Option<PathBuf>,
	/// when to try building `gpu` again
	dense_retry: Option<std::time::Instant>,
	/// why `gpu` is to be dropped at the next call ([`Self::dense_watch`])
	dense_bad: Option<String>,
	/// fastest GPU time of a step by positions since `gpu` was built, the
	/// steps since then and the slow ones in a row
	dense_base: [f64; 9],
	dense_steps: usize,
	dense_slow: usize,
	/// when `gpu` was last built again, and the retry interval's factor
	dense_since: Option<std::time::Instant>,
	dense_backoff: u64,
	/// The delta states were on a GPU that was lost: the state is no
	/// conversation's until [`Self::reset`] / [`Self::restore`].
	lost: bool,
	/// A prefill chunk's attention on a device ([`Self::set_prefill_attn`]).
	pattn: Option<Box<dyn PrefillAttn>>,
	/// `pattn`'s work in flight ([`Self::attention_begin`]): when the
	/// attention started and its query count
	pattn_flight: Option<(std::time::Instant, usize)>,
	/// an attention layer's whole past blocks on the NPU next to `pattn`
	/// (`attn_far` / `attn_farm`, loaded at the first prefill attention
	/// with them; dropped while the one-position set is active);
	/// `QWEN_FLASH_FAR_NPU=0` is the A/B
	far: Option<super::far::NpuFar>,
	/// the indexer's scores on the NPU ([`super::idx`]), loaded with `far`
	idx: Option<super::idx::NpuIdx>,
	/// the indexer's scores on the prefill attention's device instead
	/// (`QWEN_FLASH_IDX_GPU`, default on when it can)
	idx_gpu: bool,
	far_tried: bool,
	/// `QWEN_FLASH_FAR_CHECK=1`: the sub-chunk in flight's layer, p0, n,
	/// queries and whole selection, to compare the merge with the iGPU
	/// alone (diagnostic)
	far_check: Option<(usize, usize, usize, Vec<u16>, Vec<u32>)>,
	graphs: PathBuf,
	/// The prefill set's delta layers take `delta_pre_m`, their attention
	/// hyper-connection on `phc` ([`Self::set_prefill_hc`]).
	hc_m: bool,
	phc: Option<Box<dyn PrefillHc>>,
	/// The delta layers' MoE hyper-connection on the scan device's adapter
	/// (`Set::split`, [`Self::set_prefill_hcm`])
	phm: Option<Box<dyn PrefillHc>>,
	/// a chunk's dense outputs per sub-chunk; buffers outgrown are kept in
	/// `co_old`, never freed: the iGPU's imports of their pages stay
	co: Co,
	co_old: Vec<HostBuf<'static>>,
	/// a share of the prefill's routed experts ([`Self::set_prefill_exp`])
	pexp: Option<Box<dyn PrefillExp>>,
	/// the NPU / device cost factors of `pexp`'s share ([`EXP_CAL`])
	exp_cal: quant::calib::Fit,
	/// `phc`'s work in flight: (delta layer, sub-chunk position)
	phc_flight: Option<(usize, usize)>,
	/// the last expert pass's combine still owed ([`moe_defer`])
	comb: Option<Comb>,
	/// seconds in `phc` since the last chunk line
	t_hc: f64,
	/// the last [`Self::verify`] of several positions, until committed
	spec: Option<Spec>,
	/// the delta states before it (`[36][48][128][128]`, allocated by the
	/// first one)
	bak: Vec<Vec<f32>>,
	/// the vision tower and what runs its attention
	vis: Option<crate::vis::NpuVision>,
	vis_attn: Option<Box<dyn crate::vis::VisAttn>>,
	/// the images fed since the last reset, in order
	imgs: Vec<Img>,
	/// during [`Self::feed_mm`]: (first position, `[n][W]` f16 rows) per
	/// image, the rows that replace its placeholders' embeddings
	mm_emb: Vec<(usize, Vec<u16>)>,
	/// the MTP head (with the dense GPU only)
	mtp: Option<super::mtp::Mtp>,
	/// positions fed since the head last ran: (position, its streams after
	/// the last layer, its token); [`Self::drafts`] runs each one's row
	mpend: Vec<(usize, Vec<f32>, u32)>,
	/// The host attention's per-task partial sums, kept between calls (a
	/// fresh zeroed one is ~0.4 MB per query and attention layer).
	att_part: Vec<f32>,
}

// SAFETY: the engine owns its NPU handles, host buffers and devices; it is
// used from one thread at a time (the server holds it behind a Mutex).
unsafe impl Send for FlashEngine {}

impl FlashEngine {
	pub fn load(
		cfg: &Config,
		scan: Option<Box<dyn Scan>>,
		dense: Option<super::gpudec::DenseGpuNew>,
	) -> Result<FlashEngine, String> {
		Self::load_with(cfg, scan, dense, None)
	}

	/// [`Self::load`], plus a share of the prefill's routed experts
	/// ([`Self::set_prefill_exp`]): `exp` makes the device (it may wait for
	/// another thread that builds it), and each layer's pools are imported
	/// into it as soon as they are read.
	///
	/// The load's critical path is reading the 57 GB of expert pools (disk
	/// bound, ~9 s at 6.8 GB/s): the readers start right after the NPU is
	/// open, and what does not need the NPU (lm_head, the dense part on the
	/// GPU, the MTP head, the expert import) runs on other threads
	/// meanwhile; the main thread allocates the pools, then loads the graph
	/// sets and the vision tower (`QWEN_LOAD_TIMING=1` shows the spans).
	pub fn load_with(
		cfg: &Config,
		scan: Option<Box<dyn Scan>>,
		dense: Option<super::gpudec::DenseGpuNew>,
		exp: Option<PrefillExpNew>,
	) -> Result<FlashEngine, String> {
		if !(std::is_x86_feature_detected!("avx2")
			&& std::is_x86_feature_detected!("f16c")
			&& std::is_x86_feature_detected!("fma"))
		{
			return Err("flash: the engine needs AVX2, F16C and FMA".into());
		}
		let sp = quant::ltime::span(|| "flash: NPU lock".into());
		let lock = crate::lock::NpuLock::acquire()?;
		sp.end();
		let sp = quant::ltime::span(|| "flash: NPU open".into());
		let npu: &'static Npu = Box::leak(Box::new(Npu::open()?));
		sp.end();
		let t0 = std::time::Instant::now();
		let mut wb = std::collections::HashMap::new();
		// `QWEN_FLASH_HC_GPU=0`: delta_pre's hyper-connection on the NPU
		// (the A/B); else on the scan device where a `delta_pre_m` blob exists
		let hc_m = cfg.scan.is_some()
			&& cfg.graphs.join("delta_pre_m.blob").exists()
			&& std::env::var("QWEN_FLASH_HC_GPU").map_or(true, |v| v != "0");
		// `QWEN_FLASH_HCM_GPU=0`: the delta layers' MoE hyper-connection
		// inside `post_delta` on the NPU (the A/B); else on the scan
		// device between `post_delta_a` and `post_delta_b`, where their
		// blobs exist (two banks, the PLE graphs before layer 1)
		let split = hc_m
			&& ["post_delta_a", "post_delta_b"]
				.iter()
				.all(|k| cfg.graphs.join(format!("{k}.blob")).exists())
			&& ple_first()
			&& std::env::var("QWEN_FLASH_BANKS").map_or(true, |v| v.trim() != "1")
			&& std::env::var("QWEN_FLASH_HCM_GPU").map_or(true, |v| v != "0");
		// `QWEN_FLASH_GN_GPU=0`: the delta layers' gated norm and the
		// out-projection's rotation in `post_delta_a` on the NPU (the A/B);
		// else on the scan device right after the recurrence, where a
		// `post_delta_t` blob exists
		let gn = split
			&& cfg.graphs.join("post_delta_t.blob").exists()
			&& std::env::var("QWEN_FLASH_GN_GPU").map_or(true, |v| v != "0");
		// with the dense layers on a GPU the one-position graphs are not needed
		let head_f16 = std::env::var("QWEN_FLASH_HEAD").is_ok_and(|v| v == "f16");
		let head_gpu = !head_f16 && std::env::var("QWEN_FLASH_HEAD_GPU").map_or(true, |v| v != "0");
		let has_dense = dense.is_some();
		// the MTP head's dense part goes with the dense GPU's
		let mtp_dir = cfg
			.mtp
			.as_deref()
			.filter(|_| has_dense && super::gpudec::mtp_gpu());
		let wfile = |l: usize, n: &str| cfg.weights.join(format!("fl{l}.{n}.in"));
		const EFILES: [&str; 6] = ["egu.w", "egu.s", "ed.w", "ed.s", "moe.pre", "ed.pre"];
		// the expert pools, declared outside the scope: readers may still
		// write into them when the main thread leaves it with an error
		let mut bufs: Vec<HostBuf<'static>> = Vec::with_capacity(NL * EFILES.len());
		// per layer: `experts_ws`'s `gu.s2` ([`Self::ews`]), filled by the
		// reader that completes the layer
		let mut s2s: Vec<HostBuf<'static>> = Vec::with_capacity(NL);
		#[derive(Clone, Copy)]
		struct P(*mut u8, usize);
		// SAFETY: a pool buffer's address, written by one reader, then read
		// by the importer once that reader is done with it
		unsafe impl Send for P {}
		let (tx, rx) = std::sync::mpsc::channel::<(PathBuf, P, usize)>();
		let rx = std::sync::Mutex::new(rx);
		// per layer, the files read so far; a layer whose six are in goes to
		// the importer
		let done = std::sync::Mutex::new(vec![([None::<P>; 6], None::<P>); NL]);
		// per layer: `ed.s` as f32 ([`Self::ews`])
		let dsf = std::sync::Mutex::new(vec![Vec::<f32>::new(); NL]);
		let (sets, head, need, gpu, dense_retry, pexp, mtp, vis) =
			std::thread::scope(|sc| -> Result<_, String> {
				// lm_head (read, int8), then the dense part on the GPU
				let th = sc.spawn(|| -> Result<_, String> {
					let sp = quant::ltime::span(|| "flash: lm_head read".into());
					let head: Vec<u16> = read(&cfg.head)?
						.chunks_exact(2)
						.map(|c| u16::from_le_bytes([c[0], c[1]]))
						.collect();
					if head.len() != super::VOCAB * W {
						return Err(format!(
							"{}: not [{}][{W}] f16",
							cfg.head.display(),
							super::VOCAB
						));
					}
					sp.end();
					let sp = quant::ltime::span(|| "flash: lm_head int8".into());
					let head = if head_f16 {
						Head::F16(head)
					} else {
						Head::quantize(&head)
					};
					sp.end();
					let _sp = quant::ltime::span(|| "flash: dense part on the GPU".into());
					let need = match dense {
						Some(_) => {
							super::gpudec::GpuDec::need(&cfg.weights, head_gpu)?
								+ mtp_dir.map_or(0, |_| super::gpudec::GpuDec::mtp_need())
						}
						None => 0,
					};
					// a GPU that does not hold the dense part now (another
					// application has its VRAM) is not an error: decode runs
					// on the one-position graphs and the GPU is tried again
					// later ([`Self::dense_heal`])
					let mut dense_retry = None;
					let gpu = match &dense {
						Some(new) => {
							match dense_open(new, need, &cfg.weights, &head, head_gpu, mtp_dir) {
								Ok(g) => Some(g),
								Err(e) => {
									dense_retry = dense_retry_at(1);
									eprintln!(
										"flash: dense part not on the GPU ({e}){}",
										retry_note(1)
									);
									None
								}
							}
						}
						None => None,
					};
					Ok((head, need, gpu, dense_retry))
				});
				// the MTP head (drafts only while the dense GPU is there)
				let tm = sc.spawn(|| -> Result<_, String> {
					match &cfg.mtp {
						Some(d) if has_dense => {
							let _sp = quant::ltime::span(|| "flash: MTP head".into());
							Ok(Some(super::mtp::Mtp::load(d)?))
						}
						_ => Ok(None),
					}
				});
				// the experts (57 GB): the main thread allocates the pinned
				// buffers (zeMemAllocHost, ~5 s in all) while
				// `QWEN_FLASH_LOAD_THREADS` readers fill them straight from
				// the files; one buffer per file read into a `Vec` first and
				// copied was 18 + 8 + 5 s serial
				let sp_e = quant::ltime::span(|| "flash: experts".into());
				let nr = std::env::var("QWEN_FLASH_LOAD_THREADS")
					.ok()
					.and_then(|v| v.parse().ok())
					.unwrap_or(4usize)
					.max(1);
				let (ti, ri) = std::sync::mpsc::channel::<(usize, [P; 6])>();
				let readers: Vec<_> = (0..nr)
					.map(|_| {
						let (rx, done, dsf, ti) = (&rx, &done, &dsf, ti.clone());
						sc.spawn(move || -> Result<(), String> {
							use std::io::Read;
							let _sp = quant::ltime::span(|| "flash: experts: a reader".into());
							loop {
								let Ok((p, P(ptr, len), i)) = rx.lock().unwrap().recv() else {
									return Ok(());
								};
								// SAFETY: the main thread's buffer of `len` bytes,
								// written by this reader only, alive for the engine
								let dst = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
								// unbuffered: 60 GB through the file cache pushed
								// other pages out (+5-8 s in the first chunk);
								// needs 4 KiB-aligned reads (the buffers are
								// rounded up)
								let mut o = std::fs::OpenOptions::new();
								o.read(true);
								#[cfg(windows)]
								std::os::windows::fs::OpenOptionsExt::custom_flags(
									&mut o,
									0x2000_0000, // FILE_FLAG_NO_BUFFERING
								);
								let mut f =
									o.open(&p).map_err(|e| format!("{}: {e}", p.display()))?;
								let mut at = 0;
								while at < len {
									let k = f
										.read(&mut dst[at..])
										.map_err(|e| format!("{}: {e}", p.display()))?;
									if k == 0 {
										break;
									}
									at += k;
								}
								let (l, k) = (i / 6, i % 6);
								let (f, s2) = {
									let mut d = done.lock().unwrap();
									d[l].0[k] = Some(P(ptr, len));
									if !d[l].0.iter().all(Option::is_some) {
										continue;
									}
									(d[l].0.map(Option::unwrap), d[l].1.unwrap())
								};
								let _ = ti.send((l, f));
								// the layer's row scales: `gu.s2` = `egu.s`, its up
								// rows times `ed.pre`; `ed.s` as f32
								// SAFETY: the layer's files are read (by this and
								// other readers, which are done with them) and
								// `s2` is this layer's, written here only
								let h = |p: P| unsafe {
									std::slice::from_raw_parts(p.0 as *const u16, p.1 / 2)
								};
								let (gus, dpre, ds) = (h(f[1]), h(f[5]), h(f[3]));
								let d = unsafe {
									std::slice::from_raw_parts_mut(s2.0 as *mut u16, s2.1 / 2)
								};
								for e in 0..NE {
									for r in 0..2 * FE {
										let v = unf16(gus[e * 2 * FE + r]);
										d[e * 2 * FE + r] = if r < FE {
											gus[e * 2 * FE + r]
										} else {
											f16(v * unf16(dpre[e * FE + r - FE]))
										};
									}
								}
								dsf.lock().unwrap()[l] =
									ds[..NE * W].iter().map(|&h| unf16(h)).collect();
							}
						})
					})
					.collect();
				drop(ti);
				// imports each layer's pools into the device once they are
				// read (zero-copy; no reader touches the layer again)
				let imp = exp.map(|new| {
					sc.spawn(move || -> Result<Box<dyn PrefillExp + Send>, String> {
						let mut h = new()?;
						let _sp = quant::ltime::span(|| "flash: experts: import".into());
						let ne = exp_gpu_ne();
						// SAFETY: the reader of each file is done with it; the
						// pools live as long as the engine
						let u8s =
							|p: P| unsafe { std::slice::from_raw_parts(p.0 as *const u8, p.1) };
						let u16s = |p: P| unsafe {
							std::slice::from_raw_parts(p.0 as *const u16, p.1 / 2)
						};
						for (l, f) in ri {
							h.add_layer(
								l,
								&u8s(f[0])[..ne * 2 * FE * W / 2],
								&u16s(f[1])[..ne * 2 * FE],
								&u8s(f[2])[..ne * W * FE / 2],
								&u16s(f[3])[..ne * W],
								&u16s(f[5])[..ne * FE],
							)?;
						}
						Ok(h)
					})
				});
				let mut res = Ok(());
				let sp = quant::ltime::span(|| "flash: experts: zeMemAllocHost".into());
				'alloc: for l in 0..NL {
					match npu.alloc((NE * 2 * FE * 2).div_ceil(4096) * 4096) {
						Ok(mut b) => {
							done.lock().unwrap()[l].1 =
								Some(P(b.as_mut_slice().as_mut_ptr(), b.len()));
							s2s.push(b);
						}
						Err(e) => {
							res = Err(e);
							break 'alloc;
						}
					}
					for (k, n) in EFILES.iter().enumerate() {
						let p = wfile(l, n);
						let len = match std::fs::metadata(&p) {
							Ok(m) => m.len() as usize,
							Err(e) => {
								res = Err(format!("{}: {e}", p.display()));
								break 'alloc;
							}
						};
						let al = len.div_ceil(4096) * 4096;
						match npu.alloc(al) {
							Ok(mut b) => {
								let _ =
									tx.send((p, P(b.as_mut_slice().as_mut_ptr(), al), l * 6 + k));
								bufs.push(b);
							}
							Err(e) => {
								res = Err(e);
								break 'alloc;
							}
						}
					}
				}
				sp.end();
				drop(tx);
				res?;
				// the graph sets, while the readers read
				let mut sets = vec![Set::load(
					npu,
					cfg,
					"",
					(S, G, C),
					&mut wb,
					hc_m,
					split,
					gn,
				)?];
				let (head, need, gpu, dense_retry) = th
					.join()
					.map_err(|_| "flash: the lm_head / dense loader panicked".to_string())??;
				if gpu.is_none() && cfg.graphs.join("delta_pre_d1.blob").exists() {
					if let Some(s0) = sets.first_mut() {
						s0.deactivate();
					}
					sets.push(Set::load(
						npu,
						cfg,
						"_d1",
						(1, G1, 1),
						&mut wb,
						false,
						false,
						false,
					)?);
				}
				let sp = quant::ltime::span(|| "flash: vision tower".into());
				let vis = match &cfg.vision {
					Some(d) => Some(crate::vis::NpuVision::load(npu, &d.to_string_lossy())?),
					None => None,
				};
				sp.end();
				for r in readers {
					r.join()
						.map_err(|_| "flash: a weight reader panicked".to_string())??;
				}
				sp_e.end();
				let _sp = quant::ltime::span(|| "flash: join importer, MTP".into());
				let pexp = match imp {
					Some(t) => Some(
						t.join()
							.map_err(|_| "flash: the expert importer panicked".to_string())??,
					),
					None => None,
				};
				let mtp = tm
					.join()
					.map_err(|_| "flash: the MTP loader panicked".to_string())??;
				Ok((sets, head, need, gpu, dense_retry, pexp, mtp, vis))
			})?;
		let _sp = quant::ltime::span(|| "flash: engine state".into());
		// the layers' weight buffers stay alive for the recorded lists
		let wb = Box::leak(Box::new(wb));
		let mut pools = Vec::with_capacity(NL);
		let mut epre = Vec::with_capacity(NL);
		let mut host = Vec::with_capacity(NL);
		let mut it = bufs.into_iter();
		let ews: Vec<_> = s2s.into_iter().zip(dsf.into_inner().unwrap()).collect();
		for l in 0..NL {
			let mut nx = || it.next().ok_or("flash: expert buffers");
			pools.push([nx()?, nx()?, nx()?, nx()?]);
			epre.push([nx()?, nx()?]);
			let ikn = if is_attn(l) {
				read(&wfile(l, "ikn"))?
					.chunks_exact(2)
					.map(|c| unf16(u16::from_le_bytes([c[0], c[1]])))
					.collect()
			} else {
				Vec::new()
			};
			host.push(HostLayer { ikn });
		}
		if std::env::var_os("QWEN_NPU_TIMING").is_some() {
			eprintln!(
				"flash: experts loaded in {:.1} s",
				t0.elapsed().as_secs_f64()
			);
		}
		let f32s = |p: PathBuf| -> Result<Vec<f32>, String> {
			Ok(read(&p)?
				.chunks_exact(2)
				.map(|c| unf16(u16::from_le_bytes([c[0], c[1]])))
				.collect())
		};
		let hf = |n: &str| cfg.weights.join(format!("flf.hcf.{n}.in"));
		let hcf = [f32s(hf("norm"))?, f32s(hf("down"))?, f32s(hf("up"))?];
		let emb =
			std::fs::File::open(&cfg.emb).map_err(|e| format!("{}: {e}", cfg.emb.display()))?;
		let ndelta = (0..NL).filter(|&l| !is_attn(l)).count();
		// `post_delta_t`'s gated norm on the scan device: its factors
		let cvw = if sets[0].conv_c {
			(0..NL)
				.filter(|&l| !is_attn(l))
				.map(|l| conv_taps(&cfg.weights, l))
				.collect::<Result<Vec<_>, _>>()?
		} else {
			Vec::new()
		};
		let gnw = if sets[0].gn {
			(0..NL)
				.filter(|&l| !is_attn(l))
				.map(|l| gn_factors(&cfg.weights, l))
				.collect::<Result<Vec<_>, _>>()?
		} else {
			Vec::new()
		};
		let e = FlashEngine {
			npu,
			_lock: lock,
			cur: sets.len() - 1,
			wb,
			weights: cfg.weights.clone(),
			sets,
			pools: std::sync::Arc::new(ExpBufs(pools)),
			epre: std::sync::Arc::new(ExpBufs(epre)),
			exp_thread: ExpThread::new()?,
			ews,
			host,
			conv: vec![vec![0u16; 3 * NQKV]; ndelta],
			kv: vec![Kv::default(); NL],
			cstate: vec![0u16; BACK * HCW],
			nctx: ngram::Ctx::new(EOS),
			hash: ngram::Hash::new(super::VOCAB as u64, 0),
			table: std::sync::Arc::new(ngram::Table::open(&cfg.ngram)?),
			emb,
			pf: Vec::new(),
			t_inp: [0.0; 2],
			hcf,
			head: Arc::new(head),
			gnw,
			cvw,
			hstate: if scan.is_none() {
				vec![vec![0f32; H * D * D]; ndelta]
			} else {
				Vec::new()
			},
			scan,
			pos: 0,
			timing: std::env::var("QWEN_NPU_TIMING").is_ok_and(|v| v != "0"),
			coupling: None,
			meter: CostMeter::default(),
			poke: std::env::var("QWEN_FLASH_POKE").map_or(true, |v| v != "0"),
			poked: std::time::Instant::now(),
			t_att: 0.0,
			t_moe: [0.0; 5],
			xfin: Vec::new(),
			xpool: Vec::new(),
			d1_all: std::env::var("QWEN_FLASH_D1")
				.ok()
				.and_then(|v| v.trim().parse().ok())
				.unwrap_or(0),
			cpu_exp: std::env::var("QWEN_FLASH_CPU_EXP").map_or(true, |v| v != "0")
				&& std::is_x86_feature_detected!("avx2"),
			states_on_host: false,
			on_gtx: false,
			gpu,
			dense_new: dense,
			dense_need: need,
			head_gpu,
			mtp_dir: mtp_dir.map(Path::to_path_buf),
			dense_retry,
			dense_bad: None,
			dense_base: [f64::INFINITY; 9],
			dense_steps: 0,
			dense_slow: 0,
			dense_since: None,
			dense_backoff: 1,
			lost: false,
			pattn: None,
			pattn_flight: None,
			far: None,
			far_tried: false,
			far_check: None,
			idx: None,
			idx_gpu: false,
			graphs: cfg.graphs.clone(),
			hc_m,
			phc: None,
			phm: None,
			co: Co {
				bufs: Vec::new(),
				sn: 0,
			},
			co_old: Vec::new(),
			pexp: pexp.map(|h| -> Box<dyn PrefillExp> { h }),
			exp_cal: {
				// what the factors depend on beyond the executable: the NPU's
				// expert graphs and the device
				let mut parts: Vec<String> = std::fs::read_dir(&cfg.graphs)
					.into_iter()
					.flatten()
					.flatten()
					.filter(|e| e.file_name().to_string_lossy().starts_with("experts"))
					.map(|e| {
						format!(
							"{}={}",
							e.file_name().to_string_lossy(),
							quant::calib::file_id(&e.path())
						)
					})
					.collect();
				parts.sort();
				parts.push(format!("scan={:?}", cfg.scan));
				let parts: Vec<&str> = parts.iter().map(String::as_str).collect();
				quant::calib::Fit::open(
					EXP_CAL.0,
					quant::calib::fingerprint(&parts),
					&EXP_CAL.1,
					EXP_CAL_NEED,
				)
			},
			phc_flight: None,
			comb: None,
			t_hc: 0.0,
			spec: None,
			bak: Vec::new(),
			vis,
			vis_attn: None,
			imgs: Vec::new(),
			mm_emb: Vec::new(),
			mtp,
			mpend: Vec::new(),
			att_part: Vec::new(),
		};
		eprintln!("flash: loaded in {:.1} s", t0.elapsed().as_secs_f64());
		Ok(e)
	}

	pub fn pos(&self) -> usize {
		self.pos
	}

	fn rope_sc(&self, p: usize) -> [(f64, f64); RD / 2] {
		rope_sc(&self.imgs, p)
	}

	/// Unload the NPU graphs and their recorded lists (the active prefill
	/// or one-position set, the far graphs, the vision tower's), keeping
	/// every buffer and the state: they hold driver memory that every NPU
	/// user of the process shares, and with them loaded an image model's
	/// job does not fit beside them (`0x70000002`). The next chunk and the
	/// next image load them again.
	pub fn release_npu(&mut self) {
		self.far = None;
		self.far_tried = false;
		for s in &mut self.sets {
			s.deactivate();
		}
		if let Some(v) = self.vis.as_mut() {
			v.unload();
		}
	}

	/// The vision tower's configuration and its learned position grid
	/// (`[n_pos][embd]`), when the config names one (`vision = DIR`).
	pub fn vision(&self) -> Option<(&crate::vis::VisCfg, &[f32])> {
		self.vis.as_ref().map(|v| (&v.cfg, &v.pos_w[..]))
	}

	/// What runs the tower's attention ([`Self::encode_image`] needs one).
	pub fn set_vis_attn(&mut self, a: Box<dyn crate::vis::VisAttn>) {
		self.vis_attn = Some(a);
	}

	/// One image through the vision tower (`crate::vis::NpuVision::encode`):
	/// the embeddings of its tokens, `[n / merge^2][W]`. Independent of the
	/// text state.
	pub fn encode_image(
		&mut self,
		patches: &[f32],
		pos: &[f32],
		hw: &[(u32, u32)],
	) -> Result<Vec<f32>, String> {
		let v = self
			.vis
			.as_mut()
			.ok_or("this engine has no vision tower (config: vision = DIR)")?;
		let a = self
			.vis_attn
			.as_deref_mut()
			.ok_or("the vision tower has no attention device")?;
		if v.cfg.proj != W {
			return Err(format!(
				"the vision tower projects to {} dims, the model has {W}",
				v.cfg.proj
			));
		}
		v.encode(patches, pos, hw, a)
	}

	/// [`Self::feed`] for tokens with images: each span's run of placeholder
	/// tokens is fed as the tower's embeddings, at the image's MRoPE
	/// coordinates ([`mrope`]), through the prefill graphs (the one-position
	/// paths take token ids only). The n-gram context sees the placeholder
	/// ids.
	pub fn feed_mm(
		&mut self,
		ids: &[u32],
		spans: &[crate::engine::ImageSpan<'_>],
	) -> Result<(), String> {
		if spans.is_empty() {
			return self.feed(ids);
		}
		self.spec = None;
		let p0 = self.pos;
		let mut at = 0;
		for sp in spans {
			let n = sp.grid_w * sp.grid_h;
			// (an image never ends the fed tokens: a last piece of one
			// position would take the one-position path)
			if n == 0 || sp.start < at || sp.start + n >= ids.len() {
				return Err(format!(
					"feed_mm: an image of {} x {} tokens at {} of {} tokens",
					sp.grid_w,
					sp.grid_h,
					sp.start,
					ids.len()
				));
			}
			if sp.emb.len() != n * W {
				return Err(format!(
					"feed_mm: {} values for an image of {} x {} tokens",
					sp.emb.len(),
					sp.grid_w,
					sp.grid_h
				));
			}
			at = sp.start + n;
		}
		let old = self.imgs.len();
		self.imgs.extend(spans.iter().map(|sp| Img {
			start: p0 + sp.start,
			gw: sp.grid_w,
			gh: sp.grid_h,
		}));
		self.mm_emb = spans
			.iter()
			.map(|sp| (p0 + sp.start, sp.emb.iter().map(|&v| f16(v)).collect()))
			.collect();
		let bar = quant::progress::Prefill::start(ids.len());
		let mut r = Ok(());
		let ps = self.pieces(ids);
		for (i, c) in ps.iter().cloned().enumerate() {
			quant::progress::chunk(c.start, c.len());
			self.prefetch(ids, &ps, i);
			r = self.chunk(&ids[c], false).map(|_| ());
			if r.is_err() {
				break;
			}
		}
		self.mm_emb.clear();
		if r.is_err() {
			self.imgs.truncate(old);
		} else {
			bar.done();
		}
		r
	}

	/// Run the attention of prefill chunks (more than one position) on `a`
	/// instead of the cores.
	/// The prefill set takes `delta_pre_m`: [`Self::set_prefill_hc`] is
	/// needed before a prefill.
	pub fn wants_prefill_hc(&self) -> bool {
		self.hc_m
	}

	pub fn set_prefill_hc(&mut self, h: Box<dyn PrefillHc>) {
		self.phc = Some(h);
	}

	/// The prefill set's delta layers split `post` around their MoE
	/// hyper-connection: [`Self::set_prefill_hcm`] is needed before a
	/// prefill.
	pub fn wants_prefill_hcm(&self) -> bool {
		self.sets[0].split
	}

	/// `h` runs the delta layers' MoE hyper-connections (`hcm` weights).
	pub fn set_prefill_hcm(&mut self, h: Box<dyn PrefillHc>) {
		self.phm = Some(h);
	}

	/// Run a share of every prefill expert pass on `h`
	/// (`QWEN_FLASH_EXP_GPU`), after handing it every layer's pools.
	pub fn set_prefill_exp(&mut self, mut h: Box<dyn PrefillExp>) -> Result<(), String> {
		let ne = exp_gpu_ne();
		for l in 0..NL {
			let (pl, pre) = (&self.pools[l], &self.epre[l][1]);
			h.add_layer(
				l,
				&pl[0].as_slice()[..ne * 2 * FE * W / 2],
				&h16(&pl[1])[..ne * 2 * FE],
				&pl[2].as_slice()[..ne * W * FE / 2],
				&h16(&pl[3])[..ne * W],
				&h16(pre)[..ne * FE],
			)?;
		}
		self.pexp = Some(h);
		Ok(())
	}

	/// The delta recurrence's device, made after the load (the load does
	/// not use it).
	pub fn set_scan(&mut self, mut scan: Box<dyn Scan>) -> Result<(), String> {
		for (di, w) in self.gnw.iter().enumerate() {
			if !scan.set_gnorm(di, w)? {
				return Err("flash: post_delta_t needs a scan device with the gated norm 					(QWEN_FLASH_GN_GPU=0 runs it on the NPU)"
					.into());
			}
		}
		for (di, w) in self.cvw.iter().enumerate() {
			if !scan.set_conv(di, w)? {
				return Err("flash: delta_pre_c needs a scan device with the conv 					(QWEN_FLASH_CONV_GPU=0 runs it on the NPU)"
					.into());
			}
		}
		self.scan = Some(scan);
		self.hstate = Vec::new();
		Ok(())
	}

	pub fn set_prefill_attn(&mut self, a: Box<dyn PrefillAttn>) {
		self.pattn = Some(a);
	}

	/// The sampler's noise for [`Self::drafts`] (`None`: the head's argmax).
	pub fn set_coupling(&mut self, c: Option<sampler::Coupling>) {
		self.coupling = c;
	}

	pub fn reset(&mut self) -> Result<(), String> {
		self.spec = None;
		self.lost = false;
		self.mtp_clear();
		self.pos = 0;
		self.imgs.clear();
		self.conv.iter_mut().for_each(|c| c.fill(0));
		self.kv.iter_mut().for_each(|k| *k = Kv::default());
		if let Some(a) = self.pattn.as_mut() {
			a.invalidate();
		}
		if let Some(f) = self.far.as_mut() {
			f.invalidate();
		}
		if let Some(x) = self.idx.as_mut() {
			x.invalidate();
		}
		self.cstate.fill(0);
		self.nctx = ngram::Ctx::new(EOS);
		if let Some(s) = self.scan.as_mut() {
			for di in 0..self.conv.len() {
				s.set_state(di, None)?;
			}
		}
		self.states_on_host = false;
		self.on_gtx = false;
		self.hstate.iter_mut().for_each(|s| s.fill(0.0));
		Ok(())
	}

	/// A copy of the state after everything fed so far (the prefix
	/// cache's entry), as parts of [`Chunk`]s: `[pos, n-gram context]`, the
	/// PLE conv window, the delta conv windows, the delta states (f32, as
	/// the scan device keeps them), then per attention layer its cache
	/// (`k`, `v`, `ik` f16, `bk` f32). The cache's sealed segments are
	/// shared with the engine and with earlier snapshots of the same
	/// conversation, so a snapshot costs ~113 MB plus the open segments
	/// (at most 4096 positions) however long the context; the server counts
	/// a shared chunk once. Restoring it is bit-exact. Fails (instead of
	/// aborting) when the memory is not there.
	pub fn snapshot(&mut self) -> Result<Vec<Vec<Chunk>>, String> {
		self.states_to(true)?;
		let mut meta = Vec::new();
		meta.extend_from_slice(&(self.pos as u64).to_le_bytes());
		for &p in &self.nctx.prev {
			meta.extend_from_slice(&p.to_le_bytes());
		}
		meta.push(self.nctx.seg);
		let cat = |v: &mut Vec<u8>, n: usize| {
			v.try_reserve_exact(n)
				.map_err(|_| format!("flash snapshot: out of memory for {} MB", n >> 20))
		};
		let mut conv = Vec::new();
		cat(&mut conv, self.conv.iter().map(|c| c.len() * 2).sum())?;
		self.conv
			.iter()
			.for_each(|c| conv.extend_from_slice(u8s(c)));
		let mut hs = Vec::new();
		let n: usize = self.hstate.iter().map(|s| s.len()).sum();
		hs.try_reserve_exact(n)
			.map_err(|_| format!("flash snapshot: out of memory for {} MB", (n * 4) >> 20))?;
		self.hstate.iter().for_each(|s| hs.extend_from_slice(s));
		let mut parts = vec![
			snap::part(meta),
			snap::part(snap::try_copy(u8s(&self.cstate))?),
			snap::part(conv),
			vec![Arc::new(snap::Data::F32(hs))],
		];
		for kv in self.kv.iter().skip(3).step_by(4) {
			parts.push(kv.k.snapshot()?);
			parts.push(kv.v.snapshot()?);
			parts.push(kv.ik.snapshot()?);
			parts.push(kv.bk.snapshot()?);
		}
		// after images: their runs (u64 start, w, h each) and a tag
		if !self.imgs.is_empty() {
			parts.push(snap::part(
				self.imgs
					.iter()
					.flat_map(|im| [im.start, im.gw, im.gh])
					.flat_map(|v| (v as u64).to_le_bytes())
					.collect(),
			));
			parts.push(snap::part(Self::IMG_SNAP.to_vec()));
		}
		Ok(parts)
	}

	/// Last part of a snapshot of a state with images.
	const IMG_SNAP: &'static [u8] = b"flash-images";

	/// Replace the state with one from [`Self::snapshot`] at `pos` (or
	/// any chunking of its parts' bytes: one read back from disk). The
	/// cache's full segments are adopted, not copied. On an error the state
	/// is unchanged.
	pub fn restore(&mut self, pos: usize, parts: &[Vec<Chunk>]) -> Result<(), String> {
		self.spec = None;
		self.mtp_clear();
		let tag = |p: &[Chunk]| snap::concat(p)[..] == Self::IMG_SNAP[..];
		let (parts, imgs) = match parts {
			[rest @ .., im, t] if tag(t) && snap::concat(im).len().is_multiple_of(24) => {
				let v: Vec<usize> = snap::concat(im)
					.chunks_exact(8)
					.map(|c| u64::from_le_bytes(c.try_into().unwrap()) as usize)
					.collect();
				let imgs: Vec<Img> = v
					.chunks_exact(3)
					.map(|c| Img {
						start: c[0],
						gw: c[1],
						gh: c[2],
					})
					.collect();
				if imgs
					.iter()
					.any(|im| im.gw * im.gh == 0 || im.start + im.gw * im.gh > pos)
				{
					return Err("flash restore: bad image runs".into());
				}
				(rest, imgs)
			}
			_ => (parts, Vec::new()),
		};
		let nd = self.conv.len();
		let na = NL / 4;
		if parts.len() != 4 + 4 * na {
			return Err(format!("flash restore: {} parts", parts.len()));
		}
		let meta = snap::concat(&parts[0]);
		let np = self.nctx.prev.len();
		if meta.len() != 8 + 4 * np + 1 {
			return Err("flash restore: bad header".into());
		}
		let p = u64::from_le_bytes(meta[..8].try_into().unwrap()) as usize;
		if p != pos {
			return Err(format!("flash restore: snapshot at {p}, asked {pos}"));
		}
		let (cst, conv, hs) = (
			snap::concat(&parts[1]),
			snap::concat(&parts[2]),
			snap::concat(&parts[3]),
		);
		let cw = self.conv.first().map_or(0, |c| c.len());
		let sw = H * D * D;
		if cst.len() != self.cstate.len() * 2
			|| conv.len() != nd * cw * 2
			|| hs.len() != nd * sw * 4
		{
			return Err("flash restore: state sizes differ".into());
		}
		// the caches first: the only part that can fail
		let mut kvs = Vec::with_capacity(na);
		for i in 0..na {
			let b = &parts[4 + 4 * i..8 + 4 * i];
			let kv = (|| -> Result<Kv, String> {
				Ok(Kv {
					k: Col::restore(HKV * DA, &b[0], pos)?,
					v: Col::restore(HKV * DA, &b[1], pos)?,
					ik: Col::restore(ID, &b[2], pos)?,
					bk: Col::restore(ID, &b[3], pos / 4)?,
				})
			})()
			.map_err(|e| format!("flash restore: layer {i}: {e}"))?;
			kvs.push(kv);
		}
		let mut hstate = Vec::with_capacity(nd);
		for c in hs.chunks_exact(sw * 4) {
			let mut h = snap::try_vec(sw, 0f32)?;
			for (o, b) in h.iter_mut().zip(c.chunks_exact(4)) {
				*o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
			}
			hstate.push(h);
		}
		for i in 0..np {
			let o = 8 + 4 * i;
			self.nctx.prev[i] = u32::from_le_bytes(meta[o..o + 4].try_into().unwrap());
		}
		self.nctx.seg = meta[8 + 4 * np];
		u8sm(&mut self.cstate).copy_from_slice(&cst);
		for (di, c) in self.conv.iter_mut().enumerate() {
			u8sm(c).copy_from_slice(&conv[di * cw * 2..(di + 1) * cw * 2]);
		}
		self.hstate = hstate;
		// the states are on the host now; the next scan moves them back
		self.states_on_host = self.scan.is_some();
		self.on_gtx = false;
		for (kv, r) in self.kv.iter_mut().skip(3).step_by(4).zip(kvs) {
			*kv = r;
		}
		if let Some(a) = self.pattn.as_mut() {
			a.invalidate();
		}
		if let Some(f) = self.far.as_mut() {
			f.invalidate();
		}
		if let Some(x) = self.idx.as_mut() {
			x.invalidate();
		}
		self.pos = pos;
		self.imgs = imgs;
		self.lost = false;
		Ok(())
	}

	fn buf(&mut self, b: B) -> &mut HostBuf<'static> {
		&mut self.sets[self.cur].bufs[b as usize]
	}

	fn b16(&self, b: B) -> &[u16] {
		h16(&self.sets[self.cur].bufs[b as usize])
	}

	fn b8(&self, b: B) -> &[u8] {
		self.sets[self.cur].bufs[b as usize].as_slice()
	}

	/// Feed `ids` and return the last position's logits.
	pub fn forward(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
		if ids.is_empty() {
			return Err("forward: no tokens".into());
		}
		self.spec = None;
		self.dense_heal()?;
		let bar = quant::progress::Prefill::start(ids.len());
		let mut last = Vec::new();
		let ps = self.pieces(ids);
		for (i, c) in ps.iter().cloned().enumerate() {
			quant::progress::chunk(c.start, c.len());
			self.prefetch(ids, &ps, i);
			let want = c.end == ids.len();
			last = self.chunk(&ids[c], want)?;
		}
		bar.done();
		Ok(last)
	}

	pub fn feed(&mut self, ids: &[u32]) -> Result<(), String> {
		self.spec = None;
		self.dense_heal()?;
		let bar = quant::progress::Prefill::start(ids.len());
		let ps = self.pieces(ids);
		let t = std::time::Instant::now();
		for (i, c) in ps.iter().cloned().enumerate() {
			quant::progress::chunk(c.start, c.len());
			self.prefetch(ids, &ps, i);
			self.chunk(&ids[c], false)?;
		}
		bar.done();
		if self.timing {
			let (s, b) = self.pattn.as_mut().map_or((0.0, 0), |a| a.take_fill());
			eprintln!(
				"[flash] feed of {} in {} pieces: {:.0} ms, of which the attention mirror's refill {:.0} ms ({} MB)",
				ids.len(),
				ps.len(),
				t.elapsed().as_secs_f64() * 1e3,
				s * 1e3,
				b >> 20
			);
		}
		Ok(())
	}

	/// Positions per step of the one-position paths a prefill piece can
	/// take instead of a prefill chunk: the dense GPU's (up to
	/// [`super::gpudec::GpuDec::max_np`] at once), else 1 with the
	/// one-position set, else 0 (none; an image's positions need the
	/// prefill graphs' inputs).
	fn step_np(&self) -> usize {
		if !self.mm_emb.is_empty() {
			0
		} else if let Some(g) = self.gpu.as_ref() {
			g.max_np()
		} else {
			(self.sets.len() > 1) as usize
		}
	}

	/// The cost of a verify step of 1, 2, ... positions plus its drafts
	/// (`QWEN_FLASH_MTP_COST`; the last repeats; measured at run time with
	/// [`mtp_auto`]), `None` when set to `0`.
	pub fn step_costs(&self) -> Option<Vec<f32>> {
		let c = mtp_cost()?;
		Some(if mtp_auto() { self.meter.costs(&c) } else { c })
	}

	/// Tokens of a full prefill piece ([`Self::pieces`]): every piece
	/// pays ~2.7 s at 12k-17k depth whatever its length (each layer's
	/// pipeline fill and an expert pass over most experts).
	pub fn piece_len(&self) -> usize {
		S * self.sets[0].k
	}

	/// A chunk of `n` positions runs as a step ([`Self::step_gpu`]).
	fn is_step(&self, n: usize) -> bool {
		self.gpu.is_some() && n <= self.step_np().max(1)
	}

	/// `ids` cut into chunks of up to [`KSUP`] x [`S`] (sub-chunks of
	/// [`S`]); a tail of at most [`D1_MAX`] and pieces of at most
	/// [`step_max`] (with the dense GPU; [`D1_MAX`] otherwise) as steps of
	/// [`Self::step_np`] positions.
	fn pieces(&self, ids: &[u32]) -> Vec<std::ops::Range<usize>> {
		let mut v = Vec::new();
		let mut p = 0;
		let np = self.step_np();
		let short = if self.gpu.is_some() {
			step_max()
		} else {
			D1_MAX
		};
		while p < ids.len() {
			let rem = ids.len() - p;
			let n = rem.min(S * self.sets[0].k);
			// a short tail stays for the steps
			let n = if np > 0 && n == rem && n > S && n % S <= D1_MAX {
				n - n % S
			} else {
				n
			};
			// (`QWEN_FLASH_D1=N`: N positions per step, the decode path's)
			let n = if np > 0 && self.d1_all > 0 {
				self.d1_all.min(np).min(n)
			} else if np > 0 && n <= short {
				n.min(np)
			} else {
				n
			};
			v.push(p..p + n);
			p += n;
		}
		v
	}

	/// Before piece `i` of `ps` (of `ids`) runs: start reading piece
	/// `i + 1`'s n-gram rows ([`Prefetch`]), from the n-gram context the
	/// pieces up to `i` will leave, so they are read while piece `i` runs.
	/// Before the first piece also its own rows, in one batch (its
	/// sub-chunks read them 256 tokens at a time otherwise).
	fn prefetch(&mut self, ids: &[u32], ps: &[std::ops::Range<usize>], i: usize) {
		if std::env::var("QWEN_FLASH_PREFETCH").is_ok_and(|v| v == "0") {
			return;
		}
		if i == 0 {
			// (left over from a call that failed)
			self.pf.clear();
		}
		let mut c = self.nctx;
		for j in [i, i + 1] {
			let Some(p) = ps.get(j) else {
				break;
			};
			let rows: Vec<[u64; ngram::HEADS]> = ids[p.clone()]
				.iter()
				.map(|&t| self.hash.ids(c.push(t, EOS)))
				.collect();
			if p.len() <= 1 || self.is_step(p.len()) || (j == i && i > 0) {
				continue;
			}
			let Ok(h) = self.table.gather_async(&rows, 32) else {
				continue;
			};
			self.pf.push(Prefetch {
				rows,
				e: PfData::Run(h),
			});
		}
	}

	/// The n-gram embeddings of `rows` from a [`Self::pf`] that holds them
	/// (waiting for its reads), else read now.
	fn ngram_rows(&mut self, rows: &[[u64; ngram::HEADS]]) -> Result<Vec<u16>, String> {
		let w = ngram::HEADS * ngram::DIM;
		for k in 0..self.pf.len() {
			let pf = &mut self.pf[k];
			let Some(o) = pf.rows.windows(rows.len()).position(|r| r == rows) else {
				continue;
			};
			if let PfData::Run(_) = pf.e {
				let PfData::Run(h) = std::mem::replace(&mut pf.e, PfData::Failed) else {
					unreachable!()
				};
				if let Ok(e) = h.wait() {
					pf.e = PfData::Done(e);
				}
			}
			let PfData::Done(e) = &pf.e else {
				continue;
			};
			let v = e[o * w..][..rows.len() * w].to_vec();
			if o + rows.len() == pf.rows.len() {
				self.pf.remove(k);
			}
			return Ok(v);
		}
		self.table.gather(rows, 16)
	}

	/// One chunk of `n` <= S tokens through every layer.
	/// Make set `cur` the active one (0: the prefill set, whose buffers a
	/// GPU step uses too; 1: the one-position set).
	fn use_set(&mut self, cur: usize) -> Result<(), String> {
		if cur == self.cur {
			// (no-op unless [`Self::release_npu`] unloaded it)
			return self.sets[cur].activate(self.npu, &self.weights, if cur == 1 { "_d1" } else { "" }, self.wb);
		}
		let t = std::time::Instant::now();
		// the far graphs only next to the prefill set (more graphs at
		// once have lost the device)
		self.far = None;
		self.far_tried = false;
		self.sets[self.cur].deactivate();
		let sfx = if cur == 1 { "_d1" } else { "" };
		self.sets[cur].activate(self.npu, &self.weights, sfx, self.wb)?;
		self.cur = cur;
		if self.timing {
			eprintln!(
				"[flash] set {cur} recorded in {:.0} ms",
				t.elapsed().as_secs_f64() * 1e3
			);
		}
		Ok(())
	}

	fn chunk(&mut self, ids: &[u32], logits: bool) -> Result<Vec<f32>, String> {
		let n = ids.len();
		if self.is_step(n) {
			return self.step_gpu(ids, logits);
		}
		self.gtx_home()?;
		self.use_set(if n == 1 && self.sets.len() > 1 { 1 } else { 0 })?;
		let sn = self.sets[self.cur].s;
		let t0 = std::time::Instant::now();
		let pos0 = self.pos;
		// a hyper-connection left in flight by a chunk that failed is not
		// this chunk's
		self.phc_flight = None;
		self.comb = None;
		// layer-major over sub-chunks of `sn` positions: per layer every
		// sub-chunk's dense part, then one expert pass over all of them
		// (an expert job costs about the same for 1 or 32 tokens)
		let mut st: Vec<Sub> = Vec::new();
		let t_in = std::time::Instant::now();
		for (j, ids) in ids.chunks(sn).enumerate() {
			self.pos = pos0 + j * sn;
			self.keep_awake(ids.len())?;
			let nx = B::x.bytes(sn, 0, 0);
			let mut x = match self.xpool.iter().position(|b| b.len() >= nx) {
				Some(i) => self.xpool.swap_remove(i),
				None => self.npu.alloc(nx)?,
			};
			let ins = self.inputs(ids, &mut h16m(&mut x)[..nx / 2])?;
			st.push(Sub {
				j,
				n: ids.len(),
				pos: self.pos,
				x,
				ins,
			});
		}
		self.co_ensure(sn, st.len())?;
		let t_in = t_in.elapsed().as_secs_f64();
		let mut di = 0;
		let (mut t_npu, mut t_mix, mut t_exp, mut t_att, mut t_scan, mut t_wait) =
			(0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
		for l in 0..NL {
			crate::ze::trace_tag(l as u32);
			let tn0 = t_npu;
			let tl0 = std::time::Instant::now();
			let cur = self.cur;
			let nb = self.sets[cur].banks();
			let k = st.len();
			// the PLE term of every sub-chunk before the layer's dense part
			// (it reads only the streams and the n-gram rows), so the layer
			// pipelines as any delta layer; else in each sub-chunk's
			// `pre_prep`, waiting for post(j - 1) first
			let ple_first = l == PLE_LAYER && ple_first();
			if ple_first {
				self.comb_to(&mut st, usize::MAX);
				t_npu += self.ple_all(&mut st)?;
			}
			let ple_in = l == PLE_LAYER && !ple_first;
			if self.sets[cur].split && !is_attn(l) && nb == 2 && !ple_in && cur == 0 {
				let (tn, a, c, m, w) = self.delta_split(l, di, &mut st)?;
				t_npu += tn;
				t_att += a;
				t_scan += c;
				t_mix += m;
				t_wait += w;
			} else if nb == 1 {
				for j in 0..k {
					self.comb_to(&mut st, j);
					t_npu += self.pre_prep(l, di, &mut st[j], ple_in)?;
					let t = std::time::Instant::now();
					self.sets[cur].lists[l]
						.0
						.run()
						.map_err(|e| format!("set {cur} layer {l} pre: {e}"))?;
					t_npu += t.elapsed().as_secs_f64();
					self.pre_done(l, di);
					self.pos = st[j].pos;
					let (a, c, m) = self.mixer(l, di, st[j].n)?;
					t_att += a;
					t_scan += c;
					t_mix += m;
					self.bind_io(l, 1, &st[j])?;
					let t = std::time::Instant::now();
					self.sets[cur].lists[l]
						.1
						.run()
						.map_err(|e| format!("set {cur} layer {l} post: {e}"))?;
					t_npu += t.elapsed().as_secs_f64();
				}
			} else {
				// sub-chunk j on bank j % 2. Per j: stage and submit
				// pre(j + 1) (its bank may still run post(j - 1): post
				// reads its own streams `xq`, and the queue runs in order,
				// so pre(j + 1) writes z / inj after post(j - 1) read
				// them), run mixer(j), submit post(j), then wait for
				// post(j - 1) and pre(j + 1). The NPU runs post(j - 1) and
				// pre(j + 1) during mixer(j).
				let bank = |j: usize| j % nb;
				self.sets[cur].use_bank(0);
				self.comb_to(&mut st, 0);
				t_npu += self.pre_prep(l, di, &mut st[0], ple_in)?;
				if k > 1 && !ple_in {
					if !is_attn(l) {
						self.comb_to(&mut st, 1);
					}
					self.hc_begin(l, di, &st[1])?;
				}
				let t = std::time::Instant::now();
				self.sets[cur].lists[l]
					.0
					.run()
					.map_err(|e| format!("set {cur} layer {l} pre: {e}"))?;
				t_npu += t.elapsed().as_secs_f64();
				self.pre_done(l, di);
				if is_attn(l) {
					self.idx_begin(l, st[0].pos, st[0].n)?;
					self.idx_gpu_begin(l, st[0].pos, st[0].n)?;
				}
				for j in 0..k {
					// an attention layer's device attention runs while the
					// host stages pre(j + 1) and saves post(j - 1): those
					// copies are ~1/6 of the layer when the attention is the
					// longer side (the iGPU idled through them)
					let aa = is_attn(l) && self.pattn.is_some() && st[j].n > 1;
					if aa {
						self.sets[cur].use_bank(bank(j));
						self.pos = st[j].pos;
						let t = std::time::Instant::now();
						self.attention_begin(l, st[j].n)?;
						let a = t.elapsed().as_secs_f64();
						t_att += a;
						t_mix += a;
					}
					if j + 1 < k {
						self.sets[cur].use_bank(bank(j + 1));
						if ple_in && j >= 1 {
							// the PLE graph runs (and waits) in pre_prep
							self.sets[cur].lists_at(bank(j - 1))[l]
								.1
								.wait()
								.map_err(|e| format!("set {cur} layer {l} post: {e}"))?;
						}
						self.comb_to(&mut st, j + 1);
						t_npu += self.pre_prep(l, di, &mut st[j + 1], ple_in)?;
						self.sets[cur].lists[l]
							.0
							.submit()
							.map_err(|e| format!("set {cur} layer {l} pre: {e}"))?;
						// the indexer's scores of j + 1 right behind its pre
						if is_attn(l) {
							self.idx_begin(l, st[j + 1].pos, st[j + 1].n)?;
						}
						// the iGPU's hyper-connection of j + 2 runs while
						// the NPU runs post(j - 1) / pre(j + 1)
						if j + 2 < k && !ple_in {
							if !is_attn(l) {
								self.comb_to(&mut st, j + 2);
							}
							self.hc_begin(l, di, &st[j + 2])?;
						}
					}
					if aa {
						let t = std::time::Instant::now();
						if j >= 1 {
							self.sets[cur].lists_at(bank(j - 1))[l]
								.1
								.wait()
								.map_err(|e| format!("set {cur} layer {l} post: {e}"))?;
						}
						t_wait += t.elapsed().as_secs_f64();
						self.sets[cur].use_bank(bank(j));
						let t = std::time::Instant::now();
						self.attention_end()?;
						self.keep_awake(st[j].n)?;
						let a = t.elapsed().as_secs_f64();
						if std::env::var_os("QWEN_FLASH_ATT_LOG").is_some() {
							eprintln!(
								"[att] l {l} j {j}: end wait {:.1} ms at {:.1}",
								a * 1e3,
								tl0.elapsed().as_secs_f64() * 1e3
							);
						}
						t_att += a;
						t_mix += a;
						self.bind_io(l, 1, &st[j])?;
						self.sets[cur].lists[l]
							.1
							.submit()
							.map_err(|e| format!("set {cur} layer {l} post: {e}"))?;
						let t = std::time::Instant::now();
						if j + 1 < k {
							self.sets[cur].lists_at(bank(j + 1))[l]
								.0
								.wait()
								.map_err(|e| format!("set {cur} layer {l} pre: {e}"))?;
							self.sets[cur].use_bank(bank(j + 1));
							self.pre_done(l, di);
							self.idx_gpu_begin(l, st[j + 1].pos, st[j + 1].n)?;
						}
						t_wait += t.elapsed().as_secs_f64();
						continue;
					}
					self.sets[cur].use_bank(bank(j));
					self.pos = st[j].pos;
					let (a, c, m) = self.mixer(l, di, st[j].n)?;
					t_att += a;
					t_scan += c;
					t_mix += m;
					self.bind_io(l, 1, &st[j])?;
					self.sets[cur].lists[l]
						.1
						.submit()
						.map_err(|e| format!("set {cur} layer {l} post: {e}"))?;
					let t = std::time::Instant::now();
					if j >= 1 && !(ple_in && j + 1 < k) {
						self.sets[cur].lists_at(bank(j - 1))[l]
							.1
							.wait()
							.map_err(|e| format!("set {cur} layer {l} post: {e}"))?;
					}
					if j + 1 < k {
						self.sets[cur].lists_at(bank(j + 1))[l]
							.0
							.wait()
							.map_err(|e| format!("set {cur} layer {l} pre: {e}"))?;
						self.sets[cur].use_bank(bank(j + 1));
						self.pre_done(l, di);
					}
					t_wait += t.elapsed().as_secs_f64();
				}
				let t = std::time::Instant::now();
				self.sets[cur].lists_at(bank(k - 1))[l]
					.1
					.wait()
					.map_err(|e| format!("set {cur} layer {l} post: {e}"))?;
				t_wait += t.elapsed().as_secs_f64();
				self.sets[cur].use_bank(0);
			}
			if std::env::var_os("QWEN_FLASH_LAYER_LOG").is_some() {
				eprintln!(
					"[npu] l {l}: {:.1} ms npu, {:.1} ms dense part",
					(t_npu - tn0) * 1e3,
					tl0.elapsed().as_secs_f64() * 1e3
				);
			}
			if !is_attn(l) {
				di += 1;
			}
			let t = std::time::Instant::now();
			self.moe(l, &mut st)?;
			t_exp += t.elapsed().as_secs_f64();
			quant::progress::part(l + 1, NL);
		}
		self.comb_to(&mut st, usize::MAX);
		self.pos = pos0 + ids.len();
		// the last sub-chunk's streams where the head reads them; every
		// sub-chunk's for `forward_all`
		let nx = B::x.bytes(sn, 0, 0);
		let last = st.last().ok_or("chunk: no tokens")?;
		let set = &mut self.sets[self.cur];
		set.bufs[B::x as usize].as_mut_slice()[..nx].copy_from_slice(&last.x.as_slice()[..nx]);
		self.xfin.clear();
		for sb in st {
			self.xfin.push((sb.n, sb.x.as_slice()[..nx].to_vec()));
			self.xpool.push(sb.x);
		}
		if self.mtp.is_some() {
			// the last positions' streams for the MTP head's rows
			let first = ids.len().saturating_sub(mtp_ctx());
			let mut p = 0;
			for (n, xb) in &self.xfin {
				for t in 0..*n {
					if p + t >= first {
						let v: Vec<f32> = xb[t * HCW * 2..][..HCW * 2]
							.chunks_exact(2)
							.map(|c| unf16(u16::from_le_bytes([c[0], c[1]])))
							.collect();
						self.mpend.push((pos0 + p + t, v, ids[p + t]));
					}
				}
				p += n;
			}
			self.mpend_cap();
		}
		// the step line's attention time (this chunk has its own)
		self.t_att = 0.0;
		let t = std::time::Instant::now();
		let nl = ids.len() - (ids.len() - 1) / sn * sn;
		let out = if logits {
			self.head(nl - 1)
		} else {
			Vec::new()
		};
		if self.timing {
			crate::ze::trace_flush();
			eprintln!(
				"[flash] chunk {} tokens: {:.0} ms (inputs {:.0} = emb {:.0} + n-gram {:.0}, npu dense {:.0} + wait {:.0}, mixer {:.0} of which attention {:.0} scan {:.0}, hc {:.0}, moe {:.0} = record {:.0} + run {:.0} + gather {:.0} + combine {:.0} (of which device wait {:.0}), head {:.0})",
				ids.len(),
				t0.elapsed().as_secs_f64() * 1e3,
				t_in * 1e3,
				self.t_inp[0] * 1e3,
				self.t_inp[1] * 1e3,
				t_npu * 1e3,
				t_wait * 1e3,
				t_mix * 1e3,
				t_att * 1e3,
				t_scan * 1e3,
				self.t_hc * 1e3,
				t_exp * 1e3,
				self.t_moe[0] * 1e3,
				self.t_moe[1] * 1e3,
				self.t_moe[2] * 1e3,
				self.t_moe[3] * 1e3,
				self.t_moe[4] * 1e3,
				t.elapsed().as_secs_f64() * 1e3
			);
			self.t_moe = [0.0; 5];
		}
		self.t_inp = [0.0; 2];
		self.t_hc = 0.0;
		Ok(out)
	}

	/// A delta layer's dense part with `post` split around its MoE
	/// hyper-connection (`Set::split`): sub-chunk `j` on bank `j % 2`, the
	/// NPU runs `pre(j)`, `post_delta_a(j)` and two sub-chunks later
	/// `post_delta_b(j)`; the scan device's adapter runs the attention
	/// hyper-connection (`phc`), the recurrence and the MoE
	/// hyper-connection of `x1(j)` (`phm`) between them, so `post_b`
	/// never waits for it. The three are separate contexts the iGPU runs
	/// one after another in submission order, and the recurrence is the
	/// chain every sub-chunk waits for: so per `j` the scan goes first
	/// (3.1 ms on the GPU; queued behind the two hyper-connections the
	/// host waited 6.9), then `phc(j + 2)` and `phm(j - 1)` behind it.
	/// Per `j`: submit `post_b(j - 2)`, begin scan(j), `phc(j + 2)`,
	/// `phm(j - 1)`, wait for the scan and submit `post_a(j)`, collect
	/// `post_b(j - 2)` and `pre(j + 1)`, stage and submit `pre(j + 2)` on
	/// bank `j` (the queue runs it after `post_a(j)`, which reads its `z` /
	/// `inj`), wait for `post_a(j)`: save `x1(j)`, collect `phm(j - 1)`.
	/// (NPU, attention, scan, mixer, wait) seconds.
	fn delta_split(
		&mut self,
		l: usize,
		di: usize,
		st: &mut [Sub],
	) -> Result<(f64, f64, f64, f64, f64), String> {
		let cur = self.cur;
		let k = st.len();
		let bank = |j: usize| j % 2;
		let (mut tn, mut tc, mut tm, mut tw) = (0f64, 0f64, 0f64, 0f64);
		let err = |w: &'static str| move |e: String| format!("set {cur} layer {l} {w}: {e}");
		let log = std::env::var_os("QWEN_FLASH_LAYER_LOG").is_some();
		// pre(0) now, pre(1) submitted
		self.sets[cur].use_bank(0);
		self.comb_to(st, 0);
		tn += self.pre_prep(l, di, &mut st[0], false)?;
		if k > 1 {
			self.comb_to(st, 1);
			self.hc_begin(l, di, &st[1])?;
		}
		let t = std::time::Instant::now();
		self.sets[cur].lists[l].0.run().map_err(err("pre"))?;
		tn += t.elapsed().as_secs_f64();
		self.pre_done(l, di);
		if k > 1 {
			self.sets[cur].use_bank(1);
			tn += self.pre_prep(l, di, &mut st[1], false)?;
			self.sets[cur].lists[l].0.submit().map_err(err("pre"))?;
		}
		for j in 0..k {
			if j >= 2 {
				self.post_b_submit(l, &st[j - 2])?;
			}
			self.sets[cur].use_bank(bank(j));
			self.pos = st[j].pos;
			let (t, h0, c0) = (std::time::Instant::now(), self.t_hc, self.t_moe[3]);
			let pending = if self.sets[cur].gn {
				self.recurrence_begin_gn(di, st[j].n)?
			} else {
				self.recurrence_begin(di, st[j].n)?
			};
			if j + 2 < k {
				// the last expert pass's combine of j + 2 while the scan runs
				self.comb_to(st, j + 2);
				self.hc_begin(l, di, &st[j + 2])?;
			}
			if j >= 1 {
				self.hcm_begin(di, &st[j - 1])?;
			}
			if pending {
				self.recurrence_end()?;
			}
			self.poked = std::time::Instant::now();
			// the begins' host time is the hyper-connections' (`t_hc`), the
			// combine's the expert pass's
			let a = t.elapsed().as_secs_f64() - (self.t_hc - h0) - (self.t_moe[3] - c0);
			if log {
				eprintln!(
					"[scan] l {l} pos {} n {}: {:.2} ms",
					self.pos,
					st[j].n,
					a * 1e3
				);
			}
			tc += a;
			tm += a;
			self.bind_io(l, 1, &st[j])?;
			self.sets[cur].lists[l].1.submit().map_err(err("post_a"))?;
			let t = std::time::Instant::now();
			if j >= 2 {
				self.post_b_wait(l, j - 2)?;
			}
			if j + 1 < k {
				self.sets[cur].lists_at(bank(j + 1))[l]
					.0
					.wait()
					.map_err(err("pre"))?;
				self.sets[cur].use_bank(bank(j + 1));
				self.pre_done(l, di);
			}
			tw += t.elapsed().as_secs_f64();
			if j + 2 < k {
				self.sets[cur].use_bank(bank(j));
				tn += self.pre_prep(l, di, &mut st[j + 2], false)?;
				self.sets[cur].lists[l].0.submit().map_err(err("pre"))?;
			}
			let t = std::time::Instant::now();
			self.sets[cur].lists_at(bank(j))[l]
				.1
				.wait()
				.map_err(err("post_a"))?;
			tw += t.elapsed().as_secs_f64();
			if j >= 1 {
				self.hcm_end(j - 1, st[j - 1].n)?;
			}
		}
		let t = std::time::Instant::now();
		self.hcm_begin(di, &st[k - 1])?;
		if k >= 2 {
			self.post_b_submit(l, &st[k - 2])?;
		}
		self.hcm_end(k - 1, st[k - 1].n)?;
		self.post_b_submit(l, &st[k - 1])?;
		if k >= 2 {
			self.post_b_wait(l, k - 2)?;
		}
		self.post_b_wait(l, k - 1)?;
		tw += t.elapsed().as_secs_f64();
		self.sets[cur].use_bank(0);
		Ok((tn, 0.0, tc, tm, tw))
	}

	/// Start `phm`'s hyper-connection of sub-chunk `sb`'s `x1` (its
	/// `post_a` waited for), read in place from its [`Co`] slot.
	fn hcm_begin(&mut self, di: usize, sb: &Sub) -> Result<(), String> {
		let t = std::time::Instant::now();
		let h = self
			.phm
			.as_mut()
			.ok_or("flash: post_delta_a needs set_prefill_hcm")?;
		h.begin(di, u16s(self.co.at(CO_X1, sb.j)), sb.n)?;
		self.t_hc += t.elapsed().as_secs_f64();
		Ok(())
	}

	/// Collect `phm`'s hyper-connection of sub-chunk `j` (`n` positions):
	/// the mixed input straight into bank `j % 2`'s `mm` (its last
	/// `post_delta_b`, j - 2's, was waited for and `post_b(j)` is not yet
	/// submitted; nothing else binds `mm`), padding rows zero; the
	/// injection weights into its [`Co`] slot (the last expert pass's
	/// combine read the previous layer's before its pre).
	fn hcm_end(&mut self, j: usize, n: usize) -> Result<(), String> {
		let t = std::time::Instant::now();
		let cur = self.cur;
		let sn = self.sets[cur].s;
		let inj = u16sm(self.co.at_mut(CO_INJ, j));
		let set = &mut self.sets[cur];
		let was = set.bk;
		set.use_bank(j % 2);
		let m = &mut h16m(&mut set.bufs[B::mm as usize])[..sn * W];
		let r = self
			.phm
			.as_mut()
			.ok_or_else(|| "flash: post_delta_a needs set_prefill_hcm".to_string())
			.and_then(|h| h.end(m, inj));
		m[n * W..].fill(0);
		set.use_bank(was);
		r?;
		self.t_hc += t.elapsed().as_secs_f64();
		Ok(())
	}

	/// Submit `post_delta_b` of sub-chunk `sb` on its bank (its input `mm`
	/// written by [`Self::hcm_end`]), its outputs into `sb`'s [`Co`] slots.
	fn post_b_submit(&mut self, l: usize, sb: &Sub) -> Result<(), String> {
		let cur = self.cur;
		self.sets[cur].use_bank(sb.j % 2);
		self.bind_io(l, 2, sb)?;
		self.sets[cur].lists[l]
			.2
			.as_mut()
			.ok_or("flash: post_delta_b not recorded")?
			.submit()
			.map_err(|e| format!("set {cur} layer {l} post_b: {e}"))
	}

	/// Wait for `post_delta_b` of sub-chunk `j` (bank `j % 2`).
	fn post_b_wait(&mut self, l: usize, j: usize) -> Result<(), String> {
		let cur = self.cur;
		self.sets[cur].lists_at(j % 2)[l]
			.2
			.as_mut()
			.ok_or("flash: post_delta_b not recorded")?
			.wait()
			.map_err(|e| format!("set {cur} layer {l} post_b: {e}"))
	}

	/// The PLE term added to every sub-chunk's saved streams, one PLE
	/// graph after another on bank 0 (each continues the last one's conv
	/// state): NPU seconds.
	fn ple_all(&mut self, st: &mut [Sub]) -> Result<f64, String> {
		self.sets[self.cur].use_bank(0);
		let mut tn = 0.0;
		let log = std::env::var_os("QWEN_FLASH_LAYER_LOG").is_some();
		for sb in st.iter_mut() {
			let t = std::time::Instant::now();
			let a = self.pre_inputs(sb, true)?;
			tn += a;
			if log {
				eprintln!(
					"[ple] pos {} n {}: {:.2} ms npu, {:.2} ms",
					sb.pos,
					sb.n,
					a * 1e3,
					t.elapsed().as_secs_f64() * 1e3
				);
			}
		}
		Ok(tn)
	}

	/// Sub-chunk `sb`'s runtime inputs of layer `l` into the active bank,
	/// with `ple` the PLE term first (NPU seconds returned).
	fn pre_prep(&mut self, l: usize, di: usize, sb: &mut Sub, ple: bool) -> Result<f64, String> {
		let tn = self.pre_inputs(sb, ple)?;
		self.bind_io(l, 0, sb)?;
		if !is_attn(l) {
			let cp = self.conv[di].clone();
			h16m(self.buf(B::cpast)).copy_from_slice(&cp);
			if self.hc_m && self.cur == 0 {
				self.pre_hc(di, sb, ple)?;
			}
		}
		Ok(tn)
	}

	/// [`Self::pre_prep`]'s runtime inputs and PLE term, without the
	/// layer's own (conv window, hyper-connection): all [`Self::ple_all`]
	/// needs (its hyper-connection would be layer 0's, and discarded).
	fn pre_inputs(&mut self, sb: &mut Sub, ple: bool) -> Result<f64, String> {
		self.comb_check(sb)?;
		self.keep_awake(sb.n)?;
		let mut tn = 0.0;
		for (&b, v) in SUB_IN.iter().zip(&sb.ins) {
			if !matches!(b, B::e) || ple {
				self.buf(b).as_mut_slice()[..v.len()].copy_from_slice(v);
			}
		}
		if ple {
			// the PLE graph reads the bank's `x` and writes `xo`, the
			// streams with the term, back into the sub-chunk's
			let nx = B::x.bytes(self.sets[self.cur].s, 0, 0);
			self.buf(B::x).as_mut_slice()[..nx].copy_from_slice(&sb.x.as_slice()[..nx]);
			let st = self.cstate.clone();
			h16m(self.buf(B::cstate)).copy_from_slice(&st);
			let t = std::time::Instant::now();
			self.sets[self.cur]
				.ple
				.as_mut()
				.ok_or("ple list not recorded")?
				.run()
				.map_err(|e| format!("set {} ple: {e}", self.cur))?;
			tn += t.elapsed().as_secs_f64();
			sb.x.as_mut_slice()[..nx].copy_from_slice(&self.b8(B::xo)[..nx]);
			let cs = self.b16(B::cstate_out).to_vec();
			self.cstate.copy_from_slice(&cs);
		}
		Ok(tn)
	}

	/// `delta_pre_m`'s inputs `mixed` / `hinj` of the active bank for
	/// sub-chunk `sb`: from [`Self::hc_begin`]'s work if it is that one's,
	/// else from its streams now (with `ple` they hold the PLE term).
	fn pre_hc(&mut self, di: usize, sb: &Sub, ple: bool) -> Result<(), String> {
		let t = std::time::Instant::now();
		if ple || self.phc_flight != Some((di, sb.pos)) {
			let h = self
				.phc
				.as_mut()
				.ok_or("flash: delta_pre_m needs set_prefill_hc")?;
			h.begin(di, h16(&sb.x), sb.n)?;
		}
		self.phc_flight = None;
		let set = &mut self.sets[self.cur];
		let [mixed, hinj] = [B::mixed, B::hinj].map(|b| b as usize);
		let (lo, hi) = set.bufs.split_at_mut(hinj);
		let h = self
			.phc
			.as_mut()
			.ok_or("flash: delta_pre_m needs set_prefill_hc")?;
		h.end(h16m(&mut lo[mixed]), h16m(&mut hi[0]))?;
		self.t_hc += t.elapsed().as_secs_f64();
		Ok(())
	}

	/// The last expert pass's combine owed to sub-chunks up to `j` of `st`
	/// (all of them for `usize::MAX`), done now ([`moe_defer`]).
	fn comb_to(&mut self, st: &mut [Sub], j: usize) {
		let Some(c) = self.comb.as_mut() else { return };
		let j1 = j.saturating_add(1).min(c.k);
		if j1 > c.done {
			let t = std::time::Instant::now();
			comb_run(c, st, &self.co, c.done, j1);
			c.done = j1;
			self.t_moe[3] += t.elapsed().as_secs_f64();
		}
		if c.done == c.k {
			self.comb = None;
		}
	}

	/// An error if sub-chunk `sb`'s streams still lack the last expert
	/// pass ([`Self::comb_to`] not called for it).
	fn comb_check(&self, sb: &Sub) -> Result<(), String> {
		match &self.comb {
			Some(c) if sb.pos >= c.pos0 && (sb.pos - c.pos0) / c.sn >= c.done => Err(format!(
				"flash: streams at {} read before the expert pass's combine",
				sb.pos
			)),
			_ => Ok(()),
		}
	}

	/// Start delta layer `di`'s hyper-connection of sub-chunk `sb` on
	/// `phc` (its streams as saved: not at a PLE layer that adds the term
	/// in `pre_prep`), for its `pre_prep` to collect.
	fn hc_begin(&mut self, l: usize, di: usize, sb: &Sub) -> Result<(), String> {
		if !self.hc_m || self.cur != 0 || is_attn(l) {
			return Ok(());
		}
		self.comb_check(sb)?;
		let t = std::time::Instant::now();
		let h = self
			.phc
			.as_mut()
			.ok_or("flash: delta_pre_m needs set_prefill_hc")?;
		h.begin(di, h16(&sb.x), sb.n)?;
		self.phc_flight = Some((di, sb.pos));
		self.t_hc += t.elapsed().as_secs_f64();
		Ok(())
	}

	/// Point the active bank's list `w` (0 pre, 1 post, 2
	/// `post_delta_b`) of layer `l` at sub-chunk `sb`'s streams and its
	/// [`Co`] slots, before it is submitted (the list is not in flight: its
	/// last run, sub-chunk j - 2's, was waited for). Pre and post read the
	/// streams in place, so pre(j + 2) staged on post(j)'s bank needs no
	/// copy of them, and the outputs need no copy out of the bank.
	fn bind_io(&mut self, l: usize, w: usize, sb: &Sub) -> Result<(), String> {
		let Set { ri, lists, .. } = &mut self.sets[self.cur];
		let ri = &ri[l][w];
		if ri.is_empty() {
			return Ok(());
		}
		let co = &self.co;
		let a: Vec<(usize, usize, &HostBuf<'_>, usize)> = ri
			.iter()
			.map(|&(b, i)| {
				let (buf, o) = match CO.iter().position(|&c| c as usize == b as usize) {
					Some(c) => co.slot(c, sb.j),
					None => (&sb.x, 0),
				};
				(0, i, buf, o)
			})
			.collect();
		let list = match w {
			0 => &mut lists[l].0,
			1 => &mut lists[l].1,
			_ => lists[l]
				.2
				.as_mut()
				.ok_or("flash: post_delta_b not recorded")?,
		};
		list.update_at(&a)
	}

	/// [`Self::co`] for `k` sub-chunks of `sn` positions.
	fn co_ensure(&mut self, sn: usize, k: usize) -> Result<(), String> {
		self.co.sn = sn;
		for i in 0..CO.len() {
			let need = Co::stride(i, sn) * k;
			if i == self.co.bufs.len() {
				self.co.bufs.push(self.npu.alloc(need)?);
			} else if self.co.bufs[i].len() < need {
				let b = std::mem::replace(&mut self.co.bufs[i], self.npu.alloc(need)?);
				self.co_old.push(b);
			}
		}
		Ok(())
	}

	/// After the active bank's pre graph: the conv window it moved on.
	fn pre_done(&mut self, l: usize, di: usize) {
		if is_attn(l) {
			return;
		}
		if !self.sets[self.cur].conv_c {
			let co = self.b16(B::conv_out).to_vec();
			self.conv[di].copy_from_slice(&co);
			return;
		}
		// `delta_pre_c`: rows n, n + 1, n + 2 of [cpast; qkv * qs] (as
		// the graph's gather of `keep`), n from `keep`
		let bufs = &self.sets[self.cur].bufs;
		let n = i32::from_le_bytes(bufs[B::keep3 as usize].as_slice()[..4].try_into().unwrap())
			as usize;
		let qs = &self.cvw[di][4 * NQKV..];
		let (cp, qkv) = (h16(&bufs[B::cpast as usize]), h16(&bufs[B::qkv as usize]));
		let cv = &mut self.conv[di];
		for r in 0..3 {
			let i = n + r;
			let dst = &mut cv[r * NQKV..][..NQKV];
			if i < 3 {
				dst.copy_from_slice(&cp[i * NQKV..][..NQKV]);
			} else {
				let src = &qkv[(i - 3) * NQKV..][..NQKV];
				for c in 0..NQKV {
					dst[c] = f16(unf16(src[c]) * qs[c]);
				}
			}
		}
	}

	/// The mixer of the active bank's sub-chunk of `n` positions at
	/// `self.pos`: (attention, scan, all) seconds.
	fn mixer(&mut self, l: usize, di: usize, n: usize) -> Result<(f64, f64, f64), String> {
		let t = std::time::Instant::now();
		if is_attn(l) {
			self.attention(l, n)?;
			self.keep_awake(n)?;
			let a = t.elapsed().as_secs_f64();
			Ok((a, 0.0, a))
		} else {
			self.recurrence(di, n)?;
			self.poked = std::time::Instant::now();
			let a = t.elapsed().as_secs_f64();
			if std::env::var_os("QWEN_FLASH_LAYER_LOG").is_some() {
				eprintln!("[scan] l {l} pos {} n {n}: {:.2} ms", self.pos, a * 1e3);
			}
			Ok((0.0, a, a))
		}
	}

	/// An attention layer, the PLE graph or staging a chunk's inputs leave
	/// the scan device idle long enough that it drops to a power state the
	/// next scan then pays 90-520 ms for (3.7 otherwise): `Scan::poke` once
	/// it was idle for 30 ms (prefill only).
	fn keep_awake(&mut self, n: usize) -> Result<(), String> {
		if !self.poke || n <= 1 || self.poked.elapsed().as_millis() < 30 {
			return Ok(());
		}
		if let Some(sc) = self.scan.as_mut() {
			sc.poke()?;
		}
		self.poked = std::time::Instant::now();
		Ok(())
	}

	/// The runtime inputs of a chunk of `ids` at `self.pos`: the streams
	/// into `x` (embeddings repeated; padding rows zero), and the
	/// [`SUB_IN`] buffers returned (the n-gram embeddings, the masks and
	/// RoPE).
	fn inputs(&mut self, ids: &[u32], x: &mut [u16]) -> Result<Vec<Vec<u8>>, String> {
		let n = ids.len();
		let sn = self.sets[self.cur].s;
		let t = std::time::Instant::now();
		{
			let mut row = vec![0u8; W * 2];
			for (t, &id) in ids.iter().enumerate() {
				let p = self.pos + t;
				let xt = &mut x[t * HCW..][..HCW];
				if let Some((s, r)) = self
					.mm_emb
					.iter()
					.find(|(s, r)| p >= *s && p < s + r.len() / W)
				{
					xt[..W].copy_from_slice(&r[(p - s) * W..][..W]);
				} else {
					let mut done = 0;
					while done < row.len() {
						let r = self
							.emb
							.seek_read(&mut row[done..], id as u64 * (W as u64 * 2) + done as u64)
							.map_err(|e| format!("embedding table: {e}"))?;
						if r == 0 {
							return Err("embedding table: short read".into());
						}
						done += r;
					}
					for (o, b) in xt[..W].iter_mut().zip(row.chunks_exact(2)) {
						*o = u16::from_le_bytes([b[0], b[1]]);
					}
				}
				for c in 1..HC {
					xt.copy_within(..W, c * W);
				}
			}
			x[n * HCW..sn * HCW].fill(0);
		}
		self.t_inp[0] += t.elapsed().as_secs_f64();
		let t = std::time::Instant::now();
		// n-gram embeddings
		let rows: Vec<[u64; ngram::HEADS]> = ids
			.iter()
			.map(|&t| self.hash.ids(self.nctx.push(t, EOS)))
			.collect();
		let e = self.ngram_rows(&rows)?;
		self.t_inp[1] += t.elapsed().as_secs_f64();
		let le16 = |v: &mut Vec<u8>, h: u16| v.extend_from_slice(&h.to_le_bytes());
		let mut ins = Vec::with_capacity(SUB_IN.len());
		for b in SUB_IN {
			let bytes = b.bytes(sn, 0, 0);
			let mut v = Vec::with_capacity(bytes);
			match b {
				B::e => {
					for &h in e.iter().take(bytes / 2) {
						le16(&mut v, f16(f32::from_bits((h as u32) << 16)));
					}
				}
				B::valid => {
					for t in 0..sn {
						le16(&mut v, f16(if t < n { 1.0 } else { 0.0 }));
					}
				}
				B::keep3 | B::keep9 => {
					for j in 0..bytes / 4 {
						v.extend_from_slice(&((n + j) as i32).to_le_bytes());
					}
				}
				B::cos | B::sin => {
					let mut h = vec![0u16; sn * RD];
					for t in 0..n {
						let sc = self.rope_sc(self.pos + t);
						for (i, &(a, c)) in sc.iter().enumerate() {
							for j in [i, i + RD / 2] {
								h[t * RD + j] = f16(if matches!(b, B::cos) { c } else { a } as f32);
							}
						}
					}
					v.extend_from_slice(u8s(&h));
				}
				_ => unreachable!("not a sub-chunk input"),
			}
			v.resize(bytes, 0);
			ins.push(v);
		}
		Ok(ins)
	}

	/// Move the delta states to the host (`on_host`) or to the scan device.
	fn states_to(&mut self, on_host: bool) -> Result<(), String> {
		self.gtx_home()?;
		if on_host == self.states_on_host {
			return Ok(());
		}
		let Some(sc) = self.scan.as_mut() else {
			return Ok(());
		};
		let nd = self.conv.len();
		if on_host {
			if self.hstate.is_empty() {
				self.hstate = vec![vec![0f32; H * D * D]; nd];
			}
			let hs = &mut self.hstate;
			sc.get_states(nd, &mut |di, st| hs[di].copy_from_slice(st))?;
		} else {
			let hs = &self.hstate;
			sc.set_states(nd, H * D * D, &mut |di, dst| {
				dst.copy_from_slice(&hs[di]);
				true
			})?;
		}
		self.states_on_host = on_host;
		Ok(())
	}

	/// The delta states and conv windows back from the dense GPU into
	/// `hstate` / `conv`, if they are there.
	fn gtx_home(&mut self) -> Result<(), String> {
		if !self.on_gtx {
			return Ok(());
		}
		let gpu = self.gpu.as_mut().ok_or("flash: no GPU")?;
		if self.hstate.is_empty() {
			self.hstate = vec![vec![0f32; H * D * D]; self.conv.len()];
		}
		for (di, (st, cv)) in self.hstate.iter_mut().zip(self.conv.iter_mut()).enumerate() {
			gpu.delta_get(di, st, cv)?;
		}
		self.on_gtx = false;
		// in `hstate` now; the next scan moves them back
		self.states_on_host = self.scan.is_some();
		Ok(())
	}

	/// The delta states and conv windows to the dense GPU, when it runs the
	/// mixers.
	fn gtx_away(&mut self) -> Result<(), String> {
		if self.on_gtx || !self.gpu.as_ref().is_some_and(|g| g.mix_gpu()) {
			return Ok(());
		}
		if self.scan.is_some() {
			self.states_to(true)?;
		}
		if self.hstate.is_empty() {
			self.hstate = vec![vec![0f32; H * D * D]; self.conv.len()];
		}
		let gpu = self.gpu.as_mut().ok_or("flash: no GPU")?;
		for (di, (st, cv)) in self.hstate.iter().zip(&self.conv).enumerate() {
			gpu.delta_put(di, st, cv)?;
		}
		self.on_gtx = true;
		Ok(())
	}

	/// The delta recurrence of layer `di` for a GPU step on the host, in
	/// f32 from the step's f32 inputs `ins[t]` = (qn, kn, v, gb) of each
	/// position in order: two passes over each head's state instead of
	/// [`Self::recurrence`]'s four (the decay folded into the others).
	/// Returns each position's `o` `[H D]` at the one-position scale
	/// `OSCALE`, which `GpuDec::delta_gate_n`'s eps assumes. It must not go
	/// through `recurrence`, which uses the active set's scale (the prefill
	/// set's `OSCALE_P`, 64): the gate's eps is then 16x too small and decode
	/// KL vs truth goes 0.109 -> 2.5 on `ids.txt`.
	/// `backup`: copy the state to `bak[di]` first ([`Self::commit`]).
	#[allow(clippy::type_complexity)]
	fn step_recurrence(
		&mut self,
		di: usize,
		ins: &[(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)],
		backup: bool,
	) -> Result<Vec<Vec<f32>>, String> {
		if self.scan.is_some() {
			self.states_to(true)?;
		}
		if backup && self.bak.is_empty() {
			self.bak = vec![vec![0f32; H * D * D]; self.conv.len()];
		}
		// the step's own scale (`delta_gate_n` takes OSCALE), whatever set
		// is active
		let qs = OSCALE / (D as f32).sqrt();
		let n = ins.len();
		let mut o = vec![0f32; n * H * D];
		let st = &mut self.hstate[di];
		let bk = if backup {
			self.bak[di].as_mut_ptr()
		} else {
			std::ptr::null_mut()
		};
		#[derive(Clone, Copy)]
		struct P(*mut f32, *mut f32, *mut f32);
		unsafe impl Send for P {}
		unsafe impl Sync for P {}
		let p = P(st.as_mut_ptr(), o.as_mut_ptr(), bk);
		par_for(H, |h| {
			let p = p;
			// SAFETY: head h owns its state block, its backup block and
			// its output rows
			let s = unsafe { std::slice::from_raw_parts_mut(p.0.add(h * D * D), D * D) };
			if !p.2.is_null() {
				unsafe { std::slice::from_raw_parts_mut(p.2.add(h * D * D), D * D) }
					.copy_from_slice(s);
			}
			let kh = h / (H / HK);
			for (t, (qn, kn, v, gb)) in ins.iter().enumerate() {
				let oh = unsafe { std::slice::from_raw_parts_mut(p.1.add(t * H * D + h * D), D) };
				let q = &qn[kh * D..][..D];
				let k = &kn[kh * D..][..D];
				let vv = &v[h * D..][..D];
				let (eg, beta) = (gb[h].exp(), gb[H + h]);
				// kv = k (eg S); d = beta (v - kv)
				let mut kv = [0f32; D];
				for (ki, row) in k.iter().zip(s.chunks_exact(D)) {
					for (a, b) in kv.iter_mut().zip(row) {
						*a += ki * b;
					}
				}
				let mut d = [0f32; D];
				for ((a, &b), &c) in d.iter_mut().zip(vv).zip(&kv) {
					*a = (b - eg * c) * beta;
				}
				// S = eg S + k d^T, o = q S
				let mut ot = [0f32; D];
				for ((ki, qi), row) in k.iter().zip(q).zip(s.chunks_exact_mut(D)) {
					for ((a, &b), r) in row.iter_mut().zip(&d).zip(ot.iter_mut()) {
						*a = eg * *a + ki * b;
						*r += qi * *a;
					}
				}
				for (a, &b) in oh.iter_mut().zip(&ot) {
					*a = b * qs;
				}
			}
		});
		Ok(o.chunks_exact(H * D).map(|c| c.to_vec()).collect())
	}

	/// The delta recurrence of layer `di` over the chunk's `n` positions:
	/// on the scan device, but a one-position step on the host (cheaper
	/// than the round trip).
	fn recurrence(&mut self, di: usize, n: usize) -> Result<(), String> {
		self.recurrence_on(di, n, false)
	}

	/// [`Self::recurrence`], on the scan device only submitted (`begin`):
	/// then `Ok(true)`, and its `o` is there after [`Self::recurrence_end`].
	fn recurrence_begin(&mut self, di: usize, n: usize) -> Result<bool, String> {
		self.recurrence_on(di, n, true)?;
		Ok(self.scan.is_some() && !self.states_on_host)
	}

	/// [`Self::recurrence_begin`] followed by the gated norm in place of
	/// `o` (`post_delta_t`'s input; [`Scan::scan_begin_gn`]): on the scan
	/// device with it, else on the host after the host's recurrence.
	fn recurrence_begin_gn(&mut self, di: usize, n: usize) -> Result<bool, String> {
		self.gtx_home()?;
		let osc = oscale(self.sets[self.cur].s);
		if self.scan.is_some() {
			self.states_to(n == 1)?;
		}
		if let (true, false) = (self.scan.is_some(), self.states_on_host) {
			self.conv_prep_di(di, n)?;
			let sc = self.scan.as_mut().unwrap();
			let b = &mut self.sets[self.cur].bufs;
			let (qn, kn, v, gb, z) = (
				b[B::qn as usize].as_slice() as *const [u8],
				b[B::kn as usize].as_slice() as *const [u8],
				b[B::v as usize].as_slice() as *const [u8],
				b[B::gb as usize].as_slice() as *const [u8],
				b[B::z as usize].as_slice() as *const [u8],
			);
			let o = b[B::o as usize].as_mut_slice();
			// SAFETY: distinct buffers of the engine, alive while it is.
			unsafe { sc.scan_begin_gn(di, n, &*qn, &*kn, &*v, &*gb, &*z, o, osc)? };
			return Ok(true);
		}
		self.recurrence_on(di, n, false)?;
		let z = self.b16(B::z)[..n * H * D].to_vec();
		let w = std::mem::take(&mut self.gnw[di]);
		gnorm_rot(h16m(self.buf(B::o)), &z, &w, n, osc);
		self.gnw[di] = w;
		Ok(false)
	}

	/// With [`Set::conv_c`]: the active bank's `qn` / `kn` / `v` of `n`
	/// positions from its `cpast` and raw `qkv` (`delta_pre_c`), queued
	/// ahead of the next scan on the scan device ([`Scan::conv_next`]) when
	/// the states are there, else now on the host ([`conv_host`]).
	fn conv_prep_di(&mut self, di: usize, n: usize) -> Result<(), String> {
		if !self.sets[self.cur].conv_c {
			return Ok(());
		}
		let b = &mut self.sets[self.cur].bufs;
		let [cp, qkv, qn, kn, v] = [B::cpast, B::qkv, B::qn, B::kn, B::v].map(|bk| bk as usize);
		let (cp, qkv) = (
			b[cp].as_slice() as *const [u8],
			b[qkv].as_slice() as *const [u8],
		);
		let (qn, kn, v) = (
			b[qn].as_mut_slice() as *mut [u8],
			b[kn].as_mut_slice() as *mut [u8],
			b[v].as_mut_slice() as *mut [u8],
		);
		// SAFETY: distinct buffers of the engine, alive while it is.
		unsafe {
			match (self.scan.as_mut(), self.states_on_host) {
				(Some(sc), false) => sc.conv_next(di, n, &*cp, &*qkv, &mut *qn, &mut *kn, &mut *v),
				_ => {
					conv_host(
						&self.cvw[di],
						u16s(&*cp),
						u16s(&*qkv),
						n,
						u16sm(&mut *qn),
						u16sm(&mut *kn),
						u16sm(&mut *v),
					);
					Ok(())
				}
			}
		}
	}

	/// Wait for [`Self::recurrence_begin`]'s scan.
	fn recurrence_end(&mut self) -> Result<(), String> {
		match self.scan.as_mut() {
			Some(sc) => sc.scan_end(),
			None => Ok(()),
		}
	}

	fn recurrence_on(&mut self, di: usize, n: usize, begin: bool) -> Result<(), String> {
		self.gtx_home()?;
		let osc = oscale(self.sets[self.cur].s);
		if self.scan.is_some() {
			self.states_to(n == 1)?;
		}
		self.conv_prep_di(di, n)?;
		if let (Some(sc), false) = (self.scan.as_mut(), self.states_on_host) {
			let b = &mut self.sets[self.cur].bufs;
			let (qn, kn, v, gb) = (
				b[B::qn as usize].as_slice() as *const [u8],
				b[B::kn as usize].as_slice() as *const [u8],
				b[B::v as usize].as_slice() as *const [u8],
				b[B::gb as usize].as_slice() as *const [u8],
			);
			let o = b[B::o as usize].as_mut_slice();
			// SAFETY: distinct buffers of the engine, alive while it is.
			return unsafe {
				if begin {
					sc.scan_begin(di, n, &*qn, &*kn, &*v, &*gb, o, osc)
				} else {
					sc.scan(di, n, &*qn, &*kn, &*v, &*gb, o, osc)
				}
			};
		}
		let rows = |b: B, w: usize| -> Vec<f32> {
			self.b16(b)[..n * w].iter().map(|&h| unf16(h)).collect()
		};
		let (qn, kn, v, gb) = (
			rows(B::qn, HK * D),
			rows(B::kn, HK * D),
			rows(B::v, H * D),
			rows(B::gb, 2 * H),
		);
		let mut o = vec![0u16; n * H * D];
		let st = &mut self.hstate[di];
		let qs = 1.0 / (D as f32).sqrt();
		#[derive(Clone, Copy)]
		struct P(*mut f32, *mut u16);
		unsafe impl Send for P {}
		unsafe impl Sync for P {}
		let p = P(st.as_mut_ptr(), o.as_mut_ptr());
		par_for(H, |h| {
			let p = p;
			// SAFETY: head h owns its state block and its output columns
			let s = unsafe { std::slice::from_raw_parts_mut(p.0.add(h * D * D), D * D) };
			let kh = h / (H / HK);
			for t in 0..n {
				let q = &qn[t * HK * D + kh * D..][..D];
				let k = &kn[t * HK * D + kh * D..][..D];
				let vv = &v[t * H * D + h * D..][..D];
				let (g, beta) = (gb[t * 2 * H + h], gb[t * 2 * H + H + h]);
				let eg = g.exp();
				s.iter_mut().for_each(|x| *x *= eg);
				let mut kv = [0f32; D];
				for (ki, row) in k.iter().zip(s.chunks_exact(D)) {
					for (a, b) in kv.iter_mut().zip(row) {
						*a += ki * b;
					}
				}
				for (a, b) in kv.iter_mut().zip(vv) {
					*a = (b - *a) * beta;
				}
				for (ki, row) in k.iter().zip(s.chunks_exact_mut(D)) {
					for (a, b) in row.iter_mut().zip(&kv) {
						*a += ki * b;
					}
				}
				let mut ot = [0f32; D];
				for (qi, row) in q.iter().zip(s.chunks_exact(D)) {
					for (a, b) in ot.iter_mut().zip(row) {
						*a += qi * b;
					}
				}
				for (j, a) in ot.iter().enumerate() {
					// SAFETY: o[t][h][..] belongs to head h
					unsafe { *p.1.add(t * H * D + h * D + j) = f16(a * qs * osc) };
				}
			}
		});
		h16m(self.buf(B::o))[..o.len()].copy_from_slice(&o);
		Ok(())
	}

	/// Attention of layer `l` (host): append the chunk's keys, select each
	/// query's keys (the indexer past 512 complete blocks), attend.
	fn attention(&mut self, l: usize, n: usize) -> Result<(), String> {
		self.attention_begin(l, n)?;
		self.attention_end()
	}

	/// The attention's device work in flight, if [`Self::attention_begin`]
	/// left any: its output into the active bank's `o`.
	fn attention_end(&mut self) -> Result<(), String> {
		let Some((t_att, n)) = self.pattn_flight.take() else {
			return Ok(());
		};
		// straight into the active bank's `o`
		let o = &mut h16m(&mut self.sets[self.cur].bufs[B::o as usize])[..n * HQ * DA];
		match self.far.as_mut().filter(|f| f.busy()) {
			Some(f) => {
				let mut lse = vec![0f32; n * HQ];
				// QWEN_FLASH_ATT_LOG: when each device's share was done
				let log = std::env::var_os("QWEN_FLASH_ATT_LOG").is_some();
				let tn = if log {
					f.wait_jobs()?;
					t_att.elapsed().as_secs_f64()
				} else {
					0.0
				};
				let tw = std::time::Instant::now();
				self.pattn.as_mut().unwrap().attend_end(o, Some(&mut lse))?;
				let tg = tw.elapsed().as_secs_f64();
				f.end(o, &lse)?;
				if log {
					eprintln!(
						"[att] waits: npu done {:.1} ms after begin, igpu {:.1} ms, then merge {:.1} ms, {:.1} ms after begin",
						tn * 1e3,
						tg * 1e3,
						(tw.elapsed().as_secs_f64() - tg) * 1e3,
						t_att.elapsed().as_secs_f64() * 1e3
					);
				}
				if let Some((l, p0, n, q, sel)) = self.far_check.take() {
					let kv = &self.kv[l];
					let mut r = vec![0u16; n * HQ * DA];
					let (ks, vs) = (kv.k.slices(), kv.v.slices());
					self.pattn.as_mut().unwrap().attend(
						l / 4,
						&ks,
						&vs,
						p0,
						n,
						&q,
						&sel,
						&mut r,
					)?;
					let (mut md, mut ma, mut sa) = (0f32, 0f32, 0f64);
					for (a, b) in o.iter().zip(&r) {
						let (a, b) = (unf16(*a), unf16(*b));
						md = md.max((a - b).abs());
						ma = ma.max(b.abs());
						sa += (a - b).abs() as f64;
					}
					eprintln!(
						"[farcheck] l {l} p0 {p0}: max |diff| {md:.5} (max |o| {ma:.3}), mean |diff| {:.6}",
						sa / o.len() as f64
					);
				}
			}
			None => self.pattn.as_mut().unwrap().attend_end(o, None)?,
		}
		self.t_att += t_att.elapsed().as_secs_f64();
		Ok(())
	}

	/// [`Self::attention`] up to the device's work, which a prefill
	/// sub-chunk leaves in flight (the host copies other sub-chunks'
	/// buffers meanwhile); [`Self::attention_end`] collects it.
	/// Load the NPU's far attention and indexer graphs (once, on the
	/// prefill set).
	fn far_load(&mut self) -> Result<(), String> {
		if self.cur == 0 && !self.far_tried {
			self.far_tried = true;
			if std::env::var("QWEN_FLASH_FAR_NPU").map_or(true, |v| v != "0") {
				self.far = super::far::NpuFar::load(self.npu, &self.graphs, NL / 4)?;
			}
			self.idx_gpu = std::env::var("QWEN_FLASH_IDX_GPU").map_or(true, |v| v != "0")
				&& self.pattn.as_ref().is_some_and(|a| a.idx_len(0).is_some());
			if super::idx::min_blocks() != usize::MAX && !self.idx_gpu {
				self.idx = super::idx::NpuIdx::load(self.npu, &self.graphs, NL / 4)?;
			}
		}
		Ok(())
	}

	/// Queue attention layer `l`'s indexer scores of the active bank's
	/// sub-chunk of `n` positions at `p0` on the NPU, after its `attn_pre`
	/// was queued (it reads `iq` in place): the blocks before `p0` when
	/// they are many ([`super::idx::min_blocks`]).
	fn idx_begin(&mut self, l: usize, p0: usize, n: usize) -> Result<(), String> {
		if n <= 1 || self.pattn.is_none() || (p0 + n) / 4 <= 512 {
			return Ok(());
		}
		self.far_load()?;
		let cur = self.cur;
		let Some(x) = self.idx.as_mut() else {
			return Ok(());
		};
		let a = l / 4;
		if x.blocks(a) != p0 / 4 || x.blocks(a) < super::idx::min_blocks() {
			return Ok(());
		}
		x.begin(a, &self.sets[cur].bufs[B::iq as usize])
	}

	/// [`Self::idx_begin`] on the prefill attention's device
	/// ([`PrefillAttn::score_begin`]), once the sub-chunk's `attn_pre` is
	/// done (its `iq` is copied).
	fn idx_gpu_begin(&mut self, l: usize, p0: usize, n: usize) -> Result<(), String> {
		if !self.idx_gpu || n <= 1 || (p0 + n) / 4 <= 512 {
			return Ok(());
		}
		let (a, nb) = (l / 4, p0 / 4);
		let Some(pa) = self.pattn.as_mut() else {
			return Ok(());
		};
		if pa.idx_len(a).is_none_or(|h| h < nb) || nb < super::idx::min_blocks() {
			return Ok(());
		}
		let iq = &h16(&self.sets[self.cur].bufs[B::iq as usize])[..n * IH * ID];
		pa.score_begin(a, iq, n, nb)
	}

	fn attention_begin(&mut self, l: usize, n: usize) -> Result<(), String> {
		let p0 = self.pos;
		let t_att = std::time::Instant::now();
		let on_dev = n > 1 && self.pattn.is_some();
		if on_dev {
			self.far_load()?;
		}
		// the active bank's outputs of `attn_pre`, read in place (the
		// device work copies `q` at its begin)
		let bufs = &self.sets[self.cur].bufs;
		let bh = |b: B, len: usize| &h16(&bufs[b as usize])[..len];
		let qh = bh(B::q, n * HQ * DA);
		let (kb, vb, ikb, iq, q) = (
			bh(B::k, n * HKV * DA),
			bh(B::va, n * HKV * DA),
			bh(B::ik, n * ID),
			unf16v(bh(B::iq, n * IH * ID)),
			if on_dev { Vec::new() } else { unf16v(qh) },
		);
		let ikn = self.host[l].ikn.clone();
		let imgs = self.imgs.clone();
		let kv = &mut self.kv[l];
		if kv.k.rows() != p0 {
			return Err(format!(
				"flash attention: cache of {} rows at {p0}",
				kv.k.rows()
			));
		}
		kv.k.push(kb)?;
		kv.v.push(vb)?;
		kv.ik.push(ikb)?;
		// complete blocks: pooled, normed, roped at their first position
		let total = p0 + n;
		while (kv.bk.rows() + 1) * 4 <= total {
			let b = kv.bk.rows();
			let mut m = [0f32; ID];
			let ikv = kv.ik.view();
			for j in 0..4 {
				for (a, &v) in m.iter_mut().zip(ikv.row(b * 4 + j)) {
					*a += unf16(v) / 4.0;
				}
			}
			let ms = m.iter().map(|v| v * v).sum::<f32>() / ID as f32;
			let r = 1.0 / (ms + 1e-6).sqrt();
			for (a, w) in m.iter_mut().zip(&ikn) {
				*a *= r * w;
			}
			for (i, &(s, c)) in rope_sc(&imgs, b * 4).iter().enumerate() {
				let (a, bb) = (m[i], m[i + RD / 2]);
				m[i] = a * c as f32 - bb * s as f32;
				m[i + RD / 2] = bb * c as f32 + a * s as f32;
			}
			kv.bk.push(&m)?;
		}
		if let Some(x) = self.idx.as_mut() {
			let a = l / 4;
			if x.blocks(a) > kv.bk.rows() {
				x.truncate(kv.bk.rows());
			}
			let (from, to) = (x.blocks(a), kv.bk.rows());
			if to > from {
				let v = kv.bk.view();
				let mut rows = Vec::with_capacity((to - from) * ID);
				for b in from..to {
					rows.extend_from_slice(v.row(b));
				}
				x.append(a, &rows)?;
			}
		}
		// (a decode step runs with the prefill attention taken out)
		if let Some(pa) = self.pattn.as_mut().filter(|_| self.idx_gpu) {
			let a = l / 4;
			let to = kv.bk.rows();
			let from = pa.idx_len(a).unwrap_or(0).min(to);
			if to > from {
				let v = kv.bk.view();
				let mut rows = Vec::with_capacity((to - from) * ID);
				for b in from..to {
					rows.extend_from_slice(v.row(b));
				}
				pa.idx_append(a, from, &rows)?;
			}
		}
		let (kf, vf, bk) = (kv.k.view(), kv.v.view(), kv.bk.view());
		// the keys each query attends to: every complete block while there
		// are at most 512, else the indexer's top 512; then the tail
		let mut keys: Vec<Vec<u32>> = vec![Vec::new(); n];
		#[derive(Clone, Copy)]
		struct P<T>(*mut T);
		unsafe impl<T> Send for P<T> {}
		unsafe impl<T> Sync for P<T> {}
		// the blocks before the sub-chunk scored on the NPU (idx_begin), read
		// per query from its output buffers; the rest scored on the host
		let nbm = (p0 + n) / 4;
		let t_idx = std::time::Instant::now();
		let scored = match (self.idx.as_mut(), self.pattn.as_mut()) {
			(Some(x), _) if nbm > 512 => x.end(l / 4, n, nbm)?,
			(None, Some(pa)) if self.idx_gpu => match pa.score_end()? {
				Some((a, sn, nb, stride, s)) if a == l / 4 && sn == n && nbm > 512 => {
					super::idx::Scored::rows(s, nb.min(nbm), stride)
				}
				_ => super::idx::Scored::none(),
			},
			_ => super::idx::Scored::none(),
		};
		if std::env::var_os("QWEN_FLASH_ATT_LOG").is_some() {
			eprintln!(
				"[idx] l {l} p0 {p0}: {} blocks from the device, wait {:.1} ms, at {:.1} ms",
				scored.nb,
				t_idx.elapsed().as_secs_f64() * 1e3,
				t_att.elapsed().as_secs_f64() * 1e3
			);
		}
		thread_local! {
			// per worker: the order keys, the select's candidates, the blocks
			static SELBUF: std::cell::RefCell<(Vec<u32>, Vec<u64>, Vec<u32>)> =
				const { std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new())) };
		}
		let kp = P(keys.as_mut_ptr());
		let sc = &scored;
		par_for(n, |t| {
			let kp = &kp;
			let p = p0 + t;
			let nb = (p + 1) / 4;
			let mut ks: Vec<u32> = if nb <= 512 {
				(0..(nb * 4) as u32).collect()
			} else {
				SELBUF.with(|c| {
					let (k, cand, bs) = &mut *c.borrow_mut();
					k.resize(nb, 0);
					let done = sc.nb.min(nb);
					sc.keys(t, &mut k[..done]);
					let qr = &iq[t * IH * ID..][..IH * ID];
					for (b, x) in k.iter_mut().enumerate().skip(done) {
						let kr = bk.row(b);
						let v: f32 = (0..IH)
							.map(|h| dot8(&qr[h * ID..][..ID], kr).max(0.0))
							.sum();
						*x = super::idx::okey(v);
					}
					super::idx::top_m(k, 512, cand, bs);
					let mut ks = Vec::with_capacity(2048 + 4);
					ks.extend(bs.iter().flat_map(|&b| b * 4..b * 4 + 4));
					ks
				})
			};
			ks.extend((nb * 4) as u32..=p as u32);
			// SAFETY: task t owns keys[t]
			unsafe { *kp.0.add(t) = ks };
		});
		if on_dev {
			const SEL: usize = 513;
			let mut sel = vec![0u32; n * SEL];
			let plan = self.far.as_ref().map(|f| f.plan(p0, n, &keys));
			match plan.as_ref().filter(|p| p.jobs() > 0) {
				Some(p) => super::far::NpuFar::igpu_sel(p, p0, n, &keys, &mut sel),
				None => {
					// the selection as blocks (the dense ones: every key)
					for (t, ks) in keys.iter().enumerate() {
						if (p0 + t + 1) / 4 > 512 {
							sel[t * SEL] = 512;
							for b in 0..512 {
								sel[t * SEL + 1 + b] = ks[4 * b] / 4;
							}
						} else {
							sel[t * SEL] = u32::MAX;
						}
					}
				}
			}
			let kmin = plan.as_ref().map_or(0, |p| p.kmin);
			if std::env::var_os("QWEN_FLASH_FAR_CHECK").is_some()
				&& plan.as_ref().is_some_and(|p| p.jobs() > 0)
			{
				let mut full = vec![0u32; n * SEL];
				for (t, ks) in keys.iter().enumerate() {
					if (p0 + t + 1) / 4 > 512 {
						full[t * SEL] = 512;
						for b in 0..512 {
							full[t * SEL + 1 + b] = ks[4 * b] / 4;
						}
					} else {
						full[t * SEL] = u32::MAX;
					}
				}
				self.far_check = Some((l, p0, n, qh.to_vec(), full));
			}
			let th = t_att.elapsed().as_secs_f64();
			let (ks, vs) = (self.kv[l].k.slices(), self.kv[l].v.slices());
			let a = self.pattn.as_mut().unwrap();
			a.attend_begin(l / 4, &ks, &vs, p0, n, qh, &sel, kmin)?;
			if let (Some(f), Some(p)) = (self.far.as_mut(), plan.as_ref()) {
				f.begin(l / 4, p, n, qh, &keys, &self.kv[l])?;
				if std::env::var_os("QWEN_FLASH_ATT_LOG").is_some() {
					eprintln!(
						"[att] l {l} p0 {p0}: npu blocks {:?} gathered {} kmin {} est npu {:.1} igpu {:.1} ms",
						p.blocks,
						p.gather.len(),
						p.kmin,
						p.est[0],
						p.est[1]
					);
				}
			}
			if std::env::var_os("QWEN_FLASH_ATT_LOG").is_some() {
				eprintln!(
					"[att] l {l} p0 {p0}: host {:.1} ms, +upload/submit {:.1} ms",
					th * 1e3,
					(t_att.elapsed().as_secs_f64() - th) * 1e3
				);
			}
			self.pattn_flight = Some((t_att, n));
			return Ok(());
		}
		let mut o = vec![0u16; n * HQ * DA];
		let scale = 1.0 / (DA as f32).sqrt();
		// split-K per query over blocks of KC keys: a task per (query, KV
		// head, block) scoring all that head's query heads, so each key /
		// value row is read once per query; merged by log sum exp. A query's
		// result does not depend on the other queries of the step, so a
		// position attends the same in a step of any size (two waves, a
		// step started before its drafts)
		const KC: usize = 128;
		const G: usize = HQ / HKV;
		// per task: max [G], sum [G], acc [G][DA]
		let stride = 2 * G + G * DA;
		// each query's first task (and the end)
		let mut t0s = Vec::with_capacity(n + 1);
		let mut tot = 0;
		for ks in &keys {
			t0s.push(tot);
			tot += HKV * ks.len().div_ceil(KC);
		}
		t0s.push(tot);
		// (each task zeroes its own slice: on its thread, in its cache)
		let mut part = std::mem::take(&mut self.att_part);
		if part.len() < tot * stride {
			part.resize(tot * stride, 0.0);
		}
		let pp = P(part.as_mut_ptr());
		par_for(tot, |i| {
			let pp = &pp;
			let t = t0s.partition_point(|&s| s <= i) - 1;
			let ks = &keys[t];
			let nc = ks.len().div_ceil(KC);
			let (kh, c) = ((i - t0s[t]) / nc, (i - t0s[t]) % nc);
			let qt = &q[t * HQ * DA..][..HQ * DA];
			let blk = &ks[c * KC..ks.len().min((c + 1) * KC)];
			let mut sc = vec![0f32; G * blk.len()];
			for (j, &u) in blk.iter().enumerate() {
				let kr = &kf.row(u as usize)[kh * DA..][..DA];
				for g in 0..G {
					sc[g * blk.len() + j] = dot_h(&qt[(kh * G + g) * DA..][..DA], kr) * scale;
				}
			}
			// SAFETY: task i owns part[i stride ..][.. stride]
			let out = unsafe { std::slice::from_raw_parts_mut(pp.0.add(i * stride), stride) };
			out.fill(0.0);
			for g in 0..G {
				let r = &mut sc[g * blk.len()..][..blk.len()];
				let m = r.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
				let mut z = 0f32;
				for x in r.iter_mut() {
					*x = (*x - m).exp();
					z += *x;
				}
				out[g] = m;
				out[G + g] = z;
			}
			for (j, &u) in blk.iter().enumerate() {
				let vr = &vf.row(u as usize)[kh * DA..][..DA];
				for g in 0..G {
					let acc: &mut [f32; DA] =
						(&mut out[2 * G + g * DA..][..DA]).try_into().unwrap();
					axpy_h(acc, sc[g * blk.len() + j], vr);
				}
			}
		});
		let op = P(o.as_mut_ptr());
		par_for(n * HQ, |th| {
			let op = &op;
			let (t, h) = (th / HQ, th % HQ);
			let nc = keys[t].len().div_ceil(KC);
			let (kh, g) = (h / G, h % G);
			let parts = |c: usize| &part[(t0s[t] + kh * nc + c) * stride..][..stride];
			let m = (0..nc)
				.map(|c| parts(c)[g])
				.fold(f32::NEG_INFINITY, f32::max);
			let mut acc = [0f32; DA];
			let mut z = 0f32;
			for c in 0..nc {
				let pc = parts(c);
				let w = (pc[g] - m).exp();
				z += w * pc[G + g];
				axpy(&mut acc, w, &pc[2 * G + g * DA..][..DA]);
			}
			// SAFETY: task th owns o[th DA ..][.. DA]
			let ot = unsafe { std::slice::from_raw_parts_mut(op.0.add(th * DA), DA) };
			for a in acc.iter_mut() {
				*a /= z;
			}
			f16_into(ot, &acc);
		});
		self.att_part = part;
		h16m(self.buf(B::o))[..o.len()].copy_from_slice(&o);
		self.t_att += t_att.elapsed().as_secs_f64();
		Ok(())
	}

	/// Routing, the expert jobs and the combine of layer `l`: the streams
	/// for the next layer in `x`.
	/// Layer `l`'s routed experts over every sub-chunk of `st`, combined
	/// into the sub-chunks' streams ([`combine`]).
	fn moe(&mut self, l: usize, st: &mut [Sub]) -> Result<(), String> {
		self.comb_to(st, usize::MAX);
		let (sn, gn, cn) = {
			let st = &self.sets[self.cur];
			(st.s, st.g, st.c)
		};
		let npu = self.npu;
		let n = st.len() * sn;
		// routing per token in parallel (serial it was ~5 ms per layer at
		// 2048 tokens), then the tokens by their row in the expert pass's
		// `moe_q`
		let mut rs: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
		{
			#[derive(Clone, Copy)]
			struct P(*mut Vec<(usize, f32)>);
			unsafe impl Send for P {}
			unsafe impl Sync for P {}
			let rp = P(rs.as_mut_ptr());
			let co = &self.co;
			let lgs: Vec<(usize, &[u16])> = st
				.iter()
				.map(|sb| (sb.n, u16s(co.at(CO_LOGITS, sb.j))))
				.collect();
			par_for(n, |t| {
				let rp = &rp;
				let (sbn, lg) = lgs[t / sn];
				if t % sn >= sbn {
					return;
				}
				let lg = &lg[t % sn * NE..][..NE];
				let row: Vec<f32> = lg.iter().map(|&h| unf16(h)).collect();
				// SAFETY: task t owns rs[t]
				unsafe { *rp.0.add(t) = route(&row) };
			});
		}
		let mut per: Vec<Vec<(usize, f32)>> = vec![Vec::new(); NE];
		sel_dump(l, true, rs.iter().map(|r| r.as_slice()));
		for (t, r) in rs.iter().enumerate() {
			for &(i, w) in r {
				per[i].push((t, w));
			}
		}
		if n == 1 && sn == 1 && self.cpu_exp {
			let sel: Vec<(usize, f32)> = per
				.iter()
				.enumerate()
				.filter_map(|(e, t)| t.first().map(|&(_, w)| (e, w)))
				.collect();
			let (pl, pre) = (&self.pools[l], &self.epre[l][1]);
			let pool = super::cpuexp::Pool {
				gu_w: pl[0].as_slice(),
				gu_s: h16(&pl[1]),
				d_w: pl[2].as_slice(),
				d_s: h16(&pl[3]),
				d_pre: h16(pre),
			};
			let y = super::cpuexp::run(&pool, u16s(self.co.at(CO_MOEQ, 0)), &sel);
			combine(&mut st[0], &self.co, sn, &y);
			return Ok(());
		}
		// slot kinds by width: (experts per job, slots per expert, graph:
		// `None` the set's `experts`, else `gbig[i]`, outputs lack `d.s`)
		let mut kd: Vec<(usize, usize, Option<usize>, bool)> =
			std::iter::once((gn, cn, None, self.sets[self.cur].nsc))
				.chain(
					self.sets[self.cur]
						.big
						.iter()
						.enumerate()
						.map(|(i, b)| (b.1, b.2, Some(i), b.3)),
				)
				.collect();
		// the narrower kinds only for short passes: at 3220 tokens their
		// partly filled jobs cost more than they save (-0.5-1 %)
		let ntok: usize = st.iter().map(|sb| sb.n).sum();
		if ntok > NS_MAX_TOKENS {
			kd.retain(|k| k.1 >= cn);
		}
		kd.sort_by_key(|k| k.1);
		let kc: Vec<(usize, usize)> = kd.iter().map(|k| (k.0, k.1)).collect();
		let unsc: Vec<bool> = kd.iter().map(|k| k.3).collect();
		let nk = kc.len();
		let cmax = kc[nk - 1].1;
		// slots: an expert's tokens in slots of the widest kind while more
		// than that remain, the rest in one slot of the narrowest kind that
		// holds it
		let mut slots: Vec<Vec<(usize, Vec<(usize, f32)>)>> = vec![Vec::new(); nk];
		for (e, toks) in per.iter().enumerate() {
			let mut r = &toks[..];
			while r.len() > cmax {
				slots[nk - 1].push((e, r[..cmax].to_vec()));
				r = &r[cmax..];
			}
			if !r.is_empty() {
				let k = kc.iter().position(|&(_, c)| c >= r.len()).unwrap_or(nk - 1);
				slots[k].push((e, r.to_vec()));
			}
		}
		// a kind whose slots fit the free places of the next wider kind's
		// last job goes there (one job fewer)
		for k in 0..nk.saturating_sub(1) {
			let (g1, m) = (kc[k + 1].0, slots[k + 1].len());
			if !slots[k].is_empty() && slots[k].len() <= m.div_ceil(g1) * g1 - m {
				let v = std::mem::take(&mut slots[k]);
				slots[k + 1].extend(v);
			}
		}
		// a share of the slots on `pexp`, per kind in whole NPU jobs: a job
		// of `kc[k].0` slots costs the NPU the same full or not (its free
		// places run expert 0), so a kind gives the device the slots that
		// leave its job count a multiple fewer, of the experts the device
		// has; how many: by estimated time (default), or up to a fixed
		// `QWEN_FLASH_EXP_GPU` of its slots (full slots first)
		let mut gslots: Vec<(usize, Vec<(usize, f32)>)> = Vec::new();
		let frac = exp_gpu_frac();
		// the calibrated cost factors (the code's defaults until fitted)
		let (kn, kg) = (self.exp_cal.get(0), self.exp_cal.get(1));
		let npu_job_ms = |c: usize| kn * npu_job_ms(c);
		let exp_gpu_slot_ms = |n: usize| kg * exp_gpu_slot_ms(n);
		if self.pexp.is_some() && n > 1 && cmax <= super::EXP_SLOT && frac.is_none() {
			// by time: whole NPU jobs move to the device while that lowers the
			// larger of the two devices' estimated times, each time the job that
			// saves the NPU the most per ms of the device's
			let ne = exp_gpu_ne();
			// per kind its eligible slots (cheapest first), how many of them
			// moved, the NPU's jobs
			let el: Vec<Vec<usize>> = (0..nk)
				.map(|k| {
					let mut v: Vec<usize> = (0..slots[k].len())
						.filter(|&i| slots[k][i].0 < ne)
						.collect();
					v.sort_by_key(|&i| slots[k][i].1.len());
					v
				})
				.collect();
			let mut moved = vec![0usize; nk];
			let mut jobs: Vec<usize> = (0..nk).map(|k| slots[k].len().div_ceil(kc[k].0)).collect();
			let tn = |jobs: &[usize]| -> f64 {
				(0..nk).map(|k| jobs[k] as f64 * npu_job_ms(kc[k].1)).sum()
			};
			let mut tg = 0.0;
			loop {
				let cur = tn(&jobs).max(tg);
				// (NPU ms saved per device ms, kind, slots to move, their device time)
				let mut best: Option<(f64, usize, usize, f64)> = None;
				for k in 0..nk {
					if jobs[k] == 0 {
						continue;
					}
					let r = slots[k].len() - moved[k] - (jobs[k] - 1) * kc[k].0;
					if moved[k] + r > el[k].len() {
						continue;
					}
					let c: f64 = el[k][moved[k]..moved[k] + r]
						.iter()
						.map(|&i| exp_gpu_slot_ms(slots[k][i].1.len()))
						.sum::<f64>() + if tg == 0.0 { EXP_GPU_CALL_MS } else { 0.0 };
					jobs[k] -= 1;
					let m = tn(&jobs).max(tg + c);
					jobs[k] += 1;
					if m < cur && best.is_none_or(|b| npu_job_ms(kc[k].1) / c > b.0) {
						best = Some((npu_job_ms(kc[k].1) / c, k, r, c));
					}
				}
				let Some((_, k, r, c)) = best else { break };
				moved[k] += r;
				jobs[k] -= 1;
				tg += c;
			}
			for k in 0..nk {
				let mut take = el[k][..moved[k]].to_vec();
				take.sort_unstable();
				for &i in take.iter().rev() {
					gslots.push(slots[k].remove(i));
				}
			}
		} else if self.pexp.is_some() && n > 1 && cmax <= super::EXP_SLOT {
			let (ne, frac) = (exp_gpu_ne(), frac.unwrap_or(0.0));
			for k in 0..nk {
				let (g, cnt) = (kc[k].0, slots[k].len());
				let mut el: Vec<usize> = (0..cnt).filter(|&i| slots[k][i].0 < ne).collect();
				el.sort_by_key(|&i| std::cmp::Reverse(slots[k][i].1.len()));
				let cap = ((cnt as f64 * frac).round() as usize).min(el.len());
				let jobs = cnt.div_ceil(g);
				let need = (0..=jobs)
					.rev()
					.map(|jr| cnt - (jobs - jr) * g)
					.find(|&nd| nd <= cap)
					.unwrap_or(0);
				let mut take: Vec<usize> = el[..need].to_vec();
				take.sort_unstable();
				for &i in take.iter().rev() {
					gslots.push(slots[k].remove(i));
				}
			}
		}
		let (guw, gus, dw, ds) = (2 * FE * W / 2, 2 * FE * 2, W * FE / 2, W * 2);
		// jobs of both kinds, narrow first: (kind, slots); per kind its
		// experts per job and slots per expert
		let jobs: Vec<(usize, &[(usize, Vec<(usize, f32)>)])> = (0..nk)
			.flat_map(|k| slots[k].chunks(kc[k].0).map(move |j| (k, j)))
			.collect();
		let nj = jobs.len();
		let n0 = jobs.iter().filter(|j| kc[j.0].1 <= cn).count();
		// the calibration's estimates at factor 1 (the time balance only)
		let cal_est = (self.exp_cal.active() && frac.is_none() && !gslots.is_empty()).then(|| {
			(
				jobs.iter().map(|j| npu_job_ms(kc[j.0].1)).sum::<f64>() / kn,
				gslots
					.iter()
					.map(|s| exp_gpu_slot_ms(s.1.len()))
					.sum::<f64>() / kg,
			)
		});
		let t = std::time::Instant::now();
		let set = &mut self.sets[self.cur];
		// job buffers for the larger of the two kinds' rows
		let jb = kc
			.iter()
			.map(|&(g, c)| B::y.bytes(sn, g, c))
			.max()
			.unwrap_or(0);
		while set.jy.len() < nj {
			set.jx.push(npu.alloc(jb)?);
			set.jy.push(npu.alloc(jb)?);
		}
		// each job's token rows of `moe_q`, for jobs `ks` (filled after
		// both lists are recorded: the narrow jobs' while the wide list
		// runs)
		#[derive(Clone, Copy)]
		struct P(*mut u8);
		unsafe impl Send for P {}
		unsafe impl Sync for P {}
		let ps: Vec<P> = set.jx[..nj]
			.iter_mut()
			.map(|b| P(b.as_mut_slice().as_mut_ptr()))
			.collect();
		let fill = |mqa: &[u8], ks: std::ops::Range<usize>| {
			let r = W * 2;
			let k0 = ks.start;
			par_for(ks.len(), |k| {
				let ps = &ps;
				let k = k0 + k;
				let (kind, job) = jobs[k];
				let c = kc[kind].1;
				for (j, (_, toks)) in job.iter().enumerate() {
					for (i, &(t, _)) in toks.iter().enumerate() {
						// SAFETY: task k owns job buffer k, rows inside it
						let dst = unsafe {
							std::slice::from_raw_parts_mut(ps[k].0.add((j * c + i) * r), r)
						};
						dst.copy_from_slice(&mqa[t * r..][..r]);
					}
				}
			});
		};
		// the jobs of a kind in one list: one submission, no per-job
		// list / fence and no host gap between jobs
		let (graphs, gbig, jx, jy) = (&mut set.graphs, &mut set.gbig, &set.jx, &set.jy);
		let (pools, epre, (gus2, dsf)) = (&self.pools[l], &self.epre[l][1], &self.ews[l]);
		let bind = |g: &mut Graph<'static>, k: usize| -> Result<(), String> {
			let (kind, job) = jobs[k];
			let c = kc[kind].1;
			let mut no = 0;
			for i in 0..g.args.len() {
				let a = &g.args[i];
				let name = a.name.as_str();
				if a.is_output {
					// `y` [g, c, W], or per expert `y{e}` [c, W] (the blob
					// names its outputs after their ops: by their order)
					g.bind_at(i, &jy[k], no * c * W * 2)?;
					no += 1;
				} else if let Some(j) = name.strip_prefix('d').and_then(|n| n.strip_suffix(".pre"))
				{
					// the expert's own down input range, `ed.pre` [NE][FE]
					let j: usize = j.parse().map_err(|_| format!("experts: {name}"))?;
					let e = job.get(j).map_or(0, |s| s.0);
					g.bind_at(i, epre, e * FE * 2)?;
				} else if let Some(j) = name.strip_prefix('x') {
					let j: usize = j.parse().map_err(|_| format!("experts: {name}"))?;
					g.bind_at(i, &jx[k], j * c * W * 2)?;
				} else {
					// gu{j}.w / gu{j}.s / d{j}.w / d{j}.s
					let (head, part) = name.split_once('.').ok_or(format!("experts: {name}"))?;
					let (pool, j) = if let Some(j) = head.strip_prefix("gu") {
						if part == "s2" {
							let j: usize = j.parse().map_err(|_| format!("experts: {name}"))?;
							let e = job.get(j).map_or(0, |s| s.0);
							g.bind_at(i, gus2, e * 2 * FE * 2)?;
							continue;
						}
						(if part == "w" { 0 } else { 1 }, j)
					} else if let Some(j) = head.strip_prefix('d') {
						(if part == "w" { 2 } else { 3 }, j)
					} else {
						return Err(format!("experts: unknown input {name}"));
					};
					let j: usize = j.parse().map_err(|_| format!("experts: {name}"))?;
					let e = job.get(j).map_or(0, |s| s.0);
					let per = [guw, gus, dw, ds][pool];
					g.bind_at(i, &pools[pool], e * per)?;
				}
			}
			Ok(())
		};
		// the jobs in lists of consecutive jobs of one kind: the narrow ones
		// in one, the wide ones in `QWEN_FLASH_MOE_SPLIT` (default 3). All
		// recorded before any is submitted: an append while the NPU runs a
		// list waits in the driver (the wide list's record took 50-70 ms
		// instead of 1.1 behind the narrow one's submission)
		let split = std::env::var("QWEN_FLASH_MOE_SPLIT")
			.ok()
			.and_then(|v| v.parse::<usize>().ok())
			.unwrap_or(3)
			.max(1);
		let mut ranges: Vec<std::ops::Range<usize>> = Vec::new();
		// the kinds up to the set's own width in a list each, each wide
		// kind in parts by its share of the wide jobs
		let nw = jobs.iter().filter(|j| kc[j.0].1 > cn).count();
		let mut k0 = 0;
		for kind in 0..nk {
			let nkj = jobs.iter().filter(|j| j.0 == kind).count();
			if nkj == 0 {
				continue;
			}
			let parts = if kc[kind].1 <= cn {
				1
			} else {
				(split * nkj).div_ceil(nw.max(1)).min(nkj)
			};
			for p in 0..parts {
				ranges.push(k0 + p * nkj / parts..k0 + (p + 1) * nkj / parts);
			}
			k0 += nkj;
		}
		let mut lists = Vec::with_capacity(ranges.len());
		for r in &ranges {
			let k0 = r.start;
			let kind = jobs[k0].0;
			let g = match kd[kind].2 {
				None => &mut graphs[5],
				Some(i) => gbig.get_mut(i).ok_or("experts: wide graph not loaded")?,
			};
			lists.push(npu.batch_each(g, r.len(), &mut |g, k| bind(g, k0 + k))?);
		}
		let t_rec = t.elapsed().as_secs_f64();
		// per token its (job, row, weight, expert whose `d.s` the row
		// lacks or `NE`), by list
		let mut src: Vec<Vec<Vec<(usize, usize, f32, usize)>>> =
			vec![vec![Vec::new(); n]; ranges.len()];
		for (li, r) in ranges.iter().enumerate() {
			for k in r.clone() {
				let (kind, job) = jobs[k];
				let c = kc[kind].1;
				for (j, (e, toks)) in job.iter().enumerate() {
					let es = if unsc[kind] { *e } else { NE };
					for (i, &(t, w)) in toks.iter().enumerate() {
						src[li][t].push((k, j * c + i, w, es));
					}
				}
			}
		}
		let ys: Vec<*const u8> = jy[..nj].iter().map(|b| b.as_slice().as_ptr()).collect();
		let dsf = dsf.as_ptr();
		// the lists in job order on the queue: list i + 1's inputs are
		// copied while list i runs, list i's outputs gathered while list
		// i + 1 runs (unless deferred), the rest in the combine (per token
		// the sources in job order: the order of a serial gather,
		// bit-identical)
		let defer = if st.len() > 1 { moe_defer() } else { 0 };
		let mut y = if defer >= 2 {
			Vec::new()
		} else {
			vec![0f32; n * W]
		};
		#[derive(Clone, Copy)]
		struct PF(*mut f32);
		unsafe impl Send for PF {}
		unsafe impl Sync for PF {}
		let yp = PF(y.as_mut_ptr());
		let mut t_gather = 0f64;
		let mqa = &self.co.bufs[CO_MOEQ].as_slice()[..n * W * 2];
		for (b, r) in lists.iter_mut().zip(&ranges) {
			fill(mqa, r.clone());
			b.submit()?;
		}
		// the device's share, submitted behind the NPU's lists
		if !gslots.is_empty() {
			let toks: Vec<Vec<u32>> = gslots
				.iter()
				.map(|(_, ts)| ts.iter().map(|&(t, _)| t as u32).collect())
				.collect();
			let sl: Vec<(usize, &[u32])> = gslots
				.iter()
				.zip(&toks)
				.map(|((e, _), t)| (*e, &t[..]))
				.collect();
			self.pexp.as_mut().unwrap().begin(l, mqa, &sl)?;
		}
		let nl = lists.len();
		let t_sub = std::time::Instant::now();
		let mut t_lists = Vec::with_capacity(nl);
		for li in 0..nl {
			lists[li].wait()?;
			t_lists.push(t_sub.elapsed().as_secs_f64() * 1e3);
			// the scan device kept awake through the expert pass too (as
			// `keep_awake`): after the first layer's (~0.7 s with cold
			// expert pages) the next poke paid ~490 ms for its wake-up
			if self.poke && n > 1 && self.poked.elapsed().as_millis() >= 30 {
				if let Some(sc) = self.scan.as_mut() {
					sc.poke()?;
				}
				self.poked = std::time::Instant::now();
			}
			if li + 1 < nl && defer < 2 {
				let g = std::time::Instant::now();
				struct G<'a>(&'a [*const u8], *const f32);
				unsafe impl Sync for G<'_> {}
				let (src, gs) = (&src, G(&ys, dsf));
				par_for(n, |t| {
					let (yp, gs) = (&yp, &gs);
					// SAFETY: task t owns y[t][..]
					let yt = unsafe { std::slice::from_raw_parts_mut(yp.0.add(t * W), W) };
					moe_acc(yt, &src[li][t], gs.0, gs.1);
				});
				t_gather += g.elapsed().as_secs_f64();
			}
		}
		let t_run = t.elapsed().as_secs_f64() - t_rec - t_gather;
		// the lists not gathered yet, the device's rows and, per token,
		// its (row, weight) sources: the combine's
		let g = std::time::Instant::now();
		let first = if defer >= 2 { 0 } else { nl.saturating_sub(1) };
		let src: Vec<_> = src.into_iter().skip(first).collect();
		let mut gsrc: Vec<Vec<(usize, f32)>> =
			vec![Vec::new(); if gslots.is_empty() { 0 } else { n }];
		let gy: *const u16 = if gslots.is_empty() {
			std::ptr::null()
		} else {
			let mut r = 0;
			for (_, ts) in &gslots {
				for &(t, w) in ts {
					gsrc[t].push((r, w));
					r += 1;
				}
			}
			let te = std::time::Instant::now();
			let y = self.pexp.as_mut().unwrap().end()?;
			let tw = te.elapsed().as_secs_f64();
			self.t_moe[4] += tw;
			if std::env::var_os("QWEN_FLASH_MOE_LOG").is_some() {
				eprintln!("[flash] moe layer {l}: device wait {:.1} ms", tw * 1e3);
			}
			y.as_ptr()
		};
		let c = Comb {
			pos0: st[0].pos,
			sn,
			k: st.len(),
			done: 0,
			y,
			src,
			gsrc,
			gy,
			ys,
			dsf,
		};
		if defer == 0 {
			comb_run(&c, st, &self.co, 0, c.k);
		} else {
			self.comb = Some(c);
		}
		self.t_moe[3] += g.elapsed().as_secs_f64();
		drop(lists);
		// one calibration sample: the NPU's run (as the constants were
		// measured) and the device's own time, each against its estimate
		if let (Some((en, eg)), Some(gms)) = (cal_est, self.pexp.as_ref().and_then(|p| p.last_ms()))
		{
			self.exp_cal.add(&[(t_run * 1e3, en), (gms, eg)]);
		}
		if std::env::var_os("QWEN_FLASH_MOE_LOG").is_some() {
			eprintln!(
				"[flash] moe layer {l}: {} experts, {:?} slots (+{} on the device, {} rows), {nj} jobs ({} wide), record {:.1} ms, run {:.1} ms, lists done at {:.1?} ms",
				per.iter().filter(|t| !t.is_empty()).count(),
				slots.iter().map(|s| s.len()).collect::<Vec<_>>(),
				gslots.len(),
				gslots.iter().map(|s| s.1.len()).sum::<usize>(),
				nj - n0,
				t_rec * 1e3,
				t_run * 1e3,
				t_lists
			);
		}
		self.t_moe[0] += t_rec;
		self.t_moe[1] += t_run;
		self.t_moe[2] += t_gather;
		Ok(())
	}

	/// A step of `ids` (at most [`Self::spec_k`] positions) with the dense
	/// layers on the GPU ([`super::gpudec`]); the recurrence, the attention
	/// and the routed experts as in a one-position chunk (host, cores).
	/// `logits`: the last position's.
	fn step_gpu(&mut self, ids: &[u32], logits: bool) -> Result<Vec<f32>, String> {
		let mut rows = self.step_gpu_n(ids, logits as usize, false)?;
		Ok(rows.pop().unwrap_or_default())
	}

	/// [`Self::step_gpu`] on the positions of `ids` at once (at most
	/// [`Self::spec_k`]): every dense matrix and every routed expert hit is
	/// read once for all of them. `heads`: return the logits of the last
	/// `heads` positions; `save`: keep what [`Self::commit`] needs to drop a
	/// tail of them.
	fn step_gpu_n(
		&mut self,
		ids: &[u32],
		heads: usize,
		save: bool,
	) -> Result<Vec<Vec<f32>>, String> {
		let r = self.step_gpu_run(ids, heads, save);
		if let Some(t) = self.gpu.as_ref().map(|g| g.t_gpu) {
			self.dense_watch(ids.len(), t, r.as_ref().map(|_| ()).map_err(|e| e.as_str()));
		}
		r
	}

	fn step_gpu_run(
		&mut self,
		ids: &[u32],
		heads: usize,
		save: bool,
	) -> Result<Vec<Vec<f32>>, String> {
		let n = ids.len();
		let max = self.gpu.as_ref().map_or(0, |g| g.max_np());
		if n == 0 || n > max {
			return Err(format!("flash: a step of {n} positions"));
		}
		let r = self.step_gpu_with(heads, save, |me, gpu, sv| {
			let (xs, t) = me.step_layers(gpu, ids, sv)?;
			Ok((ids.to_vec(), xs, t))
		})?;
		Ok(r.1)
	}

	/// [`Self::step_gpu_run`]'s frame around `layers`, which runs a step's
	/// layers on the dense GPU (taken out of `self` meanwhile): the step's
	/// tokens, each position's streams after the last layer and (total,
	/// mixer, moe) seconds. Then the MTP head's pending rows, the positions,
	/// the logits of the last `heads` positions. Returns the tokens and the
	/// logits.
	#[allow(clippy::type_complexity)]
	fn step_gpu_with(
		&mut self,
		heads: usize,
		save: bool,
		layers: impl FnOnce(
			&mut Self,
			&mut super::gpudec::GpuDec,
			Option<&mut Spec>,
		) -> Result<(Vec<u32>, Vec<Vec<f32>>, [f64; 3]), String>,
	) -> Result<(Vec<u32>, Vec<Vec<f32>>), String> {
		self.gtx_away()?;
		let host_mix = !self.on_gtx;
		let mut gpu = self.gpu.take().ok_or("flash: no GPU")?;
		gpu.t_gpu = 0.0;
		gpu.t_site = [0.0; 5];
		let mut sv = save.then(|| Spec {
			pos0: self.pos,
			ids: Vec::new(),
			nctx: self.nctx,
			csts: vec![self.cstate.clone()],
			// (on the GPU with its mixers: `GpuDec::delta_rollback`)
			convs: if host_mix {
				self.conv.iter().map(|c| vec![c.clone()]).collect()
			} else {
				Vec::new()
			},
			rin: vec![Vec::new(); self.conv.len()],
		});
		// a step's attention runs on the cores (the device mirror is a
		// prefill's; a rollback would leave it stale)
		let pattn = self.pattn.take();
		let r = layers(self, &mut gpu, sv.as_mut());
		if pattn.is_some() {
			self.pattn = pattn;
		}
		let (t_gpu, t_site) = (gpu.t_gpu, gpu.t_site);
		self.gpu = Some(gpu);
		let (ids, xs, [t0, t_mix, t_exp]) = r?;
		let n = ids.len();
		if let Some(s) = sv.as_mut() {
			s.ids = ids.clone();
		}
		self.spec = sv;
		if self.mtp.is_some() {
			for (t, x) in xs.iter().enumerate() {
				self.mpend.push((self.pos + t, x.clone(), ids[t]));
			}
			self.mpend_cap();
		}
		let t = std::time::Instant::now();
		for (o, &v) in h16m(self.buf(B::x)).iter_mut().zip(xs.iter().flatten()) {
			*o = f16(v);
		}
		self.pos += n;
		let out = if heads == 0 {
			Vec::new()
		} else {
			let ms = self.mixed_n(&(n - heads.min(n)..n).collect::<Vec<_>>());
			let mr: Vec<&[f32]> = ms.iter().map(|v| v.as_slice()).collect();
			let gpu = self.gpu.as_mut().ok_or("flash: no GPU")?;
			// the GPU's first row blocks while the cores take the rest
			// (`QWEN_FLASH_HEAD_SPLIT` = blocks on the GPU)
			let k = head_split(ms.len()).min(gpu.head_blocks());
			if k > 0 && k < gpu.head_blocks() {
				let r0 = k * super::gpudec::HEAD_ROWS;
				gpu.head_begin(&mr, k)?;
				let mut outs = {
					let _w = quant::parallel::limit_width(quant::parallel::decode_threads());
					self.head_ms_from(&ms, r0)
				};
				let gpu = self.gpu.as_mut().ok_or("flash: no GPU")?;
				gpu.head_wait(&mut outs, r0)?;
				outs
			} else {
				match gpu.head_n(&mr)? {
					Some(v) if std::env::var("QWEN_FLASH_HEAD_CHECK").is_ok() => {
						let h = self.head(0);
						let d = v[0]
							.iter()
							.zip(&h)
							.fold(0f32, |a, (x, y)| a.max((x - y).abs()));
						eprintln!("[flash] head check: max |gpu - host| {d:.2e}");
						v
					}
					Some(v) => v,
					None => self.head_rows(&(n - heads.min(n)..n).collect::<Vec<_>>()),
				}
			}
		};
		if self.timing {
			let ph: Vec<f64> = super::cpuexp::T_PHASE
				.iter()
				.map(|a| a.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 * 1e-6)
				.collect();
			// the shared expert's wait is GPU time inside the moe span: count
			// it once (in moe)
			let shared = t_site[2];
			eprintln!(
				"[flash] step of {n}: {:.1} ms (gpu dense {:.1}, host dense {:.1}, mixer {:.1} of which attention {:.1}, moe {:.1} = setup {:.1} + experts {:.1} + shared wait {:.1}, head {:.1})",
				(t0 + t.elapsed().as_secs_f64()) * 1e3,
				(t_gpu - shared) * 1e3,
				(t0 - (t_gpu - shared) - t_mix - t_exp) * 1e3,
				t_mix * 1e3,
				std::mem::take(&mut self.t_att) * 1e3,
				t_exp * 1e3,
				ph[0],
				ph[1],
				shared * 1e3,
				t.elapsed().as_secs_f64() * 1e3
			);
			// (the head's GPU part ran after the layers' split was taken)
			let head_gpu = self.gpu.as_ref().map_or(0.0, |g| g.t_site[3]);
			let t_site = [t_site[0], t_site[1], t_site[2], head_gpu, t_site[4]];
			let [a, b, c, d, e] = t_site.map(|v| v * 1e3);
			eprintln!(
				"[flash]   gpu split: projections {a:.1}, post {b:.1}, shared wait {c:.1}, head {d:.1}, other {e:.1}"
			);
		}
		Ok((ids, out))
	}

	/// [`Self::step_gpu_n`]'s layers: each position's streams after the
	/// last layer, and (total, mixer, moe) seconds.
	#[allow(clippy::type_complexity)]
	fn step_layers(
		&mut self,
		gpu: &mut super::gpudec::GpuDec,
		ids: &[u32],
		mut sv: Option<&mut Spec>,
	) -> Result<(Vec<Vec<f32>>, [f64; 3]), String> {
		let n = ids.len();
		if n >= waves_min()
			&& n <= 2 * super::gpudec::WAVE_MAX
			&& gpu.waves()
			&& std::env::var_os("QWEN_FLASH_MIX_CHECK").is_none()
		{
			let r = if wave_pipe() {
				let was = gpu.set_shared_async(false);
				let r = self.step_waves_pipe(gpu, ids, sv);
				gpu.set_shared_async(was);
				r
			} else {
				self.step_waves(gpu, ids, sv)
			};
			gpu.set_wave(0, 0);
			return r;
		}
		let t0 = std::time::Instant::now();
		// the decode threads spin through the step's GPU waits instead of
		// parking: a dispatch after a park started 40-60 us late
		let spin = flash_spin();
		let _sp = (spin > 0).then(|| quant::parallel::spin_window(spin));
		let _w =
			(spin > 0).then(|| quant::parallel::limit_width(quant::parallel::decode_threads()));
		let (mut xs, rows, cs, sn) = self.step_inputs(ids)?;
		let mut rows = Some(rows);
		let put = |me: &mut Self, b: B, t: usize, v: &[f32]| {
			f16_into(&mut h16m(me.buf(b))[t * v.len()..], v);
		};
		let (mut t_mix, mut t_exp) = (0f64, 0f64);
		let mut di = 0;
		let mix_check = std::env::var("QWEN_FLASH_MIX_CHECK").is_ok();
		for l in 0..NL {
			if l == PLE_LAYER {
				let es = rows
					.take()
					.ok_or("flash: n-gram rows taken")
					.map(join_rows)??;
				let save = sv.as_mut().map(|s| &mut s.csts);
				gpu.ple_n(&mut xs, &es, &mut self.cstate, save)?;
			}
			// a delta layer's mixer and post in one submission
			let fused = if !is_attn(l) && gpu.mix_gpu() && !mix_check {
				let r = gpu.delta_post_n(&mut xs, l, di, sv.is_some())?;
				if r.is_some() {
					di += 1;
				}
				r
			} else {
				None
			};
			let mut post = if let Some(p) = fused {
				p
			} else {
				let (a, inj): (Vec<Vec<f32>>, Vec<[f32; HC]>) = if is_attn(l) {
					let pre = gpu.attn_pre_n(&xs, l, &cs, &sn)?;
					let t = std::time::Instant::now();
					for (ti, ([q, _, k, v, iq, ik], _)) in pre.iter().enumerate() {
						put(self, B::q, ti, q);
						put(self, B::k, ti, k);
						put(self, B::va, ti, v);
						put(self, B::iq, ti, iq);
						put(self, B::ik, ti, ik);
					}
					self.attention(l, n)?;
					let o = self.b16(B::o);
					let r = pre
						.iter()
						.enumerate()
						.map(|(ti, ([_, gate, ..], inj))| {
							let mut a = unf16v(&o[ti * HQ * DA..][..HQ * DA]);
							for (a, g) in a.iter_mut().zip(gate) {
								*a *= g;
							}
							(a, *inj)
						})
						.unzip();
					t_mix += t.elapsed().as_secs_f64();
					r
				} else if gpu.mix_gpu() {
					// `QWEN_FLASH_MIX_CHECK=1`: each layer's GPU mixer against the
					// host forms on the same state
					let mut before = None;
					if mix_check {
						let mut st = vec![0f32; H * D * D];
						let mut cv = vec![0u16; 3 * NQKV];
						gpu.delta_get(di, &mut st, &mut cv)?;
						before = Some((st, cv));
					}
					let r = gpu.delta_mix_n(&xs, l, di, sv.is_some())?;
					if let Some((st, mut cv)) = before {
						let pre = gpu.delta_pre_n(&xs, l)?;
						let pin: Vec<(&[f32], &[f32])> = pre
							.iter()
							.map(|(qkv, _, ab, _)| (qkv.as_slice(), ab.as_slice()))
							.collect();
						let ins = gpu.delta_conv_n(l, &pin, &mut cv, None);
						let keep = std::mem::replace(&mut self.hstate[di], st);
						let fl = (self.on_gtx, self.states_on_host);
						(self.on_gtx, self.states_on_host) = (false, true);
						let os = self.step_recurrence(di, &ins, false)?;
						(self.on_gtx, self.states_on_host) = fl;
						let st2 = std::mem::replace(&mut self.hstate[di], keep);
						let zs: Vec<&[f32]> = pre.iter().map(|p| p.1.as_slice()).collect();
						let a = gpu.delta_gate_n(l, &os, &zs, OSCALE);
						let mut gst = vec![0f32; H * D * D];
						let mut gcv = vec![0u16; 3 * NQKV];
						gpu.delta_get(di, &mut gst, &mut gcv)?;
						let md = |x: &[f32], y: &[f32]| {
							let m = x.iter().fold(0f32, |m, v| m.max(v.abs()));
							x.iter().zip(y).fold(0f32, |e, (a, b)| e.max((a - b).abs())) / m
						};
						eprintln!(
							"[mixcheck] l {l} pos {}: a {:.2e}, state {:.2e}, window same {}",
							self.pos,
							md(&a[0], &r.0[0]),
							md(&st2, &gst),
							cv == gcv
						);
					}
					di += 1;
					r
				} else {
					let pre = gpu.delta_pre_n(&xs, l)?;
					let t = std::time::Instant::now();
					let pin: Vec<(&[f32], &[f32])> = pre
						.iter()
						.map(|(qkv, _, ab, _)| (qkv.as_slice(), ab.as_slice()))
						.collect();
					let save = sv.as_mut().map(|s| &mut s.convs[di]);
					let ins = gpu.delta_conv_n(l, &pin, &mut self.conv[di], save);
					let os = self.step_recurrence(di, &ins, sv.is_some())?;
					if let Some(s) = sv.as_mut() {
						s.rin[di] = ins;
					}
					di += 1;
					t_mix += t.elapsed().as_secs_f64();
					let zs: Vec<&[f32]> = pre.iter().map(|p| p.1.as_slice()).collect();
					let a = gpu.delta_gate_n(l, &os, &zs, OSCALE);
					(a, pre.iter().map(|p| p.3).collect())
				};
				gpu.post_n(&mut xs, l, &a, &inj)?
			};
			let t = std::time::Instant::now();
			let sels: Vec<Vec<(usize, f32)>> = post.iter().map(|(_, _, lg, _)| route(lg)).collect();
			sel_dump(l, false, sels.iter().map(|s| s.as_slice()));
			let (pl, pre) = (&self.pools[l], &self.epre[l][1]);
			let pool = super::cpuexp::Pool {
				gu_w: pl[0].as_slice(),
				gu_s: h16(&pl[1]),
				d_w: pl[2].as_slice(),
				d_s: h16(&pl[3]),
				d_pre: h16(pre),
			};
			let mqs: Vec<&[f32]> = post.iter().map(|p| p.0.as_slice()).collect();
			let sr: Vec<&[(usize, f32)]> = sels.iter().map(|s| s.as_slice()).collect();
			// DRAM-bound: 16 shards read as fast as 24 (`cpuexp` bench)
			let ys = {
				let _w = quant::parallel::limit_width(quant::parallel::decode_threads());
				super::cpuexp::run_n(&pool, &mqs, &sr)
			};
			// the shared expert ran on the GPU meanwhile
			gpu.shared_n(&mut post)?;
			for ((x, (_, inj_m, _, shared)), y) in xs.iter_mut().zip(&post).zip(&ys) {
				for c in 0..HC {
					for i in 0..W {
						x[c * W + i] += inj_m[c] * (shared[i] + y[i]);
					}
				}
			}
			t_exp += t.elapsed().as_secs_f64();
		}
		Ok((xs, [t0.elapsed().as_secs_f64(), t_mix, t_exp]))
	}

	/// A GPU step's inputs of the tokens `ids`: each position's streams
	/// (the embedding in every stream), the n-gram rows being read (NVMe
	/// reads, ~1.5 ms, first needed at the PLE layer: read while layer 0
	/// runs), RoPE cosines and sines. Pushes the ids to the n-gram context.
	#[allow(clippy::type_complexity)]
	fn step_inputs(
		&mut self,
		ids: &[u32],
	) -> Result<
		(
			Vec<Vec<f32>>,
			ngram::Pending,
			Vec<[f32; RD]>,
			Vec<[f32; RD]>,
		),
		String,
	> {
		self.step_inputs_at(ids, 0)
	}

	/// [`Self::step_inputs`] of tokens that follow `base` positions of the
	/// step already given (their RoPE from `pos + base`).
	#[allow(clippy::type_complexity)]
	fn step_inputs_at(
		&mut self,
		ids: &[u32],
		base: usize,
	) -> Result<
		(
			Vec<Vec<f32>>,
			ngram::Pending,
			Vec<[f32; RD]>,
			Vec<[f32; RD]>,
		),
		String,
	> {
		let n = ids.len();
		let mut xs = Vec::with_capacity(n);
		for &id in ids {
			let e = self.emb_row(id)?;
			let mut x = vec![0f32; HCW];
			for c in 0..HC {
				x[c * W..][..W].copy_from_slice(&e);
			}
			xs.push(x);
		}
		let nids: Vec<_> = ids
			.iter()
			.map(|&id| self.hash.ids(self.nctx.push(id, EOS)))
			.collect();
		let rows = self.table.gather_async(&nids, 16)?;
		let (mut cs, mut sn) = (vec![[0f32; RD]; n], vec![[0f32; RD]; n]);
		for t in 0..n {
			for (i, &(a, b)) in self.rope_sc(self.pos + base + t).iter().enumerate() {
				for j in [i, i + RD / 2] {
					cs[t][j] = b as f32;
					sn[t][j] = a as f32;
				}
			}
		}
		Ok((xs, rows, cs, sn))
	}

	/// [`Self::step_layers`] as two waves (`QWEN_FLASH_WAVES`): positions
	/// `[0, na)` (A) and `[na, n)` (B), each run through every layer as a
	/// step of its own ([`super::gpudec::GpuDec::set_wave`]), B half a layer
	/// behind A, so the GPU runs one wave's dense part while the cores run
	/// the other's routed experts (a helper thread drives the pool): GPU
	/// `A.dense(L) B.dense(L) A.dense(L+1) ...`, cores `B.experts(L-1)
	/// A.experts(L) B.experts(L) ...`. Valid since a position at layer L
	/// needs only the earlier positions at layer L: the delta states (in
	/// VRAM, updated in queue order, A's before B's), the attention caches
	/// (appended A then B), the PLE window (host, A then B). Every GEMV
	/// build computes a position's values the same at 1 to 8 positions and
	/// the host attention a query's the same in a step of any size, so the
	/// waves compute what one step of all of them would: bit-identical
	/// output. Costs each
	/// dense matrix read twice; the attention layers' host work waits for
	/// the expert pass running beside it (one pool).
	#[allow(clippy::type_complexity)]
	fn step_waves(
		&mut self,
		gpu: &mut super::gpudec::GpuDec,
		ids: &[u32],
		mut sv: Option<&mut Spec>,
	) -> Result<(Vec<Vec<f32>>, [f64; 3]), String> {
		type Post = Vec<(Vec<f32>, [f32; HC], Vec<f32>, Vec<f32>)>;
		let t0 = std::time::Instant::now();
		let spin = flash_spin();
		let _sp = (spin > 0).then(|| quant::parallel::spin_window(spin));
		let _w =
			(spin > 0).then(|| quant::parallel::limit_width(quant::parallel::decode_threads()));
		let n = ids.len();
		let na = n.div_ceil(2);
		let (mut xa, rows, cs, sn) = self.step_inputs(ids)?;
		let mut xb = xa.split_off(na);
		let mut rows = Some(rows);
		let mut es_b = None;
		let (mut t_mix, mut t_exp) = (0f64, 0f64);
		// one wave's dense part of layer `l` (`base`: its first position):
		// the post's outputs, the attention's seconds
		let dense = |me: &mut Self,
		             gpu: &mut super::gpudec::GpuDec,
		             xs: &mut [Vec<f32>],
		             l: usize,
		             di: usize,
		             base: usize,
		             save: bool|
		 -> Result<(Post, f64), String> {
			if !is_attn(l) {
				let p = gpu.delta_post_n(xs, l, di, save)?;
				return Ok((
					p.ok_or("flash: two waves need the fused delta layers")?,
					0.0,
				));
			}
			let nw = xs.len();
			let pre = gpu.attn_pre_n(xs, l, &cs[base..base + nw], &sn[base..base + nw])?;
			let t = std::time::Instant::now();
			for (ti, ([q, _, k, va, iq, ik], _)) in pre.iter().enumerate() {
				for (b, v) in [(B::q, q), (B::k, k), (B::va, va), (B::iq, iq), (B::ik, ik)] {
					f16_into(&mut h16m(me.buf(b))[ti * v.len()..], v);
				}
			}
			let p = me.pos;
			me.pos = p + base;
			let r = me.attention(l, nw);
			me.pos = p;
			r?;
			let o = me.b16(B::o);
			let (a, inj): (Vec<Vec<f32>>, Vec<[f32; HC]>) = pre
				.iter()
				.enumerate()
				.map(|(ti, ([_, gate, ..], inj))| {
					let mut a = unf16v(&o[ti * HQ * DA..][..HQ * DA]);
					for (a, g) in a.iter_mut().zip(gate) {
						*a *= g;
					}
					(a, *inj)
				})
				.unzip();
			let dt = t.elapsed().as_secs_f64();
			Ok((gpu.post_n(xs, l, &a, &inj)?, dt))
		};
		let route_all = |l: usize, post: &Post| -> Vec<Vec<(usize, f32)>> {
			let sels: Vec<Vec<(usize, f32)>> = post.iter().map(|(_, _, lg, _)| route(lg)).collect();
			sel_dump(l, false, sels.iter().map(|s| s.as_slice()));
			sels
		};
		let combine_into = |xs: &mut [Vec<f32>], post: &Post, ys: &[Vec<f32>]| {
			for ((x, (_, inj_m, _, shared)), y) in xs.iter_mut().zip(post).zip(ys) {
				for c in 0..HC {
					for i in 0..W {
						x[c * W + i] += inj_m[c] * (shared[i] + y[i]);
					}
				}
			}
		};
		// the cores' pass runs on the expert thread while `self` drives the
		// GPU
		let ex = (self.pools.clone(), self.epre.clone());
		type Job = (usize, Vec<Vec<f32>>, Vec<Vec<(usize, f32)>>);
		let (idle_tx, idle_rx) = std::sync::mpsc::channel::<()>();
		let r = (|| -> Result<(), String> {
			let (jtx, jrx) = std::sync::mpsc::channel::<Job>();
			let (rtx, rrx) = std::sync::mpsc::channel::<Vec<Vec<f32>>>();
			self.exp_thread.run(Box::new(move || {
				let _idle = idle_tx;
				let ep = pool_views(&ex.0, &ex.1);
				for (l, mqs, sels) in jrx {
					let mr: Vec<&[f32]> = mqs.iter().map(|v| v.as_slice()).collect();
					let sr: Vec<&[(usize, f32)]> = sels.iter().map(|v| v.as_slice()).collect();
					// DRAM-bound: 16 shards read as fast as 24 (`cpuexp` bench)
					let ys = {
						let _w = quant::parallel::limit_width(quant::parallel::decode_threads());
						super::cpuexp::run_n(&ep[l], &mr, &sr)
					};
					if rtx.send(ys).is_err() {
						break;
					}
				}
			}))?;
			let recv = |t_exp: &mut f64| -> Result<Vec<Vec<f32>>, String> {
				let t = std::time::Instant::now();
				let ys = rrx
					.recv()
					.map_err(|_| "flash: the expert thread stopped".to_string());
				*t_exp += t.elapsed().as_secs_f64();
				ys
			};
			let job = |l: usize, post: &Post, sels: Vec<Vec<(usize, f32)>>| -> Result<(), String> {
				let mqs = post.iter().map(|p| p.0.clone()).collect();
				jtx.send((l, mqs, sels))
					.map_err(|_| "flash: the expert thread stopped".to_string())
			};
			let mut di = 0;
			// wave B's post of the layer before, its experts on the cores
			let mut b_post: Option<Post> = None;
			for l in 0..NL {
				let save = sv.is_some();
				// wave A
				gpu.set_wave(0, 0);
				if l == PLE_LAYER {
					let es = rows
						.take()
						.ok_or("flash: n-gram rows taken")
						.map(join_rows)??;
					let save = sv.as_mut().map(|s| &mut s.csts);
					gpu.ple_n(&mut xa, &es[..na], &mut self.cstate, save)?;
					es_b = Some(es[na..].to_vec());
				}
				let (mut post_a, dt) = dense(self, gpu, &mut xa, l, di, 0, save)?;
				t_mix += dt;
				let sels_a = route_all(l, &post_a);
				// wave B's experts of the layer before (they ran beside A's
				// dense part), its shared expert, combined
				if let Some(mut post_b) = b_post.take() {
					let ys = recv(&mut t_exp)?;
					let t = std::time::Instant::now();
					gpu.set_wave(1, na);
					gpu.shared_n(&mut post_b)?;
					combine_into(&mut xb, &post_b, &ys);
					t_exp += t.elapsed().as_secs_f64();
				}
				job(l, &post_a, sels_a)?;
				// wave B, beside wave A's experts
				gpu.set_wave(1, na);
				if l == PLE_LAYER {
					let es = es_b.take().ok_or("flash: n-gram rows of wave B")?;
					let save = sv.as_mut().map(|s| &mut s.csts);
					gpu.ple_n(&mut xb, &es, &mut self.cstate, save)?;
				}
				let (post_b, dt) = dense(self, gpu, &mut xb, l, di, na, save)?;
				t_mix += dt;
				let sels_b = route_all(l, &post_b);
				let ys = recv(&mut t_exp)?;
				let t = std::time::Instant::now();
				gpu.set_wave(0, 0);
				gpu.shared_n(&mut post_a)?;
				combine_into(&mut xa, &post_a, &ys);
				t_exp += t.elapsed().as_secs_f64();
				job(l, &post_b, sels_b)?;
				b_post = Some(post_b);
				if !is_attn(l) {
					di += 1;
				}
			}
			if let Some(mut post_b) = b_post.take() {
				let ys = recv(&mut t_exp)?;
				gpu.set_wave(1, na);
				gpu.shared_n(&mut post_b)?;
				combine_into(&mut xb, &post_b, &ys);
			}
			Ok(())
		})();
		// the expert thread is done with the step (its job queue closed)
		let _ = idle_rx.recv();
		r?;
		xa.append(&mut xb);
		Ok((xa, [t0.elapsed().as_secs_f64(), t_mix, t_exp]))
	}

	/// [`Self::step_waves`] with each wave's next run queued behind the
	/// other's (`QWEN_FLASH_WAVE_PIPE`, default on): the host prepares one
	/// wave's run, does its attention or reads its outputs while the GPU
	/// runs the other's, so the GPU does not wait for the host between
	/// runs. Per layer: B's experts of the layer before combined, B's run
	/// queued behind A's; (attention layers: A's pre waited, its attention,
	/// its post queued, the same for B); A's post waited, its experts on
	/// the cores while the GPU runs B's, combined; B's post waited, its
	/// experts on the cores, A's next layer queued for the GPU meanwhile.
	/// The queue still holds A's run of a layer before B's (the delta
	/// states, the caches), and each run is the one [`Self::step_waves`]
	/// submits (the shared expert in the post's submission: same kernels),
	/// so the output is bit-identical.
	#[allow(clippy::type_complexity)]
	fn step_waves_pipe(
		&mut self,
		gpu: &mut super::gpudec::GpuDec,
		ids: &[u32],
		sv: Option<&mut Spec>,
	) -> Result<(Vec<Vec<f32>>, [f64; 3]), String> {
		let t0 = std::time::Instant::now();
		let spin = flash_spin();
		let _sp = (spin > 0).then(|| quant::parallel::spin_window(spin));
		let _w =
			(spin > 0).then(|| quant::parallel::limit_width(quant::parallel::decode_threads()));
		let n = ids.len();
		let na = n.div_ceil(2);
		let (mut xa, rows, cs, sn) = self.step_inputs(ids)?;
		let mut xb = xa.split_off(na);
		let mut es: Option<Rows<'_>> = Some(Box::new(move || join_rows(rows)));
		let mut tw = [0f64; 9];
		let (_, [t_mix, t_exp]) = self.pipe_layers(
			gpu,
			&mut xa,
			&mut xb,
			(0, na),
			0..NL,
			(&cs, &sn),
			&mut es,
			sv,
			&mut tw,
			&mut |_, _| Ok(false),
		)?;
		if self.timing {
			print_waves(n, &tw);
		}
		xa.append(&mut xb);
		Ok((xa, [t0.elapsed().as_secs_f64(), t_mix, t_exp]))
	}

	/// [`Self::step_waves_pipe`]'s layers `layers` of wave A `xa` and wave B
	/// `xb` (B may be empty: A alone), at positions `ba..` and `bb..` of the
	/// step (`cs` / `sn`: RoPE per step position; the delta mixers' aux
	/// slots are the positions, so the waves of a step need not be its
	/// halves, nor start together). `es` gives the
	/// n-gram rows of A's then B's positions, taken at the PLE layer.
	/// `hook` is called after A's first run of a layer is queued
	/// ([`Hook::Begun`]), while A's experts run on the cores
	/// ([`Hook::Experts`]) and, without B, after each layer
	/// ([`Hook::LayerDone`]: `true` stops there, nothing in flight).
	/// Returns the layers done (the end of `layers` unless stopped) and
	/// (mixer, moe) seconds; `tw` sums the driver's work per kind.
	#[allow(clippy::too_many_arguments, clippy::type_complexity)]
	fn pipe_layers(
		&mut self,
		gpu: &mut super::gpudec::GpuDec,
		xa: &mut [Vec<f32>],
		xb: &mut [Vec<f32>],
		(ba, bb): (usize, usize),
		layers: std::ops::Range<usize>,
		(cs, sn): (&[[f32; RD]], &[[f32; RD]]),
		es: &mut Option<Rows<'_>>,
		mut sv: Option<&mut Spec>,
		tw: &mut [f64; 9],
		hook: &mut dyn FnMut(&mut super::gpudec::GpuDec, Hook) -> Result<bool, String>,
	) -> Result<(usize, [f64; 2]), String> {
		use super::gpudec::{GpuDec, Post};
		if layers.is_empty() {
			return Ok((layers.start, [0.0; 2]));
		}
		let na = xa.len();
		let hb = !xb.is_empty();
		let mut es_b: Option<Vec<Vec<f32>>> = None;
		let (mut t_mix, mut t_exp) = (0f64, 0f64);
		let save = sv.is_some();
		// delta index of each layer
		let dis: Vec<usize> = (0..NL)
			.scan(0, |d, l| {
				let v = *d;
				if !is_attn(l) {
					*d += 1;
				}
				Some(v)
			})
			.collect();
		let wave = |gpu: &mut GpuDec, w: usize| gpu.set_wave(w, if w == 0 { ba } else { bb });
		// wave w's attention of layer l once its pre run is done: its post
		// run queued; the attention's seconds
		let attend = |me: &mut Self,
		              gpu: &mut GpuDec,
		              xs: &[Vec<f32>],
		              w: usize,
		              l: usize|
		 -> Result<f64, String> {
			wave(gpu, w);
			let (nw, base) = (xs.len(), if w == 0 { ba } else { bb });
			let pre = gpu.attn_pre_end(nw, l, &cs[base..base + nw], &sn[base..base + nw])?;
			let t = std::time::Instant::now();
			for (ti, ([q, _, k, va, iq, ik], _)) in pre.iter().enumerate() {
				for (b, v) in [(B::q, q), (B::k, k), (B::va, va), (B::iq, iq), (B::ik, ik)] {
					f16_into(&mut h16m(me.buf(b))[ti * v.len()..], v);
				}
			}
			let p = me.pos;
			me.pos = p + base;
			let r = me.attention(l, nw);
			me.pos = p;
			r?;
			let o = me.b16(B::o);
			let (a, inj): (Vec<Vec<f32>>, Vec<[f32; HC]>) = pre
				.iter()
				.enumerate()
				.map(|(ti, ([_, gate, ..], inj))| {
					let mut a = unf16v(&o[ti * HQ * DA..][..HQ * DA]);
					for (a, g) in a.iter_mut().zip(gate) {
						*a *= g;
					}
					(a, *inj)
				})
				.unzip();
			let dt = t.elapsed().as_secs_f64();
			gpu.post_begin(xs, l, &a, &inj)?;
			Ok(dt)
		};
		let route_all = |l: usize, post: &Post| -> Vec<Vec<(usize, f32)>> {
			let sels: Vec<Vec<(usize, f32)>> = post.iter().map(|(_, _, lg, _)| route(lg)).collect();
			sel_dump(l, false, sels.iter().map(|s| s.as_slice()));
			sels
		};
		let combine_into = |xs: &mut [Vec<f32>], post: &Post, ys: &[Vec<f32>]| {
			for ((x, (_, inj_m, _, shared)), y) in xs.iter_mut().zip(post).zip(ys) {
				for c in 0..HC {
					for i in 0..W {
						x[c * W + i] += inj_m[c] * (shared[i] + y[i]);
					}
				}
			}
		};
		// the cores' pass runs on the expert thread
		let ex = (self.pools.clone(), self.epre.clone());
		type Job = (
			usize,
			Vec<Vec<f32>>,
			Vec<Vec<(usize, f32)>>,
			std::time::Instant,
		);
		let mut done = layers.start;
		let (idle_tx, idle_rx) = std::sync::mpsc::channel::<()>();
		let r = (|| -> Result<(), String> {
			let (jtx, jrx) = std::sync::mpsc::channel::<Job>();
			let (rtx, rrx) = std::sync::mpsc::channel::<(Vec<Vec<f32>>, f64, std::time::Instant)>();
			self.exp_thread.run(Box::new(move || {
				let _idle = idle_tx;
				let ep = pool_views(&ex.0, &ex.1);
				for (l, mqs, sels, sent) in jrx {
					let wake = sent.elapsed().as_secs_f64();
					let mr: Vec<&[f32]> = mqs.iter().map(|v| v.as_slice()).collect();
					let sr: Vec<&[(usize, f32)]> = sels.iter().map(|v| v.as_slice()).collect();
					// DRAM-bound: 16 shards read as fast as 24 (`cpuexp` bench)
					let ys = {
						let _w = quant::parallel::limit_width(quant::parallel::decode_threads());
						super::cpuexp::run_n(&ep[l], &mr, &sr)
					};
					if rtx.send((ys, wake, std::time::Instant::now())).is_err() {
						break;
					}
				}
			}))?;
			let recv = |t_exp: &mut f64, tw: &mut [f64; 9]| -> Result<Vec<Vec<f32>>, String> {
				let t = std::time::Instant::now();
				let (ys, wake, sent) = rrx
					.recv()
					.map_err(|_| "flash: the expert thread stopped".to_string())?;
				let e = t.elapsed().as_secs_f64();
				*t_exp += e;
				tw[3] += e;
				tw[7] += wake;
				tw[8] += sent.elapsed().as_secs_f64();
				Ok(ys)
			};
			let job = |l: usize, post: &Post, tw: &mut [f64; 9]| -> Result<(), String> {
				let t = std::time::Instant::now();
				let sels = route_all(l, post);
				let mqs = post.iter().map(|p| p.0.clone()).collect();
				let r = jtx
					.send((l, mqs, sels, std::time::Instant::now()))
					.map_err(|_| "flash: the expert thread stopped".to_string());
				tw[2] += t.elapsed().as_secs_f64();
				r
			};
			// wave w's first run of layer l queued (at the PLE layer the
			// PLE term first: a run of its own, waited for)
			let mut begin = |me: &mut Self,
			                 gpu: &mut GpuDec,
			                 xs: &mut [Vec<f32>],
			                 w: usize,
			                 l: usize,
			                 sv: Option<&mut Spec>|
			 -> Result<(), String> {
				wave(gpu, w);
				if l == PLE_LAYER {
					let es = if w == 0 {
						let mut e = es.take().ok_or("flash: n-gram rows taken")?()?;
						es_b = Some(e.split_off(na));
						e
					} else {
						es_b.take().ok_or("flash: n-gram rows of wave B")?
					};
					let save = sv.map(|s| &mut s.csts);
					gpu.ple_n(xs, &es, &mut me.cstate, save)?;
				}
				if is_attn(l) {
					gpu.attn_pre_begin(xs, l)
				} else if gpu.delta_post_begin(xs, l, dis[l], save)? {
					Ok(())
				} else {
					Err("flash: two waves need the fused delta layers".into())
				}
			};
			let tick = |tw: &mut [f64; 9], i: usize, t: std::time::Instant| {
				tw[i] += t.elapsed().as_secs_f64();
			};
			// a post run waited for, its outputs read
			let post_end = |gpu: &mut GpuDec,
			                xs: &mut [Vec<f32>],
			                l: usize,
			                site: usize,
			                tw: &mut [f64; 9]|
			 -> Result<Post, String> {
				let (t, g) = (std::time::Instant::now(), gpu.t_gpu);
				let p = gpu.post_end(xs, l, site)?;
				let w = gpu.t_gpu - g;
				tw[0] += w;
				tw[1] += t.elapsed().as_secs_f64() - w;
				Ok(p)
			};
			let t = std::time::Instant::now();
			begin(self, gpu, xa, 0, layers.start, sv.as_deref_mut())?;
			tick(tw, 5, t);
			hook(gpu, Hook::Begun)?;
			// wave B's post of the layer before, its experts on the cores
			let mut b_post: Option<Post> = None;
			for l in layers.clone() {
				if let Some(pb) = b_post.take() {
					let ys = recv(&mut t_exp, tw)?;
					let t = std::time::Instant::now();
					combine_into(xb, &pb, &ys);
					tick(tw, 4, t);
				}
				if hb {
					let t = std::time::Instant::now();
					begin(self, gpu, xb, 1, l, sv.as_deref_mut())?;
					tick(tw, 5, t);
				}
				if is_attn(l) {
					let t = std::time::Instant::now();
					t_mix += attend(self, gpu, xa, 0, l)?;
					if hb {
						t_mix += attend(self, gpu, xb, 1, l)?;
					}
					tick(tw, 6, t);
				}
				let site = if is_attn(l) { 1 } else { 0 };
				wave(gpu, 0);
				let pa = post_end(gpu, xa, l, site, tw)?;
				job(l, &pa, tw)?;
				hook(gpu, Hook::Experts)?;
				let ys = recv(&mut t_exp, tw)?;
				let t = std::time::Instant::now();
				combine_into(xa, &pa, &ys);
				tick(tw, 4, t);
				done = l + 1;
				if !hb && hook(gpu, Hook::LayerDone)? {
					break;
				}
				// (with B: A's next run queued once B's experts are on the
				// cores, which then do not wait for its preparation; B's run
				// of this layer is queued before it either way. Queuing B's
				// run after A's experts too only moves the wait to the GPU)
				if hb {
					wave(gpu, 1);
					let pb = post_end(gpu, xb, l, site, tw)?;
					job(l, &pb, tw)?;
					b_post = Some(pb);
				}
				if l + 1 < layers.end {
					let t = std::time::Instant::now();
					begin(self, gpu, xa, 0, l + 1, sv.as_deref_mut())?;
					tick(tw, 5, t);
					hook(gpu, Hook::Begun)?;
				}
			}
			if let Some(pb) = b_post.take() {
				let ys = recv(&mut t_exp, tw)?;
				combine_into(xb, &pb, &ys);
			}
			Ok(())
		})();
		// the expert thread is done with the step (its job queue closed)
		let _ = idle_rx.recv();
		r?;
		Ok((done, [t_mix, t_exp]))
	}

	/// The most positions a [`Self::verify`] step takes: the dense GPU's
	/// ([`super::gpudec::GpuDec::max_np`]), 0 without one.
	pub fn spec_k(&self) -> usize {
		self.gpu.as_ref().map_or(0, |g| g.max_np())
	}

	/// Feed `ids` (at most [`Self::spec_k`]) tentatively in one step and
	/// return every position's logits; [`Self::commit`] keeps a prefix.
	/// Without the dense GPU (dropped since the caller asked
	/// [`Self::spec_k`]) only the first of `ids` is fed: one row.
	pub fn verify(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>, String> {
		self.meter.close(0.0);
		self.spec = None;
		let t0 = std::time::Instant::now();
		self.dense_heal()?;
		if self.gpu.is_none() {
			let first = ids.first().ok_or("verify: no tokens")?;
			return Ok(vec![self.chunk(std::slice::from_ref(first), true)?]);
		}
		let r = self.step_gpu_n(ids, ids.len(), ids.len() > 1)?;
		let k = ids.len();
		if (1..=self.meter.verify.len()).contains(&k) {
			self.meter.open = Some((k, t0.elapsed().as_secs_f64() * 1e3));
		}
		if self.timing && mtp_auto() {
			self.meter.since_log += 1;
			if self.meter.since_log >= 64 {
				self.meter.since_log = 0;
				if let Some(c) = mtp_cost() {
					let m = &self.meter;
					eprintln!(
						"[flash] step costs {:?} (verify n {:?}, first {:.1} / {}, pass {:.1} / {})",
						m.costs(&c).iter().map(|v| v.round() as i32).collect::<Vec<_>>(),
						m.verify.iter().map(|e| e.n).collect::<Vec<_>>(),
						m.first.v,
						m.first.n,
						m.pass.v,
						m.pass.n
					);
				}
			}
		}
		Ok(r)
	}

	/// Between calls: drop a dense GPU that [`Self::dense_watch`] found bad
	/// (the delta states back to the host first, the one-position graphs
	/// loaded when there are none), or build one again when it is time.
	/// Fails only while the state is lost with a GPU.
	fn dense_heal(&mut self) -> Result<(), String> {
		if let Some(why) = self.dense_bad.take() {
			self.dense_drop(&why);
		} else if self.gpu.is_none()
			&& self.dense_new.is_some()
			&& self
				.dense_retry
				.is_some_and(|at| std::time::Instant::now() >= at)
		{
			let t = std::time::Instant::now();
			// (a GPU step stages its positions in the prefill set's buffers)
			let r = self.use_set(0).and_then(|()| {
				let new = self.dense_new.as_ref().unwrap();
				dense_open(
					new,
					self.dense_need,
					&self.weights,
					&self.head,
					self.head_gpu,
					self.mtp_dir.as_deref(),
				)
			});
			match r {
				Ok(g) => {
					eprintln!(
						"flash: dense part back on the GPU ({:.1} s)",
						t.elapsed().as_secs_f64()
					);
					self.gpu = Some(g);
					self.dense_since = Some(std::time::Instant::now());
					self.dense_retry = None;
					self.dense_base = [f64::INFINITY; 9];
					self.dense_steps = 0;
					self.dense_slow = 0;
					// (the head's rows are made by GPU steps only)
					self.mtp_clear();
				}
				Err(e) => {
					self.dense_retry = dense_retry_at(self.dense_backoff);
					eprintln!(
						"flash: dense part not on the GPU ({e}){}",
						retry_note(self.dense_backoff)
					);
				}
			}
		}
		if self.lost {
			return Err("flash: the decode state was lost with the GPU; start over".into());
		}
		Ok(())
	}

	fn dense_drop(&mut self, why: &str) {
		// The budget Vulkan reports counts VRAM another application holds
		// as free (WDDM can demote it), so a GPU built again while that
		// application runs pages again: back off while drops follow
		// recoveries within two minutes (x2 each, at most x32)
		let quick = self
			.dense_since
			.is_some_and(|t| t.elapsed() < std::time::Duration::from_secs(120));
		self.dense_backoff = if quick {
			(self.dense_backoff * 2).min(32)
		} else {
			1
		};
		eprintln!(
			"flash: dense GPU dropped ({why}){}",
			retry_note(self.dense_backoff)
		);
		if self.on_gtx {
			if let Err(e) = self.gtx_home() {
				eprintln!("flash: the delta states on the GPU are lost ({e})");
				self.on_gtx = false;
				self.lost = true;
			}
		}
		self.gpu = None;
		self.spec = None;
		self.mtp_clear();
		self.dense_retry = dense_retry_at(self.dense_backoff);
		// the one-position graphs, as a load without the GPU has them
		if self.sets.len() == 1 && self.graphs.join("delta_pre_d1.blob").exists() {
			let t = std::time::Instant::now();
			self.far = None;
			self.far_tried = false;
			self.sets[self.cur].deactivate();
			let cfg = Config {
				graphs: self.graphs.clone(),
				weights: self.weights.clone(),
				..Default::default()
			};
			match Set::load(
				self.npu,
				&cfg,
				"_d1",
				(1, G1, 1),
				self.wb,
				false,
				false,
				false,
			) {
				Ok(s) => {
					self.sets.push(s);
					self.cur = 1;
					eprintln!(
						"flash: one-position graphs loaded ({:.1} s)",
						t.elapsed().as_secs_f64()
					);
				}
				Err(e) => {
					eprintln!("flash: no one-position graphs ({e}); decode on the prefill graphs");
					let cur = self.cur;
					if let Err(e) = self.sets[cur].activate(self.npu, &self.weights, "", self.wb) {
						eprintln!("flash: prefill graphs not loaded again: {e}");
					}
				}
			}
		}
	}

	/// After a GPU step of `n` positions that waited `t` seconds for the
	/// GPU (`r`: its result): mark a GPU bad that failed, stalled (a step
	/// over a second: its memory paged) or stays slower than it was (8
	/// steps in a row over twice its fastest: part of its VRAM demoted to
	/// system memory). [`Self::dense_heal`] drops it at the next call.
	fn dense_watch(&mut self, n: usize, t: f64, r: Result<(), &str>) {
		if let Err(e) = r {
			self.dense_bad = Some(format!("a step failed: {e}"));
			return;
		}
		self.dense_steps += 1;
		// (the first steps after an upload warm up)
		if self.dense_steps <= 2 {
			return;
		}
		let b = &mut self.dense_base[n.min(8)];
		*b = b.min(t);
		let b = *b;
		if t > 1.0 {
			self.dense_bad = Some(format!(
				"a step of {n} waited {:.0} ms for it: its memory is being paged?",
				t * 1e3
			));
		} else if t > 2.0 * b + 0.01 {
			self.dense_slow += 1;
			if self.dense_slow >= 8 {
				self.dense_bad = Some(format!(
					"8 steps waited {:.0} ms for it, {:.0} ms when it was loaded: VRAM demoted?",
					t * 1e3,
					b * 1e3
				));
			}
		} else {
			self.dense_slow = 0;
		}
	}

	/// Keep the first `n` tokens of the last [`Self::verify`] and drop the
	/// rest: position, n-gram context and conv windows from the copies the
	/// step saved, the delta states by replaying `n` positions from the
	/// state before it, the attention caches truncated.
	pub fn commit(&mut self, n: usize) -> Result<(), String> {
		let t0 = std::time::Instant::now();
		let r = self.commit_spec(n);
		self.meter.close(t0.elapsed().as_secs_f64() * 1e3);
		r
	}

	fn commit_spec(&mut self, n: usize) -> Result<(), String> {
		let Some(sv) = self.spec.take() else {
			return Ok(());
		};
		if std::env::var_os("QWEN_FLASH_MTP_LOG").is_some() {
			eprintln!("[flash] mtp commit {n} of {}", sv.ids.len());
		}
		if n >= sv.ids.len() {
			return Ok(());
		}
		self.pos = sv.pos0 + n;
		let p = self.pos;
		self.mpend.retain(|e| e.0 < p);
		self.nctx = sv.nctx;
		for &id in &sv.ids[..n] {
			self.nctx.push(id, EOS);
		}
		self.cstate.copy_from_slice(&sv.csts[n]);
		if self.on_gtx {
			self.gpu
				.as_mut()
				.ok_or("flash: no GPU")?
				.delta_rollback(n)?;
		} else {
			for (c, s) in self.conv.iter_mut().zip(&sv.convs) {
				c.copy_from_slice(&s[n]);
			}
			for (di, rin) in sv.rin.iter().enumerate() {
				std::mem::swap(&mut self.hstate[di], &mut self.bak[di]);
				self.step_recurrence(di, &rin[..n], false)?;
			}
		}
		let p = self.pos;
		for kv in self.kv.iter_mut().skip(3).step_by(4) {
			kv.k.truncate(p)?;
			kv.v.truncate(p)?;
			kv.ik.truncate(p)?;
			kv.bk.truncate(p / 4)?;
		}
		if let Some(a) = self.pattn.as_mut() {
			a.invalidate();
		}
		if let Some(f) = self.far.as_mut() {
			f.truncate(p);
		}
		if let Some(x) = self.idx.as_mut() {
			x.truncate(p / 4);
		}
		Ok(())
	}

	/// The logits of chunk row `t`: final mixer, lm_head (host, f32).
	fn head(&self, t: usize) -> Vec<f32> {
		self.head_rows(&[t]).remove(0)
	}

	/// The final mixer's outputs of chunk rows `ts`: in groups of
	/// [`MIX_G`], each weight row of `hcf` read once for the group (per
	/// position the arithmetic of a group of one).
	fn mixed_n(&self, ts: &[usize]) -> Vec<Vec<f32>> {
		ts.chunks(MIX_G).flat_map(|g| self.mixed_g(g)).collect()
	}

	fn mixed_g(&self, ts: &[usize]) -> Vec<Vec<f32>> {
		let np = ts.len();
		let [nw, down, up] = &self.hcf;
		let xns: Vec<Vec<f32>> = ts
			.iter()
			.map(|&t| {
				let xs: Vec<f32> = self.b16(B::x)[t * HCW..][..HCW]
					.iter()
					.map(|&h| unf16(h))
					.collect();
				let mut xn = vec![0f32; HCW];
				for c in 0..HC {
					let r = &xs[c * W..][..W];
					let ms = r.iter().map(|v| v * v).sum::<f32>() / W as f32;
					let k = 1.0 / (ms + 1e-6).sqrt();
					for i in 0..W {
						xn[c * W + i] = r[i] * k * nw[c * W + i];
					}
				}
				xn
			})
			.collect();
		let rank = super::HC_RANK;
		let mut m = vec![0f32; np * rank];
		let mut u = vec![0f32; np * HCW];
		#[derive(Clone, Copy)]
		struct P(*mut f32);
		unsafe impl Send for P {}
		unsafe impl Sync for P {}
		let pm = P(m.as_mut_ptr());
		const RB: usize = 4;
		par_for(rank / RB, |b| {
			let pm = pm;
			for r in b * RB..(b + 1) * RB {
				let w = &down[r * HCW..][..HCW];
				for (p, xn) in xns.iter().enumerate() {
					// SAFETY: AVX2 / FMA checked at load; block b owns
					// rows [b RB, (b + 1) RB) of every position's m
					unsafe {
						let v = dot_f32(w, xn) / HC as f32;
						*pm.0.add(p * rank + r) = v / (1.0 + (-v).exp());
					}
				}
			}
		});
		let pu = P(u.as_mut_ptr());
		const UB: usize = 128;
		par_for(HCW / UB, |b| {
			let pu = pu;
			for j in b * UB..(b + 1) * UB {
				let w = &up[j * rank..][..rank];
				for p in 0..np {
					// SAFETY: as above; block b owns u[p][b UB..][..UB]
					unsafe {
						let v = dot_f32(w, &m[p * rank..][..rank]);
						*pu.0.add(p * HCW + j) = 1.0 / (1.0 + (-v).exp());
					}
				}
			}
		});
		(0..np)
			.map(|p| {
				let (u, xn) = (&u[p * HCW..][..HCW], &xns[p]);
				let mut mixed = vec![0f32; W];
				for c in 0..HC {
					for i in 0..W {
						let j = c * W + i;
						mixed[i] += u[j] * xn[j] / HC as f32;
					}
				}
				mixed
			})
			.collect()
	}

	/// The logits of chunk rows `ts` (one pass over the lm_head).
	fn head_rows(&self, ts: &[usize]) -> Vec<Vec<f32>> {
		let ms = self.mixed_n(ts);
		self.head_ms(&ms)
	}

	/// The logits of final mixer outputs `ms` (host lm_head).
	fn head_ms(&self, ms: &[Vec<f32>]) -> Vec<Vec<f32>> {
		self.head_ms_from(ms, 0)
	}

	/// [`Self::head_ms`] of the rows from `r0` on (the rest left zero).
	fn head_ms_from(&self, ms: &[Vec<f32>], r0: usize) -> Vec<Vec<f32>> {
		head_rows_host(&self.head, ms, r0)
	}

	/// Token `id`'s embedding row (f32).
	fn emb_row(&self, id: u32) -> Result<Vec<f32>, String> {
		emb_row_from(&self.emb, id)
	}

	fn mtp_clear(&mut self) {
		self.mpend.clear();
		if let Some(m) = self.mtp.as_mut() {
			m.clear();
		}
	}

	/// Keep the MTP head's pending rows to the last [`mtp_ctx`].
	fn mpend_cap(&mut self) {
		let c = mtp_ctx();
		if self.mpend.len() > c {
			self.mpend.drain(..self.mpend.len() - c);
		}
	}

	/// True when the engine drafts with its MTP head.
	pub fn has_mtp(&self) -> bool {
		self.mtp.is_some()
	}

	/// After [`Self::commit`]: up to `QWEN_FLASH_MTP_N` (default 3, at most
	/// [`Self::spec_k`] - 1) MTP drafts to follow `next`, the token sampled
	/// from the last position fed. The head first runs the rows of the
	/// positions fed since its last run (each position's streams with the
	/// token after it), then chains on its own streams; a chain stops
	/// before the draft at which the product of the drafts' probabilities
	/// falls below [`mtp_pmin`], and keeps the prefix that is cheapest per
	/// expected token ([`mtp_cost`]): a verify position costs ~55 % of a
	/// plain step (the MoE reads per position), so a draft pays only when
	/// likely to be accepted.
	/// Empty without a head or with nothing fed since the last call.
	///
	/// With a [`sampler::Coupling`] ([`Self::set_coupling`]) at temperature >
	/// 0 each draft is the head's coupled draw at its position: the token the
	/// sampler will draw there whenever the head's distribution is close to
	/// the model's. Drafts never change what is sampled.
	pub fn drafts(&mut self, next: u32) -> Result<Vec<u32>, String> {
		let nmax = mtp_n().min(self.spec_k().saturating_sub(1));
		if self.mtp.is_none() || self.mpend.is_empty() || nmax == 0 {
			return Ok(Vec::new());
		}
		let t0 = std::time::Instant::now();
		// the head's ~40 small dispatches per row: the decode shards kept
		// spinning between them, as in a step (`QWEN_FLASH_MTP_SPIN=0`
		// is the A/B)
		let spin = if std::env::var("QWEN_FLASH_MTP_SPIN").is_ok_and(|v| v == "0") {
			0
		} else {
			flash_spin()
		};
		let _sp = (spin > 0).then(|| quant::parallel::spin_window(spin));
		let _w =
			(spin > 0).then(|| quant::parallel::limit_width(quant::parallel::decode_threads()));
		let log = std::env::var_os("QWEN_FLASH_MTP_LOG").is_some();
		if log {
			eprintln!("[flash] mtp next {next}");
		}
		let pend = std::mem::take(&mut self.mpend);
		let toks: Vec<u32> = pend[1..].iter().map(|e| e.2).chain([next]).collect();
		let embs = toks
			.iter()
			.map(|&t| self.emb_row(t))
			.collect::<Result<Vec<_>, _>>()?;
		let rope = |me: &Self, p: usize| -> [(f32, f32); RD / 2] {
			me.rope_sc(p).map(|(s, c)| (c as f32, s as f32))
		};
		let rows: Vec<super::mtp::Row<'_>> = pend
			.iter()
			.zip(&embs)
			.map(|(e, emb)| super::mtp::Row {
				emb,
				h: &e.1,
				rope: rope(self, e.0 + 1),
			})
			.collect();
		let mut m = self.mtp.take().unwrap();
		m.trim();
		// (the head's logits at the sampler's temperature times
		// `QWEN_FLASH_COUPLED_TAU`: the head is sharper than the model)
		let coupling = self.coupling.map(|c| sampler::Coupling {
			params: sampler::SampleParams {
				temperature: c.params.temperature * coupled_tau(),
				..c.params
			},
			..c
		});
		let (mut t_run, mut t_head) = (0f64, 0f64);
		let cost = self.step_costs();
		// ms from the call's start to the end of each head pass
		let mut stamps: Vec<f64> = Vec::new();
		let mut chain = || -> Result<Vec<u32>, String> {
			let tr = std::time::Instant::now();
			let (mut h, mut mix) = m.run_on(&rows, self.gpu.as_mut())?;
			t_run += tr.elapsed().as_secs_f64();
			let base = m.len();
			let mut p = pend.last().unwrap().0 + 2;
			let (pmin, pscale) = (mtp_pmin(), mtp_pscale());
			let cal = mtp_cal();
			let qmax = mtp_accept(cal, pscale, 1.0).1;
			// the head's probability that every draft so far is accepted
			let mut pall = 1f32;
			let mut ps = Vec::new();
			let mut out = Vec::new();
			loop {
				let th = std::time::Instant::now();
				let vblk = self
					.gpu
					.as_ref()
					.map_or(0, |g| g.head_blocks())
					.min(mtp_vblk());
				let lg = if vblk > 0
					&& vblk < self.gpu.as_ref().unwrap().head_blocks()
					&& vblk * super::gpudec::HEAD_ROWS <= DRAFT_SPECIAL0
				{
					// the first `vblk` blocks on the GPU, the special
					// tokens on the cores meanwhile; the ids between are
					// never drafted
					let g = self.gpu.as_mut().unwrap();
					g.head_begin(&[&mix], vblk)?;
					let n = vblk * super::gpudec::HEAD_ROWS;
					let mut lg = vec![f32::NEG_INFINITY; super::VOCAB];
					for (r, v) in lg.iter_mut().enumerate().skip(DRAFT_SPECIAL0) {
						// SAFETY: F16C / FMA checked at load
						*v = unsafe { self.head.dot(r, &mix) };
					}
					let mut o = [std::mem::take(&mut lg)];
					self.gpu.as_mut().unwrap().head_wait(&mut o, n)?;
					let [lg] = o;
					(lg, [0..n, DRAFT_SPECIAL0..super::VOCAB])
				} else {
					let lg = match self
						.gpu
						.as_mut()
						.map(|g| g.head(&mix))
						.transpose()?
						.flatten()
					{
						Some(l) => l,
						None => self.head_ms(std::slice::from_ref(&mix)).remove(0),
					};
					(lg, [0..super::VOCAB, 0..0])
				};
				let (lg, ranges) = lg;
				let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
				for i in ranges.iter().cloned().flatten() {
					if lg[i] > bv {
						(bi, bv) = (i, lg[i]);
					}
				}
				// coupled: the head's draw with the sampler's noise at this
				// draft's position (`lg` is -inf outside `ranges`)
				if let Some(c) = coupling.filter(|c| c.params.temperature > 0.0) {
					bi = c.pick(&lg, p as u64).0 as usize;
				}
				let sum: f32 = ranges
					.iter()
					.map(|r| lg[r.clone()].iter().map(|v| (v - bv).exp()).sum::<f32>())
					.sum();
				t_head += th.elapsed().as_secs_f64();
				stamps.push(t0.elapsed().as_secs_f64() * 1e3);
				if log {
					// the head's three most likely tokens and their
					// probabilities (which candidate the sampler drew)
					let mut top = [(0usize, f32::NEG_INFINITY); 3];
					for i in ranges.iter().cloned().flatten() {
						if lg[i] > top[2].1 {
							top[2] = (i, lg[i]);
							top.sort_by(|x, y| y.1.total_cmp(&x.1));
						}
					}
					eprintln!(
						"[flash] mtp top {}",
						top.map(|(i, v)| format!("{i}:{:.4}", (v - bv).exp() / sum))
							.join(" ")
					);
				}
				pall *= mtp_accept(cal, pscale, sum).0;
				if pall < pmin {
					break;
				}
				ps.push(pall);
				out.push(bi as u32);
				if out.len() == nmax {
					break;
				}
				// the cost model would drop every longer chain: stop
				// before its drafts (`QWEN_FLASH_MTP_EARLY=0` is the A/B)
				if let Some(c) = cost.as_deref() {
					if mtp_early() && !mtp_may_grow(&ps, c, nmax - out.len(), qmax) {
						break;
					}
				}
				let emb = self.emb_row(bi as u32)?;
				let row = super::mtp::Row {
					emb: &emb,
					h: &h,
					rope: rope(self, p),
				};
				let tr = std::time::Instant::now();
				(h, mix) = m.run_on(&[row], self.gpu.as_mut())?;
				t_run += tr.elapsed().as_secs_f64();
				p += 1;
			}
			m.truncate(base);
			if log {
				eprintln!("[flash] mtp chain {out:?} p {ps:?}");
			}
			// the number of drafts with the most expected tokens per unit
			// of step cost
			if let Some(c) = cost.as_deref() {
				out.truncate(mtp_keep(&ps, c));
			}
			Ok(out)
		};
		let r = chain();
		self.mtp = Some(m);
		if r.is_ok() {
			if let (Some(&a), Some(&b)) = (stamps.first(), stamps.last()) {
				self.meter.first.add(a);
				if stamps.len() > 1 {
					self.meter.pass.add((b - a) / (stamps.len() - 1) as f64);
				}
			}
		}
		if self.timing {
			eprintln!(
				"[flash] mtp: {} rows, {:?} in {:.1} ms (layer {:.1}, head {:.1})",
				pend.len(),
				r,
				t0.elapsed().as_secs_f64() * 1e3,
				t_run * 1e3,
				t_head * 1e3
			);
		}
		r
	}

	/// Feed `ids` and return the logits of every position (a diagnostic:
	/// the KL against truth logits).
	pub fn forward_all(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>, String> {
		let mut all = Vec::with_capacity(ids.len());
		let bar = quant::progress::Prefill::start(ids.len());
		let ps = self.pieces(ids);
		for (i, c) in ps.iter().cloned().enumerate() {
			quant::progress::chunk(c.start, c.len());
			self.prefetch(ids, &ps, i);
			// a GPU step returns its logits (the decode path's head) and
			// leaves no streams in `xfin`
			if self.is_step(c.len()) {
				let n = c.len();
				all.extend(self.step_gpu_n(&ids[c], n, false)?);
				self.xfin.clear();
				continue;
			}
			self.chunk(&ids[c], false)?;
			for (n, x) in std::mem::take(&mut self.xfin) {
				self.buf(B::x).as_mut_slice()[..x.len()].copy_from_slice(&x);
				let ts: Vec<usize> = (0..n).collect();
				all.extend(self.head_rows(&ts));
			}
		}
		bar.done();
		Ok(all)
	}
}

#[cfg(test)]
mod tests {
	use super::{mrope, mtp_accept, mtp_keep, mtp_may_grow, CostMeter, Img};

	/// Measured widths replace the prior, unmeasured ones are the prior
	/// scaled by the measured widths' total; a stall counts as 3x the mean.
	#[test]
	fn cost_meter_fills_in_from_the_prior() {
		let prior = [60.0f32, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0];
		let mut m = CostMeter::default();
		assert_eq!(m.costs(&prior), prior.to_vec());
		for _ in 0..4 {
			m.first.add(10.0);
			m.pass.add(5.0);
			m.open = Some((1, 50.0));
			m.close(0.0);
			m.open = Some((4, 90.0));
			m.close(10.0);
		}
		// width 1: 50 + 10; width 4: 100 + 10 + 2 x 5
		let c = m.costs(&prior);
		assert_eq!(c[0], 60.0);
		assert_eq!(c[3], 120.0);
		// both measured = prior: the factor is 1
		assert_eq!(c[1], 80.0);
		m.open = Some((1, 5000.0));
		m.close(0.0);
		// 50 + (150 - 50) / 5 = 70 -> 70 + 10
		assert_eq!(m.costs(&prior)[0], 80.0);
		assert!((m.costs(&prior)[7] - 200.0 * (80.0 + 120.0) / 180.0).abs() < 1e-3);
	}

	/// Stopping a chain where [`mtp_may_grow`] says so keeps exactly the
	/// drafts the whole chain would keep, for chains as the engine makes
	/// them (each probability the previous one times [`mtp_accept`]'s
	/// factor, by `pscale` and by the calibration).
	#[test]
	fn early_chain_stop_keeps_the_same_drafts() {
		let costs: [&[f32]; 3] = [
			&[89.0, 143.0, 190.0, 235.0, 280.0],
			&[67.0, 95.0, 117.0, 140.0],
			&[60.0, 61.0, 62.0, 63.0],
		];
		let mut s = 0x2545_f491_4f6c_dd1du64;
		let mut rnd = || {
			s ^= s << 13;
			s ^= s >> 7;
			s ^= s << 17;
			(s >> 40) as f32 / (1u64 << 24) as f32
		};
		for (c, cal) in costs
			.iter()
			.flat_map(|&c| [(c, None), (c, Some((-1.2f32, -0.374f32)))])
		{
			let qmax = mtp_accept(cal, 0.8, 1.0).1;
			for _ in 0..20000 {
				let n = 1 + (rnd() * 7.0) as usize;
				let mut ps = Vec::new();
				let mut p = 1f32;
				for _ in 0..n {
					// sum >= 1; often close to 1 (confident drafts)
					let sum = 1.0 + (rnd() * rnd() * 4.0).powi(1 + (rnd() * 4.0) as i32);
					let q = mtp_accept(cal, 0.8, sum).0;
					assert!(q <= qmax, "{sum} {q} {qmax}");
					p *= q;
					ps.push(p);
				}
				let full = mtp_keep(&ps, c);
				let mut len = 1;
				while len < n && mtp_may_grow(&ps[..len], c, n - len, qmax) {
					len += 1;
				}
				assert_eq!(mtp_keep(&ps[..len], c), full, "{ps:?} {c:?}");
			}
		}
	}

	/// Flash-Next's MRoPE coordinates of every position equal the 27B
	/// engine's `mm_positions` (the model's `build_image_prompt`).
	#[test]
	fn image_positions_match_the_27b_engine() {
		for (n, spans) in [(6, vec![(1, 2, 2)]), (12, vec![(2, 3, 2), (9, 1, 2)])] {
			let (want, _) = crate::engine::mm_positions(0, n, &spans).unwrap();
			let imgs: Vec<Img> = spans
				.iter()
				.map(|&(start, gw, gh)| Img { start, gw, gh })
				.collect();
			for (p, w) in want.iter().enumerate() {
				assert_eq!(mrope(&imgs, p), w.map(|v| v as f64), "position {p}");
			}
		}
		assert_eq!(mrope(&[], 7), [7.0; 3]);
	}
}
