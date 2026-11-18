use super::*;
use quant::gemv_q4k;
use quant::quantize_q8k;
use quant::GgmlType;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_void;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

use ash::vk;
use std::sync::Mutex;

// ---- Fused dequant+matmul SPIR-V kernels (precompiled offline from GLSL with
// glslc; see `src/shaders/`). The dp4a Q4_K kernel is the integer-dot-product
// (VK_KHR_shader_integer_dot_product) variant, on wherever the device passes the init self-check
// (`QWEN_GPU_NODP4A` disables it).
pub(crate) fn load_spv(bytes: &[u8]) -> Vec<u32> {
	bytes
		.chunks_exact(4)
		.map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
		.collect()
}

/// Read a `QWEN_*` on/off toggle that has a **default**.
///
/// The older toggles here are presence-tested (`var(..).is_ok()`), which is
/// fine while the feature is opt-in and useless once it ships on: there is
/// no way to spell "off". `0`, `no`, `off` and `false` (any case) turn a
/// default-on toggle off; any other value, and bare presence, turns it on.
pub(crate) fn env_on(name: &str, default: bool) -> bool {
	match std::env::var(name) {
		Err(_) => default,
		Ok(v) => !matches!(
			v.trim().to_ascii_lowercase().as_str(),
			"0" | "no" | "off" | "false"
		),
	}
}

/// Upper bound on how long one GPU submission may take before the device is
/// declared lost (30 s). Prefill dispatches are the longest and finish in
/// well under a second; anything past this is a hung shader, and hanging the
/// process is strictly worse than falling back to the CPU.
const FENCE_TIMEOUT_NS: u64 = 30 * 1_000_000_000;

const QK_K: usize = 256;
const Q4K_BLOCK: usize = 144;
const Q6K_BLOCK: usize = 210;
/// VRAM stride of a Q6_K super-block. The GGUF block is 210 B, so every
/// second block would begin 2 bytes off a dword boundary and `q6k.glsl` would
/// need a two-load funnel shift for every weight fetch. `upload` repacks to
/// this 212 B (53 dword) stride instead: +0.95 % VRAM on Q6_K weights, and
/// every weight read in the shader becomes a plain aligned dword load.
const Q6K_BLOCK_PAD: usize = 212;
const QK8_0: usize = 32;
const Q8_0_BLOCK: usize = 34;
const F16_ELEM: usize = 2;
const F32_ELEM: usize = 4;

/// `QWEN_GPU_TIMING=1`: print a per-call phase breakdown of the matmul path
/// (`pack` / `alloc` / `submit` / `readback` / `free`).
///
/// The prefill wrapper turned out to cost 5x the kernel it wraps at
/// `seq = 2048`: measuring end to end could not see that, because the
/// dispatch and the 143 MB of allocation, copy and teardown around it are
/// one number there. Read once: this is checked on the hot path.
pub(crate) fn gpu_timing() -> bool {
	static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
	*ON.get_or_init(|| std::env::var("QWEN_GPU_TIMING").is_ok())
}

/// One `seq > 1` matmul for `matmul_batch_seq`.
///
/// `src` carries the weight's quantized bytes so a *non-resident* weight can
/// be streamed into VRAM for this call. It is `None` only when the caller
/// knows the weight is already resident; a job that is neither resident nor
/// carrying `src` makes the whole batch fall back to the CPU.
///
/// `rows` caps a **resident** weight to its first `rows` output rows (the
/// short-chunk prefill split hands the GPU only part of its resident
/// block); `None` computes all of them. Streamed weights ignore it: their
/// row count is `src`'s.
pub(crate) struct SeqJob<'a> {
	pub il: usize,
	pub id: WeightId,
	pub x: &'a [f32],
	pub seq: usize,
	pub src: Option<(&'a [u8], GpuKind, usize, usize)>,
	pub rows: Option<usize>,
	/// `x` already quantized to q8_K (`quant::par_quantize_q8k`), used
	/// instead of quantizing it again when the kernel reads q8_K.
	pub q8: Option<&'a [quant::Q8K]>,
	/// `(ptr, ld, col0)`: the caller's `[seq x ld]` f32 matrix this job's
	/// rows belong to, at column `col0`. With it a tiled GEMM writes
	/// there directly through a host-pointer import (`hostimp`), and
	/// `matmul_batch_seq_end_into` skips the job's copy; the caller must
	/// pass the same triple to `end_into`.
	pub dst: Option<(usize, usize, usize)>,
}

/// Which fused dequant+matmul shader a weight uses on the GPU.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuKind {
	Q4K,
	Q6K,
	Q8_0,
	F16,
	F32,
}

impl GpuKind {
	/// Short name for the `QWEN_GPU_TIMING` job dump.
	pub(crate) fn name(self) -> &'static str {
		match self {
			GpuKind::Q4K => "Q4_K",
			GpuKind::Q6K => "Q6_K",
			GpuKind::Q8_0 => "Q8_0",
			GpuKind::F16 => "F16",
			GpuKind::F32 => "F32",
		}
	}
}

/// Persistently-mapped host pointer. Raw pointers are `!Send`/`!Sync`; we
/// assert `Send`/`Sync` because the GPU is only driven from the single
/// forward thread (the `Arc<OffloadManager>` is shared but calls are
/// serialized).
#[derive(Copy, Clone)]
pub(crate) struct MappedPtr(*mut c_void);
unsafe impl Send for MappedPtr {}
unsafe impl Sync for MappedPtr {}

pub struct GpuWeight {
	kind: GpuKind,
	weight_buf: vk::Buffer,
	weight_mem: vk::DeviceMemory,
	x_buf: vk::Buffer,
	x_mem: vk::DeviceMemory,
	x_stg_buf: vk::Buffer,
	x_stg_mem: vk::DeviceMemory,
	x_stg_ptr: MappedPtr,
	y_buf: vk::Buffer,
	y_mem: vk::DeviceMemory,
	rb_buf: vk::Buffer,
	rb_mem: vk::DeviceMemory,
	rb_ptr: MappedPtr,
	params_buf: vk::Buffer,
	params_mem: vk::DeviceMemory,
	params_stg_buf: vk::Buffer,
	params_stg_mem: vk::DeviceMemory,
	desc_set: vk::DescriptorSet,
	out_dim: usize,
	in_dim: usize,
}

/// One dispatch of a single fused matmul (one job).
#[derive(Copy, Clone)]
pub(crate) struct Dispatch {
	pipeline: vk::Pipeline,
	desc_set: vk::DescriptorSet,
	x_buf: vk::Buffer,
	x_stg: vk::Buffer,
	x_len: u64,
	p_buf: vk::Buffer,
	p_stg: vk::Buffer,
	p_len: u64,
	y_buf: vk::Buffer,
	rb_buf: vk::Buffer,
	rb_ptr: MappedPtr,
	out_dim: usize,
	gx: u32,
	gy: u32,
}

/// A decode batch submitted by `matmul_batch_begin` and not yet waited on.
pub struct PendingBatch {
	cmd: vk::CommandBuffer,
	/// `cmd` belongs to `decode_cmds` and must not be freed.
	cached: bool,
	items: Vec<Dispatch>,
	t0: std::time::Instant,
}

/// A prefill batch submitted by `matmul_batch_seq_begin`, not yet waited on.
pub struct PendingSeq {
	/// One command buffer, or with a streamed batch handed to the repack
	/// thread (`follow`) the resident rows' and then the streamed rows'.
	/// The last is submitted on `fence`.
	cmds: Vec<vk::CommandBuffer>,
	/// The last of `cmds` was sent to the repack thread (`stream::Follow`):
	/// `stream_sync` must collect it before `fence` may be waited on.
	follow: bool,
	/// Bank of the batch: its fence, its host-pool slots and its stream
	/// slot pins (`matmul_batch_seq_begin_bank`).
	bank: usize,
	items: Vec<Dispatch>,
	per_call: Vec<(vk::Buffer, vk::DeviceMemory)>,
	per_call_maps: Vec<(vk::DeviceMemory, *mut c_void)>,
	pool: vk::DescriptorPool,
	t_pack: std::time::Instant,
	t_alloc: std::time::Instant,
	t_run: std::time::Instant,
	quant_secs: f64,
}

pub struct GpuDevice {
	#[allow(dead_code)]
	entry: ash::Entry,
	instance: ash::Instance,
	#[allow(dead_code)]
	physical: vk::PhysicalDevice,
	device: ash::Device,
	queue: vk::Queue,
	/// Compute queue family, and the transfer queue the streaming path
	/// uploads on. `xfer_queue` is `None` when the device exposes no
	/// transfer-only family; the copies then go to the compute queue, still
	/// asynchronously, just without a second engine behind them.
	qfamily: u32,
	xfer_queue: Option<vk::Queue>,
	xfer_family: u32,
	xfer_pool: vk::CommandPool,
	xfer_fence: vk::Fence,
	/// Signaled by a streamed upload, waited on by the dispatches that
	/// read it (`stream::Follow`).
	xfer_sem: vk::Semaphore,
	cmd_pool: vk::CommandPool,
	fence: vk::Fence,
	/// Fence of prefill batches in bank 1 (`matmul_batch_seq_begin_bank`):
	/// the two-half pipeline keeps one batch per bank in flight.
	fence2: vk::Fence,
	desc_pool: vk::DescriptorPool,
	desc_layout: vk::DescriptorSetLayout,
	pipeline_layout: vk::PipelineLayout,
	pipeline_q4k: vk::Pipeline,
	pipeline_q6k: vk::Pipeline,
	pipeline_q8_0: vk::Pipeline,
	pipeline_f16: vk::Pipeline,
	pipeline_f32: vk::Pipeline,
	pipeline_q4k_dp4a: Option<vk::Pipeline>,
	/// Decode GEMVs that read **f32** activations and do float math
	/// (`shaders/q4k_f32.glsl`, `q6k_f32.glsl`): ~2.4x / ~1.9x the q8_K
	/// integer kernels above and far closer to the exact product. Used by
	/// the per-matmul paths (`gemv_for`); the fused blocks still feed q8_K
	/// to `pipeline_q4k` / `pipeline_q6k`.
	pipeline_q4k_f32: vk::Pipeline,
	pipeline_q6k_f32: vk::Pipeline,
	/// The same float GEMVs over up to `SEQ_GEMV_NS` positions per weight
	/// read (`q4k_f32_seq.glsl`, `q6k_f32_seq.glsl`): verify batches.
	pipeline_q4k_f32_seq: vk::Pipeline,
	pipeline_q6k_f32_seq: vk::Pipeline,
	/// `false` under `QWEN_GPU_NOSEQGEMV=1`: verify batches go back to one
	/// position per `grid.y` (`gemv_dispatch`).
	seq_gemv: bool,
	/// `false` under `QWEN_GPU_Q8GEMV=1`: the per-matmul GEMV goes back to
	/// the q8_K kernels, for an A/B on one build.
	f32_gemv: bool,
	/// Tiled Q4_K GEMM for prefill (`shaders/gemm_q4k.glsl`). Distinct
	/// from `pipeline_q4k`, which is a batched GEMV: its grid contract is
	/// `gemm_grid`, not `grid_x`.
	pipeline_gemm_q4k: vk::Pipeline,
	/// `shaders/gemm_q4k_dp4a.glsl`: the same GEMM on integer dot
	/// products, ~2x `pipeline_gemm_q4k`. Present when the device has
	/// dp4a and passed its self-check; `QWEN_GPU_GEMM_FLOAT=1` is the A/B.
	pipeline_gemm_q4k_dp4a: Option<vk::Pipeline>,
	/// Tiled Q6_K GEMM for prefill (`shaders/gemm_q6k.glsl`). Same grid
	/// contract as `pipeline_gemm_q4k`; `FfnDown` is the weight that needs
	/// it, and it ran at 0.20x CPU on the GEMV kernel.
	pipeline_gemm_q6k: vk::Pipeline,
	/// `shaders/gemm_q6k_dp4a.glsl`, as `pipeline_gemm_q4k_dp4a` is to
	/// `pipeline_gemm_q4k`.
	pipeline_gemm_q6k_dp4a: Option<vk::Pipeline>,
	/// VRAM slots the prefill path streams non-resident weights through
	/// (`QWEN_GPU_STREAM=1`). Behind a `Mutex` because `matmul_batch_seq`
	/// takes `&self`: the manager is shared behind an `Arc` and its calls
	/// are serialized on the forward thread, so this never contends.
	stream: Mutex<stream::StreamPool>,
	/// Host-visible buffers `matmul_batch_seq` reuses instead of creating
	/// and destroying per call (see `pool::HostPool`). Same `&self` /
	/// serialized-caller reasoning as `stream`.
	host: Mutex<pool::HostPool>,
	/// `QWEN_GPU_NOGEMM=1` forces prefill back onto the batched-GEMV
	/// kernel, so the two can be A/B'd on the same build.
	no_gemm: bool,
	use_dp4a: bool,
	/// `VK_EXT_memory_budget` enabled (see `vram_budget`).
	has_budget_ext: bool,
	/// `VK_EXT_external_memory_host`, when enabled: the prefill split
	/// imports its output matrices and writes them in place (`hostimp`).
	host_import: Option<ash::ext::external_memory_host::Device>,
	mem_props: vk::PhysicalDeviceMemoryProperties,
	/// Set once a `VK_ERROR_DEVICE_LOST` is observed. After that every GPU
	/// matmul short-circuits to `None` so the caller can transparently fall
	/// back to the CPU path instead of faulting the whole process.
	lost: AtomicBool,
	llvmpipe: bool,
	weights: HashMap<(usize, WeightId), GpuWeight>,
	/// Recorded decode command buffers, one per job set, reused every
	/// token (`matmul_batch_begin`). A decode job's buffers are all
	/// per-weight and persistent, so the recording for a given list of
	/// weights never changes; re-recording it cost ~20 us per call, ~150
	/// calls per token. Cleared whenever `weights` changes.
	/// `QWEN_GPU_NOCMDCACHE=1` records per call again, for the A/B.
	/// Keyed by the job set and the batch size (`seq`): a verify batch of
	/// the same weights records a different grid and copy lengths.
	decode_cmds: Mutex<HashMap<(Vec<(usize, WeightId)>, u32), vk::CommandBuffer>>,
	cmd_cache: bool,
	/// Positions each weight's persistent decode buffers hold (`upload`):
	/// 1, or `1 + n_draft` when speculative decoding verifies a batch
	/// through the decode path (`set_decode_seq`).
	pub(crate) decode_seq: usize,
	/// Prefill attention kernel state (`attnp.rs`), created on first use.
	attnp: Mutex<Option<attnp::AttnRes>>,
	// ---- L2/L3 GPU-resident fused-layer (full-attn) plumbing ----
	aux_layout: vk::DescriptorSetLayout,
	aux_pipeline_layout: vk::PipelineLayout,
	pl_rms_sum: vk::Pipeline,
	pl_rms_apply: vk::Pipeline,
	pl_prep_q: vk::Pipeline,
	pl_prep_k: vk::Pipeline,
	pl_attn: vk::Pipeline,
	pl_silu: vk::Pipeline,
	pl_residual: vk::Pipeline,
	pl_q8: vk::Pipeline,
	// ---- L2/L3 SSM fused-block (gpu-resident) pipelines ----
	pl_conv1d: vk::Pipeline,
	pl_l2norm: vk::Pipeline,
	pl_ssm_beta: vk::Pipeline,
	pl_ssm_gate: vk::Pipeline,
	pl_delta_net: vk::Pipeline,
	pl_ssm_rms_o: vk::Pipeline,
	max_ctx: usize,
	block_res: Mutex<HashMap<usize, BlockRes>>,
	ssm_res: Mutex<HashMap<usize, SsmRes>>,
	/// Layers whose fused-block resources could not be allocated (VRAM is
	/// already ~96 % committed to weights at the recommended 22-layer
	/// setting). Recorded so the allocation is attempted once and the layer
	/// then stays on the CPU instead of retrying (and leaking) per token.
	block_oom: Mutex<std::collections::HashSet<usize>>,
	/// Shared device-local blocks the per-weight `x` / `y` / params buffers
	/// are bound into (see `create_small_gpu_buffer`).
	small_arena: Mutex<SmallArena>,
}

/// Bump allocator over a few large device-local allocations, for the
/// hundreds of sub-megabyte buffers the decode split creates (three per
/// weight). Nothing in it is freed before the device is dropped.
#[derive(Default)]
pub(crate) struct SmallArena {
	/// Every block, freed in `Drop`.
	blocks: Vec<vk::DeviceMemory>,
	/// The block being filled: `(memory, memory type, bytes used)`.
	cur: Option<(vk::DeviceMemory, u32, u64)>,
}

/// Per-layer resources for the GPU-resident fused full-attention block. Buffers
/// are allocated once (lazily) and reused across tokens; only the tiny norm
/// weight vectors are re-uploaded each call.
pub(crate) struct BlockRes {
	aux_pool: vk::DescriptorPool,
	param_bufs: Vec<(
		vk::Buffer,
		vk::DeviceMemory,
		vk::Buffer,
		vk::DeviceMemory,
		MappedPtr,
	)>,
	attn_norm: vk::Buffer,
	attn_norm_mem: vk::DeviceMemory,
	attn_norm_stg: vk::Buffer,
	attn_norm_stg_mem: vk::DeviceMemory,
	attn_norm_ptr: MappedPtr,
	post_norm: vk::Buffer,
	post_norm_mem: vk::DeviceMemory,
	post_norm_stg: vk::Buffer,
	post_norm_stg_mem: vk::DeviceMemory,
	post_norm_ptr: MappedPtr,
	q_norm: vk::Buffer,
	q_norm_mem: vk::DeviceMemory,
	q_norm_stg: vk::Buffer,
	q_norm_stg_mem: vk::DeviceMemory,
	q_norm_ptr: MappedPtr,
	k_norm: vk::Buffer,
	k_norm_mem: vk::DeviceMemory,
	k_norm_stg: vk::Buffer,
	k_norm_stg_mem: vk::DeviceMemory,
	k_norm_ptr: MappedPtr,
	x_in: vk::Buffer,
	x_in_mem: vk::DeviceMemory,
	x_in_stg: vk::Buffer,
	x_in_stg_mem: vk::DeviceMemory,
	x_in_ptr: MappedPtr,
	x_out: vk::Buffer,
	x_out_mem: vk::DeviceMemory,
	rb: vk::Buffer,
	rb_mem: vk::DeviceMemory,
	rb_ptr: MappedPtr,
	norm: vk::Buffer,
	norm_mem: vk::DeviceMemory,
	qfull: vk::Buffer,
	qfull_mem: vk::DeviceMemory,
	k: vk::Buffer,
	k_mem: vk::DeviceMemory,
	v: vk::Buffer,
	v_mem: vk::DeviceMemory,
	qn: vk::Buffer,
	qn_mem: vk::DeviceMemory,
	kn: vk::Buffer,
	kn_mem: vk::DeviceMemory,
	gate: vk::Buffer,
	gate_mem: vk::DeviceMemory,
	aout: vk::Buffer,
	aout_mem: vk::DeviceMemory,
	waout: vk::Buffer,
	waout_mem: vk::DeviceMemory,
	after_attn: vk::Buffer,
	after_attn_mem: vk::DeviceMemory,
	post: vk::Buffer,
	post_mem: vk::DeviceMemory,
	up: vk::Buffer,
	up_mem: vk::DeviceMemory,
	fg: vk::Buffer,
	fg_mem: vk::DeviceMemory,
	h: vk::Buffer,
	h_mem: vk::DeviceMemory,
	ffn_out: vk::Buffer,
	ffn_out_mem: vk::DeviceMemory,
	inv: vk::Buffer,
	inv_mem: vk::DeviceMemory,
	kcache: vk::Buffer,
	kcache_mem: vk::DeviceMemory,
	vcache: vk::Buffer,
	vcache_mem: vk::DeviceMemory,
	scratch: vk::Buffer,
	scratch_mem: vk::DeviceMemory,
}

/// Per-layer resources for the GPU-resident fused SSM block. Buffers are
/// allocated once (lazily, on first `ssm_block` call) and reused across
/// tokens; only the tiny norm/SSM params are re-uploaded each call. The SSM
/// recurrent state (`st_buf`) and conv window (`conv_state`) persist across
/// tokens on the GPU so `x` stays resident for the whole layer.
pub(crate) struct SsmRes {
	aux_pool: vk::DescriptorPool,
	param_bufs: Vec<(
		vk::Buffer,
		vk::DeviceMemory,
		vk::Buffer,
		vk::DeviceMemory,
		MappedPtr,
	)>,
	conv_state: vk::Buffer,
	conv_state_mem: vk::DeviceMemory,
	st_buf: vk::Buffer,
	st_mem: vk::DeviceMemory,
	conv_w: vk::Buffer,
	conv_w_mem: vk::DeviceMemory,
	conv_w_stg: vk::Buffer,
	conv_w_stg_mem: vk::DeviceMemory,
	conv_w_ptr: MappedPtr,
	ssm_norm_buf: vk::Buffer,
	ssm_norm_mem: vk::DeviceMemory,
	ssm_norm_stg: vk::Buffer,
	ssm_norm_stg_mem: vk::DeviceMemory,
	ssm_norm_ptr: MappedPtr,
	ssm_dt_buf: vk::Buffer,
	ssm_dt_mem: vk::DeviceMemory,
	ssm_dt_stg: vk::Buffer,
	ssm_dt_stg_mem: vk::DeviceMemory,
	ssm_dt_ptr: MappedPtr,
	ssm_a_buf: vk::Buffer,
	ssm_a_mem: vk::DeviceMemory,
	ssm_a_stg: vk::Buffer,
	ssm_a_stg_mem: vk::DeviceMemory,
	ssm_a_ptr: MappedPtr,
	bg: vk::Buffer,
	bg_mem: vk::DeviceMemory,
	conv_out: vk::Buffer,
	conv_out_mem: vk::DeviceMemory,
	/// SSM q/k/v. These must NOT reuse the shared `BlockRes` q/k/v: those are
	/// sized for attention (`n_head*head_dim`), while the SSM needs
	/// `key_dim = ssm_state*n_group` and `value_dim = ssm_state*dt_rank`,
	/// which are larger on the real model (a copy into them overflows).
	qn: vk::Buffer,
	qn_mem: vk::DeviceMemory,
	kn: vk::Buffer,
	kn_mem: vk::DeviceMemory,
	v: vk::Buffer,
	v_mem: vk::DeviceMemory,
	o: vk::Buffer,
	o_mem: vk::DeviceMemory,
	no: vk::Buffer,
	no_mem: vk::DeviceMemory,
	init: std::sync::atomic::AtomicBool,
}

impl GpuDevice {
	/// Whether the active device is the software `llvmpipe` ICD.
	pub(crate) fn is_llvmpipe(&self) -> bool {
		self.llvmpipe
	}
}

impl Drop for GpuDevice {
	fn drop(&mut self) {
		// A device dropped after a fence timeout (marked lost, not lost to
		// the driver) may still be running that work, and its buffers must
		// outlive it. On a truly lost device this returns at once.
		unsafe { self.device.device_wait_idle().ok() };
		self.drop_host_imports();
		self.free_stream_slots();
		self.free_host_pool();
		unsafe {
			for w in self.weights.values() {
				self.device.destroy_buffer(w.weight_buf, None);
				self.device.free_memory(w.weight_mem, None);
				// `x_mem` / `y_mem` / `params_mem` are null when the
				// buffer lives in `small_arena` (freed below), and they
				// are device-local, so never mapped.
				self.device.destroy_buffer(w.x_buf, None);
				self.device.free_memory(w.x_mem, None);
				self.device.destroy_buffer(w.x_stg_buf, None);
				self.device.unmap_memory(w.x_stg_mem);
				self.device.free_memory(w.x_stg_mem, None);
				self.device.destroy_buffer(w.y_buf, None);
				self.device.free_memory(w.y_mem, None);
				self.device.destroy_buffer(w.rb_buf, None);
				self.device.unmap_memory(w.rb_mem);
				self.device.free_memory(w.rb_mem, None);
				self.device.destroy_buffer(w.params_buf, None);
				self.device.free_memory(w.params_mem, None);
				self.device.destroy_buffer(w.params_stg_buf, None);
				self.device.unmap_memory(w.params_stg_mem);
				self.device.free_memory(w.params_stg_mem, None);
			}
			for &m in &self.small_arena.lock().unwrap().blocks {
				self.device.free_memory(m, None);
			}
			self.device.destroy_descriptor_pool(self.desc_pool, None);
			// Free per-layer fused-block resources.
			for br in self.block_res.lock().unwrap().values() {
				for (b, m, sb, sm) in [
					(
						br.attn_norm,
						br.attn_norm_mem,
						br.attn_norm_stg,
						br.attn_norm_stg_mem,
					),
					(
						br.post_norm,
						br.post_norm_mem,
						br.post_norm_stg,
						br.post_norm_stg_mem,
					),
					(br.q_norm, br.q_norm_mem, br.q_norm_stg, br.q_norm_stg_mem),
					(br.k_norm, br.k_norm_mem, br.k_norm_stg, br.k_norm_stg_mem),
					(br.x_in, br.x_in_mem, br.x_in_stg, br.x_in_stg_mem),
				] {
					self.device.destroy_buffer(b, None);
					self.device.free_memory(m, None);
					self.device.destroy_buffer(sb, None);
					self.device.free_memory(sm, None);
				}
				for (b, m, sb, sm, _p) in &br.param_bufs {
					self.device.destroy_buffer(*b, None);
					self.device.free_memory(*m, None);
					self.device.destroy_buffer(*sb, None);
					self.device.unmap_memory(*sm);
					self.device.free_memory(*sm, None);
				}
				for (b, m) in [
					(br.x_out, br.x_out_mem),
					(br.rb, br.rb_mem),
					(br.norm, br.norm_mem),
					(br.qfull, br.qfull_mem),
					(br.k, br.k_mem),
					(br.v, br.v_mem),
					(br.qn, br.qn_mem),
					(br.kn, br.kn_mem),
					(br.gate, br.gate_mem),
					(br.aout, br.aout_mem),
					(br.waout, br.waout_mem),
					(br.after_attn, br.after_attn_mem),
					(br.post, br.post_mem),
					(br.up, br.up_mem),
					(br.fg, br.fg_mem),
					(br.h, br.h_mem),
					(br.ffn_out, br.ffn_out_mem),
					(br.inv, br.inv_mem),
					(br.kcache, br.kcache_mem),
					(br.vcache, br.vcache_mem),
					(br.scratch, br.scratch_mem),
				] {
					self.device.destroy_buffer(b, None);
					self.device.free_memory(m, None);
				}
				self.device.destroy_descriptor_pool(br.aux_pool, None);
			}
			self.device.destroy_pipeline(self.pl_rms_sum, None);
			self.device.destroy_pipeline(self.pl_rms_apply, None);
			self.device.destroy_pipeline(self.pl_prep_q, None);
			self.device.destroy_pipeline(self.pl_prep_k, None);
			self.device.destroy_pipeline(self.pl_attn, None);
			self.device.destroy_pipeline(self.pl_silu, None);
			self.device.destroy_pipeline(self.pl_residual, None);
			self.device.destroy_pipeline(self.pl_q8, None);
			self.device.destroy_pipeline(self.pl_conv1d, None);
			self.device.destroy_pipeline(self.pl_l2norm, None);
			self.device.destroy_pipeline(self.pl_ssm_beta, None);
			self.device.destroy_pipeline(self.pl_ssm_gate, None);
			self.device.destroy_pipeline(self.pl_delta_net, None);
			self.device.destroy_pipeline(self.pl_ssm_rms_o, None);
			// Free per-layer SSM-block resources.
			for sr in self.ssm_res.lock().unwrap().values() {
				for (b, m) in [
					(sr.conv_state, sr.conv_state_mem),
					(sr.st_buf, sr.st_mem),
					(sr.conv_w, sr.conv_w_mem),
					(sr.ssm_norm_buf, sr.ssm_norm_mem),
					(sr.ssm_dt_buf, sr.ssm_dt_mem),
					(sr.ssm_a_buf, sr.ssm_a_mem),
					(sr.bg, sr.bg_mem),
					(sr.conv_out, sr.conv_out_mem),
					(sr.qn, sr.qn_mem),
					(sr.kn, sr.kn_mem),
					(sr.v, sr.v_mem),
					(sr.o, sr.o_mem),
					(sr.no, sr.no_mem),
					(sr.conv_w_stg, sr.conv_w_stg_mem),
					(sr.ssm_norm_stg, sr.ssm_norm_stg_mem),
					(sr.ssm_dt_stg, sr.ssm_dt_stg_mem),
					(sr.ssm_a_stg, sr.ssm_a_stg_mem),
				] {
					self.device.destroy_buffer(b, None);
					self.device.free_memory(m, None);
				}
				for (b, m, sb, sm, _p) in &sr.param_bufs {
					self.device.destroy_buffer(*b, None);
					self.device.free_memory(*m, None);
					self.device.destroy_buffer(*sb, None);
					self.device.unmap_memory(*sm);
					self.device.free_memory(*sm, None);
				}
				self.device.destroy_descriptor_pool(sr.aux_pool, None);
			}
			self.device
				.destroy_descriptor_set_layout(self.aux_layout, None);
			self.device
				.destroy_pipeline_layout(self.aux_pipeline_layout, None);
			self.device.destroy_command_pool(self.cmd_pool, None);
			self.device.destroy_command_pool(self.xfer_pool, None);
			self.device.destroy_semaphore(self.xfer_sem, None);
			self.device.destroy_device(None);
			self.instance.destroy_instance(None);
		}
	}
}

pub(crate) fn create_sm(device: &ash::Device, code: &[u32]) -> Result<vk::ShaderModule, String> {
	let info = vk::ShaderModuleCreateInfo {
		code_size: code.len() * 4,
		p_code: code.as_ptr(),
		..Default::default()
	};
	unsafe { device.create_shader_module(&info, None) }.map_err(|e| format!("shader module: {e}"))
}

/// Set at init on a device whose subgroup width varies (Intel: 8-32). The
/// kernels whose row reductions are `subgroupAdd` over a 32-lane
/// `local_size_x` (`q4k_f32`, `q6k_f32`) then require 32-wide subgroups:
/// left to itself the Intel compiler picks SIMD16/8 and each row sums only
/// part of its lanes. Only those kernels: forcing SIMD32 on everything
/// made the iGPU's prefill GEMM ~8x slower.
pub(crate) static REQUIRE_SUBGROUP_32: AtomicBool = AtomicBool::new(false);

pub(crate) fn compute_pipeline(
	device: &ash::Device,
	layout: vk::PipelineLayout,
	module: vk::ShaderModule,
	entry: &CStr,
) -> Result<vk::Pipeline, String> {
	compute_pipeline_sg(device, layout, module, entry, false)
}

/// [`compute_pipeline`] for a kernel that assumes 32-lane subgroups.
pub(crate) fn compute_pipeline_sg32(
	device: &ash::Device,
	layout: vk::PipelineLayout,
	module: vk::ShaderModule,
	entry: &CStr,
) -> Result<vk::Pipeline, String> {
	compute_pipeline_sg(device, layout, module, entry, true)
}

fn compute_pipeline_sg(
	device: &ash::Device,
	layout: vk::PipelineLayout,
	module: vk::ShaderModule,
	entry: &CStr,
	sg32: bool,
) -> Result<vk::Pipeline, String> {
	let req = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo {
		required_subgroup_size: 32,
		..Default::default()
	};
	let stage = vk::PipelineShaderStageCreateInfo {
		stage: vk::ShaderStageFlags::COMPUTE,
		module,
		p_name: entry.as_ptr(),
		p_next: if sg32 && REQUIRE_SUBGROUP_32.load(Ordering::Relaxed) {
			&req as *const _ as *const c_void
		} else {
			ptr::null()
		},
		..Default::default()
	};
	let ci = vk::ComputePipelineCreateInfo {
		stage,
		layout,
		..Default::default()
	};
	let mut v = unsafe {
		device
			.create_compute_pipelines(vk::PipelineCache::default(), &[ci], None)
			.map_err(|(p, e)| {
				let _ = p;
				format!("compute pipeline: {e:?}")
			})?
	};
	Ok(v.pop().unwrap())
}

pub(crate) fn find_memory_type(
	props: &vk::PhysicalDeviceMemoryProperties,
	req: vk::MemoryPropertyFlags,
	type_bits: u32,
) -> Option<u32> {
	for i in 0..props.memory_type_count {
		let t = props.memory_types[i as usize];
		if (type_bits & (1 << i)) != 0 && t.property_flags.contains(req) {
			return Some(i);
		}
	}
	None
}

mod attn;
mod attnp;
mod bench;
mod device;
mod hostimp;
mod matmul;
mod pool;
mod stream;
pub use attnp::{ATTN_HD, ATTN_HPG};
mod ditattn;
mod ditblk;
mod flashattn;
mod hcpre;
mod scan;
mod ssm;
mod visattn;
pub use ditattn::DitAttn;
pub use ditblk::{DitBlk, DitMid};
pub use flashattn::FlashAttn;
pub use hcpre::{HcPre, HC_MAXN};
pub use hostimp::{host_import_budget, host_imported};
pub use scan::DeltaScan;
pub use visattn::VisAttn;
mod mlpsplit;
pub use mlpsplit::MlpSplit;
mod dense;
mod experts;
pub use experts::{Experts, EXP_SLOT};
mod gemv;
pub use dense::{Dense, Region};
pub use gemv::{Gemv, GemvMat, DELTA_STATE, DELTA_WIN, GEMV_NP, GEMV_WAVE, GEMV_X, GEMV_Y};
#[cfg(test)]
mod tests;
