/// Identifies one of a layer's large matmul weight matrices.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum WeightId {
	/// full-attention Q (already fused q+gate)
	Wq,
	/// full-attention K
	Wk,
	/// full-attention V
	Wv,
	/// full-attention output projection
	Wo,
	/// SSM input projection (qkv)
	Wqkv,
	/// SSM gate projection
	WqkvGate,
	/// SSM output projection
	SsmOut,
	/// SSM beta linear (Hv x n_embd, f32)
	WssmBeta,
	/// SSM alpha linear (Hv x n_embd, f32)
	WssmAlpha,
	/// FFN up
	FfnUp,
	/// FFN gate
	FfnGate,
	/// FFN down
	FfnDown,
	/// MTP head input projection (`nextn.eh_proj`, `2 * n_embd -> n_embd`)
	EhProj,
}

/// Largest batch treated as a speculative-decoding **verify** pass rather than
/// prefill: it keeps the decode row split (`prefill_split_rows`) and the exact
/// f32-activation `ssm_beta`/`ssm_alpha` (`exact_decode`), so its logits match
/// what token-by-token decode would have produced.
pub const SPEC_MAX_SEQ: usize = 8;

/// Shortest prefill chunk that splits its resident rows onto the GPU. Each
/// split call carries ~0.4 ms of serial host cost (q8_K pack, descriptor pool,
/// submit, fence wait, copy-out), ~100 ms over a chunk's 256 calls: at 13
/// tokens that ate the whole GPU share (402-446 ms CPU-only vs 419-508 split,
/// interleaved), at 23 it was already -20 %.
#[cfg(feature = "vulkan")]
const SMALL_SPLIT_MIN_SEQ: usize = 16;

/// GPU row fraction of a short prefill chunk's split (`SMALL_SPLIT_MIN_SEQ <=
/// seq < prefill_min_seq`), capped by the resident rows (~34 %). Interpolated from
/// `gpubench --crossover` on one FFN block, resident, whole matrix, as
/// `CPU ms / (CPU ms + GPU ms)`: the share at which both sides finish together:
///
/// | seq | 8 | 12 | 16 | 20 | 24 | 32 | 48 | 64 |
/// |---|---|---|---|---|---|---|---|---|
/// | CPU ms | 3.44 | 4.59 | 5.51 | 7.91 | 8.43 | 10.44 | 15.57 | 19.78 |
/// | GPU ms | 12.87 | 13.91 | 14.53 | 13.99 | 14.12 | 14.99 | 19.06 | 20.21 |
///
/// The GPU column is nearly flat below 32 tokens: the tiled GEMM pays for a
/// whole 64-token tile's staging whatever `seq` is. The table is effectively the
/// Q4_K rate (the block is two Q4_K matrices to one Q6_K); the Q6_K GEMM stages
/// 6-bit weights into int8, which at a nearly empty tile costs more than the
/// arithmetic, and it lagged the CPU by ~1.4 ms per `FfnDown` call at the Q4_K
/// share. It gets 0.65x, where its fence wait dropped to the ~0.2 ms floor.
/// `QWEN_PREFILL_SMALL_SCALE` / `QWEN_PREFILL_SMALL_Q6K` override the two
/// factors (the host's run-to-run drift, ~35 %, is too large for an
/// end-to-end sweep to settle them; balance them on `QWEN_GPU_TIMING=1`'s
/// `[PSPLIT] ... end+copy` instead).
///
/// With the dp4a GEMMs (`dp4a`) the same measurement gives
///
/// | seq | 12 | 16 | 20 | 24 | 32 | 48 | 64 |
/// |---|---|---|---|---|---|---|---|
/// | CPU ms | 4.25 | 5.78 | 7.65 | 8.05 | 10.66 | 16.14 | 20.25 |
/// | GPU ms | 6.75 | 7.09 | 7.32 | 7.46 | 8.12 | 9.77 | 10.88 |
///
/// so the balanced share is above the resident cap at every length: these
/// chunks use all resident rows. The dp4a Q6_K GEMM no longer lags (that
/// was the float kernel's staging cost), so it gets no discount there.
#[cfg(feature = "vulkan")]
fn small_chunk_frac(seq: usize, q6k: bool, dp4a: bool) -> f64 {
	const PTS_FLOAT: [(f64, f64); 7] = [
		(8.0, 0.21),
		(12.0, 0.25),
		(16.0, 0.27),
		(20.0, 0.36),
		(32.0, 0.41),
		(48.0, 0.45),
		(64.0, 0.49),
	];
	const PTS_DP4A: [(f64, f64); 7] = [
		(12.0, 0.39),
		(16.0, 0.45),
		(20.0, 0.51),
		(24.0, 0.52),
		(32.0, 0.57),
		(48.0, 0.62),
		(64.0, 0.65),
	];
	let pts = if dp4a { &PTS_DP4A } else { &PTS_FLOAT };
	static SCALE: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
	let scale = *SCALE.get_or_init(|| {
		std::env::var("QWEN_PREFILL_SMALL_SCALE")
			.ok()
			.and_then(|v| v.trim().parse().ok())
			.unwrap_or(1.0)
	});
	let s = seq as f64;
	let f = pts
		.windows(2)
		.find(|w| s <= w[1].0)
		.map(|w| w[0].1 + (w[1].1 - w[0].1) * ((s - w[0].0) / (w[1].0 - w[0].0)).clamp(0.0, 1.0))
		.unwrap_or(pts[pts.len() - 1].1);
	static S6: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
	let s6 = *S6.get_or_init(|| {
		std::env::var("QWEN_PREFILL_SMALL_Q6K")
			.ok()
			.and_then(|v| v.trim().parse().ok())
			.unwrap_or(f64::NAN)
	});
	let s6 = if s6.is_nan() {
		if dp4a {
			1.0
		} else {
			0.65
		}
	} else {
		s6
	};
	(f * scale * if q6k { s6 } else { 1.0 }).clamp(0.0, 1.0)
}

/// Per-matmul correction to the dp4a prefill split fraction
/// (`prefill_split_rows`), added to the quant type's default on chunks of
/// `prefill_min_seq` tokens and more (not the short-chunk split). One
/// fraction per type left some matrices GPU-long and others CPU-long; the table is static so prefill output does not depend on run
/// history. `QWEN_PREFILL_DFRAC_<ID>` (e.g. `QWEN_PREFILL_DFRAC_FFNDOWN`)
/// overrides one entry; `FfnUp` and `FfnGate` share one batch and so one
/// entry (`FFNUP`), as do `Wq` / `Wk` / `Wv` (`WQ`).
#[cfg(feature = "vulkan")]
fn prefill_frac_delta(id: WeightId) -> f64 {
	let (key, dflt) = match id {
		WeightId::FfnUp | WeightId::FfnGate => ("FFNUP", 0.0),
		WeightId::FfnDown => ("FFNDOWN", 0.03),
		WeightId::Wqkv => ("WQKV", -0.03),
		WeightId::SsmOut => ("SSMOUT", -0.02),
		WeightId::Wq | WeightId::Wk | WeightId::Wv => ("WQ", 0.0),
		WeightId::Wo => ("WO", 0.0),
		_ => return 0.0,
	};
	static CELLS: std::sync::OnceLock<std::collections::HashMap<&'static str, f64>> =
		std::sync::OnceLock::new();
	let m = CELLS.get_or_init(|| {
		["FFNUP", "FFNDOWN", "WQKV", "SSMOUT", "WQ", "WO"]
			.into_iter()
			.filter_map(|k| {
				let v = std::env::var(format!("QWEN_PREFILL_DFRAC_{k}")).ok()?;
				Some((k, v.trim().parse().ok()?))
			})
			.collect()
	});
	m.get(key).copied().unwrap_or(dflt)
}

impl WeightId {
	/// Weights whose decode product is computed on the CPU with an **f32**
	/// activation (`quant::apply_linear_group_exact`), wherever the layer is
	/// placed: `ssm_beta` / `ssm_alpha`, 48 rows each. The SSM gates they feed
	/// are sensitive to q8_K activation error, and on the GPU their ~20 us of
	/// work costs a ~70 us round trip once per SSM layer. Never row-split.
	pub fn exact_decode(self) -> bool {
		matches!(self, WeightId::WssmBeta | WeightId::WssmAlpha)
	}
}

/// A single matrix product to fold into a batched GPU dispatch. `w`/`out_dim`/
/// `in_dim` are only used by the CPU fallback; on the GPU the weights already
/// live in VRAM and only `x` (the hidden-state slice) is shipped.
pub struct BatchJob<'a> {
	pub il: usize,
	pub id: WeightId,
	pub w: &'a quant::LinearWeight,
	pub x: &'a [f32],
	pub out_dim: usize,
	pub in_dim: usize,
}

/// A single batched (prefill, `seq > 1`) matrix product to fold into a GPU
/// dispatch. `seq` is the number of activation rows packed into `x`
/// (`x.len() == seq * in_dim`). `w`/`out_dim`/`in_dim` are only used by the CPU
/// fallback; on the GPU the weights already live in VRAM and only `x` is shipped.
#[derive(Clone)]
pub struct BatchJobSeq<'a> {
	pub il: usize,
	pub id: WeightId,
	pub w: &'a quant::LinearWeight,
	pub x: &'a [f32],
	pub seq: usize,
	pub out_dim: usize,
	pub in_dim: usize,
}

/// A prefill matmul group in flight on the GPU, started by
/// [`OffloadManager::matmul_q8_defer`] / [`OffloadManager::split_begin`] and
/// completed by [`OffloadManager::finish_deferred`] /
/// [`OffloadManager::split_finish`]. Every job shares one q8_K input.
pub struct Deferred<'a> {
	#[cfg(feature = "vulkan")]
	/// The batch in flight and the device it runs on: a device rebuilt in
	/// between (`OffloadManager::recover_gpu`) must not be asked to collect it.
	pending: Option<(
		std::sync::Arc<crate::gpu::GpuDevice>,
		crate::gpu::PendingSeq,
	)>,
	#[cfg(feature = "vulkan")]
	dsts: Vec<(usize, usize, usize, usize)>,
	/// Row-split jobs: the GPU took rows `[row0, row0 + g)`, the CPU takes
	/// the rest of each `out_dim` in the finish.
	#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
	parts: Vec<DeferPart<'a>>,
	/// Jobs run wholly on the CPU in the finish (`cpu_tail_ok`), outputs
	/// after `parts`'.
	#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
	tail: Vec<(&'a quant::LinearWeight, usize)>,
	#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
	seq: usize,
	#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
	in_dim: usize,
	#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
	q8: std::sync::Arc<Vec<quant::Q8K>>,
}

#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
struct DeferPart<'a> {
	w: &'a quant::LinearWeight,
	row0: usize,
	g: usize,
	out_dim: usize,
	out: Vec<f32>,
}

/// Everything the GPU-resident fused layer needs to know about one full-attention
/// layer (L2/L3 building block). Constructed by `Model` per call and passed to
/// `OffloadManager::full_attn_block`. Norm weights are uploaded each call (they
/// are tiny `f32` vectors); the big matmul weights are already in VRAM.
pub struct GpuBlockInfo<'a> {
	pub n_embd: usize,
	pub n_head: usize,
	pub n_head_kv: usize,
	pub head_dim: usize,
	pub n_ff: usize,
	pub eps: f32,
	pub pos: usize,
	pub half_nrot: usize,
	pub rope_freq_base: f32,
	pub rope_freq_scale: f32,
	pub rope_attn_factor: f32,
	pub sect_dims: f32,
	pub attn_norm: &'a [f32],
	pub post_norm: &'a [f32],
	pub q_norm: &'a [f32],
	pub k_norm: &'a [f32],
	// --- SSM-layer dims (only used by `ssm_block`; full-attention layers pass 0) ---
	pub ssm_state: usize,
	pub ssm_n_group: usize,
	pub ssm_dt_rank: usize,
	pub conv_kernel: usize,
	// --- SSM-layer GPU-resident params (only used by `ssm_block`; empty for
	// full-attention layers) ---
	pub ssm_conv1d: &'a [f32],
	pub ssm_dt: &'a [f32],
	pub ssm_a: &'a [f32],
	pub ssm_norm: &'a [f32],
}

/// Pack row-major `f16` (`u16`) weights `[out_dim x in_dim]` into the layout the
/// shader expects: each `u32` holds two consecutive `f16` values (low 16 bits =
/// even index, high 16 bits = odd index). `in_dim` must be even.
///
/// (Kept as a reference / for the documented SPIR-V fallback; the active Vulkan
/// path uses raw `ash` storage buffers directly, but the packing math is unit-tested.)
pub fn pack_f16_weights(w: &[u16], out_dim: usize, in_dim: usize) -> Vec<u32> {
	assert_eq!(w.len(), out_dim * in_dim, "weight length mismatch");
	assert!(
		in_dim.is_multiple_of(2),
		"in_dim must be even for f16 packing"
	);
	let half = in_dim / 2;
	let mut out = vec![0u32; out_dim * half];
	for o in 0..out_dim {
		let row = o * in_dim;
		let prow = o * half;
		for i in 0..half {
			let lo = w[row + 2 * i] as u32;
			let hi = w[row + 2 * i + 1] as u32;
			out[prow + i] = (hi << 16) | lo;
		}
	}
	out
}

/// CPU reference for the GPU matmul: identical to `quant::gemm_f16` (the shader
/// unpacks `f16`→`f32` and dots with `f32` `x`). Used by tests to prove the
/// packed-shader math equals the verified CPU kernel without needing a GPU.
pub fn reference_matmul(w: &[u16], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
	let mut y = vec![0.0f32; out_dim];
	for o in 0..out_dim {
		let mut acc = 0.0f32;
		let row = o * in_dim;
		for k in 0..in_dim {
			acc += quant::fp16_to_f32(w[row + k]) * x[k];
		}
		y[o] = acc;
	}
	y
}

pub struct OffloadManager {
	/// True only when the `vulkan` feature is on *and* a Vulkan adapter was
	/// successfully acquired at init.
	pub available: bool,
	/// `gpu_layers[i] == true` means layer `i`'s big matmuls are routed to the
	/// GPU (when `available`). The KV cache / SSM state for that layer always
	/// live on the CPU regardless.
	pub gpu_layers: Vec<bool>,
	n_gpu_layers: usize,
	/// Decode **row split** (`upload_split`): `(il, id) -> r` means the GPU
	/// holds rows `[0, r)` of that matrix (under the key `il + SPLIT_KEY`, so
	/// no full-matrix path can mistake it for a resident weight) and a decode
	/// matmul computes those rows there *while* the CPU computes `[r, out)`.
	split: std::collections::HashMap<(usize, WeightId), usize>,
	/// `split_layers[il]`: some matrix of layer `il` is split.
	split_layers: Vec<bool>,
	/// Forced prefill split fraction (`f64` bits, NaN = unset), replacing the
	/// default and its per-matmul delta in `prefill_split_rows`. Set only by
	/// the split tuner (`cli --tune-split`), which sweeps it at run time; the
	/// `QWEN_PREFILL_GPU_FRAC_*` overrides are read once.
	frac_force: std::sync::atomic::AtomicU64,
	/// The Vulkan device, read through [`Self::gpu`] (which the GPU tests
	/// use to drive `GpuDevice` directly). Replaceable behind `&self` because
	/// a lost device is rebuilt while the manager is shared
	/// ([`Self::recover_gpu`]); each call holds its own `Arc`, so a call still
	/// in flight on the old device finishes (on the CPU) before it is freed.
	#[cfg(feature = "vulkan")]
	gpu: std::sync::RwLock<Option<std::sync::Arc<crate::gpu::GpuDevice>>>,
	/// Every weight on the device, for uploading again after a rebuild.
	#[cfg(feature = "vulkan")]
	uploads: Vec<Upload>,
	/// `set_decode_seq`'s value, for a rebuilt device.
	#[cfg(feature = "vulkan")]
	decode_seq_n: usize,
	#[cfg(feature = "vulkan")]
	recovery: std::sync::Mutex<Recovery>,
	/// The next prefill matmul's streamed row ranges, staged by
	/// [`Self::prefetch_hint`] and uploaded while the current call runs.
	#[cfg(feature = "vulkan")]
	next: std::sync::Mutex<Option<NextUploads>>,
}

/// `(stream key, id, bytes address, length, kind)` per streamed range of a
/// hinted call, and the owner of those bytes.
#[cfg(feature = "vulkan")]
type NextUploads = (
	Vec<(usize, WeightId, usize, usize, crate::gpu::GpuKind)>,
	std::sync::Arc<dyn std::any::Any + Send + Sync>,
);

/// The prefill matmul a caller will make next, for
/// [`OffloadManager::prefetch_hint`]: the same arguments as the call itself.
pub enum NextCall<'a> {
	/// A split group: `matmul_seq_group_q8` / `matmul_batch_seq`, jobs
	/// `(id, weight, out_dim)`.
	Split(&'a [(WeightId, &'a quant::LinearWeight, usize)]),
	/// [`OffloadManager::matmul_q8_defer`]: `(id, weight, out_dim)`.
	Defer(WeightId, &'a quant::LinearWeight, usize),
	/// [`OffloadManager::matmul_q8_defer_rows`]: `(id, weight, row0, n)`.
	DeferRows(WeightId, &'a quant::LinearWeight, usize, usize),
}

/// One weight a rebuilt device must hold again: its device key (`il`, or
/// `il + SPLIT_KEY` for a decode-split row prefix) and its uploaded shape.
#[cfg(feature = "vulkan")]
struct Upload {
	key: usize,
	id: WeightId,
	ty: quant::GgmlType,
	out_dim: usize,
	in_dim: usize,
}

#[cfg(feature = "vulkan")]
#[derive(Default)]
struct Recovery {
	last: Option<std::time::Instant>,
}

/// Bytes of an `out_dim x in_dim` matrix of type `ty` in its GGUF form (0 for
/// a type the GPU does not take).
#[cfg(feature = "vulkan")]
fn quant_bytes(ty: quant::GgmlType, out_dim: usize, in_dim: usize) -> usize {
	let n = out_dim * in_dim;
	match ty {
		quant::GgmlType::Q4_K | quant::GgmlType::Q4_K_M => n / 256 * 144,
		quant::GgmlType::Q6_K => n / 256 * 210,
		quant::GgmlType::Q8_0 => n / 32 * 34,
		quant::GgmlType::F16 => n * 2,
		quant::GgmlType::F32 => n * 4,
		_ => 0,
	}
}

/// Time between attempts to rebuild a lost Vulkan device
/// (`QWEN_GPU_RECOVER_SECS`, default 15; `0` never rebuilds, the old
/// behavior). An attempt that finds the VRAM still taken costs a device init.
#[cfg(feature = "vulkan")]
fn recover_interval() -> Option<std::time::Duration> {
	static S: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
	let s = *S.get_or_init(|| {
		std::env::var("QWEN_GPU_RECOVER_SECS")
			.ok()
			.and_then(|v| v.trim().parse().ok())
			.unwrap_or(15)
	});
	(s > 0).then(|| std::time::Duration::from_secs(s))
}

impl OffloadManager {
	/// Build a manager that offloads the *last* `n_gpu_layers` of
	/// `n_layer_main` layers. With the `vulkan` feature it probes for a Vulkan
	/// adapter (via raw `ash`); on failure (or without the feature)
	/// `available` is `false` and everything runs on the CPU.
	pub fn new(n_gpu_layers: usize, n_layer_main: usize) -> OffloadManager {
		let ng = n_gpu_layers.min(n_layer_main);
		let mut gpu_layers = vec![false; n_layer_main];
		for i in (n_layer_main - ng)..n_layer_main {
			gpu_layers[i] = true;
		}
		#[cfg_attr(not(feature = "vulkan"), allow(unused_mut))]
		let mut mgr = OffloadManager {
			available: false,
			gpu_layers,
			n_gpu_layers: ng,
			split: std::collections::HashMap::new(),
			split_layers: vec![false; n_layer_main],
			frac_force: std::sync::atomic::AtomicU64::new(f64::NAN.to_bits()),
			#[cfg(feature = "vulkan")]
			gpu: std::sync::RwLock::new(None),
			#[cfg(feature = "vulkan")]
			uploads: Vec::new(),
			#[cfg(feature = "vulkan")]
			decode_seq_n: 1,
			#[cfg(feature = "vulkan")]
			recovery: std::sync::Mutex::new(Recovery::default()),
			#[cfg(feature = "vulkan")]
			next: std::sync::Mutex::new(None),
		};
		#[cfg(feature = "vulkan")]
		{
			match crate::gpu::GpuDevice::init() {
				Ok(g) => {
					mgr.gpu = std::sync::RwLock::new(Some(std::sync::Arc::new(g)));
					mgr.available = true;
				}
				Err(e) => {
					eprintln!(
						"offload: Vulkan (raw ash) device init failed ({e}); falling back to CPU for all layers"
					);
					mgr.available = false;
				}
			}
		}
		mgr
	}

	/// The Vulkan device, if there is one. It may be lost: its calls then
	/// return `None` and the caller runs on the CPU.
	#[cfg(feature = "vulkan")]
	pub(crate) fn gpu(&self) -> Option<std::sync::Arc<crate::gpu::GpuDevice>> {
		self.gpu.read().unwrap_or_else(|e| e.into_inner()).clone()
	}

	/// The device for uploads, which happen at load, before the manager is
	/// shared (so its `Arc` is unique).
	#[cfg(feature = "vulkan")]
	fn gpu_mut(&mut self) -> Option<&mut crate::gpu::GpuDevice> {
		self.gpu
			.get_mut()
			.unwrap_or_else(|e| e.into_inner())
			.as_mut()
			.and_then(std::sync::Arc::get_mut)
	}

	/// Whether the GPU path is down: the device was lost and has not been
	/// rebuilt yet. `false` when there was no GPU to begin with.
	pub fn gpu_lost(&self) -> bool {
		#[cfg(feature = "vulkan")]
		{
			self.available && self.gpu().is_none_or(|g| g.is_lost())
		}
		#[cfg(not(feature = "vulkan"))]
		{
			false
		}
	}

	/// Rebuild a lost Vulkan device and upload every weight it held again,
	/// so the GPU path comes back instead of staying off for the rest of the
	/// run. Another application claiming VRAM can make the driver lose the
	/// device, or stall a submission past the fence timeout (treated the
	/// same). `src(il, id)` returns the whole matrix's bytes and type, as
	/// `upload` / `upload_split` were given them.
	///
	/// A no-op while the device is healthy. Attempts are at least
	/// `QWEN_GPU_RECOVER_SECS` apart; one that finds too little VRAM free
	/// (the other application still holds it) gives up before uploading, and
	/// a later one tries again. Until then every matmul runs on the CPU.
	/// Returns whether the GPU is usable.
	pub fn recover_gpu(
		&self,
		src: &dyn Fn(usize, WeightId) -> Option<(Vec<u8>, quant::GgmlType)>,
	) -> bool {
		#[cfg(feature = "vulkan")]
		{
			if !self.available {
				return false;
			}
			if self.gpu().is_some_and(|g| !g.is_lost()) {
				return true;
			}
			let Some(interval) = recover_interval() else {
				return false;
			};
			// One rebuild at a time; a concurrent request stays on the CPU.
			let Ok(mut st) = self.recovery.try_lock() else {
				return false;
			};
			let now = std::time::Instant::now();
			if st.last.is_some_and(|t| now - t < interval) {
				return false;
			}
			st.last = Some(now);
			// The lost device's VRAM has to be freed before a new device can
			// take it. A call still in flight holds its own `Arc` and falls
			// back to the CPU (the device is marked lost), so that is short.
			let old = self.gpu.write().unwrap_or_else(|e| e.into_inner()).take();
			if let Some(mut old) = old {
				let until = now + std::time::Duration::from_secs(10);
				loop {
					match std::sync::Arc::try_unwrap(old) {
						Ok(d) => {
							drop(d);
							break;
						}
						Err(a) if std::time::Instant::now() < until => {
							old = a;
							std::thread::sleep(std::time::Duration::from_millis(10));
						}
						Err(_) => {
							eprintln!("offload: the lost Vulkan device is still in use; rebuilding beside it");
							break;
						}
					}
				}
			}
			let retry = interval.as_secs();
			eprintln!("offload: rebuilding the lost Vulkan device");
			let mut dev = match crate::gpu::GpuDevice::init() {
				Ok(d) => d,
				Err(e) => {
					eprintln!("offload: Vulkan device rebuild failed ({e}); CPU only, next attempt in {retry} s");
					return false;
				}
			};
			dev.decode_seq = self.decode_seq_n;
			let need: u64 = self
				.uploads
				.iter()
				.map(|u| quant_bytes(u.ty, u.out_dim, u.in_dim) as u64)
				.sum();
			let reserve: u64 = std::env::var("QWEN_VRAM_RESERVE_MB")
				.ok()
				.and_then(|s| s.parse::<u64>().ok())
				.unwrap_or(512)
				<< 20;
			if let Some((budget, usage)) = dev.vram_budget() {
				let free = budget.saturating_sub(usage);
				if free < need + reserve {
					let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
					eprintln!(
						"offload: {:.2} GiB of VRAM free, {:.2} GiB needed (+{:.2} reserve), another application holds the rest; CPU only, next attempt in {retry} s",
						gib(free),
						gib(need),
						gib(reserve),
					);
					return false;
				}
			}
			let t0 = std::time::Instant::now();
			for u in &self.uploads {
				let il = if u.key >= Self::SPLIT_KEY {
					u.key - Self::SPLIT_KEY
				} else {
					u.key
				};
				let n = quant_bytes(u.ty, u.out_dim, u.in_dim);
				let ok = match src(il, u.id) {
					Some((raw, ty)) if ty == u.ty && raw.len() >= n => {
						dev.upload(u.key, u.id, &raw[..n], ty, u.out_dim, u.in_dim)
					}
					_ => false,
				};
				if !ok || dev.is_lost() {
					eprintln!(
						"offload: uploading layer {il} {:?} again failed; CPU only, next attempt in {retry} s",
						u.id
					);
					return false;
				}
			}
			eprintln!(
				"offload: Vulkan device rebuilt; {} weights ({need} bytes) uploaded again in {:.1} s",
				self.uploads.len(),
				t0.elapsed().as_secs_f64(),
			);
			*self.gpu.write().unwrap_or_else(|e| e.into_inner()) = Some(std::sync::Arc::new(dev));
			true
		}
		#[cfg(not(feature = "vulkan"))]
		{
			let _ = src;
			false
		}
	}

	/// Key offset for row-split weights in the device's weight table. Far above
	/// any layer index, so a split weight is never found by the resident or
	/// streaming lookups, which key on the plain `(il, id)`.
	#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
	pub(crate) const SPLIT_KEY: usize = 1 << 20;

	/// Upload rows `[0, rows)` of a matrix for the decode row split. `raw` is
	/// the **whole** matrix in its GGUF form (row-major, so the first `rows`
	/// rows are a prefix of the bytes). Only `Q4_K`/`Q6_K` are split; anything
	/// else, `rows == 0`, or an unavailable GPU is a no-op (the CPU does it
	/// all). Independent of `gpu_layers`: split and layer offload are two
	/// placements, and a layer should use one or the other.
	#[cfg_attr(
		not(feature = "vulkan"),
		allow(unused_variables, clippy::needless_return)
	)]
	pub fn upload_split(
		&mut self,
		il: usize,
		id: WeightId,
		raw: &[u8],
		ty: quant::GgmlType,
		out_dim: usize,
		in_dim: usize,
		rows: usize,
	) {
		let blk = match ty {
			quant::GgmlType::Q4_K | quant::GgmlType::Q4_K_M => 144,
			quant::GgmlType::Q6_K => 210,
			_ => return,
		};
		let rows = rows.min(out_dim);
		if !self.available || rows == 0 || !in_dim.is_multiple_of(256) {
			return;
		}
		let row_bytes = in_dim / 256 * blk;
		if raw.len() != out_dim * row_bytes {
			return;
		}
		#[cfg(feature = "vulkan")]
		{
			if let Some(gpu) = self.gpu_mut() {
				if gpu.upload(
					il + Self::SPLIT_KEY,
					id,
					&raw[..rows * row_bytes],
					ty,
					rows,
					in_dim,
				) {
					self.uploads.push(Upload {
						key: il + Self::SPLIT_KEY,
						id,
						ty,
						out_dim: rows,
						in_dim,
					});
					self.split.insert((il, id), rows);
					if il < self.split_layers.len() {
						self.split_layers[il] = true;
					}
				}
			}
		}
	}

	/// Size every weight uploaded from now on for decode batches of up to `n`
	/// positions: `1 + n_draft` under speculative decoding, whose verify pass
	/// then takes the decode split's persistent buffers and recorded command
	/// buffers instead of the per-call prefill path (which allocated a
	/// descriptor pool and parameter buffers per matmul: ~2.7 ms a call, 3x a
	/// decode token over a whole verify pass). Call before any upload.
	pub fn set_decode_seq(&mut self, n: usize) {
		#[cfg(feature = "vulkan")]
		{
			self.decode_seq_n = n.clamp(1, SPEC_MAX_SEQ);
			let n = self.decode_seq_n;
			if let Some(g) = self.gpu_mut() {
				g.decode_seq = n;
			}
		}
		#[cfg(not(feature = "vulkan"))]
		let _ = n;
	}

	/// Positions a decode-split call can carry (see `set_decode_seq`).
	#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
	fn decode_seq(&self) -> usize {
		#[cfg(feature = "vulkan")]
		{
			self.decode_seq_n
		}
		#[cfg(not(feature = "vulkan"))]
		{
			1
		}
	}

	/// Whether a decode (`seq == 1`) matmul of layer `il` should go through
	/// the manager at all: the layer is offloaded, or some matrix is split.
	pub fn decode_on_gpu(&self, il: usize) -> bool {
		self.is_offloaded(il) || self.split_layers.get(il).copied().unwrap_or(false)
	}

	/// `(budget, usage)` of VRAM in bytes, from `VK_EXT_memory_budget` (see
	/// `GpuDevice::vram_budget`). `None` without a GPU or the extension.
	pub fn vram_budget(&self) -> Option<(u64, u64)> {
		#[cfg(feature = "vulkan")]
		{
			self.gpu().and_then(|g| g.vram_budget())
		}
		#[cfg(not(feature = "vulkan"))]
		{
			None
		}
	}

	/// Number of row-split matrices (for the load report).
	pub fn n_split(&self) -> usize {
		self.split.len()
	}

	/// Resident decode-split rows of `(il, id)`, if that matrix is split.
	pub fn split_rows(&self, il: usize, id: WeightId) -> Option<usize> {
		self.split.get(&(il, id)).copied()
	}

	/// Force the prefill split fraction of every K-quant job (`None` restores
	/// the defaults). For the split tuner: it replaces the per-quant default
	/// *and* the `prefill_frac_delta` correction on every chunk that takes a
	/// streamed split; short chunks keep their own rule.
	pub fn set_prefill_frac_override(&self, f: Option<f64>) {
		self.frac_force.store(
			f.unwrap_or(f64::NAN).to_bits(),
			std::sync::atomic::Ordering::Relaxed,
		);
	}

	/// `(resident rows, GPU rows)` the prefill row split gives `j` now
	/// (honoring [`Self::set_prefill_frac_override`]), or `None` if `j` does
	/// not take the split.
	pub fn prefill_split_plan(&self, j: &BatchJobSeq) -> Option<(usize, usize)> {
		#[cfg(feature = "vulkan")]
		{
			self.prefill_split_rows(j)
		}
		#[cfg(not(feature = "vulkan"))]
		{
			let _ = j;
			None
		}
	}

	/// The decode row split: the GPU computes rows `[0, r)` of every split job
	/// while this thread computes the rest (and all of any unsplit job) with
	/// the CPU's group GEMV, then the halves are joined. `None` when there is
	/// nothing to split or the GPU submission fails; the caller then runs the
	/// ordinary path.
	#[cfg(feature = "vulkan")]
	fn matmul_split(&self, jobs: &[BatchJobSeq]) -> Option<Vec<Vec<f32>>> {
		let gpu = self.gpu()?;
		let rows: Vec<usize> = jobs
			.iter()
			.map(|j| self.split.get(&(j.il, j.id)).copied().unwrap_or(0))
			.collect();
		if rows.iter().all(|&r| r == 0) {
			return None;
		}
		let gjobs: Vec<(usize, WeightId, &[f32])> = jobs
			.iter()
			.zip(&rows)
			.filter(|(_, r)| **r > 0)
			.map(|(j, _)| (j.il + Self::SPLIT_KEY, j.id, j.x))
			.collect();
		let seq = jobs[0].seq;
		if seq > 1 {
			return self.matmul_split_seq(jobs, &rows, &gjobs);
		}
		let t0 = std::time::Instant::now();
		let pending = gpu.matmul_batch_prepare(&gjobs)?;
		let t1 = std::time::Instant::now();
		// The CPU's share, concurrently with the GPU's. The GPU batch is
		// submitted from inside the CPU dispatch, once the pool's workers are
		// already computing: `vkQueueSubmit` is a ~15 us kernel transition,
		// and in front of the dispatch it delayed every CPU row by that much.
		// Jobs in one call share `x` by construction (see `cpu_fallback`); if
		// they do not, run each on its own.
		let submitted = std::sync::atomic::AtomicBool::new(false);
		let submit = || {
			if gpu.matmul_batch_submit(&pending).is_some() {
				submitted.store(true, std::sync::atomic::Ordering::Relaxed);
			}
		};
		let shared = jobs
			.windows(2)
			.all(|w| std::ptr::eq(w[0].x, w[1].x) && w[0].in_dim == w[1].in_dim);
		let cpu: Vec<Vec<f32>> = if shared {
			let g: Vec<(&quant::LinearWeight, usize, usize)> = jobs
				.iter()
				.zip(&rows)
				.map(|(j, r)| (j.w, *r, j.out_dim))
				.collect();
			quant::apply_linear_group_rows_pre(&g, jobs[0].x, jobs[0].in_dim, submit)
		} else {
			submit();
			jobs.iter()
				.zip(&rows)
				.map(|(j, r)| {
					quant::apply_linear_group_rows(&[(j.w, *r, j.out_dim)], j.x, j.in_dim)
						.pop()
						.unwrap()
				})
				.collect()
		};
		if !submitted.load(std::sync::atomic::Ordering::Relaxed) {
			eprintln!("offload: GPU submission failed during split matmul; CPU for the rest");
			return None;
		}
		let t2 = std::time::Instant::now();
		let done_before_wait = crate::gpu::gpu_timing() && gpu.fence_signaled();
		// Each output is its GPU rows then its CPU rows; the GPU's are copied
		// straight out of the mapped readback buffer into the final `Vec`.
		let mut out: Vec<Vec<f32>> = jobs.iter().map(|j| Vec::with_capacity(j.out_dim)).collect();
		let gidx: Vec<usize> = rows
			.iter()
			.enumerate()
			.filter(|(_, r)| **r > 0)
			.map(|(i, _)| i)
			.collect();
		let ended = gpu.matmul_batch_end_with(pending, |i, y| out[gidx[i]].extend_from_slice(y));
		if crate::gpu::gpu_timing() {
			let t3 = std::time::Instant::now();
			eprintln!(
				"[SPLIT] n={} prep {:6.1} us  cpu+submit {:6.1} us  wait+read {:6.1} us  done {}",
				jobs.len(),
				(t1 - t0).as_secs_f64() * 1e6,
				(t2 - t1).as_secs_f64() * 1e6,
				(t3 - t2).as_secs_f64() * 1e6,
				done_before_wait as u8,
			);
		}
		if ended.is_none() {
			eprintln!("offload: GPU device lost during split matmul; CPU for the rest");
			return None;
		}
		for (o, c) in out.iter_mut().zip(&cpu) {
			o.extend_from_slice(c);
		}
		Some(out)
	}

	/// [`Self::matmul_split`] for a speculative verify batch (`1 < seq <=
	/// decode_seq`): the GPU runs its resident rows of every position through
	/// the same persistent buffers and recorded command buffers as decode,
	/// while the CPU GEMM covers the other rows, writing straight into the
	/// `[seq x out]` results; the GPU's `[seq x r]` block is then copied in.
	#[cfg(feature = "vulkan")]
	fn matmul_split_seq(
		&self,
		jobs: &[BatchJobSeq],
		rows: &[usize],
		gjobs: &[(usize, WeightId, &[f32])],
	) -> Option<Vec<Vec<f32>>> {
		let gpu = self.gpu()?;
		let seq = jobs[0].seq;
		let t0 = std::time::Instant::now();
		let pending = gpu.matmul_batch_begin(gjobs)?;
		let t1 = std::time::Instant::now();
		let mut out: Vec<Vec<f32>> = jobs
			.iter()
			.map(|j| quant::uninit_vec(j.seq * j.out_dim))
			.collect();
		for ((j, &r), y) in jobs.iter().zip(rows).zip(out.iter_mut()) {
			let n = j.out_dim - r;
			if n == 0 {
				continue;
			}
			match j.w {
				quant::LinearWeight::Q4K { raw, .. } => {
					let rb = raw.len() / j.out_dim;
					quant::gemm_q4k_strided(n, j.in_dim, seq, &raw[r * rb..], j.x, y, j.out_dim, r)
				}
				quant::LinearWeight::Q6K { raw, .. } => {
					let rb = raw.len() / j.out_dim;
					quant::gemm_q6k_strided(n, j.in_dim, seq, &raw[r * rb..], j.x, y, j.out_dim, r)
				}
				// Only K-quants are split, so any other weight has r == 0.
				w => {
					y.copy_from_slice(&quant::apply_linear_batch(w, j.x, seq, j.out_dim, j.in_dim))
				}
			}
		}
		let t2 = std::time::Instant::now();
		let gidx: Vec<usize> = rows
			.iter()
			.enumerate()
			.filter(|(_, r)| **r > 0)
			.map(|(i, _)| i)
			.collect();
		let ended = gpu.matmul_batch_end_with(pending, |i, g| {
			let k = gidx[i];
			let (r, od) = (rows[k], jobs[k].out_dim);
			for t in 0..seq {
				out[k][t * od..t * od + r].copy_from_slice(&g[t * r..(t + 1) * r]);
			}
		});
		if crate::gpu::gpu_timing() {
			eprintln!(
				"[SPLIT] n={} seq={seq} begin {:6.1} us  cpu {:6.1} us  wait+read {:6.1} us",
				jobs.len(),
				(t1 - t0).as_secs_f64() * 1e6,
				(t2 - t1).as_secs_f64() * 1e6,
				t2.elapsed().as_secs_f64() * 1e6,
			);
		}
		if ended.is_none() {
			eprintln!("offload: GPU device lost during split matmul; CPU for the rest");
			return None;
		}
		Some(out)
	}

	/// Whether layer `il` is slated for GPU offload (regardless of whether a GPU
	/// is actually present: `matmul` still checks `available`).
	pub fn is_offloaded(&self, il: usize) -> bool {
		il < self.gpu_layers.len() && self.gpu_layers[il]
	}

	/// Does the GPU hold the uploaded weight for `(il, id)`? Always `false` when
	/// the `vulkan` feature is off or no adapter was acquired.
	#[inline]
	fn gpu_present(&self, il: usize, id: WeightId) -> bool {
		#[cfg(feature = "vulkan")]
		{
			self.gpu().is_some_and(|g| g.has(il, id))
		}
		#[cfg(not(feature = "vulkan"))]
		{
			let _ = (il, id);
			false
		}
	}

	/// Upload one layer weight matrix to the GPU **in its original quantized
	/// GGUF form** (no-op unless `available` and the layer is offloaded).
	///
	/// Only `Q4_K`/`Q4_K_M` and `Q6_K` big matmuls are uploaded: these are
	/// dequantized on the fly inside the matching fused matmul shader, so they
	/// stay compressed in VRAM (the llama.cpp mechanism), never expanded to a
	/// full `f16`/`f32` matrix. Other quant types (`F16`, `F32`, `Q8_0`, …)
	/// are not uploaded and fall back to the verified CPU path.
	pub fn upload(
		&mut self,
		il: usize,
		id: WeightId,
		raw: &[u8],
		ty: quant::GgmlType,
		out_dim: usize,
		in_dim: usize,
	) {
		if !self.available || !self.is_offloaded(il) {
			return;
		}
		#[cfg(feature = "vulkan")]
		{
			if let Some(gpu) = self.gpu_mut() {
				if gpu.upload(il, id, raw, ty, out_dim, in_dim) {
					self.uploads.push(Upload {
						key: il,
						id,
						ty,
						out_dim,
						in_dim,
					});
				}
			}
		}
		#[cfg(not(feature = "vulkan"))]
		{
			let _ = (il, id, raw, ty, out_dim, in_dim);
		}
	}

	/// Matrix product `y = W @ x` for one token (`seq_len == 1`).
	///
	/// - When `available` and layer `il` is offloaded and the weight was
	///   uploaded, the GPU (Vulkan) GEMM runs (ignoring `w`, which is already on
	///   the GPU as quantized `Q4_K_M` bytes).
	/// - Otherwise (no GPU / not offloaded / weight absent) it falls back to the
	///   verified CPU `quant::apply_linear` on `w` (fused dequant-GEMV for
	///   `Q4_K`, else `f16`/`f32`).
	///
	/// The KV cache and SSM state are never touched here: they remain CPU-side.
	///
	/// Implemented as a single-element [`matmul_batch`] (which lets the GPU fold
	/// multiple independent matmuls into one submission and overlap their H2D/D2H
	/// transfers).
	pub fn matmul(
		&self,
		il: usize,
		#[allow(unused_variables)] id: WeightId,
		w: &quant::LinearWeight,
		x: &[f32],
		out_dim: usize,
		in_dim: usize,
	) -> Vec<f32> {
		let job = BatchJob {
			il,
			id,
			w,
			x,
			out_dim,
			in_dim,
		};
		self.matmul_batch(std::slice::from_ref(&job)).pop().unwrap()
	}

	/// Dispatch a batch of independent matrix products in as few GPU submissions
	/// as possible.
	///
	/// All jobs **share the same layer/input availability** at the call site
	/// (e.g. `Wq`/`Wk`/`Wv` all consume the attention-normed hidden state, or
	/// `FfnUp`/`FfnGate` both consume the post-attention norm). They are
	/// therefore fire-and-forget from the model's perspective: we upload every
	/// `x` slice, encode every fused dequant+matmul dispatch into a *single*
	/// command encoder + *single* queue submission, copy every result back, and
	/// only then block once on `device.poll`. This overlaps the H2D uploads
	/// (they are queued back-to-back and the driver streams them while the
	/// device is otherwise idle) and the D2H readbacks (all `map_async` are
	/// requested before the single blocking poll), replacing ~N synchronous
	/// submit+map round-trips per layer with one.
	///
	/// If the GPU is unavailable, a layer isn't offloaded, or any weight wasn't
	/// uploaded, the whole batch transparently falls back to the verified CPU
	/// `quant::apply_linear` per job (order preserved).
	pub fn matmul_batch(&self, jobs: &[BatchJob]) -> Vec<Vec<f32>> {
		let seq_jobs: Vec<BatchJobSeq> = jobs
			.iter()
			.map(|j| BatchJobSeq {
				il: j.il,
				id: j.id,
				w: j.w,
				x: j.x,
				seq: 1,
				out_dim: j.out_dim,
				in_dim: j.in_dim,
			})
			.collect();
		self.matmul_batch_seq(&seq_jobs)
	}

	/// Batched (prefill, `seq > 1`) variant of [`Self::matmul_batch`]. Each job's
	/// `x` is `[seq * in_dim]` and the returned vector is `[seq * out_dim]`
	/// (seq-major). On the GPU the fused dequant+matmul shader runs over the
	/// `seq` rows in its `y` workgroup dimension; otherwise it falls back to the
	/// verified CPU `quant::apply_linear_batch` (streaming each weight once
	/// across the `seq` positions; the prefill path).
	pub fn matmul_batch_seq(&self, jobs: &[BatchJobSeq]) -> Vec<Vec<f32>> {
		if let Some(r) = Self::exact_decode(jobs) {
			return r;
		}
		#[cfg(feature = "vulkan")]
		{
			if !self.split.is_empty()
				&& !jobs.is_empty()
				&& jobs.iter().all(|j| j.seq == jobs[0].seq)
				&& jobs[0].seq <= self.decode_seq()
			{
				if let Some(r) = self.matmul_split(jobs) {
					return r;
				}
			}
			if !self.split.is_empty() && !jobs.is_empty() && jobs.iter().all(|j| j.seq > 1) {
				let is_split = |j: &BatchJobSeq| self.prefill_split_rows(j).is_some();
				if jobs.iter().any(is_split) {
					// Split jobs take the concurrent route, and the small
					// CPU-tail ones (`ssm_beta` / `ssm_alpha`) run on the CPU
					// inside it; the rest (weights with no resident rows) keep
					// the whole-matrix route, run first. Output order is the
					// jobs' order.
					let (mut sp, mut tail, mut full) = (Vec::new(), Vec::new(), Vec::new());
					for i in 0..jobs.len() {
						if is_split(&jobs[i]) {
							sp.push(i);
						} else if Self::cpu_tail_ok(&jobs[i]) {
							tail.push(i);
						} else {
							full.push(i);
						}
					}
					let pick = |ix: &[usize]| -> Vec<BatchJobSeq> {
						ix.iter().map(|&i| jobs[i].clone()).collect()
					};
					let (full_jobs, split_jobs, tail_jobs) = (pick(&full), pick(&sp), pick(&tail));
					let full_out = if full_jobs.is_empty() {
						Vec::new()
					} else {
						self.matmul_batch_seq_whole(&full_jobs)
					};
					if let Some(split_out) =
						self.matmul_split_prefill(&split_jobs, &tail_jobs, None)
					{
						let mut out: Vec<Option<Vec<f32>>> = vec![None; jobs.len()];
						for (k, v) in full.iter().zip(full_out) {
							out[*k] = Some(v);
						}
						for (k, v) in sp.iter().chain(&tail).zip(split_out) {
							out[*k] = Some(v);
						}
						return out.into_iter().map(|v| v.unwrap()).collect();
					}
				}
			}
		}
		self.matmul_batch_seq_whole(jobs)
	}

	/// A prefill job that [`Self::matmul_split_prefill`] runs on the CPU
	/// beside the split, from the split's q8_K activation, instead of as a
	/// whole-matrix call of its own: `ssm_beta` / `ssm_alpha` (48 rows each),
	/// which have no split rows. Each pair was a separate streamed GPU
	/// submission (~4 ms of serial pack + upload + wait per SSM layer, ~0.19 s
	/// per 2048-token chunk) for ~1 ms of CPU GEMM, and the CPU has slack
	/// there: `Wqkv` / `WqkvGate`'s GPU share finishes ~9 ms after the CPU's.
	#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
	fn cpu_tail_ok(j: &BatchJobSeq) -> bool {
		j.id.exact_decode()
			&& j.in_dim.is_multiple_of(256)
			&& matches!(
				j.w,
				quant::LinearWeight::Q4K { .. } | quant::LinearWeight::Q6K { .. }
			)
	}

	/// Whether [`Self::matmul_seq_group_q8`] would run this group of prefill
	/// matmuls (`(id, weight, out_dim)`, one shared input of `in_dim`) through
	/// the row split, i.e. whether a caller may hand it a q8_K activation and
	/// skip materializing the `f32` one. Mirrors `matmul_batch_seq`'s routing:
	/// not a verify-sized batch, every job takes the split (the GPU side
	/// reading q8_K at this length) or is a CPU-tail job, and at least one
	/// takes the split.
	pub fn prefill_q8_ok(
		&self,
		il: usize,
		jobs: &[(WeightId, &quant::LinearWeight, usize)],
		seq: usize,
		in_dim: usize,
	) -> bool {
		#[cfg(feature = "vulkan")]
		{
			let Some(gpu) = self.gpu() else { return false };
			if self.split.is_empty() || seq <= self.decode_seq() {
				return false;
			}
			let mut any = false;
			for &(id, w, out_dim) in jobs {
				let j = BatchJobSeq {
					il,
					id,
					w,
					x: &[],
					seq,
					out_dim,
					in_dim,
				};
				let kind = match w {
					quant::LinearWeight::Q4K { .. } => crate::gpu::GpuKind::Q4K,
					quant::LinearWeight::Q6K { .. } => crate::gpu::GpuKind::Q6K,
					_ => return false,
				};
				if self.prefill_split_rows(&j).is_some() {
					if !gpu.wants_q8k(kind, seq) {
						return false;
					}
					any = true;
				} else if !Self::cpu_tail_ok(&j) {
					return false;
				}
			}
			any
		}
		#[cfg(not(feature = "vulkan"))]
		{
			let _ = (il, jobs, seq, in_dim);
			false
		}
	}

	/// A group of prefill matmuls whose shared activation arrives already
	/// quantized (`q8`, `seq` rows of `in_dim`), e.g. `quant::par_silu_mul_q8k`
	/// for `FfnDown` or an RMS norm for `FfnUp` / `FfnGate`. Routed exactly as
	/// `matmul_batch_seq` routes an `f32` input after its own quantization, so
	/// the output is bit-identical; call only where [`Self::prefill_q8_ok`]
	/// said yes. A lost device falls back to the CPU GEMM on the same rows.
	#[cfg_attr(not(feature = "vulkan"), allow(unused_variables))]
	pub fn matmul_seq_group_q8(
		&self,
		il: usize,
		jobs: &[(WeightId, &quant::LinearWeight, usize)],
		q8: &std::sync::Arc<Vec<quant::Q8K>>,
		seq: usize,
		in_dim: usize,
	) -> Vec<Vec<f32>> {
		assert_eq!(q8.len(), seq);
		#[cfg(feature = "vulkan")]
		{
			let (mut sp, mut tail) = (Vec::new(), Vec::new());
			for (k, &(id, w, out_dim)) in jobs.iter().enumerate() {
				let j = BatchJobSeq {
					il,
					id,
					w,
					x: &[],
					seq,
					out_dim,
					in_dim,
				};
				if self.prefill_split_rows(&j).is_some() {
					sp.push((k, j));
				} else {
					tail.push((k, j));
				}
			}
			let split_jobs: Vec<BatchJobSeq> = sp.iter().map(|(_, j)| j.clone()).collect();
			let tail_jobs: Vec<BatchJobSeq> = tail.iter().map(|(_, j)| j.clone()).collect();
			if let Some(r) = self.matmul_split_prefill(&split_jobs, &tail_jobs, Some(q8)) {
				let mut out: Vec<Option<Vec<f32>>> = vec![None; jobs.len()];
				for ((k, _), v) in sp.iter().chain(&tail).zip(r) {
					out[*k] = Some(v);
				}
				return out.into_iter().map(|v| v.unwrap()).collect();
			}
		}
		jobs.iter()
			.map(|&(_, w, out_dim)| {
				let mut y = quant::uninit_vec(seq * out_dim);
				Self::gemm_q8(w, 0, out_dim, in_dim, seq, q8, &mut y, out_dim, 0);
				y
			})
			.collect()
	}

	/// Rows `[row0, row0 + n)` of a K-quant `w` on a q8_K activation, on the
	/// CPU, as a `[seq x n]` matrix: the fallback where
	/// [`Self::matmul_q8_defer_rows`] could not start.
	pub fn gemm_q8_rows(
		w: &quant::LinearWeight,
		row0: usize,
		n: usize,
		in_dim: usize,
		seq: usize,
		q8: &std::sync::Arc<Vec<quant::Q8K>>,
	) -> Vec<f32> {
		let mut y = quant::uninit_vec(seq * n);
		Self::gemm_q8(w, row0, n, in_dim, seq, q8, &mut y, n, 0);
		y
	}

	/// The CPU GEMM of rows `[row0, row0 + n)` of a K-quant `w` on a q8_K
	/// activation, into column `col0` of `y` (row stride `ld`).
	#[allow(clippy::too_many_arguments)]
	fn gemm_q8(
		w: &quant::LinearWeight,
		row0: usize,
		n: usize,
		in_dim: usize,
		seq: usize,
		q8: &std::sync::Arc<Vec<quant::Q8K>>,
		y: &mut [f32],
		ld: usize,
		col0: usize,
	) {
		// Exactly the rows asked for: a caller may run a prefix of a matrix
		// (the query rows of `Wq`, see `qgate_defer_ok`).
		match w {
			quant::LinearWeight::Q4K { raw, .. } => {
				let rb = in_dim / 256 * 144;
				quant::gemm_q4k_strided_q8(
					n,
					in_dim,
					seq,
					&raw[row0 * rb..(row0 + n) * rb],
					q8,
					y,
					ld,
					col0,
				)
			}
			quant::LinearWeight::Q6K { raw, .. } => {
				let rb = in_dim / 256 * 210;
				quant::gemm_q6k_strided_q8(
					n,
					in_dim,
					seq,
					&raw[row0 * rb..(row0 + n) * rb],
					q8,
					y,
					ld,
					col0,
				)
			}
			_ => unreachable!("q8_K GEMMs take Q4_K/Q6_K only"),
		}
	}

	/// Key offset for the *streamed* extra rows of a prefill split, so their
	/// stream slots never alias a whole-matrix stream of the same weight.
	#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
	const PSTREAM_KEY: usize = 2 << 20;

	/// Key offset for a deferred row range past the resident prefix
	/// ([`Self::matmul_q8_defer_rows`]), whose streamed rows are not the
	/// `[r, g)` range `PSTREAM_KEY` holds for the same weight.
	#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
	const RSTREAM_KEY: usize = 3 << 20;

	/// GPU share of a prefill row split for `j`: `(resident rows r, GPU rows
	/// g)` with `r <= g <= out`, or `None` if `j` does not take the split.
	///
	/// The GPU computes rows `[0, g)` (`[0, r)` from its resident decode-split
	/// rows, `[r, g)` streamed for this call) and the CPU `[g, out)`. `g` is
	/// set so both sides finish together: measured at `seq = 1529`, the GPU's
	/// tiled GEMMs run ~2.07 TFLOP/s (Q4_K) and ~2.1 (Q6_K), the CPU's ~1.67
	/// and ~1.84 since the Q6_K kernel stopped calling `_mm256_fmadd_ps` out of
	/// line. Q4_K is 0.55; Q6_K is 0.60, the measured optimum of a
	/// 0.50-0.68 sweep (the rate ratio alone says ~0.53).
	///
	/// With the dp4a GEMMs (~4.4 TFLOP/s kernel-only) the GPU
	/// takes 0.70 / 0.72 from 192 tokens and 0.65 / 0.67 below (0.60 / 0.62
	/// until the GPU stopped waiting for its uploads and readback copies the
	/// same day: 172 tokens 74.1 / **77.6** / 70.3 tok/s at 0.60 / 0.65 /
	/// 0.70, 115 tokens flat at 66-67), and streams
	/// its extra rows from 96 tokens instead of `stream_min_seq` (256).
	/// Measured end to end, tok/s: 2226 tokens flat at 75-77 over 0.68-0.76;
	/// 538 tokens 64.9 / **77.4** / 67.0 at 0.55 / 0.70 / 0.80; 172 tokens 49.1
	/// resident-only, 55.3 / **58.7** / 58.2 streamed at 0.50 / 0.60 / 0.70;
	/// 115 tokens 47.9 resident-only, **52.3** at 0.60; 71 tokens 45.7
	/// resident-only, 42.7 / 41.3 streamed at 0.50 / 0.60.
	/// With 16-byte staging loads in both dp4a GEMMs (kernel
	/// 82 -> 59 / 68.5 ms) the fractions move to 0.77 / 0.75 (2226
	/// tokens, 2 rounds: 0.70/0.72 92.3 / 93.1, 0.74/0.74 100.5, 0.76-0.78 /
	/// 0.73-0.77 flat at 101.9-103.8, 0.80/0.78 101.3-101.9), and below 192
	/// tokens to 0.72 / 0.70 (172 tokens: 0.65/0.67 79.4 / 81.1, 0.70/0.70
	/// 83.3 / 84.2, **0.72/0.70 85.4 / 87.2**, 0.75/0.73 80.0 / 81.9), and
	/// below 128 to 0.68 / 0.66 (115 tokens, 0.72/0.70 65.7 / 66.4 vs 70-74
	/// anywhere in 0.66-0.70; 0.65/0.67 before the change 68.3 / 68.6).
	/// (`QWEN_PREFILL_GPU_FRAC_Q4K` / `_Q6K` override the fraction at every
	/// length, `QWEN_GPU_STREAM_MIN_SEQ` the streaming gate.)
	/// `QWEN_GPU_PREFILL_SPLIT=0` sends everything back to the whole-matrix
	/// route.
	#[cfg(feature = "vulkan")]
	fn prefill_split_rows(&self, j: &BatchJobSeq) -> Option<(usize, usize)> {
		self.prefill_split_rows_x(j, 0.0)
	}

	/// [`Self::prefill_split_rows`] with `extra` added to the GPU fraction
	/// (dp4a GEMMs only), for a call whose CPU side also has other work.
	#[cfg(feature = "vulkan")]
	fn prefill_split_rows_x(&self, j: &BatchJobSeq, extra: f64) -> Option<(usize, usize)> {
		fn env_f(name: &str, cell: &'static std::sync::OnceLock<f64>, dflt: f64) -> f64 {
			*cell.get_or_init(|| {
				std::env::var(name)
					.ok()
					.and_then(|v| v.trim().parse().ok())
					.unwrap_or(dflt)
			})
		}
		static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
		static FQ4: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
		static FQ6: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
		if *OFF.get_or_init(|| {
			matches!(
				std::env::var("QWEN_GPU_PREFILL_SPLIT")
					.as_deref()
					.map(str::trim),
				Ok("0") | Ok("off") | Ok("no") | Ok("false")
			)
		}) {
			return None;
		}
		let dp4a = |k: crate::gpu::GpuKind| self.gpu().is_some_and(|g| g.dp4a_gemm(k));
		let (kind, f_dp4a, f_float, var, cell) = match j.w {
			quant::LinearWeight::Q4K { .. } => (
				crate::gpu::GpuKind::Q4K,
				0.73,
				0.55,
				"QWEN_PREFILL_GPU_FRAC_Q4K",
				&FQ4,
			),
			quant::LinearWeight::Q6K { .. } => (
				crate::gpu::GpuKind::Q6K,
				0.72,
				0.60,
				"QWEN_PREFILL_GPU_FRAC_Q6K",
				&FQ6,
			),
			_ => return None,
		};
		let fast = dp4a(kind);
		let dflt = match (fast, j.seq) {
			(true, s) if s < 128 => f_dp4a - 0.11,
			(true, s) if s < 192 => f_dp4a - 0.07,
			(true, _) => f_dp4a,
			(false, _) => f_float,
		};
		// An explicit override applies at every length; NaN marks "unset".
		let frac = match env_f(var, cell, f64::NAN) {
			f if f.is_nan() => dflt,
			f => f,
		} + if fast {
			prefill_frac_delta(j.id) + extra
		} else {
			0.0
		};
		let frac = match f64::from_bits(self.frac_force.load(std::sync::atomic::Ordering::Relaxed))
		{
			f if f.is_nan() => frac,
			f => f,
		};
		static STREAM_SET: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
		let stream_set =
			*STREAM_SET.get_or_init(|| std::env::var("QWEN_GPU_STREAM_MIN_SEQ").is_ok());
		let stream_min = if fast && !stream_set {
			96
		} else {
			crate::gpu::GpuDevice::stream_min_seq()
		};
		let r = self.split.get(&(j.il, j.id)).copied()?;
		if !j.in_dim.is_multiple_of(256) {
			return None;
		}
		// A speculative-decoding verify batch (`1 + n_draft` tokens) is decode
		// with a few extra positions, not prefill: it keeps exactly the decode
		// split; the resident rows on the GPU (its GEMV covers the positions
		// in `grid.y`, re-reading rows that are VRAM-resident anyway), the
		// rest through the CPU GEMM, which at 2-4 positions costs 1.06-1.23x
		// one GEMV (`quant/examples/smallseq.rs`). Without this, the batch fell
		// under `prefill_min_seq` and the GPU's third of every matrix went
		// back to the CPU.
		if j.seq <= SPEC_MAX_SEQ {
			return Some((r, r));
		}
		if j.seq < crate::gpu::GpuDevice::prefill_min_seq() {
			// A short chunk (a tool result, a follow-up turn): too short for a
			// streamed range to pay its upload, but the resident rows are
			// already in VRAM, so the GPU takes as many of them as it can
			// finish while the CPU does the rest. The tiled GEMM computes a
			// whole 64-token tile regardless (minus the all-padding warps), so
			// per row it is slower than the CPU here and its share is set from
			// the measured rates rather than `frac`.
			if j.seq < SMALL_SPLIT_MIN_SEQ || !crate::gpu::GpuDevice::gemm_on(j.seq) {
				return None;
			}
			let q6k = matches!(j.w, quant::LinearWeight::Q6K { .. });
			let g =
				((j.out_dim as f64 * small_chunk_frac(j.seq, q6k, fast)) as usize / 4 * 4).min(r);
			return (g > 0).then_some((g, g));
		}
		let mut g = ((j.out_dim as f64 * frac.clamp(0.0, 1.0)) as usize / 4 * 4).max(r);
		// Streaming the extra rows only pays past the streamed crossover.
		if g > r && (!crate::gpu::GpuDevice::stream_enabled() || j.seq < stream_min) {
			g = r;
		}
		Some((r, g.min(j.out_dim)))
	}

	/// The prefill row split: the GPU runs rows `[0, g)` of every job (tiled
	/// GEMM; the resident `[0, r)` plus `[r, g)` streamed) while this thread
	/// runs `[g, out)` through the CPU GEMM on the same row-major bytes, then
	/// the blocks are interleaved into `[seq x out]`.
	#[cfg(feature = "vulkan")]
	/// `tail` jobs (see [`Self::cpu_tail_ok`]) run entirely on the CPU after
	/// the CPU GEMMs, while the GPU finishes; their outputs follow `jobs`'.
	fn matmul_split_prefill(
		&self,
		jobs: &[BatchJobSeq],
		tail: &[BatchJobSeq],
		pre: Option<&std::sync::Arc<Vec<quant::Q8K>>>,
	) -> Option<Vec<Vec<f32>>> {
		let gpu = self.gpu()?;
		let plan: Vec<(usize, usize)> = jobs
			.iter()
			.map(|j| self.prefill_split_rows(j).unwrap())
			.collect();
		let row_bytes = |j: &BatchJobSeq| -> usize {
			j.in_dim / 256
				* match j.w {
					quant::LinearWeight::Q4K { .. } => 144,
					_ => 210,
				}
		};
		fn raw_of<'a>(j: &BatchJobSeq<'a>) -> &'a [u8] {
			match j.w {
				quant::LinearWeight::Q4K { raw, .. } | quant::LinearWeight::Q6K { raw, .. } => raw,
				_ => unreachable!("prefill_split_rows admits Q4_K/Q6_K only"),
			}
		}
		// Each distinct input quantized to q8_K once, for the GPU's staging
		// buffer and every CPU GEMM that reads it. Both sides used to
		// quantize on their own, and the CPU once more per matrix: `FfnUp` +
		// `FfnGate` and `Wqkv` + `WqkvGate` share their input, so a
		// 2048-token chunk quantized ~1.0 s of activations where ~0.4 s is
		// needed. Same kernel (`quantize_q8k`), so the bytes are identical.
		let tq = std::time::Instant::now();
		// `pre` is that quantization done by the caller
		// (`matmul_seq_group_q8`), for a shared input that was never
		// materialized (every job's `x` is then empty, so one key).
		let mut q8s: Vec<(usize, std::sync::Arc<Vec<quant::Q8K>>)> = Vec::new();
		for j in jobs.iter().chain(tail) {
			let key = j.x.as_ptr() as usize;
			if let Some(p) = pre {
				if q8s.is_empty() {
					q8s.push((key, p.clone()));
				}
			} else if !q8s.iter().any(|(k, _)| *k == key) {
				q8s.push((
					key,
					std::sync::Arc::new(quant::par_quantize_q8k(j.x, j.seq, j.in_dim)),
				));
			}
		}
		let q8_of = |j: &BatchJobSeq| -> std::sync::Arc<Vec<quant::Q8K>> {
			let key = j.x.as_ptr() as usize;
			q8s.iter().find(|(k, _)| *k == key).unwrap().1.clone()
		};
		let q8_ms = tq.elapsed().as_secs_f64() * 1e3;
		let q8_ref: Vec<std::sync::Arc<Vec<quant::Q8K>>> = jobs.iter().map(&q8_of).collect();
		// The final `[seq x out]` matrices, filled in place: the CPU GEMM
		// writes its column block `[g, out)` directly and the GPU rows `[0, r)`
		// and `[r, g)` land there too, written by the GEMM itself through a
		// host-pointer import (`SeqJob::dst`) or else copied out of the mapped
		// readback. Every element is written by exactly one of the three.
		let mut out: Vec<Vec<f32>> = jobs
			.iter()
			.map(|j| quant::uninit_vec(j.seq * j.out_dim))
			.collect();
		// GPU jobs, in order: per job its resident block, then its streamed one.
		let mut gjobs: Vec<crate::gpu::SeqJob> = Vec::new();
		let mut dsts: Vec<(usize, usize, usize, usize)> = Vec::new();
		for (((j, &(r, g)), q8), y) in jobs.iter().zip(&plan).zip(&q8_ref).zip(out.iter_mut()) {
			let yp = y.as_mut_ptr() as usize;
			if r > 0 {
				gjobs.push(crate::gpu::SeqJob {
					il: j.il + Self::SPLIT_KEY,
					id: j.id,
					x: j.x,
					seq: j.seq,
					src: None,
					rows: Some(r),
					q8: Some(&q8[..]),
					dst: Some((yp, j.out_dim, 0)),
				});
				dsts.push((yp, j.out_dim, 0, r));
			}
			if g > r {
				let kind = match j.w {
					quant::LinearWeight::Q4K { .. } => crate::gpu::GpuKind::Q4K,
					_ => crate::gpu::GpuKind::Q6K,
				};
				let rb = row_bytes(j);
				gjobs.push(crate::gpu::SeqJob {
					il: j.il + Self::PSTREAM_KEY,
					id: j.id,
					x: j.x,
					seq: j.seq,
					src: Some((&raw_of(j)[r * rb..g * rb], kind, g - r, j.in_dim)),
					rows: None,
					q8: Some(&q8[..]),
					dst: Some((yp, j.out_dim, r)),
				});
				dsts.push((yp, j.out_dim, r, g - r));
			}
		}
		let t0 = std::time::Instant::now();
		let pending = gpu.matmul_batch_seq_begin(&gjobs)?;
		// The next call's streamed rows, if this one's are already in VRAM
		// (else once the CPU rows are done, below).
		self.issue_prefetch(false);
		let t1 = std::time::Instant::now();
		for (((j, &(_, g)), y), q8) in jobs.iter().zip(&plan).zip(out.iter_mut()).zip(&q8_ref) {
			let n = j.out_dim - g;
			if n == 0 {
				continue;
			}
			Self::gemm_q8(j.w, g, n, j.in_dim, j.seq, q8, y, j.out_dim, g);
		}
		for j in tail {
			let mut y = quant::uninit_vec(j.seq * j.out_dim);
			Self::gemm_q8(
				j.w,
				0,
				j.out_dim,
				j.in_dim,
				j.seq,
				&q8_of(j),
				&mut y,
				j.out_dim,
				0,
			);
			out.push(y);
		}
		self.issue_prefetch(true);
		let t2 = std::time::Instant::now();
		// SAFETY: every destination is a column block of an `out` matrix
		// sized `seq * out_dim`, and the blocks of one matrix are disjoint.
		if unsafe { gpu.matmul_batch_seq_end_into(pending, &dsts) }.is_none() {
			eprintln!("offload: GPU device lost during split prefill; CPU for the rest");
			return None;
		}
		if crate::gpu::gpu_timing() {
			eprintln!(
				"[PSPLIT] n={} seq={} begin {:7.2} ms  cpu {:7.2} ms  end+copy {:7.2} ms  q8 {:6.2} ms",
				jobs.len(),
				jobs[0].seq,
				(t1 - t0).as_secs_f64() * 1e3,
				(t2 - t1).as_secs_f64() * 1e3,
				t2.elapsed().as_secs_f64() * 1e3,
				q8_ms,
			);
		}
		Some(out)
	}

	/// Tell the manager which prefill matmul the caller makes next, on layer
	/// `il` (`seq` positions, `in_dim` inputs): its streamed rows are then
	/// uploaded while the current call runs (as soon as the current call's
	/// own upload has landed, at the latest when its CPU rows are done)
	/// instead of in front of the next call's GPU rows. Below the two-half
	/// pipeline (`Model::layers_pipe`, which overlaps them already) every
	/// call paid its repack and DMA serially: at 140 tokens ~0.7 s of a
	/// 1.6 s GEMM phase was the CPU waiting for the GPU while the GPU waited
	/// for its uploads. `keep` owns the weights' bytes. `QWEN_PREFETCH=0`
	/// turns it off (the A/B); the hint only moves an upload, so output is
	/// bit-identical either way.
	pub fn prefetch_hint(
		&self,
		il: usize,
		call: NextCall<'_>,
		seq: usize,
		in_dim: usize,
		keep: std::sync::Arc<dyn std::any::Any + Send + Sync>,
	) {
		#[cfg(feature = "vulkan")]
		{
			static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
			let on = *ON.get_or_init(|| {
				!matches!(
					std::env::var("QWEN_PREFETCH").as_deref().map(str::trim),
					Ok("0") | Ok("off") | Ok("no") | Ok("false")
				)
			});
			if !on || self.split.is_empty() || !in_dim.is_multiple_of(256) {
				return;
			}
			let mut v = Vec::new();
			// Rows `[a, b)` of `w`, under stream key `key`.
			let mut push =
				|key: usize, id: WeightId, w: &quant::LinearWeight, a: usize, b: usize| {
					let (kind, rb, raw) = match w {
						quant::LinearWeight::Q4K { raw, .. } => {
							(crate::gpu::GpuKind::Q4K, in_dim / 256 * 144, raw)
						}
						quant::LinearWeight::Q6K { raw, .. } => {
							(crate::gpu::GpuKind::Q6K, in_dim / 256 * 210, raw)
						}
						_ => return,
					};
					if b > a && b * rb <= raw.len() {
						v.push((
							il + key,
							id,
							raw[a * rb..].as_ptr() as usize,
							(b - a) * rb,
							kind,
						));
					}
				};
			// The same ranges the call itself will stream:
			// `matmul_split_prefill`, `matmul_q8_defer`, `matmul_q8_defer_rows`.
			match call {
				NextCall::Split(jobs) => {
					for &(id, w, out_dim) in jobs {
						let j = BatchJobSeq {
							il,
							id,
							w,
							x: &[],
							seq,
							out_dim,
							in_dim,
						};
						if let Some((r, g)) = self.prefill_split_rows(&j) {
							push(Self::PSTREAM_KEY, id, w, r, g);
						}
					}
				}
				NextCall::Defer(id, w, out_dim) => {
					let j = BatchJobSeq {
						il,
						id,
						w,
						x: &[],
						seq,
						out_dim,
						in_dim,
					};
					if let (Some(frac), Some((r, g0))) =
						(Self::defer_frac(), self.prefill_split_rows(&j))
					{
						if g0 > r {
							let g = ((out_dim as f64 * frac) as usize / 4 * 4).clamp(r, out_dim);
							push(Self::PSTREAM_KEY, id, w, r, g);
						}
					}
				}
				NextCall::DeferRows(id, w, row0, n) => {
					if Self::defer_frac().is_some()
						&& self.split.get(&(il, id)).is_some_and(|&r| row0 >= r)
					{
						let g = (n as f64 * Self::qgate_frac()) as usize / 4 * 4;
						push(Self::RSTREAM_KEY, id, w, row0, row0 + g.min(n));
					}
				}
			}
			*self.next.lock().unwrap() = (!v.is_empty()).then_some((v, keep));
		}
		#[cfg(not(feature = "vulkan"))]
		{
			let _ = (il, call, seq, in_dim, keep);
		}
	}

	/// Start the hinted upload (see [`Self::prefetch_hint`]). Without
	/// `block` only if the transfer queue is idle; else it stays staged.
	#[cfg(feature = "vulkan")]
	fn issue_prefetch(&self, block: bool) {
		let mut slot = self.next.lock().unwrap();
		let Some((v, keep)) = slot.take() else { return };
		let Some(gpu) = self.gpu() else { return };
		// SAFETY: `keep` owns every range's bytes (see `prefetch_hint`).
		let needed: Vec<(usize, WeightId, &[u8], crate::gpu::GpuKind)> = v
			.iter()
			.map(|&(k, id, p, n, kind)| {
				(
					k,
					id,
					unsafe { std::slice::from_raw_parts(p as *const u8, n) },
					kind,
				)
			})
			.collect();
		if !gpu.stream_prefetch(&needed, keep.clone(), block) && !block {
			drop(needed);
			*slot = Some((v, keep));
		}
	}

	/// GPU row fraction of a [`Self::matmul_q8_defer`] call (default 0.94):
	/// most of the matrix, because the CPU is busy with the caller's own work
	/// while the GPU runs it and only takes the last rows afterwards.
	/// `QWEN_DEFER_FRAC` overrides it; `QWEN_DEFER=0` turns deferral off.
	#[cfg(feature = "vulkan")]
	fn defer_frac() -> Option<f64> {
		static F: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
		*F.get_or_init(|| {
			if matches!(
				std::env::var("QWEN_DEFER").as_deref().map(str::trim),
				Ok("0") | Ok("off") | Ok("no") | Ok("false")
			) {
				return None;
			}
			Some(
				std::env::var("QWEN_DEFER_FRAC")
					.ok()
					.and_then(|v| v.trim().parse::<f64>().ok())
					.unwrap_or(0.94)
					.clamp(0.0, 1.0),
			)
		})
	}

	/// Whether [`Self::matmul_q8_defer`] would start `job` (`(id, weight,
	/// out_dim)`) rather than return `None`, so a caller can keep it in its
	/// group otherwise.
	pub fn defer_ok(
		&self,
		il: usize,
		job: &(WeightId, &quant::LinearWeight, usize),
		seq: usize,
		in_dim: usize,
	) -> bool {
		#[cfg(feature = "vulkan")]
		{
			let &(id, w, out_dim) = job;
			let Some(gpu) = self.gpu() else { return false };
			if Self::defer_frac().is_none() || self.split.is_empty() || seq <= self.decode_seq() {
				return false;
			}
			let kind = match w {
				quant::LinearWeight::Q4K { .. } => crate::gpu::GpuKind::Q4K,
				quant::LinearWeight::Q6K { .. } => crate::gpu::GpuKind::Q6K,
				_ => return false,
			};
			let j = BatchJobSeq {
				il,
				id,
				w,
				x: &[],
				seq,
				out_dim,
				in_dim,
			};
			matches!(self.prefill_split_rows(&j), Some((r, g)) if g > r) && gpu.wants_q8k(kind, seq)
		}
		#[cfg(not(feature = "vulkan"))]
		{
			let _ = (il, job, seq, in_dim);
			false
		}
	}

	/// Start a prefill matmul whose result the caller does not need yet, on a
	/// q8_K input: rows `[0, g)` are submitted to the GPU now (resident plus
	/// streamed, as in [`Self::matmul_split_prefill`], but with
	/// `g = QWEN_DEFER_FRAC * out`), and nothing runs on the CPU until
	/// [`Self::finish_deferred`], which computes rows `[g, out)` there and
	/// then waits. Between the two the caller's CPU work overlaps the GPU's
	/// share, which in a plain split call would have left the GPU idle.
	///
	/// `None` (the caller runs the matmul normally) unless the job takes the
	/// prefill split with streamed rows. Only one GPU batch may be in flight,
	/// so no other offloaded matmul may run before `finish_deferred`.
	pub fn matmul_q8_defer<'a>(
		&self,
		il: usize,
		id: WeightId,
		w: &'a quant::LinearWeight,
		out_dim: usize,
		q8: &std::sync::Arc<Vec<quant::Q8K>>,
		seq: usize,
		in_dim: usize,
	) -> Option<Deferred<'a>> {
		#[cfg(feature = "vulkan")]
		{
			let frac = Self::defer_frac()?;
			let j = BatchJobSeq {
				il,
				id,
				w,
				x: &[],
				seq,
				out_dim,
				in_dim,
			};
			let (r, g0) = self.prefill_split_rows(&j)?;
			// Streamed rows are allowed at this length (else a short chunk:
			// the plain split is the better route there).
			if g0 <= r {
				return None;
			}
			self.defer_begin(il, id, w, 0, out_dim, r, frac, q8, seq, in_dim, 0)
		}
		#[cfg(not(feature = "vulkan"))]
		{
			let _ = (il, id, w, out_dim, q8, seq, in_dim);
			None
		}
	}

	/// GPU row fraction of the attention gate's deferral
	/// ([`Self::matmul_q8_defer_rows`]; `QWEN_DEFER_QGATE_FRAC`, default
	/// **1.0**): the CPU's attention over a 2048-token chunk takes ~48 ms per
	/// layer and the GPU's whole gate ~23, so a CPU share would only add
	/// serial time after the attention.
	#[cfg(feature = "vulkan")]
	fn qgate_frac() -> f64 {
		static F: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
		*F.get_or_init(|| {
			std::env::var("QWEN_DEFER_QGATE_FRAC")
				.ok()
				.and_then(|v| v.trim().parse::<f64>().ok())
				.unwrap_or(1.0)
				.clamp(0.0, 1.0)
		})
	}

	/// Whether a full-attention layer's prefill may split `Wq` (`w`: `n_q`
	/// query rows, then `n_q` gate rows) into a query part that runs with
	/// `Wk` / `Wv` and a gate part deferred behind the attention
	/// ([`Self::matmul_q8_defer_rows`]): q8_K input, deferral on, the query
	/// rows taking the prefill split with the resident prefix inside them,
	/// and the attention staying on the CPU (the GPU attention split needs
	/// the device while the gate holds it). `keys` is the context length
	/// after the chunk.
	#[allow(clippy::too_many_arguments)]
	pub fn qgate_defer_ok(
		&self,
		il: usize,
		w: &quant::LinearWeight,
		n_q: usize,
		seq: usize,
		in_dim: usize,
		keys: usize,
		n_head: usize,
		n_kv: usize,
		hd: usize,
	) -> bool {
		#[cfg(feature = "vulkan")]
		{
			let Some(gpu) = self.gpu() else { return false };
			if Self::defer_frac().is_none() || self.split.is_empty() || seq <= self.decode_seq() {
				return false;
			}
			if self.attn_prefill_on(seq, keys, n_head, n_kv, hd) {
				return false;
			}
			let kind = match w {
				quant::LinearWeight::Q4K { .. } => crate::gpu::GpuKind::Q4K,
				quant::LinearWeight::Q6K { .. } => crate::gpu::GpuKind::Q6K,
				_ => return false,
			};
			// (`LinearWeight::out_dim` is the GGUF's `dims[0]`, not the row
			// count, so the rows are counted from the bytes.)
			let rb = in_dim / 256
				* if kind == crate::gpu::GpuKind::Q4K {
					144
				} else {
					210
				};
			let raw_len = match w {
				quant::LinearWeight::Q4K { raw, .. } | quant::LinearWeight::Q6K { raw, .. } => {
					raw.len()
				}
				_ => 0,
			};
			if !in_dim.is_multiple_of(256) || raw_len != 2 * n_q * rb {
				return false;
			}
			let j = BatchJobSeq {
				il,
				id: WeightId::Wq,
				w,
				x: &[],
				seq,
				out_dim: n_q,
				in_dim,
			};
			matches!(self.prefill_split_rows(&j), Some((r, g)) if g > r && r <= n_q)
				&& gpu.wants_q8k(kind, seq)
		}
		#[cfg(not(feature = "vulkan"))]
		{
			let _ = (il, w, n_q, seq, in_dim, keys, n_head, n_kv, hd);
			false
		}
	}

	/// [`Self::matmul_q8_defer`] for weight rows `[row0, row0 + n)` of `id`,
	/// a range past the decode split's resident prefix, so all of the GPU's
	/// share is streamed: the attention gate, rows `[n_q, 2 n_q)` of `Wq`
	/// (see [`Self::qgate_defer_ok`]). The result is `[seq x n]`.
	#[allow(clippy::too_many_arguments)]
	pub fn matmul_q8_defer_rows<'a>(
		&self,
		il: usize,
		id: WeightId,
		w: &'a quant::LinearWeight,
		row0: usize,
		n: usize,
		q8: &std::sync::Arc<Vec<quant::Q8K>>,
		seq: usize,
		in_dim: usize,
	) -> Option<Deferred<'a>> {
		#[cfg(feature = "vulkan")]
		{
			Self::defer_frac()?;
			let r = self.split.get(&(il, id)).copied()?;
			if row0 < r {
				return None;
			}
			self.defer_begin(
				il,
				id,
				w,
				row0,
				n,
				0,
				Self::qgate_frac(),
				q8,
				seq,
				in_dim,
				0,
			)
		}
		#[cfg(not(feature = "vulkan"))]
		{
			let _ = (il, id, w, row0, n, q8, seq, in_dim);
			None
		}
	}

	/// Submit rows `[row0, row0 + g)` of a deferred matmul to the GPU, `g =
	/// frac * n` but at least the `r` resident rows (which only a range at
	/// `row0 = 0` has); the CPU takes `[row0 + g, row0 + n)` in
	/// [`Self::finish_deferred`].
	#[cfg(feature = "vulkan")]
	#[allow(clippy::too_many_arguments)]
	fn defer_begin<'a>(
		&self,
		il: usize,
		id: WeightId,
		w: &'a quant::LinearWeight,
		row0: usize,
		n: usize,
		r: usize,
		frac: f64,
		q8: &std::sync::Arc<Vec<quant::Q8K>>,
		seq: usize,
		in_dim: usize,
		bank: usize,
	) -> Option<Deferred<'a>> {
		let gpu = self.gpu()?;
		let kind = match w {
			quant::LinearWeight::Q4K { .. } => crate::gpu::GpuKind::Q4K,
			quant::LinearWeight::Q6K { .. } => crate::gpu::GpuKind::Q6K,
			_ => return None,
		};
		let rb = in_dim / 256
			* if kind == crate::gpu::GpuKind::Q4K {
				144
			} else {
				210
			};
		let raw_len = match w {
			quant::LinearWeight::Q4K { raw, .. } | quant::LinearWeight::Q6K { raw, .. } => {
				raw.len()
			}
			_ => unreachable!(),
		};
		if !gpu.wants_q8k(kind, seq) || q8.len() != seq || (row0 + n) * rb > raw_len {
			return None;
		}
		debug_assert!(row0 == 0 || r == 0);
		let g = ((n as f64 * frac) as usize / 4 * 4).clamp(r, n);
		if g == 0 {
			return None;
		}
		let key = if row0 == 0 {
			Self::PSTREAM_KEY
		} else {
			Self::RSTREAM_KEY
		};
		self.defer_submit(
			il,
			&[(id, w, row0, n, r, g, key)],
			&[],
			q8,
			seq,
			in_dim,
			bank,
		)
	}

	/// Submit the GPU rows of a group of jobs on one q8_K input: per job
	/// `(id, w, row0, n, r, g, stream key)`, the resident rows `[0, r)` and
	/// the streamed `[row0 + r, row0 + g)`. `tail` jobs run on the CPU in the
	/// finish. `None` if the GPU could not take the batch.
	#[cfg(feature = "vulkan")]
	#[allow(clippy::type_complexity)]
	fn defer_submit<'a>(
		&self,
		il: usize,
		jobs: &[(
			WeightId,
			&'a quant::LinearWeight,
			usize,
			usize,
			usize,
			usize,
			usize,
		)],
		tail: &[(&'a quant::LinearWeight, usize)],
		q8: &std::sync::Arc<Vec<quant::Q8K>>,
		seq: usize,
		in_dim: usize,
		bank: usize,
	) -> Option<Deferred<'a>> {
		let gpu = self.gpu()?;
		let mut parts = Vec::with_capacity(jobs.len());
		let mut gjobs = Vec::with_capacity(2 * jobs.len());
		let mut dsts = Vec::with_capacity(2 * jobs.len());
		for &(id, w, row0, n, r, g, key) in jobs {
			let (kind, rb, raw) = match w {
				quant::LinearWeight::Q4K { raw, .. } => {
					(crate::gpu::GpuKind::Q4K, in_dim / 256 * 144, &raw[..])
				}
				quant::LinearWeight::Q6K { raw, .. } => {
					(crate::gpu::GpuKind::Q6K, in_dim / 256 * 210, &raw[..])
				}
				_ => return None,
			};
			let mut out = quant::uninit_vec(seq * n);
			let yp = out.as_mut_ptr() as usize;
			if r > 0 {
				gjobs.push(crate::gpu::SeqJob {
					il: il + Self::SPLIT_KEY,
					id,
					x: &[],
					seq,
					src: None,
					rows: Some(r),
					q8: Some(&q8[..]),
					dst: Some((yp, n, 0)),
				});
				dsts.push((yp, n, 0, r));
			}
			if g > r {
				gjobs.push(crate::gpu::SeqJob {
					il: il + key,
					id,
					x: &[],
					seq,
					src: Some((&raw[(row0 + r) * rb..(row0 + g) * rb], kind, g - r, in_dim)),
					rows: None,
					q8: Some(&q8[..]),
					dst: Some((yp, n, r)),
				});
				dsts.push((yp, n, r, g - r));
			}
			parts.push(DeferPart {
				w,
				row0,
				g,
				out_dim: n,
				out,
			});
		}
		let pending = if gjobs.is_empty() {
			None
		} else {
			Some((gpu.clone(), gpu.matmul_batch_seq_begin_bank(&gjobs, bank)?))
		};
		if bank == 0 {
			self.issue_prefetch(false);
		}
		Some(Deferred {
			pending,
			dsts,
			parts,
			tail: tail.to_vec(),
			seq,
			in_dim,
			q8: q8.clone(),
		})
	}

	/// Complete a [`Self::matmul_q8_defer`] call: the CPU rows, then the wait
	/// for the GPU's. Returns the `[seq x out]` matrix. A lost device
	/// recomputes the GPU rows on the CPU.
	pub fn finish_deferred(&self, d: Deferred) -> Vec<f32> {
		self.split_finish(d).swap_remove(0)
	}

	/// Start a prefill matmul group (`(id, weight, out_dim)` per job, one
	/// q8_K input) whose result the caller collects later with
	/// [`Self::split_finish`]: each job's GPU rows are submitted now, at the
	/// prefill split's fraction plus `extra`, and its CPU rows run in the
	/// finish. Between the two the caller's own CPU work overlaps the GPU's
	/// share (the two-half prefill pipeline, `Model::prefill_chunk_pipe`).
	/// Jobs that do not take the split run on the CPU in the finish, as does
	/// everything if the GPU cannot take the batch, so this never fails.
	/// One batch per `bank` (0 or 1) may be in flight; nothing else may
	/// submit while a bank-1 batch is.
	#[allow(clippy::too_many_arguments)]
	pub fn split_begin<'a>(
		&self,
		il: usize,
		jobs: &[(WeightId, &'a quant::LinearWeight, usize)],
		q8: &std::sync::Arc<Vec<quant::Q8K>>,
		seq: usize,
		in_dim: usize,
		extra: f64,
		bank: usize,
	) -> Deferred<'a> {
		#[cfg(feature = "vulkan")]
		{
			let mut gj = Vec::with_capacity(jobs.len());
			let mut tail = Vec::new();
			let mut ok = true;
			for &(id, w, out_dim) in jobs {
				let j = BatchJobSeq {
					il,
					id,
					w,
					x: &[],
					seq,
					out_dim,
					in_dim,
				};
				match self.prefill_split_rows_x(&j, extra) {
					// (A job may be a row prefix of its weight, the query rows
					// of `Wq`: the resident rows must lie inside it.)
					Some((r, g)) if r <= out_dim => {
						gj.push((id, w, 0, out_dim, r, g, Self::PSTREAM_KEY))
					}
					None if Self::cpu_tail_ok(&j) => tail.push((w, out_dim)),
					_ => ok = false,
				}
			}
			if ok {
				if let Some(d) = self.defer_submit(il, &gj, &tail, q8, seq, in_dim, bank) {
					return d;
				}
			}
		}
		let _ = (il, extra, bank);
		Self::deferred_cpu(
			jobs.iter().map(|&(_, w, od)| (w, 0, od)).collect(),
			q8,
			seq,
			in_dim,
		)
	}

	/// A [`Deferred`] that runs every row on the CPU in the finish.
	fn deferred_cpu<'a>(
		jobs: Vec<(&'a quant::LinearWeight, usize, usize)>,
		q8: &std::sync::Arc<Vec<quant::Q8K>>,
		seq: usize,
		in_dim: usize,
	) -> Deferred<'a> {
		let parts = jobs
			.into_iter()
			.map(|(w, row0, out_dim)| DeferPart {
				w,
				row0,
				g: 0,
				out_dim,
				out: quant::uninit_vec(seq * out_dim),
			})
			.collect();
		Deferred {
			#[cfg(feature = "vulkan")]
			pending: None,
			#[cfg(feature = "vulkan")]
			dsts: Vec::new(),
			parts,
			tail: Vec::new(),
			seq,
			in_dim,
			q8: q8.clone(),
		}
	}

	/// [`Self::split_begin`] for rows `[row0, row0 + n)` of one weight past
	/// the decode split's resident prefix (the attention gate rows of `Wq`),
	/// the GPU taking `frac` of them, all streamed.
	#[allow(clippy::too_many_arguments)]
	pub fn split_begin_rows<'a>(
		&self,
		il: usize,
		id: WeightId,
		w: &'a quant::LinearWeight,
		row0: usize,
		n: usize,
		q8: &std::sync::Arc<Vec<quant::Q8K>>,
		seq: usize,
		in_dim: usize,
		frac: f64,
		bank: usize,
	) -> Deferred<'a> {
		#[cfg(feature = "vulkan")]
		{
			if let Some(d) = self.defer_begin(il, id, w, row0, n, 0, frac, q8, seq, in_dim, bank) {
				return d;
			}
		}
		let _ = (il, id, frac, bank);
		Self::deferred_cpu(vec![(w, row0, n)], q8, seq, in_dim)
	}

	/// Complete a [`Deferred`]: every job's CPU rows and the tail jobs on this
	/// thread, then the wait for the GPU's rows. Outputs (`[seq x out]`) in
	/// job order, the tail jobs' last. A lost device recomputes the GPU rows
	/// on the CPU.
	pub fn split_finish(&self, d: Deferred) -> Vec<Vec<f32>> {
		#[cfg(feature = "vulkan")]
		let Deferred {
			pending,
			dsts,
			mut parts,
			tail,
			seq,
			in_dim,
			q8,
		} = d;
		#[cfg(not(feature = "vulkan"))]
		let Deferred {
			mut parts,
			tail,
			seq,
			in_dim,
			q8,
		} = d;
		#[cfg_attr(not(feature = "vulkan"), allow(unused_variables))]
		let t0 = std::time::Instant::now();
		for p in parts.iter_mut() {
			if p.g < p.out_dim {
				Self::gemm_q8(
					p.w,
					p.row0 + p.g,
					p.out_dim - p.g,
					in_dim,
					seq,
					&q8,
					&mut p.out,
					p.out_dim,
					p.g,
				);
			}
		}
		let mut tails = Vec::with_capacity(tail.len());
		for &(w, od) in &tail {
			let mut y = quant::uninit_vec(seq * od);
			Self::gemm_q8(w, 0, od, in_dim, seq, &q8, &mut y, od, 0);
			tails.push(y);
		}
		#[cfg(feature = "vulkan")]
		if pending.is_some() {
			self.issue_prefetch(true);
		}
		let t1 = std::time::Instant::now();
		#[cfg(feature = "vulkan")]
		if let Some((gpu, pending)) = pending {
			// SAFETY: the destinations are disjoint column blocks of the
			// parts' `out`s, which are `seq * out_dim` and whose heap blocks
			// have not moved since the begin.
			let ok = unsafe { gpu.matmul_batch_seq_end_into(pending, &dsts) }.is_some();
			if !ok {
				eprintln!("offload: GPU device lost during a deferred matmul; CPU for the rest");
				for p in parts.iter_mut() {
					Self::gemm_q8(p.w, p.row0, p.g, in_dim, seq, &q8, &mut p.out, p.out_dim, 0);
				}
			}
			if crate::gpu::gpu_timing() {
				let g: Vec<String> = parts
					.iter()
					.map(|p| format!("{}/{}", p.g, p.out_dim))
					.collect();
				eprintln!(
					"[DEFER] seq={seq} rows {} cpu {:7.2} ms  wait {:7.2} ms",
					g.join(","),
					(t1 - t0).as_secs_f64() * 1e3,
					t1.elapsed().as_secs_f64() * 1e3,
				);
			}
		}
		let _ = t1;
		let mut outs: Vec<Vec<f32>> = parts.into_iter().map(|p| p.out).collect();
		outs.extend(tails);
		outs
	}

	/// The whole-matrix route: each job entirely on the GPU (resident or
	/// streamed) where that is measured to be faster, else on the CPU.
	fn matmul_batch_seq_whole(&self, jobs: &[BatchJobSeq]) -> Vec<Vec<f32>> {
		// A job can reach the GPU two ways: its weight is **resident** (uploaded
		// at load for one of the `--n-gpu-layers` trailing layers), or it is
		// **streamed** into a VRAM slot for this call. Streaming is what lifts
		// prefill past the 22-of-65 layers that fit in 6 GiB.
		//
		// Both routes ship **on**, so this predicate is no longer "is the GPU
		// path available" but "is the GPU path *faster here*". The two
		// thresholds are measured crossovers (`gpubench --crossover`, see
		// `GpuDevice::prefill_min_seq` / `stream_min_seq`), and they differ
		// because a streamed weight also pays its upload. Below them the GPU is
		// not marginally worse but 5-8x worse (there is no tiled GEMM under
		// `GEMM_BN` tokens and the batched GEMV re-reads every weight row per
		// position) so getting this wrong in the permissive direction is far
		// more expensive than getting it wrong in the conservative one.
		//
		// Decode (`seq == 1`) is deliberately untouched: it keeps the old
		// resident-only rule and its own kernels.
		let resident = |j: &BatchJobSeq| -> bool {
			if !(self.is_offloaded(j.il) && self.gpu_present(j.il, j.id)) {
				return false;
			}
			#[cfg(feature = "vulkan")]
			{
				if j.seq == 1 {
					return true;
				}
				// `stream_src` is doing double duty here: for a resident job
				// its bytes are not needed, but it is already the one place
				// that maps a `LinearWeight` to a `GpuKind` and rejects the
				// shapes no GPU kernel can take.
				match (self.gpu(), Self::stream_src(j)) {
					(Some(g), Some((_, kind, _, _))) => g.prefill_worth_it(kind, j.seq),
					_ => false,
				}
			}
			#[cfg(not(feature = "vulkan"))]
			{
				true
			}
		};
		let streamable = |j: &BatchJobSeq| -> bool {
			#[cfg(feature = "vulkan")]
			{
				j.seq > 1
					&& crate::gpu::GpuDevice::stream_enabled()
					&& j.seq >= crate::gpu::GpuDevice::stream_min_seq()
					&& Self::stream_src(j).is_some()
			}
			#[cfg(not(feature = "vulkan"))]
			{
				let _ = j;
				false
			}
		};
		if !jobs.is_empty() && self.available && jobs.iter().all(|j| resident(j) || streamable(j)) {
			#[cfg(feature = "vulkan")]
			{
				if let Some(gpu) = self.gpu() {
					// Decode path (every job is seq == 1) reuses each weight's
					// persistent, pre-allocated buffers: no per-submission
					// allocate/free churn, which is what keeps the GPU resident
					// across a whole trailing block of offloaded layers. Prefill
					// (seq > 1) uses the per-call path.
					let all_seq1 = jobs.iter().all(|j| j.seq == 1);
					let gpu_res = if all_seq1 {
						let slice: Vec<(usize, WeightId, &[f32])> =
							jobs.iter().map(|j| (j.il, j.id, j.x)).collect();
						gpu.matmul_batch(&slice)
					} else {
						let gpu_jobs: Vec<crate::gpu::SeqJob> = jobs
							.iter()
							.map(|j| crate::gpu::SeqJob {
								il: j.il,
								id: j.id,
								x: j.x,
								seq: j.seq,
								src: Self::stream_src(j),
								rows: None,
								q8: None,
								dst: None,
							})
							.collect();
						gpu.matmul_batch_seq(&gpu_jobs)
					};
					match gpu_res {
						Some(gpu_res) => {
							if std::env::var("QWEN_GPUCMP").is_ok() {
								for (i, j) in jobs.iter().enumerate() {
									let cpu = quant::apply_linear_batch(
										j.w, j.x, j.seq, j.out_dim, j.in_dim,
									);
									let wty = match j.w {
										quant::LinearWeight::Q4K { .. } => "Q4K",
										quant::LinearWeight::Q6K { .. } => "Q6K",
										quant::LinearWeight::F16(_) => "F16",
										quant::LinearWeight::F32(_) => "F32",
									};
									let gd = &gpu_res[i];
									let mut md = 0.0f32;
									let mut ca = 0.0f32;
									for k in 0..gd.len() {
										md = md.max((gd[k] - cpu[k]).abs());
										ca = ca.max(cpu[k].abs());
									}
									eprintln!(
										"OFFCMP L{} {:?} ty={} seq={} maxabsdiff={:.6} cpuabsmax={:.4} gpuabsmax={:.4} glen={} clen={}",
										j.il, j.id, wty, j.seq, md, ca,
										gd.iter().cloned().fold(0.0f32, f32::max),
										gd.len(), cpu.len()
									);
								}
							}
							return gpu_res;
						}
						// Device lost: transparently fall back to the CPU path so
						// generation continues instead of taking the process down.
						None => {
							eprintln!(
								"offload: GPU device lost during matmul; falling back to CPU for remaining layers"
							);
						}
					}
				}
			}
		}
		Self::cpu_fallback(jobs)
	}

	/// The CPU path for a batch the GPU declined, matching **exactly** what
	/// `Model::lin` / `lin_batch` / `lin_seq` would have run with no manager at
	/// all.
	///
	/// This used to be `apply_linear_batch` for every job regardless of `seq`,
	/// which was invisible while the manager only existed when the user asked
	/// for `--n-gpu-layers > 0`. Prefill routing ships on, so the manager is now
	/// constructed on every run, and at `seq == 1` that meant decode silently
	/// left the tuned GEMV (`apply_linear`, and `apply_linear_group` for a
	/// shared-`x` batch: one fork-join and one q8_K quantization for the whole
	/// group) for the batched GEMM. It cost ~5 % of decode and it moved the
	/// logits: `--forward 369` reported `DCR SUM -1028443.176329` where every
	/// reference in this repo says `-1034584.732451`. The argmax was unchanged,
	/// so nothing failed, which is exactly why the fallback has to be the same
	/// kernel rather than merely a correct one.
	/// Decode groups of [`WeightId::exact_decode`] weights run on the CPU
	/// with an f32 activation, whatever the placement, as `Model::lin_batch`
	/// does without a manager. `QWEN_EXACT_DECODE=1` sends every K-quant decode
	/// matmul this way: slow, but an f32-activation reference for accuracy
	/// A/Bs, where a sum of logits (`DCR SUM`) is too noisy to rank two
	/// variants. `None` for anything else.
	fn exact_decode(jobs: &[BatchJobSeq]) -> Option<Vec<Vec<f32>>> {
		if jobs.is_empty() || jobs.iter().any(|j| j.seq != jobs[0].seq) {
			return None;
		}
		// A verify batch (`seq <= SPEC_MAX_SEQ`) is decode at a few positions:
		// same weights, same exact path, one position at a time.
		let seq = jobs[0].seq;
		if seq > 1 {
			if seq > SPEC_MAX_SEQ || !jobs.iter().all(|j| j.id.exact_decode()) {
				return None;
			}
			let mut out: Vec<Vec<f32>> = jobs
				.iter()
				.map(|j| Vec::with_capacity(seq * j.out_dim))
				.collect();
			for t in 0..seq {
				let row: Vec<BatchJobSeq> = jobs
					.iter()
					.map(|j| BatchJobSeq {
						x: &j.x[t * j.in_dim..(t + 1) * j.in_dim],
						seq: 1,
						..j.clone()
					})
					.collect();
				// Rows of one call share `x` again, as `exact_decode` requires.
				let r = Self::exact_decode(&row)?;
				for (o, v) in out.iter_mut().zip(r) {
					o.extend_from_slice(&v);
				}
			}
			return Some(out);
		}
		if !jobs.iter().all(|j| j.id.exact_decode()) && !quant::exact_decode_forced() {
			return None;
		}
		let kquant = jobs.iter().all(|j| {
			matches!(
				j.w,
				quant::LinearWeight::Q4K { .. } | quant::LinearWeight::Q6K { .. }
			)
		});
		let shared = jobs
			.windows(2)
			.all(|w| std::ptr::eq(w[0].x, w[1].x) && w[0].in_dim == w[1].in_dim);
		if !kquant || !shared || !jobs[0].in_dim.is_multiple_of(quant::QK_K) {
			return None;
		}
		let g: Vec<(&quant::LinearWeight, usize)> = jobs.iter().map(|j| (j.w, j.out_dim)).collect();
		Some(quant::apply_linear_group_exact(
			&g,
			jobs[0].x,
			jobs[0].in_dim,
		))
	}

	fn cpu_fallback(jobs: &[BatchJobSeq]) -> Vec<Vec<f32>> {
		if jobs.iter().any(|j| j.seq != 1) {
			return jobs
				.iter()
				.map(|j| quant::apply_linear_batch(j.w, j.x, j.seq, j.out_dim, j.in_dim))
				.collect();
		}
		let shared = jobs.len() > 1
			&& jobs
				.windows(2)
				.all(|w| std::ptr::eq(w[0].x, w[1].x) && w[0].in_dim == w[1].in_dim);
		if shared {
			let g: Vec<(&quant::LinearWeight, usize)> =
				jobs.iter().map(|j| (j.w, j.out_dim)).collect();
			return quant::apply_linear_group(&g, jobs[0].x, jobs[0].in_dim);
		}
		jobs.iter()
			.map(|j| quant::apply_linear(j.w, j.x, j.out_dim, j.in_dim))
			.collect()
	}

	/// L2/L3 GPU-resident fused full-attention layer (decode, single token). Runs
	/// the whole layer on the GPU keeping the activation in VRAM, returning the
	/// layer output. Returns `None` (CPU fallback) when no adapter is present or
	/// the device was lost. See `GpuDevice::full_attn_block`.
	pub fn full_attn_block(&self, il: usize, x: &[f32], info: &GpuBlockInfo) -> Option<Vec<f32>> {
		if !self.available || !self.is_offloaded(il) {
			return None;
		}
		#[cfg(feature = "vulkan")]
		{
			if let Some(gpu) = self.gpu() {
				return gpu.full_attn_block(il, x, info);
			}
		}
		let _ = (il, x, info);
		None
	}

	/// L2/L3 GPU-resident fused SSM layer (decode, single token). Runs the whole
	/// SSM layer (conv1d → delta-net → gate → SsmOut → FFN) on the GPU with the
	/// recurrent state kept resident in VRAM, returning the layer output.
	/// Returns `None` (CPU fallback) when no adapter is present or the device was
	/// lost. See `GpuDevice::ssm_block`.
	pub fn ssm_block(&self, il: usize, x: &[f32], info: &GpuBlockInfo) -> Option<Vec<f32>> {
		if !self.available || !self.is_offloaded(il) {
			return None;
		}
		#[cfg(feature = "vulkan")]
		{
			if let Some(gpu) = self.gpu() {
				return gpu.ssm_block(il, x, info);
			}
		}
		let _ = (il, x, info);
		None
	}

	/// Run the ALU-ceiling probe (`gpu/bench.rs`). Returns `(seconds, ops, sink)`
	/// for one dispatch, or `None` without a Vulkan device.
	///
	/// This measures what the card's arithmetic units can sustain with no global
	/// memory traffic in the inner loop, i.e. an upper bound on any kernel that
	/// could be written for it. It is a planning gate, used by
	/// `examples/gpubench.rs --gate`; nothing in the forward pass calls it.
	#[cfg(feature = "vulkan")]
	pub fn alu_probe(
		&self,
		mode: u32,
		tile: u32,
		inner: u32,
		groups: u32,
		iters: usize,
	) -> Option<(f64, f64, f32)> {
		let g = self.gpu()?;
		let r = g.alu_probe(mode, tile, inner, groups, iters);
		Some((r.secs, r.ops, r.sink))
	}

	/// Time the prefill GEMM kernel alone (`gpu/bench.rs::gemm_probe`), with the
	/// per-call allocation, the activation quantization and the readback that
	/// `matmul_batch_seq` pays taken out of the number. Returns
	/// `(seconds_per_dispatch, first min(seq,64) tokens of y)`.
	///
	/// `spv` overrides the pipeline with SPIR-V read from disk, so a `.glsl`
	/// edit can be measured with `glslc` + rerun and no Rust rebuild; `gemv`
	/// runs the batched-GEMV kernel on the same buffers for an A/B. Bench entry
	/// point only: nothing in the forward pass calls it.
	#[cfg(feature = "vulkan")]
	#[allow(clippy::too_many_arguments)]
	pub fn gemm_probe(
		&self,
		il: usize,
		id: WeightId,
		x: &[f32],
		seq: usize,
		out_dim: usize,
		in_dim: usize,
		reps: usize,
		spv: Option<&str>,
		gemv: bool,
	) -> Option<(f64, Vec<f32>)> {
		let g = self.gpu()?;
		let kind = g.weight_kind(il, id)?;
		let pk = if gemv && !g.wants_q8k(kind, 1) {
			crate::gpu::GpuKind::F32
		} else {
			crate::gpu::GpuKind::Q4K
		};
		let packed = crate::gpu::GpuDevice::pack_x_seq(pk, in_dim, x, seq);
		let r = g.gemm_probe(il, id, &packed, seq, out_dim, in_dim, reps, spv, gemv, None);
		Some((r.secs, r.head))
	}

	/// The prefill delta-net scan alone (`GpuDevice::delta_probe`). Bench
	/// entry point only.
	#[cfg(feature = "vulkan")]
	#[allow(clippy::too_many_arguments)]
	pub fn delta_probe(
		&self,
		n: usize,
		hk: usize,
		hv: usize,
		st0: &[f32],
		k: &[f32],
		q: &[f32],
		v: &[f32],
		bg: &[f32],
		lanes: usize,
		reps: usize,
		spv: Option<&str>,
	) -> Option<(f64, Vec<f32>, Vec<f32>)> {
		let g = self.gpu()?;
		Some(g.delta_probe(n, hk, hv, st0, k, q, v, bg, lanes, reps, spv))
	}

	/// The NPU prefill's recurrence on the GPU (`GpuDevice::delta_npu_probe`).
	/// Bench entry point only.
	#[cfg(feature = "vulkan")]
	#[allow(clippy::too_many_arguments)]
	pub fn delta_npu_probe(
		&self,
		n: usize,
		st0: &[f32],
		qn: &[u8],
		kn: &[u8],
		v: &[u8],
		gb: &[u8],
		oscale: f32,
		lanes: usize,
		reps: usize,
		spv: Option<&str>,
	) -> Option<(f64, Vec<u8>, Vec<f32>)> {
		let g = self.gpu()?;
		Some(g.delta_npu_probe(n, st0, qn, kn, v, gb, oscale, lanes, reps, spv))
	}

	/// Time a decode-shaped (GEMV) kernel alone, like `gemm_probe` but with the
	/// activation format and grid chosen by the caller, so a candidate `.spv`
	/// that reads **raw f32** activations (instead of q8_K) can be measured on
	/// the same resident weight. `rows_per_wg` sets `grid.x = ceil(out/rows)`,
	/// `grid.y = seq`. `x_f32: None` uses the shipped kernel's format.
	/// Bench entry point only.
	#[cfg(feature = "vulkan")]
	#[allow(clippy::too_many_arguments)]
	pub fn gemv_probe(
		&self,
		il: usize,
		id: WeightId,
		x: &[f32],
		seq: usize,
		out_dim: usize,
		in_dim: usize,
		reps: usize,
		spv: Option<&str>,
		x_f32: Option<bool>,
		rows_per_wg: Option<u32>,
	) -> Option<(f64, Vec<f32>)> {
		let g = self.gpu()?;
		// `None` = whatever the shipped GEMV for this weight reads.
		let kind = g.weight_kind(il, id)?;
		let x_f32 = x_f32.unwrap_or_else(|| !g.wants_q8k(kind, 1));
		let packed: Vec<u8> = if x_f32 {
			x[..seq * in_dim]
				.iter()
				.flat_map(|v| v.to_le_bytes())
				.collect()
		} else {
			crate::gpu::GpuDevice::pack_x_seq(crate::gpu::GpuKind::Q4K, in_dim, x, seq)
		};
		let grid = rows_per_wg.map(|r| (out_dim.div_ceil(r as usize) as u32, seq as u32));
		let r = g.gemm_probe(il, id, &packed, seq, out_dim, in_dim, reps, spv, true, grid);
		Some((r.secs, r.head))
	}

	/// The quantized bytes behind a job's weight, for the streaming path.
	///
	/// Only `Q4_K`/`Q6_K` qualify: they are the big matmuls, they are what the
	/// tiled GEMM kernels read, and they are the only variants that keep their
	/// GGUF bytes in host RAM (`f16`/`f32` weights were expanded at load, so
	/// streaming them would ship 2-4x the bytes for a matmul the CPU already
	/// does well).
	///
	/// The dimensions come from the **job**, not from the `LinearWeight`: that
	/// struct's `out_dim`/`in_dim` are `tensor_raw`'s `(dims[0], dims[1])`, and
	/// ggml stores `ne[0]` as the contiguous axis (the *input* dim) so those
	/// two fields are transposed relative to their names. Nothing noticed
	/// because every CPU kernel is passed the dims explicitly; taking them from
	/// the weight here read `x` as `seq * 17408` floats instead of
	/// `seq * 5120` and segfaulted inside `quantize_q8k`. The `raw.len()` check
	/// below is the guard: a shape that does not match the bytes falls back to
	/// the CPU rather than dispatching against a wrong-sized buffer.
	#[cfg(feature = "vulkan")]
	fn stream_src<'a>(
		j: &BatchJobSeq<'a>,
	) -> Option<(&'a [u8], crate::gpu::GpuKind, usize, usize)> {
		const QK_K: usize = 256;
		let (raw, kind, blk) = match j.w {
			quant::LinearWeight::Q4K { raw, .. } => (raw, crate::gpu::GpuKind::Q4K, 144),
			quant::LinearWeight::Q6K { raw, .. } => (raw, crate::gpu::GpuKind::Q6K, 210),
			_ => return None,
		};
		if !j.in_dim.is_multiple_of(QK_K) || raw.len() != j.out_dim * j.in_dim / QK_K * blk {
			return None;
		}
		Some((raw.as_slice(), kind, j.out_dim, j.in_dim))
	}

	/// Drain the weight-streaming counters: `(bytes, pack_secs, copy_secs,
	/// slot_allocs)`. `gpubench --stream` uses it to separate the host repack
	/// into the staging map from the submit + DMA that follows.
	pub fn stream_take_stats(&self) -> (u64, f64, f64, u64) {
		#[cfg(feature = "vulkan")]
		{
			match self.gpu() {
				Some(g) => g.stream_take_stats(),
				None => (0, 0.0, 0.0, 0),
			}
		}
		#[cfg(not(feature = "vulkan"))]
		{
			(0, 0.0, 0.0, 0)
		}
	}

	/// Number of layers configured for offload (may exceed the GPU-present
	/// count if adapter init failed).
	pub fn n_gpu_layers(&self) -> usize {
		self.n_gpu_layers
	}

	/// True when the active Vulkan device is the software `llvmpipe` ICD (used by
	/// the WSL CI). On llvmpipe the device-init-time `aux_layout` mis-binds
	/// descriptor sets, so the production fused-block paths (full-attn / SSM)
	/// are skipped there and validated on real hardware instead.
	pub fn is_llvmpipe(&self) -> bool {
		#[cfg(feature = "vulkan")]
		{
			self.gpu().map(|g| g.is_llvmpipe()).unwrap_or(false)
		}
		#[cfg(not(feature = "vulkan"))]
		{
			false
		}
	}
}

/// KV groups the prefill attention gives the GPU (`QWEN_GPU_ATTN_GROUPS`,
/// `0` = all on the CPU); see [`OffloadManager::attn_prefill`].
#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
fn attn_gpu_groups() -> usize {
	static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
	*N.get_or_init(|| {
		std::env::var("QWEN_GPU_ATTN_GROUPS")
			.ok()
			.and_then(|s| s.trim().parse().ok())
			.unwrap_or(2)
	})
}

/// Context length (keys) from which a prefill chunk's attention is split
/// onto the GPU (`QWEN_GPU_ATTN_MIN_KEYS`): below it the K/V upload and
/// packing cost more than the GPU saves.
#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
fn attn_min_keys() -> usize {
	static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
	*N.get_or_init(|| {
		std::env::var("QWEN_GPU_ATTN_MIN_KEYS")
			.ok()
			.and_then(|s| s.trim().parse().ok())
			.unwrap_or(4096)
	})
}

/// Share of the GPU groups' attention work the GPU keeps
/// (`QWEN_GPU_ATTN_FRAC`, default 0.88); the CPU takes the rest as those
/// groups' last query rows, so both sides finish together. With all of it
/// on the GPU its two groups finished 26-28 ms after the CPU's two per call
/// at 12-19k keys (4.6 s of a 19k-token prefill); 0.88 leaves ~0.3 ms.
/// See [`attn_gpu_rows`].
#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
fn attn_gpu_frac() -> f64 {
	static F: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
	*F.get_or_init(|| {
		std::env::var("QWEN_GPU_ATTN_FRAC")
			.ok()
			.and_then(|s| s.trim().parse().ok())
			.unwrap_or(0.88f64)
			.clamp(0.0, 1.0)
	})
}

/// Query rows `[0, r)` of the GPU groups that the GPU runs: the causal work
/// of rows `[0, r)` (row `t` sees `t0 + t + 1` keys) is `frac` of all `nq`
/// rows'. Rounded down to 8 (the CPU's query tile); `nq` at `frac` 1.
#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
pub(crate) fn attn_gpu_rows(nq: usize, t0: usize, frac: f64) -> usize {
	if frac >= 1.0 {
		return nq;
	}
	// r * (t0 + r / 2) = frac * nq * (t0 + nq / 2), solved for r.
	let (a, w) = (t0 as f64, frac * nq as f64 * (t0 as f64 + nq as f64 / 2.0));
	let r = -a + (a * a + 2.0 * w).sqrt();
	((r as usize) / 8 * 8).min(nq)
}

impl OffloadManager {
	/// Whether [`Self::attn_prefill`] would put part of a chunk's attention
	/// on the GPU (`nq` queries; `keys` = context length after the chunk).
	pub fn attn_prefill_on(
		&self,
		nq: usize,
		keys: usize,
		n_head: usize,
		n_kv: usize,
		hd: usize,
	) -> bool {
		#[cfg(feature = "vulkan")]
		{
			attn_gpu_groups().min(n_kv) > 0
				&& hd == crate::gpu::ATTN_HD
				&& n_head == n_kv * crate::gpu::ATTN_HPG
				&& nq > SPEC_MAX_SEQ
				&& keys >= attn_min_keys()
				&& self.gpu().is_some()
		}
		#[cfg(not(feature = "vulkan"))]
		{
			let _ = (nq, keys, n_head, n_kv, hd);
			false
		}
	}

	/// Causal GQA attention for a prefill chunk (`quant::attn_causal`'s
	/// contract, f16 cache): the first [`attn_gpu_groups`] KV groups run on
	/// the GPU (`shaders/attn_prefill.glsl`) while the CPU runs the rest. The
	/// GPU is otherwise idle here: attention sits between the QKV and Wo
	/// matmuls. Returns `false`, having written nothing, when the split does
	/// not apply; the caller then runs `quant::attn_causal` itself.
	#[allow(clippy::too_many_arguments)]
	pub fn attn_prefill(
		&self,
		q: &[f32],
		k: &[u16],
		v: &[u16],
		nq: usize,
		t0: usize,
		n_head: usize,
		n_kv: usize,
		hd: usize,
		scale: f32,
		out: &mut [f32],
	) -> bool {
		#[cfg(feature = "vulkan")]
		{
			let ng = attn_gpu_groups().min(n_kv);
			if !self.attn_prefill_on(nq, t0 + nq, n_head, n_kv, hd) {
				return false;
			}
			let Some(gpu) = self.gpu() else { return false };
			// The GPU runs query rows [0, qg) of groups [0, ng); the CPU the
			// rest of those groups' rows and every row of the other groups.
			// A query row only sees keys up to its own position, so a row
			// range is the same call on `q` / `out` offset by `qg` rows with
			// `t0 + qg`.
			let qg = attn_gpu_rows(nq, t0, attn_gpu_frac());
			if qg == 0 {
				return false;
			}
			let qs = n_head * hd;
			let t = std::time::Instant::now();
			let Some(p) =
				gpu.attn_prefill_begin(&q[..qg * qs], k, v, qg, t0, n_head, n_kv, scale, 0, ng)
			else {
				return false;
			};
			let t1 = std::time::Instant::now();
			quant::attn_causal_groups(q, k, v, nq, t0, n_head, n_kv, hd, scale, ng..n_kv, out);
			if qg < nq {
				let (qt, ot) = (&q[qg * qs..], &mut out[qg * qs..]);
				quant::attn_causal_groups(
					qt,
					k,
					v,
					nq - qg,
					t0 + qg,
					n_head,
					n_kv,
					hd,
					scale,
					0..ng,
					ot,
				);
			}
			let t2 = std::time::Instant::now();
			if gpu.attn_prefill_end(p, out, n_head, 0).is_none() {
				quant::attn_causal_groups(
					&q[..qg * qs],
					k,
					v,
					qg,
					t0,
					n_head,
					n_kv,
					hd,
					scale,
					0..ng,
					out,
				);
			}
			if crate::gpu::gpu_timing() {
				eprintln!(
					"[ATTN] nq={nq} gpu_rows={qg} keys={} gpu_groups={ng} begin {:7.2} ms  cpu {:7.2} ms  end {:7.2} ms",
					t0 + nq,
					(t1 - t).as_secs_f64() * 1e3,
					(t2 - t1).as_secs_f64() * 1e3,
					t2.elapsed().as_secs_f64() * 1e3,
				);
			}
			return true;
		}
		#[allow(unreachable_code)]
		{
			let _ = (q, k, v, nq, t0, n_head, n_kv, hd, scale, out);
			false
		}
	}
}
