//! Resident one-position GEMVs for the Flash-Next decode's dense layers on a
//! discrete GPU (`npu::flash::gpudec`).
//!
//! WHY
//! ---
//! A Flash-Next decode step reads ~3.7 GB of dense weights (int8 mixers and
//! hyper-connections, f16 routers); on the NPU that took 131 of 182 ms. They
//! fit in the GTX 1060's 6 GiB and its 192 GB/s, so every dense matrix lives
//! in VRAM and the host runs the cheap elementwise work between GEMVs. A
//! [`Gemv::run`] is one submission of independent GEMVs on one input
//! region: the host's activation is copied to VRAM first, the results are
//! written straight into a host-visible buffer.
//!
//! Matrices are sub-allocated from blocks of at most 256 MB (hundreds of
//! separate allocations next to gigabytes of weights lost the device once),
//! first fit, a new block sized to what `need` has left; each carries its
//! own descriptor set and parameters, so a run only records dispatches.

use super::*;

/// The largest VRAM block for the matrices.
const BLOCK: u64 = 256 << 20;
/// The smallest one, and its rounding.
const BLOCK_MIN: u64 = 4 << 20;
/// Host input / output regions, in floats.
pub const GEMV_X: usize = 65536;
pub const GEMV_Y: usize = 262144;
/// Positions one run takes ([`Gemv::run_chain_n`]): position `p` has its
/// own input region at `p GEMV_X` and output region at `p GEMV_Y` (the
/// kernels' `NP`, `XS`, `YS`).
pub const GEMV_NP: usize = 8;
/// Position slots of a wave ([`Gemv::set_wave`]): wave 1 runs on slots
/// `[GEMV_WAVE, GEMV_NP)` of the same regions, with its own fence, so one
/// run of each wave can be in flight at once.
pub const GEMV_WAVE: usize = GEMV_NP / 2;
/// Staging buffer for the uploads (the largest matrix: 12288 x 2560 int8).
const STAGE: usize = 64 << 20;
const MAXMATS: usize = 2048;
/// Rows per lane group of the kernels (their `NR`, `gemv_q8.glsl`'s too).
const NR: u32 = 4;

/// A matrix to upload: int8 codes with a row scale, or f16 (scale 1).
/// `Q8` is `I8` for an input that only ever holds integers in [-128, 127]
/// (a quantized matrix's rounded input): its GEMVs of 2-8 positions run
/// on `dp4a` (`shaders/gemv_q8.glsl`) where the device has it.
pub enum GemvMat<'a> {
	I8 { w: &'a [i8], s: &'a [f32] },
	Q8 { w: &'a [i8], s: &'a [f32] },
	F16 { w: &'a [u16] },
}

struct Mat {
	/// pipeline: 0 int8, 1 f16, 2 elementwise ([`Gemv::add_act`]), 3 vector
	/// steps ([`Gemv::add_vec`]), 4 / 5 the one-position builds of 0 / 1,
	/// 6 int8 on an int8 input (`dp4a`, 4 at one position), 7 a delta
	/// mixer step ([`Gemv::add_delta`])
	kind: usize,
	/// workgroups of a run
	groups: u32,
	/// end of its output rows
	yend: usize,
	/// per wave: the input / output regions bound at the wave's first
	/// position slot
	set: [vk::DescriptorSet; 2],
	/// what `QWEN_GEMV_PROF` reports it as
	label: String,
}

/// `QWEN_GEMV_PROF=1`: GPU timestamps around every stage of every run,
/// summed per (stage, positions) and printed when the `Gemv` is dropped.
struct Prof {
	qp: vk::QueryPool,
	/// ns per timestamp tick
	period: f64,
	/// the labels of the stages of the run in flight (input copy first,
	/// output copy last)
	run: Vec<String>,
	/// the wave whose run in flight writes the queries
	owner: Option<usize>,
	sums: std::collections::HashMap<String, (u64, f64)>,
}

const PROF_Q: u32 = 64;
/// Recorded runs [`Gemv`] keeps for submitting again (a decode step has
/// ~125 distinct runs per position count and wave; past this many the rest
/// are recorded each time).
const CMD_CACHE_MAX: usize = 8192;

/// See the module docs.
pub struct Gemv {
	dev: GpuDevice,
	/// int8, f16, elementwise, vector steps, the int8 / f16 GEMVs'
	/// one-position builds (`_1`), the `dp4a` GEMV, the delta mixer, then
	/// the int8 / f16 / `dp4a` GEMVs' builds for 2 and for 3-4 positions
	/// (`_2`, `_4`)
	sm: [vk::ShaderModule; 14],
	pl: [vk::Pipeline; 14],
	/// the aux set layout plus a push constant: the run's positions
	layout: vk::PipelineLayout,
	pool: vk::DescriptorPool,
	bufs: Vec<(vk::Buffer, vk::DeviceMemory)>,
	/// VRAM blocks: buffer, size, bytes used
	blocks: Vec<(vk::Buffer, u64, u64)>,
	/// the bytes [`Self::new`] was asked for: sizes the last blocks
	need: u64,
	params: vk::Buffer,
	stage: (vk::Buffer, *mut u8),
	xin: (vk::Buffer, *mut f32),
	xdev: vk::Buffer,
	ydev: vk::Buffer,
	yout: (vk::Buffer, *const f32),
	mats: Vec<Mat>,
	align: u64,
	bytes: u64,
	/// per wave: the submission [`Gemv::run_chain_n_begin`] left in flight,
	/// and whether [`Gemv::cmds`] owns it (else it is freed once waited for)
	pending: [Option<(vk::CommandBuffer, bool)>; 2],
	/// recorded runs by everything their commands depend on (wave,
	/// positions, ordering barrier, copy ranges, ops), submitted again
	/// instead of recorded again; `None`: `QWEN_GEMV_CMDCACHE=0`
	cmds: Option<std::collections::HashMap<Vec<u64>, vk::CommandBuffer>>,
	/// the wave the runs, waits and outputs refer to ([`Gemv::set_wave`])
	wave: usize,
	/// wave 0's run in flight takes slots past [`GEMV_WAVE`]
	wide: bool,
	/// per wave: the delta mixer's first aux slot ([`Gemv::set_wave`])
	p0: [usize; 2],
	/// microseconds a wait spins before it blocks (`QWEN_GEMV_SPIN_US`)
	spin_us: u64,
	/// the delta mixers' states and aux regions ([`Gemv::delta_init`])
	delta: Option<(vk::Buffer, vk::Buffer, usize)>,
	prof: Option<Prof>,
	/// `QWEN_GEMV_NPB` (default on): runs of 2-4 positions on the GEMV
	/// builds for that many
	npb: bool,
	/// the next run's input and output float ranges ([`Gemv::io_next`]);
	/// `ranges`: `QWEN_GEMV_RANGES` (default on; `0` copies everything)
	io: Option<(Vec<std::ops::Range<usize>>, Vec<std::ops::Range<usize>>)>,
	ranges: bool,
}

/// A delta mixer's state and its backup, in floats (`gemv_delta.glsl`).
pub const DELTA_STATE: usize = 2 * 48 * 128 * 128;
/// A delta mixer's conv window `[3][10240]`, first of its aux region.
pub const DELTA_WIN: usize = 3 * 10240;
/// A delta mixer's aux region: window, saved windows, recurrence inputs.
const DELTA_AUX: usize = DELTA_WIN * 10 + 8 * (10240 + 96);

// SAFETY: used from one thread at a time (the engine's).
unsafe impl Send for Gemv {}

impl Gemv {
	/// On the Vulkan adapter `pick` names; `need` bytes of matrices must fit
	/// the device's memory budget less `reserve`.
	pub fn new(pick: Option<&str>, need: u64, reserve: u64) -> Result<Gemv, String> {
		let dev = GpuDevice::init_pick(pick)?;
		if let Some((budget, usage)) = dev.vram_budget() {
			let free = budget.saturating_sub(usage).saturating_sub(reserve);
			if free < need {
				return Err(format!(
					"gemv: {} MB of VRAM free, the dense weights need {} MB",
					free >> 20,
					need >> 20
				));
			}
		}
		let d = &dev.device;
		let main = CString::new("main").unwrap();
		// without a (working) integer dot product the int8 kernel takes Q8
		let dp4a = dev.pipeline_gemm_q4k_dp4a.is_some();
		let q8: &[u8] = if dp4a {
			quant::packed!("gemv_q8.spv")
		} else {
			quant::packed!("gemv_i8.spv")
		};
		let (q8_2, q8_4): (&[u8], &[u8]) = if dp4a {
			(
				quant::packed!("gemv_q8_2.spv"),
				quant::packed!("gemv_q8_4.spv"),
			)
		} else {
			(
				quant::packed!("gemv_i8_2.spv"),
				quant::packed!("gemv_i8_4.spv"),
			)
		};
		let sm = [
			create_sm(d, &load_spv(quant::packed!("gemv_i8.spv")))?,
			create_sm(d, &load_spv(quant::packed!("gemv_f16.spv")))?,
			create_sm(d, &load_spv(quant::packed!("gemv_act.spv")))?,
			create_sm(d, &load_spv(quant::packed!("gemv_vec.spv")))?,
			create_sm(d, &load_spv(quant::packed!("gemv_i8_1.spv")))?,
			create_sm(d, &load_spv(quant::packed!("gemv_f16_1.spv")))?,
			create_sm(d, &load_spv(q8))?,
			create_sm(d, &load_spv(quant::packed!("gemv_delta.spv")))?,
			create_sm(d, &load_spv(quant::packed!("gemv_i8_2.spv")))?,
			create_sm(d, &load_spv(quant::packed!("gemv_f16_2.spv")))?,
			create_sm(d, &load_spv(q8_2))?,
			create_sm(d, &load_spv(quant::packed!("gemv_i8_4.spv")))?,
			create_sm(d, &load_spv(quant::packed!("gemv_f16_4.spv")))?,
			create_sm(d, &load_spv(q8_4))?,
		];
		// the run's positions and its first position slot (read by the delta
		// mixer only: the other kernels see their slots through the set)
		let pc = vk::PushConstantRange {
			stage_flags: vk::ShaderStageFlags::COMPUTE,
			offset: 0,
			size: 8,
		};
		let layout = unsafe {
			d.create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo {
					set_layout_count: 1,
					p_set_layouts: &dev.aux_layout,
					push_constant_range_count: 1,
					p_push_constant_ranges: &pc,
					..Default::default()
				},
				None,
			)
		}
		.map_err(|e| format!("gemv: pipeline layout: {e}"))?;
		let pl = [
			compute_pipeline_sg32(d, layout, sm[0], &main)?,
			compute_pipeline_sg32(d, layout, sm[1], &main)?,
			compute_pipeline(d, layout, sm[2], &main)?,
			compute_pipeline(d, layout, sm[3], &main)?,
			compute_pipeline_sg32(d, layout, sm[4], &main)?,
			compute_pipeline_sg32(d, layout, sm[5], &main)?,
			compute_pipeline_sg32(d, layout, sm[6], &main)?,
			compute_pipeline(d, layout, sm[7], &main)?,
			compute_pipeline_sg32(d, layout, sm[8], &main)?,
			compute_pipeline_sg32(d, layout, sm[9], &main)?,
			compute_pipeline_sg32(d, layout, sm[10], &main)?,
			compute_pipeline_sg32(d, layout, sm[11], &main)?,
			compute_pipeline_sg32(d, layout, sm[12], &main)?,
			compute_pipeline_sg32(d, layout, sm[13], &main)?,
		];
		let sizes = [
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: (2 * 4 * MAXMATS) as u32,
			},
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::UNIFORM_BUFFER,
				descriptor_count: (2 * MAXMATS) as u32,
			},
		];
		let pool = unsafe {
			d.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo {
					max_sets: (2 * MAXMATS) as u32,
					pool_size_count: 2,
					p_pool_sizes: sizes.as_ptr(),
					..Default::default()
				},
				None,
			)
		}
		.map_err(|e| format!("gemv: descriptor pool: {e}"))?;
		let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
		let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
		let st = vk::BufferUsageFlags::STORAGE_BUFFER;
		let mut bufs = Vec::new();
		let mut mk =
			|size: usize, usage, props| -> Result<(vk::Buffer, vk::DeviceMemory), String> {
				let b = dev
					.try_create_gpu_buffer(size as u64, usage, props)
					.ok_or("gemv: buffer allocation failed")?;
				bufs.push(b);
				Ok(b)
			};
		let align = 256u64;
		let params = mk(
			MAXMATS * align as usize,
			vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
			local,
		)?;
		let stage = mk(STAGE, vk::BufferUsageFlags::TRANSFER_SRC, host)?;
		let (xn, yn) = (GEMV_NP * GEMV_X * 4, GEMV_NP * GEMV_Y * 4);
		let xin = mk(xn, vk::BufferUsageFlags::TRANSFER_SRC, host)?;
		let xdev = mk(xn, st | vk::BufferUsageFlags::TRANSFER_DST, local)?;
		// the kernels write VRAM and one copy brings the rows back: 4-byte
		// writes straight into host memory cost ~80 ns per row (a 10240-row
		// GEMV 0.9 ms)
		let ydev = mk(yn, st | vk::BufferUsageFlags::TRANSFER_SRC, local)?;
		let td = vk::BufferUsageFlags::TRANSFER_DST;
		let yout = mk(yn, td, host | vk::MemoryPropertyFlags::HOST_CACHED)
			.or_else(|_| mk(yn, td, host))?;
		let map = |m: vk::DeviceMemory| {
			unsafe { d.map_memory(m, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }
				.map_err(|e| format!("gemv: map: {e}"))
		};
		let (sp, xp, yp) = (
			map(stage.1)? as *mut u8,
			map(xin.1)? as *mut f32,
			map(yout.1)? as *const f32,
		);
		Ok(Gemv {
			sm,
			pl,
			layout,
			pool,
			blocks: Vec::new(),
			need,
			params: params.0,
			stage: (stage.0, sp),
			xin: (xin.0, xp),
			xdev: xdev.0,
			ydev: ydev.0,
			yout: (yout.0, yp),
			mats: Vec::new(),
			align,
			bytes: 0,
			pending: [None, None],
			cmds: (!std::env::var("QWEN_GEMV_CMDCACHE").is_ok_and(|v| v == "0"))
				.then(Default::default),
			wave: 0,
			wide: false,
			p0: [0; 2],
			// a decode step's runs take ~0.2-1.5 ms; a blocking wait woke
			// ~10-60 us late (`gemvbench` chain line), 96 runs per token
			spin_us: std::env::var("QWEN_GEMV_SPIN_US")
				.ok()
				.and_then(|v| v.parse().ok())
				.unwrap_or(2000),
			delta: None,
			prof: if std::env::var("QWEN_GEMV_PROF").is_ok_and(|v| v == "1") {
				let qp = unsafe {
					d.create_query_pool(
						&vk::QueryPoolCreateInfo {
							query_type: vk::QueryType::TIMESTAMP,
							query_count: PROF_Q,
							..Default::default()
						},
						None,
					)
				}
				.map_err(|e| format!("gemv: query pool: {e}"))?;
				let period = unsafe { dev.instance.get_physical_device_properties(dev.physical) }
					.limits
					.timestamp_period as f64;
				Some(Prof {
					qp,
					period,
					run: Vec::new(),
					owner: None,
					sums: Default::default(),
				})
			} else {
				None
			},
			npb: !std::env::var("QWEN_GEMV_NPB").is_ok_and(|v| v == "0"),
			io: None,
			ranges: !std::env::var("QWEN_GEMV_RANGES").is_ok_and(|v| v == "0"),
			bufs,
			dev,
		})
	}

	/// Run 2-4 positions on the builds for that many (default; the
	/// `QWEN_GEMV_NPB` switch) or on the 8-position ones: `gemvbench`'s
	/// comparison.
	pub fn set_npb(&mut self, on: bool) {
		self.npb = on;
	}

	/// Make wave `w` (0 or 1) the one the next runs, waits and outputs
	/// ([`Self::run_chain_n_begin`], [`Self::run_wait`], [`Self::y_at`])
	/// refer to. Wave 0 takes position slots `[0, GEMV_NP)`, wave 1 slots
	/// `[GEMV_WAVE, GEMV_NP)` (at most [`GEMV_WAVE`] positions a run) on its
	/// own fence: a run of each can be in flight at once, executed in
	/// submission order. A delta mixer step of wave 1 continues the
	/// recurrence inputs and saved windows of wave 0 at aux slot `p0` (the
	/// positions of wave 0's step, so a rollback replays slots `0..n` of
	/// both) and keeps the backup (`gemv_delta.glsl`'s `p0`); wave 0 takes
	/// `p0` = 0.
	pub fn set_wave(&mut self, w: usize, p0: usize) {
		assert!(w < 2 && (w == 1 || p0 == 0) && p0 <= GEMV_WAVE);
		self.wave = w;
		self.p0[w] = p0;
	}

	/// The wave [`Self::set_wave`] chose.
	pub fn wave(&self) -> usize {
		self.wave
	}

	/// Wait for both waves' runs.
	pub fn wait_all(&mut self) -> Result<(), String> {
		let w = self.wave;
		let r = (0..2).try_for_each(|i| {
			self.wave = i;
			self.run_wait()
		});
		self.wave = w;
		r
	}

	/// The next run uploads only the floats `xr` of each position's input
	/// (empty: all of it) and brings back only the floats `yr` of each
	/// output region (empty: up to its last op's rows); the rest of
	/// [`Self::y_at`] is then stale. A decode step's post run moved 144 KB
	/// in and 186 KB out per position for ~64 / ~52 KB the host reads, at
	/// ~11 GB/s on the serial chain.
	pub fn io_next(&mut self, xr: &[std::ops::Range<usize>], yr: &[std::ops::Range<usize>]) {
		if self.ranges {
			self.io = Some((xr.to_vec(), yr.to_vec()));
		}
	}

	/// VRAM bytes the matrices take.
	pub fn bytes(&self) -> u64 {
		self.bytes
	}

	/// Room for `bytes`: (buffer, offset) in the first block with room,
	/// else in a new one of [`BLOCK`] or, near the end, of what `need` has
	/// left (a last block of full size would leave most of it unused).
	fn place(&mut self, bytes: u64) -> Result<(vk::Buffer, u64), String> {
		let a = self.align;
		for (buf, size, used) in self.blocks.iter_mut() {
			let off = used.div_ceil(a) * a;
			if off + bytes <= *size {
				*used = off + bytes;
				return Ok((*buf, off));
			}
		}
		let left = self.need.saturating_sub(self.bytes) + BLOCK_MIN;
		let size = (left.div_ceil(BLOCK_MIN) * BLOCK_MIN)
			.min(BLOCK)
			.max(bytes.div_ceil(a) * a);
		let b = self
			.dev
			.try_create_gpu_buffer(
				size,
				vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
				vk::MemoryPropertyFlags::DEVICE_LOCAL,
			)
			.ok_or("gemv: VRAM allocation failed")?;
		self.bufs.push(b);
		self.blocks.push((b.0, size, bytes));
		Ok((b.0, 0))
	}

	/// Copy `data` to `dst` at `off` through the staging buffer.
	fn upload(&self, dst: vk::Buffer, off: u64, data: &[u8]) -> Result<(), String> {
		let d = &self.dev.device;
		for (i, c) in data.chunks(STAGE).enumerate() {
			// SAFETY: the staging map holds STAGE bytes
			unsafe { std::slice::from_raw_parts_mut(self.stage.1, c.len()) }.copy_from_slice(c);
			let src = self.stage.0;
			self.dev
				.submit_and_wait(|cmd| unsafe {
					d.cmd_copy_buffer(
						cmd,
						src,
						dst,
						&[vk::BufferCopy {
							src_offset: 0,
							dst_offset: off + (i * STAGE) as u64,
							size: c.len() as u64,
						}],
					)
				})
				.map_err(|e| format!("gemv: upload: {e}"))?;
		}
		Ok(())
	}

	/// Upload an `[n][k]` matrix whose GEMVs read the input at float `xoff`
	/// (a multiple of 16) and write rows to `yoff`; returns its id.
	pub fn add(
		&mut self,
		m: GemvMat<'_>,
		n: usize,
		k: usize,
		xoff: usize,
		yoff: usize,
	) -> Result<usize, String> {
		self.wait_all()?;
		let (kind, wb, sc): (usize, &[u8], Vec<f32>) = match m {
			GemvMat::I8 { w, s } | GemvMat::Q8 { w, s } => (
				if matches!(m, GemvMat::Q8 { .. }) {
					6
				} else {
					0
				},
				// SAFETY: i8 and u8 have the same layout
				unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len()) },
				s.to_vec(),
			),
			GemvMat::F16 { w } => (
				1,
				// SAFETY: plain data
				unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 2) },
				vec![1.0; n],
			),
		};
		let f16 = kind == 1;
		let per = if f16 { 8 } else { 16 };
		if !k.is_multiple_of(per)
			|| wb.len() != n * k * if f16 { 2 } else { 1 }
			|| sc.len() != n
			|| !xoff.is_multiple_of(16)
			|| xoff + k > GEMV_X
			|| yoff + n > GEMV_Y
			|| self.mats.len() >= MAXMATS
		{
			return Err(format!("gemv: bad matrix [{n}][{k}] at x {xoff} y {yoff}"));
		}
		let (wbuf, woff) = self.place(wb.len() as u64)?;
		self.upload(wbuf, woff, wb)?;
		let sb: Vec<u8> = sc.iter().flat_map(|v| v.to_le_bytes()).collect();
		let (sbuf, soff) = self.place(sb.len() as u64)?;
		self.upload(sbuf, soff, &sb)?;
		self.bytes += (wb.len() + sb.len()) as u64;
		let id = self.mats.len();
		// lanes per row: as many as the row has 16-byte loads, at most 32
		let nq = (k / per) as u32;
		let lpr = (1..=5)
			.map(|e| 1u32 << e)
			.rfind(|&l| l <= nq)
			.unwrap_or(1)
			.min(32);
		// fewer than 4 workgroups per SM (10 on the GTX 1060): the
		// workgroup's subgroups split k instead of taking rows of their own
		let groups = (n as u32).div_ceil(8 * 32 / lpr * NR);
		let (ks, groups) = if groups < 40 {
			(8, (n as u32).div_ceil(32 / lpr * NR))
		} else {
			(1, groups)
		};
		// (the int8 kernels' SiLU epilogue off: `act`, `acta`, `actdst`,
		// [`Self::set_silu`])
		let pv: Vec<u8> = [
			n as u32,
			k as u32,
			xoff as u32,
			yoff as u32,
			lpr,
			ks,
			0,
			0,
			0,
		]
		.iter()
		.flat_map(|x| x.to_le_bytes())
		.collect();
		let (xdev, ydev) = (self.xdev, self.ydev);
		let set = self.describe(
			id,
			&pv,
			[
				(wbuf, woff, wb.len() as u64),
				(xdev, 0, vk::WHOLE_SIZE),
				(ydev, 0, vk::WHOLE_SIZE),
				(sbuf, soff, sb.len() as u64),
			],
		)?;
		self.mats.push(Mat {
			kind,
			groups,
			yend: yoff + n,
			set,
			label: format!("{} [{n}][{k}]", ["i8", "f16", "", "", "", "", "q8"][kind]),
		});
		Ok(id)
	}

	/// Write op `id`'s parameters and make its descriptor sets on the
	/// storage regions `bufs` (bindings 0-3) and the input region (binding
	/// 4, the int8 kernels' SiLU epilogue): wave 0's as given, wave 1's
	/// with the input / output regions (`xdev` / `ydev` at offset 0) bound
	/// at position slot [`GEMV_WAVE`].
	fn describe(
		&mut self,
		id: usize,
		pv: &[u8],
		bufs: [(vk::Buffer, u64, u64); 4],
	) -> Result<[vk::DescriptorSet; 2], String> {
		self.wait_all()?;
		let poff = id as u64 * self.align;
		let (d, pb) = (&self.dev.device, self.params);
		self.dev
			.submit_and_wait(|cmd| unsafe { d.cmd_update_buffer(cmd, pb, poff, pv) })
			.map_err(|e| format!("gemv: params: {e}"))?;
		let layouts = [self.dev.aux_layout; 2];
		let sets = unsafe {
			d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
				descriptor_pool: self.pool,
				descriptor_set_count: 2,
				p_set_layouts: layouts.as_ptr(),
				..Default::default()
			})
		}
		.map_err(|e| format!("gemv: descriptor set: {e}"))?;
		let (xdev, ydev) = (self.xdev, self.ydev);
		let (xw, yw) = (
			(GEMV_WAVE * GEMV_X * 4) as u64,
			(GEMV_WAVE * GEMV_Y * 4) as u64,
		);
		for (w, &set) in sets.iter().enumerate() {
			let info = |(buffer, offset, range): (vk::Buffer, u64, u64)| {
				let offset = match w {
					1 if buffer == xdev && offset == 0 => xw,
					1 if buffer == ydev && offset == 0 => yw,
					_ => offset,
				};
				vk::DescriptorBufferInfo {
					buffer,
					offset,
					range,
				}
			};
			let infos = [
				info(bufs[0]),
				info(bufs[1]),
				info(bufs[2]),
				info(bufs[3]),
				info((xdev, 0, vk::WHOLE_SIZE)),
				info((self.params, poff, 64)),
			];
			let writes: Vec<vk::WriteDescriptorSet> = (0..6)
				.map(|i| vk::WriteDescriptorSet {
					dst_set: set,
					dst_binding: if i < 5 { i as u32 } else { 8 },
					descriptor_type: if i < 5 {
						vk::DescriptorType::STORAGE_BUFFER
					} else {
						vk::DescriptorType::UNIFORM_BUFFER
					},
					descriptor_count: 1,
					p_buffer_info: &infos[i],
					..Default::default()
				})
				.collect();
			unsafe { d.update_descriptor_sets(&writes, &[]) };
		}
		Ok([sets[0], sets[1]])
	}

	/// Let int8 GEMV `id` (an `I8` matrix of [`Self::add`]) also write
	/// `silu(a y)` of each output row `y` to input float `xdst + row` of its
	/// position: `gemv_act.glsl` mode 0 on its rows, computed as there, in
	/// the GEMV's dispatch (one barrier and dispatch less in a chain). The
	/// rows must lie outside the inputs the GEMVs of its stage read.
	pub fn set_silu(&mut self, id: usize, a: f32, xdst: usize) -> Result<(), String> {
		// (the end of its output rows bounds its row count)
		let yend = match self.mats.get(id) {
			Some(m) if m.kind == 0 => m.yend,
			_ => return Err(format!("gemv: no int8 GEMV {id} for a SiLU epilogue")),
		};
		if xdst + yend > GEMV_X {
			return Err(format!("gemv: SiLU epilogue at x {xdst} out of range"));
		}
		self.wait_all()?;
		let mut pv = 1u32.to_le_bytes().to_vec();
		pv.extend(a.to_le_bytes());
		pv.extend((xdst as u32).to_le_bytes());
		let (d, pb, poff) = (&self.dev.device, self.params, id as u64 * self.align + 24);
		self.dev
			.submit_and_wait(|cmd| unsafe { d.cmd_update_buffer(cmd, pb, poff, &pv) })
			.map_err(|e| format!("gemv: params: {e}"))
	}

	/// An elementwise step for [`Self::run_chain`] (`shaders/gemv_act.glsl`):
	/// mode 0 `x[dst + i] = silu(a y[src + i])`, mode 1 `y[dst + i] = a
	/// sum_c sigmoid(y[src + c n + i]) x[xsrc + c n + i]` over `hc` streams.
	#[allow(clippy::too_many_arguments)]
	pub fn add_act(
		&mut self,
		mode: u32,
		n: usize,
		src: usize,
		dst: usize,
		xsrc: usize,
		hc: usize,
		a: f32,
	) -> Result<usize, String> {
		let (xend, yend) = if mode == 0 {
			(dst + n, src + n)
		} else {
			(xsrc + hc * n, (src + hc * n).max(dst + n))
		};
		if mode > 1 || xend > GEMV_X || yend > GEMV_Y || self.mats.len() >= MAXMATS {
			return Err(format!("gemv: bad step {mode} [{n}]"));
		}
		let id = self.mats.len();
		let mut pv: Vec<u8> = [
			mode,
			n as u32,
			src as u32,
			dst as u32,
			xsrc as u32,
			hc as u32,
		]
		.iter()
		.flat_map(|x| x.to_le_bytes())
		.collect();
		pv.extend(a.to_le_bytes());
		let (xdev, ydev) = (self.xdev, self.ydev);
		let w = vk::WHOLE_SIZE;
		let set = self.describe(
			id,
			&pv,
			[(xdev, 0, w), (ydev, 0, w), (xdev, 0, w), (ydev, 0, w)],
		)?;
		self.mats.push(Mat {
			kind: 2,
			groups: n.div_ceil(256) as u32,
			yend: if mode == 0 { 0 } else { dst + n },
			set,
			label: format!("act{mode} {n}"),
		});
		Ok(id)
	}

	/// A vector step for [`Self::run_chain`] (`shaders/gemv_vec.glsl`, which
	/// documents the modes and `flags`); `w` is uploaded as the step's own
	/// weight vector, `ycopy` `None` writes no copy; `extra` is mode 5's
	/// copy of the mix (x) and mode 1's injected vector (y, flags & 8).
	#[allow(clippy::too_many_arguments)]
	pub fn add_vec(
		&mut self,
		mode: u32,
		n: usize,
		src: usize,
		dst: usize,
		aux: usize,
		hc: usize,
		a: f32,
		blk: usize,
		flags: u32,
		ycopy: Option<usize>,
		extra: Option<usize>,
		w: Option<&[f32]>,
	) -> Result<usize, String> {
		let (sy, rows) = (flags & 1 != 0, if mode == 1 || mode == 2 { hc } else { 1 });
		// mode 0's SwiGLU source, mode 5's mix: read from y
		let swi = mode == 0 && flags & 8 != 0;
		let inj = mode == 1 && flags & 8 != 0;
		let send = src
			+ if mode == 3 || swi {
				2 * n
			} else if mode == 1 || mode == 5 {
				hc * n
			} else {
				n
			};
		let from_y = (sy || mode >= 2 || swi) && !inj;
		let dend = dst + rows * n;
		let bad =
			mode > 5
				|| from_y && send > GEMV_Y
				|| !from_y && send > GEMV_X
				|| dend > GEMV_X
				|| (mode == 2 || inj) && aux + hc > GEMV_X
				|| mode == 5 && aux + hc * n > GEMV_X
				|| inj && sy || match extra {
				Some(e) if mode == 5 => e + n > GEMV_X,
				Some(e) if inj => e + n > GEMV_Y,
				Some(_) => true,
				None => inj,
			} || ycopy.is_some_and(|c| c + rows * n > GEMV_Y)
				|| blk > 256 || blk > 1 && (!blk.is_power_of_two() || !n.is_multiple_of(blk))
				|| w.is_some_and(|w| w.len() < rows * n)
				|| (flags & 4 != 0) != w.is_some() && (mode == 0 || mode == 5)
				|| self.mats.len() >= MAXMATS;
		if bad {
			return Err(format!("gemv: bad vector step {mode} [{n}]"));
		}
		let wb = match w {
			Some(w) => {
				let b: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
				let (buf, off) = self.place(b.len() as u64)?;
				self.upload(buf, off, &b)?;
				self.bytes += b.len() as u64;
				(buf, off, b.len() as u64)
			}
			None => (self.xdev, 0, vk::WHOLE_SIZE),
		};
		let id = self.mats.len();
		let mut pv: Vec<u8> = [
			mode, n as u32, src as u32, dst as u32, aux as u32, hc as u32,
		]
		.iter()
		.flat_map(|x| x.to_le_bytes())
		.collect();
		pv.extend(a.to_le_bytes());
		for v in [
			blk as u32,
			flags,
			ycopy.map_or(u32::MAX, |c| c as u32),
			extra.map_or(u32::MAX, |c| c as u32),
		] {
			pv.extend(v.to_le_bytes());
		}
		let (xdev, ydev) = (self.xdev, self.ydev);
		let wh = vk::WHOLE_SIZE;
		let set = self.describe(id, &pv, [(xdev, 0, wh), (ydev, 0, wh), wb, (ydev, 0, wh)])?;
		let groups = if mode == 1 { hc } else { n.div_ceil(256) };
		self.mats.push(Mat {
			kind: 3,
			groups: groups as u32,
			yend: ycopy.map_or(0, |c| c + rows * n),
			set,
			label: format!("vec{mode} {n}"),
		});
		Ok(id)
	}

	/// VRAM for `layers` delta mixers' states (zero): [`DELTA_STATE`] +
	/// [`DELTA_AUX`] floats each.
	pub fn delta_init(&mut self, layers: usize) -> Result<(), String> {
		if self.delta.is_some() {
			return Err("gemv: delta states twice".into());
		}
		let u = vk::BufferUsageFlags::STORAGE_BUFFER
			| vk::BufferUsageFlags::TRANSFER_DST
			| vk::BufferUsageFlags::TRANSFER_SRC;
		let mut mk = |n: usize| -> Result<vk::Buffer, String> {
			let b = self
				.dev
				.try_create_gpu_buffer(
					(layers * n * 4) as u64,
					u,
					vk::MemoryPropertyFlags::DEVICE_LOCAL,
				)
				.ok_or("gemv: delta state allocation failed")?;
			self.bufs.push(b);
			Ok(b.0)
		};
		let (s, a) = (mk(DELTA_STATE)?, mk(DELTA_AUX)?);
		let d = &self.dev.device;
		self.dev
			.submit_and_wait(|cmd| unsafe {
				d.cmd_fill_buffer(cmd, s, 0, vk::WHOLE_SIZE, 0);
				d.cmd_fill_buffer(cmd, a, 0, vk::WHOLE_SIZE, 0);
			})
			.map_err(|e| format!("gemv: delta fill: {e}"))?;
		self.bytes += (layers * (DELTA_STATE + DELTA_AUX) * 4) as u64;
		self.delta = Some((s, a, layers));
		Ok(())
	}

	/// A delta mixer step of `layer` (`shaders/gemv_delta.glsl`, which
	/// documents `mode` and `flags`): reads the gate inputs `z` at output
	/// float `yz`, `ab` at `yab`, writes the gated output at `ya`; `w` is
	/// the step's weight vector.
	#[allow(clippy::too_many_arguments)]
	pub fn add_delta(
		&mut self,
		mode: u32,
		flags: u32,
		layer: usize,
		yz: usize,
		yab: usize,
		ya: usize,
		w: &[f32],
	) -> Result<usize, String> {
		let Some((sb, ab, nl)) = self.delta else {
			return Err("gemv: no delta states".into());
		};
		if mode > 2 || layer >= nl || ya + 6144 > GEMV_Y || self.mats.len() >= MAXMATS {
			return Err(format!("gemv: bad delta step {mode}"));
		}
		let b: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
		let (wb, wo) = self.place(b.len().max(4) as u64)?;
		if !b.is_empty() {
			self.upload(wb, wo, &b)?;
		}
		self.bytes += b.len() as u64;
		let id = self.mats.len();
		let pv: Vec<u8> = [mode, flags, yz as u32, yab as u32, ya as u32]
			.iter()
			.flat_map(|x| x.to_le_bytes())
			.collect();
		let ydev = self.ydev;
		let (ss, sa) = ((DELTA_STATE * 4) as u64, (DELTA_AUX * 4) as u64);
		let set = self.describe(
			id,
			&pv,
			[
				(ydev, 0, vk::WHOLE_SIZE),
				(sb, layer as u64 * ss, ss),
				(ab, layer as u64 * sa, sa),
				(wb, wo, b.len().max(4) as u64),
			],
		)?;
		self.mats.push(Mat {
			kind: 7,
			groups: [21, 48, 20][mode as usize],
			yend: if mode == 1 { ya + 6144 } else { 0 },
			set,
			label: format!("delta{mode}"),
		});
		Ok(id)
	}

	/// Write `layer`'s delta state `[48][128][128]` and conv window
	/// [`DELTA_WIN`] (f16 values).
	pub fn delta_put(&mut self, layer: usize, state: &[f32], win: &[f32]) -> Result<(), String> {
		self.wait_all()?;
		let Some((sb, ab, nl)) = self.delta else {
			return Err("gemv: no delta states".into());
		};
		if layer >= nl || state.len() != DELTA_STATE / 2 || win.len() != DELTA_WIN {
			return Err("gemv: bad delta state".into());
		}
		let u8s =
			|v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
		self.upload(sb, (layer * DELTA_STATE * 4) as u64, u8s(state))?;
		self.upload(ab, (layer * DELTA_AUX * 4) as u64, u8s(win))
	}

	/// Read `layer`'s delta state and conv window back ([`Self::delta_put`]'s
	/// forms), through the output map (the last run's outputs are lost).
	pub fn delta_get(
		&mut self,
		layer: usize,
		state: &mut [f32],
		win: &mut [f32],
	) -> Result<(), String> {
		self.wait_all()?;
		let Some((sb, ab, nl)) = self.delta else {
			return Err("gemv: no delta states".into());
		};
		if layer >= nl || state.len() != DELTA_STATE / 2 || win.len() != DELTA_WIN {
			return Err("gemv: bad delta state".into());
		}
		let max = GEMV_NP * GEMV_Y;
		for (src, off, out) in [
			(sb, layer * DELTA_STATE, state),
			(ab, layer * DELTA_AUX, win),
		] {
			for (i, c) in out.chunks_mut(max).enumerate() {
				let (d, yout) = (&self.dev.device, self.yout.0);
				self.dev
					.submit_and_wait(|cmd| unsafe {
						d.cmd_copy_buffer(
							cmd,
							src,
							yout,
							&[vk::BufferCopy {
								src_offset: ((off + i * max) * 4) as u64,
								dst_offset: 0,
								size: (c.len() * 4) as u64,
							}],
						)
					})
					.map_err(|e| format!("gemv: delta read: {e}"))?;
				// SAFETY: the map holds GEMV_NP GEMV_Y floats; nothing in flight
				c.copy_from_slice(unsafe { std::slice::from_raw_parts(self.yout.1, c.len()) });
			}
		}
		Ok(())
	}

	/// Run the GEMVs `ids` on the input `x` (floats `[0, x.len())` of the
	/// input region); the output region afterwards in [`Self::y`].
	pub fn run(&mut self, x: &[f32], ids: &[usize]) -> Result<(), String> {
		self.run_chain(x, &[ids])
	}

	/// [`Self::run`] in stages: each stage's ops (GEMVs or [`Self::add_act`]
	/// steps) see the previous stages' results, in one submission.
	pub fn run_chain(&mut self, x: &[f32], stages: &[&[usize]]) -> Result<(), String> {
		self.run_chain_n(&[x], stages)
	}

	/// [`Self::run_chain`] on `xs.len()` positions (at most [`GEMV_NP`]) at
	/// once: every op runs on each position's own input (`xs[p]`, all of one
	/// length) and output region, each GEMV reading its matrix once for
	/// all of them; the outputs afterwards in [`Self::y_at`].
	pub fn run_chain_n(&mut self, xs: &[&[f32]], stages: &[&[usize]]) -> Result<(), String> {
		self.run_chain_n_begin(xs, stages)?;
		self.run_wait()
	}

	/// [`Self::run_chain_n`] without waiting: the host works on while the
	/// GPU runs; [`Self::run_wait`] (or the next run) waits for it, and
	/// only then are its outputs in [`Self::y_at`]. The output map is
	/// rewritten by it, so read the previous run's outputs first.
	pub fn run_chain_n_begin(&mut self, xs: &[&[f32]], stages: &[&[usize]]) -> Result<(), String> {
		self.run_wait()?;
		let n = self.mats.len();
		let np = xs.len();
		let wave = self.wave;
		// a wave-0 run past slot GEMV_WAVE takes wave 1's slots
		if wave == 0 && np > GEMV_WAVE || wave == 1 && self.wide {
			self.wait_all()?;
		}
		let base = wave * GEMV_WAVE;
		let p0 = self.p0[wave];
		if p0 + np > GEMV_NP {
			return Err("gemv: bad run (aux slots)".into());
		}
		let x = xs.first().copied().unwrap_or(&[]);
		if np == 0
			|| np > GEMV_NP - base
			|| x.len() > GEMV_X
			|| xs.iter().any(|v| v.len() != x.len())
			|| stages.iter().flat_map(|s| s.iter()).any(|&i| i >= n)
		{
			return Err("gemv: bad run".into());
		}
		let (xrs, yrs) = self.io.take().unwrap_or_default();
		#[allow(clippy::single_range_in_vec_init)] // a list of one range
		let xrs = if xrs.is_empty() {
			vec![0..x.len()]
		} else {
			xrs.into_iter()
				.map(|r| r.start.min(x.len())..r.end.min(x.len()))
				.filter(|r| !r.is_empty())
				.collect()
		};
		for (p, v) in xs.iter().enumerate() {
			// SAFETY: the map holds GEMV_NP GEMV_X floats; base + p < GEMV_NP
			let m = unsafe {
				std::slice::from_raw_parts_mut(self.xin.1.add((base + p) * GEMV_X), v.len())
			};
			for r in &xrs {
				m[r.clone()].copy_from_slice(&v[r.clone()]);
			}
		}
		let d = &self.dev.device;
		let (xin, xdev, ydev, yout, layout, pl) = (
			self.xin.0,
			self.xdev,
			self.ydev,
			self.yout.0,
			self.layout,
			self.pl,
		);
		let mats = &self.mats;
		let npb = self.npb;
		// (one run at a time holds the queries: the other wave's run in
		// flight goes unprofiled)
		let prof_qp = self
			.prof
			.as_ref()
			.filter(|p| stages.len() + 3 <= PROF_Q as usize && p.owner.is_none())
			.map(|p| p.qp);
		if let Some(pf) = self.prof.as_mut().filter(|p| p.owner.is_none()) {
			pf.run.clear();
			if prof_qp.is_some() {
				pf.owner = Some(wave);
				pf.run.push(String::new());
				pf.run
					.push(format!("x{np} copy in {} KB", (x.len() * 4) >> 10));
				for ids in stages {
					let l: Vec<&str> = ids.iter().map(|&i| mats[i].label.as_str()).collect();
					pf.run.push(format!("x{np} {}", l.join(" + ")));
				}
				let ye = stages
					.iter()
					.flat_map(|s| s.iter())
					.map(|&i| mats[i].yend)
					.max();
				pf.run
					.push(format!("x{np} copy out {} KB", (ye.unwrap_or(0) * 4) >> 10));
			}
		}
		use vk::{AccessFlags as A, PipelineStageFlags as P};
		let barrier = |cmd, a, b, sa, da| unsafe {
			d.cmd_pipeline_barrier(
				cmd,
				a,
				b,
				vk::DependencyFlags::empty(),
				&[vk::MemoryBarrier {
					src_access_mask: sa,
					dst_access_mask: da,
					..Default::default()
				}],
				&[],
				&[],
			)
		};
		let xb = (x.len().max(1) * 4) as u64;
		let yend = stages
			.iter()
			.flat_map(|s| s.iter())
			.map(|&i| self.mats[i].yend);
		let yb = (yend.max().unwrap_or(1).max(1) * 4) as u64;
		// the other wave's run in flight: this one's work starts after all
		// of it (barriers order only within submission order's scopes, and
		// a delta mixer step of wave 1 continues wave 0's state)
		let after = self.pending[1 - wave].is_some();
		let yfl = yb as usize / 4;
		#[allow(clippy::single_range_in_vec_init)] // a list of one range
		let yrs: Vec<std::ops::Range<usize>> = if yrs.is_empty() {
			vec![0..yfl]
		} else {
			yrs.iter()
				.map(|r| r.start.min(yfl)..r.end.min(yfl))
				.filter(|r| !r.is_empty())
				.collect()
		};
		// everything the recorded commands depend on (the profiled runs
		// write their own queries: recorded each time)
		let key = self.cmds.as_ref().filter(|_| prof_qp.is_none()).map(|_| {
			let mut k = vec![
				wave as u64,
				np as u64,
				p0 as u64,
				npb as u64,
				after as u64,
				x.len() as u64,
			];
			for rs in [&xrs, &yrs] {
				k.push(rs.len() as u64);
				k.extend(rs.iter().flat_map(|r| [r.start as u64, r.end as u64]));
			}
			for s in stages {
				k.push(u64::MAX);
				k.extend(s.iter().map(|&i| i as u64));
			}
			k
		});
		let hit = key
			.as_ref()
			.and_then(|k| self.cmds.as_ref()?.get(k).copied());
		let keep = hit.is_some()
			|| key.is_some() && self.cmds.as_ref().is_some_and(|c| c.len() < CMD_CACHE_MAX);
		let flags = if keep {
			vk::CommandBufferUsageFlags::empty()
		} else {
			vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
		};
		let cmd = hit.unwrap_or_else(|| {
			self.dev.record_cmd(flags, |cmd| unsafe {
				if after {
					barrier(
						cmd,
						P::COMPUTE_SHADER | P::TRANSFER,
						P::COMPUTE_SHADER | P::TRANSFER,
						A::SHADER_WRITE | A::TRANSFER_WRITE,
						A::SHADER_READ | A::SHADER_WRITE | A::TRANSFER_READ | A::TRANSFER_WRITE,
					);
				}
				let xr: Vec<vk::BufferCopy> = if x.is_empty() {
					(base..base + np)
						.map(|p| vk::BufferCopy {
							src_offset: (p * GEMV_X * 4) as u64,
							dst_offset: (p * GEMV_X * 4) as u64,
							size: xb,
						})
						.collect()
				} else {
					(base..base + np)
						.flat_map(|p| {
							xrs.iter().map(move |r| vk::BufferCopy {
								src_offset: ((p * GEMV_X + r.start) * 4) as u64,
								dst_offset: ((p * GEMV_X + r.start) * 4) as u64,
								size: (r.len() * 4) as u64,
							})
						})
						.collect()
				};
				if let Some(qp) = prof_qp {
					d.cmd_reset_query_pool(cmd, qp, 0, PROF_Q);
					d.cmd_write_timestamp(cmd, P::BOTTOM_OF_PIPE, qp, 0);
				}
				if !xr.is_empty() {
					d.cmd_copy_buffer(cmd, xin, xdev, &xr);
				}
				let mut pc = (np as u32).to_le_bytes().to_vec();
				pc.extend((p0 as u32).to_le_bytes());
				d.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, &pc);
				barrier(
					cmd,
					P::TRANSFER,
					P::COMPUTE_SHADER,
					A::TRANSFER_WRITE,
					A::SHADER_READ,
				);
				if let Some(qp) = prof_qp {
					d.cmd_write_timestamp(cmd, P::BOTTOM_OF_PIPE, qp, 1);
				}
				let mut cur = None;
				for (si, ids) in stages.iter().enumerate() {
					if si > 0 {
						barrier(
							cmd,
							P::COMPUTE_SHADER,
							P::COMPUTE_SHADER,
							A::SHADER_WRITE | A::SHADER_READ,
							A::SHADER_READ | A::SHADER_WRITE,
						);
					}
					for &i in ids.iter() {
						let m = &mats[i];
						// at one position the float kernels (`_1`) also
						// take `Q8`: its kernel gains nothing there; 2 and
						// 3-4 positions take the builds for that many
						// (`QWEN_GEMV_NPB=0`: the 8-position ones, the A/B)
						let kind = match (m.kind, np) {
							(0 | 1, 1) => m.kind + 4,
							(6, 1) => 4,
							(0 | 1, 2) if npb => 8 + m.kind,
							(6, 2) if npb => 10,
							(0 | 1, 3 | 4) if npb => 11 + m.kind,
							(6, 3 | 4) if npb => 13,
							(k, _) => k,
						};
						if cur != Some(kind) {
							d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pl[kind]);
							cur = Some(kind);
						}
						d.cmd_bind_descriptor_sets(
							cmd,
							vk::PipelineBindPoint::COMPUTE,
							layout,
							0,
							&[m.set[wave]],
							&[],
						);
						// the GEMVs loop over the positions, the other
						// steps take one per workgroup row
						let gy = if m.kind < 2 || m.kind >= 6 {
							1
						} else {
							np as u32
						};
						d.cmd_dispatch(cmd, m.groups, gy, 1);
					}
					if let Some(qp) = prof_qp {
						d.cmd_write_timestamp(cmd, P::BOTTOM_OF_PIPE, qp, si as u32 + 2);
					}
				}
				barrier(
					cmd,
					P::COMPUTE_SHADER,
					P::TRANSFER,
					A::SHADER_WRITE,
					A::TRANSFER_READ,
				);
				let yr: Vec<vk::BufferCopy> = (base..base + np)
					.flat_map(|p| {
						yrs.iter().map(move |r| vk::BufferCopy {
							src_offset: ((p * GEMV_Y + r.start) * 4) as u64,
							dst_offset: ((p * GEMV_Y + r.start) * 4) as u64,
							size: (r.len() * 4) as u64,
						})
					})
					.collect();
				if !yr.is_empty() {
					d.cmd_copy_buffer(cmd, ydev, yout, &yr);
				}
				if let Some(qp) = prof_qp {
					d.cmd_write_timestamp(cmd, P::BOTTOM_OF_PIPE, qp, stages.len() as u32 + 2);
				}
				barrier(cmd, P::TRANSFER, P::HOST, A::TRANSFER_WRITE, A::HOST_READ);
			})
		});
		if let (None, true, Some(k), Some(c)) = (hit, keep, key, self.cmds.as_mut()) {
			c.insert(k, cmd);
		}
		if let Err(e) = self.dev.submit_cmd_bank(cmd, wave) {
			if !keep {
				unsafe { d.free_command_buffers(self.dev.cmd_pool, &[cmd]) };
			}
			if let Some(pf) = self.prof.as_mut().filter(|p| p.owner == Some(wave)) {
				pf.owner = None;
				pf.run.clear();
			}
			return Err(format!("gemv: run: {e}"));
		}
		self.pending[wave] = Some((cmd, keep));
		if wave == 0 {
			self.wide = np > GEMV_WAVE;
		}
		Ok(())
	}

	/// Wait for the run [`Self::run_chain_n_begin`] left in flight on the
	/// current wave ([`Self::set_wave`]), if any.
	pub fn run_wait(&mut self) -> Result<(), String> {
		let wave = self.wave;
		match self.pending[wave].take() {
			Some((cmd, kept)) => {
				if kept {
					self.dev
						.wait_fence_spin(self.dev.bank_fence(wave), self.spin_us)
				} else {
					self.dev.wait_submitted_spin_bank(cmd, self.spin_us, wave)
				}
				.map_err(|e| format!("gemv: run: {e}"))?;
				if let Some(pf) = self.prof.as_mut().filter(|p| p.owner == Some(wave)) {
					pf.owner = None;
					let mut ts = vec![0u64; pf.run.len()];
					// SAFETY: the queries were written by the run waited for
					unsafe {
						self.dev.device.get_query_pool_results(
							pf.qp,
							0,
							&mut ts,
							vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
						)
					}
					.map_err(|e| format!("gemv: timestamps: {e}"))?;
					for (i, l) in pf.run.drain(..).enumerate().skip(1) {
						let dt = ts[i].saturating_sub(ts[i - 1]);
						let e = pf.sums.entry(l).or_default();
						e.0 += 1;
						e.1 += dt as f64 * pf.period;
					}
				}
				Ok(())
			}
			None => Ok(()),
		}
	}

	/// The output region of the last run.
	pub fn y(&self) -> &[f32] {
		self.y_at(0)
	}

	/// Position `p`'s output region of the current wave's last run.
	pub fn y_at(&self, p: usize) -> &[f32] {
		let s = self.wave * GEMV_WAVE + p;
		assert!(s < GEMV_NP);
		// SAFETY: the map holds GEMV_NP GEMV_Y floats; the wave's run is
		// waited for
		unsafe { std::slice::from_raw_parts(self.yout.1.add(s * GEMV_Y), GEMV_Y) }
	}
}

impl Drop for Gemv {
	fn drop(&mut self) {
		let _ = self.wait_all();
		if let Some(pf) = self.prof.take() {
			let mut v: Vec<_> = pf.sums.into_iter().collect();
			v.sort_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
			eprintln!("gemv: GPU time per stage (QWEN_GEMV_PROF), total ms / runs / us per run:");
			for (l, (n, ns)) in v.iter().take(60) {
				eprintln!(
					"  {:9.1} {n:7} {:8.1}  {l}",
					ns * 1e-6,
					ns * 1e-3 / *n as f64
				);
			}
			unsafe { self.dev.device.destroy_query_pool(pf.qp, None) };
		}
		let pending = std::mem::take(&mut self.pending);
		let d = &self.dev.device;
		unsafe {
			let _ = d.device_wait_idle();
			for (cmd, kept) in pending.into_iter().flatten() {
				if !kept {
					d.free_command_buffers(self.dev.cmd_pool, &[cmd]);
				}
			}
			for (_, cmd) in self.cmds.take().unwrap_or_default() {
				d.free_command_buffers(self.dev.cmd_pool, &[cmd]);
			}
			for i in 0..self.pl.len() {
				d.destroy_pipeline(self.pl[i], None);
				d.destroy_shader_module(self.sm[i], None);
			}
			d.destroy_descriptor_pool(self.pool, None);
			d.destroy_pipeline_layout(self.layout, None);
			for &(b, m) in &self.bufs {
				d.destroy_buffer(b, None);
				d.free_memory(m, None);
			}
		}
	}
}
