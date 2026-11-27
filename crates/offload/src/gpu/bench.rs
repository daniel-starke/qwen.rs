use super::*;

/// Result of one ALU-ceiling probe dispatch.
pub struct AluProbe {
	/// Wall time of one dispatch, seconds (median of the timed repeats).
	pub secs: f64,
	/// Total useful operations the dispatch performed (FLOPs for modes 0/2,
	/// integer MADs for mode 1): the host computes rates from this.
	pub ops: f64,
	/// Kept so the work cannot be dead-code eliminated.
	pub sink: f32,
}

impl GpuDevice {
	/// Measure the card's arithmetic ceiling with no global-memory traffic
	/// in the inner loop (see `shaders/alu_probe.glsl` for the modes).
	///
	/// This is a *gate*, not a kernel: it answers "what is the most any
	/// kernel on this GPU could do", so a plan that needs N GFLOP/s can be
	/// rejected before it is written. The fused matmul kernels cannot answer
	/// that question because they are bound by their access pattern.
	///
	/// `groups` workgroups of 256 invocations each run `inner` iterations.
	/// `tile` is the specialization constant (tokens held in registers);
	/// it only affects mode 2.
	pub fn alu_probe(
		&self,
		mode: u32,
		tile: u32,
		inner: u32,
		groups: u32,
		iters: usize,
	) -> AluProbe {
		const LOCAL: u32 = 256;
		// `weights` and `xin` are deliberately small (256 KiB / 64 KiB) so
		// they sit in cache and the probe measures ALUs, not the memory
		// system. Both element counts are powers of two so the shader can
		// wrap its indices with a mask.
		const W_ELEMS: u64 = 65536;
		const X_ELEMS: u64 = 16384;
		let n_inv = (groups * LOCAL) as u64;

		let (wbuf, wmem) = self.create_gpu_buffer(
			W_ELEMS * 4,
			vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
			vk::MemoryPropertyFlags::DEVICE_LOCAL,
		);
		let (xbuf, xmem, xstg, xstg_mem, xptr) =
			self.create_staged_buffer(X_ELEMS * 4, vk::BufferUsageFlags::STORAGE_BUFFER);
		let (ybuf, ymem) = self.create_gpu_buffer(
			n_inv * 4,
			vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
			vk::MemoryPropertyFlags::DEVICE_LOCAL,
		);
		let (rb, rbm) = self.create_gpu_buffer(
			n_inv * 4,
			vk::BufferUsageFlags::TRANSFER_DST,
			vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
		);
		let rbp = self.map(rbm, (n_inv * 4) as usize);
		let (pbuf, pmem, pstg, pstg_mem, pptr) =
			self.create_staged_buffer(16, vk::BufferUsageFlags::UNIFORM_BUFFER);

		// Fill with small normal magnitudes: a zeroed buffer gives the
		// driver's value analysis something to fold, and denormals/NaNs
		// would trip FFMA slow paths. Neither is representative.
		{
			let mut wsrc = vec![0u8; (W_ELEMS * 4) as usize];
			let mut s = 0x2545_F491_4F6C_DD1Du64;
			for c in wsrc.chunks_mut(4) {
				s ^= s << 13;
				s ^= s >> 7;
				s ^= s << 17;
				c.copy_from_slice(&(s as u32).to_le_bytes());
			}
			let (stg, stg_mem) = self.create_gpu_buffer(
				W_ELEMS * 4,
				vk::BufferUsageFlags::TRANSFER_SRC,
				vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
			);
			let sp = self.map(stg_mem, (W_ELEMS * 4) as usize);
			unsafe { ptr::copy_nonoverlapping(wsrc.as_ptr(), sp as *mut u8, wsrc.len()) };
			self.copy_buffer(stg, wbuf, W_ELEMS * 4);
			unsafe {
				self.device.unmap_memory(stg_mem);
				self.device.destroy_buffer(stg, None);
				self.device.free_memory(stg_mem, None);
			}

			let xsrc: Vec<f32> = (0..X_ELEMS).map(|j| 0.5 + (j % 13) as f32 * 0.01).collect();
			unsafe {
				ptr::copy_nonoverlapping(
					xsrc.as_ptr() as *const u8,
					xptr.0 as *mut u8,
					xsrc.len() * 4,
				)
			};
			self.copy_buffer(xstg, xbuf, X_ELEMS * 4);
		}
		{
			let mut pb = [0u8; 16];
			pb[0..4].copy_from_slice(&mode.to_le_bytes());
			pb[4..8].copy_from_slice(&inner.to_le_bytes());
			pb[8..12].copy_from_slice(&((W_ELEMS - 1) as u32).to_le_bytes());
			pb[12..16].copy_from_slice(&((X_ELEMS - 1) as u32).to_le_bytes());
			unsafe { ptr::copy_nonoverlapping(pb.as_ptr(), pptr.0 as *mut u8, 16) };
			self.copy_buffer(pstg, pbuf, 16);
		}

		// Pipeline, specialized on TILE.
		let sm = create_sm(&self.device, &load_spv(quant::packed!("alu_probe.spv")))
			.expect("alu_probe shader module");
		let spec_data = tile.to_le_bytes();
		let spec_entry = vk::SpecializationMapEntry {
			constant_id: 0,
			offset: 0,
			size: 4,
		};
		let spec = vk::SpecializationInfo {
			map_entry_count: 1,
			p_map_entries: &spec_entry,
			data_size: 4,
			p_data: spec_data.as_ptr() as *const c_void,
			..Default::default()
		};
		let main = CString::new("main").unwrap();
		let stage = vk::PipelineShaderStageCreateInfo {
			stage: vk::ShaderStageFlags::COMPUTE,
			module: sm,
			p_name: main.as_ptr(),
			p_specialization_info: &spec,
			..Default::default()
		};
		let pipeline = unsafe {
			self.device
				.create_compute_pipelines(
					vk::PipelineCache::default(),
					&[vk::ComputePipelineCreateInfo {
						stage,
						layout: self.pipeline_layout,
						..Default::default()
					}],
					None,
				)
				.expect("alu_probe pipeline")[0]
		};

		let desc_set = unsafe {
			self.device
				.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
					descriptor_pool: self.desc_pool,
					descriptor_set_count: 1,
					p_set_layouts: &self.desc_layout,
					..Default::default()
				})
				.expect("alu_probe desc set")[0]
		};
		// These `DescriptorBufferInfo`s must outlive `update_descriptor_sets`:
		// an inline `&vk::DescriptorBufferInfo { .. }` temporary dies before
		// the call and every binding then aliases the last one. That exact bug
		// made the fused blocks emit all zeros.
		let wbi = vk::DescriptorBufferInfo {
			buffer: wbuf,
			offset: 0,
			range: vk::WHOLE_SIZE,
			..Default::default()
		};
		let xbi = vk::DescriptorBufferInfo {
			buffer: xbuf,
			offset: 0,
			range: vk::WHOLE_SIZE,
			..Default::default()
		};
		let ybi = vk::DescriptorBufferInfo {
			buffer: ybuf,
			offset: 0,
			range: vk::WHOLE_SIZE,
			..Default::default()
		};
		let pbi = vk::DescriptorBufferInfo {
			buffer: pbuf,
			offset: 0,
			range: 16,
			..Default::default()
		};
		let writes = [
			vk::WriteDescriptorSet {
				dst_set: desc_set,
				dst_binding: 0,
				descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 1,
				p_buffer_info: &wbi,
				..Default::default()
			},
			vk::WriteDescriptorSet {
				dst_set: desc_set,
				dst_binding: 1,
				descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 1,
				p_buffer_info: &xbi,
				..Default::default()
			},
			vk::WriteDescriptorSet {
				dst_set: desc_set,
				dst_binding: 2,
				descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 1,
				p_buffer_info: &ybi,
				..Default::default()
			},
			vk::WriteDescriptorSet {
				dst_set: desc_set,
				dst_binding: 3,
				descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
				descriptor_count: 1,
				p_buffer_info: &pbi,
				..Default::default()
			},
		];
		unsafe { self.device.update_descriptor_sets(&writes, &[]) };

		let dispatch = |readback: bool| {
			let _ = self.submit_and_wait(|cmd| unsafe {
				self.device
					.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
				self.device.cmd_bind_descriptor_sets(
					cmd,
					vk::PipelineBindPoint::COMPUTE,
					self.pipeline_layout,
					0,
					&[desc_set],
					&[],
				);
				self.device.cmd_dispatch(cmd, groups, 1, 1);
				if readback {
					self.device.cmd_pipeline_barrier(
						cmd,
						vk::PipelineStageFlags::COMPUTE_SHADER,
						vk::PipelineStageFlags::TRANSFER,
						vk::DependencyFlags::empty(),
						&[vk::MemoryBarrier {
							src_access_mask: vk::AccessFlags::SHADER_WRITE,
							dst_access_mask: vk::AccessFlags::TRANSFER_READ,
							..Default::default()
						}],
						&[],
						&[],
					);
					self.device.cmd_copy_buffer(
						cmd,
						ybuf,
						rb,
						&[vk::BufferCopy {
							src_offset: 0,
							dst_offset: 0,
							size: n_inv * 4,
							..Default::default()
						}],
					);
				}
			});
		};

		// Warm-up: the first dispatch pays pipeline and descriptor warm-up.
		dispatch(true);
		let mut times = Vec::with_capacity(iters);
		for _ in 0..iters {
			let t0 = std::time::Instant::now();
			dispatch(false);
			times.push(t0.elapsed().as_secs_f64());
		}
		times.sort_by(|a, b| a.partial_cmp(b).unwrap());
		let secs = times[times.len() / 2];

		dispatch(true);
		let sink = unsafe { *(rbp as *const f32) };

		// Useful ops per invocation. Mode 0: 16 FMAs -> 32 FLOP per iter.
		// Mode 1: 16 integer MADs per iter. Mode 2: 8 nibbles x TILE FMAs
		// -> 16*TILE FLOP per iter.
		let per_inv = match mode {
			0 => 32.0 * inner as f64,
			1 => 16.0 * inner as f64,
			_ => 16.0 * tile as f64 * inner as f64,
		};
		let ops = per_inv * n_inv as f64;

		unsafe {
			self.device.destroy_pipeline(pipeline, None);
			self.device.destroy_shader_module(sm, None);
			// `desc_pool` was not created with FREE_DESCRIPTOR_SET, so the
			// set cannot be returned individually. The probe allocates one
			// set per call out of a pool of 8192 and is a bench entry point,
			// not part of the forward pass, so leaking it is fine.
			self.device.unmap_memory(rbm);
			self.device.unmap_memory(xstg_mem);
			self.device.unmap_memory(pstg_mem);
			for (b, m) in [
				(wbuf, wmem),
				(xbuf, xmem),
				(xstg, xstg_mem),
				(ybuf, ymem),
				(rb, rbm),
				(pbuf, pmem),
				(pstg, pstg_mem),
			] {
				self.device.destroy_buffer(b, None);
				self.device.free_memory(m, None);
			}
		}

		AluProbe { secs, ops, sink }
	}
}

/// Result of one `gemm_probe` run.
pub struct GemmProbe {
	/// GPU time for a single dispatch, seconds (one submit of `reps`
	/// back-to-back dispatches, divided by `reps`).
	pub secs: f64,
	/// `min(seq, 64) * out_dim` floats of the result, token-major: enough to
	/// diff against the CPU without reading back 143 MB.
	pub head: Vec<f32>,
}

impl GpuDevice {
	/// Time the prefill GEMM **kernel alone**, with every host cost taken out
	/// of the measurement.
	///
	/// `matmul_batch_seq` allocates, maps, fills and frees a device-local
	/// `y`, a host-visible readback of the same size and the activation
	/// buffers *per call*: at `seq = 2048, out_dim = 17408` that is two
	/// 143 MB allocations, a 143 MB device-to-host copy and a 143 MB memcpy
	/// into a fresh `Vec`, none of which is the kernel. Measuring through it
	/// cannot separate a slow shader from a slow wrapper, and the two want
	/// completely different fixes.
	///
	/// So: upload once, allocate once, then record **one** command buffer
	/// holding `reps` dispatches separated by COMPUTE-COMPUTE barriers (they
	/// all write the same `y`, so without a barrier the driver may overlap
	/// them and the time would be meaningless). One submit, one fence wait,
	/// divide by `reps`.
	///
	/// `spv` optionally overrides the pipeline with SPIR-V read from disk,
	/// so a `.glsl` edit can be measured with `glslc` + rerun and no Rust
	/// rebuild. `gemv` runs the batched-GEMV kernel (`q4k.glsl`) on the same
	/// buffers instead, for an A/B on identical data.
	/// Kind of a resident weight (bench helper).
	pub(crate) fn weight_kind(&self, il: usize, id: WeightId) -> Option<GpuKind> {
		self.weights.get(&(il, id)).map(|w| w.kind)
	}

	pub fn gemm_probe(
		&self,
		il: usize,
		id: WeightId,
		x_packed: &[u8],
		seq: usize,
		out_dim: usize,
		in_dim: usize,
		reps: usize,
		spv: Option<&str>,
		gemv: bool,
		grid: Option<(u32, u32)>,
	) -> GemmProbe {
		let w = &self.weights[&(il, id)];
		let kind = w.kind;
		let y_elems = (seq * out_dim) as u64;
		let head_elems = seq.min(64) * out_dim;

		let xlen = ((x_packed.len() + 3) & !3) as u64;
		let (xbuf, xmem, xstg, xstg_mem, xptr) =
			self.create_staged_buffer(xlen, vk::BufferUsageFlags::STORAGE_BUFFER);
		unsafe { ptr::copy_nonoverlapping(x_packed.as_ptr(), xptr.0 as *mut u8, x_packed.len()) };
		self.copy_buffer(xstg, xbuf, xlen);

		let (ybuf, ymem) = self.create_gpu_buffer(
			y_elems * 4,
			vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
			vk::MemoryPropertyFlags::DEVICE_LOCAL,
		);
		let (rb, rbm) = self.create_readback_buffer((head_elems * 4) as u64);
		let rbp = self.map(rbm, head_elems * 4);
		let (pbuf, pmem, pstg, pstg_mem, pptr) =
			self.create_staged_buffer(16, vk::BufferUsageFlags::UNIFORM_BUFFER);
		{
			let mut pb = [0u8; 16];
			pb[0..4].copy_from_slice(&(out_dim as u32).to_le_bytes());
			pb[4..8].copy_from_slice(&(in_dim as u32).to_le_bytes());
			pb[8..12].copy_from_slice(&(seq as u32).to_le_bytes());
			unsafe { ptr::copy_nonoverlapping(pb.as_ptr(), pptr.0 as *mut u8, 16) };
			self.copy_buffer(pstg, pbuf, 16);
		}

		// Optional shader override. Kept local to the probe: the shipped
		// pipeline is the embedded `.spv`, and a disk override is a
		// measurement affordance, not a runtime feature.
		let (pipeline, own_pipeline, own_sm) = match spv {
			Some(path) => {
				let bytes =
					std::fs::read(path).unwrap_or_else(|e| panic!("gemm_probe: read {path}: {e}"));
				let sm =
					create_sm(&self.device, &load_spv(&bytes)).expect("gemm_probe shader module");
				let main = CString::new("main").unwrap();
				// GEMV candidates reduce with `subgroupAdd` over 32 lanes,
				// so on a variable-width device (Intel 8..32) they need the
				// same required subgroup size as the shipped GEMVs.
				let p = if gemv {
					compute_pipeline_sg32(&self.device, self.pipeline_layout, sm, &main)
				} else {
					compute_pipeline(&self.device, self.pipeline_layout, sm, &main)
				}
				.expect("gemm_probe pipeline");
				(p, Some(p), Some(sm))
			}
			None if gemv => (self.gemv_for(kind), None, None),
			None => (self.gemm_pipeline_for(kind), None, None),
		};
		let (gx, gy) = if let Some(g) = grid {
			g
		} else if gemv {
			(self.grid_x(kind, out_dim), seq as u32)
		} else {
			self.gemm_grid(out_dim, seq)
		};

		let desc_set = unsafe {
			self.device
				.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
					descriptor_pool: self.desc_pool,
					descriptor_set_count: 1,
					p_set_layouts: &self.desc_layout,
					..Default::default()
				})
				.expect("gemm_probe desc set")[0]
		};
		// Must outlive `update_descriptor_sets`: see the note in `alu_probe`.
		let wbi = vk::DescriptorBufferInfo {
			buffer: w.weight_buf,
			offset: 0,
			range: vk::WHOLE_SIZE,
			..Default::default()
		};
		let xbi = vk::DescriptorBufferInfo {
			buffer: xbuf,
			offset: 0,
			range: vk::WHOLE_SIZE,
			..Default::default()
		};
		let ybi = vk::DescriptorBufferInfo {
			buffer: ybuf,
			offset: 0,
			range: vk::WHOLE_SIZE,
			..Default::default()
		};
		let pbi = vk::DescriptorBufferInfo {
			buffer: pbuf,
			offset: 0,
			range: 16,
			..Default::default()
		};
		let writes = [
			vk::WriteDescriptorSet {
				dst_set: desc_set,
				dst_binding: 0,
				descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 1,
				p_buffer_info: &wbi,
				..Default::default()
			},
			vk::WriteDescriptorSet {
				dst_set: desc_set,
				dst_binding: 1,
				descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 1,
				p_buffer_info: &xbi,
				..Default::default()
			},
			vk::WriteDescriptorSet {
				dst_set: desc_set,
				dst_binding: 2,
				descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 1,
				p_buffer_info: &ybi,
				..Default::default()
			},
			vk::WriteDescriptorSet {
				dst_set: desc_set,
				dst_binding: 3,
				descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
				descriptor_count: 1,
				p_buffer_info: &pbi,
				..Default::default()
			},
		];
		unsafe { self.device.update_descriptor_sets(&writes, &[]) };

		let dispatch = |n: usize, readback: bool| {
			let _ = self.submit_and_wait(|cmd| unsafe {
				self.device
					.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
				self.device.cmd_bind_descriptor_sets(
					cmd,
					vk::PipelineBindPoint::COMPUTE,
					self.pipeline_layout,
					0,
					&[desc_set],
					&[],
				);
				for i in 0..n {
					if i > 0 {
						self.device.cmd_pipeline_barrier(
							cmd,
							vk::PipelineStageFlags::COMPUTE_SHADER,
							vk::PipelineStageFlags::COMPUTE_SHADER,
							vk::DependencyFlags::empty(),
							&[vk::MemoryBarrier {
								src_access_mask: vk::AccessFlags::SHADER_WRITE,
								dst_access_mask: vk::AccessFlags::SHADER_WRITE
									| vk::AccessFlags::SHADER_READ,
								..Default::default()
							}],
							&[],
							&[],
						);
					}
					self.device.cmd_dispatch(cmd, gx, gy, 1);
				}
				if readback {
					self.device.cmd_pipeline_barrier(
						cmd,
						vk::PipelineStageFlags::COMPUTE_SHADER,
						vk::PipelineStageFlags::TRANSFER,
						vk::DependencyFlags::empty(),
						&[vk::MemoryBarrier {
							src_access_mask: vk::AccessFlags::SHADER_WRITE,
							dst_access_mask: vk::AccessFlags::TRANSFER_READ,
							..Default::default()
						}],
						&[],
						&[],
					);
					self.device.cmd_copy_buffer(
						cmd,
						ybuf,
						rb,
						&[vk::BufferCopy {
							src_offset: 0,
							dst_offset: 0,
							size: (head_elems * 4) as u64,
							..Default::default()
						}],
					);
				}
			});
		};

		// Warm-up (also fills `y`, so the readback below is a real result).
		dispatch(1, true);
		let t0 = std::time::Instant::now();
		dispatch(reps, false);
		let secs = t0.elapsed().as_secs_f64() / reps as f64;

		let mut head = vec![0.0f32; head_elems];
		unsafe { ptr::copy_nonoverlapping(rbp as *const f32, head.as_mut_ptr(), head_elems) };

		unsafe {
			if let Some(p) = own_pipeline {
				self.device.destroy_pipeline(p, None);
			}
			if let Some(m) = own_sm {
				self.device.destroy_shader_module(m, None);
			}
			self.device.unmap_memory(rbm);
			self.device.unmap_memory(xstg_mem);
			self.device.unmap_memory(pstg_mem);
			for (b, m) in [
				(xbuf, xmem),
				(xstg, xstg_mem),
				(ybuf, ymem),
				(rb, rbm),
				(pbuf, pmem),
				(pstg, pstg_mem),
			] {
				self.device.destroy_buffer(b, None);
				self.device.free_memory(m, None);
			}
		}

		GemmProbe { secs, head }
	}

	/// `gpubench --delta`: the prefill delta-net scan (`delta_prefill.glsl`,
	/// or the module at `spv`, built with `-DLANES=lanes`) over `n`
	/// positions. Returns (seconds per dispatch over `reps` back-to-back
	/// dispatches, `o` and the state after the first dispatch alone).
	/// Layouts as in the shader.
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
	) -> (f64, Vec<f32>, Vec<f32>) {
		const S: usize = 128;
		let b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
		let mut pr = [0u32; 16];
		pr[0] = n as u32;
		pr[1] = hk as u32;
		pr[2] = hv as u32;
		let code = match spv {
			Some(path) => {
				std::fs::read(path).unwrap_or_else(|e| panic!("delta_probe: read {path}: {e}"))
			}
			None => quant::packed!("delta_prefill.spv").to_vec(),
		};
		let (secs, o, st) = self.scan_probe(
			&code,
			(hv * S * lanes / 256) as u32,
			pr,
			st0,
			[&b(k), &b(q), &b(v), &b(bg)],
			hv * n * S * 4,
			reps,
		);
		let o = o
			.chunks_exact(4)
			.map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
			.collect();
		(secs, o, st)
	}

	/// The NPU prefill's recurrence (`delta_prefill_npu.glsl`, between the
	/// `build_delta.py --cut pre` / `--cut post` graphs) on raw f16 tensors:
	/// `qn`, `kn` [n][2048], `v` [n][6144], `gb` [n][96]; returns (seconds per
	/// dispatch, `o` [n][6144] f16 x `oscale`, the state after one dispatch).
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
	) -> (f64, Vec<u8>, Vec<f32>) {
		let mut pr = [0u32; 16];
		pr[0] = n as u32;
		pr[1] = oscale.to_bits();
		let code = match spv {
			Some(path) => {
				std::fs::read(path).unwrap_or_else(|e| panic!("delta_npu_probe: read {path}: {e}"))
			}
			None => quant::packed!("delta_prefill_npu.spv").to_vec(),
		};
		self.scan_probe(
			&code,
			(48 * 128 * lanes / 512) as u32,
			pr,
			st0,
			[qn, kn, v, gb],
			n * 6144 * 2,
			reps,
		)
	}

	/// One scan kernel on the aux layout: binding 0 = the f32 state `st0`
	/// (read / written), 1-3 and 5 = `ins`, 4 = an output of `o_bytes`.
	/// Returns (seconds per dispatch, best of 5 x `reps` back-to-back
	/// dispatches; the output and the state after the first dispatch).
	#[allow(clippy::too_many_arguments)]
	fn scan_probe(
		&self,
		code: &[u8],
		groups: u32,
		pr: [u32; 16],
		st0: &[f32],
		ins: [&[u8]; 4],
		o_bytes: usize,
		reps: usize,
	) -> (f64, Vec<u8>, Vec<f32>) {
		let use_rw = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC;
		let mut owned: Vec<(vk::Buffer, vk::DeviceMemory)> = vec![];
		let mut up = |data: &[u8], usage: vk::BufferUsageFlags| {
			let len = (data.len().max(64) as u64 + 3) & !3;
			let (b, m, sb, sm, p) = self.create_staged_buffer(len, usage);
			unsafe { ptr::copy_nonoverlapping(data.as_ptr(), p.0 as *mut u8, data.len()) };
			self.copy_buffer(sb, b, len);
			unsafe { self.device.unmap_memory(sm) };
			owned.push((b, m));
			owned.push((sb, sm));
			b
		};
		let st_raw: Vec<u8> = st0.iter().flat_map(|x| x.to_le_bytes()).collect();
		let st_b = up(&st_raw, use_rw);
		let sb = vk::BufferUsageFlags::STORAGE_BUFFER;
		let (b1, b2, b3, b5) = (
			up(ins[0], sb),
			up(ins[1], sb),
			up(ins[2], sb),
			up(ins[3], sb),
		);
		let o_b = up(&vec![0u8; o_bytes], use_rw);
		let pr_raw: Vec<u8> = pr.iter().flat_map(|x| x.to_le_bytes()).collect();
		let p_b = up(&pr_raw, vk::BufferUsageFlags::UNIFORM_BUFFER);
		let st_bytes = (st0.len() * 4) as u64;
		let o_bytes = o_bytes as u64;
		let (rb, rbm) = self.create_readback_buffer(st_bytes + o_bytes);
		let rbp = self.map(rbm, (st_bytes + o_bytes) as usize);

		let sm = create_sm(&self.device, &load_spv(code)).expect("delta_probe shader module");
		let main = CString::new("main").unwrap();
		let pipeline = compute_pipeline(&self.device, self.aux_pipeline_layout, sm, &main)
			.expect("delta_probe pipeline");
		let set = unsafe {
			self.device
				.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
					descriptor_pool: self.desc_pool,
					descriptor_set_count: 1,
					p_set_layouts: &self.aux_layout,
					..Default::default()
				})
				.expect("delta_probe desc set")[0]
		};
		let bufs = [st_b, b1, b2, b3, o_b, b5];
		let mut infos: Vec<vk::DescriptorBufferInfo> = bufs
			.iter()
			.map(|&b| vk::DescriptorBufferInfo {
				buffer: b,
				offset: 0,
				range: vk::WHOLE_SIZE,
			})
			.collect();
		infos.push(vk::DescriptorBufferInfo {
			buffer: p_b,
			offset: 0,
			range: 64,
		});
		let writes: Vec<vk::WriteDescriptorSet> = (0..7)
			.map(|i| vk::WriteDescriptorSet {
				dst_set: set,
				dst_binding: if i < 6 { i as u32 } else { 8 },
				descriptor_type: if i < 6 {
					vk::DescriptorType::STORAGE_BUFFER
				} else {
					vk::DescriptorType::UNIFORM_BUFFER
				},
				descriptor_count: 1,
				p_buffer_info: &infos[i],
				..Default::default()
			})
			.collect();
		unsafe { self.device.update_descriptor_sets(&writes, &[]) };
		let dispatch = |reps: usize, readback: bool| {
			let _ = self.submit_and_wait(|cmd| unsafe {
				let d = &self.device;
				d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
				d.cmd_bind_descriptor_sets(
					cmd,
					vk::PipelineBindPoint::COMPUTE,
					self.aux_pipeline_layout,
					0,
					&[set],
					&[],
				);
				let bar = |dst: vk::AccessFlags, dst_stage: vk::PipelineStageFlags| {
					d.cmd_pipeline_barrier(
						cmd,
						vk::PipelineStageFlags::COMPUTE_SHADER,
						dst_stage,
						vk::DependencyFlags::empty(),
						&[vk::MemoryBarrier {
							src_access_mask: vk::AccessFlags::SHADER_WRITE,
							dst_access_mask: dst,
							..Default::default()
						}],
						&[],
						&[],
					)
				};
				for i in 0..reps {
					if i > 0 {
						bar(
							vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
							vk::PipelineStageFlags::COMPUTE_SHADER,
						);
					}
					d.cmd_dispatch(cmd, groups, 1, 1);
				}
				if readback {
					bar(
						vk::AccessFlags::TRANSFER_READ,
						vk::PipelineStageFlags::TRANSFER,
					);
					let c = |_: (), off, size| vk::BufferCopy {
						src_offset: 0,
						dst_offset: off,
						size,
					};
					d.cmd_copy_buffer(cmd, st_b, rb, &[c((), 0, st_bytes)]);
					d.cmd_copy_buffer(cmd, o_b, rb, &[c((), st_bytes, o_bytes)]);
				}
			});
		};
		dispatch(1, true);
		let mut st_out = vec![0.0f32; st0.len()];
		let mut o_out = vec![0u8; o_bytes as usize];
		unsafe {
			ptr::copy_nonoverlapping(rbp as *const f32, st_out.as_mut_ptr(), st_out.len());
			ptr::copy_nonoverlapping(
				(rbp as *const u8).add(st_bytes as usize),
				o_out.as_mut_ptr(),
				o_out.len(),
			);
		}
		dispatch(2, false);
		let mut best = f64::MAX;
		for _ in 0..5 {
			let t0 = std::time::Instant::now();
			dispatch(reps, false);
			best = best.min(t0.elapsed().as_secs_f64() / reps as f64);
		}
		unsafe {
			self.device.destroy_pipeline(pipeline, None);
			self.device.destroy_shader_module(sm, None);
			self.device.unmap_memory(rbm);
			self.device.destroy_buffer(rb, None);
			self.device.free_memory(rbm, None);
			for (b, m) in owned {
				self.device.destroy_buffer(b, None);
				self.device.free_memory(m, None);
			}
		}
		(best, o_out, st_out)
	}
}
