use super::*;

/// Lazily created state of the prefill attention kernel
/// (`shaders/attn_prefill.glsl`): its own descriptor layout (4 storage
/// buffers + push constants). Its buffers are borrowed from the prefill
/// matmul pool (`host_slot`), see [`GpuDevice::attn_prefill_begin`].
pub(crate) struct AttnRes {
	pl_layout: vk::PipelineLayout,
	pipeline: vk::Pipeline,
	set: vk::DescriptorSet,
}

/// A submitted [`GpuDevice::attn_prefill_begin`]; finish with
/// [`GpuDevice::attn_prefill_end`] before any other submission.
pub struct PendingAttn {
	cmd: vk::CommandBuffer,
	nq: usize,
	/// Packed row width (this call's heads x head dim).
	qw: usize,
}

/// Workgroup shape of the kernel (`QT` positions x one KV group).
const QT: usize = 4;
/// Head dim and heads per KV group the kernel is compiled for.
pub const ATTN_HD: usize = 256;
pub const ATTN_HPG: usize = 6;

impl GpuDevice {
	fn attnp_init(&self) -> Option<AttnRes> {
		let d = &self.device;
		let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..4)
			.map(|b| vk::DescriptorSetLayoutBinding {
				binding: b,
				descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 1,
				stage_flags: vk::ShaderStageFlags::COMPUTE,
				..Default::default()
			})
			.collect();
		unsafe {
			let layout = d
				.create_descriptor_set_layout(
					&vk::DescriptorSetLayoutCreateInfo {
						binding_count: bindings.len() as u32,
						p_bindings: bindings.as_ptr(),
						..Default::default()
					},
					None,
				)
				.ok()?;
			let pcr = vk::PushConstantRange {
				stage_flags: vk::ShaderStageFlags::COMPUTE,
				offset: 0,
				size: 16,
			};
			let pl_layout = d
				.create_pipeline_layout(
					&vk::PipelineLayoutCreateInfo {
						set_layout_count: 1,
						p_set_layouts: &layout,
						push_constant_range_count: 1,
						p_push_constant_ranges: &pcr,
						..Default::default()
					},
					None,
				)
				.ok()?;
			// `QWEN_ATTN_SPV=FILE` loads the kernel from disk instead, so a
			// variant is `glslc` + rerun with no Rust rebuild (as `gpubench
			// --probe --spv` does for the GEMMs).
			let spv = match std::env::var("QWEN_ATTN_SPV") {
				Ok(p) => load_spv(&std::fs::read(&p).ok()?),
				Err(_) => load_spv(quant::packed!("attn_prefill.spv")),
			};
			let sm = create_sm(d, &spv).ok()?;
			let main = CString::new("main").unwrap();
			// Subgroup reductions assume 32 lanes (the softmax rows).
			let pipeline = compute_pipeline_sg32(d, pl_layout, sm, &main).ok()?;
			let sizes = [vk::DescriptorPoolSize {
				ty: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 4,
			}];
			let pool = d
				.create_descriptor_pool(
					&vk::DescriptorPoolCreateInfo {
						max_sets: 1,
						pool_size_count: 1,
						p_pool_sizes: sizes.as_ptr(),
						..Default::default()
					},
					None,
				)
				.ok()?;
			let set = d
				.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
					descriptor_pool: pool,
					descriptor_set_count: 1,
					p_set_layouts: &layout,
					..Default::default()
				})
				.ok()?[0];
			Some(AttnRes {
				pl_layout,
				pipeline,
				set,
			})
		}
	}

	/// Causal GQA attention of `nq` query positions (absolute `t0..t0+nq`)
	/// over KV groups `g0..g1`, on the GPU; returns without waiting. `q` is
	/// `[nq][n_head * hd]` (unscaled), `k` / `v` the f16 caches `[t][n_kv *
	/// hd]` holding at least `t0 + nq` rows. `None` when the device or VRAM
	/// cannot run it (nothing is then in flight).
	#[allow(clippy::too_many_arguments)]
	pub fn attn_prefill_begin(
		&self,
		q: &[f32],
		k: &[u16],
		v: &[u16],
		nq: usize,
		t0: usize,
		n_head: usize,
		n_kv: usize,
		scale: f32,
		g0: usize,
		g1: usize,
	) -> Option<PendingAttn> {
		let hd = ATTN_HD;
		if self.lost.load(Ordering::SeqCst) || n_head != n_kv * ATTN_HPG || g1 <= g0 || g1 > n_kv {
			return None;
		}
		let ng = g1 - g0;
		let tk = t0 + nq;
		let (qs, qw) = (n_head * hd, ng * ATTN_HPG * hd);
		let (ks, kw) = (n_kv * hd, ng * hd);
		if q.len() < nq * qs || k.len() < tk * ks || v.len() < tk * ks {
			return None;
		}
		let q_bytes = (nq * qw * 4) as u64;
		let kv_bytes = (tk * kw * 2) as u64;
		let mut guard = self.attnp.lock().unwrap();
		if guard.is_none() {
			*guard = Some(self.attnp_init()?);
		}
		let r = guard.as_ref().unwrap();
		// Borrow the prefill matmul pool's slots instead of holding buffers of
		// its own: during a prefill the card is at (or past) its VRAM budget,
		// and these slots (the GEMMs' device-local outputs, 50-100 MB each
		// at a 2048-token chunk) sit idle while attention runs. Nothing else
		// is in flight: the matmul that used them has been read back.
		let (qd, _) = self.host_slot(pool::HostUse::DeviceY, 0, q_bytes)?;
		let (kd, _) = self.host_slot(pool::HostUse::DeviceY, 1, kv_bytes)?;
		let (vd, _) = self.host_slot(pool::HostUse::DeviceY, 2, kv_bytes)?;
		let (od, _) = self.host_slot(pool::HostUse::DeviceY, 3, q_bytes)?;
		let (qs_b, qs_p) = self.host_slot(pool::HostUse::Staging, 0, q_bytes)?;
		let (ks_b, ks_p) = self.host_slot(pool::HostUse::Staging, 1, kv_bytes)?;
		let (vs_b, vs_p) = self.host_slot(pool::HostUse::Staging, 2, kv_bytes)?;
		let (rb_b, _) = self.host_slot(pool::HostUse::Readback, 0, q_bytes)?;

		// Pack this call's heads / groups; each is a contiguous run per row.
		unsafe {
			let qp = qs_p as *mut f32;
			let kp = ks_p as *mut u16;
			let vp = vs_p as *mut u16;
			let (qo, ko) = (g0 * ATTN_HPG * hd, g0 * hd);
			let (qa, ka, va) = (
				q.as_ptr() as usize,
				k.as_ptr() as usize,
				v.as_ptr() as usize,
			);
			let (qpp, kpp, vpp) = (qp as usize, kp as usize, vp as usize);
			quant::par_for(nq, |i| {
				ptr::copy_nonoverlapping(
					(qa as *const f32).add(i * qs + qo),
					(qpp as *mut f32).add(i * qw),
					qw,
				);
			});
			quant::par_for(tk.div_ceil(256), |c| {
				for t in c * 256..((c + 1) * 256).min(tk) {
					ptr::copy_nonoverlapping(
						(ka as *const u16).add(t * ks + ko),
						(kpp as *mut u16).add(t * kw),
						kw,
					);
					ptr::copy_nonoverlapping(
						(va as *const u16).add(t * ks + ko),
						(vpp as *mut u16).add(t * kw),
						kw,
					);
				}
			});
		}

		unsafe {
			let infos: Vec<vk::DescriptorBufferInfo> = [qd, kd, vd, od]
				.iter()
				.map(|&buffer| vk::DescriptorBufferInfo {
					buffer,
					offset: 0,
					range: vk::WHOLE_SIZE,
				})
				.collect();
			let writes: Vec<vk::WriteDescriptorSet> = (0..4)
				.map(|i| vk::WriteDescriptorSet {
					dst_set: r.set,
					dst_binding: i as u32,
					descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
					descriptor_count: 1,
					p_buffer_info: &infos[i],
					..Default::default()
				})
				.collect();
			self.device.update_descriptor_sets(&writes, &[]);
		}
		let pc: [u32; 4] = [nq as u32, t0 as u32, ng as u32, scale.to_bits()];
		let (pipeline, pl_layout, set) = (r.pipeline, r.pl_layout, r.set);
		let cmd = self
			.submit_nowait(|cmd| unsafe {
				let d = &self.device;
				for (src, dst, n) in [
					(qs_b, qd, q_bytes),
					(ks_b, kd, kv_bytes),
					(vs_b, vd, kv_bytes),
				] {
					d.cmd_copy_buffer(
						cmd,
						src,
						dst,
						&[vk::BufferCopy {
							src_offset: 0,
							dst_offset: 0,
							size: n,
						}],
					);
				}
				let to_compute = vk::MemoryBarrier {
					src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
					dst_access_mask: vk::AccessFlags::SHADER_READ,
					..Default::default()
				};
				d.cmd_pipeline_barrier(
					cmd,
					vk::PipelineStageFlags::TRANSFER,
					vk::PipelineStageFlags::COMPUTE_SHADER,
					vk::DependencyFlags::empty(),
					&[to_compute],
					&[],
					&[],
				);
				d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
				d.cmd_bind_descriptor_sets(
					cmd,
					vk::PipelineBindPoint::COMPUTE,
					pl_layout,
					0,
					&[set],
					&[],
				);
				d.cmd_push_constants(
					cmd,
					pl_layout,
					vk::ShaderStageFlags::COMPUTE,
					0,
					std::slice::from_raw_parts(pc.as_ptr() as *const u8, 16),
				);
				d.cmd_dispatch(cmd, nq.div_ceil(QT) as u32, ng as u32, 1);
				let to_copy = vk::MemoryBarrier {
					src_access_mask: vk::AccessFlags::SHADER_WRITE,
					dst_access_mask: vk::AccessFlags::TRANSFER_READ,
					..Default::default()
				};
				d.cmd_pipeline_barrier(
					cmd,
					vk::PipelineStageFlags::COMPUTE_SHADER,
					vk::PipelineStageFlags::TRANSFER,
					vk::DependencyFlags::empty(),
					&[to_copy],
					&[],
					&[],
				);
				d.cmd_copy_buffer(
					cmd,
					od,
					rb_b,
					&[vk::BufferCopy {
						src_offset: 0,
						dst_offset: 0,
						size: q_bytes,
					}],
				);
				let to_host = vk::MemoryBarrier {
					src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
					dst_access_mask: vk::AccessFlags::HOST_READ,
					..Default::default()
				};
				d.cmd_pipeline_barrier(
					cmd,
					vk::PipelineStageFlags::TRANSFER,
					vk::PipelineStageFlags::HOST,
					vk::DependencyFlags::empty(),
					&[to_host],
					&[],
					&[],
				);
			})
			.ok()?;
		Some(PendingAttn { cmd, nq, qw })
	}

	/// Wait for [`Self::attn_prefill_begin`] and write its heads into `out`
	/// (`[nq][n_head * hd]`, heads `g0 * HPG ..`), as the caller passed them.
	pub fn attn_prefill_end(
		&self,
		p: PendingAttn,
		out: &mut [f32],
		n_head: usize,
		g0: usize,
	) -> Option<()> {
		if self.wait_submitted(p.cmd).is_err() {
			if !self.lost.swap(true, Ordering::SeqCst) {
				eprintln!("offload: Vulkan device lost; CPU only until it is rebuilt");
			}
			return None;
		}
		let src = self
			.host_slot(pool::HostUse::Readback, 0, (p.nq * p.qw * 4) as u64)?
			.1 as *const f32;
		let qs = n_head * ATTN_HD;
		let (nq, ng_w) = (p.nq, p.qw);
		assert!(out.len() >= nq * qs);
		let qo = g0 * ATTN_HPG * ATTN_HD;
		let (sa, oa) = (src as usize, out.as_mut_ptr() as usize);
		quant::par_for(nq, |i| unsafe {
			ptr::copy_nonoverlapping(
				(sa as *const f32).add(i * ng_w),
				(oa as *mut f32).add(i * qs + qo),
				ng_w,
			);
		});
		Some(())
	}
}
