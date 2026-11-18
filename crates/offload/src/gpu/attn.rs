use super::*;

impl GpuDevice {
	/// Lazily allocate the per-layer fused-block resources. Returns `None`
	/// when the device cannot back them (the weights already commit ~96 % of
	/// VRAM at the recommended 22-layer setting), in which case the layer
	/// permanently falls back to the CPU path instead of retrying per token.
	pub(crate) fn ensure_block(&self, il: usize, info: &GpuBlockInfo, is_full: bool) -> Option<()> {
		if self.block_res.lock().unwrap().contains_key(&il) {
			return Some(());
		}
		if self.block_oom.lock().unwrap().contains(&il) {
			return None;
		}
		let r = self.ensure_block_inner(il, info, is_full);
		if r.is_none() {
			eprintln!(
					"offload: layer {il} fused-block buffers do not fit in VRAM; this layer stays on the CPU"
				);
			self.block_oom.lock().unwrap().insert(il);
		}
		r
	}

	fn ensure_block_inner(&self, il: usize, info: &GpuBlockInfo, is_full: bool) -> Option<()> {
		let n = info.n_embd;
		let nf = info.n_ff;
		let hd = info.head_dim;
		let nh = info.n_head;
		let nkv = info.n_head_kv;
		let max_ctx = self.max_ctx;
		let dev = &self.device;

		let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
		let devp = vk::MemoryPropertyFlags::DEVICE_LOCAL;

		let mk = |me: &GpuDevice,
		          bytes: u64,
		          usage: vk::BufferUsageFlags,
		          props: vk::MemoryPropertyFlags|
		 -> Option<(vk::Buffer, vk::DeviceMemory)> {
			me.try_create_gpu_buffer(bytes, usage, props)
		};
		let sbuf = vk::BufferUsageFlags::STORAGE_BUFFER;

		// mkstaged: device-local STORAGE buffer (read by shader) + host-visible
		// staging buffer (CPU writes, then copied dev); host-visible memory
		// cannot be bound as STORAGE/UNIFORM and read by shaders on the GTX 1060.
		let mkstaged = |me: &GpuDevice,
		                len: usize|
		 -> Option<(
			vk::Buffer,
			vk::DeviceMemory,
			vk::Buffer,
			vk::DeviceMemory,
			MappedPtr,
		)> {
			let (b, m) = me.try_create_gpu_buffer(
				(len * 4) as u64,
				sbuf | vk::BufferUsageFlags::TRANSFER_DST,
				devp,
			)?;
			let (sb, sm) = me.try_create_gpu_buffer(
				(len * 4) as u64,
				vk::BufferUsageFlags::TRANSFER_SRC,
				host,
			)?;
			let sp = MappedPtr(me.map(sm, len * 4));
			Some((b, m, sb, sm, sp))
		};

		let (x_in, x_in_mem, x_in_stg, x_in_stg_mem, x_in_ptr) = mkstaged(self, n)?;
		let (x_out, x_out_mem) = mk(
			self,
			(n * 4) as u64,
			sbuf | vk::BufferUsageFlags::TRANSFER_SRC,
			devp,
		)?;
		let (rb, rb_mem) = mk(
			self,
			(n * 4) as u64,
			vk::BufferUsageFlags::TRANSFER_DST,
			host,
		)?;
		let rb_ptr = MappedPtr(self.map(rb_mem, n * 4));

		// TRANSFER_DST is unconditional: several of these device-only buffers
		// are `cmd_copy_buffer` targets (a matmul `y_buf` is copied into
		// `up`/`fg`/`waout`/`ffn_out`), and a copy into a buffer created
		// without TRANSFER_DST is invalid usage.
		let dev_only = |me: &GpuDevice,
		                len: usize,
		                extra: vk::BufferUsageFlags|
		 -> Option<(vk::Buffer, vk::DeviceMemory)> {
			me.try_create_gpu_buffer(
				(len * 4) as u64,
				vk::BufferUsageFlags::STORAGE_BUFFER
					| vk::BufferUsageFlags::TRANSFER_SRC
					| vk::BufferUsageFlags::TRANSFER_DST
					| extra,
				devp,
			)
		};

		let (norm, norm_mem) = dev_only(self, n, vk::BufferUsageFlags::empty())?;
		let (qfull, qfull_mem) = dev_only(self, nh * 2 * hd, vk::BufferUsageFlags::empty())?;
		let (k, k_mem) = dev_only(self, nkv * hd, vk::BufferUsageFlags::empty())?;
		let (v, v_mem) = dev_only(self, nkv * hd, vk::BufferUsageFlags::empty())?;
		let (qn, qn_mem) = dev_only(self, nh * hd, vk::BufferUsageFlags::empty())?;
		let (kn, kn_mem) = dev_only(self, nkv * hd, vk::BufferUsageFlags::empty())?;
		let (gate, gate_mem) = dev_only(self, nh * hd, vk::BufferUsageFlags::empty())?;
		let (aout, aout_mem) = dev_only(self, nh * hd, vk::BufferUsageFlags::empty())?;
		let (waout, waout_mem) = dev_only(self, n, vk::BufferUsageFlags::empty())?;
		let (after_attn, after_attn_mem) = dev_only(self, n, vk::BufferUsageFlags::empty())?;
		let (post, post_mem) = dev_only(self, n, vk::BufferUsageFlags::empty())?;
		let (up, up_mem) = dev_only(self, nf, vk::BufferUsageFlags::empty())?;
		let (fg, fg_mem) = dev_only(self, nf, vk::BufferUsageFlags::empty())?;
		let (h, h_mem) = dev_only(self, nf, vk::BufferUsageFlags::empty())?;
		let (ffn_out, ffn_out_mem) = dev_only(self, n, vk::BufferUsageFlags::empty())?;
		let (inv, inv_mem) = dev_only(self, 4, vk::BufferUsageFlags::empty())?;

		// KV cache + attention scratch are only needed by the full-attention
		// block. For SSM (recurrent) layers allocate 1-byte dummies so the
		// (shared) `BlockRes` layout stays uniform and we don't waste ~33 MB
		// of VRAM per SSM layer on an unused KV cache.
		let dummy = || dev_only(self, 1, vk::BufferUsageFlags::empty());
		let (kcache, kcache_mem) = if is_full {
			dev_only(self, max_ctx * nkv * hd, vk::BufferUsageFlags::TRANSFER_DST)?
		} else {
			dummy()?
		};
		let (vcache, vcache_mem) = if is_full {
			dev_only(self, max_ctx * nkv * hd, vk::BufferUsageFlags::TRANSFER_DST)?
		} else {
			dummy()?
		};
		let (scratch, scratch_mem) = if is_full {
			dev_only(self, max_ctx * nh, vk::BufferUsageFlags::empty())?
		} else {
			dummy()?
		};

		let (attn_norm, attn_norm_mem, attn_norm_stg, attn_norm_stg_mem, attn_norm_ptr) =
			mkstaged(self, n)?;
		let (post_norm, post_norm_mem, post_norm_stg, post_norm_stg_mem, post_norm_ptr) =
			mkstaged(self, n)?;
		let (q_norm, q_norm_mem, q_norm_stg, q_norm_stg_mem, q_norm_ptr) = mkstaged(self, hd)?;
		let (k_norm, k_norm_mem, k_norm_stg, k_norm_stg_mem, k_norm_ptr) = mkstaged(self, hd)?;

		// Param buffers (uniform, device-local) + host-visible staging: one per aux dispatch.
		let mut param_bufs = Vec::new();
		for _ in 0..64 {
			let (b, m) = self.try_create_gpu_buffer(
				64,
				vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
				devp,
			)?;
			let (sb, sm) = self.try_create_gpu_buffer(
				64,
				vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
				host,
			)?;
			let sp = MappedPtr(self.map(sm, 64));
			param_bufs.push((b, m, sb, sm, sp));
		}

		let pool_sizes = [
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 512,
			},
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::UNIFORM_BUFFER,
				descriptor_count: 64,
			},
		];
		let pool = unsafe {
			dev.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo {
					max_sets: 64,
					pool_size_count: pool_sizes.len() as u32,
					p_pool_sizes: pool_sizes.as_ptr(),
					..Default::default()
				},
				None,
			)
		}
		.expect("block desc pool");

		self.block_res.lock().unwrap().insert(
			il,
			BlockRes {
				aux_pool: pool,
				param_bufs,
				attn_norm,
				attn_norm_mem,
				attn_norm_stg,
				attn_norm_stg_mem,
				attn_norm_ptr,
				post_norm,
				post_norm_mem,
				post_norm_stg,
				post_norm_stg_mem,
				post_norm_ptr,
				q_norm,
				q_norm_mem,
				q_norm_stg,
				q_norm_stg_mem,
				q_norm_ptr,
				k_norm,
				k_norm_mem,
				k_norm_stg,
				k_norm_stg_mem,
				k_norm_ptr,
				x_in,
				x_in_mem,
				x_in_stg,
				x_in_stg_mem,
				x_in_ptr,
				x_out,
				x_out_mem,
				rb,
				rb_mem,
				rb_ptr,
				norm,
				norm_mem,
				qfull,
				qfull_mem,
				k,
				k_mem,
				v,
				v_mem,
				qn,
				qn_mem,
				kn,
				kn_mem,
				gate,
				gate_mem,
				aout,
				aout_mem,
				waout,
				waout_mem,
				after_attn,
				after_attn_mem,
				post,
				post_mem,
				up,
				up_mem,
				fg,
				fg_mem,
				h,
				h_mem,
				ffn_out,
				ffn_out_mem,
				inv,
				inv_mem,
				kcache,
				kcache_mem,
				vcache,
				vcache_mem,
				scratch,
				scratch_mem,
			},
		);
		Some(())
	}

	/// Write a 16-u32 param block into param-buffer `idx`.
	pub(crate) fn set_param(&self, br: &BlockRes, idx: usize, arr: &[u32; 16]) {
		unsafe {
			ptr::copy_nonoverlapping(
				arr.as_ptr() as *const u8,
				br.param_bufs[idx].4 .0 as *mut u8,
				64,
			);
		}
	}

	/// Allocate (from the block's descriptor pool) an aux descriptor set binding
	/// `binds` (storage buffers) plus the uniform param buffer `pidx`.
	pub(crate) fn aux_set(
		&self,
		br: &BlockRes,
		pidx: usize,
		binds: &[(u32, vk::Buffer)],
	) -> vk::DescriptorSet {
		let set = unsafe {
			self.device
				.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
					descriptor_pool: br.aux_pool,
					descriptor_set_count: 1,
					p_set_layouts: &self.aux_layout,
					..Default::default()
				})
				.expect("aux set")[0]
		};
		// `p_buffer_info` must outlive the `update_descriptor_sets` call:
		// `&vk::DescriptorBufferInfo { .. }` written inline in the struct
		// literal is a temporary dropped at the end of the `push` statement,
		// so every storage binding ended up aliasing whichever info struct
		// happened to occupy that stack slot last (in practice: all of them
		// pointed at the final binding's buffer). Keep them in a Vec that
		// lives until the update.
		let mut infos: Vec<vk::DescriptorBufferInfo> = Vec::with_capacity(binds.len() + 1);
		for (_b, buf) in binds {
			infos.push(vk::DescriptorBufferInfo {
				buffer: *buf,
				offset: 0,
				range: vk::WHOLE_SIZE,
				..Default::default()
			});
		}
		infos.push(vk::DescriptorBufferInfo {
			buffer: br.param_bufs[pidx].0,
			offset: 0,
			range: 64,
			..Default::default()
		});
		let mut writes = Vec::with_capacity(binds.len() + 1);
		for (i, (b, _buf)) in binds.iter().enumerate() {
			writes.push(vk::WriteDescriptorSet {
				dst_set: set,
				dst_binding: *b,
				descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 1,
				p_buffer_info: &infos[i],
				..Default::default()
			});
		}
		writes.push(vk::WriteDescriptorSet {
			dst_set: set,
			dst_binding: 8,
			descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
			descriptor_count: 1,
			p_buffer_info: &infos[binds.len()],
			..Default::default()
		});
		unsafe { self.device.update_descriptor_sets(&writes, &[]) };
		set
	}

	/// Copy `src` into a host-visible probe buffer (for `QWEN_GPU_STAGE`
	/// debugging) with a COMPUTE->TRANSFER barrier so the snapshot is valid.
	/// `len` is the element count; the copy uses an explicit byte size
	/// (never `VK_WHOLE_SIZE`, which hangs the real driver on oversize copies).
	pub(crate) fn record_probe_copy(
		&self,
		cmd: vk::CommandBuffer,
		src: vk::Buffer,
		dst: vk::Buffer,
		len: usize,
	) {
		unsafe {
			self.device.cmd_copy_buffer(
				cmd,
				src,
				dst,
				&[vk::BufferCopy {
					src_offset: 0,
					dst_offset: 0,
					size: (len * 4) as u64,
					..Default::default()
				}],
			);
			let bs = [vk::BufferMemoryBarrier {
				src_access_mask: vk::AccessFlags::SHADER_WRITE,
				dst_access_mask: vk::AccessFlags::TRANSFER_READ,
				src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
				dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
				buffer: src,
				offset: 0,
				size: vk::WHOLE_SIZE,
				..Default::default()
			}];
			self.device.cmd_pipeline_barrier(
				cmd,
				vk::PipelineStageFlags::COMPUTE_SHADER,
				vk::PipelineStageFlags::TRANSFER,
				vk::DependencyFlags::empty(),
				&[],
				&bs,
				&[],
			);
		}
	}

	/// L2/L3 GPU-resident fused full-attention layer (decode, single token).
	/// Runs the entire layer on the GPU keeping the activation in VRAM: one H2D
	/// upload of `x`, then RMSNorm→QKV matmul→q/k norm+RoPE→attention (GPU KV
	/// cache)→out proj→residual→RMSNorm→FFN→residual, then one D2H readback.
	/// Returns `None` if the device was lost so the caller falls back to CPU.
	pub fn full_attn_block(&self, il: usize, x: &[f32], info: &GpuBlockInfo) -> Option<Vec<f32>> {
		if self.lost.load(Ordering::SeqCst) {
			return None;
		}
		self.ensure_block(il, info, true)?;
		if std::env::var("QWEN_GPU_NOBLOCK").is_ok() {
			return None;
		}
		let br = self.block_res.lock().unwrap();
		let br = br.get(&il).unwrap();
		let n = info.n_embd;
		let nf = info.n_ff;
		let hd = info.head_dim;
		let nh = info.n_head;
		let nkv = info.n_head_kv;
		let pos = info.pos;
		let scale = 1.0f32 / (hd as f32).sqrt();

		// Upload host inputs (x + norm weights).
		unsafe {
			ptr::copy_nonoverlapping(x.as_ptr(), br.x_in_ptr.0 as *mut f32, n);
			ptr::copy_nonoverlapping(info.attn_norm.as_ptr(), br.attn_norm_ptr.0 as *mut f32, n);
			ptr::copy_nonoverlapping(info.post_norm.as_ptr(), br.post_norm_ptr.0 as *mut f32, n);
			ptr::copy_nonoverlapping(info.q_norm.as_ptr(), br.q_norm_ptr.0 as *mut f32, hd);
			ptr::copy_nonoverlapping(info.k_norm.as_ptr(), br.k_norm_ptr.0 as *mut f32, hd);
		}

		// Stage probe (QWEN_GPU_STAGE): capture intermediate buffers so a
		// single host run can localize which fused stage produces garbage on
		// real hardware (fused-block aux shaders are skipped on llvmpipe).
		let stage_dbg = std::env::var("QWEN_GPU_STAGE").is_ok();
		let host_props =
			vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
		let mut probes: Vec<(String, vk::Buffer, vk::DeviceMemory, *const f32, usize)> = Vec::new();
		let mut mkp = |me: &GpuDevice, label: &str, len: usize| -> usize {
			let (b, m) = me.create_gpu_buffer(
				(len * 4) as u64,
				vk::BufferUsageFlags::TRANSFER_DST,
				host_props,
			);
			let p = MappedPtr(me.map(m, len * 4));
			probes.push((label.to_string(), b, m, p.0 as *const f32, len));
			probes.len() - 1
		};
		let p_norm = if stage_dbg {
			mkp(self, "norm", n)
		} else {
			usize::MAX
		};
		let p_qfull = if stage_dbg {
			mkp(self, "qfull", nh * 2 * hd)
		} else {
			usize::MAX
		};
		let p_aout = if stage_dbg {
			mkp(self, "aout", nh * hd)
		} else {
			usize::MAX
		};
		let p_after = if stage_dbg {
			mkp(self, "after_attn", n)
		} else {
			usize::MAX
		};
		let p_post = if stage_dbg {
			mkp(self, "post", n)
		} else {
			usize::MAX
		};
		let p_h = if stage_dbg {
			mkp(self, "h", nf)
		} else {
			usize::MAX
		};
		let p_xin = if stage_dbg {
			mkp(self, "x_in", n)
		} else {
			usize::MAX
		};
		let p_attnnorm = if stage_dbg {
			mkp(self, "attn_norm", n)
		} else {
			usize::MAX
		};
		let p_inv = if stage_dbg {
			mkp(self, "inv", 4)
		} else {
			usize::MAX
		};
		let p_xout = if stage_dbg {
			mkp(self, "x_out", n)
		} else {
			usize::MAX
		};

		// Resolve the 7 matmul weights (must be uploaded).
		let wq = self.weights.get(&(il, WeightId::Wq))?;
		let wk = self.weights.get(&(il, WeightId::Wk))?;
		let wv = self.weights.get(&(il, WeightId::Wv))?;
		let wo = self.weights.get(&(il, WeightId::Wo))?;
		let fup = self.weights.get(&(il, WeightId::FfnUp))?;
		let fgate = self.weights.get(&(il, WeightId::FfnGate))?;
		let fdown = self.weights.get(&(il, WeightId::FfnDown))?;

		// Param blocks.
		let eps = info.eps.to_bits();
		let rfb = info.rope_freq_base.to_bits();
		let rfs = info.rope_freq_scale.to_bits();
		let raf = info.rope_attn_factor.to_bits();
		let sd = info.sect_dims.to_bits();
		let halfn = info.half_nrot;
		let scale_bits = scale.to_bits();
		let mut p: [u32; 16];

		let mut next_p = 0usize;
		let mut pidx = || -> usize {
			let i = next_p;
			next_p += 1;
			i
		};

		let p_rms_sum_attn = {
			let i = pidx();
			p = [n as u32, eps, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
			self.set_param(br, i, &p);
			i
		};
		let p_rms_app_attn = {
			let i = pidx();
			p = [n as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
			self.set_param(br, i, &p);
			i
		};
		let p_q8_n = {
			let i = pidx();
			p = [n as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
			self.set_param(br, i, &p);
			i
		};
		let p_prep_q = {
			let i = pidx();
			p = [
				nh as u32,
				hd as u32,
				pos as u32,
				halfn as u32,
				eps,
				rfb,
				rfs,
				raf,
				0,
				sd,
				0,
				0,
				0,
				0,
				0,
				0,
			];
			self.set_param(br, i, &p);
			i
		};
		let p_prep_k = {
			let i = pidx();
			p = [
				nkv as u32,
				hd as u32,
				pos as u32,
				halfn as u32,
				eps,
				rfb,
				rfs,
				raf,
				0,
				sd,
				0,
				0,
				0,
				0,
				0,
				0,
			];
			self.set_param(br, i, &p);
			i
		};
		let p_attn = {
			let i = pidx();
			p = [
				nh as u32,
				hd as u32,
				nkv as u32,
				pos as u32,
				scale_bits,
				self.max_ctx as u32,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
			];
			self.set_param(br, i, &p);
			i
		};
		let p_silu = {
			let i = pidx();
			p = [nf as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
			self.set_param(br, i, &p);
			i
		};
		let p_res = {
			let i = pidx();
			p = [n as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
			self.set_param(br, i, &p);
			i
		};
		let p_rms_sum_post = {
			let i = pidx();
			p = [n as u32, eps, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
			self.set_param(br, i, &p);
			i
		};
		let p_rms_app_post = {
			let i = pidx();
			p = [n as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
			self.set_param(br, i, &p);
			i
		};
		let p_q8_nf = {
			let i = pidx();
			p = [nf as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
			self.set_param(br, i, &p);
			i
		};
		// Wo consumes the attention output (`n_head*head_dim`), NOT `n_embd`:
		// quantizing it with p[0]=n wrote the q8 block scales at the wrong
		// offsets and left the tail of `aout` unquantized.
		let p_q8_aout = {
			let i = pidx();
			p = [
				(nh * hd) as u32,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
			];
			self.set_param(br, i, &p);
			i
		};

		// Aux descriptor sets.
		let s_rms_sum_attn = self.aux_set(br, p_rms_sum_attn, &[(0, br.x_in), (1, br.inv)]);
		let s_rms_app_attn = self.aux_set(
			br,
			p_rms_app_attn,
			&[(0, br.x_in), (1, br.attn_norm), (2, br.norm), (3, br.inv)],
		);
		let s_q8_wq = self.aux_set(br, p_q8_n, &[(0, br.norm), (1, wq.x_buf)]);
		let s_q8_wk = self.aux_set(br, p_q8_n, &[(0, br.norm), (1, wk.x_buf)]);
		let s_q8_wv = self.aux_set(br, p_q8_n, &[(0, br.norm), (1, wv.x_buf)]);
		let s_prep_q = self.aux_set(
			br,
			p_prep_q,
			&[(0, br.qfull), (1, br.q_norm), (2, br.gate), (3, br.qn)],
		);
		let s_prep_k = self.aux_set(br, p_prep_k, &[(0, br.k), (1, br.k_norm), (2, br.kn)]);
		let s_attn = self.aux_set(
			br,
			p_attn,
			&[
				(0, br.qn),
				(1, br.kcache),
				(2, br.vcache),
				(3, br.gate),
				(4, br.aout),
				(5, br.scratch),
			],
		);
		let s_q8_wo = self.aux_set(br, p_q8_aout, &[(0, br.aout), (1, wo.x_buf)]);
		let s_res1 = self.aux_set(
			br,
			p_res,
			&[(0, br.x_in), (1, br.waout), (2, br.after_attn)],
		);
		let s_rms_sum_post = self.aux_set(br, p_rms_sum_post, &[(0, br.after_attn), (1, br.inv)]);
		let s_rms_app_post = self.aux_set(
			br,
			p_rms_app_post,
			&[
				(0, br.after_attn),
				(1, br.post_norm),
				(2, br.post),
				(3, br.inv),
			],
		);
		let s_q8_fup = self.aux_set(br, p_q8_n, &[(0, br.post), (1, fup.x_buf)]);
		let s_q8_fg = self.aux_set(br, p_q8_n, &[(0, br.post), (1, fgate.x_buf)]);
		let s_silu = self.aux_set(br, p_silu, &[(0, br.up), (1, br.fg), (2, br.h)]);
		let s_q8_fdn = self.aux_set(br, p_q8_nf, &[(0, br.h), (1, fdown.x_buf)]);
		let s_res2 = self.aux_set(
			br,
			p_res,
			&[(0, br.after_attn), (1, br.ffn_out), (2, br.x_out)],
		);

		let g = |a: usize, b: usize| a.div_ceil(b) as u32;
		// Fused-matmul grids come from `GpuDevice::grid_x` (the kernel's
		// rows-per-workgroup is per `GpuKind`); `g` is for the aux kernels only.

		// Barrier list (all buffers touched) for inter-op COMPUTE->COMPUTE sync.
		let bar: Vec<vk::Buffer> = vec![
			br.x_in,
			br.x_out,
			br.norm,
			br.qfull,
			br.k,
			br.v,
			br.qn,
			br.kn,
			br.gate,
			br.aout,
			br.waout,
			br.after_attn,
			br.post,
			br.up,
			br.fg,
			br.h,
			br.ffn_out,
			br.inv,
			br.kcache,
			br.vcache,
			br.scratch,
			br.attn_norm,
			br.post_norm,
			br.q_norm,
			br.k_norm,
			br.rb,
			wq.x_buf,
			wq.y_buf,
			wk.x_buf,
			wk.y_buf,
			wv.x_buf,
			wv.y_buf,
			wo.x_buf,
			wo.y_buf,
			fup.x_buf,
			fup.y_buf,
			fgate.x_buf,
			fgate.y_buf,
			fdown.x_buf,
			fdown.y_buf,
		];

		let ok = self.submit_and_wait(|cmd| {
			// Copy host-written inputs from staging -> device-local buffers,
			// then a TRANSFER->COMPUTE barrier (host-visible memory cannot be
			// bound as STORAGE/UNIFORM on the GTX 1060).
			// `VkBufferCopy::size` must be the real byte count: VK_WHOLE_SIZE
			// is only legal for barriers / descriptor ranges / fills, and
			// passing it here asks the DMA engine for a ~16 EiB copy, which
			// hangs the GTX 1060 without ever faulting the device. Sizes
			// mirror the `mkstaged` allocations (4-byte minimum).
			let cbytes = |len: usize| -> u64 { ((len * 4) as u64).max(4) };
			let staged_copies: Vec<(vk::Buffer, vk::Buffer, u64)> = vec![
				(br.x_in_stg, br.x_in, cbytes(n)),
				(br.attn_norm_stg, br.attn_norm, cbytes(n)),
				(br.post_norm_stg, br.post_norm, cbytes(n)),
				(br.q_norm_stg, br.q_norm, cbytes(hd)),
				(br.k_norm_stg, br.k_norm, cbytes(hd)),
			];
			let param_copies: Vec<(vk::Buffer, vk::Buffer, u64)> = br
				.param_bufs
				.iter()
				.map(|(b, _m, sb, _sm, _p)| (*sb, *b, 64u64))
				.collect();
			for (s, d, bytes) in staged_copies.iter().chain(param_copies.iter()) {
				unsafe {
					self.device.cmd_copy_buffer(
						cmd,
						*s,
						*d,
						&[vk::BufferCopy {
							src_offset: 0,
							dst_offset: 0,
							size: *bytes,
							..Default::default()
						}],
					);
				}
			}
			let dev_inputs: Vec<vk::Buffer> =
				vec![br.x_in, br.attn_norm, br.post_norm, br.q_norm, br.k_norm]
					.into_iter()
					.chain(br.param_bufs.iter().map(|(b, _m, _sb, _sm, _p)| *b))
					.collect();
			let xfer_bar: Vec<vk::BufferMemoryBarrier> = dev_inputs
				.iter()
				.map(|&b| vk::BufferMemoryBarrier {
					src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
					dst_access_mask: vk::AccessFlags::SHADER_READ,
					src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
					dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
					buffer: b,
					offset: 0,
					size: vk::WHOLE_SIZE,
					..Default::default()
				})
				.collect();
			unsafe {
				self.device.cmd_pipeline_barrier(
					cmd,
					vk::PipelineStageFlags::TRANSFER,
					vk::PipelineStageFlags::COMPUTE_SHADER,
					vk::DependencyFlags::empty(),
					&[],
					&xfer_bar,
					&[],
				);
			}
			if stage_dbg {
				self.record_probe_copy(cmd, br.x_in, probes[p_xin].1, n);
			}
			if stage_dbg {
				self.record_probe_copy(cmd, br.attn_norm, probes[p_attnnorm].1, n);
			}

			// Also orders COMPUTE writes before the TRANSFER reads of the
			// `y_buf -> scratch` copies that follow every matmul dispatch,
			// and the KV-cache appends.
			let barrier = |cmd: vk::CommandBuffer| {
				let bs: Vec<vk::BufferMemoryBarrier> = bar
					.iter()
					.map(|&b| vk::BufferMemoryBarrier {
						src_access_mask: vk::AccessFlags::SHADER_WRITE
							| vk::AccessFlags::TRANSFER_WRITE,
						dst_access_mask: vk::AccessFlags::SHADER_READ
							| vk::AccessFlags::SHADER_WRITE
							| vk::AccessFlags::TRANSFER_READ
							| vk::AccessFlags::TRANSFER_WRITE,
						src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
						dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
						buffer: b,
						offset: 0,
						size: vk::WHOLE_SIZE,
						..Default::default()
					})
					.collect();
				unsafe {
					self.device.cmd_pipeline_barrier(
						cmd,
						vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
						vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
						vk::DependencyFlags::empty(),
						&[],
						&bs,
						&[],
					);
				}
			};

			let bind = |cmd: vk::CommandBuffer,
			            pipeline: vk::Pipeline,
			            layout: vk::PipelineLayout,
			            set: vk::DescriptorSet,
			            gx: u32,
			            gy: u32| {
				unsafe {
					self.device
						.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
					self.device.cmd_bind_descriptor_sets(
						cmd,
						vk::PipelineBindPoint::COMPUTE,
						layout,
						0,
						&[set],
						&[],
					);
					self.device.cmd_dispatch(cmd, gx, gy, 1);
				}
			};
			let copy = |cmd: vk::CommandBuffer, src: vk::Buffer, dst: vk::Buffer, bytes: u64| unsafe {
				self.device.cmd_copy_buffer(
					cmd,
					src,
					dst,
					&[vk::BufferCopy {
						src_offset: 0,
						dst_offset: 0,
						size: bytes,
						..Default::default()
					}],
				);
			};

			// 0. rmsnorm_sum(x_in) -> inv
			bind(
				cmd,
				self.pl_rms_sum,
				self.aux_pipeline_layout,
				s_rms_sum_attn,
				1,
				1,
			);
			barrier(cmd);
			if stage_dbg {
				self.record_probe_copy(cmd, br.inv, probes[p_inv].1, 4);
			}
			// 1. rmsnorm_apply -> norm
			bind(
				cmd,
				self.pl_rms_apply,
				self.aux_pipeline_layout,
				s_rms_app_attn,
				g(n, 256),
				1,
			);
			barrier(cmd);
			if stage_dbg {
				self.record_probe_copy(cmd, br.norm, probes[p_norm].1, n);
			}

			// 2-3. Wq
			bind(
				cmd,
				self.pl_q8,
				self.aux_pipeline_layout,
				s_q8_wq,
				g(n, 256),
				1,
			);
			barrier(cmd);
			bind(
				cmd,
				self.pipeline_for(wq.kind),
				self.pipeline_layout,
				wq.desc_set,
				self.grid_x(wq.kind, 2 * hd * nh),
				1,
			);
			barrier(cmd);
			copy(cmd, wq.y_buf, br.qfull, (2 * hd * nh * 4) as u64);
			barrier(cmd);
			if stage_dbg {
				self.record_probe_copy(cmd, br.qfull, probes[p_qfull].1, nh * 2 * hd);
			}
			// 4-5. Wk
			bind(
				cmd,
				self.pl_q8,
				self.aux_pipeline_layout,
				s_q8_wk,
				g(n, 256),
				1,
			);
			barrier(cmd);
			bind(
				cmd,
				self.pipeline_for(wk.kind),
				self.pipeline_layout,
				wk.desc_set,
				self.grid_x(wk.kind, hd * nkv),
				1,
			);
			barrier(cmd);
			copy(cmd, wk.y_buf, br.k, (hd * nkv * 4) as u64);
			barrier(cmd);
			// 6-7. Wv
			bind(
				cmd,
				self.pl_q8,
				self.aux_pipeline_layout,
				s_q8_wv,
				g(n, 256),
				1,
			);
			barrier(cmd);
			bind(
				cmd,
				self.pipeline_for(wv.kind),
				self.pipeline_layout,
				wv.desc_set,
				self.grid_x(wv.kind, hd * nkv),
				1,
			);
			barrier(cmd);
			copy(cmd, wv.y_buf, br.v, (hd * nkv * 4) as u64);
			barrier(cmd);
			// 8. prep_q
			bind(
				cmd,
				self.pl_prep_q,
				self.aux_pipeline_layout,
				s_prep_q,
				nh as u32,
				1,
			);
			barrier(cmd);
			// 9. prep_k
			bind(
				cmd,
				self.pl_prep_k,
				self.aux_pipeline_layout,
				s_prep_k,
				nkv as u32,
				1,
			);
			barrier(cmd);
			// 10-11. append k,v to GPU KV cache at slot pos
			unsafe {
				self.device.cmd_copy_buffer(
					cmd,
					br.kn,
					br.kcache,
					&[vk::BufferCopy {
						src_offset: 0,
						dst_offset: (pos * hd * nkv * 4) as u64,
						size: (hd * nkv * 4) as u64,
						..Default::default()
					}],
				);
				self.device.cmd_copy_buffer(
					cmd,
					br.v,
					br.vcache,
					&[vk::BufferCopy {
						src_offset: 0,
						dst_offset: (pos * hd * nkv * 4) as u64,
						size: (hd * nkv * 4) as u64,
						..Default::default()
					}],
				);
			}
			barrier(cmd);
			// 12. attn
			bind(
				cmd,
				self.pl_attn,
				self.aux_pipeline_layout,
				s_attn,
				nh as u32,
				1,
			);
			barrier(cmd);
			if stage_dbg {
				self.record_probe_copy(cmd, br.aout, probes[p_aout].1, nh * hd);
			}
			// 13-14. Wo
			bind(
				cmd,
				self.pl_q8,
				self.aux_pipeline_layout,
				s_q8_wo,
				g(nh * hd, 256),
				1,
			);
			barrier(cmd);
			bind(
				cmd,
				self.pipeline_for(wo.kind),
				self.pipeline_layout,
				wo.desc_set,
				self.grid_x(wo.kind, n),
				1,
			);
			barrier(cmd);
			copy(cmd, wo.y_buf, br.waout, (n * 4) as u64);
			barrier(cmd);
			// 15. residual x_in + waout -> after_attn
			bind(
				cmd,
				self.pl_residual,
				self.aux_pipeline_layout,
				s_res1,
				g(n, 256),
				1,
			);
			barrier(cmd);
			if stage_dbg {
				self.record_probe_copy(cmd, br.after_attn, probes[p_after].1, n);
			}
			// 16. rmsnorm_sum(after_attn) -> inv
			bind(
				cmd,
				self.pl_rms_sum,
				self.aux_pipeline_layout,
				s_rms_sum_post,
				1,
				1,
			);
			barrier(cmd);
			// 17. rmsnorm_apply -> post
			bind(
				cmd,
				self.pl_rms_apply,
				self.aux_pipeline_layout,
				s_rms_app_post,
				g(n, 256),
				1,
			);
			barrier(cmd);
			if stage_dbg {
				self.record_probe_copy(cmd, br.post, probes[p_post].1, n);
			}
			// 18-19. FfnUp
			bind(
				cmd,
				self.pl_q8,
				self.aux_pipeline_layout,
				s_q8_fup,
				g(n, 256),
				1,
			);
			barrier(cmd);
			bind(
				cmd,
				self.pipeline_for(fup.kind),
				self.pipeline_layout,
				fup.desc_set,
				self.grid_x(fup.kind, nf),
				1,
			);
			barrier(cmd);
			copy(cmd, fup.y_buf, br.up, (nf * 4) as u64);
			barrier(cmd);
			// 20-21. FfnGate
			bind(
				cmd,
				self.pl_q8,
				self.aux_pipeline_layout,
				s_q8_fg,
				g(n, 256),
				1,
			);
			barrier(cmd);
			bind(
				cmd,
				self.pipeline_for(fgate.kind),
				self.pipeline_layout,
				fgate.desc_set,
				self.grid_x(fgate.kind, nf),
				1,
			);
			barrier(cmd);
			copy(cmd, fgate.y_buf, br.fg, (nf * 4) as u64);
			barrier(cmd);
			// 22. silu(up)*gate -> h
			bind(
				cmd,
				self.pl_silu,
				self.aux_pipeline_layout,
				s_silu,
				g(nf, 256),
				1,
			);
			barrier(cmd);
			if stage_dbg {
				self.record_probe_copy(cmd, br.h, probes[p_h].1, nf);
			}
			// 23-24. FfnDown
			bind(
				cmd,
				self.pl_q8,
				self.aux_pipeline_layout,
				s_q8_fdn,
				g(nf, 256),
				1,
			);
			barrier(cmd);
			bind(
				cmd,
				self.pipeline_for(fdown.kind),
				self.pipeline_layout,
				fdown.desc_set,
				self.grid_x(fdown.kind, n),
				1,
			);
			barrier(cmd);
			copy(cmd, fdown.y_buf, br.ffn_out, (n * 4) as u64);
			barrier(cmd);
			// 25. residual after_attn + ffn_out -> x_out
			bind(
				cmd,
				self.pl_residual,
				self.aux_pipeline_layout,
				s_res2,
				g(n, 256),
				1,
			);
			barrier(cmd);
			if stage_dbg {
				self.record_probe_copy(cmd, br.x_out, probes[p_xout].1, n);
			}
			// 26. x_out -> readback (COMPUTE->TRANSFER)
			unsafe {
				let bs: Vec<vk::BufferMemoryBarrier> = vec![vk::BufferMemoryBarrier {
					src_access_mask: vk::AccessFlags::SHADER_WRITE,
					dst_access_mask: vk::AccessFlags::TRANSFER_READ,
					src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
					dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
					buffer: br.x_out,
					offset: 0,
					size: vk::WHOLE_SIZE,
					..Default::default()
				}];
				self.device.cmd_pipeline_barrier(
					cmd,
					vk::PipelineStageFlags::COMPUTE_SHADER,
					vk::PipelineStageFlags::TRANSFER,
					vk::DependencyFlags::empty(),
					&[],
					&bs,
					&[],
				);
				self.device.cmd_copy_buffer(
					cmd,
					br.x_out,
					br.rb,
					&[vk::BufferCopy {
						src_offset: 0,
						dst_offset: 0,
						size: (n * 4) as u64,
						..Default::default()
					}],
				);
			}
		});

		let res = match ok {
			Ok(()) => {
				let slice = unsafe { std::slice::from_raw_parts(br.rb_ptr.0 as *const f32, n) };
				Some(slice.to_vec())
			}
			Err(_) => None,
		};
		// Stage probe summary: print checksums of each captured intermediate.
		if stage_dbg {
			for (label, _b, _m, ptr, len) in &probes {
				let s = unsafe { std::slice::from_raw_parts(*ptr, *len) };
				let sum_abs: f32 = s.iter().map(|v| v.abs()).sum();
				let max_abs = s.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
				let first = s.first().copied().unwrap_or(0.0);
				eprintln!("[GPUSTAGE] {label}: sum_abs={sum_abs:.4} max_abs={max_abs:.4} first={first:.6}");
			}
			// Dump the raw uniform param buffers (rms_sum=idx0, rms_app=idx1)
			// so we can see what the shader actually receives for p[0]=n, p[1]=eps.
			for idx in 0..2usize {
				let pptr = br.param_bufs[idx].4 .0 as *const u32;
				let s = unsafe { std::slice::from_raw_parts(pptr, 16) };
				eprintln!("[GPUPARAM] idx{idx}: {:?}", s);
			}
		}
		// Free this call's aux descriptor sets so the per-layer pool can be reused.
		let _ = unsafe {
			self.device
				.reset_descriptor_pool(br.aux_pool, vk::DescriptorPoolResetFlags::empty())
		};
		res
	}
}
