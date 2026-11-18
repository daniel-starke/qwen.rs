// The SSM/delta-net dimension names below (`S`, `Hk`, `Hv`, `N`, `Tprev`) are
// the upper-case symbols used by the reference gated-delta-net formulation; the
// math is far easier to follow against llama.cpp when they keep that spelling.
#![allow(non_snake_case)]

use super::*;

impl GpuDevice {
	/// Allocate per-layer SSM-block resources (lazy) and upload the SSM params
	/// (conv1d weight, ssm_a, ssm_dt, ssm_norm) that don't live in a `GpuWeight`.
	/// Reuses the shared `BlockRes` (from `ensure_block`) for buffers the SSM
	/// layer shares with the full-attention block; only SSM-specific buffers
	/// (conv_state, st, conv_w, …) live in `SsmRes`.
	pub(crate) fn ensure_ssm(&self, il: usize, info: &GpuBlockInfo) -> Option<()> {
		self.ensure_block(il, info, false)?;
		if self.ssm_res.lock().unwrap().contains_key(&il) {
			return Some(());
		}
		if self.block_oom.lock().unwrap().contains(&il) {
			return None;
		}
		let r = self.ensure_ssm_inner(il, info);
		if r.is_none() {
			eprintln!(
				"offload: layer {il} fused SSM buffers do not fit in VRAM; this layer stays on the CPU"
			);
			self.block_oom.lock().unwrap().insert(il);
		}
		r
	}

	fn ensure_ssm_inner(&self, il: usize, info: &GpuBlockInfo) -> Option<()> {
		let S = info.ssm_state;
		let Hk = info.ssm_n_group;
		let Hv = info.ssm_dt_rank;
		let ksz = info.conv_kernel;
		let key_dim = S * Hk;
		let value_dim = S * Hv;
		let conv_dim = key_dim * 2 + value_dim;
		let dev = &self.device;

		let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
		let devp = vk::MemoryPropertyFlags::DEVICE_LOCAL;
		// device-only buffer (GPU read+write, never CPU-written).
		let mk_dev = |me: &GpuDevice, len: usize| -> Option<(vk::Buffer, vk::DeviceMemory)> {
			me.try_create_gpu_buffer(
				(len * 4) as u64,
				vk::BufferUsageFlags::STORAGE_BUFFER
					| vk::BufferUsageFlags::TRANSFER_DST
					| vk::BufferUsageFlags::TRANSFER_SRC,
				devp,
			)
		};
		// device-local (shader-read) + host-visible staging (CPU writes each token).
		let mk_stg = |me: &GpuDevice,
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
				vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
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

		let (conv_state, conv_state_mem) = mk_dev(self, (ksz - 1) * conv_dim)?;
		let (st_buf, st_mem) = mk_dev(self, S * S * Hv)?;
		let (conv_w, conv_w_mem, conv_w_stg, conv_w_stg_mem, conv_w_ptr) =
			mk_stg(self, ksz * conv_dim)?;
		let (ssm_norm_buf, ssm_norm_mem, ssm_norm_stg, ssm_norm_stg_mem, ssm_norm_ptr) =
			mk_stg(self, S)?;
		let (ssm_dt_buf, ssm_dt_mem, ssm_dt_stg, ssm_dt_stg_mem, ssm_dt_ptr) = mk_stg(self, Hv)?;
		let (ssm_a_buf, ssm_a_mem, ssm_a_stg, ssm_a_stg_mem, ssm_a_ptr) = mk_stg(self, Hv)?;
		let (bg, bg_mem) = mk_dev(self, 2 * Hv)?;
		let (conv_out, conv_out_mem) = mk_dev(self, conv_dim)?;
		let (qn, qn_mem) = mk_dev(self, key_dim)?;
		let (kn, kn_mem) = mk_dev(self, key_dim)?;
		let (v, v_mem) = mk_dev(self, value_dim)?;
		let (o, o_mem) = mk_dev(self, value_dim)?;
		let (no, no_mem) = mk_dev(self, value_dim)?;

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
			.expect("ssm desc pool")
		};

		self.ssm_res.lock().unwrap().insert(
			il,
			SsmRes {
				aux_pool: pool,
				param_bufs,
				conv_state,
				conv_state_mem,
				st_buf,
				st_mem,
				conv_w,
				conv_w_mem,
				conv_w_stg,
				conv_w_stg_mem,
				conv_w_ptr,
				ssm_norm_buf,
				ssm_norm_mem,
				ssm_norm_stg,
				ssm_norm_stg_mem,
				ssm_norm_ptr,
				ssm_dt_buf,
				ssm_dt_mem,
				ssm_dt_stg,
				ssm_dt_stg_mem,
				ssm_dt_ptr,
				ssm_a_buf,
				ssm_a_mem,
				ssm_a_stg,
				ssm_a_stg_mem,
				ssm_a_ptr,
				bg,
				bg_mem,
				conv_out,
				conv_out_mem,
				qn,
				qn_mem,
				kn,
				kn_mem,
				v,
				v_mem,
				o,
				o_mem,
				no,
				no_mem,
				init: std::sync::atomic::AtomicBool::new(false),
			},
		);
		Some(())
	}

	pub(crate) fn ssm_set_param(&self, sr: &SsmRes, idx: usize, arr: &[u32; 16]) {
		unsafe {
			ptr::copy_nonoverlapping(
				arr.as_ptr() as *const u8,
				sr.param_bufs[idx].4 .0 as *mut u8,
				64,
			);
		}
	}

	pub(crate) fn ssm_aux_set(
		&self,
		sr: &SsmRes,
		pidx: usize,
		binds: &[(u32, vk::Buffer)],
	) -> vk::DescriptorSet {
		let set = unsafe {
			self.device
				.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
					descriptor_pool: sr.aux_pool,
					descriptor_set_count: 1,
					p_set_layouts: &self.aux_layout,
					..Default::default()
				})
				.expect("ssm aux set")[0]
		};
		// See `aux_set`: the DescriptorBufferInfo structs must outlive the
		// update call, so they are collected in a Vec first.
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
			buffer: sr.param_bufs[pidx].0,
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

	/// L2/L3 GPU-resident fused SSM layer (decode, single token). Runs the whole
	/// SSM layer on the GPU keeping `x` resident: one H2D upload of `x`, then
	/// RMSNorm→Wqkv→conv1d+SiLU→q/k l2-norm→beta/alpha gating→delta-net
	/// recurrence (GPU-persistent state)→rms-norm→gate→SsmOut→residual→RMSNorm→
	/// SwiGLU FFN→residual, then one D2H readback. Returns `None` (CPU fallback)
	/// if the device was lost or any SSM weight wasn't uploaded.
	pub fn ssm_block(&self, il: usize, x: &[f32], info: &GpuBlockInfo) -> Option<Vec<f32>> {
		if self.lost.load(Ordering::SeqCst) {
			return None;
		}
		self.ensure_ssm(il, info)?;
		if std::env::var("QWEN_GPU_NOBLOCK").is_ok() {
			return None;
		}
		let br_guard = self.block_res.lock().unwrap();
		let br = br_guard.get(&il).unwrap();
		let sr_guard = self.ssm_res.lock().unwrap();
		let sr = sr_guard.get(&il).unwrap();
		let n = info.n_embd;
		let nf = info.n_ff;
		let S = info.ssm_state;
		let Hk = info.ssm_n_group;
		let Hv = info.ssm_dt_rank;
		let ksz = info.conv_kernel;
		let key_dim = S * Hk;
		let value_dim = S * Hv;
		let conv_dim = key_dim * 2 + value_dim;
		let eps = info.eps.to_bits();

		unsafe {
			ptr::copy_nonoverlapping(x.as_ptr(), br.x_in_ptr.0 as *mut f32, n);
			ptr::copy_nonoverlapping(info.attn_norm.as_ptr(), br.attn_norm_ptr.0 as *mut f32, n);
			ptr::copy_nonoverlapping(info.post_norm.as_ptr(), br.post_norm_ptr.0 as *mut f32, n);
			ptr::copy_nonoverlapping(
				info.ssm_conv1d.as_ptr(),
				sr.conv_w_ptr.0 as *mut f32,
				info.ssm_conv1d.len(),
			);
			ptr::copy_nonoverlapping(
				info.ssm_norm.as_ptr(),
				sr.ssm_norm_ptr.0 as *mut f32,
				info.ssm_norm.len(),
			);
			ptr::copy_nonoverlapping(
				info.ssm_dt.as_ptr(),
				sr.ssm_dt_ptr.0 as *mut f32,
				info.ssm_dt.len(),
			);
			ptr::copy_nonoverlapping(
				info.ssm_a.as_ptr(),
				sr.ssm_a_ptr.0 as *mut f32,
				info.ssm_a.len(),
			);
		}

		// Stage probe (QWEN_GPU_STAGE): capture intermediates so one host run can
		// localize which fused stage produces garbage on real hardware.
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
		let pb_xin = if stage_dbg {
			mkp(self, "x_in", n)
		} else {
			usize::MAX
		};
		let pb_inv = if stage_dbg {
			mkp(self, "inv", 1)
		} else {
			usize::MAX
		};
		let pb_norm = if stage_dbg {
			mkp(self, "norm", n)
		} else {
			usize::MAX
		};
		let pb_qkv = if stage_dbg {
			mkp(self, "qkv(wqkv.y)", conv_dim)
		} else {
			usize::MAX
		};
		let pb_conv = if stage_dbg {
			mkp(self, "conv_out", conv_dim)
		} else {
			usize::MAX
		};
		let pb_qn = if stage_dbg {
			mkp(self, "qn", key_dim)
		} else {
			usize::MAX
		};
		let pb_kn = if stage_dbg {
			mkp(self, "kn", key_dim)
		} else {
			usize::MAX
		};
		let pb_v = if stage_dbg {
			mkp(self, "v", value_dim)
		} else {
			usize::MAX
		};
		let pb_bg = if stage_dbg {
			mkp(self, "bg", 2 * Hv)
		} else {
			usize::MAX
		};
		let pb_o = if stage_dbg {
			mkp(self, "o(delta)", value_dim)
		} else {
			usize::MAX
		};
		let pb_no1 = if stage_dbg {
			mkp(self, "no(rms)", value_dim)
		} else {
			usize::MAX
		};
		let pb_no2 = if stage_dbg {
			mkp(self, "no(gated)", value_dim)
		} else {
			usize::MAX
		};
		let pb_waout = if stage_dbg {
			mkp(self, "waout", n)
		} else {
			usize::MAX
		};
		let pb_after = if stage_dbg {
			mkp(self, "after_attn", n)
		} else {
			usize::MAX
		};
		let pb_post = if stage_dbg {
			mkp(self, "post", n)
		} else {
			usize::MAX
		};
		let pb_h = if stage_dbg {
			mkp(self, "h", nf)
		} else {
			usize::MAX
		};
		let pb_ffn = if stage_dbg {
			mkp(self, "ffn_out", n)
		} else {
			usize::MAX
		};
		let pb_xout = if stage_dbg {
			mkp(self, "x_out", n)
		} else {
			usize::MAX
		};

		let wqkv = self.weights.get(&(il, WeightId::Wqkv))?;
		let wqkvg = self.weights.get(&(il, WeightId::WqkvGate))?;
		let sout = self.weights.get(&(il, WeightId::SsmOut))?;
		let wbeta = self.weights.get(&(il, WeightId::WssmBeta))?;
		let walpha = self.weights.get(&(il, WeightId::WssmAlpha))?;
		let fup = self.weights.get(&(il, WeightId::FfnUp))?;
		let fgate = self.weights.get(&(il, WeightId::FfnGate))?;
		let fdown = self.weights.get(&(il, WeightId::FfnDown))?;

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
			self.ssm_set_param(sr, i, &p);
			i
		};
		let p_rms_app_attn = {
			let i = pidx();
			p = [n as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
			self.ssm_set_param(sr, i, &p);
			i
		};
		let p_q8_n = {
			let i = pidx();
			p = [n as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
			self.ssm_set_param(sr, i, &p);
			i
		};
		let p_q8_no = {
			let i = pidx();
			p = [
				value_dim as u32,
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
			self.ssm_set_param(sr, i, &p);
			i
		};
		let p_q8_nf = {
			let i = pidx();
			p = [nf as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
			self.ssm_set_param(sr, i, &p);
			i
		};
		let p_conv1d = {
			let i = pidx();
			p = [
				conv_dim as u32,
				ksz as u32,
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
			self.ssm_set_param(sr, i, &p);
			i
		};
		// p = [S, n_groups, in_off, eps, out_off, ...]: q and k are two slices of
		// the packed conv output, but each normalizes into its own offset-0 buffer.
		let p_l2_q = {
			let i = pidx();
			p = [
				S as u32, Hk as u32, 0, eps, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			];
			self.ssm_set_param(sr, i, &p);
			i
		};
		let p_l2_k = {
			let i = pidx();
			p = [
				S as u32,
				Hk as u32,
				key_dim as u32,
				eps,
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
			self.ssm_set_param(sr, i, &p);
			i
		};
		let p_ssm_beta = {
			let i = pidx();
			p = [Hv as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
			self.ssm_set_param(sr, i, &p);
			i
		};
		let p_ssm_gate = {
			let i = pidx();
			p = [Hv as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
			self.ssm_set_param(sr, i, &p);
			i
		};
		let p_delta = {
			let i = pidx();
			p = [
				S as u32, Hk as u32, Hv as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			];
			self.ssm_set_param(sr, i, &p);
			i
		};
		let p_ssm_rms_o = {
			let i = pidx();
			p = [
				S as u32, Hv as u32, eps, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			];
			self.ssm_set_param(sr, i, &p);
			i
		};
		let p_silu_no = {
			let i = pidx();
			p = [
				value_dim as u32,
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
			self.ssm_set_param(sr, i, &p);
			i
		};
		let p_rms_sum_post = {
			let i = pidx();
			p = [n as u32, eps, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
			self.ssm_set_param(sr, i, &p);
			i
		};
		let p_rms_app_post = {
			let i = pidx();
			p = [n as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
			self.ssm_set_param(sr, i, &p);
			i
		};
		let p_res1 = {
			let i = pidx();
			p = [n as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
			self.ssm_set_param(sr, i, &p);
			i
		};
		let p_res2 = {
			let i = pidx();
			p = [n as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
			self.ssm_set_param(sr, i, &p);
			i
		};
		let p_silu = {
			let i = pidx();
			p = [nf as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
			self.ssm_set_param(sr, i, &p);
			i
		};

		let s_rms_sum_attn = self.ssm_aux_set(sr, p_rms_sum_attn, &[(0, br.x_in), (1, br.inv)]);
		let s_rms_app_attn = self.ssm_aux_set(
			sr,
			p_rms_app_attn,
			&[(0, br.x_in), (1, br.attn_norm), (2, br.norm), (3, br.inv)],
		);
		let s_q8_wqkv = self.ssm_aux_set(sr, p_q8_n, &[(0, br.norm), (1, wqkv.x_buf)]);
		let s_q8_beta = self.ssm_aux_set(sr, p_q8_n, &[(0, br.norm), (1, wbeta.x_buf)]);
		let s_q8_alpha = self.ssm_aux_set(sr, p_q8_n, &[(0, br.norm), (1, walpha.x_buf)]);
		let s_q8_wg = self.ssm_aux_set(sr, p_q8_n, &[(0, br.norm), (1, wqkvg.x_buf)]);
		// conv1d reads the raw qkv straight from the matmul output: binding it to
		// `conv_out` (its own output) made the persistent conv window store the
		// silu'd value instead of the raw qkv, corrupting every token after the
		// first. q/k/v use the SSM-private buffers.
		let s_conv1d = self.ssm_aux_set(
			sr,
			p_conv1d,
			&[
				(0, sr.conv_state),
				(1, wqkv.y_buf),
				(2, sr.conv_w),
				(3, sr.conv_out),
			],
		);
		let s_l2_q = self.ssm_aux_set(sr, p_l2_q, &[(0, sr.conv_out), (1, sr.qn)]);
		let s_l2_k = self.ssm_aux_set(sr, p_l2_k, &[(0, sr.conv_out), (1, sr.kn)]);
		let s_ssm_beta = self.ssm_aux_set(sr, p_ssm_beta, &[(0, wbeta.y_buf), (1, sr.bg)]);
		let s_ssm_gate = self.ssm_aux_set(
			sr,
			p_ssm_gate,
			&[
				(0, walpha.y_buf),
				(1, sr.ssm_dt_buf),
				(2, sr.ssm_a_buf),
				(3, sr.bg),
			],
		);
		let s_delta = self.ssm_aux_set(
			sr,
			p_delta,
			&[
				(0, sr.st_buf),
				(1, sr.kn),
				(2, sr.qn),
				(3, sr.v),
				(4, sr.o),
				(5, sr.bg),
			],
		);
		let s_ssm_rms_o = self.ssm_aux_set(
			sr,
			p_ssm_rms_o,
			&[(0, sr.o), (1, sr.ssm_norm_buf), (2, sr.no)],
		);
		// aux_silu_mul is y = silu(binding1) * binding0; the SSM gate is
		// no = rms_norm(o) * silu(z), so `no` is the multiplicand and `z` the gate.
		let s_silu_no =
			self.ssm_aux_set(sr, p_silu_no, &[(0, sr.no), (1, wqkvg.y_buf), (2, sr.no)]);
		let s_q8_no = self.ssm_aux_set(sr, p_q8_no, &[(0, sr.no), (1, sout.x_buf)]);
		let s_rms_sum_post =
			self.ssm_aux_set(sr, p_rms_sum_post, &[(0, br.after_attn), (1, br.inv)]);
		let s_rms_app_post = self.ssm_aux_set(
			sr,
			p_rms_app_post,
			&[
				(0, br.after_attn),
				(1, br.post_norm),
				(2, br.post),
				(3, br.inv),
			],
		);
		let s_q8_up = self.ssm_aux_set(sr, p_q8_n, &[(0, br.post), (1, fup.x_buf)]);
		let s_q8_fg = self.ssm_aux_set(sr, p_q8_n, &[(0, br.post), (1, fgate.x_buf)]);
		let s_silu = self.ssm_aux_set(sr, p_silu, &[(0, br.up), (1, br.fg), (2, br.h)]);
		let s_res1 = self.ssm_aux_set(
			sr,
			p_res1,
			&[(0, br.x_in), (1, br.waout), (2, br.after_attn)],
		);
		let s_q8_down = self.ssm_aux_set(sr, p_q8_nf, &[(0, br.h), (1, fdown.x_buf)]);
		let s_res2 = self.ssm_aux_set(
			sr,
			p_res2,
			&[(0, br.after_attn), (1, br.ffn_out), (2, br.x_out)],
		);

		let g = |a: usize, b: usize| a.div_ceil(b) as u32;
		// Fused-matmul grids come from `GpuDevice::grid_x` (rows-per-workgroup is
		// per `GpuKind`); `g` is for the aux kernels only.

		let bar: Vec<vk::Buffer> = vec![
			br.x_in,
			br.x_out,
			br.norm,
			br.inv,
			br.attn_norm,
			br.post_norm,
			br.after_attn,
			br.post,
			br.up,
			br.fg,
			br.h,
			br.waout,
			br.ffn_out,
			br.rb,
			wqkv.x_buf,
			wqkv.y_buf,
			wqkvg.x_buf,
			wqkvg.y_buf,
			sout.x_buf,
			sout.y_buf,
			wbeta.x_buf,
			wbeta.y_buf,
			walpha.x_buf,
			walpha.y_buf,
			fup.x_buf,
			fup.y_buf,
			fgate.x_buf,
			fgate.y_buf,
			fdown.x_buf,
			fdown.y_buf,
			sr.conv_state,
			sr.conv_out,
			sr.conv_w,
			sr.ssm_norm_buf,
			sr.ssm_dt_buf,
			sr.ssm_a_buf,
			sr.bg,
			sr.st_buf,
			sr.o,
			sr.no,
			sr.qn,
			sr.kn,
			sr.v,
		];

		let ok = self.submit_and_wait(|cmd| {
			// One-time zero-init of the GPU-persistent recurrent state
			// (conv_state / st_buf are device-local and never CPU-written).
			if !sr.init.load(Ordering::SeqCst) {
				unsafe {
					self.device
						.cmd_fill_buffer(cmd, sr.conv_state, 0, vk::WHOLE_SIZE, 0);
					self.device
						.cmd_fill_buffer(cmd, sr.st_buf, 0, vk::WHOLE_SIZE, 0);
				}
				sr.init.store(true, Ordering::SeqCst);
			}
			// Copy host-written inputs from staging -> device-local buffers, then a
			// TRANSFER->COMPUTE barrier (host-visible memory cannot be bound as
			// STORAGE/UNIFORM and read by shaders on the GTX 1060).
			// `VkBufferCopy::size` must be the real byte count: VK_WHOLE_SIZE is
			// only legal for barriers / descriptor ranges / fills, and passing it
			// here asks the DMA engine for a ~16 EiB copy, which hangs the GTX
			// 1060 (the driver never faults, so even the fence timeout is not
			// reached). Sizes mirror the `mk_stg`/`mkstaged` allocations, which
			// clamp to a 4-byte minimum.
			let cbytes = |len: usize| -> u64 { ((len * 4) as u64).max(4) };
			let staged_copies: Vec<(vk::Buffer, vk::Buffer, u64)> = vec![
				(br.x_in_stg, br.x_in, cbytes(n)),
				(br.attn_norm_stg, br.attn_norm, cbytes(n)),
				(br.post_norm_stg, br.post_norm, cbytes(n)),
				(sr.conv_w_stg, sr.conv_w, cbytes(ksz * conv_dim)),
				(sr.ssm_norm_stg, sr.ssm_norm_buf, cbytes(S)),
				(sr.ssm_dt_stg, sr.ssm_dt_buf, cbytes(Hv)),
				(sr.ssm_a_stg, sr.ssm_a_buf, cbytes(Hv)),
			];
			let param_copies: Vec<(vk::Buffer, vk::Buffer, u64)> = sr
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
			let dev_inputs: Vec<vk::Buffer> = vec![
				br.x_in,
				br.attn_norm,
				br.post_norm,
				sr.conv_w,
				sr.ssm_norm_buf,
				sr.ssm_dt_buf,
				sr.ssm_a_buf,
			]
			.into_iter()
			.chain(sr.param_bufs.iter().map(|(b, _m, _sb, _sm, _p)| *b))
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

			// Also orders COMPUTE writes before the TRANSFER reads of the
			// `y_buf -> scratch` copies that follow every matmul dispatch.
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
			let copy = |cmd: vk::CommandBuffer,
			            src: vk::Buffer,
			            src_off: u64,
			            dst: vk::Buffer,
			            bytes: u64| {
				unsafe {
					self.device.cmd_copy_buffer(
						cmd,
						src,
						dst,
						&[vk::BufferCopy {
							src_offset: src_off,
							dst_offset: 0,
							size: bytes,
							..Default::default()
						}],
					);
				}
			};

			let feed = |cmd: vk::CommandBuffer,
			            kind: GpuKind,
			            gw: &GpuWeight,
			            src: vk::Buffer,
			            n_f32: usize,
			            s_q8: vk::DescriptorSet| {
				if matches!(kind, GpuKind::Q4K | GpuKind::Q6K) {
					// The q8 quantizer processes `n_f32` activation elements
					// (local_size_x = 256), so it needs `g(n_f32, 256)` workgroups,
					// NOT the matmul's group count.
					bind(
						cmd,
						self.pl_q8,
						self.aux_pipeline_layout,
						s_q8,
						g(n_f32, 256),
						1,
					);
					barrier(cmd);
				} else {
					// F16/F32/Q8_0 matmuls read f32 activation directly.
					copy(cmd, src, 0, gw.x_buf, (n_f32 * 4) as u64);
					barrier(cmd);
				}
			};

			if stage_dbg {
				self.record_probe_copy(cmd, br.x_in, probes[pb_xin].1, n);
			}
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
				self.record_probe_copy(cmd, br.inv, probes[pb_inv].1, 1);
			}
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
				self.record_probe_copy(cmd, br.norm, probes[pb_norm].1, n);
			}

			feed(cmd, wqkv.kind, wqkv, br.norm, n, s_q8_wqkv);
			bind(
				cmd,
				self.pipeline_for(wqkv.kind),
				self.pipeline_layout,
				wqkv.desc_set,
				self.grid_x(wqkv.kind, conv_dim),
				1,
			);
			barrier(cmd);
			if stage_dbg {
				self.record_probe_copy(cmd, wqkv.y_buf, probes[pb_qkv].1, conv_dim);
			}

			feed(cmd, wqkvg.kind, wqkvg, br.norm, n, s_q8_wg);
			bind(
				cmd,
				self.pipeline_for(wqkvg.kind),
				self.pipeline_layout,
				wqkvg.desc_set,
				self.grid_x(wqkvg.kind, value_dim),
				1,
			);
			barrier(cmd);

			bind(
				cmd,
				self.pl_conv1d,
				self.aux_pipeline_layout,
				s_conv1d,
				g(conv_dim, 256),
				1,
			);
			barrier(cmd);
			// one invocation per group, not per element
			bind(
				cmd,
				self.pl_l2norm,
				self.aux_pipeline_layout,
				s_l2_q,
				g(Hk, 256),
				1,
			);
			bind(
				cmd,
				self.pl_l2norm,
				self.aux_pipeline_layout,
				s_l2_k,
				g(Hk, 256),
				1,
			);
			copy(
				cmd,
				sr.conv_out,
				(2 * key_dim * 4) as u64,
				sr.v,
				(value_dim * 4) as u64,
			);
			barrier(cmd);
			if stage_dbg {
				self.record_probe_copy(cmd, sr.conv_out, probes[pb_conv].1, conv_dim);
				self.record_probe_copy(cmd, sr.qn, probes[pb_qn].1, key_dim);
				self.record_probe_copy(cmd, sr.kn, probes[pb_kn].1, key_dim);
				self.record_probe_copy(cmd, sr.v, probes[pb_v].1, value_dim);
			}

			feed(cmd, wbeta.kind, wbeta, br.norm, n, s_q8_beta);
			bind(
				cmd,
				self.pipeline_for(wbeta.kind),
				self.pipeline_layout,
				wbeta.desc_set,
				self.grid_x(wbeta.kind, Hv),
				1,
			);
			barrier(cmd);
			bind(
				cmd,
				self.pl_ssm_beta,
				self.aux_pipeline_layout,
				s_ssm_beta,
				g(Hv, 256),
				1,
			);
			barrier(cmd);

			feed(cmd, walpha.kind, walpha, br.norm, n, s_q8_alpha);
			bind(
				cmd,
				self.pipeline_for(walpha.kind),
				self.pipeline_layout,
				walpha.desc_set,
				self.grid_x(walpha.kind, Hv),
				1,
			);
			barrier(cmd);
			bind(
				cmd,
				self.pl_ssm_gate,
				self.aux_pipeline_layout,
				s_ssm_gate,
				g(Hv, 256),
				1,
			);
			barrier(cmd);

			if stage_dbg {
				self.record_probe_copy(cmd, sr.bg, probes[pb_bg].1, 2 * Hv);
			}
			// one invocation per (head, state column): Hv*S threads, not Hv
			bind(
				cmd,
				self.pl_delta_net,
				self.aux_pipeline_layout,
				s_delta,
				g(Hv * S, 256),
				1,
			);
			barrier(cmd);
			if stage_dbg {
				self.record_probe_copy(cmd, sr.o, probes[pb_o].1, value_dim);
			}

			bind(
				cmd,
				self.pl_ssm_rms_o,
				self.aux_pipeline_layout,
				s_ssm_rms_o,
				g(Hv, 256),
				1,
			);
			barrier(cmd);
			if stage_dbg {
				self.record_probe_copy(cmd, sr.no, probes[pb_no1].1, value_dim);
			}
			bind(
				cmd,
				self.pl_silu,
				self.aux_pipeline_layout,
				s_silu_no,
				g(value_dim, 256),
				1,
			);
			barrier(cmd);
			if stage_dbg {
				self.record_probe_copy(cmd, sr.no, probes[pb_no2].1, value_dim);
			}

			feed(cmd, sout.kind, sout, sr.no, value_dim, s_q8_no);
			bind(
				cmd,
				self.pipeline_for(sout.kind),
				self.pipeline_layout,
				sout.desc_set,
				self.grid_x(sout.kind, n),
				1,
			);
			barrier(cmd);
			copy(cmd, sout.y_buf, 0, br.waout, (n * 4) as u64);
			barrier(cmd);

			if stage_dbg {
				self.record_probe_copy(cmd, br.waout, probes[pb_waout].1, n);
			}
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
				self.record_probe_copy(cmd, br.after_attn, probes[pb_after].1, n);
			}

			bind(
				cmd,
				self.pl_rms_sum,
				self.aux_pipeline_layout,
				s_rms_sum_post,
				1,
				1,
			);
			barrier(cmd);
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
				self.record_probe_copy(cmd, br.post, probes[pb_post].1, n);
			}

			feed(cmd, fup.kind, fup, br.post, n, s_q8_up);
			bind(
				cmd,
				self.pipeline_for(fup.kind),
				self.pipeline_layout,
				fup.desc_set,
				self.grid_x(fup.kind, nf),
				1,
			);
			barrier(cmd);
			copy(cmd, fup.y_buf, 0, br.up, (nf * 4) as u64);
			barrier(cmd);

			feed(cmd, fgate.kind, fgate, br.post, n, s_q8_fg);
			bind(
				cmd,
				self.pipeline_for(fgate.kind),
				self.pipeline_layout,
				fgate.desc_set,
				self.grid_x(fgate.kind, nf),
				1,
			);
			barrier(cmd);
			copy(cmd, fgate.y_buf, 0, br.fg, (nf * 4) as u64);
			barrier(cmd);

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
				self.record_probe_copy(cmd, br.h, probes[pb_h].1, nf);
			}

			feed(cmd, fdown.kind, fdown, br.h, nf, s_q8_down);
			bind(
				cmd,
				self.pipeline_for(fdown.kind),
				self.pipeline_layout,
				fdown.desc_set,
				self.grid_x(fdown.kind, n),
				1,
			);
			barrier(cmd);
			copy(cmd, fdown.y_buf, 0, br.ffn_out, (n * 4) as u64);
			barrier(cmd);

			if stage_dbg {
				self.record_probe_copy(cmd, br.ffn_out, probes[pb_ffn].1, n);
			}
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
				self.record_probe_copy(cmd, br.x_out, probes[pb_xout].1, n);
			}
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
		if stage_dbg {
			for (label, b, m, ptr, len) in &probes {
				let s = unsafe { std::slice::from_raw_parts(*ptr, *len) };
				let sum_abs: f32 = s.iter().map(|v| v.abs()).sum();
				let max_abs = s.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
				let first = s.first().copied().unwrap_or(0.0);
				eprintln!("[SSMSTAGE] {label}: sum_abs={sum_abs:.4} max_abs={max_abs:.4} first={first:.6}");
				unsafe {
					self.device.unmap_memory(*m);
					self.device.destroy_buffer(*b, None);
					self.device.free_memory(*m, None);
				}
			}
		}
		let _ = unsafe {
			self.device
				.reset_descriptor_pool(sr.aux_pool, vk::DescriptorPoolResetFlags::empty())
		};
		res
	}
}
