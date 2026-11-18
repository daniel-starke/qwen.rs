// `S`/`Hk`/`Hv` below mirror the SSM dimension symbols used by the
// shaders and by `model::vec_fwd`; keep that spelling.
#![allow(non_snake_case)]

use super::*;

// The host-visible-direct-read path is unreliable under lavapipe (the
// software Vulkan ICD used for CI), so mirror the *production* upload
// path: write into a host-visible staging buffer, then `copy_buffer`
// into a device-local buffer that the shader actually reads. Outputs are
// device-local too and copied back to a host-visible readback buffer.

fn dev_buf(dev: &GpuDevice, data: &[f32]) -> (vk::Buffer, vk::DeviceMemory) {
	// Mirror the *production* upload path: the shader must read from a
	// DEVICE_LOCAL buffer (lavapipe does not expose a host-visible
	// storage buffer as readable by the compute shader, even with a
	// HOST_WRITE->SHADER_READ barrier). So stage the data through a
	// host-visible buffer and `copy_buffer` into a device-local one.
	let size = (data.len() * 4) as u64;
	let (db, dm) = dev.create_gpu_buffer(
		size,
		vk::BufferUsageFlags::STORAGE_BUFFER
			| vk::BufferUsageFlags::TRANSFER_DST
			| vk::BufferUsageFlags::TRANSFER_SRC,
		vk::MemoryPropertyFlags::DEVICE_LOCAL,
	);
	let (stg, stm) = dev.create_gpu_buffer(
		size,
		vk::BufferUsageFlags::TRANSFER_SRC,
		vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
	);
	let p = dev.map(stm, data.len() * 4);
	unsafe { ptr::copy_nonoverlapping(data.as_ptr(), p as *mut f32, data.len()) };
	dev.copy_buffer(stg, db, size);
	unsafe {
		dev.device.destroy_buffer(stg, None);
		dev.device.free_memory(stm, None);
	}
	(db, dm)
}

fn dev_param(dev: &GpuDevice, arr: &[u32; 16]) -> (vk::Buffer, vk::DeviceMemory) {
	let (db, dm) = dev.create_gpu_buffer(
		64,
		vk::BufferUsageFlags::UNIFORM_BUFFER,
		vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
	);
	let p = dev.map(dm, 64);
	unsafe { ptr::copy_nonoverlapping(arr.as_ptr() as *const u8, p as *mut u8, 64) };
	(db, dm)
}

fn read_back_f32(dev: &GpuDevice, buf: vk::Buffer, n: usize) -> Vec<f32> {
	let (rb, rm) = dev.create_gpu_buffer(
		(n * 4) as u64,
		vk::BufferUsageFlags::TRANSFER_DST,
		vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
	);
	dev.copy_buffer(buf, rb, (n * 4) as u64);
	let p = dev.map(rm, n * 4);
	let v = unsafe { std::slice::from_raw_parts(p as *const f32, n) }.to_vec();
	unsafe {
		dev.device.destroy_buffer(rb, None);
		dev.device.free_memory(rm, None);
	}
	v
}

fn read_back_u32(dev: &GpuDevice, buf: vk::Buffer, n: usize) -> Vec<u32> {
	let (rb, rm) = dev.create_gpu_buffer(
		(n * 4) as u64,
		vk::BufferUsageFlags::TRANSFER_DST,
		vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
	);
	dev.copy_buffer(buf, rb, (n * 4) as u64);
	let p = dev.map(rm, n * 4);
	let v = unsafe { std::slice::from_raw_parts(p as *const u32, n) }.to_vec();
	unsafe {
		dev.device.destroy_buffer(rb, None);
		dev.device.free_memory(rm, None);
	}
	v
}

/// Record + submit one aux-shader dispatch. `pl_layout` MUST be the
/// pipeline layout `pipeline` was created with: binding a descriptor set
/// through an incompatible layout is undefined behavior. lavapipe
/// tolerates the mismatch, the GTX 1060 returns garbage, which is why
/// this is a parameter rather than `dev.aux_pipeline_layout`.
fn dispatch(
	dev: &GpuDevice,
	pipeline: vk::Pipeline,
	pl_layout: vk::PipelineLayout,
	set: vk::DescriptorSet,
	inputs: &[vk::Buffer],
	gx: u32,
	gy: u32,
) {
	let r = dev.submit_and_wait(|cmd| {
		// Inputs may have been written by a prior transfer-copy (staging)
		// or by a prior dispatch in the same logical step; make them
		// visible to this shader's reads.
		let barriers: Vec<vk::BufferMemoryBarrier> = inputs
			.iter()
			.map(|&b| vk::BufferMemoryBarrier {
				src_access_mask: vk::AccessFlags::HOST_WRITE
					| vk::AccessFlags::TRANSFER_WRITE
					| vk::AccessFlags::SHADER_WRITE,
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
			if !barriers.is_empty() {
				dev.device.cmd_pipeline_barrier(
					cmd,
					vk::PipelineStageFlags::HOST
						| vk::PipelineStageFlags::TRANSFER
						| vk::PipelineStageFlags::COMPUTE_SHADER,
					vk::PipelineStageFlags::COMPUTE_SHADER,
					vk::DependencyFlags::empty(),
					&[],
					&barriers,
					&[],
				);
			}
			dev.device
				.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
			dev.device.cmd_bind_descriptor_sets(
				cmd,
				vk::PipelineBindPoint::COMPUTE,
				pl_layout,
				0,
				&[set],
				&[],
			);
			dev.device.cmd_dispatch(cmd, gx, gy, 1);
		}
	});
	assert!(r.is_ok(), "dispatch failed: {:?}", r);
}

// Build a *fresh* pipeline + descriptor-set layout for one aux shader.
// lavapipe mis-binds descriptor sets allocated from the device-init-time
// `aux_layout`, so tests build their own (structurally identical) layout
// to validate the shader math on the CI software Vulkan ICD. On real
// hardware the production `aux_layout` works fine.
fn fresh_aux(
	dev: &GpuDevice,
	spv: &[u8],
	nstorage: u32,
) -> (vk::Pipeline, vk::DescriptorSetLayout, vk::PipelineLayout) {
	let mut barr: [vk::DescriptorSetLayoutBinding; 9] = [vk::DescriptorSetLayoutBinding {
		binding: 0,
		descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
		descriptor_count: 1,
		stage_flags: vk::ShaderStageFlags::COMPUTE,
		p_immutable_samplers: ptr::null(),
		..Default::default()
	}; 9];
	for b in 0..nstorage {
		barr[b as usize] = vk::DescriptorSetLayoutBinding {
			binding: b,
			descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
			descriptor_count: 1,
			stage_flags: vk::ShaderStageFlags::COMPUTE,
			p_immutable_samplers: ptr::null(),
			..Default::default()
		};
	}
	barr[nstorage as usize] = vk::DescriptorSetLayoutBinding {
		binding: 8,
		descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
		descriptor_count: 1,
		stage_flags: vk::ShaderStageFlags::COMPUTE,
		p_immutable_samplers: ptr::null(),
		..Default::default()
	};
	let cnt = (nstorage + 1) as usize;
	let bindings = &barr[..cnt];
	let layout = unsafe {
		dev.device
			.create_descriptor_set_layout(
				&vk::DescriptorSetLayoutCreateInfo {
					binding_count: cnt as u32,
					p_bindings: bindings.as_ptr(),
					..Default::default()
				},
				None,
			)
			.unwrap()
	};
	let pl = unsafe {
		dev.device
			.create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo {
					set_layout_count: 1,
					p_set_layouts: &layout,
					..Default::default()
				},
				None,
			)
			.unwrap()
	};
	let code = load_spv(spv);
	let sm = unsafe {
		dev.device
			.create_shader_module(
				&vk::ShaderModuleCreateInfo {
					code_size: code.len() * 4,
					p_code: code.as_ptr(),
					..Default::default()
				},
				None,
			)
			.unwrap()
	};
	let entry = std::ffi::CString::new("main").unwrap();
	let pipe = unsafe {
		dev.device
			.create_compute_pipelines(
				vk::PipelineCache::default(),
				&[vk::ComputePipelineCreateInfo {
					stage: vk::PipelineShaderStageCreateInfo {
						stage: vk::ShaderStageFlags::COMPUTE,
						module: sm,
						p_name: entry.as_ptr(),
						..Default::default()
					},
					layout: pl,
					..Default::default()
				}],
				None,
			)
			.unwrap()[0]
	};
	(pipe, layout, pl)
}

fn fresh_set(
	dev: &GpuDevice,
	layout: vk::DescriptorSetLayout,
	pool: vk::DescriptorPool,
	binds: &[(u32, vk::Buffer)],
	pbuf: vk::Buffer,
) -> vk::DescriptorSet {
	let set = unsafe {
		dev.device
			.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
				descriptor_pool: pool,
				descriptor_set_count: 1,
				p_set_layouts: &layout,
				..Default::default()
			})
			.unwrap()[0]
	};
	let mut buf_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
	for (_b, buf) in binds {
		buf_infos.push(vk::DescriptorBufferInfo {
			buffer: *buf,
			offset: 0,
			range: vk::WHOLE_SIZE,
			..Default::default()
		});
	}
	buf_infos.push(vk::DescriptorBufferInfo {
		buffer: pbuf,
		offset: 0,
		range: 64,
		..Default::default()
	});
	let mut writes = Vec::new();
	for (i, (b, _buf)) in binds.iter().enumerate() {
		writes.push(vk::WriteDescriptorSet {
			dst_set: set,
			dst_binding: *b,
			descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
			descriptor_count: 1,
			p_buffer_info: &buf_infos[i],
			..Default::default()
		});
	}
	writes.push(vk::WriteDescriptorSet {
		dst_set: set,
		dst_binding: 8,
		descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
		descriptor_count: 1,
		p_buffer_info: &buf_infos[binds.len()],
		..Default::default()
	});
	unsafe { dev.device.update_descriptor_sets(&writes, &[]) };
	set
}

#[test]
fn gpu_rmsnorm_silu_residual_match_cpu() {
	let dev = match GpuDevice::init() {
		Ok(d) => d,
		Err(_) => {
			return;
		}
	};
	let n = 16usize;
	let x: Vec<f32> = (0..n).map(|i| (i as f32 - 8.0) * 0.3).collect();
	let w: Vec<f32> = vec![1.0; n];
	let eps = 1e-5f32;

	// RMSNorm reference.
	let mut ss = 0.0f32;
	for &v in &x {
		ss += v * v;
	}
	let inv = 1.0 / (ss / n as f32 + eps).sqrt();
	let expected: Vec<f32> = x.iter().map(|&v| v * inv).collect();

	let pool_sizes = [
		vk::DescriptorPoolSize {
			ty: vk::DescriptorType::STORAGE_BUFFER,
			descriptor_count: 64,
		},
		vk::DescriptorPoolSize {
			ty: vk::DescriptorType::UNIFORM_BUFFER,
			descriptor_count: 16,
		},
	];
	let pool = unsafe {
		dev.device
			.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo {
					max_sets: 16,
					pool_size_count: pool_sizes.len() as u32,
					p_pool_sizes: pool_sizes.as_ptr(),
					..Default::default()
				},
				None,
			)
			.unwrap()
	};

	// rmsnorm_sum -> inv, then rmsnorm_apply -> y.
	let (x_b, _x_m) = dev_buf(&dev, &x);
	let (inv_b, _inv_m) = dev_buf(&dev, &[0.0f32]);

	let p0 = [
		n as u32,
		eps.to_bits(),
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
	let (p0b, _p0m) = dev_param(&dev, &p0);
	// rmsnorm_sum -> inv, then rmsnorm_apply -> y.
	let (pl_rms_sum, l0, plyt) =
		fresh_aux(&dev, include_bytes!("../shaders/aux_rmsnorm_sum.spv"), 2);
	let s0 = fresh_set(&dev, l0, pool, &[(0u32, x_b), (1u32, inv_b)], p0b);
	dispatch(&dev, pl_rms_sum, plyt, s0, &[x_b], 1, 1);

	let (_w_b, _w_m) = dev_buf(&dev, &w);
	let (y_b, _y_m) = dev_buf(&dev, &vec![0.0f32; n]);

	let p1 = [n as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
	let (p1b, _p1m) = dev_param(&dev, &p1);
	let (pl_rms_apply, l1, plyt) =
		fresh_aux(&dev, include_bytes!("../shaders/aux_rmsnorm_apply.spv"), 4);
	let s1 = fresh_set(
		&dev,
		l1,
		pool,
		&[(0, x_b), (1, _w_b), (2, y_b), (3, inv_b)],
		p1b,
	);
	dispatch(&dev, pl_rms_apply, plyt, s1, &[x_b, _w_b, inv_b], 1, 1);

	let y = read_back_f32(&dev, y_b, n);
	for i in 0..n {
		assert!(
			(y[i] - expected[i]).abs() < 1e-3,
			"rmsnorm[{}] {} vs {}",
			i,
			y[i],
			expected[i]
		);
	}

	// silu gate: y = silu(fg) * up. Test pure silu by setting up = 1.
	let up: Vec<f32> = vec![1.0; n];
	let (up_b, _up_m) = dev_buf(&dev, &up);
	let (s_b, _s_m) = dev_buf(&dev, &x);
	let (sout_b, _sout_m) = dev_buf(&dev, &vec![0.0f32; n]);
	let p2 = [n as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
	let (p2b, _p2m) = dev_param(&dev, &p2);
	let (pl_silu, l2, plyt) = fresh_aux(&dev, include_bytes!("../shaders/aux_silu_mul.spv"), 3);
	let s2 = fresh_set(&dev, l2, pool, &[(0, up_b), (1, s_b), (2, sout_b)], p2b);
	dispatch(&dev, pl_silu, plyt, s2, &[up_b, s_b], 1, 1);
	let sout = read_back_f32(&dev, sout_b, n);
	for i in 0..n {
		let exp = x[i] / (1.0 + (-x[i]).exp());
		assert!(
			(sout[i] - exp).abs() < 1e-3,
			"silu[{}] {} vs {}",
			i,
			sout[i],
			exp
		);
	}

	// residual: a+b.
	let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
	let b2: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
	let (a_b, _a_m) = dev_buf(&dev, &a);
	let (b_b, _b_m) = dev_buf(&dev, &b2);
	let (r_b, _r_m) = dev_buf(&dev, &vec![0.0f32; n]);
	let p3 = [n as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
	let (p3b, _p3m) = dev_param(&dev, &p3);
	let (pl_residual, l3, plyt) = fresh_aux(&dev, include_bytes!("../shaders/aux_residual.spv"), 3);
	let s3 = fresh_set(&dev, l3, pool, &[(0, a_b), (1, b_b), (2, r_b)], p3b);
	dispatch(&dev, pl_residual, plyt, s3, &[a_b, b_b], 1, 1);
	let r = read_back_f32(&dev, r_b, n);
	for i in 0..n {
		assert!(
			(r[i] - (a[i] + b2[i])).abs() < 1e-4,
			"residual[{}] {} vs {}",
			i,
			r[i],
			a[i] + b2[i]
		);
	}

	unsafe {
		dev.device.destroy_descriptor_pool(pool, None);
	}
}

#[test]
fn gpu_quantize_q8k_matches_cpu() {
	let dev = match GpuDevice::init() {
		Ok(d) => d,
		Err(_) => {
			return;
		}
	};
	let n = 256usize;
	let x: Vec<f32> = (0..n)
		.map(|i| (((i * 7) % 50) as f32) / 25.0 - 1.0)
		.collect();
	let ref_q = quant::quantize_q8k(&x, n);

	let (x_b, _x_m) = dev_buf(&dev, &x);
	// out: n bytes of qs (packed four-per-u32) + 1 f32 ds.
	let out_words = (n + 4) / 4;
	let (o_b, _o_m) = dev_buf(&dev, &vec![0.0f32; out_words]);
	let p = [n as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
	let (pb, _pm) = dev_param(&dev, &p);
	let (pl_q8, layout, plyt) =
		fresh_aux(&dev, include_bytes!("../shaders/aux_quantize_q8k.spv"), 2);
	let pool_sizes = [
		vk::DescriptorPoolSize {
			ty: vk::DescriptorType::STORAGE_BUFFER,
			descriptor_count: 8,
		},
		vk::DescriptorPoolSize {
			ty: vk::DescriptorType::UNIFORM_BUFFER,
			descriptor_count: 4,
		},
	];
	let pool = unsafe {
		dev.device
			.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo {
					max_sets: 4,
					pool_size_count: pool_sizes.len() as u32,
					p_pool_sizes: pool_sizes.as_ptr(),
					..Default::default()
				},
				None,
			)
			.unwrap()
	};
	let s = fresh_set(&dev, layout, pool, &[(0, x_b), (1, o_b)], pb);
	dispatch(&dev, pl_q8, plyt, s, &[x_b], 1, 1);

	let out_words_vec = read_back_u32(&dev, o_b, out_words);
	for j in 0..n {
		let byte = (out_words_vec[j / 4] >> ((j % 4) * 8)) & 0xFF;
		let q = byte as i8;
		assert_eq!(q, ref_q.qs[j], "q8k qs[{}] {} vs {}", j, q, ref_q.qs[j]);
	}
	let ds = f32::from_bits(out_words_vec[n / 4]);
	assert!(
		(ds - ref_q.ds[0]).abs() < 1e-4,
		"q8k ds {} vs {}",
		ds,
		ref_q.ds[0]
	);

	unsafe {
		dev.device.destroy_descriptor_pool(pool, None);
	}
}

// ---- L2/L3 fused-block aux shaders: validate the novel math (RoPE +
// RMSNorm + causal attention) against CPU references. These run on the
// WSL llvmpipe ICD so the fused full-attn GPU layer has a green
// correctness gate before host validation on the GTX 1060. ----

#[test]
fn gpu_prep_q_matches_cpu() {
	let dev = match GpuDevice::init() {
		Ok(d) => d,
		Err(_) => {
			return;
		}
	};
	let nh: usize = 2;
	let hd: usize = 256;
	let halfn: usize = 128; // n_rot = 256 == head_dim -> full MRoPE
	let pos: usize = 3;
	let eps = 1e-5f32;
	let rfb = 10000.0f32;
	let rfs = 1.0f32;
	let raf = 1.0f32;
	let sect_dims = 32.0f32;

	let mut qfull: Vec<f32> = vec![0.0; nh * 2 * hd];
	let qw: Vec<f32> = vec![1.0; hd];
	for h in 0..nh {
		for d in 0..hd {
			qfull[h * hd + d] = (((h * 7 + d * 3) % 50) as f32) / 25.0 - 1.0;
			qfull[(nh + h) * hd + d] = (((h * 5 + d * 11) % 40) as f32) / 20.0 - 1.0;
		}
	}
	let (qfull_b, _m) = dev_buf(&dev, &qfull);
	let (qw_b, _m2) = dev_buf(&dev, &qw);
	let (qn_b, _m3) = dev_buf(&dev, &vec![0.0f32; nh * hd]);
	let (gate_b, _m4) = dev_buf(&dev, &vec![0.0f32; nh * hd]);

	let theta_scale = (rfb as f64).powf(-2.0 / (2.0 * halfn as f64));
	let mut exp_qn = vec![0.0f32; nh * hd];
	let mut exp_gate = vec![0.0f32; nh * hd];
	for h in 0..nh {
		let qc: Vec<f32> = (0..hd).map(|d| qfull[h * hd + d]).collect();
		let g: Vec<f32> = (0..hd).map(|d| qfull[(nh + h) * hd + d]).collect();
		let ss: f32 = qc.iter().map(|v| v * v).sum();
		let inv = 1.0 / (ss / hd as f32 + eps).sqrt();
		let qnorm: Vec<f32> = qc.iter().map(|&v| v * inv * qw[0]).collect();
		for d in 0..hd {
			let (x0, x1, pidx);
			if d < halfn {
				x0 = qnorm[d];
				x1 = qnorm[d + halfn];
				pidx = d;
			} else {
				x0 = qnorm[d - halfn];
				x1 = qnorm[d];
				pidx = d - halfn;
			}
			let theta = rfs as f64 * pos as f64 * theta_scale.powi(pidx as i32);
			let c = (theta.cos() * raf as f64) as f32;
			let s = (theta.sin() * raf as f64) as f32;
			if d < halfn {
				exp_qn[h * hd + d] = x0 * c - x1 * s;
				exp_qn[h * hd + d + halfn] = x0 * s + x1 * c;
			} else {
				exp_qn[h * hd + d - halfn] = x0 * c - x1 * s;
				exp_qn[h * hd + d] = x0 * s + x1 * c;
			}
		}
		for d in 0..hd {
			exp_gate[h * hd + d] = g[d];
		}
	}

	let p = [
		nh as u32,
		hd as u32,
		pos as u32,
		halfn as u32,
		eps.to_bits(),
		rfb.to_bits(),
		rfs.to_bits(),
		raf.to_bits(),
		0,
		sect_dims.to_bits(),
		0,
		0,
		0,
		0,
		0,
		0,
	];
	let (pb, _pm) = dev_param(&dev, &p);
	let (pl, layout, plyt) = fresh_aux(&dev, include_bytes!("../shaders/aux_prep_q.spv"), 4);
	let pool_sizes = [
		vk::DescriptorPoolSize {
			ty: vk::DescriptorType::STORAGE_BUFFER,
			descriptor_count: 16,
		},
		vk::DescriptorPoolSize {
			ty: vk::DescriptorType::UNIFORM_BUFFER,
			descriptor_count: 4,
		},
	];
	let pool = unsafe {
		dev.device
			.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo {
					max_sets: 4,
					pool_size_count: pool_sizes.len() as u32,
					p_pool_sizes: pool_sizes.as_ptr(),
					..Default::default()
				},
				None,
			)
			.unwrap()
	};
	let s = fresh_set(
		&dev,
		layout,
		pool,
		&[(0, qfull_b), (1, qw_b), (2, gate_b), (3, qn_b)],
		pb,
	);
	dispatch(&dev, pl, plyt, s, &[qfull_b, qw_b], nh as u32, 1);

	let got_qn = read_back_f32(&dev, qn_b, nh * hd);
	let got_gate = read_back_f32(&dev, gate_b, nh * hd);
	for i in 0..nh * hd {
		assert!(
			(got_qn[i] - exp_qn[i]).abs() < 1e-2,
			"prep_q qn[{}] {} vs {}",
			i,
			got_qn[i],
			exp_qn[i]
		);
		assert!(
			(got_gate[i] - exp_gate[i]).abs() < 1e-2,
			"prep_q gate[{}] {} vs {}",
			i,
			got_gate[i],
			exp_gate[i]
		);
	}
	unsafe {
		dev.device.destroy_descriptor_pool(pool, None);
	}
}

#[test]
fn gpu_prep_k_matches_cpu() {
	let dev = match GpuDevice::init() {
		Ok(d) => d,
		Err(_) => {
			return;
		}
	};
	let nkv: usize = 2;
	let hd: usize = 256;
	let halfn: usize = 128;
	let pos: usize = 3;
	let eps = 1e-5f32;
	let rfb = 10000.0f32;
	let rfs = 1.0f32;
	let raf = 1.0f32;
	let sect_dims = 32.0f32;

	let mut k: Vec<f32> = vec![0.0; nkv * hd];
	let kw: Vec<f32> = vec![1.0; hd];
	for hh in 0..nkv {
		for d in 0..hd {
			k[hh * hd + d] = (((hh * 9 + d * 5) % 50) as f32) / 25.0 - 1.0;
		}
	}
	let (k_b, _m) = dev_buf(&dev, &k);
	let (kw_b, _m2) = dev_buf(&dev, &kw);
	let (kn_b, _m3) = dev_buf(&dev, &vec![0.0f32; nkv * hd]);

	let theta_scale = (rfb as f64).powf(-2.0 / (2.0 * halfn as f64));
	let mut exp_kn = vec![0.0f32; nkv * hd];
	for hh in 0..nkv {
		let kc: Vec<f32> = (0..hd).map(|d| k[hh * hd + d]).collect();
		let ss: f32 = kc.iter().map(|v| v * v).sum();
		let inv = 1.0 / (ss / hd as f32 + eps).sqrt();
		let knorm: Vec<f32> = kc.iter().map(|&v| v * inv * kw[0]).collect();
		for d in 0..hd {
			let (x0, x1, pidx);
			if d < halfn {
				x0 = knorm[d];
				x1 = knorm[d + halfn];
				pidx = d;
			} else {
				x0 = knorm[d - halfn];
				x1 = knorm[d];
				pidx = d - halfn;
			}
			let theta = rfs as f64 * pos as f64 * theta_scale.powi(pidx as i32);
			let c = (theta.cos() * raf as f64) as f32;
			let s = (theta.sin() * raf as f64) as f32;
			if d < halfn {
				exp_kn[hh * hd + d] = x0 * c - x1 * s;
				exp_kn[hh * hd + d + halfn] = x0 * s + x1 * c;
			} else {
				exp_kn[hh * hd + d - halfn] = x0 * c - x1 * s;
				exp_kn[hh * hd + d] = x0 * s + x1 * c;
			}
		}
	}

	let p = [
		nkv as u32,
		hd as u32,
		pos as u32,
		halfn as u32,
		eps.to_bits(),
		rfb.to_bits(),
		rfs.to_bits(),
		raf.to_bits(),
		0,
		sect_dims.to_bits(),
		0,
		0,
		0,
		0,
		0,
		0,
	];
	let (pb, _pm) = dev_param(&dev, &p);
	let (pl, layout, plyt) = fresh_aux(&dev, include_bytes!("../shaders/aux_prep_k.spv"), 3);
	let pool_sizes = [
		vk::DescriptorPoolSize {
			ty: vk::DescriptorType::STORAGE_BUFFER,
			descriptor_count: 16,
		},
		vk::DescriptorPoolSize {
			ty: vk::DescriptorType::UNIFORM_BUFFER,
			descriptor_count: 4,
		},
	];
	let pool = unsafe {
		dev.device
			.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo {
					max_sets: 4,
					pool_size_count: pool_sizes.len() as u32,
					p_pool_sizes: pool_sizes.as_ptr(),
					..Default::default()
				},
				None,
			)
			.unwrap()
	};
	let s = fresh_set(&dev, layout, pool, &[(0, k_b), (1, kw_b), (2, kn_b)], pb);
	dispatch(&dev, pl, plyt, s, &[k_b, kw_b], nkv as u32, 1);

	let got_kn = read_back_f32(&dev, kn_b, nkv * hd);
	for i in 0..nkv * hd {
		assert!(
			(got_kn[i] - exp_kn[i]).abs() < 1e-2,
			"prep_k kn[{}] {} vs {}",
			i,
			got_kn[i],
			exp_kn[i]
		);
	}
	unsafe {
		dev.device.destroy_descriptor_pool(pool, None);
	}
}

#[test]
fn gpu_attn_matches_cpu() {
	let dev = match GpuDevice::init() {
		Ok(d) => d,
		Err(_) => {
			return;
		}
	};
	let nh: usize = 2;
	let nkv: usize = 1;
	let hd: usize = 16;
	let pos: usize = 3; // tseen = 4 KV positions
	let max_ctx: usize = 64;
	let scale = 1.0f32 / (hd as f32).sqrt();

	let tseen = pos + 1;
	let mut q: Vec<f32> = vec![0.0; nh * hd];
	let mut kc: Vec<f32> = vec![0.0; tseen * nkv * hd];
	let mut vc: Vec<f32> = vec![0.0; tseen * nkv * hd];
	let mut gate: Vec<f32> = vec![0.0; nh * hd];
	for h in 0..nh {
		for d in 0..hd {
			q[h * hd + d] = (((h * 3 + d * 7) % 30) as f32) / 15.0 - 1.0;
			gate[h * hd + d] = (((h * 2 + d * 5) % 20) as f32) / 10.0 - 1.0;
		}
	}
	for t in 0..tseen {
		for hh in 0..nkv {
			for d in 0..hd {
				kc[t * nkv * hd + hh * hd + d] =
					(((t * 11 + hh * 5 + d * 3) % 40) as f32) / 20.0 - 1.0;
				vc[t * nkv * hd + hh * hd + d] =
					(((t * 7 + hh * 3 + d * 9) % 50) as f32) / 25.0 - 1.0;
			}
		}
	}
	let (q_b, _m) = dev_buf(&dev, &q);
	let (kc_b, _m2) = dev_buf(&dev, &kc);
	let (vc_b, _m3) = dev_buf(&dev, &vc);
	let (gate_b, _m4) = dev_buf(&dev, &gate);
	let (aout_b, _m5) = dev_buf(&dev, &vec![0.0f32; nh * hd]);
	let (scratch_b, _m6) = dev_buf(&dev, &vec![0.0f32; max_ctx * nh]);

	// CPU reference: per head, group g = h/(nh/nkv) (here nkv=1 -> g=0).
	let mut exp = vec![0.0f32; nh * hd];
	for h in 0..nh {
		let g = h / (nh / nkv);
		let mut scores = vec![0.0f32; tseen];
		for t in 0..tseen {
			let mut s = 0.0f32;
			for d in 0..hd {
				s += q[h * hd + d] * kc[t * nkv * hd + g * hd + d];
			}
			scores[t] = s * scale;
		}
		let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
		let mut sum = 0.0f32;
		for s in &mut scores {
			*s = (*s - mx).exp();
			sum += *s;
		}
		for s in &mut scores {
			*s /= sum;
		}
		for d in 0..hd {
			let mut acc = 0.0f32;
			for t in 0..tseen {
				acc += scores[t] * vc[t * nkv * hd + g * hd + d];
			}
			let sig = 1.0 / (1.0 + (-gate[h * hd + d]).exp());
			exp[h * hd + d] = acc * sig;
		}
	}

	let p = [
		nh as u32,
		hd as u32,
		nkv as u32,
		pos as u32,
		scale.to_bits(),
		max_ctx as u32,
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
	let (pb, _pm) = dev_param(&dev, &p);
	let (pl, layout, plyt) = fresh_aux(&dev, include_bytes!("../shaders/aux_attn.spv"), 6);
	let pool_sizes = [
		vk::DescriptorPoolSize {
			ty: vk::DescriptorType::STORAGE_BUFFER,
			descriptor_count: 32,
		},
		vk::DescriptorPoolSize {
			ty: vk::DescriptorType::UNIFORM_BUFFER,
			descriptor_count: 4,
		},
	];
	let pool = unsafe {
		dev.device
			.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo {
					max_sets: 4,
					pool_size_count: pool_sizes.len() as u32,
					p_pool_sizes: pool_sizes.as_ptr(),
					..Default::default()
				},
				None,
			)
			.unwrap()
	};
	let s = fresh_set(
		&dev,
		layout,
		pool,
		&[
			(0, q_b),
			(1, kc_b),
			(2, vc_b),
			(3, gate_b),
			(4, aout_b),
			(5, scratch_b),
		],
		pb,
	);
	dispatch(&dev, pl, plyt, s, &[q_b, kc_b, vc_b, gate_b], nh as u32, 1);

	let got = read_back_f32(&dev, aout_b, nh * hd);
	for i in 0..nh * hd {
		assert!(
			(got[i] - exp[i]).abs() < 1e-2,
			"attn[{}] {} vs {}",
			i,
			got[i],
			exp[i]
		);
	}
	unsafe {
		dev.device.destroy_descriptor_pool(pool, None);
	}
}

#[test]
fn gpu_production_matmul_reads_storage() {
	// Decisive check: does the *production* GPU matmul (which reads
	// storage buffers for weights + activation) actually return non-zero
	// results on lavapipe? If this fails too, the whole GPU path is
	// broken under llvmpipe, not just the aux shaders.
	let mut mgr = OffloadManager::new(1, 1);
	if !mgr.available {
		eprintln!("GPU unavailable, skipping");
		return;
	}
	let out_dim = 4usize;
	let in_dim = 8usize;
	let w: Vec<u16> = (0..out_dim * in_dim)
		.map(|i| quant::f32_to_fp16(((i % 5) as f32 - 2.0) * 0.4))
		.collect();
	let raw: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
	mgr.upload(
		0,
		WeightId::FfnUp,
		&raw,
		quant::GgmlType::F16,
		out_dim,
		in_dim,
	);
	let x: Vec<f32> = (0..in_dim).map(|i| (i % 3) as f32 - 1.0).collect();
	let y = mgr.matmul(
		0,
		WeightId::FfnUp,
		&quant::LinearWeight::F16(w.clone()),
		&x,
		out_dim,
		in_dim,
	);
	let mut expect = vec![0.0f32; out_dim];
	quant::gemm_f16(out_dim, in_dim, 1, &w, &x, &mut expect);
	for i in 0..out_dim {
		assert!(
			(y[i] - expect[i]).abs() < 1e-2,
			"prod matmul[{}] {} vs {}",
			i,
			y[i],
			expect[i]
		);
	}
	eprintln!(
		"DEBUG production matmul on lavapipe OK: {:?}",
		&y[..out_dim]
	);
}

#[test]
fn gpu_prefill_two_jobs_share_activation() {
	// Regression test for the per-call prefill path with **more than one
	// job in a batch**: the shape every real layer produces (`Wqkv` +
	// `WqkvGate`, `FfnUp` + `FfnGate`), and the one no test covered.
	//
	// `matmul_batch_seq` allocated `n` descriptor sets from a
	// `p_set_layouts` that pointed at a *single* `VkDescriptorSetLayout`
	// field. Vulkan reads that pointer as an array of
	// `descriptor_set_count` handles, so with `n > 1` the driver
	// consumed whatever followed `desc_layout` in `GpuDevice` and
	// segfaulted inside `nvoglv64.dll`. Every single-job call read
	// exactly one handle and was fine, which is why decode, the unit
	// tests and `gpubench` all passed while `QWEN_GPU_PREFILL=1` died on
	// the first prefill call of any offloaded layer.
	//
	// `seq > 1` and two jobs sharing one `x` are the parts that matter;
	// the shapes are deliberately tiny.
	let mut mgr = OffloadManager::new(1, 1);
	if !mgr.available {
		eprintln!("GPU unavailable, skipping");
		return;
	}
	let (out_dim, in_dim, seq) = (4usize, 8usize, 3usize);
	let mk = |bias: f32| -> Vec<u16> {
		(0..out_dim * in_dim)
			.map(|i| quant::f32_to_fp16(((i % 5) as f32 - 2.0) * 0.4 + bias))
			.collect()
	};
	let (wa, wb) = (mk(0.0), mk(0.25));
	for (id, w) in [(WeightId::FfnUp, &wa), (WeightId::FfnGate, &wb)] {
		let raw: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
		mgr.upload(0, id, &raw, quant::GgmlType::F16, out_dim, in_dim);
	}
	let x: Vec<f32> = (0..seq * in_dim).map(|i| (i % 3) as f32 - 1.0).collect();
	let lwa = quant::LinearWeight::F16(wa.clone());
	let lwb = quant::LinearWeight::F16(wb.clone());
	let jobs = [
		BatchJobSeq {
			il: 0,
			id: WeightId::FfnUp,
			w: &lwa,
			x: &x,
			seq,
			out_dim,
			in_dim,
		},
		BatchJobSeq {
			il: 0,
			id: WeightId::FfnGate,
			w: &lwb,
			x: &x,
			seq,
			out_dim,
			in_dim,
		},
	];
	let got = mgr.matmul_batch_seq(&jobs);
	assert_eq!(got.len(), 2);
	for (i, w) in [(0usize, &wa), (1usize, &wb)] {
		let expect = quant::apply_linear_batch(
			&quant::LinearWeight::F16((*w).clone()),
			&x,
			seq,
			out_dim,
			in_dim,
		);
		assert_eq!(got[i].len(), seq * out_dim);
		for k in 0..seq * out_dim {
			assert!(
				(got[i][k] - expect[k]).abs() < 1e-2,
				"job{i}[{k}] {} vs {}",
				got[i][k],
				expect[k]
			);
		}
	}
}

#[test]
fn gpu_streams_non_resident_weights() {
	// The streaming prefill path: weights that were **never uploaded**
	// still run on the GPU, because `matmul_batch_seq` stages them into
	// a pool slot for the call. This is what lifts prefill past the 22
	// of 65 layers that fit in 6 GiB.
	//
	// Two jobs sharing one activation, because that is the shape every
	// real layer produces and because it is the case where the pool has
	// to hand out *distinct* slots: reusing one inside a batch would
	// have the second weight overwrite the first before the dispatch
	// reads it. `seq = 64` is `GEMM_BN`, so this also covers the tiled
	// GEMM rather than only the GEMV.
	let mgr = OffloadManager::new(0, 1);
	if !mgr.available {
		eprintln!("GPU unavailable, skipping");
		return;
	}
	let gpu = match mgr.gpu() {
		Some(g) => g,
		None => return,
	};
	let (out_dim, in_dim, seq) = (256usize, 256usize, 64usize);

	// Synthetic Q4_K blocks: any byte pattern is valid and the kernel is
	// branch-free w.r.t. the data, so only `d`/`dmin` need sane values.
	let synth = |seed: u64| -> Vec<u8> {
		let nb = out_dim * in_dim / QK_K;
		let mut raw = vec![0u8; nb * Q4K_BLOCK];
		let mut s = seed;
		for b in 0..nb {
			let o = b * Q4K_BLOCK;
			for j in 4..Q4K_BLOCK {
				s = s
					.wrapping_mul(6364136223846793005)
					.wrapping_add(1442695040888963407);
				raw[o + j] = (s >> 33) as u8;
			}
			raw[o..o + 2].copy_from_slice(&quant::f32_to_fp16(0.02).to_le_bytes());
			raw[o + 2..o + 4].copy_from_slice(&quant::f32_to_fp16(0.01).to_le_bytes());
		}
		raw
	};
	let (ra, rb) = (synth(1), synth(2));
	let x: Vec<f32> = (0..seq * in_dim)
		.map(|j| ((j % 97) as f32 - 48.0) * 0.02)
		.collect();

	let jobs = [
		SeqJob {
			il: 0,
			id: WeightId::FfnUp,
			x: &x,
			seq,
			src: Some((ra.as_slice(), GpuKind::Q4K, out_dim, in_dim)),
			rows: None,
			q8: None,
			dst: None,
		},
		SeqJob {
			il: 0,
			id: WeightId::FfnGate,
			x: &x,
			seq,
			src: Some((rb.as_slice(), GpuKind::Q4K, out_dim, in_dim)),
			rows: None,
			q8: None,
			dst: None,
		},
	];
	let got = gpu.matmul_batch_seq(&jobs).expect("streamed matmul");
	assert_eq!(got.len(), 2);
	for (i, raw) in [(0usize, &ra), (1usize, &rb)] {
		let w = quant::LinearWeight::Q4K {
			raw: (*raw).clone(),
			out_dim,
			in_dim,
		};
		let cpu = quant::apply_linear_batch(&w, &x, seq, out_dim, in_dim);
		assert_eq!(got[i].len(), cpu.len());
		let scale = cpu.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
		for k in 0..cpu.len() {
			assert!(
				(got[i][k] - cpu[k]).abs() < 1e-4 * scale,
				"job{i}[{k}] {} vs {} (scale {scale})",
				got[i][k],
				cpu[k]
			);
		}
	}
}

#[test]
fn gpu_streams_back_to_back_batches() {
	// The upload is issued to the transfer queue and **not waited for**
	// until the dispatch that reads it is recorded, so the pool can hold
	// an in-flight copy across the host work of a call. Two hazards come
	// with that, and neither shows up in a single-batch test:
	//
	//  1. the next batch must not memcpy into a staging map a live DMA
	//     is still reading from (`stream_bind` drains first), and
	//  2. a slot recycled by the round-robin must not still be claimed
	//     as holding the weight it held before.
	//
	// So: three consecutive batches over five distinct weights through a
	// pool of two slots, every result checked against the CPU. The
	// shapes are deliberately over a megabyte, which is where
	// `stage_copy` switches to the threaded copy: both for the
	// contiguous Q4_K layout and for Q6_K's 212-byte VRAM stride.
	let mgr = OffloadManager::new(0, 1);
	if !mgr.available {
		eprintln!("GPU unavailable, skipping");
		return;
	}
	let gpu = match mgr.gpu() {
		Some(g) => g,
		None => return,
	};
	let (out_dim, in_dim, seq) = (2048usize, 1024usize, 64usize);
	let mut rng = 0x9E37_79B9_7F4A_7C15u64;
	let mut next = move || {
		rng = rng
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		(rng >> 33) as u8
	};
	let mut synth_q4k = || -> Vec<u8> {
		let nb = out_dim * in_dim / QK_K;
		let mut raw = vec![0u8; nb * Q4K_BLOCK];
		for b in 0..nb {
			let o = b * Q4K_BLOCK;
			for j in 4..Q4K_BLOCK {
				raw[o + j] = next();
			}
			raw[o..o + 2].copy_from_slice(&quant::f32_to_fp16(0.02).to_le_bytes());
			raw[o + 2..o + 4].copy_from_slice(&quant::f32_to_fp16(0.01).to_le_bytes());
		}
		raw
	};
	let (up0, gate0, up1, gate1) = (synth_q4k(), synth_q4k(), synth_q4k(), synth_q4k());
	let down: Vec<u8> = {
		let nb = out_dim * in_dim / QK_K;
		let mut raw = vec![0u8; nb * 210];
		for b in 0..nb {
			let o = b * 210;
			for j in 0..208 {
				raw[o + j] = next();
			}
			raw[o + 208..o + 210].copy_from_slice(&quant::f32_to_fp16(0.02).to_le_bytes());
		}
		raw
	};
	let x: Vec<f32> = (0..seq * in_dim)
		.map(|j| ((j % 97) as f32 - 48.0) * 0.02)
		.collect();

	let batches: [Vec<(usize, WeightId, &Vec<u8>, GpuKind)>; 3] = [
		vec![
			(0, WeightId::FfnUp, &up0, GpuKind::Q4K),
			(0, WeightId::FfnGate, &gate0, GpuKind::Q4K),
		],
		vec![(0, WeightId::FfnDown, &down, GpuKind::Q6K)],
		vec![
			(1, WeightId::FfnUp, &up1, GpuKind::Q4K),
			(1, WeightId::FfnGate, &gate1, GpuKind::Q4K),
		],
	];
	for (bi, batch) in batches.iter().enumerate() {
		let jobs: Vec<SeqJob> = batch
			.iter()
			.map(|(il, id, raw, kind)| SeqJob {
				il: *il,
				id: *id,
				x: &x,
				seq,
				src: Some((raw.as_slice(), *kind, out_dim, in_dim)),
				rows: None,
				q8: None,
				dst: None,
			})
			.collect();
		let got = gpu.matmul_batch_seq(&jobs).expect("streamed matmul");
		assert_eq!(got.len(), batch.len());
		for (i, (_il, _id, raw, kind)) in batch.iter().enumerate() {
			let w = match kind {
				GpuKind::Q6K => quant::LinearWeight::Q6K {
					raw: (*raw).clone(),
					out_dim,
					in_dim,
				},
				_ => quant::LinearWeight::Q4K {
					raw: (*raw).clone(),
					out_dim,
					in_dim,
				},
			};
			let cpu = quant::apply_linear_batch(&w, &x, seq, out_dim, in_dim);
			assert_eq!(got[i].len(), cpu.len());
			let scale = cpu.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
			for k in 0..cpu.len() {
				assert!(
					(got[i][k] - cpu[k]).abs() < 1e-4 * scale,
					"batch{bi} job{i}[{k}] {} vs {} (scale {scale})",
					got[i][k],
					cpu[k]
				);
			}
		}
	}
}

#[test]
fn gpu_buffer_pool_reuses_slots_across_shapes() {
	// `matmul_batch_seq` no longer creates its `x`, `y` and readback
	// buffers per call: they come from positional slots in `HostPool`
	// that grow to the largest shape ever asked for and are then reused.
	// The hazards that introduces are all about *shape change*, and
	// `gpu_streams_back_to_back_batches` cannot see them because every
	// batch there is the same size:
	//
	//  1. a slot must grow when a wider or longer call arrives, and the
	//     descriptor must then point at the **new** buffer, not the
	//     freed one;
	//  2. a slot reused at a larger capacity must read back only this
	//     call's `seq * out_dim * 4` bytes, not the slot's capacity, and
	//     must not return the previous call's results in the tail;
	//  3. a batch with more jobs than any before it must extend the pool
	//     rather than hand two jobs the same slot.
	//
	// So: small -> large-and-wider -> medium, with the job count going
	// 1 -> 2 -> 1, every result checked against the CPU.
	let mgr = OffloadManager::new(0, 1);
	if !mgr.available {
		eprintln!("GPU unavailable, skipping");
		return;
	}
	let gpu = match mgr.gpu() {
		Some(g) => g,
		None => return,
	};
	let in_dim = 256usize;
	let mut rng = 0x2545_F491_4F6C_DD1Du64;
	let mut next = move || {
		rng = rng
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		(rng >> 33) as u8
	};
	let mut synth = |out_dim: usize| -> Vec<u8> {
		let nb = out_dim * in_dim / QK_K;
		let mut raw = vec![0u8; nb * Q4K_BLOCK];
		for b in 0..nb {
			let o = b * Q4K_BLOCK;
			for j in 4..Q4K_BLOCK {
				raw[o + j] = next();
			}
			raw[o..o + 2].copy_from_slice(&quant::f32_to_fp16(0.02).to_le_bytes());
			raw[o + 2..o + 4].copy_from_slice(&quant::f32_to_fp16(0.01).to_le_bytes());
		}
		raw
	};
	// (seq, [out_dim per job]): batch 1 grows every slot and adds one.
	let plan: [(usize, &[usize]); 3] = [(32, &[256]), (96, &[1024, 768]), (48, &[512])];
	let weights: Vec<Vec<Vec<u8>>> = plan
		.iter()
		.map(|(_, ods)| ods.iter().map(|od| synth(*od)).collect())
		.collect();

	for (bi, (seq, ods)) in plan.iter().enumerate() {
		let seq = *seq;
		let x: Vec<f32> = (0..seq * in_dim)
			.map(|j| ((j % 97) as f32 - 48.0) * 0.02 + bi as f32)
			.collect();
		let jobs: Vec<SeqJob> = ods
			.iter()
			.enumerate()
			.map(|(i, od)| SeqJob {
				il: bi,
				id: if i == 0 {
					WeightId::FfnUp
				} else {
					WeightId::FfnGate
				},
				x: &x,
				seq,
				src: Some((weights[bi][i].as_slice(), GpuKind::Q4K, *od, in_dim)),
				rows: None,
				q8: None,
				dst: None,
			})
			.collect();
		let got = gpu.matmul_batch_seq(&jobs).expect("pooled matmul");
		assert_eq!(got.len(), ods.len());
		for (i, od) in ods.iter().enumerate() {
			let w = quant::LinearWeight::Q4K {
				raw: weights[bi][i].clone(),
				out_dim: *od,
				in_dim,
			};
			// Below `gemm_min_seq` the call runs the f32-activation
			// GEMV, whose reference is the exact product; the tiled
			// GEMM (every seq here, including the padded 32/48-token
			// tiles) reads q8_K, whose reference is the CPU q8_K kernel.
			let cpu = if GpuDevice::gemm_on(seq) {
				quant::apply_linear_batch(&w, &x, seq, *od, in_dim)
			} else {
				let deq = quant::dequantize_q4_k(&weights[bi][i], *od * in_dim);
				let mut r = vec![0.0f32; seq * od];
				for p in 0..seq {
					for o in 0..*od {
						r[p * od + o] = deq[o * in_dim..(o + 1) * in_dim]
							.iter()
							.zip(&x[p * in_dim..(p + 1) * in_dim])
							.map(|(a, b)| *a as f64 * *b as f64)
							.sum::<f64>() as f32;
					}
				}
				r
			};
			// The length is the point of hazard 2: a grown slot holds
			// more than this call produced.
			assert_eq!(got[i].len(), seq * od);
			assert_eq!(got[i].len(), cpu.len());
			let scale = cpu.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
			for k in 0..cpu.len() {
				assert!(
					(got[i][k] - cpu[k]).abs() < 1e-4 * scale,
					"batch{bi} job{i}[{k}] {} vs {} (scale {scale})",
					got[i][k],
					cpu[k]
				);
			}
		}
	}
}

#[test]
fn gpu_short_chunk_gemm_row_cap() {
	// The short-chunk prefill split: a **resident** weight run for only
	// its first `rows` output rows (`SeqJob::rows`), through the tiled
	// GEMM at a chunk far below one 64-token tile, where the warps whose
	// token group is all padding skip their arithmetic. 200 rows cuts the
	// second 128-row tile; 20 tokens leaves 5 of 8 warps idle and one
	// (tokens 16..23) half live. Both GEMMs, since both shaders changed.
	let mut mgr = OffloadManager::new(1, 1);
	if !mgr.available {
		eprintln!("GPU unavailable, skipping");
		return;
	}
	let (out_dim, in_dim, seq, rows) = (384usize, 512usize, 20usize, 200usize);
	let mut rng = 0x2545_F491_4F6C_DD1Du64;
	let mut next = move || {
		rng = rng
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		(rng >> 33) as u8
	};
	let nb = out_dim * in_dim / QK_K;
	let mut q4 = vec![0u8; nb * Q4K_BLOCK];
	for b in 0..nb {
		let o = b * Q4K_BLOCK;
		for j in 4..Q4K_BLOCK {
			q4[o + j] = next();
		}
		q4[o..o + 2].copy_from_slice(&quant::f32_to_fp16(0.02).to_le_bytes());
		q4[o + 2..o + 4].copy_from_slice(&quant::f32_to_fp16(0.01).to_le_bytes());
	}
	let mut q6 = vec![0u8; nb * 210];
	for b in 0..nb {
		let o = b * 210;
		for j in 0..208 {
			q6[o + j] = next();
		}
		q6[o + 208..o + 210].copy_from_slice(&quant::f32_to_fp16(0.02).to_le_bytes());
	}
	mgr.upload(
		0,
		WeightId::FfnUp,
		&q4,
		quant::GgmlType::Q4_K,
		out_dim,
		in_dim,
	);
	mgr.upload(
		0,
		WeightId::FfnDown,
		&q6,
		quant::GgmlType::Q6_K,
		out_dim,
		in_dim,
	);
	let gpu = mgr.gpu().unwrap();
	assert!(GpuDevice::gemm_on(seq) && seq < GpuDevice::GEMM_BN);
	let x: Vec<f32> = (0..seq * in_dim)
		.map(|j| ((j % 89) as f32 - 44.0) * 0.03)
		.collect();
	for (id, raw, is6) in [
		(WeightId::FfnUp, &q4, false),
		(WeightId::FfnDown, &q6, true),
	] {
		let job = SeqJob {
			il: 0,
			id,
			x: &x,
			seq,
			src: None,
			rows: Some(rows),
			q8: None,
			dst: None,
		};
		let got = gpu
			.matmul_batch_seq(&[job])
			.expect("resident short-chunk matmul");
		let head = raw[..raw.len() / out_dim * rows].to_vec();
		let w = if is6 {
			quant::LinearWeight::Q6K {
				raw: head,
				out_dim: rows,
				in_dim,
			}
		} else {
			quant::LinearWeight::Q4K {
				raw: head,
				out_dim: rows,
				in_dim,
			}
		};
		let cpu = quant::apply_linear_batch(&w, &x, seq, rows, in_dim);
		assert_eq!(got[0].len(), seq * rows);
		let scale = cpu.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
		for k in 0..cpu.len() {
			assert!(
				(got[0][k] - cpu[k]).abs() < 1e-4 * scale,
				"{id:?}[{k}] {} vs {} (scale {scale})",
				got[0][k],
				cpu[k]
			);
		}
	}
}

#[test]
fn gpu_min_layout_rmsnorm() {
	// Isolate whether the 9-binding `aux_layout` is what breaks storage
	// reads on lavapipe (vs the 4-binding production `desc_layout` which
	// works). Build a *minimal* layout with only the bindings the shader
	// actually uses (0,1 storage + 8 uniform) and run the same shader.
	let dev = match GpuDevice::init() {
		Ok(d) => d,
		Err(_) => {
			return;
		}
	};
	let n = 16usize;
	let x: Vec<f32> = (0..n).map(|i| (i as f32 - 8.0) * 0.3).collect();
	let eps = 1e-5f32;
	let ss: f32 = x.iter().map(|v| v * v).sum();
	let inv = 1.0 / (ss / n as f32 + eps).sqrt();

	let (x_b, _x_m) = dev_buf(&dev, &x);
	let (inv_b, _inv_m) = dev_buf(&dev, &[0.0f32]);

	let bindings = [
		vk::DescriptorSetLayoutBinding {
			binding: 0,
			descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
			descriptor_count: 1,
			stage_flags: vk::ShaderStageFlags::COMPUTE,
			p_immutable_samplers: ptr::null(),
			..Default::default()
		},
		vk::DescriptorSetLayoutBinding {
			binding: 1,
			descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
			descriptor_count: 1,
			stage_flags: vk::ShaderStageFlags::COMPUTE,
			p_immutable_samplers: ptr::null(),
			..Default::default()
		},
		vk::DescriptorSetLayoutBinding {
			binding: 8,
			descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
			descriptor_count: 1,
			stage_flags: vk::ShaderStageFlags::COMPUTE,
			p_immutable_samplers: ptr::null(),
			..Default::default()
		},
	];
	let min_layout = unsafe {
		dev.device
			.create_descriptor_set_layout(
				&vk::DescriptorSetLayoutCreateInfo {
					binding_count: bindings.len() as u32,
					p_bindings: bindings.as_ptr(),
					..Default::default()
				},
				None,
			)
			.unwrap()
	};
	let min_pl = unsafe {
		dev.device
			.create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo {
					set_layout_count: 1,
					p_set_layouts: &min_layout,
					..Default::default()
				},
				None,
			)
			.unwrap()
	};
	let code = load_spv(include_bytes!("../shaders/aux_rmsnorm_sum.spv"));
	let sm = unsafe {
		dev.device
			.create_shader_module(
				&vk::ShaderModuleCreateInfo {
					code_size: code.len() * 4,
					p_code: code.as_ptr(),
					..Default::default()
				},
				None,
			)
			.unwrap()
	};
	let entry = std::ffi::CString::new("main").unwrap();
	let pl = unsafe {
		dev.device
			.create_compute_pipelines(
				vk::PipelineCache::default(),
				&[vk::ComputePipelineCreateInfo {
					stage: vk::PipelineShaderStageCreateInfo {
						stage: vk::ShaderStageFlags::COMPUTE,
						module: sm,
						p_name: entry.as_ptr(),
						..Default::default()
					},
					layout: min_pl,
					..Default::default()
				}],
				None,
			)
			.map_err(|(p, e)| {
				let _ = p;
				format!("{e:?}")
			})
			.unwrap()[0]
	};

	let pool_sizes = [
		vk::DescriptorPoolSize {
			ty: vk::DescriptorType::STORAGE_BUFFER,
			descriptor_count: 8,
		},
		vk::DescriptorPoolSize {
			ty: vk::DescriptorType::UNIFORM_BUFFER,
			descriptor_count: 4,
		},
	];
	let pool = unsafe {
		dev.device
			.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo {
					max_sets: 4,
					pool_size_count: pool_sizes.len() as u32,
					p_pool_sizes: pool_sizes.as_ptr(),
					..Default::default()
				},
				None,
			)
			.unwrap()
	};
	let set = unsafe {
		dev.device
			.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
				descriptor_pool: pool,
				descriptor_set_count: 1,
				p_set_layouts: &min_layout,
				..Default::default()
			})
			.unwrap()[0]
	};
	let p = [
		n as u32,
		eps.to_bits(),
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
	let (pb, _pm) = dev_param(&dev, &p);
	let writes = [
		vk::WriteDescriptorSet {
			dst_set: set,
			dst_binding: 0,
			descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
			descriptor_count: 1,
			p_buffer_info: &vk::DescriptorBufferInfo {
				buffer: x_b,
				offset: 0,
				range: vk::WHOLE_SIZE,
				..Default::default()
			},
			..Default::default()
		},
		vk::WriteDescriptorSet {
			dst_set: set,
			dst_binding: 1,
			descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
			descriptor_count: 1,
			p_buffer_info: &vk::DescriptorBufferInfo {
				buffer: inv_b,
				offset: 0,
				range: vk::WHOLE_SIZE,
				..Default::default()
			},
			..Default::default()
		},
		vk::WriteDescriptorSet {
			dst_set: set,
			dst_binding: 8,
			descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
			descriptor_count: 1,
			p_buffer_info: &vk::DescriptorBufferInfo {
				buffer: pb,
				offset: 0,
				range: 64,
				..Default::default()
			},
			..Default::default()
		},
	];
	unsafe { dev.device.update_descriptor_sets(&writes, &[]) };
	dispatch(&dev, pl, min_pl, set, &[x_b], 1, 1);
	let r = read_back_f32(&dev, inv_b, 1)[0];
	eprintln!("DEBUG min-layout rmsnorm_sum inv={} (expected {})", r, inv);
	assert!(
		(r - inv).abs() < 1e-3,
		"min-layout rmsnorm inv {} vs {}",
		r,
		inv
	);
}

#[test]
fn gpu_probe_storage_binding_limit() {
	let dev = match GpuDevice::init() {
		Ok(d) => d,
		Err(_) => {
			return;
		}
	};
	let n = 16usize;
	let x: Vec<f32> = (0..n).map(|i| (i as f32 - 8.0) * 0.3).collect();
	let eps = 1e-5f32;
	let ss: f32 = x.iter().map(|v| v * v).sum();
	let inv = 1.0 / (ss / n as f32 + eps).sqrt();
	for nstor in 2u32..=6u32 {
		let (x_b, _x_m) = dev_buf(&dev, &x);
		let (inv_b, _inv_m) = dev_buf(&dev, &[0.0f32]);
		let mut bindings = Vec::new();
		for b in 0..nstor {
			bindings.push(vk::DescriptorSetLayoutBinding {
				binding: b,
				descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 1,
				stage_flags: vk::ShaderStageFlags::COMPUTE,
				p_immutable_samplers: ptr::null(),
				..Default::default()
			});
		}
		bindings.push(vk::DescriptorSetLayoutBinding {
			binding: 8,
			descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
			descriptor_count: 1,
			stage_flags: vk::ShaderStageFlags::COMPUTE,
			p_immutable_samplers: ptr::null(),
			..Default::default()
		});
		let min_layout = unsafe {
			dev.device
				.create_descriptor_set_layout(
					&vk::DescriptorSetLayoutCreateInfo {
						binding_count: bindings.len() as u32,
						p_bindings: bindings.as_ptr(),
						..Default::default()
					},
					None,
				)
				.unwrap()
		};
		let min_pl = unsafe {
			dev.device
				.create_pipeline_layout(
					&vk::PipelineLayoutCreateInfo {
						set_layout_count: 1,
						p_set_layouts: &min_layout,
						..Default::default()
					},
					None,
				)
				.unwrap()
		};
		let code = load_spv(include_bytes!("../shaders/aux_rmsnorm_sum.spv"));
		let sm = unsafe {
			dev.device
				.create_shader_module(
					&vk::ShaderModuleCreateInfo {
						code_size: code.len() * 4,
						p_code: code.as_ptr(),
						..Default::default()
					},
					None,
				)
				.unwrap()
		};
		let entry = std::ffi::CString::new("main").unwrap();
		let pl = unsafe {
			dev.device
				.create_compute_pipelines(
					vk::PipelineCache::default(),
					&[vk::ComputePipelineCreateInfo {
						stage: vk::PipelineShaderStageCreateInfo {
							stage: vk::ShaderStageFlags::COMPUTE,
							module: sm,
							p_name: entry.as_ptr(),
							..Default::default()
						},
						layout: min_pl,
						..Default::default()
					}],
					None,
				)
				.map_err(|(p, e)| {
					let _ = p;
					format!("{e:?}")
				})
				.unwrap()[0]
		};
		let pool_sizes = [
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: nstor * 4,
			},
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::UNIFORM_BUFFER,
				descriptor_count: 4,
			},
		];
		let pool = unsafe {
			dev.device
				.create_descriptor_pool(
					&vk::DescriptorPoolCreateInfo {
						max_sets: 4,
						pool_size_count: pool_sizes.len() as u32,
						p_pool_sizes: pool_sizes.as_ptr(),
						..Default::default()
					},
					None,
				)
				.unwrap()
		};
		let set = unsafe {
			dev.device
				.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
					descriptor_pool: pool,
					descriptor_set_count: 1,
					p_set_layouts: &min_layout,
					..Default::default()
				})
				.unwrap()[0]
		};
		let p = [
			n as u32,
			eps.to_bits(),
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
		let (pb, _pm) = dev_param(&dev, &p);
		let writes = [
			vk::WriteDescriptorSet {
				dst_set: set,
				dst_binding: 0,
				descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 1,
				p_buffer_info: &vk::DescriptorBufferInfo {
					buffer: x_b,
					offset: 0,
					range: vk::WHOLE_SIZE,
					..Default::default()
				},
				..Default::default()
			},
			vk::WriteDescriptorSet {
				dst_set: set,
				dst_binding: 1,
				descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 1,
				p_buffer_info: &vk::DescriptorBufferInfo {
					buffer: inv_b,
					offset: 0,
					range: vk::WHOLE_SIZE,
					..Default::default()
				},
				..Default::default()
			},
			vk::WriteDescriptorSet {
				dst_set: set,
				dst_binding: 8,
				descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
				descriptor_count: 1,
				p_buffer_info: &vk::DescriptorBufferInfo {
					buffer: pb,
					offset: 0,
					range: 64,
					..Default::default()
				},
				..Default::default()
			},
		];
		unsafe { dev.device.update_descriptor_sets(&writes, &[]) };
		dispatch(&dev, pl, min_pl, set, &[x_b], 1, 1);
		let r = read_back_f32(&dev, inv_b, 1)[0];
		let ok = (r - inv).abs() < 1e-3;
		eprintln!("DEBUG nstor={} inv={} ok={}", nstor, r, ok);
		unsafe {
			dev.device.destroy_descriptor_pool(pool, None);
			dev.device.destroy_pipeline_layout(min_pl, None);
			dev.device.destroy_descriptor_set_layout(min_layout, None);
			dev.device.destroy_pipeline(pl, None);
			dev.device.destroy_shader_module(sm, None);
		}
	}
}

/// End-to-end check of the L2/L3 GPU-resident fused SSM block: feed a tiny
/// SSM layer through `OffloadManager::ssm_block` (the whole conv1d →
/// delta-net → gate → rmsnorm → SsmOut → SwiGLU FFN path, decode, single
/// token) and compare against a CPU reference that mirrors the model's
/// `forward_step` SSM branch exactly. Weights are uploaded as F16 so the
/// GPU matmul reads f32 activation (the test's `reference_matmul` widens
/// f16→f32 identically). Skips if no device.
///
/// Passes on the GTX 1060; two fused-block bugs once broke it (the
/// hang was a `VK_WHOLE_SIZE` `VkBufferCopy`; the all-zero output was a
/// dangling `DescriptorBufferInfo` in `ssm_aux_set`).
#[test]
fn gpu_ssm_block_matches_cpu() {
	let mut mgr = OffloadManager::new(1, 1);
	if !mgr.available {
		eprintln!("GPU unavailable, skipping ssm_block test");
		return;
	}
	// The production fused SSM block uses the device-init-time `aux_layout`,
	// which the software `llvmpipe` ICD mis-binds (causing a crash). It is
	// validated on real hardware (GTX 1060) instead; skip on CI.
	if mgr.is_llvmpipe() {
		eprintln!("llvmpipe: skipping fused ssm_block end-to-end (aux_layout mis-bind); validated on host");
		return;
	}
	// Tiny SSM dims.
	let n: usize = 16;
	let S: usize = 4;
	let Hk: usize = 2;
	let Hv: usize = 3;
	let ksz: usize = 4;
	let key_dim = S * Hk; // 8
	let value_dim = S * Hv; // 12
	let conv_dim = key_dim * 2 + value_dim; // 28
	let nf: usize = 16;
	let eps = 1e-5f32;

	let mut rng = 12345u64;
	let mut rnd = || {
		rng = rng
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		((rng >> 33) as f32 / (1u64 << 31) as f32) - 1.0
	};

	let f16_bytes = |v: &[f32]| -> Vec<u8> {
		let mut b = Vec::with_capacity(v.len() * 2);
		for &x in v {
			b.extend_from_slice(&quant::f32_to_fp16(x).to_le_bytes());
		}
		b
	};
	let mut mat = |od: usize, id: usize| -> Vec<f32> { (0..od * id).map(|_| rnd()).collect() };
	let upload_f16 = |mgr: &mut OffloadManager, id: WeightId, w: &[f32], od: usize, idim: usize| {
		mgr.upload(0, id, &f16_bytes(w), quant::GgmlType::F16, od, idim);
	};

	let wqkv = mat(conv_dim, n);
	let wqkvg = mat(value_dim, n);
	let sout_w = mat(n, value_dim);
	let wbeta = mat(Hv, n);
	let walpha = mat(Hv, n);
	let fup_w = mat(nf, n);
	let fgate_w = mat(nf, n);
	let fdown_w = mat(n, nf);

	upload_f16(&mut mgr, WeightId::Wqkv, &wqkv, conv_dim, n);
	upload_f16(&mut mgr, WeightId::WqkvGate, &wqkvg, value_dim, n);
	upload_f16(&mut mgr, WeightId::SsmOut, &sout_w, n, value_dim);
	upload_f16(&mut mgr, WeightId::WssmBeta, &wbeta, Hv, n);
	upload_f16(&mut mgr, WeightId::WssmAlpha, &walpha, Hv, n);
	upload_f16(&mut mgr, WeightId::FfnUp, &fup_w, nf, n);
	upload_f16(&mut mgr, WeightId::FfnGate, &fgate_w, nf, n);
	upload_f16(&mut mgr, WeightId::FfnDown, &fdown_w, n, nf);

	let conv1d: Vec<f32> = (0..ksz * conv_dim).map(|_| rnd()).collect(); // row-major [c][f] = w[f + c*ksz]
	let ssm_dt: Vec<f32> = (0..Hv).map(|_| rnd()).collect();
	let ssm_a: Vec<f32> = (0..Hv).map(|_| rnd()).collect();
	let ssm_norm: Vec<f32> = (0..S).map(|_| rnd()).collect();
	let attn_norm: Vec<f32> = (0..n).map(|_| rnd()).collect();
	let post_norm: Vec<f32> = (0..n).map(|_| rnd()).collect();
	let x: Vec<f32> = (0..n).map(|_| rnd()).collect();

	let info = GpuBlockInfo {
		n_embd: n,
		// An SSM layer never runs the attention path, but `ensure_block`
		// still sizes the shared full-attention buffers from these dims,
		// so they must be the layer's real (non-zero) values, exactly as
		// `Model` passes them. Zeroing them asks for 0-byte buffers and
		// 0-sized dispatches, which hangs the GTX 1060.
		n_head: 2,
		n_head_kv: 1,
		head_dim: 4,
		n_ff: nf,
		eps,
		pos: 0,
		half_nrot: 2,
		rope_freq_base: 0.0,
		rope_freq_scale: 0.0,
		rope_attn_factor: 0.0,
		sect_dims: 0.0,
		attn_norm: &attn_norm,
		post_norm: &post_norm,
		q_norm: &[],
		k_norm: &[],
		ssm_state: S,
		ssm_n_group: Hk,
		ssm_dt_rank: Hv,
		conv_kernel: ksz,
		ssm_conv1d: &conv1d,
		ssm_dt: &ssm_dt,
		ssm_a: &ssm_a,
		ssm_norm: &ssm_norm,
	};

	let gpu = mgr
		.ssm_block(0, &x, &info)
		.expect("ssm_block should return GPU output");

	// ---- CPU reference (mirrors model/src/forward.rs SSM decode path) ----
	let l2_norm = |v: &[f32]| -> Vec<f32> {
		let ss: f32 = v.iter().map(|a| a * a).sum();
		let inv = 1.0 / ss.sqrt().max(eps);
		v.iter().map(|a| a * inv).collect()
	};
	let rms_norm = |v: &[f32], w: &[f32]| -> Vec<f32> {
		let ss: f32 = v.iter().map(|a| a * a).sum();
		let inv = 1.0 / (ss / v.len() as f32 + eps).sqrt();
		v.iter().zip(w).map(|(a, b)| a * inv * b).collect()
	};
	// f16-weight matmul matching the GPU F16 shader (f16->f32 * f32, f32 acc)
	let matmul = |w: &[f32], xv: &[f32], od: usize, id: usize| -> Vec<f32> {
		let mut y = vec![0.0f32; od];
		for o in 0..od {
			let mut acc = 0.0f32;
			for k in 0..id {
				acc += quant::fp16_to_f32(quant::f32_to_fp16(w[o * id + k])) * xv[k];
			}
			y[o] = acc;
		}
		y
	};
	let silu = |a: f32| a / (1.0 + (-a).exp());
	let sigmoid = |a: f32| 1.0 / (1.0 + (-a).exp());
	let softplus = |a: f32| if a > 20.0 { a } else { (1.0 + a.exp()).ln() };

	let norm = rms_norm(&x, &attn_norm);
	let qkv = matmul(&wqkv, &norm, conv_dim, n);
	// conv: rolling window [zeros(ksz-1 rows); qkv]; initial conv_state = 0
	let mut input = vec![0.0f32; ksz * conv_dim];
	input[(ksz - 1) * conv_dim..].copy_from_slice(&qkv);
	let mut conv_out = vec![0.0f32; conv_dim];
	for c in 0..conv_dim {
		let mut s = 0.0f32;
		for f in 0..ksz {
			s += conv1d[f + c * ksz] * input[f * conv_dim + c];
		}
		conv_out[c] = silu(s);
	}
	let mut q = vec![0.0f32; key_dim];
	let mut k = vec![0.0f32; key_dim];
	let mut v = vec![0.0f32; value_dim];
	q.copy_from_slice(&conv_out[0..key_dim]);
	k.copy_from_slice(&conv_out[key_dim..2 * key_dim]);
	v.copy_from_slice(&conv_out[2 * key_dim..2 * key_dim + value_dim]);
	let mut qn = vec![0.0f32; key_dim];
	let mut kn = vec![0.0f32; key_dim];
	for h in 0..Hk {
		qn[h * S..(h + 1) * S].copy_from_slice(&l2_norm(&q[h * S..(h + 1) * S]));
		kn[h * S..(h + 1) * S].copy_from_slice(&l2_norm(&k[h * S..(h + 1) * S]));
	}
	let beta_raw = matmul(&wbeta, &norm, Hv, n);
	let beta_s: Vec<f32> = beta_raw.iter().map(|b| sigmoid(*b)).collect();
	let alpha = matmul(&walpha, &norm, Hv, n);
	// gate_h[h] = exp(softplus(alpha[h] + ssm_dt[h]) * ssm_a[h])
	let gate_h: Vec<f32> = (0..Hv)
		.map(|h| (softplus(alpha[h] + ssm_dt[h]) * ssm_a[h]).exp())
		.collect();

	let mut st = vec![0.0f32; Hv * S * S]; // zero initial SSM state
	let mut o = vec![0.0f32; value_dim];
	let qscale = 1.0 / (S as f32).sqrt();
	for h in 0..Hv {
		let kh = h % Hk;
		let g = gate_h[h];
		let beta = beta_s[h];
		let kvec = &kn[kh * S..(kh + 1) * S];
		let qvec = &qn[kh * S..(kh + 1) * S];
		let vvec = &v[h * S..(h + 1) * S];
		let st0 = h * S * S;
		for i in 0..S * S {
			st[st0 + i] *= g;
		}
		let mut sk = vec![0.0f32; S];
		for i in 0..S {
			let ki = kvec[i];
			let base = i * S;
			for j in 0..S {
				sk[j] += st[st0 + base + j] * ki;
			}
		}
		let mut d = vec![0.0f32; S];
		for j in 0..S {
			d[j] = (vvec[j] - sk[j]) * beta;
		}
		for i in 0..S {
			let ki = kvec[i];
			let base = i * S;
			for j in 0..S {
				st[st0 + base + j] += ki * d[j];
			}
		}
		for i in 0..S {
			let qi = qvec[i] * qscale;
			let base = i * S;
			for j in 0..S {
				o[h * S + j] += st[st0 + base + j] * qi;
			}
		}
	}
	let z = matmul(&wqkvg, &norm, value_dim, n);
	let mut no = vec![0.0f32; value_dim];
	for h in 0..Hv {
		let rs = rms_norm(&o[h * S..(h + 1) * S], &ssm_norm);
		for d in 0..S {
			no[h * S + d] = rs[d] * silu(z[h * S + d]);
		}
	}
	let aout = matmul(&sout_w, &no, n, value_dim);
	let after_attn: Vec<f32> = x.iter().zip(&aout).map(|(a, b)| a + b).collect();
	let post = rms_norm(&after_attn, &post_norm);
	let up = matmul(&fup_w, &post, nf, n);
	let fg = matmul(&fgate_w, &post, nf, n);
	let hvec: Vec<f32> = (0..nf).map(|i| silu(fg[i]) * up[i]).collect();
	let ffn_out = matmul(&fdown_w, &hvec, n, nf);
	let expected: Vec<f32> = after_attn
		.iter()
		.zip(&ffn_out)
		.map(|(a, b)| a + b)
		.collect();

	for i in 0..n {
		assert!(
			(gpu[i] - expected[i]).abs() < 1e-2,
			"ssm_block[{}] gpu={} cpu={}",
			i,
			gpu[i],
			expected[i]
		);
	}
	eprintln!("DEBUG gpu_ssm_block_matches_cpu OK");
}

/// Validate the new per-head SSM RMSNorm shader (`aux_ssm_rms_o`) against
/// a CPU reference. Unlike the fused block, this uses a fresh aux layout
/// (the lavapipe workaround) so it runs on the WSL CI software ICD.
#[test]
fn gpu_ssm_rms_o_matches_cpu() {
	let dev = match GpuDevice::init() {
		Ok(d) => d,
		Err(_) => {
			return;
		}
	};
	let S: usize = 4;
	let Hv: usize = 3;
	let n = Hv * S; // 12
	let eps = 1e-5f32;

	let o: Vec<f32> = (0..n)
		.map(|i| (((i * 7) % 50) as f32) / 25.0 - 1.0)
		.collect();
	let w: Vec<f32> = (0..S)
		.map(|i| (((i * 3) % 20) as f32) / 10.0 + 0.5)
		.collect();

	let mut exp = vec![0.0f32; n];
	for h in 0..Hv {
		let mut ss = 0.0f32;
		for d in 0..S {
			ss += o[h * S + d] * o[h * S + d];
		}
		let inv = 1.0 / (ss / S as f32 + eps).sqrt();
		for d in 0..S {
			exp[h * S + d] = o[h * S + d] * inv * w[d];
		}
	}

	let (o_b, _o_m) = dev_buf(&dev, &o);
	let (w_b, _w_m) = dev_buf(&dev, &w);
	let (no_b, _no_m) = dev_buf(&dev, &vec![0.0f32; n]);
	let p = [
		S as u32,
		Hv as u32,
		eps.to_bits(),
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
	let (pb, _pm) = dev_param(&dev, &p);
	let (pl, layout, plyt) = fresh_aux(&dev, include_bytes!("../shaders/aux_ssm_rms_o.spv"), 3);
	let pool_sizes = [
		vk::DescriptorPoolSize {
			ty: vk::DescriptorType::STORAGE_BUFFER,
			descriptor_count: 16,
		},
		vk::DescriptorPoolSize {
			ty: vk::DescriptorType::UNIFORM_BUFFER,
			descriptor_count: 4,
		},
	];
	let pool = unsafe {
		dev.device
			.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo {
					max_sets: 4,
					pool_size_count: pool_sizes.len() as u32,
					p_pool_sizes: pool_sizes.as_ptr(),
					..Default::default()
				},
				None,
			)
			.unwrap()
	};
	let s = fresh_set(&dev, layout, pool, &[(0, o_b), (1, w_b), (2, no_b)], pb);
	dispatch(&dev, pl, plyt, s, &[o_b, w_b], Hv as u32, 1);
	let got = read_back_f32(&dev, no_b, n);
	for i in 0..n {
		assert!(
			(got[i] - exp[i]).abs() < 1e-3,
			"ssm_rms_o[{}] {} vs {}",
			i,
			got[i],
			exp[i]
		);
	}
	unsafe {
		dev.device.destroy_descriptor_pool(pool, None);
	}
}
