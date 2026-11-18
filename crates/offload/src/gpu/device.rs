use super::*;

impl GpuDevice {
	/// Acquire the Vulkan adapter/device via raw `ash` and build the fused
	/// dequant+matmul compute pipelines from precompiled SPIR-V.
	pub fn init() -> Result<GpuDevice, String> {
		Self::init_pick(None)
	}

	/// [`Self::init`] on the adapter `pick` names (index or name
	/// substring, as `QWEN_VK_DEVICE`, which it overrides).
	pub fn init_pick(pick: Option<&str>) -> Result<GpuDevice, String> {
		let entry =
			unsafe { ash::Entry::load() }.map_err(|e| format!("failed to load vulkan-1: {e:?}"))?;
		// Vulkan 1.1 is requested so kernels can use subgroup arithmetic
		// (`subgroupAdd`, SPIR-V 1.3) for their row reductions instead of a
		// shared-memory tree with a barrier per level.
		let app_info = vk::ApplicationInfo {
			api_version: vk::make_api_version(0, 1, 1, 0),
			..Default::default()
		};
		let instance = unsafe {
			entry
				.create_instance(
					&vk::InstanceCreateInfo {
						p_application_info: &app_info,
						enabled_extension_count: 0,
						pp_enabled_extension_names: ptr::null(),
						enabled_layer_count: 0,
						pp_enabled_layer_names: ptr::null(),
						..Default::default()
					},
					None,
				)
				.map_err(|e| format!("Vulkan instance creation failed: {e}"))?
		};

		let pds = unsafe {
			instance
				.enumerate_physical_devices()
				.map_err(|e| format!("no Vulkan physical devices: {e}"))?
		};
		if pds.is_empty() {
			return Err("no Vulkan physical devices available".into());
		}
		// `QWEN_VK_DEVICE`: pick an adapter by enumeration index or by a
		// case-insensitive substring of its name (e.g. `intel` for the
		// Arrow Lake iGPU). Default: discrete over integrated.
		let pick = pick
			.map(str::to_string)
			.or_else(|| std::env::var("QWEN_VK_DEVICE").ok())
			.map(|s| s.trim().to_ascii_lowercase());
		let name_of = |pd: &vk::PhysicalDevice| {
			let p = unsafe { instance.get_physical_device_properties(*pd) };
			unsafe {
				CStr::from_ptr(p.device_name.as_ptr())
					.to_string_lossy()
					.to_ascii_lowercase()
			}
		};
		let chosen = pick.as_ref().and_then(|s| match s.parse::<usize>() {
			Ok(i) => pds.get(i).copied(),
			Err(_) => pds
				.iter()
				.find(|pd| name_of(pd).contains(s.as_str()))
				.copied(),
		});
		if let (Some(p), None) = (&pick, chosen) {
			return Err(format!("QWEN_VK_DEVICE={p} matches no Vulkan adapter"));
		}
		let physical = chosen.unwrap_or_else(|| {
			pds.iter()
				.copied()
				.max_by_key(|pd| {
					let p = unsafe { instance.get_physical_device_properties(*pd) };
					match p.device_type {
						vk::PhysicalDeviceType::DISCRETE_GPU => 2i32,
						vk::PhysicalDeviceType::INTEGRATED_GPU => 1i32,
						_ => 0i32,
					}
				})
				.unwrap()
		});
		let props = unsafe { instance.get_physical_device_properties(physical) };
		let dev_name = unsafe {
			CStr::from_ptr(props.device_name.as_ptr())
				.to_string_lossy()
				.into_owned()
		};
		eprintln!("offload: Vulkan device: {dev_name}");
		let llvmpipe = dev_name.contains("llvmpipe");

		let qf = unsafe { instance.get_physical_device_queue_family_properties(physical) };
		let qfamily = qf
			.iter()
			.position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
			.ok_or("no compute-capable queue family")? as u32;

		// Optional DP4A (integer dot product) support.
		let exts = unsafe {
			instance
				.enumerate_device_extension_properties(physical)
				.map_err(|e| format!("device ext props: {e}"))?
		};
		let dp4a_ext_name = CString::new("VK_KHR_shader_integer_dot_product").unwrap();
		let has_dp4a_ext = exts.iter().any(|e| unsafe {
			CStr::from_ptr(e.extension_name.as_ptr()) == dp4a_ext_name.as_c_str()
		});
		// L4: DP4A (integer dot product) Q4_K is the default when the device
		// advertises it. `QWEN_GPU_NODP4A` forces the scalar Q4_K shader. The
		// chosen path is verified by a runtime self-check at the end of init;
		// any mismatch silently falls back to scalar (no behavior change).
		let want_dp4a = has_dp4a_ext && std::env::var("QWEN_GPU_NODP4A").is_err();
		// `VK_EXT_memory_budget`: how much VRAM this process may actually
		// use right now. The heap *size* is not that number on Windows:
		// the desktop compositor and other apps hold part of the 6 GiB, and
		// WDDM does not fail an allocation past the real budget, it quietly
		// demotes memory to system RAM and every dispatch touching it
		// crawls over PCIe (decode fell from ~3.6 to 0.8 tok/s at 5.37 GB).
		let budget_ext_name = CString::new("VK_EXT_memory_budget").unwrap();
		let has_budget_ext = exts.iter().any(|e| unsafe {
			CStr::from_ptr(e.extension_name.as_ptr()) == budget_ext_name.as_c_str()
		});

		// A **transfer-only** queue family is the card's DMA engine, and it
		// is what makes a weight upload overlap instead of serialize: the
		// streaming path submits its copies there and does not wait for them
		// until the dispatch that reads them is about to be recorded. On the
		// GTX 1060 that is family 1 (`QUEUE_TRANSFER_BIT | SPARSE_BINDING`);
		// a device that offers no such family (llvmpipe) keeps using the
		// compute queue, which still decouples the upload from the host but
		// cannot run it beside a dispatch.
		let xfamily = qf
			.iter()
			.position(|f| {
				f.queue_flags.contains(vk::QueueFlags::TRANSFER)
					&& !f.queue_flags.contains(vk::QueueFlags::COMPUTE)
					&& !f.queue_flags.contains(vk::QueueFlags::GRAPHICS)
			})
			.map(|i| i as u32);

		let queue_priority = 1.0f32;
		let mut qcis = vec![vk::DeviceQueueCreateInfo {
			queue_family_index: qfamily,
			queue_count: 1,
			p_queue_priorities: &queue_priority,
			..Default::default()
		}];
		if let Some(xf) = xfamily {
			qcis.push(vk::DeviceQueueCreateInfo {
				queue_family_index: xf,
				queue_count: 1,
				p_queue_priorities: &queue_priority,
				..Default::default()
			});
		}

		let dp4a_feat = vk::PhysicalDeviceShaderIntegerDotProductFeatures {
			shader_integer_dot_product: vk::TRUE,
			..Default::default()
		};

		// std430 layout on uniform blocks (our `Params` UBO) requires the
		// scalar-block-layout feature, so enable it unconditionally and chain
		// any extra features (dp4a) after it.
		let mut scalar_feat = vk::PhysicalDeviceScalarBlockLayoutFeatures {
			scalar_block_layout: vk::TRUE,
			p_next: if want_dp4a {
				&dp4a_feat as *const _ as *mut c_void
			} else {
				ptr::null_mut()
			},
			..Default::default()
		};

		// Variable subgroup width (Intel 8..32): require 32 on every
		// pipeline (see `REQUIRE_SUBGROUP_32`). NVIDIA is fixed at 32.
		let sgs_ext_name = CString::new("VK_EXT_subgroup_size_control").unwrap();
		let has_sgs_ext = exts.iter().any(|e| unsafe {
			CStr::from_ptr(e.extension_name.as_ptr()) == sgs_ext_name.as_c_str()
		});
		let mut sgs_props = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
		if has_sgs_ext {
			let mut p2 = vk::PhysicalDeviceProperties2 {
				p_next: &mut sgs_props as *mut _ as *mut c_void,
				..Default::default()
			};
			unsafe { instance.get_physical_device_properties2(physical, &mut p2) };
		}
		let want_sg32 = has_sgs_ext
			&& sgs_props.min_subgroup_size < 32
			&& sgs_props.max_subgroup_size >= 32
			&& sgs_props
				.required_subgroup_size_stages
				.contains(vk::ShaderStageFlags::COMPUTE);
		let mut sgs_feat = vk::PhysicalDeviceSubgroupSizeControlFeatures {
			subgroup_size_control: vk::TRUE,
			p_next: scalar_feat.p_next,
			..Default::default()
		};
		if want_sg32 {
			scalar_feat.p_next = &mut sgs_feat as *mut _ as *mut c_void;
		}
		REQUIRE_SUBGROUP_32.store(want_sg32, std::sync::atomic::Ordering::Relaxed);
		if want_sg32 {
			eprintln!(
				"offload: subgroup size {}..{}, requiring 32 for the subgroupAdd GEMVs",
				sgs_props.min_subgroup_size, sgs_props.max_subgroup_size
			);
		}

		let mut ext_ptrs: Vec<*const std::ffi::c_char> = Vec::new();
		if want_sg32 {
			ext_ptrs.push(sgs_ext_name.as_ptr());
		}
		if want_dp4a {
			ext_ptrs.push(dp4a_ext_name.as_ptr());
		}
		if has_budget_ext {
			ext_ptrs.push(budget_ext_name.as_ptr());
		}
		// `VK_EXT_external_memory_host`: the prefill split's GEMMs write
		// straight into the caller's output matrix (`hostimp`).
		let host_ext_name = CString::new("VK_EXT_external_memory_host").unwrap();
		let has_host_ext = exts.iter().any(|e| unsafe {
			CStr::from_ptr(e.extension_name.as_ptr()) == host_ext_name.as_c_str()
		});
		let mut host_props = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
		if has_host_ext {
			let mut p2 = vk::PhysicalDeviceProperties2 {
				p_next: &mut host_props as *mut _ as *mut c_void,
				..Default::default()
			};
			unsafe { instance.get_physical_device_properties2(physical, &mut p2) };
			ext_ptrs.push(host_ext_name.as_ptr());
		}
		let device = unsafe {
			instance
				.create_device(
					physical,
					&vk::DeviceCreateInfo {
						queue_create_info_count: qcis.len() as u32,
						p_queue_create_infos: qcis.as_ptr(),
						enabled_extension_count: ext_ptrs.len() as u32,
						pp_enabled_extension_names: if ext_ptrs.is_empty() {
							ptr::null()
						} else {
							ext_ptrs.as_ptr()
						},
						p_enabled_features: ptr::null(),
						p_next: &scalar_feat as *const _ as *mut c_void,
						..Default::default()
					},
					None,
				)
				.map_err(|e| format!("Vulkan device creation failed: {e}"))?
		};
		let mut use_dp4a = want_dp4a;
		let queue = unsafe { device.get_device_queue(qfamily, 0) };
		let xfer_queue = xfamily.map(|f| unsafe { device.get_device_queue(f, 0) });

		// SPIR-V shader modules.
		let sm_q4k = create_sm(&device, &load_spv(include_bytes!("../shaders/q4k.spv")))?;
		let sm_q6k = create_sm(&device, &load_spv(include_bytes!("../shaders/q6k.spv")))?;
		let sm_q4k_f32 = create_sm(&device, &load_spv(include_bytes!("../shaders/q4k_f32.spv")))?;
		let sm_q6k_f32 = create_sm(&device, &load_spv(include_bytes!("../shaders/q6k_f32.spv")))?;
		let sm_q4k_f32_seq = create_sm(
			&device,
			&load_spv(include_bytes!("../shaders/q4k_f32_seq.spv")),
		)?;
		let sm_q6k_f32_seq = create_sm(
			&device,
			&load_spv(include_bytes!("../shaders/q6k_f32_seq.spv")),
		)?;
		let sm_q8_0 = create_sm(&device, &load_spv(include_bytes!("../shaders/q8_0.spv")))?;
		let sm_f16 = create_sm(&device, &load_spv(include_bytes!("../shaders/f16.spv")))?;
		let sm_f32 = create_sm(&device, &load_spv(include_bytes!("../shaders/f32.spv")))?;
		let sm_gemm_q4k = create_sm(
			&device,
			&load_spv(include_bytes!("../shaders/gemm_q4k.spv")),
		)?;
		let sm_gemm_q6k = create_sm(
			&device,
			&load_spv(include_bytes!("../shaders/gemm_q6k.spv")),
		)?;
		// DP4A Q4_K is optional: if the precompiled SPIR-V fails to load
		// (e.g. a driver that rejects `OpSDotKHR`), disable it and keep the
		// validated scalar Q4_K path. This must not break the whole device.
		let sm_q4k_dp4a = if use_dp4a {
			match create_sm(
				&device,
				&load_spv(include_bytes!("../shaders/dp4a_q4k.spv")),
			) {
				Ok(sm) => Some(sm),
				Err(e) => {
					eprintln!("offload: dp4a Q4_K shader load failed ({e}); using scalar Q4_K");
					use_dp4a = false;
					None
				}
			}
		} else {
			None
		};

		// Descriptor set layout: (0) weight storage, (1) x storage,
		// (2) y storage (rw), (3) params uniform.
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
				binding: 2,
				descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 1,
				stage_flags: vk::ShaderStageFlags::COMPUTE,
				p_immutable_samplers: ptr::null(),
				..Default::default()
			},
			vk::DescriptorSetLayoutBinding {
				binding: 3,
				descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
				descriptor_count: 1,
				stage_flags: vk::ShaderStageFlags::COMPUTE,
				p_immutable_samplers: ptr::null(),
				..Default::default()
			},
		];
		let desc_layout = unsafe {
			device
				.create_descriptor_set_layout(
					&vk::DescriptorSetLayoutCreateInfo {
						binding_count: bindings.len() as u32,
						p_bindings: bindings.as_ptr(),
						..Default::default()
					},
					None,
				)
				.map_err(|e| format!("desc layout: {e}"))?
		};
		let pipeline_layout = unsafe {
			device
				.create_pipeline_layout(
					&vk::PipelineLayoutCreateInfo {
						set_layout_count: 1,
						p_set_layouts: &desc_layout,
						..Default::default()
					},
					None,
				)
				.map_err(|e| format!("pipeline layout: {e}"))?
		};

		let main = CString::new("main").unwrap();
		let pipeline_q4k = compute_pipeline(&device, pipeline_layout, sm_q4k, &main)?;
		let pipeline_q6k = compute_pipeline(&device, pipeline_layout, sm_q6k, &main)?;
		let pipeline_q4k_f32 = compute_pipeline_sg32(&device, pipeline_layout, sm_q4k_f32, &main)?;
		let pipeline_q6k_f32 = compute_pipeline_sg32(&device, pipeline_layout, sm_q6k_f32, &main)?;
		let pipeline_q4k_f32_seq =
			compute_pipeline_sg32(&device, pipeline_layout, sm_q4k_f32_seq, &main)?;
		let pipeline_q6k_f32_seq =
			compute_pipeline_sg32(&device, pipeline_layout, sm_q6k_f32_seq, &main)?;
		let pipeline_q8_0 = compute_pipeline(&device, pipeline_layout, sm_q8_0, &main)?;
		let pipeline_f16 = compute_pipeline(&device, pipeline_layout, sm_f16, &main)?;
		let pipeline_f32 = compute_pipeline(&device, pipeline_layout, sm_f32, &main)?;
		let pipeline_gemm_q4k = compute_pipeline(&device, pipeline_layout, sm_gemm_q4k, &main)?;
		let pipeline_gemm_q6k = compute_pipeline(&device, pipeline_layout, sm_gemm_q6k, &main)?;
		// The dp4a GEMMs (~2x the float ones) are optional the same way
		// as the dp4a GEMV; `QWEN_GPU_GEMM_FLOAT=1` keeps the float GEMMs.
		let dp4a_gemm = |spv: &[u8], name: &str| -> Option<vk::Pipeline> {
			if !use_dp4a || std::env::var("QWEN_GPU_GEMM_FLOAT").is_ok() {
				return None;
			}
			match create_sm(&device, &load_spv(spv))
				.and_then(|m| compute_pipeline(&device, pipeline_layout, m, &main))
			{
				Ok(p) => Some(p),
				Err(e) => {
					eprintln!("offload: dp4a {name} GEMM unavailable ({e}); using the float GEMM");
					None
				}
			}
		};
		let pipeline_gemm_q4k_dp4a =
			dp4a_gemm(include_bytes!("../shaders/gemm_q4k_dp4a.spv"), "Q4_K");
		let pipeline_gemm_q6k_dp4a =
			dp4a_gemm(include_bytes!("../shaders/gemm_q6k_dp4a.spv"), "Q6_K");
		let pipeline_q4k_dp4a = if use_dp4a {
			if let Some(m) = sm_q4k_dp4a {
				match compute_pipeline_sg32(&device, pipeline_layout, m, &main) {
					Ok(p) => Some(p),
					Err(e) => {
						eprintln!("offload: dp4a Q4_K pipeline failed ({e}); using scalar Q4_K");
						use_dp4a = false;
						None
					}
				}
			} else {
				None
			}
		} else {
			None
		};

		// ---- L2/L3 aux shaders (rmsnorm/rope/attn/silu/residual/quantize) ----
		// NOTE: lavapipe (the CI software Vulkan ICD) mis-binds descriptor sets
		// with 8 storage-buffer bindings; it works with at most 6 storage
		// bindings (the most any aux shader uses; attn's scratch is binding 5)
		// plus the uniform at binding 8. Keep this layout minimal and matching
		// the shaders (storage 0..=5, uniform 8).
		let mk_binding = |b: u32| vk::DescriptorSetLayoutBinding {
			binding: b,
			descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
			descriptor_count: 1,
			stage_flags: vk::ShaderStageFlags::COMPUTE,
			p_immutable_samplers: ptr::null(),
			..Default::default()
		};
		let aux_bindings: [vk::DescriptorSetLayoutBinding; 7] = [
			mk_binding(0),
			mk_binding(1),
			mk_binding(2),
			mk_binding(3),
			mk_binding(4),
			mk_binding(5),
			vk::DescriptorSetLayoutBinding {
				binding: 8,
				descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
				descriptor_count: 1,
				stage_flags: vk::ShaderStageFlags::COMPUTE,
				p_immutable_samplers: ptr::null(),
				..Default::default()
			},
		];
		let aux_layout = unsafe {
			device
				.create_descriptor_set_layout(
					&vk::DescriptorSetLayoutCreateInfo {
						binding_count: aux_bindings.len() as u32,
						p_bindings: aux_bindings.as_ptr(),
						..Default::default()
					},
					None,
				)
				.map_err(|e| format!("aux desc layout: {e}"))?
		};
		let aux_pipeline_layout = unsafe {
			device
				.create_pipeline_layout(
					&vk::PipelineLayoutCreateInfo {
						set_layout_count: 1,
						p_set_layouts: &aux_layout,
						..Default::default()
					},
					None,
				)
				.map_err(|e| format!("aux pipeline layout: {e}"))?
		};
		let sm_rms_sum = create_sm(
			&device,
			&load_spv(include_bytes!("../shaders/aux_rmsnorm_sum.spv")),
		)?;
		let sm_rms_apply = create_sm(
			&device,
			&load_spv(include_bytes!("../shaders/aux_rmsnorm_apply.spv")),
		)?;
		let sm_prep_q = create_sm(
			&device,
			&load_spv(include_bytes!("../shaders/aux_prep_q.spv")),
		)?;
		let sm_prep_k = create_sm(
			&device,
			&load_spv(include_bytes!("../shaders/aux_prep_k.spv")),
		)?;
		let sm_attn = create_sm(
			&device,
			&load_spv(include_bytes!("../shaders/aux_attn.spv")),
		)?;
		let sm_silu = create_sm(
			&device,
			&load_spv(include_bytes!("../shaders/aux_silu_mul.spv")),
		)?;
		let sm_residual = create_sm(
			&device,
			&load_spv(include_bytes!("../shaders/aux_residual.spv")),
		)?;
		let sm_q8 = create_sm(
			&device,
			&load_spv(include_bytes!("../shaders/aux_quantize_q8k.spv")),
		)?;
		let pl_rms_sum = compute_pipeline(&device, aux_pipeline_layout, sm_rms_sum, &main)?;
		let pl_rms_apply = compute_pipeline(&device, aux_pipeline_layout, sm_rms_apply, &main)?;
		let pl_prep_q = compute_pipeline(&device, aux_pipeline_layout, sm_prep_q, &main)?;
		let pl_prep_k = compute_pipeline(&device, aux_pipeline_layout, sm_prep_k, &main)?;
		let pl_attn = compute_pipeline(&device, aux_pipeline_layout, sm_attn, &main)?;
		let pl_silu = compute_pipeline(&device, aux_pipeline_layout, sm_silu, &main)?;
		let pl_residual = compute_pipeline(&device, aux_pipeline_layout, sm_residual, &main)?;
		let pl_q8 = compute_pipeline(&device, aux_pipeline_layout, sm_q8, &main)?;
		let sm_conv1d = create_sm(
			&device,
			&load_spv(include_bytes!("../shaders/aux_conv1d.spv")),
		)?;
		let sm_l2norm = create_sm(
			&device,
			&load_spv(include_bytes!("../shaders/aux_l2norm.spv")),
		)?;
		let sm_ssm_beta = create_sm(
			&device,
			&load_spv(include_bytes!("../shaders/aux_ssm_beta.spv")),
		)?;
		let sm_ssm_gate = create_sm(
			&device,
			&load_spv(include_bytes!("../shaders/aux_ssm_gate.spv")),
		)?;
		let sm_delta_net = create_sm(
			&device,
			&load_spv(include_bytes!("../shaders/aux_delta_net.spv")),
		)?;
		let pl_conv1d = compute_pipeline(&device, aux_pipeline_layout, sm_conv1d, &main)?;
		let pl_l2norm = compute_pipeline(&device, aux_pipeline_layout, sm_l2norm, &main)?;
		let pl_ssm_beta = compute_pipeline(&device, aux_pipeline_layout, sm_ssm_beta, &main)?;
		let pl_ssm_gate = compute_pipeline(&device, aux_pipeline_layout, sm_ssm_gate, &main)?;
		let pl_delta_net = compute_pipeline(&device, aux_pipeline_layout, sm_delta_net, &main)?;
		let sm_ssm_rms_o = create_sm(
			&device,
			&load_spv(include_bytes!("../shaders/aux_ssm_rms_o.spv")),
		)?;
		let pl_ssm_rms_o = compute_pipeline(&device, aux_pipeline_layout, sm_ssm_rms_o, &main)?;

		// Persistent descriptor pool for per-weight (decode path) sets.
		let pool_sizes = [
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 8192 * 3,
			},
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::UNIFORM_BUFFER,
				descriptor_count: 8192,
			},
		];
		let desc_pool = unsafe {
			device
				.create_descriptor_pool(
					&vk::DescriptorPoolCreateInfo {
						max_sets: 8192,
						pool_size_count: pool_sizes.len() as u32,
						p_pool_sizes: pool_sizes.as_ptr(),
						..Default::default()
					},
					None,
				)
				.map_err(|e| format!("desc pool: {e}"))?
		};

		let cmd_pool = unsafe {
			device
				.create_command_pool(
					&vk::CommandPoolCreateInfo {
						queue_family_index: qfamily,
						..Default::default()
					},
					None,
				)
				.map_err(|e| format!("cmd pool: {e}"))?
		};
		let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
			.map_err(|e| format!("fence: {e}"))?;
		let fence2 = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
			.map_err(|e| format!("fence: {e}"))?;

		// Command pool + fence for the transfer queue. Both are created even
		// when there is no dedicated family (the pool then sits on the
		// compute family), so the streaming path has exactly one code path.
		let xfer_family = xfamily.unwrap_or(qfamily);
		let xfer_pool = unsafe {
			device
				.create_command_pool(
					&vk::CommandPoolCreateInfo {
						queue_family_index: xfer_family,
						..Default::default()
					},
					None,
				)
				.map_err(|e| format!("xfer cmd pool: {e}"))?
		};
		let xfer_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
			.map_err(|e| format!("xfer fence: {e}"))?;
		let xfer_sem =
			unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
				.map_err(|e| format!("xfer semaphore: {e}"))?;
		if xfamily.is_some() {
			eprintln!("offload: transfer queue family {xfer_family} (dedicated DMA engine)");
		}

		let mem_props = unsafe { instance.get_physical_device_memory_properties(physical) };

		let host_import =
			has_host_ext.then(|| ash::ext::external_memory_host::Device::new(&instance, &device));
		let mut dev = GpuDevice {
			entry,
			instance,
			physical,
			device,
			queue,
			qfamily,
			xfer_queue,
			xfer_family,
			xfer_pool,
			xfer_fence,
			xfer_sem,
			cmd_pool,
			fence,
			fence2,
			desc_pool,
			desc_layout,
			pipeline_layout,
			pipeline_q4k,
			pipeline_q6k,
			pipeline_q8_0,
			pipeline_f16,
			pipeline_f32,
			pipeline_q4k_dp4a,
			pipeline_q4k_f32,
			pipeline_q6k_f32,
			pipeline_q4k_f32_seq,
			pipeline_q6k_f32_seq,
			seq_gemv: std::env::var("QWEN_GPU_NOSEQGEMV").is_err(),
			f32_gemv: std::env::var("QWEN_GPU_Q8GEMV").is_err(),
			pipeline_gemm_q4k,
			pipeline_gemm_q4k_dp4a,
			pipeline_gemm_q6k_dp4a,
			pipeline_gemm_q6k,
			stream: Mutex::new(stream::StreamPool::new()),
			host: Mutex::new(pool::HostPool::new()),
			no_gemm: std::env::var("QWEN_GPU_NOGEMM").is_ok(),
			use_dp4a,
			has_budget_ext,
			host_import,
			mem_props,
			lost: AtomicBool::new(false),
			llvmpipe,
			weights: HashMap::new(),
			decode_cmds: Mutex::new(HashMap::new()),
			cmd_cache: std::env::var("QWEN_GPU_NOCMDCACHE").is_err(),
			decode_seq: 1,
			aux_layout,
			aux_pipeline_layout,
			pl_rms_sum,
			pl_rms_apply,
			pl_prep_q,
			pl_prep_k,
			pl_attn,
			pl_silu,
			pl_residual,
			pl_q8,
			pl_conv1d,
			pl_l2norm,
			pl_ssm_beta,
			pl_ssm_gate,
			pl_delta_net,
			pl_ssm_rms_o,
			max_ctx: std::env::var("QWEN_GPU_CTX")
				.ok()
				.and_then(|s| s.parse::<usize>().ok())
				.unwrap_or(8192),
			block_res: Mutex::new(HashMap::new()),
			ssm_res: Mutex::new(HashMap::new()),
			block_oom: Mutex::new(Default::default()),
			small_arena: Mutex::new(Default::default()),
			attnp: Mutex::new(None),
		};
		if has_host_ext {
			dev.register_host_import(host_props.min_imported_host_pointer_alignment as usize);
		}

		// L4 self-check: if DP4A is selected, run a tiny Q4_K matmul through
		// the integer-dot shader and compare it against the trusted CPU
		// `gemv_q4k`. On any mismatch fall back to the scalar Q4_K shader.
		if dev.pipeline_q4k_dp4a.is_some() {
			if dev.selfcheck_dp4a() {
				eprintln!("offload: dp4a Q4_K self-check passed; using integer-dot Q4_K");
			} else {
				eprintln!("offload: dp4a Q4_K self-check FAILED; falling back to scalar Q4_K");
				dev.use_dp4a = false;
			}
		}
		// The self-check recorded a decode command buffer for its probe
		// weight, and may have changed which Q4_K pipeline is used.
		dev.clear_decode_cmds();
		Ok(dev)
	}

	/// Runtime correctness gate for the DP4A Q4_K path: dispatches a small
	/// known Q4_K matmul through the integer-dot shader and checks it equals
	/// `quant::gemv_q4k`. Returns `false` on any mismatch (or if a dispatch
	/// could not be performed). Leaks a few tiny temp buffers (one-time, at
	/// init) rather than disturbing the persistent weight map.
	pub(crate) fn selfcheck_dp4a(&mut self) -> bool {
		let out_rows = 8u32;
		let in_dim: usize = 256;
		let nb = in_dim / QK_K;
		let row_bytes = nb * Q4K_BLOCK;
		let mut q4k = vec![0u8; (out_rows as usize) * row_bytes];
		// Deterministic pseudo-random fill; any valid Q4_K byte pattern works
		// because the shader and `gemv_q4k` parse the same layout.
		for (i, b) in q4k.iter_mut().enumerate() {
			*b = ((i * 31 + 7) % 251) as u8;
		}
		// Pin each block's (d, dmin) header to finite f16s: random bytes can
		// decode to inf/NaN, and a NaN difference never trips the check.
		for blk in q4k.chunks_mut(Q4K_BLOCK) {
			blk[0..2].copy_from_slice(&0x2C00u16.to_le_bytes()); // d = 0.0625
			blk[2..4].copy_from_slice(&0x2800u16.to_le_bytes()); // dmin = 0.03125
		}
		let x: Vec<f32> = (0..in_dim)
			.map(|i| (((i * 13) % 100) as f32) / 50.0 - 1.0)
			.collect();

		self.upload(
			99999,
			WeightId::FfnUp,
			&q4k,
			GgmlType::Q4_K_M,
			out_rows as usize,
			in_dim,
		);
		// `matmul` dispatches `gemv_for`, which is the f32 GEMV unless
		// `QWEN_GPU_Q8GEMV`; force the q8_K route so the dp4a kernel runs.
		let f32_gemv = std::mem::replace(&mut self.f32_gemv, false);
		let y_gpu = self.matmul(99999, WeightId::FfnUp, &x, out_rows as usize, in_dim);
		self.f32_gemv = f32_gemv;
		let y_gpu = match y_gpu {
			Some(v) => v,
			None => {
				self.weights.remove(&(99999, WeightId::FfnUp));
				return false;
			}
		};
		if y_gpu.len() != out_rows as usize {
			self.weights.remove(&(99999, WeightId::FfnUp));
			return false;
		}
		let mut y_ref = vec![0f32; out_rows as usize];
		gemv_q4k(out_rows as usize, in_dim, &x, &q4k, &mut y_ref);

		// Integer parts are exact on both sides; only the per-block float
		// accumulation order differs, so the tolerance is relative.
		let mut ok = true;
		let mut maxdiff = 0.0f32;
		for i in 0..(out_rows as usize) {
			let d = (y_gpu[i] - y_ref[i]).abs() / (1.0 + y_ref[i].abs());
			if !d.is_finite() || d > maxdiff {
				maxdiff = if d.is_finite() { d } else { f32::INFINITY };
			}
		}
		if maxdiff > 1e-4 {
			ok = false;
			eprintln!(
				"offload: dp4a Q4_K self-check failed (maxdiff={:.4}); using scalar Q4_K",
				maxdiff
			);
		}
		self.weights.remove(&(99999, WeightId::FfnUp));
		ok
	}

	/// How many output rows one workgroup of the fused dequant-matmul shader
	/// covers. This is the **single source of truth** for the host-side grid:
	/// the scalar Q4_K kernel gives each output row its own 32-lane team
	/// (`shaders/q4k.glsl`, `local_size = (32, ROWS)`), so a workgroup covers
	/// `ROWS` rows; every other kernel is still one row per thread at
	/// `local_size_x = 64`. Dispatching the wrong count silently computes only
	/// part of the output, so all call sites must go through `grid_x`.
	pub(crate) fn rows_per_wg(&self, kind: GpuKind) -> usize {
		match kind {
			// q4k/q6k.glsl: 4 row teams of 32 lanes. q4k/q6k_f32.glsl and
			// dp4a_q4k.glsl: one warp computing NR = 4 rows (WARPS = 1) -
			// the same 4, so every variant shares this contract. Change the
			// `#define`s and this together.
			GpuKind::Q4K | GpuKind::Q6K => 4,
			_ => 64,
		}
	}

	/// Workgroup count in x for a fused matmul producing `out_dim` rows.
	pub(crate) fn grid_x(&self, kind: GpuKind, out_dim: usize) -> u32 {
		let r = self.rows_per_wg(kind);
		out_dim.div_ceil(r) as u32
	}

	/// Output rows and tokens one workgroup of `shaders/gemm_q4k.glsl`
	/// covers (`BM` / `BN` in that file). Kept beside `grid_x` because they
	/// are the same class of contract and a mismatch fails the same silent
	/// way: the dispatch computes part of the output and leaves the rest
	/// stale.
	pub(crate) const GEMM_BM: usize = 128;
	pub(crate) const GEMM_BN: usize = 64;

	/// Whether a `seq > 1` Q4_K matmul should use the tiled GEMM instead of
	/// the batched-GEMV kernel.
	///
	/// See `gemm_on` for the threshold. Decode (`seq == 1`) is never
	/// eligible: it is DRAM-bound on the CPU at 97 % of this machine's
	/// ceiling and is deliberately left alone.
	pub(crate) fn use_gemm(&self, kind: GpuKind, seq: usize) -> bool {
		!self.no_gemm && matches!(kind, GpuKind::Q4K | GpuKind::Q6K) && Self::gemm_on(seq)
	}

	/// Whether `seq` is long enough for the tiled GEMM. Below `GEMM_BN` it
	/// computes one padded tile, which since the all-padding warps skip
	/// their FMAs still beats the batched GEMV (0.12x CPU at 16 tokens) by ~3x
	/// (`gpubench --crossover`, one FFN block: 0.33x at 12, 0.38x at 16,
	/// 0.57x at 20). Up to `SPEC_MAX_SEQ` a batch is a speculative-decoding
	/// verify pass, which keeps the decode GEMV (`QWEN_GPU_GEMM_MIN_SEQ`
	/// overrides).
	pub fn gemm_on(seq: usize) -> bool {
		seq >= Self::gemm_min_seq()
	}

	pub fn gemm_min_seq() -> usize {
		static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
		*N.get_or_init(|| {
			std::env::var("QWEN_GPU_GEMM_MIN_SEQ")
				.ok()
				.and_then(|s| s.parse().ok())
				.unwrap_or(crate::SPEC_MAX_SEQ + 1)
		})
	}

	/// Chunk length below which a **resident** weight's prefill matmul is
	/// left on the CPU (`QWEN_GPU_PREFILL_MIN_SEQ`).
	///
	/// GPU prefill ships on, so this is now a *routing* decision rather than
	/// an opt-in, and it has to be right in both directions. Below
	/// `GEMM_BN` there is no tiled GEMM and the batched GEMV re-reads each
	/// weight row per position, which is not a small loss: measured with
	/// `gpubench --crossover` on one FFN block, resident:
	///
	/// | seq | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
	/// |---|---|---|---|---|---|---|---|---|
	/// | x CPU | 0.12 | 0.14 | **1.20** | 1.17 | 1.30 | 1.32 | 1.32 | 1.40 |
	///
	/// The step at 64 is the GEMV/GEMM switch, so the default threshold is
	/// `GEMM_BN` itself: an 8x loss on one side of it and a 1.2x win on the
	/// other. The attention group (`Wq`+`Wk`+`Wv`, then `Wo`) crosses at the
	/// same place with the same shape (0.16x at 32, 1.10x at 64), so one
	/// threshold covers both: the narrow `Wk`/`Wv` shapes do not need their
	/// own.
	/// Is routing this prefill matmul to the GPU faster than leaving it on
	/// the CPU? The **resident** half of the manager's routing gate.
	///
	/// Consulting `use_gemm` rather than re-testing its parts keeps
	/// `QWEN_GPU_NOGEMM` honest: with the tiled GEMM forced off, the GPU is
	/// running the batched GEMV, which is where the 0.12-0.14x figures
	/// below come from; that toggle should make the routing conservative,
	/// not make it dispatch the slow kernel.
	pub(crate) fn prefill_worth_it(&self, kind: GpuKind, seq: usize) -> bool {
		self.use_gemm(kind, seq) && seq >= Self::prefill_min_seq()
	}

	pub fn prefill_min_seq() -> usize {
		static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
		*N.get_or_init(|| {
			std::env::var("QWEN_GPU_PREFILL_MIN_SEQ")
				.ok()
				.and_then(|s| s.parse().ok())
				.unwrap_or(Self::GEMM_BN)
		})
	}

	/// Whether `gemm_pipeline_for(kind)` is the dp4a GEMM (~2x the float
	/// one), which moves the prefill split's balance point.
	pub(crate) fn dp4a_gemm(&self, kind: GpuKind) -> bool {
		self.use_dp4a
			&& match kind {
				GpuKind::Q6K => self.pipeline_gemm_q6k_dp4a.is_some(),
				_ => self.pipeline_gemm_q4k_dp4a.is_some(),
			}
	}

	/// The tiled-GEMM pipeline for `kind`. Only valid where `use_gemm` said
	/// yes; the two are always chosen together with `gemm_grid`.
	pub(crate) fn gemm_pipeline_for(&self, kind: GpuKind) -> vk::Pipeline {
		match kind {
			// `use_dp4a` is cleared if the device fails the dp4a self-check.
			GpuKind::Q6K => match self.pipeline_gemm_q6k_dp4a {
				Some(p) if self.use_dp4a => p,
				_ => self.pipeline_gemm_q6k,
			},
			_ => match self.pipeline_gemm_q4k_dp4a {
				Some(p) if self.use_dp4a => p,
				_ => self.pipeline_gemm_q4k,
			},
		}
	}

	/// Workgroup grid for the tiled Q4_K GEMM: rows in x, token tiles in y.
	/// Note y is a token *tile* here, unlike the GEMV kernels where y is a
	/// single position.
	pub(crate) fn gemm_grid(&self, out_dim: usize, seq: usize) -> (u32, u32) {
		(
			out_dim.div_ceil(Self::GEMM_BM) as u32,
			seq.div_ceil(Self::GEMM_BN) as u32,
		)
	}

	/// Whether the per-matmul GEMV for `kind` reads raw f32 activations
	/// (`q4k_f32` / `q6k_f32`) rather than q8_K.
	///
	/// dp4a does not change this. Its kernel is ~1.4x `q4k_f32` alone
	/// (157 vs 110-130 GB/s on FfnUp), but decode did not move (4.05/4.57
	/// vs 4.08/4.71 tok/s interleaved): the row split's GPU share is
	/// capped by VRAM, so the GPU already waits on the CPU rows. And the
	/// q8_K activation costs accuracy: `--forward 369` logits vs
	/// `QWEN_EXACT_DECODE`: RMS 0.165 vs 0.125. dp4a replaces the *q8_K*
	/// kernel (`pipeline_for`), where the numerics are the same.
	pub(crate) fn gemv_x_f32(&self, kind: GpuKind) -> bool {
		match kind {
			GpuKind::Q4K | GpuKind::Q6K => self.f32_gemv,
			_ => true,
		}
	}

	/// Whether a (kind, seq) matmul through `matmul_batch*` wants its
	/// activation packed as q8_K. The tiled GEMMs always do.
	pub(crate) fn wants_q8k(&self, kind: GpuKind, seq: usize) -> bool {
		matches!(kind, GpuKind::Q4K | GpuKind::Q6K)
			&& (self.use_gemm(kind, seq) || !self.gemv_x_f32(kind))
	}

	/// The GEMV pipeline the per-matmul paths dispatch; its activation
	/// format is `gemv_x_f32(kind)`. The fused blocks use `pipeline_for`.
	pub(crate) fn gemv_for(&self, kind: GpuKind) -> vk::Pipeline {
		match kind {
			GpuKind::Q4K if self.gemv_x_f32(kind) => self.pipeline_q4k_f32,
			GpuKind::Q6K if self.gemv_x_f32(kind) => self.pipeline_q6k_f32,
			_ => self.pipeline_for(kind),
		}
	}

	/// Positions the multi-position GEMVs (`q4k_f32_seq.glsl`,
	/// `q6k_f32_seq.glsl`) cover per workgroup: their `NS`.
	pub(crate) const SEQ_GEMV_NS: usize = 8;

	/// Pipeline and grid of a GEMV over `seq` positions. Pipeline and grid
	/// are chosen together because the kernels have different contracts:
	/// `q4k_f32` takes one position per `grid.y`, the `_seq` variants
	/// `SEQ_GEMV_NS`. For `seq > 1` (a speculative verify batch) the `_seq`
	/// kernels read each weight once for all positions instead of once per
	/// position: 1.2-1.5x (Q4_K) / 1.8x (Q6_K) at seq 8 on the GTX 1060;
	/// `QWEN_GPU_NOSEQGEMV=1` is the A/B.
	pub(crate) fn gemv_dispatch(
		&self,
		kind: GpuKind,
		out_dim: usize,
		seq: usize,
	) -> (vk::Pipeline, u32, u32) {
		let gx = self.grid_x(kind, out_dim);
		if seq > 1 && self.seq_gemv && self.gemv_x_f32(kind) {
			let gy = seq.div_ceil(Self::SEQ_GEMV_NS) as u32;
			match kind {
				GpuKind::Q4K => return (self.pipeline_q4k_f32_seq, gx, gy),
				GpuKind::Q6K => return (self.pipeline_q6k_f32_seq, gx, gy),
				_ => {}
			}
		}
		(self.gemv_for(kind), gx, seq as u32)
	}

	/// The q8_K-activation pipeline for `kind` (fused blocks, which
	/// quantize on the GPU with `aux_quantize_q8k`).
	pub(crate) fn pipeline_for(&self, kind: GpuKind) -> vk::Pipeline {
		match kind {
			GpuKind::Q4K => {
				if self.use_dp4a {
					self.pipeline_q4k_dp4a.unwrap()
				} else {
					self.pipeline_q4k
				}
			}
			GpuKind::Q6K => self.pipeline_q6k,
			GpuKind::Q8_0 => self.pipeline_q8_0,
			GpuKind::F16 => self.pipeline_f16,
			GpuKind::F32 => self.pipeline_f32,
		}
	}

	/// Create a buffer + device memory of the requested properties and bind.
	/// Allocate a device-local GPU buffer (with `usage`, which should include
	/// `TRANSFER_DST`) plus a host-visible staging buffer (mapped) for CPU
	/// writes. Host-visible memory on this GPU does NOT support
	/// STORAGE_BUFFER/UNIFORM_BUFFER usage, so shader-readable buffers must be
	/// device-local; the CPU writes the staging buffer, then we copy it to the
	/// device buffer before dispatching.
	pub(crate) fn create_staged_buffer(
		&self,
		size: u64,
		usage: vk::BufferUsageFlags,
	) -> (
		vk::Buffer,
		vk::DeviceMemory,
		vk::Buffer,
		vk::DeviceMemory,
		MappedPtr,
	) {
		let (dev_buf, dev_mem) = self.create_gpu_buffer(
			size,
			usage | vk::BufferUsageFlags::TRANSFER_DST,
			vk::MemoryPropertyFlags::DEVICE_LOCAL,
		);
		let (stg_buf, stg_mem) = self.create_gpu_buffer(
			size,
			vk::BufferUsageFlags::TRANSFER_SRC,
			vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
		);
		let stg_ptr = MappedPtr(self.map(stg_mem, size as usize));
		(dev_buf, dev_mem, stg_buf, stg_mem, stg_ptr)
	}

	/// [`Self::create_staged_buffer`] whose device-local half comes from
	/// [`Self::create_small_gpu_buffer`]; its memory handle is then null.
	pub(crate) fn create_staged_small(
		&self,
		size: u64,
		usage: vk::BufferUsageFlags,
	) -> (
		vk::Buffer,
		vk::DeviceMemory,
		vk::Buffer,
		vk::DeviceMemory,
		MappedPtr,
	) {
		let (dev_buf, dev_mem) =
			self.create_small_gpu_buffer(size, usage | vk::BufferUsageFlags::TRANSFER_DST);
		let (stg_buf, stg_mem) = self.create_gpu_buffer(
			size,
			vk::BufferUsageFlags::TRANSFER_SRC,
			vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
		);
		let stg_ptr = MappedPtr(self.map(stg_mem, size as usize));
		(dev_buf, dev_mem, stg_buf, stg_mem, stg_ptr)
	}

	/// A small device-local buffer bound into a shared block of
	/// `small_arena` instead of its own allocation. Returns a null memory
	/// handle in that case (the arena owns the memory); a buffer too large
	/// for a block gets its own allocation as before.
	///
	/// **Why:** the decode split creates three device-local buffers per
	/// weight (`x`, `y`, params; ~1200 for the whole model), and on the
	/// GTX 1060 under WDDM that many separate sub-megabyte allocations
	/// make the device **lost** once ~2.4-3 GB of weights sit beside them -
	/// well inside the reported budget, with no TDR logged and no error
	/// before the next `vkQueueSubmit`. Reproduced in raw Vulkan
	/// (`examples/vramprobe.rs`): the small buffers as separate
	/// allocations failed 7 of 7 runs at ~2.5 GB; the same buffers in one
	/// block filled 5 GB 6 of 6. The lost device then silently sent every
	/// decode matmul back to the CPU (3.4 instead of ~4.6 tok/s).
	pub(crate) fn create_small_gpu_buffer(
		&self,
		size: u64,
		usage: vk::BufferUsageFlags,
	) -> (vk::Buffer, vk::DeviceMemory) {
		const BLOCK: u64 = 32 << 20;
		let size = size.max(4);
		let buf = unsafe {
			self.device
				.create_buffer(
					&vk::BufferCreateInfo {
						size,
						usage,
						..Default::default()
					},
					None,
				)
				.expect("create_buffer")
		};
		let req = unsafe { self.device.get_buffer_memory_requirements(buf) };
		let mt = find_memory_type(
			&self.mem_props,
			vk::MemoryPropertyFlags::DEVICE_LOCAL,
			req.memory_type_bits,
		);
		let Some(mt) = mt.filter(|_| req.size <= BLOCK / 4) else {
			unsafe { self.device.destroy_buffer(buf, None) };
			return self.create_gpu_buffer(size, usage, vk::MemoryPropertyFlags::DEVICE_LOCAL);
		};
		let mut a = self.small_arena.lock().unwrap();
		let fits = |&(_, t, used): &(vk::DeviceMemory, u32, u64)| {
			t == mt && used.next_multiple_of(req.alignment) + req.size <= BLOCK
		};
		if !a.cur.as_ref().is_some_and(fits) {
			let mem = unsafe {
				self.device
					.allocate_memory(
						&vk::MemoryAllocateInfo {
							allocation_size: BLOCK,
							memory_type_index: mt,
							..Default::default()
						},
						None,
					)
					.unwrap_or_else(|e| panic!("small-buffer block ({BLOCK} bytes): {e:?}"))
			};
			a.blocks.push(mem);
			a.cur = Some((mem, mt, 0));
		}
		let (mem, _, used) = a.cur.as_mut().unwrap();
		let off = used.next_multiple_of(req.alignment);
		unsafe { self.device.bind_buffer_memory(buf, *mem, off) }.expect("bind_buffer_memory");
		*used = off + req.size;
		(buf, vk::DeviceMemory::null())
	}

	pub(crate) fn create_gpu_buffer(
		&self,
		size: u64,
		usage: vk::BufferUsageFlags,
		props: vk::MemoryPropertyFlags,
	) -> (vk::Buffer, vk::DeviceMemory) {
		self.try_create_gpu_buffer(size, usage, props)
			.unwrap_or_else(|| {
				panic!("create_gpu_buffer({size} bytes, props {props:?}): out of device memory")
			})
	}

	/// Fallible `create_gpu_buffer`. Returns `None` when the driver cannot
	/// back the allocation, so lazily-allocated per-layer resources can decline
	/// the GPU path instead of aborting the process (the fused L2/L3 blocks
	/// allocate on top of an already near-full VRAM budget).
	pub(crate) fn try_create_gpu_buffer(
		&self,
		size: u64,
		usage: vk::BufferUsageFlags,
		props: vk::MemoryPropertyFlags,
	) -> Option<(vk::Buffer, vk::DeviceMemory)> {
		self.try_create_buffer_shared(size, usage, props, false)
	}

	/// `try_create_gpu_buffer`, but `concurrent` puts the buffer in
	/// `SHARING_MODE_CONCURRENT` across the compute and transfer families.
	///
	/// A streamed weight is written by the transfer queue and read by the
	/// compute queue. With `EXCLUSIVE` sharing that costs a queue-family
	/// **ownership transfer** (a release barrier on one queue and a matching
	/// acquire on the other) and without it the buffer's contents are
	/// undefined after the transfer, which on this driver shows up as a
	/// dispatch reading zeros rather than as an error. Concurrent sharing
	/// makes the handoff implicit. It can cost the driver some compression,
	/// which is irrelevant for buffers holding already-quantized bytes.
	pub(crate) fn try_create_buffer_shared(
		&self,
		size: u64,
		usage: vk::BufferUsageFlags,
		props: vk::MemoryPropertyFlags,
		concurrent: bool,
	) -> Option<(vk::Buffer, vk::DeviceMemory)> {
		// `VkBufferCreateInfo::size` must be > 0 (Vulkan spec). A degenerate
		// dimension (e.g. an SSM-only layer whose attention dims are 0) would
		// otherwise ask for a 0-byte buffer: lavapipe tolerates it, the GTX
		// 1060 fails the backing allocation with ERROR_OUT_OF_DEVICE_MEMORY.
		let size = size.max(4);
		let families = [self.qfamily, self.xfer_family];
		let shared = concurrent && self.qfamily != self.xfer_family;
		let buf = unsafe {
			self.device
				.create_buffer(
					&vk::BufferCreateInfo {
						size,
						usage,
						sharing_mode: if shared {
							vk::SharingMode::CONCURRENT
						} else {
							vk::SharingMode::EXCLUSIVE
						},
						queue_family_index_count: if shared { 2 } else { 0 },
						p_queue_family_indices: if shared {
							families.as_ptr()
						} else {
							ptr::null()
						},
						..Default::default()
					},
					None,
				)
				.ok()?
		};
		let req = unsafe { self.device.get_buffer_memory_requirements(buf) };
		let mt = match find_memory_type(&self.mem_props, props, req.memory_type_bits) {
			Some(mt) => mt,
			None => {
				unsafe { self.device.destroy_buffer(buf, None) };
				return None;
			}
		};
		let mem = match unsafe {
			self.device.allocate_memory(
				&vk::MemoryAllocateInfo {
					allocation_size: req.size,
					memory_type_index: mt,
					..Default::default()
				},
				None,
			)
		} {
			Ok(m) => m,
			Err(_) => {
				unsafe { self.device.destroy_buffer(buf, None) };
				return None;
			}
		};
		if unsafe { self.device.bind_buffer_memory(buf, mem, 0) }.is_err() {
			unsafe {
				self.device.destroy_buffer(buf, None);
				self.device.free_memory(mem, None);
			}
			return None;
		}
		Some((buf, mem))
	}

	/// Host-visible buffer the CPU **reads** (a matmul result coming back).
	///
	/// The plain `HOST_VISIBLE | HOST_COHERENT` type this driver hands out
	/// is write-combined: fine to write, uncached to read, so a `memcpy`
	/// *out* of it runs at ~0.2 GB/s. Measured on the GTX 1060 that made the
	/// 143 MB prefill readback **670 ms**, four times the 172 ms kernel it
	/// was reading; it was 75 % of `matmul_batch_seq` and it is why the
	/// prefill GEMM looked like 12 % of the card's ALU ceiling when the
	/// shader itself was at 57 %.
	///
	/// `HOST_CACHED` is the same system memory made cacheable for reads. It
	/// is `HOST_COHERENT` as well on this driver, so no explicit
	/// `invalidate_mapped_memory_ranges` is needed; fall back to the
	/// uncached type on an adapter that offers no cached one (and then
	/// `HOST_COHERENT` still makes the mapping valid, just slow).
	pub(crate) fn create_readback_buffer(&self, size: u64) -> (vk::Buffer, vk::DeviceMemory) {
		self.try_create_readback_buffer(size)
			.expect("readback buffer")
	}

	/// [`Self::create_readback_buffer`], returning `None` instead of
	/// panicking so the host pool can fall back to a per-call allocation.
	pub(crate) fn try_create_readback_buffer(
		&self,
		size: u64,
	) -> Option<(vk::Buffer, vk::DeviceMemory)> {
		let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
		// STORAGE_BUFFER: a prefill GEMM writes its result straight into
		// it (`y_direct`).
		let usage = vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER;
		self.try_create_gpu_buffer(size, usage, host | vk::MemoryPropertyFlags::HOST_CACHED)
			.or_else(|| self.try_create_gpu_buffer(size, usage, host))
	}

	pub(crate) fn map(&self, mem: vk::DeviceMemory, size: usize) -> *mut c_void {
		unsafe {
			self.device
				.map_memory(mem, 0, size as u64, vk::MemoryMapFlags::empty())
				.expect("map_memory")
		}
	}

	/// Record a one-shot copy and block until complete.
	pub(crate) fn copy_buffer(&self, src: vk::Buffer, dst: vk::Buffer, size: u64) {
		if self
			.submit_and_wait(|cmd| unsafe {
				self.device.cmd_copy_buffer(
					cmd,
					src,
					dst,
					&[vk::BufferCopy {
						src_offset: 0,
						dst_offset: 0,
						size,
						..Default::default()
					}],
				);
			})
			.is_err()
		{
			// Said once: everything after this runs on the CPU, and a
			// silent loss once cost decode a quarter of its speed unnoticed.
			if !self.lost.swap(true, Ordering::SeqCst) {
				eprintln!("offload: Vulkan device lost; CPU only until it is rebuilt");
			}
		}
	}

	/// Allocate a command buffer, run `record`, submit, block, free.
	/// Returns `Err(vk::Result::ERROR_DEVICE_LOST)` if the device was lost so
	/// callers can transparently fall back to the CPU path.
	pub(crate) fn submit_and_wait<F: FnOnce(vk::CommandBuffer)>(
		&self,
		record: F,
	) -> Result<(), vk::Result> {
		let cmd = self.submit_nowait(record)?;
		self.wait_submitted(cmd)
	}

	/// Allocate a command buffer, run `record` and submit it on
	/// `self.fence` **without waiting**. Exactly one submission may be in
	/// flight on that fence: every `submit_nowait` must be paired with a
	/// `wait_submitted` before the next one. This is what lets the decode
	/// path run the GPU's share of a matmul while the CPU computes its own.
	pub(crate) fn submit_nowait<F: FnOnce(vk::CommandBuffer)>(
		&self,
		record: F,
	) -> Result<vk::CommandBuffer, vk::Result> {
		let cmd = self.record_cmd(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT, record);
		if let Err(e) = self.submit_cmd(cmd) {
			unsafe { self.device.free_command_buffers(self.cmd_pool, &[cmd]) };
			return Err(e);
		}
		Ok(cmd)
	}

	/// Allocate a command buffer from `cmd_pool` and record `record` into it.
	pub(crate) fn record_cmd<F: FnOnce(vk::CommandBuffer)>(
		&self,
		flags: vk::CommandBufferUsageFlags,
		record: F,
	) -> vk::CommandBuffer {
		let cmd = unsafe {
			self.device
				.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
					command_pool: self.cmd_pool,
					level: vk::CommandBufferLevel::PRIMARY,
					command_buffer_count: 1,
					..Default::default()
				})
				.expect("alloc cmd")[0]
		};
		unsafe {
			self.device
				.begin_command_buffer(
					cmd,
					&vk::CommandBufferBeginInfo {
						flags,
						..Default::default()
					},
				)
				.ok();
			record(cmd);
			self.device.end_command_buffer(cmd).ok();
		}
		cmd
	}

	/// The fence of prefill bank `bank` (0 = `self.fence`).
	pub(crate) fn bank_fence(&self, bank: usize) -> vk::Fence {
		if bank == 0 {
			self.fence
		} else {
			self.fence2
		}
	}

	/// Submit a recorded command buffer on `self.fence` (one in flight).
	pub(crate) fn submit_cmd(&self, cmd: vk::CommandBuffer) -> Result<(), vk::Result> {
		self.submit_cmd_on(cmd, self.fence)
	}

	/// [`Self::submit_cmd`] on bank `bank`'s fence ([`Self::bank_fence`]).
	pub(crate) fn submit_cmd_bank(
		&self,
		cmd: vk::CommandBuffer,
		bank: usize,
	) -> Result<(), vk::Result> {
		self.submit_cmd_on(cmd, self.bank_fence(bank))
	}

	/// [`Self::wait_submitted_spin`] for a submission on bank `bank`'s fence.
	pub(crate) fn wait_submitted_spin_bank(
		&self,
		cmd: vk::CommandBuffer,
		spin_us: u64,
		bank: usize,
	) -> Result<(), vk::Result> {
		let r = self.wait_fence_spin(self.bank_fence(bank), spin_us);
		unsafe { self.device.free_command_buffers(self.cmd_pool, &[cmd]) };
		r
	}

	/// Allocate, record and submit on bank `bank`'s fence without
	/// waiting (`submit_nowait` for the prefill banks).
	pub(crate) fn submit_nowait_bank<F: FnOnce(vk::CommandBuffer)>(
		&self,
		bank: usize,
		record: F,
	) -> Result<vk::CommandBuffer, vk::Result> {
		let cmd = self.record_cmd(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT, record);
		if let Err(e) = self.submit_cmd_on(cmd, self.bank_fence(bank)) {
			unsafe { self.device.free_command_buffers(self.cmd_pool, &[cmd]) };
			return Err(e);
		}
		Ok(cmd)
	}

	fn submit_cmd_on(&self, cmd: vk::CommandBuffer, fence: vk::Fence) -> Result<(), vk::Result> {
		unsafe {
			self.device.reset_fences(&[fence]).ok();
			let r = self.device.queue_submit(
				self.queue,
				&[vk::SubmitInfo {
					command_buffer_count: 1,
					p_command_buffers: &cmd,
					..Default::default()
				}],
				fence,
			);
			if let Err(e) = r {
				if e == vk::Result::ERROR_DEVICE_LOST {
					// Said once: everything after this runs on the CPU, and a
					// silent loss once cost decode a quarter of its speed unnoticed.
					if !self.lost.swap(true, Ordering::SeqCst) {
						eprintln!("offload: Vulkan device lost; CPU only until it is rebuilt");
					}
				}
				return Err(e);
			}
		}
		Ok(())
	}

	/// Submit a recorded command buffer with no fence: a batch whose
	/// completion is observed through a later submission's fence (fence
	/// signals cover every earlier submission to the queue).
	pub(crate) fn submit_cmd_nofence(&self, cmd: vk::CommandBuffer) -> Result<(), vk::Result> {
		let r = unsafe {
			self.device.queue_submit(
				self.queue,
				&[vk::SubmitInfo {
					command_buffer_count: 1,
					p_command_buffers: &cmd,
					..Default::default()
				}],
				vk::Fence::null(),
			)
		};
		if r == Err(vk::Result::ERROR_DEVICE_LOST) && !self.lost.swap(true, Ordering::SeqCst) {
			eprintln!("offload: Vulkan device lost; CPU only until it is rebuilt");
		}
		r
	}

	/// Wait for a prefill batch's command buffers (`PendingSeq::cmds`)
	/// and free them. With `follow`, the last one was submitted by the
	/// repack thread, so that thread's batch is collected first: until
	/// then `fence` may not even have been reset for it.
	pub(crate) fn wait_cmds(
		&self,
		cmds: &[vk::CommandBuffer],
		follow: bool,
		bank: usize,
	) -> Result<(), vk::Result> {
		let mut r = Ok(());
		if follow {
			// The other bank's `stream_bind` may already have collected
			// this batch's follow-up; draining again would wait for the
			// other bank's upload instead.
			let collected = self
				.stream
				.lock()
				.map(|p| p.follow_ok[bank].is_some())
				.unwrap_or(false);
			if !collected {
				self.stream_sync();
			}
			let ok = self
				.stream
				.lock()
				.map(|mut p| p.follow_ok[bank].take() == Some(true))
				.unwrap_or(false);
			if !ok {
				// Never submitted: nothing will signal `fence`. Anything
				// submitted before it still has to finish before its
				// buffers are freed.
				unsafe { self.device.queue_wait_idle(self.queue).ok() };
				r = Err(vk::Result::ERROR_DEVICE_LOST);
			}
		}
		if r.is_ok() {
			r = self.wait_fence_on(self.bank_fence(bank));
		}
		unsafe { self.device.free_command_buffers(self.cmd_pool, cmds) };
		self.stream_unpin(bank);
		r
	}

	/// Free every cached decode command buffer. Needs `&mut self`: nothing
	/// can be in flight, and every weight change goes through here.
	pub(crate) fn clear_decode_cmds(&mut self) {
		let cmds: Vec<vk::CommandBuffer> = self
			.decode_cmds
			.get_mut()
			.unwrap()
			.drain()
			.map(|(_, c)| c)
			.collect();
		if !cmds.is_empty() {
			unsafe { self.device.free_command_buffers(self.cmd_pool, &cmds) };
		}
	}

	/// Wait for the submission `submit_nowait` returned, then free it.
	pub(crate) fn wait_submitted(&self, cmd: vk::CommandBuffer) -> Result<(), vk::Result> {
		self.wait_submitted_spin(cmd, 200)
	}

	/// [`Self::wait_submitted`] spinning up to `spin_us` before it blocks.
	pub(crate) fn wait_submitted_spin(
		&self,
		cmd: vk::CommandBuffer,
		spin_us: u64,
	) -> Result<(), vk::Result> {
		let r = self.wait_fence_spin(self.fence, spin_us);
		unsafe { self.device.free_command_buffers(self.cmd_pool, &[cmd]) };
		r
	}

	/// Wait for the submission in flight on `self.fence`. A timeout is
	/// reported as a device loss.
	pub(crate) fn wait_fence(&self) -> Result<(), vk::Result> {
		self.wait_fence_on(self.fence)
	}

	pub(crate) fn wait_fence_on(&self, fence: vk::Fence) -> Result<(), vk::Result> {
		self.wait_fence_spin(fence, 200)
	}

	/// [`Self::wait_fence_on`] spinning up to `spin_us` before it blocks.
	pub(crate) fn wait_fence_spin(&self, fence: vk::Fence, spin_us: u64) -> Result<(), vk::Result> {
		unsafe {
			// Bounded wait: a shader that never terminates would otherwise
			// block the whole process forever with no way out. Treat a
			// timeout like a device loss so the caller falls back to the CPU
			// path instead of hanging.
			// Spin on the fence for a short while first: in the decode row
			// split the GPU's share is usually done by the time the CPU's
			// is, and a blocking `vkWaitForFences` pays a kernel wake-up
			// (~50 us on Windows) even for an already-signaled fence's
			// neighborhood. Only then fall back to the bounded wait.
			let spin_until = std::time::Instant::now() + std::time::Duration::from_micros(spin_us);
			let mut signaled = false;
			while std::time::Instant::now() < spin_until {
				if let Ok(true) = self.device.get_fence_status(fence) {
					signaled = true;
					break;
				}
				std::hint::spin_loop();
			}
			let w = if signaled {
				Ok(())
			} else {
				self.device
					.wait_for_fences(&[fence], true, FENCE_TIMEOUT_NS)
			};
			match w {
				Ok(()) => {}
				Err(vk::Result::ERROR_DEVICE_LOST) => {
					// Said once: everything after this runs on the CPU, and a
					// silent loss once cost decode a quarter of its speed unnoticed.
					if !self.lost.swap(true, Ordering::SeqCst) {
						eprintln!("offload: Vulkan device lost; CPU only until it is rebuilt");
					}
					return w;
				}
				Err(e) => {
					eprintln!(
							"offload: GPU submission did not complete within {} s ({e:?});                              disabling the GPU path and continuing on the CPU",
							FENCE_TIMEOUT_NS / 1_000_000_000
						);
					self.lost.store(true, Ordering::SeqCst);
					return Err(vk::Result::ERROR_DEVICE_LOST);
				}
			}
		}
		Ok(())
	}

	/// Record all dispatches, copy each `y` -> its readback buffer, and
	/// submit without waiting (`run_batch_end` blocks and reads results).
	/// `None` if the device was lost (so the caller can fall back to the
	/// CPU path).
	pub(crate) fn run_batch_begin(
		&self,
		items: &[Dispatch],
		bank: usize,
	) -> Option<vk::CommandBuffer> {
		if self.lost.load(Ordering::SeqCst) {
			return None;
		}
		self.submit_nowait_bank(bank, |cmd| self.record_batch(cmd, items))
			.ok()
	}

	/// The command buffer for a decode job set, recorded but not
	/// submitted: looked up in `decode_cmds` under `key` (recorded on first
	/// use). Returns it and whether it is a cached one, which must not be
	/// freed; `submit_prepared` submits it. Only valid for items whose every
	/// buffer is persistent per weight: `matmul_batch_begin`'s.
	pub(crate) fn prepare_batch_cached(
		&self,
		key: &[(usize, WeightId)],
		seq: u32,
		items: &[Dispatch],
	) -> (vk::CommandBuffer, bool) {
		let one_shot = || {
			self.record_cmd(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT, |cmd| {
				self.record_batch(cmd, items)
			})
		};
		if !self.cmd_cache {
			return (one_shot(), false);
		}
		let mut cache = self.decode_cmds.lock().unwrap();
		let key = (key.to_vec(), seq);
		if let Some(&c) = cache.get(&key) {
			return (c, true);
		}
		let c = self.record_cmd(vk::CommandBufferUsageFlags::empty(), |cmd| {
			self.record_batch(cmd, items)
		});
		cache.insert(key, c);
		(c, true)
	}

	/// Submit a `prepare_batch_cached` command buffer on `self.fence`. On
	/// failure a non-cached one is freed here.
	pub(crate) fn submit_prepared(&self, cmd: vk::CommandBuffer, cached: bool) -> Option<()> {
		if self.lost.load(Ordering::SeqCst) {
			if !cached {
				unsafe { self.device.free_command_buffers(self.cmd_pool, &[cmd]) };
			}
			return None;
		}
		if self.submit_cmd(cmd).is_err() {
			if !cached {
				unsafe { self.device.free_command_buffers(self.cmd_pool, &[cmd]) };
			}
			return None;
		}
		Some(())
	}

	/// Record `items`: stage each `x` (and params) into device-local
	/// memory, dispatch, and copy each `y` to its readback buffer.
	pub(crate) fn record_batch(&self, cmd: vk::CommandBuffer, items: &[Dispatch]) {
		self.record_batch_skip(cmd, items, &[])
	}

	/// `record_batch`, minus the staging copies of the `x` buffers in
	/// `skip_x`: already copied by an earlier submission on this queue,
	/// which the global barrier below orders before these dispatches.
	pub(crate) fn record_batch_skip(
		&self,
		cmd: vk::CommandBuffer,
		items: &[Dispatch],
		skip_x: &[vk::Buffer],
	) {
		{
			// Each job's activation lives in a host-visible staging buffer;
			// copy it into the device-local x_buf (which shaders can read),
			// then barrier TRANSFER->COMPUTE so the matmul sees it.
			let mut seen: std::collections::HashSet<vk::Buffer> = skip_x.iter().copied().collect();
			for d in items {
				if seen.insert(d.x_buf) {
					unsafe {
						self.device.cmd_copy_buffer(
							cmd,
							d.x_stg,
							d.x_buf,
							&[vk::BufferCopy {
								src_offset: 0,
								dst_offset: 0,
								size: d.x_len,
								..Default::default()
							}],
						);
					}
				}
			}
			let xbs: Vec<vk::Buffer> = items.iter().map(|d| d.x_buf).collect();
			let xbarriers: Vec<vk::BufferMemoryBarrier> = xbs
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
			// The global barrier covers the *streamed weight* buffers, which
			// were written by a copy submitted to the transfer queue. The
			// host fence wait in `stream_sync` (or, for a `stream::Follow`
			// batch, its semaphore) orders the two submissions;
			// this is the memory dependency that makes those writes visible
			// to the shader. It costs nothing extra: the per-buffer `x`
			// barriers below already use exactly these stage masks.
			let gbarrier = [vk::MemoryBarrier {
				src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
				dst_access_mask: vk::AccessFlags::SHADER_READ,
				..Default::default()
			}];
			unsafe {
				self.device.cmd_pipeline_barrier(
					cmd,
					vk::PipelineStageFlags::TRANSFER,
					vk::PipelineStageFlags::COMPUTE_SHADER,
					vk::DependencyFlags::empty(),
					&gbarrier,
					&xbarriers,
					&[],
				);
			}
			// Copy each job's params staging buffer into the device-local
			// uniform buffer (host-visible memory can't be bound as UNIFORM).
			let mut pseen: std::collections::HashSet<vk::Buffer> = Default::default();
			for d in items {
				if d.p_len > 0 && pseen.insert(d.p_buf) {
					unsafe {
						self.device.cmd_copy_buffer(
							cmd,
							d.p_stg,
							d.p_buf,
							&[vk::BufferCopy {
								src_offset: 0,
								dst_offset: 0,
								size: d.p_len,
								..Default::default()
							}],
						);
					}
				}
			}
			let pbs: Vec<vk::Buffer> = items
				.iter()
				.filter(|d| d.p_len > 0)
				.map(|d| d.p_buf)
				.collect();
			let pbarriers: Vec<vk::BufferMemoryBarrier> = pbs
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
			if !pbarriers.is_empty() {
				unsafe {
					self.device.cmd_pipeline_barrier(
						cmd,
						vk::PipelineStageFlags::TRANSFER,
						vk::PipelineStageFlags::COMPUTE_SHADER,
						vk::DependencyFlags::empty(),
						&[],
						&pbarriers,
						&[],
					);
				}
			}

			for d in items {
				unsafe {
					self.device
						.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, d.pipeline);
					self.device.cmd_bind_descriptor_sets(
						cmd,
						vk::PipelineBindPoint::COMPUTE,
						self.pipeline_layout,
						0,
						&[d.desc_set],
						&[],
					);
					self.device.cmd_dispatch(cmd, d.gx, d.gy, 1);
				}
			}

			// A dispatch whose `y` *is* its readback buffer (`y_direct`)
			// wrote the host memory itself: it needs only the
			// device-to-host memory dependency, no copy.
			if items.iter().any(|d| d.y_buf == d.rb_buf) {
				unsafe {
					self.device.cmd_pipeline_barrier(
						cmd,
						vk::PipelineStageFlags::COMPUTE_SHADER,
						vk::PipelineStageFlags::HOST,
						vk::DependencyFlags::empty(),
						&[vk::MemoryBarrier {
							src_access_mask: vk::AccessFlags::SHADER_WRITE,
							dst_access_mask: vk::AccessFlags::HOST_READ,
							..Default::default()
						}],
						&[],
						&[],
					);
				}
			}
			let items: Vec<&Dispatch> = items.iter().filter(|d| d.y_buf != d.rb_buf).collect();
			let ybarriers: Vec<vk::BufferMemoryBarrier> = items
				.iter()
				.map(|d| vk::BufferMemoryBarrier {
					src_access_mask: vk::AccessFlags::SHADER_WRITE,
					dst_access_mask: vk::AccessFlags::TRANSFER_READ,
					src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
					dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
					buffer: d.y_buf,
					offset: 0,
					size: vk::WHOLE_SIZE,
					..Default::default()
				})
				.collect();
			unsafe {
				self.device.cmd_pipeline_barrier(
					cmd,
					vk::PipelineStageFlags::COMPUTE_SHADER,
					vk::PipelineStageFlags::TRANSFER,
					vk::DependencyFlags::empty(),
					&[],
					&ybarriers,
					&[],
				);
				for d in items {
					self.device.cmd_copy_buffer(
						cmd,
						d.y_buf,
						d.rb_buf,
						&[vk::BufferCopy {
							src_offset: 0,
							dst_offset: 0,
							size: (d.out_dim * 4) as u64,
							..Default::default()
						}],
					);
				}
			}
		}
	}

	/// Wait for `run_batch_begin`'s submission and copy each result
	/// straight from its mapped readback buffer into a caller matrix:
	/// item `i` is `[rows x ncols]` and row `t` goes to
	/// `dst + (t * ld + col0)` floats. No intermediate `Vec`: a fresh one
	/// per result costs demand-zero page faults on top of the copy.
	///
	/// # Safety
	/// Each `(dst, ld, col0, ncols)` must describe writable memory for
	/// every row of its item, disjoint from the other items'.
	pub(crate) unsafe fn run_batch_end_into(
		&self,
		cmds: &[vk::CommandBuffer],
		follow: bool,
		bank: usize,
		items: &[Dispatch],
		dsts: &[(usize, usize, usize, usize)],
	) -> Option<()> {
		self.wait_cmds(cmds, follow, bank).ok()?;
		for (d, &(dst, ld, col0, ncols)) in items.iter().zip(dsts) {
			// Written in place by the GEMM (`SeqJob::dst`): nothing to copy.
			if d.rb_ptr.0.is_null() {
				continue;
			}
			let rows = d.out_dim / ncols;
			let src = d.rb_ptr.0 as usize;
			quant::par_for(rows, move |t| unsafe {
				std::ptr::copy_nonoverlapping(
					(src as *const f32).add(t * ncols),
					(dst as *mut f32).add(t * ld + col0),
					ncols,
				);
			});
		}
		Some(())
	}

	/// Wait for `run_batch_begin`'s submission and read every result back.
	pub(crate) fn run_batch_end(
		&self,
		cmd: vk::CommandBuffer,
		items: &[Dispatch],
		t_submit: std::time::Instant,
	) -> Option<Vec<Vec<f32>>> {
		self.wait_submitted(cmd).ok()?;
		self.read_batch(items, t_submit)
	}

	/// Read every result of a completed batch back into fresh `Vec`s.
	pub(crate) fn read_batch(
		&self,
		items: &[Dispatch],
		t_submit: std::time::Instant,
	) -> Option<Vec<Vec<f32>>> {
		let t_read = std::time::Instant::now();
		let out: Vec<Vec<f32>> = items
			.iter()
			.map(|d| read_back(d.rb_ptr.0 as *const f32, d.out_dim))
			.collect();
		if gpu_timing() {
			let bytes: usize = items.iter().map(|d| d.out_dim * 4).sum();
			eprintln!(
				"[GPUT] submit {:7.2} ms   readback {:7.2} ms ({:.1} MB, {:5.2} GB/s host copy)",
				(t_read - t_submit).as_secs_f64() * 1e3,
				t_read.elapsed().as_secs_f64() * 1e3,
				bytes as f64 / 1e6,
				bytes as f64 / t_read.elapsed().as_secs_f64() / 1e9,
			);
		}
		Some(out)
	}

	/// `(budget, usage)` in bytes of the largest device-local heap, from
	/// `VK_EXT_memory_budget`: what this process may use without WDDM
	/// demoting memory to system RAM, and what it uses now. `None` when the
	/// extension is missing.
	pub fn vram_budget(&self) -> Option<(u64, u64)> {
		if !self.has_budget_ext {
			return None;
		}
		let mut b = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
		let mut p2 = vk::PhysicalDeviceMemoryProperties2 {
			p_next: &mut b as *mut _ as *mut c_void,
			..Default::default()
		};
		unsafe {
			self.instance
				.get_physical_device_memory_properties2(self.physical, &mut p2)
		};
		let mp = &p2.memory_properties;
		(0..mp.memory_heap_count as usize)
			.filter(|&h| {
				mp.memory_heaps[h]
					.flags
					.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
			})
			.max_by_key(|&h| mp.memory_heaps[h].size)
			.map(|h| (b.heap_budget[h], b.heap_usage[h]))
	}

	/// Whether the in-flight submission on `self.fence` has completed
	/// (diagnostics for the decode row split).
	pub(crate) fn fence_signaled(&self) -> bool {
		matches!(
			unsafe { self.device.get_fence_status(self.fence) },
			Ok(true)
		)
	}

	/// A `VK_ERROR_DEVICE_LOST` or a fence timeout was seen: every GPU call
	/// now returns `None` (`OffloadManager::recover_gpu` rebuilds it).
	pub(crate) fn is_lost(&self) -> bool {
		self.lost.load(Ordering::SeqCst)
	}

	/// Mark the device lost, as a `VK_ERROR_DEVICE_LOST` would.
	#[cfg(test)]
	pub(crate) fn mark_lost(&self) {
		self.lost.store(true, Ordering::SeqCst);
	}

	pub fn has(&self, il: usize, id: WeightId) -> bool {
		self.weights.contains_key(&(il, id))
	}
}

/// Copy one matmul result out of its mapped readback buffer.
///
/// This is the D2H mirror of `stage_copy`, and it was the largest host cost left
/// in `matmul_batch_seq` once the upload moved to the transfer queue: 285 MB on
/// the `FfnUp`+`FfnGate` call at `seq = 2048`, 67.6 ms, ~10 % of the FFN block.
///
/// `HOST_CACHED` already fixed the pathological version (0.21 GB/s out of
/// write-combined memory), but 4.2 GB/s is still far under this machine's
/// 60.8 GB/s DRAM ceiling, and the reason is the *destination*: `slice.to_vec()`
/// allocates a fresh 142 MB `Vec`, so the copy is demand-zero page faults at 4 KB
/// granularity, serialized on one core. Splitting by megabyte faults and copies
/// on the shared pool instead. Small results stay inline, where the fork-join
/// would cost more than the copy.
fn read_back(src: *const f32, n: usize) -> Vec<f32> {
	/// 1 MB, matching `stage_copy`'s split.
	const PAR_MIN: usize = (1 << 20) / 4;
	if n < PAR_MIN {
		return unsafe { std::slice::from_raw_parts(src, n) }.to_vec();
	}
	// Every one of the `n` elements is written by the `par_for` below before
	// `out` is read or returned, and `par_for` joins its workers.
	let mut out = quant::uninit_vec(n);
	let (d, s) = (out.as_mut_ptr() as usize, src as usize);
	quant::par_for(n.div_ceil(PAR_MIN), move |c| {
		let off = c * PAR_MIN;
		let len = PAR_MIN.min(n - off);
		unsafe {
			ptr::copy_nonoverlapping((s as *const f32).add(off), (d as *mut f32).add(off), len)
		};
	});
	out
}
