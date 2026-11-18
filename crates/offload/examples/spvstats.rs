//! Compile SPIR-V compute shaders on the real device and print the driver's
//! statistics for them (`VK_KHR_pipeline_executable_properties`): register
//! count, shared memory, spills; whatever the driver reports. Timings alone
//! cannot tell "this variant spilled" from "this variant is slower"; on the
//! GTX 1060 a kernel above 128 registers per thread also halves the
//! workgroups an SM holds for a 256-thread workgroup.
//!
//! ```text
//! cargo run --release -p offload --features vulkan --example spvstats -- \
//!     [--bind ssss] [--device NAME] FILE.spv [FILE.spv ...]
//! ```
//!
//! `--bind` gives the descriptor type of bindings 0, 1, ... of set 0 (`s`
//! storage buffer, `u` uniform buffer; default `ssssssss`): the pipeline
//! layout must match what the shader declares (the GEMMs are `sssu`). Push
//! constants of up to 128 bytes are always declared.

use ash::vk;
use std::ffi::{c_char, CStr};

fn cstr(b: &[c_char]) -> String {
	unsafe { CStr::from_ptr(b.as_ptr()) }
		.to_string_lossy()
		.into_owned()
}

fn main() {
	let mut bind = "ssssssss".to_string();
	let mut want_dev: Option<String> = None;
	let mut files = Vec::new();
	let mut args = std::env::args().skip(1);
	while let Some(a) = args.next() {
		match a.as_str() {
			"--bind" => bind = args.next().expect("--bind TYPES"),
			"--device" => want_dev = Some(args.next().expect("--device NAME").to_lowercase()),
			_ => files.push(a),
		}
	}
	if files.is_empty() {
		eprintln!("usage: spvstats [--bind ssss] [--device NAME] FILE.spv ...");
		std::process::exit(2);
	}
	unsafe {
		let entry = ash::Entry::load().expect("vulkan-1");
		let app = vk::ApplicationInfo {
			api_version: vk::make_api_version(0, 1, 1, 0),
			..Default::default()
		};
		let instance = entry
			.create_instance(
				&vk::InstanceCreateInfo {
					p_application_info: &app,
					..Default::default()
				},
				None,
			)
			.expect("instance");
		let pdevs = instance.enumerate_physical_devices().expect("devices");
		let pick = pdevs
			.iter()
			.copied()
			.find(|&p| {
				let pr = instance.get_physical_device_properties(p);
				match &want_dev {
					Some(n) => cstr(&pr.device_name).to_lowercase().contains(n.as_str()),
					None => pr.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
				}
			})
			.or_else(|| pdevs.first().copied())
			.expect("no Vulkan device");
		let props = instance.get_physical_device_properties(pick);
		println!("device: {}", cstr(&props.device_name));
		let qf = instance
			.get_physical_device_queue_family_properties(pick)
			.iter()
			.position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
			.expect("compute queue") as u32;
		let prio = [1.0f32];
		let qci = vk::DeviceQueueCreateInfo {
			queue_family_index: qf,
			queue_count: 1,
			p_queue_priorities: prio.as_ptr(),
			..Default::default()
		};
		let exts = [ash::khr::pipeline_executable_properties::NAME.as_ptr()];
		let mut feat = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR {
			pipeline_executable_info: vk::TRUE,
			..Default::default()
		};
		let dci = vk::DeviceCreateInfo {
			p_next: &mut feat as *mut _ as *mut std::ffi::c_void,
			queue_create_info_count: 1,
			p_queue_create_infos: &qci,
			enabled_extension_count: exts.len() as u32,
			pp_enabled_extension_names: exts.as_ptr(),
			..Default::default()
		};
		let device = instance
			.create_device(pick, &dci, None)
			.expect("device (does the driver offer VK_KHR_pipeline_executable_properties?)");
		let pep = ash::khr::pipeline_executable_properties::Device::new(&instance, &device);

		let bindings: Vec<vk::DescriptorSetLayoutBinding> = bind
			.chars()
			.enumerate()
			.map(|(i, c)| vk::DescriptorSetLayoutBinding {
				binding: i as u32,
				descriptor_type: if c == 'u' {
					vk::DescriptorType::UNIFORM_BUFFER
				} else {
					vk::DescriptorType::STORAGE_BUFFER
				},
				descriptor_count: 1,
				stage_flags: vk::ShaderStageFlags::COMPUTE,
				..Default::default()
			})
			.collect();
		let dsl = device
			.create_descriptor_set_layout(
				&vk::DescriptorSetLayoutCreateInfo {
					binding_count: bindings.len() as u32,
					p_bindings: bindings.as_ptr(),
					..Default::default()
				},
				None,
			)
			.expect("set layout");
		let pcr = vk::PushConstantRange {
			stage_flags: vk::ShaderStageFlags::COMPUTE,
			offset: 0,
			size: 128,
		};
		let pl = device
			.create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo {
					set_layout_count: 1,
					p_set_layouts: &dsl,
					push_constant_range_count: 1,
					p_push_constant_ranges: &pcr,
					..Default::default()
				},
				None,
			)
			.expect("pipeline layout");
		let main = c"main";
		for f in &files {
			let bytes = std::fs::read(f).unwrap_or_else(|e| panic!("{f}: {e}"));
			let words: Vec<u32> = bytes
				.chunks_exact(4)
				.map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
				.collect();
			let sm = device
				.create_shader_module(
					&vk::ShaderModuleCreateInfo {
						code_size: words.len() * 4,
						p_code: words.as_ptr(),
						..Default::default()
					},
					None,
				)
				.expect("shader module");
			let ci = vk::ComputePipelineCreateInfo {
				flags: vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR,
				stage: vk::PipelineShaderStageCreateInfo {
					stage: vk::ShaderStageFlags::COMPUTE,
					module: sm,
					p_name: main.as_ptr(),
					..Default::default()
				},
				layout: pl,
				..Default::default()
			};
			let pipe = match device.create_compute_pipelines(vk::PipelineCache::null(), &[ci], None)
			{
				Ok(p) => p[0],
				Err((_, e)) => {
					println!("{f}: pipeline creation failed: {e:?}");
					continue;
				}
			};
			println!("{f}:");
			let execs = pep
				.get_pipeline_executable_properties(&vk::PipelineInfoKHR {
					pipeline: pipe,
					..Default::default()
				})
				.expect("executable properties");
			for (i, ex) in execs.iter().enumerate() {
				println!("  [{i}] {} (subgroup {})", cstr(&ex.name), ex.subgroup_size);
				let stats = pep
					.get_pipeline_executable_statistics(&vk::PipelineExecutableInfoKHR {
						pipeline: pipe,
						executable_index: i as u32,
						..Default::default()
					})
					.expect("statistics");
				for s in &stats {
					let v = match s.format {
						vk::PipelineExecutableStatisticFormatKHR::BOOL32 => {
							format!("{}", s.value.b32 != 0)
						}
						vk::PipelineExecutableStatisticFormatKHR::INT64 => {
							format!("{}", s.value.i64)
						}
						vk::PipelineExecutableStatisticFormatKHR::UINT64 => {
							format!("{}", s.value.u64)
						}
						_ => format!("{}", s.value.f64),
					};
					println!("      {:<32} {v}", cstr(&s.name));
				}
			}
			device.destroy_pipeline(pipe, None);
			device.destroy_shader_module(sm, None);
		}
		device.destroy_pipeline_layout(pl, None);
		device.destroy_descriptor_set_layout(dsl, None);
		device.destroy_device(None);
		instance.destroy_instance(None);
	}
}
