//! Every Vulkan adapter's memory heaps with their `VK_EXT_memory_budget`
//! budget and this process's usage, optionally after allocating and
//! touching `MB` of host-visible memory on the adapter whose name contains
//! `PICK` (held `SECS` seconds): what WDDM lets one more process map next
//! to a running server.
//!
//!     heapprobe [PICK MB [SECS]]
//!
//!     cargo run --release -p offload --features vulkan --example heapprobe -- intel 4096 30
use ash::vk;

fn heaps(inst: &ash::Instance, pd: vk::PhysicalDevice) {
	let mut b = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
	let mut p2 = vk::PhysicalDeviceMemoryProperties2 {
		p_next: &mut b as *mut _ as *mut std::ffi::c_void,
		..Default::default()
	};
	unsafe { inst.get_physical_device_memory_properties2(pd, &mut p2) };
	let mp = &p2.memory_properties;
	for h in 0..mp.memory_heap_count as usize {
		let g = |v: u64| v as f64 / (1u64 << 30) as f64;
		println!(
			"  heap {h}: size {:.2} GiB, budget {:.2} GiB, usage {:.2} GiB{}",
			g(mp.memory_heaps[h].size),
			g(b.heap_budget[h]),
			g(b.heap_usage[h]),
			if mp.memory_heaps[h]
				.flags
				.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
			{
				" (device local)"
			} else {
				""
			}
		);
	}
}

fn main() {
	let a: Vec<String> = std::env::args().collect();
	let pick = a.get(1).map(|s| s.to_lowercase());
	let mb: u64 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
	let secs: u64 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
	unsafe {
		let entry = ash::Entry::load().unwrap();
		let app = vk::ApplicationInfo {
			api_version: vk::make_api_version(0, 1, 1, 0),
			..Default::default()
		};
		let inst = entry
			.create_instance(
				&vk::InstanceCreateInfo {
					p_application_info: &app,
					..Default::default()
				},
				None,
			)
			.unwrap();
		let mut target = None;
		for pd in inst.enumerate_physical_devices().unwrap() {
			let p = inst.get_physical_device_properties(pd);
			let name = std::ffi::CStr::from_ptr(p.device_name.as_ptr())
				.to_string_lossy()
				.into_owned();
			println!("{name}");
			heaps(&inst, pd);
			if pick
				.as_ref()
				.is_some_and(|k| name.to_lowercase().contains(k.as_str()))
			{
				target = Some(pd);
			}
		}
		let (Some(pd), true) = (target, mb > 0) else {
			return;
		};
		let qf = inst
			.get_physical_device_queue_family_properties(pd)
			.iter()
			.position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
			.unwrap() as u32;
		let prio = [1.0f32];
		let qci = vk::DeviceQueueCreateInfo {
			queue_family_index: qf,
			queue_count: 1,
			p_queue_priorities: prio.as_ptr(),
			..Default::default()
		};
		let ext = [c"VK_EXT_memory_budget".as_ptr()];
		let dev = inst
			.create_device(
				pd,
				&vk::DeviceCreateInfo {
					queue_create_info_count: 1,
					p_queue_create_infos: &qci,
					enabled_extension_count: 1,
					pp_enabled_extension_names: ext.as_ptr(),
					..Default::default()
				},
				None,
			)
			.unwrap();
		let mp = inst.get_physical_device_memory_properties(pd);
		let want = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
		let ty = (0..mp.memory_type_count)
			.find(|&i| mp.memory_types[i as usize].property_flags.contains(want))
			.unwrap();
		const CHUNK: u64 = 256 << 20;
		let mut held = Vec::new();
		let mut done = 0u64;
		while done < mb << 20 {
			match dev.allocate_memory(
				&vk::MemoryAllocateInfo {
					allocation_size: CHUNK,
					memory_type_index: ty,
					..Default::default()
				},
				None,
			) {
				Ok(m) => {
					let p = dev
						.map_memory(m, 0, CHUNK, vk::MemoryMapFlags::empty())
						.unwrap() as *mut u8;
					std::ptr::write_bytes(p, 1, CHUNK as usize);
					held.push(m);
					done += CHUNK;
				}
				Err(e) => {
					println!("allocation failed at {} MB: {e}", done >> 20);
					break;
				}
			}
		}
		println!("holding {} MB for {secs} s", done >> 20);
		heaps(&inst, pd);
		std::thread::sleep(std::time::Duration::from_secs(secs));
		for m in held {
			dev.free_memory(m, None);
		}
		dev.destroy_device(None);
	}
}
