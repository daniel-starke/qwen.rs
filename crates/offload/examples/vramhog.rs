//! Another application claiming VRAM: allocates `MB` of device-local memory
//! in 64 MB buffers, writes it once, holds it for `SECS` seconds, then exits.
//! The repro for a server losing its GPU path to a competing process.
//!
//!     vramhog <MB> <SECS> [active]
//!
//! `active` rewrites every buffer once a second, as a game or another
//! inference process would, so WDDM keeps the hog's memory resident and
//! evicts the other process's instead.
//!
//!     cargo run --release -p offload --features vulkan --example vramhog -- 3000 120
use ash::vk;
fn main() {
	let a: Vec<String> = std::env::args().collect();
	let total: u64 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(3000);
	let secs: u64 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(60);
	let active = a.get(3).is_some_and(|s| s == "active");
	const CHUNK: u64 = 64;
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
		let pd = inst
			.enumerate_physical_devices()
			.unwrap()
			.into_iter()
			.find(|&p| inst.get_physical_device_properties(p).vendor_id == 0x10de)
			.unwrap();
		let qf = inst
			.get_physical_device_queue_family_properties(pd)
			.iter()
			.position(|q| q.queue_flags.contains(vk::QueueFlags::TRANSFER))
			.unwrap() as u32;
		let prio = [1.0f32];
		let qci = vk::DeviceQueueCreateInfo {
			queue_family_index: qf,
			queue_count: 1,
			p_queue_priorities: prio.as_ptr(),
			..Default::default()
		};
		let dev = inst
			.create_device(
				pd,
				&vk::DeviceCreateInfo {
					queue_create_info_count: 1,
					p_queue_create_infos: &qci,
					..Default::default()
				},
				None,
			)
			.unwrap();
		let q = dev.get_device_queue(qf, 0);
		let mp = inst.get_physical_device_memory_properties(pd);
		let cpool = dev
			.create_command_pool(
				&vk::CommandPoolCreateInfo {
					queue_family_index: qf,
					..Default::default()
				},
				None,
			)
			.unwrap();
		let mut held = 0u64;
		let mut bufs = Vec::new();
		while held < total {
			let size = CHUNK << 20;
			let b = dev
				.create_buffer(
					&vk::BufferCreateInfo {
						size,
						usage: vk::BufferUsageFlags::TRANSFER_DST,
						..Default::default()
					},
					None,
				)
				.unwrap();
			let r = dev.get_buffer_memory_requirements(b);
			let ti = (0..mp.memory_type_count)
				.find(|&i| {
					r.memory_type_bits & (1 << i) != 0
						&& mp.memory_types[i as usize]
							.property_flags
							.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
				})
				.unwrap();
			let m = match dev.allocate_memory(
				&vk::MemoryAllocateInfo {
					allocation_size: r.size,
					memory_type_index: ti,
					..Default::default()
				},
				None,
			) {
				Ok(m) => m,
				Err(e) => {
					eprintln!("vramhog: allocation failed at {held} MB: {e:?}");
					dev.destroy_buffer(b, None);
					break;
				}
			};
			dev.bind_buffer_memory(b, m, 0).unwrap();
			// Touch it so the driver really backs it with VRAM.
			let cmd = dev
				.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
					command_pool: cpool,
					level: vk::CommandBufferLevel::PRIMARY,
					command_buffer_count: 1,
					..Default::default()
				})
				.unwrap()[0];
			dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
				.unwrap();
			dev.cmd_fill_buffer(cmd, b, 0, vk::WHOLE_SIZE, 0x5a5a5a5a);
			dev.end_command_buffer(cmd).unwrap();
			dev.queue_submit(
				q,
				&[vk::SubmitInfo {
					command_buffer_count: 1,
					p_command_buffers: &cmd,
					..Default::default()
				}],
				vk::Fence::null(),
			)
			.unwrap();
			dev.queue_wait_idle(q).unwrap();
			dev.free_command_buffers(cpool, &[cmd]);
			bufs.push((b, m));
			held += CHUNK;
		}
		eprintln!("vramhog: holding {held} MB for {secs} s");
		let end = std::time::Instant::now() + std::time::Duration::from_secs(secs);
		while std::time::Instant::now() < end {
			if active {
				let cmd = dev
					.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
						command_pool: cpool,
						level: vk::CommandBufferLevel::PRIMARY,
						command_buffer_count: 1,
						..Default::default()
					})
					.unwrap()[0];
				dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
					.unwrap();
				for &(b, _) in &bufs {
					dev.cmd_fill_buffer(cmd, b, 0, vk::WHOLE_SIZE, 0x5a5a5a5a);
				}
				dev.end_command_buffer(cmd).unwrap();
				if let Err(e) = dev
					.queue_submit(
						q,
						&[vk::SubmitInfo {
							command_buffer_count: 1,
							p_command_buffers: &cmd,
							..Default::default()
						}],
						vk::Fence::null(),
					)
					.and_then(|_| dev.queue_wait_idle(q))
				{
					eprintln!("vramhog: {e:?}");
					break;
				}
				dev.free_command_buffers(cpool, &[cmd]);
			}
			std::thread::sleep(std::time::Duration::from_secs(1));
		}
		for (b, m) in bufs {
			dev.destroy_buffer(b, None);
			dev.free_memory(m, None);
		}
		dev.destroy_command_pool(cpool, None);
		dev.destroy_device(None);
		inst.destroy_instance(None);
		eprintln!("vramhog: released");
	}
}
