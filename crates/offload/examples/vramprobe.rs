//! Raw-Vulkan repro for the decode split's device loss (see
//! `GpuDevice::create_small_gpu_buffer`). Fills VRAM with `chunk` MB
//! device-local buffers up to `total` MB, each by a staging copy plus a wait;
//! the first `skip` MB are allocated but never written.
//!
//!     vramprobe <total MB> <chunk MB> <skip MB> [fresh|stale] [extras] [kinds] [sub]
//!
//! `fresh` uses a new staging buffer per chunk (as `GpuDevice::upload` does).
//! `extras` adds per chunk the small buffers one decode-split weight carries,
//! `kinds` picks which by index (0 x, 2 y, 4 params: device-local; 1, 3, 5:
//! their mapped host-side twins), and `sub` binds the device-local ones into
//! one shared allocation. On the GTX 1060:
//! `4600 8 0 fresh extras 024` -> `FAIL at ~2450 MB: ERROR_DEVICE_LOST`
//! (4 of 4), `4000 8 0 fresh extras 024 sub` -> `OK 4000` (3 of 3).
//!
//!     cargo run --release -p offload --features vulkan --example vramprobe -- 4600 8 0 fresh extras 024
use ash::vk;
fn main() {
	let a: Vec<String> = std::env::args().collect();
	let total: u64 = a[1].parse().unwrap();
	let chunk: u64 = a[2].parse().unwrap();
	let skip: u64 = a[3].parse().unwrap();
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
			.position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
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
		let find = |bits: u32, f: vk::MemoryPropertyFlags| {
			(0..mp.memory_type_count)
				.find(|&i| {
					bits & (1 << i) != 0 && mp.memory_types[i as usize].property_flags.contains(f)
				})
				.unwrap()
		};
		let mk = |size: u64, usage: vk::BufferUsageFlags, f: vk::MemoryPropertyFlags| {
			let b = dev
				.create_buffer(
					&vk::BufferCreateInfo {
						size,
						usage,
						..Default::default()
					},
					None,
				)
				.unwrap();
			let r = dev.get_buffer_memory_requirements(b);
			let m = dev
				.allocate_memory(
					&vk::MemoryAllocateInfo {
						allocation_size: r.size,
						memory_type_index: find(r.memory_type_bits, f),
						..Default::default()
					},
					None,
				)
				.unwrap();
			dev.bind_buffer_memory(b, m, 0).unwrap();
			(b, m)
		};
		let cs = chunk << 20;
		let (sb, sm) = mk(
			cs,
			vk::BufferUsageFlags::TRANSFER_SRC,
			vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
		);
		let p = dev
			.map_memory(sm, 0, cs, vk::MemoryMapFlags::empty())
			.unwrap() as *mut u8;
		std::ptr::write_bytes(p, 0x11, cs as usize);
		let pool = dev
			.create_command_pool(
				&vk::CommandPoolCreateInfo {
					queue_family_index: qf,
					flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
					..Default::default()
				},
				None,
			)
			.unwrap();
		let cmd = dev
			.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
				command_pool: pool,
				level: vk::CommandBufferLevel::PRIMARY,
				command_buffer_count: 1,
				..Default::default()
			})
			.unwrap()[0];
		let fence = dev
			.create_fence(&vk::FenceCreateInfo::default(), None)
			.unwrap();
		let mut keep = Vec::new();
		let sub = a.get(7).is_some_and(|v| v == "sub");
		let arena = if sub {
			let tb = dev
				.create_buffer(
					&vk::BufferCreateInfo {
						size: 1 << 20,
						usage: vk::BufferUsageFlags::STORAGE_BUFFER
							| vk::BufferUsageFlags::TRANSFER_DST
							| vk::BufferUsageFlags::TRANSFER_SRC,
						..Default::default()
					},
					None,
				)
				.unwrap();
			let r = dev.get_buffer_memory_requirements(tb);
			Some(
				dev.allocate_memory(
					&vk::MemoryAllocateInfo {
						allocation_size: 1 << 30,
						memory_type_index: find(
							r.memory_type_bits,
							vk::MemoryPropertyFlags::DEVICE_LOCAL,
						),
						..Default::default()
					},
					None,
				)
				.unwrap(),
			)
		} else {
			None
		};
		let mut aoff = 0u64;
		let mut mb = 0u64;
		while mb + chunk <= total {
			let (b, m) = mk(
				cs,
				vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
				vk::MemoryPropertyFlags::DEVICE_LOCAL,
			);
			keep.push((b, m));
			mb += chunk;
			if a.get(5).is_some_and(|v| v == "extras") {
				let hv =
					vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
				for (sz, f, mapit) in [
					(163840u64, vk::MemoryPropertyFlags::DEVICE_LOCAL, false),
					(163840, hv, true),
					(557056, vk::MemoryPropertyFlags::DEVICE_LOCAL, false),
					(557056, hv | vk::MemoryPropertyFlags::HOST_CACHED, true),
					(16, vk::MemoryPropertyFlags::DEVICE_LOCAL, false),
					(16, hv, true),
				]
				.into_iter()
				.enumerate()
				.filter(|(i, _)| a.get(6).is_none_or(|m| m.contains(&i.to_string())))
				.map(|(_, t)| t)
				{
					let usage = vk::BufferUsageFlags::STORAGE_BUFFER
						| vk::BufferUsageFlags::TRANSFER_DST
						| vk::BufferUsageFlags::TRANSFER_SRC;
					if let (Some(am), false) = (arena, mapit) {
						let b2 = dev
							.create_buffer(
								&vk::BufferCreateInfo {
									size: sz,
									usage,
									..Default::default()
								},
								None,
							)
							.unwrap();
						let r = dev.get_buffer_memory_requirements(b2);
						aoff = aoff.div_ceil(r.alignment) * r.alignment;
						dev.bind_buffer_memory(b2, am, aoff).unwrap();
						aoff += r.size;
						continue;
					}
					let (b2, m2) = mk(sz, usage, f);
					if mapit {
						dev.map_memory(m2, 0, sz, vk::MemoryMapFlags::empty())
							.unwrap();
					}
					keep.push((b2, m2));
				}
			}
			if mb <= skip {
				continue;
			}
			let fresh = a.get(4).is_some_and(|v| v == "fresh");
			let (sb, sm2) = if fresh {
				let (b2, m2) = mk(
					cs,
					vk::BufferUsageFlags::TRANSFER_SRC,
					vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
				);
				let p2 = dev
					.map_memory(m2, 0, cs, vk::MemoryMapFlags::empty())
					.unwrap() as *mut u8;
				std::ptr::write_bytes(p2, 0x11, cs as usize);
				(b2, Some(m2))
			} else {
				(sb, None)
			};
			dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
				.unwrap();
			dev.cmd_copy_buffer(
				cmd,
				sb,
				b,
				&[vk::BufferCopy {
					src_offset: 0,
					dst_offset: 0,
					size: cs,
				}],
			);
			dev.end_command_buffer(cmd).unwrap();
			dev.reset_fences(&[fence]).unwrap();
			let r = dev
				.queue_submit(
					q,
					&[vk::SubmitInfo {
						command_buffer_count: 1,
						p_command_buffers: &cmd,
						..Default::default()
					}],
					fence,
				)
				.and_then(|_| dev.wait_for_fences(&[fence], true, u64::MAX));
			if let Err(e) = r {
				println!("FAIL at {mb} MB: {e:?}");
				return;
			}
			if let Some(m2) = sm2 {
				dev.destroy_buffer(sb, None);
				dev.free_memory(m2, None);
			}
		}
		println!("OK {mb} MB ({} chunks)", keep.len());
	}
}
