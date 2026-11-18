use super::*;

/// One reusable host-visible buffer, allocated once and kept mapped.
pub(crate) struct HostSlot {
	buf: vk::Buffer,
	mem: vk::DeviceMemory,
	ptr: MappedPtr,
	cap: u64,
}

impl HostSlot {
	const EMPTY: HostSlot = HostSlot {
		buf: vk::Buffer::null(),
		mem: vk::DeviceMemory::null(),
		ptr: MappedPtr(ptr::null_mut()),
		cap: 0,
	};
}

/// The host-visible buffers `matmul_batch_seq` used to create and destroy on
/// every call.
///
/// Those are not small: at `seq = 2048` the `FfnUp`+`FfnGate` call maps 285 MB
/// of readback and 11 MB of packed activation, and the `FfnDown` call maps
/// 42 MB and 36 MB. Creating, mapping, unmapping and destroying them cost
/// **~30 ms per FFN block** (`alloc` 8-16 ms, `free` ~12 ms, plus the
/// `vkCreateBuffer`/`vkMapMemory` half of `stage`) against a block that had
/// come down to ~620 ms. It is the same problem `StreamSlot` already solves for
/// the weight staging buffers, and the same fix: allocate once, reuse, grow only
/// when a call is wider or longer than any before it.
///
/// Only **host** memory is pooled. The device-local `x` and `y` buffers are
/// still per call: VRAM is ~96 % committed at `--n-gpu-layers 22`, and a pool
/// holds every slot's high-water mark at once, where per-call allocation holds
/// only the shape in flight.
///
/// Slots are indexed positionally (readback slot `i` always serves job `i`,
/// staging slot `g` always serves activation group `g`) so a call's slots are
/// distinct from each other, which is what matters: the whole batch is in flight
/// in one submission. Reuse *across* calls is safe because `run_batch` blocks on
/// the fence and copies every result out before returning.
pub(crate) struct HostPool {
	/// One per job, holding that job's `seq * out_dim * 4` result.
	rb: Vec<HostSlot>,
	/// One per shared-`x` group, holding that group's packed activation.
	xs: Vec<HostSlot>,
	/// Device-local `y` (one per job) and `x` (one per group). These are VRAM,
	/// so unlike the host families they are given up for good the first time an
	/// allocation fails: see `dev_off`.
	yd: Vec<HostSlot>,
	xd: Vec<HostSlot>,
	/// Per-job uniform params, staging and device-local. 16 bytes each, but
	/// created per call they were two `vkAllocateMemory` + two
	/// `vkFreeMemory` per job: ~0.5 ms of a short-chunk split call's ~1.5 ms
	/// of host overhead, serial with the CPU's share.
	ps: Vec<HostSlot>,
	pd: Vec<HostSlot>,
	/// Set once a device-local slot could not be allocated. The card is ~96 %
	/// committed at `--n-gpu-layers 22`, and a pool holds every slot's
	/// high-water mark at once where per-call allocation holds only the shape in
	/// flight, so on a tight card the right answer is to hand the VRAM back and
	/// go back to allocating per call, not to retry (and fall back) every call.
	dev_off: bool,
	/// Slot (re)allocations, for the `QWEN_GPU_TIMING` line.
	pub(crate) allocs: u64,
}

impl HostPool {
	pub(crate) fn new() -> HostPool {
		HostPool {
			rb: Vec::new(),
			xs: Vec::new(),
			yd: Vec::new(),
			xd: Vec::new(),
			ps: Vec::new(),
			pd: Vec::new(),
			dev_off: false,
			allocs: 0,
		}
	}
}

/// Which of the two slot families a request is for.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum HostUse {
	/// D2H: `HOST_CACHED`, so the host `memcpy` *out* of it is not uncached.
	Readback,
	/// H2U: plain host-visible, written once and read by the copy engine.
	Staging,
	/// Device-local `y`, written by the shader and read by the D2H copy.
	DeviceY,
	/// Device-local `x`, written by the H2D copy and read by the shader.
	DeviceX,
	/// A job's 16-byte uniform params: the host-side staging copy ...
	ParamsStg,
	/// ... and the device-local uniform buffer the shader binds.
	ParamsDev,
}

impl HostUse {
	fn device_local(self) -> bool {
		matches!(
			self,
			HostUse::DeviceY | HostUse::DeviceX | HostUse::ParamsDev
		)
	}
}

impl GpuDevice {
	/// `QWEN_GPU_NOPOOL=1` goes back to allocating the host buffers per
	/// call, so the pool can be A/B'd on one binary. Read once.
	pub(crate) fn host_pool_disabled() -> bool {
		static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
		*OFF.get_or_init(|| std::env::var("QWEN_GPU_NOPOOL").is_ok())
	}

	/// Slot `i` of the given family, at least `size` bytes, mapped.
	///
	/// Returns `None` when the pool is disabled, so the caller falls back to
	/// its own per-call allocation and the two paths stay one code path.
	pub(crate) fn host_slot(
		&self,
		which: HostUse,
		i: usize,
		size: u64,
	) -> Option<(vk::Buffer, *mut c_void)> {
		if Self::host_pool_disabled() {
			return None;
		}
		let mut pool = self.host.lock().ok()?;
		let pool = &mut *pool;
		if which.device_local() && pool.dev_off {
			return None;
		}
		let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
		let dev = vk::MemoryPropertyFlags::DEVICE_LOCAL;
		let (slots, usage, props) = match which {
			HostUse::Readback => (
				&mut pool.rb,
				vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
				host,
			),
			HostUse::Staging => (&mut pool.xs, vk::BufferUsageFlags::TRANSFER_SRC, host),
			HostUse::DeviceY => (
				&mut pool.yd,
				vk::BufferUsageFlags::STORAGE_BUFFER
					| vk::BufferUsageFlags::TRANSFER_SRC
					| vk::BufferUsageFlags::TRANSFER_DST,
				dev,
			),
			HostUse::DeviceX => (
				&mut pool.xd,
				vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
				dev,
			),
			HostUse::ParamsStg => (&mut pool.ps, vk::BufferUsageFlags::TRANSFER_SRC, host),
			HostUse::ParamsDev => (
				&mut pool.pd,
				vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
				dev,
			),
		};
		while slots.len() <= i {
			slots.push(HostSlot::EMPTY);
		}
		if slots[i].cap < size {
			self.free_host_slot(&mut slots[i]);
			let made = match which {
				HostUse::Readback => self.try_create_readback_buffer(size),
				_ => self.try_create_gpu_buffer(size, usage, props),
			};
			let (buf, mem) = match made {
				Some(v) => v,
				None => {
					if which.device_local() {
						// Give the VRAM back and stop trying: this card
						// cannot hold the pool's high-water mark on top of
						// the weights, and per-call allocation can.
						eprintln!(
								"offload: VRAM too tight for the device-local buffer pool; allocating per call"
							);
						pool.dev_off = true;
						for s in pool
							.yd
							.iter_mut()
							.chain(pool.xd.iter_mut())
							.chain(pool.pd.iter_mut())
						{
							self.free_host_slot(s);
						}
						pool.yd.clear();
						pool.xd.clear();
						pool.pd.clear();
					}
					return None;
				}
			};
			// Device-local memory is not mappable, and nothing reads these
			// two families from the host.
			let ptr = if which.device_local() {
				MappedPtr(ptr::null_mut())
			} else {
				MappedPtr(self.map(mem, size as usize))
			};
			slots[i] = HostSlot {
				buf,
				mem,
				ptr,
				cap: size,
			};
			pool.allocs += 1;
		}
		Some((slots[i].buf, slots[i].ptr.0))
	}

	fn free_host_slot(&self, s: &mut HostSlot) {
		if s.cap == 0 {
			return;
		}
		unsafe {
			if !s.ptr.0.is_null() {
				self.device.unmap_memory(s.mem);
			}
			self.device.destroy_buffer(s.buf, None);
			self.device.free_memory(s.mem, None);
		}
		s.cap = 0;
		s.ptr = MappedPtr(ptr::null_mut());
	}

	/// Drop every pooled host buffer (called from `Drop`).
	pub(crate) fn free_host_pool(&self) {
		if let Ok(mut pool) = self.host.lock() {
			let pool = &mut *pool;
			for s in pool
				.rb
				.iter_mut()
				.chain(pool.xs.iter_mut())
				.chain(pool.yd.iter_mut())
				.chain(pool.xd.iter_mut())
				.chain(pool.ps.iter_mut())
				.chain(pool.pd.iter_mut())
			{
				self.free_host_slot(s);
			}
			pool.rb.clear();
			pool.xs.clear();
			pool.yd.clear();
			pool.xd.clear();
			pool.ps.clear();
			pool.pd.clear();
		}
	}

	/// Pooled-slot (re)allocations since the last call, for the timing line.
	pub(crate) fn host_pool_allocs(&self) -> u64 {
		match self.host.lock() {
			Ok(mut p) => std::mem::take(&mut p.allocs),
			Err(_) => 0,
		}
	}
}
