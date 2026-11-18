//! Host memory imported as GPU buffers (`VK_EXT_external_memory_host`), so
//! the prefill split's GEMMs write their rows straight into the caller's
//! `[seq x out]` matrix.
//!
//! WHY
//! ---
//! The split's GPU rows used to land in a pooled readback buffer and be
//! copied into the output matrix after the wait: ~0.93 s per 2048-token chunk
//! (~22 GB of f32 at ~24 GB/s on the whole pool), serial with everything else.
//! The GEMM already writes host memory over PCIe (`y_direct`); pointing it at
//! the final matrix instead makes the copy disappear.
//!
//! LIFETIME
//! --------
//! An import pins the pages of a heap block and must be destroyed before
//! the block goes back to the OS. The output matrices are ordinary `Vec`s from
//! the `cli` binary's recycling allocator, which hands the same blocks back
//! layer after layer, so an import is made once per block and kept; the
//! allocator's release hook (`quant::alloc::set_release_hook`) destroys it
//! when the block is really freed. Without that allocator installed
//! (`release_tracked() == false`, e.g. under `cargo test`) nothing is
//! imported and the readback copy stays.
//!
//! An import covers the block's pages, rounded out to the import alignment
//! (4 KiB here). A heap block of >= 1 MiB is its own `VirtualAlloc` region on
//! Windows, so the rounding stays inside committed memory of that block.
//!
//! `QWEN_GPU_Y_INPLACE=0` keeps the readback copy, for the A/B.

use super::*;
use std::sync::atomic::AtomicUsize;

struct Import {
	base: usize,
	len: usize,
	buf: vk::Buffer,
	mem: vk::DeviceMemory,
}

struct DevTable {
	handle: vk::Device,
	dev: ash::Device,
	align: usize,
	entries: Vec<Import>,
}

/// Every live import, per device. A global because the allocator's release
/// hook is a plain `fn`.
static TABLE: Mutex<Vec<DevTable>> = Mutex::new(Vec::new());

/// The bytes of every live import of the process ([`GpuDevice::import_host_range`],
/// whatever its caller), by memory handle, and their sum.
static LIVE: Mutex<Vec<(u64, usize)>> = Mutex::new(Vec::new());
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Host memory imported as GPU buffers by the whole process, in bytes.
pub fn host_imported() -> usize {
	LIVE_BYTES.load(Ordering::Relaxed)
}

/// `QWEN_GPU_IMPORT_MB`: what an optional import (one a model can do
/// without) may bring [`host_imported`] up to, default 24 GiB. Flash-Next
/// imports 22.8 GB of NPU expert pools into the iGPU; the image model's
/// iGPU block share 6.9 GB of NPU weights on top (29.6 GB) lost the NPU
/// for good (`0x70000001` on every later job of the process) at the chat
/// engine's next prefill set switch; at 25.2 GB both ran.
pub fn host_import_budget() -> usize {
	static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
	*B.get_or_init(|| {
		std::env::var("QWEN_GPU_IMPORT_MB")
			.ok()
			.and_then(|v| v.trim().parse::<usize>().ok())
			.map_or(24 << 30, |mb| mb << 20)
	})
}

/// Destroy an import made by [`GpuDevice::import_host_range`] (its buffer
/// and memory) and take it off [`host_imported`].
///
/// # Safety
/// No submission may still use the buffer.
pub(crate) unsafe fn free_import(dev: &ash::Device, buf: vk::Buffer, mem: vk::DeviceMemory) {
	unsafe {
		dev.destroy_buffer(buf, None);
		dev.free_memory(mem, None);
	}
	let h = vk::Handle::as_raw(mem);
	if let Ok(mut l) = LIVE.lock() {
		if let Some(i) = l.iter().position(|e| e.0 == h) {
			LIVE_BYTES.fetch_sub(l.swap_remove(i).1, Ordering::Relaxed);
		}
	}
}

/// The allocator's release hook: destroy every import overlapping the block
/// being freed. Runs inside `dealloc`, so it must not free a large block.
fn release_hook(ptr: usize, size: usize) {
	let Ok(mut t) = TABLE.lock() else { return };
	for dt in t.iter_mut() {
		let (lo, hi) = (
			ptr & !(dt.align - 1),
			(ptr + size + dt.align - 1) & !(dt.align - 1),
		);
		let dev = &dt.dev;
		dt.entries.retain(|e| {
			if e.base < hi && lo < e.base + e.len {
				unsafe { free_import(dev, e.buf, e.mem) };
				false
			} else {
				true
			}
		});
	}
}

pub(crate) fn y_inplace() -> bool {
	static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
	*ON.get_or_init(|| env_on("QWEN_GPU_Y_INPLACE", true))
}

impl GpuDevice {
	/// Register this device for host-pointer imports (called once from
	/// `init` when the extension was enabled).
	pub(crate) fn register_host_import(&self, align: usize) {
		if !align.is_power_of_two() {
			return;
		}
		quant::alloc::set_release_hook(release_hook);
		if let Ok(mut t) = TABLE.lock() {
			t.push(DevTable {
				handle: self.device.handle(),
				dev: self.device.clone(),
				align,
				entries: Vec::new(),
			});
		}
	}

	/// Destroy this device's imports (from `Drop`, before the device goes).
	pub(crate) fn drop_host_imports(&self) {
		let Ok(mut t) = TABLE.lock() else { return };
		if let Some(i) = t.iter().position(|d| d.handle == self.device.handle()) {
			let dt = t.swap_remove(i);
			for e in &dt.entries {
				unsafe { free_import(&self.device, e.buf, e.mem) };
			}
		}
	}

	/// A storage buffer over the host bytes `[ptr, ptr + bytes)` and the
	/// offset of `ptr` in it, importing the block on first use. `None` when
	/// imports are off or unsupported here, or the import failed; the caller
	/// then uses the readback copy.
	///
	/// The memory must stay allocated while the GPU writes it; the release
	/// hook covers everything after that.
	pub(crate) fn import_host(&self, ptr: usize, bytes: usize) -> Option<(vk::Buffer, u64)> {
		self.host_import.as_ref()?;
		if !y_inplace() || bytes < quant::alloc::MIN_BLOCK || !quant::alloc::release_tracked() {
			return None;
		}
		let mut t = TABLE.lock().ok()?;
		let dt = t.iter_mut().find(|d| d.handle == self.device.handle())?;
		let a = dt.align;
		let (base, end) = (ptr & !(a - 1), (ptr + bytes + a - 1) & !(a - 1));
		if let Some(e) = dt
			.entries
			.iter()
			.find(|e| e.base <= base && end <= e.base + e.len)
		{
			return Some((e.buf, (ptr - e.base) as u64));
		}
		match self.import_host_range(base, end - base) {
			Ok((buf, mem)) => {
				dt.entries.push(Import {
					base,
					len: end - base,
					buf,
					mem,
				});
				Some((buf, (ptr - base) as u64))
			}
			Err(e) => {
				self.host_import_failed(&e);
				None
			}
		}
	}

	/// The host-pointer import alignment, when this device imports host
	/// memory at all.
	pub(crate) fn host_import_align(&self) -> Option<usize> {
		self.host_import.as_ref()?;
		let t = TABLE.lock().ok()?;
		t.iter()
			.find(|d| d.handle == self.device.handle())
			.map(|d| d.align)
	}

	/// Import the host bytes `[base, base + len)` (both aligned to
	/// [`Self::host_import_align`]) as one storage buffer, untracked: the
	/// caller destroys it with [`free_import`] before the memory goes away.
	pub(crate) fn import_host_range(
		&self,
		base: usize,
		len: usize,
	) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
		let ext = self
			.host_import
			.as_ref()
			.ok_or("no VK_EXT_external_memory_host")?;
		let ht = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
		let mut hp = vk::MemoryHostPointerPropertiesEXT::default();
		let r = unsafe {
			(ext.fp().get_memory_host_pointer_properties_ext)(
				self.device.handle(),
				ht,
				base as *const c_void,
				&mut hp,
			)
		};
		if r != vk::Result::SUCCESS {
			return Err(format!("vkGetMemoryHostPointerPropertiesEXT: {r}"));
		}
		let ext_info = vk::ExternalMemoryBufferCreateInfo {
			handle_types: ht,
			..Default::default()
		};
		let buf = unsafe {
			self.device.create_buffer(
				&vk::BufferCreateInfo {
					p_next: &ext_info as *const _ as *const c_void,
					size: len as u64,
					usage: vk::BufferUsageFlags::STORAGE_BUFFER,
					sharing_mode: vk::SharingMode::EXCLUSIVE,
					..Default::default()
				},
				None,
			)
		}
		.map_err(|e| format!("import buffer: {e}"))?;
		let req = unsafe { self.device.get_buffer_memory_requirements(buf) };
		let bits = hp.memory_type_bits & req.memory_type_bits;
		if bits == 0 || req.size > len as u64 {
			unsafe { self.device.destroy_buffer(buf, None) };
			return Err(format!(
				"no memory type (host bits {:#x}, buffer bits {:#x}, size {} of {len})",
				hp.memory_type_bits, req.memory_type_bits, req.size
			));
		}
		let import = vk::ImportMemoryHostPointerInfoEXT {
			handle_type: ht,
			p_host_pointer: base as *mut c_void,
			..Default::default()
		};
		let mem = match unsafe {
			self.device.allocate_memory(
				&vk::MemoryAllocateInfo {
					p_next: &import as *const _ as *const c_void,
					allocation_size: len as u64,
					memory_type_index: bits.trailing_zeros(),
					..Default::default()
				},
				None,
			)
		} {
			Ok(m) => m,
			Err(e) => {
				unsafe { self.device.destroy_buffer(buf, None) };
				return Err(format!("import of {len} bytes: {e}"));
			}
		};
		if let Err(e) = unsafe { self.device.bind_buffer_memory(buf, mem, 0) } {
			unsafe {
				self.device.destroy_buffer(buf, None);
				self.device.free_memory(mem, None);
			}
			return Err(format!("bind import: {e}"));
		}
		if let Ok(mut l) = LIVE.lock() {
			l.push((vk::Handle::as_raw(mem), len));
			LIVE_BYTES.fetch_add(len, Ordering::Relaxed);
		}
		Ok((buf, mem))
	}

	fn host_import_failed(&self, why: &str) {
		static SAID: AtomicBool = AtomicBool::new(false);
		if !SAID.swap(true, Ordering::Relaxed) {
			eprintln!(
				"offload: host-pointer import failed ({why}); prefill keeps the readback copy"
			);
		}
	}
}
