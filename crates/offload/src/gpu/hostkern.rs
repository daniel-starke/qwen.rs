//! The plumbing of kernels that run on tensors in NPU host buffers
//! (`DitBlk`, `H3Blk`, `H3Vae`): one descriptor set layout of six storage
//! buffers, the pipelines, host ranges imported once
//! (`VK_EXT_external_memory_host`), descriptor sets cached by their
//! buffers, and submissions without a fence until [`Rig::end`].

use super::*;
use std::collections::HashMap;

/// Descriptor sets the pool holds.
const SETS: usize = 1024;

struct Imported {
	base: usize,
	len: usize,
	buf: vk::Buffer,
	mem: vk::DeviceMemory,
	rows: bool,
}

/// A page-aligned host vector of `T`, imported once.
pub(super) struct HostVec<T> {
	v: Vec<T>,
	off: usize,
	pub len: usize,
}

impl<T: Copy + Default> HostVec<T> {
	pub fn new(len: usize) -> HostVec<T> {
		let sz = std::mem::size_of::<T>();
		let v = vec![T::default(); len + 4096 / sz];
		let off = ((v.as_ptr() as usize).next_multiple_of(4096) - v.as_ptr() as usize) / sz;
		HostVec { v, off, len }
	}
	pub fn s(&self) -> &[T] {
		&self.v[self.off..self.off + self.len]
	}
	pub fn m(&mut self) -> &mut [T] {
		&mut self.v[self.off..self.off + self.len]
	}
	/// `(address, bytes)`.
	pub fn range(&self) -> (usize, usize) {
		(
			self.s().as_ptr() as usize,
			self.len * std::mem::size_of::<T>(),
		)
	}
}

pub(super) struct Rig {
	pub dev: GpuDevice,
	what: &'static str,
	dsl: vk::DescriptorSetLayout,
	pub layout: vk::PipelineLayout,
	sm: Vec<vk::ShaderModule>,
	pub pl: Vec<vk::Pipeline>,
	pool: vk::DescriptorPool,
	imports: Vec<Imported>,
	align: usize,
	sets: HashMap<[usize; 6], vk::DescriptorSet>,
	/// Submissions in flight (no fence; [`Rig::end`] fences them).
	pending: Vec<vk::CommandBuffer>,
	/// Imports kept across passes: the own buffers' addresses, and ranges of
	/// these lengths (the weights).
	pub own: Vec<usize>,
	pub keep_lens: Vec<usize>,
}

impl Rig {
	/// On the Vulkan adapter `pick` names, one pipeline per SPIR-V module,
	/// `pc` bytes of push constants.
	pub fn new(
		pick: Option<&str>,
		spvs: &[&[u8]],
		pc: u32,
		what: &'static str,
	) -> Result<Rig, String> {
		let dev = GpuDevice::init_pick(pick)?;
		let align = dev
			.host_import_align()
			.ok_or("the Vulkan device has no VK_EXT_external_memory_host")?;
		let d = &dev.device;
		let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..6)
			.map(|b| vk::DescriptorSetLayoutBinding {
				binding: b,
				descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 1,
				stage_flags: vk::ShaderStageFlags::COMPUTE,
				..Default::default()
			})
			.collect();
		let dsl = unsafe {
			d.create_descriptor_set_layout(
				&vk::DescriptorSetLayoutCreateInfo {
					binding_count: bindings.len() as u32,
					p_bindings: bindings.as_ptr(),
					..Default::default()
				},
				None,
			)
		}
		.map_err(|e| format!("{what}: {e}"))?;
		let pcr = vk::PushConstantRange {
			stage_flags: vk::ShaderStageFlags::COMPUTE,
			offset: 0,
			size: pc,
		};
		let layout = unsafe {
			d.create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo {
					set_layout_count: 1,
					p_set_layouts: &dsl,
					push_constant_range_count: 1,
					p_push_constant_ranges: &pcr,
					..Default::default()
				},
				None,
			)
		}
		.map_err(|e| format!("{what}: {e}"))?;
		let main = CString::new("main").unwrap();
		let mut sm = Vec::new();
		let mut pl = Vec::new();
		for s in spvs {
			let m = create_sm(d, &load_spv(s))?;
			pl.push(compute_pipeline(d, layout, m, &main)?);
			sm.push(m);
		}
		let sizes = [vk::DescriptorPoolSize {
			ty: vk::DescriptorType::STORAGE_BUFFER,
			descriptor_count: (6 * SETS) as u32,
		}];
		let pool = unsafe {
			d.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo {
					max_sets: SETS as u32,
					pool_size_count: sizes.len() as u32,
					p_pool_sizes: sizes.as_ptr(),
					..Default::default()
				},
				None,
			)
		}
		.map_err(|e| format!("{what} descriptor pool: {e}"))?;
		Ok(Rig {
			dev,
			what,
			dsl,
			layout,
			sm,
			pl,
			pool,
			imports: Vec::new(),
			align,
			sets: HashMap::new(),
			pending: Vec::new(),
			own: Vec::new(),
			keep_lens: Vec::new(),
		})
	}

	/// Record kernel `k` over set `set` with push constants `pc`, a grid of
	/// `gx` x `gy`, and a barrier after it.
	pub fn dispatch(
		&self,
		cmd: vk::CommandBuffer,
		k: usize,
		set: vk::DescriptorSet,
		pc: &[u32],
		gx: usize,
		gy: usize,
	) {
		self.dispatch3(cmd, k, set, pc, [gx, gy, 1]);
	}

	/// [`Rig::dispatch`] over a grid of three dimensions.
	pub fn dispatch3(
		&self,
		cmd: vk::CommandBuffer,
		k: usize,
		set: vk::DescriptorSet,
		pc: &[u32],
		g: [usize; 3],
	) {
		let d = &self.dev.device;
		unsafe {
			d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pl[k]);
			d.cmd_bind_descriptor_sets(
				cmd,
				vk::PipelineBindPoint::COMPUTE,
				self.layout,
				0,
				&[set],
				&[],
			);
			let b: Vec<u8> = pc.iter().flat_map(|x| x.to_le_bytes()).collect();
			d.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, &b);
			d.cmd_dispatch(cmd, g[0] as u32, g[1] as u32, g[2] as u32);
		}
		use vk::{AccessFlags as A, PipelineStageFlags as P};
		self.barrier(
			cmd,
			P::COMPUTE_SHADER,
			P::COMPUTE_SHADER,
			A::SHADER_WRITE,
			A::SHADER_READ | A::SHADER_WRITE,
		);
	}

	pub fn barrier(
		&self,
		cmd: vk::CommandBuffer,
		a: vk::PipelineStageFlags,
		b: vk::PipelineStageFlags,
		sa: vk::AccessFlags,
		da: vk::AccessFlags,
	) {
		unsafe {
			self.dev.device.cmd_pipeline_barrier(
				cmd,
				a,
				b,
				vk::DependencyFlags::empty(),
				&[vk::MemoryBarrier {
					src_access_mask: sa,
					dst_access_mask: da,
					..Default::default()
				}],
				&[],
				&[],
			)
		}
	}

	/// Record a submission (after the host's writes and the submissions
	/// before it; the host reads its writes after [`Rig::end`]) and submit
	/// it without a fence.
	pub fn submit(&mut self, rec: impl FnOnce(&Rig, vk::CommandBuffer)) -> Result<(), String> {
		use vk::{AccessFlags as A, PipelineStageFlags as P};
		let me = &*self;
		let cmd = me
			.dev
			.record_cmd(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT, |cmd| {
				me.barrier(
					cmd,
					P::HOST | P::COMPUTE_SHADER,
					P::COMPUTE_SHADER,
					A::HOST_WRITE | A::SHADER_WRITE,
					A::SHADER_READ | A::SHADER_WRITE,
				);
				rec(me, cmd);
				me.barrier(
					cmd,
					P::COMPUTE_SHADER,
					P::HOST,
					A::SHADER_WRITE,
					A::HOST_READ,
				);
			});
		if let Err(e) = self.dev.submit_cmd_nofence(cmd) {
			unsafe {
				self.dev
					.device
					.free_command_buffers(self.dev.cmd_pool, &[cmd])
			};
			return Err(format!("{}: {e}", self.what));
		}
		self.pending.push(cmd);
		Ok(())
	}

	/// Submissions in flight.
	pub fn in_flight(&self) -> usize {
		self.pending.len()
	}

	/// Wait for the submissions in flight (none: Ok): an empty one on the
	/// fence after them (a fence covers every earlier submission).
	pub fn end(&mut self) -> Result<(), String> {
		if self.pending.is_empty() {
			return Ok(());
		}
		let r = match self.dev.submit_nowait(|_| {}) {
			Ok(cmd) => self.dev.wait_submitted(cmd),
			Err(e) => {
				unsafe { self.dev.device.queue_wait_idle(self.dev.queue).ok() };
				Err(e)
			}
		};
		let cmds: Vec<vk::CommandBuffer> = std::mem::take(&mut self.pending);
		unsafe {
			self.dev
				.device
				.free_command_buffers(self.dev.cmd_pool, &cmds)
		};
		r.map_err(|e| format!("{}: {e}", self.what))
	}

	/// Drop the imports of the pass' buffers (before they are freed); the
	/// weights' and the own buffers' stay.
	pub fn forget_rows(&mut self) {
		let _ = self.end();
		let d = &self.dev.device;
		unsafe {
			let _ = d.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty());
		}
		self.sets.clear();
		let mut keep = Vec::new();
		for e in self.imports.drain(..) {
			if e.rows {
				unsafe { hostimp::free_import(d, e.buf, e.mem) };
			} else {
				keep.push(e);
			}
		}
		self.imports = keep;
	}

	/// The descriptor set over host ranges `b` (bindings 0..6).
	pub fn set(&mut self, b: [(usize, usize); 6]) -> Result<vk::DescriptorSet, String> {
		let key = b.map(|x| x.0);
		if let Some(s) = self.sets.get(&key) {
			return Ok(*s);
		}
		if self.sets.len() == SETS {
			// the sets of the submissions in flight go with the pool
			self.end()?;
			unsafe {
				self.dev
					.device
					.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
			}
			.map_err(|e| format!("{}: {e}", self.what))?;
			self.sets.clear();
		}
		let mut infos = [vk::DescriptorBufferInfo::default(); 6];
		for (i, &(ptr, len)) in b.iter().enumerate() {
			let (buf, off) = self.import(ptr, len)?;
			infos[i] = vk::DescriptorBufferInfo {
				buffer: buf,
				offset: off,
				range: len as u64,
			};
		}
		let d = &self.dev.device;
		let set = unsafe {
			d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
				descriptor_pool: self.pool,
				descriptor_set_count: 1,
				p_set_layouts: &self.dsl,
				..Default::default()
			})
		}
		.map_err(|e| format!("{} descriptor set: {e}", self.what))?[0];
		let writes: Vec<vk::WriteDescriptorSet> = (0..6)
			.map(|i| vk::WriteDescriptorSet {
				dst_set: set,
				dst_binding: i as u32,
				descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 1,
				p_buffer_info: &infos[i],
				..Default::default()
			})
			.collect();
		unsafe { d.update_descriptor_sets(&writes, &[]) };
		self.sets.insert(key, set);
		Ok(set)
	}

	/// A buffer over host bytes `[ptr, ptr + len)` and `ptr`'s offset in it,
	/// importing the covering pages on first use. Ranges of [`Self::keep_lens`]
	/// lengths and the own buffers are kept across passes.
	fn import(&mut self, ptr: usize, len: usize) -> Result<(vk::Buffer, u64), String> {
		let a = self.align;
		let (base, end) = (ptr & !(a - 1), (ptr + len + a - 1) & !(a - 1));
		if let Some(e) = self
			.imports
			.iter()
			.find(|e| e.base <= base && end <= e.base + e.len)
		{
			return Ok((e.buf, (ptr - e.base) as u64));
		}
		let keep = self.keep_lens.contains(&len) || self.own.contains(&ptr);
		let (buf, mem) = self.dev.import_host_range(base, end - base)?;
		self.imports.push(Imported {
			base,
			len: end - base,
			buf,
			mem,
			rows: !keep,
		});
		Ok((buf, (ptr - base) as u64))
	}
}

impl Drop for Rig {
	fn drop(&mut self) {
		let _ = self.end();
		let d = &self.dev.device;
		unsafe {
			let _ = d.device_wait_idle();
			let _ = d.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty());
			for e in self.imports.drain(..) {
				hostimp::free_import(d, e.buf, e.mem);
			}
			for &p in &self.pl {
				d.destroy_pipeline(p, None);
			}
			for &m in &self.sm {
				d.destroy_shader_module(m, None);
			}
			d.destroy_descriptor_pool(self.pool, None);
			d.destroy_pipeline_layout(self.layout, None);
			d.destroy_descriptor_set_layout(self.dsl, None);
		}
	}
}
