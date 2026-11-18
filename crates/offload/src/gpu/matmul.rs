use super::*;

impl GpuDevice {
	/// Upload one weight. Returns whether the device now holds it: an
	/// unsupported type or size, VRAM that cannot back the weight (on a
	/// rebuilt device another application may still hold part of it), or
	/// a lost device leave it on the CPU.
	pub fn upload(
		&mut self,
		il: usize,
		id: WeightId,
		raw: &[u8],
		ty: GgmlType,
		out_dim: usize,
		in_dim: usize,
	) -> bool {
		if self.lost.load(Ordering::SeqCst) {
			return false;
		}
		let (kind, expected) = match ty {
			GgmlType::Q4_K | GgmlType::Q4_K_M => {
				(GpuKind::Q4K, out_dim * in_dim / QK_K * Q4K_BLOCK)
			}
			GgmlType::Q6_K => (GpuKind::Q6K, out_dim * in_dim / QK_K * Q6K_BLOCK),
			GgmlType::Q8_0 => (GpuKind::Q8_0, out_dim * in_dim / QK8_0 * Q8_0_BLOCK),
			GgmlType::F16 => (GpuKind::F16, out_dim * in_dim * F16_ELEM),
			GgmlType::F32 => (GpuKind::F32, out_dim * in_dim * F32_ELEM),
			other => {
				eprintln!("offload: {il:?}/{id:?} quant {other:?} not offloaded; CPU fallback");
				return false;
			}
		};
		if raw.len() != expected {
			eprintln!(
					"offload: {il:?}/{id:?} {ty:?} raw len {} != expected {}; skipping GPU (CPU fallback)",
					raw.len(), expected
				);
			return false;
		}
		// Q6_K is repacked to a dword-aligned stride in VRAM (see
		// `Q6K_BLOCK_PAD`); the GGUF bytes themselves are untouched, so the
		// CPU fallback still reads the original 210 B blocks.
		let repacked: Vec<u8> = if kind == GpuKind::Q6K {
			let nb = raw.len() / Q6K_BLOCK;
			let mut v = vec![0u8; nb * Q6K_BLOCK_PAD];
			for b in 0..nb {
				let (d, s) = (b * Q6K_BLOCK_PAD, b * Q6K_BLOCK);
				v[d..d + Q6K_BLOCK].copy_from_slice(&raw[s..s + Q6K_BLOCK]);
			}
			v
		} else {
			Vec::new()
		};
		let raw: &[u8] = if kind == GpuKind::Q6K { &repacked } else { raw };

		let padded_len = (raw.len() + 3) & !3;

		// Weights: device-local, copied via a host-visible staging buffer.
		let Some((weight_buf, weight_mem)) = self.try_create_gpu_buffer(
			padded_len as u64,
			vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
			vk::MemoryPropertyFlags::DEVICE_LOCAL,
		) else {
			eprintln!("offload: no VRAM for {il:?}/{id:?} ({padded_len} bytes); CPU fallback");
			return false;
		};
		{
			let Some((stg_buf, stg_mem)) = self.try_create_gpu_buffer(
				padded_len as u64,
				vk::BufferUsageFlags::TRANSFER_SRC,
				vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
			) else {
				unsafe {
					self.device.destroy_buffer(weight_buf, None);
					self.device.free_memory(weight_mem, None);
				}
				return false;
			};
			let sp = self.map(stg_mem, padded_len);
			unsafe {
				ptr::copy_nonoverlapping(raw.as_ptr(), sp as *mut u8, raw.len());
			}
			self.copy_buffer(stg_buf, weight_buf, padded_len as u64);
			unsafe {
				self.device.destroy_buffer(stg_buf, None);
				self.device.free_memory(stg_mem, None);
			}
		}

		// Sized for `decode_seq` positions: a speculative verify batch runs
		// through these same buffers (`matmul_batch_prepare`).
		let ds = self.decode_seq.max(1);
		let in_dim_u = (in_dim * ds) as u64;
		let out_dim_u = (out_dim * ds) as u64;
		// The small buffers share a few large blocks rather than taking
		// one allocation each (see `create_small_gpu_buffer`).
		let (x_buf, x_mem, x_stg_buf, x_stg_mem, x_stg_ptr) =
			self.create_staged_small(in_dim_u * 4, vk::BufferUsageFlags::STORAGE_BUFFER);
		let (y_buf, y_mem) = self.create_small_gpu_buffer(
			out_dim_u * 4,
			vk::BufferUsageFlags::STORAGE_BUFFER
				| vk::BufferUsageFlags::TRANSFER_SRC
				| vk::BufferUsageFlags::TRANSFER_DST,
		);
		let (rb_buf, rb_mem) = self.create_readback_buffer(out_dim_u * 4);
		let rb_ptr = self.map(rb_mem, out_dim * ds * 4);
		let (params_buf, params_mem, params_stg_buf, params_stg_mem, params_stg_ptr) =
			self.create_staged_small(16, vk::BufferUsageFlags::UNIFORM_BUFFER);
		{
			let mut pb = [0u8; 16];
			pb[0..4].copy_from_slice(&(out_dim as u32).to_le_bytes());
			pb[4..8].copy_from_slice(&(in_dim as u32).to_le_bytes());
			// Only a bound (`si < seq`): the grid's y decides how many
			// positions a dispatch computes.
			pb[8..12].copy_from_slice(&(ds as u32).to_le_bytes());
			unsafe {
				ptr::copy_nonoverlapping(pb.as_ptr(), params_stg_ptr.0 as *mut u8, 16);
			}
			self.copy_buffer(params_stg_buf, params_buf, 16);
		}

		let desc_set = unsafe {
			self.device
				.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
					descriptor_pool: self.desc_pool,
					descriptor_set_count: 1,
					p_set_layouts: &self.desc_layout,
					..Default::default()
				})
				.expect("alloc desc set")[0]
		};
		let wbi = vk::DescriptorBufferInfo {
			buffer: weight_buf,
			offset: 0,
			range: vk::WHOLE_SIZE,
			..Default::default()
		};
		let xbi = vk::DescriptorBufferInfo {
			buffer: x_buf,
			offset: 0,
			range: vk::WHOLE_SIZE,
			..Default::default()
		};
		let ybi = vk::DescriptorBufferInfo {
			buffer: y_buf,
			offset: 0,
			range: vk::WHOLE_SIZE,
			..Default::default()
		};
		let pbi = vk::DescriptorBufferInfo {
			buffer: params_buf,
			offset: 0,
			range: 16,
			..Default::default()
		};
		let writes = [
			vk::WriteDescriptorSet {
				dst_set: desc_set,
				dst_binding: 0,
				descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 1,
				p_buffer_info: &wbi,
				..Default::default()
			},
			vk::WriteDescriptorSet {
				dst_set: desc_set,
				dst_binding: 1,
				descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 1,
				p_buffer_info: &xbi,
				..Default::default()
			},
			vk::WriteDescriptorSet {
				dst_set: desc_set,
				dst_binding: 2,
				descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 1,
				p_buffer_info: &ybi,
				..Default::default()
			},
			vk::WriteDescriptorSet {
				dst_set: desc_set,
				dst_binding: 3,
				descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
				descriptor_count: 1,
				p_buffer_info: &pbi,
				..Default::default()
			},
		];
		unsafe { self.device.update_descriptor_sets(&writes, &[]) };

		self.clear_decode_cmds();
		self.weights.insert(
			(il, id),
			GpuWeight {
				kind,
				weight_buf,
				weight_mem,
				x_buf,
				x_mem,
				x_stg_buf,
				x_stg_mem,
				x_stg_ptr,
				y_buf,
				y_mem,
				rb_buf,
				rb_mem,
				rb_ptr: MappedPtr(rb_ptr),
				params_buf,
				params_mem,
				params_stg_buf,
				params_stg_mem,
				desc_set,
				out_dim,
				in_dim,
			},
		);
		!self.lost.load(Ordering::SeqCst)
	}

	pub fn matmul(
		&self,
		il: usize,
		id: WeightId,
		x: &[f32],
		_out_dim: usize,
		_in_dim: usize,
	) -> Option<Vec<f32>> {
		self.matmul_batch(&[(il, id, x)])
			.map(|mut v| v.pop().unwrap())
	}

	pub(crate) fn pack_x(kind: GpuKind, in_dim: usize, x: &[f32]) -> Vec<u8> {
		match kind {
			GpuKind::Q4K | GpuKind::Q6K => {
				debug_assert_eq!(
					in_dim % 256,
					0,
					"q4/q6 matmul in_dim must be a QK_K multiple"
				);
				let q8k = quantize_q8k(x, in_dim);
				let nb = in_dim / 256;
				let mut buf = vec![0u8; (in_dim + nb * 4 + 3) & !3];
				for j in 0..in_dim {
					buf[j] = q8k.qs[j] as u8;
				}
				for b in 0..nb {
					let fb = q8k.ds[b].to_le_bytes();
					buf[in_dim + b * 4..in_dim + b * 4 + 4].copy_from_slice(&fb);
				}
				buf
			}
			_ => {
				let mut buf = vec![0u8; x.len() * 4];
				let ptr = buf.as_mut_ptr();
				unsafe {
					ptr::copy_nonoverlapping(x.as_ptr() as *const u8, ptr, x.len() * 4);
				}
				buf
			}
		}
	}

	/// Write pre-quantized q8_K rows into a staging map in `pack_x_seq`'s
	/// layout: per row, `in_dim` `qs` bytes then `in_dim / 256` f32 deltas.
	fn stage_q8k(dst: *mut u8, q8: &[quant::Q8K], in_dim: usize) {
		let nb = in_dim / 256;
		let stride = in_dim + nb * 4;
		let d = dst as usize;
		let row = move |s: usize| unsafe {
			let p = (d as *mut u8).add(s * stride);
			ptr::copy_nonoverlapping(q8[s].qs.as_ptr() as *const u8, p, in_dim);
			ptr::copy_nonoverlapping(q8[s].ds.as_ptr() as *const u8, p.add(in_dim), nb * 4);
		};
		if q8.len() * stride < 256 * 1024 {
			(0..q8.len()).for_each(row);
		} else {
			quant::par_for(q8.len(), row);
		}
	}

	pub(crate) fn pack_x_seq(kind: GpuKind, in_dim: usize, x: &[f32], seq: usize) -> Vec<u8> {
		match kind {
			GpuKind::Q4K | GpuKind::Q6K => {
				debug_assert_eq!(
					in_dim % 256,
					0,
					"q4/q6 matmul in_dim must be a QK_K multiple"
				);
				let nb = in_dim / 256;
				let stride = in_dim + nb * 4;
				// Every byte of every row is written below (`stride` is
				// exactly `qs` + `ds`), so the buffer is never read before
				// it is initialized. `vec![0u8; ..]` would zero 36 MB on one
				// core first, which is most of what this used to cost.
				let mut buf: Vec<u8> = Vec::with_capacity(seq * stride);
				#[allow(clippy::uninit_vec)]
				unsafe {
					buf.set_len(seq * stride)
				};
				// `qs` goes straight into the row's `in_dim` bytes (`i8`
				// and `u8` are the same bytes); only the `nb` deltas need a
				// detour, because the row base is not 4-byte aligned in a
				// `Vec<u8>`. Both come from `quantize_q8k_into`, which
				// allocates nothing: the allocations the `Q8K`-returning
				// form does per row were what kept this loop off the
				// memory roofline.
				thread_local! {
					static DS: std::cell::RefCell<Vec<f32>> =
						const { std::cell::RefCell::new(Vec::new()) };
				}
				let row = |s: usize, dst: *mut u8| {
					let xpos = &x[s * in_dim..(s + 1) * in_dim];
					DS.with(|ds| {
						let mut ds = ds.borrow_mut();
						if ds.len() < nb {
							ds.resize(nb, 0.0);
						}
						let qs = unsafe { std::slice::from_raw_parts_mut(dst as *mut i8, in_dim) };
						quant::quantize_q8k_into(xpos, in_dim, &mut ds, qs);
						unsafe {
							ptr::copy_nonoverlapping(
								ds.as_ptr() as *const u8,
								dst.add(in_dim),
								nb * 4,
							);
						}
					});
				};
				// Threading this is worth ~5 % of a streamed FFN block: at
				// `seq = 2048` the `FfnDown` activation is 142 MB of f32 to
				// quantize, and serially that was 22-27 ms; by then the
				// single largest host cost in the call. The rows are
				// independent and write disjoint `stride`-byte spans.
				// Below a dispatch's worth of traffic (the same 256 KB
				// cutoff `par_rows` uses) the fork-join costs more than the
				// quantize, which is every `seq == 1` call.
				if seq * in_dim * 4 < 256 * 1024 {
					for s in 0..seq {
						row(s, unsafe { buf.as_mut_ptr().add(s * stride) });
					}
				} else {
					let d = buf.as_mut_ptr() as usize;
					quant::par_for(seq, |s| row(s, unsafe { (d as *mut u8).add(s * stride) }));
				}
				buf
			}
			_ => {
				let mut buf = vec![0u8; x.len() * 4];
				let ptr = buf.as_mut_ptr();
				unsafe {
					ptr::copy_nonoverlapping(x.as_ptr() as *const u8, ptr, x.len() * 4);
				}
				buf
			}
		}
	}

	/// Decode path (seq == 1): each job reuses its weight's pre-allocated,
	/// persistently-mapped buffers (no per-submission allocate/free churn;
	/// this is what lets the GPU path stay resident across a whole trailing
	/// block of offloaded layers without re-mapping buffers every token).
	/// Returns `None` if the device was lost so the caller can fall back to
	/// the CPU path.
	pub fn matmul_batch(&self, jobs: &[(usize, WeightId, &[f32])]) -> Option<Vec<Vec<f32>>> {
		let p = self.matmul_batch_begin(jobs)?;
		self.matmul_batch_end(p)
	}

	/// Wait for a `matmul_batch_begin` submission and return its outputs.
	pub fn matmul_batch_end(&self, p: PendingBatch) -> Option<Vec<Vec<f32>>> {
		if !p.cached {
			return self.run_batch_end(p.cmd, &p.items, p.t0);
		}
		self.wait_fence().ok()?;
		self.read_batch(&p.items, p.t0)
	}

	/// `matmul_batch_end` without the result `Vec`s: waits, then hands
	/// `sink` each job's output in the mapped readback buffer, in job
	/// order, so the caller can copy it straight to where it belongs.
	pub fn matmul_batch_end_with(
		&self,
		p: PendingBatch,
		mut sink: impl FnMut(usize, &[f32]),
	) -> Option<()> {
		if p.cached {
			self.wait_fence().ok()?;
		} else {
			self.wait_submitted(p.cmd).ok()?;
		}
		for (i, d) in p.items.iter().enumerate() {
			// The fence wait made the readback copy visible; `rb_ptr`
			// maps `out_dim` floats of HOST_COHERENT memory.
			sink(i, unsafe {
				std::slice::from_raw_parts(d.rb_ptr.0 as *const f32, d.out_dim)
			});
		}
		Some(())
	}

	/// `matmul_batch` without the wait: stage every `x`, submit, return.
	/// The caller must pass the result to `matmul_batch_end` before any
	/// other submission on this device (one in flight on `self.fence`).
	pub fn matmul_batch_begin(&self, jobs: &[(usize, WeightId, &[f32])]) -> Option<PendingBatch> {
		let p = self.matmul_batch_prepare(jobs)?;
		self.matmul_batch_submit(&p)?;
		Some(p)
	}

	/// Submit a `matmul_batch_prepare`d batch. `None` if the submission
	/// failed; the batch must then not be passed to `matmul_batch_end*`.
	pub fn matmul_batch_submit(&self, p: &PendingBatch) -> Option<()> {
		self.submit_prepared(p.cmd, p.cached)
	}

	/// `matmul_batch_begin` up to, not including, the submit: stage every
	/// `x` and look up (or record) the command buffer. Split out so the
	/// decode row split can issue the ~15 us `vkQueueSubmit` from inside
	/// the CPU dispatch (`quant::apply_linear_group_rows_pre`) instead of
	/// in front of it.
	pub fn matmul_batch_prepare(&self, jobs: &[(usize, WeightId, &[f32])]) -> Option<PendingBatch> {
		if self.lost.load(Ordering::SeqCst) {
			return None;
		}
		let t0 = std::time::Instant::now();
		let mut items = Vec::with_capacity(jobs.len());
		// Every job of a batch covers the same positions (1, or a
		// speculative verify batch of up to `decode_seq`).
		let seq = jobs
			.first()
			.map_or(1, |(il, id, x)| x.len() / self.weights[&(*il, *id)].in_dim);
		if seq == 0 || seq > self.decode_seq.max(1) {
			return None;
		}
		for (il, id, x) in jobs {
			let w = &self.weights[&(*il, *id)];
			if x.len() != seq * w.in_dim {
				return None;
			}
			let x_len = if self.wants_q8k(w.kind, 1) {
				let bytes = if seq == 1 {
					Self::pack_x(w.kind, w.in_dim, x)
				} else {
					Self::pack_x_seq(w.kind, w.in_dim, x, seq)
				};
				unsafe {
					ptr::copy_nonoverlapping(bytes.as_ptr(), w.x_stg_ptr.0 as *mut u8, bytes.len());
				}
				bytes.len()
			} else {
				// Raw f32: straight into the staging map, no packed copy.
				unsafe {
					ptr::copy_nonoverlapping(x.as_ptr(), w.x_stg_ptr.0 as *mut f32, x.len());
				}
				x.len() * 4
			};
			let (pipeline, gx, gy) = self.gemv_dispatch(w.kind, w.out_dim, seq);
			items.push(Dispatch {
				pipeline,
				desc_set: w.desc_set,
				x_buf: w.x_buf,
				x_stg: w.x_stg_buf,
				x_len: x_len as u64,
				// The params were copied to `params_buf` once, at upload,
				// and nothing rewrites them.
				p_buf: w.params_buf,
				p_stg: w.params_stg_buf,
				p_len: 0,
				y_buf: w.y_buf,
				rb_buf: w.rb_buf,
				rb_ptr: w.rb_ptr,
				out_dim: w.out_dim * seq,
				gx,
				gy,
			});
		}
		let key: Vec<(usize, WeightId)> = jobs.iter().map(|(il, id, _)| (*il, *id)).collect();
		let (cmd, cached) = self.prepare_batch_cached(&key, seq as u32, &items);
		Some(PendingBatch {
			cmd,
			cached,
			items,
			t0,
		})
	}

	/// Prefill path (seq > 1): shared-`x` jobs grouped; per-call buffers so the
	/// activation is uploaded once per group. Returns `None` if the device was
	/// lost so the caller can fall back to the CPU path.
	pub fn matmul_batch_seq(&self, jobs: &[SeqJob]) -> Option<Vec<Vec<f32>>> {
		let p = self.matmul_batch_seq_begin(jobs)?;
		self.matmul_batch_seq_end(p)
	}

	/// `matmul_batch_seq` up to and including the submit, without the wait.
	/// Everything the call does on the host *before* the dispatch (stream
	/// bind, activation pack on the shared `quant` pool, staging) happens
	/// here, so a caller can then run CPU work on that pool concurrently
	/// with the GPU and collect with `matmul_batch_seq_end`. One in flight
	/// per device, like `matmul_batch_begin`.
	/// The tiled GEMMs write their result straight into the host-visible
	/// readback buffer instead of into VRAM followed by a copy into it.
	///
	/// The copy ran on the GPU *after* the last dispatch, at PCIe speed:
	/// ~16 ms for one 2048-token `FfnUp`+`FfnGate` call (0.70 of 2 x 17408
	/// rows x 2048 positions of f32 = 200 MB), ~1.7 s of serial GPU time
	/// per chunk over all matmuls, and `FfnUp` was the call where the GPU
	/// finished last. Written from the shader, the same bytes cross the
	/// bus while the GEMM computes (~1.7 GB/s over its ~120 ms, each
	/// element written once, 32 rows per warp store). Also frees the
	/// device-local `y` buffer, which prefill was holding over the VRAM
	/// budget. `QWEN_GPU_Y_DIRECT=0` is the A/B.
	pub(crate) fn y_direct() -> bool {
		static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
		*ON.get_or_init(|| env_on("QWEN_GPU_Y_DIRECT", true))
	}

	pub fn matmul_batch_seq_begin(&self, jobs: &[SeqJob]) -> Option<PendingSeq> {
		self.matmul_batch_seq_begin_bank(jobs, 0)
	}

	/// `matmul_batch_seq_begin` in prefill bank `bank` (0 or 1). Each bank
	/// has its own fence, host-pool slots and stream slot pins, so a batch
	/// of one bank may be submitted while the other bank's is still in
	/// flight (the two-half prefill pipeline); within a bank, one at a
	/// time. Only bank 0 may be in flight when anything else submits.
	pub fn matmul_batch_seq_begin_bank(&self, jobs: &[SeqJob], bank: usize) -> Option<PendingSeq> {
		let bank = bank.min(1);
		// Host-pool slot index offset of this bank.
		let bo = bank * 16;
		if self.lost.load(Ordering::SeqCst) {
			return None;
		}
		struct D<'a> {
			il: usize,
			id: WeightId,
			x: *const f32,
			seq: usize,
			kind: GpuKind,
			in_dim: usize,
			out_dim: usize,
			q8: Option<&'a [quant::Q8K]>,
			dst: Option<(usize, usize, usize)>,
		}
		// Binding 0 comes from one of two places: the weight is resident
		// (uploaded once at load, `--n-gpu-layers`) or it is streamed into a
		// pool slot for this call. Resolving both here keeps the rest of the
		// function identical for the two cases.
		let mut descs = Vec::with_capacity(jobs.len());
		let mut wbufs: Vec<vk::Buffer> = Vec::with_capacity(jobs.len());
		let mut to_stream: Vec<(usize, WeightId, &[u8], GpuKind)> = Vec::new();
		let mut stream_slot: Vec<Option<usize>> = Vec::with_capacity(jobs.len());
		let mut follow_tx: Option<stream::FollowTx> = None;
		for j in jobs {
			match self.weights.get(&(j.il, j.id)) {
				Some(w) => {
					descs.push(D {
						il: j.il,
						id: j.id,
						x: j.x.as_ptr(),
						seq: j.seq,
						kind: w.kind,
						in_dim: w.in_dim,
						out_dim: j.rows.map_or(w.out_dim, |n| n.min(w.out_dim)),
						q8: j.q8,
						dst: j.dst,
					});
					wbufs.push(w.weight_buf);
					stream_slot.push(None);
				}
				None => {
					let (raw, kind, out_dim, in_dim) = j.src?;
					descs.push(D {
						il: j.il,
						id: j.id,
						x: j.x.as_ptr(),
						seq: j.seq,
						kind,
						in_dim,
						out_dim,
						q8: j.q8,
						dst: j.dst,
					});
					wbufs.push(vk::Buffer::null());
					stream_slot.push(Some(to_stream.len()));
					to_stream.push((j.il, j.id, raw, kind));
				}
			}
		}
		if !to_stream.is_empty() {
			let (bufs, ftx) = self.stream_bind(&to_stream, true, bank)?;
			follow_tx = ftx;
			for (ji, si) in stream_slot.iter().enumerate() {
				if let Some(si) = si {
					wbufs[ji] = bufs[*si];
				}
			}
		}

		// 0 = q8_K activation, 1 = raw f32. Depends on `seq` too: the tiled
		// GEMMs read q8_K, the f32 GEMVs raw floats.
		let pack_key = |k: GpuKind, seq: usize| -> u8 {
			if self.wants_q8k(k, seq) {
				0
			} else {
				1
			}
		};

		// Group jobs by shared activation.
		let mut gkeys: Vec<(usize, usize, usize, u8)> = Vec::new();
		let mut gjobs: Vec<Vec<usize>> = Vec::new();
		for (ji, d) in descs.iter().enumerate() {
			let pk = pack_key(d.kind, d.seq);
			let mut found = None;
			for (gi, (p, s, idim, pkk)) in gkeys.iter().enumerate() {
				if *p == d.x as usize && *s == d.seq && *idim == d.in_dim && *pkk == pk {
					found = Some(gi);
					break;
				}
			}
			match found {
				Some(gi) => gjobs[gi].push(ji),
				None => {
					gkeys.push((d.x as usize, d.seq, d.in_dim, pk));
					gjobs.push(vec![ji]);
				}
			}
		}

		let t_pack = std::time::Instant::now();
		// `pack` is two unrelated costs (quantizing `x` to q8_K and
		// creating/filling its staging buffer) and they take different
		// fixes, so they are timed apart.
		let mut quant_secs = 0.0f64;

		// Per-call resources.
		let mut group_x: Vec<(vk::Buffer, vk::Buffer, u64)> = Vec::new();
		let mut per_call: Vec<(vk::Buffer, vk::DeviceMemory)> = Vec::new();
		let mut per_call_maps: Vec<(vk::DeviceMemory, *mut c_void)> = Vec::new();

		for gj in &gjobs {
			let rep = &descs[gj[0]];
			let rep_kind = if pack_key(rep.kind, rep.seq) == 0 {
				GpuKind::Q4K
			} else {
				GpuKind::F32
			};
			let t_q = std::time::Instant::now();
			// Already quantized by the caller: written straight into the
			// staging map below, with no intermediate buffer.
			let pre_q8 = rep
				.q8
				.filter(|q| rep_kind == GpuKind::Q4K && q.len() == rep.seq);
			let xbytes = match pre_q8 {
				Some(_) => Vec::new(),
				None => Self::pack_x_seq(
					rep_kind,
					rep.in_dim,
					unsafe { std::slice::from_raw_parts(rep.x, rep.seq * rep.in_dim) },
					rep.seq,
				),
			};
			quant_secs += t_q.elapsed().as_secs_f64();
			let size = match pre_q8 {
				Some(_) => (rep.seq * (rep.in_dim + rep.in_dim / 256 * 4)) as u64,
				None => xbytes.len() as u64,
			};
			// The device-local half is per call (VRAM); only the host-side
			// staging buffer comes from the pool.
			let gi = group_x.len();
			let xb = match self.host_slot(pool::HostUse::DeviceX, bo + gi, size) {
				Some((b, _)) => b,
				None => {
					let (b, m) = self.create_gpu_buffer(
						size,
						vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
						vk::MemoryPropertyFlags::DEVICE_LOCAL,
					);
					per_call.push((b, m));
					b
				}
			};
			let (xb_stg, xb_stg_ptr) = match self.host_slot(pool::HostUse::Staging, bo + gi, size) {
				Some((b, p)) => (b, p),
				None => {
					let (b, m) = self.create_gpu_buffer(
						size,
						vk::BufferUsageFlags::TRANSFER_SRC,
						vk::MemoryPropertyFlags::HOST_VISIBLE
							| vk::MemoryPropertyFlags::HOST_COHERENT,
					);
					let p = self.map(m, size as usize);
					per_call.push((b, m));
					per_call_maps.push((m, p));
					(b, p)
				}
			};
			match pre_q8 {
				Some(q8) => Self::stage_q8k(xb_stg_ptr as *mut u8, q8, rep.in_dim),
				None => stream::par_memcpy(xb_stg_ptr as *mut u8, &xbytes),
			}
			// Only ever unmap memory that was actually mapped: recording
			// the *device-local* `xm` here once made the teardown call
			// `vkUnmapMemory` on memory that was never mapped (undefined
			// behavior; it segfaulted inside `nvoglv64.dll`) and leave the
			// staging allocation mapped until it was freed. Harmless on a
			// 5-token prompt, fatal on a real one: `QWEN_GPU_PREFILL=1` with
			// a 462-token prompt crashed in `matmul_batch_seq` every time.
			group_x.push((xb, xb_stg, size));
		}

		let n = jobs.len();
		let pool_sizes = [
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: (n * 3) as u32,
			},
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::UNIFORM_BUFFER,
				descriptor_count: n as u32,
			},
		];
		let pool = unsafe {
			self.device
				.create_descriptor_pool(
					&vk::DescriptorPoolCreateInfo {
						max_sets: n as u32,
						pool_size_count: pool_sizes.len() as u32,
						p_pool_sizes: pool_sizes.as_ptr(),
						..Default::default()
					},
					None,
				)
				.expect("per-call desc pool")
		};
		// `p_set_layouts` is an **array** of `descriptor_set_count` layouts,
		// not one layout applied `n` times. Passing `&self.desc_layout` with
		// `n > 1` made the driver read past that field and use whatever
		// followed it in `GpuDevice` as a `VkDescriptorSetLayout`: a
		// segfault inside `nvoglv64.dll`, and the reason `QWEN_GPU_PREFILL=1`
		// died on the *first* prefill call of any layer with two or more
		// grouped matmuls (`Wqkv` + `WqkvGate`). Single-job calls read
		// exactly one handle and were fine, which is why the decode path and
		// the single-shape benches never saw it.
		let layouts = vec![self.desc_layout; n];
		let sets = unsafe {
			self.device
				.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
					descriptor_pool: pool,
					descriptor_set_count: n as u32,
					p_set_layouts: layouts.as_ptr(),
					..Default::default()
				})
				.expect("per-call desc sets")
		};

		let t_alloc = std::time::Instant::now();
		let mut items = Vec::with_capacity(n);
		for (ji, d) in descs.iter().enumerate() {
			let y_size = (d.seq * d.out_dim * 4) as u64;
			let direct = self.use_gemm(d.kind, d.seq) && Self::y_direct();
			// The GEMM writes the caller's matrix itself: an imported view
			// of the whole `[seq x ld]` matrix, bound at this job's column.
			// The storage-buffer offset must be 16-byte aligned (heap
			// blocks are; `col0` is a multiple of 4 rows).
			let inplace = match d.dst {
				Some((p, ld, col0)) if direct => self
					.import_host(p, d.seq * ld * 4)
					.map(|(b, off)| (b, off + (col0 * 4) as u64, ld))
					.filter(|&(_, off, _)| off % 16 == 0),
				_ => None,
			};
			let (rb, rbp) = match (
				inplace,
				self.host_slot(pool::HostUse::Readback, bo + ji, y_size),
			) {
				(Some((b, _, _)), _) => (b, ptr::null_mut()),
				(None, Some((b, p))) => (b, p),
				(None, None) => {
					let (b, m) = self.create_readback_buffer(y_size);
					let p = self.map(m, y_size as usize);
					per_call.push((b, m));
					per_call_maps.push((m, p));
					(b, p)
				}
			};
			let yb = match self.host_slot(pool::HostUse::DeviceY, bo + ji, y_size) {
				_ if direct => rb,
				Some((b, _)) => b,
				None => {
					let (b, m) = self.create_gpu_buffer(
						y_size,
						vk::BufferUsageFlags::STORAGE_BUFFER
							| vk::BufferUsageFlags::TRANSFER_SRC
							| vk::BufferUsageFlags::TRANSFER_DST,
						vk::MemoryPropertyFlags::DEVICE_LOCAL,
					);
					per_call.push((b, m));
					b
				}
			};
			let (pb, pb_stg, pbp_stg) = match (
				self.host_slot(pool::HostUse::ParamsDev, bo + ji, 16),
				self.host_slot(pool::HostUse::ParamsStg, bo + ji, 16),
			) {
				(Some((b, _)), Some((s, p))) => (b, s, MappedPtr(p)),
				_ => {
					let (pb, pbm, pb_stg, pb_stg_mem, pbp_stg) =
						self.create_staged_buffer(16, vk::BufferUsageFlags::UNIFORM_BUFFER);
					per_call.push((pb, pbm));
					per_call.push((pb_stg, pb_stg_mem));
					per_call_maps.push((pb_stg_mem, pbp_stg.0 as *mut c_void));
					(pb, pb_stg, pbp_stg)
				}
			};
			{
				let mut pb_bytes = [0u8; 16];
				pb_bytes[0..4].copy_from_slice(&(d.out_dim as u32).to_le_bytes());
				pb_bytes[4..8].copy_from_slice(&(d.in_dim as u32).to_le_bytes());
				pb_bytes[8..12].copy_from_slice(&(d.seq as u32).to_le_bytes());
				if let Some((_, _, ld)) = inplace {
					pb_bytes[12..16].copy_from_slice(&(ld as u32).to_le_bytes());
				}
				unsafe {
					ptr::copy_nonoverlapping(pb_bytes.as_ptr(), pbp_stg.0 as *mut u8, 16);
				}
			}
			let gi = gjobs.iter().position(|g| g.contains(&ji)).unwrap();
			let (xb, xb_stg, xlen) = group_x[gi];

			let wbi = vk::DescriptorBufferInfo {
				buffer: wbufs[ji],
				offset: 0,
				range: vk::WHOLE_SIZE,
				..Default::default()
			};
			let xbi = vk::DescriptorBufferInfo {
				buffer: xb,
				offset: 0,
				range: vk::WHOLE_SIZE,
				..Default::default()
			};
			let ybi = vk::DescriptorBufferInfo {
				buffer: yb,
				offset: inplace.map_or(0, |(_, off, _)| off),
				range: vk::WHOLE_SIZE,
				..Default::default()
			};
			let pbi = vk::DescriptorBufferInfo {
				buffer: pb,
				offset: 0,
				range: 16,
				..Default::default()
			};
			let writes = [
				vk::WriteDescriptorSet {
					dst_set: sets[ji],
					dst_binding: 0,
					descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
					descriptor_count: 1,
					p_buffer_info: &wbi,
					..Default::default()
				},
				vk::WriteDescriptorSet {
					dst_set: sets[ji],
					dst_binding: 1,
					descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
					descriptor_count: 1,
					p_buffer_info: &xbi,
					..Default::default()
				},
				vk::WriteDescriptorSet {
					dst_set: sets[ji],
					dst_binding: 2,
					descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
					descriptor_count: 1,
					p_buffer_info: &ybi,
					..Default::default()
				},
				vk::WriteDescriptorSet {
					dst_set: sets[ji],
					dst_binding: 3,
					descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
					descriptor_count: 1,
					p_buffer_info: &pbi,
					..Default::default()
				},
			];
			unsafe { self.device.update_descriptor_sets(&writes, &[]) };

			// The tiled GEMM and the batched GEMV have *different* grid
			// contracts (rows/tokens vs rows/position), so the pipeline and
			// the grid must be chosen together: picking one without the
			// other silently computes a fraction of the output.
			let (pipeline, gx, gy) = if self.use_gemm(d.kind, d.seq) {
				let (gx, gy) = self.gemm_grid(d.out_dim, d.seq);
				(self.gemm_pipeline_for(d.kind), gx, gy)
			} else {
				self.gemv_dispatch(d.kind, d.out_dim, d.seq)
			};
			items.push(Dispatch {
				pipeline,
				desc_set: sets[ji],
				x_buf: xb,
				x_stg: xb_stg,
				x_len: xlen,
				p_buf: pb,
				p_stg: pb_stg,
				p_len: 16,
				y_buf: yb,
				rb_buf: rb,
				rb_ptr: MappedPtr(rbp),
				out_dim: d.seq * d.out_dim,
				gx,
				gy,
			});
		}

		if gpu_timing() {
			for (ji, d) in descs.iter().enumerate() {
				eprintln!(
					"[GPUT] job{ji} il={} id={:?} kind={} seq={} in={} out={} gx={} gy={}",
					d.il,
					d.id,
					d.kind.name(),
					d.seq,
					d.in_dim,
					d.out_dim,
					items[ji].gx,
					items[ji].gy,
				);
			}
		}
		// Everything the dispatch needs from the host is now recorded, so
		// this is the latest possible moment to wait for the weight upload
		// issued at the top of the call, and the whole point of issuing it
		// there rather than here.
		let follow = follow_tx.is_some();
		let cmds = match follow_tx {
			// The upload went to the repack thread: submit the resident
			// rows now and let that thread submit the streamed ones behind
			// the upload (`stream::Follow`), so neither this thread nor
			// the resident rows wait for it.
			Some(tx) => {
				let (res, strm): (Vec<usize>, Vec<usize>) =
					(0..items.len()).partition(|&ji| stream_slot[ji].is_none());
				let pick =
					|ix: &[usize]| -> Vec<Dispatch> { ix.iter().map(|&i| items[i]).collect() };
				let (a, b) = (pick(&res), pick(&strm));
				let mut cmds = Vec::with_capacity(2);
				if !a.is_empty() {
					let ca = self.record_cmd(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT, |c| {
						self.record_batch(c, &a)
					});
					if self.lost.load(Ordering::SeqCst) || self.submit_cmd_nofence(ca).is_err() {
						unsafe { self.device.free_command_buffers(self.cmd_pool, &[ca]) };
						drop(tx);
						self.stream_sync();
						self.stream_unpin(bank);
						self.free_per_call(&per_call, &per_call_maps, pool);
						return None;
					}
					cmds.push(ca);
				}
				let skip: Vec<vk::Buffer> = a.iter().map(|d| d.x_buf).collect();
				let cb = self.record_cmd(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT, |c| {
					self.record_batch_skip(c, &b, &skip)
				});
				cmds.push(cb);
				if tx.send(Some(cb)).is_err() {
					// The worker is gone; `wait_cmds` sees `follow_ok`
					// unset and reports the loss.
				}
				cmds
			}
			None => {
				self.stream_sync();
				match self.run_batch_begin(&items, bank) {
					Some(c) => vec![c],
					None => {
						self.stream_unpin(bank);
						self.free_per_call(&per_call, &per_call_maps, pool);
						return None;
					}
				}
			}
		};
		let t_run = std::time::Instant::now();
		Some(PendingSeq {
			cmds,
			follow,
			bank,
			items,
			per_call,
			per_call_maps,
			pool,
			t_pack,
			t_alloc,
			t_run,
			quant_secs,
		})
	}

	fn free_per_call(
		&self,
		per_call: &[(vk::Buffer, vk::DeviceMemory)],
		per_call_maps: &[(vk::DeviceMemory, *mut c_void)],
		pool: vk::DescriptorPool,
	) {
		unsafe {
			for (mem, _p) in per_call_maps {
				self.device.unmap_memory(*mem);
			}
			for (b, m) in per_call {
				self.device.destroy_buffer(*b, None);
				self.device.free_memory(*m, None);
			}
			self.device.destroy_descriptor_pool(pool, None);
		}
	}

	/// `matmul_batch_seq_end`, but each job's `[seq x out]` result is
	/// copied into `dsts[i] = (ptr, ld, col0, out)` in place (see
	/// `run_batch_end_into`).
	///
	/// # Safety
	/// As `run_batch_end_into`.
	pub unsafe fn matmul_batch_seq_end_into(
		&self,
		p: PendingSeq,
		dsts: &[(usize, usize, usize, usize)],
	) -> Option<()> {
		let PendingSeq {
			cmds,
			follow,
			bank,
			items,
			per_call,
			per_call_maps,
			pool,
			..
		} = p;
		let r = unsafe { self.run_batch_end_into(&cmds, follow, bank, &items, dsts) };
		self.free_per_call(&per_call, &per_call_maps, pool);
		r
	}

	/// Wait for a `matmul_batch_seq_begin` submission, read the outputs
	/// back and release the call's per-call resources.
	pub fn matmul_batch_seq_end(&self, p: PendingSeq) -> Option<Vec<Vec<f32>>> {
		let PendingSeq {
			cmds,
			follow,
			bank,
			items,
			per_call,
			per_call_maps,
			pool,
			t_pack,
			t_alloc,
			t_run,
			quant_secs,
		} = p;
		let outputs = match self.wait_cmds(&cmds, follow, bank) {
			Ok(()) => self.read_batch(&items, t_run),
			Err(_) => None,
		};

		let t_free = std::time::Instant::now();
		self.free_per_call(&per_call, &per_call_maps, pool);
		if gpu_timing() {
			if let Ok(mut sp) = self.stream.lock() {
				if sp.bytes > 0 {
					// `wait` is what the upload actually cost: the DMA runs
					// while this call packs `x` and allocates, so anything
					// it did not cover shows up here and nowhere else.
					eprintln!(
							"[GPUT] stream repack {:7.2} ms ({:8.1} MB, {:5.2} GB/s)   wait {:7.2} ms   {} slot (re)allocs",
							sp.pack_secs * 1e3,
							sp.bytes as f64 / 1e6,
							sp.bytes as f64 / sp.pack_secs / 1e9,
							sp.copy_secs * 1e3,
							sp.allocs,
						);
					sp.bytes = 0;
					sp.secs = 0.0;
					sp.pack_secs = 0.0;
					sp.copy_secs = 0.0;
				}
			}
			eprintln!(
					"[GPUT] pack {:7.2} ms (q8_K {:6.2} + stage {:6.2})   alloc {:7.2} ms ({} host (re)allocs)   run_batch {:7.2} ms   free {:7.2} ms",
					(t_alloc - t_pack).as_secs_f64() * 1e3,
					quant_secs * 1e3,
					(t_alloc - t_pack).as_secs_f64() * 1e3 - quant_secs * 1e3,
					(t_run - t_alloc).as_secs_f64() * 1e3,
					self.host_pool_allocs(),
					(t_free - t_run).as_secs_f64() * 1e3,
					t_free.elapsed().as_secs_f64() * 1e3,
				);
		}

		outputs
	}
}
