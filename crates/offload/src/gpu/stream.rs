use super::*;

/// A VRAM slot the prefill path streams one weight through.
///
/// Slots are allocated once and reused. That matters: `create_gpu_buffer` +
/// `allocate_memory` + `map_memory` for a 50-73 MB weight costs single-digit
/// milliseconds, and a streaming prefill touches ~500 weights per chunk; paying
/// allocation per weight would cost more than the DMA it wraps.
pub(crate) struct StreamSlot {
	dev_buf: vk::Buffer,
	dev_mem: vk::DeviceMemory,
	stg_buf: vk::Buffer,
	stg_mem: vk::DeviceMemory,
	stg_ptr: MappedPtr,
	cap: u64,
	/// Which weight the slot currently holds, and how many of its bytes, so a
	/// weight used twice in a row is not re-uploaded. The length is part of
	/// the key: a split streams a row *range* whose end moves with the chunk
	/// length, and a longer range must not count as held by a shorter one.
	holds: Option<(usize, WeightId, usize)>,
	/// Prefill banks (bit per bank) whose batch in flight reads this slot:
	/// another bank's `stream_bind` must not upload into it.
	pins: u8,
}

/// Round-robin pool of streaming slots.
///
/// **Why streaming at all.** The prefill GEMMs beat the CPU by ~1.4-1.6x, but
/// only 22 of 65 layers' weights fit in 6 GiB, so two thirds of the GEMMs stayed
/// on the CPU and the end-to-end win was +6 % instead of the ~1.3x the kernels
/// support. Residency is the wrong model for prefill: a chunk reads each weight
/// **once**, so the weight only has to be in VRAM for the length of one matmul.
///
/// **Why it pays.** Measured on this host, an FFN block (up+gate+down, 173 MB)
/// uploads in ~33 ms while its GEMMs at `seq = 2048` take ~680 ms: under 5 %.
/// It does *not* pay on short chunks, where the same 33 ms buys much less
/// compute; `MIN_SEQ` is that cutoff.
pub(crate) struct StreamPool {
	slots: Vec<StreamSlot>,
	next: usize,
	/// Bytes uploaded and slots (re)allocated, for the `QWEN_GPU_TIMING` line.
	pub(crate) bytes: u64,
	pub(crate) allocs: u64,
	pub(crate) secs: f64,
	/// `secs` split into its two halves: the host copy/repack into the staging
	/// map, and the submit + DMA that follows it. They are separated because
	/// they need *different* fixes (the first is a CPU memcpy, the second is
	/// the copy engine) and until they are measured apart it is not obvious
	/// which one the ~33 ms per FFN block actually is.
	pub(crate) pack_secs: f64,
	pub(crate) copy_secs: f64,
	/// Command buffer of a transfer submission that has been issued but not yet
	/// waited for. `Some` between `stream_bind` and `stream_sync`.
	pending: Option<vk::CommandBuffer>,
	/// The repack+submit thread, and whether it currently owns a batch.
	///
	/// `Some(worker)` once the first streamed batch has been bound; `None` when
	/// the worker is disabled (`QWEN_GPU_STREAM_WORKER=0`, or no dedicated
	/// transfer queue family) and the repack runs inline as it used to.
	worker: Option<XferWorker>,
	/// A batch is in the worker's hands: `stream_drain` must collect its result
	/// before it may touch `pending` or any slot.
	worker_busy: bool,
	/// The worker was asked for once and refused (toggled off, no transfer
	/// queue family, or the thread would not spawn). Do not keep retrying.
	worker_disabled: bool,
	/// Outcome of the last batch's follow-up compute submission (see
	/// `Follow`): `Some(false)` if the worker could not submit it, so the
	/// caller must not wait on `fence` for it. Set by `stream_drain`.
	pub(crate) follow_ok: [Option<bool>; 2],
	/// Bank of the batch last handed to the worker.
	cur_bank: usize,
	/// Owner of the bytes an outstanding prefetch upload reads
	/// (`stream_prefetch`): the repack runs after the hint's caller has
	/// returned, so the weights are kept alive until the upload is drained.
	keep: Option<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
	/// The follow-up channel of an outstanding prefetch upload: the call
	/// whose rows it holds sends its streamed command buffer through it
	/// (`stream_bind`), so that buffer runs behind the upload on the GPU and
	/// neither the call nor its CPU rows wait for the upload on the host.
	/// Anything else that needs the worker sends `None` first.
	pf_follow: Option<FollowTx>,
	/// A `None` was sent through `pf_follow`: the next collected batch's
	/// follow-up outcome belongs to no call.
	pf_unused: bool,
}

/// The compute submission that reads a batch's streamed weights, handed to
/// the repack thread so it can submit it right after the upload.
///
/// Before this the caller waited for the upload on the host (`stream_sync`)
/// before it submitted anything, so at `seq = 2048` every split matmul
/// started its CPU share ~2-8 ms late (the repack (one thread, ~7 GB/s) plus
/// the DMA, ~0.6 s per 2048-token chunk) and the GPU's *resident* rows,
/// which need no upload at all, waited with it. Now the caller submits the
/// resident rows at once, records the streamed rows' dispatches, and sends
/// that command buffer here; the worker submits it waiting on `sem`, which
/// its own transfer submission signals. A binary semaphore is enough because
/// one thread submits both the signal and the wait, in that order.
struct Follow {
	/// The recorded command buffer, or `None` if the caller gave up (the
	/// sender dropped): the worker then submits an empty batch that only waits
	/// on `sem`, so the semaphore is unsignaled again for the next batch.
	rx: std::sync::mpsc::Receiver<Option<vk::CommandBuffer>>,
	queue: vk::Queue,
	fence: vk::Fence,
	sem: vk::Semaphore,
}

/// Caller half of a `Follow`.
pub(crate) type FollowTx = std::sync::mpsc::Sender<Option<vk::CommandBuffer>>;

/// One batch of weight uploads handed to the repack thread.
///
/// The source and destination are carried as `usize`, not slices: `raw` borrows
/// the model's weight bytes for the length of the `matmul_batch_seq` call, and
/// the worker is joined (`stream_drain`) before that call returns, so the
/// referent always outlives the thread, but the borrow checker cannot see that
/// through a channel. This is the same lifetime erasure `par_memcpy` and
/// `quant::parallel::dispatch` already make for the same reason.
struct XferJob {
	/// `(staging map, source bytes, length, kind)` per weight to upload.
	packs: Vec<(usize, usize, usize, GpuKind)>,
	/// `(staging buffer, device buffer, bytes)` per copy to record.
	copies: Vec<(vk::Buffer, vk::Buffer, u64)>,
	/// Spread the repack across the shared thread pool (`QWEN_GPU_STREAM_WORKER=par`).
	par: bool,
	/// A compute submission to issue once the upload is submitted.
	follow: Option<Follow>,
}

/// Handle to the repack+submit thread.
///
/// **Why a thread and not just a queue.** The DMA was taken off the critical
/// path by submitting it to the transfer queue and awaiting it at the dispatch,
/// which left the host `memcpy` into the staging map: ~7 ms per
/// FFN block even threaded across the pool, because it happens *before* the copy
/// can be recorded. No amount of queue overlap removes that; it is CPU work, and
/// the only way to stop paying for it serially is to run it beside the caller's
/// own host work (quantizing `x`, allocating this call's buffers, writing its
/// descriptor sets: 11-39 ms at `seq = 2048`).
///
/// The thread owns cloned Vulkan handles rather than a borrow of `GpuDevice`:
/// `ash::Device` is a cheap clonable handle wrapper, and the queue, command pool
/// and fence are plain `u64` handles. The device outlives the thread because
/// `free_stream_slots` (called from `Drop`) shuts it down and joins first.
struct XferWorker {
	tx: std::sync::mpsc::Sender<XferJob>,
	/// `(transfer cmd, repack secs, follow-up submitted)`.
	rx: std::sync::mpsc::Receiver<(Option<vk::CommandBuffer>, f64, Option<bool>)>,
	handle: Option<std::thread::JoinHandle<()>>,
}

impl XferWorker {
	/// Stop the thread and wait for it. Called before the device is destroyed.
	fn shutdown(&mut self) {
		// Dropping the sender is what ends the worker's `for job in rx` loop.
		let (dead, _) = std::sync::mpsc::channel();
		let _ = std::mem::replace(&mut self.tx, dead);
		if let Some(h) = self.handle.take() {
			let _ = h.join();
		}
	}
}

impl StreamPool {
	pub(crate) fn new() -> StreamPool {
		StreamPool {
			slots: Vec::new(),
			next: 0,
			bytes: 0,
			allocs: 0,
			secs: 0.0,
			pack_secs: 0.0,
			copy_secs: 0.0,
			pending: None,
			worker: None,
			worker_busy: false,
			worker_disabled: false,
			follow_ok: [None, None],
			cur_bank: 0,
			keep: None,
			pf_follow: None,
			pf_unused: false,
		}
	}
}

impl GpuDevice {
	/// Stream a layer's weights through VRAM for prefill instead of
	/// requiring them to be resident. **On by default**;
	/// `QWEN_GPU_STREAM=0` turns it off. Read once.
	///
	/// Shipping it on is what makes GPU prefill cover the whole model
	/// instead of the 22 of 65 layers that fit in 6 GiB, and it is gated by
	/// `stream_min_seq` so a chunk too short to amortize the upload still
	/// stays on the CPU.
	pub fn stream_enabled() -> bool {
		static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
		*ON.get_or_init(|| env_on("QWEN_GPU_STREAM", true))
	}

	/// Chunk length below which streaming is refused, so a short prompt is
	/// not charged an upload it cannot amortize (`QWEN_GPU_STREAM_MIN_SEQ`).
	///
	/// The default is a measured break-even, not a guess. `gpubench
	/// --crossover` times one FFN block on both sides at each chunk length
	/// (against the *rewritten* 1836 GFLOP/s CPU GEMM, not the
	/// 966 the older numbers here assumed):
	///
	/// | seq | 64 | 128 | 256 | 512 | 1024 | 2048 |
	/// |---|---|---|---|---|---|---|
	/// | streamed x CPU | 0.63 | 0.82 | **1.05** | 1.19 | 1.25 | 1.38 |
	///
	/// 256 is where it crosses 1.0. It is also at or above `GEMM_BN`, so a
	/// streamed matmul always runs the tiled GEMM rather than the batched
	/// GEMV: streaming the GEMV would be strictly worse than the CPU.
	pub fn stream_min_seq() -> usize {
		static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
		*N.get_or_init(|| {
			std::env::var("QWEN_GPU_STREAM_MIN_SEQ")
				.ok()
				.and_then(|s| s.parse().ok())
				.unwrap_or(256)
		})
	}

	/// `QWEN_GPU_STREAM_SYNC=1`: wait for a batch's upload inside
	/// `stream_bind` instead of just before the dispatch that reads it.
	///
	/// That is the original behavior, kept because it is the only honest
	/// way to measure the overlap on this host: prefill here varies by more
	/// than the effect between runs, so the A and the B have to be the same
	/// binary on the same load, one environment variable apart.
	pub(crate) fn stream_sync_mode() -> bool {
		static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
		*ON.get_or_init(|| std::env::var("QWEN_GPU_STREAM_SYNC").is_ok())
	}

	/// `QWEN_GPU_STREAM_WORKER=0`: repack and submit inline on the calling
	/// thread, the way `stream_bind` did before the worker existed.
	///
	/// Kept for the same reason `stream_sync_mode` is: the effect is a few
	/// milliseconds per FFN block and this host's run-to-run spread is
	/// larger than that, so the A and the B have to be the same binary on
	/// the same load, one variable apart.
	pub(crate) fn stream_worker_enabled() -> bool {
		static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
		*ON.get_or_init(|| env_on("QWEN_GPU_STREAM_WORKER", true))
	}

	/// `QWEN_GPU_STREAM_FOLLOW=0`: wait for a batch's upload on the host
	/// before submitting the dispatches that read it, and submit the
	/// resident ones with them; the behavior before `Follow`, kept as
	/// its A/B.
	pub(crate) fn stream_follow_enabled() -> bool {
		static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
		*ON.get_or_init(|| env_on("QWEN_GPU_STREAM_FOLLOW", true))
	}

	/// `QWEN_GPU_STREAM_WORKER=par`: let the repack thread use the shared
	/// pool instead of copying on its own thread. Off by default: see
	/// `stage_copy_with`.
	pub(crate) fn stream_worker_par() -> bool {
		static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
		*ON.get_or_init(|| {
			std::env::var("QWEN_GPU_STREAM_WORKER")
				.map(|v| v.trim().eq_ignore_ascii_case("par"))
				.unwrap_or(false)
		})
	}

	/// Spawn the repack+submit thread, or return `None` to keep the work
	/// inline.
	///
	/// Without a dedicated transfer queue family the worker would submit to
	/// the *compute* queue from a second thread. A `VkQueue` must be
	/// externally synchronized, and while `stream_sync` does order this
	/// thread against `run_batch`'s submission, that ordering is a property
	/// of the current call sequence rather than something enforced here: so
	/// the worker simply does not engage on such a device.
	fn spawn_xfer_worker(&self) -> Option<XferWorker> {
		if !Self::stream_worker_enabled() {
			return None;
		}
		let queue = self.xfer_queue?;
		let device = self.device.clone();
		let pool = self.xfer_pool;
		let fence = self.xfer_fence;
		let (tx, job_rx) = std::sync::mpsc::channel::<XferJob>();
		let (done_tx, rx) =
			std::sync::mpsc::channel::<(Option<vk::CommandBuffer>, f64, Option<bool>)>();
		let handle = std::thread::Builder::new()
			.name("qwen-stream-xfer".into())
			.spawn(move || {
				for job in job_rx {
					let t0 = std::time::Instant::now();
					repack_batch(&job);
					let secs = t0.elapsed().as_secs_f64();
					let sem = job.follow.as_ref().map(|f| f.sem);
					let cmd = record_and_submit(&device, pool, queue, fence, &job.copies, sem);
					let follow = job
						.follow
						.map(|f| submit_follow(&device, &f, cmd.is_some()));
					if done_tx.send((cmd, secs, follow)).is_err() {
						return;
					}
				}
			})
			.ok()?;
		Some(XferWorker {
			tx,
			rx,
			handle: Some(handle),
		})
	}

	/// Bytes a weight occupies in VRAM. Q6_K is repacked to a dword-aligned
	/// 212-byte stride on the way in (`Q6K_BLOCK_PAD`), so it is larger there
	/// than on disk; everything else is uploaded as-is.
	pub(crate) fn vram_len(kind: GpuKind, raw_len: usize) -> u64 {
		match kind {
			GpuKind::Q6K => ((raw_len / Q6K_BLOCK) * Q6K_BLOCK_PAD) as u64,
			_ => ((raw_len + 3) & !3) as u64,
		}
	}

	/// Make `needed` weights readable by the shader, uploading any that are
	/// not already in a slot, and return one buffer per entry **in order**.
	///
	/// The upload is **issued, not awaited**: the copies go to the transfer
	/// queue in one submission and the caller waits for them with
	/// `stream_sync` just before it records the dispatch that reads them.
	/// Everything between (quantizing and uploading `x`, allocating this
	/// call's buffers, writing its descriptor sets) then run beside the
	/// DMA instead of after it, and at `seq = 2048` that host work is longer
	/// than the DMA, so the upload effectively costs nothing but its host
	/// repack. Waiting here instead made it pure serial time.
	///
	/// All entries get *distinct* slots, because the caller binds them into
	/// one submission: reusing a slot inside a batch would have the second
	/// weight overwrite the first before the dispatch reads it. That is why
	/// the pool grows to `needed.len()` rather than being a fixed ring.
	///
	/// Returns `None` if a slot cannot be allocated (VRAM exhausted), so the
	/// caller falls back to the CPU instead of dispatching against a buffer
	/// that does not exist.
	///
	/// With `follow`, and when the upload goes to the repack thread, also
	/// returns a `FollowTx`: the caller sends the command buffer that reads
	/// the streamed weights through it instead of calling `stream_sync`
	/// and submitting it itself (see `Follow`). It must then not wait on
	/// `fence` before `stream_sync` has collected the batch.
	pub(crate) fn stream_bind(
		&self,
		needed: &[(usize, WeightId, &[u8], GpuKind)],
		follow: bool,
		bank: usize,
	) -> Option<(Vec<vk::Buffer>, Option<FollowTx>)> {
		let t0 = std::time::Instant::now();
		let mut pool = self.stream.lock().ok()?;
		let n = needed.len();
		// This bank's previous batch has been waited for; the other
		// bank's may still be reading its slots.
		let other = 1u8 << (1 - bank.min(1));
		for s in pool.slots.iter_mut() {
			s.pins &= !(1u8 << bank.min(1));
		}

		// Pass 1: keep whatever is already resident in a slot. `claimed`
		// stops pass 2 from handing the same slot to two entries.
		let mut assign: Vec<Option<usize>> = vec![None; n];
		let mut claimed: Vec<bool> = pool.slots.iter().map(|s| s.pins & other != 0).collect();
		for (i, (il, id, raw, kind)) in needed.iter().enumerate() {
			let want = Self::vram_len(*kind, raw.len());
			for (si, slot) in pool.slots.iter().enumerate() {
				// A slot the other bank reads is shared, not claimed: its
				// upload has landed by the `stream_drain` below.
				let shared = slot.pins & other != 0;
				if (!claimed[si] || shared)
					&& slot.holds == Some((*il, *id, raw.len()))
					&& slot.cap >= want
				{
					claimed[si] = true;
					assign[i] = Some(si);
					break;
				}
			}
		}

		// Pass 2: everything else takes the next unclaimed slot, growing the
		// pool if this batch is wider than it has ever been.
		for i in 0..n {
			if assign[i].is_some() {
				continue;
			}
			let free = (0..pool.slots.len()).any(|si| !claimed[si]);
			let si = loop {
				if pool.slots.len() < n || !free {
					let si = pool.slots.len();
					pool.slots.push(StreamSlot {
						dev_buf: vk::Buffer::null(),
						dev_mem: vk::DeviceMemory::null(),
						stg_buf: vk::Buffer::null(),
						stg_mem: vk::DeviceMemory::null(),
						stg_ptr: MappedPtr(ptr::null_mut()),
						cap: 0,
						holds: None,
						pins: 0,
					});
					claimed.push(true);
					break si;
				}
				let si = pool.next % pool.slots.len();
				pool.next = pool.next.wrapping_add(1);
				if !claimed[si] {
					claimed[si] = true;
					break si;
				}
			};
			assign[i] = Some(si);
		}

		// Any transfer still in flight from the previous batch reads from
		// these same staging maps and may target these same slots, so it has
		// to be finished before anything is written into them. In the normal
		// flow `stream_sync` already consumed it before the dispatch that
		// followed; this covers the paths that bailed out instead.
		//
		// Except a prefetched batch (`stream_prefetch`) that holds every
		// entry: its follow-up then carries this call's streamed rows, so
		// they run behind its upload on the GPU instead of after a host
		// wait. (Bank 0 only: the follow-up signals bank 0's fence.)
		if let Some(tx) = pool.pf_follow.take() {
			let all = needed.iter().enumerate().all(|(i, (il, id, raw, _))| {
				pool.slots[assign[i].unwrap()].holds == Some((*il, *id, raw.len()))
			});
			if all && follow && bank == 0 {
				pool.cur_bank = 0;
				pool.follow_ok[0] = None;
				for a in &assign {
					pool.slots[a.unwrap()].pins |= 1;
				}
				let out = assign
					.iter()
					.map(|a| pool.slots[a.unwrap()].dev_buf)
					.collect();
				pool.secs += t0.elapsed().as_secs_f64();
				return Some((out, Some(tx)));
			}
			let _ = tx.send(None);
			pool.pf_unused = true;
		}
		self.stream_drain(&mut pool);

		let mut copies: Vec<(vk::Buffer, vk::Buffer, u64)> = Vec::new();
		let mut staged: Vec<(usize, (usize, WeightId, usize))> = Vec::new();
		let mut packs: Vec<(usize, usize, usize, GpuKind)> = Vec::new();
		for (i, (il, id, raw, kind)) in needed.iter().enumerate() {
			let si = assign[i].unwrap();
			if pool.slots[si].holds == Some((*il, *id, raw.len())) {
				continue; // pass-1 hit: already uploaded
			}
			let want = Self::vram_len(*kind, raw.len());
			if pool.slots[si].cap < want {
				self.grow_slot(&mut pool, si, want)?;
			}

			packs.push((
				pool.slots[si].stg_ptr.0 as usize,
				raw.as_ptr() as usize,
				raw.len(),
				*kind,
			));
			// A weight counts as held only once its copy has been
			// *submitted*; until then a later batch must not believe it is
			// in VRAM.
			pool.slots[si].holds = None;
			copies.push((pool.slots[si].stg_buf, pool.slots[si].dev_buf, want));
			staged.push((si, (*il, *id, raw.len())));
			pool.bytes += want;
		}

		let mut ret_follow: Option<FollowTx> = None;
		if !copies.is_empty() {
			// One submission for the whole batch, on the transfer queue, and
			// **not waited on here**. The caller quantizes and uploads `x`,
			// allocates this call's buffers and writes its descriptor sets
			// before it needs the weights (~25 ms of host work at
			// seq = 2048 against ~14 ms of DMA) so by the time
			// `stream_sync` runs the copy has usually already landed.
			// Waiting here instead is what made the upload pure serial time.
			//
			// The repack into the staging map goes to the worker thread with
			// it, so the same host work also covers the `memcpy` the queue
			// could not. `holds` is set optimistically here rather than
			// after the submit: `stream_drain` is the only thing that can
			// observe a slot again, and it always collects the worker first.
			if pool.worker.is_none() && !pool.worker_disabled {
				match self.spawn_xfer_worker() {
					Some(w) => pool.worker = Some(w),
					None => pool.worker_disabled = true,
				}
			}
			// On a send error the job comes back in the `Err`, so `packs` is
			// handed back rather than lost: the inline fallback below still
			// has everything it needs.
			let mut follow_tx = None;
			let follow = (follow && !Self::stream_sync_mode() && Self::stream_follow_enabled())
				.then(|| {
					let (tx, rx) = std::sync::mpsc::channel();
					follow_tx = Some(tx);
					Follow {
						rx,
						queue: self.queue,
						fence: self.bank_fence(bank),
						sem: self.xfer_sem,
					}
				});
			let sent = match &pool.worker {
				Some(w) => match w.tx.send(XferJob {
					packs: std::mem::take(&mut packs),
					copies: copies.clone(),
					par: Self::stream_worker_par(),
					follow,
				}) {
					Ok(()) => {
						ret_follow = follow_tx.take();
						pool.cur_bank = bank.min(1);
						pool.follow_ok[bank.min(1)] = None;
						true
					}
					Err(std::sync::mpsc::SendError(job)) => {
						packs = job.packs;
						false
					}
				},
				None => false,
			};
			if sent {
				pool.worker_busy = true;
			} else {
				// No worker (disabled, or it died): repack and submit here.
				let t_pack = std::time::Instant::now();
				for (dst, src, len, kind) in &packs {
					let raw = unsafe { std::slice::from_raw_parts(*src as *const u8, *len) };
					stage_copy(*dst as *mut u8, raw, *kind);
				}
				pool.pack_secs += t_pack.elapsed().as_secs_f64();
				pool.pending = Some(self.stream_submit(&copies)?);
			}
			for (si, h) in staged {
				pool.slots[si].holds = Some(h);
			}
			if Self::stream_sync_mode() {
				let t0 = std::time::Instant::now();
				self.stream_drain(&mut pool);
				pool.copy_secs += t0.elapsed().as_secs_f64();
			}
		}

		for a in &assign {
			pool.slots[a.unwrap()].pins |= 1u8 << bank.min(1);
		}
		let out = assign
			.iter()
			.map(|a| pool.slots[a.unwrap()].dev_buf)
			.collect();
		pool.secs += t0.elapsed().as_secs_f64();
		Some((out, ret_follow))
	}

	/// (Re)allocate slot `si` to hold `want` bytes. Slot sizes converge
	/// after the first few weights, so this runs a handful of times per
	/// process, not per weight. `None` if VRAM is exhausted.
	fn grow_slot(&self, pool: &mut StreamPool, si: usize, want: u64) -> Option<()> {
		self.free_slot(&mut pool.slots[si]);
		let (dev_buf, dev_mem) = self.try_create_buffer_shared(
			want,
			vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
			vk::MemoryPropertyFlags::DEVICE_LOCAL,
			true,
		)?;
		let (stg_buf, stg_mem) = match self.try_create_buffer_shared(
			want,
			vk::BufferUsageFlags::TRANSFER_SRC,
			vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
			true,
		) {
			Some(v) => v,
			None => {
				unsafe {
					self.device.destroy_buffer(dev_buf, None);
					self.device.free_memory(dev_mem, None);
				}
				return None;
			}
		};
		let stg_ptr = MappedPtr(self.map(stg_mem, want as usize));
		pool.slots[si] = StreamSlot {
			dev_buf,
			dev_mem,
			stg_buf,
			stg_mem,
			stg_ptr,
			cap: want,
			holds: None,
			pins: 0,
		};
		pool.allocs += 1;
		Some(())
	}

	/// Upload `needed` into free slots *ahead of* the call that will read
	/// them, so that call's `stream_bind` finds them held and its streamed
	/// rows do not wait for a repack and a DMA first
	/// (`OffloadManager::prefetch_hint`). Slots a batch in flight has
	/// pinned are left alone; nothing is pinned here.
	///
	/// One upload is outstanding at a time: with `block` false this
	/// returns `false`, doing nothing, while the previous one (the current
	/// call's own) has not landed; with `block` it waits for it. Also
	/// `false` if a slot cannot be allocated. `keep` owns the bytes of
	/// `needed` until the upload has been drained.
	pub(crate) fn stream_prefetch(
		&self,
		needed: &[(usize, WeightId, &[u8], GpuKind)],
		keep: std::sync::Arc<dyn std::any::Any + Send + Sync>,
		block: bool,
	) -> bool {
		let Ok(mut pool) = self.stream.lock() else {
			return false;
		};
		if self.lost.load(Ordering::SeqCst) || (!block && !self.stream_idle(&mut pool)) {
			return false;
		}
		self.stream_drain(&mut pool);
		let mut claimed: Vec<bool> = pool.slots.iter().map(|s| s.pins != 0).collect();
		let mut packs: Vec<(usize, usize, usize, GpuKind)> = Vec::new();
		let mut copies: Vec<(vk::Buffer, vk::Buffer, u64)> = Vec::new();
		let mut staged: Vec<(usize, (usize, WeightId, usize))> = Vec::new();
		for (il, id, raw, kind) in needed {
			let key = (*il, *id, raw.len());
			let want = Self::vram_len(*kind, raw.len());
			if let Some(si) = (0..pool.slots.len())
				.find(|&si| pool.slots[si].holds == Some(key) && pool.slots[si].cap >= want)
			{
				claimed[si] = true;
				continue;
			}
			// A free slot already large enough, round robin; else any free
			// slot, grown; else a new one.
			let n = pool.slots.len();
			let start = pool.next;
			let free = |big: bool| {
				(0..n)
					.map(|k| (start + k) % n)
					.find(|&si| !claimed[si] && (!big || pool.slots[si].cap >= want))
			};
			let si = match free(true).or_else(|| free(false)) {
				Some(si) => si,
				None => {
					pool.slots.push(StreamSlot {
						dev_buf: vk::Buffer::null(),
						dev_mem: vk::DeviceMemory::null(),
						stg_buf: vk::Buffer::null(),
						stg_mem: vk::DeviceMemory::null(),
						stg_ptr: MappedPtr(ptr::null_mut()),
						cap: 0,
						holds: None,
						pins: 0,
					});
					claimed.push(false);
					n
				}
			};
			pool.next = si + 1;
			claimed[si] = true;
			if pool.slots[si].cap < want && self.grow_slot(&mut pool, si, want).is_none() {
				for (si, _) in staged {
					pool.slots[si].holds = None;
				}
				return false;
			}
			packs.push((
				pool.slots[si].stg_ptr.0 as usize,
				raw.as_ptr() as usize,
				raw.len(),
				*kind,
			));
			pool.slots[si].holds = None;
			copies.push((pool.slots[si].stg_buf, pool.slots[si].dev_buf, want));
			staged.push((si, key));
			pool.bytes += want;
		}
		if copies.is_empty() {
			return true;
		}
		if pool.worker.is_none() && !pool.worker_disabled {
			match self.spawn_xfer_worker() {
				Some(w) => pool.worker = Some(w),
				None => pool.worker_disabled = true,
			}
		}
		let mut pf_tx = None;
		let follow = (!Self::stream_sync_mode() && Self::stream_follow_enabled()).then(|| {
			let (tx, rx) = std::sync::mpsc::channel();
			pf_tx = Some(tx);
			Follow {
				rx,
				queue: self.queue,
				fence: self.bank_fence(0),
				sem: self.xfer_sem,
			}
		});
		let sent = match &pool.worker {
			Some(w) => match w.tx.send(XferJob {
				packs: std::mem::take(&mut packs),
				copies: copies.clone(),
				par: Self::stream_worker_par(),
				follow,
			}) {
				Ok(()) => true,
				Err(std::sync::mpsc::SendError(job)) => {
					packs = job.packs;
					false
				}
			},
			None => false,
		};
		if sent {
			pool.worker_busy = true;
			pool.pf_follow = pf_tx;
			pool.pf_unused = false;
		} else {
			let t_pack = std::time::Instant::now();
			for (dst, src, len, kind) in &packs {
				let raw = unsafe { std::slice::from_raw_parts(*src as *const u8, *len) };
				stage_copy(*dst as *mut u8, raw, *kind);
			}
			pool.pack_secs += t_pack.elapsed().as_secs_f64();
			match self.stream_submit(&copies) {
				Some(c) => pool.pending = Some(c),
				None => return false,
			}
		}
		for (si, h) in staged {
			pool.slots[si].holds = Some(h);
		}
		pool.keep = Some(keep);
		true
	}

	/// Whether no weight upload is outstanding, without blocking: collects
	/// a finished worker batch (as `stream_drain` would) and polls the
	/// transfer fence.
	fn stream_idle(&self, pool: &mut StreamPool) -> bool {
		if pool.worker_busy {
			match pool.worker.as_ref().map(|w| w.rx.try_recv()) {
				Some(Ok(msg)) => {
					pool.worker_busy = false;
					self.stream_collect(pool, Some(msg));
				}
				Some(Err(std::sync::mpsc::TryRecvError::Empty)) => return false,
				// A dead worker: `stream_drain` reports it.
				_ => return true,
			}
		}
		match pool.pending {
			None => true,
			Some(_) => unsafe { self.device.get_fence_status(self.xfer_fence) }.unwrap_or(true),
		}
	}

	/// Record every copy of one batch into a single command buffer and submit
	/// it to the transfer queue **without waiting**. Returns the in-flight
	/// command buffer, or `None` (device marked lost) if the submission
	/// failed. The caller stores it in `pool.pending`; taking the pool lock
	/// here would deadlock, since `stream_bind` holds it across this call.
	fn stream_submit(&self, copies: &[(vk::Buffer, vk::Buffer, u64)]) -> Option<vk::CommandBuffer> {
		let r = record_and_submit(
			&self.device,
			self.xfer_pool,
			self.xfer_queue.unwrap_or(self.queue),
			self.xfer_fence,
			copies,
			None,
		);
		if r.is_none() {
			self.lost.store(true, Ordering::SeqCst);
		}
		r
	}

	/// Wait for the in-flight weight transfer, if any. Called immediately
	/// before recording the dispatch that reads those weights.
	///
	/// The time spent here is the part of the upload the host work did *not*
	/// cover, so it is charged to `copy_secs`: that counter now reports the
	/// residual serial cost of streaming rather than the raw DMA time.
	pub(crate) fn stream_sync(&self) {
		if let Ok(mut pool) = self.stream.lock() {
			// `worker_busy` counts as pending: with the repack thread the
			// submission has not been recorded yet, so `pending` is still
			// `None` at this point and testing it alone would let the
			// dispatch run against buffers nothing had copied into.
			if pool.pending.is_none() && !pool.worker_busy {
				return;
			}
			let t0 = std::time::Instant::now();
			self.stream_drain(&mut pool);
			pool.copy_secs += t0.elapsed().as_secs_f64();
		}
	}

	/// Release bank `bank`'s stream slots once its batch has completed.
	pub(crate) fn stream_unpin(&self, bank: usize) {
		if let Ok(mut pool) = self.stream.lock() {
			for s in pool.slots.iter_mut() {
				s.pins &= !(1u8 << bank.min(1));
			}
		}
	}

	fn stream_drain(&self, pool: &mut StreamPool) {
		// Collect the worker's batch first: until it answers, its repack may
		// still be writing into these staging maps and its submission has
		// not been recorded, so `pending` is not yet the whole story.
		// A prefetch's worker waits for its follow-up: release it.
		if let Some(tx) = pool.pf_follow.take() {
			let _ = tx.send(None);
			pool.pf_unused = true;
		}
		if pool.worker_busy {
			pool.worker_busy = false;
			let msg = pool.worker.as_ref().and_then(|w| w.rx.recv().ok());
			self.stream_collect(pool, msg);
		}
		self.stream_finish(pool);
	}

	/// Take a worker batch's result: its transfer command buffer (now
	/// `pending`) and its follow-up's outcome. `None`: the thread died.
	#[allow(clippy::type_complexity)]
	fn stream_collect(
		&self,
		pool: &mut StreamPool,
		msg: Option<(Option<vk::CommandBuffer>, f64, Option<bool>)>,
	) {
		{
			match msg {
				Some((cmd, pack_secs, follow)) => {
					pool.pack_secs += pack_secs;
					if std::mem::take(&mut pool.pf_unused) {
						// An unclaimed prefetch's empty follow-up.
					} else if follow.is_some() {
						let b = pool.cur_bank;
						pool.follow_ok[b] = follow;
					}
					if follow == Some(false) {
						eprintln!("offload: streamed-weight dispatch submission failed; disabling the GPU path");
						self.lost.store(true, Ordering::SeqCst);
					}
					match cmd {
						Some(c) => pool.pending = Some(c),
						None => {
							eprintln!(
								"offload: weight upload submission failed; disabling the GPU path"
							);
							self.lost.store(true, Ordering::SeqCst);
						}
					}
				}
				// The thread died mid-batch. Its staging writes are of
				// unknown completeness, so nothing in a slot can be trusted.
				None => {
					eprintln!("offload: weight upload thread stopped; disabling the GPU path");
					self.lost.store(true, Ordering::SeqCst);
					pool.worker = None;
					pool.worker_disabled = true;
				}
			}
		}
	}

	/// Wait for the submitted transfer, if any, and release what it read.
	fn stream_finish(&self, pool: &mut StreamPool) {
		let cmd = match pool.pending.take() {
			Some(c) => c,
			None => {
				if self.lost.load(Ordering::SeqCst) {
					for s in pool.slots.iter_mut() {
						s.holds = None;
					}
				}
				return;
			}
		};
		unsafe {
			match self
				.device
				.wait_for_fences(&[self.xfer_fence], true, FENCE_TIMEOUT_NS)
			{
				Ok(()) => {}
				Err(e) => {
					eprintln!(
						"offload: weight upload did not complete ({e:?}); disabling the GPU path"
					);
					self.lost.store(true, Ordering::SeqCst);
				}
			}
			self.device.free_command_buffers(self.xfer_pool, &[cmd]);
		}
		pool.keep = None;
		// What the slots hold is only trustworthy if the copy landed.
		if self.lost.load(Ordering::SeqCst) {
			for s in pool.slots.iter_mut() {
				s.holds = None;
			}
		}
	}

	/// Drain the streaming counters: `(bytes, pack_secs, copy_secs, allocs)`.
	/// Used by `gpubench --stream`, which needs the split for one call rather
	/// than the running total `QWEN_GPU_TIMING` prints.
	pub(crate) fn stream_take_stats(&self) -> (u64, f64, f64, u64) {
		match self.stream.lock() {
			Ok(mut p) => {
				let r = (p.bytes, p.pack_secs, p.copy_secs, p.allocs);
				p.bytes = 0;
				p.secs = 0.0;
				p.pack_secs = 0.0;
				p.copy_secs = 0.0;
				p.allocs = 0;
				r
			}
			Err(_) => (0, 0.0, 0.0, 0),
		}
	}

	fn free_slot(&self, s: &mut StreamSlot) {
		if s.cap == 0 {
			return;
		}
		unsafe {
			self.device.unmap_memory(s.stg_mem);
			self.device.destroy_buffer(s.stg_buf, None);
			self.device.free_memory(s.stg_mem, None);
			self.device.destroy_buffer(s.dev_buf, None);
			self.device.free_memory(s.dev_mem, None);
		}
		s.cap = 0;
		s.holds = None;
	}

	/// Drop every streaming slot (called from `Drop`).
	pub(crate) fn free_stream_slots(&self) {
		if let Ok(mut pool) = self.stream.lock() {
			self.stream_drain(&mut pool);
			// The worker holds cloned handles to the device that is about to
			// be destroyed, so it has to be gone before we return.
			if let Some(w) = pool.worker.as_mut() {
				w.shutdown();
			}
			pool.worker = None;
			for s in pool.slots.iter_mut() {
				self.free_slot(s);
			}
			pool.slots.clear();
		}
	}
}

/// Copy one batch's weights into their staging maps, on the transfer thread.
///
/// **One thread, deliberately.** The single-threaded copy reaches ~13 GB/s of
/// this host's ~25 GB/s, so the obvious move is to split it, and both ways of
/// splitting it are worse:
///
/// | repack | host GB/s | FFN block ms (best of 5, mean of rounds) |
/// |---|---|---|
/// | inline on the caller, pooled (before) | 24.8 | 582.3 |
/// | worker, `quant` pool (`QWEN_GPU_STREAM_WORKER=par`) | 24.9 | 585.0 |
/// | worker, 3 scoped threads per batch | 8.6-10.9 | 583.6 |
/// | **worker, one thread** | 12.9 | **580.4** |
///
/// The win over inline is **~0.3 %** of the block, ahead in 3 of 4 interleaved
/// pairs. That is the honest size of this lever: the repack was ~1 % of the
/// block to begin with, and the caller's host work between `stream_bind` and
/// `stream_sync` is shorter than a single-threaded 174 MB copy, so part of it
/// still lands as a wait at the sync.
///
/// The pooled variant is a wash because `quant`'s pool runs one dispatch at a
/// time process-wide, so it interleaves with the caller's own pooled work
/// instead of overlapping it. Scoped threads are worse still: a streamed prefill
/// binds a batch every few hundred milliseconds across ~65 layers, and spawning
/// per batch costs more than the bandwidth it buys; the *measured* rate falls.
/// A slower copy that genuinely runs beside the caller beats a faster one that
/// takes the caller's cores away from it.
fn repack_batch(job: &XferJob) {
	for (dst, src, len, kind) in &job.packs {
		let raw = unsafe { std::slice::from_raw_parts(*src as *const u8, *len) };
		stage_copy_with(*dst as *mut u8, raw, *kind, job.par);
	}
}

/// Record one batch's buffer copies into a fresh command buffer and submit it to
/// `queue` **without waiting**. Returns the in-flight command buffer, or `None`
/// if the submission failed.
///
/// A free function rather than a method because the repack thread runs it too,
/// and it owns cloned handles rather than a `&GpuDevice`. Nothing here touches
/// the slot pool or the `lost` flag; the caller owns both.
fn record_and_submit(
	device: &ash::Device,
	pool: vk::CommandPool,
	queue: vk::Queue,
	fence: vk::Fence,
	copies: &[(vk::Buffer, vk::Buffer, u64)],
	signal: Option<vk::Semaphore>,
) -> Option<vk::CommandBuffer> {
	let cmd = match unsafe {
		device.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
			command_pool: pool,
			level: vk::CommandBufferLevel::PRIMARY,
			command_buffer_count: 1,
			..Default::default()
		})
	} {
		Ok(v) => v[0],
		Err(_) => return None,
	};
	unsafe {
		device
			.begin_command_buffer(
				cmd,
				&vk::CommandBufferBeginInfo {
					flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
					..Default::default()
				},
			)
			.ok();
		for (src, dst, size) in copies {
			device.cmd_copy_buffer(
				cmd,
				*src,
				*dst,
				&[vk::BufferCopy {
					src_offset: 0,
					dst_offset: 0,
					size: *size,
					..Default::default()
				}],
			);
		}
		device.end_command_buffer(cmd).ok();
		device.reset_fences(&[fence]).ok();
		if device
			.queue_submit(
				queue,
				&[vk::SubmitInfo {
					command_buffer_count: 1,
					p_command_buffers: &cmd,
					signal_semaphore_count: signal.is_some() as u32,
					p_signal_semaphores: signal.as_ref().map_or(ptr::null(), |s| s as *const _),
					..Default::default()
				}],
				fence,
			)
			.is_err()
		{
			device.free_command_buffers(pool, &[cmd]);
			return None;
		}
	}
	Some(cmd)
}

/// Submit a batch's follow-up compute command buffer (see `Follow`) on the
/// compute queue, waiting on the semaphore the upload just signaled.
/// `uploaded` is whether that upload was submitted at all; if not, nothing
/// will signal the semaphore and the command buffer is not submitted.
/// Returns whether it was submitted; when the caller sent none, an empty batch
/// consumes the signal and the result is `true`.
fn submit_follow(device: &ash::Device, f: &Follow, uploaded: bool) -> bool {
	let cmd = f.rx.recv().ok().flatten();
	if !uploaded {
		return cmd.is_none();
	}
	let stage = vk::PipelineStageFlags::ALL_COMMANDS;
	let n = cmd.is_some() as u32;
	let cmd_ptr = cmd.as_ref().map_or(ptr::null(), |c| c as *const _);
	unsafe {
		if cmd.is_some() {
			device.reset_fences(&[f.fence]).ok();
		}
		device
			.queue_submit(
				f.queue,
				&[vk::SubmitInfo {
					wait_semaphore_count: 1,
					p_wait_semaphores: &f.sem,
					p_wait_dst_stage_mask: &stage,
					command_buffer_count: n,
					p_command_buffers: cmd_ptr,
					..Default::default()
				}],
				if cmd.is_some() {
					f.fence
				} else {
					vk::Fence::null()
				},
			)
			.is_ok()
	}
}

/// Copy one weight's quantized bytes into a slot's staging map, repacking Q6_K
/// to the 212-byte VRAM stride on the way (`Q6K_BLOCK_PAD`).
///
/// Threaded across the shared pool. This is the half of the upload that queue
/// overlap cannot hide (it happens before the copy is even recorded) and at
/// `seq = 2048` it was 12.4 ms of the 26.6 ms an FFN block cost: a plain
/// single-threaded `memcpy` of 174 MB into write-combined memory. Split by
/// super-block range it is a few ms. Small weights stay inline, where the
/// fork-join would cost more than the copy.
fn stage_copy(dst: *mut u8, raw: &[u8], kind: GpuKind) {
	stage_copy_with(dst, raw, kind, true)
}

/// `par == false` keeps the copy on the calling thread instead of handing it to
/// `quant::par_for`.
///
/// That is what the repack thread wants. `quant`'s pool admits one dispatch at a
/// time process-wide, so a repack that goes through it does not run *beside* the
/// caller's own pooled work (quantizing `x`, packing its staging buffer): the
/// two interleave and neither finishes sooner. Measured on one streamed FFN
/// block at `seq = 2048`, best of 5, three interleaved rounds: pooled repack on
/// the worker 584.97 ms mean against 584.2 inline, i.e. nothing; the same worker
/// copying on its own threads, 580.3.
fn stage_copy_with(dst: *mut u8, raw: &[u8], kind: GpuKind, par: bool) {
	const PAR_MIN: usize = 1 << 20;
	if kind != GpuKind::Q6K {
		// Contiguous kinds: the VRAM layout is the GGUF layout.
		if par {
			par_memcpy(dst, raw);
		} else {
			unsafe { ptr::copy_nonoverlapping(raw.as_ptr(), dst, raw.len()) };
		}
		return;
	}
	// Q6_K is block-strided in VRAM, so the split has to be by whole
	// super-blocks. The two padding bytes per block are never read (`d` is the
	// low half of dword 52), so they are left undefined rather than zeroed.
	let nb = raw.len() / Q6K_BLOCK;
	let per = PAR_MIN / Q6K_BLOCK;
	let copy_blocks = |b0: usize, b1: usize, d: usize, s: usize| {
		for b in b0..b1 {
			unsafe {
				ptr::copy_nonoverlapping(
					(s as *const u8).add(b * Q6K_BLOCK),
					(d as *mut u8).add(b * Q6K_BLOCK_PAD),
					Q6K_BLOCK,
				)
			};
		}
	};
	let (d, s) = (dst as usize, raw.as_ptr() as usize);
	if nb <= per || !par {
		copy_blocks(0, nb, d, s);
		return;
	}
	quant::par_for(nb.div_ceil(per), move |c| {
		copy_blocks(c * per, ((c + 1) * per).min(nb), d, s);
	});
}

/// `memcpy` into a host-visible mapping, split across the shared pool.
///
/// Every host-visible buffer this path writes is tens of megabytes (a weight
/// into a slot's staging map, a packed activation into its own) and a
/// single-threaded `memcpy` of that size is milliseconds the GPU spends idle.
/// Small copies stay inline, where the fork-join would cost more than the copy.
pub(crate) fn par_memcpy(dst: *mut u8, src: &[u8]) {
	const PAR_MIN: usize = 1 << 20;
	let n = src.len();
	if n < PAR_MIN {
		unsafe { ptr::copy_nonoverlapping(src.as_ptr(), dst, n) };
		return;
	}
	let (d, s) = (dst as usize, src.as_ptr() as usize);
	quant::par_for(n.div_ceil(PAR_MIN), move |c| {
		let off = c * PAR_MIN;
		let len = PAR_MIN.min(n - off);
		unsafe {
			ptr::copy_nonoverlapping((s as *const u8).add(off), (d as *mut u8).add(off), len)
		};
	});
}
