# Optimization techniques

> [!NOTE]
> Items in this file never include absolute numbers. A result
> appears at most as a relative improvement (a percentage or a factor against
> the previous state). A design parameter appears only where the technique
> cannot be reproduced without it.

This file records the optimization techniques that have proven successful, and
the ones that were measured and rejected, so that optimizing a new model path
takes fewer development cycles. Start from what already worked, and skip what
already failed unless the new path removes the reason.

Relative gains describe the engine a technique was measured on. Read them as
the order of magnitude to expect, not as a promise. Tags name that engine:
**GGUF** (quantized weights on the cores and a discrete GPU), **NPU** (the
dense model on the NPU), **Flash** (the MoE model on the NPU and iGPU for
prefill, the discrete GPU and the cores for decode) and **Image** (the
diffusion transformer and VAE).

See [Method](#method) · [CPU kernels](#cpu-kernels) ·
[GPU kernels](#gpu-kernels) · [NPU graphs](#npu-graphs) ·
[Low-precision accuracy](#low-precision-accuracy) ·
[Memory and data movement](#memory-and-data-movement) ·
[Splitting work across devices](#splitting-work-across-devices) ·
[Scheduling and pipelining](#scheduling-and-pipelining) ·
[Speculative decoding](#speculative-decoding) ·
[Serving and prefix cache](#serving-and-prefix-cache) ·
[Rejected](#rejected)

## Method

- **Find the bound first.** Decode is DRAM-bound on every engine and prefill is
  compute-bound. A lever that does not cut DRAM bytes per token does not move
  decode.
- **Take the largest measured time share next.** Then ask what bounds the
  phase: a device's queue, its kernel, or the host waiting for either.
- **Test whether a phase is critical by injecting delay.** If added delay does
  not change the total, speeding that phase up will not either (Flash).
- **Do not trust a bound taken by removing work.** Dropping an operation can
  change the work downstream and overstate the possible gain (Flash).
- **Run interleaved A/B rounds, never one run against one run.** A host can
  have per-process performance states far apart, about 35 % on ours. One binary
  and one environment variable apart is the cleanest comparison.
- **Freeze adaptive splits during comparisons.** A refit changes the work split
  and with it the logits.
- **Make speedups bit-identical where possible** (same operations in the same
  order per output) and check by dumping logits or output checksums. Give lossy
  changes a KL gate against the full-precision reference.
- **Check attention with KL over many teacher-forced positions.** A
  single-token check does not exercise it, because a softmax over one score is
  trivially exact.
- **Check an exact-in-intent change to an MoE prefill at the changed op.**
  Tokens are coupled through the expert pass, so whole-model KL moves anyway.
- **Keep speed-neutral efficiency gains** and record what they save: CPU time,
  bytes moved, memory or energy. They free a resource for the next change.
- **Let work splits learn their device scale from their own timings.** Keep the
  cost shape in code, fit one factor per device, freeze it, and key it to the
  executable, so a rebuild needs no hand retuning.
- **Tune profiles per thinking mode and context depth on the target machine**,
  +3-10 % over code defaults (Flash).

## CPU kernels

Decode:
- **Keep weights in their stored quantization** and dequantize in registers.
  An f32 output head kept quantized is about 5x smaller and 4.7x faster (GGUF).
- **Use an integer dot product on quantized activations** (`maddubs`), about
  18x the float dequantization kernel (GGUF).
- **Never concatenate weights per token.** Removing a per-token matrix fusion
  gave about 8x, and two GEMVs also beat one fused GEMV (GGUF).
- **Enable every feature the intrinsics need.** Without `fma` in
  `#[target_feature]`, each `_mm256_fmadd_ps` was an out-of-line call, +53 %
  once fixed (GGUF).
- **Vectorize the activation quantizer and read CPU features once**, about 10x
  (GGUF).
- **Use a work-stealing pool on hybrid cores.** Equal stripes finish at E-core
  speed, while chunks stolen from a shared cursor do not (GGUF).
- **Run small-batch elementwise passes inline on the caller.** Pool barriers
  dominate them, -6 % on a verify step (GGUF).
- **Use fewer threads on DRAM-bound decode.** Two thirds of the threads decode
  as fast as all of them, with -32 % CPU time (GGUF).
- **Store the KV cache in f16** (F16C, round to nearest even), +12 % decode at
  depth, -47 % cache state, KL unchanged (GGUF).
- **Sample in one pass with a running top-k threshold**, about 150x on a large
  vocabulary, where it had been a large share of a speculative step (Flash).

Prefill GEMM, in the order the gains were found (GGUF):
- **Hoist the scale unpack out of the position loop**, 2.7-4.2x, bit-exact.
- **Tile the position dimension outside the row loop**, so activations are not
  re-read per output row, 2.2x.
- **Reduce horizontally in registers.** Store plus scalar sum was a
  store-forwarding stall, +15 %.
- **Keep per-position vector accumulators on block-major, 64-byte-aligned
  packed tiles**, +14-30 %. The alignment also removed a bimodality between
  processes.
- **Iterate block-outer over 8-row groups** so the tile's block slice stays in
  L1, +28-30 %, -15-19 % CPU time.
- **Use AVX-VNNI with two sub-blocks per scale multiply**, +4-8 %, -5-6 % CPU
  time.

Other prefill passes:
- **Tile causal attention and use an online softmax**, about 17x (GGUF).
- **Split a recurrent block into parallel, recurrence-only and parallel
  passes**, and scan per head and column block over all tokens, 4x,
  bit-identical (GGUF).
- **Watch for cache-set aliasing.** A power-of-two stride that mapped a state
  copy to a quarter of the L1 sets cost 36 % of the scan (GGUF).
- **Allocate nothing per row in parallel passes.** Small allocations serialize
  on the heap lock, about 7x on the norm pass (GGUF).
- **Fuse elementwise steps with quantization row by row**, so large float
  intermediates are never written (GGUF).
- **Do one cheap select instead of a full sort.** A histogram pass over
  order-preserving keys, then a select within the boundary bucket, replaced a
  score matrix and per-query vectors, -12 % CPU time (Flash).
- **Do not compute precision the output cannot show.** A cubic `2^f` is enough
  for 8-bit probabilities, about -4 % on the attention kernel (Image).
- **Use integer attention where the error budget allows** (keys less their mean,
  per-head rotation, per-key value scales, which keep the attention sinks),
  3.2x on AVX-VNNI (Image).

## GPU kernels

- **Coalesce memory access.** A team of lanes per output row that splits one
  contiguous super-block load gave 2-5.3x on a GEMV. One thread per row
  collapsed at high workgroup counts (GGUF).
- **Avoid emulated instructions on the target GPU.** Converting nibbles through
  the float mantissa avoids slow integer paths on Pascal, 1.8-2.4x and more
  accurate than the integer form (GGUF).
- **Read each weight block once for all positions of a verify step**, -33 %
  (GGUF).
- **Tile prefill GEMMs**, about 12x (GGUF).
- **Use packed dot products** (`dotPacked4x8EXT`) with wide staging loads, 2x on
  the kernel and a further -33 % (GGUF).
- **Repack weights to a dword-aligned stride at upload**, for under 1 % more
  VRAM (GGUF).
- **When a kernel is bound by shared-memory loads, widen the loads.** Vector
  loads with several value columns per lane took a recurrence -38 %. Check
  register use, since a wider SIMD mode that spills is several times slower
  (Flash).
- **Reuse weights per thread tile**, -34 % on the hyper-connection call (Flash).
- **Keep shared-memory operands in the cheapest form.** An int8 mirror of pooled
  keys scored about 10x faster than an f32 form (Flash).

## NPU graphs

- **Bind weights as graph inputs instead of constants.** It is as fast, lets
  layers share one compiled graph, and lifts the constant size limit (NPU).
- **Share weight buffers between graph sets**, -50 % peak RAM and -36 % load
  time (NPU).
- **Read weight inputs swizzled**, +17 % prefill and +10 % decode (NPU), up to
  -18 % per dense graph (Flash).
- **Fold scales and signs into neighboring weights**, and move ops the NPU does
  poorly to a neighbor, -17 % per block (Image).
- **Replace a serial op chain through DDR with one custom kernel.** Norms and
  activations in one SHAVE kernel took -17 % per graph (Flash) and -7 % (Image).
  Stock chains that already stay in CMX gain nothing.
- **Keep intermediates in CMX, group recurrences by head, and match the
  compiler's fused forms**, +67 % over a series of steps (NPU).
- **Drop passes another stage redoes.** A softmax without its normalize pass,
  when the host merge divides anyway, took -26 % per block (NPU).
- **Record command lists once and mutate their arguments.** Re-recording per
  chunk cost about 20 % (NPU).
- **Record lists on their own thread**, because an append waits while the NPU
  runs (Image).
- **Size job slots by fill.** Wide slots for full batches and narrow ones for
  short pieces took the expert pass -27 % (Flash).
- **Run VAE-style convolutions as halo tiles on the NPU**, with norms on the
  cores, 2.9x (Image).

## Low-precision accuracy

- **Find which error dominates before raising precision.** Activation error,
  not int8 weights, was the KL, and rotating the activations cut it about 26x
  (NPU).
- **Rotate before quantizing.** Int4 per-row GPTQ on block-Hadamard-rotated
  weights is faster than the grouped export, with about half the KL (NPU).
- **Use calibrated static int8 ranges** folded into norm weights and row scales,
  which removes per-token rescale passes, +43 % for +9 % KL (NPU).
- **Calibrate on frozen texts of several kinds, and judge on a long text.** Mixing
  code, prose and poetry in three languages instead of drawing on live source
  files makes builds reproducible. KL fell 5 % on a long text and rose 7-40 % on
  a short one, so a short text alone misjudges a calibration (Flash).
- **Bound every intermediate of an f16 computation.** A chunked (WY) gated delta
  rule with a blocked inverse keeps all intermediates in range, about 18x lower
  KL than a Neumann product whose partial products overflowed (NPU).
- **Guard f16 range on the NPU.** Floor epsilons, pre-scale l2 norms by max-abs,
  accumulate cumulative sums in fp32, and scale up subnormals the device flushes
  (NPU).
- **Keep several scales per group.** One scale per stream lost visible KL on the
  hyper-connections (Flash).
- **Center keys cheaply.** A softmax ignores per-channel key offsets, so the
  previous step's means work as well as the block's own (Image).

## Memory and data movement

Host:
- **Recycle large blocks.** The Windows heap returned every large buffer to the
  OS and each layer re-faulted it, +40 % prefill (GGUF).
- **Read large weight files unbuffered and in parallel.** The file cache pushed
  out pinned pages. A few readers saturate the NVMe, and both fewer and more were
  slower, about 4x on load (Flash).
- **Give parallel readers their own file handles.** Windows serializes
  synchronous reads on one handle, over 1000x on cold rows (Flash).
- **Write buffers once, where they are read.** Bind outputs at their consumer's
  slot and remove zero fills that the next write overwrites. Mostly speed
  neutral, with -7-11 % host wait on the NPU (Flash), -10 % on a decode step
  (Flash) and -4 % on the VAE (Image).
- **Keep threads persistent** instead of spawning them per step (Flash).
- **Store long-lived state in immutable shared segments**, so a snapshot's size
  does not grow with the conversation (Flash).
- **Keep the display and system awake while serving.** In display standby the
  iGPU stalled at random points, a 44-57 % higher prefill rate when kept awake
  (NPU).
- **Keep the iGPU out of its idle power state** during long NPU work, +41 %
  (Flash).

GPU:
- **Choose memory types deliberately.** Keep shader-read buffers device-local
  and readback buffers host-cached, 3.8x on a matmul call. Reuse the readback
  destination, since a fresh allocation is limited by page faults, about 6x
  (GGUF).
- **Pool per-call buffers**, 3.6x on allocation and about 9x on release (GGUF).
- **Sub-allocate small buffers from larger blocks.** Many small allocations next
  to large weights lost the device, +39 % decode (GGUF).
- **Stream weights through reused VRAM slots for prefill**, +63 %, because a
  chunk reads each weight once (GGUF).
- **Overlap uploads on the transfer queue with host work**, about 4x on upload,
  and prefetch the next matmul's rows, +11-14 % (GGUF).
- **Import host memory as GPU memory** and write outputs in place (GGUF, Flash).
- **Record decode command buffers once and resubmit them**, +3 % (GGUF).
- **Respect the VRAM budget.** Past it, WDDM silently demotes VRAM to system RAM
  and decode collapses, so plan from `VK_EXT_memory_budget` less a reserve
  (GGUF).
- **Count all devices against one residency limit.** What the NPU and GPUs keep
  resident shares one WDDM limit. Importing only part of the expert pool avoided
  device loss at full context, with speed within 1 % (Flash).
- **Recover from device loss in-process** by rebuilding and re-uploading (GGUF),
  and drop and re-upload when another application takes the VRAM (Flash).

## Splitting work across devices

- **Only bytes resident in another device's memory leave the DRAM bus.** The
  gain comes from where the bytes live, not from faster compute.
- **Split matmuls by rows.** Keep the first rows of every big matmul in VRAM and
  compute the rest on the CPU, +33 % decode, with fewer VRAM bytes than whole-layer
  offload (GGUF).
- **Add a streamed share for prefill**, resident, streamed and CPU rows with
  fractions from measured rates, +30 % (GGUF).
- **Put a share of the attention on the GPU**, balanced by query rows, -27 %
  (GGUF).
- **Keep decode MLPs on the cores plus VRAM rows when the NPU is slower there**,
  +26-39 % (NPU).
- **Put the recurrence on the iGPU between NPU graphs**, with host buffers
  imported zero-copy, +26 % (NPU).
- **Move whatever bounds the NPU to the iGPU while the iGPU has slack**, until
  both are bound: hyper-connections +7-8 %, gated norm and rotation +4 %,
  attention -23 % on its layers, indexer scores +2-3 % at depth (Flash).
- **Split experts between devices by estimated time per device**, +9 % (Flash).
- **Split far attention into key blocks merged by log-sum-exp**, which reaches
  long contexts and lets a second device share the blocks, +9 % (NPU).
- **Add an iGPU share of the blocks**, spreading its chunks through each block,
  -15-18 % (Image).
- **Run the vision tower's per-patch work on the NPU and its attention on the
  iGPU**, 3-4x over the cores (NPU).
- **Use int8 prefix K / V only when they do not fit**, about -30 % memory with
  many references (Image).
- **Keep the cores and discrete GPU free in prefill when other work needs them**,
  by placing it on the NPU and iGPU only (Flash).

## Scheduling and pipelining

- **Reorder the outer loops around the costly shared pass.** Layer-major prefill
  with one expert pass per large piece gave +76 %, and larger pieces +11 % more
  (Flash).
- **Pipeline sub-chunks over two buffer banks**, +11 % (Flash).
- **Overlap one half's GPU work with the other half's CPU phases**, +4-5 %, and
  keep two batches in flight, +2-4 % (GGUF).
- **Defer work behind phases that already run**, +1.5-2 % (GGUF).
- **Overlap the NPU and the cores chunk by chunk**, -32 % (Image).
- **Run two waves per decode step**, half the positions a layer behind, so the
  GPU runs one wave's dense part while the cores run the other's experts, -13 %.
  Queuing each wave's next GPU run behind the other's gave a further -7 % (Flash).
- **Submit the critical kernel first.** The iGPU runs separate contexts in
  submission order, and queue priorities changed nothing, +3-6 % (Flash).
- **Do deferred work just before its consumer, while devices are busy.** Moving
  a gather and combine into the next layer took the MoE pass -12 %, +3-4 % overall
  (Flash).
- **Prepare inputs while the devices run.** Packing keys chunk by chunk cut an
  idle block boundary by 55 % (Image).
- **Spin through short device waits.** Blocking waits and parked workers wake too
  late for short gaps, +6 % decode (Flash).
- **Merge small steps into the neighboring dispatch**, and copy only the ranges
  the host reads, several percent each (Flash).
- **Read each row once for all positions** in a step's mixer and head, -29 %, and
  use one attention path for every query count, 2.6x (Flash).
- **Compute only what later rows need.** Speculative head rows before the last as
  key and value only took -19 %, and 2.6x on catch-up (Flash).
- **Route short pieces through a cheaper path** than a full prefill chunk, up to
  2.7x on very short pieces (Flash).
- **Do the expert pass in one dispatch**, with partials summed in slot order,
  +11 % and bit-identical (Flash).

## Speculative decoding

More tokens per weight read is the decode lever, because verifying several
positions costs about as much as decoding one.

- **Use the model's own next-token head (MTP)** where it exists, about 2x on code
  with greedy output byte-identical (GGUF).
- **Use a block drafter** that drafts from the target's features, +25 % over MTP
  on code (GGUF).
- **Draft from context without a model** by copying what followed an earlier
  occurrence of the last tokens, 2.6-2.9x on edit-style output and neutral on
  prose (GGUF).
- **Choose draft length by expected value.** Learn acceptance per match length
  online and draft only when it beats the alternative, +2-12 % on agentic turns
  (GGUF).
- **Reduce the draft vocabulary** to frequent ids, specials and ids seen,
  2.5-3.2x per draft and lossless (GGUF, Flash).
- **Roll back by replay** with double-buffered recurrent state instead of
  per-token copies, -33 % per verified token (Flash).
- **Price drafts with a cost model** and stop a chain once no longer chain can
  win, +5-6 % (Flash).
- **Measure step costs at run time**, +1.6-2.5 % at shallow depth. It lost up to
  0.9 % at depth, so use it only where it wins (Flash).
- **Couple drafter and target sampling** with shared Gumbel noise, an exact
  sample with more accepted drafts at high temperature, +1.6-4.8 % (Flash).
- **Calibrate acceptance at the mode's own temperature.** Head confidence is badly
  overconfident at high temperature (Flash).
- **Defer rollback with low-rank factors** so verify starts from one state, +29 %
  (NPU).

## Serving and prefix cache

- **Restore the longest kept prefix and prefill only the rest**, about 15x on a
  follow-up turn (GGUF).
- **Share snapshots of common prefixes and persist them**, about 50x for a new
  session on a known preamble (GGUF).
- **Share KV pages between checkpoints by content**, so a checkpoint costs one
  recurrent state plus its new rows.
- **Evict by cost** (recency, hits, class and learned replay time per freed byte)
  instead of LRU, and gate disk writes by the prefill time they save.
- **Drop checkpoints that would add a prefill piece**, -19 % on the turn that
  crosses one (Flash).
- **Continue live.** A prompt that extends the engine's tokens needs no reset,
  restore or re-prefill, -56 % to first content on agentic turns (Flash).
- **Key images by a pixel hash**, so an image no longer bypasses the cache, 3.5x
  (GGUF).
- **Avoid per-request deep copies and quadratic validation of large requests**
  under the engine lock.
- **Enforce tool calls structurally** by seeding the assistant turn with the
  call's opening tags, and take argument types from the schema.
- **Apply no repetition penalty to tool calls**, because the call format repeats
  tokens by construction.
- **Pick special tokens by token type, not by name shape.** Otherwise they are
  split by BPE and the model misbehaves.

## Rejected

Kernels and quantization:
- **NPU as a matmul offload.** It cannot consume int4, and the cores outrun it.
- **AVX-VNNI with one sub-block per scale multiply.** The saved scale step
  returns as a widen and multiply.
- **Packed dot products as the decode GEMV.** Fastest kernel, but decode stays
  bound by the VRAM cap and the integer activation costs accuracy.
- **Register blocking of the GEMM accumulators**, -41 %, because the unpack hoist
  already uses the registers.
- **Tree reduction in the GEMM.** Within noise, and it loses precision.
- **GPU scale hoist.** It adds many weight bytes to save ALU work, and coalescing
  was the real problem.
- **Activation sparsity.** Few matrices tolerate it, and scattered reads are slower
  than sequential ones until sparsity is very high.
- **Entropy coding or lossless repacking of weights.** Quantized weights are
  already within 3-15 % of their entropy.
- **More registers, retiled GPU attention, more slot kinds.** They spill or show
  no gain.

Device placement:
- **Whole GPU layers.** Correct, but not competitive with the row split.
- **LRU weight cache or streaming for decode.** A dense model scans cyclically,
  and the bus is slower than DRAM.
- **Hot experts in VRAM.** Too little VRAM is left to cover enough reads, for at
  most about 5 % of decode.
- **Large pages for shared expert buffers.** A real gain on the expert pass, but
  the NPU's allocations cannot use them, and a separate copy costs too much RAM.
- **The iGPU as the main compute device.** It shares DRAM with the cores and was
  about 6x slower than the discrete GPU path.
- **iGPU share of attention next to the cores.** It competes with them (Image).
- **Norms inside NPU convolution graphs**, 3-4x slower than on the cores.
- **Weight-side dequantization on SHAVE.** Far slower than DPU dequantization.
- **SwiGLU as a SHAVE kernel.** Stock elementwise chains already stay in CMX.

Scheduling:
- **Longer spins, `UMWAIT`, or shorter spin windows.** `UMWAIT` lost 1.5-2.4 %,
  and windows short enough to save CPU pay the wake-up on most dispatches.
- **Host attention on a side pool beside the experts.** Oversubscription cost up
  to 45 %.
- **Preparing the next sub-chunk's attention early.** It delays the NPU's next
  job, 4-6 % slower.
- **Rebalancing or adding NPU attention jobs at depth.** Gathering the extra blocks
  on the host costs what the iGPU saves.
- **Queuing the next wave after experts are sent.** The GPU becomes the wait.
- **Compiler option sweeps.** Noise.
- **Recurrence reformulations** (unrolled loops, transpose-free inverse, other
  chunk sizes). Slower or less accurate.

Speculation:
- **Tree drafting and sibling verification.** At most 1-3 %, about the cost of the
  extra position.
- **Drafters with fewer routed experts.** Acceptance falls more than step time.
- **Block verification of long copy runs.** Such runs are too rare in agentic output.
- **Retuning cost vectors from a single run.** Tune only with interleaved rounds in
  every mode.
- **EAGLE-3, T-MAC, FastMTP.** No drafter for this target, no gain at 4 bits on
  AVX2, or training required.

Accuracy shortcuts:
- **Static int8 activations in decode graphs.** Wrong tokens.
- **Skipping far blocks by softmax mass.** Attention is too diffuse at block
  granularity.
- **Int4 transformer weights.** No faster on the NPU, with 3-4x the error (Image).
- **Block cache across diffusion steps.** Visible changes on some prompts (Image).
- **Block-sparse attention for many references.** About 1.4x at best (Image).
- **Resampled step schedules.** They redraw the image and look harsh (Image).
