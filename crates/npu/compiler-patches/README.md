# NPU compiler patches

Patches to the Intel NPU MLIR compiler. The NPU engine's graphs are compiled
with our own build of that compiler (`vpux-opt` / `vpux-translate`, driven
by `crates/npu/scripts/vpuxc.ps1`) instead of the compiler that ships inside
the NPU driver. `../NPU.md` describes how the hardware and the compiler behave.

**License.** `npu_compiler` is licensed under the Apache License 2.0. These
patches are modifications of its source and are provided under the same
license (Apache-2.0), not under this repository's MIT license. Each file
states which files it changes.

## What they apply to

| Item | Value |
|---|---|
| Component | [`openvinotoolkit/npu_compiler`](https://github.com/openvinotoolkit/npu_compiler) |
| Version | tag `npu_ud_2025_44_rc1`, commit `a934b15d7494c4961afd51cf6c896b15d1fabd8c` |
| Built against | OpenVINO commit `80e587bccec` (pinned by that tag's `validation/openvino_config.json`) |
| Targets built | `vpux-opt`, `vpux-translate` (plus `openvino_ir_frontend`, needed by `--import-IE`) |
| Matching driver | Intel NPU driver 32.0.100.4404, whose compiler is built from the same OpenVINO commit. Unpatched, our build produces blobs bit-identical to the driver's. |

The patches touch only the compiler source. Nothing here changes the
installed driver. The driver's own compiler stays unpatched, and blobs that
need a fix must come from our build. The driver can sideload a self-built
compiler DLL (`NPU_ALT_DEPENDENCY_PATH`, see
`src/vpux_driver_compiler/docs/test_and_debug/enable_sideloading.md`), but
nothing in this repository relies on that.

## Applying

```sh
git clone https://github.com/openvinotoolkit/npu_compiler
cd npu_compiler
git checkout npu_ud_2025_44_rc1        # a934b15d7
git apply /path/to/qwen.rs/crates/npu/compiler-patches/*.patch
# then build vpux-opt and vpux-translate against OpenVINO 80e587bccec
# (Release, Ninja, MSVC 2022; ~40 min for the first build, minutes after a patch)
```

`git apply --check` on a clean checkout of that tag is the quick test that a
patch still applies. The patches are independent of each other.

## The patches

Each section names the files the patch changes, the problem, the change, its
switch (environment variables read by `vpux-opt`, and with them unset, blobs are
byte-identical to the unpatched compiler's) and what it is used for here.

### 0001-i4-zero-point-elf-export.patch (required for i4 weight inputs)

**Files.** `src/vpux_compiler/src/dialect/VPUMI37XX/blob_writer.cpp`,
`createTensorRef`.

**Problem.** The ELF export serializes a quantized tensor's zero points as
`uint8_t` through `checked_cast`. An i4 (signed) quantized type can carry a
negative zero point, so exporting any graph with **int4 weights as a graph
input** (not a constant) fails:

```
Can not safely cast -1 from __int64 to unsigned char
```

The driver's compiler fails the same way (`pfnCreate2` returns
`0x78000004`).

**Change.** The zero point is stored as its two's-complement byte
(`static_cast<uint8_t>(static_cast<int8_t>(zp))`), which is what the
hardware reads for a signed type. Always on.

**Use.** The int4 weight set (per-row GPTQ) is bound as i4 **inputs** to the
prefill graphs. One compiled graph then serves all 47 shared delta-net
layers, and the weights stay outside the NPU's ~14 GB limit on graph
constants. Without this patch, only int8 weight inputs (twice the bytes) or
int4 constants (one graph per layer) compile.

### 0002-reduce-sum-accumulate-multi-row.patch (speed, speculative verify)

**Files.**
`src/vpux_compiler/src/dialect/IE/transforms/passes/unroll_fully_connected.cpp`,
`isBeneficialToUseReduceSumForAccumulate`.

**Problem.** `UnrollFullyConnected` splits a grouped-int4 FullyConnected
into 128-deep sub-convolutions along K. It adds the partial sums with one
ReduceSum only when the FC has exactly one row. With more rows (a
4-position speculative verify) it adds them with a chain of NCE Adds (252
for 4 FFN pairs), which slows the layer down.

**Change.** With `NPU_RS_ACC_MAX=N` the ReduceSum form is taken up to N rows
(default 1, the unpatched behavior).

**Measured** on 4 int4 g128 FFN pairs at s = 1 / 2 / 4 / 8, the times are 10.1 / 11.2 /
13.4 / 15.9 ms, against 10.3 / 15.2 / 16.4 / 18.3 unpatched.

**Use.** Only graphs with weights in groups of 128 (g128). The per-row int4
weight set has no K split. s = 4 verify graphs of that kind are compiled
with `NPU_RS_ACC_MAX=16`.

### 0003-dpu-profiling-buffer-x4.patch (diagnostics only)

**Files.** `src/vpux_compiler/src/dialect/VPUIP/utils/utils.cpp`,
`getDPUProfMaxBufferSize`.

**Problem.** DPU profiling stores task records in a CMX buffer sized for 64
DPU tasks. Our per-layer graphs have thousands of DPU tasks, and a profiled
compile overflows that buffer.

**Change.** The buffer is 4x larger on NPU37XX / NPU40XX. It affects only
blobs compiled with `dpu-profiling=true`.

**Use.** A partial aid. Profiled builds of the large delta-net graphs still
crash in `--lower-VPUIP-to-ELF`.

### 0004-si-input-multi-consumer.patch (static ranges without `sep_scale`)

**Files.** `src/vpux_compiler/src/dialect/IE/utils/quantization.cpp`,
`findNCEOpCandidatesWithWeights`.

**Problem.** With i4 weight inputs, a static-range int8 FakeQuantize that
the compiler fuses into the output of the preceding MatMul (the block
rotation) must stay **i8**. `ConvertWeightsToU8` keeps an activation signed
only when `keepIntTypeForSIWeightsAsInput` finds that every consumer is an
NCE op with signed (i4) weights as an input. The search follows
pass-through ops (views, Convert, Transpose, FakeQuantize) **only while
they have one use**. When one quantized activation feeds several int4
MatMuls through their own reshapes (our split projection and gate / up
MatMuls), it gives up, the activation becomes u8, and `ConvertLayers2VPU`
fails with `failed to legalize operation
'builtin.unrealized_conversion_cast'` (u8 with zero point 128 to i8).

**Change.** The search follows every branch of a multi-use pass-through op
and returns all NCE candidates found (failure if any branch has none).
Always on.

**Use.** `build_delta.py --nosep` folds each static range's 127 / r into the
per-channel input in front of the rotation (ln1 / ln2, ...) instead of a
scalar SHAVE multiply after it. This saves 1.9 ms per delta-net layer, together with
the out / down inputs, which compile without the patch (one consumer).
Without the patch, `build_delta.py` avoids the fusion with `sep_scale`. The
range scale is applied as a run-time scalar Multiply after the rotation.

### 0005-scheduler-engine-overlap.patch (DPU / SHAVE concurrency, opt-in, no gain measured)

**Files.** `src/vpux_compiler/src/core/feasible_memory_scheduler.cpp` (and
`.hpp`), `src/vpux_compiler/src/dialect/VPU/IR/attributes.cpp`.

**Problem.** The feasible memory scheduler never runs a DPU op and a SHAVE
op at the same time (`../NPU.md` §5).

**Change.** Three switches let it overlap them, and one prints diagnostics:

- `NPU_RS_NOALIGN=1` skips `alignExecutors()` on every memory free (which
  moves every engine's cycle to the end of each op that frees memory).
  Instead each freed CMX range records the cycle it became free (its
  buffer's last use, or a spill write's end). Once a round's buffers have
  addresses, every op pushed in that round is moved to begin no earlier
  than the free cycle of the memory it now occupies (`retimeRound`, in push
  order, propagated to its dependents and its queue). Simply dropping the
  alignment corrupts `ssm_out`. With the retiming, a reused range's next
  user cannot be simulated as overlapping its previous user, so the memory
  control edges keep their direction and are not dropped as "coexisting".
- With `NPU_RS_LOOKAHEAD=K`, a compute queue whose IR-order head is not ready
  (it waits for the other engine) takes the first ready op among its next K
  instead. When that op does not fit in CMX, it takes the next ready one.
- With `NPU_RS_CMX_FRAC=F`, the tiling / placement passes see F x the real CMX
  size (`getTotalCMXSize`, `getTotalCMXFragmentationAwareSize`). The
  allocator still uses all of it, so two tiled ops can be resident at once.
- `NPU_RS_SCHEDSTAT=1` prints diagnostics on stderr. Per round, it shows why an idle
  queue's head was not scheduled (`SCHEDSTAT idle ... first ready at +N`,
  `SCHEDSTAT nofit ... free B`). Per schedule, it shows the simulated busy cycles per
  engine and for DPU and SHAVE together (`SCHEDSUM`).

**Measured** (256-position delta layer 1 and the 64-layer g0 set). Outputs stay
bit-identical with every combination, but no speedup. Layer 52.2 ms to
52.0-52.2 (`NOALIGN` + `LOOKAHEAD=64`), and 55.8 / 52.6 / 53.6 with `CMX_FRAC`
0.5 / 0.6 / 0.75 (smaller tiles cost more than they overlap). 64-layer
prefill 85.4 to 85.6 tok/s (`NOALIGN` + `LOOKAHEAD=64`), 84.8 with
`CMX_FRAC=0.6`. The schedule does change (simulated DPU and SHAVE overlap
1.2M to 2.1M cycles in the main function), but on the hardware DPU and SHAVE
overlap only 0.05-0.4 ms per layer. In the recurrence window (7-30 ms) the
DMA engine runs alone for ~11 of 23 ms (per-op layout copies, spills, slice
gathers), so little DPU or SHAVE work is left to overlap.

### 0006-grouped-matmul-37xx.patch (batched MatMuls as one DPU op, for speed)

**Files.** `NPU37XX/dialect/VPUIP/transforms/pipelines.cpp`,
`core/execution_group_analysis.cpp`, `core/barrier_info.cpp`,
`utils/wlm_legalization_utils.cpp`, `dialect/IE/utils/matmul.cpp`.

**Problem.** On NPU37XX a batched MatMul (`[.., B, M, K] x [.., B, K, N]`,
e.g. the delta-net recurrence's `[12, 2, 128, 128]` products) is unrolled
into B 2-D MatMuls in IE (`MatMulInputsTo2d`), each with its own slice DMAs,
output spill and a `slice_gather` concat copy. That costs ~5 us per slice,
so a 100 MFLOP batched MatMul takes 0.12 ms where the same FLOPs as one 2-D
FC take 0.03. The compiler can keep it as one grouped NCE MatMul
(`enable-grouped-matmul=true`), but on 37XX that fails the ELF export
(`Can't get ODU permutation by output dimsOrder: 'GNHWC'`), because only the
40XX pipeline runs `BatchMatMulToMatMul`, which turns the 5-D task into one
4-D DPU task per group on sub-views of the same CMX buffers (no slicing
DMAs, no concat).

**Change.**

1. **The pass** is added to the 37XX VPUIP pipeline after
   `UnrollDistributedOps`, as on 40XX. It acts only on 5-D NCE tasks, which
   exist only with `enable-grouped-matmul=true`, so other blobs are
   unchanged.
2. **Descriptor-ring deadlock** (`NPU_RS_DPU1LIST=1`). With many split
   tasks the device is lost (`0x70000001`, a hang) whenever an unrelated
   multi-cluster DPU op precedes them. On 37XX all DPU invariants of both
   tiles share **one** descriptor list (every invariant is `<0:0:i>`, and the
   ring holds 32, loaded in halves of 16), and so do the ActShave
   invocations. The runtime reloads half k + 2 only once **every** task of
   half k has had its `clean_after` barrier consumed. The non-WLM
   legalization (`legalizeScheduleForNonWlm`) forms its descriptor groups
   per tile and gives only a group's *last* task an early barrier, so a
   barrier produced by 24 split tasks that straddle three halves can never
   be consumed. With the variable set, the groups are formed over the
   merged list (`rsMergedQueues`). Every task of group g whose update
   barriers all have a producer or consumer in group g + 2 or later gets
   one extra barrier, consumed by the first task of g + 1 (it becomes its
   `clean_after`), and the insertion repeats after the task reordering
   until nothing changes. `scripts/ringsim.py ELF.mlir` simulates that rule
   over a lowered graph (`vpuxc.ps1 -Keep`) and reproduces every observed
   pass and fail case.
3. **`NPU_RS_GMM_MAXG=N`** allows no grouped MatMul with more than N groups. 112
   groups of `[64, 64]` run, but 128 still lose the device, for a reason not
   found (not the ring, by the same simulation). Use 96.

**Measured.** Batched outputs are **bit-identical** to the unrolled compile and
repeatable (`npubench blob --check`). Delta layer 1 50.85 to 49.31 ms at the
shipped flags. It also makes `build_delta.py --c 64` pay off (46.6 ms,
`--rgroups 6` 45.8, against 57.5 unrolled). 64-layer 256-position prefill
84.9 to **95.0 tok/s** with `--c 64 --rgroups 6`, KL vs BF16 0.0334 to
0.0335 (0.0154 to 0.0162 on text 2, from the `--c 64` rounding).

**Use** all three together:

```sh
NPU_RS_DPU1LIST=1 NPU_RS_GMM_MAXG=96 NPUC_OPTS="enable-grouped-matmul=true" sh compile_lean.sh SET
```

### 0007-cmx-view-chains.patch (intermediates stay in CMX, for speed)

**Files.** `dialect/VPU/transforms/passes/make_ops_with_distributed_tensor.cpp`,
`dialect/VPUIP/transforms/passes/optimize_copies.cpp`.

**Problem.** Every op of the delta-net recurrence writes its output to DDR
and the next op reads it back, although both ops keep the data split the
same way over the two tiles. There are two causes, and each has a switch.

**Change.**

1. **`NPU_RS_VIEWFOLD=1`.** The MatMul lowering wraps each NCE MatMul in
   reshape / permute-cast chains (4-D to 5-D grouped form and back).
   Between two MatMuls those chains are a net identity, but they sit on DDR
   tensors between the CMX-to-DDR and DDR-to-CMX copies, so the copies never
   meet. Before `MakeOpsWithDistributedTensor`, a chain of pure view ops
   (`AffineReshape`, `PermuteCast`, `ShapeCast`, `Reshape`, `Squeeze`,
   `Unsqueeze`, `LayoutCast`) that ends in exactly the type it started from
   is dropped.
2. **`NPU_RS_DISTVIEW=1`.** `MovePureViewOpBeforeCopy` refuses to move a
   view onto a SEGMENTED buffer whenever the view changes the rank (a TODO
   in the source), which is every edge between a 4-D eltwise / SHAVE op and
   a 5-D grouped MatMul. A new `OptimizeCopies` pattern
   (`RsDistributedViewChain`) matches a copy from distributed CMX to DDR,
   pure views, and a copy from DDR back to distributed CMX. If every
   cluster's segment of the source and of the destination is the **same
   contiguous byte range** of the compact DDR tensor, the destination
   becomes a `VPUIP.ViewOp` of the source. The checks are SEGMENTED,
   compute shapes equal to memory shapes, compact strides, the segment
   contiguous in memory order, and equal per-cluster offsets and sizes.
   `RsDeadSpillCopy` then drops the DDR copy if nothing reads it any more.

**Measured** (delta layer 1, shipped flags `--c 64 --rgroups 6 --gsum1`,
with 0006). The layer went from 45.9 to 45.7 ms (VIEWFOLD) to **43.9 ms** (both). Recurrence DMA
task time per head group went from 2.5 to 1.6 ms. **64 layers at 256 positions went from 95.2 to
99.2 tok/s, residuals bit-identical.**

**Use:**

```sh
NPU_RS_DPU1LIST=1 NPU_RS_GMM_MAXG=96 NPU_RS_VIEWFOLD=1 NPU_RS_DISTVIEW=1 \
NPUC_OPTS="enable-grouped-matmul=true" sh compile_lean.sh SET
```

### 0008-kernel-dir-override.patch (own SHAVE kernels)

**Files.** `act-kernels/shave_binary_resources.cpp`, `getElf`.

**Problem.** The SHAVE kernels a blob can use are fixed when the compiler is
built.

**Change.** With `NPU_RS_KERNEL_DIR=DIR` set, `DIR/<kernelPath>.elf` (e.g.
`log_softmax_3720xx.elf` and `log_softmax_3720xx_lsu0_wo.elf`) replaces the
ACT-SHAVE kernel the compiler would embed under that name. Unset, or with no
such file, blobs are unchanged.

**Use.** This is how our own kernels get into a blob. Build one with our
kernel compiler (`shave build KERNEL.sk OUT.elf`, `../SHAVE-DSL.md`), name
it after a kernel the graph does not otherwise use, and give the graph that
op as a carrier. `build_delta.py --shinv BS` puts `crates/npu/shave/trinv.c`
(built with `-DBS=BS`) on a `LogSoftmax` over the flattened `[C*C]` axis.
The kernel ABI our compiler emits has the entry `controlled_act(u32 params)`,
packed 0x24-byte `MemRefData` records (data +0, numDims +8, dims +0xc,
strides +0x10), inputs then outputs.

### 0009-swizzled-weight-inputs.patch (weights as inputs at constant speed)

**Files.** `dialect/VPUIP/transforms/passes/swizzling.cpp`,
`dialect/VPUIP/transforms/passes/resolve_dma_with_swizzling.cpp`.

**Problem.** The DPU reads its weights fastest from **swizzled** CMX buffers
(the RAM-cut address stagger of the 37XX CMX). The compiler swizzles
constant weights at compile time, but never a weight that is a graph input,
and the 37XX DMA cannot swizzle while copying. So a weight bound as an input
runs ~22 % slower than the same weight as a constant (gate / up shape, 11.85
vs 9.2 ms, and constants with `enable-weights-swizzling=false` are exactly as
slow as inputs).

**Change.**

- With `NPU_RS_SWZIN=1`, an NCE op's weights may also be swizzled when they are
  copied into CMX from a graph input. This applies only to quantized (int4 /
  int8) inputs, only to compact (unstrided) copies, and only to real graph
  inputs, meaning arguments of the entry function, or of an outlined function whose
  every call site passes one. The pass then marks the CMX buffer swizzled
  and leaves the bytes alone. **The host must supply such an input already
  swizzled** (`crates/npu/scripts/swz.py`). `ResolveDMAWithSwizzling`
  checks that every graph input is read only swizzled or only plain, and
  that every swizzled copy starts at a multiple of the swizzle period and is
  512-byte aligned. Otherwise the compile fails.
- `NPU_RS_SWZKEY=K` (1 to 5, default 5) sets the swizzle key of all weights. The
  pattern rotates the 16-byte blocks of each 512-byte row by
  `(row mod 2^K) * 2^(5-K)` blocks and repeats every `512 << K` bytes. One
  input buffer can serve graphs with different weight tiles when every tile
  starts at a multiple of the period. The decode graphs copy 272-row tiles
  (696,320 bytes, not a multiple of 16 KB), so the shared prefill / decode
  weights use key 2 (2 KB). Keys 2-5 run equally fast, and key 1 is ~8 % slower.
- `NPU_RS_SWZIN_LOG=FILE` writes the network-input indices that were
  swizzled.

**Measured.** FC microbenchmark 11.85 to 9.14 ms (constants 9.11), outputs
bit-identical. Delta layer 41.6 to 37.2 ms, attention layer 34.1 to 29.9 ms
(256 positions), bit-identical on real weights. `down_proj` (K = 17408) is
copied in strided K chunks and stays plain.

**Use.** `scripts/swzset.py` builds a set with its blobs recompiled this way
and its swizzled weight files. `scripts/swzcheck.py` checks that every
weight buffer shared between prefill and decode graphs is swizzled in all
of them or in none.
