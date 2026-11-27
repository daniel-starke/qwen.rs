# Compute shaders

Every kernel is GLSL, **precompiled offline** to SPIR-V and embedded in the
binary with `include_bytes!` (see `gpu/device.rs`). The shipped `.exe`
therefore needs no shader compiler at run time and no Vulkan SDK to build,
only the system `vulkan-1` loader, which `ash` opens with `Entry::load()`.

## Rebuilding a kernel

```
glslc -fshader-stage=compute -O <name>.glsl -o <name>.spv
```

Kernels that use subgroup operations (`subgroupAdd`, `subgroupBroadcast`)
also need `--target-env=vulkan1.1`, as the table below marks. The device
is created at API 1.1 for this reason.

No build script compiles shaders (`build.rs` only deflates the `.spv`
files, embedded with `quant::packed!`). A `.spv` is rebuilt only when the
command above runs, so always recompile and commit the `.spv` together with its
`.glsl` edit.

## Kernels

The kernels this file describes are below. Every other kernel (attention,
delta-net, auxiliary and GEMV helpers) explains itself in its header
comment. "Vulkan 1.1" marks the kernels that use subgroup operations and
need `--target-env=vulkan1.1`.

| Kernels | Purpose | Vulkan 1.1 | Check / benchmark |
|---|---|---|---|
| `q4k_f32`, `q6k_f32` | Decode GEMVs on an f32 activation (see below) | yes | `gpubench --gemv-probe` |
| `q4k_f32_seq`, `q6k_f32_seq` | GEMVs for 2-8 positions (see below) | yes | `gpubench` |
| `q4k`, `q6k` | GEMVs on a q8_K activation, for the fused blocks and `QWEN_GPU_Q8GEMV=1` | | `gpubench --check` |
| `dp4a_q4k` | Q4_K GEMV on packed 4x8 integer dot products | yes | self-check at device init |
| `gemm_q4k`, `gemm_q6k` | Prefill GEMMs (see below) | | `gpubench --probe` |
| `q8_0`, `f16`, `f32` | GEMVs for the remaining weight types | | `gpu_*_matmul_matches_cpu` tests |
| `gemm_f32` | Dense f32 GEMM for GPTQ's factorization, not inference | | `densebench` |
| `hc_*_npu`, `hc_*_dp4a` | Flash-Next prefill hyper-connections on the iGPU | `hc_down_*`, `hc_up_*` | `hcbench` |
| `exp_gemm`, `exp_act`, `exp_xq` | Flash-Next prefill routed experts on the iGPU | yes | `expbench` |
| `idx_score_npu` | Flash-Next prefill block indexer on the iGPU | yes | `indexer` test |
| `dit_gemm_npu`, `dit_row_npu`, `dit_act_npu`, `dit_qkv_npu` | Qwen-Image transformer blocks on the iGPU | yes, except `dit_qkv` | `dit_mid` test |
| `h3_row_npu`, `h3_act_npu`, `h3_qkv_npu` (+ `dit_gemm_npu`) | MiniMax-H3-Image transformer blocks on the iGPU | yes, except `h3_qkv` | `h3_mid` test |
| `h3te_act_npu` (+ `h3_row_npu`, `h3_qkv_npu` at the text encoder's widths) | MiniMax-H3-Image text encoder layers on the iGPU | yes, except `h3te_qkv` | `h3_mid` test |
| `h3v_gemm`, `h3v_attn`, `h3v_qk`, `h3v_norm`, `h3v_elem` | MiniMax-H3-Image VAE decoder tiles on the iGPU | yes | `h3_vae` test |
| `h3_attn` | MiniMax-H3-Image transformer attention on the GTX 1060 | yes | `h3_attention_matches_reference`, `h3_attention_bench` |
| `vis_rope_npu`, `vis_qk_npu`, `vis_sm_npu`, `vis_pv_npu` | Attention of the NPU engine's vision tower on the iGPU | | `visbench` |
| `aux_rmsnorm_sum` | Deliberately single-threaded (see the comment in the file) | | |
| `alu_probe` | ALU ceiling measurement (`gpubench --gate`), no matmul | | |

Benchmarks are examples of the `offload` crate, run with
`cargo run --release -p offload --features vulkan --example NAME -- ARGS`.
Tests that need a specific adapter take it from `QWEN_TEST_ADAPTER`, e.g.
`QWEN_TEST_ADAPTER=intel cargo test --release -p offload --features vulkan
indexer -- --include-ignored --nocapture`.

### Notes per kernel

- **`dp4a_q4k`** is the q8_K Q4_K GEMV on packed 4x8 integer dot products,
  in `q4k_f32`'s layout. `GpuDevice::init` checks it against
  `quant::gemv_q4k`. When it passes, it replaces `q4k` wherever a q8_K Q4_K
  GEMV runs (fused blocks, `QWEN_GPU_Q8GEMV=1`). For the kernel alone on FfnUp, the GTX 1060 reaches
  155-163 GB/s against `q4k_f32`'s 108-130, and the iGPU 55 against 15.
- **`gemm_f32`** computes a dense f32 `C (-)= A B^T` on regions of VRAM
  buffers (`gpu/dense.rs`, `offload::Dense`), from which `crates/npubuild`
  builds GPTQ's Hessian factorization. It has the same tile shape as
  `gemm_q4k`. `densebench` checks it against f64 sums and times it (GTX
  1060, 700 GFLOP/s at k = 256 and 1340 at k = 4096-8192).
- **`hc_norm_npu`, `hc_down_npu`, `hc_up_npu`** are the Flash-Next delta
  layers' attention hyper-connection in prefill (`gpu/hcpre.rs`,
  `offload::HcPre`). They apply a norm per stream, the down matrix with the injection
  rows (split-k in shared memory) plus SiLU / sigmoid, and the up matrix
  plus the gated stream mean, as f32 sums over the graphs' int8 weights.
  **`hc_norm_dp4a`, `hc_down_dp4a`, `hc_up_dp4a`** do the same on int8 inputs
  and are the default where `dp4a` works (`QWEN_FLASH_HC_DP4A=0` selects the
  float kernels). The normed streams and the down output are rounded in
  groups of 16 with a scale each, and the sums per group are exact int32.
  `hcbench adapter wdir [layer]` checks them against f64 sums and times a
  call. It takes 4.0 ms per call on int8 inputs and 5.9 ms on f32, at 256 positions on
  the Arrow Lake iGPU. One scale per stream or per position costs a KL of
  0.05-0.06 (outlier channels).
- **`exp_gemm`, `exp_act`** run a share of the Flash-Next prefill's routed
  experts (`gpu/experts.rs`, `offload::Experts`). `exp_gemm` is a grouped
  int8 x int4 `dp4a` GEMM over slots of one expert's tokens, built in four
  forms. `-DGU` gives gate / up (`exp_gu.spv`), without it down
  (`exp_down.spv`), and `-DTOK=32u` / `16u` / `8u` give kernels for slots of
  at most that many tokens (`exp_gu32.spv` / `exp_down32.spv`, and so on). A
  workgroup computes all `TOK` tokens, so a slot of 3 tokens on the 32-token
  kernel costs as much as a full one. `exp_xq` rounds the MoE input to int8
  once per token before them, and `exp_act` computes SwiGLU, the down input's
  range, the Hadamard transform and the int8 rounding.
  `expbench adapter wdir [layer] [tokens]` checks them against the host on a
  layer's pools and times them. They reach 1.65 TOPS with the activations read by
  `subgroupBroadcast` and 1.34 with both operands in shared memory. Larger
  thread tiles spill on the iGPU.
- **`idx_score_npu`** is the Flash-Next prefill's block indexer
  (`gpu/flashattn.rs`, `FlashAttn::score_begin`). Per query and pooled block
  key, it computes the sum over 4 heads of `max(q . k, 0)`, as an int8 x int8 `dp4a`
  GEMM with a scale per row in `exp_gemm`'s scheme (keys in shared memory,
  query heads by `subgroupBroadcast`). The `indexer` test checks it against
  a host model and times 256 queries. It takes 1.46 ms over 13312 blocks, against
  14.5 ms for an f32 form with both operands in shared memory.
- **`dit_gemm_npu`, `dit_row_npu`, `dit_act_npu`, `dit_qkv_npu`** run the
  Qwen-Image transformer's `dit_mid` job (`gpu/ditblk.rs`,
  `offload::DitBlk`). `dit_gemm_npu` is an int8 x int8 `dp4a` GEMM on the
  NPU's row-major `[n][k]` codes (`exp_gemm`'s scheme, 64 tokens x 64 rows)
  with a weight row stride `ldw`, since `d`'s rows carry the NPU graph's 32
  zero pad columns. `dit_row_npu` is the row work around it, built three
  times (`-DMODE=0` / `1` / `2` gives `dit_row0_npu.spv` to
  `dit_row2_npu.spv`), `dit_act_npu` is SwiGLU with the per-row int8, and
  `dit_qkv_npu` writes q / k / v. The `dit_mid` test
  (`QWEN_TEST_ADAPTER=intel`) checks a job against a host model and times
  it.
- **`h3_row_npu`, `h3_act_npu`, `h3_qkv_npu`** run MiniMax-H3-Image's
  `h3/mid` job (`gpu/h3blk.rs`, `offload::H3Blk`) with `dit_gemm_npu` for
  the matmuls: the stored residual times `xr` plus the out projection,
  RMSNorm times the modulation entry plus its shift, the up half's 1 / 256
  before SwiGLU. `h3_row_npu` is built three times: `-DMODE=0 -DW=7168u`
  gives `h3_row0_npu.spv` (the attention width), `-DMODE=1` / `-DMODE=2`
  give `h3_row1_npu.spv` / `h3_row2_npu.spv`. The text encoder's layers
  (`H3Blk` at its widths: d 5120, attention 8192, K / V 1024, MLP 25600)
  use `h3_row_npu` with `-DMODE=0 -DW=8192u` / `-DMODE=1 -DW=5120u` /
  `-DMODE=2 -DW=5120u` (`h3te_row0_npu.spv` to `h3te_row2_npu.spv`),
  `h3_qkv_npu` with `-DA=8192u -DKV=1024u` (`h3te_qkv_npu.spv`) and
  `h3te_act_npu` (its hidden row is past the 64 KB of shared memory, so it
  is rotated in two halves written back over the gate half of `y`). The
  `h3_mid` test (`QWEN_TEST_ADAPTER=intel`) checks a job of both shapes
  against a host model and times it.
- **`h3v_gemm`, `h3v_attn`, `h3v_qk`, `h3v_norm`, `h3v_elem`** run whole
  tiles of MiniMax-H3-Image's VAE decoder (`gpu/h3vae.rs`,
  `offload::H3Vae`) in f32 on the NPU's f16 weights: a 64 x 64 tiled GEMM
  with bias and residual add, attention within a tile (one query per
  thread, keys in chunks of 32), q / k RMSNorm and RoPE, the norms
  (`-DMODE=0` RMSNorm, `-DMODE=1` the head's LayerNorm: `h3v_norm0.spv`,
  `h3v_norm1.spv`) and the elementwise steps (`-DMODE=0` f16 in, `1`
  SwiGLU, `2` f16 out: `h3v_elem0.spv` to `h3v_elem2.spv`). The `h3_vae`
  test checks two tiles through one layer and the head against a host
  model.
- **`vis_rope_npu`, `vis_qk_npu`, `vis_sm_npu`, `vis_pv_npu`** are the
  attention of the NPU engine's vision tower (`gpu/visattn.rs`,
  `offload::VisAttn`). They compute 2-D RoPE into f32 buffers per head, the scores of a
  tile of queries, their exponentials, and P V. They read and write the
  NPU's host buffers (f16) in place. `visbench [adapter] [gw] [gh]` checks
  them against f64 sums and times a block, and `QWEN_VIS_ATTN_TIMING=1` splits
  the time by pass. The kernels' comments record the slower forms.

## Decode GEMVs with f32 activations (`q4k_f32`, `q6k_f32`)

These run every single-position GEMV (`matmul_batch`, and `matmul_batch_seq`
at seq 1). `GpuDevice::gemv_for` / `wants_q8k` pick the kernel and the
activation format together. The q8_K kernels `q4k` / `q6k` remain for the
fused blocks (which quantize on the GPU) and for `QWEN_GPU_Q8GEMV=1`.

Kernel-only rates on the GTX 1060 (`gpubench --gemv-probe`), against a
pure-read ceiling of ~160 GB/s for the same access pattern:

| Weight | q8_K integer kernel | f32 float kernel |
|---|---|---|
| FfnUp `[17408x5120]` Q4_K | 49-51 GB/s | 117-130 GB/s |
| Wq / Wo Q4_K | 46-51 GB/s | 99-127 GB/s |
| FfnDown `[5120x17408]` Q6_K | 78-80 GB/s | 133-149 GB/s |

What makes the difference, in order of size:

1. Float math instead of 32-bit integer multiplies, which Pascal emulates.
2. A nibble becomes a float with a shift and a LOP3, never an `I2F`, as in
   `uintBitsToFloat(0x3F800000 | n << k)` = `1 + n/2^(23-k)`, with the bias
   removed once per sub-block through the activation sum.
3. One shift serves both nibbles of a byte (Q4_K).
4. 32 weights per lane, so the scale decode is amortized.
5. 4 rows per warp share the activation loads.

Precision is *better* than with the q8_K kernels. The maximum error against
an exact f64 product is 0.003 on outputs of ~900, where the q8_K path is off
by ~4.6.

As the grid contract, one warp (`WARPS = 1`) computes `NR = 4` rows, so a workgroup
covers 4 rows, the same `rows_per_wg` as the q8_K kernels. Change the
`#define`s and `rows_per_wg` together.

## Multi-position GEMVs (`q4k_f32_seq`, `q6k_f32_seq`)

`GpuDevice::gemv_dispatch` uses them for every float GEMV with `seq > 1`
(speculative verify batches, and the prefill GEMV below `gemm_min_seq`).
Each warp covers NR = 4 rows x NS = 8 positions. A block is read and
dequantized to float once and applied to every position, where `q4k_f32` /
`q6k_f32` (one position per `grid.y`) read the whole weight again for each
position. For the kernel alone at seq 8, FfnUp Q4_K goes 3075 -> 2100-2520 us, FfnDown
Q6_K 3713 -> 2073-2086 us, and the maximum absolute difference against an f64
reference 0.0030 -> 0.0002 (Q4_K). In the grid contract, x = `grid_x` (NR = 4 like
the others), y = ceil(seq / 8). `QWEN_GPU_NOSEQGEMV=1` runs the
one-position kernels instead.

## Two kernels per quant type, GEMV and GEMM

Each of Q4_K and Q6_K has **two** kernels, and `GpuDevice::use_gemm(kind,
seq)` picks one:

| Phase | Kernel | Shape it is for |
|---|---|---|
| Decode and MTP verify (`seq <= 8`) | `q4k`, `q6k` | one output row per 32-lane team, sequence **position** in `gl_WorkGroupID.y` |
| Prefill (`seq >= 9`) | `gemm_q4k`, `gemm_q6k` | a 128-row x 64-token **tile** per workgroup, token *tile* in `gl_WorkGroupID.y` |

Below 64 tokens the GEMM runs one padded tile. Warps whose 8 tokens are all
padding skip their FMAs (a warp is one token group), which makes the GEMM
~3x the GEMV there.

The GEMV kernels read a weight row again for every (row, position) pair,
which is correct but costs `seq` times the weight traffic. That is fine at
`seq = 1` (decode is DRAM-bound anyway) and catastrophic at `seq = 2048`.
Measured on the GTX 1060, the tiled GEMMs are **12.1x** (Q4_K) and **10.8x**
(Q6_K) faster there, which takes both from ~0.2x the CPU's prefill rate to
~2.2x.

## Thread layouts and grids

`q4k` and `q6k` use **one 32-lane team per output row**, four rows per
workgroup (`local_size = (32, 4)`). The lanes split the *super-block* so
their weight loads form one contiguous 128-byte transaction, and a
shared-memory tree reduces the 32 partial sums at the end. `q4k_f32`,
`q6k_f32` and `dp4a_q4k` use one warp per 4 rows (`subgroupAdd`). `q8_0`,
`f16` and `f32` use **one row per thread** at `local_size_x = 64`.

The tiled GEMMs use a third layout, `local_size = (32, 8)` = 256 threads
covering `BM = 128` rows x `BN = 64` tokens, 4 rows and 8 tokens per thread.

Because these layouts need different workgroup counts, the host must never
compute the grid inline:

- **`GpuDevice::grid_x(kind, out_dim)` is the single source of truth for the
  GEMV kernels**, and `rows_per_wg` next to it holds the per-`GpuKind`
  value.
- **`GpuDevice::gemm_grid(out_dim, seq)` is the source of truth for the
  GEMMs**, whose `GEMM_BM` / `GEMM_BN` must match `BM` / `BN` in the
  `.glsl`.
- Pipeline and grid are always chosen together (`use_gemm`, then
  `gemm_pipeline_for` and `gemm_grid`). Picking one without the other fails
  the same silent way.

A mismatch computes a fraction of the output and leaves the rest at
whatever the buffer held. If you change a kernel's `local_size`, `ROWS`,
`BM` or `BN`, change the host constant in the same commit.

`q6k` and `gemm_q6k` also assume their weights are **repacked to a
212-byte (53-dword) stride** in VRAM instead of the GGUF's 210.
`GpuDevice::upload` does that, and the bytes on disk and the CPU fallback are
untouched. Without it every second block would start 2 bytes off a dword
boundary, and each weight fetch would need a two-load funnel shift.

## Measuring

Measure any change to the GEMV kernels with

```
cargo run --release -p offload --features vulkan --example gpubench -- --check
```

which times them on the real model's matmul shapes and compares the result
against `quant::gemv_q4k` / `gemv_q6k`. `--sweep` varies `out_dim` at a
fixed `in_dim`, which shows the cache collapse of one-row-per-thread
layouts.

For the **GEMM** kernels use `--probe`:

```
glslc -fshader-stage=compute -O gemm_q6k.glsl -o /tmp/v.spv
cargo run --release -p offload --features vulkan --example gpubench -- --probe --q6k --spv /tmp/v.spv
```

`--probe` times the kernel *alone* next to the same shape through
`matmul_batch_seq`, and compares against `quant::apply_linear_batch`.
`--spv` loads the module from disk, so the loop is `glslc` plus a rerun,
with no Rust rebuild. `--q6k` selects the Q6_K `FfnDown` shape, and
`--gemv` adds the GEMV kernel for a comparison on the same buffers. **Do not
judge one of these kernels by an end-to-end timing**, because the wrapper around a
dispatch can cost several times the kernel itself, and an end-to-end number
cannot tell the two apart.
