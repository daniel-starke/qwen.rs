# Architecture

qwen.rs is one Rust workspace that turns a model file into an HTTP endpoint.
Everything between the file and the socket is implemented here, including GGUF
loading, the K-quant kernels, the model graph, the GPU offload, the NPU
engine, the tokenizer, the sampler, the server and the image generator.
llama.cpp is the reference for the algorithms and the correctness baseline, but
none of its code is used.

```
                   ┌───────────────────────────────────────────────┐
  client ─────────▶│ server: HTTP/1.1, Anthropic + OpenAI APIs     │
                   │ prefix cache, tool-call translation, SSE      │
                   └──────────────────────┬────────────────────────┘
                                          │ token ids
                   ┌──────────────────────▼────────────────────────┐
                   │ decode driver: sampler, context drafts,       │
                   │ MTP / DFlash speculative decoding             │
                   └──────┬───────────────────────────────┬────────┘
                          │ GGUF model                    │ .npu model
             ┌────────────▼───────────────┐   ┌───────────▼────────────────┐
             │ model graph (CPU, Vec<f32>)│   │ NPU engine (Level Zero)    │
             │ + offload manager          │   │ + iGPU / GPU helpers       │
             └──────┬──────────────┬──────┘   └────────────────────────────┘
                    │              │
          ┌─────────▼──────┐ ┌─────▼──────────────────┐
          │ quant: AVX2 /  │ │ offload: ash (Vulkan), │
          │ AVX-VNNI GEMM  │ │ precompiled SPIR-V     │
          └────────────────┘ └────────────────────────┘
```

## Crates

| crate | role |
|---|---|
| `gguf` | GGUF v3 parser for the header, typed metadata, tensor infos and memory-mapped tensor data. Nothing is copied, and tensors are read lazily. |
| `quant` | K-quant formats (Q4_K, Q6_K, Q8_0, F16, F32), the q8_K activation format, and the CPU kernels for dequantization, decode GEMVs, prefill GEMMs, causal attention and the delta-net scan. Also the worker pool and the recycling allocator. |
| `model` | The model graph on the CPU, the run state (KV cache, recurrent state), chunked prefill, the two-half prefill pipeline and the speculative decoders (MTP, DFlash, context drafts). |
| `offload` | The Vulkan device (raw `ash`), its kernels and the offload manager that splits matmuls between CPU and GPU. Also the iGPU / GPU helpers of the NPU engine. |
| `tokenizer` | Byte-level BPE from the GGUF vocabulary and merges, the chat template, and the streaming decoder. |
| `sampler` | Temperature, top-k, top-p, min-p, repetition / presence / frequency penalties (on the generated tokens only). |
| `imgio` / `vision` | Image decoding and resampling, the Qwen3-VL vision tower and projector, and the splicing of an image into a prompt. See [vision.md](vision.md). |
| `server` | Thread-per-connection HTTP/1.1, a hand-written JSON codec, the two chat APIs, tool use, profiles, MCP and the Images API. No third-party runtime dependencies. See [server.md](server.md). |
| `prefixcache` | The prefix cache of every engine, with states as parts, KV in shared pages, checkpoints, cost-based eviction and the disk tier. |
| `imagegen` | Image generation: Qwen-Image-2.1 (`qimage`, [image.md](image.md)) and MiniMax-H3-Image (`h3`, [h3image.md](h3image.md)), each with its configs, transformer, VAE and NPU backend, over shared parts at the root (the job interface, the Qwen3-VL text encoder, condition images, ops, weight cache, NPU helpers). |
| `npu` | The NPU engine through Level Zero (the 27B's in `engine.rs`, Qwen3.8-Flash-Next's in `flash/`), and the SHAVE kernel toolchain (`crates/npu/NPU.md`, `crates/npu/SHAVE-DSL.md`). |
| `npubuild` | `cli BF16_DIR --npu-build OUT` runs calibration and GPTQ and builds the graph sets the NPU engine loads, for each model (picked by the checkpoint's configs, and also the image model's NPU set). |
| `ov` | Runs an OpenVINO export through the OpenVINO runtime, which is loaded at run time. Experimental and slower than the NPU engine. |
| `cli` | The binary, with argument parsing, the modes, and the choice of engine based on the model argument. See `crates/cli/README.md` for the reference. |

`imgio` sits below `vision` so that container formats and the tower's
geometry can be tested apart. `vision` is the only crate that knows both the
tokenizer and the model, because splicing an image into a prompt is where
the chat template, the placeholder run and the MRoPE positions must agree.

## The model

The GGUF's `general.architecture` is `qwen35`, a hybrid model:

- 64 decoder layers. Every fourth (`full_attention_interval`) is full causal
  attention with grouped KV heads (24 query heads, 4 KV heads) and an output
  gate. The rest are **gated delta-net** layers. Each has a short causal conv1d and then a
  linear-attention recurrence with a 128-wide state per head.
- A 65th block (`blk.64`, `nextn_predict_layers = 1`) is the
  multi-token-prediction head used by `--mtp`.
- Interleaved MRoPE, where rotary pairs cycle through the (t, h, w) coordinates.
  For text all three are the sequence position, while an image's patches share one
  `t` (see [vision.md](vision.md)).
- RMSNorm, SwiGLU MLPs, a 248k-row output head, and a 262144-token trained
  context.

The forward pass is plain `Vec<f32>` code. A decode step takes O(1) work per
token. KV rows and the delta-net and conv states live in a `RunState` that
persists across tokens, so generation is O(n) in the number of tokens. The
large matmul weights stay quantized as they are in the file. Decode quantizes
the activation to q8_K and runs fused dequant-dot GEMVs. Prefill runs tiled
GEMMs on packed q8_K activation tiles (position tiles, row groups, AVX-VNNI
where present).

## CPU + GPU offload

The 16.8 GB model does not fit in the 6 GiB card, so no single backend holds
it. Instead every large matmul is **split by rows**:

- **Decode.** At load, the offload manager measures the free VRAM
  (`VK_EXT_memory_budget`) and keeps the first part of each matrix's rows
  resident on the GPU, about a third of every matrix on the reference card.
  Each decode matmul computes those rows on the GPU while the CPU computes the
  rest, and both write into one output.
- **Prefill.** A larger share of rows goes to the GPU, namely the resident rows plus
  a range streamed through VRAM per call. The shares are per quant type and
  chunk length, set from measured GPU and CPU GEMM rates (`cli --tune-split`
  measures them). The GPU GEMMs use integer dot products (dp4a) where the
  device passes a self-check.
- **Long chunks** (from 512 tokens) run as a two-half pipeline. The positions are
  split into halves, and each layer's GEMM stages run on one half while the
  CPU does the other half's non-GEMM work (scans, attention, norms). Two GEMM
  batches stay in flight, so the GPU never waits for the host.
- **Attention** of long prompts is also split. The GPU runs some KV groups and
  the CPU the rest.

Only activations cross the bus. Results are written straight into the
caller's matrix (`VK_EXT_external_memory_host`) where available. A device
that is lost (for example, another application claims the VRAM) is rebuilt
and its weights uploaded again, and the CPU carries the whole load meanwhile.

All kernels are GLSL compiled offline to SPIR-V and embedded. No shader
compiler is needed to build. Regenerating them, and the measuring tools
(`gpubench`, `spvstats`), are described in
`crates/offload/src/shaders/README.md`.

Environment variables that tune or switch off each of these mechanisms are
listed under "Environment variables" in `crates/cli/README.md`.

## The NPU engine

`cli MODEL.npu` runs the model on the Intel NPU through Level Zero, on graphs
that `--npu-build` makes from the BF16 checkpoint:

- int4 weights, one scale per row, quantized by GPTQ on block-Hadamard-rotated
  matrices, with static int8 activation ranges from a calibration run.
- Per-layer graphs at 256 and 64 positions for prefill, a 4-position verify
  set plus the MTP block for decode. The weights are bound inputs, so prefill
  and decode share one copy.
- Optional helpers run on Vulkan devices, with the delta-net recurrence and part of the
  long-context attention on the iGPU, and a share of the decode MLPs on the
  discrete GPU. The other decode MLPs run on the CPU cores.
- A 262144-position context. Keys and values past the decode window are kept
  on the host in blocks, and each block's attention is merged by log-sum-exp.
  The iGPU runs part of those blocks, and the CPU cores can take a share too
  (opt-in, `QWEN_NPU_FAR_CPU=1`).
- Speculative decoding with the MTP block and context drafts, the prefix
  cache (bit-exact snapshots of the engine's state) and images (the vision
  tower's per-patch work as NPU graphs, its attention on the iGPU).
- Custom SHAVE kernels, written in the kernel language of
  `crates/npu/SHAVE-DSL.md` and built by our own compiler, assembler and
  linker.

### Building the NPU graph sets

`cli BF16_DIR --npu-build OUT` (`crates/npubuild`) makes everything the
engine loads from the BF16 checkpoint, in five steps:

| Step | Output | What it does |
|---|---|---|
| `emb` | `emb_table.f16`, `tokenizer/` | The embedding table as f16, and the checkpoint's `tokenizer.json`. |
| `weights` | `cal/`, `w/` | The float model runs a calibration text layer by layer. Each layer's four quantized MatMul groups get the static int8 range of their input and **int4 codes with one scale per row, by GPTQ** on the block-Hadamard-rotated weights against the input's Hessian. The MTP block follows the same way, and the lm_head becomes int8. 15 s per layer, 16 min in all (21 min without the GPU). |
| `graphs` | `graphs/**.xml` | The 291 OpenVINO-format graphs cover prefill layers at 256 and 64 positions (whole, and cut around the delta recurrence and the attention), the verify set's layers, their far attention, the heads, the MTP block, and the vision tower (28 graphs over 256 patches, cut at each block's attention, with f16 weights that take 1.8 GB with their blobs). |
| `compile` | `graphs/**.blob` | 259 graphs go through our build of the NPU compiler with our SHAVE kernels (built in-process), six at a time. The two heads, the MTP graphs and the vision tower go through the driver's compiler. 2 min. |
| `sets` | `sets/`, `model*.npu` | The sets as hard links, swizzled copies (`wz/`) of the weight files that graphs read swizzled, and the three `.npu` files. |

In a `--features vulkan` build, the largest factorization of GPTQ runs on
the discrete GPU. It is the inverse Cholesky factor of `down_proj`'s 17408 x
17408 input Hessian, which is most of a layer's quantization time (blocked,
every large step one f32 GEMM in VRAM, `gemm_f32.glsl`). The cores quantize
the layer's other matrices meanwhile, which takes 9.1 s per layer against 13.4 s. It
needs ~4 GB of free VRAM. Without it, or in a plain build, the cores do
everything.

The build makes the sets of the three NPU setups. The engine also accepts a
one-position decode set (`decode` without `mtp`), windowed prefill sets and
64-position ones (`small`, `long`, `long_small`), which the build does not
make.

The hardware and compiler behavior this relies on is documented in
`crates/npu/NPU.md`. The patches to the Intel NPU compiler are described in
`crates/npu/compiler-patches/README.md`.

## Qwen3.8-Flash-Next

A `.npu` file with `arch = flash` runs `Qwen/Qwen3.8-Flash-Next` on the same
engine machinery (`crates/npu/src/flash`). It is a 125B mixture of experts
with 6B active per token. It has 48 layers (3 delta-net, then 1 attention with a
block indexer that selects keys past 2048 positions), 512 routed experts
(top-10) plus a shared expert in every layer, a 4-stream residual
(hyper-connections), an n-gram embedding at layer 1 and one MTP layer. The
tokenizer and chat template are the 27B's.

- **Weights.** Dense matrices are int8 per row, routed experts int4 by GPTQ (~60 GB,
  resident in host memory and bound to the NPU per job), lm_head int8. The
  n-gram table (~100 GB, BF16) stays on the SSD and is read per token,
  prefetched a piece ahead in prefill.
- **Prefill** runs on the NPU and the iGPU only, layer by layer over pieces of
  up to 4096 tokens. Each 256-position sub-chunk runs an NPU graph before and
  after the mixer, with the delta-net recurrence or the attention on the iGPU
  in between. The sub-chunks are pipelined over two buffer banks. One pass of
  expert jobs then covers the whole piece, so each expert's weights are read
  once per piece.
- **Prefill experts** are split between the NPU and the iGPU (int8 x int4
  dp4a on the NPU's host buffers, imported zero-copy). Whole NPU jobs move to
  the iGPU as long as that lowers the larger of the two devices' estimated
  times. The estimates keep their shape in code and learn one scale factor
  per device from the run's own timings (`quant::calib`, stored per
  executable and frozen once fitted). Without an iGPU the cores run the
  recurrence and the attention.
- **Decode** with a discrete GPU (`dense = geforce`) keeps every dense
  weight, the delta-net mixers and their states in VRAM and runs a delta
  layer in one GPU submission. The routed experts run on the CPU cores
  (DRAM-bound) while the GPU runs the shared expert, and the lm_head is split
  between the GPU and the cores. Without the GPU, a one-position NPU graph
  set decodes.
- **Verify steps** of up to 8 positions check the MTP head's drafts (the head
  runs on the cores) and context drafts. A step of 2 or more positions runs
  as two waves a layer apart, each wave's next GPU run queued behind the
  other's, so the GPU computes one wave's dense part while the cores run the
  other wave's routed experts and the host its attention. Every query takes
  the same split-K attention path whatever the step size, so a verify step
  computes exactly what one-position decode does. Prompt suffixes of up to 96
  tokens run as such steps instead of as a prefill piece.
- Snapshots, the prefix cache and images work as on the 27B's engine.

The `.npu` keys and the measured speeds are in the README ("Device setups"
and "Run").

## Speculative decoding

Every speculative path drafts some tokens, verifies `1 + N` positions in one
batched pass of the main model, and keeps the longest prefix the main model
agrees with. Every emitted token is still sampled from the main model's
logits.

- **Context drafts** (on by default) are continuations copied from an earlier
  occurrence of the last few tokens in the prompt or the output, and on the
  server from earlier responses. Large gains on edits and tool calls.
- **MTP** (`--mtp [N]`) has the model's own next-token head draft a chain,
  scored over a reduced vocabulary.
- **DFlash2** (`--dflash FILE`) is a separately downloaded block drafter that
  reads hidden states of five target layers and drafts a block in one pass.

`--mtp` and `--dflash` are for a GGUF. The NPU engines draft with their own
MTP block when the `.npu` file names one, and use context drafts too.

On Flash-Next the drafting is priced. The profile holds the measured cost of
a verify step per number of positions, an MTP chain is cut to the prefix with
the most expected tokens per millisecond, and a context draft is verified only
when its expected tokens per cost (learned per match length and position from
every draft that could have been proposed) beat the MTP steps' recent rate.
Thinking requests use **coupled sampling**. Tokens are drawn by Gumbel-max with
noise fixed by the request's seed and the position, and the MTP head drafts
with the same noise, so a draft matches the sample more often without
changing the distribution sampled from.

Batched verification sums in a different order than one-token decode, so
greedy output can differ at a near-tie token.

## Correctness checks

- Unit tests pin the dequantization, the GEMMs and the GEMVs against scalar
  references, and pin the GPU kernels against the CPU.
- `cli MODEL --forward 369` must give argmax 264 on the reference GGUF.
- The chat template renderer is checked byte for byte against Jinja2 on the
  template copied out of the GGUF (`crates/tokenizer/tests`).
- `QWEN_EXACT_DECODE=1` runs every decode matmul with f32 activations on the
  CPU, as an accuracy reference to compare logit dumps against.
- The NPU engines are measured against BF16 truth logits. `QWEN_NPU_IDS`
  feeds a fixed token file and `QWEN_NPU_DUMP_RESID` / `QWEN_FLASH_DUMP_ALL`
  dump what the engine computed, for a KL divergence against the truth.
