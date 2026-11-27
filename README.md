# qwen.rs

A from-scratch Rust inference server for **Qwen3.8-27B**, **Qwen3.8-Flash-Next**,
**Qwen-Image-2.1-Turbo** and **MiniMax-H3-Image**. It serves an **Anthropic Messages API** and an
**OpenAI Chat Completions / llama-server**-compatible HTTP endpoint, so coding
agents such as Zed and Claude Code can use a local model, and it generates
images through **MCP** and the **OpenAI Images API**.

It uses no llama.cpp, Candle or Burn. Model loading, the quantized kernels,
the model graphs, the vision tower, the tokenizer, the sampler, speculative
decoding, the NPU engine with its kernel compiler, and the HTTP server are all
implemented here.

**It is built for one machine**, and every number in this file was measured
on it:

| Part | Machine |
|---|---|
| CPU | Intel Core Ultra 285K (Arrow Lake, 24 cores, AVX2 and AVX-VNNI, no AVX-512) |
| NPU | Intel AI Boost (Arrow Lake NPU) |
| iGPU | Arrow Lake Xe |
| GPU | NVIDIA GTX 1060 6 GiB |
| RAM | 128 GiB |
| OS | Windows 11 x64 |

The engines that use the NPU support only the Arrow Lake NPU. Everything was
tested only on Windows 11.

## What it does

- **Chat models.** Qwen3.8-27B (dense, hybrid attention and linear-attention
  layers) and Qwen3.8-Flash-Next (125B mixture of experts, 6B active per
  token), both with a 262144-token context.
- **APIs.** Anthropic `POST /v1/messages` (streaming, `count_tokens`), OpenAI
  `POST /v1/chat/completions`, llama-server's `GET /props`, and `GET /health`.
- **Tool use** on both APIs, with forced `tool_choice` and argument types
  taken from each tool's JSON schema.
- **Thinking** per request, with an effort level and a token budget.
- **Images in the conversation** (PNG, JPEG, BMP, Netpbm) on every setup.
- **Image generation** with Qwen-Image-2.1-Turbo (edits with up to 10
  reference images) or MiniMax-H3-Image (4 MP by default, edits with up to 16
  reference images), covering text to image and edits, as MCP tools for
  coding agents and as the OpenAI Images API.
- **Prefix cache.** A follow-up turn, an edited message or a new conversation
  with the same tools prefills only what is new, also across server restarts
  with `--prompt-cache-dir`.
- **Speculative decoding.** Drafts come from earlier text in the conversation (on
  by default), the model's own multi-token-prediction head, or the DFlash2
  drafter. Every token is still sampled from the main model.
- **Profiles.** Sampling defaults and decoding settings per thinking mode and
  context depth, tuned on this machine.

See [docs/architecture.md](docs/architecture.md) for how it works.

## Which setup to use

The model file given as the first argument picks the engine. A `.npu` file
runs the NPU engine, and a `.gguf` file runs the CPU + GPU engine.

**Qwen3.8-27B:**

| Setup | Devices | Prefill tok/s | Decode tok/s | What you need |
|---|---|---|---|---|
| **NPU + CPU + GPU** (recommended) | NPU, iGPU, cores, GTX 1060 | ~130 (~108 at 17k tokens) | ~9.8 code / ~6.3 prose | the BF16 checkpoint (52 GB), our NPU compiler build, a 19-min `--npu-build` |
| NPU + iGPU | NPU, iGPU | ~130 | ~5.5-7 code / ~3.6-3.9 prose | the same build output |
| NPU only | NPU | ~103 (~96 at 12k tokens) | ~5.5 code (~2.9 without drafts from the model's head) | the same build output |
| CPU + GPU | cores, GTX 1060 | ~120-132 | ~4.9, or ~12 code / ~10 prose with DFlash2 | two GGUF downloads (17.7 GB), optionally the DFlash2 drafter (3.8 GB), with nothing to build |

**Qwen3.8-Flash-Next** (NPU engine only):

| Setup | Devices | Prefill tok/s | Decode tok/s | What you need |
|---|---|---|---|---|
| **NPU + CPU + GPU** (recommended) | NPU, iGPU, cores, GTX 1060 | ~238 at 19k tokens, ~235 at 81k | ~17, or ~20-25 with drafts (a Zed tool loop at 12k tokens reaches ~20 thinking and ~23 non-thinking, and ~18 at 55k) | the BF16 checkpoint (336 GB), our NPU compiler build, a `--npu-build` with 165 GB of output (its `weights` step alone ~50 min), ~75 GB of RAM while running |
| NPU + iGPU | NPU, iGPU | ~149 at 3.2k | ~5.1 | the same build output |
| NPU only | NPU | ~146 at 3.2k | ~5.2 | the same build output |

- The NPU setups leave devices free. NPU + iGPU keeps the cores and the GTX
  1060 free for other work, and NPU only also frees the iGPU.
- **Images in the conversation** work on every setup. On the NPU engines the
  vision tower runs on the NPU, with its attention on the iGPU (on the cores
  without one). A 240-token image takes 0.7 s, a 1000-token one 6.5 s. On CPU
  \+ GPU they take 2.9 s and 20 s.
- **The CPU + GPU setup is the quickest start.** Download two files, build
  `cli`, run.

## Quick start (CPU + GPU)

1. Install Rust with the `x86_64-pc-windows-gnu` target and mingw-w64 (see
   [Building](#building)).
2. Build with `cargo build --release -p cli --features vulkan`.
3. Download `Qwen3.8-27B-Q4_K_M.gguf` and `mmproj-Qwen3.8-27B-BF16.gguf` (see
   [Model files](#model-files)).
4. Serve:

   ```bash
   target/release/cli.exe path/to/Qwen3.8-27B-Q4_K_M.gguf --serve 8080 --ctx 262144 \
       --mmproj path/to/mmproj-Qwen3.8-27B-BF16.gguf --prompt-cache-dir path/to/prompt-cache
   ```

5. Connect a client, either [Zed](#zed) or [Claude Code](#claude-code).

For the NPU setups, build the NPU compiler and the graph sets first. See
[NPU graph sets](#npu-graph-sets-qwen38-27b).

## Building

| Item | Requirement |
|---|---|
| Rust | 2021 edition, with `cargo`, developed with a nightly toolchain (1.96) |
| Linker | the `x86_64-pc-windows-gnu` target and a mingw-w64 linker (`x86_64-w64-mingw32-gcc`) |
| Vulkan | nothing to build, since the shaders are precompiled and embedded. At run time it uses the GPU driver's `vulkan-1.dll` |

```bash
# every setup (the Vulkan path drives the GTX 1060 and the iGPU)
cargo build --release -p cli --features vulkan

# NPU only, or CPU only
cargo build --release -p cli
```

The binary is `target/release/cli.exe`. It is statically linked and loads
everything else (Vulkan, the NPU's Level Zero loader, OpenVINO) at run time,
only when a setup uses it. `cargo build --profile deploy -p cli --features vulkan`
builds the same binary with link-time optimization and without debug
symbols, the form to install.

What each setup needs at run time:

| Setup | Needs |
|---|---|
| CPU + GPU | the NVIDIA driver (Vulkan 1.1). Without a Vulkan device the run stays on the CPU. |
| NPU setups | the [Intel NPU driver](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html) 32.0.100.4404 (the version our compiler build matches, which you pick in the page's version selector) and its Level Zero loader `ze_loader.dll`. For the iGPU and GTX 1060 helpers, a `--features vulkan` build and the GPU drivers. No OpenVINO. |
| OpenVINO export | the OpenVINO runtime (2026.3.1 tested). See [The NPU through OpenVINO](#the-npu-through-openvino-experimental). |

## Model files

### CPU + GPU (Qwen3.8-27B GGUF)

| File | Needed for | Download |
|---|---|---|
| `Qwen3.8-27B-Q4_K_M.gguf` (16.8 GB) | always | [lmstudio-community/Qwen3.8-27B-GGUF][gguf-q4] |
| `mmproj-Qwen3.8-27B-BF16.gguf` (931 MB) | images (`--mmproj`) | [lmstudio-community/Qwen3.8-27B-GGUF][gguf-mmproj] |
| `model.safetensors` (3.8 GB) + `config.json` | the DFlash2 drafter (`--dflash`) | [z-lab/Qwen3.8-27B-DFlash2][dflash-st], [config][dflash-cfg] |

The model's own multi-token-prediction head (`--mtp`) is part of the main
GGUF and needs no extra file.

The links are pinned to the Hugging Face commits this code was tested against,
so they keep serving the same bytes if the repositories are updated:

| File | Bytes | SHA-256 |
|---|---|---|
| `Qwen3.8-27B-Q4_K_M.gguf` | 16810714336 | `e00082f779fa385cee8c68a3ec8833a75778cc87272240b942f74e0b8243e520` |
| `mmproj-Qwen3.8-27B-BF16.gguf` | 931145856 | `97ba9d70e7407f08c880def231fd360a312c76d0053387733f10dcf6affd75a1` |
| DFlash2 `model.safetensors` | 3848817896 | `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c` |

[gguf-q4]: https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF/resolve/5a7da681f60570ab5b439a587e912d2e5eddb582/Qwen3.8-27B-Q4_K_M.gguf
[gguf-mmproj]: https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF/resolve/5a7da681f60570ab5b439a587e912d2e5eddb582/mmproj-Qwen3.8-27B-BF16.gguf
[dflash-st]: https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/resolve/50307d4c4cde6860d4eee73e2547cd786fe8e8a4/model.safetensors
[dflash-cfg]: https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/resolve/50307d4c4cde6860d4eee73e2547cd786fe8e8a4/config.json

The DFlash loader reads `<name>.json` beside the `.safetensors` file, else
`config.json` in its directory, so either keep the two files together or
rename the config to match (e.g. `Qwen3.8-27B-DFlash2.safetensors` and
`Qwen3.8-27B-DFlash2.json`).

**Memory.** A load peaks at ≈21.4 GiB of RAM. The KV cache costs **64 KiB per
token of `--ctx`** (f16), committed up front, so 1 GiB at 16k, 8 GiB at 128k,
16 GiB at the native 262144. A state the server keeps in its prefix cache
costs 64 KiB per position it holds (pages shared with the conversation's
other checkpoints) plus ~150 MB of recurrent state.

### The NPU compiler

The NPU setups run graphs compiled by our build of Intel's NPU compiler,
[`openvinotoolkit/npu_compiler`](https://github.com/openvinotoolkit/npu_compiler)
at tag `npu_ud_2025_44_rc1` with the patches in `crates/npu/compiler-patches`,
built against OpenVINO commit `80e587bccec` (MSVC 2022, CMake, Ninja, ~40 min
for the first build). The steps are in
[crates/npu/compiler-patches/README.md](crates/npu/compiler-patches/README.md).
Only `vpux-opt` and `vpux-translate` are needed, and only to build graph sets,
not to run them. A directory named `npu-compiler` beside `cli.exe` is used
when neither `--npu-compiler` nor `NPUC_BIN` is given.

### NPU graph sets (Qwen3.8-27B)

The NPU engine does not read the GGUF. Everything it loads is built from the
original **BF16** checkpoint, [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
(revision `1d4bf0f2ff60`, 18 safetensors shards, 52 GB, with its
`tokenizer.json` and `config.json`), by one command:

```bash
cli path/to/Qwen3.8-27B-BF16 --npu-build path/to/npu \
    --npu-compiler path/to/npu-compiler/bin
```

It needs the checkpoint, ~25 GB for the output, the NPU compiler (see above, though
`NPUC_BIN` can replace `--npu-compiler`) and the NPU with its driver. No
Python and no OpenVINO. It takes 19 minutes. Each step is skipped when its
output exists, and an interrupted build continues after the last finished
layer. [docs/architecture.md](docs/architecture.md) describes each step
("Building the NPU graph sets"), and `crates/cli/README.md` lists every option.

The output holds the graph sets and one `.npu` file per NPU setup
(`model.npu`, `model_igpu.npu`, `model_npu.npu`), a text file that names the
sets with paths relative to itself. The output directory can be moved as a
whole, but `sets/` cannot be moved without the rest. `w/` holds the weights
as plain files, and decoding reads them from there.

Accuracy against the BF16 model (KL divergence of the next-token
distribution over a 256-token text that is not the calibration text) is 0.037,
with the same top-1 token at 93-94 % of the positions.

### Flash-Next graph sets

The same command on the BF16 checkpoint of
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(revision `de4b8e4d43b9`, 131 shards, 336 GB) detects the model and builds
its files:

```bash
cli path/to/Qwen3.8-Flash-Next-BF16 --npu-build path/to/flash \
    --npu-compiler path/to/npu-compiler/bin --npu-calib-chunks 24
```

It calibrates on the frozen texts in `crates/npubuild/calib` (code and
public-domain poetry), built into the binary, so the result does not depend
on the source tree.

The output is ~165 GB, made up of the n-gram embedding table (96 GB, read from the SSD
per token while running, so keep it on a fast local drive) and the weights
(62 GB). The `weights` step reads every expert of the checkpoint and takes
~50 min. It also resumes after the last finished layer. It writes the same
three `.npu` files.

| Step | Output | Contents |
|---|---|---|
| `emb` | `emb_table.f16`, `tokenizer/` | the embedding table and the tokenizer |
| `ngram` | `ngram.bf16` | the n-gram embedding shards concatenated (320M rows of 160 BF16) |
| `weights` | `cal/`, `w/` | calibration through the float model, dense matrices int8 per row, routed experts int4 per row by GPTQ, and routers, norms and hyper-connection matrices f16 |
| `head` | `lm_head.f16` | the lm_head as f16 (the GPU decode path quantizes it to int8 at load) |
| `graphs`, `compile` | `graphs/flash/*` | the per-layer prefill and one-position graphs and the expert jobs |
| `vis` | `sets/vis` | the vision tower (the 27B's, merger output 2560) |
| `mtp` | `mtp/` | the multi-token-prediction layer, int8 per row (routed experts int4) |
| `sets` | `model*.npu` | the three `.npu` files |

### Qwen-Image

| File | Download |
|---|---|
| `Qwen/Qwen-Image-2.1-Turbo` (BF16 diffusers layout, 30.2 GB without `assets/`) | [Qwen/Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo), revision `fc13ba1986fc` |

The CPU backend reads the checkpoint directory as is. `--npu-build` on it
writes the NPU set (`image.npu`, ~6 min with the NPU compiler).

### MiniMax-H3-Image

| File | Download |
|---|---|
| `MiniMaxAI/MiniMax-H3-Image` (diffusers layout, ~135 GB) | [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image), revision `8ced60068623` |

Only the diffusers layout is read (`transformer_diffusers/`,
`vae_diffusers/`, `text_encoder/`, `tokenizer/`, `processor/` and the JSON
files); the native layout (`transformer/`, `vae/`,
`pdd_fused_heads.safetensors`) can be left out. The CPU backend converts the
weights to Q6_K at load; `--image-cache DIR` keeps the converted weights
(~34 GB) so later loads skip the conversion. `--npu-build` on it writes the
NPU set (`image.npu`, ~2 h with the NPU compiler, ~45 GB of weights).

## Running

The model path is always the **first** argument and paths below are
placeholders. Every example serves `http://localhost:8080` at the largest
context its setup has, ready for [Zed](#zed). Replace `--serve 8080` with
`--interactive` for a chat in the terminal, or with `--generate "PROMPT"
--max-new 64 --temp 0` for one answer.

### Qwen3.8-27B, NPU + CPU + GPU (recommended)

```bash
cli path/to/npu/model.npu --serve 8080 --host 127.0.0.1 --prompt-cache-dir path/to/prompt-cache \
    --image-max-tokens 512
```

`model.npu`, as `--npu-build` writes it:

```ini
# prefill: 256 positions, then the far sets for positions past the window
prefill   = sets/g1c
set       = sets/f1c
set       = sets/f1h
# decode: 4-position verify set + MTP drafts, far form past its 4096 window
decode    = sets/dec4vm
mtp       = sets/mtp
fardec    = sets/f4d
emb       = emb_table.f16
tokenizer = tokenizer
# images: the tower's per-patch work on the NPU, its attention on the `scan` adapter (else the cores)
vision    = sets/vis
# helpers: delta recurrence and far blocks on the iGPU, MLP share on the discrete GPU
scan      = intel
mlpgpu    = geforce
```

`scan` and `mlpgpu` name a Vulkan adapter by a part of its device name and
need a `--features vulkan` build. Which device does what:

| Phase | NPU | iGPU (`scan`) | CPU cores | GTX 1060 (`mlpgpu`) |
|---|---|---|---|---|
| an image | the vision tower's per-patch work (28 graphs of 256 patches) | the tower's attention | resize and patches | none |
| prefill | every layer's matrix multiplications, norms, attention | the linear-attention recurrence, a share of the long-context attention | merging the long-context attention's results | none |
| decode | each layer's attention or recurrence, the output head, the drafts | none | each layer's MLP | a share of the MLP's rows (10240 of 17408, held in VRAM) |

- The context is 262144 positions. It is fixed by the graph sets, so there is
  no `--ctx`. Past 4096 positions the MLPs run on the cores alone (the GTX
  share would arrive too late).
- NPU device memory holds three prefill sets next to the verify set, the
  drafts' set and `fardec`. A fourth fails to initialize.
- When another application takes the GTX 1060's VRAM, the cores take over
  its share, and it is uploaded again once the memory is back.
- **Images.** `--image FILE` (repeatable) with `--generate`, `/image FILE` in
  `--interactive`, and image blocks over both APIs. There is no `--mmproj`, because
  the tower comes from the checkpoint. `--image-max-tokens N` (default 1024)
  caps each image. An image that stays in the conversation is encoded and
  prefilled once, because the prefix cache keys it by a hash of its pixels.

```bash
cli path/to/npu/model.npu --generate "Describe this image." --image pic.png --image-max-tokens 256
```

### Qwen3.8-27B, NPU + iGPU

`model_igpu.npu` is `model.npu` without `mlpgpu`, with a decode set that
keeps the MLPs on the NPU (`sets/dec4vww`). The cores and the GTX 1060 stay
free.

```bash
cli path/to/npu/model_igpu.npu --serve 8080 --host 127.0.0.1 --prompt-cache-dir path/to/prompt-cache
```

### Qwen3.8-27B, NPU only

`model_npu.npu` is `model_igpu.npu` without `scan`, with the same sets and the
same context. The NPU then also runs the recurrence and all of the
long-context attention. The cores only merge the attention's results (they
need AVX2, F16C and FMA). A build without `--features vulkan` is enough.

```bash
cli path/to/npu/model_npu.npu --serve 8080 --host 127.0.0.1 --prompt-cache-dir path/to/prompt-cache
```

A 5.8k-token prompt prefills at 103 tok/s (130 with the iGPU) and decodes at 5.6
tok/s. The vision tower's attention runs on the cores, so a 240-token image takes
1.1 s, a 1000-token one 12 s.

### Qwen3.8-Flash-Next

```bash
cli path/to/flash/model.npu --serve 8080 --host 127.0.0.1 --prompt-cache-dir path/to/prompt-cache
```

Its `model.npu` starts with `arch = flash`:

```ini
# Flash-Next, NPU + CPU + GPU: decode's dense part and lm_head on the GTX 1060, routed experts on the cores
arch      = flash
graphs    = graphs/flash
weights   = w
ngram     = ngram.bf16
emb       = emb_table.f16
head      = lm_head.f16
tokenizer = tokenizer
# images: the tower's per-patch work on the NPU, its attention on the `scan` adapter (else the cores)
vision    = sets/vis
# helper: delta recurrence and prefill attention on the iGPU
scan      = intel
dense     = geforce
# MTP drafts (on the cores, needs `dense`)
mtp       = mtp
```

- **Prefill** runs on the NPU and the iGPU, layer by layer over pieces of up
  to 4096 tokens. The cores and the GTX 1060 stay free. Each layer's experts
  are split between the NPU and the iGPU by estimated time. The split learns
  from the run's own timings and stores what it learned (`QWEN_CALIB`). It
  learns again after a rebuild.
- **Decode** runs the dense matrices and the output head on the GTX 1060
  (`dense = geforce`) and the 10 active experts per layer on the cores.
  Without `dense` the dense part runs on the NPU, and the
  multi-token-prediction drafts are not available.
- **RAM.** The working set is ~75 GB. The experts stay resident.
- **GPU-shared memory limit.** What the NPU, the iGPU and the GTX 1060
  keep mapped counts against one Windows limit for all of them together,
  dxdiag's "Shared Memory" (by default 80 % of RAM, or ~102 GiB with
  128 GiB). Past it a device is lost and every later request fails until a
  restart. Flash-Next holds ~88 GiB after loading and ~94 GiB at the full
  262144-token context. For more headroom (e.g. image jobs late in a long
  conversation), raise the limit and reboot:
  ```bat
  reg add HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\MemoryManager /v SystemPartitionCommitLimitPercentageMax /t REG_DWORD /d 90 /f
  ```
  With 128 GiB this gives ~111.5 GiB (RAM minus 16 GB). Check dxdiag. The
  Intel driver's "Shared GPU/NPU Memory Override" sets only the minimum
  and does not raise this limit. Microsoft warns that changing the value
  can destabilize the system. `reg delete` the value to undo.
- Accuracy against the BF16 model is KL 0.05 on a 3220-token text, with the same
  top-1 token at ~94 % of the positions, and KL 0.11-0.13 on a 256-token text, at
  ~85 %.
- Images, the prefix cache, `--interactive`, tool use and both APIs work as
  on the 27B. There is no GGUF and no OpenVINO path for this model.

### Qwen3.8-27B, CPU + GPU

```bash
M=path/to/Qwen3.8-27B-Q4_K_M.gguf
cli "$M" --serve 8080 --host 127.0.0.1 \
    --ctx 262144 \
    --prompt-cache-mb 65536 \
    --prompt-cache-dir path/to/prompt-cache \
    --dflash path/to/Qwen3.8-27B-DFlash2.safetensors \
    --mmproj path/to/mmproj-Qwen3.8-27B-BF16.gguf --image-max-tokens 512
```

Drop `--mmproj` for text only, and `--dflash` without the drafter (`--mtp`
instead needs no extra file).

- `--ctx 262144` is the model's trained context. Its KV cache (up to 16 GiB)
  lives in RAM, so the GPU's size does not limit it. On a machine with less
  RAM, lower `--ctx` (it costs RAM, not speed) or `--prompt-cache-mb`.
- `--prompt-cache-mb 65536` keeps a few long conversations. The cache is
  charged 64 KiB per position a state holds (pages shared between a
  conversation's checkpoints) plus ~150 MB of recurrent state per checkpoint.
- `--image-max-tokens 512` halves the default per-image cost (the tower runs
  at ~34 tok/s, so a 512-token image takes ~15 s plus its prefill) and keeps
  screenshots legible. Use 256 for photos and diagrams, 1024 for dense text.
- Leave `--n-gpu-layers` at 0. The default split of every matrix between the
  GPU and the cores uses the VRAM better than whole layers do.

With an image, from the terminal:

```bash
cli "$M" --generate "Describe this image." \
    --mmproj path/to/mmproj-Qwen3.8-27B-BF16.gguf --image pic.png
```

### Image generation (Qwen-Image-2.1)

`Qwen/Qwen-Image-2.1-Turbo` does text to image and edits with up to 10 reference
images, 9 denoising steps by default. From the checkpoint directory it runs
on the cores. The `image.npu` set runs the transformer, the VAE's
convolutions and the vision tower on the NPU, the attention on the cores and
a share of the blocks on the iGPU. One 1 MP image on `image.npu` takes
~1.5 min, an edit with one reference ~2 min. Peak RAM is ~21-24 GB.

```bash
# the NPU set (~6 min; needs the BF16 checkpoint, the NPU compiler and the NPU)
cli path/to/Qwen-Image-2.1-Turbo --npu-build path/to/qimage --npu-compiler path/to/npu-compiler/bin

# one image, or an edit with --image (repeatable)
cli path/to/qimage/image.npu --txt2img "a lighthouse at dusk, oil painting" -o out.png
cli path/to/qimage/image.npu --txt2img "make it winter" --image in.png -o out.png

# served next to a chat model (loaded on the first image job, dropped after --image-idle)
cli path/to/npu/model.npu --serve 8080 --image-model path/to/qimage/image.npu --image-out-dir path/to/images
```

- **MCP** (`POST /mcp`, Streamable HTTP) offers the tools `generate_image` and
  `edit_image` (aspect or size, seed, steps, transparent background, and an
  `output_path` inside the client's roots or `--image-out-dir`), progress per
  step, cancellation. `cli --mcp-stdio [URL]` bridges clients that only start
  commands.
- **OpenAI Images API.** `POST /v1/images/generations` and
  `/v1/images/edits` (`b64_json` responses).
- One job runs at a time across the devices. Chat requests wait while an
  image job runs, and vice versa. `cli path/to/qimage/image.npu --serve 8080`
  serves the image routes alone.

See [docs/image.md](docs/image.md) for details.

### Image generation (MiniMax-H3-Image)

`MiniMaxAI/MiniMax-H3-Image` does text to image and edits with up to 16
reference images (`<Picture N>` in the prompt), 6 fixed steps, 4 MP by
default and up to 16 MP. The same commands, MCP tools and routes as
Qwen-Image apply; the path picks the model, one image model per server. From
the checkpoint directory it runs on the cores (Q6_K, a 4 MP image ~10 min).
The `image.npu` set runs the text encoder, the transformer and the VAE
decoder on the NPU, with shares on the iGPU and on a discrete Vulkan card
(a 4 MP image ~3 min). Resident memory is ~35 GB.

```bash
# the NPU set (~2 h; needs the BF16 checkpoint, the NPU compiler and the NPU)
cli path/to/MiniMax-H3-Image --npu-build path/to/h3 --npu-compiler path/to/npu-compiler/bin --image-cache path/to/h3-cache

cli path/to/h3/image.npu --txt2img "a lighthouse at dusk, oil painting" -o out.png
cli path/to/npu/model.npu --serve 8080 --image-model path/to/h3/image.npu --image-out-dir path/to/images
```

See [docs/h3image.md](docs/h3image.md) for details.

### The NPU through OpenVINO (experimental)

Given an OpenVINO export directory instead of a `.npu` file, `cli` runs the
export through the OpenVINO runtime (`openvino_c.dll`, loaded at run time, with
`--ov-lib DIR` if it is not in the default install). It needs no graph sets
and no NPU compiler, but it is far slower than the NPU engine (~1.6 tok/s
decode, ~11 tok/s prefill), text only, without a prefix cache, and with a
context compiled into its graphs.

```bash
cli path/to/Qwen3.8-27B-int4sym-ov --serve 8080 --host 127.0.0.1 \
    --ov-device NPU --ov-ctx 4096 --ov-chunk 32 --ov-cache path/to/ovcache
```

A first NPU compile takes several minutes per graph. Later starts load from
`--ov-cache` (blobs are 1-14 GB each, so keep it off a small system drive).
The export is made from the BF16 checkpoint with `optimum-intel`, as
symmetric int4 (the form the NPU dequantizes itself):

```bash
optimum-cli export openvino -m path/to/Qwen3.8-27B-BF16 --task image-text-to-text \
    --weight-format int4 --sym --group-size 128 --ratio 1.0 path/to/Qwen3.8-27B-int4sym-ov
```

## Clients

### Zed

The **Anthropic-compatible** provider supports everything (thinking toggle
and effort, thinking blocks, tools, images, `count_tokens`). In
`settings.json`:

```json
    "language_models": {
        "anthropic_compatible": {
            "qwen.rs": {
                "api_url": "http://localhost:8080",
                "available_models": [
                    {
                        "name": "Qwen3.8",
                        "display_name": "Qwen3.8",
                        "max_tokens": 262144,
                        "max_output_tokens": 32768,
                        "mode": {
                            "type": "adaptive"
                        },
                        "capabilities": {
                            "tools": true,
                            "images": true,
                            "prompt_caching": false
                        }
                    }
                ]
            }
        }
    }
```

- The same entry serves every model. The server does not check the request's
  `model` and echoes it back.
- Without `mode`, Zed shows no thinking toggle.
- The API key (Settings > AI > LLM Providers > qwen.rs) can be any text.
- Zed sends `temperature: 1.0` on every request. The server drops it by
  default (`--client-ignore`), so the profile's temperature applies.

The **llama.cpp** provider works too (API URL `http://localhost:8080`, and Zed
reads the context window from `/props`), without the thinking toggle.

**Image generation as Zed tools.** Start the server with `--image-model` and
with `--image-out-dir DIR` pointing at a directory that holds your projects
(Zed declares no MCP roots, so without it no `output_path` is writable), and
add:

```json
"context_servers": { "qwen-image": { "url": "http://localhost:8080/mcp" } },
"context_server_timeout": 600
```

Zed's default timeout of 60 s cancels 1-megapixel jobs. 600 s is its
maximum.

### Claude Code

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 \
ANTHROPIC_API_KEY=sk-local-dummy \
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929 \
ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-sonnet-4-5-20250929 \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
claude
```

- `ANTHROPIC_MODEL` must be an id **Claude Code recognizes**, because it rejects
  unknown ones before any HTTP call. The server ignores the field and echoes
  it back.
- Claude Code sizes its context to the Claude model it thinks it is talking
  to (200k), not to the server's. A request longer than the context is
  refused with a 400 rather than compacted, so serve the full 262144 or
  `/compact` before the conversation outgrows the window.

## Configuration

### Main flags

| Flag | Default | Effect |
|---|---|---|
| `--serve [PORT]`, `--host H` | 8080, 127.0.0.1 | HTTP server |
| `--interactive`, `--generate [PROMPT]` | | terminal chat / one answer (`@FILE` reads the prompt from a file) |
| `--system TEXT` | | system prompt (CLI) |
| `--think off\|low\|medium\|xhigh`, `--no-preserve-thinking` | CLI `medium`, server off on `/v1/messages` and on on `/v1/chat/completions` | thinking and its effort. With `--serve`, the default of requests that do not say |
| `--max-new`, `--temp`, `--top-k`, `--top-p`, `--min-p`, `--presence-penalty`, `--rep`, `--seed` | the profile's | sampling (CLI) |
| `--profile FILE` | `QWEN_PROFILE`, else `<model>.profile` beside the model, else built in | sampling and decoding settings per thinking mode and context depth |
| `--tune-profile OUT` | | measure this machine's decoding settings and write a profile |
| `--client-ignore PREFIX=FIELD[,FIELD...]` | `*=temperature` | on the server, drop these request fields of clients whose `User-Agent` starts with `PREFIX` (`*` = all, `off` = none) |
| `--prompt-cache-mb N` | 32768 | server prefix-cache budget, `0` disables it |
| `--prompt-cache-dir DIR` | off | disk tier, where shared prefixes persist across restarts |
| `--prompt-cache-disk-mb N` | `--prompt-cache-mb` | disk tier budget |
| `--prompt-cache-spill` | off | states evicted from RAM go to the disk tier (only those worth their SSD wear) |
| `--image-model DIR\|image.npu`, `--image-out-dir DIR`, `--image-idle SECS` | off, none, 300 | image generation next to chat on the server |
| `--progress`, `--no-keep-awake` | off, keep awake | progress bars on stderr, and let Windows sleep (by default the process keeps the display on, because the iGPU stalls in standby) |
| `--record-requests DIR` | off | save every POST request to `DIR` on the server (`crates/server/examples/replay.rs` posts them again) |
| `--ctx N` | 262144 | GGUF context window. **Always set it for the server** (Zed reads it from `/props`) |
| `--batch N` | 2048 | GGUF prefill chunk |
| `--mtp [N]` | off (bare = 3) | GGUF drafts from the model's multi-token-prediction head |
| `--dflash FILE` | off | the DFlash2 drafter for GGUF, exclusive with `--mtp` |
| `--mmproj FILE` | off | GGUF vision projector. Without it an image is an error |
| `--image FILE` | | image for `--generate` / the first chat turn (repeatable) |
| `--image-max-tokens N` | 1024 | per-image cap in tokens (the vision cost setting) |
| `--n-gpu-layers N` | 0 | whole GGUF layers on the GPU. The default row split is faster |

[crates/cli/README.md](crates/cli/README.md) lists every flag and environment variable.

### Profiles

Sampling defaults and the speculative-decoding settings come from a profile,
per thinking mode and context depth. Without one, the model cards' values
apply. Thinking uses temperature 1.0, top-p 0.95, top-k 20, and non-thinking uses 0.7,
0.8, 20 and a presence penalty of 1.5 on the generated tokens. A request for
temperature 0 gets no penalties it did not ask for. A request's own `top_p`
and other sampling fields take precedence, but its `temperature` is dropped
by default so the profile's applies (`--serve --client-ignore off` honors it,
as `docs/server.md` explains).

`profiles/` holds the profiles tuned on this machine. A file named like the
model with `.profile` beside it is picked up by `--serve`, `--generate` and
`--interactive`. To measure a new one (it runs a coding-agent conversation
up to the context size, which takes hours at full context, so use `--tune-quick` and
`--tune-depths 1,16384` for a first look):

```bash
cli "$M" --tune-profile "${M%.*}.profile" --ctx 262144 --mtp
cli MODEL.npu --tune-profile MODEL.profile
```

`crates/server/src/profile.rs` documents the format.

### Thinking

Thinking is off by default on the Anthropic API, because reasoning tokens
cost decode time. A request enables it with `"thinking": {"type":
"enabled"}`. The OpenAI / llama-server API (`/v1/chat/completions`) turns it
on by default, as llama-server does (Zed's llama.cpp provider sends no
switch). A request turns it off with `chat_template_kwargs.enable_thinking:
false`. `--serve --think off` makes off the default of both.

### The NPU is an exclusive device

A process on the NPU engine holds a machine-wide lock until it exits. A
second one prints `npu: waiting for another process to release the NPU` and
waits. The NPU engine serves one request at a time.

## Caveats

- **Slow first turn.** Prefill on the 27B is ~120-130 tok/s, so an agent's
  ~30k-token preamble costs ~4 minutes the first time. With
  `--prompt-cache-dir` that happens once per machine rather than once per
  session.
- **Decode slows as the context grows** (on CPU + GPU, ~4.9 tok/s short and ~4.1
  at 20k tokens), since every token reads the keys of the attention layers.
- **One request at a time on the NPU engines**, and a second one waits. On a
  GGUF, requests run concurrently. An image job holds every device, so chat
  waits while it runs.
- **Parallel tool calls are parsed but rarely produced.** The model nearly
  always emits one call per turn.
- **No server-side tools.** `web_search` and other tools declared by `type`
  without an `input_schema` are skipped, since nothing here could run them.
- **Three models only.** The graphs, the chat template and the vision tower
  are written for Qwen3.8-27B, Qwen3.8-Flash-Next and Qwen-Image-2.1-Turbo.
  Other GGUFs or checkpoints will not load usefully.
- **Vision limitations.** An mmproj (or a checkpoint) using deepstack layers
  is refused. The NPU engine's tower computes in half precision, so its
  embeddings differ from the CPU + GPU setup's by 0.4-0.8 % (cosine
  0.99997). Images are resized with a Pillow-compatible bicubic filter (as
  the Hugging Face processor does) where llama.cpp uses bilinear, so
  pixel-exact agreement with llama.cpp is not expected.
- **The server does not fetch.** Remote `http(s)` image URLs are refused, so
  send base64 or a `data:` URL. There is no authentication, no rate limiting
  and no request size policy, so bind the server to localhost.
- **Speculative and batched paths are not bit-exact** against one-token
  decoding. They sum in a different order, so greedy output can differ at a
  near-tie token. Every token is still sampled from the main model.

## Development

To run the tests, use `cargo test --workspace --release` and
`cargo test -p offload --features vulkan -- --test-threads=1` (each GPU test
builds its own device, and they do not fit the card concurrently).

Cross-compiling to Windows from WSL needs incremental compilation off, or the
build fails on a session-lock error. The binary then lands under
`target/x86_64-pc-windows-gnu/release/`:

```bash
cargo build --release -p cli --features vulkan --config 'profile."*".incremental=false'
```

The crates the build pulls in are `ash` 0.38 (only with the `vulkan` feature),
`fancy-regex` 0.13 (the tokenizer's pre-tokenizer) and `image` 0.25 (PNG /
JPEG / BMP / PNM decoding only), plus 26 transitive crates pinned in
`Cargo.lock`. No C code is in the build; the build scripts only write the
linker map and deflate embedded data (shaders, tables, texts) with
`miniz_oxide`, which `image` links anyway. Binaries carry no build-machine
paths (cargo `trim-paths`, nightly).

Needed only for specific tasks:

| Task | Needs |
|---|---|
| Editing a compute shader (`crates/offload/src/shaders/*.glsl`) | `glslc` (shaderc v2026.3 or newer, for `GL_EXT_integer_dot_product`). See `crates/offload/src/shaders/README.md` |
| GPU tests and benchmarks (`cargo test -p offload --features vulkan`, `gpubench`, `spvstats`) | a Vulkan device, and for `spvstats` also `VK_KHR_pipeline_executable_properties` |
| Regenerating test fixtures (`crates/*/tests/*.py`) | Python 3 with `jinja2` (chat template), `Pillow` and `numpy` (image / vision references) |
| Correctness reference | a llama.cpp build (`llama-cli`, or LM Studio's `llama-server` for vision) |
| OpenVINO experiments (`crates/ov/scripts`) | Python 3 with `openvino` (2026.4.0 wheel), `numpy`, `openvino_genai` (`genai_ref.py`) |
| Image generation reference dumps (`crates/imagegen/scripts/qimage_refdump.py`) | Python 3 with `diffusers` 0.42 and `transformers` 5 (ComfyUI's embedded Python has both) |
| NPU experiments (`crates/npu/scripts`, which rewrite an OpenVINO export's graphs and which nothing else needs) | Python 3 with `openvino` and `numpy`, an OpenVINO export, a POSIX `sh` and PowerShell |
| Building or writing NPU SHAVE kernels (`crates/npu/shave/*.sk`) | nothing extra, because our own kernel compiler, assembler and linker are included (`cargo build --release -p npu --example shave`, with the language in `crates/npu/SHAVE-DSL.md`) |
| Adding an instruction to the SHAVE kernel language | the NPU, because `crates/npu/scripts/shavedev.py` measures it on the device (`crates/npu/NPU.md` §2 "Provenance") |

## Documentation

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | How it works, covering crates, the model, the CPU + GPU row split, the NPU engine and its build, Qwen3.8-Flash-Next, speculative decoding |
| [docs/server.md](docs/server.md) | The HTTP APIs, tool use and the model's call format, the prefix cache, clients |
| [docs/image.md](docs/image.md) | Image generation, covering the model, the NPU and CPU backends, MCP and the Images API |
| [docs/h3image.md](docs/h3image.md) | Image generation with MiniMax-H3-Image, covering the model, the NPU and CPU backends and their measurements |
| [docs/vision.md](docs/vision.md) | Running images through the model, and the invariants of the vision path |
| [docs/optimizations.md](docs/optimizations.md) | Optimization techniques that proved successful and those measured and rejected, with relative gains, to shorten work on a new model path |
| [crates/cli/README.md](crates/cli/README.md) | Every command-line flag and environment variable |
| [crates/offload/src/shaders/README.md](crates/offload/src/shaders/README.md) | The compute shaders, how to rebuild them, and the measuring tools |
| [crates/npu/NPU.md](crates/npu/NPU.md) | The NPU hardware and compiler as this project uses them |
| [crates/npu/SHAVE-DSL.md](crates/npu/SHAVE-DSL.md) | The SHAVE kernel language and its compiler |
| [crates/npu/compiler-patches/README.md](crates/npu/compiler-patches/README.md) | The patches to the Intel NPU compiler, and how to build it |

## License

See [LICENSE](LICENSE).

## Disclaimers

### AI assistance

This software was developed with the assistance of AI-based tools.
All design decisions and final content remain the responsibility of the human author.

Users should be aware that AI-assisted code generation may introduce subtle errors,
security vulnerabilities or suboptimal patterns that standard review processes
might not catch. Independent review and thorough testing before use in production
environments is strongly recommended.

### Affiliation

This project is an independent, open source project and is **not affiliated, associated,
authorized, endorsed by or in any way officially connected with Alibaba Group Holding Limited**,
Alibaba Cloud or any of its subsidiaries or affiliates.

All product names, logos, brands, trademarks, and registered trademarks mentioned or referenced in
this project are the property of their respective owners.
