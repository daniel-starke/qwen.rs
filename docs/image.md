# Image generation (Qwen-Image-2.1)

Text to image and edits with up to 10 reference images (`crates/imagegen`),
on the command line (`--txt2img`) or served as MCP and the OpenAI Images API
(`--image-model`, `--mcp-stdio`). There are two backends, the quantized CPU
backend, which reads the checkpoint directory, and the NPU backend, which
runs the transformer, the VAE's convolutions and the vision tower on the NPU
from the set that `cli DIR --npu-build OUT` writes (`OUT/image.npu`).

## Usage

```
# one image, or an edit with --image (repeatable), on the CPU backend (checkpoint directory)
cli QWEN_IMAGE_DIR --txt2img "PROMPT" -o out.png [--size WxH] [--seed N] [--steps N]
cli QWEN_IMAGE_DIR --txt2img "PROMPT" --image A.png [--image B.png]... -o out.png [--resolution 1024]

# the NPU set (~6 min, 7.0 GB), then the same commands on image.npu
cli QWEN_IMAGE_DIR --npu-build OUT [--npu-compiler DIR]
cli OUT/image.npu --txt2img "PROMPT" -o out.png

# served next to a chat model, alone, or over stdio
cli MODEL --serve 8080 --image-model QWEN_IMAGE_DIR|OUT/image.npu [--image-out-dir DIR]... [--image-idle 300]
cli QWEN_IMAGE_DIR --serve 8080
cli --mcp-stdio http://127.0.0.1:8080
```

## Model and files

The model is **`Qwen/Qwen-Image-2.1-Turbo`**, the newest open release (Qwen's
own 8-step distillation of `Qwen/Qwen-Image-2.1`, BF16, license
`qwen-research`). Qwen-Image-3.0 has **no open weights** and is an API model
only (`qwen-image-3.0` / `-pro` on Alibaba Cloud Model Studio and Qwen
Studio). The code is organized so that a later open release needs new
configs plus the blocks it changes.

- **Checkpoint.** `Qwen/Qwen-Image-2.1-Turbo` in the BF16 diffusers layout
  (`model_index.json`, `processor/`, `scheduler/`, `text_encoder/` in one
  safetensors file, `transformer/` in two shards, `vae/`), commit
  `fc13ba1986fc`, 30.2 GB without `assets/`.
- **Reference code.** The reference is diffusers 0.42.1's `QwenImage21Pipeline` for the
  accuracy gates, and ComfyUI 0.37's port as a second reading
  (`comfy/ldm/qwen_image21/model.py`, `comfy/text_encoders/qwen_image21.py`,
  `comfy/ldm/wan/vae2_2.py`, `QwenImage21` in `comfy/latent_formats.py`).
  ComfyUI's embedded Python ships both.
- **Not used.** The unsloth Q4_K_M GGUF and ComfyUI's
  `qwen3vl_8b_int8_convrot` text encoder (third-party requantizations of the
  same checkpoint, while the BF16 is quantized here), and `Qwen-Image-2.1-PE-T2I` /
  `-PE-I2I` (9B prompt rewriters, because the calling agent is an LLM and writes the
  detailed prompt itself).

## The model

**Text encoder.** Qwen3-VL (`text_encoder/`, 36 layers, 4096 wide, GQA
32/8, head 128, vocab 151936, vision tower 27 blocks, 1152 wide, patch 16,
merge 2, **deepstack** mergers). Prompt:

```
<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n
<|im_start|>user\n{<imageN><|vision_start|><|image_pad|><|vision_end|> ...}{prompt}<|im_end|>\n
<|im_start|>assistant\n
```

The output is the **last layer's hidden states without the final RMSNorm**
(transformers 4.57's `hidden_states[-1]`, which Qwen tuned to), and the system
turn (everything before the second `<|im_start|>`) is dropped. For an edit,
the reference images run through the vision tower so the text tokens see
them, then their vision tokens are **dropped** and each image's position in
the remaining sequence is recorded (`image_slots`). In the DiT the image is
represented by its VAE latents at that slot instead.

**Transformer** (`QwenImage21Transformer2DModel`, 7.1B) has 32
**single-stream** blocks, 4096 wide, 32 heads x 128, SwiGLU MLP 12288
(`gate_up` fused), QK RMSNorm, 3-axis RoPE (16 / 56 / 56, theta 10000).
- `txt_in` is zero-centered RMSNorm (stored weight + 1) -> Linear -> GELU(tanh)
  -> Linear. `img_in` is Linear 64 -> 4096 (patch size 1, so one token per
  latent pixel, i.e. per 16x16 image pixels).
- Sequence = text with each reference image's latent tokens spliced in at
  its slot, then the target image's tokens. The RoPE ids are text `(p, p, p)` and
  an image `(p, h, w)` with h / w centered on 0 on its own grid, after which
  `p += max(h, w)` (details in "Edits").
- **One modulation for all blocks.** `SiLU(temb) -> Linear(4096, 4 x 4096)`
  = scale1, gate1, scale2, gate2 (gates `tanh`), no shift. A block computes
  `x += gate1 * attn(LN(x) * (1 + scale1))`, `x += gate2 * mlp(LN(x) * (1 +
  scale2))`, LayerNorm without affine.
- **Prefix rows (text + references) modulate with t = 0**, the target rows
  with the sampled t. Attention is **block-causal**. Text is causal, and an image
  block sees everything up to its end. So the prefix does not depend on the
  target or the step. Its K / V per layer are computed **once per image**
  and every step runs the target rows only, attending to `[prefix K/V,
  target]` (`causal_condition: true`, "prefix KV caching" on the card).
- The output is LayerNorm * (1 + scale) (scale from `norm_out.linear`), Linear 4096
  -> 64.

**VAE** (`AutoencoderKLQwenImage21`) uses the Wan-2.2 VAE layout with
temporal kernel 1 (all convs effectively 2-D), encoder width 96, decoder
width 144, `dim_mult` 1/2/4/8/8, 2 res blocks, z 64, **4 image channels
(RGBA)**, 16x spatial. Latents are normalized per channel with the
config's `latents_mean` / `latents_std`.

**Sampling** (Turbo) uses flow matching, Euler, CFG 1 (no negative pass) and the
8 sigmas saved in `model_index.json` `sample_sigmas`, `1.0 0.978453 0.95418
0.926626 0.89508 0.845148 0.704534 0.414568` (then 0), **plus one step at
0.3 by default** (9 steps, see "Steps" below). The velocity form is
`x_{i+1} = x_i + (s_{i+1} - s_i) * v(x_i, s_i)`, timestep fed as `s`
(x1000 rounding as ComfyUI). The base model would use 40 steps with the
dynamic exponential shift (`base_shift 0.5 @ 256`, `max_shift 0.9 @ 8192`
tokens, `shift_terminal 0.02`), which is supported by the same scheduler code, not a
default.

### Steps

`steps` (MCP and OpenAI requests, `--txt2img --steps N`,
`imagegen::Options::steps`, `Schedule::for_steps`) works as follows. **0 / absent = 9**,
the saved sigmas plus `REFINE_SIGMA` 0.3 inside the last interval, 8 =
the saved schedule, and any other count (1-50) resamples the saved curve,
terminal 0 included, piecewise linear in the step fraction.
`QWEN_IMAGE_SIGMAS=a,b,...` overrides the schedule (an experiment).

Turbo's last saved sigma is 0.41, and that jump to 0 smooths fine texture
away (waxy skin, soft fur and cobblestones). Measured at 1 MP on the NPU
backend, seed 42, a photo / a night street with lettering / a watercolor,
detail = std of the luminance Laplacian, drift = RMS difference to the
8-step image at 1/16 scale:

| Schedule | Transformer | Detail vs 8 | Drift | Look |
|---|---|---|---|---|
| 8 (saved) | 78 s | baseline | baseline | clean, skin waxy |
| 6 (0.978, 0.927 dropped) | 60 s | -11 to +2 % | 0.04-0.10 | composition changes, worse |
| **9 (+ 0.3)** | **89 s** | **+9 %** (photo) | **0.009** | **texture, nothing overdone** |
| 9 (+ 0.2) | 89 s | +5 to +10 % | 0.011-0.014 | skin slightly cracked |
| 9 (+ 0.1) | 89 s | +4 % (photo) | 0.009 | weaker |
| 10 (+ 0.56, 0.2) | 99 s | +9 to +15 % | 0.016-0.020 | as 9, slower |
| 12 / 16 / 25 resampled | 118 / 157 / 241 s | +8 to +28 % | 0.03-0.15 | another image, harsh from ~12, over-sharpened at 25 |

The model card says other schedules are not evaluated, and the one published
test of more steps (20 at 4 MP) "overcooks". Splitting only the last
interval keeps every step the model was distilled on and the composition.

## Interface

### MCP (Streamable HTTP, `POST /mcp`)

The chat server (`crates/server/src/mcp.rs`) speaks JSON-RPC 2.0, MCP
protocol 2025-06-18 (older versions accepted), with `initialize` (it returns an
`Mcp-Session-Id`), `notifications/initialized`, `tools/list`, `tools/call`,
`ping` and `notifications/cancelled`, and it asks the client `roots/list`.
`cli --mcp-stdio [URL]` is a stdio-to-HTTP bridge for clients that only
start commands, and that process loads no model.

Tools:

- `generate_image` `{prompt, aspect?: "1:1"|"4:3"|"3:4"|"3:2"|"2:3"|"16:9"|"9:16", width?, height?, seed?, steps?, transparent?, output_path?, return?: "path"|"inline"|"both"}`
- `edit_image` takes the same plus `images: [string]` (1-10 absolute paths or
  `data:` URLs).

**Defaults.** **1 MP** (aspect 1:1 = 1024x1024, other aspects the same area,
with sides that are multiples of 32, and an edit without a size takes its last
image's aspect), a random seed (returned), and `return` = `both` when
`output_path` is given, else `inline`.

**Results.** A text block (path, size, seed, seconds, and the model load
when this call paid for it) and an `image` block. The file is the
full-resolution PNG (RGBA when `transparent`, while `transparent: false`
composites the output over white and writes RGB). The inline image is a
preview scaled to at most 1024 px on its long side (Pillow-exact Lanczos),
so it does not flood the agent's context.

**Progress and cancellation.** `tools/call` answers as an SSE stream when
the client accepts one, else as one JSON body. Progress
(`notifications/progress` with the request's `progressToken`) goes out per
load quarter, stage and denoising step, with `progress` strictly increasing
over a `total` of 11 (load 0-1, text encoder 1, VAE encoder 1.5, step `i` at
`1 + i`, VAE 10). `notifications/cancelled` stops the job at the next stage
or step, and the request gets no response (a stream just ends), and a client
that disconnects mid-stream cancels too.

**Write paths.** `output_path` must lie inside one of the client's MCP roots
or under `--image-out-dir DIR` (repeatable). Anything else is a tool error,
reported before any work. The server asks for the roots with `roots/list`
inside the call's stream when the client declared the capability, and the
answer arrives as its own POST, is cached per session and is dropped on
`notifications/roots/list_changed`. Relative paths resolve against the first
root. Without `output_path` and with `--image-out-dir`, the server names the
file in the first such directory.

Client configuration:

```jsonc
// Zed settings.json
"context_servers": { "qwen-image": { "url": "http://localhost:8080/mcp" } },
"context_server_timeout": 600   // seconds; Zed's default of 60 cancels image jobs, 600 is its maximum
// OpenCode opencode.json
"mcp": { "qwen-image": { "type": "remote", "url": "http://localhost:8080/mcp" } }
// stdio-only clients
"command": "cli", "args": ["--mcp-stdio", "http://localhost:8080"]
```

### OpenAI Images API

`POST /v1/images/generations` `{prompt, n?, size?: "WxH"|"auto", seed?, steps?,
background?: "transparent"|"opaque"|"auto", response_format?: "b64_json"}`
and `POST /v1/images/edits` (multipart `image` / `image[]` plus `prompt`, or
the same JSON with `images` as `data:` URLs) return
`{created, data: [{b64_json, seed}]}` (`crates/server/src/images.rs`).
`n` is 1-4 (seeds `seed`, `seed + 1`, ...), `size` is `WxH` (multiples of 32,
256-2048) or `auto`, and `response_format` is `b64_json` only, so there are no
`url` responses, since the server stores nothing.

## Server behavior

- `--image-model DIR|image.npu` names the model (the checkpoint directory
  for the CPU backend, the build output's `image.npu` for the NPU backend).
  It is **loaded on the first image request**,
  not at startup. `cli QWEN_IMAGE_DIR|image.npu --serve` serves the image
  routes alone.
- **Chat waits.** One job at a time runs on the shared devices. An image job
  queues behind running chat requests and vice versa
  (`server::images::DEVICES`, which chat requests share and an image job holds
  alone).
- **VRAM.** None is held (GTX 1060 unused), and iGPU buffers are freed after each job.
- **RAM.** The model (on the CPU backend, Q6_K text encoder 5.7 GB + transformer
  5.7 GB, and on the NPU backend, the text encoder + 7 GB of NPU weights)
  stays **`--image-idle` seconds (default 300) after the last image job**,
  then is dropped, and the recycling allocator's cached blocks with it
  (`quant::alloc::trim`, since otherwise up to `QWEN_ALLOC_CACHE_MB` of freed
  weights would stay resident).
- **NPU.** The image graphs load next to the chat engine's sets. Graphs
  and recorded command lists of both share one driver memory pool, and a
  1 MP job does not fit beside Flash-Next's prefill set, recorded lists and
  vision tower (`0x70000002`). So each image job first has the chat engine
  unload its NPU graphs (`Engine::release_devices`; Flash-Next keeps every
  buffer and its state, and loads the graphs again at its next prefill
  chunk or image, at no measurable prefill cost), and after a job the
  model unloads its NPU graphs in turn: the transformer's (its weights
  stay), the VAE decoder and the vision tower, all loaded again by the
  next job; with them loaded, the chat engine's next prefill fails the same
  way. A job that
  fails with an error unloads the whole model, so the next chat request
  has the device's memory. Each layer's command list is recorded per run
  and dropped after it, so the image model holds no lists between jobs. The NPU lock
  (`Global\qwen-rs-npu`) is taken on a thread of its own
  (`NpuLock::shared`), because the idle unload drops the model on another
  thread than the one that loaded it and a mutex belongs to its thread. It
  is not taken at all when this process holds it already (an NPU chat
  engine).
- **iGPU.** The transformer's iGPU block share imports every block's int8
  matrices (7.0 GB) and is built only while the process's host imports stay
  within `QWEN_GPU_IMPORT_MB` (default 24 GiB). Next to Flash-Next (22.7 GB
  of expert pools imported) it is not built, and the NPU runs every chunk
  (512x512 transformer ~31 s instead of ~25 s). Past that limit (29.6 GB of
  imports) the NPU is lost for good at the chat engine's next prefill set
  switch.

## Cost and device placement

Image generation is compute-bound throughout, so **the NPU takes the
compute-bound work, and the GTX 1060 is not used** (it holds no VRAM). At
1 MP (64 x 64 = 4096 target tokens), each step takes matmuls of 2 x 7.0e9 x 4096
= 57 TFLOP and attention of ~9 TFLOP. 8 steps take ~530 TFLOP, plus the prefix (once)
and the VAE decode. The measured time is ~1.5 min per 1 MP image ("NPU performance", and a
2048x2048 image is 4x the tokens and ~5-6x the time).

Where each part runs on the NPU backend:

| Part | Device | Why |
|---|---|---|
| Text encoder (Qwen3-VL prefill) | cores (Q6_K) | 0.3 s per prompt, 6-8 s with one 1 MP reference, 83 s with ten (10.4k tokens) |
| Reference images, vision tower | **NPU** + attention on the cores ("NPU performance") | 5.0-6.0 s per 1 MP image |
| Reference images, VAE encoder | **NPU** 3x3 convs + cores | 2.3 s per 1 MP image |
| DiT per-token work (modulated LN, QKV, out proj, MLP, gated residuals) | **NPU**, W8A8 ("NPU backend") | |
| DiT q / k row scales, QK norm, RoPE | cores, f32, with the attention | ~3 ms of SHAVE work per NPU job, ~1 s of the cores per 1 MP image |
| DiT attention (target x [prefix, target]) | **cores**, int8 on AVX-VNNI, overlapped with the NPU (`quant::attn_block_rows_k8` on a `quant::K8` packed per block from the NPU's f16 K / V) | ~4.3 TFLOP/s (f32 on the f16 K / V reaches ~1.6), and an iGPU share is slower next to the cores (see "Rejected") |
| Modulation, timestep embedding, `txt_in`, `img_in`, `proj_out`, Euler | cores | tiny |
| VAE decoder | **NPU** 3x3 convs over halo tiles + cores | 5.9-6.3 s at 1 MP |

Transformer weights are int8 per row, with no calibration (int4 per row is no
faster and has 3-4x the error, see "Rejected").

## Speed and memory

Speed:
- **Prefix K/V once per image.** Text + reference rows run through the 32
  blocks once, and every step runs only the target rows (8 steps x 4096 rows at
  1 MP instead of 8 x (4096 + prefix)).
- No CFG pass (Turbo, CFG 1), so one DiT pass per step.
- The NPU's 256-row graphs over the target rows, overlapped with the
  cores' attention chunk by chunk ("NPU performance").
- The text encoder needs no decode, no KV beyond the prompt, no lm_head.
- The CPU backend uses `quant`'s VNNI GEMMs on quantized weights,
  never f32 GEMMs on dequantized ones.

The memory budget while loaded at 1 MP is **<= 24 GB RAM, 0 VRAM**.
- Weights (NPU backend) are DiT int8 7.0 GB in NPU host buffers, text
  encoder Q6_K 5.7 GB (no lm_head), vision tower and VAE convolutions f16
  in NPU host buffers.
- Never a whole model as f32 or BF16 in RAM. The CPU reference converts
  BF16 per layer from a memory map, and the built weights are read straight
  into their buffers (as Flash-Next's graph sets).
- Activations are processed in 256-row chunks, so the MLP hidden state of a
  chunk is 256 x 24576 x 2 B = 12 MB, not 4096 x 24576 x 4 B = 400 MB.
  Prefix K/V takes 32 x 2 x prefix x 4096 x 2 B (~160 MB at 300 tokens).
  VAE decoding by row tiles with halos (the last level at 1 MP would be
  1024 x 1024 x 144 x 2 B = 300 MB per tensor untiled).
- With many reference images, the prefix K / V (~2.2 GB per 1 MP reference
  in f16) go int8 and the text encoder is dropped for the denoising when
  they do not fit ("NPU performance"). **An edit with 5 or more
  references at 1 MP may exceed the budget** (10 references take
  25.8 GB = 12 GB resident + 11.1 GB int8 K / V + 2.7 GB prefix-pass
  buffers), while 1-4 references stay within 24 GB.
- Idle unload (server) frees RAM after 5 min and iGPU buffers after each job.

## CPU backend

### Reference forward (f32)

`crates/imagegen` has `config` (the JSONs), `te`, `dit`, `vae`, `sched`, `rng`,
`pipeline` (`txt2img`), `refdump` (reads the dumps). f32 on the cores, and every
weight is read from the BF16 checkpoint per layer (per use, as the DiT's 32
blocks are re-read each step), so RSS stays at one layer plus activations.

Facts the port pins (each checked by `crates/imagegen/tests/qimage_reference.rs`):

- **Text encoder output** is the last layer **before** the final RMSNorm
  (transformers 5.x normalizes it, giving wrong conditioning and no error). The system
  turn is dropped up to the second `<|im_start|>` (14 tokens). Text-only
  positions are equal on all MRoPE axes, so this is plain rotate-half RoPE, theta 5e6.
- `txt_in.text_norm` is **zero-centered** (weight + 1), and the DiT's QK norms and
  the TE's norms are plain RMSNorm.
- Prefix rows modulate with **t = 0**, the target with the step's t, and
  modulation rows are `[scale1, gate1, scale2, gate2]`, gates `tanh`. The
  timestep embedding is `[cos, sin]` of `1000 t` (cos first), `linear_1 ->
  SiLU -> linear_2`, with no biases anywhere in the DiT.
- DiT RoPE uses the adjacent-pair (complex) layout, axes 16 / 56 / 56 = 8 / 28 / 28
  pairs (frame, height, width), theta 10000, and target ids `(txt_len, y, x)`
  with `y` in `-(h - h/2) .. h/2`. Text causal, the target block sees
  everything.
- Turbo's sigmas come from `model_index.json` `sample_sigmas` (scheduler
  `use_dynamic_shifting: false`, shift 1), so a step count does not change
  them. The model's timestep is `(sigma * 1000) / 1000` in f32, and Euler is `x +=
  (s[i+1] - s[i]) v`.
- **Initial noise** is torch's CPU `randn((1, 1, 64, h, w), generator=
  manual_seed(seed))` (`imagegen::rng`, mt19937, 24-bit uniforms, Box-Muller
  per 16), so a seed gives the diffusers / ComfyUI image (rel. diff 3e-8 from
  SLEEF vs libm in the last bit).
- **VAE.** Latents are de-normalized per channel (`latents_std`, `latents_mean`),
  `post_quant_conv`, then the decoder. RMS norms are `F.normalize` over
  channels x sqrt(C) x gamma. Up blocks 0-2 are `upsample3d` (they hold a
  `time_conv`, **unused** for one frame), block 3 `upsample2d`, block 4 none, and
  each upsampling block adds `DupUp3D` of its input, with output channel `c` at
  `(2y+a, 2x+b)` = input channel `(c ft 4 + (ft-1) 4 + 2a + b) / repeats`
  (for the first chunk, the second temporal half when `ft` = 2). Output RGBA clamped
  to [-1, 1], bytes `round((v/2 + 0.5) 255)`.

Gate measurements (512x512 = 1024 target tokens, 8 steps, 50
prompt tokens of which 36 kept, CPU f32, 24 threads):

| Check | Result |
|---|---|
| per module vs diffusers (each fed the reference's inputs) | rel. error of TE layers <= 2.2e-6, `txt_in` / temb / modulation <= 2.4e-7, DiT blocks (step 0) <= 1.4e-6, velocity steps 0-7 <= 3.0e-6, VAE stages <= 5.4e-6 (gate 1e-3) |
| image vs diffusers, same seed (own noise) | max abs pixel diff **2 / 255**, mean 0.0023, 0.001 % of values > 1 |
| time (`cli --txt2img`) | text encoder 6.9 s, transformer 264 s (step 0 with the prefix ~41 s, then ~32 s per step), VAE 5.0 s. diffusers f32 on the same cores takes ~10 s per step, VAE 4.5 s |
| peak working set | **3.67 GB** |

The reference re-reads and widens 7.1B BF16 weights each step and runs f32
GEMMs, and the CPU backend (quantized) and the NPU backend replace both.

### Edits

`imagegen::cond` prepares the condition images,
`vision::tower::forward_deep` and `npubuild::vis::model_deepstack` run the
text encoder's vision tower, `TextEncoder::encode_images` (`imagegen::qimage::te`) the text stack with them,
`qimage::vae::Vae::encode` their latents, `qimage::dit::layout` the joint sequence, and
`qimage::pipeline::generate` drives text-to-image and edits alike. Facts the port
pins (each checked by `crates/imagegen/tests/qimage_edit.rs`):

- **One resize feeds both encoders.** Each image is converted to RGBA and
  resized with Pillow's **Lanczos** to `calculate_dimensions(res^2,
  aspect)` (`res` = the pipeline's `output_resolution`, default 1024, sides
  `round(x / 32) * 32` with Python's half-to-even `round`). Pillow resizes
  RGBA through the **premultiplied** `RGBa` form in **fixed point** (22-bit
  taps, the horizontal pass stored back to 8 bits), and `imgio::pil` does the
  same, byte for byte. The VAE reads all four channels as `2 v / 255 - 1`, and
  the vision tower reads the copy composited **over white** (Pillow's
  `paste` with the alpha as mask, `DIV255` rounding). Qwen3-VL's processor
  resize is the identity at these sizes. The target has the **last**
  image's original aspect at `res^2` pixels unless a size is given.
- **Vision tower.** The 27B's Qwen3-VL tower plus **deepstack**. After
  blocks 8 / 16 / 24 comes a merger each (2x2 merge, LayerNorm over the merged
  4608 width, fc1, GELU, fc2 -> 4096), and every merger's GELU (the final one's
  too) is the **exact erf form** (`nn.GELU()`, while the GGUF path keeps
  llama.cpp's tanh form, `VisionModel::gelu_erf`).
- **Text encoder with images.** The user turn starts with
  `<image1><|vision_start|>` + one `<|image_pad|>` per merged token +
  `<|vision_end|>` (` <image2>...` for more, and `<imageN>` is plain text). The
  pads' embeddings are the tower's rows, and the deepstack rows are **added at
  the pads after text layers 0, 1, 2**. In MRoPE, an image's tokens sit at `(t,
  t + row, t + col)`, raster over the merged grid, then `t += max(rows,
  cols)`; interleaved sections 24 / 20 / 20: rotary pair `i` reads height
  for `i % 3 == 1`, width for `i % 3 == 2` (both below 60), time otherwise.
  The output keeps the pad rows (`image_pad`), because the transformer replaces
  them.
- **VAE encoder** (one frame, 2-D). `conv_in` 4 -> 96, five down blocks of
  2 residual blocks (96, 192, 384, 768, 768). The first four downsample
  (zero pad right / bottom, 3x3 stride 2) and add `AvgDown3D` of their
  input, with channel groups of `ft * 2 * 2` values per input channel, averaged
  `in * ft * 4 / out` at a time, where `ft = 2` (blocks 1-3) puts a **zero
  frame first**, so half of each group is zeros. `downsample3d`'s
  `time_conv` is not used on the first frame. Mid block, RMS norm, SiLU,
  `conv_out` -> 128, `quant_conv`, and the latent is the **mean** (first 64
  channels), normalized `(x - latents_mean) / latents_std`.
- **Transformer sequence.** Each text-encoder pad = a **2x2 group of latent
  tokens**, so an image's `n` pads become its `4 n = h w` latent tokens in
  **raster order** (`img_in` rows), and the text rows stay. The RoPE ids
  (`QwenImage21Rope`) are text `(p, p, p)`, `p` advancing, and an image block
  takes `p` where it starts as its frame and its **own** grid centered on 0
  (`-(h - h / 2) .. h / 2`, integer as in diffusers, while ComfyUI's port shifts by
  half a token where the parity differs from the target's, not followed),
  then `p += max(h, w)`, and the target likewise after the last text.
  Attention in the prefix runs by segments, with text causal and each image block
  sees everything before it and all of itself (two images are separate
  blocks even when adjacent), and the target sees everything. The whole prefix
  (text and condition latents) modulates with `t = 0` and is computed once.

Gate measurements (one 512x384 RGBA condition image with an
alpha gradient, `--resolution 512` = 576x448 condition and target (36x28
= 1008 latent tokens each, 252 vision tokens, 1008 patches), 8 steps, 300
text-encoder tokens of which 286 kept, prefix 1042 rows, CPU f32, 24
threads):

| Check | Result |
|---|---|
| preprocessing vs Pillow | resized RGBA **byte-identical** (1032192 bytes), VAE pixels exact, vision pixels rel. 3e-8 |
| per module vs diffusers / transformers (each fed the reference's inputs) | rel. error of vision embeddings 4.3e-5, deepstack 4.7e-6 to 1.8e-5, TE layers <= 3.1e-6 (out 3.2e-6), VAE encoder stages <= 2.2e-5 (latents 5.3e-6), DiT RoPE <= 1.5e-8, blocks (step 0) <= 1.6e-6, velocity steps 0-7 <= 8.2e-7, VAE decoder 7.5e-7 (gate 1e-3) |
| image vs diffusers, same seed (own noise) | max abs pixel diff **1 / 255**, mean 0.0001 |
| time (`cli --txt2img --image`) | text encoder with the vision tower 15.4 s, VAE encoder 1.5 s, transformer 301 s (step 0 with the 1042-row prefix ~70 s, then ~33-38 s per step), VAE 4.9 s, and diffusers f32 takes ~14 s per step |
| peak working set | **3.27 GB** |

Several references were tested with three images (1024x1024, 1024x1024,
1097x617) at `--resolution 384` = 384x384, 384x384, 512x288 (the target
512x288), "Put the fox from image 1 into the room of image 2, in the style
of image 3". Every resized image is byte-identical to Pillow, with text encoder
3.1e-6, vision embeddings 2.5e-5 / deepstack <= 1.8e-5, VAE encoder
stages <= 1.4e-7 rel. at `conv_in`, velocity steps 0-7 <= 3.3e-5, VAE
decoder 7.2e-7 (`tests/qimage_edit.rs` on a `qimage_refdump.py --image A --image B
--image C` dump, 278 s).

### Quantized backend

In `imagegen::qimage::Model` (`QWEN_IMAGE_WEIGHTS`), the text
encoder's and the transformer's matmuls are converted from BF16 to **Q6_K
once at load** (`imagegen::ops::Lin`, ~18 s for both, 5.7 + 5.7 GB) and
run by `quant`'s Q6_K GEMMs (AVX-VNNI) on q8_K activations, while the vision
tower (1152 wide, not a multiple of 256) and the VAE stay f32, read from
the checkpoint per job. Measured on the CPU reference's dumps (512x512,
every step fed the reference's latents):

| Weights | Text encoder out rel. error | Velocity rel. error (max of 8 steps) | Transformer step (1024 tokens) | Resident |
|---|---|---|---|---|
| Q6_K, unrotated | 1.1e-1 | 2.7e-2 | 9.0 s | 11.4 GB |
| **Q6_K, rotated** (default) | 7e-2 | 2.4e-2 | 9.0 s | 11.4 GB |
| Q4_K, rotated | 3.2e-1 | 6.1e-2 | 8.7 s | 7.8 GB |

The 512x512 image end to end (own noise, seed 42) against the f32
reference gives Q6_K rotated PSNR 23.8 dB with 14 % of values off by more than 8,
unrotated 20.0 dB / 18 %, and Q4_K 20.3 dB / 35 %.

- **Rotation.** Every 256-wide block of a weight row and of an activation
  row is multiplied by fixed random signs and the normalized Hadamard
  matrix (`npubuild::la::fwht_blocks`, `npubuild::gptq::signs`) before
  quantization, and the product is unchanged. Qwen3-VL's hidden states carry a
  few channels hundreds of times the rest, and q8_K scales each 256-block
  by its largest value, so unrotated such a block loses the other 255.
  The text encoder's error grows from ~1-2 % per layer to 7 % at the last
  layer either way. This is Q6_K's own weight error, which the diffusion then turns
  into different details, not worse ones. Q4_K saves 3.6 GB for 2.5x the
  velocity error and 3 % of the time, which is not worth it on the cores.
- **Image vs the f32 reference** (512x512, seed 42, the reference's
  prompt) shows the same composition, subject, pose, lighting and lettering, but fine details differ
  (PSNR 23.8 dB), judged by eye as equal quality.
- The MLPs run by 256-row chunks (`ops::MLP_CHUNK`, the gate / up rows of a
  chunk only), and the down projection's q8_K input is formed straight
  from `silu(gate) * up` (rotated in the same pass).
- **Attention.** With one head per KV group (the transformer, the VAE's
  mid block) `quant::attn_*` blocks query rows by position (up to 6
  positions of one head per K / V load), bit-identical per row, and a task
  takes 384 positions (`TQ_MHA`, because a task streams its head's whole K / V
  once, so the tile is what amortizes it). At the 1 MP
  shape (`quant` `bench_attn_block_mha`), it reaches 295 GFLOP/s at 12 positions, 959
  at 48, 1531 at 192, 1596 at 384, 1608 at 768, and takes 6.3 s per 1 MP step.
- **VAE.** A 3x3 convolution builds its padded input per band of rows
  (`qimage::vae::conv3_rows`, 64 MB bands) from a row source, and an upsampling
  block's conv reads the low-resolution map directly (`conv_up2`), so
  neither a whole padded copy (2.4 GB at the last upsampling block at 1
  MP) nor the 2x upsampled map exists.

1 MP (1024x1024, 4096 target tokens, 8 steps, 36 prompt tokens kept, 24
threads), `cli --txt2img`, `QWEN_IMAGE_TIMING=1`:

| Stage | CPU backend |
|---|---|
| model load (Q6_K, both) | 13.5-22 s (from the page cache / cold) |
| text encoder | 0.3 s |
| transformer step, QKV / norm + RoPE / attention / out / MLP | 5.1 / 0.4 / 6.3 / 1.9 / 15.2 s |
| transformer, 8 steps | **233 s** |
| VAE decoder | 17.3 s |
| peak working set | **17.8 GB** (denoise 13.0, VAE 17.4) |

The MLP (40 TFLOP per step at 2.6 TFLOP/s) and QKV are the time, set by the
Q6_K GEMMs at their rate. Inside a running Flash-Next server the same jobs
run ~1.3x slower (1376x768 takes the transformer 311 s). An edit
at 1 MP with one condition image (1024 latent prefix rows plus the text) takes
text encoder with the vision tower 33.6 s, VAE encoder 6.5 s, transformer
412.5 s, VAE 23.6 s.

## NPU backend

Build steps (`npubuild::qimage::build`), each skipped when its output
exists:

- `cal` holds the activation ranges (`OUT/cal/dit.cal`, `imagegen::qimage::npucal`). The
  Q6_K CPU backend generates four calibration images (512x512 / 640x384 /
  448x576) while every block records the largest magnitude of its rotated
  QKV, out and gate / up inputs over the prefix and target rows of all
  steps, with x1.25 headroom, in ~4 min.
- `weights` writes `OUT/w/dit{L}.*.in`, ~1 s per block from the page cache.
- `graphs` and `compile` write `OUT/graphs/dit`, with our compiler, the prefill
  options and no swizzling, in ~15 s.
- `sets` writes `OUT/image.npu` with `arch = qwen-image`, `checkpoint` (absolute,
  symlinks kept), `graphs` and `weights`.

The text encoder, the VAE and the
transformer's small weights (`txt_in`, `img_in`, timestep embedding,
modulation, `norm_out`, `proj_out`) come from the checkpoint as on the CPU
backend.

**Graphs** (`npubuild::qimage`). The 32 blocks are cut at each attention, with
three graphs over 256 rows, one blob each for all blocks (every per-block
value is a bound input), `dit_pre` (block 0 up to its attention, q, k, v),
`dit_mid` (block l - 1 after its attention + block l up to its own),
`dit_post` (block 31 after its attention). The step's modulation (`1 +
scale1`, `gate1`, `1 + scale2`, `gate2`) is an input too, and the prefix
runs through the same graphs at its own modulation (t = 0). q / k / v
leave the graph raw (before their row scales), and the cores apply q / k's
row scales, head norms and RoPE in f32 (`imagegen::qimage::npudit::qk_finish`,
block l's K once all its chunks are done, each chunk's q with its
attention), and v's row scales are folded into `o.pre` (a per-channel
factor of V passes through the attention).

The engine (`imagegen::qimage::npudit`) runs one submission of every chunk per
layer (`batch_each`, with each argument is bound at its chunk's offset in
image-sized host buffers, so nothing is copied between jobs). The block's
attention runs on the cores, overlapped chunk by chunk (`quant::attn_block`
on the f16 K / V where the NPU wrote them, and the prefix's K / V per block are
kept from the first step and put in front, see "NPU performance").

Facts the build pins:

- **W8A8 with a Hadamard rotation.** Each matrix is stored as `W R` (`R` is
  the 27B's random signs, normalized Hadamard blocks of 256), int8 per row
  (no calibration for weights). The graph rotates its input the same way
  and quantizes it, at a calibrated range for QKV, out and gate / up
  (`pre` = signs x `127 / 128 / r`, the range folded into the row scale),
  per row for the down projection's input (the MLP hidden state, whose
  outliers vary, `127 / 128 / max |row|`, the output x `max |row| 128 /
  127`, and the rotation's signs are folded into the up rows, the up half's
  row scales into the down columns, so only the gate half is scaled in
  the graph, `gu.sg`). The residual stream carries a few channels hundreds
  of times the rest (1407, 1669, 3028 reach ~400 against an RMS of ~6 in the
  target rows, and the prefix's reach 9000), so without the rotation one scale
  per tensor would lose every other channel.
- **The activation grid is `k / 128`, not the integers.** With integer
  codes the MatMul's f16 output (before the row scale) of the outlier
  channel overflows to inf in block 0 at the last step of a 512x512 image
  (17 rows of NaN velocity, an empty image).
- **RoPE.** The checkpoint rotates adjacent pairs, and each head's q / k rows
  and QK norm weights are stored permuted (evens, then odds), so the cores
  rotate halves (pair `i` at lanes `i` and `64 + i`), and dot products are
  unchanged.
- QKV are one int8 matrix (rows q, k, v), gate and up another.
- The weights live in NPU host buffers (`zeMemAllocHost`, 6.98 GB), because our
  Level Zero wrapper cannot bind a memory map, so they are read into the
  buffers at load (2.4 s from the page cache).

**Graph times** (one job = one block over 256 rows, `npubench blob`):

- `dit_mid` takes 43.6 ms with f16 activations and int8 weights
  (2.6 TFLOP/s), 25.3 ms as W8A8, and **21.0 ms with the folds** (5.3 TOPS).
  The four MatMuls run at 9.6-10 TOPS of the NPU's 13.1 (2 tiles x 2048 MAC
  x 1.6 GHz) and take 11.7 ms. The rest is 3.1 ms with only DMA active, the
  gate's row scale 0.7 ms, the down input's per-row quantization ~1.5 ms,
  the LayerNorms, the rotations and SwiGLU. `dit_post` takes 17.1 ms (18.4
  unfolded), `dit_pre` 4.5 ms (7.5).
- Without the folds (`QWEN_IMAGE_FOLD=0` at build time, in its own output
  directory, where the engine tells the forms apart by the graphs' inputs), QK
  norm and RoPE cost 2.1 ms of each job, qkv's row scale 0.7 ms and the gu
  fold 1.3 ms.
- 128 / 512 rows per job take 10.75 / 56.6 ms (the 512-row graph spills).
- In the profiled job (19.2 ms), the four MatMuls take 12.3 ms, and between gate / up
  and down 4.35 ms (the gate's scale, Swish and the product on the SHAVEs
  1.75, the rotation 0.5, the per-row quantization 2.1).

**The down input's per-row quantization** runs in a SHAVE kernel
(`shave/rowq.sk`, the default, while `QWEN_IMAGE_ROWQ=0` at build time selects
the stock operations, in its own output directory). A static range cannot
replace it. Over the calibration images the median row's largest value
lies 4-1000x (~1000x in blocks 5-6) below the block's range, which gives
10-1000x the per-row form's quantization noise of 1e-2. A SoftMax over the
row carries the kernel (the compiler never splits a softmax's axis, while an
elementwise carrier reaches the kernel as 64-row x 3080-column tiles). One
pass scales the row to a maximum of 127 / 128 and writes the maximum into
32 pad lanes, and the down MatMul reads the pad lanes against zero columns
(`d.w` is [4096, 12288 + 32]), and the output's rescale slices the maximum
from the kernel's f16 output (slicing the quantized tensor does not
compile). Against the stock operations, `dit_mid` goes 21.0 -> 19.45 ms
(`dit_post` likewise), 1 MP text to image 75.7 -> 74.0 s, with the iGPU's
share now the bound (fewer iGPU chunks, 4 of 16, take 89-90 s), and s512 velocity is
2.0-2.7e-2 (2.0-3.0e-2).

To profile a graph, run `vpuxc.ps1 -Profile -Opts enable-grouped-matmul=true`
with `NPU_RS_DPU1LIST=1 NPU_RS_VIEWFOLD=1 NPU_RS_DISTVIEW=1
NPU_RS_GMM_MAXG=96`, then `NPU_TASKS=1 npubench blob X --profile` and
`scripts/tasks.py`. `examples/qigraph` writes the graphs.

**Per-stage reference data** (`crates/imagegen/examples/qimage_stageref.rs`).
`qimage_stageref record REF QWEN_IMAGE_DIR [--size N]` writes the f32
reference's stages (text encoder output, prefix rows, modulation, the
target rows after every block of the first and the last step, every
step's latents and velocity, the final latents and image), and `qimage_stageref
check REF MODEL [--free]` runs a backend (a checkpoint directory = the
Q6_K CPU backend, or an `image.npu`) on the recorded inputs, stage by
stage, with times, and with `--free` the whole denoising from the recorded
noise against the reference image. The sets used here are `s256` (one NPU
chunk, 132 MB, recorded in 2 min) and `s512` (four chunks, 519 MB, 5 min),
with the prompt "A red fox sitting in a snowy forest at dawn, soft golden
light, photograph" and seed 42. `QWEN_IMAGE_NANCHECK=1` prints the
non-finite values per block of the NPU forward. `examples/qimage_ditstats.rs` prints the
residual stream's largest values per block.

The gate (`qimage_stageref check s512 MODEL --free`, 512x512 = 1024 target tokens,
24 prompt tokens kept, 8 steps, the transformer fed the recorded f32
text-encoder output):

| Check | NPU int8 | CPU Q6_K |
|---|---|---|
| target rows after each block, step 0 (rel. error) | 8.2e-3 to 1.4e-1, 4.0e-2 (block 31) | 1.3e-3 to 5.4e-2, 1.7e-2 |
| the same, step 7 | 4.9e-3 to 7.6e-2, 2.6e-2 | 8.4e-4 to 3.8e-2, 1.4e-2 |
| velocity, steps 0-7 (each fed the reference's latents) | 1.9e-2 to 3.1e-2 (unfolded 1.8e-2 to 3.1e-2) | 8.8e-3 to 1.3e-2 |
| free-running latents / image vs the f32 reference | 6.2e-2 / **PSNR 31.2 dB** | 6.9e-2 / 29.9 dB |
| transformer per step | **2.48 s** with the iGPU's share (3.01 s without, prefix once 0.78 s) | 8.39 s |

The free-running image error is where the trajectory drifts, not the
per-step error, and it is chaotic. A 1-ulp f32 change of every attention
output alone moves it by ~3 dB, and other summation orders give
28.8-32.2 dB, so compare the per-step numbers. The NPU's image is as close to the
reference as the Q6_K one (judged by eye to be the same image). The text encoder
(Q6_K in both) is 9.2e-2 off the f32 one. s256 (one chunk) has velocity
1.7-3.9e-2, VAE decoder 67.7 dB on the recorded latents, free-running
image 30.2 dB. Inside a Flash-Next server the job runs as fast as alone
(512x512 takes 32.2 vs 31.5 s).

## NPU performance

1 MP (1024x1024, 4096 target tokens, 8 steps, `cli image.npu --txt2img`,
seed 42, edit = one 1 MP image, "Make it night, with a full moon", 1060
text-encoder tokens, a 4118-row prefix):

| Stage | Time | How |
|---|---|---|
| model load | ~9 s | text encoder Q6_K ~7 s, NPU weights ~2 s |
| text to image, transformer | **77.9-78.5 s**, 9 steps (the default) **78.7-83.5 s** | NPU, iGPU and attention overlapped (iGPU off 93.6-94.0 s and NPU-bound, with NPU waits of 62 s and int8 attention 30.5 s, f32 attention 95.2-95.5 s) |
| edit, transformer, 9 steps | **100.3-104.9 s** | NPU, iGPU and attention, 16 attention threads (iGPU off 125.3-126.7 s, NPU waits 52-56 s, attention 67-72 s) |
| VAE decoder | **5.9-6.3 s** | 3x3 convs on the NPU |
| edit, vision tower | **5.0-6.0 s** | NPU graphs + deepstack, attention on the cores |
| edit, text encoder (with the tower's rows) | 6.2-7.9 s | cores (Q6_K) |
| edit, VAE encoder | **2.3 s** | stride-1 3x3 convs on the NPU |
| peak working set, text to image / edit | 21.0 / **23.7 GB** | (at the VAE decoder) |

**NPU and attention overlapped** (`imagegen::qimage::npudit::pipeline`). Per
block, once every chunk has its q / k / v, the cores run chunk c's
attention and the next graph's chunk c goes to the NPU, which runs it
while the cores do chunk c + 1. K / V are double-buffered (block l + 1's
written while block l's are read). The prefix pass runs through the same
pipeline (segments of causal text and block-causal images).
- An NPU append (`pfnAppendGraphExecute`) waits while the NPU runs a job
  (~13 ms of a 25 ms job), for any graph instance, so the lists are recorded
  and submitted on a thread of their own (on the main thread appends
  cost 49.6 s of a 1 MP image).
- The descriptor heap does not hold a step's 512 recorded lists
  (`0x70000002` at append), so there is one list per chunk, recorded per use, at most
  32 alive (`MAX_LISTS`, since a 10-reference prefix has 162 chunks per
  block).
- Attention width (`QWEN_IMAGE_ATTN_THREADS`). The NPU's jobs slow down
  next to more than 16 busy cores, and not from memory bandwidth (the DiT
  uses ~15-20 of 57 GB/s). `dit_mid` alone takes 25.3 ms, next to 16 / 24
  threads of cache-resident work 25.1 / 32.7 ms, next to DRAM streaming
  on 4 / 24 threads 27.9 / 33.5. With the folds text to image is
  NPU-bound with the int8 attention (8 / 12 / 16 threads 93.8 / 94.0 /
  93.9 s), and so are edits at 16 threads, with one reference at 16 / 12 / 24 threads
  110.7 / 110.8 / 126.0 s, three (a 12.3k-row prefix) 16 / 24 147.9 /
  170.0 s. Default 16, every core for a layer's first chunk (the NPU
  waits for it), and with the f32 attention (`QWEN_IMAGE_ATTN_I8=0`) every
  core when the prefix is over a third of the rows (24 threads 158.7 s,
  16 183.6).

**iGPU share of the blocks** (`offload::DitBlk`, `QWEN_IMAGE_GPU_BLK=F`,
default 0.3125, `0` for the NPU alone). The iGPU runs the NPU's `dit_mid` job
(block l - 1 after its attention, block l up to its attention) for `F`
of each block's chunks of the target, each submitted once its attention
is done, while the NPU takes the rest. It uses the same int8
weights (the NPU's host buffers, imported zero-copy) and the folded
graph's math, `dit_gemm_npu.glsl` (int8 x int8 `dp4a` on the `[n][k]`
codes, exp_gemm's scheme, exact sums), `dit_row_npu.glsl` (residual +
gate, LayerNorm x modulation, pre-scale, Hadamard, int8),
`dit_act_npu.glsl` (SwiGLU, per-row int8), `dit_qkv_npu.glsl`, with f32
between the matmuls. One job against a host model of the math differs by 2e-4
(the f16 outputs). For 256 rows, the four matmuls take 45.5 ms (2.0-2.7 TOPS,
tiles of 64 tokens x 64 rows, 128 rows or 32 tokens slower) and the job
~50 ms, against the NPU's 21 ms. The GEMM is bound by its shared-memory
weight reads, not the dp4a rate (the same dispatch on registers only takes
~9 of its ~41 ms of a 300-row job). 8 / 16 / 32 tokens per lane, 32 /
64 / 128 rows per lane group, weights straight from global memory,
token tiles adjacent in the grid and non-saturating dot products were
all no faster (more registers spill).
- Which chunks (`qimage::npudit::gpu_chunk`). The NPU takes chunk 0, and the iGPU's
  are spread from chunk 1 on (5 of 16, namely 1, 4, 7, 10, 13), so both start
  after the first chunks' attention. Its first `F` chunks as one
  submission after their attention left the NPU and the iGPU idle for
  ~45 ms of each ~310 ms block (1 MP text to image, 9 steps, 89.3-91.7 s
  against 78.7-83.5 s spread, NPU waits 13.6-13.8 -> 8.7-9.1 s, iGPU
  waits 36.5-41.8 -> 33.1-37.9 s).
- 1 MP text to image (16 chunks, 9 steps) with a share of 3 / 4 / 5 / 6 of 16 takes
  108.4 / 84.9-104.0 / 78.7-83.5 / 84.1 s. One-reference edit share 0 /
  2 / 3 / 5 / 6 / 7 of 16 takes 125.3-126.5 / 130.6 / 129.7 / 100.3-104.9 /
  108.3 / 124.1 s. Fewer chunks than 5 are slower than their share
  predicts. Each iGPU chunk takes ~65-140 ms instead of ~45, with idle
  gaps between them (likely the iGPU's clock). Image vs the NPU alone
  38.8 dB (int8 vs f32 attention 35.9), text to image spread vs the
  first chunks 30.8 dB (fine texture), edit with vs without the share
  42.3 dB, s512 gate velocity 1.85-3.23e-2, image 30.0 dB, 2.54 s per
  step.
- Only with a prefix of at most twice the target rows (text to image,
  one reference). Edits with more references are attention-bound and
  keep the NPU alone.
- The prefix pass and `dit_pre` / `dit_post` stay on the NPU.
- The cores would be faster still (int8 VNNI GEMM on the same row-major
  codes, 4 x 3 tiles, with the four matmuls at 15.9 ms on 16 threads, 7.4 TOPS),
  but the iGPU takes the share so that the cores stay free.

**Int8 attention** (`quant::K8`, `quant::attn_block_rows_k8`,
`QWEN_IMAGE_ATTN_I8` with `kv` default, `k` Q.K only, `0` f32 on the f16
K / V) is SageAttention's recipe on AVX-VNNI. Per block, once its K is
final, the keys (prefix + target) are packed, with K less its per-(head,
channel) mean over the keys (softmax ignores the shift), rotated per head
by the normalized Hadamard matrix (as are the queries, so the dots are
unchanged), int8 per (key, head), and V int8 per (key, head) with the scale
folded into the probabilities (`P'_j = P_j s_j`), which are u8 per
(query row, 64-key block) at the row's block maximum. Q.K and P V are
`vpdpbusd` (queries offset to u8, the offset's term removed per key).
Packing takes ~5-9 ms per block at 1 MP, which the NPU and the iGPU would
wait for at every block boundary. From the second step on, a block's keys
are therefore packed chunk by chunk as its jobs finish
(`QWEN_IMAGE_KPACK_EARLY`, default on, `0` for all at the block boundary), centered at the previous step's channel means
(any per-channel offset gives the same softmax, and the mean only centers
the int8 range), the prefix's rows first, during the wait for the block's
last jobs. Only the last chunk's packing and the first chunk's attention
stay between two blocks. For 1 MP text to image, 9 steps, the boundary time goes 4.9
-> 2.2 s and the transformer 77.7-78.0 -> 75.1 s. For a one-reference edit, 11.6 / 9.4
-> 4.5-4.7 s, 94.8-98.7 -> 90.4-91.2 s, and s512 velocity is 2.0-2.9e-2
(1.85-3.0e-2), image 31.2 dB (30.7).
- Kernel at 4096 queries x 4132 keys x 32 heads (`quant` `bench_attn_k8`) takes
  f32 on f16 K / V 173-181 ms, int8 Q.K 108-117, int8 Q.K + P V 55-57.
- With int8 V the softmax and the u8 probabilities are one pass
  (`softmax_p8_row`, with block max in AVX2, probabilities kept in registers,
  `exp(s - m)` as `2^(s log2 e - m log2 e)`, one FMA and a cubic, 7.5e-5
  relative error, since the probabilities become u8, with the kernel at 55-58 -> 53-55
  ms, and codes rounded to nearest even, which is one instruction, where
  `f32::round` is a `roundf` library call per element). A one-reference 1 MP
  edit is then balanced between the cores' attention (60-69 s) and the NPU (54-60 s).
- Attention output error vs f32 on the same K / V (`QWEN_IMAGE_ATTN_CHECK=1`
  prints it per block, s512) is `kv` 0.6-1.2e-2, `k` 0.4-1.0e-2. The first
  blocks put ~95 % of the weight on a few text keys (attention sinks)
  whose V is ~14x smaller than the rest. A V scale per channel (over all
  keys or per 64-key block) gave 5-13e-2 there, one per key 1e-2, and without
  the rotation Q.K alone is 6-11e-3.
- The gate (`qimage_stageref check s512 --free`) gives velocity steps 0-7 1.97-2.74e-2
  (f32 1.86-2.89e-2), image 32.6 dB (33.3), and `k` 1.93-3.23e-2, 31.0 dB.

**VAE on the NPU** (`npubuild::qvae` for graphs, `imagegen::qimage::npuvae` for the
engine). Every 3x3 stride-1 convolution of the decoder and the encoder runs as
f16 graphs over 32x32 tiles with a one-pixel halo (16 kinds by channel
counts). RMS norm, SiLU, up- / downsampling, `conv_in` and the
attention stay on the cores, which build the tiles (norm + SiLU applied
while copying) and unpack the results. In the 1 MP decoder, the NPU waited 3.8 s and the
cores 1.1 s. The 512x512 decoder vs the f32 one on the same latents reaches 65.9 dB, and
NPU encoder latents vs the cores' f32 differ by rel. 4.05e-4.
- The RMS norm inside a conv graph made a 144-channel job 6-8 ms instead
  of 2.2, so the cores do it.
- CPU forms are slower, with rotated Q6_K im2col GEMMs (8.0 s at 512x512, 48 dB, since
  the quant GEMM is weak at K 1.5-2.8k), f32 im2col (14.3 s).

**Vision tower on the NPU** (`vision = sets/vis` in `image.npu`, built by
the `vis` step) uses the 27B's tower graphs (`npubuild::vis`) with the three
deepstack mergers as extra outputs `deep{j}` and the erf GELU, compiled by
the driver, with the attention on the cores (`quant::attn_block`, heads padded
72 -> 80, because its SIMD path needs a head width that is a multiple of 16 and the
scalar one is ~5x slower, with F16C conversions on the pool). Embeddings vs
the cores' tower rel. 3.2e-2, deepstack 0.6 / 1.3 / 2.2e-2. The edit
images with either tower reach PSNR 40.2 dB (512x512), far inside the
transformer's own trajectory noise. Loaded on the first edit and kept.

**Many reference images** (up to `imagegen::qimage::MAX_IMAGES` = 10 in every
interface, and each keeps its `--resolution` area, never shrunk). The prefix
grows by ~4.1k rows per 1 MP reference, its K / V by ~2.2 GB in f16
(`32 x 2 x rows x 4096 x 2 B`). In `qimage::pipeline::prefix_plan`, the K / V plus
the prefix pass' buffers may take `QWEN_IMAGE_PREFIX_MB` (default 6144)
next to the text encoder. Beyond that the prefix K / V are **int8 with a
scale per row and head, rotated per head** (`H (s x)`, `H` the normalized
Hadamard matrix of 128, undone when written back as f16 in front of the
target's K / V each block), and when even those do not fit, the **text
encoder is dropped** for the denoising and loaded again by the next job.
So 1-2 references stay f16 (bit-identical), 3-4 are int8, 5-10 int8
without the text encoder, and with it go the NPU vision tower and the VAE's
NPU weights (reloaded in under a second) and the allocator's cached
blocks (`quant::alloc::trim`).
- The int8 reconstruction error (512x512 edit, per block) is K 0.63e-2 rotated
  (0.83-1.3e-2 unrotated, since K has a few large channels), V 0.65e-2.
- Velocity with int8 vs f16 prefix K / V (`QWEN_IMAGE_KV8_CHECK=1`, where each
  step runs both) differs by 1.4-3.8e-2. A 1-ulp change of every other f16
  K / V value gives the same 1.4-3.3e-2 (`QWEN_IMAGE_KV8_ONLY=ulp`), because any
  perturbation moves the W8A8 graphs' activation codes by that much, so
  int8 is inside the backend's own noise.
- 10 references (1 MP each, 1024x1024 x 2, 1097x617 x 2, 1920x1080 x
  6, target 1376x768, 10395 text-encoder tokens, a 41293-row prefix) take
  vision tower 44.7-46.5 s, text encoder 83-87 s, VAE encoder 21.5 s,
  transformer 532.8 s (the prefix pass ~224 s, then ~44 s per step,
  attention on the cores 495.5 s, NPU waits 13.8 s, f32 attention
  1087-1211 s), VAE 7.0 s. Peak
  working set **25.4-25.8 GB**, with 10.3 GB before the prefix pass (NPU weights and
  the transformer's small ones), + 2.7 GB of prefix-pass buffers, + 11.1
  GB of int8 K / V. This exceeds the 24 GB budget, and int4 K / V and a
  disk-backed prefix are not used instead.

## Rejected

**Int4 transformer weights** (round to nearest per row, `QWEN_IMAGE_W4=1`
at build time, in its own output directory) take 25.0 ms per `dit_mid` job, no
faster (compute-bound), with velocity error 7-9e-2 (3-4x int8's), free-running
image PSNR 19.9 dB. GPTQ could only buy back the accuracy, for 3.5 GB of
RAM.

**Block cache across steps** (DBCache, `QWEN_IMAGE_CACHE=F,B,sSTEPS` or
`F,B,tX[,R]`, off by default, `QWEN_IMAGE_CACHE_LOG=1`,
`QWEN_IMAGE_CACHE_EXT=k`). On a cached step only blocks `0 .. F` and the
last `B` run, the middle ones replaced by their residual change at the
last full step. The middle blocks' change moves 17-47 % (rel. L2)
between Turbo's steps. A cached step's velocity error is 5.6-6.4e-2 for
F + B from 1 + 1 to 8 + 4 (the backend's own 1.9-3.1e-2), and extrapolating
from the last two full steps predicts worse (step 4 has 0.207 reused, 0.231
/ 0.304 at k 0.5 / 1). 1 MP, 1 + 1 blocks, steps 4 / 3, 5 / 2, 4, 6
cached take 83.5 / 72.8 / 62.1 s (of 94.0), with images vs uncached 39.5 / 37.0
/ 32.2 dB (a text prompt gives 34.5 / 27.6 dB, visibly changed at two
steps).

**iGPU share of the attention** (`offload::DitAttn`,
`dit_attn_npu.glsl`, `QWEN_IMAGE_ATTN_GPU=F`, default 0). The iGPU takes
the last `F` of each block's query chunks while the cores run the rest.
The kernel (G = 8 queries of one head per workgroup, keys through the
caches, the layout of `flash_attn_npu.glsl`) runs 0.47 TFLOP/s alone
(256 queries x 8192 keys take 72 ms, while staging 32 x 64 tiles in 43.5 KB of
shared memory instead reaches 0.13, one workgroup per Xe core). Next to the cores'
attention it is several times slower and slows the cores. A 1 MP edit's transformer takes 151.7 s
without, 158.8 s with 1/16 of the chunks on the iGPU, 196.2 s with 3/16
(the cores' attention 114.1 -> 120.3 / 133.8 s for less work).

**Block-sparse attention** (SpargeAttn's block-mean Q.K
predictor, top-CDF selection, self-similarity guard). Measured on the
10-reference edit's target attention (41.3k prefix + 4.1k target keys,
64-key blocks). The exact weights need, per query row, 28 % of the key
blocks for 99 % of the weight (heads 3-74 %), but a block of 16 / 64
query rows needs the union, 57 / 70 %, because the rows of a block attend
to different keys, and the target's own keys (9 % of them) take 62 % of the
weight. The predictor at 64 query rows has kept share 0.80 / 0.95 / 0.99 ->
density 0.44 / 0.49 / 0.58, output error 2.3e-1 / 1.4e-1 / 9.3e-2 (the
int8 attention's own 1e-2), the masked kernel 32-39 ms against ~45 ms
dense.

## Checks

| Part | Gate |
|---|---|
| CPU reference (text to image) | per-module tensors vs diffusers on the same inputs (rel. error < 1e-3 in f32), and a 512x512 image matching diffusers' for the same seed and noise (`tests/qimage_reference.rs`) |
| Edits | the same for preprocessing (Pillow-exact), vision tower with deepstack, text encoder with images, VAE encoder, slots, reference RoPE, with one and several references (`tests/qimage_edit.rs`) |
| Server | MCP (tools, progress, cancel, roots), `/v1/images/*`, `--mcp-stdio` end to end with a client (Zed) |
| CPU and NPU backends | `qimage_stageref check REF MODEL --free` compares every stage vs the f32 reference ("NPU backend") |

Reference tensors for the gates come from diffusers in ComfyUI's embedded
Python (`python_embeded`, CPU, f32) through
`crates/imagegen/scripts/qimage_refdump.py`, dumped to a scratch directory.

The server is checked by `cargo test -p server` (`images::tests`,
`mcp::tests`, covering sizes, write paths, multipart, base64, the JSON-RPC routes
and argument errors) and end to end against a Flash-Next chat server with
`--image-model`:

- **Zed 1.27** (its agent on the same server's Flash-Next) ran `generate_image`
  (16:9 -> 1376x768, `output_path` `art/cat.png`, a new subdirectory of the
  `--image-out-dir`) and `edit_image` (an absolute path in, 1024x1024 out,
  the inline preview shown in the thread and described correctly by the
  model).
- Zed declares **no roots**, so writes need `--image-out-dir`. The server
  warns at startup without one, and a refused path says so. Zed sends no
  `progressToken` and **cancels a tool call after 60 s** by default (it
  drops the request without `notifications/cancelled`, and the job then
  finishes and writes its file anyway). Set `"context_server_timeout": 600`
  (Zed's maximum) in its settings. It takes effect when the server
  reconnects. A tool call needs the user's approval unless Zed's agent
  profile allows it.
- Zed's transport sends one message at a time and waits for each
  response's head, so a `tools/call` stream opens at once, not at the first
  progress message.
- Every response, streams included, says `Connection: close`, because Zed pools
  connections, and a pooled chat stream whose handler is still storing its
  prefix-cache state would take the next tool call and reset it, which
  ends Zed's MCP session for good ("server shut down" on every later
  call).
- A scripted client over HTTP had `roots/list` asked inside the call's stream
  and answered by a separate POST, the file written inside that root,
  progress per step, `notifications/cancelled` after step 3 (no response,
  job stopped at the next step), a path outside the roots refused.
- `cli --mcp-stdio` ran initialize, `tools/list` and a refused path.
- OpenAI ran `/v1/images/generations` (JSON) and `/v1/images/edits`
  (multipart `image[]`).
- The idle unload after 300 s releases the model and 1.7 GB of the
  allocator's cached blocks, and the next call loads it again (17.6 s).
- The OpenCode configuration above is not checked end to end.

## Code layout

- `crates/imagegen` has configs (the checkpoint's JSONs), scheduler, RoPE
  ids, sequence building, the CPU reference forward of all three models,
  PNG output, the pipeline driving whichever backend.
- `crates/npubuild` (`qimage`) has `cli QWEN_IMAGE_DIR --npu-build OUT`
  (calibration ranges, int8 weights, graphs, compile, `image.npu`).
- `crates/imagegen` (`npudit`, `npucal`, `npuvae`) has the NPU transformer's
  and VAE's host side and the calibration (it needs the CPU backend, which
  `npubuild` cannot depend on).
- `crates/npubuild` has the VAE's conv graphs (`qvae`) and (`vis`, `sets`) the
  vision tower's set with deepstack outputs.
- `crates/offload` (`DitBlk`, `dit_gemm_npu.glsl`, `dit_row_npu.glsl`,
  `dit_act_npu.glsl`, `dit_qkv_npu.glsl`) has the iGPU's share of the
  blocks' chunks ("NPU performance"), and (`DitAttn`, `dit_attn_npu.glsl`) the
  iGPU attention share, off by default (`QWEN_IMAGE_ATTN_GPU`, see
  "Rejected").
- `crates/server` has `mcp.rs` (JSON-RPC, SSE, roots), `images.rs` (OpenAI
  routes), the device queue and the idle unload.
- `crates/cli` has `--image-model`, `--image-out-dir`, `--mcp-stdio`,
  `--txt2img PROMPT -o FILE` for one image without a server.

### Shared components

The image path is built on components the chat engines use too:

| Need | Code |
|---|---|
| safetensors / JSON configs | `npubuild::st` (`Json`, `parse_json`, `Checkpoint`) |
| Tokenizer | `tokenizer::hfjson` (`tokenizer.json` BPE) |
| Image decode, Pillow-exact resize | `imgio` |
| Vision tower (edits) | `vision::tower` (CPU, with deepstack), `npu::vis` (NPU, deepstack outputs `deep{j}`) |
| f32 GEMM for the reference | `npubuild::la::linear` (800+ GFLOP/s) |
| Quantized CPU GEMMs (fallback backend) | `quant::gemm` (AVX2 / VNNI), `quant::parallel` pool, `quant::alloc` |
| Attention (CPU) | `quant::attn` |
| f16 conversion, SiLU / activations | `quant::fp16`, `quant::act` |
| int8 per-row weights, GPTQ, Hadamard rotation | `npubuild::gptq`, `npubuild::la` (`fwht_*`), Flash-Next's dense int8 (`npubuild::flash`) |
| NPU graph building, compile, sets | `npubuild::{g, sets}`, the compiler flags (`npubuild::sets`) |
| NPU execution, host buffers, NPU lock | `npu::{ze, engine}`, `ov::lock` |
| iGPU attention / imports | `offload::VisAttn`, `offload::DitAttn`, `gpu/hostimp.rs` |
| VAE 3x3 convs on the NPU (graphs / engine) | `npubuild::qvae`, `imagegen::qimage::npuvae` (the only convolution graphs, since the 27B's tower graphs are matmuls only) |
| HTTP, hand-written JSON, SSE | `crates/server` (`lib.rs`, `openai.rs`) |
| Progress output, load timing | `quant::progress`, `quant::ltime` |
| Block-causal attention (DiT image blocks, VAE mid attention) | `quant::attn_block` (the tiled `attn_causal` kernel with every key of the query's block visible) |
| diffusers component dirs (`transformer/`, `vae/`, `text_encoder/`) | `npubuild::st::Checkpoint::open_dir` (any `*.safetensors.index.json`, else one `*.safetensors`), and large BF16 tensors widen on the pool |
| PNG output | `imgio::save_png_rgba` (`image`'s PNG encoder, already in the `png` feature) |
