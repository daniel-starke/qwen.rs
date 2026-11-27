# Image generation (MiniMax-H3-Image)

Text to image and edits with up to 16 reference images, on the interfaces of
Qwen-Image-2.1 (`docs/image.md`): `--txt2img`, MCP and the OpenAI Images
API. The server picks the backend from the checkpoint the `--image-model`
path names (`model_index.json` class); one image model per server.

Status: M1 (f32 reference, per-layer gates), M2 (Q6_K CPU backend,
`--txt2img`, server / MCP) and M3 (the transformer and the VAE decoder on
the NPU) done.

## Model and files

- **Checkpoint.** `MiniMaxAI/MiniMax-H3-Image`, commit `8ced60068623`, the
  diffusers layout: `transformer_diffusers/` (12 shards, BF16 with the
  patch projection, timestep MLP and output heads in F32),
  `vae_diffusers/` (F32), `text_encoder/` (14 shards, BF16; the repository
  keeps its own copy of MiniMax-H3's), `tokenizer/`, `processor/`,
  `modular_model_index.json` (the sampling grid). The native layout
  (`transformer/`, `vae/`, `pdd_fused_heads.safetensors`) is the same
  weights for other frameworks and is not read.
- **Reference code.** diffusers main (`MiniMaxH3ImageTransformer2DModel`,
  `AutoencoderKLMiniMaxH3Image`, `MiniMaxH3Scheduler`,
  `modular_pipelines/minimax_h3_image`), commit `750b3be2c019`; diffusers
  0.42.1 has only the H3 video model. transformers 5 for the Qwen3-VL
  text encoder.
- **Not used.** MiniMax-H3-Image-PE (prompt expander): the calling agent is
  an LLM and writes the detailed English description the model expects.

## The model

**Text encoder.** Qwen3-VL with the 32B shape (64 layers, 5120 wide, GQA
64/8, head 128, FFN 25600, MRoPE 24/20/20 interleaved, theta 5e6; vision
tower 27 blocks, 1152 wide, patch 16, merge 2, deepstack after blocks
8/16/24, out 5120). The conditioning is `hidden_states[50]`: decoder layer
49's output, **not normalized**; layers 50-63, the final norm and the
lm_head never run. The prompt goes in verbatim: no chat template, no
special tokens, at most 4096 tokens. An edit prepends `Picture i: ` and a
vision block (`<|vision_start|>`, one `<|image_pad|>` per merged token,
`<|vision_end|>`) per reference. Every row is kept, the vision blocks too
(tagged as image rows).

**Transformer** (33.1B; 19.3B of it in the block matmuls, 13B in the
per-block AdaLN projections): one packed sequence `[text | references |
target]` with **full bidirectional attention**, no cross-attention.
- Text rows: `context_embedder` (5120 -> 5376, bias), a 2-block token
  refiner (pre-norm RMSNorm, full attention over the text, no RoPE, no
  modulation), RMSNorm. Independent of the step.
- Image rows: latents patchified 2x2 (24 x 4 = 96 values per row, a row
  covers 32x32 pixels), `proj_in` (96 -> 5376, bias).
- 50 blocks, 5376 wide, 56 heads x 128 (attention width 7168), SwiGLU
  14336 (`net.0.proj`: value half, then gate half), no biases. `x +=
  gate * f(RMSNorm(x) * (1 + scale) + shift)` for attention and MLP; the
  gates are not squashed. The six modulation rows come per row from a
  `(timestep, modality)` table, each block's own `adaln_proj` (Linear 2688
  -> 3 modalities x 6 x 5376, bias) of `SiLU(temb)`; modality 0 image, 1
  text, 2 audio (unused).
- q / k: per-head RMSNorm, then rotate-half RoPE over the first 96 of 128
  channels (axes t, h, w with one shared list of 16 frequencies, theta
  1e4), the other 32 pass through. Rotary positions: text `(p, 0, 0)`;
  each image is one frame `T + j` (`T` = text rows), its `(h, w)` an
  aspect-normalized grid centered on the unit square scaled by 32 (a square
  spans `[0, 32)`), built in f64 as numpy's `linspace(endpoint=False)`.
- Timestep: `t = 1 - sigma` in [0, 1] (1 = clean), **unscaled**,
  `[cos, sin]` sinusoid (256), `linear_1 -> SiLU -> linear_2` (biases).
- Output: `norm_out` (RMSNorm * (1 + scale) + shift per timestep), then
  the step's own head `pdd_proj_out[i]` (5376 -> 96, bias): one output head
  per sampling step (Parallel Decoding Distillation).

**Sampling.** 6 fixed steps over `sample_sigmas` `1.0 0.9675 0.8875 0.75
0.5328 0.2320 0`, CFG baked in. The velocity is data-ward: `x += (sigma_i -
sigma_{i+1}) v` in f32. Each sigma is rounded through its f32 timestep and
back. Text rows take the step's `t`; references are noised once to `t =
0.999` (`0.999 x + 0.001 noise`) and held there. A step count other than 6
does not exist (one head per step).

**Canvas.** A request below `min_generation_area` (4 MP) is generated on a
canvas of the same aspect at 4 MP and resized back with Lanczos; sides are
multiples of 32. The model is trained for 4-20 MP and looks best at 16 MP.

**VAE.** Encoder: MiniMax-H3's causal video CNN over one frame (only the
last temporal slice of each 3x3x3 kernel acts), reflect padding,
GroupNorm(32), levels 128/256/256/512/512/1024, 16x down, 48 moments; the
latent is a posterior **sample** off the request's generator, rounded to
f16, normalized per channel. Decoder: a 24-layer 2-D ViT (1024 wide, 16
heads x 64, 4 registers + one zero token, 2-axis RoPE over half of each
head, residuals scaled per channel), one token per latent pixel, 16x16x3
pixels out per token, ImageNet-normalized RGB. Both run in **256-pixel
tiles with at least 64 pixels of overlap, blended linearly**; tiling is
part of the released model (it changes the output).

**References** keep their own size: resized only when the short edge is
outside 1024-2048 for the VAE, to exactly 1024 for the vision tower. The
generator draws in the pipeline's order: VAE posterior samples, target
noise, reference noise.

## Cost and placement

At the 4 MP default (2048x2048 = 64x64 = 4096 target rows, ~100 text
rows): block matmuls 2 x 19.3e9 x 4200 = 160 TFLOP per step, attention
~25 TFLOP. 16 MP (16384 rows) is ~1.1 PFLOP per step, attention a third
of it.

- **No prefix caching.** Unlike Qwen-Image the attention is bidirectional
  and the text rows take the step's timestep, so every step runs every
  row: each reference costs its 1024-4096 rows in all 6 steps.
- **AdaLN tables instead of 13B parameters.** The modulation depends only
  on the timestep, and the timesteps are fixed (6 + the reference level):
  a table of 7 x 50 x 3 x 6 x 5376 f32 (~135 MB) replaces the projections
  at run time.

| Part | CPU backend | NPU backend (`image.npu`) |
|---|---|---|
| text encoder (50 layers) | cores, Q6_K | **NPU**, W8A8; a quarter of the 256-row chunks on the **iGPU**; attention on the cores |
| vision tower, VAE encoder (edits) | cores | cores |
| transformer blocks | cores, Q6_K | **NPU**, W8A8; 4 of 16 chunks on the **iGPU** |
| attention (all rows x all rows) | cores, f32 | cores, f32, overlapped with the NPU; about 60 % of the query rows on the **GTX 1060** (f32) |
| VAE decoder (24-layer ViT, 256 px tiles) | cores, f32 | **NPU**, f16; 30 % of the tiles on the **iGPU** (f32) |

## NPU backend

`cli H3_DIR --npu-build OUT [--npu-compiler DIR]` (`npubuild::h3img`),
each step skipped when its output exists:

- `cal`: the Q6_K CPU backend (from the weight cache) generates five
  frozen jobs (two 512 px images, a 768x1024 portrait, a 2048x2048 image,
  a one-reference edit of the first image) while every block records the
  largest magnitude of its block input, its rotated QKV, out and gate / up
  inputs (x1.25 headroom), its output and its MLP's gate and hidden state
  (`cal/h3dit.cal`), and the Hessians `X^T X` of its QKV, out, gate / up
  and down inputs (`cal/h3hess`, the lower triangles in f32, 32 GB; ~47k
  rows), and writes the AdaLN tables (`cal/h3adaln.bin`), ~35 min.
- `weights`: per block the int8 matrices (GPTQ on the rotated rows
  against the rotated Hessian, the rows' scale `max|w| / 127` as round to
  nearest's; `QWEN_H3_GPTQ=0` rounds to nearest and needs no Hessians),
  the q / k row scales and head norms for the cores, and the modulation
  entries (`w/h3dit{L}.*.in`, 19.3 GB), ~10 min (~1 min to nearest).
- `graphs`, `compile`: the transformer's `h3/pre`, `h3/mid`, `h3/post`
  (256 rows a job, the `rowq` SHAVE kernel), the same over 64 rows
  (`h3/pre64`, `h3/mid64`, `h3/post64`; a set without them runs 256-row
  jobs only; a later `--npu-build` compiles just the missing blobs) and
  the VAE decoder's
  `h3vae/dec` (one ViT layer over 4 tiles) and `h3vae/out` (LayerNorm and
  `proj_out`), ~2 min.
- `te`: the text encoder's layers (`npubuild::h3te`,
  `imagegen::h3::tecal`): the float layers run one at a time over 12
  frozen prompts (lettering and Chinese among them), two edits (one and two
  references, made by the Q6_K backend) and two calibration texts, ~6700
  rows; each layer records its ranges and the Hessians of its four
  inputs, then is rounded by int8 GPTQ (`w/te`, 24.4 GB), ~65 min,
  resumed per layer.
- `sets`: `OUT/image.npu` (`arch = h3-image`, `checkpoint`, `graphs`,
  `weights`, `vae`, and `te`, `te_graphs` when the text encoder is
  built), which every command takes in place of the checkpoint.

Transformer, as Qwen-Image's NPU backend (`docs/image.md` "NPU backend"):
the blocks are cut at the attention, W8A8 with the 256-wide Hadamard
rotation, static calibrated ranges for QKV, out and gate / up and a
per-row scale for down, q / k / v raw out of the graph, the attention of
chunk c on the cores while the NPU runs the next graph's chunk c - 1.
What H3 needs beyond it:

- **The residual is stored scaled.** H3's residual stream reaches 3e7
  (block 39 on: a few channels), past f16. Each block's output is kept as
  `x_l / c_l`, `c_l` = the block's calibrated maximum / 8192; the RMSNorms
  ignore the scale, `gate / c_l` is folded into the out and down
  projections' row scales, and the step from `c_{l-1}` to `c_l` is one
  multiply. `norm_out` takes the last block's scaled rows as they are.
- **The MLP's up half is scaled by 1/256** before the SwiGLU product (the
  hidden state reaches 3e5), the down projection's per-row scale undoes it.
- **Rows are ordered by modulation entry.** Text rows, image rows at the
  step's timestep and references at `REF_T` have different modulation, and
  it differs per block (folded with the RMSNorm weights into one f16 row
  per part, entry and block). The engine groups the rows by entry, pads
  each group to whole 256-row chunks and binds a chunk's entry; full
  attention does not depend on the keys' order, and the cores take only
  the valid rows as keys.
- The text rows' refiner, `proj_in`, `norm_out` and the output heads run on
  the cores (`Dit::load_small`).
- **Small jobs at a block's edges.** Every key of block l must exist
  before any of its attention runs, so the NPU idles during a block's
  first chunk of attention and the cores during its last job. Each group
  of rows is padded to 64 rows, the rest past whole 256-row chunks goes
  into 64-row jobs, and a block starts with one and ends with two of them
  (`QWEN_H3_SMALL=0`: 256-row jobs only).
- **An iGPU share of the blocks** (`offload::H3Blk`, `QWEN_H3_GPU_BLK`,
  default 0.25 of the 256-row chunks, spread from the second on): the
  iGPU runs the `h3/mid` job of those chunks on the same int8 weights and
  host buffers (imported zero-copy, 19.3 GB within `QWEN_GPU_IMPORT_MB`),
  in f32 between the int8 matmuls; the step's modulation entries are
  copied in as f32. Built only for the default static activations.
- **A share of the attention on the discrete card** (`offload::H3Attn`,
  `h3_attn.glsl`, `QWEN_H3_ATTN_GPU_SHARE`): the GTX 1060 takes 256-row
  chunks from the back of each block (the cores keep the first and the
  small last ones), in batches of two chunks, up to three in flight. Unlike
  the iGPU it cannot read host memory per key block (PCIe), so a block's K
  and V go to VRAM once, staged row by row as the cores finish them, with
  the block's first batch; queries go up finished (row scales, head norm,
  RoPE) as f32, outputs come back as f16. The share is the one that ends
  the card and the cores together, from their seconds per unit of
  attention (`quant::calib` fit `h3-attn`, the card timed by GPU
  timestamps; frozen, so the split and the output are reproducible).
  VRAM is held only during a job: K / V plus three slots, ~190 MB at 4 MP,
  refused past `QWEN_H3_ATTN_GPU_MB` (default 1024) or past what
  `VK_EXT_memory_budget` leaves less 512 MB for other applications; the
  attention then stays on the cores. The kernel (one head x 48 query rows
  a workgroup, keys in blocks of 64, fp32, 3 x 4 / 3 x 2 register tiles,
  software-pipelined staging) runs 1.45 TFLOP/s; the cores' attention
  about 1.

Text encoder (`h3/npute.rs`, `QWEN_H3_TE_NPU=0` keeps it on the cores):
a layer is a transformer block's graphs with one modulation entry (the
RMSNorm weights) and grouped queries, on the same W8A8 scheme, scaled
residual and SwiGLU scaling. The attention is causal, so chunk c of layer
l needs only the keys of chunks 0..c: each chunk walks through the layers
on its own (a wavefront), its next job sent as soon as its attention is
done, and no device waits for a whole layer. Rows go in 256-row chunks,
the rest in 64-row ones. The cores finish each chunk's keys (row scales,
head norm, MRoPE) into a compact per-layer copy and run the attention;
after layers 0-2 the image rows take their deepstack features between a
`post` and a `pre` job. The iGPU (`H3Blk` at the encoder's widths,
`QWEN_H3_TE_GPU`, default a quarter) runs the `mid` jobs of the last
256-row chunks, which never wait for keys; its f32 row work is also more
accurate than the NPU's f16 (an edit's step-0 velocity with its text 5.8e-2
against 8.1e-2 NPU-only). The early layers' folded down-projection scales
are ~60x the transformer's, so the encoder's graphs apply the row maximum
before the row scale (the other order overflows f16).

Not taken: int4 MLP weights in the text encoder (`QWEN_H3_TE_W4=mlp` at
build time, 13.6 instead of 24.4 GB). A short prompt's encode took 17 %
less (64-row jobs read the weights), a one-reference edit's 20 % more
(the iGPU kernels are int8 only), and step 0's velocity error doubled
(4.8e-2 -> 9.4e-2, the Q6_K encoder's 5.9e-2) with the free-running image
5 dB further off.

VAE decoder: the cores make each tile's 261 tokens (`post_quant_conv`,
`proj_in`, registers, zero token), the NPU runs the 24 layers with the
attention inside the graph (f16 weights from the checkpoint at load, the
norms and per-channel residual scales folded in, the residual x16), and
the head; at most 64 executions are recorded at once (the driver's
descriptor memory). A canvas side under 256 px decodes on the cores.
Tiles are independent through the decoder, so the iGPU
(`offload::H3Vae`, `QWEN_H3_VAE_GPU`, default 0.3 of the tiles in whole
NPU jobs) takes whole tiles through every layer and the head at the same
time, in f32 on the NPU's f16 weights, reading and writing the NPU's
buffers.

GPTQ against round to nearest on s512: block errors 3-7 % lower, the
mean velocity error per step the same within noise (+3 %), the
free-running image 19.3 -> 21.9 dB, and a 4 MP image 24.8 -> 26.1 dB
against the Q6_K backend's; the transformer time is unchanged.

Not taken: per-row activations for QKV, out and gate / up (the down
projection's form, `QWEN_H3_ACT` at build time). Per-row out and gate / up
raised the velocity error to 7.4-10.6e-2 per step; per-row QKV left its
mean unchanged (4.9e-2).

Not taken: int8 attention on the cores (`quant::K8`, `QWEN_H3_ATTN_I8=1`).
On s512 it doubled step 2's velocity error on the NPU backend (6.6e-2 ->
1.1e-1) and moved the CPU backend's free-running image from 23.4 to 18.7
dB, for 5 % of a 4 MP job on the NPU backend (the attention is hidden
behind the NPU).

## Interfaces

- `cli H3_DIR --txt2img PROMPT -o FILE.png [--image REF]... [--size WxH]
  [--seed N] [--image-cache DIR]`; `cli MODEL --serve PORT --image-model
  H3_DIR` (or `cli H3_DIR --serve PORT`, images only): MCP and the OpenAI
  Images API. The path picks the model (`imagegen::Kind::of`); one image
  model per server.
- Default canvas 2048x2048, or the last reference's aspect at 4 MP for an
  edit; explicit sides 256-8192 (multiples of 32), at most 16 MP. No
  `steps` (fixed 6), no alpha. Up to 16 references, `<Picture N>` in the
  prompt. The MCP tools state the cost of a reference.

## Memory: weight cache and the text encoder

- **Weight cache** (`--image-cache DIR`, `QWEN_IMAGE_CACHE_DIR`;
  `imagegen::wcache`): the Q6_K text encoder (18.6 GB) and transformer with
  its AdaLN tables (15.5 GB) are written once; a load then reads them (13 s
  for both) instead of converting ~92 GB of BF16. A file carries a key
  (shard names, sizes, mtimes, format, rotation, timesteps) and is made
  again when it differs. Without a cache directory nothing is written.
- **Text encoder resident while memory allows.** After encoding, a job
  estimates the denoising's working memory and drops the text encoder when
  the available physical memory would fall below `QWEN_IMAGE_TE_RESERVE_MB`
  (default 8192); the next job loads it again (~7 s from the cache).
  Resident: ~35 GB in all; beside Flash-Next (~75 GB) the encoder is
  dropped per job.

## Measured

4 MP, the s512 reference's prompt, seed 42 (`QWEN_IMAGE_TIMING=1`):

| | CPU backend (Q6_K) | NPU backend |
|---|---|---|
| per step | 89 s | 24 s |
| transformer (6 steps) | 535 s | 150-154 s (NPU wait 59-69 s, iGPU wait 1-2 s, card wait 4-9 s, attention and K / V on the cores 71-80 s) |
| VAE decoder | 47.5 s | 20.3 s |

The NPU backend's transformer is bound by the NPU: the iGPU's share and
the small edge jobs took 197 -> 180 s (interleaved A/B, the NPU wait
46 -> 14 s), the attention then binding on the cores (slowed 150 ->
157-163 s next to the iGPU, shared memory bandwidth); the card's share
of the attention (10 of the 16 big chunks a block) took 175.3 -> 144.5-146
s. iGPU block shares of 0.1875 / 0.4375 took 184.5 / 207 s without the
card; with it 0.375 / 0.4375 take 165.6 / 185.6 s (the iGPU then finishes
last). An iGPU share of 0.3125 took 161-164 s against 150-154 s at the
default 0.25 (interleaved): the iGPU finished each block last (iGPU wait
21-24 s), and the NPU sat idle 36 s of its 183 s span, 26 s of it while
the iGPU still ran. The two do not slow each other much: an NPU job
overlapped by iGPU jobs takes 2 % longer than with the iGPU off (ETW
`DxgKrnl` packets), so they keep running side by side rather than taking
turns. The VAE decoder's iGPU share took 25.4 -> 20.3 s; shares
of 0.4 / 0.5 were slower (the iGPU then finishes last).

512x512 (344 rows): 6.5 s per step on the cores, 3.6 s on the NPU. The 4
MP images of both backends show the same scene, light and lettering;
details differ (the NPU backend's sign is wider).

Accuracy against the f32 reference (s512, every step fed the reference's
latents and text rows):

| Check | NPU backend | CPU backend (Q6_K) |
|---|---|---|
| target rows through blocks 0 / 24 / 49 of step 0 (chained) | 1.1e-2 / 1.8e-2 / 1.1e-1 | 8.3e-3 / 1.1e-2 / 7.9e-2 |
| velocity, steps 0-5 (mean) | 3.9e-2 to 6.6e-2 (5.0e-2) | 3.4e-2 to 5.3e-2 (4.1e-2) |
| VAE decoder on the reference's latents | 4.9e-3, PSNR 54.2 dB | |
| free-running image (the reference's noise) | PSNR 21.9 dB | 23.4 dB |

The free-running image drifts with any change of rounding (the CPU
backend with int8 attention: 18.7 dB; every target chunk on the iGPU's
f32 row work instead of the NPU's f16: 16.2 dB, per-step velocity mean
5.55e-2 against 5.16e-2, block errors equal through block 12 and then
better or worse by block; the card's attention, equal to the cores' to
one f16 ulp of its output, moves steps 1-2 from 6.9 / 6.7e-2 to 12.9 /
9.7e-2 with 256-row chunks, and a one-ulp nudge of the cores' own output
moves steps 1-3 to 8.3 / 7.9 / 8.2e-2), so the per-step numbers rank the
backends, not small changes of rounding. A
4 MP image with the default iGPU share shows the same scene, light and
lettering as without it; 22.8 dB against the Q6_K backend's image
(27.4 dB without the share). The text encoder (Q6_K) is 7.6e-3 off the f32 one, the text rows
2.3e-2.

Text encoder, NPU (int8) against the cores (Q6_K), `QWEN_CALIB=0`:

| | cores, Q6_K | NPU + iGPU |
|---|---|---|
| s512 prompt (88 rows) | 1.99 s | 1.81 s |
| e512 one-reference edit (1057 rows) | 23.7 s | 8.7 s (10.8 s NPU only) |
| two-reference edit, 1024x1024 (2094 rows): text-encoder stage with the vision tower | 62.9 / 80.9 s | 40.1 / 40.1 s |
| 19-row prompt | 0.6 s | 1.1 / 1.4 s |
| step 0 velocity with its text rows, s512 / e512 (reference's text 4.35 / 5.77e-2) | 5.85 / 6.48e-2 | 4.78 / 5.81e-2 |
| free-running image, s512 / e512 (reference's text 21.9 / 32.2 dB) | 19.5 / 33.2 dB | 20.2 / 32.1 dB |

A short prompt is slower on the NPU: a 64-row job per layer reads all
24.4 GB of weights, which the cores' Q6_K encoder outruns for a handful of
rows.

## Milestones

1. **M1, f32 reference and per-layer gates** (done). `crates/imagegen/src/h3`
   (config, layout, transformer, VAE; the text stack is `te.rs`'s
   `encode_ids` cut to 50 layers), `scripts/h3refdump.py`,
   `tests/h3ref.rs`. Every gate within 6e-6 rel. (text encoder 1.7e-7,
   vision tower 1.4e-5), the reference resize byte-identical.
2. **M2, Q6_K CPU backend** (done): the pipeline with edits, the weight
   cache, `--txt2img`, server / MCP.
3. **M3, NPU backend** (done): the transformer's blocks and the VAE
   decoder on the NPU, the build (`npubuild::h3img`), the gates.
4. **M4, iGPU shares and small edge jobs** (done): GPTQ rounding, the
   iGPU's share of the blocks and of the VAE decoder's tiles, 64-row jobs
   at a block's edges.
5. **M5, attention on the GTX 1060** (done): a share of each block's
   query chunks, sized by a frozen fit of both devices' rates.
6. **M6, text encoder on the NPU** (done): the `te` step, the causal
   wavefront over chunks, an iGPU share.
7. Next: int16 attention on the cores (AVX-VNNI), the vision tower and
   the VAE encoder of edits on the NPU.

## Checks

`scripts/h3refdump.py` runs the official modules on the CPU in f32, each
decoder layer / block materialized from the shards right before it runs
(RSS: one layer plus activations; the text encoder alone is ~98 GB in f32),
and dumps every layer's output: the text encoder's 50 layers, the refiner,
the timestep embeddings, the AdaLN tables of blocks 0 and 49, the rotary
tables, every block of the first and last step (blocks 0 / 24 / 49 of the
others), velocities and latents of every step, the VAE's stages on the
first tile, the image. `tests/h3ref.rs` feeds each layer the reference's
input and compares its output (gate 1e-3 rel.), so one layer can be checked
in seconds (`QWEN_TEST_H3_BLOCKS=0,24,49`).

```
python h3refdump.py --model H3_DIR --out REF/s512 --size 512 [--pylib DIFFUSERS_DIR]   # 256 target rows
python h3refdump.py --model H3_DIR --out REF/s1024 --size 1024          # 1024 rows
python h3refdump.py --model H3_DIR --out REF/e512 --size 512 --image A.png --prompt "Turn <Picture 1> into ..."
QWEN_TEST_H3_MODEL=H3_DIR QWEN_TEST_H3_REF=REF/s512 cargo test --release -p imagegen --test h3ref -- --test-threads 1 --nocapture
QWEN_TEST_H3_QUANT=1 ...                                                # + the Q6_K backend against the dump
QWEN_TEST_H3_CHAIN=1 ...                                                # the Q6_K backend through chained blocks, every step, free-running
QWEN_TEST_H3_NPU=OUT/image.npu ...                                      # the NPU backend: blocks, every step, free-running, VAE decoder
QWEN_TEST_H3_NPU_VAE=OUT/graphs/h3vae ...                               # the NPU VAE decoder alone, and its 4 MP time
```

`QWEN_IMAGE_NANCHECK=1` prints the NPU residual's non-finite values and
largest magnitude per block.

`--pylib` puts a diffusers checkout first on `sys.path` (an embedded
Python with a `._pth` file ignores `PYTHONPATH`). An edit dump adds the
references' gates (resize, vision tower, VAE encoder, posterior sample,
noise). `QWEN_IMAGE_MIN_AREA=1024` lets `h3gen` / `--txt2img` generate a
dump's small canvas as given, for an image comparison.

`--min-area` defaults to 1024 in the dump script, so small canvases are
generated as given (the pipeline's default is 4 MP).
