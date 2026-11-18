# `cli` command-line reference

The front-end binary is `target/release/cli.exe` (build with
`cargo build --release -p cli --features vulkan`).

The **first argument selects the backend by what it is**:

| Argument | Backend |
|---|---|
| `<model.gguf>` | the GGUF engine |
| `<openvino-export-dir>` (holds `openvino_language_model.xml`) | the OpenVINO engine (`crates/ov`) |
| `<model.npu>` (text file of `key = path` lines) | the NPU engine (`crates/npu`), and `arch = flash` in the file selects the Flash-Next engine (`crates/npu/src/flash`) |
| `<bf16-checkpoint-dir>` + `--npu-build OUT` | build the NPU engine's files (no inference) |
| `<qwen-image-dir>` (holds `model_index.json`) + `--txt2img` / `--serve` | image generation (`crates/imagegen`), either one image or an images-only server |
| `--mcp-stdio [URL]` (no model) | stdio bridge to a server's MCP endpoint |
| `--version` / `-V` (no model) | print `qwen.rs <version>` |

`--image` / image blocks on an NPU engine use the tower named by `vision = DIR`
in the `.npu` file (no `--mmproj`). On the OpenVINO backend images are not
supported at all. `--mtp` / `--dflash` are an error on an NPU engine run (the
engine drafts with its own MTP head).

---

## Modes (model argument `<model.gguf>`)

| Flag | Effect |
|---|---|
| *(none)* | Dump GGUF metadata, arch highlights and the tensor index. |
| `--dequant <tensor>` | Read one tensor, print its shape and dequantized min / max / mean. |
| `--forward <t0,t1,...>` | One forward over the given token ids. Prints logit sums, argmax and top-20 (the `DCR SUM` / `ARGMAX` check line). Long prompts use `@FILE`. |
| `--generate [<prompt>]` | One-shot generation. A bare `--generate` reads the prompt from stdin. `@FILE` reads the prompt from a file (long prompts overflow the Windows command line). Prints a `[perf]` line (prefill / decode tok/s) plus an `[eff]` line (CPU-seconds, joules and tok/J per phase, see `QWEN_ENERGY`). |
| `--interactive` | Multi-turn chat on stdin, where only each turn's new tokens are prefilled. `/image <file>` attaches an image to your next message, `/exit` quits. |
| `--image-probe` | Run only the vision tower (no text model). With the positional path the mmproj itself, `cli.exe <mmproj.gguf> --image-probe --image x.png` works. |
| `--tokmeta` | Print the tokenizer-related GGUF metadata (`tokenizer.*`, `general.*`). |
| `--embedtok <ids>` | Print embedding statistics (norm, min / max, samples) for the given token ids. |
| `--tokenize <text>` | Print the token ids of `text`, one per line. Literal `\n` is expanded, and `@FILE` reads the text from a file. |
| `--serve [PORT]` | Start the HTTP server (default port **8080**) with Anthropic `/v1/messages` (SSE), `count_tokens`, OpenAI `/v1/chat/completions`, `/v1/models`, `/props`, `/health`. The model is loaded once and shared. |
| `--kld-eval <text-file> [--kld-tokens N] [--sparse-sweep SPEC1;SPEC2...]` | Teacher-forced accuracy of a decode-path variant against the unmodified model, reporting mean / p99 / max KL, top-1 agreement and perplexity. `--sparse-sweep` takes semicolon-separated sparsity specs (`a=0.4,o=0.4,f=0.4,d=0.5`, also `all=…`). |
| `--tune-profile OUT` | Measure this machine's decode knobs for agentic coding (a Claude-Code-like conversation grown to each depth, in process) and write a profile. Load flags as for `--serve`, plus `--tune-depths D,D,..`, `--tune-modes think,nothink`, `--tune-tokens N`, `--tune-rounds N`, `--tune-corpus DIR`, `--tune-quick`. Any engine. |
| `--tune-split [SEQ,SEQ,...] [--tune-rounds N]` | Measure the prefill CPU/GPU row split on this host and recommend the `QWEN_PREFILL_GPU_FRAC_*` / `QWEN_PREFILL_DFRAC_*` fractions (default seqs `2048,160,100`, 3 rounds). Measures matmuls in isolation. |

## Generation options

Defaults shown. All apply to `--generate` / `--interactive` (and `--serve`,
where per-request JSON wins).

| Flag | Default | Meaning |
|---|---|---|
| `--chat` | off | Render the prompt through the model's chat template (on automatically for `--interactive`, and whenever `--system`, `--think` or an image is present). |
| `--system <text>` | none | System prompt, rendered as a system turn. |
| `--think off\|low\|medium\|xhigh` | `medium` | Thinking and its effort (the template's `enable_thinking` / `reasoning_effort`, and `medium` adds no instruction). With `--serve`, it sets the default of requests that do not say (else off on `/v1/messages`, on on `/v1/chat/completions`). |
| `--no-preserve-thinking` | none | Past turns carry no reasoning (`--interactive` rewinds to the previous reply's header each turn). With `--serve` it sets the default of requests. |
| `--profile FILE` | `QWEN_PROFILE`, else `<model>.profile` beside the model, else built in | Sampling defaults and decode knobs per thinking mode and context depth (`crates/server/src/profile.rs`). |
| `--no-keep-awake` | none | Let Windows put the display and the system to sleep while `cli` runs. By default the process holds both on until it exits (`SetThreadExecutionState`), because with the display in standby the iGPU stalls 0.4-0.5 s at random points of an NPU-engine prefill (81-88 instead of 127 tok/s). Also `QWEN_KEEP_AWAKE=0`. |
| `--progress` | off | A progress bar per prefill on stderr (tokens done, tok/s, ETA), redrawn at most once per second. A prefill under a second draws none. Decode shows tokens so far, tok/s, elapsed time and (NPU engines drafting) tok/step, in the console window's title with `--generate` / `--interactive` (the answer streams to the console, and the old title comes back afterwards) and as a stderr line with `--serve`. Also `QWEN_PROGRESS=1`. GGUF and NPU engine (`MODEL.npu`, the 27B and Flash-Next), not the OpenVINO export. |
| `--max-new N` | 16 | Max tokens to generate. |
| `--temp T`, `--top-k K`, `--top-p P`, `--min-p P` | the profile's | Sampling (built in, thinking 1.0 / 20 / 0.95 and non-thinking 0.7 / 20 / 0.8). `--temp 0` is greedy and keeps only the penalties given explicitly. |
| `--presence-penalty X`, `--rep R` | the profile's | Presence / repetition penalty on the generated tokens (built in, 1.5 presence without thinking, else none). |
| `--seed S` | 1234 | Sampler seed (reproduces a run). |
| `--mtp [N]` | off | Speculative decoding with the GGUF's MTP head, which drafts N tokens per step (bare = 3, cut short by `QWEN_MTP_PMIN`), verified in one batch. Mutually exclusive with `--dflash`. |
| `--dflash <file>` | none | DFlash2 block drafter (`model.safetensors` of `z-lab/Qwen3.8-27B-DFlash2`, with `<name>.json` beside it or `config.json` in its directory, quantized to Q4_K at load). Mutually exclusive with `--mtp`. |
| `--layers N` | none | Run only the first N transformer layers (`--forward` only). |

## Model / resource options

| Flag | Default | Meaning |
|---|---|---|
| `--n-gpu-layers N` | 0 | How many *last* layers (and their weights) live on the GPU. 0 = decode fully on CPU, but prefill still streams weights through the GPU when it helps (see `QWEN_GPU_PREFILL`). |
| `--ctx N` | model native (262144) | Context window in tokens. On an OpenVINO export it is `--ov-ctx` unless that is given too. |
| `--batch N` | 2048 | Prefill chunk size (positions per chunk). |
| `--mmproj <mmproj.gguf>` | none | Load the vision projector (enables `--image` / image blocks). Without it, an image request is an error, not a silent drop. |
| `--image <file>` | none | Attach an image (repeatable). Formats are PNG, baseline JPEG, BMP, Netpbm. |
| `--image-max-tokens N` | 1024 | Cost knob, in *merged* tokens, for how many text tokens the tower's output occupies (4N patches through the tower). |
| `--image-dump <file.ppm>` | none | Write the exact resized pixels the tower saw. |
| `--prompt-cache-mb N` | 32768 | MiB of states `--serve` keeps for the prefix cache (`crates/prefixcache`). 0 disables it. |
| `--prompt-cache-dir DIR` | none | Disk tier of `--serve`. Shared prefixes are written there (pack files, keyed on the model's contents) and found again after a restart. Writes are limited by flash wear (`QWEN_CACHE_WEAR_S_PER_GB`, `QWEN_CACHE_WRITE_GB_DAY`). |
| `--prompt-cache-disk-mb N` | `--prompt-cache-mb` | Budget of the disk tier. |
| `--prompt-cache-spill` | off | States evicted from RAM go to the disk tier (when worth their writes) and are read back when that beats the prefill. |

## Server options

| Flag | Default | Meaning |
|---|---|---|
| `--host H` | 127.0.0.1 | Bind address for `--serve`. |
| `--client-ignore PREFIX=FIELD[,FIELD...]` | `*=temperature` | Repeatable. Drop these top-level request fields from clients whose `User-Agent` starts with `PREFIX` (`*` = every client), so the profile's values apply. Giving the flag replaces the default, and `off` clears it. |
| `--record-requests DIR` | none | Make the server write every POST to `DIR` as it arrives (`{unix ms}-{n}.req` with the request line, the headers without the API key, and the body). `cargo run --release -p server --example replay -- DIR [HOST:PORT]` posts them to a running server again, one after the other without the original gaps, and prints each one's time to first content and total. |

Run `--serve` with an explicit `--ctx`, because Zed reads the context window from
`/props`.

## OpenVINO export (model argument `<openvino-export-dir>`)

Runs the export through the OpenVINO runtime with its own weights and
`tokenizer.json`. It runs one request at a time, and each request is reset and prefilled
from scratch (no prefix cache, no drafts). Use the symmetric int4 export
(`Qwen3.8-27B-int4sym-ov`), because the hub's asymmetric one is ~3x slower on the NPU.

| Flag | Default | Meaning |
|---|---|---|
| `--ov-device D` | NPU | `NPU` is the static stateless engine. `CPU` / `GPU` run the stateful export as-is (a reference only, with CPU at ~3.2 tok/s decode). |
| `--ov-ctx N` | 4096 | Context in tokens. On the NPU it is the **static KV capacity** compiled into the graphs, so a different value is a different compile. |
| `--ov-chunk N` | 0 | Positions per prefill `infer`. 0 = every prompt token through the 1-position graph (~0.63 s each). **Use 32** (~11 tok/s prefill). A shorter remainder runs right-padded through the same graph. |
| `--ov-cache DIR` | `%TEMP%\qwen-rs-ovcache` | Compiled-blob cache. An NPU compile is 4-14 min per graph, and a cached load is ~10-15 s. |
| `--ov-lib DIR` | none | Directory with `openvino_c.dll` (else `OPENVINO_LIB`, `INTEL_OPENVINO_DIR`, the default install). |
| `--ov-prop KEY=VALUE` | none | Extra compile property, appended last (overrides the NPUW defaults). Repeatable. |

## NPU engine (model argument `<model.npu>`)

`model.npu` is a text file of `key = path` lines, with prefill sets
(`prefill`, `small`, `long`, `long_small`, `set`), `decode`, `mtp`, `emb`,
`tokenizer`, optional `vision = DIR` (the vision tower set), `scan = intel`
(the Vulkan adapter name for the iGPU recurrence) and `mlpgpu = geforce`
(the MLP split on the GTX 1060). It runs one request at a time, and the process holds
the machine-wide NPU lock.

With `arch = flash` (Qwen3.8-Flash-Next) the keys are instead `graphs` (the
compiled blobs), `weights` (the `fl{L}.*.in` files), `ngram` (the n-gram
table), `head` (the lm_head as f16), `emb`, `tokenizer`, `mtp = DIR` (the MTP
head, drafts only with `dense`), `vision = DIR`, `scan = ADAPTER` and
`dense = ADAPTER` (decode's dense part, the hyper-connections and the lm_head
on that GPU). Unknown keys are an error.

The `--serve` / `--prompt-cache-*` / `--image` flags apply as above.

## Image generation (model argument `<qwen-image-dir>`, or `--image-model`)

Qwen-Image-2.1 (`docs/image.md`). The model argument is the checkpoint
directory (the CPU backend, with the text encoder's and the transformer's
matmuls as Q6_K, `QWEN_IMAGE_WEIGHTS`) or the `image.npu` that `--npu-build`
writes from it (the NPU backend).

| Flag | Default | Effect |
|---|---|---|
| `--txt2img PROMPT -o FILE.png` | none | One image without a server. `--image FILE` (repeatable) makes it an edit. The other options are `--size WxH` (multiples of 32), `--resolution N` (edit inputs and an unsized target at N^2 pixels, default 1024), `--seed N`, `--steps N` (0 = default 9, the 8 distilled steps plus one refinement step, 8 = the saved schedule, range 1-50), `--noise FILE.f32`. |
| `--serve [PORT]` | 8080 | With a `<qwen-image-dir>` first, it serves the image routes only (`POST /mcp`, `/v1/images/generations`, `/v1/images/edits`). |
| `--image-model DIR\|image.npu` | none | With any other `--serve`, it serves the image routes next to chat. The model loads on the first image job and chat waits while one runs. |
| `--image-out-dir DIR` | none | Repeatable. Where MCP clients may write besides their roots. The first takes files without an `output_path`. |
| `--image-idle SECS` | 300 | Drop the image model after this long without a job. |
| `--mcp-stdio [URL]` | `http://127.0.0.1:8080` | First argument, no model. One JSON-RPC message per stdin line is POSTed to `URL/mcp`, with answers (and SSE events) one per stdout line. |

## `--npu-build` (model argument `<bf16-checkpoint-dir>`)

Build the NPU engine's files from the BF16 checkpoint (`crates/npubuild`),
with no Python, no OpenVINO and no export, in ~20 min and ~25 GB. A step is skipped when
its output already exists. For the Flash-Next checkpoint the step list is
`emb,ngram,weights,head,graphs,compile,vis,mtp,sets` (~165 GB in all), and for
the Qwen-Image checkpoint `cal,weights,graphs,compile,vis,sets` (~6 min,
writes `image.npu`).

```
cli.exe Qwen3.8-27B-BF16 --npu-build OUT \
    [--npu-calib FILE]... [--npu-calib-chunks N] \
    [--npu-steps emb,weights,graphs,compile,sets] \
    [--npu-compiler DIR] [--npu-jobs N] [--npu-no-gpu]
```

| Flag | Default | Meaning |
|---|---|---|
| `--npu-build OUT` | none | Output directory (required). |
| `--npu-calib FILE` | built-in frozen texts (`crates/npubuild/calib`), source code and prose for the 27B (9 chunks), and also code and public-domain poetry for Flash-Next | Calibration text file for the per-layer Hessians (repeatable, with chunks taken from each file in turn). |
| `--npu-calib-chunks N` | 16 | Max 256-token calibration chunks. |
| `--npu-steps a,b,..` | all | Steps to run, from `emb`, `weights`, `graphs`, `compile`, `sets` (27B, with `ngram`, `head`, `vis`, `mtp` added for Flash-Next). `emb`, `weights` and `graphs` need neither the compiler nor the NPU. |
| `--npu-compiler DIR` | `NPUC_BIN`, else `npu-compiler` beside `cli.exe` | Directory with `vpux-translate.exe` / `vpux-opt.exe` (our patched compiler). |
| `--npu-jobs N` | 6 | Parallel graph-compilation jobs. |
| `--npu-no-gpu` | off | Don't use the GPU for the inverse-Cholesky factor of GPTQ (~4 s/layer slower). |

---

## Environment variables

All `QWEN_*` (plus the three OpenVINO-runtime variables). Set them in the
shell that starts `cli.exe`. Defaults are the shipped ones, and `0` / `off` /
`no` / `false` mean "off" wherever a value is read. Most are performance
tuning and switches that turn one optimization off, so it can be compared
with the default. For those the table says what the other value runs.

### Threads / CPU

| Variable | Default | Effect |
|---|---|---|
| `QWEN_THREADS` | cores (24) | Worker-pool size. Diagnostic only. |
| `QWEN_DECODE_THREADS` | 2/3 of pool (16) | Shards a decode step uses. Decode is DRAM-bound, so fewer threads are as fast with less CPU time. |
| `QWEN_GEMM_PTILE` | 32 | Position-tile width for the CPU prefill GEMMs. The largest single CPU prefill lever (2.4x). Bit-exact. |
| `QWEN_GEMM_RGROUP` | 8 | Weight rows per block-outer group in the CPU prefill GEMMs. 1 means one row at a time. Bit-identical. |
| `QWEN_GEMM_VNNI` | on where the CPU has AVX-VNNI | CPU prefill GEMMs on AVX-VNNI (4 uops vs 6 on AVX2). Not bit-identical (different f32 lane rounding). |
| `QWEN_GEMM_HSUM` | off | Horizontal-sum variant of the Q4_K row GEMM, for comparison. |
| `QWEN_ALLOC_CACHE_MB` | 4096 | Bound on the binary's recycling allocator (frees ≥1 MiB blocks for reuse). 0 disables it. |
| `QWEN_KEEP_AWAKE` | on | `0` is the same as `--no-keep-awake` (allow display / system sleep). |
| `QWEN_CALIB` | on | Work-split calibration (`quant::calib`). A split keeps its cost shape in code and learns one scale factor per device from the run's own timings, then freezes it (printed as `calib: NAME: fitted factors ...`). Entries are keyed by the executable (size + mtime) and the split's inputs, so a rebuild or new NPU graphs refit automatically, and a frozen entry that drifts > 15 % warns once. Today it covers the Flash-Next prefill's NPU / iGPU expert split (NPU run time and iGPU timestamps, 96 layers). `0` uses code defaults only, and `reset` refits. Entries in `QWEN_CALIB_DIR` (default `%LOCALAPPDATA%\qwen-rs\calib`). |
| `QWEN_ENERGY` | on | The `[eff]` line's energy part, with joules and tok/J per phase, package watts (cores, iGPU, rest incl. NPU) from the RAPL `Energy Meter` counters, GTX 1060 watts from NVML (power integrated every 100 ms on Pascal). Machine-wide, not per process, and meaningless for phases under ~100 ms. `0` turns it off. |
| `QWEN_POOL_STATS` | unset | With `=1`, `--generate` prints `[pool]` lines (shard / spin / join time, per call site) after `[eff]` (for Flash-Next `--generate`, the decode's), and so does an NPU-engine `--serve` with `QWEN_NPU_TIMING`, per request (prefill and decode). |

### GPU offload (prefill routing)

| Variable | Default | Effect |
|---|---|---|
| `QWEN_GPU_PREFILL` | on | Route prefill's big matmuls through the GPU (streaming included). Permission only, since per-call crossovers decide. |
| `QWEN_GPU_PREFILL_MIN_SEQ` | 64 | Chunk length below which a *resident* weight's prefill matmul stays on CPU. 0 disables the gate. |
| `QWEN_GPU_GEMM_MIN_SEQ` | 9 | Shortest `seq` that uses the tiled GEMMs instead of the batched GEMV. `99999` turns tiling off everywhere (for comparison with the short-chunk split). |
| `QWEN_GPU_STREAM` | on | Stream a layer's weights through VRAM for prefill instead of requiring them resident. |
| `QWEN_GPU_STREAM_MIN_SEQ` | 256 | Chunk length below which a *streamed* weight stays on CPU (it also pays its upload). |
| `QWEN_GPU_STREAM_WORKER` | transfer thread | `=0` puts the staging repack on the calling thread, and `=par` on the shared pool (a wash). |
| `QWEN_GPU_SPLIT` | auto | Decode **row split**. Each decode matmul keeps its first ~34 % of rows in VRAM on the GPU and the CPU does the rest concurrently. Number in (0, 1] forces the fraction. |
| `QWEN_GPU_PREFILL_SPLIT` | on | Prefill **row split**, with rows `[0, g)` on the GPU, `[g, out)` on the CPU, in place. Needs the decode split for its resident rows. |
| `QWEN_PREFILL_GPU_FRAC_Q4K` / `QWEN_PREFILL_GPU_FRAC_Q6K` | 0.73 / 0.72 | GPU row fraction of a prefill split (0.66 / 0.65 below 192 tokens, 0.62 / 0.61 below 128, 0.55 / 0.60 on the float GEMMs). From the measured GPU vs CPU GEMM rates and confirmed by sweeps. |
| `QWEN_PREFILL_DFRAC_<ID>` | FFNDOWN +0.03, WQKV −0.03, SSMOUT −0.02, rest 0 | Per-matmul correction on 64+ token chunks (`prefill_frac_delta`). An entry set to 0 applies no correction. |
| `QWEN_GPU_GEMM_FLOAT` | unset | When set, prefill GEMMs run on the float kernels instead of the dp4a ones (~2x kernel-only), and the split fractions fall back with it. |
| `QWEN_GPU_NODP4A` | unset | When set, the float Q4_K kernels run instead of the integer-dot-product (dp4a) ones. |
| `QWEN_GPU_Q8GEMV` | unset | When set, per-matmul decode GEMVs run on the q8_K kernels instead of f32. |
| `QWEN_GPU_ATTN_GROUPS` | 2 | KV groups (of 4) of the prefill attention that the GPU runs while the CPU runs the rest. 0 = CPU only. |
| `QWEN_GPU_ATTN_FRAC` | 0.88 | Share of those groups' causal work the GPU keeps, while the CPU runs the last query rows. 1 keeps all of them. |
| `QWEN_GPU_ATTN_MIN_KEYS` | 4096 | Context length from which the attention split applies. |
| `QWEN_GPU_CTX` | 8192 | Context the GPU attention buffers are sized for. |
| `QWEN_VRAM_RESERVE_MB` | 512 | VRAM left free for prefill streaming slots and per-call buffers. Too small and WDDM silently demotes VRAM to system RAM, and
decode collapses rather than fails. |
| `QWEN_VRAM_GB` | unset | Static VRAM budget for layer placement (used only without the `VK_EXT_memory_budget` extension). |
| `QWEN_GPU_RECOVER_SECS` | 15 | Seconds between attempts to rebuild a lost Vulkan device (after another app claimed VRAM). 0 = never. |
| `QWEN_GPU_IMPORT_MB` | 24576 | Process-wide cap on optional host-memory imports into a Vulkan device (the image model's iGPU block share next to Flash-Next's imported expert pools). An import past it is skipped and the NPU runs that work. Past ~29 GB the NPU was lost for good. |
| `QWEN_GPU_Y_DIRECT` | on | Prefill GEMMs write straight into the host-visible readback buffer (no trailing GPU copy). Bit-identical. |
| `QWEN_GPU_Y_INPLACE` | on | The prefill split's GEMMs write their rows directly into the caller's output matrix via `VK_EXT_external_memory_host`. Bit-identical. |
| `QWEN_GPU_STREAM_FOLLOW` | on | A split prefill matmul submits the streamed rows behind the upload on the transfer thread instead of waiting host-side. Bit-identical. |
| `QWEN_GPU_STREAM_SYNC` | unset | When set, a host-side wait comes before one submission instead. |
| `QWEN_DEFER` / `QWEN_DEFER_FRAC` | on / 0.94 | Defer a layer's `WqkvGate` (and the attention output gate) onto the GPU, running while the CPU does the other passes. Superseded by the pipeline on long chunks. |
| `QWEN_DEFER_QGATE_FRAC` | 1.0 | GPU share of the deferred output-gate rows. |
| `QWEN_PREFETCH` | on | Below the pipeline (96-511 tokens), upload the next prefill matmul's streamed rows while the current call runs. Bit-identical. |
| `QWEN_PIPE` / `QWEN_PIPE_MIN` | on / 512 | Two-half prefill pipeline, where each layer's five GEMM stages run on one half while the CPU does the other's phase. Supersedes `QWEN_DEFER` on those chunks. Bit-identical with the right row placement. |
| `QWEN_PIPE_OVERLAP` | on | Keep two GEMM batches in flight in the pipeline. Bit-identical. |
| `QWEN_PIPE_A<j>` / `QWEN_PIPE_B<j>` | A `0.035 0.11 0.03 0 0.005`, B `0.075 −0.02 0.03 0.005 0` | Per-stage GPU-fraction extras of the pipeline (stage 0-4, per half). Tune from `QWEN_GPU_TIMING`'s `[DEFER] ... wait`. |
| `QWEN_PIPE_FA0` / `QWEN_PIPE_FB0` | 0.05 / 0.22 | Attention-stage 0 extras (B overlaps A's attention). |
| `QWEN_PIPE_QGATE_A` / `QWEN_PIPE_QGATE_B` | 1.0 / 0.7 | Gate-row fractions of `Wq` per half. |
| `QWEN_SSM_COLS` | 32 | Value-column block per prefill SSM-scan task (multiple of 8 dividing the state width). |
| `QWEN_FUSED_SWIGLU` | on | Where a prefill matmul group takes the row split, its q8_K input is quantized row by row as it is computed (SwiGLU straight from `silu(gate) * up`, norms straight from `norm_seq_in`) instead of a full f32 intermediate. Bit-identical. |
| `QWEN_PREFILL_SMALL_SCALE` / `QWEN_PREFILL_SMALL_Q6K` | 1.0 / 0.65 | Short-chunk (16-63 token) prefill split, with a scale on the Q4_K GPU share and the Q6_K factor on it. |

### GPU decode / layer blocks

| Variable | Default | Effect |
|---|---|---|
| `QWEN_GPU_LAYER` | unset | When set, runs the fused GPU layer block for one decode step (text MRoPE only, falling back once an image is in the context). Disables the server's prefix cache. |
| `QWEN_GPU_LAYERCMP` | unset | Keep the fused layer output aside and run the CPU path too, reporting the divergence. |
| `QWEN_GPU_NOBLOCK` | unset | Disable the fused GPU attention/SSM blocks. |
| `QWEN_GPU_NOGEMM` | unset | Disable the GPU GEMMs entirely. |
| `QWEN_GPU_NOSEQGEMV` | unset | Disable the seq GEMV path. |
| `QWEN_GPU_NOCMDCACHE` | unset | Disable the decode command cache. |
| `QWEN_GPU_NOPOOL` | unset | Disable the host buffer pool. |
| `QWEN_GPU_STAGE` | unset | Capture fused-block intermediates to localize which stage produces garbage (fused-block aux shaders are skipped on llvmpipe). |
| `QWEN_GPU_TIMING` | unset | Per-call phase split of the GPU matmul path (`pack / alloc / run_batch / free`, `submit / readback` with the readback GB/s). The tool to reach for on any GPU perf question. |
| `QWEN_VK_DEVICE` | unset | Select the Vulkan adapter by name. |
| `QWEN_GPUCMP` | unset | Compare every GPU prefill matmul against the CPU (`worst |GPU-CPU|`). |
| `QWEN_ATTN_SPV` | unset | Load the prefill attention kernel from this `.spv` file instead of the embedded one (iterate without a Rust rebuild). |

### Speculative decoding

| Variable | Default | Effect |
|---|---|---|
| `QWEN_MTP_VOCAB` | 32768 | Reduced `--mtp` draft vocabulary (rows `[0, N)` + special + every id seen). 0 = full head. Lossless. |
| `QWEN_MTP_PMIN` | 0.8 | `--mtp` stops chaining drafts once the MTP head's top probability falls below this. |
| `QWEN_DFLASH_N` | 4 (max 7) | Drafts `--dflash` verifies per step. |
| `QWEN_DFLASH_PMIN` | 0 | `--dflash` verifies only up to the first draft whose drafter probability is below this. |
| `QWEN_DFLASH_QUANT` / `QWEN_DFLASH_HEAD` | `q4k` | Format of the drafter's matmuls / its private lm_head copy (or `q6k`). |
| `QWEN_DFLASH_VOCAB` | 151643 | Reduced `--dflash` drafter-head vocabulary. 0 = full head. |
| `QWEN_DFLASH_BLOCK` | `N + 1` (max 8) | Rows the drafter runs per draft. |
| `QWEN_DFLASH_DEBUG` | unset | Per-draft log lines (pos, ctx, drafts, probabilities). |
| `QWEN_SUFFIX` | on | Context drafts copy up to 7 tokens from an earlier occurrence of the last 3+ tokens and verify them like MTP drafts. |
| `QWEN_SUFFIX_GLOBAL_TOKENS` | 1048576 | Server only. Tokens of earlier responses kept for context drafts across requests. 0 disables it. |
| `QWEN_SUFFIX_PLAIN_MIN` / `QWEN_SUFFIX_MIN` / `QWEN_SUFFIX_ENGINE_MIN` | 3 / 12 / 6 | Match length a context draft needs without a model drafter / to replace an MTP-DFlash draft / for an engine that drafts on its own (the NPU's MTP). |
| `QWEN_SUFFIX_EV` | on | On engines that report step costs (Flash-Next), a context draft is verified only when its expected tokens per step cost, learned per match-length bucket and draft position, beat the MTP steps' recent rate, and cut to the best length. `0` = the fixed `QWEN_SUFFIX_ENGINE_MIN` rule. |
| `QWEN_FLASH_COUPLED` / `QWEN_FLASH_COUPLED_TAU` | off (on in the Flash-Next profile's `[think]`) / 1 | Coupled sampling on a drafting NPU engine at temperature > 0. Tokens are drawn by Gumbel-max with noise fixed by the request's seed and position (the same distribution), and the MTP head drafts with the same noise, so a draft matches the sample more often. `_TAU` scales the head's temperature for its draw. Changes the sampled text, not its distribution. |
| `QWEN_DRAFT_LOG` | unset | `1` makes an NPU-engine server print one log line per drafted decode step (sources, drafts, accepted). |

### NPU engine (`model.npu` runs)

| Variable | Default | Effect |
|---|---|---|
| `QWEN_NPU_MTP_CTX` | 64 | Prompt positions the MTP catches up over after a prefill. |
| `QWEN_NPU_PAIR` | on | Two consecutive prefill chunks run in flight together, so the NPU never waits for the iGPU recurrence. |
| `QWEN_NPU_FAR_GPU` | on | Far blocks (past the 4096 window) run on the iGPU while the NPU runs the first ones. |
| `QWEN_NPU_FAR_CPU` | off | `=1` runs far blocks on the CPU cores too (a share below the iGPU's, prefill only, off so the cores stay free). |
| `QWEN_NPU_FAR_FUSED` | on | The iGPU's far kernel with the softmax in its scores pass (`attn_far_qks_npu.glsl` + `attn_far_pv_npu.glsl`). `0` puts the softmax in the P V pass instead. |
| `QWEN_NPU_FAR_GPU_LEAD` | 2.0 | Split point (measured-rate multiple) for the far-block division. |
| `QWEN_NPU_FAR_MIN` | unset | (test) force the far set from position N on. |
| `QWEN_NPU_FAR_PREALLOC` | unset | (test) preallocate K/V blocks for N positions. |
| `QWEN_NPU_FARDEC` | unset | (test) every decode step on the far blocks. |
| `QWEN_NPU_FARM` | on | One `farm.blob` per far job. `0` uses one block each. |
| `QWEN_NPU_FAR_DUMP` | unset | `DIR:BLOCKS` dumps every attention call's blocks as raw f16 for offline checks. |
| `QWEN_NPU_FAR_STATS` | unset | File to append per-call far stats to. |
| `QWEN_NPU_KQ` / `QWEN_NPU_QQ` | unset | `BITS[:RANGE]` runs an accuracy simulation of integer far keys / queries (rounded to BITS-bit ints per 256-row, own scale or fixed RANGE), not a speed-up. |
| `QWEN_NPU_SCAN_IMPORTS` | driver limit | At most N K/V block imports on the scan device (lower it to test eviction). |
| `QWEN_NPU_SCAN_TS` | unset | Each prefill recurrence on the scan device prints its GPU time (timestamps) next to the host's wait for it (`[scants]`). The difference is time queued behind other work on the iGPU. On Flash-Next it also prints `[igts]` lines, the absolute GPU intervals of the scan submission (conv / recurrence / gated norm) and the delta layers' hyper-connections on the iGPU's one clock. |
| `QWEN_NPU_SUBMIT_LOG=FILE` | unset | One line per NPU queue submission, `FILETIME tag file:line` (UTC 100 ns ticks, where the tag is the Flash-Next prefill's layer and the location is the submitting caller). Matched in order to an ETW `Microsoft-Windows-DxgKrnl` dump's NPU DMA packets, it gives each graph's NPU busy time. No measurable cost. |
| `QWEN_NPU_TIMING` | unset | Per-phase prefill/decode timing lines. With an NPU-engine `--serve`, it prints per request the prefill steps (reset, restore, each feed, snapshot), a `request timeline` line (body read, parse + render + tokenize, lock wait, prefill, first token, decode split into host / context drafts / accept / verify, time after decode, idle since the previous request) and an `[eff]` line (CPU-s, cores busy, CPU-ms per token for prefill and decode). On Flash-Next it also prints each feed's time and the iGPU attention mirror's refill, and a `drafts:` line per request (engine and context drafts verified / accepted, steps whose first draft was accepted). |
| `QWEN_ENGINE_LIVE` | on | On an NPU-engine `--serve`, a request whose prompt extends the previous request's prompt and reply continues the engine's state (no restore, the reply not prefilled again). `0` restores a prefix-cache checkpoint every time. |
| `QWEN_NPU_DUMP` | unset | Raw f32 first-token logits. |
| `QWEN_NPU_DUMP_RESID` | unset | Append each prefill chunk's final residual rows (f16) for the KL eval. |
| `QWEN_NPU_IDS` | unset | Use these token ids (whitespace-separated) instead of the prompt's, and run the KL eval through the engine. |
| `QWEN_NPU_SPEC_DEBUG` | unset | Speculative-decoding step log. |
| `QWEN_NPU_NOPREFILL` | unset | Every token through decode/verify steps (separates prefill error from decode error). |
| `QWEN_NPU_CONT` | unset | `N,flag` runs N snapshot-reset-restore round trips before the rest (the prefix cache's path), to compare against one prefill. |
| `QWEN_NPU_MLP_THREADS` | 12 | Threads per MLP pass on the cores. |
| `QWEN_NPU_MLP_VNNI` | on where available | AVX-VNNI in the core MLP matmuls. |
| `QWEN_NPU_MLP_TIMING` | unset | NPU / CPU ms per MLP pass. |
| `QWEN_NPU_MLP_GPU_FRAC` | `0.9,0.8` | Share of the MLP rows the GTX 1060 takes, at 2 and at 3-4 positions. |
| `QWEN_NPU_MLP_GPU_RESERVE_MB` | 512 | VRAM left free beside the GPU MLP share (room for other applications). |
| `QWEN_NPU_MLP_GPU_ROWS` | 17408 | Total rows considered for the GPU share (10240 of them live in VRAM). |
| `QWEN_NPU_MLP_GPU_SYNC` | unset | Synchronize after each GPU MLP share (diagnostic). |
| `QWEN_NPU_VIS_TIMING` | unset | NPU-job and iGPU-attention time per image. |
| `QWEN_VIS_ATTN_TIMING` | unset | Runs every pass of the iGPU tower attention as its own submission and prints its time. |

### OpenVINO export (`<export-dir>` runs)

| Variable | Default | Effect |
|---|---|---|
| `OPENVINO_LIB` | unset | Directory with `openvino_c.dll` (before `INTEL_OPENVINO_DIR` and the default install). |
| `INTEL_OPENVINO_DIR` | unset | Install root. `<dir>/runtime/bin/intel64/Release` is tried. |
| `QWEN_OV_NPU_LOCK` | on | `0` skips both machine-wide NPU locks (compile slots + device mutex). Not on a shared machine, because a `--serve` process holds the NPU for its whole life. |
| `QWEN_OV_STATIC_DEVICE` | NPU | With `--ov-device NPU`, compile the static stateless graphs for `DEV` instead (e.g. `CPU`, the lock-free reference). |
| `QWEN_OV_DELTA` | `chunk` | How the prefill graph replaces the delta-net Loops. `chunk` is the chunked WY form, `unroll` runs one recurrent step per position (slower), and `loop` keeps the Loops (CPU only). |
| `QWEN_OV_DELTA_C` | largest of 32/16/8/4/2/1 dividing `--ov-chunk` | Positions per sub-chunk of the above (at most 16 with `neumann`). |
| `QWEN_OV_DELTA_BATCH` | on | Batch the non-sequential sub-chunk ops. |
| `QWEN_OV_DELTA_INV` | `blocked` | How `(I + A)^-1` is formed. `blocked` is the block-mask form, with all intermediates \|T\| ≤ 1, and `neumann` is the old factor chain (f16 overflow, with partial products reaching 1e4x the result). |
| `QWEN_OV_CONVSCALE` | auto (≤5) | `K` scales the conv1d weights x2^K and the outputs x2^-K, which is the NPU accuracy fix (int8-asym conv scales below f16 normal range come out zero otherwise). 0 = off. |
| `QWEN_OV_DSCALE` | `8,4,4;6,8,6` | Per-layer delta-net power-of-two scales (`kq,kv,ko;…`) that stop the NPU flushing subnormal products in the recurrence. 0 = off. Ignored when `QWEN_OV_QSCALE` / `QWEN_OV_VSCALE` are set. |
| `QWEN_OV_QSCALE` / `QWEN_OV_VSCALE` | 64 / 1 | Uniform power-of-two scales in place of `QWEN_OV_DSCALE`. Setting either selects them. |
| `QWEN_OV_PAD` | on | Right-pad the prompt remainder through the chunk graph (one call) instead of one position at a time. |
| `QWEN_OV_PAD_MIN` | `ceil(chunk/7)` (min 2) | Shortest remainder that is padded. 1 also sends decode steps through it (a test). |
| `QWEN_OV_LAST_LOGITS` | on | Only the last prompt position goes through the lm_head (keeps the head's NPU blob under 2 GiB so it caches, since the graph recomputes without it). |
| `QWEN_OV_ROPE` | `graph` | RoPE handling in the export (`graph` = in the graph). |

The OpenVINO diagnostic and comparison variables are `QWEN_OV_DUMP=FILE` (every sampled
position's logits), `QWEN_OV_PROBES=names` + `QWEN_OV_PROBE_DUMP=FILE`,
`QWEN_OV_TF_DUMP=FILE`, `QWEN_OV_STATE_SAVE=DIR@POS` / `QWEN_OV_STATE_LOAD[_SKIP]` /
`QWEN_OV_STATE_LOAD_ONLY` (saving / loading model states),
`QWEN_OV_KEEP_XML=DIR` / `QWEN_OV_XML_ONLY=DIR` (keep / only write the rewritten
graphs, for the micro-graph scripts, debug only).

### NPU build (`--npu-build`)

| Variable | Effect |
|---|---|
| `NPUC_BIN` | Directory with the patched compiler's `vpux-translate.exe` / `vpux-opt.exe`, used when `--npu-compiler` is not given (before `npu-compiler` beside `cli.exe`). |
| `NPUC_TBB` | The compiler's TBB bin directory (default the compiler's directory when it holds `tbb12.dll`, else `…/temp/Windows_AMD64/tbb/bin` beside the compiler). |

### Flash-Next (`arch = flash` runs)

| Variable | Default | Effect |
|---|---|---|
| `QWEN_FLASH_KSUP` | 16 | 256-position sub-chunks per super-chunk (one routed-expert pass per layer over the piece). 16 instead of 8 gives 146.5 instead of 131.9 tok/s at 3.2k. |
| `QWEN_FLASH_PREFETCH` | on | The next 4096-token piece's n-gram rows read on a background thread while the current piece runs, and every reader thread has its own handle to `ngram.bf16`. 0 = no prefetch. |
| `QWEN_FLASH_POKE` | on | Keep the scan device awake between jobs (expert passes included, since a wake-up costs ~0.5 s). `0` turns it off. |
| `QWEN_FLASH_PLE_FIRST` | on | Layer 1's PLE graphs run for every sub-chunk before the layer, not inside its pipeline. `0` runs them inside it. |
| `QWEN_FLASH_HC_GPU` | on | A delta layer's attention hyper-connection on the iGPU (`offload::HcPre`), and `delta_pre_m` starts from its output. `0` turns it off. |
| `QWEN_FLASH_CONV_GPU` | on | The delta layers' short conv, Swish and q / k l2 norms on the iGPU ahead of the recurrence (`conv_npu.glsl`), and the NPU's `delta_pre_c` ends at the projections. Needs `QWEN_FLASH_HC_GPU` and the `delta_pre_c` blob (`--npu-steps graphs,compile`). `0` runs `delta_pre_m` on the NPU. |
| `QWEN_FLASH_GN_GPU` | on | The delta layers' gated norm and the out-projection's input rotation on the iGPU, in the recurrence's submission (`gn_npu.glsl`, in place of its output), and the `post_delta_t` NPU graph (out-projection, injection) in place of `post_delta_a`. Needs `QWEN_FLASH_HCM_GPU` and the `post_delta_t` blob (`--npu-steps graphs,compile`). `0` uses `post_delta_a` instead. |
| `QWEN_FLASH_IDX_MIN` | 1024 | Blocks of 4 positions before a prefill attention sub-chunk from which the block indexer's scores of them run on a device (the iGPU, see `QWEN_FLASH_IDX_GPU`) instead of the cores. `0` = always on the cores. |
| `QWEN_FLASH_IDX_GPU` | on (with `scan`) | The indexer's scores run on the iGPU (int8 `dp4a` kernel over an int8 mirror of the pooled keys), queued after the sub-chunk's projections. `0` = the NPU's `idx_score` graph (needs its blob, `--npu-steps graphs,compile`). |
| `QWEN_FLASH_HCM_GPU` | on | The delta layers' MoE hyper-connection on the iGPU between the `post_delta_a` / `post_delta_b` NPU graphs (`FlashEngine::delta_split`). Needs `QWEN_FLASH_HC_GPU`, two banks and the PLE graphs. `0` turns it off. |
| `QWEN_FLASH_HC_IMPORT` | on | The prefill hyper-connections on the iGPU read their streams from the NPU's buffers in place (imported once) instead of copying them per call. `0` copies them. Bit-identical. |
| `QWEN_FLASH_HC_DP4A` | on | Both hyper-connections' down / up matrices on dp4a with int8 inputs in groups of 16 (`hc_*_dp4a.glsl`, 4.0 instead of 5.4 ms a call). `0` uses the float kernels. |
| `QWEN_FLASH_EXP_GPU` | balance | Routed-expert slots the iGPU takes (`offload::Experts`, int8 x int4 dp4a) while the NPU runs the rest. When unset, whole NPU jobs move to the iGPU while that lowers the larger of the two devices' estimated times. A number `F` takes a fixed share `F` of each slot kind instead, and `0` = none. The estimates' per-device scale is fitted by `QWEN_CALIB`. |
| `QWEN_FLASH_EXP_GPU_NE` | 224 | Only experts `[0, N)` have their pools imported for the iGPU share (0.110 GiB an expert, 24.7 GiB at 224). Everything the NPU, the iGPU and the GTX 1060 keep resident in system memory shares one WDDM limit (dxdiag's "Shared Memory", by default 80 % of RAM, ~102 GiB with 128 GiB, and the registry value `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\MemoryManager\SystemPartitionCommitLimitPercentageMax` = 90 plus a reboot raises it to RAM minus 16 GB, ~111.5 GiB, and the Intel driver's "Shared GPU/NPU Memory Override" sets only the minimum), and the iGPU's attention mirrors add ~24 KiB a context position. At 320 a 117k-token conversation reached the limit and lost the GTX 1060, then the NPU. Raise it only with a shorter context in mind. |
| `QWEN_FLASH_CPU_EXP` | on | A one-position decode step's routed experts on the CPU cores (int4 x int8) instead of an NPU job. `0` uses an NPU job. |
| `QWEN_FLASH_HEAD_GPU` | on | With `dense`, the lm_head runs on the GPU too. `0` keeps it on the cores. |
| `QWEN_FLASH_HEAD` | unset | `f16` uses the host lm_head as is (else int8 per row, half the bytes read per decode step). For comparison with `QWEN_FLASH_HEAD_GPU`. |
| `QWEN_FLASH_CHAIN` | on | Chained hyper-connections on the GPU decode path (one submission per layer). `0` uses six submissions per layer. |
| `QWEN_FLASH_C64` | on | The wide expert job slots. `0` puts every expert in slots of `c`. |
| `QWEN_FLASH_PE` | on | The per-expert wide graph (`experts_w`). `0` uses `experts_c64` instead. |
| `QWEN_FLASH_WS` | on | The expert graphs without the row-scale Multiply (`experts_ns` / `experts_ws`). `0` puts it on the NPU. |
| `QWEN_FLASH_ATTN_GPU` | on | Prefill attention on the iGPU (`offload::FlashAttn`). `0` runs it on the cores. |
| `QWEN_FLASH_FAR_NPU` | on | Prefill attention runs whole key blocks on the NPU while the iGPU runs the rest, merged by log sum exp (`crates/npu/src/flash/far.rs`). `0` runs all of it on the iGPU. |
| `QWEN_FLASH_FAR_COST` | `12,4.6,7.1,0.0146,0.023` | The far split's cost model in ms, with NPU work per sub-chunk, dense / masked block job, iGPU per key of a dense / selected query (per 256 queries). |
| `QWEN_FLASH_FAR_SLOTS` | `2` | Most NPU jobs per attention sub-chunk (1-8). |
| `QWEN_FLASH_BANKS` | 2 | Buffer banks a layer's sub-chunks are pipelined over. `1` runs them one after the other. |
| `QWEN_FLASH_MOE_DEFER` | 2 | In the prefill expert pass, `2` gathers and combines the experts' outputs into each sub-chunk's streams in the next layer, just before it reads them (while the devices run), `1` defers the combine only, and `0` does both right after the pass. Bit-identical. |
| `QWEN_FLASH_MOE_SPLIT` | 3 | Number of lists the wide expert jobs are submitted in (a single list's driver record takes 50-70 ms). |
| `QWEN_FLASH_DENSE_RESERVE_MB` | 256 | With `dense`, VRAM left free when the dense weights are uploaded. A GPU without the room, or one that fails or slows down (another application took its VRAM), leaves decode on the one-position NPU graphs until a retry every `QWEN_GPU_RECOVER_SECS` succeeds. |
| `QWEN_FLASH_LOAD_THREADS` | 4 | Threads that fill the expert weight buffers in parallel while the main thread allocates the pinned buffers (4 reach the NVMe's ~6.7 GB/s, while 2 and 8 were slower). |
| `QWEN_LOAD_TIMING` | unset | `[load] START -> END (DUR s) WHAT` per piece of the Flash-Next load, on one clock from process start, showing what overlaps and what the critical path is (the expert reads). |
| `QWEN_FLASH_MTP_N` / `_PMIN` / `_PSCALE` / `_COST` / `_CTX` | 3 / 0.2 / 0.8 / measured / 64 | The MTP head with `dense`, in order the drafts per chain, the probability product a chain stops under, the factor on each draft's probability (the head is overconfident), the step costs (ms for 1, 2, … positions, where `0` keeps the whole chain) by which the draft prefix with the most expected tokens per ms is kept, and the prompt positions the head catches up over after a prefill. |

The Flash-Next checks and dumps are `QWEN_FLASH_D1=N` (every position through decode
steps of N, where `1` is the decode-path KL check, and every N dumps the same
logits as `1`), `QWEN_FLASH_DUMP_ALL=FILE` /
`QWEN_FLASH_DUMP_LAST=FILE` (every / the last prompt position's logits, raw
f32, for a KL check), `QWEN_FLASH_SNAPCHECK=N` (snapshot / restore round trip
after all but N prompt tokens, and the max diff must be 0),
`QWEN_FLASH_HEAD_CHECK`, `QWEN_FLASH_FAR_CHECK=1` (each far merge against the
iGPU alone), `QWEN_FLASH_SEL_DUMP=FILE` (every routing decision as
`[layer, e0..e9]` u16, for a study of expert reuse), and the log lines
`QWEN_FLASH_MTP_LOG`, `QWEN_FLASH_MOE_LOG`, `QWEN_FLASH_ATT_LOG`,
`QWEN_FLASH_LAYER_LOG`, `QWEN_FLASH_ARGS`.

### Flash-Next decode (`dense` GPU steps)

All default on unless noted, and `0` turns each one off.

| Variable | Default | Effect |
|---|---|---|
| `QWEN_FLASH_MIX_GPU` | on | A delta layer's whole mixer (conv, norms, gates, recurrence, gated norm) on the GPU, the states resident in VRAM. `QWEN_FLASH_MIX_CHECK=1` compares it with the host forms. |
| `QWEN_FLASH_FUSE` | on | A delta layer's mixer and post run (output projection, injection, MoE hyper-connection, router) as one submission. |
| `QWEN_FLASH_MERGE` | on | The small vector steps between the GPU decode's GEMVs merged into the dispatches next to them (SiLU, mix, injection + norm, SwiGLU), with the same values. |
| `QWEN_FLASH_SHARED_ASYNC` | on | The shared expert on the GPU as its own submission while the cores run the routed experts. |
| `QWEN_FLASH_WAVES` / `QWEN_FLASH_WAVES_MIN` | 2 / 2 | A GPU step of 2-8 positions as two waves, the GPU on one wave's dense part while the cores run the other's routed experts (same output). `1` runs one. |
| `QWEN_FLASH_WAVE_PIPE` | on | In a two-wave step, each wave's next GPU run is queued behind the other's, so the host's work for one wave (attention, reading outputs, preparing inputs) overlaps the GPU's for the other (same output). `0` = one run in flight at a time. |
| `QWEN_FLASH_HEAD_SPLIT` | by step size | The lm_head's first N of 16 row blocks on the GPU, the rest on the cores. The default is 12 for steps of 1-2 positions, 10 for 3-4, 9 for 5, 8 for 6-8 (the cores read each row once for all positions). A number sets every step's split, and `16` = all on the GPU. |
| `QWEN_FLASH_SPIN` | 200000 | Pause rounds the pool's decode shards spin before parking (~2 ms), so dispatches after a GPU wait start warm. `0` = the pool's 20 us. |
| `QWEN_GEMV_SPIN_US` | 2000 | Microseconds a GEMV run's fence wait spins before blocking. |
| `QWEN_GEMV_RANGES` / `QWEN_GEMV_NPB` | on / on | A run uploads / reads back only the ranges its caller names, and runs of 2 / 3-4 positions use GEMV builds for that many. `QWEN_GEMV_PROF=1` prints GPU timestamps per stage at exit. |
| `QWEN_GEMV_CMDCACHE` | on | A GEMV run's recorded command buffer is kept by everything it depends on and submitted again instead of recorded again. `0` records every run. |
| `QWEN_FLASH_MTP_GPU` | on | With `mtp` + `dense`, the MTP head's dense part (89 MB) runs on the GPU, and its attention and routed experts on the cores. `0` keeps the whole head on the cores. |
| `QWEN_FLASH_MTP_VBLK` / `QWEN_FLASH_MTP_SPIN` | 6 / on | MTP drafts over the first N of the head's 16 row blocks (lossless), and the drafts run in the step's spin window. |
| `QWEN_FLASH_MTP_AUTO` | off (the profile sets it from 8192 to 32767 prompt tokens) | The step costs measured on the running engine (verify + commit per width, the MTP call as first pass + per draft, running means of weight 1/8) in place of `QWEN_FLASH_MTP_COST`, which stays the prior for widths not measured yet, scaled by the measured ones. With `QWEN_NPU_TIMING=1` a `[flash] step costs` line every 64 verify steps. |
| `QWEN_FLASH_MTP_EARLY` | on | A draft chain stops once the cost model would drop every longer one (lossless). |
| `QWEN_FLASH_MTP_CAL` | off (the profile's `[think]` sets it) | `a,b` models draft acceptance as `sigmoid(a + b ln(1 - p))` instead of `PSCALE x p`. The shipped profile fits it for thinking with coupled drafting (`-0.583,-0.540`), and non-thinking keeps `PSCALE`. |
| `QWEN_FLASH_STEP_MAX` | 96 | Prefill pieces of at most N tokens run as GPU steps of up to 8 positions instead of a prefill chunk (~1.6 s however short). |
| `QWEN_FLASH_NS8` | on where the blobs exist | Expert passes of at most 1024 tokens put experts with at most 8 / 16 tokens in narrow job slots. |
| `QWEN_FLASH_D1` | unset | `N` runs every position through decode steps of N positions (`1` is the decode-path KL check, and every N dumps the same logits as `1`). |

### Prefix cache, profiles, server

| Variable | Default | Effect |
|---|---|---|
| `QWEN_PROFILE` | unset | The profile file when `--profile` is not given. |
| `QWEN_MTP_N` / `QWEN_DFLASH_N` | the load's | Drafts per step for a request (profile knobs), at most what `--mtp N` / `--dflash` loaded. |
| `QWEN_CACHE_INTERVAL` | 4096 | A prefix-cache checkpoint every N prompt tokens. 0 = none. |
| `QWEN_CACHE_BETA` / `QWEN_CACHE_HALF_LIFE` / `QWEN_CACHE_SHARED_WEIGHT` | 0.3 / 64 / 2 | Eviction score, with the weight of a state a kept state extends, cache operations for the recency weight to halve, and the weight of shared prefixes. |
| `QWEN_CACHE_WEAR_S_PER_GB` / `QWEN_CACHE_WRITE_GB_DAY` | 30 / 64 | For the disk tier, the prefill seconds a write must save per GB of new bytes, and the GB written per day at most (counted across restarts). |
| `QWEN_CACHE_WAIT` | 30 | Seconds a request waits for a longer state another request is storing. |
| `QWEN_PROGRESS` | unset | `1` is the same as `--progress`. |

### Image generation

| Variable | Default | Effect |
|---|---|---|
| `QWEN_IMAGE_WEIGHTS` | `q6k` | On the CPU backend, `q6k` / `q4k` matmuls quantized at load, or `f32` (the reference, BF16 widened per use). |
| `QWEN_IMAGE_ROT` | on | On the CPU backend, weights and activations are Hadamard-rotated per 256-wide block before quantization. `0` leaves them unrotated. |
| `QWEN_IMAGE_TIMING` | unset | `1` prints per-stage times of the text encoder, transformer, VAE and the memory at each stage. |
| `QWEN_IMAGE_ATTN_THREADS` | 16 | On the NPU backend, the pool width of the attention overlapped with the NPU. `0` = all. |
| `QWEN_IMAGE_ATTN_I8` | `kv` | On the NPU backend, the attention runs on the cores in int8 (`k` = Q.K only, `0` = f32). `QWEN_IMAGE_ATTN_CHECK=1` prints its error against f32. |
| `QWEN_IMAGE_GPU_BLK` | 0.3125 | On the NPU backend, the share of each block's target chunks the iGPU runs (with a short prefix only). `0` = none. |
| `QWEN_IMAGE_PREFIX_MB` | 6144 | On the NPU backend, the room for the prefix's K / V. Past it they are int8, and further past it the text encoder, tower and VAE weights are dropped during the denoising. |
| `QWEN_IMAGE_CACHE` | off | `F,B,SPEC` enables a block cache across steps (only the first F and last B blocks run on cached steps). |
| `QWEN_IMAGE_SIGMAS` | unset | The denoising sigmas as given (an experiment that overrides `steps`). |

Further image switches (`QWEN_IMAGE_KV8_*`, `QWEN_IMAGE_ATTN_GPU`,
`QWEN_IMAGE_NANCHECK`, and the build-time `QWEN_IMAGE_FOLD` /
`QWEN_IMAGE_W4`) are described in `docs/image.md`.

### Measurement / debug (any mode)

| Variable | Effect |
|---|---|
| `QWEN_PHASE=1` | Per-token `PHASE` lines during decode, per-chunk `PREFILL` lines during prefill (gemm / ssmscan / attn / norm / elem). The first tool to reach for on any perf question. |
| `QWEN_DUMP=FILE` | Write `--forward`'s logits (raw f32). |
| `QWEN_PREFILL_DUMP=FILE` | Write the last prompt position's logits after prefill (the decode-path `QWEN_DUMP` cannot compare prefill variants). |
| `QWEN_PREFILL_CHECK` | Re-run the prefill sequentially as a reference and report the logit divergence. |
| `QWEN_VISION_TIMING=1` | Per-phase split of the vision tower (norm / qkv+rope / attn / proj / mlp + patch count). |
| `QWEN_SERVER_DEBUG=1` | Print the exact prompt and the raw completion for every `/v1/messages` request. Look at it first when a tool call does not come back. |
| `QWEN_SPARSE` | Activation sparsity (TEAL-style, measurement only), with fractions zeroed per site, e.g. `a=0.4,o=0.4,f=0.4,d=0.5` (also `all=…`). Off unless set. |
| `QWEN_SPARSE_FREQ` | (kld-eval) enable frequency-based sparsity. |
| `QWEN_KLD_BASE=FILE` | (kld-eval) cross-binary baseline that reuses the baseline's log-softmax rows from the file. |
| `QWEN_KLD_REF=exact` | (kld-eval) run the reference pass with `QWEN_EXACT_DECODE` (the noise floor of the shipped decode). |
| `QWEN_EXACT_DECODE=1` | Run every K-quant decode matmul on the CPU with an f32 activation. Slow. This is the accuracy **reference**, not a mode. |
| `QWEN_SPEC_CHECK` | Self-check speculative-decoding verify/commit paths. |
| `QWEN_TREE_LOG` | Log MTP draft tree decisions. |
| `QWEN_FEAT_DUMP=DIR` | Dump the per-position feature rows (DFlash drafter inputs) after a generate. |
| `QWEN_AVXVNNI` | An alternative AVX-VNNI GEMM form, slower than `QWEN_GEMM_VNNI`'s. |
| `QWEN_DEBUG` | Verbose load / forward diagnostics. |
| `QWEN_TOKDUMP` / `QWEN_TENSORDUMP` / `QWEN_HIDDEN` / `QWEN_LAYERINP` / `QWEN_OFFSET` | Raw dumps of token embeddings, the first 288 bytes of `token_embd`, the final normed hidden state, per-layer inputs. `QWEN_OFFSET` shifts `token_embd` reads (mmap debugging). `QWEN_RAWEMB=FILE` (dump one token's embedding row) / `QWEN_REFRAM=FILE` (correlate it with a reference row) / `QWEN_NORMW=FILE` (dump the output-norm weights) belong to the same early loader checks. |
| `QWEN_NOCACHE` / `QWEN_CACHE_MAX=N` | Disable / cap the f16 layer-weight cache (memory testing). |
| `QWEN_MROPE_CONTIGUOUS` | (debug) force the contiguous MRoPE position layout. |
