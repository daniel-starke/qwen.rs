# Vision

The model reads images through the Qwen3-VL vision tower, which has 27 transformer
blocks over square patches (the mmproj's `clip.vision.patch_size`), 2-D
RoPE, an interpolated learned position embedding, and a merger that turns
each 2x2 group of patches into one token of the text model.

The path is opt-in. On a GGUF nothing loads the tower unless `--mmproj` is
given. On the NPU engine the tower is a graph set named in the `.npu` file
(`vision = sets/vis`). The per-patch work runs on the NPU and the attention
on the iGPU or the CPU cores, and `--image`, `/image` and image blocks work
without `--mmproj`. Qwen3.8-Flash-Next has the same tower with a
2560-wide merger output, and `--npu-build` makes its set the same way. Its
n-gram embedding hashes the `<|image_pad|>` id at image positions, as the
reference implementation does.

```bash
# one answer about an image
cli MODEL.gguf --generate "Describe this image." \
    --mmproj mmproj-Qwen3.8-27B-BF16.gguf --image pic.png \
    --image-max-tokens 256 --max-new 60 --temp 0

# chat, /image FILE attaches one mid-conversation
cli MODEL.gguf --interactive --mmproj mmproj-Qwen3.8-27B-BF16.gguf

# the tower alone, without the text model; writes the pixels it saw
cli mmproj-Qwen3.8-27B-BF16.gguf --image-probe --image pic.jpg --image-dump seen.ppm
```

Over HTTP, images are content blocks (see [server.md](server.md)).

## Cost

`--image-max-tokens N` (default 1024) caps each image in *merged* tokens and
is the cost knob. An image of N tokens is 4N patches through the tower, whose
attention is O(patches²) and whose matmuls are O(patches), followed by N
tokens of text-model prefill. `QWEN_VISION_TIMING=1` prints the tower's
per-phase split. Which phase dominates depends on image size.

## Preprocessing

The image is resized so both sides are whole merge blocks (patch size x
merge size, which is one merged token) within the token cap, using the same
Pillow-compatible bicubic filter as the Hugging Face processor
(`crates/imgio/src/resize.rs`). llama.cpp resizes bilinearly, so pixel-exact agreement with llama.cpp is not expected.

The accepted formats are non-interlaced PNG (any color type and bit depth),
baseline JPEG (progressive JPEG is refused with an error), BMP, Netpbm.
mmproj files that use deepstack layers are refused rather than run without
them.

## Four invariants with no visible failure mode

If one of these is wrong, the model gives confident nonsense instead of an
error. Each is pinned by tests and called out in the source.

1. **Patch order is merge-block major**, not raster. It goes by block row, block
   column, then the 2x2 inside the block. The projector's reshape, the 2-D
   RoPE ids and the position embedding all assume this order
   (`vision::preprocess`).
2. **The patch embedding is a Conv3d with temporal patch size 2.** A still
   image is fed as the same frame twice, so the two temporal slices
   (`v.patch_embd.weight` and `.weight.1`) collapse to one 2-D kernel equal
   to their **sum**. The loader folds them at load time.
3. **The learned position embedding is interpolated with
   `linspace(0, 47, n)`**, endpoints included. A center-aligned resample
   would be off by half a cell at the edges.
4. **MRoPE positions are not sequence positions.** An image's tokens share
   one `t`, while `h` and `w` sweep the merged grid. The text after the image
   resumes one past the largest coordinate the image used. A 1024-token
   image therefore advances the sequence position by 1024 but the MRoPE
   position only by `max(grid_h, grid_w)`. The run state carries both
   positions (`pos` and `mpos`), and mixing them up is the trap.

## Prompt assembly

An image becomes `<|vision_start|>`, one `<|image_pad|>` per merged token,
and `<|vision_end|>` in the rendered prompt. The tower's output rows replace
the placeholders' embeddings. A literal `<|image_pad|>` elsewhere in the text,
such as in a quoted file, is encoded as ordinary characters, so it cannot be
mistaken for an image. Each image is encoded once per conversation because the
prefix cache keys it by a hash of its pixels.

## Checking against llama.cpp

LM Studio's prebuilt `llama-server` (with `mtmd`) runs the same GGUF and
mmproj. It is a reference for the tower's output, within the resize
difference noted above.
