"""Per-layer reference dumps for the MiniMax-H3-Image gates (docs/h3image.md).

Runs diffusers' MiniMax-H3-Image pieces on the CPU in f32 from the BF16
checkpoint, one layer in RAM at a time, and writes every tensor the Rust
port is compared against as raw little-endian f32 (`NAME.f32`) plus
`manifest.json` (name -> shape, and the run's settings).

The models are built on the meta device from the official classes
(transformers' Qwen3VLModel, diffusers' MiniMaxH3ImageTransformer2DModel and
AutoencoderKLMiniMaxH3Image), and each decoder layer / transformer block is
materialized in f32 from the safetensors shards right before it runs and
dropped right after, so the official forward code runs unchanged while RSS
stays at one layer plus activations (the text encoder alone is ~98 GB in
f32, the transformer ~132 GB). Every layer's output is a checkpoint: a gate
can feed a recorded input to one layer and compare its output.

Needs torch, transformers >= 5 (Qwen3-VL) and a diffusers with
MiniMaxH3ImageTransformer2DModel (main after 0.42.1): put such a checkout
first on PYTHONPATH, or name it with `--pylib DIR` (an embedded Python with a
`._pth` file ignores PYTHONPATH).

    python h3refdump.py --model H3_DIR --out DIR [--prompt TEXT | --prompt-file F]
        [--size 512 | --width W --height H] [--min-area 1024] [--seed 42]
        [--stages te,dit,vae] [--image FILE ...] [--blocks all|first-last]

`--min-area` is the pipeline's `min_generation_area` (default 1024 here, so
`--size` is the canvas; the pipeline's own default is 2048 * 2048). With
`--image` (repeatable) the run is an edit: stage `venc` runs before `dit`
and the request's generator is carried between the stages
(`gen_state.bin`), in the pipeline's order: VAE posterior samples, target
noise, reference noise.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

DEFAULT_PROMPT = (
    "A photograph of a red fox sitting in fresh snow at dawn in a quiet pine forest. Soft golden light comes "
    "from the low sun on the left and rims the fox's fur, the snow sparkles, and long blue shadows stretch to the "
    "right. Beside the fox stands a small weathered wooden sign with carved letters that read \"QWEN.RS\". Shallow "
    "depth of field, the trees in the background are softly blurred."
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Dumper:
    def __init__(self, out):
        self.out = out
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, "manifest.json")
        self.manifest = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {"tensors": {}}

    def put(self, name, t):
        a = t.detach().float().cpu().contiguous().numpy() if torch.is_tensor(t) else np.asarray(t, dtype=np.float32)
        a.astype("<f4").tofile(os.path.join(self.out, name + ".f32"))
        self.manifest["tensors"][name] = list(a.shape)

    def get(self, name):
        shape = self.manifest["tensors"][name]
        a = np.fromfile(os.path.join(self.out, name + ".f32"), dtype="<f4").reshape(shape)
        return torch.from_numpy(a.copy())

    def save(self):
        with open(os.path.join(self.out, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(self.manifest, f, indent=1)

    def put_gen(self, g):
        g.get_state().numpy().tofile(os.path.join(self.out, "gen_state.bin"))

    def get_gen(self):
        g = torch.Generator("cpu")
        state = np.fromfile(os.path.join(self.out, "gen_state.bin"), dtype=np.uint8)
        g.set_state(torch.from_numpy(state))
        return g


class Weights:
    """BF16/F32 tensors of one safetensors directory, read on demand as f32."""

    def __init__(self, folder):
        from safetensors import safe_open
        self.folder = folder
        idx = [f for f in os.listdir(folder) if f.endswith(".safetensors.index.json")]
        if idx:
            self.map = json.load(open(os.path.join(folder, idx[0]), encoding="utf-8"))["weight_map"]
        else:
            single = [f for f in os.listdir(folder) if f.endswith(".safetensors")][0]
            with safe_open(os.path.join(folder, single), "pt") as f:
                self.map = {k: single for k in f.keys()}
        self.files = {}
        self.open = safe_open

    def get(self, name):
        shard = self.map[name]
        if shard not in self.files:
            self.files[shard] = self.open(os.path.join(self.folder, shard), "pt")
        return self.files[shard].get_tensor(name).float()

    def materialize(self, mod, prefix):
        """Allocate `mod`'s parameters on the CPU in f32 and load them from `prefix`."""
        mod.to_empty(device="cpu")
        mod.to(torch.float32)
        sd = {k: self.get(prefix + k) for k in mod.state_dict().keys()}
        mod.load_state_dict(sd, strict=True)

    @staticmethod
    def drop(mod):
        mod.to("meta")


class Done(Exception):
    pass


def canvas(args, d):
    """(height, width) of the canvas, as MiniMaxH3ImagePrepareCanvasStep."""
    from diffusers.modular_pipelines.minimax_h3_image.before_denoise import get_image_canvas_size
    h = args.height or args.size
    w = args.width or args.size
    ch, cw = get_image_canvas_size(h, w, args.min_area, 32)
    d.manifest.update(output_height=h, output_width=w, height=ch, width=cw)
    return ch, cw


def load_refs(args, d):
    """The edit's references as MiniMaxH3ImageEditResizeStep prepares them."""
    from PIL import Image, ImageOps
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.modular_pipelines.minimax_h3_image.before_encoder import bound_short_edge
    proc = VaeImageProcessor(vae_scale_factor=16, do_normalize=False)

    def resize(img, lo, hi):
        size = bound_short_edge(*img.size, lo, hi, 32)
        return img if img.size == size else proc.resize(img, height=size[1], width=size[0])

    refs, vis = [], []
    for f in args.image:
        img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        refs.append(resize(img, 1024, 2048))
        vis.append(resize(img, 1024, 1024))
    d.manifest["images"] = [{"file": os.path.abspath(f), "vae": list(r.size), "vision": list(v.size)}
                            for f, r, v in zip(args.image, refs, vis)]
    return refs, vis


def stage_te(args, d):
    from transformers import AutoConfig, Qwen2TokenizerFast, Qwen3VLProcessor, AutoModel
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    te_dir = os.path.join(args.model, "text_encoder")
    W = Weights(te_dir)
    tok = Qwen2TokenizerFast.from_pretrained(os.path.join(args.model, "tokenizer"))
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(args.model, "processor"))
    cfg = AutoConfig.from_pretrained(te_dir)
    layer = 50  # MiniMaxH3ImageModularPipeline.text_encoder_layer: hidden_states[50]

    # the presentation (encoders.py): no chat template, no special tokens
    vision = None
    prompt_ids = tok(args.prompt, add_special_tokens=False)["input_ids"][:4096]
    if args.image:
        _, vis_imgs = load_refs(args, d)
        vision = processor.image_processor(images=vis_imgs, return_tensors="pt")
        merge = processor.image_processor.merge_size ** 2
        ids, tags = [], []
        for i, grid in enumerate(vision["image_grid_thw"]):
            label = tok(f"Picture {i + 1}: ", add_special_tokens=False)["input_ids"][:4096]
            vids = ([tok.convert_tokens_to_ids("<|vision_start|>")]
                    + [tok.convert_tokens_to_ids("<|image_pad|>")] * (int(grid.prod()) // merge)
                    + [tok.convert_tokens_to_ids("<|vision_end|>")])
            ids += label + vids
            tags += [1] * len(label) + [0] * len(vids)
        ids += prompt_ids
        tags += [1] * len(prompt_ids)
        d.put("vis_pixel_values", vision["pixel_values"])
        d.put("vis_grid_thw", vision["image_grid_thw"].float())
    else:
        ids, tags = prompt_ids, [1] * len(prompt_ids)
    d.put("te_input_ids", torch.tensor(ids, dtype=torch.float32))
    d.put("te_token_tags", torch.tensor(tags, dtype=torch.float32))
    log(f"te: {len(ids)} tokens ({len(prompt_ids)} prompt)")

    with torch.device("meta"):
        model = Qwen3VLModel(cfg)
    lm = model.language_model
    lm.rotary_emb = type(lm.rotary_emb)(config=cfg.text_config)
    W.materialize(lm.embed_tokens, "model.language_model.embed_tokens.")
    if vision is not None:
        visual = AutoModel.from_config(cfg.vision_config, dtype=torch.float32)
        visual.load_state_dict({k: W.get("model.visual." + k) for k in visual.state_dict().keys()}, strict=True)
        model.visual = visual.eval()

        def vis_out(m, a, kw, o):
            d.put("vis_embeds", torch.cat(list(o.pooler_output), dim=0) if isinstance(o.pooler_output, (list, tuple))
                  else o.pooler_output)
            for j, f in enumerate(o.deepstack_features):
                d.put(f"vis_deep{j}", torch.cat(list(f), dim=0) if isinstance(f, (list, tuple)) else f)
        visual.register_forward_hook(vis_out, with_kwargs=True)
        for j in sorted({0, 8, len(visual.blocks) - 1}):
            visual.blocks[j].register_forward_hook(
                lambda m, a, o, j=j: d.put(f"vis_block{j}", o[0] if isinstance(o, tuple) else o))

    def lm_in(m, a, kw):
        pos = kw.get("position_ids")
        if pos is not None:
            d.put("te_position_ids", pos[:, 0] if pos.dim() == 3 else pos)
        emb = kw.get("inputs_embeds")
        if emb is not None:
            d.put("te_embed", emb[0])
    lm.register_forward_pre_hook(lm_in, with_kwargs=True)

    out = {}
    for i, dec in enumerate(lm.layers):
        def pre(m, a, kw, i=i):
            if i >= layer:
                raise Done()
            t0 = time.time()
            W.materialize(m, f"model.language_model.layers.{i}.")
            m.eval()
            out["t_load"] = time.time() - t0

        def post(m, a, kw, o, i=i):
            h = o[0] if isinstance(o, tuple) else o
            d.put(f"te_layer{i}", h[0])
            out["last"] = h[0].clone()
            Weights.drop(m)
            log(f"te: layer {i} (load {out['t_load']:.1f} s)")
        dec.register_forward_pre_hook(pre, with_kwargs=True)
        dec.register_forward_hook(post, with_kwargs=True)

    input_ids = torch.tensor([ids], dtype=torch.long)
    mm = torch.tensor(processor.create_mm_token_type_ids([ids]), dtype=torch.long)
    kw = {}
    if vision is not None:
        kw = {"pixel_values": vision["pixel_values"].float(), "image_grid_thw": vision["image_grid_thw"]}
    t0 = time.time()
    with torch.no_grad():
        try:
            model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), mm_token_type_ids=mm,
                  use_cache=False, **kw)
        except Done:
            pass
    # hidden_states[50] is decoder layer 49's output: not normalized, after the deepstack layers (0-2)
    d.put("te_out", out["last"])
    log(f"te: out {tuple(out['last'].shape)} in {time.time() - t0:.0f} s")


def stage_venc(args, d):
    from diffusers import AutoencoderKLMiniMaxH3Image

    refs, _ = load_refs(args, d)
    log("venc: loading VAE (f32)")
    vae = AutoencoderKLMiniMaxH3Image.from_pretrained(os.path.join(args.model, "vae_diffusers"),
                                                     torch_dtype=torch.float32).eval()
    g = torch.Generator("cpu").manual_seed(args.seed)
    mean_px = torch.tensor((0.485, 0.456, 0.406)).view(1, -1, 1, 1)
    std_px = torch.tensor((0.229, 0.224, 0.225)).view(1, -1, 1, 1)
    lmean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1)
    lstd = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1)
    enc = vae.encoder
    for i, img in enumerate(refs):
        px = torch.from_numpy(np.array(img)).permute(2, 0, 1)[None]
        px = px[..., : px.shape[-2] // 16 * 16, : px.shape[-1] // 16 * 16]
        d.put(f"ref{i}_rgb", px[0].float())  # [3][h][w] bytes
        px = (px.float() / 255.0 - mean_px) / std_px
        first = {"tile": True}
        hooks = []

        def keep(name):
            def f(m, a, o):
                if first["tile"]:
                    d.put(f"ref{i}_venc_{name}", o[0, :, -1])
            return f
        hooks.append(enc.conv_in.register_forward_hook(keep("conv_in")))
        for j, blk in enumerate(enc.down_blocks):
            hooks.append(blk.register_forward_hook(keep(f"down{j}")))
        hooks.append(enc.conv_out.register_forward_hook(keep("conv_out")))
        hooks.append(enc.register_forward_hook(lambda m, a, o: first.update(tile=False)))
        t0 = time.time()
        with torch.no_grad():
            post = vae.encode(px, return_dict=False)[0]
            lat = post.sample(generator=g).to(torch.float16).float()
        for hk in hooks:
            hk.remove()
        d.put(f"ref{i}_moments", post.parameters[0])
        lat = lat[..., : lat.shape[-2] // 2 * 2, : lat.shape[-1] // 2 * 2]
        lat = (lat - lmean) / lstd
        d.put(f"ref{i}_latents", lat[0])  # [24][lh][lw]
        log(f"venc: image {i} {img.size} -> latents {tuple(lat.shape[-2:])} in {time.time() - t0:.1f} s")
    d.put_gen(g)


def stage_dit(args, d):
    from diffusers import MiniMaxH3ImageTransformer2DModel, MiniMaxH3Scheduler
    from diffusers.modular_pipelines.minimax_h3_image.before_denoise import (
        build_image_packed_sequence, patchify_video_latents)

    dit_dir = os.path.join(args.model, "transformer_diffusers")
    W = Weights(dit_dir)
    ch, cw = canvas(args, d)
    lh, lw = ch // 16, cw // 16
    nref = len(d.manifest.get("images") or [])
    g = d.get_gen() if nref else torch.Generator("cpu").manual_seed(args.seed)
    embeds = d.get("te_out")[None]
    tags = d.get("te_token_tags").long()
    refs = [d.get(f"ref{i}_latents")[None] for i in range(nref)]

    pos, token_tags, image_idx, text_idx, nref_rows = build_image_packed_sequence(
        tags, [tuple(r.shape[-2:]) for r in refs], lh, lw, (1, 2, 2), 0)
    d.put("dit_position_ids", pos)
    d.put("dit_token_tags", token_tags.float())
    d.manifest.update(dit_text_rows=int(text_idx.numel()), dit_ref_rows=int(nref_rows),
                      dit_target_rows=int(image_idx.numel()) - int(nref_rows))

    noise = torch.randn((1, 24, 1, lh, lw), generator=g, dtype=torch.float32)
    latents = patchify_video_latents(noise, (1, 2, 2))
    d.put("noise", latents)  # (rows, 96), row-major over the patch grid
    if nref:
        sched0 = MiniMaxH3Scheduler.from_pretrained(os.path.join(args.model, "scheduler"))
        rows = []
        for r in refs:
            n = torch.randn(r.shape, generator=g, dtype=torch.float32)
            noised = sched0.scale_noise(r, torch.tensor(0.999), n)
            rows.append(patchify_video_latents(noised[:, :, None], (1, 2, 2)))
        rows = torch.cat(rows)
        d.put("dit_ref_rows", rows)
        latents = torch.cat([rows, latents])

    sched = MiniMaxH3Scheduler.from_pretrained(os.path.join(args.model, "scheduler"))
    sigmas_cfg = json.load(open(os.path.join(args.model, "modular_model_index.json"), encoding="utf-8"))["sample_sigmas"]
    ts = torch.tensor([1.0 - s for s in sigmas_cfg[:-1]], dtype=torch.float32)
    sched.set_timesteps(sigmas=torch.cat([1.0 - ts, torch.zeros(1)]))
    d.put("sched_sigmas", sched.sigmas)
    d.put("sched_timesteps", sched.timesteps)
    seq = int(image_idx.numel() + text_idx.numel())
    plan = []
    for t in sched.timesteps:
        row_t = torch.full((seq,), float(t), dtype=torch.float32)
        row_t[image_idx[:nref_rows]] = 0.999
        plan.append(torch.unique(row_t, sorted=True, return_inverse=True))

    log("dit: building transformer on meta")
    cfg = MiniMaxH3ImageTransformer2DModel.load_config(dit_dir)
    with torch.device("meta"):
        dit = MiniMaxH3ImageTransformer2DModel.from_config(cfg)
    dit.rope = type(dit.rope)(rope_freq_dim=dit.config.rope_freq_dim, rope_theta=dit.config.rope_theta)
    for name in ("proj_in", "context_embedder", "time_embedder", "token_refiner", "norm_out", "pdd_proj_out"):
        W.materialize(getattr(dit, name), name + ".")
    dit.eval()

    step = {"i": 0}
    nsteps = len(sched.timesteps)
    all_blocks = args.blocks == "all"

    def keep_block(L):
        return all_blocks or step["i"] in (0, nsteps - 1) or L in (0, 24, 49)

    def once(name):
        def f(m, a, o):
            if step["i"] == 0:
                d.put(name, o[0] if o.dim() == 3 else o)
        return f
    dit.context_embedder.register_forward_hook(once("dit_ctx"))
    for j, b in enumerate(dit.token_refiner.refiner_blocks):
        b.register_forward_hook(once(f"dit_refiner{j}"))
    dit.token_refiner.register_forward_hook(once("dit_txt"))
    dit.proj_in.register_forward_hook(
        lambda m, a, o: d.put(f"dit_img_in{step['i']}", o[0]) if step["i"] in (0, nsteps - 1) else None)
    dit.time_embedder.register_forward_hook(lambda m, a, o: d.put(f"dit_temb{step['i']}", o))
    def rope_out(m, a, o):
        # a forward hook's non-None return replaces the output: return nothing
        if step["i"] == 0:
            d.put("dit_rope_cos", o[0])
            d.put("dit_rope_sin", o[1])
    dit.rope.register_forward_hook(rope_out)
    dit.norm_out.register_forward_hook(lambda m, a, kw, o: d.put(f"dit_norm_out{step['i']}", o[0])
                                       if step["i"] in (0, nsteps - 1) else None, with_kwargs=True)

    tl = {}
    for L, blk in enumerate(dit.transformer_blocks):
        def pre(m, a, kw, L=L):
            t0 = time.time()
            W.materialize(m, f"transformer_blocks.{L}.")
            m.eval()
            tl["load"] = tl.get("load", 0.0) + time.time() - t0
            if L == 0 and step["i"] == 0:
                d.put("dit_x_in", a[0][0])  # the packed sequence entering block 0

        def post(m, a, kw, o, L=L):
            if keep_block(L):
                d.put(f"dit_s{step['i']}_block{L}", o[0])
            Weights.drop(m)
        blk.register_forward_pre_hook(pre, with_kwargs=True)
        blk.register_forward_hook(post, with_kwargs=True)
        # the block's AdaLN table (rows [t0_mod0, t0_mod1, t0_mod2, t1_mod0, ...]) for the first / last block
        if L in (0, 49):
            blk.adaln_proj.register_forward_hook(
                lambda m, a, o, L=L: d.put(f"dit_s{step['i']}_adaln{L}", torch.stack(o, dim=1))
                if step["i"] in (0, nsteps - 1) else None)

    with torch.no_grad():
        for i, t in enumerate(sched.timesteps):
            step["i"] = i
            tl.clear()
            t0 = time.time()
            uniq, tidx = plan[i]
            v = dit(hidden_states=latents[None], encoder_hidden_states=embeds, timestep=uniq, timestep_indices=tidx,
                    token_tags=token_tags, position_ids=pos.to(torch.float32), image_indices=image_idx,
                    text_indices=text_idx, pdd_step_index=i, return_dict=False)[0]
            d.put(f"dit_v{i}", v[0, nref_rows:])
            latents[nref_rows:] = sched.step(v[0, nref_rows:].float(), t, latents[nref_rows:], return_dict=False)[0]
            d.put(f"latents{i + 1}", latents[nref_rows:])
            log(f"dit: step {i} t {float(t):.6f} {time.time() - t0:.0f} s (weights {tl.get('load', 0):.0f} s)")
            d.save()
    rows = latents[nref_rows:].reshape(1, lh // 2, lw // 2, 24, 2, 2).permute(0, 3, 1, 4, 2, 5)
    d.put("latents_final", rows.reshape(24, lh, lw))


def stage_vae(args, d):
    from diffusers import AutoencoderKLMiniMaxH3Image
    from diffusers.image_processor import VaeImageProcessor

    log("vae: loading (f32)")
    vae = AutoencoderKLMiniMaxH3Image.from_pretrained(os.path.join(args.model, "vae_diffusers"),
                                                     torch_dtype=torch.float32).eval()
    lat = d.get("latents_final")[None]
    lmean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1)
    lstd = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1)
    z = lat * lstd + lmean
    d.put("vae_z", z[0])

    dec = vae.decoder
    first = {"tile": True}
    hooks = []

    def keep(name):
        def f(m, a, o):
            if first["tile"]:
                d.put(f"vae_tile0_{name}", o[0])
        return f
    hooks.append(vae.post_quant_conv.register_forward_hook(keep("post_quant")))
    hooks.append(dec.proj_in.register_forward_hook(keep("proj_in")))
    for j in range(len(dec.transformer_blocks)):
        hooks.append(dec.transformer_blocks[j].register_forward_hook(keep(f"block{j}")))
    hooks.append(dec.norm_out.register_forward_hook(keep("norm_out")))
    hooks.append(dec.register_forward_hook(keep("out")))
    hooks.append(dec.register_forward_hook(lambda m, a, o: first.update(tile=False)))
    t0 = time.time()
    with torch.no_grad():
        img = vae.decode(z, return_dict=False)[0]
    for hk in hooks:
        hk.remove()
    log(f"vae: decoded {tuple(img.shape)} in {time.time() - t0:.1f} s")
    d.put("vae_out", img[0])

    # MiniMaxH3ImageDecodeStep: ImageNet de-normalization, 8-bit truncation, Lanczos back to the requested size
    mean_px = torch.tensor((0.485, 0.456, 0.406)).view(1, -1, 1, 1)
    std_px = torch.tensor((0.229, 0.224, 0.225)).view(1, -1, 1, 1)
    img = (img.float() * std_px + mean_px).clamp(0, 1)
    img = (img * 255).clamp(0, 255).to(torch.uint8).float() / 255
    proc = VaeImageProcessor(vae_scale_factor=16, do_normalize=False)
    oh, ow = d.manifest["output_height"], d.manifest["output_width"]
    if (oh, ow) != tuple(img.shape[-2:]):
        pil = proc.postprocess(img, output_type="pil")
        pil = [proc.resize(p, oh, ow) for p in pil]
        img = proc.numpy_to_pt(proc.pil_to_numpy(pil))
    d.put("image", img[0] * 255)
    pil = proc.postprocess(img, output_type="pil")[0]
    pil.save(os.path.join(args.out, "ref.png"))
    log(f"vae: wrote ref.png ({pil.mode} {pil.size})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="MiniMax-H3-Image directory (diffusers layout)")
    ap.add_argument("--out", required=True, help="directory for the dumps")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--min-area", type=int, default=1024, help="the pipeline's min_generation_area")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stages", default=None, help="default te,dit,vae (te,venc,dit,vae with --image)")
    ap.add_argument("--image", action="append", default=[], help="reference image (repeatable): an edit")
    ap.add_argument("--pylib", default=None, help="directory put first on sys.path (a diffusers checkout)")
    ap.add_argument("--blocks", default="first-last",
                    help="first-last: every block of the first and last step, blocks 0/24/49 of the others; all")
    args = ap.parse_args()
    if args.pylib:
        sys.path.insert(0, args.pylib)
    if args.prompt_file:
        args.prompt = open(args.prompt_file, encoding="utf-8").read().strip()
    args.prompt = args.prompt or DEFAULT_PROMPT
    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count() or 1)
    d = Dumper(args.out)
    d.manifest.update(prompt=args.prompt, seed=args.seed, min_area=args.min_area)
    stages = args.stages or ("te,venc,dit,vae" if args.image else "te,dit,vae")
    for s in stages.split(","):
        {"te": stage_te, "venc": stage_venc, "dit": stage_dit, "vae": stage_vae}[s](args, d)
        d.save()
        log(f"stage {s} done")


if __name__ == "__main__":
    main()
