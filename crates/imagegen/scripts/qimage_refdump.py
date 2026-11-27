"""Reference dumps for the imagegen gates (docs/image.md "Checks").

Runs diffusers' QwenImage21Pipeline pieces on the CPU in f32 from the BF16
checkpoint, one model in RAM at a time (text encoder, transformer, VAE), and
writes every tensor the Rust port is compared against as raw little-endian
f32 (`NAME.f32`) plus `manifest.json` (name -> shape, and the run's
settings). Needs torch, transformers >= 5, diffusers with
QwenImage21Pipeline (0.42+): ComfyUI's embedded Python has them.

    python qimage_refdump.py --model QWEN_IMAGE_DIR --out SCRATCH_DIR
        [--prompt TEXT] [--size 512] [--seed 42] [--stages te,dit,vae]
        [--image FILE ... --resolution 512]

With `--image` (repeatable) the run is an edit, as the pipeline's
`image=`: each condition image is resized to ~`--resolution`^2 pixels
(the target takes the last image's aspect, `--size` is ignored), the text
encoder sees them through its vision tower, the VAE encoder makes their
latents (stage `venc`, before `dit`), and the transformer gets them at
their slots. Stages read the previous stage's dumps from --out, so they can
be rerun one at a time. Peak RAM ~36 GB (the text encoder in f32).
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

DEFAULT_PROMPT = ("A red fox sitting in fresh snow at dawn, soft golden light, "
                  "a wooden sign beside it that reads \"QWEN.RS\"")


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


def target_hw(args, d):
    """Target size in latent tokens: --size, or the last condition image's."""
    if d.manifest.get("images"):
        tw, th = d.manifest["target"]
        return th // 16, tw // 16
    return args.size // 16, args.size // 16


def load_images(args):
    """The condition images as the pipeline prepares them: RGBA, resized to
    ~resolution^2 pixels (sides x32) with Pillow's Lanczos. Returns the
    resized PIL images, their VAE tensors, their sizes and the target size."""
    from PIL import Image
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
    proc = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64)
    res = args.resolution
    pils, vae_px, sizes = [], [], []
    for f in args.image:
        img = Image.open(f)
        img.load()
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        w, h, _ = calculate_dimensions(res * res, img.size[0] / img.size[1])
        sizes.append((w, h))
        pils.append(proc.resize(img, width=w, height=h))
        vae_px.append(proc.preprocess(img, width=w, height=h).unsqueeze(2))
    last = Image.open(args.image[-1])
    tw, th, _ = calculate_dimensions(res * res, last.size[0] / last.size[1])
    return pils, vae_px, sizes, (tw // 32 * 32, th // 32 * 32)


def stage_te(args, d):
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    from diffusers import QwenImage21Pipeline, FlowMatchEulerDiscreteScheduler

    log("te: loading processor + text encoder (f32)")
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(args.model, "processor"))
    te = Qwen3VLForConditionalGeneration.from_pretrained(
        os.path.join(args.model, "text_encoder"), dtype=torch.float32, low_cpu_mem_usage=True).eval()
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(args.model, "scheduler"))
    pipe = QwenImage21Pipeline(scheduler=sched, vae=None, text_encoder=te, processor=processor,
                               transformer=None, sample_sigmas=None)

    images = None
    hooks = []
    if args.image:
        pils, vae_px, sizes, (tw, th) = load_images(args)
        images = pils
        d.manifest["images"] = [{"file": os.path.abspath(f), "width": w, "height": h}
                                for f, (w, h) in zip(args.image, sizes)]
        d.manifest["target"] = [tw, th]
        for i, (p, v) in enumerate(zip(pils, vae_px)):
            d.put(f"ref{i}_rgba", torch.from_numpy(np.asarray(p, dtype=np.float32)))  # [h][w][4] bytes
            d.put(f"ref{i}_vae_px", v[0, :, 0])  # [4][h][w] in [-1, 1]
        vis = te.model.visual

        def vis_in(m, a, kw):
            d.put("vis_pixel_values", a[0])
            d.put("vis_grid_thw", kw["grid_thw"].float())

        def vis_out(m, a, kw, o):
            d.put("vis_embeds", o.pooler_output)
            for j, f in enumerate(o.deepstack_features):
                d.put(f"vis_deep{j}", f)
        hooks.append(vis.register_forward_pre_hook(vis_in, with_kwargs=True))
        hooks.append(vis.register_forward_hook(vis_out, with_kwargs=True))
        for j in sorted({0, 8, len(vis.blocks) - 1}):
            hooks.append(vis.blocks[j].register_forward_hook(
                lambda m, a, o, j=j: d.put(f"vis_block{j}", o[0] if isinstance(o, tuple) else o)))
    else:
        text = pipe.prompt_template_t2i.format(args.prompt)
        ids = processor(text=[text], return_tensors="pt").input_ids[0]
        d.put("te_input_ids", ids.float())
    d.manifest["drop_idx"] = pipe._drop_idx

    # every layer's output (pre final norm for the last), for bisecting the Rust port
    text_model = te.model.language_model
    layer_out = []
    hooks += [layer.register_forward_hook(lambda m, a, o: layer_out.append(o[0] if isinstance(o, tuple) else o))
              for layer in text_model.layers]

    def lm_in(m, a, kw):
        pos = kw.get("position_ids")
        if pos is not None:
            d.put("te_position_ids", pos[:, 0] if pos.dim() == 3 else pos)
        emb = kw.get("inputs_embeds")
        if emb is not None:
            d.put("te_embed", emb[0])
    hooks.append(text_model.register_forward_pre_hook(lm_in, with_kwargs=True))

    def top_in(m, a, kw):
        d.put("te_input_ids", kw["input_ids"][0].float())
    hooks.append(te.register_forward_pre_hook(top_in, with_kwargs=True))
    with torch.no_grad():
        embeds, mask, pad_mask = pipe.encode_prompt(prompt=args.prompt, image=images, device=torch.device("cpu"))
    for h in hooks:
        h.remove()
    for i in sorted({0, 1, 2, 3, len(layer_out) // 2, len(layer_out) - 2, len(layer_out) - 1}):
        d.put(f"te_layer{i}", layer_out[i][0])
    d.put("te_out", embeds[0])
    d.put("te_image_pad_mask", pad_mask[0].float())
    d.manifest["te_mask_none"] = mask is None
    d.manifest["te_image_pad"] = int(pad_mask.sum())
    log(f"te: out {tuple(embeds.shape)}, {int(pad_mask.sum())} image slots")


def stage_venc(args, d):
    from diffusers import AutoencoderKLQwenImage21

    log("venc: loading VAE (f32)")
    vae = AutoencoderKLQwenImage21.from_pretrained(os.path.join(args.model, "vae"), torch_dtype=torch.float32).eval()
    enc = vae.encoder
    mean = torch.tensor(vae.config.latents_mean).view(1, 64, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, 64, 1, 1, 1)
    for i in range(len(d.manifest["images"])):
        x = d.get(f"ref{i}_vae_px").unsqueeze(0).unsqueeze(2)
        hooks = []
        for name, mod in enc.named_children():
            if isinstance(mod, torch.nn.ModuleList):
                for j, sub in enumerate(mod):
                    hooks.append(sub.register_forward_hook(
                        lambda m, a, o, n=f"ref{i}_venc_{name}{j}":
                            d.put(n, (o[0] if isinstance(o, tuple) else o)[0, :, -1])))
            elif sum(p.numel() for p in mod.parameters()) > 0:
                hooks.append(mod.register_forward_hook(
                    lambda m, a, o, n=f"ref{i}_venc_{name}":
                        d.put(n, (o[0] if isinstance(o, tuple) else o)[0, :, -1])))
        t0 = time.time()
        with torch.no_grad():
            lat = vae.encode(x).latent_dist.mode()
        for hk in hooks:
            hk.remove()
        d.put(f"ref{i}_venc_mean", lat[0, :, 0])
        lat = (lat - mean) / std
        _, c, _, lh, lw = lat.shape
        d.put(f"ref{i}_latents", lat.reshape(1, c, lh * lw).transpose(1, 2)[0])  # (h*w, 64), token-major
        log(f"venc: image {i} -> latents {lh}x{lw} in {time.time() - t0:.1f} s")


def stage_dit(args, d):
    from diffusers import QwenImage21Transformer2DModel, FlowMatchEulerDiscreteScheduler
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import retrieve_timesteps, calculate_shift

    embeds = d.get("te_out").unsqueeze(0)
    h, w = target_hw(args, d)
    refs = d.manifest.get("images") or []
    log("dit: loading transformer (f32)")
    dit = QwenImage21Transformer2DModel.from_pretrained(
        os.path.join(args.model, "transformer"), torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(args.model, "scheduler"))
    sigmas = json.load(open(os.path.join(args.model, "model_index.json"), encoding="utf-8"))["sample_sigmas"]

    # the pipeline's randn_tensor on the CPU with a seeded generator, packed
    g = torch.Generator("cpu").manual_seed(args.seed)
    noise = torch.randn((1, 1, 64, h, w), generator=g, dtype=torch.float32)
    latents = noise.view(1, 64, h * w).transpose(1, 2)
    d.put("noise", latents[0])  # (h*w, 64), token-major

    mu = calculate_shift(h * w, sched.config.get("base_image_seq_len", 256), sched.config.get("max_image_seq_len", 4096),
                         sched.config.get("base_shift", 0.5), sched.config.get("max_shift", 1.15))
    timesteps, _ = retrieve_timesteps(sched, None, "cpu", sigmas=sigmas, mu=mu)
    d.put("sched_sigmas", sched.sigmas)
    d.put("sched_timesteps", timesteps)

    if refs:
        pad = d.get("te_image_pad_mask").bool().unsqueeze(0)
        ref_lat = torch.cat([d.get(f"ref{i}_latents") for i in range(len(refs))], dim=0).unsqueeze(0)
    else:
        pad = torch.zeros(1, embeds.shape[1], dtype=torch.bool)
        ref_lat = None
    img_mask = torch.cat([pad, torch.ones(1, h * w // 4, dtype=torch.bool)], dim=1)
    img_shapes = [[*[(1, r["height"] // 16, r["width"] // 16) for r in refs], (1, h, w)]]
    cache = QwenImage21KVCache(len(dit.transformer_blocks))

    # step-0 internals
    step = {"i": 0}

    def keep(name):
        def f(m, a, o):
            if step["i"] == 0:
                d.put(name, o[0] if o.dim() == 3 else o)
        return f
    hooks = [dit.txt_in.register_forward_hook(keep("dit_txt_in")),
             dit.img_in.register_forward_hook(keep("dit_img_in")),
             dit.time_text_embed.register_forward_hook(keep("dit_temb")),
             dit.modulation.register_forward_hook(keep("dit_modulation"))]
    for i, b in enumerate(dit.transformer_blocks):
        hooks.append(b.register_forward_hook(keep(f"dit_block{i}")))
    hooks.append(dit.norm_out.register_forward_hook(keep("dit_norm_out")))

    def rope_out(m, a, o):
        if step["i"] == 0:
            d.put("dit_rope_cos", o.real)
            d.put("dit_rope_sin", o.imag)
    hooks.append(dit.pos_embed.register_forward_hook(rope_out))

    sched.set_begin_index(0)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            step["i"] = i
            t0 = time.time()
            mode = "extract" if i == 0 else "cached"
            inp = latents if ref_lat is None else torch.cat([ref_lat, latents], dim=1)
            v = dit(hidden_states=inp, timestep=t.expand(1).to(latents.dtype) / 1000,
                    encoder_hidden_states=embeds, encoder_hidden_states_mask=None, img_shapes=img_shapes,
                    img_mask=img_mask, kv_cache=cache, kv_cache_mode=mode, return_dict=False)[0]
            v = v[:, -latents.size(1):]
            d.put(f"dit_v{i}", v[0])
            latents = sched.step(v, t, latents, return_dict=False)[0]
            d.put(f"latents{i + 1}", latents[0])
            log(f"dit: step {i} t {float(t):.4f} {time.time() - t0:.1f} s")
            d.save()
    for hk in hooks:
        hk.remove()
    d.put("latents_final", latents[0])


def stage_vae(args, d):
    from diffusers import AutoencoderKLQwenImage21
    from diffusers.image_processor import VaeImageProcessor

    h, w = target_hw(args, d)
    log("vae: loading (f32)")
    vae = AutoencoderKLQwenImage21.from_pretrained(os.path.join(args.model, "vae"), torch_dtype=torch.float32).eval()
    lat = d.get("latents_final")  # (h*w, 64)
    lat = lat.transpose(0, 1).reshape(1, 64, 1, h, w)
    mean = torch.tensor(vae.config.latents_mean).view(1, 64, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, 64, 1, 1, 1)
    z = lat * std + mean
    d.put("vae_z", z[0, :, 0])

    hooks = []
    dec = vae.decoder
    for name, mod in dec.named_children():
        if isinstance(mod, torch.nn.ModuleList):
            for j, sub in enumerate(mod):
                hooks.append(sub.register_forward_hook(
                    lambda m, a, o, n=f"vae_dec_{name}{j}": d.put(n, (o[0] if isinstance(o, tuple) else o)[0, :, -1])))
        elif sum(p.numel() for p in mod.parameters()) > 0:
            hooks.append(mod.register_forward_hook(
                lambda m, a, o, n=f"vae_dec_{name}": d.put(n, (o[0] if isinstance(o, tuple) else o)[0, :, -1])))
    t0 = time.time()
    with torch.no_grad():
        img = vae.decode(z, return_dict=False)[0][:, :, 0]
    for hk in hooks:
        hk.remove()
    log(f"vae: decoded {tuple(img.shape)} in {time.time() - t0:.1f} s")
    d.put("vae_out", img[0])
    proc = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64)
    pil = proc.postprocess(img, output_type="pil")[0]
    pil.save(os.path.join(args.out, "ref.png"))
    log(f"vae: wrote ref.png ({pil.mode} {pil.size})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Qwen-Image-2.1 checkpoint directory (diffusers layout)")
    ap.add_argument("--out", required=True, help="scratch directory for the dumps")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stages", default=None, help="default te,dit,vae (te,venc,dit,vae with --image)")
    ap.add_argument("--image", action="append", default=[], help="condition image (repeatable): an edit")
    ap.add_argument("--resolution", type=int, default=512,
                    help="with --image: the pipeline's output_resolution (images and target ~N^2 pixels)")
    args = ap.parse_args()
    if args.size % 32:
        sys.exit("--size must be a multiple of 32")
    torch.manual_seed(0)
    d = Dumper(args.out)
    d.manifest.update(prompt=args.prompt, size=args.size, seed=args.seed)
    if args.image:
        d.manifest["resolution"] = args.resolution
    stages = args.stages or ("te,venc,dit,vae" if args.image else "te,dit,vae")
    for s in stages.split(","):
        {"te": stage_te, "venc": stage_venc, "dit": stage_dit, "vae": stage_vae}[s](args, d)
        d.save()
        log(f"stage {s} done")


if __name__ == "__main__":
    main()
