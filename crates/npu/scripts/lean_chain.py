"""Teacher-forced prefill through the lean per-layer graphs (`build_delta.py`
/ `build_attn.py` output, DIR/lean{L}.xml, one S-position chunk, zero initial
state) on the CPU (exact f32; FakeQuantize simulates the int8 activations) or
any OpenVINO device, then the decode head on every position: the
end-to-end accuracy gate of the lean prefill blobs (compare with `--ref`,
the exact chain of the original cut layers, `chain.py`).

    python lean_chain.py DIR IDS OUT --head DIR [--ref REF.f32] [--device CPU]
                         [--save-inputs INDIR --cap 256] [--resid FILE]

`--save-inputs INDIR`: write the chunk's inputs (x = the embeddings, mask
for attention graphs of `--cap`, cos, sin, valid, keep; raw f16 / i32) for
`npubench prefill DIR --inputs INDIR`, and stop. `--resid FILE`: skip the
layers and run the head on that final residual (raw f16 [S, 5120], from
`npubench prefill --dump FILE`).

Layers L % 4 == 3 are full attention (cap = whatever the graph was built
with: past rows all masked), the rest delta-net."""
import os
import sys
import time
import numpy as np
import openvino as ov

a = sys.argv[1:]


def opt(name, d, conv=str):
    if name in a:
        i = a.index(name)
        v = a[i + 1]
        del a[i:i + 2]
        return conv(v)
    return d


export = opt("--export", os.environ.get("OV_EXPORT")) or sys.exit("--export DIR or OV_EXPORT: the OpenVINO export")
ref = opt("--ref", None)
device = opt("--device", "CPU")
headdir = opt("--head", None)
nl = opt("--layers", 64, int)
save_in = opt("--save-inputs", None)
cap = opt("--cap", 256, int)
resid = opt("--resid", None)
# `--cmp-layers DIR`: compare each layer's output with DIR/res{L}.f16
# (`npubench prefill --dump-layers DIR`), relative error per layer.
cmpd = opt("--cmp-layers", None)
d, idsf, out = a
ids = np.array([int(v) for v in open(idsf).read().split()], np.int64)
S, W = len(ids), 5120
core = ov.Core()
EXACT = {"INFERENCE_PRECISION_HINT": "f32"}
emb = core.compile_model(core.read_model(export + r"\openvino_text_embeddings_model.xml"), "CPU")
x = list(emb({0: ids[None, :]}).values())[0].reshape(1, S, W).astype(np.float32)
# rotary tables (theta 1e7, 64 rotary dims, text positions 0..S-1)
inv = 1e7 ** (-np.arange(0, 64, 2) / 64.0)
ang = np.arange(S)[:, None] * inv[None, :]
ang = np.concatenate([ang, ang], 1).reshape(1, S, 64)

if save_in:
    os.makedirs(save_in, exist_ok=True)
    mask = np.full([1, 1, S, cap + S], -30000.0, np.float32)
    mask[0, 0][:, cap:] = np.where(np.tril(np.ones((S, S), bool)), 0.0, -30000.0)
    for nm, v in [("x", x), ("mask", mask), ("cos", np.cos(ang)), ("sin", np.sin(ang)), ("valid", np.ones([1, S, 1]))]:
        v.astype(np.float16).tofile(os.path.join(save_in, nm + ".in"))
    np.arange(S, S + 3, dtype=np.int32).tofile(os.path.join(save_in, "keep.in"))
    raise SystemExit(f"inputs -> {save_in}")
if resid:
    x = np.fromfile(resid, np.float16).astype(np.float32).reshape(1, S, W)
    nl = 0
t0 = time.time()
for L in range(nl):
    cm = core.compile_model(core.read_model(f"{d}\\lean{L}.xml"), device, EXACT if device == "CPU" else {})
    f = {}
    for p in cm.inputs:
        nm = p.get_any_name()
        shp = list(p.get_shape())
        own = f"{d}/lean{L}.{nm}.in"  # weights / per-layer constants (--wai, --allin)
        if nm != "x" and os.path.exists(own):
            if p.get_element_type() == ov.Type.i4:
                # packed nibbles (`wq.pack4`)
                t = ov.Tensor(ov.Type.i4, ov.Shape(shp))
                np.copyto(np.asarray(t.data).reshape(-1), np.fromfile(own, np.uint8))
                f[nm] = t
                continue
            dt = {ov.Type.i8: np.int8, ov.Type.f16: np.float16, ov.Type.f32: np.float32}[p.get_element_type()]
            f[nm] = np.fromfile(own, dt).reshape(shp)
            continue
        if nm == "x":
            v = x
        elif nm in ("conv", "ssm", "kpast", "vpast"):
            v = np.zeros(shp, np.float32)
        elif nm == "valid":
            v = np.ones(shp, np.float32)
        elif nm == "keep":
            v = np.arange(S, S + 3, dtype=np.int32)
        elif nm == "mask":
            cap = shp[-1] - S
            v = np.full(shp, -30000.0, np.float32)
            v[0, 0][:, cap:] = np.where(np.tril(np.ones((S, S), bool)), 0.0, -30000.0)
        elif nm == "cos":
            v = np.cos(ang)
        elif nm == "sin":
            v = np.sin(ang)
        else:
            raise SystemExit(f"no rule for {nm}")
        f[nm] = v.astype(np.float16) if p.get_element_type() == ov.Type.f16 else v
    r = cm(f)
    y = next(r[o] for o in cm.outputs if o.get_any_name() == "y")
    x = np.asarray(y, np.float32).reshape(1, S, W)
    extra = ""
    if cmpd:
        z = np.fromfile(f"{cmpd}/res{L}.f16", np.float16).astype(np.float32).reshape(x.shape)
        bad = ~np.isfinite(z)
        rel = np.linalg.norm(np.where(bad, 0, z) - x) / np.linalg.norm(x)
        extra = f"  npu rel {rel:.3e} non-finite {bad.sum()}"
        if bad.any():
            extra += f" (rows {np.unique(np.nonzero(bad)[1])[:8]})"
    print(f"layer{L}: |x| {np.abs(x).max():.2f}  {time.time() - t0:.0f} s{extra}", flush=True)

hm = core.compile_model(core.read_model(f"{headdir}\\head.xml"), "CPU", EXACT)
f16 = hm.inputs[0].get_element_type() == ov.Type.f16
logits = np.stack([np.asarray(list(hm({"x": x[:, i:i + 1].astype(np.float16) if f16 else x[:, i:i + 1]}).values())[0], np.float32).ravel()
                   for i in range(S)])
logits.tofile(out)
print(f"logits {logits.shape} -> {out}; argmax of last {int(logits[-1].argmax())}")
if ref:
    rf = np.fromfile(ref, np.float32).reshape(logits.shape)

    def lsm(z):
        z = z - z.max(-1, keepdims=True)
        return z - np.log(np.exp(z).sum(-1, keepdims=True))

    lp, lq = lsm(rf.astype(np.float64)), lsm(logits.astype(np.float64))
    kl = (np.exp(lp) * (lp - lq)).sum(-1)
    top1 = (rf.argmax(-1) == logits.argmax(-1)).mean()
    nxt = ids[1:]
    ppl_r = np.exp(-lp[np.arange(S - 1), nxt].mean())
    ppl_t = np.exp(-lq[np.arange(S - 1), nxt].mean())
    print(f"vs {ref}: mean KL {kl.mean():.5f} (max {kl.max():.4f}), top-1 agree {100 * top1:.1f} %, ppl {ppl_r:.3f} -> {ppl_t:.3f}")
