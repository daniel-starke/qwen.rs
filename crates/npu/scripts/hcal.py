"""Input Hessians H = sum X^T X of every weight MatMul the lean prefill
graphs quantize, for GPTQ (`wq.py`, NPU_WQ=gptq:DIR): run the exact chain of
cut layers (`cut_all.py`, DIR/layer{L}.xml, S-position graphs, zero state per
chunk) over a calibration text cut into S-token chunks and tap the inputs of
the projection (delta in_proj_qkv / attention q_proj), the output projection
(out_proj / o_proj), gate_proj and down_proj.

    python hcal.py CUTDIR OUTDIR IDS [--layers 64] [--chunks N]

Writes OUTDIR/L{L}.{proj,out,gu,down}.npy (f32 [k, k], unrotated: the
builders rotate it with the weight) and prints each one's token count."""
import os
import sys
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
NL = opt("--layers", 64, int)
NCH = opt("--chunks", 99, int)
d, out, idsf = a
os.makedirs(out, exist_ok=True)
core = ov.Core()
EXACT = {"INFERENCE_PRECISION_HINT": "f32"}
emb = core.compile_model(core.read_model(export + r"\openvino_text_embeddings_model.xml"), "CPU")


def tril(c, incl):
    return np.tril(np.ones((c, c), np.float32), 0 if incl else -1)


def blk(c, b):
    i, j = np.meshgrid(np.arange(c), np.arange(c), indexing="ij")
    return ((i // (2 * b) == j // (2 * b)) & (i // b > j // b)).astype(np.float32)


def feed_for(cm, x, S):
    f = {}
    for p in cm.inputs:
        nm = p.get_any_name()
        dd = list(p.get_shape())
        if nm == "x":
            v = x
        elif nm.startswith("cache_params.past."):
            v = np.zeros(dd, np.float32)
        elif nm == "qwen_rs.delta_tril_strict":
            v = tril(dd[0], False)
        elif nm == "qwen_rs.delta_tril_incl":
            v = tril(dd[0], True)
        elif nm.startswith("qwen_rs.delta_blk"):
            v = blk(dd[0], int(nm[len("qwen_rs.delta_blk"):]))
        elif nm == "qwen_rs.valid":
            v = np.ones(dd, np.float32)
        elif nm == "qwen_rs.conv_keep":
            v = np.arange(S, S + dd[0], dtype=np.int32)
        elif nm == "qwen_rs.conv_unscale":
            v = np.full(dd, 2.0 ** -5, np.float32)
        elif nm == "attention_mask":
            v = np.zeros(dd, np.int64)
            v[..., -S:] = 1
        elif nm == "position_ids":
            v = np.broadcast_to(np.arange(0, S, dtype=np.int64), dd).copy()
        else:
            raise SystemExit(f"no rule for input {nm} {dd}")
        f[nm] = v.astype(np.float16) if p.get_element_type() == ov.Type.f16 else v
    return f


TAPS = {
    "proj": ("linear_attn.in_proj_qkv/", "self_attn.q_proj/"),
    "out": ("linear_attn.out_proj/", "self_attn.o_proj/"),
    "gu": ("mlp.gate_proj/",),
    "down": ("mlp.down_proj/",),
}
ids = np.array([int(v) for v in open(idsf).read().split()], np.int64)
S = 256
chunks = [ids[i:i + S] for i in range(0, len(ids) - S + 1, S)][:NCH]
xs = [list(emb({0: c[None, :]}).values())[0].reshape(1, S, 5120).astype(np.float32) for c in chunks]
print(f"{len(chunks)} chunks of {S}", flush=True)
for L in range(NL):
    m = core.read_model(f"{d}\\layer{L}.xml")
    names = []
    for o in m.get_ordered_ops():
        if o.get_type_name() != "MatMul":
            continue
        fn = o.get_friendly_name()
        for k, pats in TAPS.items():
            if any(p in fn for p in pats) and k not in names and "/ov_ext::linear/" in fn:
                m.add_results([ov.opset13.result(o.input_value(0))])
                names.append(k)
    assert sorted(names) == sorted(TAPS), (L, names)
    cm = core.compile_model(m, "CPU", EXACT)
    H = {}
    for ci in range(len(xs)):
        r = cm(feed_for(cm, xs[ci], S))
        outs = cm.outputs
        for nm, o in zip(names, outs[len(outs) - len(names):]):
            X = np.asarray(r[o], np.float64).reshape(S, -1)
            H[nm] = H.get(nm, 0) + X.T @ X
        y = next(r[o] for o in outs if list(o.get_partial_shape().get_shape()) == [1, S, 5120])
        xs[ci] = np.asarray(y, np.float32).reshape(1, S, 5120)
    for nm, h in H.items():
        np.save(os.path.join(out, f"L{L}.{nm}.npy"), (h / (S * len(xs))).astype(np.float32))
    print(f"layer {L}: " + " ".join(f"{nm} {h.shape[0]}" for nm, h in H.items()), flush=True)
