"""Static int8 ranges for the norm-fed MatMuls of the lean prefill graphs
(`build_delta.py` / `build_attn.py --srange FILE.json`): run the exact chain
of cut layers (`cut_all.py`, DIR/layer{L}.xml, zero state) on calibration
texts, tap the input of each layer's first projection (delta: in_proj_qkv,
attention: q_proj) and of gate_proj, rotate it as the lean graph does
(`quantrot.block_factors(k, ROTB)`) and record the per-token max |x R|.

    python rcal.py DIR OUT.json IDS [IDS ...] [--rotb 256] [--q 100] [--margin 1.0]

OUT.json: {"L": {"proj": r, "gu": r, "proj_p50": .., "gu_p50": ..}}, r = the
`--q` percentile over all calibration tokens of the per-token max, times
`--margin` (values beyond r are clipped by the static FakeQuantize)."""
import os
import json
import sys
import numpy as np
import openvino as ov
from quantrot import block_factors, rotate_any

a = sys.argv[1:]


def opt(name, d, conv=str):
    if name in a:
        i = a.index(name)
        v = a[i + 1]
        del a[i:i + 2]
        return conv(v)
    return d


export = opt("--export", os.environ.get("OV_EXPORT")) or sys.exit("--export DIR or OV_EXPORT: the OpenVINO export")
ROTB = opt("--rotb", 256, int)
Q = opt("--q", 100.0, float)
MARGIN = opt("--margin", 1.0, float)
NL = opt("--layers", 64, int)
ALL = "--all" in a
a = [v for v in a if v != "--all"]
d, out, *idsfs = a
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


TAPS = {"proj": ("linear_attn.in_proj_qkv/", "self_attn.q_proj/"), "gu": ("mlp.gate_proj/",)}
# `--all`: also the gated inputs (out: linear_attn.out_proj / o_proj, down: down_proj)
if ALL:
    TAPS.update({"out": ("linear_attn.out_proj/", "self_attn.o_proj/"), "down": ("mlp.down_proj/",)})
# one compiled model per layer, tapped once, reused for every text
models = []
for L in range(NL):
    m = core.read_model(f"{d}\\layer{L}.xml")
    names = []
    for o in m.get_ordered_ops():
        if o.get_type_name() != "MatMul":
            continue
        fn = o.get_friendly_name()
        for k, pats in TAPS.items():
            if any(p in fn for p in pats) and k not in names:
                m.add_results([ov.opset13.result(o.input_value(0))])
                names.append(k)
    assert sorted(names) == sorted(TAPS), (L, names)
    models.append((core.compile_model(m, "CPU", EXACT), names))
    print(f"layer {L} compiled", flush=True)

tokmax = {L: {k: [] for k in TAPS} for L in range(NL)}
for idsf in idsfs:
    ids = np.array([int(v) for v in open(idsf).read().split()], np.int64)
    S, W = len(ids), 5120
    x = list(emb({0: ids[None, :]}).values())[0].reshape(1, S, W).astype(np.float32)
    for L, (cm, names) in enumerate(models):
        r = cm(feed_for(cm, x, S))
        outs = cm.outputs
        for nm, o in zip(names, outs[len(outs) - len(names):]):
            X = np.asarray(r[o], np.float32).reshape(S, -1)
            Xr = rotate_any(X, block_factors(X.shape[1], ROTB))
            tokmax[L][nm].append(np.abs(Xr).max(1))
        y = next(r[o] for o in outs if list(o.get_partial_shape().get_shape()) == [1, S, W])
        x = np.asarray(y, np.float32).reshape(1, S, W)
    print(f"{idsf}: done", flush=True)

res = {}
for L in range(NL):
    e = {}
    for nm in TAPS:
        t = np.concatenate(tokmax[L][nm])
        e[nm] = float(np.percentile(t, Q) * MARGIN)
        e[nm + "_p50"] = float(np.median(t))
        e[nm + "_max"] = float(t.max())
    res[str(L)] = e
    print(f"L{L:2d} proj r {e['proj']:8.3f} (p50 {e['proj_p50']:.3f} max {e['proj_max']:.3f})  "
          f"gu r {e['gu']:8.3f} (p50 {e['gu_p50']:.3f} max {e['gu_max']:.3f})")
json.dump(res, open(out, "w"), indent=1)
print(f"-> {out}")
