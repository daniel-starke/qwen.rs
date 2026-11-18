"""Teacher-forced prefill through per-layer cut graphs on the CPU: embed the
prompt `IDS` (one token id per line, S of them), run DIR/layer0..63.xml (S
positions each, zero initial state, `cut_all.py --s S`), then the decode head
DIR_HEAD/head.xml on every position. Writes OUT (f32 logits, S x vocab) --
compare two runs with `kl.py`-style statistics via `--ref REF.f32`.

    python chain.py DIR IDS OUT [--head DIR] [--layers 64] [--ref REF.f32]
                    [--export DIR] [--device CPU]

Masks as the engine sets them for a full, unpadded chunk."""
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
nl = opt("--layers", 64, int)
ref = opt("--ref", None)
device = opt("--device", "CPU")
headdir = opt("--head", None)
# `--calib OUT.json`: record max |input| of every weight MatMul (static
# activation ranges for `w8.py --act cal:OUT.json`).
calib = opt("--calib", None)
cal = {}
d, idsf, out = a
headdir = headdir or d
ids = np.array([int(v) for v in open(idsf).read().split()], np.int64)
S, W = len(ids), 5120
core = ov.Core()
EXACT = {"INFERENCE_PRECISION_HINT": "f32"}

emb = core.compile_model(core.read_model(export + r"\openvino_text_embeddings_model.xml"), "CPU")
x = list(emb({0: ids[None, :]}).values())[0].reshape(1, S, W).astype(np.float32)


def tril(c, incl):
    return np.tril(np.ones((c, c), np.float32), 0 if incl else -1)


def blk(c, b):
    i, j = np.meshgrid(np.arange(c), np.arange(c), indexing="ij")
    return ((i // (2 * b) == j // (2 * b)) & (i // b > j // b)).astype(np.float32)


def feed_for(cm, x):
    f = {}
    for p in cm.inputs:
        nm = p.get_any_name()
        d = list(p.get_shape())
        f16 = p.get_element_type() == ov.Type.f16
        if nm == "x":
            v = x
        elif nm.startswith("cache_params.past."):
            v = np.zeros(d, np.float32)
        elif nm == "qwen_rs.delta_tril_strict":
            v = tril(d[0], False)
        elif nm == "qwen_rs.delta_tril_incl":
            v = tril(d[0], True)
        elif nm.startswith("qwen_rs.delta_blk"):
            v = blk(d[0], int(nm[len("qwen_rs.delta_blk"):]))
        elif nm == "qwen_rs.valid":
            v = np.ones(d, np.float32)
        elif nm == "qwen_rs.conv_keep":
            v = np.arange(S, S + d[0], dtype=np.int32)
        elif nm == "qwen_rs.conv_unscale":
            v = np.full(d, 2.0 ** -5, np.float32)
        elif nm == "attention_mask":
            # past slots empty (zero state), the S new ones valid
            v = np.zeros(d, np.int64)
            v[..., -S:] = 1
        elif nm == "position_ids":
            v = np.broadcast_to(np.arange(0, S, dtype=np.int64), d).copy()
        else:
            raise SystemExit(f"no rule for input {nm} {d}")
        f[nm] = v.astype(np.float16) if f16 else v
    return f


def const_only(node, seen=None):
    seen = set() if seen is None else seen
    if id(node) in seen:
        return True
    seen.add(id(node))
    t = node.get_type_name()
    if t == "Constant":
        return True
    if t == "Parameter":
        return False
    return all(const_only(v.get_node(), seen) for v in node.input_values())


t0 = time.time()
for L in range(nl):
    m = core.read_model(f"{d}\\layer{L}.xml")
    probes = []
    if calib:
        for o in m.get_ordered_ops():
            if o.get_type_name() == "MatMul" and const_only(o.input_value(1).get_node()):
                rr = ov.opset13.result(o.input_value(0))
                m.add_results([rr])
                probes.append(o.get_friendly_name())
    cm = core.compile_model(m, device, EXACT if device == "CPU" else {})
    r = cm(feed_for(cm, x))
    for nm, o in zip(probes, cm.outputs[len(cm.outputs) - len(probes):]):
        cal[nm] = float(np.abs(np.asarray(r[o])).max())
    y = next(r[o] for o in cm.outputs if o.get_any_name() == "y" or (o.get_partial_shape().is_static and list(o.get_shape()) == [1, S, W]))
    x = np.asarray(y, np.float32).reshape(1, S, W)
    print(f"layer{L}: |x| {np.abs(x).max():.2f}  {time.time() - t0:.0f} s", flush=True)
if calib:
    import json
    json.dump(cal, open(calib, "w"), indent=0)
    print(f"{len(cal)} MatMul input ranges -> {calib}")
hm = core.compile_model(core.read_model(f"{headdir}\\head.xml"), "CPU", EXACT)
logits = np.stack([np.asarray(list(hm({"x": x[:, i:i + 1].astype(np.float16) if hm.inputs[0].get_element_type() == ov.Type.f16 else x[:, i:i + 1]}).values())[0], np.float32).ravel() for i in range(S)])
logits.tofile(out)
print(f"logits {logits.shape} -> {out}; argmax of last {int(logits[-1].argmax())}")
if ref:
    r = np.fromfile(ref, np.float32).reshape(logits.shape)

    def lsm(z):
        z = z - z.max(-1, keepdims=True)
        return z - np.log(np.exp(z).sum(-1, keepdims=True))

    lp, lq = lsm(r.astype(np.float64)), lsm(logits.astype(np.float64))
    kl = (np.exp(lp) * (lp - lq)).sum(-1)
    top1 = (r.argmax(-1) == logits.argmax(-1)).mean()
    nxt = ids[1:]
    ppl_r = np.exp(-lp[np.arange(S - 1), nxt].mean())
    ppl_t = np.exp(-lq[np.arange(S - 1), nxt].mean())
    print(f"vs {ref}: mean KL {kl.mean():.5f} (max {kl.max():.4f}), top-1 agree {100 * top1:.1f} %, ppl {ppl_r:.3f} -> {ppl_t:.3f}")
