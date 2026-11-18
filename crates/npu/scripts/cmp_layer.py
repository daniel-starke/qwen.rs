"""Run cut layer IRs (`cut_layer.py`, optionally rewritten by `w8.py`) on the
CPU with the same realistic inputs and compare every output against the
first IR (exact f32): relative L2 error and max abs diff per output.

    python cmp_layer.py REF.xml TEST.xml [TEST2.xml ...] [--seed 1] [--save DIR]

Inputs: `x` ~ N(0, 1) (the residual; the layer normalizes it), past states
small noise (SSM 0.02, conv / KV 0.5), the `qwen_rs.*` masks as the engine
sets them for a full chunk (no padding), attention mask all ones, positions
`cap .. cap + s`. `--save DIR` writes the inputs and REF's outputs as raw
f16 files named after the ports (for an NPU run of a blob)."""
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


seed = opt("--seed", 1, int)
save = opt("--save", None)
xmls = a
core = ov.Core()


def tril(c, incl):
    return np.tril(np.ones((c, c), np.float32), 0 if incl else -1)


def blk(c, b):
    i, j = np.meshgrid(np.arange(c), np.arange(c), indexing="ij")
    return ((i // (2 * b) == j // (2 * b)) & (i // b > j // b)).astype(np.float32)


def make_inputs(m):
    ins = {}
    shp = {p.get_any_name(): list(p.get_partial_shape().get_shape()) for p in m.inputs}
    s = shp["x"][1]
    for nm, d in shp.items():
        rng = np.random.default_rng([seed] + list(nm.encode()))
        if nm == "x":
            v = rng.standard_normal(d).astype(np.float32)
        elif nm.startswith("cache_params.past.ssm"):
            v = (0.02 * rng.standard_normal(d)).astype(np.float32)
        elif nm.startswith("cache_params.past."):
            v = (0.5 * rng.standard_normal(d)).astype(np.float32)
        elif nm == "qwen_rs.delta_tril_strict":
            v = tril(d[0], False)
        elif nm == "qwen_rs.delta_tril_incl":
            v = tril(d[0], True)
        elif nm.startswith("qwen_rs.delta_blk"):
            v = blk(d[0], int(nm[len("qwen_rs.delta_blk"):]))
        elif nm == "qwen_rs.valid":
            v = np.ones(d, np.float32)
        elif nm == "qwen_rs.conv_keep":
            v = np.arange(s, s + d[0], dtype=np.int32)
        elif nm == "qwen_rs.conv_unscale":
            v = np.full(d, 2.0 ** -5, np.float32)
        elif nm == "attention_mask":
            v = np.ones(d, np.int64)
        elif nm == "position_ids":
            cap = shp["attention_mask"][1] - s if "attention_mask" in shp else 0
            v = np.broadcast_to(np.arange(cap, cap + s, dtype=np.int64), d).copy()
        else:
            raise SystemExit(f"no rule for input {nm} {d}")
        ins[nm] = v
    return ins


def run(xml, ins):
    m = core.read_model(xml)
    c = core.compile_model(m, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    feed = {}
    for p in c.inputs:
        nm = p.get_any_name()
        feed[nm] = ins[nm].astype(np.float16) if p.get_element_type() == ov.Type.f16 else ins[nm]
    r = c(feed)
    return {o.get_any_name(): np.asarray(r[o], np.float32) for o in c.outputs}, m


ins = make_inputs(core.read_model(xmls[0]))
ref, _ = run(xmls[0], ins)
for x in xmls[1:]:
    out, _ = run(x, make_inputs(core.read_model(x)))
    print(x)
    for k, r in ref.items():
        o = out.get(k)
        if o is None:
            print(f"  {k}: missing")
            continue
        rel = np.linalg.norm(o - r) / max(np.linalg.norm(r), 1e-30)
        print(f"  {k:40s} rel {rel:.2e}  max|d| {np.abs(o - r).max():.3e}  max|ref| {np.abs(r).max():.3e}")
if save:
    import os
    os.makedirs(save, exist_ok=True)
    for k, v in ins.items():
        v.astype(np.float16 if v.dtype == np.float32 else v.dtype).tofile(os.path.join(save, k + ".in"))
    for k, v in ref.items():
        v.astype(np.float32).tofile(os.path.join(save, k + ".ref"))
