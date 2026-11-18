"""Check a `build_delta.py` graph against the cut layer it was built from, on
the CPU (exact f32 for the reference; the built graph as compiled: its
FakeQuantize ops simulate int8): same x / states, mapped between the two
interfaces. Prints relative errors of y and both states.

    python cmp_delta.py CUT.xml BUILT.xml [--n N] [--seed 1] [--save DIR]

`--n`: real positions (the rest padding). `--save DIR`: write the built
graph's inputs (raw, named after its inputs + `.in`) and the reference
outputs mapped to its layout (`.ref`, f32), for checking an NPU run."""
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


seed = opt("--seed", 1, int)
n_real = opt("--n", None, int)
save = opt("--save", None)
cut, built = a
core = ov.Core()
rng = np.random.default_rng(seed)
mr = core.read_model(cut)
shp = {p.get_any_name(): list(p.get_partial_shape().get_shape()) for p in mr.inputs}
S = shp["x"][1]
n = n_real or S
x = rng.standard_normal([1, S, 5120]).astype(np.float32)
conv4 = (0.5 * rng.standard_normal([1, 10240, 4])).astype(np.float32)
ssm = (0.02 * rng.standard_normal([1, 48, 128, 128])).astype(np.float32)
C = None
feed = {}
for nm, d in shp.items():
    if nm == "x":
        v = x
    elif nm.startswith("cache_params.past.ssm"):
        v = ssm
    elif nm.startswith("cache_params.past.conv"):
        v = conv4
    elif nm == "qwen_rs.delta_tril_strict":
        v = np.tril(np.ones(d, np.float32), -1)
    elif nm == "qwen_rs.delta_tril_incl":
        v = np.tril(np.ones(d, np.float32), 0)
    elif nm.startswith("qwen_rs.delta_blk"):
        b = int(nm[len("qwen_rs.delta_blk"):])
        i, j = np.meshgrid(np.arange(d[0]), np.arange(d[0]), indexing="ij")
        v = ((i // (2 * b) == j // (2 * b)) & (i // b > j // b)).astype(np.float32)
    elif nm == "qwen_rs.valid":
        v = (np.arange(S) < n).astype(np.float32).reshape(d)
    elif nm == "qwen_rs.conv_keep":
        v = np.arange(n, n + d[0], dtype=np.int32)
    elif nm == "qwen_rs.conv_unscale":
        v = np.full(d, 2.0 ** -5, np.float32)
    elif nm == "beam_idx":
        v = np.zeros(d, np.int32)
    else:
        raise SystemExit(f"no rule for {nm}")
    feed[nm] = v
cr = core.compile_model(mr, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
rr = cr(feed)
ref = {}
for o in cr.outputs:
    nm = o.get_any_name()
    v = np.asarray(rr[o], np.float32)
    if ".ssm." in nm:
        ref["ssm_out"] = v
    elif ".conv." in nm:
        ref["conv_out"] = v.transpose(0, 2, 1)[:, 1:, :]
    else:
        ref["y"] = v

mb = core.read_model(built)
cb = core.compile_model(mb, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
bf = {
    "x": x.astype(np.float16),
    "conv": conv4.transpose(0, 2, 1)[:, 1:, :].astype(np.float16),
    "ssm": ssm.astype(np.float16),
    "valid": (np.arange(S) < n).astype(np.float16).reshape(1, S, 1),
    "keep": np.arange(n, n + 3, dtype=np.int32),
}
rb = cb(bf)
for o in cb.outputs:
    nm = o.get_any_name()
    v = np.asarray(rb[o], np.float32)
    r = ref[nm]
    if nm == "y":
        v, r = v[:, :n], r[:, :n]
    rel = np.linalg.norm(v - r) / max(np.linalg.norm(r), 1e-30)
    print(f"  {nm:10s} rel {rel:.3e}  max|d| {np.abs(v - r).max():.3e}  max|ref| {np.abs(r).max():.3e}")
if save:
    os.makedirs(save, exist_ok=True)
    for k, v in bf.items():
        v.tofile(os.path.join(save, k + ".in"))
    for k, v in ref.items():
        v.astype(np.float32).tofile(os.path.join(save, k + ".ref"))
