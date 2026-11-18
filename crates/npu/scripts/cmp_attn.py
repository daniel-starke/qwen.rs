"""Check a `build_attn.py` graph against the cut attention layer it was built
from, on the CPU (exact f32 reference; the built graph as compiled, its
FakeQuantize ops simulating int8): random x and past KV, `--pos P` past rows
valid. Prints relative errors of y and the new K / V rows.

    python cmp_attn.py CUT.xml BUILT.xml [--pos 300] [--seed 1] [--save DIR]

`--save DIR`: the built graph's inputs (`NAME.in`, raw f16) and the
references in its layout (`NAME.ref`, f32), for `cmp_out.py` on an NPU run."""
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
P = opt("--pos", 300, int)
save = opt("--save", None)
cut, built = a
core = ov.Core()
rng = np.random.default_rng(seed)
mr = core.read_model(cut)
shp = {p.get_any_name(): list(p.get_partial_shape().get_shape()) for p in mr.inputs}
S = shp["x"][1]
kname = [n for n in shp if ".key." in n][0]
vname = [n for n in shp if ".value." in n][0]
CAP = shp[kname][2]
x = rng.standard_normal([1, S, 5120]).astype(np.float32)
kp = rng.standard_normal([1, 4, CAP, 256]).astype(np.float32)
vp = (0.5 * rng.standard_normal([1, 4, CAP, 256])).astype(np.float32)
kp[:, :, P:] = 0
vp[:, :, P:] = 0
am = np.zeros([1, CAP + S], np.int64)
am[0, :P] = 1
am[0, CAP:] = 1
pos = np.arange(P, P + S)
feed = {"x": x, kname: kp, vname: vp, "attention_mask": am,
        "position_ids": np.broadcast_to(pos, [4, 1, S]).astype(np.int64).copy()}
cr = core.compile_model(mr, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
rr = cr(feed)
ref = {}
for o in cr.outputs:
    nm = o.get_any_name()
    v = np.asarray(rr[o], np.float32)
    if ".key." in nm:
        ref["knew"] = v[:, :, CAP:].transpose(0, 2, 1, 3)
    elif ".value." in nm:
        ref["vnew"] = v[:, :, CAP:].transpose(0, 2, 1, 3)
    else:
        ref["y"] = v

# rotary tables: text positions, theta from the cut's inv_freq
ops = {o.get_friendly_name(): o for o in mr.get_ordered_ops()}
mm = [o for n, o in ops.items() if n.endswith("rotary_emb/aten::matmul/MatMul")][0]
sub = ov.Model([ov.opset13.result(mm.input_value(0))], [], "w")
inv = core.compile_model(sub, "CPU")({})[0].reshape(3, -1)[0].astype(np.float64)
ang = pos[:, None] * inv[None, :]
ang = np.concatenate([ang, ang], 1)
mask = np.full([1, 1, S, CAP + S], -30000.0, np.float32)
mask[..., :P] = 0
mask[0, 0][np.tril(np.ones((S, S), bool))[:, :] .nonzero()[0], CAP + np.tril(np.ones((S, S), bool)).nonzero()[1]] = 0
bf = {
    "x": x.astype(np.float16),
    "kpast": kp.transpose(0, 2, 1, 3).astype(np.float16),
    "vpast": vp.transpose(0, 2, 1, 3).astype(np.float16),
    "mask": mask.astype(np.float16),
    "cos": np.cos(ang).reshape(1, S, -1).astype(np.float16),
    "sin": np.sin(ang).reshape(1, S, -1).astype(np.float16),
}
mb = core.read_model(built)
cb = core.compile_model(mb, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
rb = cb(bf)
for o in cb.outputs:
    nm = o.get_any_name()
    v = np.asarray(rb[o], np.float32)
    r = ref[nm]
    rel = np.linalg.norm(v - r) / max(np.linalg.norm(r), 1e-30)
    print(f"  {nm:6s} rel {rel:.3e}  max|d| {np.abs(v - r).max():.3e}  max|ref| {np.abs(r).max():.3e}")
if save:
    os.makedirs(save, exist_ok=True)
    for k, v in bf.items():
        v.tofile(os.path.join(save, k + ".in"))
    for k, v in ref.items():
        np.ascontiguousarray(v, np.float32).tofile(os.path.join(save, k + ".ref"))
