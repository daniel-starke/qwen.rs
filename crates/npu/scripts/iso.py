"""Per-layer NPU error in isolation: run lean layer L on the CPU (exact f32,
FakeQuantize simulating int8) from the NPU's own input to it (DUMPS/res{L-1}.f16,
`npubench prefill --dump-layers DUMPS`) and compare with the NPU's output
DUMPS/res{L}.f16, relative to the layer's residual update.

    python iso.py DIR DUMPS INDIR L [L ...]     (INDIR: npubench --inputs)"""
import os
import sys
import numpy as np
import openvino as ov

d, dumps, ind, *ls = sys.argv[1:]
core = ov.Core()
S, W = 256, 5120
for L in map(int, ls):
    x = np.fromfile(f"{dumps}/res{L - 1}.f16" if L else f"{ind}/x.in", np.float16).reshape(1, S, W)
    cm = core.compile_model(core.read_model(f"{d}/lean{L}.xml"), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    f = {}
    for p in cm.inputs:
        nm, shp = p.get_any_name(), list(p.get_shape())
        own = f"{d}/lean{L}.{nm}.in"
        dt = {ov.Type.i8: np.int8, ov.Type.f16: np.float16, ov.Type.i32: np.int32}[p.get_element_type()]
        if nm == "x":
            v = x
        elif os.path.exists(own):
            v = np.fromfile(own, dt).reshape(shp)
        elif os.path.exists(f"{ind}/{nm}.in") and os.path.getsize(f"{ind}/{nm}.in") == np.prod(shp) * np.dtype(dt).itemsize:
            v = np.fromfile(f"{ind}/{nm}.in", dt).reshape(shp)
        else:
            v = np.zeros(shp, dt)
        f[nm] = v
    r = cm(f)
    y = next(r[o] for o in cm.outputs if o.get_any_name() == "y").reshape(S, W).astype(np.float64)
    n = np.fromfile(f"{dumps}/res{L}.f16", np.float16).astype(np.float64).reshape(S, W)
    dl = y - x.reshape(S, W).astype(np.float64)
    e = n - y
    rr = np.linalg.norm(e, axis=1) / np.maximum(np.linalg.norm(dl, axis=1), 1e-9)
    print(f"L{L:2d} rel(update) {np.linalg.norm(e) / np.linalg.norm(dl):.4f}  rows p50 {np.median(rr):.4f} max {rr.max():.4f} @ {int(rr.argmax())}", flush=True)
