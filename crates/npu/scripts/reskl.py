r"""KL between final residuals of two engine runs over the same tokens.

    python reskl.py REF.res VAR.res... [--head DIR] [--tail N] [--step K]

REF / VAR: `QWEN_NPU_DUMP_RESID` files (f16 [S, 5120]). The decode head
(DIR\head.xml, on the CPU) runs on every K-th of the last N positions of
each; prints mean / median / max KL(REF || VAR), top-1 agreement and the
relative rms difference of the residual rows. For long-context checks
(far blocks), where no BF16 truth exists.
"""
import os
import sys
import numpy as np
import openvino as ov

a = sys.argv[1:]


def opt(k, d, t=str):
    if k in a:
        i = a.index(k)
        v = t(a[i + 1])
        del a[i:i + 2]
        return v
    return d


head = opt("--head", os.environ.get("NPU_HEAD")) or sys.exit("--head DIR or NPU_HEAD: the decode head's directory")
tail = opt("--tail", 1024, int)
step = opt("--step", 4, int)
W = 5120
hm = ov.Core().compile_model(ov.Core().read_model(head + r"\head.xml"), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
f16 = hm.inputs[0].get_element_type() == ov.Type.f16


def rows(f):
    x = np.fromfile(f, np.float16).reshape(-1, W)
    return x[max(0, len(x) - tail)::step]


def lsm(x):
    z = np.stack([np.asarray(list(hm({"x": (r if f16 else r.astype(np.float32)).reshape(1, 1, W)}).values())[0], np.float64).ravel() for r in x])
    z -= z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


xr = rows(a[0])
lp = lsm(xr)
for f in a[1:]:
    x = rows(f)
    lq = lsm(x)
    kl = (np.exp(lp) * (lp - lq)).sum(-1)
    d = x.astype(np.float32) - xr.astype(np.float32)
    rel = np.sqrt((d * d).sum() / (xr.astype(np.float32) ** 2).sum())
    print(f"{f}: {len(x)} rows, KL {kl.mean():.5f} (median {np.median(kl):.5f}, max {kl.max():.4f}), top-1 {100 * (lp.argmax(-1) == lq.argmax(-1)).mean():.1f} %, resid rel rms {rel:.4f}", flush=True)
