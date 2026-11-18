"""Write small elementwise / norm micro-graphs (one op family each, applied
`--rep` times in a chain so the op dominates), for `npubench ir`.

    python ewbench.py OUTDIR [--s 256] [--n 5120] [--rep 4]"""
import os
import sys
import numpy as np
import openvino as ov
import openvino.opset13 as op

a = sys.argv[1:]


def opt(name, d, conv=str):
    if name in a:
        i = a.index(name)
        v = a[i + 1]
        del a[i:i + 2]
        return conv(v)
    return d


S = opt("--s", 256, int)
N = opt("--n", 5120, int)
R = opt("--rep", 4, int)
out = a[0]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)


def c16(v):
    return op.constant(np.asarray(v, np.float16))


def ci64(v):
    return op.constant(np.asarray(v, np.int64))


def variant(name, f):
    x = op.parameter([1, S, N], ov.Type.f16, name="x")
    y = x
    for _ in range(R):
        y = f(y)
    r = op.result(y)
    r.output(0).get_tensor().set_names({"y"})
    ov.save_model(ov.Model([r], [x], name), os.path.join(out, name + ".xml"), compress_to_fp16=False)


vec = (1 + 0.01 * rng.standard_normal([1, 1, N])).astype(np.float16)
variant("mul_scalar", lambda y: op.multiply(y, c16(0.999)))
variant("mul_vec", lambda y: op.multiply(y, c16(vec)))
variant("mul_col", lambda y: op.multiply(y, op.reduce_max(y, ci64([-1]), True)))
variant("add_self", lambda y: op.add(y, y))
variant("mul_self", lambda y: op.multiply(y, y))
variant("swish", lambda y: op.swish(y, c16(1.0)))
variant("reduce_mean_sq", lambda y: op.multiply(y, op.reduce_mean(op.multiply(y, y), ci64([-1]), True)))
variant("reduce_l2", lambda y: op.divide(y, op.reduce_l2(y, ci64([-1]), True)))
variant("mvn", lambda y: op.mvn(y, ci64([-1]), True, 1e-6, "inside_sqrt"))
variant("to_f32_back", lambda y: op.convert(op.multiply(op.convert(y, "f32"), op.constant(np.float32(0.999))), "f16"))
print("ok")
