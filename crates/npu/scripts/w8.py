"""Rewrite the weight MatMuls of a cut layer IR (`cut_layer.py`) to int8 x
int8 on the NPU's DPU: each int4-g128 weight is dequantized on the host and
requantized to int8 with one scale per output row (i8 [n, k] -> f16 x scale
[n, 1], the form the compiler lowers to a quantized convolution with a
per-channel weight table, no K split), and the MatMul's activation is
quantized per token in the graph:

    s = max(|x|, axis=-1) / 127;  y = MatMul(FakeQuantize(x / s, +-127), W8) * s

(`--act dyn`, default). `--act none` keeps f16 activations (int8 weights
only), `--act fq:R` uses a static FakeQuantize over [-R, R] (no per-token
scale; R must cover the activations).

    python w8.py IN.xml OUT.xml [--act dyn|none|fq:R] [--min-n 256]
                 [--smooth A --stats CAL.npz]

Weight MatMuls with fewer than `--min-n` output rows (the delta-net's
in_proj_a / in_proj_b, 48 rows) are left alone. `--smooth A`: SmoothQuant
migration strength: per input channel c_j = max|x_j|^A / max|W_j|^(1-A),
max|x_j| from `actq.py --stats` (calibration text), max|W_j| over every
weight MatMul reading that x; x / c is quantized, W * c requantized (in a
lean graph 1 / c folds into the preceding norm weight / up_proj rows)."""
import sys
import numpy as np
from quantrot import rot_factors, rotate, rotate_graph
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


act = opt("--act", "dyn")
cal_margin = opt("--margin", 1.0, float)
# `--f16-act a,b`: MatMuls whose name contains one of these keep f16
# activations (int8 weights only).
f16_act = [v for v in opt("--f16-act", "").split(",") if v]
min_n = opt("--min-n", 256, int)
smooth = opt("--smooth", 0.0, float)
# `--rot`: rotate every dynamically quantized activation (quantrot.py) and
# fold the rotation into the weight (instead of / before --smooth).
ROT = "--rot" in a
a = [v for v in a if v != "--rot"]
stats = dict(np.load(opt("--stats", None))) if smooth else {}
inp, outp = a

core = ov.Core()
m = core.read_model(inp)


def const_only(node, seen=None):
    """True if every leaf feeding `node` is a Constant."""
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


def evaluate(output):
    sub = ov.Model([op.result(output)], [], "w")
    c = core.compile_model(sub, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    return c({})[0]


quant_cache = {}
calr = {}
if act.startswith("cal:"):
    import json
    calr = json.load(open(act[4:]))


def consumers(x):
    return [t.get_node().get_friendly_name() for t in x.get_target_inputs() if t.get_node().get_type_name() == "MatMul"]


def wmat(mm):
    W = evaluate(mm.input_value(1)).astype(np.float32)
    if not mm.get_attributes()["transpose_b"]:
        W = W.T
    return W.reshape(-1, W.shape[-1])


smooth_c = {}


def smoothing(x):
    """c [k] for the int8 MatMuls reading x (None without --smooth)."""
    if not smooth:
        return None
    key = (x.get_node().get_friendly_name(), x.get_index())
    if key not in smooth_c:
        mms = [t.get_node() for t in x.get_target_inputs() if t.get_node().get_type_name() == "MatMul"
               and const_only(t.get_node().input_value(1).get_node())]
        ws = [wmat(mm) for mm in mms]
        ws = [w for w in ws if w.shape[0] >= min_n]
        xmax = max((stats[mm.get_friendly_name()] for mm in mms if mm.get_friendly_name() in stats), key=len, default=None)
        if xmax is None or not ws:
            smooth_c[key] = None
        else:
            wmax = np.max([np.abs(w).max(0) for w in ws], axis=0)
            c = np.maximum(xmax, 1e-5) ** smooth / np.maximum(wmax, 1e-8) ** (1 - smooth)
            smooth_c[key] = (c / np.exp(np.log(c).mean())).astype(np.float32)
    return smooth_c[key]


def quantize_act(x, act):
    """Per-token dynamic int8: (FakeQuantize(x / s), s), x in f16."""
    key = (x.get_node().get_friendly_name(), x.get_index(), act)
    if key in quant_cache:
        return quant_cache[key]
    xh = op.convert(x, "f16") if x.get_element_type() != ov.Type.f16 else x
    c = smoothing(x) if act == "dyn" else None
    if c is not None:
        xh = op.multiply(xh, op.constant((1.0 / c).astype(np.float16)))
    if act == "dyn" and ROT:
        xh = rotate_graph(op, xh, rot_factors(x.get_partial_shape()[-1].get_length()))
    if act == "none":
        r = (xh, None)
    elif act.startswith("cal:"):
        # static per-tensor range from `chain.py --calib`, keyed by the
        # MatMul consuming x (all consumers of one x share it: take the max)
        lim = max(calr.get(n, 0.0) for n in consumers(x)) * cal_margin
        lo, hi = op.constant(np.float16(-lim)), op.constant(np.float16(lim))
        r = (op.fake_quantize(xh, lo, hi, lo, hi, 255), None)
    elif act.startswith("fq:"):
        lim = float(act[3:])
        lo, hi = op.constant(np.float16(-lim)), op.constant(np.float16(lim))
        r = (op.fake_quantize(xh, lo, hi, lo, hi, 255), None)
    else:
        mx = op.reduce_max(op.absolute(xh), op.constant(np.array([-1], np.int64)), True)
        mx = op.maximum(mx, op.constant(np.float16(1e-4)))
        s = op.multiply(mx, op.constant(np.float16(1.0 / 127.0)))
        xs = op.divide(xh, s)
        lo, hi = op.constant(np.float16(-127.0)), op.constant(np.float16(127.0))
        r = (op.fake_quantize(xs, lo, hi, lo, hi, 255), s)
    quant_cache[key] = r
    return r


n_done, err = 0, []
for mm in [o for o in m.get_ordered_ops() if o.get_type_name() == "MatMul"]:
    w = mm.input_value(1)
    if not const_only(w.get_node()):
        continue
    at = mm.get_attributes()
    W = evaluate(w).astype(np.float32)
    if not at["transpose_b"]:
        W = W.T
    W = W.reshape(-1, W.shape[-1])
    n, k = W.shape
    if n < min_n:
        continue
    x = mm.input_value(0)
    act_mm = "none" if any(p in mm.get_friendly_name() for p in f16_act) else act
    c = smoothing(x) if act_mm == "dyn" else None
    if c is not None:
        W = W * c.reshape(1, -1)
    if act_mm == "dyn" and ROT:
        W = rotate(W, rot_factors(k)).astype(np.float32)
    amax = np.abs(W).max(axis=1, keepdims=True)
    sc = np.maximum(amax, 1e-12) / 127.0
    q = np.clip(np.rint(W / sc), -127, 127).astype(np.int8)
    rel = np.linalg.norm(q * sc - W) / max(np.linalg.norm(W), 1e-30)
    err.append(rel)
    wq = op.multiply(op.convert(op.constant(q), "f16"), op.constant(sc.astype(np.float16)))
    xq, s = quantize_act(x, act_mm)
    y = op.matmul(xq, wq, at["transpose_a"], True)
    if s is not None:
        y = op.multiply(y, s)
    if mm.get_output_element_type(0) != ov.Type.f16:
        y = op.convert(y, mm.get_output_element_type(0))
    for t in list(mm.output(0).get_target_inputs()):
        t.replace_source_output(y.output(0) if hasattr(y, "output") else y)
    n_done += 1
    print(f"  {mm.get_friendly_name()}: [{n} x {k}] int8, requant rel err {rel:.4f}", flush=True)

m.validate_nodes_and_infer_types()
ov.save_model(m, outp, compress_to_fp16=False)
print(f"{n_done} MatMuls -> int8 weights, activations {act}; mean requant rel err {np.mean(err):.4f} -> {outp}")
