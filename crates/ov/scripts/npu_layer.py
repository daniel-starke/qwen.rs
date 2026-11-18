"""Decoder layers of the real export as a standalone graph.

The full NPU graph takes 4-6 min to compile and its error depends on how NPUW
partitions it (probes move it). This cuts layers L..Lb out of the engine's
stateless IR (int4 weights, delta-net Loops, convs, attention, MLPs, exactly
as the engine compiles them) with the residual before L as a new Parameter
and the residual after Lb plus the present states as Results, and replays a
residual stream through it step by step (recurrent states and KV carried on
the host in f16, as the engine does). The same cut compiled on the exact CPU,
on the same inputs, is the reference, so the error is these layers' alone.

    python npu_layer.py XML RES.bin L[-Lb][,...] [--dev NPU] [--plain]
           [--full NPU_RES.bin --feed full] [--flush-sub] [--cap N] [--prop K=V ...]

XML: a stateless IR the engine wrote (`QWEN_OV_KEEP_XML=DIR`), read with the
export's `.bin` (`--export DIR`). RES.bin: a `QWEN_OV_PROBE_DUMP` of residual
probes `0,1,...,63` from the exact CPU (f32 x 5120 per probe and step).
`--full`: the same dump from the full NPU graph; with `--feed full` its
residuals are the inputs, so the full graph's own error at these layers (in
situ) is printed beside the cut's. A single layer is never folded by NPUW;
spans of 2+ repeated blocks (e.g. 11-19) are, and reproduce the full graph.
`--flush-sub`: the device model's subnormal f16 constants set to zero (a CPU
stand-in for a device that flushes them). `--cap N`: KV capacity (default
80). `--plain`: the NPU without NPUW. Found the conv weight-scale flush
(`stateless::scale_conv_weights`)."""
import os
import re, sys, time
import numpy as np
import openvino as ov
from openvino.preprocess import PrePostProcessor

args = sys.argv[1:]
def opt(name, d, conv=str):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return conv(v)
    return d
dev = opt("--dev", "NPU")
export = opt("--export", os.environ.get("OV_EXPORT")) or sys.exit("--export DIR or OV_EXPORT: the OpenVINO export")
full = opt("--full", None)
feed = opt("--feed", "cpu")
steps = opt("--steps", 0, int)
CAP = opt("--cap", 80, int)  # static KV capacity of the attention layers' past
flush = "--flush-sub" in args
if flush: args.remove("--flush-sub")
plain = "--plain" in args
if plain: args.remove("--plain")
props = {}
while "--prop" in args:
    i = args.index("--prop"); k, v = args[i + 1].split("=", 1); props[k] = v; del args[i:i + 2]
xml, resf = args[0], args[1]
# "8" or a span "8-10" (input before 8, output after 10)
layers = [tuple(int(y) for y in (x.split("-") * 2)[:2]) for x in args[2].split(",")]
if dev == "NPU" and not plain:
    base = {"NPU_USE_NPUW": "YES", "NPUW_ONLINE_PIPELINE": "REP", "NPUW_FOLD": "YES", "NPUW_DQ": "YES",
            "NPUW_FUNCALL_ASYNC": "YES", "NPUW_WEIGHTS_BANK": "shared"}
    props = {**base, **props}

W = 5120
def load_res(f):
    raw = np.fromfile(f, np.float32)
    n = len(raw) // (64 * W)
    return raw[: n * 64 * W].reshape(n, 64, W).astype(np.float64)
R = load_res(resf)
F = load_res(full) if full else None
n = R.shape[0] if not steps else min(steps, R.shape[0])
assert n <= CAP, f"{n} steps need --cap >= {n}"

core = ov.Core()
LAY = re.compile(r"language_model\.layers\.(\d+)[./]")

def layer_of(node):
    m = LAY.search(node.get_friendly_name())
    return int(m.group(1)) if m else None

def cut(L, Lb, flush=False):
    """Layer L as a model: x -> (out, present states). Rebuilt from a fresh
    read each time, since replacing inputs mutates the graph."""
    m = core.read_model(xml, export + r"\openvino_language_model.bin")
    ops = {o.get_friendly_name(): o for o in m.get_ops()}
    out = ops[f"__module.model.model.language_model.layers.{Lb}/aten::add/Add_1"]
    prev = ops[f"__module.model.model.language_model.layers.{L - 1}/aten::add/Add_1"]
    x = ov.opset13.parameter([1, 1, W], ov.Type.f32, name="x")
    x.output(0).get_tensor().set_names({"x"})
    for t in list(prev.output(0).get_target_inputs()):
        if layer_of(t.get_node()) == L:
            t.replace_source_output(x.output(0))
    # Every Parameter the layer's output reaches: its past ports (behind the
    # rewrite's beam_idx Gather), beam_idx, and inputs_embeds (shape only).
    seen, stack, reach = set(), [out], []
    while stack:
        nd = stack.pop()
        if id(nd) in seen: continue
        seen.add(id(nd))
        if nd.get_type_name() == "Parameter": reach.append(nd); continue
        stack.extend(v.get_node() for v in nd.input_values())
    pasts = [p for p in reach if p.output(0).get_any_name().startswith("cache_params.past.")]
    others = [p for p in reach if p not in pasts and p.get_friendly_name() != "x"]
    results = [ov.opset13.result(out.output(0))]
    results[0].output(0).get_tensor().set_names({"y"})
    for p in pasts:
        pn = p.output(0).get_any_name().replace(".past.", ".present.")
        r = next(r for r in m.get_results() if pn in r.output(0).get_names())
        rr = ov.opset13.result(r.input_value(0))
        rr.output(0).get_tensor().set_names({pn})
        results.append(rr)
    if flush:
        # `--flush-sub`: the device model's f16 constants in these layers with
        # their subnormals set to zero; what a device that flushes them
        # would compute (a CPU simulation of the NPU).
        nf = 0
        for o in m.get_ops():
            if o.get_type_name() == "Constant" and o.get_output_element_type(0) == ov.Type.f16 and (layer_of(o) or -1) in range(L, Lb + 1):
                d = o.get_data()
                sub_m = (d != 0) & (np.abs(d.astype(np.float32)) < 6.1035e-5)
                if sub_m.any():
                    nf += int(sub_m.sum())
                    c = ov.opset13.constant(np.where(sub_m, np.float16(0), d))
                    o.output(0).replace(c.output(0))
        print(f"  layers {L}-{Lb}: {nf} subnormal f16 constants flushed")
    sub = ov.Model(results, [x] + pasts + others, f"layers{L}-{Lb}")
    shapes = {"x": [1, 1, W]}
    for p in others:
        nm = p.output(0).get_any_name()
        shapes[nm] = {"beam_idx": [1], "attention_mask": [1, CAP + 1], "position_ids": [4, 1, 1]}.get(nm, [1, 1, W])
    for p in pasts:
        d = list(p.get_partial_shape().get_min_shape())
        d[0] = 1
        if ".key." in p.output(0).get_any_name() or ".value." in p.output(0).get_any_name():
            d[2] = CAP
        shapes[p.output(0).get_any_name()] = d
    sub.reshape(shapes)
    ppp = PrePostProcessor(sub)
    for p in pasts:
        nm = p.output(0).get_any_name()
        ppp.input(nm).tensor().set_element_type(ov.Type.f16)
        ppp.output(nm.replace(".past.", ".present.")).tensor().set_element_type(ov.Type.f16)
    print(f"  layer {L}: inputs x + {[p.output(0).get_any_name() for p in pasts + others]}")
    return ppp.build(), [(p.output(0).get_any_name(), shapes[p.output(0).get_any_name()]) for p in pasts]

rel = lambda a, b: np.sqrt(((a - b) ** 2).sum() / max((b ** 2).sum(), 1e-300))
OUT = [3994, 3456]  # outlier channels (|x| ~550), reported apart
keep = np.ones(W, bool); keep[OUT] = False
EXACT = {"DYNAMIC_QUANTIZATION_GROUP_SIZE": "0", "INFERENCE_PRECISION_HINT": "f32"}

def run(cm, X):
    req = cm.create_infer_request()
    names = {i.get_any_name() for i in cm.inputs}
    st = {nm: np.zeros(d, np.float16) for nm, d in pasts}
    Y = np.zeros((len(X), W))
    for s in range(len(X)):
        feed = {"x": X[s].reshape(1, 1, W).astype(np.float32), **st}
        if "beam_idx" in names: feed["beam_idx"] = np.zeros(1, np.int32)
        if "inputs_embeds" in names: feed["inputs_embeds"] = np.zeros((1, 1, W), np.float32)
        if "attention_mask" in names:
            mk = np.zeros((1, CAP + 1), np.int64); mk[0, :s] = 1; mk[0, CAP] = 1
            feed["attention_mask"] = mk
        if "position_ids" in names: feed["position_ids"] = np.full((4, 1, 1), s, np.int64)
        r = req.infer(feed)
        Y[s] = r["y"].ravel()
        for nm, _ in pasts:
            o = r[nm.replace(".past.", ".present.")]
            if ".key." in nm or ".value." in nm:
                # the engine's KV: a fixed past of CAP slots, the new row at
                # index CAP of present goes to slot s.
                st[nm][:, :, s] = o[:, :, CAP]
            else:
                st[nm] = o.copy()
    return Y

feed_full = feed == "full"
X_all = F if feed_full else R
print(f"{dev}{' plain' if plain else ''} {props}; {n} steps; inputs: {'the full NPU graph' if feed_full else 'the exact CPU'}'s residuals")
print("error added by layer L alone, vs the exact CPU on the same inputs: rel of the increment, of the output w/o outlier channels")
print(" L     compile  ms/step |  micro: incr   out" + ("   | in situ (full graph): incr   out" if feed_full else ""))
for L, Lb in layers:
    if L == 0:
        print(f"{L:2d}  skipped (no residual input)"); continue
    sub, pasts = cut(L, Lb)
    X = X_all[:n, L - 1]
    Yc = run(core.compile_model(sub, "CPU", EXACT), X)
    if flush:
        sub, _ = cut(L, Lb, True)
    t = time.time()
    cm = core.compile_model(sub, dev, props)
    tc = time.time() - t
    t = time.time()
    Yd = run(cm, X)
    ti = (time.time() - t) / n
    line = f"{L:2d}-{Lb:<2d} {tc:6.1f}s  {ti * 1e3:6.1f} |        {rel(Yd - X, Yc - X):.4f} {rel(Yd[:, keep], Yc[:, keep]):.4f}"
    if feed_full:
        FY = F[:n, Lb]
        line += f"   |                       {rel(FY - X, Yc - X):.4f} {rel(FY[:, keep], Yc[:, keep]):.4f}"
    print(line, flush=True)
