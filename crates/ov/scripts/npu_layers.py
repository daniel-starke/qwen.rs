"""Where does the NPU lose accuracy? One decode step of the static stateless LM
on the NPU and on the CPU, with extra outputs at layer residuals, the final
norm and the logits; prints the relative error and the magnitude per output.

python npu_layers.py TOKEN_ID [KEY=VALUE ...]   ($OV_EXPORT picks the export)"""
import os, sys, time
import numpy as np
import openvino as ov
from openvino._offline_transformations import stateful_to_stateless_transformation
from openvino.preprocess import PrePostProcessor

P = 64
tok = int(sys.argv[1])
props = dict(a.split("=", 1) for a in sys.argv[2:])
export = os.environ.get("OV_EXPORT") or sys.exit("OV_EXPORT: the OpenVINO export")
# $PROBES: ";"-separated op names below the language_model prefix instead.
LAYERS = [int(x) for x in os.environ.get("LAYERS", "0,1,2,3,7,15,31,47,63").split(",")]
core = ov.Core()

emb_m = core.compile_model(os.path.join(export, "openvino_text_embeddings_model.xml"), "CPU")
x = emb_m({"input": np.array([[tok]], np.int64)})[0].astype(np.float32)


def rescale_gated_norms(m, ops, c):
    """`linear_attn.norm` sees the delta-net output, small enough per head that
    `mean(x^2) + eps` is subnormal in f16. RMSNorm is scale-invariant, so feed
    it `c * x` and `c^2 * eps`: the same function, in f16's normal range."""
    import openvino.opset13 as ops13
    n = 0
    for name, op in ops.items():
        if not (name.endswith("linear_attn.norm/aten::pow/Power")):
            continue
        base = name[: -len("aten::pow/Power")]
        x = op.input_value(0)
        xs = ops13.multiply(x, ops13.constant(np.array(c, np.float32)))
        op.input(0).replace_source_output(xs.output(0))
        mul = ops[base + "aten::mul/Multiply"]
        for i in range(2):
            if mul.input_value(i).get_node() is x.get_node() and mul.input_value(i).get_index() == x.get_index():
                mul.input(i).replace_source_output(xs.output(0))
        add = ops[base + "aten::add/Add"]
        for i in range(2):
            src = add.input_value(i).get_node()
            if src.get_type_name() == "Constant":
                eps = src.get_data().astype(np.float32) * c * c
                add.input(i).replace_source_output(ops13.constant(eps).output(0))
        n += 1
    m.validate_nodes_and_infer_types()
    return n


def scale_q(m, ops, c):
    """The delta-net output is linear in q, and its only reader is the gated
    RMSNorm. Scale q by `c` and the norm's eps by `c^2`: the same function,
    with the output lifted out of f16's subnormal range (which the NPU flushes
    to zero) before the norm amplifies it."""
    import openvino.opset13 as ops13
    n = 0
    for name, op in ops.items():
        if not name.endswith("linear_attn/aten::mul/Multiply_3"):
            continue
        base = name[: -len("aten::mul/Multiply_3")]
        q = ops13.multiply(op.output(0), ops13.constant(np.array(c, np.float32)))
        for t in list(op.output(0).get_target_inputs()):
            if t.get_node().get_type_name() == "Loop":
                t.replace_source_output(q.output(0))
        add = ops[base[:-1] + ".norm/aten::add/Add"]
        for i in range(2):
            src = add.input_value(i).get_node()
            if src.get_type_name() == "Constant":
                add.input(i).replace_source_output(ops13.constant(src.get_data().astype(np.float32) * c * c).output(0))
        n += 1
    m.validate_nodes_and_infer_types()
    return n


def build():
    m = core.read_model(os.path.join(export, "openvino_language_model.xml"))
    stateful_to_stateless_transformation(m)
    ops = {o.get_friendly_name(): o for o in m.get_ops()}
    cq = float(os.environ.get("QSCALE", "0"))
    if cq:
        print("q rescaled:", scale_q(m, ops, cq), flush=True)
    c = float(os.environ.get("NORM_SCALE", "0"))
    if c:
        print("gated norms rescaled:", rescale_gated_norms(m, ops, c), flush=True)
    pre = "__module.model.model.language_model."
    if os.environ.get("PROBES"):
        probes = [pre + n for n in os.environ["PROBES"].split(";")]
    else:
        probes = [f"{pre}layers.{i}/aten::add/Add_1" for i in LAYERS]
        probes.append(pre + "norm/aten::mul/Multiply_1")
    for n in probes:
        m.add_outputs([ops[n].output(0)])
    shapes = {}
    for i in m.inputs:
        n = i.get_any_name()
        if n == "inputs_embeds":
            shapes[n] = [1, 1, 5120]
        elif n == "attention_mask":
            shapes[n] = [1, P + 1]
        elif n == "position_ids":
            shapes[n] = [4, 1, 1]
        else:
            d = [v.get_length() if v.is_static else -1 for v in i.get_partial_shape()]
            d[0] = 1
            shapes[n] = [P if v == -1 else v for v in d]
    m.reshape(shapes)
    ppp = PrePostProcessor(m)
    for i in m.inputs:
        n = i.get_any_name()
        if n.startswith("past_key_values") or n.startswith("cache_params"):
            ppp.input(n).tensor().set_element_type(ov.Type.f16)
    for o in m.outputs:
        if o.get_names() and any(n.startswith("present") or n.startswith("cache_params") for n in o.get_names()):
            ppp.output(o.get_any_name()).tensor().set_element_type(ov.Type.f16)
    return ppp.build(), probes


def run(device, p):
    m, probes = build()
    t = time.time()
    c = core.compile_model(m, device, p)
    print(f"{device}: compiled in {time.time() - t:.1f}s", flush=True)
    r = c.create_infer_request()
    feed = {}
    for i in c.inputs:
        n = i.get_any_name()
        sh = list(i.get_shape())
        if n == "attention_mask":
            a = np.zeros(sh, np.int64); a[0, P:] = 1; feed[n] = a
        elif n == "position_ids":
            feed[n] = np.zeros(sh, np.int64)
        elif n == "inputs_embeds":
            feed[n] = x.reshape(sh)
        elif n == "beam_idx":
            feed[n] = np.zeros(sh, np.int32)
        else:
            feed[n] = np.zeros(sh, np.float16 if i.get_element_type() == ov.Type.f16 else np.float32)
    r.infer(feed)
    out = {}
    for k, o in enumerate(c.outputs):
        first = len(c.outputs) - len(probes)
        if k >= first:
            out[probes[k - first]] = r.get_output_tensor(k).data.astype(np.float64).copy()
        elif "logits" in o.get_names():
            out["logits"] = r.get_output_tensor(k).data.astype(np.float64).copy()
    return out, probes


npu_props = {"NPU_USE_NPUW": "YES", "NPUW_ONLINE_PIPELINE": "REP", "NPUW_FOLD": "YES", "NPUW_DQ": "YES", "NPUW_FUNCALL_ASYNC": "YES"}
npu_props.update(props)
a, probes = run("CPU", {})
b, _ = run("NPU", npu_props)
for k in list(probes) + ["logits"]:
    u, v = a[k].ravel(), b[k].ravel()
    rel = np.sqrt(((u - v) ** 2).mean()) / max(np.sqrt((u ** 2).mean()), 1e-30)
    print(f"{k.split('language_model.')[-1]:40s} rel_rms {rel:.4f}  max|cpu| {np.abs(u).max():10.1f}  max|npu| {np.abs(v).max():10.1f}  argmax-diff {int(np.abs(u - v).argmax())}")
if os.environ.get("SAVE"):
    np.savez(os.environ["SAVE"], **{"cpu:" + k: v for k, v in a.items()}, **{"npu:" + k: v for k, v in b.items()})
