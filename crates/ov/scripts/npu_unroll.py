"""Prefill on the NPU without the delta-net Loop.

The NPU compiler rejects the static stateless graph for S > 1 positions
("SetupLocationVerifier: Found 24*S duplicated names"; S = 1 compiles). Each
of the 48 Loops is the recurrent gated delta rule, one position per
iteration, so for a static S it is replaced by S explicit steps:

    S <- S * exp(g_t);  d = (v_t - k_t^T S) * beta_t;  S <- S + k_t d^T;  o_t = q_t^T S

Compares the unrolled graph on DEV (default NPU) against the Loop graph on
the CPU for one S-position step from a zero state, and times it.

python npu_unroll.py S P [KEY=VALUE ...]   ($OV_EXPORT, $DEV, $QSCALE)"""
import os, sys, time
import numpy as np
import openvino as ov
import openvino.opset13 as op
from openvino._offline_transformations import stateful_to_stateless_transformation
from openvino.preprocess import PrePostProcessor

S, P = int(sys.argv[1]), int(sys.argv[2])
props = dict(a.split("=", 1) for a in sys.argv[3:])
export = os.environ.get("OV_EXPORT") or sys.exit("OV_EXPORT: the OpenVINO export")
DEV = os.environ.get("DEV", "NPU")
QSCALE = float(os.environ.get("QSCALE", "64"))
core = ov.Core()


def c(v, t=np.int64):
    return op.constant(np.array(v, t))


def scale_q(m, c_):
    ops = {o.get_friendly_name(): o for o in m.get_ops()}
    for name, o in ops.items():
        if not name.endswith("linear_attn/aten::mul/Multiply_3"):
            continue
        base = name[: -len("aten::mul/Multiply_3")]
        q = op.multiply(o.output(0), c(c_, np.float32))
        for t in list(o.output(0).get_target_inputs()):
            if t.get_node().get_type_name() == "Loop":
                t.replace_source_output(q.output(0))
        add = ops[base[:-1] + ".norm/aten::add/Add"]
        for i in range(2):
            src = add.input_value(i).get_node()
            if src.get_type_name() == "Constant":
                add.input(i).replace_source_output(c(src.get_data().astype(np.float32) * c_ * c_, np.float32).output(0))


def unroll(m, s):
    loops = [o for o in m.get_ops() if o.get_type_name() == "Loop"]
    ax2 = c(2)
    for l in loops:
        iv = l.input_values()
        # Loop inputs: 0 trip count, 1 condition, 2 q, 3 k, 4 v, 5 g, 6 beta (all
        # sliced on axis 2), 7 state (merged), 8 output buffer.
        q, k, v, g, beta, st = iv[2], iv[3], iv[4], iv[5], iv[6], iv[7]
        outs = []
        for t in range(s):
            i = c([t])
            qt, kt, vt = (op.gather(x, i, ax2) for x in (q, k, v))  # [B,48,1,128]
            gt = op.unsqueeze(op.gather(g, i, ax2), c(-1))  # [B,48,1,1]
            bt = op.unsqueeze(op.gather(beta, i, ax2), c(-1))
            st = op.multiply(st, op.exp(gt))
            kv = op.matmul(kt, st, False, False)  # [B,48,1,128]
            d = op.multiply(op.subtract(vt, kv), bt)
            st = op.add(st, op.matmul(kt, d, True, False))  # [B,48,128,128]
            outs.append(op.matmul(qt, st, False, False))
        out = op.concat(outs, 2) if s > 1 else outs[0]
        # Keep the tensor names (the SSM state's is `cache_params.present.ssm.N`).
        out.output(0).get_tensor().add_names(l.output(0).get_names())
        st.output(0).get_tensor().add_names(l.output(1).get_names())
        for t in list(l.output(0).get_target_inputs()):
            t.replace_source_output(out.output(0))
        for t in list(l.output(1).get_target_inputs()):
            t.replace_source_output(st.output(0))
    m.validate_nodes_and_infer_types()
    return len(loops)


def build(unrolled):
    m = core.read_model(os.path.join(export, "openvino_language_model.xml"))
    stateful_to_stateless_transformation(m)
    shapes = {}
    for i in m.inputs:
        n = i.get_any_name()
        if n == "inputs_embeds":
            shapes[n] = [1, S, 5120]
        elif n == "attention_mask":
            shapes[n] = [1, P + S]
        elif n == "position_ids":
            shapes[n] = [4, 1, S]
        else:
            d = [x.get_length() if x.is_static else -1 for x in i.get_partial_shape()]
            d[0] = 1
            shapes[n] = [P if x == -1 else x for x in d]
    m.reshape(shapes)
    if unrolled:
        if QSCALE:
            scale_q(m, QSCALE)
        print("unrolled", unroll(m, S), "loops", flush=True)
    ppp = PrePostProcessor(m)
    # Match on every name: an unrolled state node carries the Add's own name
    # too, and get_any_name() may return that one.
    state = lambda names: any(n.startswith(("past_key_values", "present", "cache_params")) for n in names)
    for k, i in enumerate(m.inputs):
        if state(i.get_names()):
            ppp.input(k).tensor().set_element_type(ov.Type.f16)
    for k, o in enumerate(m.outputs):
        if state(o.get_names()):
            ppp.output(k).tensor().set_element_type(ov.Type.f16)
    return ppp.build()


emb = core.compile_model(os.path.join(export, "openvino_text_embeddings_model.xml"), "CPU")
ids = (np.arange(S, dtype=np.int64) * 7919 + 1000) % 150000
x = emb({"input": ids[None]})[0].astype(np.float32)


def run(m, dev, p):
    t = time.time()
    cm = core.compile_model(m, dev, p)
    print(f"{dev}: compiled in {time.time() - t:.1f}s", flush=True)
    r = cm.create_infer_request()
    feed = {}
    for i in cm.inputs:
        n = i.get_any_name()
        sh = list(i.get_shape())
        if n == "attention_mask":
            a = np.zeros(sh, np.int64); a[0, P:] = 1; feed[n] = a
        elif n == "position_ids":
            feed[n] = np.zeros(sh, np.int64) + np.arange(S)
        elif n == "inputs_embeds":
            feed[n] = x.reshape(sh)
        elif n == "beam_idx":
            feed[n] = np.zeros(sh, np.int32)
        else:
            feed[n] = np.zeros(sh, np.float16 if i.get_element_type() == ov.Type.f16 else np.float32)
    for k in range(3):
        t = time.time()
        r.infer(feed)
        print(f"{dev} infer {k}: {1000 * (time.time() - t):.1f} ms", flush=True)
    out = {"logits": r.get_tensor("logits").data.astype(np.float64).copy()}
    for o in cm.outputs:
        n = o.get_any_name()
        n = next((x for x in o.get_names() if x.startswith("cache_params.present.ssm")), n)
        if n.startswith("cache_params.present.ssm") and n.endswith((".0", ".47")):
            out[n] = r.get_tensor(n).data.astype(np.float64).copy()
    return out


ref = run(build(False), "CPU", {})
npu_props = {"NPU_USE_NPUW": "YES", "NPUW_ONLINE_PIPELINE": "REP", "NPUW_FOLD": "YES", "NPUW_DQ": "YES", "NPUW_FUNCALL_ASYNC": "YES"} if DEV == "NPU" else {}
npu_props["CACHE_DIR"] = os.environ.get("OV_CACHE", "ovcache_py")
npu_props.update(props)
got = run(build(True), DEV, npu_props)
for k in ref:
    u, w = ref[k].ravel(), got[k].ravel()
    rel = np.sqrt(((u - w) ** 2).mean()) / max(np.sqrt((u ** 2).mean()), 1e-30)
    print(f"{k:40s} rel_rms {rel:.4f}")
a, b = ref["logits"][0], got["logits"][0]
for j in range(S):
    x_, y_ = a[j], b[j]
    px = np.exp(x_ - x_.max()); px /= px.sum(); py = np.exp(y_ - y_.max()); py /= py.sum()
    print(j, "top", int(x_.argmax()), int(y_.argmax()), "KL %.4f" % (px * (np.log(px + 1e-300) - np.log(py + 1e-300))).sum())
