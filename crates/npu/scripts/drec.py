"""Rewrite the unrolled delta-net recurrence of a decode / verify cut graph
(`qwen_rs.unroll.*{i}`, one step per position) into the form the NPU runs
fastest, with the recurrent state stored **transposed** (St[h, v, k]):

    kv   = (k @ St^T) * exp(g)        (decay on the small mat-vec result)
    St'  = St * exp(g) + delta^T k    (outer product as a K = 1 MatMul)
    o    = q @ St'^T

Every MatMul then takes its data operand as `transpose_b` (no state
permutes). Decode step 1.88 -> 1.09 ms (48 heads, isolated). The graph's
state input and outputs keep their names and shapes; their *layout* changes,
so prefill (`build_delta.py --sttrans`), decode and verify graphs must all
use it together.

    python drec.py IN.xml OUT.xml"""
import sys
import openvino as ov
import openvino.opset13 as op

inp, outp = sys.argv[1:3]
m = ov.Core().read_model(inp)
by = {o.get_friendly_name().split("/")[-1]: o for o in m.get_ordered_ops()}
n = 0
while f"qwen_rs.unroll.kv{n}" in by:
    n += 1
if n == 0:
    sys.exit("no qwen_rs.unroll.kv0: not a delta-net decode graph")
for i in range(n):
    exp_, dec, kv, kd, st, out = (by[f"qwen_rs.unroll.{nm}{i}"] for nm in ("exp", "decay", "kv", "kd", "state", "out"))
    delta = by[f"qwen_rs.unroll.delta{i}"]
    S = dec.input_value(0)          # previous state (St layout)
    eg = exp_.output(0)
    k = kv.input_value(0)
    q = out.input_value(0)
    kvn = op.multiply(op.matmul(k, S, False, True), eg)
    kv.output(0).replace(kvn.output(0))
    Sn = op.add(op.multiply(S, eg), op.matmul(delta.output(0), kd.input_value(0), True, False))
    st.output(0).replace(Sn.output(0))
    on = op.matmul(q, Sn, False, True)
    out.output(0).replace(on.output(0))
    Sn.set_friendly_name(f"qwen_rs.unroll.state{i}")
m.validate_nodes_and_infer_types()
ov.save_model(m, outp, compress_to_fp16=False)
print(f"{n} steps rewritten -> {outp}")
