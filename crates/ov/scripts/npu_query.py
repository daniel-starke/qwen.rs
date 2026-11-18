"""Make the LM stateless + static (seq S, past P) and ask the NPU which ops it supports."""
import os
import sys, time, collections
import openvino as ov
from openvino._offline_transformations import stateful_to_stateless_transformation

S = int(sys.argv[1]) if len(sys.argv) > 1 else 1
P = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
core = ov.Core()
t = time.time()
m = core.read_model(os.path.join(os.environ.get("OV_EXPORT") or sys.exit("OV_EXPORT: the OpenVINO export"), "openvino_language_model.xml"))
stateful_to_stateless_transformation(m)
print(f"read+stateless {time.time()-t:.1f}s")
shapes = {}
for i in m.inputs:
    n = i.get_any_name()
    ps = i.get_partial_shape()
    if n == "inputs_embeds":
        shapes[n] = [1, S, 5120]
    elif n == "attention_mask":
        shapes[n] = [1, P + S]
    elif n == "position_ids":
        shapes[n] = [4, 1, S]
    elif n == "beam_idx":
        shapes[n] = [1]
    else:
        d = [x.get_length() if x.is_static else -1 for x in ps]
        d[0] = 1
        d = [P if x == -1 else x for x in d]
        shapes[n] = d
seen = collections.Counter()
for n, s in shapes.items():
    k = n.split(".")[2] if n.startswith("past") or "cache" in n else n
    if seen[k] < 1:
        print("  in", n, s)
    seen[k] += 1
print("  inputs per kind", dict(seen))
for o in list(m.outputs)[:6]:
    print("  out", o.get_any_name(), o.get_partial_shape())
m.reshape(shapes)
print("reshaped; outputs:", [(o.get_any_name(), o.get_partial_shape()) for o in list(m.outputs)[:4]])
t = time.time()
sup = core.query_model(m, "NPU")
print(f"query {time.time()-t:.1f}s")
uns = collections.Counter()
for op in m.get_ordered_ops():
    if op.get_friendly_name() not in sup:
        uns[op.get_type_name()] += 1
print("unsupported on NPU:", dict(uns))
