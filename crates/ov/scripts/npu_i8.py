"""Prototype: asym u4 (+u4 zero point, f16 group scale) -> exact sym i8 (q - zp), then
stateless + static + f16 state I/O, compile on NPU, time decode.
python npu_i8.py P [KEY=VALUE ...]"""
import os
import sys, time, collections
import numpy as np
import openvino as ov
import openvino.opset13 as ops
from openvino._offline_transformations import stateful_to_stateless_transformation
from openvino.preprocess import PrePostProcessor

P = int(sys.argv[1])
props = dict(a.split("=", 1) for a in sys.argv[2:])
core = ov.Core()
m = core.read_model(os.path.join(os.environ.get("OV_EXPORT") or sys.exit("OV_EXPORT: the OpenVINO export"), "openvino_language_model.xml"))


def unpack_u4(c):
    b = np.array(c.data).view(np.uint8).ravel()
    v = np.empty(b.size * 2, np.uint8)
    v[0::2] = b & 15
    v[1::2] = b >> 4
    return v.reshape(list(c.get_output_shape(0)))


t = time.time()
n = 0
seen = collections.Counter()
for op in m.get_ordered_ops():
    if op.get_type_name() != "Subtract":
        continue
    a, z = [op.input_value(i).get_node() for i in range(2)]
    if a.get_type_name() != "Convert" or z.get_type_name() != "Convert":
        seen["sub-not-convert"] += 1
        continue
    wc = a.input_value(0).get_node()
    zc = z.input_value(0).get_node()
    if wc.get_type_name() != "Constant" or zc.get_type_name() != "Constant":
        seen["not-const"] += 1
        continue
    if wc.get_element_type().get_type_name() != "u4" or zc.get_element_type().get_type_name() != "u4":
        seen["not-u4"] += 1
        continue
    w = unpack_u4(wc).astype(np.int16)
    zp = unpack_u4(zc).astype(np.int16)
    q = (w - zp).astype(np.int8)
    newc = ops.constant(q, ov.Type.i8)
    cvt = ops.convert(newc, a.get_output_element_type(0))
    # Every consumer of the Subtract now reads the converted i8 weight.
    for tin in list(op.output(0).get_target_inputs()):
        tin.replace_source_output(cvt.output(0))
    n += 1
print(f"converted {n} weights in {time.time()-t:.1f} s; skipped {dict(seen)}", flush=True)

stateful_to_stateless_transformation(m)
shapes = {}
for i in m.inputs:
    nm = i.get_any_name()
    if nm == "inputs_embeds":
        shapes[nm] = [1, 1, 5120]
    elif nm == "attention_mask":
        shapes[nm] = [1, P + 1]
    elif nm == "position_ids":
        shapes[nm] = [4, 1, 1]
    else:
        d = [x.get_length() if x.is_static else -1 for x in i.get_partial_shape()]
        d[0] = 1
        shapes[nm] = [P if x == -1 else x for x in d]
m.reshape(shapes)
ppp = PrePostProcessor(m)
for i in m.inputs:
    nm = i.get_any_name()
    if nm.startswith("past_key_values") or nm.startswith("cache_params"):
        ppp.input(nm).tensor().set_element_type(ov.Type.f16)
for o in m.outputs:
    nm = o.get_any_name()
    if nm.startswith("present") or nm.startswith("cache_params"):
        ppp.output(nm).tensor().set_element_type(ov.Type.f16)
m = ppp.build()
print("props", props, flush=True)
t = time.time()
c = core.compile_model(m, "NPU", props)
print(f"compiled in {time.time()-t:.1f}s", flush=True)
del m
r = c.create_infer_request()
feed = {}
for i in c.inputs:
    nm = i.get_any_name()
    sh = list(i.get_shape())
    if nm == "attention_mask":
        a = np.zeros(sh, np.int64); a[0, P:] = 1; feed[nm] = a
    elif nm == "position_ids":
        feed[nm] = np.zeros(sh, np.int64)
    elif nm == "inputs_embeds":
        feed[nm] = (np.random.randn(*sh) * 0.01).astype(np.float32)
    else:
        feed[nm] = np.zeros(sh, np.float16 if i.get_element_type() == ov.Type.f16 else np.float32)
for k in range(4):
    t = time.time(); pc = time.process_time()
    r.infer(feed)
    print(f"infer {k}: {1000*(time.time()-t):.1f} ms wall, {1000*(time.process_time()-pc):.1f} ms process CPU", flush=True)
lg = r.get_tensor("logits").data
print("logits", lg.shape, float(np.abs(lg).mean()), np.isfinite(lg).all())
