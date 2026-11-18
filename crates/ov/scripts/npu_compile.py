"""Compile the static stateless LM on the NPU and time one inference.
python npu_compile.py S P [KEY=VALUE ...]   ($OV_EXPORT picks the export dir)"""
import os, sys, time
import numpy as np
import openvino as ov
from openvino._offline_transformations import stateful_to_stateless_transformation

S, P = int(sys.argv[1]), int(sys.argv[2])
props = dict(a.split("=", 1) for a in sys.argv[3:])
core = ov.Core()
export = os.environ.get("OV_EXPORT") or sys.exit("OV_EXPORT: the OpenVINO export")
m = core.read_model(os.path.join(export, "openvino_language_model.xml"))
stateful_to_stateless_transformation(m)
if os.environ.get("UNIQUE_LOOP_NAMES"):
    # The NPU compiler rejects S > 1 with "SetupLocationVerifier: Found N
    # duplicated names": the 48 delta-net Loop bodies use the same op names.
    for i, op in enumerate(o for o in m.get_ops() if o.get_type_name() == "Loop"):
        for b in op.get_function().get_ops():
            b.set_friendly_name(f"{op.get_friendly_name()}/{b.get_friendly_name()}")
    print("loop body names made unique", flush=True)
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
import os
if os.environ.get("F16STATE", "1") == "1":
    from openvino.preprocess import PrePostProcessor
    ppp = PrePostProcessor(m)
    for i in m.inputs:
        n = i.get_any_name()
        if n.startswith("past_key_values") or n.startswith("cache_params"):
            ppp.input(n).tensor().set_element_type(ov.Type.f16)
    for o in m.outputs:
        n = o.get_any_name()
        if n.startswith("present") or n.startswith("cache_params"):
            ppp.output(n).tensor().set_element_type(ov.Type.f16)
    m = ppp.build()
    print("state I/O -> f16", flush=True)
print("props", props, flush=True)
t = time.time()
c = core.compile_model(m, "NPU", props)
print(f"compiled in {time.time()-t:.1f}s", flush=True)
r = c.create_infer_request()
feed = {}
for i in c.inputs:
    n = i.get_any_name()
    sh = list(i.get_shape())
    if n == "attention_mask":
        a = np.zeros(sh, np.int64); a[0, P:] = 1; feed[n] = a
    elif n == "position_ids":
        feed[n] = np.zeros(sh, np.int64) + np.arange(S)
    elif n == "inputs_embeds":
        feed[n] = (np.random.randn(*sh) * 0.01).astype(np.float32)
    else:
        feed[n] = np.zeros(sh, np.float16 if i.get_element_type() == ov.Type.f16 else np.float32)
for k in range(4):
    t = time.time(); pc = time.process_time()
    r.infer(feed)
    print(f"infer {k}: {1000*(time.time()-t):.1f} ms wall, {1000*(time.process_time()-pc):.1f} ms process CPU", flush=True)
lg = r.get_tensor("logits").data
print("logits", lg.shape, float(np.abs(lg).mean()), np.isfinite(lg).all())
