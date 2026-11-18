"""Make every delta-net Loop of a cut layer IR match the develop NPU
compiler's `FuseGDNLoop` pattern, so it becomes one fused `GatedDeltaNet`
op (a SHAVE kernel, chunk 64) instead of a Loop / our chunked matmul form.

The pattern wants the Loop's query input to be the export's literal
`q / Power(Convert(Gather(ShapeOf(q), 3)), 0.5)`; our graphs have that folded
to a constant (and the delta-net q scale multiplied in). This rewrites the
query input to `Divide(Multiply(q_in, sqrt(D)), Power(Convert(Gather(
ShapeOf(q_in), 3)), 0.5))`: the same value, in the matched shape. Run it
last (no constant folding afterwards).

    python gdn_prep.py IN.xml OUT.xml"""
import sys
import numpy as np
import openvino as ov
import openvino.opset13 as op

inp, outp = sys.argv[1:3]
m = ov.Core().read_model(inp)
n = 0
for lp in [o for o in m.get_ordered_ops() if o.get_type_name() == "Loop"]:
    q = lp.input_value(2)
    d = q.get_partial_shape()[3].get_length()
    et = q.get_element_type()
    qs = op.multiply(q, op.constant(np.array(np.sqrt(d), dtype=np.float32 if et == ov.Type.f32 else np.float16)))
    hs = op.gather(op.shape_of(q), op.constant(np.int64(3)), op.constant(np.int64(0)))
    hs = op.convert(hs, et)
    sc = op.power(hs, op.constant(np.array(0.5, dtype=np.float32 if et == ov.Type.f32 else np.float16)))
    lp.input(2).replace_source_output(op.divide(qs, sc).output(0))
    n += 1
m.validate_nodes_and_infer_types()
ov.save_model(m, outp, compress_to_fp16=False)
print(f"{n} Loops prepared -> {outp}")
