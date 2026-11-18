"""A graph holding only a cut layer's delta-net Loop (its q, k, v, g, beta,
state as f16 Parameters), for isolating the fused GatedDeltaNet kernel.

    python gdn_only.py LAYER.xml OUT.xml

Run `gdn_prep.py` on the result."""
import sys
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


inp, outp = a
m = ov.Core().read_model(inp)
lp = next(o for o in m.get_ordered_ops() if o.get_type_name() == "Loop")
params = []
for i, nm in enumerate(["q", "k", "v", "g", "beta", "state"]):
    v = lp.input_value(2 + i)
    p = op.parameter(list(v.get_partial_shape().get_shape()), v.get_element_type(), name=nm)
    p.output(0).get_tensor().set_names({nm})
    lp.input(2 + i).replace_source_output(p.output(0))
    params.append(p)
r0 = op.result(lp.output(0))
r0.output(0).get_tensor().set_names({"out"})
r1 = op.result(lp.output(1))
r1.output(0).get_tensor().set_names({"state_out"})
sub = ov.Model([r0, r1], params, "gdn")
sub.validate_nodes_and_infer_types()
ov.save_model(sub, outp, compress_to_fp16=False)
print(f"{[list(p.get_shape()) for p in params]} -> {outp}")
