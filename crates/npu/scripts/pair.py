"""Put two cut layer IRs side by side in one graph (independent inputs and
outputs, names suffixed `.a` / `.b`): does the NPU compiler overlap one's
DMA-bound ops with the other's DPU-bound matmuls?

    python pair.py A.xml B.xml OUT.xml"""
import sys
import openvino as ov

a, b, outp = sys.argv[1:4]
core = ov.Core()
params, results = [], []
for path, suf in ((a, ".a"), (b, ".b")):
    m = core.read_model(path)
    for p in m.get_parameters():
        nm = p.output(0).get_any_name() + suf
        p.set_friendly_name(nm)
        p.output(0).get_tensor().set_names({nm})
        params.append(p)
    for r in m.get_results():
        nm = r.output(0).get_any_name() + suf
        r.set_friendly_name(nm)
        r.output(0).get_tensor().set_names({nm})
        r.input_value(0).get_tensor().set_names({nm})
        results.append(r)
    for o in m.get_ordered_ops():
        if o.get_type_name() not in ("Parameter", "Result"):
            o.set_friendly_name(o.get_friendly_name() + suf)
pm = ov.Model(results, params, "pair")
ov.save_model(pm, outp, compress_to_fp16=False)
print(f"{len(params)} inputs, {len(results)} outputs -> {outp}")
