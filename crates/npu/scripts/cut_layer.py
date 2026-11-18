"""Cut decoder layers L..Lb out of the engine's stateless decode IR and save
them as a standalone OpenVINO IR (own .bin), static shapes, f16 state I/O:
the input of `npubench ir`, which compiles it with the NPU driver through
Level Zero (no NPUW) and times it.

    python cut_layer.py STATELESS.xml L[-Lb] OUT.xml [--export DIR] [--cap 4096] [--s 1]

The residual before L becomes Parameter `x` [1, s, 5120] (f16), the residual
after Lb Result `y`; the layers' past states stay Parameters and their
present states Results (f16). Attention masks / position ids are Parameters
of their static shapes. Cut logic as in crates/ov/scripts/npu_layer.py."""
import os
import re, sys
import openvino as ov
from openvino.preprocess import PrePostProcessor

args = sys.argv[1:]
def opt(name, d, conv=str):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return conv(v)
    return d
export = opt("--export", os.environ.get("OV_EXPORT")) or sys.exit("--export DIR or OV_EXPORT: the OpenVINO export")
CAP = opt("--cap", 4096, int)
S = opt("--s", 1, int)
args = [a for a in args if a not in ("--f16mm", "--cf-only")]
xml, span, outp = args
L, Lb = (int(v) for v in (span.split("-") * 2)[:2])
W = 5120
LAY = re.compile(r"language_model\.layers\.(\d+)[./]")

def layer_of(node):
    m = LAY.search(node.get_friendly_name())
    return int(m.group(1)) if m else None

core = ov.Core()
m = core.read_model(xml, export + r"\openvino_language_model.bin")
# Static shapes + constant folding first: the export computes split sizes
# and reshapes from ShapeOf chains through earlier layers, which a cut would
# otherwise drag along.
full = {}
for p in m.inputs:
    nm = p.get_any_name()
    if nm.startswith("cache_params.past."):
        d = list(p.get_partial_shape().get_min_shape())
        d[0] = 1
        if ".key." in nm or ".value." in nm:
            d[2] = CAP
        full[nm] = d
    else:
        # Keep statically-shaped extra inputs (the chunked-prefill masks
        # qwen_rs.delta_tril_* / delta_blkB etc.) as-is; only the known
        # dynamic ones get an S-dependent shape.
        ps = p.get_partial_shape()
        if nm in ("beam_idx", "attention_mask", "position_ids", "inputs_embeds") or ps.is_dynamic:
            full[nm] = {"beam_idx": [1], "attention_mask": [1, CAP + S], "position_ids": [4, 1, S]}.get(nm, [1, S, W])
        else:
            full[nm] = list(ps.get_shape())
m.reshape(full)
if "--cf-only" in sys.argv:
    # Plain constant folding: keeps the Loop bodies' Squeeze / Unsqueeze
    # (the MOC pipeline turns them into Reshapes, which the develop
    # compiler's GatedDeltaNet fusion pattern does not match).
    from openvino.passes import Manager, ConstantFolding
    pm = Manager()
    pm.register_pass(ConstantFolding())
    pm.run_passes(m)
else:
    from openvino._offline_transformations import apply_moc_transformations
    apply_moc_transformations(m, cf=True)
ops = {o.get_friendly_name(): o for o in m.get_ops()}
out = ops[f"__module.model.model.language_model.layers.{Lb}/aten::add/Add_1"]
prev = ops[f"__module.model.model.language_model.layers.{L - 1}/aten::add/Add_1"]
x = ov.opset13.parameter([1, S, W], ov.Type.f32, name="x")
x.output(0).get_tensor().set_names({"x"})
for t in list(prev.output(0).get_target_inputs()):
    lt = layer_of(t.get_node())
    print(f"  consumer of the residual: {t.get_node().get_friendly_name()} (layer {lt})")
    if lt is None or lt >= L:
        t.replace_source_output(x.output(0))
def walk(starts):
    seen, stack, reach = set(), list(starts), []
    while stack:
        nd = stack.pop()
        if id(nd) in seen:
            continue
        seen.add(id(nd))
        if nd.get_type_name() == "Parameter":
            reach.append(nd)
            continue
        stack.extend(v.get_node() for v in nd.input_values())
    return reach
reach = walk([out])
pasts = [p for p in reach if p.output(0).get_any_name().startswith("cache_params.past.")]
results = [ov.opset13.result(out.output(0))]
results[0].output(0).get_tensor().set_names({"y"})
for p in pasts:
    pn = p.output(0).get_any_name().replace(".past.", ".present.")
    r = next(r for r in m.get_results() if pn in r.output(0).get_names())
    rr = ov.opset13.result(r.input_value(0))
    rr.output(0).get_tensor().set_names({pn})
    results.append(rr)
# The present states can reach inputs the residual does not (the padded
# graph's conv-state gather, `qwen_rs.conv_keep`).
reach = walk([r.input_value(0).get_node() for r in results])
others = [p for p in reach if p not in pasts and p.get_friendly_name() != "x"]
sub = ov.Model(results, [x] + pasts + others, f"layers{L}-{Lb}")
if "--f16mm" in sys.argv:
    # Weight MatMuls in f16: MatMul(a_f32, Convert_f32(W_f16)) ->
    # Convert_f32(MatMul(Convert_f16(a), W_f16)). Everything else (norms,
    # the recurrence) stays f32: x^2 in the RMSNorms would overflow f16.
    import openvino.opset13 as op
    n = 0
    for o in sub.get_ordered_ops():
        if o.get_type_name() != "MatMul":
            continue
        w = o.input_value(1).get_node()
        if w.get_type_name() != "Convert" or w.input_value(0).get_element_type() != ov.Type.f16:
            continue
        a16 = op.convert(o.input_value(0), "f16")
        mm = op.matmul(a16, w.input_value(0), o.get_attributes()["transpose_a"], o.get_attributes()["transpose_b"])
        back = op.convert(mm, "f32")
        for t in list(o.output(0).get_target_inputs()):
            t.replace_source_output(back.output(0))
        n += 1
    print(f"  {n} weight MatMuls moved to f16")
shapes = {"x": [1, S, W]}
for p in others:
    nm = p.output(0).get_any_name()
    ps = p.get_partial_shape()
    if nm in ("beam_idx", "attention_mask", "position_ids", "inputs_embeds") or ps.is_dynamic:
        shapes[nm] = {"beam_idx": [1], "attention_mask": [1, CAP + S], "position_ids": [4, 1, S]}.get(nm, [1, S, W])
    else:
        shapes[nm] = list(ps.get_shape())
for p in pasts:
    d = list(p.get_partial_shape().get_min_shape())
    d[0] = 1
    nm = p.output(0).get_any_name()
    if ".key." in nm or ".value." in nm:
        d[2] = CAP
    shapes[nm] = d
sub.reshape(shapes)
ppp = PrePostProcessor(sub)
ppp.input("x").tensor().set_element_type(ov.Type.f16)
ppp.output("y").tensor().set_element_type(ov.Type.f16)
for p in pasts:
    nm = p.output(0).get_any_name()
    ppp.input(nm).tensor().set_element_type(ov.Type.f16)
    ppp.output(nm.replace(".past.", ".present.")).tensor().set_element_type(ov.Type.f16)
sub = ppp.build()
ov.save_model(sub, outp, compress_to_fp16=False)
print(f"layers {L}-{Lb}: inputs {[i.get_any_name() for i in sub.inputs]}")
print(f"  outputs {[o.get_any_name() for o in sub.outputs]} -> {outp}")
