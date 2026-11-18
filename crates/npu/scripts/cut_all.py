"""Cut every decoder layer (and the norm+lm_head tail) of the engine's
stateless decode IR into standalone static IRs, reading the 14 GB model once.

    python cut_all.py STATELESS.xml OUTDIR [--export DIR] [--cap 4096] [--s 1]
                      [--layers A-B] [--no-head]

Writes OUTDIR/layer{0..63}.xml (+ .bin) and OUTDIR/head.xml. Each layer L:
Parameter `x` [1,1,5120] f16 (inputs_embeds for L=0, else layer L-1's residual)
-> Result `y` + the layer's conv/ssm or key/value present states (f16).
head: layer 63's residual -> `logits`. Static shapes, constant-folded once.
Feed them with `npubench chain`/a token runner, states on the host.
"""
import os, sys, re
import openvino as ov
from openvino.preprocess import PrePostProcessor

a = sys.argv[1:]
def opt(name, d, conv=str):
    if name in a:
        i = a.index(name); v = a[i + 1]; del a[i:i + 2]; return conv(v)
    return d
export = opt("--export", os.environ.get("OV_EXPORT")) or sys.exit("--export DIR or OV_EXPORT: the OpenVINO export")
CAP = opt("--cap", 4096, int)
S = opt("--s", 1, int)
span = opt("--layers", "0-63")
LA, LB = (int(v) for v in span.split("-"))
nohead = "--no-head" in a
a = [v for v in a if v != "--no-head"]
xml, outdir = a
W = 5120
os.makedirs(outdir, exist_ok=True)
LAY = re.compile(r"language_model\.layers\.(\d+)[./]")
def layer_of(node):
    m = LAY.search(node.get_friendly_name())
    return int(m.group(1)) if m else None

core = ov.Core()
base = core.read_model(xml, export + r"\openvino_language_model.bin")
# One static reshape + constant fold for the whole model (the expensive part).
full = {}
for p in base.inputs:
    nm = p.get_any_name()
    ps = p.get_partial_shape()
    if nm.startswith("cache_params.past."):
        d = list(ps.get_min_shape()); d[0] = 1
        if ".key." in nm or ".value." in nm:
            d[2] = CAP
        full[nm] = d
    elif nm in ("beam_idx", "attention_mask", "position_ids", "inputs_embeds") or ps.is_dynamic:
        full[nm] = {"beam_idx": [1], "attention_mask": [1, CAP + S], "position_ids": [4, 1, S]}.get(nm, [1, S, W])
    else:
        full[nm] = list(ps.get_shape())
base.reshape(full)
from openvino._offline_transformations import apply_moc_transformations
apply_moc_transformations(base, cf=True)
print("folded; cutting", flush=True)

def save(sub, pasts, name):
    shapes = {"x": [1, S, W]}
    for p in sub.inputs:
        nm = p.get_any_name()
        if nm == "x":
            continue
        ps = p.get_partial_shape()
        if nm in ("beam_idx", "attention_mask", "position_ids", "inputs_embeds") or ps.is_dynamic:
            shapes[nm] = {"beam_idx": [1], "attention_mask": [1, CAP + S], "position_ids": [4, 1, S]}.get(nm, [1, S, W])
        elif nm.startswith("cache_params.past."):
            d = list(ps.get_min_shape()); d[0] = 1
            if ".key." in nm or ".value." in nm:
                d[2] = CAP
            shapes[nm] = d
        else:
            shapes[nm] = list(ps.get_shape())
    sub.reshape(shapes)
    ppp = PrePostProcessor(sub)
    ppp.input("x").tensor().set_element_type(ov.Type.f16)
    for o in sub.outputs:
        if o.get_any_name() == "y":
            ppp.output("y").tensor().set_element_type(ov.Type.f16)
    for nm in pasts:
        ppp.input(nm).tensor().set_element_type(ov.Type.f16)
        ppp.output(nm.replace(".past.", ".present.")).tensor().set_element_type(ov.Type.f16)
    sub = ppp.build()
    ov.save_model(sub, f"{outdir}\\{name}.xml", compress_to_fp16=False)

def reach_params(out):
    seen, stack, reach = set(), [out], []
    while stack:
        nd = stack.pop()
        if id(nd) in seen:
            continue
        seen.add(id(nd))
        if nd.get_type_name() == "Parameter":
            reach.append(nd); continue
        stack.extend(v.get_node() for v in nd.input_values())
    return reach

for L in range(LA, LB + 1):
    m = base.clone()
    ops = {o.get_friendly_name(): o for o in m.get_ops()}
    out = ops[f"__module.model.model.language_model.layers.{L}/aten::add/Add_1"]
    x = ov.opset13.parameter([1, S, W], ov.Type.f32, name="x")
    x.output(0).get_tensor().set_names({"x"})
    if L == 0:
        src = next(i for i in m.inputs if i.get_any_name() == "inputs_embeds")
        for t in list(src.get_target_inputs()):
            if layer_of(t.get_node()) in (None, 0):
                t.replace_source_output(x.output(0))
    else:
        prev = ops[f"__module.model.model.language_model.layers.{L - 1}/aten::add/Add_1"]
        for t in list(prev.output(0).get_target_inputs()):
            if layer_of(t.get_node()) in (None, L):
                t.replace_source_output(x.output(0))
    reach = reach_params(out)
    pasts = [p.output(0).get_any_name() for p in reach if p.output(0).get_any_name().startswith("cache_params.past.")]
    results = [ov.opset13.result(out.output(0))]
    results[0].output(0).get_tensor().set_names({"y"})
    for pn in pasts:
        presentn = pn.replace(".past.", ".present.")
        r = next(r for r in m.get_results() if presentn in r.output(0).get_names())
        rr = ov.opset13.result(r.input_value(0))
        rr.output(0).get_tensor().set_names({presentn})
        results.append(rr)
    # The present states can reach inputs the residual does not (the padded
    # graph's conv-state gather, `qwen_rs.conv_keep`).
    reach = []
    for r in results:
        reach += [p for p in reach_params(r.input_value(0).get_node()) if p not in reach]
    others = [p for p in reach if p.output(0).get_any_name() not in pasts and p.get_friendly_name() != "x"]
    others_p = [p for p in m.get_parameters() if p.output(0).get_any_name() in pasts] + others
    sub = ov.Model(results, [x] + others_p, f"layer{L}")
    save(sub, pasts, f"layer{L}")
    print(f"layer{L}: {len(pasts)} states, +{len(others)} other", flush=True)

# head: layer 63 residual -> logits
if nohead:
    raise SystemExit(0)
m = base.clone()
ops = {o.get_friendly_name(): o for o in m.get_ops()}
log = next(r for r in m.get_results() if r.output(0).get_any_name() == "logits")
prev = ops["__module.model.model.language_model.layers.63/aten::add/Add_1"]
x = ov.opset13.parameter([1, S, W], ov.Type.f32, name="x")
x.output(0).get_tensor().set_names({"x"})
for t in list(prev.output(0).get_target_inputs()):
    t.replace_source_output(x.output(0))
sub = ov.Model([log], [x], "head")
ppp = PrePostProcessor(sub)
ppp.input("x").tensor().set_element_type(ov.Type.f16)
sub = ppp.build()
ov.save_model(sub, f"{outdir}\\head.xml", compress_to_fp16=False)
print("head: -> logits", flush=True)
