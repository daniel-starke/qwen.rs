"""The export's MTP block (`openvino_mtp_model.xml`, stateful) as a static
stateless decode graph for the NPU: S positions, KV window CAP, f16 I/O.

    python cut_mtp.py OUT.xml [--s 1] [--cap 4096] [--export DIR]

Inputs: hidden [1, S, 5120] (the main model's output-normed hidden state at
the previous position), emb [1, S, 5120] (the embedding of the token),
attention_mask [1, CAP + S] i64, position_ids [1, S] i64, past_key /
past_value [1, 4, CAP, 256]. Outputs: h [1, S, 5120] (after the MTP block's
shared head norm: the lm_head input), present_key / present_value
[1, 4, CAP + S, 256]."""
import os
import sys
import openvino as ov
import openvino.opset13 as op
from openvino._offline_transformations import apply_moc_transformations, stateful_to_stateless_transformation

a = sys.argv[1:]
def opt(name, d, conv=str):
    if name in a:
        i = a.index(name); v = a[i + 1]; del a[i:i + 2]; return conv(v)
    return d
export = opt("--export", os.environ.get("OV_EXPORT")) or sys.exit("--export DIR or OV_EXPORT: the OpenVINO export")
S = opt("--s", 1, int)
CAP = opt("--cap", 4096, int)
out = a[0]
core = ov.Core()
m = core.read_model(export + r"\openvino_mtp_model.xml")
stateful_to_stateless_transformation(m)
shapes = {}
for p in m.get_parameters():
    n = p.get_friendly_name()
    if n == "hidden_states" or n == "inputs_embeds":
        shapes[p] = [1, S, 5120]
    elif n == "attention_mask":
        shapes[p] = [1, CAP + S]
    elif n == "position_ids":
        shapes[p] = [1, S]
    elif "past_key_values" in n:
        shapes[p] = [1, 4, CAP, 256]
m.reshape({p.output(0): ov.PartialShape(s) for p, s in shapes.items()})
apply_moc_transformations(m, cf=True)
# f16 boundaries
newp, outs = [], []
rename = {"hidden_states": "hidden", "inputs_embeds": "emb"}
for p in list(m.get_parameters()):
    n = p.get_friendly_name()
    if p.get_element_type() == ov.Type.f32:
        nn = rename.get(n, "past_key" if ".key" in n else "past_value" if ".value" in n else n)
        q = op.parameter(list(p.get_shape()), ov.Type.f16, name=nn)
        q.output(0).get_tensor().set_names({nn})
        c = op.convert(q, "f32")
        for t in list(p.output(0).get_target_inputs()):
            t.replace_source_output(c.output(0))
        newp.append(q)
    else:
        newp.append(p)
names = ["h", "present_key", "present_value"]
for r, nm in zip(m.get_results(), names):
    rr = op.result(op.convert(r.input_value(0), "f16"))
    rr.output(0).get_tensor().set_names({nm})
    outs.append(rr)
mm = ov.Model(outs, newp, f"mtp_s{S}")
mm.validate_nodes_and_infer_types()
ov.save_model(mm, out, compress_to_fp16=False)
for p in mm.get_parameters(): print("in ", p.get_friendly_name(), p.get_shape(), p.get_element_type())
for r in mm.get_results(): print("out", sorted(r.output(0).get_names()), r.get_output_shape(0))
