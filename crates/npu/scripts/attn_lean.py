"""Rewrite a full-attention decode / verify cut graph (KV window CAP, export
form: f32 up-convert of the whole window, Concat of the new rows, SDPA over
K / V broadcast to 24 heads, int64 attention_mask) into a lean form:

  - scores against the past window and the new rows apart (no Concat of the
    window), GQA by stacking each KV head's 6 query heads as rows;
  - an f16 additive bias `attn_bias` [1, 1, 1, CAP] for the window replaces
    `attention_mask` (the new rows are causal among themselves);
  - outputs only the new K / V rows (`present.key/value` [1, 4, S, 256]), so
    the host keeps the window as a ring buffer (keys are stored rotated:
    slot order does not matter);
  - `--vt`: the value window as Vt [1, 4, 256, CAP] (the P @ V MatMul then
    reads it as transpose_b, no permute of the window).

    python attn_lean.py IN.xml OUT.xml [--vt]"""
import sys
import numpy as np
import openvino as ov
import openvino.opset13 as op

a = sys.argv[1:]
VT = "--vt" in a
a = [v for v in a if v != "--vt"]
inp, outp = a
m = ov.Core().read_model(inp)
allops = m.get_ordered_ops()
sd = next(o for o in allops if o.get_type_name() == "ScaledDotProductAttention")
params = {p.get_friendly_name(): p for p in m.get_parameters()}
pk = next(p for n, p in params.items() if n.startswith("cache_params.past.key"))
pv = next(p for n, p in params.items() if n.startswith("cache_params.past.value"))


def window_concat(prm):
    """The Concat that appends the new rows to this past window (its first
    input is the window, possibly through a Convert)."""
    for o in allops:
        if o.get_type_name() != "Concat":
            continue
        src = o.input_value(0).get_node()
        if src.get_type_name() == "Convert":
            src = src.input_value(0).get_node()
        if src is prm or src.get_friendly_name() == prm.get_friendly_name():
            return o
    sys.exit(f"no window Concat for {prm.get_friendly_name()}")


cat = {"Concat_4": window_concat(pk), "Concat_5": window_concat(pv)}
_, HKV, CAP, D = [d.get_length() for d in pk.get_partial_shape()]
q = sd.input_value(0)  # [1, 24, S, D] f32
HQ, S = q.get_partial_shape()[1].get_length(), q.get_partial_shape()[2].get_length()
G = HQ // HKV
scale = sd.input_value(4)
f16 = ov.Type.f16
c64 = lambda v: op.constant(np.asarray(v, np.int64))
q16 = op.convert(q, f16)
# [1, 24, S, D] -> [1, 4, 6 S, D]: query head h reads KV head h // 6
qg = op.reshape(q16, c64([1, HKV, G * S, D]), False)
kn = op.convert(cat["Concat_4"].input_value(1), f16)  # [1, 4, S, D]
vn = op.convert(cat["Concat_5"].input_value(1), f16)
sc = op.convert(scale, f16)
bias = op.parameter([1, 1, 1, CAP], f16, name="attn_bias")  # the same window for every new row
sp = op.add(op.multiply(op.matmul(qg, pk, False, True), sc),
            op.reshape(op.broadcast(op.unsqueeze(bias, c64(2)), c64([1, 1, G, S, CAP])), c64([1, 1, G * S, CAP]), False))
sn = op.multiply(op.matmul(qg, kn, False, True), sc)  # [1, 4, 6 S, S]
if S > 1:
    tri = np.where(np.tril(np.ones((S, S), bool)), 0.0, -30000.0).astype(np.float16)
    sn = op.add(sn, op.constant(np.tile(tri, (G, 1)).reshape(1, 1, G * S, S)))
p = op.softmax(op.concat([sp, sn], 3), 3)
pp, pn = op.variadic_split(p, c64(3), c64([CAP, S])).outputs()
if VT:
    pvt = op.parameter([1, HKV, D, CAP], f16, name=pv.get_friendly_name())
    opast = op.matmul(pp, pvt, False, True)
else:
    pvt = pv
    opast = op.matmul(pp, pv, False, False)
o = op.add(opast, op.matmul(pn, vn, False, False))  # [1, 4, 6 S, D]
o = op.convert(op.reshape(o, c64([1, HQ, S, D]), False), sd.get_output_element_type(0))
sd.output(0).replace(o.output(0))
# results: the present windows -> the new rows only
for r in m.get_results():
    src = r.input_value(0).get_node()
    names = r.output(0).get_names()
    if src.get_type_name() == "Convert":
        src = src.input_value(0).get_node()
    if src is cat["Concat_4"]:
        r.input(0).replace_source_output(kn.output(0))
    elif src is cat["Concat_5"]:
        r.input(0).replace_source_output(vn.output(0))
newp = [p_ for p_ in m.get_parameters() if p_.get_friendly_name() != "attention_mask" and not (VT and p_ is pv)]
newp.append(bias)
if VT:
    newp.insert(newp.index(pk) + 1, pvt)
m2 = ov.Model(m.get_results(), newp, m.get_friendly_name())
m2.validate_nodes_and_infer_types()
ov.save_model(m2, outp, compress_to_fp16=False)
print(f"S={S} CAP={CAP} G={G} vt={VT}: inputs {[p_.get_friendly_name() for p_ in newp]} -> {outp}")
