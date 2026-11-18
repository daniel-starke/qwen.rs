"""Heads for NPU speculative decoding, from the decode cut's head.xml
(final norm + u8 asymmetric lm_head):
- OUT/head.xml: the head plus a second output `hn` = the output-normed hidden
  state (f16), which the MTP block takes as its `hidden` input;
- OUT/mtphead.xml: lm_head rows [0, V) only (`--vocab`, default 32768), on an
  already normed x [1, 1, 5120] f16 (the MTP block's `h`): draft logits.

    python heads.py SRC_HEAD.xml OUTDIR [--vocab 32768]"""
import sys
import numpy as np
import openvino as ov
import openvino.opset13 as op

a = sys.argv[1:]
V = int(a[a.index("--vocab") + 1]) if "--vocab" in a else 32768
src, outd = a[0], a[1]
core = ov.Core()
m = core.read_model(src)
ops = {o.get_friendly_name(): o for o in m.get_ordered_ops()}
norm = next(o for n, o in ops.items() if n.endswith("language_model.norm/aten::mul/Multiply_1"))
mm = next(o for n, o in ops.items() if n.endswith("lm_head/ov_ext::linear/MatMul"))
res = m.get_results()[0]
hn = op.result(op.convert(norm.output(0), "f16"))
hn.output(0).get_tensor().set_names({"hn"})
res.output(0).get_tensor().set_names({"logits"})
m2 = ov.Model([res, hn], m.get_parameters(), "head_hn")
ov.save_model(m2, f"{outd}/head.xml", compress_to_fp16=False)

w = next(o for n, o in ops.items() if n.endswith("lm_head._openvino_orig_weight")).get_data()
zp = next(o for n, o in ops.items() if n.endswith("lm_head._openvino_orig_weight/zero_point")).get_data()
sc = next(o for n, o in ops.items() if n.endswith("lm_head._openvino_orig_weight/scale")).get_data()
x = op.parameter([1, 1, 5120], ov.Type.f16, name="x")
wq = op.multiply(op.subtract(op.convert(op.constant(np.ascontiguousarray(w[:V])), "f16"),
                             op.convert(op.constant(np.ascontiguousarray(zp[:V])), "f16")),
                 op.constant(np.ascontiguousarray(sc[:V])))
lg = op.matmul(x, wq, False, True)
r = op.result(op.convert(lg, "f32"))
r.output(0).get_tensor().set_names({"logits"})
ov.save_model(ov.Model([r], [x], "mtphead"), f"{outd}/mtphead.xml", compress_to_fp16=False)
print("ok", V)
