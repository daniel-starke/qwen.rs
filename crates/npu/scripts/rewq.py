"""Rewrite the weight MatMuls of a decode cut graph (`cut_all.py`,
DIR/layer{L}.xml, int4 g128 constants of the export) to the requantized set
the lean prefill graphs use (`wq.py` GPTQ codes, CODES/L{L}.{name}.npz):
per-row int4 of W R, with the activation rotated in the graph by the same
block Hadamard R (`quantrot.block_factors(k, 256)`, x R (W R)^T = x W^T).
One weight set then serves prefill, verify and decode.

    python rewq.py IN.xml OUT.xml CODES [--src BF16DIR] [--rotb 256]

Fused prefill matrices are split back per module: delta `proj` = in_proj_qkv
| in_proj_z (in_proj_a / in_proj_b, the last 96 rows, are int8 from BF16:
`wq.TAIL8`), attention `proj` = q rows of every head | gate rows of every
head | k | v, `gu` = gate | up."""
import os
import re
import sys
import numpy as np
import openvino as ov
import openvino.opset13 as op
from quantrot import block_factors, rotate_any, rotate_graph_any
import wq
import bf16src

a = sys.argv[1:]


def opt(name, d, conv=str):
    if name in a:
        i = a.index(name)
        v = a[i + 1]
        del a[i:i + 2]
        return conv(v)
    return d


SRC = opt("--src", os.environ.get("NPU_SRC")) or sys.exit("--src DIR or NPU_SRC: the BF16 checkpoint")
ROTB = opt("--rotb", 256, int)
# `--float`: BF16 weights as f16 constants, no rotation (the accuracy truth)
FLOAT = "--float" in a
a = [v for v in a if v != "--float"]
inp, outp, codes = a
L = int(re.search(r"layer(\d+)\.xml", inp.replace("\\", "/")).group(1))
bf16src.SRC = SRC
core = ov.Core()
m = core.read_model(inp)


def load(name):
    z = np.load(os.path.join(codes, f"L{L}.{name}.npz"))
    return wq.unpack4(z["q"]), z["s"]


cache = {}


def rows(name, lo, hi):
    if name not in cache:
        cache[name] = load(name)
    q, s = cache[name]
    return q[lo:hi], s[lo:hi]


def attn_q_rows(part):
    """Rows of the fused attention `proj` holding q_proj's rows in q_proj's
    order (per head: 256 q, 256 gate)."""
    idx = np.empty(24 * 512, np.int64)
    for h in range(24):
        idx[h * 512:h * 512 + 256] = h * 256 + np.arange(256)
        idx[h * 512 + 256:h * 512 + 512] = 24 * 256 + h * 256 + np.arange(256)
    return idx


def codes_for(mod):
    """(q int8 [n, k] or None, scale [n, 1], int8?) for module `mod`."""
    if mod == "linear_attn.in_proj_qkv":
        return rows("proj", 0, 10240)
    if mod == "linear_attn.in_proj_z":
        return rows("proj", 10240, 16384)
    if mod in ("linear_attn.in_proj_a", "linear_attn.in_proj_b"):
        return None
    if mod in ("linear_attn.out_proj", "self_attn.o_proj"):
        return rows("out", 0, None)
    if mod == "mlp.gate_proj":
        return rows("gu", 0, 17408)
    if mod == "mlp.up_proj":
        return rows("gu", 17408, None)
    if mod == "mlp.down_proj":
        return rows("down", 0, None)
    if mod == "self_attn.q_proj":
        q, s = rows("proj", 0, None)
        i = attn_q_rows(0)
        return q[i], s[i]
    if mod == "self_attn.k_proj":
        return rows("proj", 24 * 512, 24 * 512 + 1024)
    if mod == "self_attn.v_proj":
        return rows("proj", 24 * 512 + 1024, 24 * 512 + 2048)
    raise SystemExit(f"no rule for {mod}")


MOD = re.compile(r"layers\.\d+\.([a-z_]+\.[a-z_]+)/ov_ext::linear/MatMul$")
n_rw = 0
for o in list(m.get_ordered_ops()):
    if o.get_type_name() != "MatMul":
        continue
    mm = MOD.search(o.get_friendly_name())
    if not mm:
        continue
    mod = mm.group(1)
    assert o.get_attributes()["transpose_b"], mod
    wout = o.input_value(1)
    n, k = (int(d) for d in wout.get_shape())
    f = block_factors(k, ROTB)
    if FLOAT:
        wf = bf16src.layer_weight(L, mod + "/").reshape(n, k).astype(np.float16)
        w = op.constant(wf)
        if wout.get_element_type() != ov.Type.f16:
            w = op.convert(w, wout.get_element_type())
        o.input(1).replace_source_output(w.output(0))
        n_rw += 1
        continue
    c = codes_for(mod)
    if c is None:
        # int8 per row, round to nearest, from BF16 (rotated like the rest)
        wr = rotate_any(bf16src.layer_weight(L, mod + "/").reshape(n, k).astype(np.float32), f)
        q, s = wq.quant_i8(wr)
        wi = op.convert(op.constant(q), ov.Type.f16)
        w = op.multiply(wi, op.constant(s.astype(np.float16)))
    else:
        q, s = c
        assert q.shape == (n, k), (mod, q.shape, n, k)
        # i4 constant: OpenVINO takes packed bytes for 4-bit types
        t = ov.Tensor(ov.Type.i4, ov.Shape([n, 1, k]))
        np.copyto(np.asarray(t.data).reshape(-1), wq.pack4(q).reshape(-1))
        ci = ov.op.Constant(t)
        w = op.multiply(op.convert(ci, ov.Type.f16), op.constant(s.reshape(n, 1, 1).astype(np.float16)))
        w = op.reshape(w, op.constant(np.array([n, k], np.int64)), False)
    et = wout.get_element_type()
    if et != ov.Type.f16:
        w = op.convert(w, et)
    x = o.input_value(0)
    xt = x.get_element_type()
    xr = rotate_graph_any(op, x, f, np.float32 if xt == ov.Type.f32 else np.float16)
    o.input(0).replace_source_output(xr.output(0) if hasattr(xr, "output") else xr)
    o.input(1).replace_source_output(w.output(0))
    n_rw += 1
m.validate_nodes_and_infer_types()
ov.save_model(m, outp, compress_to_fp16=False)
print(f"layer {L}: {n_rw} MatMuls rewritten -> {outp}")
