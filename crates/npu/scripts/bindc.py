"""Turn a graph's large weight-code constants (int4 / uint8 / int8 feeding a
decompression Convert: the lm_head, the MTP block, the draft head) into
graph INPUTS whose bytes live in WDIR/<name>.in. The engine binds an input
named `w.*` to that file's buffer (one buffer per name, shared by every graph
that names it), so the blob carries no weights: the driver compile takes
seconds instead of minutes and copies nothing, and a different weight file
(another quantization of the same shape) is tried without recompiling.

Scales and zero points stay constants (a scale as an input runs the
dequant on SHAVE: 4x slower). An
asymmetric u8 weight (the export's lm_head and MTP down_proj) is stored as
signed codes q - 128 with the row scale after the MatMul and the zero point
as a `sum(x)` term: as a plain u8 input the lm_head took 1954 ms, this way
25.5 (the constant blob: 25.5). Int4 g128 weights stay constants (--i4).

Names are content-addressed, `w.<constant name>.<sha1[:10]>`, so graphs
sharing a weight share its file and buffer. A constant whose bytes are a
prefix of a file already in WDIR (the draft head's first 32768 lm_head rows)
takes that file's name: the engine binds a buffer at least as large as the
input.

    python bindc.py IN.xml OUT.xml WDIR [--min BYTES]"""
import hashlib
import os
import re
import sys

import numpy as np
import openvino as ov
import openvino.opset13 as op

a = sys.argv[1:]
MIN = 1 << 20
if "--min" in a:
    i = a.index("--min")
    MIN = int(a[i + 1])
    del a[i:i + 2]
# int4 constants stay unless --i4: the MTP's g128 groups as inputs run
# mtp_s4 15 -> 40 ms, s1 9.5 -> 30.6 (the group dequant leaves the DPU); a
# group-major batched MatMul over the 40 groups was worse (377 ms)
KEEP4 = "--i4" not in a
a = [v for v in a if v != "--i4"]
inp, outp, wdir = a[:3]
os.makedirs(wdir, exist_ok=True)
m = ov.Core().read_model(inp)

CODES = {ov.Type.i4, ov.Type.u4, ov.Type.i8, ov.Type.u8}


def raw(c):
    return np.frombuffer(bytes(c.get_data()), np.uint8)[: c.get_byte_size()]


def prefix_of(b):
    """Name of a WDIR file that starts with the bytes b, else None."""
    h = hashlib.sha1(b).digest()
    for f in sorted(os.listdir(wdir)):
        p = os.path.join(wdir, f)
        if f.endswith(".in") and os.path.getsize(p) > b.size:
            with open(p, "rb") as fh:
                if hashlib.sha1(fh.read(b.size)).digest() == h:
                    return f[:-3]
    return None


def only(out):
    t = [x.get_node() for x in out.get_target_inputs()]
    return t[0] if len(t) == 1 else None


def other(node, src):
    """The input of a binary node that is not `src`."""
    i = 1 if node.input_value(0).get_node() is src else 0
    return node.input_value(i).get_node()


def zp_chain(c):
    """u8 codes -> Convert -> Subtract(Convert(zp)) -> Multiply(scale)
    [-> Convert] -> MatMul(x, w^T): (zp [n], scale [n], MatMul), else None."""
    cv = only(c.output(0))
    sub = cv and only(cv.output(0))
    if not sub or sub.get_type_name() != "Subtract":
        return None
    mul = only(sub.output(0))
    if not mul or mul.get_type_name() != "Multiply":
        return None
    nx = only(mul.output(0))
    if nx and nx.get_type_name() == "Convert":
        nx = only(nx.output(0))
    if not nx or nx.get_type_name() != "MatMul" or not nx.get_attributes()["transpose_b"]:
        return None
    zc, sc = other(sub, cv), other(mul, sub)
    if zc.get_type_name() == "Convert":
        zc = zc.input_value(0).get_node()
    if zc.get_type_name() != "Constant" or sc.get_type_name() != "Constant":
        return None
    n = c.get_output_shape(0)[0]
    zp = np.asarray(zc.get_data(), np.float32).reshape(-1)
    s = np.asarray(sc.get_data(), np.float32).reshape(-1)
    if zp.size != n or s.size != n or len(c.get_output_shape(0)) != 2:
        return None
    return zp, s, nx


def store(b, tag):
    name = f"w.{tag}.{hashlib.sha1(b).hexdigest()[:10]}"
    f = os.path.join(wdir, name + ".in")
    if not os.path.exists(f):
        pre = prefix_of(b)
        if pre is not None:
            return pre
        b.tofile(f)
    return name


c16 = lambda v: op.constant(np.asarray(v, np.float16))
params, moved = [], 0
for c in m.get_ordered_ops():
    if c.get_type_name() != "Constant" or c.get_element_type() not in CODES or c.get_byte_size() < MIN:
        continue
    if KEEP4 and c.get_element_type() in (ov.Type.i4, ov.Type.u4):
        continue
    cons = [t.get_node() for t in c.output(0).get_target_inputs()]
    if len(cons) != 1 or cons[0].get_type_name() != "Convert":
        continue
    b = raw(c)
    tag = re.sub(r"[^A-Za-z0-9_.]+", "_", c.get_friendly_name().replace("self.", "").replace("._openvino_orig_weight", ""))
    zc = zp_chain(c) if c.get_element_type() == ov.Type.u8 else None
    if zc is not None:
        # asymmetric u8 as an input dequantizes on SHAVE (lm_head 1954 ms).
        # Signed codes q - 128 with the row scale after the MatMul, as the
        # prefill's i8 weights: (q - zp) s . x = s (q' . x) - s (zp - 128) sum(x)
        zp, s, mm = zc
        b = (b.astype(np.int16) - 128).astype(np.int8).view(np.uint8)
        name = store(b, tag + ".i8")
        n, k = c.get_output_shape(0)
        prm = op.parameter([n, k], ov.Type.i8, name=name)
        prm.output(0).get_tensor().set_names({name})
        x = mm.input_value(0)
        w = op.multiply(op.convert(prm, ov.Type.f16), c16(2.0 ** -8))
        if x.get_element_type() != ov.Type.f16:
            w = op.convert(w, x.get_element_type())
        y = op.matmul(x, w, False, True)
        ft = np.float16 if x.get_element_type() == ov.Type.f16 else np.float32
        rk = len(x.get_partial_shape())
        y = op.multiply(y, op.constant((s * 2.0 ** 8).astype(ft).reshape([1] * (rk - 1) + [n])))
        sx = op.reduce_sum(x, op.constant(np.array([-1], np.int64)), True)
        y = op.subtract(y, op.multiply(sx, op.constant((s * (zp - 128)).astype(ft).reshape([1] * (rk - 1) + [n]))))
        if y.get_output_element_type(0) != mm.get_output_element_type(0):
            y = op.convert(y, mm.get_output_element_type(0))
        mm.output(0).replace(y.output(0))
        params.append(prm)
        moved += b.size
        print(f"  {c.get_friendly_name()} {[n, k]} u8 zp -> i8 {name}, scale after the MatMul")
        continue
    name = store(b, tag)
    prm = op.parameter(list(c.get_output_shape(0)), c.get_element_type(), name=name)
    prm.output(0).get_tensor().set_names({name})
    c.output(0).replace(prm.output(0))
    params.append(prm)
    moved += b.size
    print(f"  {c.get_friendly_name()} {list(c.get_output_shape(0))} {c.get_element_type()} -> {name}")

m2 = ov.Model(m.get_results(), list(m.get_parameters()) + params, m.get_friendly_name())
m2.validate_nodes_and_infer_types()
ov.save_model(m2, outp, compress_to_fp16=False)
print(f"{inp}: {len(params)} constants, {moved / 2**20:.1f} MiB -> inputs; {outp}")
