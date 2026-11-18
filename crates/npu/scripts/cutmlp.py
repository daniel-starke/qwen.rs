"""Cut the MLP off a decode / verify layer graph (`bindw.py` form: int4
weight inputs), so the CPU cores run it while the NPU keeps the mixer: the
MLP's three matrices are 70 % of a layer's weight bytes and decode is bound
by reading them, at ~36 GB/s on the NPU and ~58 GB/s on the cores.

    python cutmlp.py IN.xml OUT.xml WDIR [--check]

OUT.xml: the graph up to the residual before the MLP (`y` = h_mid, the MLP's
weight inputs removed). OUT.mlp (beside it): what the engine's CPU MLP
(`crates/npu/src/mlp.rs`) needs besides the codes; a text line

    mlp1 eps=E gate=w16,w18 up=w20,w22 down=w24 wdir=WDIR had=0|1

then f32 arrays: norm weight [5120], d1 [5120], H1 [256 x 256], d2, H2,
d3 [17408], H3, gate row scales [17408], up row scales, down row scales
[5120]. The MLP is

    n = x * rsqrt(mean(x^2) + eps) * nw
    g = W_gate rot1(n * d1) * sg,  u = W_up rot2(n * d2) * su
    y = x + W_down rot3(silu(g) * u * d3) * sd

with rot(v) = v as [blocks, 256] @ H. The codes are WDIR's PLAIN (not
swizzled) `lean{L}.<name>.in` files. `had=1`: every H is the Sylvester
Hadamard matrix / 16 (the engine then uses a fast transform).
`--check`: the cut graph + this MLP in numpy against IN.xml on the CPU."""
import os
import re
import sys
import numpy as np
import openvino as ov
import openvino.opset13 as op

a = sys.argv[1:]
CHECK = "--check" in a
a = [v for v in a if v != "--check"]
inp, outp, wdir = a
L = int(re.search(r"layer(\d+)\.xml$", inp.replace("\\", "/")).group(1))
core = ov.Core()
m = core.read_model(inp)


def src(node, i):
    return node.input_value(i).get_node()


def is_const(n):
    return n.get_type_name() == "Constant"


def cval(n):
    return np.asarray(n.get_data(), np.float32)


def split_const(node):
    """(the non-constant input, the constant's values) of a binary op."""
    x, y = src(node, 0), src(node, 1)
    return (y, cval(x)) if is_const(x) else (x, cval(y))


def fc(node):
    """A scaled int4 FC ending in `node`: (weight input name, row scales, data input)."""
    scale = None
    if node.get_type_name() == "Multiply":
        node, c = split_const(node)
        scale = c.reshape(-1).astype(np.float64)
    assert node.get_type_name() == "MatMul", node.get_type_name()
    data, w = src(node, 0), src(node, 1)
    ws = 1.0
    while w.get_type_name() != "Parameter":
        t = w.get_type_name()
        if t == "Convert":
            w = src(w, 0)
        elif t == "Multiply":
            w, c = split_const(w)
            ws = ws * c.reshape(-1).astype(np.float64)
        elif t == "Reshape":
            w = src(w, 0)
        else:
            raise SystemExit(f"weight chain: {t}")
    n = w.get_output_shape(0)[0]
    s = np.ones(n) * ws
    if scale is not None:
        s = s * scale
    return w.get_friendly_name(), s.astype(np.float32), data


def fcs(node):
    """One FC or a Concat of them: (names, scales, data input)."""
    parts = [src(node, i) for i in range(node.get_input_size())] if node.get_type_name() == "Concat" else [node]
    names, scales, data = [], [], None
    for p in parts:
        nm, s, d = fc(p)
        names.append(nm)
        scales.append(s)
        assert data is None or data is d
        data = d
    return names, np.concatenate(scales), data


def rot(node):
    """Reshape(MatMul(Reshape(Multiply(v, d)), H)): (v, d, H)."""
    assert node.get_type_name() == "Reshape", node.get_type_name()
    mm = src(node, 0)
    assert mm.get_type_name() == "MatMul"
    h = cval(src(mm, 1))
    assert not mm.get_attributes()["transpose_b"]
    r = src(mm, 0)
    assert r.get_type_name() == "Reshape"
    v, d = split_const(src(r, 0))
    return v, d.reshape(-1), h


res = [r for r in m.get_results() if "y" in r.input_value(0).get_names()][0]
add = src(res, 0)
while add.get_type_name() == "Convert":
    add = src(add, 0)
assert add.get_type_name() == "Add"
hmid, dn = add.input_value(0), src(add, 1)
if hmid.get_node().get_type_name() in ("Multiply", "MatMul"):
    hmid, dn = add.input_value(1), src(add, 0)
down, sd, x3 = fcs(dn)
prod, d3, H3 = rot(x3)
assert prod.get_type_name() == "Multiply"
sw, upc = src(prod, 0), src(prod, 1)
if sw.get_type_name() != "Swish":
    sw, upc = upc, sw
assert sw.get_type_name() == "Swish"
gate, sg, x1 = fcs(src(sw, 0))
up, su, x2 = fcs(upc)
n1, d1, H1 = rot(x1)
n2, d2, H2 = rot(x2)
assert n1 is n2
nrm, nw = split_const(n1)
assert nrm.get_type_name() == "Multiply"
div = src(nrm, 1) if src(nrm, 0) is hmid.get_node() else src(nrm, 0)
assert div.get_type_name() == "Divide", div.get_type_name()
sq = src(div, 1)
assert sq.get_type_name() == "Sqrt"
_, eps = split_const(src(sq, 0))
eps = float(eps.reshape(-1)[0])
nw = nw.reshape(-1)

i, j = np.arange(256)[:, None], np.arange(256)[None, :]
syl = np.where(np.vectorize(lambda v: bin(v).count("1") & 1)(i & j), -1.0, 1.0) / 16.0
had = int(all(np.allclose(h, syl, atol=1e-6) for h in (H1, H2, H3)))

mlp = os.path.splitext(outp)[0] + ".mlp"
with open(mlp, "wb") as f:
    f.write(f"mlp1 eps={eps!r} gate={','.join(gate)} up={','.join(up)} down={','.join(down)} wdir={wdir} had={had}\n".encode())
    for v in (nw, d1, H1, d2, H2, d3, H3, sg, su, sd):
        f.write(np.ascontiguousarray(v, np.float32).tobytes())

# the cut graph
ref = core.read_model(inp) if CHECK else None
y = op.convert(hmid, res.input_value(0).get_element_type())
y.output(0).get_tensor().set_names({"y"})
res.input(0).replace_source_output(y.output(0))
m.validate_nodes_and_infer_types()
for p in list(m.get_parameters()):
    if p.get_friendly_name() in gate + up + down:
        m.remove_parameter(p)
ov.save_model(m, outp, compress_to_fp16=False)
print(f"layer {L}: gate {gate} up {up} down {down} eps {eps} had {had}")

if CHECK:
    rng = np.random.default_rng(0)

    def codes(nm):
        return np.fromfile(os.path.join(wdir, f"lean{L}.{nm}.in"), np.uint8)

    def unpack(b, rows):
        lo, hi = (b & 15).astype(np.int8), (b >> 4).astype(np.int8)
        w = np.stack([lo, hi], -1).reshape(rows, -1)
        return np.where(w > 7, w - 16, w).astype(np.float32)

    feeds = {}
    for p in ref.get_parameters():
        nm, shp, et = p.get_friendly_name(), list(p.get_output_shape(0)), p.get_element_type()
        if et == ov.Type.i4:
            feeds[nm] = ov.Tensor(et, shp)
            np.frombuffer(feeds[nm].data, np.uint8)[:] = 0 if False else 0
            raw = codes(nm)
            ov_t = ov.Tensor(et, shp)
            memoryview(ov_t.data).cast("B")[:] = raw.tobytes()[: ov_t.byte_size]
            feeds[nm] = ov_t
        elif et.is_integral():
            feeds[nm] = np.zeros(shp, et.to_dtype())
        else:
            feeds[nm] = (rng.standard_normal(shp) * (1.0 if nm == "x" else 0.05)).astype(et.to_dtype())
    rr = core.compile_model(ref, "CPU").create_infer_request()
    rr.infer(feeds)
    yref = [rr.get_tensor(o).data.astype(np.float32) for o in ref.outputs if "y" in o.get_names()][0]
    cut = core.compile_model(core.read_model(outp), "CPU").create_infer_request()
    cut.infer({k: v for k, v in feeds.items() if k not in gate + up + down})
    x = [cut.get_tensor(o).data.astype(np.float32) for o in cut.model_outputs if "y" in o.get_names()][0]
    xs = x.reshape(-1, x.shape[-1])
    n = xs / np.sqrt((xs * xs).mean(-1, keepdims=True) + eps) * nw
    r = lambda v, d, H: ((v * d).reshape(v.shape[0], -1, 256) @ H).reshape(v.shape[0], -1)
    W = lambda names, rows: np.concatenate([unpack(codes(nm), rows // len(names)) for nm in names])
    g = r(n, d1, H1) @ W(gate, sg.size).T * sg
    u = r(n, d2, H2) @ W(up, su.size).T * su
    yy = xs + r(g / (1 + np.exp(-g)) * u, d3, H3) @ W(down, sd.size).T * sd
    d = np.abs(yy - yref.reshape(yy.shape)).max()
    print(f"  check: max |y - ref| {d:.3e}, max |mlp| {np.abs(yref.reshape(yy.shape) - xs).max():.3e}")
