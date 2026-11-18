"""Rewrite a decode / verify layer graph (`rewq.py` int4 constants, then
`drec.py` / `wyrec.py` / `attn_lean.py`) so its int4 weight codes are graph
INPUTS named after a prefill set's bound weight files (`lean{L}.wN.in`):
the engine then binds the prefill's host buffers, and the decode set no
longer copies ~14 GB of constants into the driver.

Each decode MatMul's i4 constant is matched byte for byte against the
prefill files' rows (whole files in row order, a row range of a file, or the
attention q rows' per-head interleave of `w0`). Per prefill file the graph
gets one MatMul over all of its rows, `x @ (w * 2^-8)^T * (s * 2^8)` with the
decode's row scales after the MatMul (as the prefill graphs apply them; a
scale before it would dequantize the weight on SHAVE), and each original
MatMul's output becomes slices / gathers of those. Small weights (the int8
in_proj_a / in_proj_b rows) stay constants.

    python bindw.py IN.xml PREFILL_DIR OUT.xml"""
import os
import re
import sys
import numpy as np
import openvino as ov
import openvino.opset13 as op

a = sys.argv[1:]
# `--post`: the row scale after the MatMul (as the prefill graphs do);
# default: on the weight before it, the decode constants' own pattern
# (i4 [n, 1, k] x s [n, 1, 1], reshaped), which the compiler turns into the
# int4 FC's dequant (s = 4 verify delta layer 7.2 ms vs 11.3-12.7 post)
POST = "--post" in a
a = [v for v in a if v != "--post"]
inp, pdir, outp = a[:3]
L = int(re.search(r"layer(\d+)\.xml$", inp.replace("\\", "/")).group(1))
m = ov.Core().read_model(inp)

# prefill weight files of layer L: name -> rows [r, rowbytes]
files = {}
for f in sorted(os.listdir(pdir)):
    mm = re.fullmatch(rf"lean{L}\.(w\d+)\.in", f)
    if mm:
        files[mm.group(1)] = np.fromfile(os.path.join(pdir, f), np.uint8)


def consumers(out):
    return [t.get_node() for t in out.get_target_inputs()]


# decode matrices: (MatMul, codes [n, k/2] bytes, scale [n], k)
mats = []
for c in m.get_ordered_ops():
    if c.get_type_name() != "Constant" or c.get_element_type() != ov.Type.i4:
        continue
    shp = list(c.get_output_shape(0))
    n, k = shp[0], shp[-1]
    node, scale = c, None
    while True:
        nx = consumers(node.output(0))
        assert len(nx) == 1, (c.get_friendly_name(), [x.get_type_name() for x in nx])
        node = nx[0]
        if node.get_type_name() == "Multiply" and scale is None:
            other = node.input_value(1).get_node() if node.input_value(0).get_node() is not c and node.input_value(0).get_node().get_type_name() == "Convert" else node.input_value(0).get_node()
            scale = np.asarray(other.get_data(), np.float32).reshape(-1)
            assert scale.size == n, (c.get_friendly_name(), scale.size, n)
        if node.get_type_name() == "MatMul":
            break
    assert node.get_attributes()["transpose_b"], c.get_friendly_name()
    codes = np.frombuffer(bytes(c.get_data()), np.uint8)[: n * k // 2].reshape(n, k // 2)
    mats.append((node, codes, scale, k))


def q_interleave():
    idx = np.empty(24 * 512, np.int64)
    for h in range(24):
        idx[h * 512:h * 512 + 256] = h * 256 + np.arange(256)
        idx[h * 512 + 256:h * 512 + 512] = 24 * 256 + h * 256 + np.arange(256)
    return idx


def rowmap(codes):
    """[(file, file rows)] covering the matrix's rows in order."""
    n, rb = codes.shape
    # whole files, contiguous, in order
    out, r = [], 0
    while r < n:
        for nm, fb in files.items():
            if fb.size % rb:
                continue
            fr = fb.reshape(-1, rb)
            if r + fr.shape[0] <= n and np.array_equal(codes[r:r + fr.shape[0]], fr):
                out.append((nm, np.arange(fr.shape[0])))
                r += fr.shape[0]
                break
        else:
            break
    if r == n:
        return out
    # a row range of one file, or the interleaved attention q rows
    for nm, fb in files.items():
        if fb.size % rb:
            continue
        fr = fb.reshape(-1, rb)
        hits = np.where((fr == codes[0]).all(1))[0]
        for h in hits:
            if h + n <= fr.shape[0] and np.array_equal(fr[h:h + n], codes):
                return [(nm, np.arange(h, h + n))]
        if n == 24 * 512 and fr.shape[0] >= n:
            idx = q_interleave()
            if np.array_equal(fr[idx], codes):
                return [(nm, idx)]
    raise SystemExit(f"layer {L}: a [{n}, {rb * 2}] matrix matches no prefill weight file")


# per file: its matrices' rows and scales; per matrix: (MatMul, parts)
use, mparts = {}, []
for mm_, codes, scale, k in mats:
    if codes.shape[0] * k < 1 << 20:
        continue  # small: keep the constant
    parts = rowmap(codes)
    r = 0
    for nm, rows in parts:
        e = use.setdefault(nm, {"k": k, "scale": np.full(files[nm].size * 2 // k, np.nan, np.float32), "x": mm_.input_value(0), "mats": []})
        assert e["k"] == k
        e["scale"][rows] = scale[r:r + len(rows)]
        r += len(rows)
    for nm, _ in parts:
        use[nm]["mats"].append(mm_)
    mparts.append((mm_, parts))

c16 = lambda v: op.constant(np.asarray(v, np.float16))
ci64 = lambda v: op.constant(np.asarray(v, np.int64))
yf, params = {}, []
for nm, e in use.items():
    assert not np.isnan(e["scale"]).any(), f"layer {L}: {nm} has rows no decode matrix reads"
    n = e["scale"].size
    prm = op.parameter([n, 1, e["k"]] if not POST else [n, e["k"]], ov.Type.i4, name=nm)
    params.append(prm)
    # the matrices sharing a file (attention q / k / v) each rotate the same
    # normed input with the same factors: the first one's serves all
    x = e["x"]
    if not POST:
        w = op.multiply(op.convert(prm, ov.Type.f16), c16(e["scale"].reshape(n, 1, 1)))
        w = op.reshape(w, ci64([n, e["k"]]), False)
        if x.get_element_type() != ov.Type.f16:
            w = op.convert(w, x.get_element_type())
        yf[nm] = op.matmul(x, w, False, True)
        continue
    w = op.multiply(op.convert(prm, ov.Type.f16), c16(2.0 ** -8))
    if x.get_element_type() != ov.Type.f16:
        w = op.convert(w, x.get_element_type())
    y = op.matmul(x, w, False, True)
    s = op.constant((e["scale"] * 2.0 ** 8).astype(np.float16 if x.get_element_type() == ov.Type.f16 else np.float32).reshape(1, 1, n) if len(x.get_partial_shape()) == 3 else (e["scale"] * 2.0 ** 8).astype(np.float32).reshape(1, n))
    yf[nm] = op.multiply(y, s)

for mm_, parts in mparts:
    pieces = []
    for nm, rows in parts:
        y = yf[nm]
        n_ = use[nm]["scale"].size
        ax = len(y.get_output_partial_shape(0)) - 1
        if len(rows) == n_ and np.array_equal(rows, np.arange(n_)):
            pieces.append(y)
        elif np.array_equal(rows, np.arange(rows[0], rows[0] + len(rows))):
            pieces.append(op.slice(y, ci64([rows[0]]), ci64([rows[0] + len(rows)]), ci64([1]), ci64([ax])))
        else:
            pieces.append(op.gather(y, ci64(rows), ci64(ax)))
    out = pieces[0] if len(pieces) == 1 else op.concat(pieces, -1)
    if out.get_output_element_type(0) != mm_.get_output_element_type(0):
        out = op.convert(out, mm_.get_output_element_type(0))
    mm_.output(0).replace(out.output(0))

m2 = ov.Model(m.get_results(), list(m.get_parameters()) + params, m.get_friendly_name())
m2.validate_nodes_and_infer_types()
ov.save_model(m2, outp, compress_to_fp16=False)
print(f"layer {L}: {len(mparts)} MatMuls -> inputs {sorted(use, key=lambda v: int(v[1:]))} -> {outp}")
