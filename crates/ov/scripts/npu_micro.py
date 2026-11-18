"""One delta-net recurrence step as a standalone graph, on a real device.

The full-model NPU graph takes 4-6 min to compile, and every probe or scale
change is a new compile. This compiles just the gated delta rule of one
linear-attention layer (48 heads, 128 x 128 state; seconds to compile) and
replays a probe dump through it step by step, the state carried on the host
in f16 between steps as the engine does. Its output is compared with an f64
replay of the same inputs, so the error is the device's arithmetic alone.

    S <- S * exp(g);  d = (v - k^T S) * beta;  S <- S + k d^T;  o = q^T S

Inputs come from a `QWEN_OV_PROBE_DUMP` of the CPU (exact) run with the probe
list of `probes.txt` (layer L's conv output, scaled query, g, beta; see
`opcmp.py` for the layout), so q already carries the graph's query scale.

python npu_micro.py PROBES.bin LOG LAYER_SLOT [--dev NPU] [--form mulsum|matmul]
       [--q EXP] [--s EXP] [--prop K=V ...]

LAYER_SLOT: index of the layer's first probe (0 for layer 0, 18 for layer 8
in the default list). --q / --s: extra power-of-two exponents on the query
and on the value (so the state); the output is scaled back on the host.
Prints the relative error of o and of the state, overall and per magnitude
band of the heads, and the error a gated RMSNorm would see."""
import re, sys, time
import numpy as np
import openvino as ov
import openvino.opset13 as op

args = sys.argv[1:]
def opt(name, d, conv=str):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return conv(v)
    return d
dev = opt("--dev", "NPU")
form = opt("--form", "mulsum")
eq = opt("--q", 0, int)
es = opt("--s", 0, int)
props = {}
while "--prop" in args:
    i = args.index("--prop"); k, v = args[i + 1].split("=", 1); props[k] = v; del args[i:i + 2]
pfile, log, slot = args[0], args[1], int(args[2])

sizes, names = {}, None
for l in open(log, encoding="utf-8", errors="replace"):
    m = re.search(r"probe qwen_rs\.probe\.(\d+) shape \[([\d, ]+)\]", l)
    if m:
        sizes[int(m.group(1))] = int(np.prod([int(x) for x in m.group(2).split(",")]))
    m = re.search(r"npu: \d+ probes: (.*)", l)
    if m:
        names = m.group(1).strip().split(",")
sz = [sizes.get(i, 5120) for i in range(len(names))]
off = np.cumsum([0] + sz)
raw = np.fromfile(pfile, np.float32)
n = len(raw) // off[-1]
A = raw[: n * off[-1]].reshape(n, off[-1]).astype(np.float64)
P = lambda i: A[:, off[i]:off[i + 1]]

H, D = 48, 128
def l2n(x):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + 1e-6)
conv = P(slot + 2)
k = np.repeat(l2n(conv[:, 2048:4096].reshape(n, 16, D)), 3, axis=1)
v = conv[:, 4096:].reshape(n, H, D)
q = P(slot + 4).reshape(n, H, D)
g = P(slot + 5).reshape(n, H)
beta = P(slot + 6).reshape(n, H)
cq, cs = 2.0 ** eq, 2.0 ** es

# --- the graph: one step, f32 I/O, the state port f16 like the engine's.
pq = op.parameter([H, D], ov.Type.f32, name="q")
pk = op.parameter([H, D], ov.Type.f32, name="k")
pv = op.parameter([H, D], ov.Type.f32, name="v")
pg = op.parameter([H, 1, 1], ov.Type.f32, name="g")
pb = op.parameter([H, 1], ov.Type.f32, name="beta")
ps = op.parameter([H, D, D], ov.Type.f16, name="s")
c32 = lambda x: op.constant(np.array(x, np.float32))
s0 = op.convert(ps, ov.Type.f32)
qx = op.multiply(pq, c32(cq)) if eq else pq
vx = op.multiply(pv, c32(cs)) if es else pv
s1 = op.multiply(s0, op.exp(pg))
k3 = op.unsqueeze(pk, op.constant(np.array([2], np.int64)))  # [H, D, 1]
if form == "mulsum":
    # the export's Loop body: elementwise products, then a ReduceSum.
    kv = op.reduce_sum(op.multiply(s1, k3), op.constant(np.array([1], np.int64)), False)  # [H, D]
else:
    kv = op.squeeze(op.matmul(op.unsqueeze(pk, op.constant(np.array([1], np.int64))), s1, False, False), op.constant(np.array([1], np.int64)))
d = op.multiply(op.subtract(vx, kv), pb)
d3 = op.unsqueeze(d, op.constant(np.array([1], np.int64)))  # [H, 1, D]
s2 = op.add(s1, op.multiply(k3, d3))
q3 = op.unsqueeze(qx, op.constant(np.array([2], np.int64)))
if form == "mulsum":
    o = op.reduce_sum(op.multiply(s2, q3), op.constant(np.array([1], np.int64)), False)
else:
    o = op.squeeze(op.matmul(op.unsqueeze(qx, op.constant(np.array([1], np.int64))), s2, False, False), op.constant(np.array([1], np.int64)))
ro = op.result(o); ro.output(0).get_tensor().set_names({"o"})
rs = op.result(op.convert(s2, ov.Type.f16)); rs.output(0).get_tensor().set_names({"s_out"})
model = ov.Model([ro, rs], [pq, pk, pv, pg, pb, ps], "delta_step")

core = ov.Core()
t = time.time()
cm = core.compile_model(model, dev, props)
print(f"{dev} {form} q x2^{eq} s x2^{es}: compiled in {time.time() - t:.1f} s")
req = cm.create_infer_request()

# --- replay: device vs f64, both from the same inputs; device state carried in f16.
S = np.zeros((H, D, D), np.float16)
Sx = np.zeros((H, D, D))
out, ref = np.zeros((n, H, D)), np.zeros((n, H, D))
for t in range(n):
    r = req.infer({"q": q[t].astype(np.float32), "k": k[t].astype(np.float32), "v": v[t].astype(np.float32),
                   "g": g[t].reshape(H, 1, 1).astype(np.float32), "beta": beta[t].reshape(H, 1).astype(np.float32), "s": S})
    out[t] = r["o"] / (cq * cs)
    S = r["s_out"]
    Sx = Sx * np.exp(g[t])[:, None, None]
    dd = (v[t] - np.einsum("hk,hkv->hv", k[t], Sx)) * beta[t][:, None]
    Sx = Sx + k[t][:, :, None] * dd[:, None, :]
    ref[t] = np.einsum("hk,hkv->hv", q[t], Sx)
    Sx = Sx.astype(np.float16).astype(np.float64)  # the port is f16 on every device

rel = lambda a, b: np.sqrt(((a - b) ** 2).sum() / max((b ** 2).sum(), 1e-300))
print(f"  o rel {rel(out, ref):.2e}   final state rel {rel(S.astype(np.float64) / cs, Sx):.2e}"
      f"   state |x|<6.1e-5: f64 {(np.abs(Sx * cs) < 6.1e-5).mean() * 100:.1f} %, device zeros {(S == 0).mean() * 100:.1f} %")
rr = np.sqrt((ref ** 2).mean(-1)).ravel()
er = np.sqrt(((out - ref) ** 2).mean(-1)).ravel()
for lo, hi in ((0, 1e-3), (1e-3, 1e-2), (1e-2, 0.1), (0.1, 1e9)):
    m = (rr >= lo) & (rr < hi)
    if m.any():
        print(f"  head rows rms [{lo:g},{hi:g}): {m.sum():5d}  median rel err {np.median(er[m] / rr[m]):.4f}")
