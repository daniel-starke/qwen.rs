"""Activation-quantization error of every weight MatMul on real activations:
run the exact per-layer chain (`chain.py` inputs: DIR/layer*.xml cut at S
positions, zero state) and, in the layers `--layers`, take each weight
MatMul's input X [S, k] and weight W [n, k] and compare X @ W^T against int8
activation schemes (weights exact, so only the activation error shows):

    tok      per-token max-abs int8 (what build_*.py `--act dyn` does)
    sq<A>    SmoothQuant: X / c, W * c with c_j = max|X_j|^A / max|W_j|^(1-A)
             (c from this prompt's own statistics), then per token
    g<G>     per token and per group of G input channels

    python actq.py DIR IDS [--layers 0,3,8,31,63] [--sq 0.5,0.8] [--g 128]
                   [--stats OUT.npz]

`--stats OUT.npz`: instead, record every weight MatMul's per-input-channel
max |X| over all layers (merged by max into OUT.npz if it exists); the
calibration `w8.py --smooth A --stats` reads."""
import os
import sys
import numpy as np
import openvino as ov

a = sys.argv[1:]


def opt(name, d, conv=str):
    if name in a:
        i = a.index(name)
        v = a[i + 1]
        del a[i:i + 2]
        return conv(v)
    return d


export = opt("--export", os.environ.get("OV_EXPORT")) or sys.exit("--export DIR or OV_EXPORT: the OpenVINO export")
want = {int(v) for v in opt("--layers", "0,3,8,31,63").split(",")}
sqs = [float(v) for v in opt("--sq", "0.5,0.8").split(",") if v]
gs = [int(v) for v in opt("--g", "128").split(",") if v]
stats = opt("--stats", None)
# `--cal CAL.npz --alpha A`: also the deployable scheme (SmoothQuant with c
# from calibration maxima (another text), int8 per-row weights of W * c)
# next to int8 weights alone (`w8`, f16 activations).
calf = opt("--cal", None)
alpha = opt("--alpha", 0.65, float)
cal = dict(np.load(calf)) if calf else {}
# `--rot`: per-token int8 after a Kronecker rotation (folded into W).
ROT = "--rot" in a
# `--rotb 128,256`: block-diagonal Hadamard rotations of block b instead.
ROTB = [int(v) for v in opt("--rotb", "").split(",") if v]
# `--static`: with --rotb, also one static range for all tokens
STATICQ = "--static" in a
a = [v for v in a if v != "--static"]
a = [v for v in a if v != "--rot"]
d, idsf = a
if stats:
    want = set(range(64))
st = {}
ids = np.array([int(v) for v in open(idsf).read().split()], np.int64)
S, W = len(ids), 5120
core = ov.Core()
EXACT = {"INFERENCE_PRECISION_HINT": "f32"}
emb = core.compile_model(core.read_model(export + r"\openvino_text_embeddings_model.xml"), "CPU")
x = list(emb({0: ids[None, :]}).values())[0].reshape(1, S, W).astype(np.float32)
# --- from chain.py ---
def tril(c, incl):
    return np.tril(np.ones((c, c), np.float32), 0 if incl else -1)


def blk(c, b):
    i, j = np.meshgrid(np.arange(c), np.arange(c), indexing="ij")
    return ((i // (2 * b) == j // (2 * b)) & (i // b > j // b)).astype(np.float32)


def feed_for(cm, x):
    f = {}
    for p in cm.inputs:
        nm = p.get_any_name()
        d = list(p.get_shape())
        f16 = p.get_element_type() == ov.Type.f16
        if nm == "x":
            v = x
        elif nm.startswith("cache_params.past."):
            v = np.zeros(d, np.float32)
        elif nm == "qwen_rs.delta_tril_strict":
            v = tril(d[0], False)
        elif nm == "qwen_rs.delta_tril_incl":
            v = tril(d[0], True)
        elif nm.startswith("qwen_rs.delta_blk"):
            v = blk(d[0], int(nm[len("qwen_rs.delta_blk"):]))
        elif nm == "qwen_rs.valid":
            v = np.ones(d, np.float32)
        elif nm == "qwen_rs.conv_keep":
            v = np.arange(S, S + d[0], dtype=np.int32)
        elif nm == "qwen_rs.conv_unscale":
            v = np.full(d, 2.0 ** -5, np.float32)
        elif nm == "attention_mask":
            # past slots empty (zero state), the S new ones valid
            v = np.zeros(d, np.int64)
            v[..., -S:] = 1
        elif nm == "position_ids":
            v = np.broadcast_to(np.arange(0, S, dtype=np.int64), d).copy()
        else:
            raise SystemExit(f"no rule for input {nm} {d}")
        f[nm] = v.astype(np.float16) if f16 else v
    return f


def const_only(node, seen=None):
    seen = set() if seen is None else seen
    if id(node) in seen:
        return True
    seen.add(id(node))
    t = node.get_type_name()
    if t == "Constant":
        return True
    if t == "Parameter":
        return False
    return all(const_only(v.get_node(), seen) for v in node.input_values())




def q8(t, axis=-1):
    s = np.maximum(np.abs(t).max(axis=axis, keepdims=True), 1e-8) / 127.0
    return np.clip(np.rint(t / s), -127, 127) * s


from quantrot import hadamard, rot_factors, rotate  # noqa: E402


def rel(y, r):
    return np.linalg.norm(y - r) / max(np.linalg.norm(r), 1e-30)


full = {}
for L in range(max(want) + 1):
    m = core.read_model(f"{d}\\layer{L}.xml")
    probes = []
    if L in want:
        for o in m.get_ordered_ops():
            if o.get_type_name() == "MatMul" and const_only(o.input_value(1).get_node()):
                m.add_results([ov.opset13.result(o.input_value(0))])
                wr = ov.opset13.result(o.input_value(1))
                tb = o.get_attributes()["transpose_b"]
                wv = core.compile_model(ov.Model([wr], [], "w"), "CPU", EXACT)({})[0].astype(np.float32)
                wv = wv.reshape(-1, wv.shape[-1])
                short = o.get_friendly_name().split("layers.")[-1].split("/")[0]
                full[short] = o.get_friendly_name()
                probes.append((short, wv if tb else wv.T))
    cm = core.compile_model(m, "CPU", EXACT)
    r = cm(feed_for(cm, x))
    for (nm, Wm), o in zip(probes, cm.outputs[len(cm.outputs) - len(probes):]):
        X = np.asarray(r[o], np.float32).reshape(-1, Wm.shape[1])
        if stats:
            st[full[nm]] = np.abs(X).max(0)
            continue
        if Wm.shape[0] < 256:
            continue
        Y = X @ Wm.T
        out = [f"tok {rel(q8(X) @ Wm.T, Y):.4f}"]
        for A in sqs:
            c = np.maximum(np.abs(X).max(0), 1e-5) ** A / np.maximum(np.abs(Wm).max(0), 1e-8) ** (1 - A)
            out.append(f"sq{A} {rel(q8(X / c) @ (Wm * c).T, Y):.4f}")
        for G in gs:
            Xg = q8(X.reshape(X.shape[0], -1, G)).reshape(X.shape)
            out.append(f"g{G} {rel(Xg @ Wm.T, Y):.4f}")
        if full[nm] in cal:
            c = np.maximum(cal[full[nm]], 1e-5) ** alpha / np.maximum(np.abs(Wm).max(0), 1e-8) ** (1 - alpha)
            out.append(f"w8 {rel(X @ q8(Wm).T, Y):.4f}  cal+w8 {rel(q8(X / c) @ q8(Wm * c).T, Y):.4f}")
        if ROT and X.shape[1] % 1024 == 0:
            f = rot_factors(X.shape[1])
            Xr, Wr = rotate(X, f), rotate(Wm, f)
            out.append(f"rot {rel(q8(Xr) @ Wr.T, Y):.4f}  rot+w8 {rel(q8(Xr) @ q8(Wr).T, Y):.4f}")
        for b in ROTB:
            sg = np.random.default_rng(0).choice([-1.0, 1.0], X.shape[1])
            hb = hadamard(b)
            rb = lambda t: ((t * sg).reshape(-1, b) @ hb).reshape(t.shape)
            Xr, Wr = rb(X), rb(Wm)
            out.append(f"hb{b}+w8 {rel(q8(Xr) @ q8(Wr).T, Y):.4f}")
            if STATICQ:
                # one range for all tokens: per-token max spread, and the
                # error with the range at the max / the 99th percentile
                tm = np.abs(Xr).max(1)
                out.append(f"tokmax p50 {np.median(tm):.2f} max/p50 {tm.max() / np.median(tm):.2f} min/p50 {tm.min() / np.median(tm):.2f}")
                for qq in (100, 99):
                    rq = np.percentile(tm, qq)
                    sq = rq / 127.0
                    Xs = np.clip(np.rint(Xr / sq), -127, 127) * sq
                    out.append(f"st{qq} {rel(Xs @ q8(Wr).T, Y):.4f}")
        cmax = np.abs(X).max(0)
        print(f"L{L:2d} {nm:28s} k={X.shape[1]:5d} chmax/med {cmax.max() / np.median(cmax):7.1f}  " + "  ".join(out), flush=True)
    y = next(r[o] for o in cm.outputs if list(o.get_partial_shape().get_shape()) == [1, S, W])
    x = np.asarray(y, np.float32).reshape(1, S, W)
if stats:
    import os
    if os.path.exists(stats):
        old = dict(np.load(stats))
        for k, v in old.items():
            st[k] = np.maximum(st[k], v) if k in st else v
    np.savez(stats, **st)
    print(f"{len(st)} MatMul channel maxima -> {stats}")
