"""Build a lean prefill graph for one full-attention layer from a cut layer IR
(`cut_layer.py --s S`; only its constants are used), the attention-layer
counterpart of `build_delta.py`.

    python build_attn.py CUT.xml OUT.xml [--s 256] [--cap 4096] [--act mixed|dyn|none]

Interface (all f16):
    x      [1, S, 5120]        residual in
    kpast  [1, CAP, 4, 256]    past keys, token-major (rows >= pos are masked)
    vpast  [1, CAP, 4, 256]    past values, token-major
    mask   [1, 1, S, CAP + S]  additive (0 / -30000): past rows < pos, causal new
    cos    [1, S, 64]          rotary tables of the S positions
    sin    [1, S, 64]
  -> y [1, S, 5120], knew [1, S, 4, 256], vnew [1, S, 4, 256] (the host
     appends them at row pos: one contiguous range in the token-major cache).

Design: q (with its per-head gate) / k / v fused into one int8 MatMul, gate /
up into one; per-head q / k RMSNorm pre-scaled like the residual norms; the 6
query heads of a KV head stacked into one [6 S, 256] matrix so GQA is plain
MatMuls; the softmax scale folded into q."""
import sys
import numpy as np
import wq as wqmod
import bf16src
from quantrot import block_factors, rot_factors, rotate_any, rotate_graph_any
import openvino as ov
import openvino.opset13 as _op
import os as _os

# `NPU_TAGS=1`: op names get their region as a prefix (as in build_delta.py)
_REGION = ["pre"]
_TAGGED = []


class _OpTag:
    def __getattr__(self, name):
        f = getattr(_op, name)
        if not callable(f) or name in ("parameter", "result"):
            return f

        def g(*a_, **k_):
            r = f(*a_, **k_)
            _TAGGED.append((_REGION[0], r))
            return r
        return g


op = _OpTag()


def region(n):
    _REGION[0] = n

a = sys.argv[1:]


def opt(name, d, conv=str):
    if name in a:
        i = a.index(name)
        v = a[i + 1]
        del a[i:i + 2]
        return conv(v)
    return d


S = opt("--s", 256, int)
CAP = opt("--cap", 4096, int)
ACT = opt("--act", "mixed")
SQ_ALPHA = opt("--alpha", 0.65, float)
SQ_STATS = opt("--stats", None)
# `--rotb B`: with `--act rot`, block-diagonal Hadamard rotations of block B
# (one MatMul) instead of the Kronecker rotation; `--rotb -B`: the same B
# signs in every block, folded into the matrix (no sign multiply).
ROTB = opt("--rotb", 0, int)
# `--wai`: the int8 weights as graph inputs `w0, w1, ...` (raw i8 in
# OUT.wN.in) instead of constants: they then live in host buffers the
# runner binds, outside the NPU's ~14 GB cap on graph constants.
WAI = "--wai" in a
a = [v for v in a if v != "--wai"]
# `--act rot` applies the RMSNorm weight in the graph and rotates /
# quantizes the weighted input; `--fold` folds the weight into the MatMul
# instead (worse: 256-token chain KL 0.0034 vs 0.0017, max 0.23 vs 0.03).
NOFOLD = "--fold" not in a
a = [v for v in a if v not in ("--fold", "--nofold")]
RSTAT = opt("--rstat", 16.0, float)
PRE = opt("--pre", 1.0 / 16, float)
# `--kvsplit`: score the past and the new keys as two MatMuls and one
# Softmax over their concat, instead of concatenating K / V first.
KVSPLIT = "--kvsplit" in a
a = [v for v in a if v != "--kvsplit"]
# `--part all|pre|core|post`: the whole layer, or one of three blobs that
# chain: `pre` (norm, q/gate/k/v projection, head norms, RoPE: x, cos, sin ->
# qg [1, 4, 6 S, 256], gate [1, S, 6144], knew, vnew), `core` (attention,
# no weights, shared by every attention layer: qg, kpast, vpast, knew, vnew,
# mask -> o [1, S, 6144]) and `post` (gate, out_proj, FFN: x, o, gate -> y).
PART = opt("--part", "all")
# `--sm`: `softmax` (default), `none` (no normalization; a timing probe only).
SM = opt("--sm", "softmax")
# `--w0 N`: the first weight input's index (`--part post --w0 1` names its
# weights as the whole layer's graph does, so both bind the same files)
W0 = opt("--w0", 0, int)
NBLK = int(opt("--nblk", "1"))
# `--kq RK,RQ` (with `--part far`): the block's keys as int8 weight inputs
# `kq0..kq3` ([CAP, 256] i8 per KV head, k = code * RK / 127) and the
# queries through a static FakeQuantize over +-RQ, so that Q K runs as the
# DPU's integer fully-connected layer instead of f16 x f16.
KQ = opt("--kq", None)
FARX = [v for v in a if v == "--nolse"]
a = [v for v in a if v != "--nolse"]
# `--nomask`: no attention mask at all (a timing probe only)
NOMASK = "--nomask" in a
a = [v for v in a if v != "--nomask"]
# `--srange FILE.json | R`: as in build_delta.py, with `--act rot`, the
# projection and gate / up quantize their rotated input with a static range
# (FILE.json from `rcal.py`, entry of layer L from the cut's name
# `layerL.xml`): a FakeQuantize of range +-r, in the input's own units.
# (Folding 127 / r into the norm weight and r / 127 into the constant row
# scales, as build_delta.py --allin does with its row-scale inputs, put the
# f16 row scales near the subnormal range: NPU error per layer x1.2.)
SRANGE = opt("--srange", None)
SONLY = opt("--sonly", None)
GUSPLIT = opt("--gusplit", 1, int)
# `--sgated FILE.json:THR`: static input ranges also for the gated MatMuls
# (out_proj / o_proj: "out", down_proj: "down"; `rcal.py --all`), each only
# where that input's per-token max over its median is <= THR in this layer
SGATED = opt("--sgated", None)
ROUT = RDOWN = None
if SGATED:
    import json as _json
    import re as _re
    _f, _thr = SGATED.rsplit(":", 1)
    _e = _json.load(open(_f))[_re.search(r"layer(\d+)\.xml$", a[0].replace("\\", "/")).group(1)]
    ROUT = _e["out"] if _e["out"] / _e["out_p50"] <= float(_thr) else None
    RDOWN = _e["down"] if _e["down"] / _e["down_p50"] <= float(_thr) else None
# `--rmsk ln,hn`: ln1 / ln2 (ln) and the q / k head norms (hn) in the form the
# compiler fuses into one IE.RMS SHAVE op (as in build_delta.py)
RMSK = set(opt("--rmsk", "").split(",")) - {""}
# `--lnin`: the RMSNorm weights as graph inputs `ln1` / `ln2` (OUT.lnN.in),
# as build_delta.py --allin has them: with a constant weight before the
# rotation and a static FakeQuantize the compiler folds them together and
# gate / up lost accuracy on the NPU (KL 0.0040 vs 0.0023).
LNIN = "--lnin" in a
a = [v for v in a if v != "--lnin"]
inp, outp = a
RPROJ = RGU = None
if SRANGE is not None:
    import json
    import re
    try:
        RPROJ = RGU = float(SRANGE)
    except ValueError:
        _lay = re.search(r"layer(\d+)\.xml$", inp.replace("\\", "/")).group(1)
        _r = json.load(open(SRANGE))[_lay]
        RPROJ, RGU = float(_r["proj"]), float(_r["gu"])
    # `--sonly proj|gu`: the static range on that MatMul only
    if SONLY == "gu":
        RPROJ = None
    elif SONLY == "proj":
        RGU = None
W, HQ, HKV, D, RD = 5120, 24, 4, 256, 64
G = HQ // HKV
core = ov.Core()
m = core.read_model(inp)
ops = {o.get_friendly_name(): o for o in m.get_ordered_ops()}
F16 = ov.Type.f16


def evaluate(output):
    sub = ov.Model([op.result(output)], [], "w")
    return core.compile_model(sub, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})({})[0].astype(np.float32)


def find(suffix):
    hits = [o for n, o in ops.items() if n.endswith(suffix)]
    if len(hits) != 1:
        raise SystemExit(f"{suffix}: {len(hits)} matches")
    return hits[0]


def const_in(suffix, k=1):
    return evaluate(find(suffix).input_value(k))


def _weight_export(suffix):
    o = find(suffix)
    w = evaluate(o.input_value(1))
    if not o.get_attributes()["transpose_b"]:
        w = w.T
    return w.reshape(-1, w.shape[-1])


def _export_shape(suffix):
    """[n, k] of the export's MatMul weight, without evaluating it."""
    o = find(suffix)
    sh = list(o.input_value(1).get_partial_shape().get_shape())
    if not o.get_attributes()["transpose_b"]:
        sh = sh[::-1]
    return (int(np.prod(sh[:-1])), int(sh[-1]))


def weight(suffix):
    if not bf16src.SRC:
        return _weight_export(suffix)
    import re
    lay = int(re.search(r"layer(\d+)\.xml$", inp.replace("\\", "/")).group(1))
    wb = bf16src.layer_weight(lay, suffix).reshape(_export_shape(suffix))
    if __import__("os").environ.get("NPU_SRC_CHECK") == "1":
        # dequantizes the export's int4 matrix on the CPU: slow, a check only
        w = _weight_export(suffix)
        print(f"bf16 {suffix.split('/')[0]}: export vs bf16 rel {np.linalg.norm(w - wb) / np.linalg.norm(wb):.4f}", flush=True)
    return wb


ln1 = const_in("input_layernorm/aten::mul/Multiply_1").reshape(-1)
eps1 = float(const_in("input_layernorm/aten::add/Add").reshape(-1)[0])
Wq = weight("self_attn.q_proj/ov_ext::linear/MatMul")  # [24 * 512, 5120]: per head q, gate
Wk = weight("self_attn.k_proj/ov_ext::linear/MatMul")
Wv = weight("self_attn.v_proj/ov_ext::linear/MatMul")
qnw = const_in("self_attn.q_norm/aten::mul/Multiply_1").reshape(-1)
qeps = float(const_in("self_attn.q_norm/aten::add/Add").reshape(-1)[0])
knw = const_in("self_attn.k_norm/aten::mul/Multiply_1").reshape(-1)
keps = float(const_in("self_attn.k_norm/aten::add/Add").reshape(-1)[0])
scale = float(const_in("scaled_dot_product_attention/ScaledDotProductAttention", 4).reshape(-1)[0])
Wo = weight("self_attn.o_proj/ov_ext::linear/MatMul")
ln2 = const_in("post_attention_layernorm/aten::mul/Multiply_1").reshape(-1)
eps2 = float(const_in("post_attention_layernorm/aten::add/Add").reshape(-1)[0])
Wg = weight("mlp.gate_proj/ov_ext::linear/MatMul")
Wu = weight("mlp.up_proj/ov_ext::linear/MatMul")
Wd = weight("mlp.down_proj/ov_ext::linear/MatMul")
# q rows reordered to every head's query, then every head's gate
Wq4 = Wq.reshape(HQ, 2, D, W)
Wproj = np.concatenate([Wq4[:, 0].reshape(-1, W), Wq4[:, 1].reshape(-1, W), Wk, Wv], 0)
Wgu = np.concatenate([Wg, Wu], 0)
Wproj_n = Wproj * ln1.reshape(1, -1)
Wgu_n = Wgu * ln2.reshape(1, -1)
# `--act sq`: the norm-fed MatMuls (projection, gate / up) per-token int8 on
# SmoothQuant-scaled inputs (`--stats CAL.npz` from `actq.py --stats`,
# `--alpha`), the gated-product inputs (out / o / down projection, whose
# outlier channels move per token) f16 activations with int8 weights.
# Measured on the CPU chain (256 tokens): mean KL 0.0026 vs 0.045 for
# per-token int8 everywhere.
GATED = "none" if ACT == "sq" else None
if ACT == "sq":
    from smooth import smooth_c
    _st = dict(np.load(SQ_STATS))
    c1 = smooth_c(_st, find("self_attn.q_proj/ov_ext::linear/MatMul").get_friendly_name(), Wproj, SQ_ALPHA)
    c2 = smooth_c(_st, find("mlp.gate_proj/ov_ext::linear/MatMul").get_friendly_name(), Wgu, SQ_ALPHA)
    Wproj_s, ln1_s = Wproj * c1.reshape(1, -1), ln1 / c1
    Wgu_s, ln2_s = Wgu * c2.reshape(1, -1), ln2 / c2
NQ, NK = HQ * D, HKV * D
print(f"weights read; scale {scale} eps {eps1} {qeps} {keps}", flush=True)


def c16(v):
    return op.constant(np.asarray(v, np.float16))


def ci64(v):
    return op.constant(np.asarray(v, np.int64))


WPARAMS = []
_RET_TUPLE = False
# `--gusplit P`: the gate / up MatMul as P MatMuls over row blocks (cached
# int4 codes only, `GUSPLIT`), as in build_delta.py


class LazyW:
    """A weight formed on first call (`rotate_any`), with its shape known."""

    def __init__(self, shape, fn):
        self.shape, self.fn = tuple(shape), fn

    def __call__(self):
        return self.fn()


def i8_weight(w, fold=1.0, H=None, name=None):
    c = wqmod.cached(name, w.shape) if WAI and __import__("os").environ.get("NPU_WQ_STORE") == "i4" and wqmod.MODE != "f" else None
    if c is not None:
        # cached GPTQ codes: bound as stored, the weight never formed
        packed, sc = c
        if name == "gu" and GUSPLIT > 1:
            n = w.shape[0] // GUSPLIT
            res = []
            for i in range(GUSPLIT):
                prm = op.parameter([n, w.shape[1]], ov.Type.i4, name=f"w{W0 + len(WPARAMS)}")
                WPARAMS.append((prm, np.ascontiguousarray(packed[i * n:(i + 1) * n])))
                res.append(op.multiply(op.convert(prm, "f16"), c16(sc[i * n:(i + 1) * n] * fold)))
            return res
        prm = op.parameter(list(w.shape), ov.Type.i4, name=f"w{W0 + len(WPARAMS)}")
        WPARAMS.append((prm, packed))
        return op.multiply(op.convert(prm, "f16"), c16(sc * fold))
    if callable(w):
        w = w()
    if wqmod.MODE == "f":
        # float weights (accuracy reference)
        wexp = op.constant(np.asarray(w * fold, np.float16))
        return (wexp, None) if isinstance(_RET_TUPLE, bool) and _RET_TUPLE else wexp
    q, sc = wqmod.quant(w, H, name)
    if sc.shape[1] > 1:
        # int4 groups (accuracy simulation only): q x per-group scale, no
        # row-scale factor after the MatMul
        n, k = q.shape
        gs = sc.shape[1]
        prm = op.parameter([n, gs, k // gs], ov.Type.i8, name=f"w{W0 + len(WPARAMS)}")
        WPARAMS.append((prm, q.reshape(n, gs, k // gs)))
        wexp = op.reshape(op.multiply(op.convert(prm, "f16"), op.constant((sc * fold).reshape(n, gs, 1).astype(np.float16))),
                          op.constant(np.array([n, k], np.int64)), False)
        return wexp
    sc = sc * fold
    if WAI:
        # an i8 graph input (as fast as a constant; the scale must stay a
        # constant), written to OUT.wN.in
        if __import__("os").environ.get("NPU_WQ_STORE") == "i4" and q.min() >= -8 and q.max() <= 7:
            # int4 storage: packed nibbles in OUT.wN.in (`wq.pack4`)
            prm = op.parameter(list(q.shape), ov.Type.i4, name=f"w{W0 + len(WPARAMS)}")
            WPARAMS.append((prm, wqmod.pack4(q)))
        else:
            prm = op.parameter(list(q.shape), ov.Type.i8, name=f"w{W0 + len(WPARAMS)}")
            WPARAMS.append((prm, q))
        return op.multiply(op.convert(prm, "f16"), c16(sc))
    return op.multiply(op.convert(op.constant(q), "f16"), c16(sc))


_wcache = {}
_LAYER = int(__import__("re").search(r"layer(\d+)\.xml$", inp.replace("\\", "/")).group(1)) if wqmod.HDIR else None


def _wname(w):
    return {id(Wproj): "proj", id(Wo): "out", id(Wgu): "gu", id(Wd): "down"}.get(id(w))


def _hess(w, f):
    """GPTQ Hessian (NPU_WQ=gptq:DIR) of the MatMul input `w` reads."""
    nm = _wname(w)
    return None if nm is None or nm in wqmod.KEEP8 else wqmod.hessian(_LAYER, nm, f, rotate_any)


def linear(x, w, static=None, mode=None, srange=None):
    """x [.., k] f16 @ w[n, k]^T, int8 weights, per-token int8 activations
    (or a static per-tensor range `static`). `rot`: per token after a
    Kronecker rotation of x, folded into the weight (quantrot.py).
    `srange` r: `rot` with a static range +-r."""
    mode = mode or ACT
    if mode == "rot":
        f = block_factors(w.shape[1], abs(ROTB), periodic=ROTB < 0) if ROTB else rot_factors(w.shape[1])
        x = rotate_graph_any(op, x, f)
        key = (id(w), "rot")
        if key not in _wcache:
            _wcache[key] = i8_weight(LazyW(w.shape, lambda: rotate_any(w, f)), H=_hess(w, f), name=_wname(w))
        wq = _wcache[key]
        if srange is not None:
            lo, hi = c16(-srange), c16(srange)
            xq = op.fake_quantize(x, lo, hi, lo, hi, 255)
            if isinstance(wq, list):
                # `--gusplit`: one MatMul per row block, no split after them
                return [op.matmul(xq, w_, False, True) for w_ in wq]
            return op.matmul(xq, wq, False, True)
        mode = "dyn"
    else:
        if id(w) not in _wcache:
            _wcache[id(w)] = i8_weight(w)
        wq = _wcache[id(w)]
    if mode == "none":
        return op.matmul(x, wq, False, True)
    if static is not None and mode == "mixed":
        lo, hi = c16(-static), c16(static)
        return op.matmul(op.fake_quantize(x, lo, hi, lo, hi, 255), wq, False, True)
    mx = op.maximum(op.reduce_max(op.absolute(x), ci64([-1]), True), c16(1e-4))
    lo, hi = c16(-127.0), c16(127.0)
    xq = op.fake_quantize(op.multiply(x, op.divide(c16(127.0), mx)), lo, hi, lo, hi, 255)
    if isinstance(wq, list):
        post = op.multiply(mx, c16(1.0 / 127.0))
        return [op.multiply(op.matmul(xq, w_, False, True), post) for w_ in wq]
    return op.multiply(op.matmul(xq, wq, False, True), op.multiply(mx, c16(1.0 / 127.0)))


def rsqrt(t, eps):
    t = op.add(t, c16(eps)) if eps >= 6.2e-5 else op.maximum(t, c16(6.2e-5))
    return op.divide(c16(1.0), op.sqrt(t))


def sumsq_mean(t):
    n = t.get_output_partial_shape(0)[-1].get_length()
    return op.matmul(op.power(t, c16(2.0)), c16(np.full((n, 1), 1.0 / n, np.float16)), False, False)


LNPARAMS = []


def rms_fused(x, eps):
    """x / sqrt(mean(x^2) + eps), as the compiler's FuseRMSNorm matches it."""
    ms = op.reduce_mean(op.power(x, c16(2.0)), ci64([-1]), True)
    return op.multiply(x, op.divide(c16(1.0), op.sqrt(op.add(ms, c16(eps)))))


def rmsnorm(x, w, eps, pre=PRE, name=None):
    if "ln" in RMSK:
        pre = 1.0
    xs = op.multiply(x, c16(pre)) if pre != 1.0 else x
    y = rms_fused(xs, eps) if "ln" in RMSK else op.multiply(xs, rsqrt(sumsq_mean(xs), eps * pre * pre))
    if w is None:
        return y
    if LNIN and name:
        prm = op.parameter([len(w)], F16, name=name)
        LNPARAMS.append((prm, np.asarray(w, np.float16)))
        return op.multiply(y, prm)
    return op.multiply(y, c16(w))


def headnorm(t, w, eps):
    """Per-head RMSNorm with a per-row max-abs pre-scale (q / k rows can be
    small or large; this keeps the mean square in f16's normal range)."""
    if "hn" in RMSK:
        return op.multiply(rms_fused(t, eps), c16(w))
    mx = op.maximum(op.reduce_max(op.absolute(t), ci64([-1]), True), c16(1e-4))
    ts = op.multiply(t, op.divide(c16(1.0), mx))
    ms = sumsq_mean(ts)
    # eps relative to the pre-scaled rows: eps / mx^2, negligible unless the
    # row is ~0, where the floor in rsqrt takes over.
    return op.multiply(op.multiply(ts, rsqrt(ms, 0.0)), c16(w))


def rope(t, cos, sin):
    """t [1, S, n, 256]: rotate the first 64 channels (rotate-half within
    them); cos / sin [1, S, 1, 64]."""
    r, p = op.variadic_split(t, ci64(3), ci64([RD, D - RD])).outputs()
    r1, r2 = op.variadic_split(r, ci64(3), ci64([RD // 2, RD // 2])).outputs()
    rot = op.concat([op.negative(r2), r1], 3)
    if S == 1:
        # one position: the compiler broadcasts [1, 1, 1, 64] over the heads
        # wrongly (every second head came out wrong); spell the copies out
        n = t.get_output_partial_shape(0)[2].get_length()
        cos, sin = op.concat([cos] * n, 2), op.concat([sin] * n, 2)
    return op.concat([op.add(op.multiply(r, cos), op.multiply(rot, sin)), p], 3)


def param(shape, name):
    return op.parameter(shape, F16, name=name)


x = op.parameter([1, S, W], F16, name="x")
kpast = op.parameter([1, CAP, HKV, D], F16, name="kpast")
vpast = op.parameter([1, CAP, HKV, D], F16, name="vpast")
mask = op.parameter([1, 1, S, CAP + S], F16, name="mask")
cos = op.parameter([1, S, RD], F16, name="cos")
sin = op.parameter([1, S, RD], F16, name="sin")
params = [x, kpast, vpast, mask, cos, sin]

def pre():
    if ACT == "sq":
        proj = linear(rmsnorm(x, ln1_s, eps1), Wproj_s)
    elif ACT == "mixed" or (ACT == "rot" and not NOFOLD):
        proj = linear(rmsnorm(x, None, eps1), Wproj_n, RSTAT)
    else:
        proj = linear(rmsnorm(x, ln1, eps1, name="ln1"), Wproj, srange=RPROJ)
    q, gate, k, v = op.variadic_split(proj, ci64(2), ci64([NQ, NQ, NK, NK])).outputs()
    cs = op.unsqueeze(cos, ci64(2))
    sn = op.unsqueeze(sin, ci64(2))
    q = rope(headnorm(op.reshape(q, ci64([1, S, HQ, D]), False), qnw * scale, qeps), cs, sn)
    knew = rope(headnorm(op.reshape(k, ci64([1, S, HKV, D]), False), knw, keps), cs, sn)
    vnew = op.reshape(v, ci64([1, S, HKV, D]), False)
    # q [1, S, 24, D] -> [1, 4, 6 S, D] (query head h reads KV head h // 6)
    qg = op.reshape(op.transpose(op.reshape(q, ci64([1, S, HKV, G, D]), False), ci64([0, 2, 3, 1, 4])),
                    ci64([1, HKV, G * S, D]), False)
    return qg, op.sigmoid(gate), knew, vnew


def heads(t):
    return op.transpose(t, ci64([0, 2, 1, 3]))  # [1, n, 4, D] -> [1, 4, n, D]


def core(qg, knew, vnew):
    region("core")
    mk = op.reshape(op.broadcast(op.unsqueeze(mask, ci64(2)), ci64([1, 1, G, S, CAP + S])),
                    ci64([1, 1, G * S, CAP + S]), False)
    if KVSPLIT:
        sp = op.matmul(qg, heads(kpast), False, True)
        sn_ = op.matmul(qg, heads(knew), False, True)
        p = op.softmax(op.add(op.concat([sp, sn_], 3), mk), 3)
        pp, pn = op.variadic_split(p, ci64(3), ci64([CAP, S])).outputs()
        o = op.add(op.matmul(pp, heads(vpast), False, False), op.matmul(pn, heads(vnew), False, False))
    else:
        K = heads(op.concat([kpast, knew], 1))
        V = heads(op.concat([vpast, vnew], 1))
        sc = op.matmul(qg, K, False, True)
        if not NOMASK:
            sc = op.add(sc, mk)
        p = op.softmax(sc, 3) if SM == "softmax" else sc
        o = op.matmul(p, V, False, False)  # [1, 4, 6 S, D]
    return op.reshape(op.transpose(op.reshape(o, ci64([1, HKV, G, S, D]), False), ci64([0, 3, 1, 2, 4])),
                      ci64([1, S, HQ * D]), False)


def post(o, sgate):
    region("wo")
    x1 = op.add(x, linear(op.multiply(o, sgate), Wo, mode=GATED, srange=ROUT))
    region("gu")
    if ACT == "sq":
        gu = linear(rmsnorm(x1, ln2_s, eps2), Wgu_s)
    elif ACT == "mixed" or (ACT == "rot" and not NOFOLD):
        gu = linear(rmsnorm(x1, None, eps2), Wgu_n, RSTAT)
    else:
        gu = linear(rmsnorm(x1, ln2, eps2, name="ln2"), Wgu, srange=RGU)
    if isinstance(gu, list):
        h_ = len(gu) // 2
        gg = gu[0] if h_ == 1 else op.concat(gu[:h_], 2)
        uu = gu[h_] if h_ == 1 else op.concat(gu[h_:], 2)
    else:
        gg, uu = op.variadic_split(gu, ci64(2), ci64([Wg.shape[0], Wg.shape[0]])).outputs()
    region("down")
    return op.add(x1, linear(op.multiply(op.swish(gg, c16(1.0)), uu), Wd, mode=GATED, srange=RDOWN))


if PART == "all":
    qg, sgate, knew, vnew = pre()
    res = [(post(core(qg, knew, vnew), sgate), "y"), (knew, "knew"), (vnew, "vnew")]
elif PART == "pre":
    params = [x, cos, sin]
    qg, sgate, knew, vnew = pre()
    res = [(qg, "qg"), (sgate, "gate"), (knew, "knew"), (vnew, "vnew")]
elif PART == "core":
    qg_, kn_, vn_ = param([1, HKV, G * S, D], "qg"), param([1, S, HKV, D], "knew"), param([1, S, HKV, D], "vnew")
    params = [qg_, kpast, vpast, kn_, vn_, mask]
    res = [(core(qg_, kn_, vn_), "o")]
elif PART == "post":
    o_, g_ = param([1, S, HQ * D], "o"), param([1, S, HQ * D], "gate")
    params = [x, o_, g_]
    res = [(post(o_, g_), "y")]
elif PART == "near":
    # the attention over the chunk's own keys and one partly filled block of
    # past keys (CAP rows, the last two not keys): as `far`, with the mask
    # [1, 1, S, S + CAP] (new keys first)
    qg_, kn_, vn_ = param([1, HKV, G * S, D], "qg"), param([1, S, HKV, D], "knew"), param([1, S, HKV, D], "vnew")
    params = [qg_, kpast, vpast, kn_, vn_, mask]
    mk = op.reshape(op.broadcast(op.unsqueeze(mask, ci64(2)), ci64([1, 1, G, S, CAP + S])),
                    ci64([1, 1, G * S, CAP + S]), False)
    sc = op.add(op.matmul(qg_, heads(op.concat([kn_, kpast], 1)), False, True), mk)
    p = op.softmax(sc, 3)
    res = [(op.matmul(p, heads(op.concat([vn_, vpast], 1)), False, False), "o"),
           (op.variadic_split(p, ci64(3), ci64([CAP + S - 2, 2])).output(1), "lse")]
elif PART == "near0":
    # `near` for a decode step: the new key is already in the block (the
    # host put it there), mask [1, 1, S, CAP]
    qg_ = param([1, HKV, G * S, D], "qg")
    mask = op.parameter([1, 1, S, CAP], F16, name="mask")
    params = [qg_, kpast, vpast, mask]
    mk = op.reshape(op.broadcast(op.unsqueeze(mask, ci64(2)), ci64([1, 1, G, S, CAP])), ci64([1, 1, G * S, CAP]), False)
    p = op.softmax(op.add(op.matmul(qg_, heads(kpast), False, True), mk), 3)
    res = [(op.matmul(p, heads(vpast), False, False), "o"),
           (op.variadic_split(p, ci64(3), ci64([CAP - 2, 2])).output(1), "lse")]
elif PART == "far":
    # one block of CAP - 2 past keys, no mask, no weights (one blob for
    # every attention layer): qg, kblk, vblk -> o [1, 4, 6 S, 256] normalized
    # within the block, lse [1, 4, 6 S, 2] = (max score, sum exp(score - max));
    # the host merges blocks with it
    qg_ = param([1, HKV, G * S, D], "qg")
    kb, vb = param([1, CAP, HKV, D], "kblk"), param([1, CAP, HKV, D], "vblk")
    params = [qg_, kb, vb]
    if NBLK > 1:
        # `--nblk N` (decode steps: a job per block is job overhead): N
        # blocks kblk0.. / vblk0.. in one graph, o [1, 4 N, 6 S, 256] and
        # lse [1, 4 N, 6 S, 2] block-major; the host ignores unused blocks
        kbs = [param([1, CAP, HKV, D], f"kblk{i}") for i in range(NBLK)]
        vbs = [param([1, CAP, HKV, D], f"vblk{i}") for i in range(NBLK)]
        params = [qg_] + kbs + vbs
        ps = [op.softmax(op.matmul(qg_, heads(k_), False, True), 3) for k_ in kbs]
        res = [(op.concat([op.matmul(p_, heads(v_), False, False) for p_, v_ in zip(ps, vbs)], 1), "o"),
               (op.concat([op.variadic_split(p_, ci64(3), ci64([CAP - 2, 2])).output(1) for p_ in ps], 1), "lse")]
    if KQ:
        rk, rq = [float(v) for v in KQ.split(",")]
        kqs = [op.parameter([CAP, D], ov.Type.i8, name=f"kq{h}") for h in range(HKV)]
        params = [qg_] + kqs + [vb]
        lo, hi = c16(-rq), c16(rq)
        qs = op.split(op.fake_quantize(qg_, lo, hi, lo, hi, 255), ci64(1), HKV).outputs()
        # per KV head down to P V: one concat of scores would make the
        # softmax wait for every head's Q K
        ps = [op.softmax(op.matmul(q_, op.multiply(op.convert(k_, "f16"), c16(rk / 127.0)), False, True), 3) for q_, k_ in zip(qs, kqs)]
        vs = op.split(heads(vb), ci64(1), HKV).outputs()
        res = [(op.concat([op.matmul(p_, v_, False, False) for p_, v_ in zip(ps, vs)], 1), "o"),
               (op.concat([op.variadic_split(p_, ci64(3), ci64([CAP - 2, 2])).output(1) for p_ in ps], 1), "lse")]
    else:
        sc = op.matmul(qg_, heads(kb), False, True)
        p = op.softmax(sc, 3)
    res = res if NBLK > 1 or KQ else [(op.matmul(p, heads(vb), False, False), "o")]
    if NBLK == 1 and not KQ and "--nolse" not in FARX:
        # compiled with shave/softmax_lse.sk as the Softmax kernel: the last
        # two key slots of the block are not keys (their value rows are
        # zero), p's last two columns come back as max score and sum exp
        res += [(op.variadic_split(p, ci64(3), ci64([CAP - 2, 2])).output(1), "lse")]
else:
    raise SystemExit(f"--part {PART}?")

outs = []
for t, nm in res:
    r = op.result(t)
    r.output(0).get_tensor().set_names({nm})
    outs.append(r)
model = ov.Model(outs, params + [prm for prm, _ in WPARAMS + LNPARAMS], "attn")
for prm, q in WPARAMS + LNPARAMS:
    q.tofile(outp[:-4] + "." + prm.get_friendly_name() + ".in")
if _os.environ.get("NPU_TAGS") == "1":
    for _rg, _n in _TAGGED:
        _n = _n.get_node() if hasattr(_n, "get_node") else _n
        if _n.get_type_name() not in ("Parameter", "Result", "Constant"):
            _n.set_friendly_name(f"{_rg}.{_n.get_friendly_name()}")
model.validate_nodes_and_infer_types()
ov.save_model(model, outp, compress_to_fp16=False)
print(f"S={S} cap={CAP} act={ACT} kvsplit={KVSPLIT} part={PART}: {len(model.get_ops())} ops -> {outp}")
