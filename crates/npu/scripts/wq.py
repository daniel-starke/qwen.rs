"""Weight requantization for the lean NPU graphs (`build_delta.py`,
`build_attn.py`): one scale per output row, values in an int8 container.

    NPU_WQ=i8         (default) int8, scale = max|w| / 127
    NPU_WQ=i4         int4 [-8, 7], round to nearest, scale per row from a
                      clip search (min squared error over clip ratios)
    NPU_WQ=gptq:DIR   int4 [-8, 7] by GPTQ against the input Hessians in DIR
                      (`hcal.py`; the builders rotate them with the weight),
                      the same per-row scales as `i4`

Per-row int4 is the form the NPU's DPU runs at full rate at every s (no
128-deep K split; `npubench fc --group 0`): 11.3 TOPS at s = 256 and
42 GB/s at s = 1 / 4, against int8 weights' 8.3 TOPS / 2x the bytes."""
import os
import re
import sys
import numpy as np

MODE = os.environ.get("NPU_WQ", "i8")
HDIR = MODE.split(":", 1)[1] if MODE.startswith("gptq:") else None
# NPU_WQ_G=G: int4 with one scale per G input columns (0 = per row)
G = int(os.environ.get("NPU_WQ_G", "0"))
# NPU_WQ_KEEP8=down,out: these matrices (proj / out / gu / down) stay int8
TAIL8 = int(os.environ.get("NPU_WQ_TAIL8", "96"))
KEEP8 = [v for v in os.environ.get("NPU_WQ_KEEP8", "").split(",") if v]
DAMP = float(os.environ.get("NPU_WQ_DAMP", "0.01"))


def _scales(w, g, lo=-8, hi=7, grid=21):
    """Clip-searched scales, one per g columns of each row: [n, k / g]."""
    n, k = w.shape
    if g and g < k:
        return _row_scales(w.reshape(n * (k // g), g), lo, hi, grid).reshape(n, k // g)
    return _row_scales(w, lo, hi, grid)


def _row_scales(w, lo=-8, hi=7, grid=21):
    amax = np.maximum(np.abs(w).max(axis=1, keepdims=True), 1e-12)
    best_e = np.full((w.shape[0], 1), np.inf, np.float32)
    best_s = amax / hi
    for r in np.linspace(0.5, 1.0, grid):
        sc = amax * r / hi
        e = np.empty_like(best_e)
        for i in range(0, w.shape[0], 2048):
            ws, ss = w[i:i + 2048], sc[i:i + 2048]
            d = np.clip(np.rint(ws / ss), lo, hi) * ss - ws
            e[i:i + 2048] = np.einsum("ij,ij->i", d, d)[:, None]
        m = e < best_e
        best_e = np.where(m, e, best_e)
        best_s = np.where(m, sc, best_s)
    return best_s.astype(np.float32)


def gptq(w, H, sc, lo=-8, hi=7, bs=128):
    """GPTQ (Frantar et al.) with fixed scales `sc` [n, k / g] (static groups),
    columns in descending Hessian-diagonal order. Returns the codes [n, k]."""
    W = np.array(w, np.float32)
    n, k = W.shape
    H = np.array(H, np.float64)
    dead = np.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    perm = np.argsort(-np.diag(H), kind="stable")
    W = W[:, perm]
    H = H[perm][:, perm]
    gidx = perm // (k // sc.shape[1])
    H[np.diag_indices(k)] += DAMP * np.mean(np.diag(H))
    # upper U with H^-1 = U^T U: U = J inv(chol(J H J)) J (J = reversal),
    # one Cholesky and one inverse instead of chol, inv, product, chol
    U = np.ascontiguousarray(np.linalg.inv(np.linalg.cholesky(H[::-1, ::-1]))[::-1, ::-1]).astype(np.float32)
    del H
    Q = np.empty((n, k), np.int8)
    for i1 in range(0, k, bs):
        i2 = min(i1 + bs, k)
        W1 = W[:, i1:i2].copy()
        E1 = np.empty_like(W1)
        U1 = U[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = W1[:, i]
            s_ = sc[:, gidx[i1 + i]]
            q = np.clip(np.rint(col / s_), lo, hi)
            Q[:, i1 + i] = q
            err = (col - q * s_) / U1[i, i]
            W1[:, i:] -= np.outer(err, U1[i, i:])
            E1[:, i] = err
        if i2 < k:
            W[:, i2:] -= E1 @ U[i1:i2, i2:]
    out = np.empty_like(Q)
    out[:, perm] = Q
    return out


def quant(w, H=None, name=None):
    """w [n, k] float (H [k, k]: its input's Hessian, for gptq; name: proj /
    out / gu / down) -> (q int8 [n, k], scale f32 [n, k / g])."""
    w = np.asarray(w, np.float32)
    if MODE == "i8" or name in KEEP8 or name == "ab":
        sc = np.maximum(np.abs(w).max(axis=1, keepdims=True), 1e-12) / 127.0
        return np.clip(np.rint(w / sc), -127, 127).astype(np.int8), sc
    if name == "proj" and TAIL8 and w.shape[0] in (16480,):
        # the delta-net gate rows (in_proj_a / in_proj_b, the last 96) int8
        qa, sa = quant_body(w[:-TAIL8], H, name)
        qb = quant_i8(w[-TAIL8:])
        return (np.concatenate([qa, qb[0]]), np.concatenate([sa, np.broadcast_to(qb[1], (TAIL8, sa.shape[1]))])) if sa.shape[1] == 1 else None
    return quant_body(w, H, name)


def quant_i8(w):
    sc = np.maximum(np.abs(w).max(axis=1, keepdims=True), 1e-12) / 127.0
    return np.clip(np.rint(w / sc), -127, 127).astype(np.int8), sc


# GPTQ results are also written to CODES/L{layer}.{name}.npz (packed int4
# codes `q` [n, k / 2] u8, low nibble first; scales `s` [n, k / g] f32) for
# the decode graphs (`rewq.py`). Default: HDIR/../codes.
CODES = os.environ.get("NPU_WQ_CODES") or (os.path.join(os.path.dirname(os.path.normpath(HDIR)), "codes") if HDIR else None)
_m = re.search(r"layer(\d+)\.xml", " ".join(sys.argv).replace("\\", "/"))
LAYER = int(_m.group(1)) if _m else None


def pack4(q):
    u = (q.astype(np.int16) & 15).astype(np.uint8)
    return u[:, 0::2] | (u[:, 1::2] << 4)


def unpack4(p):
    lo = (p & 15).astype(np.int8)
    hi = (p >> 4).astype(np.int8)
    q = np.empty((p.shape[0], p.shape[1] * 2), np.int8)
    q[:, 0::2] = np.where(lo > 7, lo - 16, lo)
    q[:, 1::2] = np.where(hi > 7, hi - 16, hi)
    return q


def _save(name, q, sc):
    if CODES and name and LAYER is not None:
        os.makedirs(CODES, exist_ok=True)
        np.savez(os.path.join(CODES, f"L{LAYER}.{name}.npz"), q=pack4(q), s=sc.astype(np.float32))


def cached(name, shape):
    """The cached GPTQ codes of this layer's `name` for a [n, k] weight as
    stored (packed int4 [n, k / 2] u8, scales f32 [n, 1]), or None. Lets a
    builder bind them without forming (rotating, requantizing) the weight."""
    if not (HDIR and CODES and name and LAYER is not None) or name in KEEP8 or MODE != "gptq" and not MODE.startswith("gptq"):
        return None
    pth = os.path.join(CODES, f"L{LAYER}.{name}.npz")
    if not os.path.exists(pth):
        return None
    z = np.load(pth)
    if z["q"].shape != (shape[0], shape[1] // 2) or z["s"].shape[1] != 1:
        return None
    return z["q"], z["s"]


def quant_body(w, H, name=None):
    if HDIR and H is not None and CODES and name and LAYER is not None:
        # codes from `gptq_gpu.py` (or an earlier run): layer, name and shape
        pth = os.path.join(CODES, f"L{LAYER}.{name}.npz")
        if os.path.exists(pth):
            z = np.load(pth)
            if z["q"].shape == (w.shape[0], w.shape[1] // 2):
                return unpack4(z["q"]), z["s"]
    sc = _scales(w, G)
    if MODE == "i4" or H is None:
        se = np.repeat(sc, w.shape[1] // sc.shape[1], axis=1)
        return np.clip(np.rint(w / se), -8, 7).astype(np.int8), sc
    if HDIR:
        # reuse a dumped result for this layer / matrix (same scales: the
        # clip search is deterministic)
        pth = os.path.join(CODES, f"L{LAYER}.{name}.npz") if CODES and name and LAYER is not None else None
        if pth and os.path.exists(pth):
            z = np.load(pth)
            if z["q"].shape == (w.shape[0], w.shape[1] // 2):
                return unpack4(z["q"]), z["s"]
        q = gptq(w, H, sc)
        _save(name, q, sc)
        return q, sc
    raise SystemExit(f"NPU_WQ={MODE}?")


def hessian(layer, name, f, rotate_any):
    """The rotated Hessian R^T H R of layer `layer`'s `name` input, or None."""
    if not HDIR:
        return None
    if CODES and os.path.exists(os.path.join(CODES, f"L{layer}.{name}.npz")):
        return np.empty(0)  # codes cached (`quant_body` loads them): skip the Hessian
    p = os.path.join(HDIR, f"L{layer}.{name}.npy")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}")
    h = np.load(p).astype(np.float64)
    h = rotate_any(h, f)
    return rotate_any(np.ascontiguousarray(h.T), f)
