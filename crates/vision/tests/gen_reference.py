"""NumPy reference for the vision tower, written from the Qwen3-VL spec.

Weights come from the same integer LCG the Rust test uses, so nothing has to be
serialized: both sides generate bit-identical inputs and only the *result* is
checked in. Run from the repo root; writes crates/vision/tests/data/tiny.ref.
"""
import os
import numpy as np

E = 16          # n_embd
NH = 2          # heads  -> head_dim 8
NFF = 24
NL = 2
PATCH = 4
MERGE = 2
PROJ = 8
SIDE = 4        # position grid side -> n_pos 16
GH, GW = 4, 6   # patch grid (multiples of MERGE)

class Lcg:
    def __init__(self, seed): self.s = seed
    def f(self):
        self.s = (self.s * 1103515245 + 12345) & 0x7fffffff
        return np.float32((self.s / 0x7fffffff) * 2.0 - 1.0)
    def arr(self, *shape):
        n = int(np.prod(shape))
        return np.array([self.f() for _ in range(n)], dtype=np.float32).reshape(shape)

r = Lcg(0x12345678)
PLEN = 3 * PATCH * PATCH
patch_w = r.arr(E, PLEN)      # [out, in]
patch_b = r.arr(E)
pos_w = r.arr(SIDE * SIDE, E)
blocks = []
for _ in range(NL):
    blocks.append(dict(
        ln1_w=r.arr(E), ln1_b=r.arr(E), ln2_w=r.arr(E), ln2_b=r.arr(E),
        qkv_w=r.arr(3 * E, E), qkv_b=r.arr(3 * E),
        out_w=r.arr(E, E), out_b=r.arr(E),
        up_w=r.arr(NFF, E), up_b=r.arr(NFF),
        down_w=r.arr(E, NFF), down_b=r.arr(E),
    ))
post_w, post_b = r.arr(E), r.arr(E)
MERGED = E * MERGE * MERGE
mm0_w, mm0_b = r.arr(MERGED, MERGED), r.arr(MERGED)
mm2_w, mm2_b = r.arr(PROJ, MERGED), r.arr(PROJ)

N = GH * GW
patches = r.arr(N, PLEN)      # already in merge-block order

def ln(x, w, b, eps=1e-6):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.float32(0.79788456) * (x + 0.044715 * x ** 3)))

# --- position embedding, fast_pos_embed_interpolate ------------------------
def axis(n):
    out = []
    for i in range(n):
        v = 0.0 if n <= 1 else i * (SIDE - 1) / (n - 1)
        lo = int(v)
        hi = min(lo + 1, SIDE - 1)
        out.append((lo, hi, v - lo))
    return out

hs, ws = axis(GH), axis(GW)
pos = np.zeros((N, E), dtype=np.float32)
s = 0
order = []          # (row, col) per sequence slot: merge-block order
for by in range(GH // MERGE):
    for bx in range(GW // MERGE):
        for iy in range(MERGE):
            for ix in range(MERGE):
                h, w = by * MERGE + iy, bx * MERGE + ix
                order.append((h, w))
                h0, h1, dh = hs[h]
                w0, w1, dw = ws[w]
                for idx, wt in ((h0 * SIDE + w0, (1 - dh) * (1 - dw)),
                                (h0 * SIDE + w1, (1 - dh) * dw),
                                (h1 * SIDE + w0, dh * (1 - dw)),
                                (h1 * SIDE + w1, dh * dw)):
                    if wt:
                        pos[s] += pos_w[idx] * np.float32(wt)
                s += 1

x = patches @ patch_w.T + patch_b
x = x + pos

# --- 2-D rope --------------------------------------------------------------
HD = E // NH
half = HD // 2
quarter = half // 2
cos = np.zeros((N, half), dtype=np.float32)
sin = np.zeros((N, half), dtype=np.float32)
for s, (h, w) in enumerate(order):
    for j in range(half):
        p, k = (h, j) if j < quarter else (w, j - quarter)
        inv = 1.0 / (10000.0 ** (2.0 * k / half))
        cos[s, j] = np.cos(p * inv)
        sin[s, j] = np.sin(p * inv)

def rope(t):  # t: [N, NH, HD]
    a, b = t[..., :half], t[..., half:]
    c = cos[:, None, :]
    s_ = sin[:, None, :]
    return np.concatenate([a * c - b * s_, b * c + a * s_], axis=-1)

for blk in blocks:
    h = ln(x, blk["ln1_w"], blk["ln1_b"])
    qkv = h @ blk["qkv_w"].T + blk["qkv_b"]
    q, k, v = qkv[:, :E], qkv[:, E:2 * E], qkv[:, 2 * E:]
    q = rope(q.reshape(N, NH, HD))
    k = rope(k.reshape(N, NH, HD))
    v = v.reshape(N, NH, HD)
    scores = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(HD)
    scores = scores - scores.max(-1, keepdims=True)
    scores = np.exp(scores)
    scores /= scores.sum(-1, keepdims=True)
    a = np.einsum("hqk,khd->qhd", scores, v).reshape(N, E)
    x = x + (a @ blk["out_w"].T + blk["out_b"])

    h = ln(x, blk["ln2_w"], blk["ln2_b"])
    up = gelu(h @ blk["up_w"].T + blk["up_b"])
    x = x + (up @ blk["down_w"].T + blk["down_b"])

x = ln(x, post_w, post_b)
y = x.reshape(N // (MERGE * MERGE), MERGED)
y = gelu(y @ mm0_w.T + mm0_b)
out = y @ mm2_w.T + mm2_b

d = os.path.join("crates", "vision", "tests", "data")
os.makedirs(d, exist_ok=True)
out.astype("<f4").tofile(os.path.join(d, "tiny.ref"))
print("tokens", out.shape, "checksum", float(out.astype(np.float64).sum()))
