"""Inputs of one right-padded 256-position prefill chunk for the first n
tokens of a prompt (`npubench prefill DIR --inputs OUT --dump-states ...`):
x (embeddings from TABLE, [vocab, 5120] f16), mask (attention, cap 256: the
past all masked, causal within the chunk), cos / sin (theta 1e7, 64 rotary
dims, positions 0..255), valid (1 for the n real rows), keep (the conv
window rows of the last 3 real positions).

    python e2e_in.py IDS.txt N OUTDIR --emb TABLE.f16 [--cap 256]"""
import os
import sys
import numpy as np

a = sys.argv[1:]
emb = a[a.index("--emb") + 1]
cap = int(a[a.index("--cap") + 1]) if "--cap" in a else 256
idsf, n, out = a[0], int(a[1]), a[2]
ids = [int(v) for v in open(idsf).read().split()][:n]
S, W = 256, 5120
assert 3 <= n <= S, n
os.makedirs(out, exist_ok=True)
tab = np.memmap(emb, np.float16, "r").reshape(-1, W)
x = np.zeros((1, S, W), np.float16)
x[0, :n] = tab[ids]
mask = np.full((1, 1, S, cap + S), -30000.0, np.float32)
mask[0, 0][:, cap:] = np.where(np.tril(np.ones((S, S), bool)), 0.0, -30000.0)
inv = 1e7 ** (-np.arange(0, 64, 2) / 64.0)
ang = np.arange(S)[:, None] * inv[None, :]
ang = np.concatenate([ang, ang], 1).reshape(1, S, 64)
valid = np.zeros((1, S, 1)); valid[0, :n] = 1
for nm, v in [("x", x), ("mask", mask), ("cos", np.cos(ang)), ("sin", np.sin(ang)), ("valid", valid)]:
    v.astype(np.float16).tofile(os.path.join(out, nm + ".in"))
np.arange(n, n + 3, dtype=np.int32).tofile(os.path.join(out, "keep.in"))
print(f"{n} tokens -> {out}")
