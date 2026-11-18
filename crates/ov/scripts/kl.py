"""Teacher-forced error per position between two `QWEN_OV_TF_DUMP` files
(raw f32 logits rows, the model's padded vocab each) of the same prompt.

python kl.py REF.tf TEST.tf [--vocab N] [--row V] [--every K] [--from P] [--ref-from R]

`--ref-from R`: skip the first R rows of REF (TEST started at position R,
e.g. with `QWEN_OV_STATE_LOAD_SKIP`).

KL(ref || test) of the softmax over the first N logits (the tokenizer's
vocab; the lm_head is padded past it), top-1 agreement and the max |logit|
difference, per position; prints every K-th position and a binned summary."""
import sys
import numpy as np

args = sys.argv[1:]
def opt(name, default):
    if name in args:
        i = args.index(name)
        v = args[i + 1]
        del args[i:i + 2]
        return int(v)
    return default
nv = opt("--vocab", 248077)
row = opt("--row", 248320)
every = opt("--every", 32)
start = opt("--from", 0)
ref_from = opt("--ref-from", 0)
ref, test = args[:2]
a = np.memmap(ref, np.float32, "r").reshape(-1, row)[ref_from:]
b = np.memmap(test, np.float32, "r").reshape(-1, row)
n = min(len(a), len(b))
if len(a) != len(b):
    print(f"rows differ: {len(a)} vs {len(b)}; comparing the first {n}")

def lsm(x):
    x = x.astype(np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())

kl = np.zeros(n); top = np.zeros(n, bool); mx = np.zeros(n)
for i in range(n):
    x, y = a[i, :nv], b[i, :nv]
    la, lb = lsm(x), lsm(y)
    kl[i] = float((np.exp(la) * (la - lb)).sum())
    top[i] = x.argmax() == y.argmax()
    mx[i] = float(np.abs(x.astype(np.float64) - y).max())
for i in range(start, n):
    if (i - start) % every == 0 or i == n - 1:
        print(f"pos {i:5d}  KL {kl[i]:.4f}  top1 {'=' if top[i] else 'X'}  max|dlogit| {mx[i]:.3f}")
print("bins (positions: mean KL / max KL / top-1 flips):")
edges = [0, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1 << 30]
for lo, hi in zip(edges, edges[1:]):
    if lo >= n:
        break
    s = slice(lo, min(hi, n))
    print(f"  [{lo:4d},{min(hi, n):4d})  {kl[s].mean():.4f}  {kl[s].max():.4f}  {(~top[s]).sum()}")
print(f"all: mean KL {kl.mean():.4f}, median {np.median(kl):.4f}, top-1 {top.mean() * 100:.1f} %")
