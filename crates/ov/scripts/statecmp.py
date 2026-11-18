"""Compare two `QWEN_OV_STATE_SAVE` dumps (f16 state files) of the same
prompt and position, state by state: relative RMS error of TEST against REF,
and how many entries each holds as zero / f16 subnormal.

python statecmp.py REF_DIR TEST_DIR [POS]

KV files are the whole `[1, 4, cap, 256]` past buffer; with POS only the
first POS slots (the filled ones) are compared."""
import os, re, sys
import numpy as np

ref, test = sys.argv[1:3]
pos = int(sys.argv[3]) if len(sys.argv) > 3 else None

def key(n):
    m = re.match(r"cache_params\.past\.(\w+)\.(\d+)\.f16", n)
    return (m.group(1), int(m.group(2))) if m else (n, 0)

def sub(x):
    a = np.abs(x)
    return int(((a > 0) & (a < 6.104e-5)).sum()), int((x == 0).sum())

rows = []
for n in sorted(os.listdir(ref), key=key):
    a = np.fromfile(os.path.join(ref, n), np.float16)
    b = np.fromfile(os.path.join(test, n), np.float16)
    kind, layer = key(n)
    if kind in ("key", "value") and pos:
        cap = a.size // (4 * 256)
        a = a.reshape(4, cap, 256)[:, :pos]
        b = b.reshape(4, cap, 256)[:, :pos]
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    rel = np.sqrt(((a64 - b64) ** 2).mean() / max((a64 ** 2).mean(), 1e-300))
    sa, za = sub(a)
    sb, zb = sub(b)
    rows.append((kind, layer, rel, np.sqrt((a64 ** 2).mean()), sa, sb, za, zb, a.size))
for kind in ("key", "value", "conv", "ssm"):
    print(f"== {kind}: layer  relRMS  rms(ref)  subnormal ref/test  zero ref/test  (of N)")
    for r in rows:
        if r[0] == kind:
            print(f"  {r[1]:3d}  {r[2]:.4f}  {r[3]:.3e}  {r[4]:7d} / {r[5]:7d}  {r[6]:7d} / {r[7]:7d}  ({r[8]})")
