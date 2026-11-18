"""Compare an NPU run's outputs (`npubench blob ... --dump-dir DIR`, f16 raw
`*.out`) with the references `cmp_delta.py --save DIR` wrote (`*.ref`,
f32), matching them by element count (names do not survive the compile).

    python cmp_out.py DIR [--n N]"""
import glob
import os
import sys
import numpy as np

d = sys.argv[1]
n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else None
refs = {os.path.basename(f)[:-4]: np.fromfile(f, np.float32) for f in glob.glob(os.path.join(d, "*.ref"))}
outs = {os.path.basename(f)[:-4]: np.fromfile(f, np.float16).astype(np.float32) for f in glob.glob(os.path.join(d, "*.out"))}
for rn, r in refs.items():
    cands = [k for k, v in outs.items() if v.size == r.size]
    if not cands:
        print(f"  {rn}: no output of {r.size} elements")
        continue
    # Several outputs of one size (k / v rows): the closest one, named.
    best = None
    for c in cands:
        o, rr = outs[c], r
        if rn == "y" and n:
            o, rr = o.reshape(-1, 5120)[:n], rr.reshape(-1, 5120)[:n]
        bad = ~np.isfinite(o)
        rel = np.linalg.norm(np.where(bad, 0, o) - rr) / max(np.linalg.norm(rr), 1e-30)
        if best is None or rel < best[0]:
            best = (rel, np.abs(np.where(bad, 0, o) - rr).max(), np.abs(rr).max(), bad.sum(), c)
    rel, md, mr, nb, c = best
    amb = f", best of {len(cands)}" if len(cands) > 1 else ""
    print(f"  {rn:10s} rel {rel:.3e}  max|d| {md:.3e}  max|ref| {mr:.3e}  non-finite {nb}  ({c}{amb})")
