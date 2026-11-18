"""swzcheck.py PREFILL_SET... --decode DECODE_SET...: every weight buffer
the engine shares must be read in one form by every graph that binds it.

Per layer L and input name N with a prefill file lean{L}.N.in: the graphs
of every prefill set (lean{L}'s blob, via lean{L}.use) and every decode
set (layer{L}.blob) that take N must all list it in their .swz file, or
none of them. Exit 1 on any mismatch."""
import os
import sys

a = sys.argv[1:]
i = a.index("--decode")
pre, dec = a[:i], a[i + 1:]


def swz(d, stem, L):
    u = os.path.join(d, f"{stem}{L}.use")
    g = open(u).read().strip() if os.path.exists(u) else f"{stem}{L}"
    return set(open(os.path.join(d, f"{g}.swz")).read().split())


def inputs(d, stem, L):
    import openvino as ov
    u = os.path.join(d, f"{stem}{L}.use")
    g = open(u).read().strip() if os.path.exists(u) else f"{stem}{L}"
    return {p.get_friendly_name() for p in ov.Core().read_model(os.path.join(d, f"{g}.xml")).get_parameters()}


bad = 0
nchk = nswz = 0
for L in range(64):
    files = {f[len(f"lean{L}."):-3] for f in os.listdir(pre[0]) if f.startswith(f"lean{L}.") and f.endswith(".in")}
    state = {}
    for d, stem in [(p, "lean") for p in pre] + [(p, "layer") for p in dec]:
        s, ins = swz(d, stem, L), inputs(d, stem, L)
        for n in files & ins:
            state.setdefault(n, set()).add(n in s)
    for n, v in sorted(state.items()):
        nchk += 1
        nswz += True in v
        if len(v) > 1:
            print(f"layer {L} {n}: swizzled in some graphs, plain in others")
            bad += 1
print(f"swzcheck: {nchk} weight inputs ({nswz} swizzled):", "OK" if bad == 0 else f"{bad} mismatches")
sys.exit(1 if bad else 0)
