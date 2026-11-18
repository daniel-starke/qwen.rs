"""Summarize an `NPU_TASKS=1 npubench ... --profile` dump: busy time and task
count per (op, engine), heaviest first, plus the critical-path view (union of
busy intervals per engine).

    python tasks.py DUMP.txt [N]"""
import re, sys, collections
f, top = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 25
rows = []
for line in open(f, encoding="utf-8", errors="replace"):
    if not line.startswith("T "):
        continue
    _, eng, st, du, rest = line.rstrip("\n").split(" ", 4)
    name = rest.rsplit(" [", 1)[0]
    op = name.split("?", 1)[0]
    op = re.sub(r"__module\.model\.model\.language_model\.layers\.\d+\.", "", op)
    rows.append((eng, int(st), int(du), op))
agg = collections.defaultdict(lambda: [0, 0])
for eng, st, du, op in rows:
    a = agg[(op, eng)]; a[0] += du; a[1] += 1
span = max(s + d for _, s, d, _ in rows) - min(s for _, s, _, _ in rows)
print(f"{len(rows)} tasks, span {span/1e6:.3f} ms")
for eng in ("DMA", "DPU", "SW"):
    iv = sorted((s, s + d) for e, s, d, _ in rows if e == eng)
    u, cur = 0, None
    for a, b in iv:
        if cur is None or a > cur[1]:
            if cur: u += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur: u += cur[1] - cur[0]
    print(f"  {eng}: some task active {u/1e6:.3f} ms ({100*u/span:.0f} % of span)")
for (op, eng), (du, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:top]:
    print(f"  {du/1e6:8.3f} ms {n:5d}  {eng:3s} {op}")

# Timeline: per op, window [first start, last end] and engines, in start order.
if "--timeline" in sys.argv:
    t0 = min(s for _, s, _, _ in rows)
    win = {}
    for eng, st, du, op in rows:
        w = win.setdefault(op, [st, st + du, set(), 0])
        w[0] = min(w[0], st); w[1] = max(w[1], st + du); w[2].add(eng); w[3] += du
    print("timeline (ms): start   end   busy  op")
    for op, (a, b, e, du) in sorted(win.items(), key=lambda kv: kv[1][0]):
        if b - a > 20000:
            print(f"  {(a-t0)/1e6:7.3f} {(b-t0)/1e6:7.3f} {du/1e6:6.3f}  {'/'.join(sorted(e)):11s} {op}")
