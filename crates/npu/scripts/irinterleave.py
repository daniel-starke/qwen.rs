"""Reorder the ops of an IE-dialect MLIR file (`vpux-translate --import-IE
--mlir-print-debuginfo`) so that independent head groups of the delta-net
recurrence (`build_delta.py --rgroups G` with NPU_TAGS=1: op names `rec0.*`,
`rec1.*`, ...) interleave op by op instead of following each other.

    python irinterleave.py IN.ie.mlir OUT.ie.mlir

The NPU compiler's feasible memory scheduler takes each executor's compute
ops strictly in IR order (`_computeOpOrder`): with group-major IR, the next
SHAVE op always belongs to the group the DPU is working on, so DPU and SHAVE
never overlap. Here the k-th op of group g is placed at the position of
group 0's k-th op (ties by g), and every other op keeps its position; the
result is a topological order (a ready-list sort on those keys)."""
import heapq
import re
import sys

inp, outp = sys.argv[1:3]
lines = open(inp).read().split("\n")
locs = {}
for ln in lines:
    m = re.match(r'#(loc\d+) = loc\((.*)\)$', ln)
    if m:
        locs[m.group(1)] = m.group(2)


def name(l, d=0):
    v = locs.get(l, "")
    m = re.search(r'"([^"]+)"', v)
    if m:
        return m.group(1)
    m = re.search(r'#(loc\d+)', v)
    return name(m.group(1), d + 1) if m and d < 10 else ""


# the body of @main: one op per line between the func header and `return`
start = next(i for i, ln in enumerate(lines) if re.match(r'\s*func\.func @main', ln)) + 1
end = next(i for i in range(start, len(lines)) if re.match(r'\s*return\b', lines[i]))
body = lines[start:end]
assert all(re.match(r'\s*%', ln) for ln in body), "multi-line op in the body"
defs, ops = {}, []
for i, ln in enumerate(body):
    lhs = ln.split(" = ", 1)[0].strip()
    for r in re.findall(r'%[\w]+', lhs):
        defs[r] = i
    rhs = ln.split(" = ", 1)[1]
    uses = {defs[u] for u in re.findall(r'%[\w]+', rhs) if u in defs}
    m = re.search(r'loc\(#(loc\d+)\)\s*$', ln)
    ops.append((name(m.group(1)) if m else "", uses))
# key: rec{g}'s k-th op sits at rec0's k-th op
grp = {}
for i, (n, _) in enumerate(ops):
    m = re.match(r'rec(\d+)\.', n)
    if m:
        grp.setdefault(int(m.group(1)), []).append(i)
key = {i: (i, -1) for i in range(len(ops))}
if 0 in grp:
    for g, idx in grp.items():
        if len(idx) != len(grp[0]):
            sys.exit(f"group {g}: {len(idx)} ops vs {len(grp[0])} in group 0")
        for k, i in enumerate(idx):
            key[i] = (grp[0][k], g)
users = [[] for _ in ops]
indeg = [0] * len(ops)
for i, (_, uses) in enumerate(ops):
    for u in uses:
        users[u].append(i)
        indeg[i] += 1
# an untagged op (e.g. the split that feeds every group) moves up to its
# earliest user, so it does not pin the later groups behind it
for i in reversed(range(len(ops))):
    if not re.match(r'rec\d+\.', ops[i][0]) and users[i]:
        k = min(key[u] for u in users[i])
        if k < key[i]:
            key[i] = (k[0], k[1] - 0.5)
ready = [(key[i], i) for i in range(len(ops)) if indeg[i] == 0]
heapq.heapify(ready)
order = []
while ready:
    i = heapq.heappop(ready)[1]
    order.append(i)
    for j in users[i]:
        indeg[j] -= 1
        if indeg[j] == 0:
            heapq.heappush(ready, (key[j], j))
assert len(order) == len(ops)
lines[start:end] = [body[i] for i in order]
open(outp, "w").write("\n".join(lines))
print(f"{len(ops)} ops, {len(grp)} groups x {len(grp.get(0, []))}; {sum(a != b for a, b in enumerate(order))} moved")
