"""Rewrite the K-position verify graph of a delta-net layer (`verify_cut.py`:
unrolled steps `qwen_rs.unroll.*{i}`, a state output `state{i}` per position)
into the deferred-rollback WY form:

  S0  = alpha * sbase + (kp * w)^T dp       (the previous step's rollback,
                                             rank <= K, done here, not on the host)
  G   = cumsum(g); A = exp(G); M_jm = exp(G_j - G_m) (m <= j)
  D   = (I + tril_-1(beta (K K^T) * M))^-1 (beta (V - A * K S0))
  o   = A * Q S0 + ((Q K^T) * M) D

Outputs (replacing state0..K-1): `sbase` = S0, `kcur` = K, `dnew` = D,
`gcum` = G. The host picks the accepted length a and sets the next step's
inputs: sbase <- this sbase, kp <- kcur, dp <- dnew, alpha = exp(G_(a-1)),
w_m = exp(G_(a-1) - G_m) for m < a, else 0. One state write per step instead
of K. Inputs added: `qwen_rs.kp`, `qwen_rs.dp` [1, H, K, D], `qwen_rs.w`
[1, H, K, 1], `qwen_rs.alpha` [1, H, 1, 1]; the ssm past input is `sbase`.

    python wyrec.py IN.xml OUT.xml"""
import sys
import numpy as np
import openvino as ov
import openvino.opset13 as op

inp, outp = sys.argv[1:3]
m = ov.Core().read_model(inp)
by = {o.get_friendly_name().split("/")[-1]: o for o in m.get_ordered_ops()}
N = 0
while f"qwen_rs.unroll.kv{N}" in by:
    N += 1
assert N > 1, "not a K-position delta verify graph"
f32 = ov.Type.f32
c = lambda v: op.constant(np.asarray(v, np.float32))
c64 = lambda v: op.constant(np.asarray(v, np.int64))
u = lambda nm, i: by[f"qwen_rs.unroll.{nm}{i}"]
S = u("decay", 0).input_value(0)                    # [1, H, D, D], S[h, k, v]
H, D = S.get_partial_shape()[1].get_length(), S.get_partial_shape()[2].get_length()
rows = lambda f: op.concat([f(i) for i in range(N)], 2)
K = rows(lambda i: u("kv", i).input_value(0))       # [1, H, N, D]
Q = rows(lambda i: u("out", i).input_value(0))
V = rows(lambda i: u("vmkv", i).input_value(0))
g = rows(lambda i: u("exp", i).input_value(0))      # [1, H, N, 1] log decay
b = rows(lambda i: u("delta", i).input_value(1))    # beta
P = lambda n, s: op.parameter(s, ov.Type.f16, name=n)
kp, dp, w, al = P("qwen_rs.kp", [1, H, N, D]), P("qwen_rs.dp", [1, H, N, D]), P("qwen_rs.w", [1, H, N, 1]), P("qwen_rs.alpha", [1, H, 1, 1])
cv = lambda t: op.convert(t, f32)
S0 = op.add(op.multiply(S, cv(al)), op.matmul(op.multiply(cv(kp), cv(w)), cv(dp), True, False))
tri = np.tril(np.ones((N, N), np.float32))
G = op.matmul(c(tri), g, False, False)
A = op.exp(G)
# mask the exponent first: above the diagonal G_j - G_m > 0 can overflow f16
M = op.multiply(op.exp(op.multiply(op.add(G, op.transpose(op.multiply(G, c(-1.0)), c64([0, 1, 3, 2]))), c(tri))), c(tri))
KS = op.matmul(K, S0, False, False)
QS = op.matmul(Q, S0, False, False)
# Bn = -beta_j (K K^T)_jm M_jm, strictly lower (the sign folded into the mask)
Bn = op.multiply(op.multiply(op.multiply(op.matmul(K, K, False, True), M), c(-np.tril(np.ones((N, N), np.float32), -1))), b)
rhs = op.multiply(op.add(V, op.multiply(op.multiply(A, c(-1.0)), KS)), b)
# D = (I - Bn)^-1 rhs = rhs + Bn (rhs + Bn (rhs + ...)): exact after N - 1
# terms (Bn^N = 0); tiny MatMuls and Adds, no Subtract (each lowers to a
# DPU negate + add) and no doubling
Dn = rhs
for _ in range(N - 1):
    Dn = op.add(rhs, op.matmul(Bn, Dn, False, False))
O = op.add(op.multiply(A, QS), op.matmul(op.multiply(op.matmul(Q, K, False, True), M), Dn, False, False))
for i in range(N):
    oi = op.slice(O, c64([i]), c64([i + 1]), c64([1]), c64([2]))
    u("out", i).output(0).replace(oi.output(0))
# results: drop state{i}, add the factors
keep = []
for r in m.get_results():
    names = r.output(0).get_names() | r.input_value(0).get_names()
    if any(n.startswith("state") for n in names):
        continue
    keep.append(r)
new = []
for t, nm in ((S0, "sbase"), (K, "kcur"), (Dn, "dnew"), (G, "gcum")):
    r = op.result(op.convert(t, ov.Type.f16) if nm != "gcum" else t)
    r.output(0).get_tensor().set_names({nm})
    new.append(r)
pp = m.get_parameters() + [kp, dp, w, al]
m2 = ov.Model(keep + new, pp, m.get_friendly_name())
m2.validate_nodes_and_infer_types()
ov.save_model(m2, outp, compress_to_fp16=False)
print(f"K={N}: results {[next(iter(r.output(0).get_names()), r.get_friendly_name()) for r in m2.get_results()]} -> {outp}")
