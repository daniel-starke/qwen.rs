"""Kronecker rotations for int8 activations (QuaRot-style): x R per token is
quantized instead of x, and the weight is stored as W R (exact: x R (W R)^T
= x W^T). R = diag(sign) (Ra kron Hb kron Hb) over k = a * b * b channels
(b = 32, Ra a random orthogonal [a, a], Hb a Hadamard): it spreads the
per-token outlier channels the gated products (out / o / down projection
inputs) have, and costs ~2 (a + 2 b) FLOP per element as three small
MatMuls in the graph. Measured on real activations (`actq.py --rot`): per-
token int8 error of down_proj 9.9 % -> 0.6 % with int8 weights."""
import numpy as np

B = 32


def hadamard(n):
    h = np.ones((1, 1))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def rot_factors(k, seed=0):
    a = k // (B * B)
    assert a * B * B == k, k
    rng = np.random.default_rng(seed)
    ra = np.linalg.qr(rng.standard_normal((a, a)))[0] if a > 1 else np.ones((1, 1))
    return rng.choice([-1.0, 1.0], k), ra, hadamard(B)


def rotate(X, f):
    """X [.., k] @ R (numpy)."""
    sg, ra, hb = f
    a = ra.shape[0]
    t = (X * sg).reshape(-1, a, B, B)
    t = np.einsum("nijk,il->nljk", t, ra)
    t = np.einsum("nijk,jl->nilk", t, hb)
    t = np.einsum("nijk,kl->nijl", t, hb)
    return t.reshape(X.shape)


def rotate_graph(op, x, f, dt=np.float16):
    """x [1, S, k] @ R as OpenVINO ops: sign multiply, then the three factors
    as MatMuls with constant left / right operands."""
    sg, ra, hb = f
    a = ra.shape[0]
    k = sg.size
    shp = list(x.get_output_partial_shape(0).get_shape()) if hasattr(x, "get_output_partial_shape") else list(x.get_partial_shape().get_shape())
    c = lambda v: op.constant(np.ascontiguousarray(v, dt))  # a transposed view comes out as i16
    i64 = lambda v: op.constant(np.asarray(v, np.int64))
    n = int(np.prod(shp[:-1]))
    t = op.multiply(x, c(sg))
    t = op.reshape(t, i64([n, a, B, B]), False)
    t = op.matmul(t, c(hb), False, False)  # last axis
    t = op.matmul(c(hb.T), t, False, False)  # axis 2
    if a > 1:
        t = op.reshape(op.matmul(c(ra.T), op.reshape(t, i64([n, a, B * B]), False), False, False), i64([n, a, B, B]), False)
    return op.reshape(t, i64(shp), False)


def block_factors(k, b, seed=0, periodic=False):
    """Block-diagonal Hadamard of block b (random input signs): one MatMul
    with K = b in the graph, which maps onto the DPU far better than the
    32-wide Kronecker factors (layer 8: 109 vs 61 ms unrotated). Accuracy on
    real activations close to the full rotation at b = 256-512."""
    assert k % b == 0, (k, b)
    if periodic:
        # the same b signs in every block, folded into the matrix: one MatMul,
        # no element-wise sign multiply (a SHAVE pass over the activation)
        sg = np.random.default_rng(seed).choice([-1.0, 1.0], b)
        return ("block", np.ones(k), sg.reshape(-1, 1) * hadamard(b))
    return ("block", np.random.default_rng(seed).choice([-1.0, 1.0], k), hadamard(b))


def rotate_any(X, f):
    if isinstance(f[0], str):
        _, sg, hb = f
        return ((X * sg).reshape(-1, hb.shape[0]) @ hb).reshape(X.shape)
    return rotate(X, f)


def rotate_graph_any(op, x, f, dt=np.float16):
    if not isinstance(f[0], str):
        return rotate_graph(op, x, f, dt)
    _, sg, hb = f
    b = hb.shape[0]
    shp = list(x.get_output_partial_shape(0).get_shape()) if hasattr(x, "get_output_partial_shape") else list(x.get_partial_shape().get_shape())
    i64 = lambda v: op.constant(np.asarray(v, np.int64))
    t = x if np.all(sg == 1) else op.multiply(x, op.constant(np.ascontiguousarray(sg, dt)))
    t = op.matmul(op.reshape(t, i64([-1, b]), False), op.constant(np.ascontiguousarray(hb, dt)), False, False)
    return op.reshape(t, i64(shp), False)
