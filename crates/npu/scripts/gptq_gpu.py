"""GPTQ of every lean-graph weight matrix on the GPU (PyTorch, CUDA), writing
the per-layer codes `wq.py` reads (CODES/L{L}.{name}.npz: packed int4 `q`,
scales `s`), so the builders (`NPU_WQ=gptq:HDIR`) only load them. The same
algorithm as `wq.gptq` (static per-row / per-group scales from
`wq._scales`, descending-diagonal column order, lazy 128-column blocks);
the numpy one takes ~15 min per layer, this one ~20 s.

    python gptq_gpu.py HDIR CODES [--layers A-B] [--src BF16DIR] [--rotb 256]

Matrices are built from the BF16 original as the builders build them (rows
fused in the same order, rotated by `quantrot.block_factors(k, ROTB)`); the
builders take codes and scales from the file (`wq.quant`, matched by layer,
name and shape).
Runs in the main Python (torch); needs no OpenVINO."""
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bf16src  # noqa: E402
import wq  # noqa: E402
from quantrot import block_factors, rotate_any  # noqa: E402

a = sys.argv[1:]


def opt(name, d, conv=str):
    if name in a:
        i = a.index(name)
        v = a[i + 1]
        del a[i:i + 2]
        return conv(v)
    return d


SRC = opt("--src", os.environ.get("NPU_SRC")) or sys.exit("--src DIR or NPU_SRC: the BF16 checkpoint")
ROTB = opt("--rotb", 256, int)
LA, LB = (int(v) for v in opt("--layers", "0-63").split("-"))
DAMP = float(os.environ.get("NPU_WQ_DAMP", "0.01"))
hdir, codes = a
os.makedirs(codes, exist_ok=True)
bf16src.SRC = SRC
dev = torch.device("cuda")


def w_(L, mod):
    return bf16src.tensor(f"model.language_model.layers.{L}.{mod}.weight").astype(np.float32)


def matrices(L):
    """{name: unrotated fused weight [n, k]} as the builders fuse them."""
    if L % 4 == 3:
        Wq = w_(L, "self_attn.q_proj")
        Wq4 = Wq.reshape(24, 2, 256, 5120)
        proj = np.concatenate([Wq4[:, 0].reshape(-1, 5120), Wq4[:, 1].reshape(-1, 5120),
                               w_(L, "self_attn.k_proj"), w_(L, "self_attn.v_proj")], 0)
        out = w_(L, "self_attn.o_proj")
    else:
        # in_proj_a / b (the last 96 rows of the builders' fused matrix) stay int8
        proj = np.concatenate([w_(L, "linear_attn.in_proj_qkv"), w_(L, "linear_attn.in_proj_z")], 0)
        out = w_(L, "linear_attn.out_proj")
    gu = np.concatenate([w_(L, "mlp.gate_proj"), w_(L, "mlp.up_proj")], 0)
    return {"proj": proj, "out": out, "gu": gu, "down": w_(L, "mlp.down_proj")}


@torch.no_grad()
def scales(w, lo=-8, hi=7, grid=21):
    """`wq._row_scales` on the GPU (per row; clip ratios 0.5 .. 1.0)."""
    W = torch.tensor(w, device=dev)
    amax = W.abs().amax(1, keepdim=True).clamp_min(1e-12)
    best_e = torch.full_like(amax, float("inf"))
    best_s = amax / hi
    for r in np.linspace(0.5, 1.0, grid):
        sc = amax * float(r) / hi
        e = ((torch.clamp(torch.round(W / sc), lo, hi) * sc - W) ** 2).sum(1, keepdim=True)
        m = e < best_e
        best_e = torch.where(m, e, best_e)
        best_s = torch.where(m, sc, best_s)
    return best_s.cpu().numpy().astype(np.float32)


def rot_h(H, f):
    """R^T H R on the GPU (R: block Hadamard with signs, `rotate_any`)."""
    _, sg, hb = f
    b = hb.shape[0]
    sg = torch.tensor(sg, dtype=torch.float32, device=dev)
    hb = torch.tensor(hb, dtype=torch.float32, device=dev)
    k = H.shape[0]
    t = (H * sg[None, :]).reshape(k, k // b, b) @ hb
    t = t.reshape(k, k).T.contiguous()
    t = (t * sg[None, :]).reshape(k, k // b, b) @ hb
    return t.reshape(k, k)


@torch.no_grad()
def gptq(w, H, sc, lo=-8, hi=7, bs=128):
    W = torch.tensor(w, device=dev)
    n, k = W.shape
    S = torch.tensor(sc, device=dev)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    perm = torch.argsort(torch.diag(H), descending=True, stable=True)
    W = W[:, perm]
    H = H[perm][:, perm]
    H += DAMP * torch.mean(torch.diag(H)) * torch.eye(k, device=dev)
    # the factorization on the CPU: for k = 17408 the chain holds ~4 x 1.2 GB
    # and the 6 GB card pages to system memory (down_proj 252 s -> ~10 s)
    Hc = H.cpu()
    del H
    torch.cuda.empty_cache()
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hc)), upper=True).to(dev)
    del Hc
    gidx = (perm // (k // S.shape[1])).tolist()
    Q = torch.empty((n, k), dtype=torch.int8, device=dev)
    for i1 in range(0, k, bs):
        i2 = min(i1 + bs, k)
        W1 = W[:, i1:i2].clone()
        E1 = torch.empty_like(W1)
        U1 = U[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = W1[:, i]
            s_ = S[:, gidx[i1 + i]]
            q = torch.clamp(torch.round(col / s_), lo, hi)
            Q[:, i1 + i] = q.to(torch.int8)
            err = (col - q * s_) / U1[i, i]
            W1[:, i:] -= err[:, None] * U1[i, i:][None, :]
            E1[:, i] = err
        if i2 < k:
            W[:, i2:] -= E1 @ U[i1:i2, i2:]
    out = torch.empty_like(Q)
    out[:, perm] = Q
    return out.cpu().numpy()


for L in range(LA, LB + 1):
    t0 = time.time()
    for name, w in matrices(L).items():
        dst = os.path.join(codes, f"L{L}.{name}.npz")
        if os.path.exists(dst):
            continue
        f = block_factors(w.shape[1], ROTB)
        wr = np.asarray(rotate_any(w, f), np.float32)
        assert wq.G == 0, "groups: not on the GPU path"
        sc = scales(wr)
        H = torch.tensor(np.load(os.path.join(hdir, f"L{L}.{name}.npy")), device=dev)
        q = gptq(wr, rot_h(H, f), sc)
        del H
        torch.cuda.empty_cache()
        np.savez(dst, q=wq.pack4(q), s=sc.astype(np.float32))
    print(f"layer {L}: {time.time() - t0:.0f} s", flush=True)
