"""Qwen3.8-Flash-Next (qwen4_exp) in f32 from the BF16 checkpoint, one layer at a
time: the truth logits the Flash-Next builder and engine are checked against.

A self-contained re-implementation of transformers' `modular_qwen4_exp.py`
(no transformers needed; torch + safetensors). One sequence from zero state,
no padding, text positions only (MRoPE t = h = w).

    python flash_ref.py BF16_DIR IDS.txt OUT.f32 [--layers N] [--dump DIR]

IDS.txt: one token id per line. OUT.f32: logits `[T][vocab]` f32 (the
format `nbcheck` reads). `--dump DIR` writes the streams after every layer
(`L{l}.f32`, `[T][10240]`) and the n-gram ids (`ngram_ids.i64`, `[T][16]`).
"""

import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)

P = "model.language_model."


class Slice:
    """Rows `[a:b]` of a tensor along dim 0, read from the file (no mmap)."""

    def __init__(self, ck, name):
        self.ck, self.name = ck, name
        self.fn, self.dtype, self.shape, self.off = ck.info(name)

    def get_shape(self):
        return self.shape

    def __getitem__(self, s):
        row = 1
        for d in self.shape[1:]:
            row *= d
        a, b = s.start, s.stop
        return self.ck.read(self.fn, self.dtype, self.off, a * row, (b - a) * row).reshape(b - a, *self.shape[1:])


class Ck:
    DT = {"BF16": (torch.bfloat16, 2), "F32": (torch.float32, 4), "I64": (torch.int64, 8), "F16": (torch.float16, 2)}

    def __init__(self, d):
        self.d = d
        self.map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.hdr = {}
        self.fh = {}

    def info(self, name):
        fn = self.map[name]
        if fn not in self.hdr:
            with open(os.path.join(self.d, fn), "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                self.hdr[fn] = (8 + n, json.loads(f.read(n)))
        base, h = self.hdr[fn]
        e = h[name]
        return fn, e["dtype"], e["shape"], base + e["data_offsets"][0]

    def read(self, fn, dtype, off, first, count):
        tdt, sz = self.DT[dtype]
        if fn not in self.fh:
            self.fh[fn] = open(os.path.join(self.d, fn), "rb")
        f = self.fh[fn]
        f.seek(off + first * sz)
        buf = bytearray(f.read(count * sz))
        return torch.frombuffer(buf, dtype=tdt)

    def t(self, name):
        fn, dt, shape, off = self.info(name)
        n = 1
        for d in shape:
            n *= d
        x = self.read(fn, dt, off, 0, n).reshape(shape)
        return x if dt == "I64" else x.float()

    def sl(self, name):
        return Slice(self, name)


def rms(x, w=None, eps=1e-6, group=None):
    """Qwen4ExpTextRMSNorm: weight stored less one; `group` norms each slice."""
    sh = x.shape
    if group is not None:
        x = x.reshape(*sh[:-1], -1, group)
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    y = y.reshape(sh)
    return y if w is None else y * (1.0 + w)


def rope(x, cos, sin):
    """x [T, H, D]; cos / sin [T, rd]: rotate the first rd dims (rotate_half)."""
    rd = cos.shape[-1]
    xr, xp = x[..., :rd], x[..., rd:]
    h = rd // 2
    rot = torch.cat([-xr[..., h:], xr[..., :h]], dim=-1)
    xr = xr * cos[:, None, :] + rot * sin[:, None, :]
    return torch.cat([xr, xp], dim=-1)


class Model:
    def __init__(self, d):
        self.ck = Ck(d)
        self.cfg = json.load(open(os.path.join(d, "config.json")))["text_config"]
        c = self.cfg
        self.W = c["hidden_size"]
        self.HC = c["hc_count"]
        self.eos = c["eos_token_id"] if not isinstance(c["eos_token_id"], list) else c["eos_token_id"][0]
        rd = int(c["head_dim"] * c["rope_parameters"]["partial_rotary_factor"])
        theta = c["rope_parameters"]["rope_theta"]
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, rd, 2, dtype=torch.float64) / rd))

    def cos_sin(self, pos):
        f = pos[:, None].double() * self.inv_freq[None, :]
        f = torch.cat([f, f], dim=-1)
        return f.cos().float(), f.sin().float()

    # --- hyper-connections ---------------------------------------------------
    def hc(self, pre, x, combine=True):
        ck = self.ck
        xn = rms(x, ck.t(pre + "hc_norm.weight"), group=self.W)
        m = F.silu((xn @ ck.t(pre + "input_mix_weight_down.weight").T) / self.HC)
        m = torch.sigmoid(m @ ck.t(pre + "input_mix_weight_up.weight").T)
        T = x.shape[0]
        mixed = (m.view(T, self.HC, self.W) * xn.view(T, self.HC, self.W)).mean(dim=1)
        if not combine:
            return mixed
        inj = 2 * torch.sigmoid((xn @ ck.t(pre + "block_inject_weight.weight").T) / self.HC)
        return mixed, inj

    def inject(self, x, y, inj):
        T = x.shape[0]
        return x + (y[:, None, :] * inj[:, :, None]).reshape(T, self.HC * self.W)

    # --- PLE / n-gram embedding ----------------------------------------------
    def ngram_ids(self, l, ids):
        ck, c = self.ck, self.cfg
        pre = P + f"layers.{l}.ple.ple_embedding."
        mult = ck.t(pre + "layer_multipliers")
        sizes = ck.t(pre + "ngram_heads_vocab_sizes")
        offs = ck.t(pre + "ngram_heads_offsets")
        n = c["ngram_size"]
        hpn = c["heads_per_ngram"]
        hist = torch.cat([torch.full((n - 1,), self.eos, dtype=torch.long), ids.long()])
        L = hist.shape[0]
        posn = torch.arange(L)
        eosp = torch.where(hist == self.eos, posn, torch.full_like(posn, -1))
        prev_incl = torch.cummax(eosp, dim=0).values
        prev = torch.cat([torch.tensor([-1]), prev_incl[:-1]])
        pin = posn - (prev + 1)

        def shifted(s):
            src = posn - s
            g = hist[src.clamp_min(0)]
            ok = (pin >= s) & (src >= 0)
            return torch.where(ok, g, torch.full_like(g, self.eos))

        sh = [shifted(s) for s in range(n)]
        blocks = []
        for g in range(2, n + 1):
            a = (g - 2) * hpn
            mixed = sh[0] * mult[0]
            for p in range(1, g):
                mixed = torch.bitwise_xor(mixed, sh[p] * mult[p])
            blocks.append(torch.remainder(mixed[:, None], sizes[a : a + hpn][None, :]) + offs[a : a + hpn][None, :])
        return torch.cat(blocks, dim=-1)[n - 1 :]

    def ngram_rows(self, l, nid):
        """Rows of the 128-shard table for ids [T, 16] -> [T, 16 * 160]."""
        pre = P + f"layers.{l}.ple.ple_embedding.ngram_embedding."
        s0 = self.ck.sl(pre + "shard_0.weight")
        per = s0.get_shape()[0]
        T, Hn = nid.shape
        out = None
        for t in range(T):
            for h in range(Hn):
                r = int(nid[t, h])
                row = self.ck.sl(pre + f"shard_{r // per}.weight")[r % per : r % per + 1].float()[0]
                if out is None:
                    out = torch.zeros(T, Hn, row.shape[0])
                out[t, h] = row
        return out.reshape(T, -1)

    def ple(self, l, x, ids, dump):
        ck = self.ck
        pre = P + f"layers.{l}.ple."
        nid = self.ngram_ids(l, ids)
        if dump:
            nid.numpy().tofile(os.path.join(dump, "ngram_ids.i64"))
        e = self.ngram_rows(l, nid)
        T = x.shape[0]
        key = rms(e @ ck.t(pre + "key_proj.weight").T, ck.t(pre + "norm_key.weight"), group=self.W).view(T, self.HC, self.W)
        val = e @ ck.t(pre + "value_proj.weight").T
        q = rms(x, ck.t(pre + "norm_query.weight"), group=self.W).view(T, self.HC, self.W)
        gate = (key * q).sum(-1, keepdim=True) / math.sqrt(self.W)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gv = (torch.sigmoid(gate) * val[:, None, :]).reshape(T, -1)
        gvn = rms(gv, ck.t(pre + "norm_conv.weight"), group=self.W)
        w = ck.t(pre + "conv1d.weight")  # [C, 1, K]
        K = w.shape[-1]
        dil = self.cfg["ngram_size"]
        pad = (K - 1) * dil
        h = F.pad(gvn.T[None], (pad, 0))
        conv = F.silu(F.conv1d(h, w, groups=w.shape[0], dilation=dil))[0].T
        return gv + conv

    # --- mixers --------------------------------------------------------------
    def delta(self, l, x):
        ck, c = self.ck, self.cfg
        pre = P + f"layers.{l}.linear_attn."
        T = x.shape[0]
        hk, hv, dk, dv = c["linear_num_key_heads"], c["linear_num_value_heads"], c["linear_key_head_dim"], c["linear_value_head_dim"]
        qkv = x @ ck.t(pre + "in_proj_qkv.weight").T
        z = (x @ ck.t(pre + "in_proj_z.weight").T).view(T, hv, dv)
        b = x @ ck.t(pre + "in_proj_b.weight").T
        a = x @ ck.t(pre + "in_proj_a.weight").T
        w = ck.t(pre + "conv1d.weight")
        K = w.shape[-1]
        qkv = F.silu(F.conv1d(F.pad(qkv.T[None], (K - 1, 0)), w, groups=w.shape[0]))[0].T
        q, k, v = torch.split(qkv, [hk * dk, hk * dk, hv * dv], dim=-1)
        q = q.reshape(T, hk, dk)
        k = k.reshape(T, hk, dk)
        v = v.reshape(T, hv, dv)
        beta = torch.sigmoid(b)
        g = -ck.t(pre + "A_log").exp() * F.softplus(a + ck.t(pre + "dt_bias"))
        rep = hv // hk
        q = q.repeat_interleave(rep, dim=1)
        k = k.repeat_interleave(rep, dim=1)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) / math.sqrt(dk)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        S = torch.zeros(hv, dk, dv)
        o = torch.zeros(T, hv, dv)
        for t in range(T):
            S = S * g[t].exp()[:, None, None]
            kv = (S * k[t][:, :, None]).sum(1)
            d = (v[t] - kv) * beta[t][:, None]
            S = S + k[t][:, :, None] * d[:, None, :]
            o[t] = (S * q[t][:, :, None]).sum(1)
        var = o.pow(2).mean(-1, keepdim=True)
        o = o * torch.rsqrt(var + c["rms_norm_eps"]) * ck.t(pre + "norm.weight")
        act = c.get("output_gate_type") or c["hidden_act"]
        o = o * (torch.sigmoid(z) if act == "sigmoid" else F.silu(z))
        return o.reshape(T, -1) @ ck.t(pre + "out_proj.weight").T

    def attn(self, l, x, pos):
        ck, c = self.ck, self.cfg
        pre = P + f"layers.{l}.self_attn."
        T = x.shape[0]
        hq, hkv, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        cos, sin = self.cos_sin(pos)
        allowed = self.indexer(l, x, pos, cos, sin)
        qg = (x @ ck.t(pre + "q_proj.weight").T).view(T, hq, 2 * d)
        q, gate = qg[..., :d], qg[..., d:].reshape(T, -1)
        q = rms(q, ck.t(pre + "q_norm.weight"))
        k = rms((x @ ck.t(pre + "k_proj.weight").T).view(T, hkv, d), ck.t(pre + "k_norm.weight"))
        v = (x @ ck.t(pre + "v_proj.weight").T).view(T, hkv, d)
        q = rope(q, cos, sin)
        k = rope(k, cos, sin)
        rep = hq // hkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        s = torch.einsum("thd,shd->hts", q, k) / math.sqrt(d)
        s = s.masked_fill(~allowed[None], float("-inf"))
        p = torch.softmax(s, dim=-1)
        o = torch.einsum("hts,shd->thd", p, v).reshape(T, -1)
        o = o * torch.sigmoid(gate)
        return o @ ck.t(pre + "o_proj.weight").T

    def indexer(self, l, x, pos, cos, sin):
        """QSA: [T, T] bool of the keys each query attends to (causal included)."""
        ck, c = self.ck, self.cfg
        pre = P + f"layers.{l}.self_attn.indexer."
        T = x.shape[0]
        nh, hd = c["indexer_n_heads"], c["indexer_head_dim"]
        cr = c["indexer_compress_ratio"]
        topb = c["indexer_budget"] // cr
        qk = x @ ck.t(pre + "index_qk_proj.weight").T
        q = rms(qk[:, : nh * hd].view(T, nh, hd), ck.t(pre + "q_layernorm.weight"))
        q = rope(q, cos, sin)
        raw = qk[:, nh * hd :]
        nbt = T // cr
        pooled = raw[: nbt * cr].view(nbt, cr, hd).mean(1)
        bk = rms(pooled, ck.t(pre + "k_layernorm.weight"))
        starts = torch.arange(nbt) * cr
        bk = rope(bk[:, None, :], cos[starts], sin[starts])[:, 0]
        sc = torch.relu(torch.einsum("thd,bd->tbh", q, bk)).sum(-1) / math.sqrt(hd)
        allowed = torch.zeros(T, T, dtype=torch.bool)
        for t in range(T):
            nb = (t + 1) // cr
            if nb > 0:
                sel = sc[t, :nb].topk(min(topb, nb)).indices
                idx = (sel[:, None] * cr + torch.arange(cr)[None]).flatten()
                allowed[t, idx] = True
            allowed[t, nb * cr : t + 1] = True
        return allowed

    def moe(self, pre, x):
        ck, c = self.ck, self.cfg
        T = x.shape[0]
        logits = x @ ck.t(pre + "gate.weight").T
        probs = torch.softmax(logits, dim=-1)
        tv, ti = probs.topk(c["num_experts_per_tok"], dim=-1)
        tv = tv / tv.sum(-1, keepdim=True)
        gu = ck.sl(pre + "experts.gate_up_proj")
        dn = ck.sl(pre + "experts.down_proj")
        out = torch.zeros_like(x)
        for e in torch.unique(ti).tolist():
            tok, slot = torch.where(ti == e)
            wgu = gu[e : e + 1].float()[0]
            wd = dn[e : e + 1].float()[0]
            h = x[tok] @ wgu.T
            ff = h.shape[-1] // 2
            h = F.silu(h[:, :ff]) * h[:, ff:]
            out.index_add_(0, tok, (h @ wd.T) * tv[tok, slot][:, None])
        se = pre + "shared_expert."
        sh = (F.silu(x @ ck.t(se + "gate_proj.weight").T) * (x @ ck.t(se + "up_proj.weight").T)) @ ck.t(se + "down_proj.weight").T
        sh = torch.sigmoid(x @ ck.t(pre + "shared_expert_gate.weight").T) * sh
        return out + sh, ti

    # --- whole model -----------------------------------------------------------
    def forward(self, ids, nlayers, dump):
        ck, c = self.ck, self.cfg
        T = ids.shape[0]
        pos = torch.arange(T)
        emb = ck.sl(P + "embed_tokens.weight")
        x0 = torch.stack([emb[i : i + 1].float()[0] for i in ids.tolist()])
        x = x0.repeat(1, self.HC)
        ple_ids = [] if os.environ.get("FLASH_NO_PLE") else [i - 1 for i in c.get("ple_layer_ids", [])]
        hist = {}
        for l in range(nlayers):
            t0 = time.time()
            lp = P + f"layers.{l}."
            if l in ple_ids:
                x = x + self.ple(l, x, ids, dump)
            h, inj = self.hc(lp + "attn_hyper_connection.", x)
            if c["layer_types"][l] == "linear_attention":
                y = self.delta(l, h)
            else:
                y = self.attn(l, h, pos)
            x = self.inject(x, y, inj)
            h, inj = self.hc(lp + "mlp_hyper_connection.", x)
            y, ti = self.moe(lp + "mlp.", h)
            x = self.inject(x, y, inj)
            hist[l] = torch.bincount(ti.flatten(), minlength=c["num_experts"]).tolist()
            if dump:
                x.numpy().tofile(os.path.join(dump, f"L{l}.f32"))
            print(f"layer {l:2d} {c['layer_types'][l][:6]} |x| {x.abs().mean():.4f} max {x.abs().max():.1f}  {time.time() - t0:.1f} s", flush=True)
        h = self.hc(P + "hyper_connection_mixer.", x, combine=False)
        if dump:
            h.numpy().tofile(os.path.join(dump, "final.f32"))
            json.dump(hist, open(os.path.join(dump, "experts.json"), "w"))
        return h @ ck.t("lm_head.weight").T


def main():
    a = sys.argv[1:]
    nl = None
    dump = None
    if "--layers" in a:
        i = a.index("--layers")
        nl = int(a[i + 1])
        del a[i : i + 2]
    if "--dump" in a:
        i = a.index("--dump")
        dump = a[i + 1]
        del a[i : i + 2]
        os.makedirs(dump, exist_ok=True)
    d, idf, out = a
    ids = torch.tensor([int(s) for s in open(idf).read().split()], dtype=torch.long)
    m = Model(d)
    nl = m.cfg["num_hidden_layers"] if nl is None else nl
    logits = m.forward(ids, nl, dump)
    logits.numpy().astype("float32").tofile(out)
    top = logits.argmax(-1)
    print("argmax of the last 8 positions:", top[-8:].tolist())


if __name__ == "__main__":
    main()
