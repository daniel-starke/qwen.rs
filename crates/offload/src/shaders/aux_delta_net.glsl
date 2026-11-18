#version 450
#extension GL_EXT_scalar_block_layout : enable
// Gated delta-net per-head recurrence (SSM). Persistent state `st` is [Hv][S][S]
// (row-major), `k`/`q` are [Hk][S], `v`/`o` are [Hv][S]. Implements
// delta_net_head exactly:
//   st *= g;  sk = stᵀ k;  d = (v - sk) * beta;
//   st += k ⊗ d;  o = stᵀ q / sqrt(S)
// `bg` is length 2*Hv: bg[h]=beta_s, bg[Hv+h]=gate_h.
//
// One invocation per (head, column) pair: index = h*S + j, dispatch
// ceil(Hv*S / 256) workgroups. Every phase above is column-local: column j only
// ever needs its own `sk[j]`/`d[j]`, so no barrier and no cross-thread traffic
// are required, and adjacent j read adjacent `st` addresses (coalesced). The
// earlier one-invocation-per-head version left Hv (32) threads doing S*S work
// each with a strided walk over a 2 MB state, which dominated fused SSM decode.
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) buffer St { float st[]; };
layout(set = 0, binding = 1, std430) readonly buffer Kk { float kk[]; };
layout(set = 0, binding = 2, std430) readonly buffer Qq { float qq[]; };
layout(set = 0, binding = 3, std430) readonly buffer Vv { float vv[]; };
layout(set = 0, binding = 4, std430) writeonly buffer Oo { float oo[]; };
layout(set = 0, binding = 5, std430) readonly buffer Bg { float bg[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };
void main() {
    uint S = p[0];
    uint Hk = p[1];
    uint Hv = p[2];
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= Hv * S) return;
    uint h = idx / S;
    uint j = idx - h * S;
    uint kh = h % Hk;
    float beta = bg[h];
    float g = bg[Hv + h];
    uint ks = kh * S;
    uint st0 = h * S * S;
    float qscale = 1.0 / sqrt(float(S));

    // st[:, j] *= g  and  sk = sum_i st[i][j] * k[i]
    float acc = 0.0;
    for (uint i = 0u; i < S; i++) {
        float sv = st[st0 + i * S + j] * g;
        st[st0 + i * S + j] = sv;
        acc += sv * kk[ks + i];
    }
    float d = (vv[h * S + j] - acc) * beta;
    // st[:, j] += k ⊗ d  and  o[j] = sum_i st[i][j] * q[i] / sqrt(S)
    float o = 0.0;
    for (uint i = 0u; i < S; i++) {
        float sv = st[st0 + i * S + j] + kk[ks + i] * d;
        st[st0 + i * S + j] = sv;
        o += sv * qq[ks + i];
    }
    oo[h * S + j] = o * qscale;
}
