#version 450
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_clustered : enable
// Gated delta-net over a whole prefill chunk of N positions, the recurrence
// `quant::delta_net_cols` runs per position (per head h, key head kh = h % Hk):
//   st *= g;  sk = st^T k;  d = (v - sk) * beta;  st += k (x) d;  o = st^T q / sqrt(S)
//
// Column j of a head's state only mixes with itself, so the state never leaves
// registers: one column is held by LANES consecutive invocations, lane l owning
// rows l, l + LANES, ... (R = S / LANES floats), and the two dot products over
// the key dim are clustered subgroup adds. A workgroup is COLS = 256 / LANES
// columns of one head; dispatch Hv * S / COLS workgroups. k / q of TB positions
// are staged in shared memory (q pre-scaled), read as broadcasts.
//
// Layouts (as the CPU prefill's pass 2): st [Hv][S][S] (row = key dim) read at
// the start and written at the end; k, q [Hk][N][S]; v, o [Hv][N][S];
// bg = beta [N][Hv] then gate [N][Hv] (gate already exp'd).
#ifndef LANES
#define LANES 8
#endif
#define S 128
#define R (S / LANES)
#define COLS (256 / LANES)
#ifndef TB
#define TB 16
#endif
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) buffer St { float st[]; };
layout(set = 0, binding = 1, std430) readonly buffer Kk { float kk[]; };
layout(set = 0, binding = 2, std430) readonly buffer Qq { float qq[]; };
layout(set = 0, binding = 3, std430) readonly buffer Vv { float vv[]; };
layout(set = 0, binding = 4, std430) writeonly buffer Oo { float oo[]; };
layout(set = 0, binding = 5, std430) readonly buffer Bg { float bg[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };

shared float sk_[TB * S];
shared float sq_[TB * S];

void main() {
    uint N = p[0];
    uint Hk = p[1];
    uint Hv = p[2];
    uint nb = S / COLS;
    uint h = gl_WorkGroupID.x / nb;
    uint j = (gl_WorkGroupID.x % nb) * COLS + gl_LocalInvocationID.x / LANES;
    uint l = gl_LocalInvocationID.x % LANES;
    uint tid = gl_LocalInvocationID.x;
    uint kh = h % Hk;
    uint st0 = h * S * S + j;
    float qscale = 1.0 / sqrt(float(S));

    float s[R];
    for (uint m = 0u; m < R; m++) s[m] = st[st0 + (l + LANES * m) * S];

    for (uint t0 = 0u; t0 < N; t0 += TB) {
        barrier();
        for (uint e = tid; e < TB * S; e += 256u) {
            uint t = t0 + e / S;
            uint i = e % S;
            float kv = 0.0, qv = 0.0;
            if (t < N) {
                kv = kk[(kh * N + t) * S + i];
                qv = qq[(kh * N + t) * S + i] * qscale;
            }
            sk_[e] = kv;
            sq_[e] = qv;
        }
        barrier();
        uint tn = min(uint(TB), N - t0);
        for (uint tt = 0u; tt < tn; tt++) {
            uint t = t0 + tt;
            float beta = bg[t * Hv + h];
            float g = bg[N * Hv + t * Hv + h];
            float vj = vv[(h * N + t) * S + j];
            uint b = tt * S + l;
            float acc = 0.0;
            for (uint m = 0u; m < R; m++) {
                s[m] *= g;
                acc += s[m] * sk_[b + LANES * m];
            }
            acc = subgroupClusteredAdd(acc, LANES);
            float d = (vj - acc) * beta;
            float o = 0.0;
            for (uint m = 0u; m < R; m++) {
                s[m] += sk_[b + LANES * m] * d;
                o += s[m] * sq_[b + LANES * m];
            }
            o = subgroupClusteredAdd(o, LANES);
            if (l == 0u) oo[(h * N + t) * S + j] = o;
        }
    }
    for (uint m = 0u; m < R; m++) st[st0 + (l + LANES * m) * S] = s[m];
}
