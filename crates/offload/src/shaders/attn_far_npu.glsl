#version 450
#extension GL_EXT_scalar_block_layout : enable
// Far attention of the NPU engine on the scan device (the iGPU), pass 2 of 2
// (`attn_far_qk_npu.glsl` is pass 1): softmax of one block's scores against
// the running maximum, and O += P V. The running (O, max, sum) live in
// buffers across the blocks of a query (one pair of dispatches per block):
//   sc  [24 * S][CAP] f32: the block's scores (overwritten: probabilities)
//   v   [CAP][4][256] f16, one V block
//   o   [24 * S][256] f32, sum of exp(score - m) * v, NOT normalized
//   ml  [24 * S][2] f32: m (running max), l (sum of exp(score - m));
//       the host sets m = -1e30 before a query's first block
// The host merges o / l with the NPU's blocks by log sum exp (m + ln l).
// One workgroup = one KV group x QT = 4 positions = R = 24 rows; in P V an
// invocation owns one dim-quad of one position for all 6 heads (10 loads per
// 96 multiply-adds, V read straight from its buffer). No subgroup operations
// (the subgroup size is 8 / 16 / 32 on Xe).
#define HD 256u
#define HPG 6u
#define QT 4u
#define R 24u
// invocations per row in the softmax
#define TPR 10u

layout(local_size_x = 256) in;

layout(set = 0, binding = 2, std430) readonly buffer Vb { uvec2 v2[]; };
layout(set = 0, binding = 3, std430) buffer Ob { vec4 o4[]; };
layout(set = 0, binding = 4, std430) buffer Mb { vec2 ml[]; };
layout(set = 0, binding = 5, std430) buffer Sb { vec4 sc4[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };

shared float sm[R];
shared float sl[R];
shared float sf[R];
shared float pr_[R * TPR];

vec4 half4(uvec2 u) {
    return vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y));
}

void main() {
    uint S = p[0];
    uint cap4 = p[2] / 4u;
    uint tid = gl_LocalInvocationID.x;
    uint tb = gl_WorkGroupID.x * QT;
    uint g = gl_WorkGroupID.y;

    // row r = position * 6 + head
    if (tid < R) {
        vec2 s = ml[(g * HPG + tid % HPG) * S + tb + tid / HPG];
        sm[tid] = s.x;
        sl[tid] = s.y;
    }
    // P V roles: dim-quad d4, position pp, all 6 heads.
    uint d4 = tid & 63u;
    uint pp = tid >> 6u;
    uint row0 = g * HPG * S + tb + pp;
    vec4 acc[HPG];
    for (uint h = 0u; h < HPG; h++) {
        uint row = row0 + h * S;
        acc[h] = (ml[row].x < -1e29) ? vec4(0.0) : o4[row * (HD / 4u) + d4];
    }

    // softmax roles: row sr, score quads [s0, s1)
    uint sr = tid / TPR;
    uint per = (cap4 + TPR - 1u) / TPR;
    uint sbase = ((g * HPG + sr % HPG) * S + tb + sr / HPG) * cap4;
    uint s0 = min((tid % TPR) * per, cap4);
    uint s1 = min(s0 + per, cap4);
    if (sr < R) {
        vec4 m4 = vec4(-1e30);
        for (uint i = s0; i < s1; i++) {
            m4 = max(m4, sc4[sbase + i]);
        }
        pr_[tid] = max(max(m4.x, m4.y), max(m4.z, m4.w));
    }
    barrier();
    float mn = 0.0;
    if (sr < R) {
        mn = sm[sr];
        for (uint i = 0u; i < TPR; i++) {
            mn = max(mn, pr_[sr * TPR + i]);
        }
    }
    barrier();
    if (sr < R) {
        vec4 t4 = vec4(0.0);
        for (uint i = s0; i < s1; i++) {
            vec4 e = exp(sc4[sbase + i] - mn);
            sc4[sbase + i] = e;
            t4 += e;
        }
        pr_[tid] = (t4.x + t4.y) + (t4.z + t4.w);
    }
    memoryBarrierBuffer();
    barrier();
    if (sr < R && tid % TPR == 0u) {
        float sum = 0.0;
        for (uint i = 0u; i < TPR; i++) {
            sum += pr_[tid + i];
        }
        float f = exp(sm[sr] - mn);
        sf[sr] = f;
        sl[sr] = sl[sr] * f + sum;
        sm[sr] = mn;
    }
    barrier();
    for (uint h = 0u; h < HPG; h++) {
        acc[h] *= sf[pp * HPG + h];
    }
    uint v0i = g * (HD / 4u) + d4;
    for (uint kq = 0u; kq < cap4; kq++) {
        uint vi = v0i + kq * 4u * HD;
        vec4 a0 = half4(v2[vi]);
        vec4 a1 = half4(v2[vi + HD]);
        vec4 a2 = half4(v2[vi + 2u * HD]);
        vec4 a3 = half4(v2[vi + 3u * HD]);
        for (uint h = 0u; h < HPG; h++) {
            vec4 pr = sc4[(row0 + h * S) * cap4 + kq];
            acc[h] += pr.x * a0 + pr.y * a1 + pr.z * a2 + pr.w * a3;
        }
    }

    for (uint h = 0u; h < HPG; h++) {
        o4[(row0 + h * S) * (HD / 4u) + d4] = acc[h];
    }
    barrier();
    if (tid < R) {
        ml[(g * HPG + tid % HPG) * S + tb + tid / HPG] = vec2(sm[tid], sl[tid]);
    }
}
