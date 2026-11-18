#version 450
#extension GL_EXT_integer_dot_product : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
// The block indexer's scores of a Qwen3.8-Flash-Next prefill attention
// sub-chunk on the scan device (the iGPU, `offload::FlashAttn::score_begin`):
// per query and pooled block key, sum over the 4 indexer heads of
// max(q_h . k_b, 0). The NPU ran it as a graph behind the sub-chunk's
// projections, ~6.6 ms of a ~24 ms NPU cycle per sub-chunk at 53k
// positions, while the iGPU idled between attentions.
// int8 x int8 `dp4a` with a scale per key row and per query head row
// (absmax / 127, made on the host). The layout of `exp_gemm.glsl`: a
// workgroup is NG groups of 32 lanes over RW = 32 RT blocks; lane l holds
// blocks l + 32 i and its group's 8 query heads (2 queries x 4 heads),
// the keys in shared memory (the whole 128-wide row: 8 uvec4, rows 9
// apart), the query heads in registers, lane j's read by the others with
// `subgroupBroadcast`. A group's 32 lanes are whole subgroups of 8, 16 or
// 32. The f32 form (both operands in shared memory) ran 240 GFLOP/s.
//   q   [n][4][32] int8 words, qs [n][4] f32 scales
//   k   [cap][32]  int8 words, ks [cap] f32 scales (the layer's pooled,
//       normed, roped block keys)
//   o   [n][stride] u32: the score's f32 bits (>= +0, so they order as
//       integers), blocks [0, nb)
//   p   n, nb, stride
// Grid: x = blocks / RW, y = queries / (2 NG).
#define RT 2u
#define NG 8u
#define LS (32u * NG)
#define RW (32u * RT)
#define QT (2u * NG)
#define ST 9u
layout(local_size_x = LS) in;
layout(set = 0, binding = 0, std430) readonly buffer Qb { uvec4 q4[]; };
layout(set = 0, binding = 1, std430) readonly buffer Kb { uvec4 k4[]; };
layout(set = 0, binding = 2, std430) readonly buffer Ksb { float ksc[]; };
layout(set = 0, binding = 3, std430) readonly buffer Qsb { float qsc[]; };
layout(set = 0, binding = 4, std430) writeonly buffer Ob { uint o[]; };
layout(set = 0, binding = 8, std140) uniform Params {
    uint n;
    uint nb;
    uint stride;
};

shared uvec4 sk[RW * ST];

void main() {
    uint t = gl_LocalInvocationID.x;
    uint b0 = gl_WorkGroupID.x * RW;
    uint t0 = gl_WorkGroupID.y * QT;
    uint ln = t % 32u, tg = t / 32u;
    for (uint r = t; r < RW * 8u; r += LS) {
        uint b = r / 8u;
        sk[b * ST + r % 8u] = b0 + b < nb ? k4[(b0 + b) * 8u + r % 8u] : uvec4(0u);
    }
    // this lane's query head (the subgroup's lanes j < 8 hold heads tg 8 + j)
    uint qh = t0 * 4u + tg * 8u + gl_SubgroupInvocationID % 8u;
    bool tv = qh < n * 4u;
    uvec4 xv[8];
    [[unroll]] for (uint p = 0u; p < 8u; p++) {
        xv[p] = tv ? q4[qh * 8u + p] : uvec4(0u);
    }
    barrier();
    int acc[RT * 8u];
    [[unroll]] for (uint i = 0u; i < RT * 8u; i++) { acc[i] = 0; }
    [[unroll]] for (uint p = 0u; p < 8u; p++) {
        ivec4 wv[RT];
        [[unroll]] for (uint i = 0u; i < RT; i++) { wv[i] = ivec4(sk[(ln + 32u * i) * ST + p]); }
#define QJ(j) { \
            ivec4 x = ivec4(subgroupBroadcast(xv[p], j)); \
            [[unroll]] for (uint i = 0u; i < RT; i++) { \
                int a = dotPacked4x8AccSatEXT(wv[i].x, x.x, acc[i * 8u + j]); \
                a = dotPacked4x8AccSatEXT(wv[i].y, x.y, a); \
                a = dotPacked4x8AccSatEXT(wv[i].z, x.z, a); \
                acc[i * 8u + j] = dotPacked4x8AccSatEXT(wv[i].w, x.w, a); \
            } }
        QJ(0u) QJ(1u) QJ(2u) QJ(3u) QJ(4u) QJ(5u) QJ(6u) QJ(7u)
    }
    [[unroll]] for (uint a = 0u; a < 2u; a++) {
        uint tq = t0 + tg * 2u + a;
        if (tq >= n) {
            continue;
        }
        vec4 sq = vec4(qsc[tq * 4u], qsc[tq * 4u + 1u], qsc[tq * 4u + 2u], qsc[tq * 4u + 3u]);
        [[unroll]] for (uint i = 0u; i < RT; i++) {
            uint b = b0 + ln + 32u * i;
            if (b < nb) {
                vec4 d = vec4(acc[i * 8u + a * 4u], acc[i * 8u + a * 4u + 1u],
                              acc[i * 8u + a * 4u + 2u], acc[i * 8u + a * 4u + 3u]);
                vec4 m = max(d * sq, vec4(0.0));
                float s = (m.x + m.y + m.z + m.w) * ksc[b];
                o[tq * stride + b] = floatBitsToUint(s) & 0x7fffffffu;
            }
        }
    }
}
