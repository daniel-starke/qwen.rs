#version 450
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
// Far attention of the NPU engine on the scan device (the iGPU), pass 1 of 2
// of the fused form (`attn_far_pv_npu.glsl` is pass 2; the older pair is
// `attn_far_qk_npu.glsl` + `attn_far_npu.glsl`): the scores Q K^T of one K
// block AND the softmax update against the running maximum, so the second
// pass is P V alone. A workgroup owns every key of one (position, KV group)
// for its 6 heads, so the block's row maximum is a reduction over its own
// KSTEP invocations, and each invocation then turns exactly the scores it
// wrote into probabilities (its own writes: no barrier between them). The
// scores are swept twice by their writer instead of three times by another
// workgroup with 240 of 256 invocations busy.
// The reductions are subgroup operations: KSTEP = 8 is at most the smallest
// subgroup of every device here (Xe 8..32), and a subgroup never spans two
// workgroups, so the workgroup is one subgroup whatever width the compiler
// picks. The first form, through shared memory and two barriers, ran 13.9 ms
// per block on the iGPU against the two-pass 9.0. No subgroup size is
// required (forcing SIMD32 made the iGPU's prefill GEMM ~8x slower).
//   q   [4][6 * S][256] f16 (the pre graph's qg), row (g * 6 + h) * S + pos
//   k   [CAP][4][256] f16, one K block; keys [0, nk) count
//   ml  [24 * S][2] f32: m (running max), l (sum of exp(score - m)), updated;
//       the host sets m = -1e30 before a query's first block
//   sc  [24 * S][CAP] f32 out: exp(score - m_new), 0 for keys >= nk; then
//       at vec4 p[3] on, f = exp(m_old - m_new) per row (the factor pass 2
//       applies to o)
// Scores as `attn_far_qk_npu.glsl`: one invocation = one position x one KV
// group x 4 keys at a time, all 6 heads (10 loads per 24 dot products).
#define HD 256u
#define HPG 6u
#ifndef KSTEP
#define KSTEP 8u
#endif
layout(local_size_x = KSTEP) in;
layout(set = 0, binding = 0, std430) readonly buffer Qb { uvec2 q2[]; };
layout(set = 0, binding = 1, std430) readonly buffer Kb { uvec2 k2[]; };
layout(set = 0, binding = 4, std430) buffer Mb { vec2 ml[]; };
layout(set = 0, binding = 5, std430) buffer Sb { vec4 sc4[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };

vec4 half4(uvec2 u) {
    return vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y));
}

void main() {
    uint S = p[0];
    uint nk = p[1];
    uint cap4 = p[2] / 4u;
    uint fb = p[3];
    uint tid = gl_LocalInvocationID.x;
    uint pos = gl_WorkGroupID.y;
    uint g = gl_WorkGroupID.z;
    uint q0 = (g * HPG * S + pos) * (HD / 4u);
    uint row0 = g * HPG * S + pos;
    float lm[HPG];
    for (uint h = 0u; h < HPG; h++) {
        lm[h] = -1e30;
    }
    for (uint kq = tid; kq < cap4; kq += KSTEP) {
        uint k0 = kq * 4u * HD + g * (HD / 4u);
        vec4 sv[HPG];
        for (uint h = 0u; h < HPG; h++) {
            sv[h] = vec4(0.0);
        }
        for (uint dq = 0u; dq < HD / 4u; dq++) {
            vec4 ka = half4(k2[k0 + dq]);
            vec4 kc = half4(k2[k0 + HD + dq]);
            vec4 ke = half4(k2[k0 + 2u * HD + dq]);
            vec4 kg = half4(k2[k0 + 3u * HD + dq]);
            for (uint h = 0u; h < HPG; h++) {
                vec4 qv = half4(q2[q0 + h * S * (HD / 4u) + dq]);
                sv[h] += vec4(dot(qv, ka), dot(qv, kc), dot(qv, ke), dot(qv, kg));
            }
        }
        bvec4 live = lessThan(uvec4(kq * 4u) + uvec4(0u, 1u, 2u, 3u), uvec4(nk));
        for (uint h = 0u; h < HPG; h++) {
            vec4 s = mix(vec4(-1e30), sv[h], live);
            sc4[(row0 + h * S) * cap4 + kq] = s;
            lm[h] = max(lm[h], max(max(s.x, s.y), max(s.z, s.w)));
        }
    }
    // the new running maximum of each head's row, and its old one
    float mo[HPG];
    float mn[HPG];
    for (uint h = 0u; h < HPG; h++) {
        mo[h] = ml[row0 + h * S].x;
        mn[h] = max(mo[h], subgroupMax(lm[h]));
    }
    float t[HPG];
    for (uint h = 0u; h < HPG; h++) {
        t[h] = 0.0;
    }
    for (uint kq = tid; kq < cap4; kq += KSTEP) {
        for (uint h = 0u; h < HPG; h++) {
            uint i = (row0 + h * S) * cap4 + kq;
            vec4 e = exp(sc4[i] - mn[h]);
            sc4[i] = e;
            t[h] += (e.x + e.y) + (e.z + e.w);
        }
    }
    for (uint h = 0u; h < HPG; h++) {
        float sum = subgroupAdd(t[h]);
        if (tid == h) {
            uint row = row0 + h * S;
            float f = exp(mo[h] - mn[h]);
            ml[row] = vec2(mn[h], ml[row].y * f + sum);
            sc4[fb + row / 4u][row % 4u] = f;
        }
    }
}
