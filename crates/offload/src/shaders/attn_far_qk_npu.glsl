#version 450
#extension GL_EXT_scalar_block_layout : enable
// Far attention of the NPU engine on the scan device (the iGPU), pass 1 of 2:
// the scores Q K^T of one K block; `attn_far_npu.glsl` is pass 2.
//   q   [4][6 * S][256] f16 (the pre graph's qg), row (g * 6 + h) * S + pos
//   k   [CAP][4][256] f16, one K block; keys [0, nk) count
//   sc  [24 * S][CAP] f32 out: scores, -1e30 for keys >= nk
// One invocation = one position x one KV group x 4 keys at a time, all 6 heads of the
// group: 10 loads per 24 dot products, everything else in registers. No
// shared memory and no barriers: on Xe shared memory is as slow as a buffer
// (the Pascal-style kernel with K / V staged in shared memory ran 45 ms per
// block).
#define HD 256u
#define HPG 6u
// invocations per (position, KV group): each walks every KSTEP-th key quad
#ifndef KSTEP
#define KSTEP 8u
#endif
layout(local_size_x = KSTEP) in;
layout(set = 0, binding = 0, std430) readonly buffer Qb { uvec2 q2[]; };
layout(set = 0, binding = 1, std430) readonly buffer Kb { uvec2 k2[]; };
layout(set = 0, binding = 5, std430) writeonly buffer Sb { vec4 sc4[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };

vec4 half4(uvec2 u) {
    return vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y));
}

void main() {
    uint S = p[0];
    uint nk = p[1];
    uint cap4 = p[2] / 4u;
    uint pos = gl_GlobalInvocationID.y;
    uint g = gl_GlobalInvocationID.z;
    uint q0 = (g * HPG * S + pos) * (HD / 4u);
    for (uint kq = gl_GlobalInvocationID.x; kq < cap4; kq += KSTEP) {
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
            sc4[((g * HPG + h) * S + pos) * cap4 + kq] = mix(vec4(-1e30), sv[h], live);
        }
    }
}
