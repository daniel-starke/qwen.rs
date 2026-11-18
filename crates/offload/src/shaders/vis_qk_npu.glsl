#version 450
#extension GL_EXT_scalar_block_layout : enable
// Attention of the NPU engine's vision tower on the scan device (the iGPU),
// pass 2 of 4: the scores Q K^T of one tile of queries against every key.
//   q   [H][nq][18] vec4 (rotated, scaled), k the same (`vis_rope_npu.glsl`)
//   mx  [H][tq][16] f32 out: [0, 8) the largest score of a row among the
//       keys of each invocation ([8, 16): `vis_sm_npu.glsl`)
//   sc  [H][tq][nq / 4] vec4 out: the tile's scores, -1e30 for keys >= n
// One invocation = one head x 8 queries x 4 keys at a time (12 loads per 32
// dot products of 4), walking every KSTEP-th key quad. The device is bound
// by its loads, as long as everything else stays in registers. Per block
// at 4096 patches: 4 queries x 4 keys 71 ms, 8 x 4 54 ms, 8 x 8 81 ms; a
// kernel that kept no scores (scores, exponentials and P V of a query in
// one sweep: Q, K and V loaded for every 4 keys) 580 ms against 155 ms for
// the three passes. No shared memory: on Xe it is as slow as a buffer
// (`attn_far_qk_npu.glsl`).
#define HD4 18u
#define KSTEP 8u
#define QN 8u
layout(local_size_x = KSTEP) in;
layout(set = 0, binding = 0, std430) readonly buffer Qb { vec4 q4[]; };
layout(set = 0, binding = 1, std430) readonly buffer Kb { vec4 k4[]; };
layout(set = 0, binding = 4, std430) writeonly buffer Mb { float mx[]; };
layout(set = 0, binding = 5, std430) writeonly buffer Sb { vec4 sc4[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };

void main() {
    uint n = p[0];
    uint nq = p[2];
    uint n4 = nq / 4u;
    uint t0 = p[3];
    uint tq = p[4];
    uint qy = gl_GlobalInvocationID.y * QN;
    uint h = gl_GlobalInvocationID.z;
    uint q0 = (h * nq + t0 + qy) * HD4;
    uint s0 = (h * tq + qy) * n4;
    float rm[QN];
    for (uint i = 0u; i < QN; i++) {
        rm[i] = -1e30;
    }
    for (uint kq = gl_GlobalInvocationID.x; kq < n4; kq += KSTEP) {
        uint k0 = (h * nq + kq * 4u) * HD4;
        vec4 sv[QN];
        for (uint i = 0u; i < QN; i++) {
            sv[i] = vec4(0.0);
        }
        for (uint dq = 0u; dq < HD4; dq++) {
            vec4 ka = k4[k0 + dq];
            vec4 kc = k4[k0 + HD4 + dq];
            vec4 ke = k4[k0 + 2u * HD4 + dq];
            vec4 kg = k4[k0 + 3u * HD4 + dq];
            for (uint i = 0u; i < QN; i++) {
                vec4 qv = q4[q0 + i * HD4 + dq];
                sv[i] += vec4(dot(qv, ka), dot(qv, kc), dot(qv, ke), dot(qv, kg));
            }
        }
        bvec4 live = lessThan(uvec4(kq * 4u) + uvec4(0u, 1u, 2u, 3u), uvec4(n));
        for (uint i = 0u; i < QN; i++) {
            vec4 a = mix(vec4(-1e30), sv[i], live);
            sc4[s0 + i * n4 + kq] = a;
            rm[i] = max(rm[i], max(max(a.x, a.y), max(a.z, a.w)));
        }
    }
    for (uint i = 0u; i < QN; i++) {
        mx[(h * tq + qy + i) * 16u + gl_GlobalInvocationID.x] = rm[i];
    }
}
