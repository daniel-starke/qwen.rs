#version 450
#extension GL_EXT_scalar_block_layout : enable
// Attention of the NPU engine's vision tower on the scan device (the iGPU),
// pass 4 of 4: O = P V of a tile of queries, written as the f16 rows the
// NPU reads.
//   v   [H][nq][18] vec4 (`vis_rope_npu.glsl`; zero for keys >= n)
//   o   [n][H][72] f16 out
//   mx  [H][tq][16] f32: [8, 16) the parts of a row's sum (`vis_sm_npu.glsl`)
//   sc  [H][tq][nq / 4] vec4: exp(score - max)
// One invocation = one head x 8 queries x 2 dim-quads, 4 keys at a time (16
// loads per 256 multiply-adds), everything else in registers; no shared
// memory. Neighboring invocations are neighboring dim-quad pairs.
#define HD4 18u
#define DP 9u
layout(local_size_x = 8) in;
layout(set = 0, binding = 2, std430) readonly buffer Vb { vec4 v4[]; };
layout(set = 0, binding = 3, std430) writeonly buffer Ob { uvec2 o2[]; };
layout(set = 0, binding = 4, std430) readonly buffer Mb { float mx[]; };
layout(set = 0, binding = 5, std430) readonly buffer Sb { vec4 sc4[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };

void main() {
    uint n = p[0];
    uint heads = p[1];
    uint nq = p[2];
    uint n4 = nq / 4u;
    uint t0 = p[3];
    uint tq = p[4];
    uint d4 = (gl_GlobalInvocationID.x % DP) * 2u;
    uint qy = (gl_GlobalInvocationID.x / DP) * 8u;
    if (qy >= tq) {
        return;
    }
    uint h = gl_GlobalInvocationID.z;
    uint v0 = h * nq * HD4 + d4;
    uint row = h * tq + qy;
    uint pbase = row * n4;
    vec4 aa[8];
    vec4 ab[8];
    for (uint i = 0u; i < 8u; i++) {
        aa[i] = vec4(0.0);
        ab[i] = vec4(0.0);
    }
    for (uint kq = 0u; kq < n4; kq++) {
        uint vi = v0 + kq * 4u * HD4;
        vec4 a0 = v4[vi];
        vec4 b0 = v4[vi + 1u];
        vec4 a1 = v4[vi + HD4];
        vec4 b1 = v4[vi + HD4 + 1u];
        vec4 a2 = v4[vi + 2u * HD4];
        vec4 b2 = v4[vi + 2u * HD4 + 1u];
        vec4 a3 = v4[vi + 3u * HD4];
        vec4 b3 = v4[vi + 3u * HD4 + 1u];
        for (uint i = 0u; i < 8u; i++) {
            vec4 pr = sc4[pbase + i * n4 + kq];
            aa[i] += pr.x * a0 + pr.y * a1 + pr.z * a2 + pr.w * a3;
            ab[i] += pr.x * b0 + pr.y * b1 + pr.z * b2 + pr.w * b3;
        }
    }
    for (uint i = 0u; i < 8u; i++) {
        uint t = t0 + qy + i;
        if (t < n) {
            float s = 0.0;
            for (uint j = 8u; j < 16u; j++) {
                s += mx[(row + i) * 16u + j];
            }
            s = 1.0 / s;
            vec4 r = aa[i] * s;
            uint o = (t * heads + h) * HD4 + d4;
            o2[o] = uvec2(packHalf2x16(r.xy), packHalf2x16(r.zw));
            r = ab[i] * s;
            o2[o + 1u] = uvec2(packHalf2x16(r.xy), packHalf2x16(r.zw));
        }
    }
}
