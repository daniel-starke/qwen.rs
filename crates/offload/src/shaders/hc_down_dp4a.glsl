#version 450
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_integer_dot_product : require
// Pass 2 of 3 (`hc_norm_dp4a.glsl`): the down matrix with the injection
// rows folded in (`dinj`, int8 [324][10240], row scales in `pw`) on the
// int8 `xq` (dp4a, exact int32 sums per group of 16, then its scale) -> `m`
// [n][324] f32: rows 0..320 SiLU(t / 4), rows 320..324 the injection
// weights 2 sigmoid(t / 4).
// A workgroup takes 32 rows x 4 PT positions; its 8 subgroups of 32
// threads split k (each thread 4 rows x PT positions over 1280 inputs, half
// of one stream), summed in shared memory. PT 4 -> 8 took the pass 1.77 ->
// 1.34 ms at 256 positions (6 and 12 are slower).
layout(local_size_x = 256) in;

layout(set = 0, binding = 1, std430) readonly buffer W { uvec4 w4[]; };
layout(set = 0, binding = 2, std430) readonly buffer Pw { float pw[]; };
layout(set = 0, binding = 3, std430) readonly buffer Xq { uvec4 xq4[]; };
layout(set = 0, binding = 3, std430) readonly buffer Xs { float xs[]; };
layout(set = 0, binding = 4, std430) writeonly buffer M { float m[]; };
layout(set = 0, binding = 8, std140) uniform Params {
    uint n;
    uint wofs;
    uint pofs;
};

#define R 324u
#define XS 655360u
#define XG (XS + 2048u)
#ifndef PT
#define PT 8u
#endif
#define PW (4u * PT)
shared float red[8][32u * PW];

#define DOT(q, a) (dotPacked4x8EXT(int(q.x), int(a.x)) + dotPacked4x8EXT(int(q.y), int(a.y)) \
    + dotPacked4x8EXT(int(q.z), int(a.z)) + dotPacked4x8EXT(int(q.w), int(a.w)))

void main() {
    uint t = gl_LocalInvocationID.x;
    uint ks = t / 32u, tile = t % 32u;
    uint rg = tile % 8u, pg = tile / 8u;
    uint r0 = gl_WorkGroupID.x * 32u + rg * 4u;
    uint p0 = gl_WorkGroupID.y * PW + pg * PT;
    float acc[4u * PT];
    [[unroll]] for (uint i = 0u; i < 4u * PT; i++) { acc[i] = 0.0; }
    uint pp[PT];
    [[unroll]] for (uint j = 0u; j < PT; j++) { pp[j] = min(p0 + j, n - 1u) * 640u; }
    uint rr[4];
    [[unroll]] for (uint i = 0u; i < 4u; i++) { rr[i] = wofs + min(r0 + i, R - 1u) * 640u; }
    for (uint c = ks * 80u; c < ks * 80u + 80u; c++) {
        uvec4 q[4];
        [[unroll]] for (uint i = 0u; i < 4u; i++) { q[i] = w4[rr[i] + c]; }
        [[unroll]] for (uint j = 0u; j < PT; j++) {
            uvec4 a = xq4[pp[j] + c];
            float sc = xs[XG + pp[j] + c];
            [[unroll]] for (uint i = 0u; i < 4u; i++) { acc[j * 4u + i] += float(DOT(q[i], a)) * sc; }
        }
    }
    [[unroll]] for (uint i = 0u; i < 4u * PT; i++) { red[ks][tile * 4u * PT + i] = acc[i]; }
    barrier();
    for (uint e = t; e < 32u * PW; e += 256u) {
        float a = 0.0;
        [[unroll]] for (uint g = 0u; g < 8u; g++) { a += red[g][e]; }
        uint tl = e / (4u * PT), j = (e % (4u * PT)) / 4u, i = e % 4u;
        uint r = gl_WorkGroupID.x * 32u + (tl % 8u) * 4u + i;
        uint p = gl_WorkGroupID.y * PW + (tl / 8u) * PT + j;
        if (r < R && p < n) {
            float v = a * pw[pofs + 10240u + r] * 0.25;
            m[p * R + r] = r < 320u ? v / (1.0 + exp(-v)) : 2.0 / (1.0 + exp(-v));
        }
    }
}
