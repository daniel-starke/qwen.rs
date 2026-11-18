#version 450
#extension GL_EXT_control_flow_attributes : require
// Pass 2 of 3 (`hc_norm_npu.glsl`): the down matrix with the injection
// rows folded in (`dinj`, int8 [324][10240], row scales in `pw`) on `xn`
// -> `m` [n][324] f32: rows 0..320 SiLU(t / 4), rows 320..324 the
// injection weights 2 sigmoid(t / 4).
// A workgroup takes 32 rows x 16 positions; its 8 subgroups of 32 threads
// split k (each thread 4 rows x 4 positions over 1280 inputs), summed in
// shared memory.
layout(local_size_x = 256) in;

layout(set = 0, binding = 1, std430) readonly buffer W { uvec4 w4[]; };
layout(set = 0, binding = 2, std430) readonly buffer Pw { float pw[]; };
layout(set = 0, binding = 3, std430) readonly buffer Xn { vec4 xn4[]; };
layout(set = 0, binding = 4, std430) writeonly buffer M { float m[]; };
layout(set = 0, binding = 8, std140) uniform Params {
    uint n;
    uint wofs;
    uint pofs;
};

#define R 324u
shared float red[8][512];

vec4 b4(uint u) {
    return vec4(ivec4(int(u << 24u) >> 24, int(u << 16u) >> 24, int(u << 8u) >> 24, int(u) >> 24));
}

void main() {
    uint t = gl_LocalInvocationID.x;
    uint ks = t / 32u, tile = t % 32u;
    uint rg = tile % 8u, pg = tile / 8u;
    uint r0 = gl_WorkGroupID.x * 32u + rg * 4u;
    uint p0 = gl_WorkGroupID.y * 16u + pg * 4u;
    float acc[16];
    [[unroll]] for (uint i = 0u; i < 16u; i++) { acc[i] = 0.0; }
    uint pp[4];
    [[unroll]] for (uint j = 0u; j < 4u; j++) { pp[j] = min(p0 + j, n - 1u) * 2560u; }
    uint rr[4];
    [[unroll]] for (uint i = 0u; i < 4u; i++) { rr[i] = wofs + min(r0 + i, R - 1u) * 640u; }
    for (uint c = ks * 80u; c < ks * 80u + 80u; c++) {
        uvec4 q[4];
        [[unroll]] for (uint i = 0u; i < 4u; i++) { q[i] = w4[rr[i] + c]; }
        [[unroll]] for (uint j = 0u; j < 4u; j++) {
            vec4 a0 = xn4[pp[j] + 4u * c];
            vec4 a1 = xn4[pp[j] + 4u * c + 1u];
            vec4 a2 = xn4[pp[j] + 4u * c + 2u];
            vec4 a3 = xn4[pp[j] + 4u * c + 3u];
            [[unroll]] for (uint i = 0u; i < 4u; i++) {
                acc[j * 4u + i] += dot(b4(q[i].x), a0) + dot(b4(q[i].y), a1)
                    + dot(b4(q[i].z), a2) + dot(b4(q[i].w), a3);
            }
        }
    }
    [[unroll]] for (uint i = 0u; i < 16u; i++) { red[ks][tile * 16u + i] = acc[i]; }
    barrier();
    for (uint e = t; e < 512u; e += 256u) {
        float a = 0.0;
        [[unroll]] for (uint g = 0u; g < 8u; g++) { a += red[g][e]; }
        uint tl = e / 16u, j = (e % 16u) / 4u, i = e % 4u;
        uint r = gl_WorkGroupID.x * 32u + (tl % 8u) * 4u + i;
        uint p = gl_WorkGroupID.y * 16u + (tl / 8u) * 4u + j;
        if (r < R && p < n) {
            float v = a * pw[pofs + 10240u + r] * 0.25;
            m[p * R + r] = r < 320u ? v / (1.0 + exp(-v)) : 2.0 / (1.0 + exp(-v));
        }
    }
}
