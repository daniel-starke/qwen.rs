#version 450
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_shuffle : require
// Pass 3 of 3 (`hc_norm_npu.glsl`): the up matrix (int8 [10240][320], row
// scales in `pw`) on `m` and the gated mean of the streams: mixed[p][c] =
// 1/4 sum_s sigmoid(u[p][s 2560 + c]) xn[p][s 2560 + c], f16 pairs.
// A thread takes one column c (all 4 streams' rows) x 4 positions.
layout(local_size_x = 256) in;

layout(set = 0, binding = 1, std430) readonly buffer W { uvec4 w4[]; };
layout(set = 0, binding = 2, std430) readonly buffer Pw { float pw[]; };
layout(set = 0, binding = 3, std430) readonly buffer Xn { float xn[]; };
layout(set = 0, binding = 4, std430) readonly buffer M { vec4 m4[]; };
layout(set = 0, binding = 5, std430) writeonly buffer Out { uint mixed[]; };
layout(set = 0, binding = 8, std140) uniform Params {
    uint n;
    uint wofs;
    uint pofs;
};

vec4 b4(uint u) {
    return vec4(ivec4(int(u << 24u) >> 24, int(u << 16u) >> 24, int(u << 8u) >> 24, int(u) >> 24));
}

void main() {
    uint col = gl_GlobalInvocationID.x;
    uint p0 = gl_WorkGroupID.y * 4u;
    // the up matrix after the 324 x 10240 down rows
    uint ub = wofs + 207360u;
    float acc[16];
    [[unroll]] for (uint i = 0u; i < 16u; i++) { acc[i] = 0.0; }
    uint pp[4];
    [[unroll]] for (uint j = 0u; j < 4u; j++) { pp[j] = min(p0 + j, n - 1u) * 81u; }
    for (uint c = 0u; c < 20u; c++) {
        uvec4 q[4];
        [[unroll]] for (uint s = 0u; s < 4u; s++) { q[s] = w4[ub + (s * 2560u + col) * 20u + c]; }
        [[unroll]] for (uint j = 0u; j < 4u; j++) {
            vec4 a0 = m4[pp[j] + 4u * c];
            vec4 a1 = m4[pp[j] + 4u * c + 1u];
            vec4 a2 = m4[pp[j] + 4u * c + 2u];
            vec4 a3 = m4[pp[j] + 4u * c + 3u];
            [[unroll]] for (uint s = 0u; s < 4u; s++) {
                acc[j * 4u + s] += dot(b4(q[s].x), a0) + dot(b4(q[s].y), a1)
                    + dot(b4(q[s].z), a2) + dot(b4(q[s].w), a3);
            }
        }
    }
    [[unroll]] for (uint j = 0u; j < 4u; j++) {
        uint p = p0 + j;
        float y = 0.0;
        [[unroll]] for (uint s = 0u; s < 4u; s++) {
            uint jr = s * 2560u + col;
            float u = acc[j * 4u + s] * pw[pofs + 10576u + jr];
            y += xn[min(p, n - 1u) * 10240u + jr] / (1.0 + exp(-u));
        }
        y *= 0.25;
        float o = subgroupShuffleXor(y, 1u);
        if ((col & 1u) == 0u && p < n) {
            mixed[p * 1280u + col / 2u] = packHalf2x16(vec2(y, o));
        }
    }
}
