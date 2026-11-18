#version 450
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_integer_dot_product : require
#extension GL_KHR_shader_subgroup_shuffle : require
// Pass 3 of 3 (`hc_norm_dp4a.glsl`): the up matrix (int8 [10240][320], row
// scales in `pw`) on `m` rounded to int8 in groups of 16 (dp4a, exact int32
// sums per group, then its scale; one scale per position lost KL 0.06), and
// the gated mean of the streams: mixed[p][c] = 1/4 sum_s
// sigmoid(u[p][s 2560 + c]) xn[p][s 2560 + c], with xn formed from the f16
// streams, the norm's rsqrt and weight; f16 pairs out.
// A workgroup takes PP positions (their `m` rounded in shared memory), a
// thread one column c (all 4 streams' rows) x the PP positions. Each
// thread reads its rows from memory once per PP positions: PP 4 -> 16 took
// the pass 2.32 -> 1.35 ms at 256 positions (12 is as fast, 20 and 32
// spill).
layout(local_size_x = 256) in;

layout(set = 0, binding = 0, std430) readonly buffer Xin { uint x[]; };
layout(set = 0, binding = 1, std430) readonly buffer W { uvec4 w4[]; };
layout(set = 0, binding = 2, std430) readonly buffer Pw { float pw[]; };
layout(set = 0, binding = 3, std430) readonly buffer Xs { float xs[]; };
layout(set = 0, binding = 4, std430) readonly buffer M { float m[]; };
layout(set = 0, binding = 5, std430) writeonly buffer Out { uint mixed[]; };
layout(set = 0, binding = 8, std140) uniform Params {
    uint n;
    uint wofs;
    uint pofs;
};

#define XS 655360u
#ifndef PP
#define PP 16u
#endif
shared float mv[PP][320];
shared float msc[PP][20];
shared uint mq[PP][80];

#define DOT(q, a) (dotPacked4x8EXT(int(q.x), int(a.x)) + dotPacked4x8EXT(int(q.y), int(a.y)) \
    + dotPacked4x8EXT(int(q.z), int(a.z)) + dotPacked4x8EXT(int(q.w), int(a.w)))

void main() {
    uint t = gl_LocalInvocationID.x;
    uint col = gl_GlobalInvocationID.x;
    uint p0 = gl_WorkGroupID.y * PP;
    for (uint e = t; e < PP * 320u; e += 256u) {
        uint j = e / 320u;
        mv[j][e % 320u] = m[min(p0 + j, n - 1u) * 324u + e % 320u];
    }
    barrier();
    for (uint e = t; e < PP * 20u; e += 256u) {
        uint j = e / 20u, g = e % 20u;
        float mx = 0.0;
        [[unroll]] for (uint i = 0u; i < 16u; i++) { mx = max(mx, abs(mv[j][16u * g + i])); }
        msc[j][g] = mx > 0.0 ? mx / 127.0 : 1.0;
    }
    barrier();
    for (uint e = t; e < PP * 80u; e += 256u) {
        uint j = e / 80u, w = e % 80u;
        uint u = 0u;
        [[unroll]] for (uint b = 0u; b < 4u; b++) {
            int k = clamp(int(roundEven(mv[j][4u * w + b] / msc[j][w / 4u])), -127, 127);
            u |= (uint(k) & 255u) << (8u * b);
        }
        mq[j][w] = u;
    }
    barrier();
    // the up matrix after the 324 x 10240 down rows
    uint ub = wofs + 207360u;
    float acc[4u * PP];
    [[unroll]] for (uint i = 0u; i < 4u * PP; i++) { acc[i] = 0.0; }
    for (uint c = 0u; c < 20u; c++) {
        uvec4 q[4];
        [[unroll]] for (uint s = 0u; s < 4u; s++) { q[s] = w4[ub + (s * 2560u + col) * 20u + c]; }
        [[unroll]] for (uint j = 0u; j < PP; j++) {
            uvec4 a = uvec4(mq[j][4u * c], mq[j][4u * c + 1u], mq[j][4u * c + 2u], mq[j][4u * c + 3u]);
            float sc = msc[j][c];
            [[unroll]] for (uint s = 0u; s < 4u; s++) { acc[j * 4u + s] += float(DOT(q[s], a)) * sc; }
        }
    }
    [[unroll]] for (uint j = 0u; j < PP; j++) {
        uint p = p0 + j, pc = min(p, n - 1u);
        float y = 0.0;
        [[unroll]] for (uint s = 0u; s < 4u; s++) {
            uint jr = s * 2560u + col;
            float u = acc[j * 4u + s] * pw[pofs + 10576u + jr];
            uint xi = pc * 10240u + jr;
            vec2 h = unpackHalf2x16(x[xi / 2u]);
            float xn = ((xi & 1u) == 0u ? h.x : h.y) * xs[XS + 1024u + 4u * pc + s] * pw[pofs + jr];
            y += xn / (1.0 + exp(-u));
        }
        y *= 0.25;
        float o = subgroupShuffleXor(y, 1u);
        if ((col & 1u) == 0u && p < n) {
            mixed[p * 1280u + col / 2u] = packHalf2x16(vec2(y, o));
        }
    }
}
