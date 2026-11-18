#version 450
#extension GL_EXT_control_flow_attributes : require
// Between the two expert GEMMs (`exp_gemm.glsl`): per slot row of `G`
// (gate 640 then up 640, f32) `h = silu(g) u d.pre`, the Hadamard / sqrt(128)
// on blocks of 128, rounded to int8 and packed per 8 elements as (even,
// odd) words for the down GEMM -> `Hq` [row][160]. `npu::flash::cpuexp`'s
// math. Grid: x = 64 (token in the slot), y = slots.
layout(local_size_x = 128) in;

layout(set = 0, binding = 0, std430) readonly buffer Dp { uint dp[]; };
layout(set = 0, binding = 1, std430) readonly buffer Sl { uint sl[]; };
layout(set = 0, binding = 2, std430) readonly buffer G { float gb[]; };
layout(set = 0, binding = 3, std430) writeonly buffer Hq { uint hq[]; };
layout(set = 0, binding = 8, std140) uniform Params {
    uint wofs;
    uint pofs;   // d.pre's offset in f16 pairs (even)
    uint sbase;  // the dispatch's first slot
};

#define FE 640u
shared float v[FE];

int q8(float x) {
    return int(clamp(sign(x) * floor(abs(x) + 0.5), -128.0, 127.0));
}

void main() {
    uint t = gl_LocalInvocationID.x;
    uint s = sbase + gl_WorkGroupID.y, tk = gl_WorkGroupID.x;
    uint e = sl[4u * s], nt = sl[4u * s + 1u], orow = sl[4u * s + 3u];
    if (tk >= nt) { return; }
    uint row = orow + tk;
    [[unroll]] for (uint b = 0u; b < 5u; b++) {
        uint i = b * 128u + t;
        float g = gb[row * 2u * FE + i], u = gb[row * 2u * FE + FE + i];
        uint pi = pofs / 2u + (e * FE + i) / 2u;
        float p = unpackHalf2x16(dp[pi])[i & 1u];
        v[i] = g / (1.0 + exp(-g)) * u * p;
    }
    barrier();
    // butterflies of stride h within each block of 128: thread t takes
    // pair t % 64 of blocks t / 64, t / 64 + 2, + 4 (< 5)
    for (uint h = 1u; h < 128u; h *= 2u) {
        uint pr = t % 64u;
        uint j = (pr / h) * 2u * h + pr % h;
        for (uint b = t / 64u; b < 5u; b += 2u) {
            uint a = b * 128u + j;
            float x = v[a], y = v[a + h];
            v[a] = x + y;
            v[a + h] = x - y;
        }
        barrier();
    }
    if (t < 80u) {
        const float r = 0.08838834764831845;
        int q[8];
        [[unroll]] for (uint k = 0u; k < 8u; k++) { q[k] = q8(v[t * 8u + k] * r); }
        hq[row * 160u + 2u * t] = (uint(q[0]) & 0xFFu) | ((uint(q[2]) & 0xFFu) << 8u)
            | ((uint(q[4]) & 0xFFu) << 16u) | (uint(q[6]) << 24u);
        hq[row * 160u + 2u * t + 1u] = (uint(q[1]) & 0xFFu) | ((uint(q[3]) & 0xFFu) << 8u)
            | ((uint(q[5]) & 0xFFu) << 16u) | (uint(q[7]) << 24u);
    }
}
