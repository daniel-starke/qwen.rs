#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_integer_dot_product : require
// int4 (per-row scale) x int8 GEMV for the NPU engine's decode MLP share on
// a discrete GPU (`gpu/mlpsplit.rs`), up to MAXS positions per pass over
// the weights.
//
// A weight byte holds elements 2k (low nibble) and 2k + 1 (high), each
// biased by 8 (n = w + 8). The activation is int8 with one scale per block
// of 256 elements (`inv`), stored per 32 elements as 16 even bytes then 16
// odd bytes, so a 16-byte weight load masks into two uvec4 whose bytes pair
// with the activation's: 8 `dp4a` per 32 weights and position. `corr`
// holds -8 inv sum(x) per block (the nibble bias), up to 128 blocks.
//
// A first form with f32 activations (nibbles in the mantissa, as
// `q4k_f32.glsl`) spent its time loading activations: 4 positions x 32
// floats per 32 weights and 2 rows (4 rows of floats do not fit the
// registers). This one loads 2 uvec4 per position for NR rows.
//
// Grid: x = out_dim / NR.

#define NR 8
#define MAXS 4

layout(local_size_x = 32) in;

layout(set = 0, binding = 0, std430) readonly buffer Weights { uvec4 w4[]; };
layout(set = 0, binding = 1, std430) readonly buffer Xin { uvec4 xq[]; };
layout(set = 0, binding = 2, std430) writeonly buffer Yout { float yout[]; };
layout(set = 0, binding = 3, std430) readonly buffer Scale { float rs[]; };
layout(set = 0, binding = 4, std430) readonly buffer Inv { float inv[]; };
layout(set = 0, binding = 5, std430) readonly buffer Corr { float corr[]; };
layout(set = 0, binding = 8, std140) uniform Params {
    uint out_dim;
    uint in_dim;
    uint seq;
    uint wq;        // a weight row's stride in uvec4 (>= in_dim / 32)
};

void main() {
    uint lane = gl_LocalInvocationID.x;
    uint o0 = gl_WorkGroupID.x * NR;
    uint nq = in_dim / 32u;
    uint nblk = in_dim / 256u;

    float acc[NR][MAXS];
    for (uint r = 0u; r < NR; r++) {
        for (uint s = 0u; s < MAXS; s++) { acc[r][s] = 0.0; }
    }
    for (uint iq = lane; iq < nq; iq += 32u) {
        ivec4 lo[NR], hi[NR];
        for (uint r = 0u; r < NR; r++) {
            uvec4 q = w4[(o0 + r) * wq + iq];
            lo[r] = ivec4(q & uvec4(0x0F0F0F0Fu));
            hi[r] = ivec4((q >> 4u) & uvec4(0x0F0F0F0Fu));
        }
        uint b = iq >> 3u;
        for (uint s = 0u; s < seq; s++) {
            ivec4 xe = ivec4(xq[(s * nq + iq) * 2u]);
            ivec4 xo = ivec4(xq[(s * nq + iq) * 2u + 1u]);
            float iv = inv[s * nblk + b];
            for (uint r = 0u; r < NR; r++) {
                int d = dotPacked4x8EXT(lo[r].x, xe.x) + dotPacked4x8EXT(hi[r].x, xo.x)
                      + dotPacked4x8EXT(lo[r].y, xe.y) + dotPacked4x8EXT(hi[r].y, xo.y)
                      + dotPacked4x8EXT(lo[r].z, xe.z) + dotPacked4x8EXT(hi[r].z, xo.z)
                      + dotPacked4x8EXT(lo[r].w, xe.w) + dotPacked4x8EXT(hi[r].w, xo.w);
                acc[r][s] += float(d) * iv;
            }
        }
    }
    for (uint s = 0u; s < seq; s++) {
        // only this call's blocks: entries past them are stale
        float cl = 0.0;
        for (uint k = lane; k < min(nblk, 128u); k += 32u) { cl += corr[s * 128u + k]; }
        float c = subgroupAdd(cl);
        for (uint r = 0u; r < NR; r++) {
            float v = subgroupAdd(acc[r][s]) + c;
            if (lane == 0u) {
                yout[s * out_dim + o0 + r] = v * rs[o0 + r];
            }
        }
    }
}
