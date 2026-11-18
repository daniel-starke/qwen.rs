#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_integer_dot_product : require
// Q4_K x q8_K GEMV with packed 4x8 integer dots (DP4A, `OpSDotKHR` via
// VK_KHR_shader_integer_dot_product). Pascal has native DP4A (sm_61).
//
// History: the first version of this kernel gave each thread a whole output
// row (the layout `q4k.glsl` abandoned at ~6 GB/s) and read the **high**
// nibble for even sub-blocks, so its init self-check failed with
// maxdiff ~6.19e6, which was misread as defective Pascal silicon. With the
// nibble order fixed it was correct but 2.7 GB/s on FfnUp.
//
// This revision uses `q4k_f32.glsl`'s layout (lane = 8*bi + g: 4 blocks per
// warp step, 32 weights per lane, NR rows per warp sharing the activation
// loads and the per-sub-block activation sums), with the dot done as four
// `dotPacked4x8EXT` per nibble half. Measured on the GTX 1060
// (`gpubench --gemv-probe`), kernel only: FfnUp 155-163 GB/s against
// `q4k_f32`'s 108-130, i.e. at the ~160 GB/s read ceiling.
//
// Numerics: the integer dots are exact; error vs an exact product is the
// q8_K activation's (max ~4.6 on outputs ~950), the same as the CPU
// `gemv_q4k` that computes the other rows of a decode split.
//
// Activation: q8_K packed `[qs int8 (in_dim) | ds f32 (nb)]` per sequence
// row (`pack_x` / `aux_quantize_q8k`), read as uints so the row stride
// `in_dim + 4*nb` need only be dword aligned.
//
// Grid: x = ceil(out_dim / NR), y = seq; subgroup size must be 32.

#ifndef NR
#define NR 4
#endif

layout(local_size_x = 32) in;

layout(set = 0, binding = 0, std430) readonly buffer Weights { uvec4 w4[]; };
layout(set = 0, binding = 1, std430) readonly buffer Xin { uint xin[]; };
layout(set = 0, binding = 2, std430) writeonly buffer Yout { float yout[]; };
layout(set = 0, binding = 3, std140) uniform Params {
    uint out_dim;
    uint in_dim;
    uint seq;
};

void main() {
    uint lane = gl_LocalInvocationID.x;
    uint o0 = gl_WorkGroupID.x * NR;
    uint si = gl_WorkGroupID.y;
    if (o0 >= out_dim || si >= seq) { return; }

    uint nb = in_dim / 256u;
    uint bi = lane >> 3u;
    uint g = lane & 7u;
    uint p = g >> 1u;
    uint t = g & 1u;
    uint slo = 2u * p;
    bool lo_form = p < 2u;
    int jb = int((slo & 3u) * 8u);

    uint sx = in_dim + nb * 4u;
    uint qs_dw = si * sx / 4u;
    uint ds_dw = qs_dw + in_dim / 4u;
    uint xlo = qs_dw + slo * 8u + t * 4u;

    float acc[NR];
    for (uint r = 0u; r < NR; r++) { acc[r] = 0.0; }

    for (uint ib = bi; ib < nb; ib += 4u) {
        uint xo = xlo + ib * 64u;
        int l0 = int(xin[xo]), l1 = int(xin[xo + 1u]), l2 = int(xin[xo + 2u]), l3 = int(xin[xo + 3u]);
        int h0 = int(xin[xo + 8u]), h1 = int(xin[xo + 9u]), h2 = int(xin[xo + 10u]), h3 = int(xin[xo + 11u]);
        float xd = uintBitsToFloat(xin[ds_dw + ib]);
        const int ONES = 0x01010101;
        int sl = dotPacked4x8EXT(ONES, l0) + dotPacked4x8EXT(ONES, l1)
               + dotPacked4x8EXT(ONES, l2) + dotPacked4x8EXT(ONES, l3);
        int sh = dotPacked4x8EXT(ONES, h0) + dotPacked4x8EXT(ONES, h1)
               + dotPacked4x8EXT(ONES, h2) + dotPacked4x8EXT(ONES, h3);

        for (uint r = 0u; r < NR; r++) {
            uint o = o0 + r;
            if (o >= out_dim) { break; }
            uint wb = (o * nb + ib) * 9u;
            uvec4 hd = w4[wb];
            uvec4 qv = w4[wb + 1u + g];

            vec2 dm = unpackHalf2x16(hd.x);
            uint scv = lo_form ? (hd.y & 0x3F3F3F3Fu)
                               : ((hd.w & 0x0F0F0F0Fu) | ((hd.y >> 2u) & 0x30303030u));
            uint mv = lo_form ? (hd.z & 0x3F3F3F3Fu)
                              : (((hd.w >> 4u) & 0x0F0F0F0Fu) | ((hd.z >> 2u) & 0x30303030u));
            int sc0 = int(bitfieldExtract(scv, jb, 8));
            int sc1 = int(bitfieldExtract(scv, jb + 8, 8));
            int mm0 = int(bitfieldExtract(mv, jb, 8));
            int mm1 = int(bitfieldExtract(mv, jb + 8, 8));

            const uint LO = 0x0F0F0F0Fu;
            int dl = dotPacked4x8EXT(int(qv.x & LO), l0) + dotPacked4x8EXT(int(qv.y & LO), l1)
                   + dotPacked4x8EXT(int(qv.z & LO), l2) + dotPacked4x8EXT(int(qv.w & LO), l3);
            int dh = dotPacked4x8EXT(int((qv.x >> 4u) & LO), h0) + dotPacked4x8EXT(int((qv.y >> 4u) & LO), h1)
                   + dotPacked4x8EXT(int((qv.z >> 4u) & LO), h2) + dotPacked4x8EXT(int((qv.w >> 4u) & LO), h3);

            acc[r] += xd * (dm.x * float(sc0 * dl + sc1 * dh) - dm.y * float(mm0 * sl + mm1 * sh));
        }
    }

    for (uint r = 0u; r < NR; r++) {
        float s = subgroupAdd(acc[r]);
        uint o = o0 + r;
        if (lane == 0u && o < out_dim) {
            yout[si * out_dim + o] = s;
        }
    }
}
