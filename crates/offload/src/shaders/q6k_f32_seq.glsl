#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require
// Q6_K x f32 GEMV over **up to NS positions per weight read**: the Q6_K
// counterpart of `q4k_f32_seq.glsl` (read that header for why, including why
// the weights are dequantized to float once), with the lane layout of
// `q6k_f32.glsl`. FfnDown seq 8: 2073 us (conversion per position: 2185 us).
//
// Grid: x = ceil(out_dim / NR), y = ceil(seq / NS).

#ifndef NR
#define NR 4
#endif
#ifndef NS
#define NS 8
#endif

layout(local_size_x = 32) in;

layout(set = 0, binding = 0, std430) readonly buffer Weights { uint w[]; };
layout(set = 0, binding = 1, std430) readonly buffer Xin { vec4 xv[]; };
layout(set = 0, binding = 2, std430) writeonly buffer Yout { float yout[]; };
layout(set = 0, binding = 3, std140) uniform Params {
    uint out_dim;
    uint in_dim;
    uint seq;
};

float f6(uint b, uint k) {
    uint m = (8u * k <= 17u) ? (b << (17u - 8u * k)) : (b >> (8u * k - 17u));
    return uintBitsToFloat((m & 0x007E0000u) | 0x3F800000u);
}
vec4 f6x4(uint b) { return vec4(f6(b, 0u), f6(b, 1u), f6(b, 2u), f6(b, 3u)); }

void main() {
    uint lane = gl_LocalInvocationID.x;
    uint o0 = gl_WorkGroupID.x * NR;
    uint p0 = gl_WorkGroupID.y * NS;
    if (o0 >= out_dim || p0 >= seq) { return; }
    uint ns = min(seq - p0, uint(NS));

    uint nb = in_dim / 256u;
    uint bi = lane >> 3u;
    uint c8 = lane & 7u;
    uint n = c8 >> 2u;
    uint c = c8 & 3u;
    uint grp = c >> 1u;

    uint ql0 = n * 16u + 2u * c;
    uint ql1 = ql0 + 8u;
    uint qh0 = 32u + n * 8u + 2u * c;
    uint scd = 48u + 2u * n;
    int sb0 = int(grp * 8u);
    int sb2 = sb0 + 16;

    uint xrow = in_dim / 4u;
    uint xbase = p0 * xrow + n * 32u + 2u * c;

    float acc[NR][NS];
    for (uint r = 0u; r < NR; r++) {
        for (uint s = 0u; s < NS; s++) { acc[r][s] = 0.0; }
    }

    const uint M4 = 0x0F0F0F0Fu, M2 = 0x30303030u;
    for (uint ib = bi; ib < nb; ib += 4u) {
        // The NR rows' 6-bit values (byte-parallel, 8 dwords = 32 weights) and
        // scales for this block: loaded and assembled once for every position.
        // This lane's 32 weights of each row, dequantized once: d*sc*(q - 32).
        vec4 wq[NR][8];
        for (uint r = 0u; r < NR; r++) {
            uint o = min(o0 + r, out_dim - 1u);
            uint wb = (o * nb + ib) * 53u;
            uint la0 = w[wb + ql0], la1 = w[wb + ql0 + 1u];
            uint lb0 = w[wb + ql1], lb1 = w[wb + ql1 + 1u];
            uint h0 = w[wb + qh0], h1 = w[wb + qh0 + 1u];
            uint sw0 = w[wb + scd], sw1 = w[wb + scd + 1u];
            float d = unpackHalf2x16(w[wb + 52u]).x;
            vec4 sc = d * vec4(float(bitfieldExtract(int(sw0), sb0, 8)),
                               float(bitfieldExtract(int(sw0), sb2, 8)),
                               float(bitfieldExtract(int(sw1), sb0, 8)),
                               float(bitfieldExtract(int(sw1), sb2, 8)));
            uint q[8];
            q[0] = (la0 & M4) | ((h0 << 4u) & M2);
            q[1] = (la1 & M4) | ((h1 << 4u) & M2);
            q[2] = (lb0 & M4) | ((h0 << 2u) & M2);
            q[3] = (lb1 & M4) | ((h1 << 2u) & M2);
            q[4] = ((la0 >> 4u) & M4) | (h0 & M2);
            q[5] = ((la1 >> 4u) & M4) | (h1 & M2);
            q[6] = ((lb0 >> 4u) & M4) | ((h0 >> 2u) & M2);
            q[7] = ((lb1 >> 4u) & M4) | ((h1 >> 2u) & M2);
            for (uint j = 0u; j < 8u; j++) {
                uint b = q[j];
                wq[r][j] = sc[j >> 1u]
                    * (vec4(uvec4(b, b >> 8u, b >> 16u, b >> 24u) & 0xFFu) - 32.0);
            }
        }
        for (uint s = 0u; s < NS; s++) {
            if (s >= ns) { break; }
            uint xo = xbase + s * xrow + ib * 64u;
            vec4 x00 = xv[xo],       x01 = xv[xo + 1u];
            vec4 x10 = xv[xo + 8u],  x11 = xv[xo + 9u];
            vec4 x20 = xv[xo + 16u], x21 = xv[xo + 17u];
            vec4 x30 = xv[xo + 24u], x31 = xv[xo + 25u];
            for (uint r = 0u; r < NR; r++) {
                acc[r][s] += dot(wq[r][0], x00) + dot(wq[r][1], x01)
                           + dot(wq[r][2], x10) + dot(wq[r][3], x11)
                           + dot(wq[r][4], x20) + dot(wq[r][5], x21)
                           + dot(wq[r][6], x30) + dot(wq[r][7], x31);
            }
        }
    }

    for (uint r = 0u; r < NR; r++) {
        uint o = o0 + r;
        for (uint s = 0u; s < NS; s++) {
            if (s >= ns) { break; }
            float v = subgroupAdd(acc[r][s]);
            if (lane == 0u && o < out_dim) {
                yout[(p0 + s) * out_dim + o] = v;
            }
        }
    }
}
