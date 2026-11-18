#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require
// Q6_K x **f32** GEMV for decode: the float-math counterpart of `q4k_f32.glsl`
// (read that file's header first; the reasoning is the same).
//
// Q6_K block in VRAM (212-byte dword-aligned stride, `Q6K_BLOCK_PAD`):
//   ql[128] (dwords 0..31), qh[64] (32..47), scales i8[16] (48..51), d f16 (52).
// For half n (0/1) and l in 0..31, with ql/qh/sc advanced by 64/32/8 per half:
//   y[l+ 0] = d*sc[l/16+0]*((ql[l]    & 15) | ((qh[l] >> 0) & 3) << 4) - 32)
//   y[l+32] = d*sc[l/16+2]*((ql[l+32] & 15) | ((qh[l] >> 2) & 3) << 4) - 32)
//   y[l+64] = d*sc[l/16+4]*((ql[l]    >> 4) | ((qh[l] >> 4) & 3) << 4) - 32)
//   y[l+96] = d*sc[l/16+6]*((ql[l+32] >> 4) | ((qh[l] >> 6) & 3) << 4) - 32)
//
// The four 6-bit values of a dword are assembled byte-parallel (2 ops per 4
// weights), then each byte is shifted to mantissa bits 17..22 and read as
// f = 1 + q/64. With q = 64*(f-1): sum x*w = d*sc*(64*sum x*f - 96*sum x).
//
// Layout: lane = 8*bi + c8. `bi` (0..3) picks one of 4 consecutive blocks; `c8`
// = 4n + c picks half n and l in [8c, 8c+8): two dwords each of ql[l], ql[l+32]
// and qh[l], i.e. 32 weights in 4 runs of 8, each run inside one 16-element
// scale group (group l/16 = c/2).
//
// Grid: x = ceil(out_dim / (NR * WARPS)), y = seq.

#ifndef NR
#define NR 4
#endif
#ifndef WARPS
#define WARPS 1
#endif

layout(local_size_x = 32, local_size_y = WARPS) in;

layout(set = 0, binding = 0, std430) readonly buffer Weights { uint w[]; };
layout(set = 0, binding = 1, std430) readonly buffer Xin { vec4 xv[]; };
layout(set = 0, binding = 2, std430) writeonly buffer Yout { float yout[]; };
layout(set = 0, binding = 3, std140) uniform Params {
    uint out_dim;
    uint in_dim;
    uint seq;
};

// Byte k of b (a 6-bit value) as 1 + q/64; k is constant after unrolling.
float f6(uint b, uint k) {
    uint m = (8u * k <= 17u) ? (b << (17u - 8u * k)) : (b >> (8u * k - 17u));
    return uintBitsToFloat((m & 0x007E0000u) | 0x3F800000u);
}
vec4 f6x4(uint b) { return vec4(f6(b, 0u), f6(b, 1u), f6(b, 2u), f6(b, 3u)); }

void main() {
    uint lane = gl_LocalInvocationID.x;
    uint warp = gl_LocalInvocationID.y;
    uint o0 = (gl_WorkGroupID.x * WARPS + warp) * NR;
    uint si = gl_WorkGroupID.y;
    if (o0 >= out_dim || si >= seq) { return; }

    uint nb = in_dim / 256u;
    uint bi = lane >> 3u;
    uint c8 = lane & 7u;
    uint n = c8 >> 2u;
    uint c = c8 & 3u;
    uint grp = c >> 1u;                 // l/16 for l in [8c, 8c+8)

    uint ql0 = n * 16u + 2u * c;        // ql[l]    dwords (and +1)
    uint ql1 = ql0 + 8u;                // ql[l+32] dwords
    uint qh0 = 32u + n * 8u + 2u * c;   // qh[l]    dwords
    uint scd = 48u + 2u * n;            // sc[n*8 + 0..3], [n*8 + 4..7] in scd+1
    int sb0 = int(grp * 8u);            // byte of sc[n*8 + grp] in dword scd
    int sb2 = sb0 + 16;                 // byte of sc[n*8 + grp + 2]

    uint xbase = si * (in_dim / 4u) + n * 32u + 2u * c;

    float acc[NR];
    for (uint r = 0u; r < NR; r++) { acc[r] = 0.0; }

    for (uint ib = bi; ib < nb; ib += 4u) {
        uint xo = xbase + ib * 64u;
        vec4 x00 = xv[xo],       x01 = xv[xo + 1u];     // l .. l+7
        vec4 x10 = xv[xo + 8u],  x11 = xv[xo + 9u];     // l+32
        vec4 x20 = xv[xo + 16u], x21 = xv[xo + 17u];    // l+64
        vec4 x30 = xv[xo + 24u], x31 = xv[xo + 25u];    // l+96
        vec4 one = vec4(1.0);
        float s0 = dot(x00 + x01, one), s1 = dot(x10 + x11, one);
        float s2 = dot(x20 + x21, one), s3 = dot(x30 + x31, one);

        for (uint r = 0u; r < NR; r++) {
            uint o = o0 + r;
            if (o >= out_dim) { break; }
            uint wb = (o * nb + ib) * 53u;
            uint la0 = w[wb + ql0], la1 = w[wb + ql0 + 1u];
            uint lb0 = w[wb + ql1], lb1 = w[wb + ql1 + 1u];
            uint h0 = w[wb + qh0], h1 = w[wb + qh0 + 1u];
            uint sw0 = w[wb + scd], sw1 = w[wb + scd + 1u];
            float d = unpackHalf2x16(w[wb + 52u]).x;

            float sc0 = float(bitfieldExtract(int(sw0), sb0, 8));
            float sc1 = float(bitfieldExtract(int(sw0), sb2, 8));
            float sc2 = float(bitfieldExtract(int(sw1), sb0, 8));
            float sc3 = float(bitfieldExtract(int(sw1), sb2, 8));

            const uint M4 = 0x0F0F0F0Fu, M2 = 0x30303030u;
            float f0 = dot(x00, f6x4((la0 & M4) | ((h0 << 4u) & M2)))
                     + dot(x01, f6x4((la1 & M4) | ((h1 << 4u) & M2)));
            float f1 = dot(x10, f6x4((lb0 & M4) | ((h0 << 2u) & M2)))
                     + dot(x11, f6x4((lb1 & M4) | ((h1 << 2u) & M2)));
            float f2 = dot(x20, f6x4(((la0 >> 4u) & M4) | (h0 & M2)))
                     + dot(x21, f6x4(((la1 >> 4u) & M4) | (h1 & M2)));
            float f3 = dot(x30, f6x4(((lb0 >> 4u) & M4) | ((h0 >> 2u) & M2)))
                     + dot(x31, f6x4(((lb1 >> 4u) & M4) | ((h1 >> 2u) & M2)));

            acc[r] += d * (sc0 * (64.0 * f0 - 96.0 * s0) + sc1 * (64.0 * f1 - 96.0 * s1)
                         + sc2 * (64.0 * f2 - 96.0 * s2) + sc3 * (64.0 * f3 - 96.0 * s3));
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
