#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require
// Q4_K x **f32** GEMV for decode: float math, no q8_K activation.
//
// Why a second Q4_K GEMV: `q4k.glsl` does the dot in 32-bit integers against a
// q8_K activation. On Pascal a 32-bit integer multiply is emulated (XMAD x3-4)
// and each lane decoded a block's scales to use them on only 8 weights, so the
// kernel was ALU-bound at ~50 GB/s on a 192 GB/s card. This one:
//
//   * reads the activation as f32 (more accurate than q8_K, and no host pack);
//   * turns nibbles into floats by placing them in the mantissa of 1.0,
//     never an I2F, which is quarter rate on Pascal. One shift puts a byte at
//     bits 15..22, so its high nibble reads as 1 + n/16 and its low nibble as
//     1 + n/256 (`fhi` / `flo`): SHF/2 + LOP3 + FFMA per weight;
//   * corrects the `1 +` bias once per sub-block segment with the segment's
//     activation sum, which does not depend on the row and is shared by the
//     `NR` rows a warp computes;
//   * gives each lane 32 weights of one block (a 16-byte `uvec4`), so the scale
//     decode is amortized over 32 weights instead of 8;
//   * reduces with `subgroupAdd` (the warp owns its rows; no shared memory).
//
// Math, per sub-block s with scale sc and min m (w = d*sc*n - dmin*m):
//   sum x*w = d*sc*sum x*n - dmin*m*sum x,  and with f = 1 + n/K
//   (K = 256 for low nibbles, 16 for high), sum x*n = K*(sum x*f - sum x), so
//   sum x*w = K*d*sc * sum x*f  -  (K*d*sc + dmin*m) * sum x.
//
// Layout: lane = 8*bi + g. `bi` (0..3) is which of 4 consecutive blocks the
// lane takes per step; `g` (0..7) is its 16-byte slice of the block's 128 qs
// bytes, i.e. chunk p = g/2 (sub-blocks 2p, 2p+1) and half t = g%2 (positions
// 16t..16t+15 of both sub-blocks). A warp's 32 qs loads are 4 x 128 contiguous
// bytes. The header (d, dmin, 12 scale bytes) is the block's first `uvec4`.
//
// Grid: x = ceil(out_dim / (NR * WARPS)), y = seq.

#ifndef NR
#define NR 4
#endif
#ifndef WARPS
#define WARPS 1
#endif

layout(local_size_x = 32, local_size_y = WARPS) in;

layout(set = 0, binding = 0, std430) readonly buffer Weights { uvec4 w4[]; };
layout(set = 0, binding = 1, std430) readonly buffer Xin { vec4 xv[]; };
layout(set = 0, binding = 2, std430) writeonly buffer Yout { float yout[]; };
layout(set = 0, binding = 3, std140) uniform Params {
    uint out_dim;
    uint in_dim;
    uint seq;
};

// Byte `k` of `q` shifted so its two nibbles sit at mantissa bits 15..22:
// the high nibble at 19..22 reads as 1 + n/16, the low one at 15..18 as
// 1 + n/256. One shift serves two weights; `k` is constant after unrolling.
uint at15(uint q, uint k) {
    return (k < 2u) ? (q << (15u - 8u * k)) : (q >> (8u * k - 15u));
}
float fhi(uint t) { return uintBitsToFloat((t & 0x00780000u) | 0x3F800000u); }
float flo(uint t) { return uintBitsToFloat((t & 0x00078000u) | 0x3F800000u); }

// Dot of one qs dword's 4 low nibbles with `xl` and 4 high nibbles with `xh`.
void dot8(uint q, vec4 xl, vec4 xh, inout float fl, inout float fh) {
    uint t0 = at15(q, 0u), t1 = at15(q, 1u), t2 = at15(q, 2u), t3 = at15(q, 3u);
    fl += dot(xl, vec4(flo(t0), flo(t1), flo(t2), flo(t3)));
    fh += dot(xh, vec4(fhi(t0), fhi(t1), fhi(t2), fhi(t3)));
}

void main() {
    uint lane = gl_LocalInvocationID.x;
    uint warp = gl_LocalInvocationID.y;
    uint o0 = (gl_WorkGroupID.x * WARPS + warp) * NR;
    uint si = gl_WorkGroupID.y;
    if (o0 >= out_dim || si >= seq) { return; }   // whole warp exits together

    uint nb = in_dim / 256u;
    uint bi = lane >> 3u;
    uint g = lane & 7u;
    uint p = g >> 1u;
    uint t = g & 1u;
    uint slo = 2u * p;
    uint shi = slo + 1u;

    // `get_scale_min_k4` has two forms (sub-block < 4 and >= 4); a lane's two
    // sub-blocks 2p, 2p+1 always share one, so the form is fixed per lane.
    // Both forms are decoded for all four sub-blocks at once as packed bytes
    // and the lane's pair is picked by byte offset `jb`.
    bool lo_form = p < 2u;
    int jb = int((slo & 3u) * 8u);

    // vec4 index of this lane's low-sub-block activation slice in block 0.
    uint xbase = si * (in_dim / 4u) + slo * 8u + t * 4u;

    float acc[NR];
    for (uint r = 0u; r < NR; r++) { acc[r] = 0.0; }

    for (uint ib = bi; ib < nb; ib += 4u) {
        uint xo = xbase + ib * 64u;
        vec4 xl0 = xv[xo], xl1 = xv[xo + 1u], xl2 = xv[xo + 2u], xl3 = xv[xo + 3u];
        vec4 xh0 = xv[xo + 8u], xh1 = xv[xo + 9u], xh2 = xv[xo + 10u], xh3 = xv[xo + 11u];
        vec4 one = vec4(1.0);
        float sxl = dot(xl0 + xl1 + xl2 + xl3, one);
        float sxh = dot(xh0 + xh1 + xh2 + xh3, one);

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
            float sc0 = float(bitfieldExtract(scv, jb, 8));
            float sc1 = float(bitfieldExtract(scv, jb + 8, 8));
            float mm0 = float(bitfieldExtract(mv, jb, 8));
            float mm1 = float(bitfieldExtract(mv, jb + 8, 8));

            // Low nibbles -> sub-block slo (1 + n/256), high -> shi (1 + n/16).
            float fl = 0.0, fh = 0.0;
            dot8(qv.x, xl0, xh0, fl, fh);
            dot8(qv.y, xl1, xh1, fl, fh);
            dot8(qv.z, xl2, xh2, fl, fh);
            dot8(qv.w, xl3, xh3, fl, fh);

            float dl = 256.0 * dm.x * sc0;
            float dh = 16.0 * dm.x * sc1;
            acc[r] += dl * fl - (dl + dm.y * mm0) * sxl
                    + dh * fh - (dh + dm.y * mm1) * sxh;
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
