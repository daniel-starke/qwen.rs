#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require
// Q4_K x f32 GEMV over **up to NS positions per weight read**: the
// speculative-decoding verify batch (`seq` 2..8).
//
// `q4k_f32.glsl` maps the position to `gl_WorkGroupID.y`, so a batch of `seq`
// positions reads every resident weight `seq` times; the dispatch walks x
// fastest, so a row is long gone from the 1.5 MB L2 when the next position
// wants it. At ~130 GB/s that is ~38 ms per position for the decode split's
// ~5 GB of resident rows, which is what made each extra verify row cost
// 16-43 ms and capped speculation at ~4 rows.
//
// Here a warp dequantizes each weight to float **once** (d*sc*q - dmin*m)
// and applies it to every position with one FMA per weight: the weight
// traffic is that of one GEMV, and the activation (seq x in_dim f32,
// <= 160 KB) stays in L2.
//
// One Q4_K block per warp step: lane l holds the 8 weights of
// qs bytes [4l, 4l+4) (4 low nibbles of sub-block 2(l/8), 4 high of the
// next) for each of NR rows, and reads 2 vec4 of x per position. The first
// version spread a warp over 4 blocks with 32 weights per lane (`q4k_f32`'s
// layout): 128 weight registers per lane at NR = 4, low occupancy. This one
// holds 32, and is 10-31 % faster at every seq 2-8 (FfnUp kernel-only,
// `gpubench --gemv-probe --xf32 --rows 4`: seq 5 1700 -> 1160 us, seq 8
// 2100 -> 1640 us; Wq / Wo seq 5 ~495 us). NR = 2 / 3 are slower, and NR
// >= 6 spill (acc[NR][NS] dominates the registers then).
//
// Not `q4k_f32.glsl`'s `1 + n/256` float trick: re-running that conversion
// per position left an early version of this kernel ALU-bound, and its
// rounding is 14x the error of a plain dequantize (gpubench --gemv-probe
// maxabsdiff 0.0030 -> 0.0002 at seq 8).
//
// Grid: x = ceil(out_dim / NR), y = ceil(seq / NS).

#ifndef NR
#define NR 4
#endif
#ifndef NS
#define NS 8
#endif
layout(local_size_x = 32) in;
layout(set = 0, binding = 0, std430) readonly buffer Weights { uvec4 w4[]; };
layout(set = 0, binding = 1, std430) readonly buffer Xin { vec4 xv[]; };
layout(set = 0, binding = 2, std430) writeonly buffer Yout { float yout[]; };
layout(set = 0, binding = 3, std140) uniform Params {
    uint out_dim;
    uint in_dim;
    uint seq;
};

void main() {
    uint lane = gl_LocalInvocationID.x;
    uint o0 = gl_WorkGroupID.x * NR;
    uint s0 = gl_WorkGroupID.y * NS;
    if (o0 >= out_dim || s0 >= seq) { return; }
    uint ns = min(seq - s0, uint(NS));
    uint nb = in_dim / 256u;
    uint p = lane >> 3u;
    uint e = lane & 7u;
    uint slo = 2u * p;
    bool lo_form = p < 2u;
    int jb = int((slo & 3u) * 8u);
    uint xrow = in_dim / 4u;
    uint xb = s0 * xrow + 16u * p + e;
    uint qi = lane >> 2u;
    uint qc = lane & 3u;

    float acc[NR][NS];
    for (uint r = 0u; r < NR; r++) {
        for (uint s = 0u; s < NS; s++) { acc[r][s] = 0.0; }
    }
    for (uint ib = 0u; ib < nb; ib++) {
        vec4 wl[NR];
        vec4 wh[NR];
        for (uint r = 0u; r < NR; r++) {
            uint o = min(o0 + r, out_dim - 1u);
            uint wb = (o * nb + ib) * 9u;
            uvec4 hd = w4[wb];
            uint qq = w4[wb + 1u + qi][qc];
            vec2 dm = unpackHalf2x16(hd.x);
            uint scv = lo_form ? (hd.y & 0x3F3F3F3Fu)
                               : ((hd.w & 0x0F0F0F0Fu) | ((hd.y >> 2u) & 0x30303030u));
            uint mv = lo_form ? (hd.z & 0x3F3F3F3Fu)
                              : (((hd.w >> 4u) & 0x0F0F0F0Fu) | ((hd.z >> 2u) & 0x30303030u));
            float dl = dm.x * float(bitfieldExtract(scv, jb, 8));
            float dh = dm.x * float(bitfieldExtract(scv, jb + 8, 8));
            float ml = dm.y * float(bitfieldExtract(mv, jb, 8));
            float mh = dm.y * float(bitfieldExtract(mv, jb + 8, 8));
            vec4 lo = vec4(uvec4(qq, qq >> 8u, qq >> 16u, qq >> 24u) & 0x0Fu);
            vec4 hi = vec4(uvec4(qq >> 4u, qq >> 12u, qq >> 20u, qq >> 28u) & 0x0Fu);
            wl[r] = dl * lo - ml;
            wh[r] = dh * hi - mh;
        }
        uint xo = xb + ib * 64u;
        for (uint s = 0u; s < NS; s++) {
            if (s >= ns) { break; }
            vec4 xl = xv[xo + s * xrow];
            vec4 xh = xv[xo + s * xrow + 8u];
            for (uint r = 0u; r < NR; r++) {
                acc[r][s] += dot(wl[r], xl) + dot(wh[r], xh);
            }
        }
    }
    for (uint r = 0u; r < NR; r++) {
        uint o = o0 + r;
        for (uint s = 0u; s < NS; s++) {
            if (s >= ns) { break; }
            float v = subgroupAdd(acc[r][s]);
            if (lane == 0u && o < out_dim) {
                yout[(s0 + s) * out_dim + o] = v;
            }
        }
    }
}
