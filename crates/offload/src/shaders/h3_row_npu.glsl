#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : require
// MiniMax-H3-Image's transformer: the row work around its int8 matmuls on a
// Vulkan device (`gpu/h3blk.rs`; the NPU graphs' math,
// `npubuild::h3img::dit_graph`): one row of W per workgroup, then
// `rotate(t)` in Hadamard blocks of 256 (normalized) and int8 codes on the
// grid 1 / 128 (`fake_quantize(t, -1, 127 / 128, 256)`). The residual is
// the stored `x_l / c_l` (f16); `o.s` and `d.s` (the gates and `1 / c_l`)
// are applied by the matmuls.
//
//   MODE 0 (W = inner): t = a o.pre                                   (the out projection's input)
//   MODE 1 (W = D):     x1 = x xr + o -> X1; t = (rms(x1) ms2 + sh2) gu.pre (the MLP's input)
//   MODE 2 (W = D):     xo = x1 + dn -> XO (f16); t = (rms(xo) ms1 + sh1) qkv.pre (the next block's QKV input)
#ifndef W
#define W 5376u
#endif
#define LS 256u
#define PER (W / LS)
#define EPS 6.2e-5
layout(local_size_x = LS) in;

// f16 rows: `a` (MODE 0) or the residual `x` (MODE 1)
layout(set = 0, binding = 0, std430) readonly buffer Xh { uint xh[]; };
// the last matmul's f32 output [row][W]
layout(set = 0, binding = 1, std430) readonly buffer Y { float y[]; };
// x1 f32 [row][W]
layout(set = 0, binding = 2, std430) buffer X1 { float x1[]; };
// per-block vectors and the modulation entries, f32
layout(set = 0, binding = 3, std430) readonly buffer V { float vv[]; };
layout(set = 0, binding = 4, std430) writeonly buffer Xq { uint xq[]; };
// the new residual, f16 rows (MODE 2)
layout(set = 0, binding = 5, std430) writeonly buffer Xo { uint xo[]; };
layout(push_constant) uniform Pc {
    uint rin;   // row of Xh / Xo of this workgroup's row 0
    uint pre;   // offset of the pre vector in V
    uint xr;    // offset of xr (MODE 1)
    uint ms;    // offset of the RMSNorm's modulation (ms2 / ms1)
    uint sh;    // offset of the shift (sh2 / sh1)
};

shared float s[W];
shared float red[LS / 8u];

float hsum(float v) {
    v = subgroupAdd(v);
    if (gl_SubgroupInvocationID == 0u) { red[gl_SubgroupID] = v; }
    barrier();
    float t = 0.0;
    for (uint i = 0u; i < gl_NumSubgroups; i++) { t += red[i]; }
    barrier();
    return t;
}

float xh_at(uint row, uint e) {
    uint w = xh[(row * W + e) / 2u];
    return unpackHalf2x16(w)[e & 1u];
}

void main() {
    uint t = gl_LocalInvocationID.x, r = gl_WorkGroupID.x;
    float v[PER];
#if MODE == 0
    [[unroll]] for (uint j = 0u; j < PER; j++) {
        uint e = t + LS * j;
        v[j] = xh_at(rin + r, e) * vv[pre + e];
    }
#else
    float sq = 0.0;
    [[unroll]] for (uint j = 0u; j < PER; j++) {
        uint e = t + LS * j;
#if MODE == 1
        v[j] = xh_at(rin + r, e) * vv[xr + e] + y[r * W + e];
        x1[r * W + e] = v[j];
#else
        v[j] = x1[r * W + e] + y[r * W + e];
#endif
        sq += v[j] * v[j];
    }
    float inv = inversesqrt(hsum(sq) / float(W) + EPS);
#if MODE == 2
    // the new residual as f16, pairs from shared memory
    [[unroll]] for (uint j = 0u; j < PER; j++) { s[t + LS * j] = v[j]; }
    barrier();
    for (uint wi = t; wi < W / 2u; wi += LS) {
        xo[((rin + r) * W) / 2u + wi] = packHalf2x16(vec2(s[2u * wi], s[2u * wi + 1u]));
    }
    barrier();
#endif
    [[unroll]] for (uint j = 0u; j < PER; j++) {
        uint e = t + LS * j;
        v[j] = (v[j] * inv * vv[ms + e] + vv[sh + e]) * vv[pre + e];
    }
#endif
    [[unroll]] for (uint j = 0u; j < PER; j++) { s[t + LS * j] = v[j]; }
    barrier();
    // Hadamard blocks of 256: W / 2 butterflies a stage
    for (uint h = 1u; h < 256u; h *= 2u) {
        for (uint p = t; p < W / 2u; p += LS) {
            uint b = p / 128u, q = p % 128u;
            uint i = b * 256u + (q / h) * 2u * h + q % h;
            float a0 = s[i], a1 = s[i + h];
            s[i] = a0 + a1;
            s[i + h] = a0 - a1;
        }
        barrier();
    }
    for (uint u = t; u < W / 4u; u += LS) {
        uint w = 0u;
        [[unroll]] for (uint c = 0u; c < 4u; c++) {
            float q = clamp(roundEven(s[4u * u + c] * (128.0 / 16.0)), -128.0, 127.0);
            w |= (uint(int(q)) & 0xFFu) << (8u * c);
        }
        xq[r * (W / 4u) + u] = w;
    }
}
