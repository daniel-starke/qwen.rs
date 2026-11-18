#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : require
// The Qwen-Image transformer's row work around its int8 matmuls on a
// Vulkan device (`gpu/ditblk.rs`; the NPU graphs' math,
// `npubuild::qimage::graph_form`, folded form): one row of D = 4096 per
// workgroup, then `rotate(t)` in Hadamard blocks of 256 (normalized) and
// int8 codes on the grid 1 / 128 (`fake_quantize(t, -1, 127 / 128, 256)`).
//
//   MODE 0: t = a * o.pre                              (the out projection's input)
//   MODE 1: x1 = x + o mg1 -> X1; t = mvn(x1) ms2 gu.pre (the MLP's input)
//   MODE 2: x2 = x1 + dn mg2 -> XO (f16); t = mvn(x2) ms1 qkv.pre (the next block's QKV input)
#define D 4096u
#define LS 256u
#define PER (D / LS)
layout(local_size_x = LS) in;

// f16 rows: `a` (MODE 0) or the residual `x` (MODE 1)
layout(set = 0, binding = 0, std430) readonly buffer Xh { uint xh[]; };
// the last matmul's f32 output [row][ldy]
layout(set = 0, binding = 1, std430) readonly buffer Y { float y[]; };
// x1 f32 [row][D]
layout(set = 0, binding = 2, std430) buffer X1 { float x1[]; };
// per-block vectors and the step's modulation, f32
layout(set = 0, binding = 3, std430) readonly buffer V { float vv[]; };
layout(set = 0, binding = 4, std430) writeonly buffer Xq { uint xq[]; };
// the new residual, f16 rows (MODE 2)
layout(set = 0, binding = 5, std430) writeonly buffer Xo { uint xo[]; };
layout(push_constant) uniform Pc {
    uint rin;   // row of Xh / Xo of this workgroup's row 0
    uint ldy;
    uint pre;   // offset of the pre vector in V
    uint mg;    // offset of the gate (MODE 1: mg1, MODE 2: mg2)
    uint ms;    // offset of the LayerNorm's 1 + scale (ms2 / ms1)
};

shared float s[D];
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
    uint w = xh[(row * D + e) / 2u];
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
    float sm = 0.0;
    [[unroll]] for (uint j = 0u; j < PER; j++) {
        uint e = t + LS * j;
#if MODE == 1
        float b = xh_at(rin + r, e);
#else
        float b = x1[r * D + e];
#endif
        v[j] = b + y[r * ldy + e] * vv[mg + e];
#if MODE == 1
        x1[r * D + e] = v[j];
#endif
        sm += v[j];
    }
    float mean = hsum(sm) / float(D);
    float sq = 0.0;
    [[unroll]] for (uint j = 0u; j < PER; j++) { float d = v[j] - mean; sq += d * d; }
    float inv = inversesqrt(hsum(sq) / float(D) + 1e-6);
#if MODE == 2
    // the new residual as f16, pairs from shared memory
    [[unroll]] for (uint j = 0u; j < PER; j++) { s[t + LS * j] = v[j]; }
    barrier();
    [[unroll]] for (uint j = 0u; j < PER / 2u; j++) {
        uint wi = t + LS * j;
        xo[((rin + r) * D) / 2u + wi] = packHalf2x16(vec2(s[2u * wi], s[2u * wi + 1u]));
    }
    barrier();
#endif
    [[unroll]] for (uint j = 0u; j < PER; j++) {
        uint e = t + LS * j;
        v[j] = (v[j] - mean) * inv * vv[ms + e] * vv[pre + e];
    }
#endif
    [[unroll]] for (uint j = 0u; j < PER; j++) { s[t + LS * j] = v[j]; }
    barrier();
    // Hadamard blocks of 256: D / 2 butterflies a stage
    for (uint h = 1u; h < 256u; h *= 2u) {
        [[unroll]] for (uint j = 0u; j < D / 2u / LS; j++) {
            uint p = t + LS * j;
            uint b = p / 128u, q = p % 128u;
            uint i = b * 256u + (q / h) * 2u * h + q % h;
            float a0 = s[i], a1 = s[i + h];
            s[i] = a0 + a1;
            s[i + h] = a0 - a1;
        }
        barrier();
    }
    [[unroll]] for (uint j = 0u; j < D / 4u / LS; j++) {
        uint u = t + LS * j;
        uint w = 0u;
        [[unroll]] for (uint c = 0u; c < 4u; c++) {
            float q = clamp(roundEven(s[4u * u + c] * (128.0 / 16.0)), -128.0, 127.0);
            w |= (uint(int(q)) & 0xFFu) << (8u * c);
        }
        xq[r * (D / 4u) + u] = w;
    }
}
