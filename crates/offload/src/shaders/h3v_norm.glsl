#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : require
// MiniMax-H3-Image's VAE decoder norms (`gpu/h3vae.rs`), one row of D per
// workgroup:
//   MODE 0: h = x / sqrt(mean(x^2) + eps)            (the layers' RMSNorm; weights folded)
//   MODE 1: h = (x - mean) / sqrt(var + eps) g + b   (the head's LayerNorm, latent rows only:
//           out row o = tile 256 + i reads row tile S + i)
#ifndef D
#define D 1024u
#endif
#define LS 256u
#define PER (D / LS)
layout(local_size_x = LS) in;

layout(set = 0, binding = 1, std430) readonly buffer X { float x[]; };
layout(set = 0, binding = 2, std430) writeonly buffer H { float hh[]; };
// the head's g, b (f16 pairs)
layout(set = 0, binding = 4, std430) readonly buffer G { uint g[]; };
layout(set = 0, binding = 5, std430) readonly buffer Bb { uint bb[]; };
layout(push_constant) uniform Pc {
    float eps;
    uint S;
};

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

float f16at(uint e, bool isg) {
    uint w = isg ? g[e / 2u] : bb[e / 2u];
    return unpackHalf2x16(w)[e & 1u];
}

void main() {
    uint t = gl_LocalInvocationID.x, o = gl_WorkGroupID.x;
#if MODE == 0
    uint r = o;
#else
    uint r = (o / 256u) * S + o % 256u;
#endif
    float v[PER];
    [[unroll]] for (uint j = 0u; j < PER; j++) { v[j] = x[r * D + t + LS * j]; }
#if MODE == 0
    float sq = 0.0;
    [[unroll]] for (uint j = 0u; j < PER; j++) { sq = fma(v[j], v[j], sq); }
    float inv = inversesqrt(hsum(sq) / float(D) + eps);
    [[unroll]] for (uint j = 0u; j < PER; j++) { hh[o * D + t + LS * j] = v[j] * inv; }
#else
    float sm = 0.0;
    [[unroll]] for (uint j = 0u; j < PER; j++) { sm += v[j]; }
    float mean = hsum(sm) / float(D);
    float sq = 0.0;
    [[unroll]] for (uint j = 0u; j < PER; j++) { float d = v[j] - mean; sq = fma(d, d, sq); }
    float inv = inversesqrt(hsum(sq) / float(D) + eps);
    [[unroll]] for (uint j = 0u; j < PER; j++) {
        uint e = t + LS * j;
        hh[o * D + e] = (v[j] - mean) * inv * f16at(e, true) + f16at(e, false);
    }
#endif
}
