#version 450
#extension GL_EXT_control_flow_attributes : require
// MiniMax-H3-Image's VAE decoder q / k finish (`gpu/h3vae.rs`; the NPU
// graph `npubuild::h3img::vdec_graph`): per row and head of q and k,
// RMSNorm over the 64 channels (no weight), the rotate-half RoPE over the
// first 2 x 16 channels with the token's `cos` / `sin` (token = row % S),
// q times `qs`. In place. One thread per (row, q / k, head).
#define HD 64u
#define HALF 16u
layout(local_size_x = 64) in;

layout(set = 0, binding = 1, std430) buffer Qkv { float qkv[]; };
// cos [S][HALF] at `cs`, sin [S][HALF] at `sn`
layout(set = 0, binding = 3, std430) readonly buffer V { float vv[]; };
layout(push_constant) uniform Pc {
    uint rows;
    uint S;
    uint D;
    uint cs;
    uint sn;
    float eps;
    float qs;
};

void main() {
    uint i = gl_GlobalInvocationID.x;
    uint heads = D / HD;
    if (i >= rows * 2u * heads) { return; }
    uint r = i / (2u * heads), j = i % (2u * heads);
    uint isk = j / heads, h = j % heads;
    uint b = r * 3u * D + isk * D + h * HD;
    float v[HD];
    float ss = 0.0;
    [[unroll]] for (uint c = 0u; c < HD; c++) { v[c] = qkv[b + c]; ss = fma(v[c], v[c], ss); }
    float inv = inversesqrt(ss / float(HD) + eps) * (isk == 0u ? qs : 1.0);
    uint tk = (r % S) * HALF;
    [[unroll]] for (uint c = 0u; c < HALF; c++) {
        float a = v[c], bb = v[c + HALF];
        float co = vv[cs + tk + c], si = vv[sn + tk + c];
        v[c] = a * co - bb * si;
        v[c + HALF] = bb * co + a * si;
    }
    [[unroll]] for (uint c = 0u; c < HD; c++) { qkv[b + c] = v[c] * inv; }
}
