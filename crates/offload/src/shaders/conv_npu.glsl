#version 450
#extension GL_EXT_scalar_block_layout : enable
// Flash-Next prefill: a delta layer's short conv, Swish and the l2 norms of
// q / k, ahead of the recurrence (`delta_prefill_npu.glsl`, same
// submission), from the NPU pre graph's raw qkv projection
// (`graphs::delta_pre_c`):
//   y[t][c] = sum_k w[c][k] * x[t + k][c],   x = [cpast; qkv * qs]
//   s = y * sigmoid(y);  q, k: s / sqrt(sum_head(s^2) + 1e-6), v: s
//   w     [10240][4] f32: the conv taps, then qs [10240] f32 (qkv's row scale)
//   cpast [3][10240] f16: the last three positions before the chunk (scaled)
//   qkv   [N][10240] f16: the raw MatMul output (no row scale)
//   qn, kn [N][2048], v [N][6144] f16 out
// One group per (head row of 128 channels: 16 q, 16 k, 48 v, position); a
// thread per channel pair. f16 tensors are read as packed uints.
#define C 10240u
#define QK 2048u
#define HV 6144u
layout(local_size_x = 64) in;
layout(set = 0, binding = 0, std430) readonly buffer Ww { float w[]; };
layout(set = 0, binding = 1, std430) readonly buffer Cp { uint cp[]; };
layout(set = 0, binding = 2, std430) readonly buffer Xx { uint xx[]; };
layout(set = 0, binding = 3, std430) writeonly buffer Qq { uint qq[]; };
layout(set = 0, binding = 4, std430) writeonly buffer Kk { uint kk[]; };
layout(set = 0, binding = 5, std430) writeonly buffer Vv { uint vv[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };

shared float ss[64];

// row r of x = [cpast; qkv * qs], channels c, c + 1
vec2 row(int r, uint c, vec2 qs) {
    if (r < 3) {
        return unpackHalf2x16(cp[(uint(r) * C + c) >> 1u]);
    }
    return unpackHalf2x16(xx[(uint(r - 3) * C + c) >> 1u]) * qs;
}

void main() {
    uint i = gl_LocalInvocationID.x;
    uint hr = gl_WorkGroupID.x;
    uint t = gl_WorkGroupID.y;
    uint c = hr * 128u + 2u * i;
    vec2 qs = vec2(w[4u * C + c], w[4u * C + c + 1u]);
    vec4 w0 = vec4(w[4u * c], w[4u * c + 1u], w[4u * c + 2u], w[4u * c + 3u]);
    vec4 w1 = vec4(w[4u * c + 4u], w[4u * c + 5u], w[4u * c + 6u], w[4u * c + 7u]);
    vec2 y = vec2(0.0);
    for (int k = 0; k < 4; k++) {
        vec2 x = row(int(t) + k, c, qs);
        y += vec2(w0[k], w1[k]) * x;
    }
    vec2 s = y / (1.0 + exp(-y));
    if (hr >= 32u) {
        vv[(t * HV + c - 2u * QK) >> 1u] = packHalf2x16(s);
        return;
    }
    ss[i] = dot(s, s);
    barrier();
    for (uint h = 32u; h > 0u; h >>= 1u) {
        if (i < h) ss[i] += ss[i + h];
        barrier();
    }
    s *= inversesqrt(ss[0] + 1e-6);
    if (hr < 16u) {
        qq[(t * QK + c) >> 1u] = packHalf2x16(s);
    } else {
        kk[(t * QK + c - QK) >> 1u] = packHalf2x16(s);
    }
}
