#version 450
#extension GL_EXT_scalar_block_layout : enable
// Flash-Next prefill: a delta layer's gated norm after the recurrence
// (`delta_prefill_npu.glsl`, same submission), in place of its `o`, so the
// NPU's `post_delta_t` starts at the out-projection's quantized input:
//   y = o / sqrt(mean_head(o^2) + 1e-6 oscale^2) * w0 * sigmoid(z * w1)
// per (position, head of 128), then rotated in Hadamard blocks of 256
// (Sylvester order, / 16: `graphs::rotate`).
//   o  [N][6144] f16 in/out (the scan's output, x oscale)
//   z  [N][6144] f16, the gate input as the pre graph leaves it (no row scale)
//   w  [2][6144] f32: gnw * out.pre per channel, then z's row scale
// One group per (block of 256 channels = 2 heads, position); a thread per
// channel. f16 tensors are read as packed uints.
#define C 6144u
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer Ww { float w[]; };
layout(set = 0, binding = 1, std430) buffer Oo { uint oo[]; };
layout(set = 0, binding = 2, std430) readonly buffer Zz { uint zz[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };

#define LD(buf, e) (unpackHalf2x16(buf[(e) >> 1u])[(e) & 1u])

shared float sh[256];
shared float ss[256];

void main() {
    float oscale = uintBitsToFloat(p[1]);
    uint i = gl_LocalInvocationID.x;
    uint c = gl_WorkGroupID.x * 256u + i;
    uint e = gl_WorkGroupID.y * C + c;
    float o = LD(oo, e);
    float z = LD(zz, e);
    ss[i] = o * o;
    barrier();
    // sums of squares per head (threads 0..127, 128..255)
    for (uint h = 64u; h > 0u; h >>= 1u) {
        if ((i & 127u) < h) ss[i] += ss[i + h];
        barrier();
    }
    float inv = inversesqrt(ss[i & 128u] / 128.0 + 1e-6 * oscale * oscale);
    float g = 1.0 / (1.0 + exp(-z * w[C + c]));
    float x = o * inv * w[c] * g;
    for (uint h = 1u; h < 256u; h <<= 1u) {
        sh[i] = x;
        barrier();
        float y = sh[i ^ h];
        x = (i & h) != 0u ? y - x : x + y;
        barrier();
    }
    sh[i] = x * 0.0625;
    barrier();
    if (i < 128u) {
        uint b = (gl_WorkGroupID.y * C + gl_WorkGroupID.x * 256u) >> 1u;
        oo[b + i] = packHalf2x16(vec2(sh[2u * i], sh[2u * i + 1u]));
    }
}
