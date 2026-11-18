#version 450
#extension GL_EXT_scalar_block_layout : enable
// RMSNorm reduction pass: writes inv = 1/sqrt(ss/n + eps) into a 1-float buffer.
//
// One workgroup of 256 invocations: each strides over the row, then a shared
// tree reduction. Every `barrier()` sits at workgroup scope (never inside the
// divergent `if`), which is what made the earlier reduction fragile across ICDs.
// Dispatch with (1,1,1).
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer Xin { float x[]; };
layout(set = 0, binding = 1, std430) writeonly buffer Yout { float y[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };
shared float sh[256];
void main() {
    uint n = p[0];
    float eps = uintBitsToFloat(p[1]);
    uint t = gl_LocalInvocationID.x;
    float s = 0.0;
    for (uint i = t; i < n; i += 256u) { float v = x[i]; s += v * v; }
    sh[t] = s;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (t < stride) { sh[t] += sh[t + stride]; }
        barrier();
    }
    if (t == 0u) { y[0] = 1.0 / sqrt(sh[0] / float(n) + eps); }
}
