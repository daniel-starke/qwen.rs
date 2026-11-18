#version 450
#extension GL_EXT_scalar_block_layout : enable
// RMSNorm apply pass: y[i] = x[i] * inv[0] * w[i].
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer Xin { float x[]; };
layout(set = 0, binding = 1, std430) readonly buffer Win { float w[]; };
layout(set = 0, binding = 2, std430) writeonly buffer Yout { float y[]; };
layout(set = 0, binding = 3, std430) readonly buffer Inv { float inv[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };
void main() {
    uint n = p[0];
    uint i = gl_GlobalInvocationID.x;
    if (i >= n) return;
    y[i] = x[i] * inv[0] * w[i];
}
