#version 450
#extension GL_EXT_scalar_block_layout : enable
// residual add: y[i] = a[i] + b[i].
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer A { float a[]; };
layout(set = 0, binding = 1, std430) readonly buffer B { float b[]; };
layout(set = 0, binding = 2, std430) writeonly buffer Y { float y[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };
void main() {
    uint n = p[0];
    uint i = gl_GlobalInvocationID.x;
    if (i >= n) return;
    y[i] = a[i] + b[i];
}
