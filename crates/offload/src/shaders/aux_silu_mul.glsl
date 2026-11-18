#version 450
#extension GL_EXT_scalar_block_layout : enable
// silu(fg) * up  ->  y   (FFN gate activation). y[i] = silu(fg[i]) * up[i].
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer Up { float up[]; };
layout(set = 0, binding = 1, std430) readonly buffer Fg { float fg[]; };
layout(set = 0, binding = 2, std430) writeonly buffer Y { float y[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };
void main() {
    uint n = p[0];
    uint i = gl_GlobalInvocationID.x;
    if (i >= n) return;
    float f = fg[i];
    float s = f / (1.0 + exp(-f));
    y[i] = s * up[i];
}
