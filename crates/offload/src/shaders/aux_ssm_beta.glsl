#version 450
#extension GL_EXT_scalar_block_layout : enable
// SSM beta: beta_s[h] = sigmoid(beta_raw[h]); written into the low half of the
// combined beta+gate buffer `bg` (length 2*Hv) consumed by the delta-net shader.
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer Bin { float inb[]; };
layout(set = 0, binding = 1, std430) writeonly buffer Bg { float bg[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };
void main() {
    uint Hv = p[0];
    uint h = gl_GlobalInvocationID.x;
    if (h >= Hv) return;
    float b = inb[h];
    bg[h] = 1.0 / (1.0 + exp(-b));
}
