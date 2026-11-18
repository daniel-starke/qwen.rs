#version 450
#extension GL_EXT_scalar_block_layout : enable
// L2 normalize `ngroups` contiguous blocks of `S` elements each, reading at
// element offset `in_off` and writing at element offset `out_off`:
//   y[out_off + g*S + d] = x[in_off + g*S + d] / max(sqrt(ss), eps)
// Used for SSM q/k per-group normalization, where q and k are two slices of one
// packed conv output but land in separate (offset-0) buffers.
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer Xin { float x[]; };
layout(set = 0, binding = 1, std430) writeonly buffer Yout { float y[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };
void main() {
    uint S = p[0];
    uint ng = p[1];
    uint in_off = p[2];
    uint eps = p[3];
    uint out_off = p[4];
    uint g = gl_GlobalInvocationID.x;
    if (g >= ng) return;
    uint ibase = in_off + g * S;
    uint obase = out_off + g * S;
    float ss = 0.0;
    for (uint d = 0u; d < S; d++) { float v = x[ibase + d]; ss += v * v; }
    float inv = 1.0 / max(sqrt(ss), uintBitsToFloat(eps));
    for (uint d = 0u; d < S; d++) { y[obase + d] = x[ibase + d] * inv; }
}
