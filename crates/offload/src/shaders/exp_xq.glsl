#version 450
// Before the gate / up GEMM (`exp_gemm.glsl -DGU`): the f16 MoE input rows
// (`moe_q`, prescaled to the int8 grid) rounded to int8 and packed per 8
// elements as (even, odd) words -> `Xq` [token][640]. Once per token, not
// per GEMM workgroup. Grid: x = tokens * 320 / 64.
layout(local_size_x = 64) in;

layout(set = 0, binding = 0, std430) readonly buffer X { uvec4 x4[]; };
layout(set = 0, binding = 1, std430) writeonly buffer Xq { uint xq[]; };
layout(set = 0, binding = 8, std140) uniform Params {
    uint n;      // tokens * 320 (groups of 8)
};

int q8(float v) {
    return int(clamp(sign(v) * floor(abs(v) + 0.5), -128.0, 127.0));
}

uint pk(int a, int b, int c, int d) {
    return (uint(a) & 0xFFu) | ((uint(b) & 0xFFu) << 8u) | ((uint(c) & 0xFFu) << 16u)
        | (uint(d) << 24u);
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= n) { return; }
    uvec4 u = x4[i];
    vec2 a = unpackHalf2x16(u.x), b = unpackHalf2x16(u.y);
    vec2 c = unpackHalf2x16(u.z), d = unpackHalf2x16(u.w);
    xq[2u * i] = pk(q8(a.x), q8(b.x), q8(c.x), q8(d.x));
    xq[2u * i + 1u] = pk(q8(a.y), q8(b.y), q8(c.y), q8(d.y));
}
