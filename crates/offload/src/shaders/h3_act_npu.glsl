#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : require
// MiniMax-H3-Image's MLP activation on a Vulkan device (`gpu/h3blk.rs`;
// `npubuild::h3img::dit_graph`): one row of the gate / up matmul's output
// per workgroup, `hid = swish(g sg) (u / 256)` (the up rows carry the down
// rotation's signs; the 1 / 256 keeps the NPU's f16 in range and is undone
// by `d.s`), `rotate(hid)` in Hadamard blocks of 256, int8 per row at
// `127 / 128 / max |row|` (codes on the grid 1 / 128; the `rowq` kernel's
// floor 1e-3 on the normalized maximum), the row's factor
// `max |row| 128 / 127` -> V.
#ifndef FF
#define FF 14336u
#endif
#define LS 256u
#define PER (FF / LS)
layout(local_size_x = LS) in;

layout(set = 0, binding = 1, std430) readonly buffer Y { float y[]; };
layout(set = 0, binding = 3, std430) buffer V { float vv[]; };
layout(set = 0, binding = 4, std430) writeonly buffer Xq { uint xq[]; };
layout(push_constant) uniform Pc {
    uint sg;    // offset of gu.sg in V
    uint rowf;  // offset of the rows' factors in V
};

shared float s[FF];
shared float red[LS / 8u];

void main() {
    uint t = gl_LocalInvocationID.x, r = gl_WorkGroupID.x;
    [[unroll]] for (uint j = 0u; j < PER; j++) {
        uint e = t + LS * j;
        float g = y[r * 2u * FF + e] * vv[sg + e];
        s[e] = g / (1.0 + exp(-g)) * (y[r * 2u * FF + FF + e] * (1.0 / 256.0));
    }
    barrier();
    for (uint h = 1u; h < 256u; h *= 2u) {
        [[unroll]] for (uint j = 0u; j < FF / 2u / LS; j++) {
            uint p = t + LS * j;
            uint b = p / 128u, q = p % 128u;
            uint i = b * 256u + (q / h) * 2u * h + q % h;
            float a0 = s[i], a1 = s[i + h];
            s[i] = a0 + a1;
            s[i + h] = a0 - a1;
        }
        barrier();
    }
    float m = 0.0;
    [[unroll]] for (uint j = 0u; j < PER; j++) { m = max(m, abs(s[t + LS * j])); }
    m = subgroupMax(m);
    if (gl_SubgroupInvocationID == 0u) { red[gl_SubgroupID] = m; }
    barrier();
    m = 0.0;
    for (uint i = 0u; i < gl_NumSubgroups; i++) { m = max(m, red[i]); }
    // the normalized rotation's 1 / 16
    m = max(m / 16.0, 1e-3);
    float f = 127.0 / m / 16.0;
    [[unroll]] for (uint j = 0u; j < FF / 4u / LS; j++) {
        uint u = t + LS * j;
        uint w = 0u;
        [[unroll]] for (uint c = 0u; c < 4u; c++) {
            float q = clamp(roundEven(s[4u * u + c] * f), -128.0, 127.0);
            w |= (uint(int(q)) & 0xFFu) << (8u * c);
        }
        xq[r * (FF / 4u) + u] = w;
    }
    if (t == 0u) { vv[rowf + r] = m * 128.0 / 127.0; }
}
