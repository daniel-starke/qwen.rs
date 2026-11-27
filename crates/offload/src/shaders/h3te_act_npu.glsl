#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : require
// `h3_act_npu` for an MLP whose hidden row does not fit shared memory
// (the H3 text encoder's 25600): the row in HALVES pieces of whole
// Hadamard blocks, each rotated in shared memory and written back over
// the gate half of its own `y` row (its gate values are read by then),
// then the row's maximum and the int8 codes from there. The math is
// `h3_act_npu`'s: `hid = swish(g sg) (u / 256)`, `rotate(hid)` in blocks
// of 256, int8 per row at `127 / 128 / max |row|` (floor 1e-3 on the
// normalized maximum), the row's factor `max |row| 128 / 127` -> V.
#ifndef FF
#define FF 25600u
#endif
#ifndef HALVES
#define HALVES 2u
#endif
#define LS 256u
#define CH (FF / HALVES)
#define PER (CH / LS)
layout(local_size_x = LS) in;

layout(set = 0, binding = 1, std430) coherent buffer Y { float y[]; };
layout(set = 0, binding = 3, std430) buffer V { float vv[]; };
layout(set = 0, binding = 4, std430) writeonly buffer Xq { uint xq[]; };
layout(push_constant) uniform Pc {
    uint sg;    // offset of gu.sg in V
    uint rowf;  // offset of the rows' factors in V
};

shared float s[CH];
shared float red[LS / 8u];

void main() {
    uint t = gl_LocalInvocationID.x, r = gl_WorkGroupID.x;
    uint row = r * 2u * FF;
    float m = 0.0;
    for (uint hv = 0u; hv < HALVES; hv++) {
        uint o = hv * CH;
        [[unroll]] for (uint j = 0u; j < PER; j++) {
            uint e = t + LS * j;
            float g = y[row + o + e] * vv[sg + o + e];
            s[e] = g / (1.0 + exp(-g)) * (y[row + FF + o + e] * (1.0 / 256.0));
        }
        barrier();
        for (uint h = 1u; h < 256u; h *= 2u) {
            for (uint p = t; p < CH / 2u; p += LS) {
                uint b = p / 128u, q = p % 128u;
                uint i = b * 256u + (q / h) * 2u * h + q % h;
                float a0 = s[i], a1 = s[i + h];
                s[i] = a0 + a1;
                s[i + h] = a0 - a1;
            }
            barrier();
        }
        [[unroll]] for (uint j = 0u; j < PER; j++) {
            uint e = t + LS * j;
            m = max(m, abs(s[e]));
            y[row + o + e] = s[e];
        }
        barrier();
    }
    m = subgroupMax(m);
    if (gl_SubgroupInvocationID == 0u) { red[gl_SubgroupID] = m; }
    memoryBarrierBuffer();
    barrier();
    m = 0.0;
    for (uint i = 0u; i < gl_NumSubgroups; i++) { m = max(m, red[i]); }
    // the normalized rotation's 1 / 16
    m = max(m / 16.0, 1e-3);
    float f = 127.0 / m / 16.0;
    for (uint u = t; u < FF / 4u; u += LS) {
        uint w = 0u;
        [[unroll]] for (uint c = 0u; c < 4u; c++) {
            float q = clamp(roundEven(y[row + 4u * u + c] * f), -128.0, 127.0);
            w |= (uint(int(q)) & 0xFFu) << (8u * c);
        }
        xq[r * (FF / 4u) + u] = w;
    }
    if (t == 0u) { vv[rowf + r] = m * 128.0 / 127.0; }
}
