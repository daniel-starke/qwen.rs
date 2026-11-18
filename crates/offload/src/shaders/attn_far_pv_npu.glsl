#version 450
#extension GL_EXT_scalar_block_layout : enable
// Far attention of the NPU engine on the scan device (the iGPU), pass 2 of 2
// of the fused form (`attn_far_qks_npu.glsl` is pass 1 and has done the
// softmax): O = f O + P V, no shared memory, no barriers.
//   sc  [24 * S][CAP] f32: the block's probabilities exp(score - m_new);
//       at vec4 p[3] on, f = exp(m_old - m_new) per row
//   v   [CAP][4][256] f16, one V block
//   o   [24 * S][256] f32, sum of exp(score - m) * v, NOT normalized
// f is 0 exactly on a query's first block (m_old = -1e30): o is then not read,
// it may hold anything. The host merges o / l with the NPU's blocks by log
// sum exp (m + ln l).
// One workgroup = one KV group x QT = 4 positions; an invocation owns one
// dim-quad of one position for all 6 heads (10 loads per 96 multiply-adds,
// V read straight from its buffer), as in `attn_far_npu.glsl`.
#define HD 256u
#define HPG 6u
#define QT 4u

layout(local_size_x = 256) in;

layout(set = 0, binding = 2, std430) readonly buffer Vb { uvec2 v2[]; };
layout(set = 0, binding = 3, std430) buffer Ob { vec4 o4[]; };
layout(set = 0, binding = 5, std430) readonly buffer Sb { vec4 sc4[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };

vec4 half4(uvec2 u) {
    return vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y));
}

void main() {
    uint S = p[0];
    uint cap4 = p[2] / 4u;
    uint fb = p[3];
    uint tid = gl_LocalInvocationID.x;
    uint tb = gl_WorkGroupID.x * QT;
    uint g = gl_WorkGroupID.y;
    uint d4 = tid & 63u;
    uint pp = tid >> 6u;
    uint row0 = g * HPG * S + tb + pp;
    vec4 acc[HPG];
    for (uint h = 0u; h < HPG; h++) {
        uint row = row0 + h * S;
        float f = sc4[fb + row / 4u][row % 4u];
        acc[h] = (f == 0.0) ? vec4(0.0) : o4[row * (HD / 4u) + d4] * f;
    }
    uint v0i = g * (HD / 4u) + d4;
    for (uint kq = 0u; kq < cap4; kq++) {
        uint vi = v0i + kq * 4u * HD;
        vec4 a0 = half4(v2[vi]);
        vec4 a1 = half4(v2[vi + HD]);
        vec4 a2 = half4(v2[vi + 2u * HD]);
        vec4 a3 = half4(v2[vi + 3u * HD]);
        for (uint h = 0u; h < HPG; h++) {
            vec4 pr = sc4[(row0 + h * S) * cap4 + kq];
            acc[h] += pr.x * a0 + pr.y * a1 + pr.z * a2 + pr.w * a3;
        }
    }
    for (uint h = 0u; h < HPG; h++) {
        o4[(row0 + h * S) * (HD / 4u) + d4] = acc[h];
    }
}
