#version 450
#extension GL_EXT_scalar_block_layout : enable
// Attention of the NPU engine's vision tower on the scan device (the iGPU),
// pass 1 of 4: 2-D RoPE of one block's queries and keys, from the NPU's
// rows into f32 buffers laid out per head (`vis_qk_npu.glsl`,
// `vis_sm_npu.glsl` and `vis_pv_npu.glsl` are passes 2 to 4).
//   x   [n][3][H][72] f16: per patch q, k, v (the NPU graph's `qkv`)
//   cs  [n][18] vec4: per patch 9 quads of cos, 9 of sin (NEOX pairing
//       (j, j + 36) inside a head; the row's angles then the column's)
//   q   [H][nq][18] vec4 out, x 1 / sqrt(72); k the same, unscaled
//   v   [H][nq][18] vec4 out (in `x` a head's values of two patches are
//       7 KB apart, and pass 4 walks them patch by patch)
// Rows n .. nq (nq = n rounded up to 8) are zero. One invocation = one
// patch x one head x one quad pair.
#define HD4 18u
#define HALF4 9u
layout(local_size_x = 48) in;
layout(set = 0, binding = 0, std430) readonly buffer Xb { uvec2 x2[]; };
layout(set = 0, binding = 1, std430) readonly buffer Cb { vec4 cs[]; };
layout(set = 0, binding = 2, std430) writeonly buffer Qb { vec4 q4[]; };
layout(set = 0, binding = 3, std430) writeonly buffer Kb { vec4 k4[]; };
layout(set = 0, binding = 4, std430) writeonly buffer Vb { vec4 v4[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };

vec4 half4(uvec2 u) {
    return vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y));
}

void main() {
    uint n = p[0];
    uint heads = p[1];
    uint nq = p[2];
    uint t = gl_GlobalInvocationID.y;
    uint h = gl_GlobalInvocationID.x / HALF4;
    uint j = gl_GlobalInvocationID.x % HALF4;
    if (h >= heads) {
        return;
    }
    uint dst = (h * nq + t) * HD4 + j;
    if (t >= n) {
        q4[dst] = vec4(0.0);
        q4[dst + HALF4] = vec4(0.0);
        k4[dst] = vec4(0.0);
        k4[dst + HALF4] = vec4(0.0);
        v4[dst] = vec4(0.0);
        v4[dst + HALF4] = vec4(0.0);
        return;
    }
    vec4 c = cs[t * HD4 + j];
    vec4 s = cs[t * HD4 + HALF4 + j];
    uint row = heads * HD4;
    uint src = t * 3u * row + h * HD4 + j;
    vec4 a = half4(x2[src]);
    vec4 b = half4(x2[src + HALF4]);
    float scale = uintBitsToFloat(p[5]);
    q4[dst] = (a * c - b * s) * scale;
    q4[dst + HALF4] = (b * c + a * s) * scale;
    a = half4(x2[src + row]);
    b = half4(x2[src + row + HALF4]);
    k4[dst] = a * c - b * s;
    k4[dst + HALF4] = b * c + a * s;
    v4[dst] = half4(x2[src + 2u * row]);
    v4[dst + HALF4] = half4(x2[src + 2u * row + HALF4]);
}
