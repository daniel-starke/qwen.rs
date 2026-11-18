#version 450
#extension GL_EXT_scalar_block_layout : enable
// Attention of the NPU engine's vision tower on the scan device (the iGPU),
// pass 3 of 4: exp(score - max) of a tile's scores, in place.
//   mx  [H][tq][16] f32: [0, 8) parts of a row's largest score
//       (`vis_qk_npu.glsl`); [8, 16) out: parts of the row's sum
//   sc  [H][tq][nq / 4] vec4: the tile's scores (overwritten)
// One invocation = an eighth of the keys of one row. No shared memory and
// no barriers: as one workgroup per 8 rows that reduced the largest score
// and the sum through shared memory, this pass took 60 to 430 ms per block
// at 4096 patches from one run to the next.
#define PARTS 8u
layout(local_size_x = PARTS) in;
layout(set = 0, binding = 4, std430) buffer Mb { float mx[]; };
layout(set = 0, binding = 5, std430) buffer Sb { vec4 sc4[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };

// exp(x) for x <= 0, relative error 2e-7: 2^floor(y) 2^(y - floor(y)) with
// y = x log2(e), the second factor as sqrt(2) times the degree-6 Taylor
// polynomial of exp around 0 (its argument is within ln(2) / 2). Below
// x = -87 the result stays exp(-87) = 1.6e-38 (a key past the last one: its
// V is zero).
vec4 exp_neg(vec4 x) {
    vec4 y = max(x, vec4(-87.0)) * 1.4426950408889634;
    vec4 fl = floor(y);
    vec4 g = (y - fl - 0.5) * 0.6931471805599453;
    vec4 r = (((((g * (1.0 / 720.0) + 1.0 / 120.0) * g + 1.0 / 24.0) * g + 1.0 / 6.0) * g + 0.5) * g + 1.0) * g + 1.0;
    return r * 1.4142135623730951 * intBitsToFloat((ivec4(fl) + 127) << 23);
}

void main() {
    uint n4 = p[2] / 4u;
    uint tq = p[4];
    uint part = gl_GlobalInvocationID.x;
    uint row = gl_GlobalInvocationID.z * tq + gl_GlobalInvocationID.y;
    float mn = -1e30;
    for (uint i = 0u; i < PARTS; i++) {
        mn = max(mn, mx[row * 16u + i]);
    }
    uint per = (n4 + PARTS - 1u) / PARTS;
    uint s0 = min(part * per, n4);
    uint s1 = min(s0 + per, n4);
    uint sbase = row * n4;
    vec4 t4 = vec4(0.0);
    for (uint i = s0; i < s1; i++) {
        vec4 e = exp_neg(sc4[sbase + i] - mn);
        sc4[sbase + i] = e;
        t4 += e;
    }
    mx[row * 16u + PARTS + part] = (t4.x + t4.y) + (t4.z + t4.w);
}
