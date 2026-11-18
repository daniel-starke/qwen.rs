#version 450
#extension GL_EXT_scalar_block_layout : enable
// quantize_q8k: pack f32 activation `x` (length in_dim) into the q8_K layout the
// fused matmul shaders expect: in_dim int8 `qs` bytes followed by one f32 `d` per
// 256-block. One workgroup per 256-block. Mirrors quant::quantize_q8k exactly
// (d = 1/iscale, iscale = -127/amax, qs = round(iscale*x) clamped to [-128,127])
// so the matmul output matches the CPU reference to q8 precision.
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer Xin { float x[]; };
layout(set = 0, binding = 1, std430) buffer Out { uint o[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };
shared float shx[256];
shared int qi[256];
void main() {
    uint in_dim = p[0];
    uint b = gl_WorkGroupID.x;
    uint j = gl_LocalInvocationID.x;
    uint base = b * 256u;
    float v = x[base + j];
    shx[j] = abs(v);
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (j < stride) { shx[j] = max(shx[j], shx[j + stride]); }
        barrier();
    }
    float amax = shx[0];
    float iscale = (amax > 0.0) ? (-127.0 / amax) : 0.0;
    int qj = 0;
    if (amax > 0.0) {
        float r = iscale * v;
        float rr = (r >= 0.0) ? floor(r + 0.5) : -floor(-r + 0.5);
        qj = int(clamp(rr, -128.0, 127.0));
    }
    qi[j] = qj;
    barrier();
    if ((j & 3u) == 0u) {
        uint w = 0u;
        for (uint k = 0u; k < 4u; k++) {
            w |= uint(qi[j + k] & 0xFF) << (k * 8u);
        }
        o[(base + j) / 4u] = w;
    }
    if (j == 0u) {
        float ds = (amax > 0.0) ? (1.0 / iscale) : 0.0;
        o[(in_dim + b * 4u) / 4u] = floatBitsToUint(ds);
    }
}
