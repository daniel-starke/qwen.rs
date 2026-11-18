#version 450
// f16 x f32 matvec/matmul: one output row per thread.
//
// The half->float conversion is a hand routine, not `unpackHalf2x16`; the two
// agree on every value a weight can hold. Infinities and NaNs are not handled:
// a weight tensor containing them is already broken upstream.

layout(local_size_x = 64) in;

layout(set = 0, binding = 0, std430) readonly buffer Weights { uint weights[]; };
layout(set = 0, binding = 1, std430) readonly buffer Xin { float xin[]; };
layout(set = 0, binding = 2, std430) writeonly buffer Yout { float yout[]; };
layout(set = 0, binding = 3, std140) uniform Params {
    uint out_dim;
    uint in_dim;
    uint seq;
};

float f16_to_f32(uint h) {
    uint sgn = (h >> 15u) & 1u;
    uint exp = (h >> 10u) & 0x1Fu;
    uint mant = h & 0x3FFu;
    if (exp == 0u) {
        if (mant == 0u) {
            return sgn == 1u ? -0.0 : 0.0;
        }
        // Subnormal: no implicit leading 1, and the exponent is fixed at 2^-24
        // per mantissa step.
        float mag = float(mant) * exp2(-24.0);
        return sgn == 1u ? -mag : mag;
    }
    uint outbits = (sgn << 31u) | ((exp + 112u) << 23u) | (mant << 13u);
    return uintBitsToFloat(outbits);
}

// One byte out of the `uint` weight array.
uint rd(uint off) {
    uint w = weights[off / 4u];
    return (w >> ((off % 4u) * 8u)) & 0xFFu;
}

void main() {
    uint o = gl_GlobalInvocationID.x;
    uint si = gl_GlobalInvocationID.y;
    if (o >= out_dim || si >= seq) {
        return;
    }
    float acc = 0.0;
    uint row_base = o * in_dim;
    for (uint k = 0u; k < in_dim; k++) {
        uint e = (row_base + k) * 2u;
        uint h = rd(e) | (rd(e + 1u) << 8u);
        acc += f16_to_f32(h) * xin[si * in_dim + k];
    }
    yout[si * out_dim + o] = acc;
}
