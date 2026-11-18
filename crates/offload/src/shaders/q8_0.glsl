#version 450
// Fused Q8_0 dequant-matmul: one output row per thread.
//
// Q8_0 block (32 weights, 34 bytes): an f16 scale `d` followed by 32 signed
// bytes. Unlike the K-quants this kernel takes `x` as raw f32, not q8_K: the
// host packs by `GpuKind` and only Q4_K/Q6_K get the quantized activation.

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

const uint QK = 32u;
const uint BLK = 34u;

void main() {
    uint o = gl_GlobalInvocationID.x;
    uint si = gl_GlobalInvocationID.y;
    if (o >= out_dim || si >= seq) {
        return;
    }
    float acc = 0.0;
    uint row_base = o * in_dim;
    for (uint k = 0u; k < in_dim; k++) {
        uint e = row_base + k;
        uint b = e / QK;
        uint q = e % QK;
        uint base = b * BLK;
        float d = f16_to_f32(rd(base) | (rd(base + 1u) << 8u));
        uint qb = rd(base + 2u + q);
        float qs = float(qb) - (qb >= 128u ? 256.0 : 0.0);
        acc += d * qs * xin[si * in_dim + k];
    }
    yout[si * out_dim + o] = acc;
}
