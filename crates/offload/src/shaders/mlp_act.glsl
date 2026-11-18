#version 450
// The NPU engine's decode MLP share on a discrete GPU, between the gate /
// up GEMVs and the down GEMV (`gpu/mlpsplit.rs`): p = H (silu(g) * u * d3)
// per block of 256 (H = Sylvester Hadamard / 16), quantized to int8 with
// the block's own scale and written as `i4_dp4a.glsl` reads an activation
// (per 32 elements 16 even bytes, then 16 odd bytes; `inv` per block;
// `corr` = -8 inv sum(q) per block).
//
// Grid: x = rows / 256, y = positions.

layout(local_size_x = 256) in;

layout(set = 0, binding = 0, std430) readonly buffer G { float g[]; };
layout(set = 0, binding = 1, std430) readonly buffer U { float u[]; };
layout(set = 0, binding = 2, std430) readonly buffer D { float d3[]; };
layout(set = 0, binding = 3, std430) writeonly buffer X { uint xq[]; };
layout(set = 0, binding = 4, std430) writeonly buffer Inv { float inv[]; };
layout(set = 0, binding = 5, std430) writeonly buffer Corr { float corr[]; };
layout(set = 0, binding = 8, std140) uniform Params {
    uint rows;
    uint in_dim;
    uint seq;
};

shared float sh[256];
shared int qi[256];
shared int si[256];

void main() {
    uint t = gl_LocalInvocationID.x;
    uint blk = gl_WorkGroupID.x;
    uint s = gl_WorkGroupID.y;
    uint e = blk * 256u + t;
    float gv = g[s * rows + e];
    sh[t] = gv / (1.0 + exp(-gv)) * u[s * rows + e] * d3[e];
    memoryBarrierShared();
    barrier();
    for (uint len = 1u; len < 256u; len <<= 1u) {
        float a = sh[t & ~len], c = sh[t | len];
        barrier();
        sh[t] = ((t & len) == 0u) ? a + c : a - c;
        memoryBarrierShared();
        barrier();
    }
    float val = sh[t] * 0.0625;
    barrier();
    sh[t] = abs(val);
    memoryBarrierShared();
    barrier();
    for (uint len = 1u; len < 256u; len <<= 1u) {
        if ((t & (2u * len - 1u)) == 0u) { sh[t] = max(sh[t], sh[t + len]); }
        memoryBarrierShared();
        barrier();
    }
    float mx = sh[0];
    int q = (mx > 0.0) ? int(round(val * (127.0 / mx))) : 0;
    qi[t] = q;
    si[t] = q;
    memoryBarrierShared();
    barrier();
    if (t < 64u) {
        // uint t of the block: weight load t / 8, even / odd half, 4 bytes
        uint par = (t >> 2u) & 1u;
        uint e0 = (t >> 3u) * 32u + (t & 3u) * 8u + par;
        uint v = uint(qi[e0] & 255) | (uint(qi[e0 + 2u] & 255) << 8u) | (uint(qi[e0 + 4u] & 255) << 16u) | (uint(qi[e0 + 6u] & 255) << 24u);
        xq[(s * (rows / 32u) + blk * 8u) * 8u + t] = v;
    }
    for (uint len = 1u; len < 256u; len <<= 1u) {
        if ((t & (2u * len - 1u)) == 0u) { si[t] += si[t + len]; }
        memoryBarrierShared();
        barrier();
    }
    if (t == 0u) {
        float iv = mx / 127.0;
        inv[s * (rows / 256u) + blk] = iv;
        corr[s * 128u + blk] = -8.0 * iv * float(si[0]);
    }
}
