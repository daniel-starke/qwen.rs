#version 450
#extension GL_EXT_control_flow_attributes : require
// MiniMax-H3-Image's VAE decoder attention (`gpu/h3vae.rs`): full attention
// within each tile of S tokens, 64-wide heads, q / k finished (`h3v_qk`, q
// already scaled). One query per thread, keys and values in chunks of 32
// through shared memory, the softmax online per chunk.
// Grid: x = ceil(S / 64), y = head, z = tile.
#define HD 64u
#define KC 32u
layout(local_size_x = 64) in;

// [rows][3 D]: q, k, v
layout(set = 0, binding = 1, std430) readonly buffer Qkv { float qkv[]; };
// [rows][D]
layout(set = 0, binding = 2, std430) writeonly buffer O { float o[]; };
layout(push_constant) uniform Pc {
    uint S;
    uint D;
};

shared float ks[KC][HD];
shared float vs[KC][HD];

void main() {
    uint t = gl_LocalInvocationID.x;
    uint qi = gl_WorkGroupID.x * 64u + t, h = gl_WorkGroupID.y;
    uint base = gl_WorkGroupID.z * S;
    uint ld = 3u * D;
    bool valid = qi < S;
    float q[HD], acc[HD];
    uint qr = (base + min(qi, S - 1u)) * ld + h * HD;
    [[unroll]] for (uint c = 0u; c < HD; c++) { q[c] = qkv[qr + c]; acc[c] = 0.0; }
    float m = -1e30, l = 0.0;
    for (uint k0 = 0u; k0 < S; k0 += KC) {
        uint n = min(KC, S - k0);
        for (uint i = t; i < KC * HD; i += 64u) {
            uint kr = i / HD, c = i % HD;
            uint row = (base + k0 + min(kr, n - 1u)) * ld + h * HD + c;
            ks[kr][c] = qkv[row + D];
            vs[kr][c] = qkv[row + 2u * D];
        }
        barrier();
        float sc[KC];
        float mc = m;
        for (uint j = 0u; j < KC; j++) {
            float s = 0.0;
            [[unroll]] for (uint c = 0u; c < HD; c++) { s = fma(q[c], ks[j][c], s); }
            sc[j] = j < n ? s : -1e30;
            mc = max(mc, sc[j]);
        }
        float corr = exp(m - mc);
        l *= corr;
        [[unroll]] for (uint c = 0u; c < HD; c++) { acc[c] *= corr; }
        for (uint j = 0u; j < KC; j++) {
            float p = exp(sc[j] - mc);
            l += p;
            [[unroll]] for (uint c = 0u; c < HD; c++) { acc[c] = fma(p, vs[j][c], acc[c]); }
        }
        m = mc;
        barrier();
    }
    if (valid) {
        float il = 1.0 / l;
        uint orow = (base + qi) * D + h * HD;
        [[unroll]] for (uint c = 0u; c < HD; c++) { o[orow + c] = acc[c] * il; }
    }
}
