#version 450
// Flash-Next prefill, attention hyper-connection of a delta layer on the
// iGPU (`gpu/hcpre.rs`), pass 1 of 3: the streams `x` [n][4][2560] f16
// normed per stream with the layer's weight -> `xn` [n][10240] f32.
// One workgroup per (position, stream).
layout(local_size_x = 256) in;

layout(set = 0, binding = 0, std430) readonly buffer Xin { uint x[]; };
layout(set = 0, binding = 2, std430) readonly buffer Pw { float pw[]; };
layout(set = 0, binding = 3, std430) writeonly buffer Xn { float xn[]; };
layout(set = 0, binding = 8, std140) uniform Params {
    uint n;
    uint wofs;  // the layer's weights in `w4` (uvec4)
    uint pofs;  // its norm / scales in `pw`: norm [10240], down [336], up [10240]
};

shared float red[256];

void main() {
    uint p = gl_WorkGroupID.x / 4u, s = gl_WorkGroupID.x % 4u, t = gl_LocalInvocationID.x;
    uint base = p * 5120u + s * 1280u;
    vec2 v[5];
    float ss = 0.0;
    for (uint i = 0u; i < 5u; i++) {
        v[i] = unpackHalf2x16(x[base + t + 256u * i]);
        ss += dot(v[i], v[i]);
    }
    red[t] = ss;
    barrier();
    for (uint o = 128u; o > 0u; o >>= 1u) {
        if (t < o) { red[t] += red[t + o]; }
        barrier();
    }
    float r = inversesqrt(red[0] / 2560.0 + 1e-6);
    for (uint i = 0u; i < 5u; i++) {
        uint e = s * 2560u + 2u * (t + 256u * i);
        xn[p * 10240u + e] = v[i].x * r * pw[pofs + e];
        xn[p * 10240u + e + 1u] = v[i].y * r * pw[pofs + e + 1u];
    }
}
