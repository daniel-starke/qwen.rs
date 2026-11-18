#version 450
#extension GL_EXT_control_flow_attributes : require
// Flash-Next prefill, a delta layer's hyper-connection on the iGPU
// (`gpu/hcpre.rs`, the dp4a form), pass 1 of 3: the streams `x`
// [n][4][2560] f16 normed per stream with the layer's weight, rounded to
// int8 in groups of 16 -> `xq` [n][10240] int8 with the groups' scales
// `xs[XG + 640 p + g]` (max |xn| / 127; one scale per stream lost KL 0.05:
// the streams' outlier channels) and the norm's rsqrt `xs[XS + 1024 + 4 p +
// s]` (pass 3 forms the float `xn` again from `x`).
// One workgroup per (position, stream); threads 0..160 take a group each.
layout(local_size_x = 256) in;

layout(set = 0, binding = 0, std430) readonly buffer Xin { uvec4 x4[]; };
layout(set = 0, binding = 2, std430) readonly buffer Pw { float pw[]; };
layout(set = 0, binding = 3, std430) writeonly buffer Xq { uvec4 xq4[]; };
layout(set = 0, binding = 3, std430) writeonly buffer Xs { float xs[]; };
layout(set = 0, binding = 8, std140) uniform Params {
    uint n;
    uint wofs;
    uint pofs;  // the layer's norm / scales in `pw`: norm [10240], down [336], up [10240]
};

// the scales after HC_MAXN x 10240 int8 values
#define XS 655360u
#define XG (XS + 2048u)

shared float red[256];

void main() {
    uint p = gl_WorkGroupID.x / 4u, s = gl_WorkGroupID.x % 4u, t = gl_LocalInvocationID.x;
    bool on = t < 160u;
    float v[16];
    float ss = 0.0;
    if (on) {
        uint b = p * 1280u + s * 320u + 2u * t;
        [[unroll]] for (uint h = 0u; h < 2u; h++) {
            uvec4 u = x4[b + h];
            [[unroll]] for (uint c = 0u; c < 4u; c++) {
                vec2 f = unpackHalf2x16(u[c]);
                v[h * 8u + 2u * c] = f.x;
                v[h * 8u + 2u * c + 1u] = f.y;
            }
        }
        [[unroll]] for (uint i = 0u; i < 16u; i++) { ss += v[i] * v[i]; }
    } else {
        [[unroll]] for (uint i = 0u; i < 16u; i++) { v[i] = 0.0; }
    }
    red[t] = ss;
    barrier();
    for (uint o = 128u; o > 0u; o >>= 1u) {
        if (t < o) { red[t] += red[t + o]; }
        barrier();
    }
    float r = inversesqrt(red[0] / 2560.0 + 1e-6);
    barrier();
    float mx = 0.0;
    uint e0 = s * 2560u + 16u * t;
    if (on) {
        [[unroll]] for (uint i = 0u; i < 16u; i++) {
            v[i] *= r * pw[pofs + e0 + i];
            mx = max(mx, abs(v[i]));
        }
    }
    float sc = mx > 0.0 ? mx / 127.0 : 1.0;
    if (on) {
        uvec4 q;
        [[unroll]] for (uint c = 0u; c < 4u; c++) {
            uint w = 0u;
            [[unroll]] for (uint b = 0u; b < 4u; b++) {
                int k = clamp(int(roundEven(v[4u * c + b] / sc)), -127, 127);
                w |= (uint(k) & 255u) << (8u * b);
            }
            q[c] = w;
        }
        xq4[p * 640u + s * 160u + t] = q;
        xs[XG + p * 640u + s * 160u + t] = sc;
    }
    if (t == 0u) {
        xs[XS + 1024u + 4u * p + s] = r;
    }
}
