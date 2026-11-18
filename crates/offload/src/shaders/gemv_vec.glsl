#version 450
// Vector steps between the Flash-Next decode's GEMVs (`gpu/gemv.rs`
// `Gemv::add_vec`), so a layer's mixer input and its whole post-mixer part
// (output projection -> injection -> norm -> hyper-connection -> router /
// shared expert -> SwiGLU -> down) chain in one submission each. `w` is the
// step's own weight vector (binding 2).
//   mode 0 (rot):    v = src[i] (* w[i]); FWHT over blocks of `blk` (0 =
//                    none), x 1/sqrt(blk); y[ycopy + i] = v (if ycopy !=
//                    ~0); x[dst + i] = round(v) clamped to int8 (flags & 2)
//                    or v. src is y with flags & 1, else x; with flags & 8
//                    src[i] is mode 3's SwiGLU of y[src]. Grid ceil(n / 256).
//   mode 5 (mix + rot): src[i] is mode 1 of `gemv_act.glsl` (a sum_{c < hc}
//                    sigmoid(y[src + c n + i]) x[aux + c n + i]), written
//                    as is to x[extra + i] (if extra != ~0), then mode 0.
//   mode 1 (norm):   stream c (workgroup c): x[dst + c n + i] = s[c n + i] /
//                    sqrt(mean_i s^2 + a) * w[c n + i], s = src. Grid hc.
//                    With flags & 8 mode 2 first, on the streams at src
//                    (x): s = src[c n + i] + y[extra + i] x[aux + c],
//                    stored back and to y[ycopy + c n + i].
//   mode 2 (inject): x[dst + c n + i] += y[src + i] x[aux + c] for c < hc,
//                    also to y[ycopy + c n + i]. Grid ceil(n / 256).
//   mode 3 (swiglu): x[dst + i] = silu(y[src + i]) y[src + n + i].
//   mode 4 (inj):    x[dst + i] = 2 sigmoid(a y[src + i]) (a
//                    hyper-connection's injection weights, `gpudec::inj_w`).
// Modes 5, 1 with flags & 8 and 0 with flags & 8 are the steps they merge
// in one dispatch (one barrier less each), each value's arithmetic the
// separate steps' in their order.
// Grid y: positions, each with its own x / y region (XS / YS floats apart,
// `Gemv::run_chain_n`); `w` is shared.
// The FWHT's butterflies are `npu::flash::cpuexp::fwht`'s, in its order.

#define XS 65536u
#define YS 262144u

layout(local_size_x = 256) in;

layout(set = 0, binding = 0, std430) buffer Xin { float x[]; };
layout(set = 0, binding = 1, std430) buffer Yout { float y[]; };
layout(set = 0, binding = 2, std430) readonly buffer Wv { float w[]; };
layout(set = 0, binding = 8, std140) uniform Params {
    uint mode;
    uint n;
    uint src;
    uint dst;
    uint aux;
    uint hc;
    float a;
    uint blk;
    uint flags;
    uint ycopy;
    uint extra;
};

shared float s[256];

float sigm(float v) { return 1.0 / (1.0 + exp(-v)); }

void main() {
    uint lid = gl_LocalInvocationID.x;
    uint g = gl_WorkGroupID.x;
    uint xo = gl_WorkGroupID.y * XS, yo = gl_WorkGroupID.y * YS;
    if (mode == 0u || mode == 5u) {
        uint i = g * 256u + lid;
        float v = 0.0;
        if (i < n) {
            if (mode == 5u) {
                float s = 0.0;
                for (uint c = 0u; c < hc; c++) {
                    s += sigm(y[yo + src + c * n + i]) * x[xo + aux + c * n + i];
                }
                v = a * s;
                if (extra != 0xffffffffu) { x[xo + extra + i] = v; }
            } else if ((flags & 8u) != 0u) {
                float gt = y[yo + src + i];
                v = gt * sigm(gt) * y[yo + src + n + i];
            } else {
                v = (flags & 1u) != 0u ? y[yo + src + i] : x[xo + src + i];
            }
            if ((flags & 4u) != 0u) { v *= w[i]; }
        }
        if (blk > 1u) {
            s[lid] = v;
            for (uint h = 1u; h < blk; h *= 2u) {
                barrier();
                if ((lid & h) == 0u) {
                    float p = s[lid], q = s[lid + h];
                    s[lid] = p + q;
                    s[lid + h] = p - q;
                }
            }
            barrier();
            v = s[lid] * inversesqrt(float(blk));
        }
        if (i >= n) { return; }
        if (ycopy != 0xffffffffu) { y[yo + ycopy + i] = v; }
        if ((flags & 2u) != 0u) { v = clamp(roundEven(v), -128.0, 127.0); }
        x[xo + dst + i] = v;
    } else if (mode == 1u) {
        uint b = g * n;
        float acc = 0.0;
        for (uint i = lid; i < n; i += 256u) {
            float v;
            if ((flags & 8u) != 0u) {
                v = x[xo + src + b + i] + y[yo + extra + i] * x[xo + aux + g];
                x[xo + src + b + i] = v;
                if (ycopy != 0xffffffffu) { y[yo + ycopy + b + i] = v; }
            } else {
                v = (flags & 1u) != 0u ? y[yo + src + b + i] : x[xo + src + b + i];
            }
            acc += v * v;
        }
        s[lid] = acc;
        for (uint h = 128u; h > 0u; h >>= 1u) {
            barrier();
            if (lid < h) { s[lid] += s[lid + h]; }
        }
        barrier();
        float r = inversesqrt(s[0] / float(n) + a);
        for (uint i = lid; i < n; i += 256u) {
            float v = (flags & 1u) != 0u ? y[yo + src + b + i] : x[xo + src + b + i];
            x[xo + dst + b + i] = v * r * w[b + i];
        }
    } else if (mode == 2u) {
        uint i = g * 256u + lid;
        if (i >= n) { return; }
        float o = y[yo + src + i];
        for (uint c = 0u; c < hc; c++) {
            float v = x[xo + dst + c * n + i] + o * x[xo + aux + c];
            x[xo + dst + c * n + i] = v;
            if (ycopy != 0xffffffffu) { y[yo + ycopy + c * n + i] = v; }
        }
    } else if (mode == 4u) {
        uint i = g * 256u + lid;
        if (i >= n) { return; }
        x[xo + dst + i] = 2.0 * sigm(a * y[yo + src + i]);
    } else {
        uint i = g * 256u + lid;
        if (i >= n) { return; }
        float gt = y[yo + src + i];
        x[xo + dst + i] = gt * sigm(gt) * y[yo + src + n + i];
    }
}
