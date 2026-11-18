#version 450
// Elementwise steps between the Flash-Next decode's GEMVs (`gpu/gemv.rs`),
// so a hyper-connection's down -> up -> mix chain is one submission.
//   mode 0: x[dst + i] = silu(a y[src + i])                        (i < n)
//   mode 1: y[dst + i] = a sum_{c < hc} sigmoid(y[src + c n + i]) x[xsrc + c n + i]
//
// Grid: x = ceil(n / 256), y = positions (each with its own x / y region,
// XS / YS floats apart: `Gemv::run_chain_n`).
// The merged forms compute the same values (`QWEN_FLASH_MERGE`): mode 0 is
// `gemv_i8.glsl`'s SiLU epilogue, mode 1 `gemv_vec.glsl`'s mode 5; change
// the arithmetic of all three together.

#define XS 65536u
#define YS 262144u

layout(local_size_x = 256) in;

layout(set = 0, binding = 0, std430) buffer Xin { float x[]; };
layout(set = 0, binding = 1, std430) buffer Yout { float y[]; };
layout(set = 0, binding = 8, std140) uniform Params {
    uint mode;
    uint n;
    uint src;
    uint dst;
    uint xsrc;
    uint hc;
    float a;
};

float sigm(float v) { return 1.0 / (1.0 + exp(-v)); }

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= n) { return; }
    uint xo = gl_WorkGroupID.y * XS, yo = gl_WorkGroupID.y * YS;
    if (mode == 0u) {
        float v = a * y[yo + src + i];
        x[xo + dst + i] = v * sigm(v);
    } else {
        float s = 0.0;
        for (uint c = 0u; c < hc; c++) {
            s += sigm(y[yo + src + c * n + i]) * x[xo + xsrc + c * n + i];
        }
        y[yo + dst + i] = a * s;
    }
}
