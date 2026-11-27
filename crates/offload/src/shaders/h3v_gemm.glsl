#version 450
#extension GL_EXT_control_flow_attributes : require
// MiniMax-H3-Image's VAE decoder (`gpu/h3vae.rs`): `y[m][n] = x[m] . w[n] +
// b[n]` (ADD: `y[m][n] +=` that, the residual), the weights the NPU's f16
// copies `[N][K]` (row-major, imported zero-copy), `x` f32 `[M][K]`, f32
// sums. 64 x 64 outputs a workgroup, 4 x 4 a thread, K in steps of 16
// through shared memory. Grid: x = N / 64, y = ceil(M / 64).
layout(local_size_x = 256) in;

layout(set = 0, binding = 0, std430) readonly buffer W { uint w[]; };
layout(set = 0, binding = 1, std430) readonly buffer X { float x[]; };
layout(set = 0, binding = 2, std430) buffer Y { float y[]; };
layout(set = 0, binding = 3, std430) readonly buffer B { uint bias[]; };
layout(push_constant) uniform Pc {
    uint K;
    uint N;
    uint M;
    uint add;
};

shared float as_[16][68];
shared float bs[16][68];

void main() {
    uint t = gl_LocalInvocationID.x;
    uint tm = gl_WorkGroupID.y * 64u, tn = gl_WorkGroupID.x * 64u;
    uint tx = t % 16u, ty = t / 16u;
    float acc[4][4];
    [[unroll]] for (uint i = 0u; i < 4u; i++) {
        [[unroll]] for (uint j = 0u; j < 4u; j++) { acc[i][j] = 0.0; }
    }
    uint kh = K / 2u;
    for (uint k0 = 0u; k0 < K; k0 += 16u) {
        [[unroll]] for (uint i = 0u; i < 4u; i++) {
            uint q = t + 256u * i;
            uint r = q / 16u, c = q % 16u;
            uint m = tm + r;
            as_[c][r] = m < M ? x[m * K + k0 + c] : 0.0;
        }
        [[unroll]] for (uint i = 0u; i < 2u; i++) {
            uint q = t + 256u * i;
            uint r = q / 8u, c = q % 8u;
            vec2 v = unpackHalf2x16(w[(tn + r) * kh + k0 / 2u + c]);
            bs[2u * c][r] = v.x;
            bs[2u * c + 1u][r] = v.y;
        }
        barrier();
        [[unroll]] for (uint kk = 0u; kk < 16u; kk++) {
            vec4 a = vec4(as_[kk][ty * 4u], as_[kk][ty * 4u + 1u], as_[kk][ty * 4u + 2u], as_[kk][ty * 4u + 3u]);
            vec4 b = vec4(bs[kk][tx * 4u], bs[kk][tx * 4u + 1u], bs[kk][tx * 4u + 2u], bs[kk][tx * 4u + 3u]);
            [[unroll]] for (uint i = 0u; i < 4u; i++) {
                [[unroll]] for (uint j = 0u; j < 4u; j++) { acc[i][j] = fma(a[i], b[j], acc[i][j]); }
            }
        }
        barrier();
    }
    [[unroll]] for (uint i = 0u; i < 4u; i++) {
        uint m = tm + ty * 4u + i;
        if (m >= M) { continue; }
        [[unroll]] for (uint j = 0u; j < 4u; j++) {
            uint n = tn + tx * 4u + j;
            float v = acc[i][j] + unpackHalf2x16(bias[n / 2u])[n & 1u];
            if (add != 0u) { v += y[m * N + n]; }
            y[m * N + n] = v;
        }
    }
}
