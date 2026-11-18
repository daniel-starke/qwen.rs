#version 450
#extension GL_EXT_integer_dot_product : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
// The Qwen-Image transformer's int8 x int8 matmuls on a Vulkan device (the
// iGPU, `gpu/ditblk.rs`): `y[m][n] = (xq[m] . w[n]) 2^-14 s[n] f[m]`, the
// weights the NPU's int8 codes `[N][K]` (row-major, imported zero-copy),
// `xq` int8 rows `[M][K]`, exact int32 sums; `y = sum 2^-14 s[n] f[m]`
// (codes and activations both on grids of 1 / 128).
//
// exp_gemm.glsl's scheme: a workgroup is TOK / 8 token groups of 32 lanes;
// lane l of group tg holds rows l + 32 i (i < RT) and the group's 8
// tokens tg 8 + j. The weights go through shared memory (K in steps of 64
// bytes, rows ST uvec4 apart); lane l loads token tg 8 + l % 8's 64 bytes
// into registers and the others read it by `subgroupBroadcast`.
// Grid: x = N / RW, y = M / TOK (rows past M read row M - 1, not written).
#ifndef RT
#define RT 2u
#endif
#ifndef TOK
#define TOK 64u
#endif
#define NG (TOK / 8u)
#define LS (32u * NG)
#define RW (32u * RT)
#define ST 5u
layout(local_size_x = LS) in;

layout(set = 0, binding = 0, std430) readonly buffer Wc { uvec4 wc[]; };
layout(set = 0, binding = 1, std430) readonly buffer Xq { uvec4 xq[]; };
layout(set = 0, binding = 2, std430) writeonly buffer Y { float y[]; };
// f32 vectors: the row scales s[n] at `sofs`, the token factors f[m] at
// `fofs` (~0: none)
layout(set = 0, binding = 3, std430) readonly buffer V { float vv[]; };
layout(push_constant) uniform Pc {
    uint K;      // bytes per row (multiple of 64)
    uint N;
    uint M;
    uint ldy;    // y's row stride in floats
    uint sofs;
    uint fofs;
    uint ldw;    // the weights' row stride in bytes (multiple of 16; 0: K)
};

shared uvec4 sw[RW * ST];
shared float ep[TOK * (RW + 1u)];

void main() {
    uint t = gl_LocalInvocationID.x;
    uint r0 = gl_WorkGroupID.x * RW, m0 = gl_WorkGroupID.y * TOK;
    uint kv = K / 16u;
    uint kw = ldw == 0u ? kv : ldw / 16u;
    uint ln = t % 32u, tg = t / 32u;
    uint tk = tg * 8u + gl_SubgroupInvocationID % 8u;
    uint xr = min(m0 + tk, M - 1u) * kv;
    int acc[RT * 8u];
    [[unroll]] for (uint i = 0u; i < RT * 8u; i++) { acc[i] = 0; }
    for (uint k0 = 0u; k0 < kv; k0 += 4u) {
        // RW rows x 4 uvec4: LS threads, RW * 4 / LS loads each
        [[unroll]] for (uint h = 0u; h < RW * 4u / LS; h++) {
            uint q = t + h * LS;
            uint r = q / 4u, c = q % 4u;
            sw[r * ST + c] = wc[(r0 + r) * kw + k0 + c];
        }
        uvec4 xv[4];
        [[unroll]] for (uint p = 0u; p < 4u; p++) { xv[p] = xq[xr + k0 + p]; }
        barrier();
        [[unroll]] for (uint p = 0u; p < 4u; p++) {
            ivec4 wv[RT];
            [[unroll]] for (uint i = 0u; i < RT; i++) { wv[i] = ivec4(sw[(ln + 32u * i) * ST + p]); }
#define TOKJ(j) { ivec4 x = ivec4(subgroupBroadcast(xv[p], j)); \
                [[unroll]] for (uint i = 0u; i < RT; i++) { \
                    int a = dotPacked4x8AccSatEXT(wv[i].x, x.x, acc[i * 8u + j]); \
                    a = dotPacked4x8AccSatEXT(wv[i].y, x.y, a); \
                    a = dotPacked4x8AccSatEXT(wv[i].z, x.z, a); \
                    acc[i * 8u + j] = dotPacked4x8AccSatEXT(wv[i].w, x.w, a); } }
            TOKJ(0u) TOKJ(1u) TOKJ(2u) TOKJ(3u) TOKJ(4u) TOKJ(5u) TOKJ(6u) TOKJ(7u)
        }
        barrier();
    }
    [[unroll]] for (uint i = 0u; i < RT; i++) {
        float s = (sofs == ~0u ? 1.0 : vv[sofs + r0 + ln + 32u * i]) * (1.0 / 16384.0);
        [[unroll]] for (uint j = 0u; j < 8u; j++) {
            ep[(tg * 8u + j) * (RW + 1u) + ln + 32u * i] = float(acc[i * 8u + j]) * s;
        }
    }
    barrier();
    for (uint q = t; q < TOK * RW; q += LS) {
        uint k = q / RW, r = q % RW;
        if (m0 + k < M) {
            float f = fofs == ~0u ? 1.0 : vv[fofs + m0 + k];
            y[(m0 + k) * ldy + r0 + r] = ep[k * (RW + 1u) + r] * f;
        }
    }
}
