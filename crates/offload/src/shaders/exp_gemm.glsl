#version 450
#extension GL_EXT_integer_dot_product : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
// Routed experts of the Qwen3.8-Flash-Next NPU engine's prefill on a Vulkan
// device (the iGPU, `gpu/experts.rs`): a grouped int8 x int4 GEMM over
// slots of up to 64 tokens of one expert, built twice:
//
//   -DGU : gate / up, `g[row] = (xq . w) / 256 * s` for the 1280 rows of
//          `gu`, xq the int8 MoE input (`exp_xq.glsl`) -> `G` f32
//   (no) : down, `y[row] = (hq . w) / 256 * s` for the 2560 rows of `d`,
//          hq the int8 output of `exp_act.glsl` -> `Y` f16
//
// The math of `npu::flash::cpuexp`. A code byte holds element 2k in its
// low nibble and 2k + 1 in its high one, two's complement; a 32-bit word
// of 8 codes sign-extends into two words of 4 int8 (even, odd elements),
// the activation is packed the same way per 8 elements, so a word pair is
// two `dp4a`. Exact int32 sums.
//
// A workgroup is TOK / 8 token groups of 32 lanes; lane l of group tg
// holds rows l + 32 i (i < RT) and the group's 8 tokens tg 8 + j. The
// weights go through shared memory (K in steps of 64, rows 5 uvec4
// apart); the activations do not: lane l loads token tg 8 + l % 8 into
// registers and the others read it by `subgroupBroadcast`, so one shared
// uvec4 feeds 32 `dp4a` instead of the 8 of a 4 x 4 thread tile with both
// operands in shared memory (1.34 -> 1.65 TOPS on the Arrow Lake iGPU,
// `expbench`). A group's 32 lanes are whole subgroups of 8, 16 or 32.
// RT 2 for both slot sizes (64 rows; the compiler picks SIMD16). Measured
// and slower: RT 1 / 4, 16 tokens a lane (two broadcast banks), the next
// K step prefetched into registers (spills).
// Grid: x = rows / RW, y = slots from `sbase`.
#ifndef RT
#define RT 2u
#endif
#ifndef TOK
#define TOK 64u
#endif
#define NG (TOK / 8u)
#define LS (32u * NG)
#define RW (32u * RT)
layout(local_size_x = LS) in;

#ifdef GU
#define K 2560u
#define NROW 1280u
#else
#define K 640u
#define NROW 2560u
#endif
#define KW (K / 8u)
#define ST 5u

layout(set = 0, binding = 0, std430) readonly buffer Wc { uint wc[]; };
layout(set = 0, binding = 1, std430) readonly buffer Ws { uint ws[]; };
// int8 rows [row][K / 4 words], per 8 elements (even, odd): the MoE input
// per token (gate / up) or the down input per slot row
layout(set = 0, binding = 2, std430) readonly buffer Xq { uvec4 xq4[]; };
// slots: (expert, tokens, token list offset, first output row), then the
// token lists
layout(set = 0, binding = 3, std430) readonly buffer Sl { uint sl[]; };
#ifdef GU
layout(set = 0, binding = 4, std430) writeonly buffer G { float gb[]; };
#else
layout(set = 0, binding = 4, std430) writeonly buffer Y { uint yb[]; };
#endif
layout(set = 0, binding = 8, std140) uniform Params {
    uint wofs;   // codes' offset in words
    uint sofs;   // scales' offset in f16 pairs (even)
    uint sbase;  // the dispatch's first slot
};

// [row][group pair] (lo0, hi0, lo1, hi1): 8 groups of 8 = 4 uvec4 a row
shared uvec4 sw[RW * ST];
// the epilogue's [token][row]
shared float ep[TOK * (RW + 1u)];

uint sx4(uint w) {
    // nibbles -> int8, sign-extended (0x8..0xF -> 0xF8..0xFF)
    return w | ((w & 0x08080808u) * 0x1Eu);
}

float scale(uint i) {
    uint u = ws[sofs / 2u + i / 2u];
    return unpackHalf2x16(u)[i & 1u];
}

void main() {
    uint t = gl_LocalInvocationID.x;
    uint s = sbase + gl_WorkGroupID.y;
    uint e = sl[4u * s], nt = sl[4u * s + 1u], to = sl[4u * s + 2u], orow = sl[4u * s + 3u];
    uint r0 = gl_WorkGroupID.x * RW;
    uint ln = t % 32u, tg = t / 32u;
    // weight loads: word lg of rows lr + (LS / 8) h
    uint lg = t % 8u, lr = t / 8u;
    uint wr0 = wofs + (e * NROW + r0 + lr) * KW + lg;
    // this lane's token (the subgroup's lanes j < 8 are tokens tg 8 + j)
    uint tk = tg * 8u + gl_SubgroupInvocationID % 8u;
    bool tv = tk < nt;
#ifdef GU
    uint xr = tv ? sl[to + tk] * (K / 16u) : 0u;
#else
    uint xr = (orow + min(tk, nt - 1u)) * (K / 16u);
#endif
    int acc[RT * 8u];
    [[unroll]] for (uint i = 0u; i < RT * 8u; i++) { acc[i] = 0; }
    for (uint k0 = 0u; k0 < KW; k0 += 8u) {
        uint c = 2u * (lg % 2u), q = lg / 2u;
        [[unroll]] for (uint h = 0u; h < RW / (LS / 8u); h++) {
            uint w = wc[wr0 + h * (LS / 8u) * KW + k0];
            sw[(lr + (LS / 8u) * h) * ST + q][c] = sx4(w & 0x0F0F0F0Fu);
            sw[(lr + (LS / 8u) * h) * ST + q][c + 1u] = sx4((w >> 4u) & 0x0F0F0F0Fu);
        }
        uvec4 xv[4];
        [[unroll]] for (uint p = 0u; p < 4u; p++) {
            xv[p] = tv ? xq4[xr + k0 / 2u + p] : uvec4(0u);
        }
        barrier();
        [[unroll]] for (uint p = 0u; p < 4u; p++) {
            ivec4 wv[RT];
            [[unroll]] for (uint i = 0u; i < RT; i++) { wv[i] = ivec4(sw[(ln + 32u * i) * ST + p]); }
            // token j of the group from lane j (subgroupBroadcast takes a
            // literal lane); the accumulating form chains the four dp4a
            // (saturation never engages: |sum| < 2560 x 8 x 128)
#define TOKJ(j) {                 ivec4 x = ivec4(subgroupBroadcast(xv[p], j));                 [[unroll]] for (uint i = 0u; i < RT; i++) {                     int a = dotPacked4x8AccSatEXT(wv[i].x, x.x, acc[i * 8u + j]);                     a = dotPacked4x8AccSatEXT(wv[i].y, x.y, a);                     a = dotPacked4x8AccSatEXT(wv[i].z, x.z, a);                     acc[i * 8u + j] = dotPacked4x8AccSatEXT(wv[i].w, x.w, a);                 } }
            TOKJ(0u) TOKJ(1u) TOKJ(2u) TOKJ(3u) TOKJ(4u) TOKJ(5u) TOKJ(6u) TOKJ(7u)
        }
        barrier();
    }
    // [token][row] through shared memory, written row-contiguous
    [[unroll]] for (uint i = 0u; i < RT; i++) {
        float sc = scale(e * NROW + r0 + ln + 32u * i) * (1.0 / 256.0);
        [[unroll]] for (uint j = 0u; j < 8u; j++) {
            ep[(tg * 8u + j) * (RW + 1u) + ln + 32u * i] = float(acc[i * 8u + j]) * sc;
        }
    }
    barrier();
#ifdef GU
    for (uint q = t; q < TOK * RW; q += LS) {
        uint k = q / RW, r = q % RW;
        if (k < nt) { gb[(orow + k) * NROW + r0 + r] = ep[k * (RW + 1u) + r]; }
    }
#else
    for (uint q = t; q < TOK * RW / 2u; q += LS) {
        uint k = q / (RW / 2u), r = 2u * (q % (RW / 2u));
        if (k < nt) {
            yb[((orow + k) * NROW + r0 + r) / 2u] =
                packHalf2x16(vec2(ep[k * (RW + 1u) + r], ep[k * (RW + 1u) + r + 1u]));
        }
    }
#endif
}
