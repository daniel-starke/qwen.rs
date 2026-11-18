#version 450
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_control_flow_attributes : require
// `gemv_i8.glsl` for an f16 matrix (the routers, the injection vectors):
// y[yoff + r] = rs[r] * sum_i w[r][i] x[xoff + i], 8 weights per lane and
// step.
//
// Each lane-group takes NR rows that share the activation loads.
//
// Grid: x = ceil(n / (8 * 32 / lpr * NR)), with ks = 8 ceil(n / (32 / lpr * NR)).

// Up to NP positions per dispatch (`np`, a push constant; `Gemv::run_chain_n`):
// each weight load is used for every position, whose input region is
// XS floats after the previous one's and output region YS floats. The
// sums are reduced as in `gemv_i8.glsl` (reduce-scatter, the same pairs):
// kernel alone [512][2560] at 2 / 4 / 8 positions 35 / 46 / 84 -> 30 / 38
// / 62 us (it reduced each sum on its own).

#define NR 4u
// `gemv_f16_1.spv` is this kernel built with -DNP1, for runs of one
// position (the loop over 8 cost a one-position decode step 36 -> 42 ms);
// `_2.spv` / `_4.spv` with -DNP2 / -DNP4 for runs of 2 / 3-4.
#ifdef NP1
#define NP 1u
#elif defined(NP2)
#define NP 2u
#elif defined(NP4)
#define NP 4u
#else
#define NP 8u
#endif
#define XS 65536u
#define YS 262144u
#define PER 8u
layout(local_size_x = 256) in;

layout(set = 0, binding = 0, std430) readonly buffer Weights { uvec4 w4[]; };
layout(set = 0, binding = 1, std430) readonly buffer Xin { vec4 x4[]; };
layout(set = 0, binding = 2, std430) writeonly buffer Yout { float yout[]; };
layout(set = 0, binding = 3, std430) readonly buffer Scale { float rs[]; };
layout(set = 0, binding = 8, std140) uniform Params {
    uint n;
    uint k;
    uint xoff;      // in floats, a multiple of 16
    uint yoff;
    uint lpr;       // lanes per row: a power of two <= 32
    uint ks;        // 1, or 8: the subgroups split k (few rows, long rows)
};
layout(push_constant) uniform Pc { uint np; };

shared float part[8][32 * NR];
#ifndef NP1
// k-split partial sums at lpr >= 4: (32 / lpr) row groups x NR x NP
shared float pall[8][8u * NR * NP];
#endif

#define DOT dot(vec4(unpackHalf2x16(q.x), unpackHalf2x16(q.y)), x0)           + dot(vec4(unpackHalf2x16(q.z), unpackHalf2x16(q.w)), x1)

void main() {
    uint lane = gl_SubgroupInvocationID;
    uint sl = lane & (lpr - 1u);
    uint rpw = 32u / lpr;
    uint sg = gl_SubgroupID;
    uint grp = ks == 1u ? gl_WorkGroupID.x * 8u + sg : gl_WorkGroupID.x;
    uint row0 = (grp * rpw + lane / lpr) * NR;
    uint i0 = ks == 1u ? sl : sg * lpr + sl;
    uint nq = k / PER;
    uint xb = xoff / 4u;
    float acc[NR * NP];
    [[unroll]] for (uint r = 0u; r < NR * NP; r++) { acc[r] = 0.0; }
    if (row0 < n) {
        for (uint i = i0; i < nq; i += lpr * ks) {
#ifdef NP1
            vec4 x0 = x4[xb + 2u * i + 0u];
            vec4 x1 = x4[xb + 2u * i + 1u];
            for (uint r = 0u; r < NR; r++) {
                if (row0 + r < n) {
                    uvec4 q = w4[(row0 + r) * nq + i];
                    acc[r] += DOT;
                }
            }
#else
            uvec4 qs[NR];
            [[unroll]] for (uint r = 0u; r < NR; r++) {
                qs[r] = row0 + r < n ? w4[(row0 + r) * nq + i] : uvec4(0u);
            }
            [[unroll]] for (uint p = 0u; p < NP; p++) {
                if (p >= np) { continue; }
                uint xp = xb + p * (XS / 4u) + 2u * i;
                vec4 x0 = x4[xp + 0u];
                vec4 x1 = x4[xp + 1u];
                [[unroll]] for (uint r = 0u; r < NR; r++) {
                    uvec4 q = qs[r];
                    acc[p * NR + r] += DOT;
                }
            }
#endif
        }
    }
#ifndef NP1
    if (ks == 1u || lpr >= 4u) {
        // reduce-scatter: `gemv_i8.glsl` (the same sums as the one
        // reduction per value below)
        uint idx = 0u;
        [[unroll]] for (uint s = 0u; s < 5u; s++) {
            uint o = lpr >> (s + 1u);
            if (o == 0u) { break; }
            uint h = (NR * NP) >> (s + 1u);
            if (h == 0u) {
                acc[0] += subgroupShuffleXor(acc[0], o);
                continue;
            }
            bool up = (sl & o) != 0u;
            [[unroll]] for (uint j = 0u; j < h; j++) {
                float a = up ? acc[j] : acc[j + h];
                acc[j] = (up ? acc[j + h] : acc[j]) + subgroupShuffleXor(a, o);
            }
            if (up) { idx += h; }
        }
        // values per lane, and whether this lane writes them (with fewer
        // values than lanes, the lanes of a value all hold its sum)
        uint cnt = max((NR * NP) / lpr, 1u);
        bool wr = NR * NP >= lpr || (sl & (lpr / (NR * NP) - 1u)) == 0u;
        if (ks == 1u) {
            [[unroll]] for (uint j = 0u; j < NR * NP; j++) {
                if (j >= cnt || !wr) { break; }
                uint p = (idx + j) / NR, r = (idx + j) % NR;
                if (p < np && row0 + r < n) { yout[yoff + p * YS + row0 + r] = acc[j] * rs[row0 + r]; }
            }
            return;
        }
        // the subgroups' partial sums of every (row group, position, row)
        [[unroll]] for (uint j = 0u; j < NR * NP; j++) {
            if (j >= cnt || !wr) { break; }
            pall[sg][(lane / lpr) * NR * NP + idx + j] = acc[j];
        }
        barrier();
        for (uint j = gl_LocalInvocationID.x; j < rpw * NR * NP; j += 256u) {
            uint g0 = j / (NR * NP), e = j % (NR * NP);
            uint p = e / NR, row = gl_WorkGroupID.x * rpw * NR + g0 * NR + e % NR;
            if (p < np && row < n) {
                float a = 0.0;
                for (uint g = 0u; g < 8u; g++) { a += pall[g][j]; }
                yout[yoff + p * YS + row] = a * rs[row];
            }
        }
        return;
    }
#endif
    [[unroll]] for (uint p = 0u; p < NP; p++) {
        if (p >= np) { continue; }
        uint yp = yoff + p * YS;
        [[unroll]] for (uint r = 0u; r < NR; r++) {
            float a = acc[p * NR + r];
            for (uint o = lpr / 2u; o > 0u; o >>= 1u) { a += subgroupShuffleXor(a, o); }
            if (ks == 1u) {
                if (sl == 0u && row0 + r < n) { yout[yp + row0 + r] = a * rs[row0 + r]; }
            } else if (sl == 0u) {
                part[sg][(lane / lpr) * NR + r] = a;
            }
        }
        if (ks != 1u) {
            barrier();
            uint j = gl_LocalInvocationID.x;
            uint row = gl_WorkGroupID.x * rpw * NR + j;
            if (j < rpw * NR && row < n) {
                float a = 0.0;
                for (uint g = 0u; g < 8u; g++) { a += part[g][j]; }
                yout[yp + row] = a * rs[row];
            }
            barrier();
        }
    }
}
