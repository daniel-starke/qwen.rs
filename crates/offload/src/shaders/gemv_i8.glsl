#version 450
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_control_flow_attributes : require
// One-position GEMV of an int8 matrix (per-row scale) and an f32 activation
// for the Flash-Next decode's dense layers on a discrete GPU
// (`gpu/gemv.rs`): y[yoff + r] = rs[r] * sum_i w[r][i] x[xoff + i].
// `lpr` lanes take a row (16 weights each per step), a subgroup 32 / lpr
// row groups of NR, a workgroup 8 subgroups (256 threads: full occupancy on Pascal,
// where 32-thread workgroups left half the warps idle at 9-78 GB/s).
//
// Each lane-group takes NR rows that share the activation loads.
//
// Grid: x = ceil(n / (8 * 32 / lpr * NR)), with ks = 8 ceil(n / (32 / lpr * NR)).

// Up to NP positions per dispatch (`np`, a push constant; `Gemv::run_chain_n`):
// each weight load is used for every position, whose input region is
// XS floats after the previous one's and output region YS floats. The
// NR * NP sums are reduced by a reduce-scatter (the same pairs as one
// reduction per sum, so the same result): kernel alone at 8 positions
// [10240][320] 172 -> 121 us, [320][10240] 128 -> 90, [10240][2560] 714
// -> 559; with fewer sums than the row's lanes (2 positions at 32 lanes,
// and the [320][10240] at 4) the scatter now runs until a lane holds one
// sum and the plain reduction does the rest, where each sum was reduced
// on its own (per position, two barriers): 2 / 4 positions [320][10240]
// 42 / 66 -> 38 / 55 us, [10240][320] 54 -> 46, [16384][2560] 347 / 624
// -> 315 / 540. Every position's input in shared memory (and each weight
// converted once) gained only on short rows and cost the long ones
// registers: [10240][320] 124 -> 94, [10240][2560] 559 -> 594.

#define NR 4u
// `gemv_i8_1.spv` is this kernel built with -DNP1, for runs of one
// position (the loop over 8 cost a one-position decode step 36 -> 42 ms);
// `gemv_i8_2.spv` / `_4.spv` with -DNP2 / -DNP4 for runs of 2 / 3-4 (verify
// steps: 8 positions' registers and reduce-scatter cost a step of 2 the
// hyper-connection GEMVs at 1.6-1.9x their one-position time).
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
#define PER 16u
layout(local_size_x = 256) in;

layout(set = 0, binding = 0, std430) readonly buffer Weights { uvec4 w4[]; };
layout(set = 0, binding = 1, std430) readonly buffer Xin { vec4 x4[]; };
layout(set = 0, binding = 2, std430) writeonly buffer Yout { float yout[]; };
layout(set = 0, binding = 3, std430) readonly buffer Scale { float rs[]; };
// the input region again, for the SiLU epilogue (`act`; rows past `xoff`
// + k, never read by this dispatch)
layout(set = 0, binding = 4, std430) writeonly buffer Xout { float xout[]; };
layout(set = 0, binding = 8, std140) uniform Params {
    uint n;
    uint k;
    uint xoff;      // in floats, a multiple of 16
    uint yoff;
    uint lpr;       // lanes per row: a power of two <= 32
    uint ks;        // 1, or 8: the subgroups split k (few rows, long rows)
    // != 0: also xout[actdst + p XS + r] = silu(acta y) of every output y
    // (`gemv_act.glsl` mode 0, the step it replaces: `Gemv::set_silu`)
    uint act;
    float acta;
    uint actdst;
};
layout(push_constant) uniform Pc { uint np; };

shared float part[8][32 * NR];
#ifndef NP1
// k-split partial sums at lpr >= 4: (32 / lpr) row groups x NR x NP
shared float pall[8][8u * NR * NP];
#endif

float sigm(float v) { return 1.0 / (1.0 + exp(-v)); }

// position p's output row `row`
void put(uint p, uint row, float v) {
    yout[yoff + p * YS + row] = v;
    if (act != 0u) {
        float t = acta * v;
        xout[actdst + p * XS + row] = t * sigm(t);
    }
}

vec4 b4(uint u) {
    return vec4(ivec4(int(u << 24u) >> 24, int(u << 16u) >> 24, int(u << 8u) >> 24, int(u) >> 24));
}
#define DOT dot(b4(q.x), x0) + dot(b4(q.y), x1) + dot(b4(q.z), x2) + dot(b4(q.w), x3)

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
            vec4 x0 = x4[xb + 4u * i + 0u];
            vec4 x1 = x4[xb + 4u * i + 1u];
            vec4 x2 = x4[xb + 4u * i + 2u];
            vec4 x3 = x4[xb + 4u * i + 3u];
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
                uint xp = xb + p * (XS / 4u) + 4u * i;
                vec4 x0 = x4[xp + 0u];
                vec4 x1 = x4[xp + 1u];
                vec4 x2 = x4[xp + 2u];
                vec4 x3 = x4[xp + 3u];
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
        // reduce-scatter over the row's lpr lanes: each step halves the
        // values a lane holds, 31 shuffles in all where one reduction per
        // value took NR * NP log2(lpr) (up to 160 at 8 positions); with
        // fewer values than lanes (NR NP < lpr) the steps left once a lane
        // holds one value add it up as one reduction does. The same pairs
        // as one reduction per value, so the same sums.
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
                if (p < np && row0 + r < n) { put(p, row0 + r, acc[j] * rs[row0 + r]); }
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
                put(p, row, a * rs[row]);
            }
        }
        return;
    }
#endif
    [[unroll]] for (uint p = 0u; p < NP; p++) {
        if (p >= np) { continue; }
        [[unroll]] for (uint r = 0u; r < NR; r++) {
            float a = acc[p * NR + r];
            for (uint o = lpr / 2u; o > 0u; o >>= 1u) { a += subgroupShuffleXor(a, o); }
            if (ks == 1u) {
                if (sl == 0u && row0 + r < n) { put(p, row0 + r, a * rs[row0 + r]); }
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
                put(p, row, a * rs[row]);
            }
            barrier();
        }
    }
}
