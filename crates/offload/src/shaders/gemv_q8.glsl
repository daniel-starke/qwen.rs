#version 450
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_integer_dot_product : require
// `gemv_i8.glsl` for an input whose values are integers in [-128, 127]
// (a quantized matrix's rounded input, `GemvMat::Q8`). The workgroup packs
// the input to int8 in shared memory, TC 16-element chunks of every
// position at a time (v + 1.5 * 2^23 puts the two's-complement byte in the
// mantissa's low bits), and each weight load costs 4 `dp4a` per position.
// `gemv_i8.glsl` reads 64 bytes of float input per position for every 64
// weight bytes, which bound its 8-position verify runs; here it is 16 bytes
// from shared memory: 8 positions on [10240][2560] 821 -> 373 us, on
// [2560][6144] 475 -> 229. The sum is exact in int32 (the float kernel's
// is too while partial sums stay below 2^24). At one position this kernel
// is no faster, so `Gemv` runs `gemv_i8_1.spv` there. The sums are
// reduced as in `gemv_i8.glsl` (reduce-scatter): kernel alone at 8
// positions [10240][2560] 296 -> 195 us, [10240][320] 85 -> 41; also with
// fewer sums than lanes (2-4 positions): [16480][2560] at 4 positions 318
// -> 274 us, [2560][6144] 117 -> 105, [2560][640] 27 -> 17.
//
// Grid and parameters: `gemv_i8.glsl`.

#define NR 4u
// `gemv_q8_2.spv` / `_4.spv`: built with -DNP2 / -DNP4, for runs of 2 / 3-4
// positions (`gemv_i8.glsl`)
#if defined(NP2)
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

layout(set = 0, binding = 0, std430) readonly buffer Weights { ivec4 w4[]; };
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

shared int part[8][32 * NR];
shared int pall[8][8u * NR * NP];
// TC chunks of 16 inputs per tile (32 / 64 / 128 / 256: 377 / 379 / 410 /
// 471 us at 8 positions on [10240][2560]); lanes past it skip a tile
#define TC 64u
shared ivec4 xs[NP][TC];

int pk(vec4 v) {
    uvec4 u = floatBitsToUint(v + 12582912.0) & 255u;
    return int(u.x | (u.y << 8u) | (u.z << 16u) | (u.w << 24u));
}

#define DOT(q, a) (dotPacked4x8EXT(q.x, a.x) + dotPacked4x8EXT(q.y, a.y) \
    + dotPacked4x8EXT(q.z, a.z) + dotPacked4x8EXT(q.w, a.w))

void main() {
    uint lane = gl_SubgroupInvocationID;
    uint sl = lane & (lpr - 1u);
    uint rpw = 32u / lpr;
    uint sg = gl_SubgroupID;
    uint lid = gl_LocalInvocationID.x;
    uint grp = ks == 1u ? gl_WorkGroupID.x * 8u + sg : gl_WorkGroupID.x;
    uint row0 = (grp * rpw + lane / lpr) * NR;
    uint i0 = ks == 1u ? sl : sg * lpr + sl;
    uint nq = k / PER;
    uint xb = xoff / 4u;
    int acc[NR * NP];
    [[unroll]] for (uint r = 0u; r < NR * NP; r++) { acc[r] = 0; }
    for (uint t0 = 0u; t0 < nq; t0 += TC) {
        uint tc = min(TC, nq - t0);
        barrier();
        for (uint j = lid; j < np * tc; j += 256u) {
            uint p = j / tc, c = j - p * tc;
            uint xp = xb + p * (XS / 4u) + 4u * (t0 + c);
            xs[p][c] = ivec4(pk(x4[xp]), pk(x4[xp + 1u]), pk(x4[xp + 2u]), pk(x4[xp + 3u]));
        }
        barrier();
        if (row0 < n) {
            for (uint c = i0; c < tc; c += lpr * ks) {
                ivec4 qs[NR];
                [[unroll]] for (uint r = 0u; r < NR; r++) {
                    qs[r] = row0 + r < n ? w4[(row0 + r) * nq + t0 + c] : ivec4(0);
                }
                [[unroll]] for (uint p = 0u; p < NP; p++) {
                    if (p >= np) { continue; }
                    ivec4 a = xs[p][c];
                    [[unroll]] for (uint r = 0u; r < NR; r++) {
                        acc[p * NR + r] += DOT(qs[r], a);
                    }
                }
            }
        }
    }
    if (ks == 1u || lpr >= 4u) {
        // reduce-scatter: `gemv_i8.glsl`
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
                int a = up ? acc[j] : acc[j + h];
                acc[j] = (up ? acc[j + h] : acc[j]) + subgroupShuffleXor(a, o);
            }
            if (up) { idx += h; }
        }
        uint cnt = max((NR * NP) / lpr, 1u);
        bool wr = NR * NP >= lpr || (sl & (lpr / (NR * NP) - 1u)) == 0u;
        if (ks == 1u) {
            [[unroll]] for (uint j = 0u; j < NR * NP; j++) {
                if (j >= cnt || !wr) { break; }
                uint p = (idx + j) / NR, r = (idx + j) % NR;
                if (p < np && row0 + r < n) { yout[yoff + p * YS + row0 + r] = float(acc[j]) * rs[row0 + r]; }
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
                int a = 0;
                for (uint g = 0u; g < 8u; g++) { a += pall[g][j]; }
                yout[yoff + p * YS + row] = float(a) * rs[row];
            }
        }
        return;
    }
    [[unroll]] for (uint p = 0u; p < NP; p++) {
        if (p >= np) { continue; }
        uint yp = yoff + p * YS;
        [[unroll]] for (uint r = 0u; r < NR; r++) {
            int a = acc[p * NR + r];
            for (uint o = lpr / 2u; o > 0u; o >>= 1u) { a += subgroupShuffleXor(a, o); }
            if (ks == 1u) {
                if (sl == 0u && row0 + r < n) { yout[yp + row0 + r] = float(a) * rs[row0 + r]; }
            } else if (sl == 0u) {
                part[sg][(lane / lpr) * NR + r] = a;
            }
        }
        if (ks != 1u) {
            barrier();
            uint j = gl_LocalInvocationID.x;
            uint row = gl_WorkGroupID.x * rpw * NR + j;
            if (j < rpw * NR && row < n) {
                int a = 0;
                for (uint g = 0u; g < 8u; g++) { a += part[g][j]; }
                yout[yp + row] = float(a) * rs[row];
            }
            barrier();
        }
    }
}
