#version 450
#extension GL_EXT_integer_dot_product : require
// Tiled Q6_K x q8_K GEMM for prefill with an integer-dot (dp4a) inner loop:
// `gemm_q4k_dp4a.glsl`'s counterpart for Q6_K weights, same bindings, tile
// and grid contract as `gemm_q6k.glsl` (`GpuDevice::gemm_grid`).
//
// The 6-bit weights are unpacked at staging to signed int8 (q - 32), which
// is what `gemm_q6k.glsl` already did; there they were then converted to
// float per use. Here each 16-weight scale group is 4 `dotPacked4x8EXT` per
// (row, token), and the scale `d * sc` and the q8_K block scale are applied
// once per group. Half a super-block (128 k) per staging step.
//
// FfnDown [5120 x 17408] kernel-only (`gpubench --probe --q6k --spv`),
// seq 20 4.96 -> 2.40 ms, 64 6.21 -> 2.83, 512 42.1 -> 20.6,
// 2048 168.5 -> 82-84 ms (4.4 TFLOP/s), same maxabsdiff vs the CPU.
// Staging decodes 4 output dwords per (qh dword, 2 ql dwords): with one
// output dword per thread (2 global loads each) it was 118 ms. Folding the
// q8_K scale in per super-block through a second accumulator spilled (138).
// 16-byte activation loads and one scale-header decode per row (instead of
// one per (row, group), 8 global loads per thread): 82 -> 68.5
// ms at seq 2048. 16-byte weight loads would need a 224-byte VRAM stride; a
// timing-only probe of that gained nothing (67-69 ms), so the stride stays.
//
// Activation scale folded out of the epilogue: `acc` is held in
// units of the current q8_K block's scale `dx` and multiplied by
// `dx_prev / dx` when the block changes, then by the last `dx` at the end --
// one FMUL per (row, token) per super-block instead of one per 16-weight
// group (3 -> 2 epilogue instructions per 4 dp4a). All-zero blocks (dx = 0)
// keep the previous scale; their sums are 0. 69.1 -> 67.0 ms at seq 2048, 512
// 17.3 -> 16.8, 64 2.11 -> 2.06. Folding the rescale into the first group's
// FMA (the Q4_K form) was slower here (68.1). The remaining I2F + FFMA per
// group is ~6 ms (epilogue once per step: 60.6 ms, wrong output).
// Assumes no block scale below ~1e-34 relative to the token's output (the
// rescaled accumulator would overflow), which q8_K of real activations is not.
#define TM 4u
#define BM 128u
#define BN 64u
#define TN 8u

layout(local_size_x = 32, local_size_y = 8) in;

layout(set = 0, binding = 0, std430) readonly buffer Weights { uint weights[]; };
layout(set = 0, binding = 1, std430) readonly buffer Xin { uint xin[]; };
layout(set = 0, binding = 1, std430) readonly buffer Xin4 { uvec4 xin4[]; };
layout(set = 0, binding = 2, std430) writeonly buffer Yout { float yout[]; };
layout(set = 0, binding = 3, std140) uniform Params {
    uint out_dim;
    uint in_dim;
    uint seq;
    // Floats between consecutive tokens' rows of `yout`; 0 = `out_dim`. The
    // prefill split binds the caller's whole `[seq x out]` matrix here (at
    // its column offset) so the GPU's rows land in place, no readback copy.
    uint ld;
};

const uint QL_DW = 0u;
const uint QH_DW = 32u;
const uint SC_DW = 48u;
const uint D_DW = 52u;

shared uint wq8[32][BM + 1u];
shared uvec4 xs[32][BN / 4u + 1u];
shared float dsc[8][BM];
shared float rsx[BN];

void main() {
    uint rg = gl_LocalInvocationID.x;
    uint tg = gl_LocalInvocationID.y;
    uint tid = rg + 32u * tg;
    uint o0 = gl_WorkGroupID.x * BM;
    uint t0 = gl_WorkGroupID.y * BN;
    bool live = t0 + tg * TN < seq;
    // The q8_K scale of the last nonzero activation block of this thread's
    // token (threads tid < BN load token t0 + tid every step).
    float dprev = 0.0;

    uint nb = in_dim / 256u;
    uint sxd = (in_dim + nb * 4u) / 4u;

    float acc[TM][TN];
    for (uint m = 0u; m < TM; m++) {
        for (uint t = 0u; t < TN; t++) {
            acc[m][t] = 0.0;
        }
    }

    for (uint step = 0u; step < nb * 2u; step++) {
        uint b = step >> 1u;
        uint jj = step & 1u;

        barrier();

        // Each thread decodes 4 output dwords from one qh dword and two ql
        // dwords: ql[d] and ql[d + 8] (low / high 32 bytes of the half), both
        // nibbles, and the qh bit pairs at shifts 0/2 (low nibbles) and 4/6.
        for (uint l = 0u; l < 4u; l++) {
            uint idx = tid + l * 256u;
            uint r = idx >> 3u;
            uint d = idx & 7u;
            uint orow = o0 + r;
            uint qa = 0u, qb = 0u, qh = 0u;
            if (orow < out_dim) {
                uint wb = (orow * nb + b) * 53u;
                qa = weights[wb + QL_DW + jj * 16u + d];
                qb = weights[wb + QL_DW + jj * 16u + d + 8u];
                qh = weights[wb + QH_DW + jj * 8u + d];
            }
            for (uint k = 0u; k < 4u; k++) {
                uint hi = k >> 1u;
                uint up = k & 1u;
                uint qlw = up == 0u ? qa : qb;
                uint lo4 = (qlw >> (hi * 4u)) & 0x0F0F0F0Fu;
                uint top = ((qh >> (up * 2u + hi * 4u)) & 0x03030303u) << 4u;
                // Bytewise q - 32 as int8, q in 0..63 (a packed subtract would
                // borrow across bytes): q ^ 0x20 is q - 32 for q >= 32 and
                // q + 32 for q < 32, and the latter has bit 5 set; setting bits
                // 6..7 there subtracts 64, giving q - 32 mod 256.
                uint f = (lo4 | top) ^ 0x20202020u;
                wq8[hi * 16u + up * 8u + d][r] = (orow < out_dim) ? (f | ((f & 0x20202020u) * 6u)) : 0u;
            }
        }

        if (tid < BM) {
            uint orow = o0 + tid;
            float d = 0.0;
            uint s0 = 0u, s1 = 0u;
            if (orow < out_dim) {
                uint wb = (orow * nb + b) * 53u;
                d = unpackHalf2x16(weights[wb + D_DW]).x;
                s0 = weights[wb + SC_DW + jj * 2u];
                s1 = weights[wb + SC_DW + jj * 2u + 1u];
            }
            for (uint g = 0u; g < 8u; g++) {
                int sc = bitfieldExtract(int(g < 4u ? s0 : s1), int((g & 3u) * 8u), 8);
                dsc[g][tid] = d * float(sc);
            }
        }

        if ((sxd & 3u) == 0u) {
            for (uint l = 0u; l < 2u; l++) {
                uint idx = tid + l * 256u;
                uint q = idx & 7u;
                uint t = idx >> 3u;
                uint tgl = t0 + t;
                uvec4 v = (tgl < seq) ? xin4[(tgl * sxd + b * 64u + jj * 32u) / 4u + q] : uvec4(0u);
                xs[4u * q][t >> 2u][t & 3u] = v.x;
                xs[4u * q + 1u][t >> 2u][t & 3u] = v.y;
                xs[4u * q + 2u][t >> 2u][t & 3u] = v.z;
                xs[4u * q + 3u][t >> 2u][t & 3u] = v.w;
            }
        } else {
            for (uint l = 0u; l < 8u; l++) {
                uint idx = tid + l * 256u;
                uint kd = idx & 31u;
                uint t = idx >> 5u;
                uint tgl = t0 + t;
                xs[kd][t >> 2u][t & 3u] = (tgl < seq) ? xin[tgl * sxd + b * 64u + jj * 32u + kd] : 0u;
            }
        }
        if (tid < BN && jj == 0u) {
            uint tgl = t0 + tid;
            float dn = (tgl < seq) ? uintBitsToFloat(xin[tgl * sxd + in_dim / 4u + b]) : 0.0;
            float r = 1.0;
            if (dn != 0.0) {
                r = (dprev != 0.0) ? dprev / dn : 1.0;
                dprev = dn;
            }
            rsx[tid] = r;
        }

        barrier();

        if (live) {
            // `acc` is kept in units of the current block's activation scale
            // and rescaled when the block changes: one FMUL per (row, token)
            // per super-block instead of one per 16-weight group.
            if (jj == 0u) {
                for (uint t = 0u; t < TN; t++) {
                    float r = rsx[tg * TN + t];
                    for (uint m = 0u; m < TM; m++) {
                        acc[m][t] *= r;
                    }
                }
            }
            for (uint g = 0u; g < 8u; g++) {
                int isum[TM][TN];
                for (uint m = 0u; m < TM; m++) {
                    for (uint t = 0u; t < TN; t++) {
                        isum[m][t] = 0;
                    }
                }
                for (uint dd = 0u; dd < 4u; dd++) {
                    uint dw2 = g * 4u + dd;
                    uvec4 xa = xs[dw2][tg * 2u];
                    uvec4 xb = xs[dw2][tg * 2u + 1u];
                    for (uint m = 0u; m < TM; m++) {
                        int w = int(wq8[dw2][rg + 32u * m]);
                        isum[m][0] += dotPacked4x8EXT(w, int(xa.x));
                        isum[m][1] += dotPacked4x8EXT(w, int(xa.y));
                        isum[m][2] += dotPacked4x8EXT(w, int(xa.z));
                        isum[m][3] += dotPacked4x8EXT(w, int(xa.w));
                        isum[m][4] += dotPacked4x8EXT(w, int(xb.x));
                        isum[m][5] += dotPacked4x8EXT(w, int(xb.y));
                        isum[m][6] += dotPacked4x8EXT(w, int(xb.z));
                        isum[m][7] += dotPacked4x8EXT(w, int(xb.w));
                    }
                }
                for (uint m = 0u; m < TM; m++) {
                    float ws = dsc[g][rg + 32u * m];
                    for (uint t = 0u; t < TN; t++) {
                        acc[m][t] = fma(ws, float(isum[m][t]), acc[m][t]);
                    }
                }
            }
        }
    }

    barrier();
    if (tid < BN) {
        rsx[tid] = dprev;
    }
    barrier();

    uint ldy = (ld == 0u) ? out_dim : ld;
    for (uint m = 0u; m < TM; m++) {
        uint row = o0 + rg + 32u * m;
        if (row >= out_dim) {
            continue;
        }
        for (uint t = 0u; t < TN; t++) {
            uint tok = t0 + tg * TN + t;
            if (tok < seq) {
                yout[tok * ldy + row] = acc[m][t] * rsx[tg * TN + t];
            }
        }
    }
}
