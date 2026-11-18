#version 450
#extension GL_EXT_integer_dot_product : require
// Tiled Q4_K x q8_K GEMM for prefill with an integer-dot (dp4a) inner loop.
// Same bindings, tile (BM = 128 rows x BN = 64 tokens) and grid contract as
// `gemm_q4k.glsl` (`GpuDevice::gemm_grid`); selected instead of it when the
// device passed the dp4a self-check (`GpuDevice::gemm_pipeline_for`).
//
// WHY
// ---
// `gemm_q4k.glsl` converts every nibble and activation to float and spends
// one FFMA per multiply-add, at 84-87 % of the card's FP32 peak. The products
// are small integers (nibble 0..15 x q8 -128..127), so `dotPacked4x8EXT`
// (OpSDotKHR, dp4a on Pascal) does four of them per instruction, exactly; the
// weight scale `d * sc` and the min term `dmin * m * sum(q8)` are applied once
// per 32-weight sub-block, the q8_K block scale once per sub-block too.
// FfnUp [17408 x 5120] kernel-only (`gpubench --probe --spv`):
//
//     seq    float GEMM          this kernel
//     20     4.33 ms              2.29 ms
//     64     6.47 ms              2.65 ms
//     512    41.4 ms (2.20 TF/s)  20.6 ms (4.43 TF/s)
//     2048   166.8 ms (2.19)      82.3 ms (4.44)
//
// with the same maxabsdiff against the CPU (0.001465 at seq 2048).
//
// SHAPE NOTES
// -----------
// One whole Q4_K super-block (256 k) per staging step: its header is decoded
// once (8 sub-block scales per row) instead of once per 64-k step, and there
// are 3 barriers per 256 k instead of 12; 82 vs 95-102 ms with the 64-k
// step. ~44 KB of shared memory, 2 workgroups per SM. TM = 8 spilled (335
// ms), as did folding the q8_K scale in once per super-block through a second
// accumulator array (137 ms).
//
// Staging loads are 16-byte (`uvec4`): 81-86 -> 58.3 ms at seq
// 2048 (seq 20 1.87 -> 1.30, 64 2.61 -> 1.89, 512 20.6 -> 14.7). Staging
// alone measured 44.7 ms and the dot products alone 50 ms, yet together they
// took 86: with 16 warps per SM the 32 scalar loads per thread per step were
// the bottleneck, not the dp4a. A register prefetch of the next block on top
// of this spilled (92 ms). Weight rows are 144 bytes, so always aligned; the
// activation row (`sxd` dwords) is aligned when the block count is a multiple
// of 4 (every shape in this model), else the scalar loop below runs.
//
// Half a super-block (128 k) per staging step with the nibbles split at
// staging: 58.6 -> 55.7 ms at seq 2048 (seq 20 1.61 -> 1.45, 64
// 1.94-2.33 -> 1.80, 512 14.8 -> 14.1). The kernel is instruction-issue
// bound (a bare dp4a loop on this card does the FfnUp dp4a count in ~33 ms)
// and each weight word in `wq` is read by all 8 token groups, so the
// shift-and-mask that picked its nibble ran 8 times per word per workgroup.
// Splitting once at staging needs twice the words per sub-block, which only
// fits shared memory at half a block per step (~34 KB now). Measured and not
// kept on top of it: sub-block sums from a subgroup shuffle at staging
// instead of the `msum` pass (one barrier fewer, 56.6 ms), the I2F replaced
// by a float-bias trick (56.6 ms; also flat on the old kernel), and an
// XOR-swizzled `xs` against its 8-way staging-store bank conflict (flat).
//
// Activation scale folded out of the epilogue: `acc` is held in
// units of the current q8_K block's scale `dx` and multiplied by
// `dx_prev / dx` when the block changes (inside the first sub-block's FMA),
// then by the last `dx` at the end; one FMUL per (row, token) per
// super-block instead of one per sub-block. All-zero blocks (dx = 0) keep the
// previous scale; their sums are 0. 56.3 -> 55.2 ms at seq 2048, 512 14.1 ->
// 13.9. As a separate rescale loop at the block change it spilled (91 ms).
// Assumes no block scale below ~1e-34 relative to the token's output (the
// rescaled accumulator would overflow), which q8_K of real activations is not.
#define TM 4u
#define BM 128u
#define BN 64u
#define TN 8u

layout(local_size_x = 32, local_size_y = 8) in;

layout(set = 0, binding = 0, std430) readonly buffer Weights4 { uvec4 weights4[]; };
layout(set = 0, binding = 1, std430) readonly buffer Xin { uint xin[]; };
layout(set = 0, binding = 1, std430) readonly buffer Xin4 { uvec4 xin4[]; };
layout(set = 0, binding = 2, std430) writeonly buffer Yout { float yout[]; };
layout(set = 0, binding = 3, std140) uniform Params {
    uint out_dim;
    uint in_dim;
    uint seq;
    uint ld;
};

// Half a super-block (128 k, 4 sub-blocks) per staging step, nibbles split
// at staging: wq[sl * 8 + dwi][row] holds local sub-block sl, 4 weights as bytes.
shared uint wq[32][BM + 1u];
shared uvec4 xs[32][BN / 4u + 1u];
shared float dsc[8][BM];
shared float dmm[8][BM];
shared float msum[4][BN];
shared float rsx[BN];

uint pick(uint w0, uint w1, uint w2, uint k) {
    return (k == 0u) ? w0 : ((k == 1u) ? w1 : w2);
}

void main() {
    uint rg = gl_LocalInvocationID.x;
    uint tg = gl_LocalInvocationID.y;
    uint tid = rg + 32u * tg;
    uint o0 = gl_WorkGroupID.x * BM;
    uint t0 = gl_WorkGroupID.y * BN;
    bool live = t0 + tg * TN < seq;
    // The q8_K scale of the last nonzero activation block of this thread's
    // token (threads tid >= 192 load token t0 + tid - 192 every block).
    float dprev = 0.0;

    uint nb = in_dim / 256u;
    uint sxd = (in_dim + nb * 4u) / 4u;

    float acc[TM][TN];
    for (uint m = 0u; m < TM; m++) {
        for (uint t = 0u; t < TN; t++) {
            acc[m][t] = 0.0;
        }
    }

    for (uint b = 0u; b < nb; b++) {
      for (uint hb = 0u; hb < 2u; hb++) {
        barrier();

        // This half's qs: 4 uvec4 per row (qs uvec4 4*hb .. 4*hb+3).
        for (uint l = 0u; l < 2u; l++) {
            uint idx = tid + l * 256u;
            uint r = idx >> 2u;
            uint q = idx & 3u;
            uint orow = o0 + r;
            uvec4 v = (orow < out_dim) ? weights4[(orow * nb + b) * 9u + 1u + 4u * hb + q] : uvec4(0u);
            // word w = 4q + c of the half; pair jj = w >> 3, dwi = w & 7.
            uint jj = q >> 1u;
            uint d0 = (q & 1u) * 4u;
            uint lo0 = (2u * jj) * 8u + d0;
            uint hi0 = lo0 + 8u;
            wq[lo0][r] = v.x & 0x0F0F0F0Fu;
            wq[lo0 + 1u][r] = v.y & 0x0F0F0F0Fu;
            wq[lo0 + 2u][r] = v.z & 0x0F0F0F0Fu;
            wq[lo0 + 3u][r] = v.w & 0x0F0F0F0Fu;
            wq[hi0][r] = (v.x >> 4u) & 0x0F0F0F0Fu;
            wq[hi0 + 1u][r] = (v.y >> 4u) & 0x0F0F0F0Fu;
            wq[hi0 + 2u][r] = (v.z >> 4u) & 0x0F0F0F0Fu;
            wq[hi0 + 3u][r] = (v.w >> 4u) & 0x0F0F0F0Fu;
        }
        if (hb == 0u && tid < BM) {
            uint orow = o0 + tid;
            float d = 0.0, dmn = 0.0;
            uint sw0 = 0u, sw1 = 0u, sw2 = 0u;
            if (orow < out_dim) {
                uvec4 hd = weights4[(orow * nb + b) * 9u];
                vec2 dm = unpackHalf2x16(hd.x);
                d = dm.x;
                dmn = dm.y;
                sw0 = hd.y;
                sw1 = hd.z;
                sw2 = hd.w;
            }
            for (uint sb = 0u; sb < 8u; sb++) {
                bool lo = sb < 4u;
                uint ia = lo ? sb : sb - 4u, ib = lo ? sb + 4u : sb, ic = sb + 4u;
                uint a = bitfieldExtract(pick(sw0, sw1, sw2, ia >> 2u), int((ia & 3u) * 8u), 8);
                uint bb = bitfieldExtract(pick(sw0, sw1, sw2, ib >> 2u), int((ib & 3u) * 8u), 8);
                uint c = bitfieldExtract(pick(sw0, sw1, sw2, ic >> 2u), int((ic & 3u) * 8u), 8);
                float sc = float(lo ? (a & 63u) : ((c & 0xFu) | ((a >> 6u) << 4u)));
                float mm = float(lo ? (bb & 63u) : ((c >> 4u) | ((bb >> 6u) << 4u)));
                dsc[sb][tid] = d * sc;
                dmm[sb][tid] = -dmn * mm;
            }
        }
        if ((sxd & 3u) == 0u) {
            for (uint l = 0u; l < 2u; l++) {
                uint idx = tid + l * 256u;
                uint q = idx & 7u;
                uint t = idx >> 3u;
                uint tgl = t0 + t;
                uvec4 v = (tgl < seq) ? xin4[(tgl * sxd + b * 64u + hb * 32u) / 4u + q] : uvec4(0u);
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
                xs[kd][t >> 2u][t & 3u] = (tgl < seq) ? xin[tgl * sxd + b * 64u + hb * 32u + kd] : 0u;
            }
        }
        if (hb == 0u && tid >= 192u) {
            uint t = tid - 192u;
            uint tgl = t0 + t;
            float dn = (tgl < seq) ? uintBitsToFloat(xin[tgl * sxd + in_dim / 4u + b]) : 0.0;
            float r = 1.0;
            if (dn != 0.0) {
                r = (dprev != 0.0) ? dprev / dn : 1.0;
                dprev = dn;
            }
            rsx[t] = r;
        }

        barrier();

        {
            uint sl = tid / BN;
            uint t = tid % BN;
            int q = 0;
            for (uint i = 0u; i < 8u; i++) {
                q += dotPacked4x8EXT(int(xs[sl * 8u + i][t >> 2u][t & 3u]), 0x01010101);
            }
            msum[sl][t] = float(q);
        }

        barrier();

        if (live) {
            float rr[TN];
            for (uint t = 0u; t < TN; t++) {
                rr[t] = (hb == 0u) ? rsx[tg * TN + t] : 1.0;
            }
            for (uint sl = 0u; sl < 4u; sl++) {
                uint sb = 4u * hb + sl;
                int isum[TM][TN];
                for (uint m = 0u; m < TM; m++) {
                    for (uint t = 0u; t < TN; t++) {
                        isum[m][t] = 0;
                    }
                }
                for (uint dwi = 0u; dwi < 8u; dwi++) {
                    uvec4 xa = xs[sl * 8u + dwi][tg * 2u];
                    uvec4 xb = xs[sl * 8u + dwi][tg * 2u + 1u];
                    for (uint m = 0u; m < TM; m++) {
                        int w = int(wq[sl * 8u + dwi][rg + 32u * m]);
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
                    uint r = rg + 32u * m;
                    float ws = dsc[sb][r];
                    float wm = dmm[sb][r];
                    for (uint t = 0u; t < TN; t++) {
                        acc[m][t] = fma(ws, float(isum[m][t]), (sl == 0u) ? acc[m][t] * rr[t] : acc[m][t]);
                        acc[m][t] = fma(wm, msum[sl][tg * TN + t], acc[m][t]);
                    }
                }
            }
        }
      }
    }

    barrier();
    if (tid >= 192u) {
        rsx[tid - 192u] = dprev;
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
