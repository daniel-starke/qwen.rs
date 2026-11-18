#version 450
// Tiled Q4_K x q8_K GEMM for prefill.
//
// WHY THIS EXISTS
// ---------------
// `q4k.glsl` maps the sequence position to `gl_WorkGroupID.y`, so every
// (row, position) pair re-reads the weight row from VRAM. That is a *batched
// GEMV*: the weight is fetched `seq` times instead of once, and measured on the
// GTX 1060 it holds ~1.37 ms/token flat from seq=1 to seq=2048 (13 % of the
// CPU's rate); no reuse at all. This kernel gives a workgroup a real 2-D tile
// so a weight byte fetched once is used for BN tokens.
//
// WHY FLOAT AND NOT DP4A / INT32
// ------------------------------
// Measured on this card with `gpubench --gate` (`shaders/alu_probe.glsl`):
//
//     fp32 FFMA          3.2-3.9 TFLOP/s
//     int32 MAD          0.59-0.71 G MAD/s  (~1.2-1.4 TFLOP/s equivalent)
//     dp4a (OpSDotKHR)   not measured here. Recorded as broken in silicon
//                        earlier; that was a nibble-order bug in
//                        `dp4a_q4k.glsl`, and dp4a works on this card.
//
// So the int32 path tops out near 1.4x the CPU, which is not worth streaming
// weights over PCIe for. The float path is ~3.4x the CPU and is what this
// kernel uses; a dp4a GEMM has not been tried.
//
// Accumulating the *integer products* in f32 is not a precision compromise: the
// nibble is 0..15 and the q8 activation is -128..127, so each product is exact
// and a 32-term sub-block sum peaks at ~61k; far inside f32's exact-integer
// range of 2^24. (The `ds` activation scale is folded in during staging, which
// costs one rounding per staged value, ~1e-7 relative.)
//
// TILE SHAPE
// ----------
//     BM = 128 output rows   BN = 64 tokens   BK = 64 k per staging step
//     256 threads = 32 row groups x 8 token groups
//     TM = 4 rows/thread     TN = 8 tokens/thread
//
// TN = 8 is chosen from the probe's amortization curve: the 2-op nibble unpack
// is spread over TN FMAs, and TN=8 measured 84-87 % of the card's ALU peak
// (TN=4 only 59-67 %, TN=16 would want twice the registers).
//
// BK = 64 is one Q4_K *sub-block pair*: sub-blocks 2j and 2j+1 share the same
// 32 contiguous `qs` bytes (low nibbles are 2j, high nibbles are 2j+1), so each
// row's staging read is one 32-byte transaction and no byte is fetched twice.
//
// Weight traffic is `W * seq/BN` and activation traffic `A * out_dim/BM`; at the
// FFN shape [17408x5120] with seq=2048 that is 2.0 GB + 1.4 GB against ~110 ms
// of arithmetic, i.e. ~20 %: the tile is wide enough that this is not the
// limiter.
//
// GRID CONTRACT (read before changing any constant here)
// ------------------------------------------------------
// Dispatch `ceil(out_dim/BM)` x `ceil(seq/BN)` workgroups. This kernel does NOT
// go through `GpuDevice::grid_x` / `rows_per_wg`; those describe the GEMV
// kernels, whose contract is rows-per-workgroup in x and *position* in y. Here y
// is a token *tile*. `GpuDevice::gemm_grid` is the single source of truth for
// this one. A mismatch is silent: the dispatch computes part of the output and
// leaves the rest stale (a 300 GB/s "result" was measured from exactly that).

#define BM 128u
#define BN 64u
#define BK 64u
#define TM 4u
#define TN 8u

layout(local_size_x = 32, local_size_y = 8) in;

layout(set = 0, binding = 0, std430) readonly buffer Weights { uint weights[]; };
layout(set = 0, binding = 1, std430) readonly buffer Xin { uint xin[]; };
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

// `[8][BM+1]`, not `[8][BM]`: the staging loop writes with the dword index
// varying fastest (that is what makes the *global* read coalesced), so the
// shared stride must be odd in dwords or all 32 lanes hit one bank.
shared uint wq[8][BM + 1u];
// Token-contiguous so the inner loop reads a thread's 8 tokens as two `vec4`s,
// and every lane of a warp shares the same token group -> broadcast, no conflict.
shared vec4 af[BK][BN / 4u];
shared float dsc[2][BM];
shared float dmm[2][BM];
shared float msum[2][BN];

// Select one of the block's three packed-scale dwords without a dynamically
// indexed array (which the driver would back with scratch memory).
uint pick(uint w0, uint w1, uint w2, uint k) {
    return (k == 0u) ? w0 : ((k == 1u) ? w1 : w2);
}

void main() {
    uint rg = gl_LocalInvocationID.x;              // 0..31, row group
    uint tg = gl_LocalInvocationID.y;              // 0..7, token group
    uint tid = rg + 32u * tg;                      // 0..255
    uint o0 = gl_WorkGroupID.x * BM;               // first output row of the tile
    uint t0 = gl_WorkGroupID.y * BN;               // first token of the tile
    // A warp is one token group (`local_size_x` = 32 = the row groups), so
    // this is warp-uniform: in a partial tile (a short prefill chunk, or the
    // tail of a long one) the warps whose 8 tokens are all padding skip the
    // FMAs and leave the SM to the live ones. They still stage and hit every
    // barrier. A 20-token chunk costs 3 of 8 warps' arithmetic, not 8.
    bool live = t0 + tg * TN < seq;

    uint nb = in_dim / 256u;
    uint sx = in_dim + nb * 4u;                    // q8_K per-token stride, bytes
    uint sxd = sx / 4u;                            // ... in dwords (in_dim % 256 == 0)

    float acc[TM][TN];
    for (uint m = 0u; m < TM; m++) {
        for (uint t = 0u; t < TN; t++) {
            acc[m][t] = 0.0;
        }
    }

    // One step per Q4_K sub-block pair: `nb` super-blocks x 4 pairs each.
    for (uint step = 0u; step < nb * 4u; step++) {
        uint b = step >> 2u;                       // super-block
        uint j = step & 3u;                        // sub-block pair within it

        barrier();                                 // previous iteration's reads are done

        // ---- stage weights: 128 rows x 8 dwords ----------------------------
        // `dw` varies fastest across `tid`, so 8 consecutive threads read the 32
        // contiguous bytes of one row and a warp covers 4 rows in 4 transactions.
        for (uint l = 0u; l < 4u; l++) {
            uint idx = tid + l * 256u;             // 0..1023
            uint r = idx >> 3u;
            uint dw = idx & 7u;
            uint orow = o0 + r;
            wq[dw][r] = (orow < out_dim)
                ? weights[(orow * nb + b) * 36u + 4u + j * 8u + dw]
                : 0u;
        }

        // ---- stage the two sub-blocks' scales, one per thread ---------------
        {
            uint r = tid & 127u;
            uint s = tid >> 7u;
            uint sb = 2u * j + s;                  // sub-block 0..7 within the super-block
            uint orow = o0 + r;
            float sc = 0.0, mm = 0.0, d = 0.0, dmn = 0.0;
            if (orow < out_dim) {
                uint base = (orow * nb + b) * 36u;
                vec2 dm = unpackHalf2x16(weights[base]);
                d = dm.x;
                dmn = dm.y;
                uint sw0 = weights[base + 1u], sw1 = weights[base + 2u], sw2 = weights[base + 3u];
                // `get_scale_min_k4`, same two forms as the CPU
                // `unpack_q4_scales` and as `q4k.glsl`.
                bool lo = sb < 4u;
                uint ia = lo ? sb : sb - 4u, ib = lo ? sb + 4u : sb, ic = sb + 4u;
                uint a = bitfieldExtract(pick(sw0, sw1, sw2, ia >> 2u), int((ia & 3u) * 8u), 8);
                uint bb = bitfieldExtract(pick(sw0, sw1, sw2, ib >> 2u), int((ib & 3u) * 8u), 8);
                uint c = bitfieldExtract(pick(sw0, sw1, sw2, ic >> 2u), int((ic & 3u) * 8u), 8);
                sc = float(lo ? (a & 63u) : ((c & 0xFu) | ((a >> 6u) << 4u)));
                mm = float(lo ? (bb & 63u) : ((c >> 4u) | ((bb >> 6u) << 4u)));
            }
            dsc[s][r] = d * sc;
            dmm[s][r] = -dmn * mm;
        }

        // ---- stage activations: 64 tokens x 64 k ---------------------------
        // `kd` varies fastest, so 16 consecutive threads read one token's 64
        // contiguous q8 bytes. The shared write scatters 4 consecutive k for one
        // token (a 4-way bank conflict), which is ~3 % of this step's slots
        // against 2048 FMAs per thread, not worth transposing for.
        for (uint l = 0u; l < 4u; l++) {
            uint idx = tid + l * 256u;             // 0..1023
            uint kd = idx & 15u;
            uint t = idx >> 4u;
            uint tgl = t0 + t;
            uint v = 0u;
            float ds = 0.0;
            if (tgl < seq) {
                uint qbase = tgl * sxd + (b * 256u + j * 64u) / 4u;
                v = xin[qbase + kd];
                ds = uintBitsToFloat(xin[tgl * sxd + in_dim / 4u + b]);
            }
            for (uint c = 0u; c < 4u; c++) {
                af[kd * 4u + c][t >> 2u][t & 3u] =
                    float(bitfieldExtract(int(v), int(c * 8u), 8)) * ds;
            }
        }

        barrier();

        // ---- per-(sub-block, token) activation sums -------------------------
        // The Q4_K min term is constant across k within a sub-block, so it
        // contracts to `dmm[s][row] * sum(af over the sub-block)`: a per-token
        // quantity, computed once here instead of TM times in the inner loop.
        if (tid < 128u) {
            uint s = tid >> 6u;
            uint t = tid & 63u;
            float acc_m = 0.0;
            for (uint i = 0u; i < 32u; i++) {
                acc_m += af[s * 32u + i][t >> 2u][t & 3u];
            }
            msum[s][t] = acc_m;
        }

        barrier();

        // ---- compute --------------------------------------------------------
        if (live)
        for (uint s = 0u; s < 2u; s++) {
            // `psum` holds the raw integer dot for this sub-block; the weight
            // scale is applied once at the end of it rather than once per
            // weight, which keeps the inner loop pure FMA.
            float psum[TM][TN];
            for (uint m = 0u; m < TM; m++) {
                for (uint t = 0u; t < TN; t++) {
                    psum[m][t] = 0.0;
                }
            }
            for (uint dwi = 0u; dwi < 8u; dwi++) {
                // Rows are assigned to threads with stride 32 (not 4) so these
                // four reads are consecutive across the lanes of a warp.
                uint w0 = wq[dwi][rg];
                uint w1 = wq[dwi][rg + 32u];
                uint w2 = wq[dwi][rg + 64u];
                uint w3 = wq[dwi][rg + 96u];
                for (uint c = 0u; c < 4u; c++) {
                    uint kb = s * 32u + dwi * 4u + c;
                    int sh = int(c * 8u + s * 4u);
                    vec4 a0 = af[kb][tg * 2u];
                    vec4 a1 = af[kb][tg * 2u + 1u];
                    float n0 = float(bitfieldExtract(w0, sh, 4));
                    float n1 = float(bitfieldExtract(w1, sh, 4));
                    float n2 = float(bitfieldExtract(w2, sh, 4));
                    float n3 = float(bitfieldExtract(w3, sh, 4));
                    psum[0][0] = fma(n0, a0.x, psum[0][0]);
                    psum[0][1] = fma(n0, a0.y, psum[0][1]);
                    psum[0][2] = fma(n0, a0.z, psum[0][2]);
                    psum[0][3] = fma(n0, a0.w, psum[0][3]);
                    psum[0][4] = fma(n0, a1.x, psum[0][4]);
                    psum[0][5] = fma(n0, a1.y, psum[0][5]);
                    psum[0][6] = fma(n0, a1.z, psum[0][6]);
                    psum[0][7] = fma(n0, a1.w, psum[0][7]);
                    psum[1][0] = fma(n1, a0.x, psum[1][0]);
                    psum[1][1] = fma(n1, a0.y, psum[1][1]);
                    psum[1][2] = fma(n1, a0.z, psum[1][2]);
                    psum[1][3] = fma(n1, a0.w, psum[1][3]);
                    psum[1][4] = fma(n1, a1.x, psum[1][4]);
                    psum[1][5] = fma(n1, a1.y, psum[1][5]);
                    psum[1][6] = fma(n1, a1.z, psum[1][6]);
                    psum[1][7] = fma(n1, a1.w, psum[1][7]);
                    psum[2][0] = fma(n2, a0.x, psum[2][0]);
                    psum[2][1] = fma(n2, a0.y, psum[2][1]);
                    psum[2][2] = fma(n2, a0.z, psum[2][2]);
                    psum[2][3] = fma(n2, a0.w, psum[2][3]);
                    psum[2][4] = fma(n2, a1.x, psum[2][4]);
                    psum[2][5] = fma(n2, a1.y, psum[2][5]);
                    psum[2][6] = fma(n2, a1.z, psum[2][6]);
                    psum[2][7] = fma(n2, a1.w, psum[2][7]);
                    psum[3][0] = fma(n3, a0.x, psum[3][0]);
                    psum[3][1] = fma(n3, a0.y, psum[3][1]);
                    psum[3][2] = fma(n3, a0.z, psum[3][2]);
                    psum[3][3] = fma(n3, a0.w, psum[3][3]);
                    psum[3][4] = fma(n3, a1.x, psum[3][4]);
                    psum[3][5] = fma(n3, a1.y, psum[3][5]);
                    psum[3][6] = fma(n3, a1.z, psum[3][6]);
                    psum[3][7] = fma(n3, a1.w, psum[3][7]);
                }
            }
            for (uint m = 0u; m < TM; m++) {
                uint r = rg + 32u * m;
                float ws = dsc[s][r];
                float wm = dmm[s][r];
                for (uint t = 0u; t < TN; t++) {
                    uint tk = tg * TN + t;
                    acc[m][t] = fma(ws, psum[m][t], acc[m][t]);
                    acc[m][t] = fma(wm, msum[s][tk], acc[m][t]);
                }
            }
        }
    }

    uint ldy = (ld == 0u) ? out_dim : ld;
    for (uint m = 0u; m < TM; m++) {
        uint row = o0 + rg + 32u * m;
        if (row >= out_dim) {
            continue;
        }
        for (uint t = 0u; t < TN; t++) {
            uint tok = t0 + tg * TN + t;
            if (tok < seq) {
                yout[tok * ldy + row] = acc[m][t];
            }
        }
    }
}
