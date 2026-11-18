#version 450
// Tiled Q6_K x q8_K GEMM for prefill: the Q6_K twin of `gemm_q4k.glsl`.
//
// WHY THIS EXISTS
// ---------------
// `FfnDown` is the only Q6_K matmul in a layer and it is a third of the FFN's
// FLOPs. Until this kernel existed it ran `q6k.glsl`, which maps the sequence
// position to `gl_WorkGroupID.y` and so re-reads the weight row for every
// (row, position) pair. Measured on the GTX 1060 at seq=462 that was **433 ms,
// 190 GFLOP/s, 0.20x the CPU**, as much as 7.4 `FfnUp`s for the same FLOP
// count, and single-handedly what made GPU prefill a net loss even after the
// Q4_K GEMM reached 2.3x CPU.
//
// TILE SHAPE: deliberately identical to `gemm_q4k.glsl`
// ------------------------------------------------------
//     BM = 128 output rows   BN = 64 tokens   BK = 64 k per staging step
//     256 threads = 32 row groups x 8 token groups
//     TM = 4 rows/thread     TN = 8 tokens/thread
//
// Same numbers, same `af` layout, same reduction, so the two kernels can be read
// against each other. TN = 8 is from the probe's amortization curve
// (`gpubench --gate`): the per-weight unpack is spread over TN FMAs and TN=8
// measured 82-87 % of the card's ALU peak against 60 % at TN=4.
//
// WHAT IS DIFFERENT FROM Q4_K, AND WHY
// ------------------------------------
// A Q6_K weight is `ql` (4 bits) + `qh` (2 bits) - 32, scaled by a **signed**
// group scale every 16 elements. Assembling that costs ~4 integer ops per
// weight against Q4_K's 1, and in a Q4_K-shaped inner loop each weight would be
// assembled once per *token group*, i.e. 8 times per workgroup.
//
// So the staging pass **unpacks to signed int8 once** (four weights per dword,
// `q - 32` in each byte) and the inner loop only sign-extends and converts. The
// group scale is not folded in: it is applied to the group's partial sum, which
// keeps the inner loop pure FMA and is exact; `q` is -32..31 and the q8
// activation is -128..127, so a 16-term group sum peaks at 65536, well inside
// f32's exact-integer range of 2^24.
//
// STEP DECOMPOSITION
// ------------------
// A Q6_K super-block is 256 weights = two 128-element halves (`jj`), and each
// half's 64 `ql` bytes hold its first 64 elements in their low nibbles and its
// last 64 in their high nibbles (`hi`). One step covers 64 elements = one
// (half, nibble) pair = **4 group scales**, so a super-block is 4 steps and the
// step count matches `gemm_q4k.glsl` exactly.
//
// The `ql`/`qh` dwords of a half are therefore read twice per super-block, once
// for `hi = 0` and once for `hi = 1`. That is deliberate: consuming both nibbles
// in one step would make the step 128 elements wide, and `af` would need 32 KB
// instead of 16 KB, which costs more occupancy than the second read costs
// bandwidth, since it lands in L2 behind the first.
//
// GRID CONTRACT (read before changing any constant here)
// ------------------------------------------------------
// Dispatch `ceil(out_dim/BM)` x `ceil(seq/BN)` workgroups via
// `GpuDevice::gemm_grid`; the same contract as `gemm_q4k.glsl`, and **not**
// `grid_x`/`rows_per_wg`, which describe the GEMV kernels (whose `y` is a single
// position, not a token tile). `use_gemm` must pick the pipeline and the grid
// together. A mismatch is silent: it computes part of the output and leaves the
// rest stale (a flattering 300 GB/s was measured from exactly that).
//
// VRAM LAYOUT
// -----------
// Blocks are the host-repacked **212-byte (53 dword)** stride (`Q6K_BLOCK_PAD`),
// so every load below is an aligned dword read:
//   dwords  0..31  ql; 4-bit low halves, two elements per byte
//   dwords 32..47  qh; 2-bit high halves, four elements per byte
//   dwords 48..51  16 signed 8-bit group scales (one per 16 elements)
//   dword     52   the f16 block scale `d` in its low half

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

// Dword offsets of the sections within a repacked 53-dword Q6_K block.
const uint QL_DW = 0u;
const uint QH_DW = 32u;
const uint SC_DW = 48u;
const uint D_DW = 52u;

// Unpacked weights for the step: `[16 dwords of k][row]`, four signed int8 per
// dword. `BM + 1u`, not `BM`: the staging loop writes with the *row* index
// varying fastest across a warp, and the compute loop reads with it varying
// fastest too, so the stride must be odd in dwords or all 32 lanes hit one bank.
shared uint wq8[16][BM + 1u];
// Token-contiguous so the inner loop reads a thread's 8 tokens as two `vec4`s,
// and every lane of a warp shares the same token group -> broadcast, no conflict.
shared vec4 af[BK][BN / 4u];
// `d * group_scale` per (group in step, row). Read with `g` uniform and the row
// varying, so a stride of BM dwords is already conflict-free.
shared float dsc[4][BM];

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

    // One step per (super-block, half, nibble-set): 64 weights, 4 group scales.
    for (uint step = 0u; step < nb * 4u; step++) {
        uint b = step >> 2u;                       // super-block
        uint jj = (step >> 1u) & 1u;               // which 128-element half
        uint hi = step & 1u;                       // 0 = low nibbles, 1 = high
        uint base_g = jj * 8u + hi * 4u;           // first group scale of the step

        barrier();                                 // previous iteration's reads are done

        // ---- stage weights: 128 rows x 16 dwords, unpacked to signed int8 ----
        // `dw` varies fastest across `tid`, so 16 consecutive threads read one
        // row's 16 contiguous `ql` dwords (64 bytes) and a warp covers 2 rows.
        for (uint l = 0u; l < 8u; l++) {
            uint idx = tid + l * 256u;             // 0..2047
            uint r = idx >> 4u;
            uint dw = idx & 15u;
            uint orow = o0 + r;
            uint packed = 0u;
            if (orow < out_dim) {
                uint wb = (orow * nb + b) * 53u;
                uint qlw = weights[wb + QL_DW + jj * 16u + dw];
                uint qhw = weights[wb + QH_DW + jj * 8u + (dw & 7u)];
                // `ql` byte `i` of this dword is element `4*dw + i` of the half
                // (low nibble) or `+ 64` (high nibble); both take their two top
                // bits from the same `qh` byte, in fields 2 apart.
                int qh_sh = int(((dw >> 3u) & 1u) * 2u + hi * 4u);
                int nib_sh = int(hi * 4u);
                for (uint i = 0u; i < 4u; i++) {
                    int sh = int(i * 8u);
                    uint nib = bitfieldExtract(qlw, sh + nib_sh, 4);
                    uint qhb = bitfieldExtract(qhw, sh, 8);
                    uint top = bitfieldExtract(qhb, qh_sh, 2);
                    // -32 folded in here so the inner loop is a plain
                    // sign-extend; the byte then holds a value in -32..31.
                    uint q = uint(int(nib | (top << 4u)) - 32) & 0xFFu;
                    packed |= q << sh;
                }
            }
            wq8[dw][r] = packed;
        }

        // ---- stage this step's four group scales, one per thread -------------
        {
            uint r = tid & 127u;
            uint g = (tid >> 7u) * 2u;             // threads cover g and g+1
            uint orow = o0 + r;
            float d = 0.0;
            int sc0 = 0, sc1 = 0;
            if (orow < out_dim) {
                uint wb = (orow * nb + b) * 53u;
                d = unpackHalf2x16(weights[wb + D_DW]).x;
                uint i0 = base_g + g, i1 = base_g + g + 1u;
                sc0 = bitfieldExtract(int(weights[wb + SC_DW + (i0 >> 2u)]), int((i0 & 3u) * 8u), 8);
                sc1 = bitfieldExtract(int(weights[wb + SC_DW + (i1 >> 2u)]), int((i1 & 3u) * 8u), 8);
            }
            dsc[g][r] = d * float(sc0);
            dsc[g + 1u][r] = d * float(sc1);
        }

        // ---- stage activations: 64 tokens x 64 k ---------------------------
        // `kd` varies fastest, so 16 consecutive threads read one token's 64
        // contiguous q8 bytes. The shared write scatters 4 consecutive k for one
        // token, which is a bank conflict, but it is 4 vector stores per thread
        // per step against 2048 FMAs: measured at a few percent, not worth
        // transposing for (`gemm_q4k.glsl` carries the same note).
        for (uint l = 0u; l < 4u; l++) {
            uint idx = tid + l * 256u;             // 0..1023
            uint kd = idx & 15u;
            uint t = idx >> 4u;
            uint tgl = t0 + t;
            uint v = 0u;
            float ds = 0.0;
            if (tgl < seq) {
                uint qbase = tgl * sxd + b * 64u + jj * 32u + hi * 16u;
                v = xin[qbase + kd];
                ds = uintBitsToFloat(xin[tgl * sxd + in_dim / 4u + b]);
            }
            for (uint c = 0u; c < 4u; c++) {
                af[kd * 4u + c][t >> 2u][t & 3u] =
                    float(bitfieldExtract(int(v), int(c * 8u), 8)) * ds;
            }
        }

        barrier();

        // ---- compute --------------------------------------------------------
        // Four groups of 16 k. `psum` holds the raw integer dot for one group;
        // its scale is applied once at the end rather than once per weight,
        // which keeps the inner loop pure FMA.
        if (live)
        for (uint g = 0u; g < 4u; g++) {
            float psum[TM][TN];
            for (uint m = 0u; m < TM; m++) {
                for (uint t = 0u; t < TN; t++) {
                    psum[m][t] = 0.0;
                }
            }
            for (uint dd = 0u; dd < 4u; dd++) {
                uint dw = g * 4u + dd;
                // Rows are assigned to threads with stride 32 (not 4) so these
                // four reads are consecutive across the lanes of a warp.
                uint w0 = wq8[dw][rg];
                uint w1 = wq8[dw][rg + 32u];
                uint w2 = wq8[dw][rg + 64u];
                uint w3 = wq8[dw][rg + 96u];
                for (uint c = 0u; c < 4u; c++) {
                    uint kb = dw * 4u + c;
                    int sh = int(c * 8u);
                    vec4 a0 = af[kb][tg * 2u];
                    vec4 a1 = af[kb][tg * 2u + 1u];
                    float n0 = float(bitfieldExtract(int(w0), sh, 8));
                    float n1 = float(bitfieldExtract(int(w1), sh, 8));
                    float n2 = float(bitfieldExtract(int(w2), sh, 8));
                    float n3 = float(bitfieldExtract(int(w3), sh, 8));
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
                float ws = dsc[g][rg + 32u * m];
                for (uint t = 0u; t < TN; t++) {
                    acc[m][t] = fma(ws, psum[m][t], acc[m][t]);
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
