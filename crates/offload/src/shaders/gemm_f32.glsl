#version 450
// Dense f32 GEMM on sub-matrices of large buffers: C (-)= A B^T.
//
// WHY THIS EXISTS
// ---------------
// Building the NPU graph sets (`crates/npubuild`) quantizes every weight
// matrix with GPTQ, which needs the inverse Cholesky factor of a 17408 x
// 17408 input Hessian per layer: ~7 TFLOP of blocked factorization whose
// every large step is a product of this form. The card's fp32 FMA peak is
// 3.2-3.9 TFLOP/s (`gemm_q4k.glsl`), several times what the CPU cores reach
// in f32, so the factorization lives in VRAM and only its 256 x 256 diagonal
// blocks visit the host (`gpu/dense.rs`).
//
// THE PRODUCT
// -----------
//     C[i][j] (=|-=) sum_t A[i][t] * B[j][t]      i < m, j < n, t < k
//
// A, B and C are regions of three buffers (the same buffer may be bound to
// several bindings: a blocked Cholesky updates the trailing part of the
// matrix its panel is read from), given as an offset and a row stride in
// floats. `mode` bit 0: subtract from C instead of storing; bit 1: B is
// read transposed (`B[j][t]` at `b_off + t * ldb + j`), which lets the
// triangular inverse multiply by a matrix it holds row-major on the other
// side without a transposed copy; bit 2: only the tiles that reach the lower
// triangle of C (`j <= i`) are computed; a symmetric update whose upper
// half nobody reads.
//
// TILE SHAPE (as `gemm_q4k.glsl`, whose measurements chose it)
// ------------------------------------------------------------
//     BM = 128 rows of A    BN = 64 rows of B    BK = 32 t per staging step
//     256 threads = 32 row groups x 8 column groups
//     TM = 4 rows / thread  TN = 8 columns / thread: 32 accumulators
//
// Dispatch `ceil(m / BM)` x `ceil(n / BN)` workgroups.

#define BM 128u
#define BN 64u
#define BK 32u
#define TM 4u
#define TN 8u

layout(local_size_x = 32, local_size_y = 8) in;

layout(set = 0, binding = 0, std430) readonly buffer A_ { float a[]; };
layout(set = 0, binding = 1, std430) readonly buffer B_ { float b[]; };
layout(set = 0, binding = 2, std430) buffer C_ { float c[]; };
layout(set = 0, binding = 8, std140) uniform Params {
    uint a_off;
    uint lda;
    uint b_off;
    uint ldb;
    uint c_off;
    uint ldc;
    uint m;
    uint n;
    uint k;
    uint mode;
};

// A thread reads its 4 rows of A as one vec4 and its 8 columns as two.
shared vec4 sa[BK][BM / 4u];
shared vec4 sb[BK][BN / 4u];

void main() {
    uint rg = gl_LocalInvocationID.x;              // 0..31, row group
    uint cg = gl_LocalInvocationID.y;              // 0..7, column group
    uint tid = rg + 32u * cg;                      // 0..255
    uint i0 = gl_WorkGroupID.x * BM;
    uint j0 = gl_WorkGroupID.y * BN;
    bool tb = (mode & 2u) != 0u;
    if ((mode & 4u) != 0u && j0 >= i0 + BM) {
        return;                                    // wholly above the diagonal
    }

    vec4 acc[TM][2];
    for (uint r = 0u; r < TM; r++) {
        acc[r][0] = vec4(0.0);
        acc[r][1] = vec4(0.0);
    }

    for (uint k0 = 0u; k0 < k; k0 += BK) {
        barrier();                                 // the previous step's reads are done

        // ---- stage A: 128 rows x 32 t. `t` varies fastest, so a warp reads
        // 32 contiguous floats of one row.
        for (uint l = 0u; l < 16u; l++) {
            uint idx = tid + l * 256u;             // 0..4095
            uint t = idx & 31u;
            uint r = idx >> 5u;
            float v = 0.0;
            if (i0 + r < m && k0 + t < k) {
                v = a[a_off + (i0 + r) * lda + k0 + t];
            }
            sa[t][r >> 2u][r & 3u] = v;
        }
        // ---- stage B: 64 rows x 32 t
        for (uint l = 0u; l < 8u; l++) {
            uint idx = tid + l * 256u;             // 0..2047
            uint t = tb ? (idx >> 6u) : (idx & 31u);
            uint r = tb ? (idx & 63u) : (idx >> 5u);
            float v = 0.0;
            if (j0 + r < n && k0 + t < k) {
                v = tb ? b[b_off + (k0 + t) * ldb + j0 + r] : b[b_off + (j0 + r) * ldb + k0 + t];
            }
            sb[t][r >> 2u][r & 3u] = v;
        }

        barrier();

        for (uint t = 0u; t < BK; t++) {
            vec4 av = sa[t][rg];
            vec4 b0 = sb[t][2u * cg];
            vec4 b1 = sb[t][2u * cg + 1u];
            acc[0][0] += av.x * b0;
            acc[0][1] += av.x * b1;
            acc[1][0] += av.y * b0;
            acc[1][1] += av.y * b1;
            acc[2][0] += av.z * b0;
            acc[2][1] += av.z * b1;
            acc[3][0] += av.w * b0;
            acc[3][1] += av.w * b1;
        }
    }

    bool sub = (mode & 1u) != 0u;
    for (uint r = 0u; r < TM; r++) {
        uint row = i0 + rg * TM + r;
        if (row >= m) {
            continue;
        }
        for (uint t = 0u; t < TN; t++) {
            uint col = j0 + cg * TN + t;
            if (col >= n) {
                continue;
            }
            uint idx = c_off + row * ldc + col;
            float v = acc[r][t >> 2u][t & 3u];
            c[idx] = sub ? c[idx] - v : v;
        }
    }
}
