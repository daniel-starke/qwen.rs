#version 450
// Fused Q6_K x q8_K dequant-GEMM: coalesced, one 32-lane row team per output row.
//
// Same restructuring as `q4k.glsl`, for the same reason: the previous revision
// staged the 210-byte Q6_K block into a `uint wblk[53]` local array whose
// dynamic indexing forced it into local (scratch) memory, and gave each thread a
// whole output row so the 32 lanes of a warp walked 32 divergent weight streams.
// It reached 10.4 GB/s on a GTX 1060 (`--example gpubench`), ~5 % of the card.
//
// Layout of a Q6_K super-block (256 weights):
//   bytes   0..127  ql; 4-bit low halves, two elements per byte
//   bytes 128..191  qh; 2-bit high halves, four elements per byte
//   bytes 192..207  16 signed group scales (one per 16 elements)
//   bytes 208..209  the f16 block scale `d`
//
// The host repacks blocks to a **212-byte (53 dword) stride** in VRAM
// (`Q6K_BLOCK_PAD`), so a block base is always dword aligned and every load
// below is a plain aligned dword read instead of a funnel shift.
//
// Lane mapping: lane `j` owns ql dword `j`; that is ql bytes `4j..4j+3`, whose
// low nibbles are elements `p = jj*128 + c + m` and whose high nibbles are
// `p + 64` (`jj = 4j/64`, `c = 4j % 64`, `m = 0..3`). So the team's 32 ql loads
// are one contiguous 128-byte transaction, both nibble halves of each byte are
// consumed in the same pass, and the two elements share a qh byte (different
// 2-bit fields) and a single qh dword per lane. Each lane touches exactly two of
// the 16 group scales, and both indices are fixed for the whole kernel.
//
// Numerics: the integer accumulation is exact and associative, so splitting it
// across lanes is lossless; only the per-block float term is summed in a
// different order than the CPU `gemv_q6k` (tree over lanes rather than a single
// running sum), at f32 rounding level. `gpubench --check` and `cli --forward`
// both verify it.
//
// The host must dispatch `ceil(out_dim / ROWS)` workgroups in x: see
// `GpuDevice::grid_x`, the single source of truth for that.

#define ROWS 4u

layout(local_size_x = 32, local_size_y = 4) in;

layout(set = 0, binding = 0, std430) readonly buffer Weights { uint weights[]; };
layout(set = 0, binding = 1, std430) readonly buffer Xin { uint xin[]; };
layout(set = 0, binding = 2, std430) writeonly buffer Yout { float yout[]; };
layout(set = 0, binding = 3, std140) uniform Params {
    uint out_dim;
    uint in_dim;
    uint seq;
};

shared float part[ROWS][32];

// Dword offsets of the sections within a repacked 53-dword Q6_K block.
const uint QL_DW = 0u;      // bytes   0..127
const uint QH_DW = 32u;     // bytes 128..191
const uint SC_DW = 48u;     // bytes 192..207
const uint D_DW = 52u;      // bytes 208..209 (low half of the dword)

void main() {
    uint lane = gl_LocalInvocationID.x;
    uint row = gl_LocalInvocationID.y;
    uint o = gl_WorkGroupID.x * ROWS + row;
    uint si = gl_WorkGroupID.y;
    bool live = (o < out_dim) && (si < seq);

    uint nb = in_dim / 256u;
    uint sx = in_dim + nb * 4u;
    uint qbase = si * sx;
    uint qs_dw = qbase / 4u;
    uint ds_dw = (qbase + in_dim) / 4u;

    // Everything derived from `lane` alone is loop-invariant.
    uint ql_byte = 4u * lane;
    uint jj = ql_byte >> 6u;             // 0 or 1: which 128-element half
    uint c = ql_byte & 63u;              // byte within that half
    uint qh_dw = QH_DW + jj * 8u + ((c & 31u) >> 2u);
    uint qh_sh = (c >> 5u) * 2u;         // 2-bit field of the low element
    uint p_lo = jj * 128u + c;           // first element of the low-nibble run
    uint g0 = p_lo >> 4u;                // its 16-element group
    uint g1 = (p_lo + 64u) >> 4u;        // group of the high-nibble run
    uint x_lo_dw = p_lo >> 2u;
    uint x_hi_dw = (p_lo + 64u) >> 2u;

    float acc = 0.0;
    if (live) {
        for (uint i = 0u; i < nb; i++) {
            uint wb = (o * nb + i) * 53u;
            float d = uintBitsToFloat(xin[ds_dw + i]) * unpackHalf2x16(weights[wb + D_DW]).x;

            uint qlw = weights[wb + QL_DW + lane];
            uint qhw = weights[wb + qh_dw];
            // Signed 8-bit group scales.
            int sc0 = bitfieldExtract(int(weights[wb + SC_DW + (g0 >> 2u)]), int((g0 & 3u) * 8u), 8);
            int sc1 = bitfieldExtract(int(weights[wb + SC_DW + (g1 >> 2u)]), int((g1 & 3u) * 8u), 8);

            uint xblk = qs_dw + i * 64u;
            uint xlo = xin[xblk + x_lo_dw];
            uint xhi = xin[xblk + x_hi_dw];

            int a0 = 0, a1 = 0;
            for (int m = 0; m < 4; m++) {
                int sh = m * 8;
                uint qhb = bitfieldExtract(qhw, sh, 8);
                int q0 = int(bitfieldExtract(qlw, sh, 4)
                             | (bitfieldExtract(qhb, int(qh_sh), 2) << 4u)) - 32;
                int q1 = int(bitfieldExtract(qlw, sh + 4, 4)
                             | (bitfieldExtract(qhb, int(qh_sh) + 4, 2) << 4u)) - 32;
                a0 += q0 * bitfieldExtract(int(xlo), sh, 8);
                a1 += q1 * bitfieldExtract(int(xhi), sh, 8);
            }
            acc += d * float(sc0 * a0 + sc1 * a1);
        }
    }

    // Tree-reduce the row team. Every invocation reaches every barrier.
    part[row][lane] = acc;
    barrier();
    for (uint s = 16u; s > 0u; s >>= 1u) {
        if (lane < s) { part[row][lane] += part[row][lane + s]; }
        barrier();
    }
    if (live && lane == 0u) {
        yout[si * out_dim + o] = part[row][0];
    }
}
