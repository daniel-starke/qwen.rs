#version 450
// Fused Q4_K x q8_K dequant-GEMM: coalesced, one 32-lane row team per output row.
//
// History (measured on a GTX 1060, `cargo run -p offload --features vulkan
// --example gpubench`):
//
//   * The original kernel gave each thread a whole output row and staged the
//     144-byte Q4_K block into a `uint wblk[36]` local array. The nibble loop
//     indexes that array dynamically, so the driver had to back it with *local
//     (scratch) memory* and every weight byte took a global round-trip.
//     17.3 GB/s on the 17408x5120 FFN weight.
//   * Reading `weights[]` directly, a dword at a time, removed the spill but
//     kept one-thread-per-row. Throughput then *peaked* at ~21 GB/s for ~64-96
//     workgroups and collapsed to 6 GB/s at 272 workgroups: with a row per
//     thread, the 32 lanes of a warp walk 32 weight streams 144*nb bytes apart,
//     so once every SM is full the card is juggling thousands of divergent
//     streams and the cache stops helping.
//
// This revision fixes the access pattern instead. `ROWS` output rows share a
// workgroup; each row is owned by a 32-lane team, and the team splits the *block*
// rather than the row:
//
//   lane j  <->  qs dword j  of every block (bytes 16+4j .. 19+4j)
//
// so the team's 32 loads are one contiguous 128-byte transaction per block. Each
// qs dword holds 4 weight bytes = 8 nibbles: the low nibbles belong to sub-block
// `2p` and the high nibbles to `2p+1`, where `p = j / 8`. A lane therefore needs
// only its own two (scale, min) pairs, not all eight.
//
// Numerics: the integer parts are unchanged and exact; `sc[s] * dot` summed
// across lanes equals `sc[s] * (sum of dots)` because integer arithmetic is
// associative. The per-block float term `d*sumi + dmin*sum_m` is accumulated per
// lane and tree-reduced at the end, so the float summation *order* differs from
// the CPU `gemv_q4k` (as it already did); the difference stays at f32 rounding
// level, which `gpubench --check` and `cli --forward` both verify.
//
// Alignment: a Q4_K block is 144 B = 36 dwords, so a block base is dword aligned.
// The q8_K activation stride is `in_dim + nb*4` with `in_dim % 256 == 0`, so the
// activation dword indices below are exact. Both hold by construction in
// `pack_x` / `pack_x_seq`.
//
// The host must dispatch `ceil(out_dim / ROWS)` workgroups in x: see
// `GpuDevice::grid_x`, which is the single source of truth for that.

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

// Select one of the block's three packed-scale dwords. Written as explicit
// selects rather than an indexed array so it never becomes a scratch-memory
// access, and so the three dwords are loaded once per block instead of once per
// scale byte (the compiler cannot CSE loads at six different dynamic indices).
uint pick(uint w0, uint w1, uint w2, uint k) {
    return (k == 0u) ? w0 : ((k == 1u) ? w1 : w2);
}

void main() {
    uint lane = gl_LocalInvocationID.x;          // 0..31, splits the block
    uint row = gl_LocalInvocationID.y;           // 0..ROWS-1
    uint o = gl_WorkGroupID.x * ROWS + row;      // output row
    uint si = gl_WorkGroupID.y;                  // sequence position
    bool live = (o < out_dim) && (si < seq);

    uint nb = in_dim / 256u;
    uint sx = in_dim + nb * 4u;
    uint base = si * sx;
    uint qs_dw = base / 4u;
    uint ds_dw = (base + in_dim) / 4u;

    // This lane owns qs dword `lane`, i.e. sub-blocks 2p and 2p+1 at byte offset
    // `q` inside each sub-block's 32-byte activation span.
    uint p = lane >> 3u;
    uint q = lane & 7u;
    uint slo = 2u * p;          // sub-block of this lane's low nibbles
    uint shi = 2u * p + 1u;     // sub-block of its high nibbles

    // `get_scale_min_k4` has two forms, for sub-block < 4 and >= 4. A lane's two
    // sub-blocks are `2p` and `2p+1`, which always land in the *same* form
    // (p < 2 -> both < 4), so the choice is fixed for the whole kernel: hoist the
    // three scale-byte indices each form needs and select branchlessly below,
    // instead of re-deciding it inside every block iteration.
    bool lo_form = p < 2u;
    uint ia0 = lo_form ? slo : slo - 4u, ib0 = lo_form ? slo + 4u : slo, ic0 = slo + 4u;
    uint ia1 = lo_form ? shi : shi - 4u, ib1 = lo_form ? shi + 4u : shi, ic1 = shi + 4u;
    // Split each index into (which of the three dwords, bit offset within it).
    uint wa0 = ia0 >> 2u, wb0 = ib0 >> 2u, wc0 = ic0 >> 2u;
    uint wa1 = ia1 >> 2u, wb1 = ib1 >> 2u, wc1 = ic1 >> 2u;
    int ha0 = int((ia0 & 3u) * 8u), hb0 = int((ib0 & 3u) * 8u), hc0 = int((ic0 & 3u) * 8u);
    int ha1 = int((ia1 & 3u) * 8u), hb1 = int((ib1 & 3u) * 8u), hc1 = int((ic1 & 3u) * 8u);

    float acc = 0.0;
    if (live) {
        for (uint i = 0u; i < nb; i++) {
            uint wb = (o * nb + i) * 36u;

            float xd = uintBitsToFloat(xin[ds_dw + i]);
            // The block header is exactly the two f16 scales (d, dmin) packed in
            // one dword, so `unpackHalf2x16` decodes both in a single native
            // instruction (the hand-rolled f16->f32 it replaces was ~20 ALU ops
            // *per lane per block*, and every lane of the team decodes it).
            vec2 dm = unpackHalf2x16(weights[wb]);
            float d = xd * dm.x;
            float dmin = -xd * dm.y;
            // This lane's two (scale, min) pairs, with the same 6-bit + 2-bit-carry
            // split as the CPU `unpack_q4_scales` / `get_scale_min_k4`.
            uint sw0 = weights[wb + 1u], sw1 = weights[wb + 2u], sw2 = weights[wb + 3u];
            uint a0 = bitfieldExtract(pick(sw0, sw1, sw2, wa0), ha0, 8);
            uint b0 = bitfieldExtract(pick(sw0, sw1, sw2, wb0), hb0, 8);
            uint c0 = bitfieldExtract(pick(sw0, sw1, sw2, wc0), hc0, 8);
            uint a1 = bitfieldExtract(pick(sw0, sw1, sw2, wa1), ha1, 8);
            uint b1 = bitfieldExtract(pick(sw0, sw1, sw2, wb1), hb1, 8);
            uint c1 = bitfieldExtract(pick(sw0, sw1, sw2, wc1), hc1, 8);
            int sc0 = int(lo_form ? (a0 & 63u) : ((c0 & 0xFu) | ((a0 >> 6u) << 4u)));
            int mm0 = int(lo_form ? (b0 & 63u) : ((c0 >> 4u) | ((b0 >> 6u) << 4u)));
            int sc1 = int(lo_form ? (a1 & 63u) : ((c1 & 0xFu) | ((a1 >> 6u) << 4u)));
            int mm1 = int(lo_form ? (b1 & 63u) : ((c1 >> 4u) | ((b1 >> 6u) << 4u)));

            uint xblk = qs_dw + i * 64u;
            uint qw = weights[wb + 4u + lane];
            uint xw0 = xin[xblk + slo * 8u + q];
            uint xw1 = xin[xblk + shi * 8u + q];

            int dot0 = 0, dot1 = 0, sm0 = 0, sm1 = 0;
            for (int k = 0; k < 4; k++) {
                int sh = k * 8;
                // `bitfieldExtract` is one instruction each and, on the signed
                // overload, sign-extends the q8 byte for free: replacing the
                // shift/mask/compare/subtract chain the scalar version used.
                int nlo = int(bitfieldExtract(qw, sh, 4));
                int nhi = int(bitfieldExtract(qw, sh + 4, 4));
                int v0 = bitfieldExtract(int(xw0), sh, 8);
                int v1 = bitfieldExtract(int(xw1), sh, 8);

                dot0 += nlo * v0;
                sm0 += v0;
                dot1 += nhi * v1;
                sm1 += v1;
            }

            int sumi = sc0 * dot0 + sc1 * dot1;
            int sum_m = mm0 * sm0 + mm1 * sm1;
            acc += d * float(sumi) + dmin * float(sum_m);
        }
    }

    // Tree-reduce the 32 lanes of each row team. Every invocation reaches every
    // barrier (inactive rows carry acc = 0), which the spec requires.
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
