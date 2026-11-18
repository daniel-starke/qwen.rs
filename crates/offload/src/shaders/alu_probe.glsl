#version 450
// ALU-ceiling probe: measures what this GPU can *arithmetically* sustain, with
// no global-memory traffic in the inner loop. It exists to answer one question
// before any tiled-GEMM work is written: is the GTX 1060's compute ceiling far
// enough above the CPU's measured 966 GFLOP/s prefill GEMM rate to be worth
// streaming weights over PCIe for?
//
// The fused `q4k.glsl` / `q6k.glsl` kernels cannot answer that. They map the
// sequence position to `gl_WorkGroupID.y`, so a "GEMM" re-reads the weight row
// once per token: they measure the memory system, not the ALUs. This probe
// isolates the ALUs instead, so the number it reports is an *upper bound* on any
// kernel that could be written for this card.
//
// Modes (selected by the `mode` uniform, so the branch is uniform across the
// dispatch and costs nothing):
//
//   0  FP32 FFMA chains: 16 independent `fma` chains. Intended as the FFMA
//                             ceiling, but measured on a GTX 1060 it lands ~15 %
//                             *below* mode 2 at TILE=16: all 16 FMAs read the
//                             same two source registers and Pascal's operand
//                             collector cannot feed that at full rate. Kept as a
//                             cross-check, not as the peak: the host takes the
//                             best rate across all modes as the peak.
//   1  INT32 MAD peak: the same shape with `u = u*m + n`. On Pascal
//                             (sm_61) a 32-bit integer multiply is emulated with
//                             XMAD, so this is expected to be several times
//                             slower than mode 0; it is the ceiling for any
//                             kernel that accumulates the Q4_K dot in `int`,
//                             which is what `q4k.glsl` does today.
//   2  Tiled Q4_K model: the inner loop a real tiled GEMM would run: fetch
//                             one weight dword (8 nibbles), and for each nibble
//                             do `TILE` FFMAs against `TILE` register-resident
//                             activations. `TILE` is the number of tokens a
//                             workgroup keeps in registers, i.e. exactly the
//                             reuse a GEMM buys over a GEMV. Sweeping `TILE`
//                             shows how fast the dequant cost amortizes, and
//                             `TILE = 1` reproduces the GEMV case for reference.
//
// `TILE` is a specialization constant so `acc[]`/`xr[]` stay in registers: a
// runtime-length loop over a local array would be backed by scratch memory,
// which is the exact pessimization that cost the first `q4k.glsl` revision 2x
// (see the history note in that file).
//
// Bindings match the fused-matmul descriptor layout (`GpuDevice::desc_layout`)
// so the probe can reuse that pipeline layout unchanged.

layout(local_size_x = 256) in;

layout(constant_id = 0) const int TILE = 8;

layout(set = 0, binding = 0, std430) readonly buffer Weights { uint weights[]; };
layout(set = 0, binding = 1, std430) readonly buffer Xin { float xin[]; };
layout(set = 0, binding = 2, std430) writeonly buffer Yout { float yout[]; };
layout(set = 0, binding = 3, std140) uniform Params {
    uint mode;
    uint inner;    // inner-loop trip count
    uint wmask;    // index mask into `weights` (a power-of-two element count)
    uint xmask;    // index mask into `xin`
};

void main() {
    uint gid = gl_GlobalInvocationID.x;
    float sink = 0.0;

    if (mode == 0u) {
        // ---- FP32 FFMA peak ----------------------------------------------
        // 16 chains so the ~6-cycle FFMA latency is fully covered by
        // throughput; `b`/`c` come from memory so nothing is constant-folded.
        float b = xin[gid & xmask];
        float c = xin[(gid + 1u) & xmask];
        float a0=b, a1=b+1.0, a2=b+2.0, a3=b+3.0, a4=b+4.0, a5=b+5.0, a6=b+6.0, a7=b+7.0;
        float a8=b+8.0, a9=b+9.0, aa=b+10.0, ab=b+11.0, ac=b+12.0, ad=b+13.0, ae=b+14.0, af=b+15.0;
        // Unrolled 4x so the loop's own compare/branch is amortized to the same
        // degree as mode 2's (which issues 8*TILE FMAs per trip). Without this
        // the "peak" comes out *below* mode 2 at TILE >= 8, which would be
        // nonsense: the probe would be reporting loop overhead, not the ALUs.
        for (uint i = 0u; i < inner; i += 4u) {
            a0 = fma(a0, b, c); a1 = fma(a1, b, c); a2 = fma(a2, b, c); a3 = fma(a3, b, c);
            a4 = fma(a4, b, c); a5 = fma(a5, b, c); a6 = fma(a6, b, c); a7 = fma(a7, b, c);
            a8 = fma(a8, b, c); a9 = fma(a9, b, c); aa = fma(aa, b, c); ab = fma(ab, b, c);
            ac = fma(ac, b, c); ad = fma(ad, b, c); ae = fma(ae, b, c); af = fma(af, b, c);
            a0 = fma(a0, b, c); a1 = fma(a1, b, c); a2 = fma(a2, b, c); a3 = fma(a3, b, c);
            a4 = fma(a4, b, c); a5 = fma(a5, b, c); a6 = fma(a6, b, c); a7 = fma(a7, b, c);
            a8 = fma(a8, b, c); a9 = fma(a9, b, c); aa = fma(aa, b, c); ab = fma(ab, b, c);
            ac = fma(ac, b, c); ad = fma(ad, b, c); ae = fma(ae, b, c); af = fma(af, b, c);
            a0 = fma(a0, b, c); a1 = fma(a1, b, c); a2 = fma(a2, b, c); a3 = fma(a3, b, c);
            a4 = fma(a4, b, c); a5 = fma(a5, b, c); a6 = fma(a6, b, c); a7 = fma(a7, b, c);
            a8 = fma(a8, b, c); a9 = fma(a9, b, c); aa = fma(aa, b, c); ab = fma(ab, b, c);
            ac = fma(ac, b, c); ad = fma(ad, b, c); ae = fma(ae, b, c); af = fma(af, b, c);
            a0 = fma(a0, b, c); a1 = fma(a1, b, c); a2 = fma(a2, b, c); a3 = fma(a3, b, c);
            a4 = fma(a4, b, c); a5 = fma(a5, b, c); a6 = fma(a6, b, c); a7 = fma(a7, b, c);
            a8 = fma(a8, b, c); a9 = fma(a9, b, c); aa = fma(aa, b, c); ab = fma(ab, b, c);
            ac = fma(ac, b, c); ad = fma(ad, b, c); ae = fma(ae, b, c); af = fma(af, b, c);
        }
        sink = ((a0+a1)+(a2+a3)) + ((a4+a5)+(a6+a7))
             + ((a8+a9)+(aa+ab)) + ((ac+ad)+(ae+af));
    } else if (mode == 1u) {
        // ---- INT32 MAD peak ----------------------------------------------
        uint m = weights[gid & wmask] | 1u;
        uint n = weights[(gid + 1u) & wmask];
        uint u0=m, u1=m+1u, u2=m+2u, u3=m+3u, u4=m+4u, u5=m+5u, u6=m+6u, u7=m+7u;
        uint u8=m+8u, u9=m+9u, ua=m+10u, ub=m+11u, uc=m+12u, ud=m+13u, ue=m+14u, uf=m+15u;
        for (uint i = 0u; i < inner; i += 4u) {   // 4x unrolled, as in mode 0
            u0 = u0*m + n; u1 = u1*m + n; u2 = u2*m + n; u3 = u3*m + n;
            u4 = u4*m + n; u5 = u5*m + n; u6 = u6*m + n; u7 = u7*m + n;
            u8 = u8*m + n; u9 = u9*m + n; ua = ua*m + n; ub = ub*m + n;
            uc = uc*m + n; ud = ud*m + n; ue = ue*m + n; uf = uf*m + n;
            u0 = u0*m + n; u1 = u1*m + n; u2 = u2*m + n; u3 = u3*m + n;
            u4 = u4*m + n; u5 = u5*m + n; u6 = u6*m + n; u7 = u7*m + n;
            u8 = u8*m + n; u9 = u9*m + n; ua = ua*m + n; ub = ub*m + n;
            uc = uc*m + n; ud = ud*m + n; ue = ue*m + n; uf = uf*m + n;
            u0 = u0*m + n; u1 = u1*m + n; u2 = u2*m + n; u3 = u3*m + n;
            u4 = u4*m + n; u5 = u5*m + n; u6 = u6*m + n; u7 = u7*m + n;
            u8 = u8*m + n; u9 = u9*m + n; ua = ua*m + n; ub = ub*m + n;
            uc = uc*m + n; ud = ud*m + n; ue = ue*m + n; uf = uf*m + n;
            u0 = u0*m + n; u1 = u1*m + n; u2 = u2*m + n; u3 = u3*m + n;
            u4 = u4*m + n; u5 = u5*m + n; u6 = u6*m + n; u7 = u7*m + n;
            u8 = u8*m + n; u9 = u9*m + n; ua = ua*m + n; ub = ub*m + n;
            uc = uc*m + n; ud = ud*m + n; ue = ue*m + n; uf = uf*m + n;
        }
        uint s = ((u0+u1)+(u2+u3)) + ((u4+u5)+(u6+u7))
               + ((u8+u9)+(ua+ub)) + ((uc+ud)+(ue+uf));
        sink = float(s & 0xFFFFu) * 1e-6;
    } else {
        // ---- Tiled Q4_K GEMM inner loop ----------------------------------
        // One weight dword = 8 packed 4-bit weights. Each is unpacked once
        // (bitfieldExtract + I2F) and then reused across `TILE` tokens, which is
        // the whole point of a GEMM: at TILE=1 the unpack costs as much as the
        // arithmetic, at TILE=16 it is noise.
        float acc[TILE];
        float xr[TILE];
        for (int t = 0; t < TILE; t++) {
            acc[t] = 0.0;
            xr[t] = xin[(gid + uint(t) * 37u) & xmask];
        }
        for (uint i = 0u; i < inner; i++) {
            // Indexed so consecutive lanes read consecutive dwords (coalesced)
            // and the value changes per iteration, so the unpack cannot be
            // hoisted out of the loop. `wmask` keeps the footprint in L1/L2:
            // this probe is about ALU throughput, not the memory system.
            uint qw = weights[(gid + i * 256u) & wmask];
            for (int k = 0; k < 8; k++) {
                float w = float(bitfieldExtract(qw, k * 4, 4));
                for (int t = 0; t < TILE; t++) {
                    acc[t] = fma(w, xr[t], acc[t]);
                }
            }
        }
        for (int t = 0; t < TILE; t++) {
            sink += acc[t];
        }
    }

    // One store per invocation, outside the timed loop's critical path, purely
    // so the whole computation is observable and cannot be dead-code eliminated.
    yout[gid] = sink;
}
