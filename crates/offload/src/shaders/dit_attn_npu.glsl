#version 450
// Attention of the Qwen-Image transformer on the scan device (the iGPU,
// `offload::DitAttn`): 32 heads of 128, every query over keys [0, nk), no
// mask (the target rows; `imagegen::qimage::npudit` gives the iGPU a share of
// each block's query chunks while the cores run the rest). One pass, an
// online softmax over tiles of 128 keys; the layout of
// `flash_attn_npu.glsl` with G queries of one head per workgroup in place
// of G heads of one query.
// A workgroup is (G queries, head), 64 invocations. Per tile:
// - scores: each invocation scores 2 keys for the G queries (the queries
//   in shared memory, scaled: every invocation reads the same element, a
//   broadcast, used for both keys); key rows come through the caches;
// - the tile's max and sum per query as two-level reductions over shared
//   memory (8 G invocations, then G);
// - P V: each invocation accumulates 2 dims of every query over the
//   tile's values (consecutive invocations read consecutive 4 bytes of a
//   value row; a shared vec4 of 4 keys' weights per query).
// The first form (a tile of 32 queries x 64 keys, keys and values staged
// in 43.5 KB of shared memory, 4 x 4 scores per invocation) ran 0.13
// TFLOP/s: one workgroup per Xe core.
//   q    [*][32][128] f16, the queries at rows q0 .. q0 + nq
//   k, v [nk][32][128] f16
//   o    [*][32][128] f16 out, rows o0 .. o0 + nq
//   p    q0, nq, nk, scale (f32 bits), o0
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_control_flow_attributes : enable
#define HD4 32u
#define ROW4 1024u
#ifndef G
#define G 8u
#endif
#define NI 64u
#define TK 128u
layout(local_size_x = 64) in;
layout(set = 0, binding = 0, std430) readonly buffer Qb { uvec2 q2[]; };
layout(set = 0, binding = 1, std430) readonly buffer Kb { uvec2 k2[]; };
layout(set = 0, binding = 2, std430) readonly buffer Vb { uint v1[]; };
layout(set = 0, binding = 3, std430) writeonly buffer Ob { uint o1[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };

shared vec4 qs[G * HD4];
shared vec2 sc2[G * NI];
shared float red[G * NI];
shared float red2[G * 8u];
shared float mxs[G];
shared float fac[G];

vec4 half4(uvec2 u) {
    return vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y));
}

void main() {
    uint q0 = p[0];
    uint nq = p[1];
    uint nk = p[2];
    float scale = uintBitsToFloat(p[3]);
    uint t0 = gl_WorkGroupID.x * G;
    uint h = gl_WorkGroupID.y;
    uint i = gl_LocalInvocationID.x;
    for (uint r = i; r < G * HD4; r += NI) {
        uint qr = t0 + r / HD4;
        qs[r] = qr < nq ? half4(q2[(q0 + qr) * ROW4 + h * HD4 + r % HD4]) * scale : vec4(0.0);
    }
    vec2 acc[G];
    [[unroll]] for (uint g = 0u; g < G; g++) {
        acc[g] = vec2(0.0);
    }
    float l = 0.0;
    if (i < G) {
        mxs[i] = -1e30;
    }
    barrier();
    for (uint base = 0u; base < nk; base += TK) {
        uint j = base + 2u * i;
        float s0[G];
        float s1[G];
        [[unroll]] for (uint g = 0u; g < G; g++) {
            s0[g] = -1e30;
            s1[g] = -1e30;
        }
        if (j < nk) {
            uint kr0 = j * ROW4 + h * HD4;
            uint kr1 = min(j + 1u, nk - 1u) * ROW4 + h * HD4;
            [[unroll]] for (uint g = 0u; g < G; g++) {
                s0[g] = 0.0;
                s1[g] = 0.0;
            }
            for (uint d = 0u; d < HD4; d++) {
                vec4 ka = half4(k2[kr0 + d]);
                vec4 kb = half4(k2[kr1 + d]);
                [[unroll]] for (uint g = 0u; g < G; g++) {
                    vec4 qv = qs[g * HD4 + d];
                    s0[g] += dot(qv, ka);
                    s1[g] += dot(qv, kb);
                }
            }
            if (j + 1u >= nk) {
                [[unroll]] for (uint g = 0u; g < G; g++) {
                    s1[g] = -1e30;
                }
            }
        }
        [[unroll]] for (uint g = 0u; g < G; g++) {
            red[g * NI + i] = max(s0[g], s1[g]);
        }
        barrier();
        if (i < G * 8u) {
            uint g = i / 8u;
            uint r0 = g * NI + (i % 8u) * 8u;
            float v = -1e30;
            [[unroll]] for (uint x = 0u; x < 8u; x++) {
                v = max(v, red[r0 + x]);
            }
            red2[i] = v;
        }
        barrier();
        if (i < G) {
            float m0 = mxs[i];
            float v = m0;
            [[unroll]] for (uint x = 0u; x < 8u; x++) {
                v = max(v, red2[i * 8u + x]);
            }
            fac[i] = exp(m0 - v);
            mxs[i] = v;
        }
        barrier();
        [[unroll]] for (uint g = 0u; g < G; g++) {
            float m = mxs[g];
            vec2 e = vec2(exp(s0[g] - m), exp(s1[g] - m));
            sc2[g * NI + i] = e;
            red[g * NI + i] = e.x + e.y;
        }
        barrier();
        if (i < G * 8u) {
            uint r0 = (i / 8u) * NI + (i % 8u) * 8u;
            float v = 0.0;
            [[unroll]] for (uint x = 0u; x < 8u; x++) {
                v += red[r0 + x];
            }
            red2[i] = v;
        }
        barrier();
        if (i < G) {
            float v = 0.0;
            [[unroll]] for (uint x = 0u; x < 8u; x++) {
                v += red2[i * 8u + x];
            }
            l = l * fac[i] + v;
        }
        [[unroll]] for (uint g = 0u; g < G; g++) {
            acc[g] *= fac[g];
        }
        uint cnt = min(TK, nk - base);
        for (uint jj = 0u; jj < cnt; jj += 4u) {
            uint u = base + jj;
            vec2 v0 = unpackHalf2x16(v1[u * 2048u + h * 64u + i]);
            vec2 va = unpackHalf2x16(v1[min(u + 1u, nk - 1u) * 2048u + h * 64u + i]);
            vec2 vb = unpackHalf2x16(v1[min(u + 2u, nk - 1u) * 2048u + h * 64u + i]);
            vec2 vc = unpackHalf2x16(v1[min(u + 3u, nk - 1u) * 2048u + h * 64u + i]);
            uint si = jj / 2u;
            [[unroll]] for (uint g = 0u; g < G; g++) {
                vec4 w = vec4(sc2[g * NI + si], sc2[g * NI + si + 1u]);
                acc[g] += w.x * v0 + w.y * va + w.z * vb + w.w * vc;
            }
        }
        barrier();
    }
    if (i < G) {
        fac[i] = 1.0 / l;
    }
    barrier();
    [[unroll]] for (uint g = 0u; g < G; g++) {
        if (t0 + g < nq) {
            o1[(p[4] + t0 + g) * 2048u + h * 64u + i] = packHalf2x16(acc[g] * fac[g]);
        }
    }
}
