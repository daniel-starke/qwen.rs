#version 450
// Prefill attention of the Qwen3.8-Flash-Next NPU engine on the scan device
// (the iGPU, `offload::FlashAttn`): 24 query heads of 256 over 2 KV heads,
// each query over its own key list; every earlier position while the
// indexer selects all complete blocks (at most 512), else its 512 selected
// blocks of 4 keys plus the incomplete tail block. One pass, an online
// softmax over tiles of 256 keys.
// A workgroup is one (query, KV head, half of its 12 query heads), so a key
// and value row is read once for G = 6 heads. Per tile:
// - scores: each invocation scores 2 keys of one block for the 12 heads
//   (the queries in shared memory: every invocation reads the same
//   element, a broadcast, used for both keys);
// - the tile's max and sum per head as two-level reductions over shared
//   memory (8 G invocations, then G);
// - P V: each invocation accumulates 2 dims of every head over the tile's
//   values (consecutive invocations read consecutive 4 bytes of a value
//   row; a shared vec4 of 4 keys' weights per head and 4 keys).
// iGPU, 256 queries at 2048 keys (`flash_attention_bench`): 28 ms (34.7 with
// 512 selected blocks). The first form (64 invocations, one key per
// invocation, a serial softmax per head) ran 107 ms; this one with G = 12
// 40.7 (SIMD8, 768 B of spills); G = 4 32.8; 4 keys per invocation 27.9 /
// 43.1 (G = 6); 256 invocations with G = 12 (one dim each in P V, 49 KB of
// shared memory) 95.
// Keys come in blocks of 4 consecutive positions (the selection's blocks
// and the tail both start at a multiple of 4), so a pair or quad of keys
// is one index lookup. Keys past `pos` get weight 0 and read row `pos`.
//   q    [n][24][256] f16
//   k, v [cap][2][256] f16 (the layer's cache; positions [0, p0 + n) set)
//   sel  [n][513] u32: count (0xffffffff = every position from kmin, else
//        the number of listed blocks, at most 512), then the listed
//        blocks, ascending; the incomplete tail block always follows
//   o    [n][24][256] f16 out
//   lse  [n][24] f32 out: max + ln(sum) of the scaled scores, for merging
//        with the keys before kmin / outside the list (the NPU's blocks)
//   p    p0, n, scale (f32 bits), kmin
#extension GL_EXT_scalar_block_layout : enable
#define HD4 64u
#ifndef G
#define G 6u
#endif
#define NI 128u
#define TK 256u
layout(local_size_x = 128) in;
layout(set = 0, binding = 0, std430) readonly buffer Qb { uvec2 q2[]; };
layout(set = 0, binding = 1, std430) readonly buffer Kb { uvec2 k2[]; };
layout(set = 0, binding = 2, std430) readonly buffer Vb { uint v1[]; };
layout(set = 0, binding = 3, std430) readonly buffer Sb { uint sel[]; };
layout(set = 0, binding = 4, std430) writeonly buffer Ob { uint o1[]; };
layout(set = 0, binding = 5, std430) writeonly buffer Lb { float lse[]; };
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
    uint p0 = p[0];
    float scale = uintBitsToFloat(p[2]);
    uint t = gl_WorkGroupID.x;
    uint kh = gl_WorkGroupID.y;
    uint i = gl_LocalInvocationID.x;
    uint pos = p0 + t;
    uint qrow = t * 24u + kh * 12u + gl_WorkGroupID.z * G;
    for (uint r = i; r < G * HD4; r += NI) {
        qs[r] = half4(q2[(qrow + r / HD4) * HD4 + r % HD4]) * scale;
    }
    uint sb = t * 513u;
    uint raw = sel[sb];
    bool dense = raw == 0xffffffffu;
    uint kmin = p[3];
    uint nsel = dense ? 0u : raw;
    uint nb = (pos + 1u) / 4u;
    uint ns4 = nsel * 4u;
    uint nk = dense ? pos + 1u - kmin : ns4 + (pos + 1u - nb * 4u);
    vec2 acc[G];
    for (uint h = 0u; h < G; h++) {
        acc[h] = vec2(0.0);
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
        for (uint h = 0u; h < G; h++) {
            s0[h] = -1e30;
            s1[h] = -1e30;
        }
        if (j < nk) {
            uint u = dense ? kmin + j : (j < ns4 ? sel[sb + 1u + j / 4u] * 4u + (j & 3u) : nb * 4u + (j - ns4));
            uint kr0 = (u * 2u + kh) * HD4;
            uint kr1 = (min(u + 1u, pos) * 2u + kh) * HD4;
            for (uint h = 0u; h < G; h++) {
                s0[h] = 0.0;
                s1[h] = 0.0;
            }
            for (uint d = 0u; d < HD4; d++) {
                vec4 ka = half4(k2[kr0 + d]);
                vec4 kb = half4(k2[kr1 + d]);
                for (uint h = 0u; h < G; h++) {
                    vec4 qv = qs[h * HD4 + d];
                    s0[h] += dot(qv, ka);
                    s1[h] += dot(qv, kb);
                }
            }
            if (j + 1u >= nk) {
                for (uint h = 0u; h < G; h++) {
                    s1[h] = -1e30;
                }
            }
        }
        for (uint h = 0u; h < G; h++) {
            red[h * NI + i] = max(s0[h], s1[h]);
        }
        barrier();
        if (i < G * 8u) {
            uint h = i / 8u;
            uint r0 = h * NI + (i % 8u) * 16u;
            float v = -1e30;
            for (uint x = 0u; x < 16u; x++) {
                v = max(v, red[r0 + x]);
            }
            red2[i] = v;
        }
        barrier();
        if (i < G) {
            float m0 = mxs[i];
            float v = m0;
            for (uint x = 0u; x < 8u; x++) {
                v = max(v, red2[i * 8u + x]);
            }
            fac[i] = exp(m0 - v);
            mxs[i] = v;
        }
        barrier();
        for (uint h = 0u; h < G; h++) {
            float m = mxs[h];
            vec2 e = vec2(exp(s0[h] - m), exp(s1[h] - m));
            sc2[h * NI + i] = e;
            red[h * NI + i] = e.x + e.y;
        }
        barrier();
        if (i < G * 8u) {
            uint r0 = (i / 8u) * NI + (i % 8u) * 16u;
            float v = 0.0;
            for (uint x = 0u; x < 16u; x++) {
                v += red[r0 + x];
            }
            red2[i] = v;
        }
        barrier();
        if (i < G) {
            float v = 0.0;
            for (uint x = 0u; x < 8u; x++) {
                v += red2[i * 8u + x];
            }
            l = l * fac[i] + v;
        }
        for (uint h = 0u; h < G; h++) {
            acc[h] *= fac[h];
        }
        uint cnt = min(TK, nk - base);
        for (uint jj = 0u; jj < cnt; jj += 4u) {
            uint j2 = base + jj;
            uint u = dense ? kmin + j2 : (j2 < ns4 ? sel[sb + 1u + j2 / 4u] * 4u : nb * 4u + (j2 - ns4));
            vec2 v0 = unpackHalf2x16(v1[(u * 2u + kh) * 128u + i]);
            vec2 va = unpackHalf2x16(v1[(min(u + 1u, pos) * 2u + kh) * 128u + i]);
            vec2 vb = unpackHalf2x16(v1[(min(u + 2u, pos) * 2u + kh) * 128u + i]);
            vec2 vc = unpackHalf2x16(v1[(min(u + 3u, pos) * 2u + kh) * 128u + i]);
            uint si = jj / 2u;
            for (uint h = 0u; h < G; h++) {
                vec4 w = vec4(sc2[h * NI + si], sc2[h * NI + si + 1u]);
                acc[h] += w.x * v0 + w.y * va + w.z * vb + w.w * vc;
            }
        }
        barrier();
    }
    if (i < G) {
        fac[i] = 1.0 / l;
        lse[qrow + i] = mxs[i] + log(l);
    }
    barrier();
    for (uint h = 0u; h < G; h++) {
        o1[(qrow + h) * 128u + i] = packHalf2x16(acc[h] * fac[h]);
    }
}
