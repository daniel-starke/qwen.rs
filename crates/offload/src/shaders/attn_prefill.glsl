#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require
// Causal GQA attention for a prefill chunk (flash-attention style, fp32),
// the GPU half of `OffloadManager::attn_prefill`: the CPU runs the other KV
// groups through `quant::attn_causal_groups` at the same time.
//
// One workgroup = one KV group x QT = 4 query positions = R = 24 query rows
// (the group's 6 heads per position), so every K/V byte staged into shared
// memory serves 24 rows. Keys are walked in BK = 64 blocks with an online
// softmax:
//   S  = Q K^T for the block, K staged in 4 chunks of 64 dims;
//   P  = exp(S - m), row max / sum carried in `sm` / `sl`;
//   O += P V, V staged in 4 chunks of 16 keys, O in registers.
// Everything stays fp32 in shared memory: Pascal converts f16 at a fraction
// of its FMA rate, so the f16 cache is widened once at staging.
// Shared memory is ~47 KB of the 48 KB limit.
//
// Buffers hold only this dispatch's `ng` KV groups (the host packs them):
// Q and out f32 [nq][ng * HPG * HD], K / V the f16 cache rows [t][ng * HD]
// as packed pairs. Query position i is absolute position t0 + i.
#define HD 256u
#define HPG 6u
#define QT 4u
#define R 24u
#define BK 64u
#define VK 16u
#define KS (BK + 1u)

layout(local_size_x = 256) in;

layout(set = 0, binding = 0, std430) readonly buffer Qb { vec4 q4[]; };
layout(set = 0, binding = 1, std430) readonly buffer Kb { uvec2 k2[]; };
layout(set = 0, binding = 2, std430) readonly buffer Vb { uvec2 v2[]; };
layout(set = 0, binding = 3, std430) writeonly buffer Ob { vec4 o4[]; };
layout(push_constant) uniform P {
    uint nq;
    uint t0;
    uint ng;
    float scale;
} pc;

shared vec4 Qs[R][HD / 4u];
// K chunk: [16 dim-quads][KS keys]; V chunk: [VK keys][64 dim-quads].
shared vec4 KV[16u * KS];
// Scores / probabilities as vec4 over 4 keys, so PV reads 4 at once; the
// extra quad per row staggers the rows across banks.
shared vec4 Ss[R][BK / 4u + 1u];
shared float sm[R];
shared float sl[R];
shared float sf[R];

vec4 half4(uvec2 u) {
    return vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y));
}

// The K/V staging is software-pipelined: a block is 8 stages (K dim-chunks
// 0..3, then V key-chunks 0..3), and the global loads of stage n + 1 are
// issued into registers before stage n is computed, so their latency hides
// behind the arithmetic. Without it the kernel waited on every load (removing
// the K loads alone saved 19 %, the QK math only 15 %).
// Element `l` of a thread's share of stage `st` of block `blk`.
uvec2 stage_load(uint blk, uint st, uint l, uint tid, uint kend, uint ks4, uint g) {
    uint idx = tid + l * 256u;
    uint t, off;
    if (st < 4u) {
        t = blk * BK + (idx >> 4u);
        off = st * 16u + (idx & 15u);
    } else {
        t = blk * BK + (st - 4u) * VK + (idx >> 6u);
        off = idx & 63u;
    }
    if (t >= kend) {
        return uvec2(0u);
    }
    return st < 4u ? k2[t * ks4 + g * (HD / 4u) + off] : v2[t * ks4 + g * (HD / 4u) + off];
}

void stage_store(uint st, uint l, uint tid, uvec2 raw) {
    uint idx = tid + l * 256u;
    if (st < 4u) {
        KV[(idx & 15u) * KS + (idx >> 4u)] = half4(raw);
    } else {
        KV[(idx >> 6u) * 64u + (idx & 63u)] = half4(raw);
    }
}

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint tb = gl_WorkGroupID.x * QT;
    uint g = gl_WorkGroupID.y;                 // group within the packed buffers
    uint qs4 = pc.ng * HPG * HD / 4u;          // q / out row stride, vec4
    uint ks4 = pc.ng * HD / 4u;                // cache row stride, uvec2 (4 halves)

    for (uint i = tid; i < R * (HD / 4u); i += 256u) {
        uint r = i / (HD / 4u);
        uint d4 = i % (HD / 4u);
        uint p = r / HPG;
        uint h = r % HPG;
        Qs[r][d4] = (tb + p < pc.nq)
            ? q4[(tb + p) * qs4 + (g * HPG + h) * (HD / 4u) + d4] * pc.scale
            : vec4(0.0);
    }
    if (tid < R) {
        sm[tid] = -1e30;
        sl[tid] = 0.0;
    }

    // QK roles: keys j and j + 32, rows (pos qp, heads qh0..qh0+2).
    uint j = tid & 31u;
    uint qr = tid >> 5u;                       // 0..7
    uint qp = qr >> 1u;
    uint qh0 = (qr & 1u) * 3u;
    // PV roles: dim-quad d4, position pp, all 6 heads.
    uint d4 = tid & 63u;
    uint pp = tid >> 6u;
    vec4 acc[HPG];
    for (uint h = 0u; h < HPG; h++) {
        acc[h] = vec4(0.0);
    }

    uint qlast = min(tb + QT, pc.nq);          // exclusive
    uint kend = pc.t0 + qlast;                 // keys [0, kend)
    uint nblk = (kend + BK - 1u) / BK;

    uvec2 raw[4];
    for (uint l = 0u; l < 4u; l++) {
        raw[l] = stage_load(0u, 0u, l, tid, kend, ks4, g);
    }
    for (uint blk = 0u; blk < nblk; blk++) {
        uint kb = blk * BK;
        float s0[3], s1[3];
        for (uint h = 0u; h < 3u; h++) {
            s0[h] = 0.0;
            s1[h] = 0.0;
        }
        for (uint c = 0u; c < 4u; c++) {
            barrier();
            for (uint l = 0u; l < 4u; l++) {
                stage_store(c, l, tid, raw[l]);
            }
            barrier();
            for (uint l = 0u; l < 4u; l++) {
                raw[l] = stage_load(blk, c + 1u, l, tid, kend, ks4, g);
            }
            for (uint dq = 0u; dq < 16u; dq++) {
                vec4 ka = KV[dq * KS + j];
                vec4 kc = KV[dq * KS + j + 32u];
                for (uint h = 0u; h < 3u; h++) {
                    vec4 qv = Qs[qp * HPG + qh0 + h][c * 16u + dq];
                    s0[h] += dot(qv, ka);
                    s1[h] += dot(qv, kc);
                }
            }
        }
        // Causal mask: query position tb + qp sees keys <= t0 + tb + qp.
        uint qabs = pc.t0 + tb + qp;
        for (uint h = 0u; h < 3u; h++) {
            Ss[qp * HPG + qh0 + h][j >> 2u][j & 3u] = (kb + j <= qabs) ? s0[h] : -1e30;
            Ss[qp * HPG + qh0 + h][(j + 32u) >> 2u][j & 3u] = (kb + j + 32u <= qabs) ? s1[h] : -1e30;
        }
        barrier();
        // Online softmax: warp w owns rows 3w .. 3w + 2.
        {
            uint w = tid >> 5u;
            uint lane = tid & 31u;
            for (uint i = 0u; i < 3u; i++) {
                uint r = w * 3u + i;
                float a = Ss[r][lane >> 2u][lane & 3u];
                float b = Ss[r][(lane + 32u) >> 2u][lane & 3u];
                float mo = sm[r];
                float mn = max(mo, subgroupMax(max(a, b)));
                float pa = exp(a - mn);
                float pb = exp(b - mn);
                Ss[r][lane >> 2u][lane & 3u] = pa;
                Ss[r][(lane + 32u) >> 2u][lane & 3u] = pb;
                float sum = subgroupAdd(pa + pb);
                if (lane == 0u) {
                    float f = exp(mo - mn);
                    sf[r] = f;
                    sl[r] = sl[r] * f + sum;
                    sm[r] = mn;
                }
            }
        }
        barrier();
        for (uint h = 0u; h < HPG; h++) {
            acc[h] *= sf[pp * HPG + h];
        }
        for (uint vc = 0u; vc < BK / VK; vc++) {
            barrier();
            for (uint l = 0u; l < 4u; l++) {
                stage_store(4u + vc, l, tid, raw[l]);
            }
            barrier();
            for (uint l = 0u; l < 4u; l++) {
                raw[l] = (vc < 3u)
                    ? stage_load(blk, 5u + vc, l, tid, kend, ks4, g)
                    : stage_load(blk + 1u, 0u, l, tid, kend, ks4, g);
            }
            for (uint kq = 0u; kq < VK; kq += 4u) {
                vec4 v0 = KV[kq * 64u + d4];
                vec4 v1 = KV[(kq + 1u) * 64u + d4];
                vec4 v2 = KV[(kq + 2u) * 64u + d4];
                vec4 v3 = KV[(kq + 3u) * 64u + d4];
                for (uint h = 0u; h < HPG; h++) {
                    uint r = pp * HPG + h;
                    uint c = vc * VK + kq;
                    vec4 pr = Ss[r][c >> 2u];
                    acc[h] += pr.x * v0 + pr.y * v1 + pr.z * v2 + pr.w * v3;
                }
            }
        }
    }

    if (tb + pp < pc.nq) {
        for (uint h = 0u; h < HPG; h++) {
            float inv = 1.0 / sl[pp * HPG + h];
            o4[(tb + pp) * qs4 + (g * HPG + h) * (HD / 4u) + d4] = acc[h] * inv;
        }
    }
}
