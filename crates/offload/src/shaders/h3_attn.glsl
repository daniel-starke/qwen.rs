#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : enable
// MiniMax-H3-Image transformer attention on the GTX 1060
// (`offload::H3Attn`): every query over keys [0, nk), heads of 128, no
// mask. `imagegen::h3::npudit` gives the card query chunks of each block
// while the cores run the others.
//
// One workgroup = one head x R = 48 query rows (256 invocations). Keys are
// walked in BK = 64 blocks with an online softmax, all in fp32 (Pascal
// converts f16 at a quarter of its FMA rate, so K / V are widened once at
// staging):
//   S  = Q K^T, K staged in 4 chunks of 32 dims;
//   P  = exp(S - m), the row max / sum carried in `sm` / `sl`;
//   O += P V, V staged in 4 chunks of 16 keys, O in registers.
// Register tiles: QK 3 rows x 4 keys, PV 3 rows x 2 dim-quads per
// invocation. A warp covers 4 rows x 8 keys (or dim-quads), so a shared
// load has at most 8 distinct addresses and serves 4 invocations each.
// The staging is software-pipelined: the global loads of stage n + 1 are
// issued before stage n is computed (8 stages a key block).
// Shared memory: 47.3 KB of the 48 KB limit.
//
//   q  f32 [nq][ROW] (scaled here), rows 0 .. nq
//   k  f16 [nk][ROW] as packed quads
//   v  f16 [nk][ROW] as packed quads
//   o  f16 [nq][ROW] out
//   ROW = heads x 128 (push constant `row4` = ROW / 4)
#define HD4 32u
#define R 48u
#define BK 64u
#define VK 16u
#define KS (BK + 1u)
#define QS (HD4 + 1u)
#define SS (BK / 4u + 1u)

layout(local_size_x = 256) in;

layout(set = 0, binding = 0, std430) readonly buffer Qb { vec4 q4[]; };
layout(set = 0, binding = 1, std430) readonly buffer Kb { uvec2 k2[]; };
layout(set = 0, binding = 2, std430) readonly buffer Vb { uvec2 v2[]; };
layout(set = 0, binding = 3, std430) writeonly buffer Ob { uvec2 o2[]; };
layout(push_constant) uniform P {
    uint nq;
    uint nk;
    float scale;
    uint row4;
} pc;

shared vec4 Qs[R * QS];
// K chunk: [8 dim-quads][KS keys]; V chunk: [VK keys][32 dim-quads].
shared vec4 KV[8u * KS];
// Scores / probabilities as vec4 over 4 keys.
shared vec4 Ss[R * SS];
shared float sm[R];
shared float sl[R];
shared float sf[R];

vec4 half4(uvec2 u) {
    return vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y));
}

// Element `l` (0, 1) of an invocation's share of stage `st` of key block
// `blk`: stages 0..3 are K dims 32 st .. 32 st + 31 of 64 keys, 4..7 V
// keys 16 (st - 4) .. + 15 at all dims.
uvec2 stage_load(uint blk, uint st, uint l, uint tid, uint h) {
    uint idx = tid + l * 256u;
    uint t, off;
    if (st < 4u) {
        t = blk * BK + (idx >> 3u);
        off = st * 8u + (idx & 7u);
    } else {
        t = blk * BK + (st - 4u) * VK + (idx >> 5u);
        off = idx & 31u;
    }
    if (st >= 8u || t >= pc.nk) {
        return uvec2(0u);
    }
    uint a = t * pc.row4 + h * HD4 + off;
    return st < 4u ? k2[a] : v2[a];
}

void stage_store(uint st, uint l, uint tid, uvec2 raw) {
    uint idx = tid + l * 256u;
    if (st < 4u) {
        KV[(idx & 7u) * KS + (idx >> 3u)] = half4(raw);
    } else {
        KV[(idx >> 5u) * HD4 + (idx & 31u)] = half4(raw);
    }
}

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint lane = tid & 31u;
    uint w = tid >> 5u;
    uint t0 = gl_WorkGroupID.x * R;
    uint h = gl_WorkGroupID.y;

    for (uint i = tid; i < R * HD4; i += 256u) {
        uint r = i / HD4;
        uint d4 = i % HD4;
        Qs[r * QS + d4] = (t0 + r < pc.nq) ? q4[(t0 + r) * pc.row4 + h * HD4 + d4] * pc.scale : vec4(0.0);
    }
    if (tid < R) {
        sm[tid] = -1e30;
        sl[tid] = 0.0;
    }

    // tile roles: column c (keys c + 16 i, or dim-quads c + 16 m), rows
    // rg + 16 r
    uint c = (w & 1u) * 8u + (lane & 7u);
    uint rg = (w >> 1u) * 4u + (lane >> 3u);
    vec4 acc[3][2];
    [[unroll]] for (uint r = 0u; r < 3u; r++) {
        acc[r][0] = vec4(0.0);
        acc[r][1] = vec4(0.0);
    }

    uint nblk = (pc.nk + BK - 1u) / BK;
    uvec2 raw[2];
    raw[0] = stage_load(0u, 0u, 0u, tid, h);
    raw[1] = stage_load(0u, 0u, 1u, tid, h);
    for (uint blk = 0u; blk < nblk; blk++) {
        uint kb = blk * BK;
        float s[3][4];
        [[unroll]] for (uint r = 0u; r < 3u; r++) {
            [[unroll]] for (uint i = 0u; i < 4u; i++) {
                s[r][i] = 0.0;
            }
        }
        for (uint ch = 0u; ch < 4u; ch++) {
            barrier();
            stage_store(ch, 0u, tid, raw[0]);
            stage_store(ch, 1u, tid, raw[1]);
            barrier();
            raw[0] = stage_load(blk, ch + 1u, 0u, tid, h);
            raw[1] = stage_load(blk, ch + 1u, 1u, tid, h);
            [[unroll]] for (uint dq = 0u; dq < 8u; dq++) {
                vec4 ka[4];
                [[unroll]] for (uint i = 0u; i < 4u; i++) {
                    ka[i] = KV[dq * KS + c + 16u * i];
                }
                [[unroll]] for (uint r = 0u; r < 3u; r++) {
                    vec4 qv = Qs[(rg + 16u * r) * QS + ch * 8u + dq];
                    [[unroll]] for (uint i = 0u; i < 4u; i++) {
                        s[r][i] += dot(qv, ka[i]);
                    }
                }
            }
        }
        [[unroll]] for (uint r = 0u; r < 3u; r++) {
            [[unroll]] for (uint i = 0u; i < 4u; i++) {
                uint key = c + 16u * i;
                Ss[(rg + 16u * r) * SS + (key >> 2u)][key & 3u] = (kb + key < pc.nk) ? s[r][i] : -1e30;
            }
        }
        barrier();
        // online softmax: warp w owns rows 6 w .. 6 w + 5
        [[unroll]] for (uint i = 0u; i < 6u; i++) {
            uint r = w * 6u + i;
            float a = Ss[r * SS + (lane >> 2u)][lane & 3u];
            float b = Ss[r * SS + ((lane + 32u) >> 2u)][lane & 3u];
            float mo = sm[r];
            float mn = max(mo, subgroupMax(max(a, b)));
            float pa = exp(a - mn);
            float pb = exp(b - mn);
            Ss[r * SS + (lane >> 2u)][lane & 3u] = pa;
            Ss[r * SS + ((lane + 32u) >> 2u)][lane & 3u] = pb;
            float sum = subgroupAdd(pa + pb);
            if (lane == 0u) {
                float f = exp(mo - mn);
                sf[r] = f;
                sl[r] = sl[r] * f + sum;
                sm[r] = mn;
            }
        }
        barrier();
        [[unroll]] for (uint r = 0u; r < 3u; r++) {
            float f = sf[rg + 16u * r];
            acc[r][0] *= f;
            acc[r][1] *= f;
        }
        for (uint vc = 0u; vc < BK / VK; vc++) {
            barrier();
            stage_store(4u + vc, 0u, tid, raw[0]);
            stage_store(4u + vc, 1u, tid, raw[1]);
            barrier();
            uint nb = vc < 3u ? blk : blk + 1u;
            uint ns = vc < 3u ? 5u + vc : 0u;
            raw[0] = stage_load(nb, ns, 0u, tid, h);
            raw[1] = stage_load(nb, ns, 1u, tid, h);
            [[unroll]] for (uint kq = 0u; kq < VK; kq += 4u) {
                vec4 v[4][2];
                [[unroll]] for (uint j = 0u; j < 4u; j++) {
                    v[j][0] = KV[(kq + j) * HD4 + c];
                    v[j][1] = KV[(kq + j) * HD4 + c + 16u];
                }
                [[unroll]] for (uint r = 0u; r < 3u; r++) {
                    vec4 pr = Ss[(rg + 16u * r) * SS + ((vc * VK + kq) >> 2u)];
                    [[unroll]] for (uint m = 0u; m < 2u; m++) {
                        acc[r][m] += pr.x * v[0][m] + pr.y * v[1][m] + pr.z * v[2][m] + pr.w * v[3][m];
                    }
                }
            }
        }
    }

    [[unroll]] for (uint r = 0u; r < 3u; r++) {
        uint row = rg + 16u * r;
        if (t0 + row < pc.nq) {
            float inv = 1.0 / sl[row];
            [[unroll]] for (uint m = 0u; m < 2u; m++) {
                vec4 o = acc[r][m] * inv;
                o2[(t0 + row) * pc.row4 + h * HD4 + c + 16u * m] = uvec2(packHalf2x16(o.xy), packHalf2x16(o.zw));
            }
        }
    }
}
