#version 450
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_clustered : enable
// `delta_prefill.glsl` on the NPU prefill's tensors: the recurrence between the
// `build_delta.py --cut pre` and `--cut post` graphs. Same scan (the state in
// registers, LANES invocations per value column, clustered adds over the key
// dim); what differs is the interface:
//   q, k  [N][16 * S] f16, l2-normalized, token-major (the pre graph's qn, kn);
//         value head h reads key head h / 3 (the export's order)
//   v     [N][48 * S] f16;  gb [N][96] f16: log decay g (h), then beta (48 + h)
//   o     [N][48 * S] f16 out, x oscale (the post graph wants the full graph's
//         internal scale)
//   st    [48][S][S] f32 (row = key dim), true scale, read at the start and
//         written at the end; it stays on the GPU between chunks
// f16 tensors are read as packed uints (no 16-bit storage feature needed). A
// lane group owns four adjacent value columns and lane l the key rows
// 4 (l + LANES m) + c (c < 4), so a step reads k and q from shared memory as
// R / 4 vec4 loads each and every load feeds 4 columns: the scan is bound by
// shared-memory messages (two columns and scalar loads: 3.1 ms per 256
// positions, this 1.95). `v` is two packed loads per step and lane 0 writes
// the four columns' `o` as two packed words (collecting `o` in shared
// memory for packing instead cost 2x on the iGPU). The state, 64 floats a
// lane, keeps the compiler at SIMD8 (SIMD16 spills: 7x slower).
#ifndef LANES
#define LANES 8
#endif
#define S 128
#define R (S / LANES)
#define COLS (4 * 256 / LANES)
#ifndef TB
#define TB 16
#endif
#define HV 48u
#define HK 16u
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) buffer St { float st[]; };
layout(set = 0, binding = 1, std430) readonly buffer Qq { uint qq[]; };
layout(set = 0, binding = 2, std430) readonly buffer Kk { uint kk[]; };
layout(set = 0, binding = 3, std430) readonly buffer Vv { uint vv[]; };
layout(set = 0, binding = 4, std430) writeonly buffer Oo { uint oo[]; };
layout(set = 0, binding = 5, std430) readonly buffer Gb { uint gb[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };

#define LD(buf, e) (unpackHalf2x16(buf[(e) >> 1u])[(e) & 1u])

shared vec4 sk_[TB * S / 4];
shared vec4 sq_[TB * S / 4];

void main() {
    uint N = p[0];
    float oscale = uintBitsToFloat(p[1]);
    uint nb = S / COLS;
    uint h = gl_WorkGroupID.x / nb;
    uint tid = gl_LocalInvocationID.x;
    uint l = tid % LANES;
    // columns j .. j + 3
    uint j = (gl_WorkGroupID.x % nb) * COLS + 4u * (tid / LANES);
    uint kh = h / (HV / HK);
    uint st0 = h * S * S + j;
    float qscale = 1.0 / sqrt(float(S));

    // lane l owns key rows 4 (l + LANES m) + c, c < 4: one vec4 load each
    vec4 s[R];
    for (uint m = 0u; m < R; m++) {
        uint r = st0 + (4u * (l + LANES * (m >> 2u)) + (m & 3u)) * S;
        s[m] = vec4(st[r], st[r + 1u], st[r + 2u], st[r + 3u]);
    }

    for (uint t0 = 0u; t0 < N; t0 += TB) {
        barrier();
        for (uint e = tid; e < TB * S / 4u; e += 256u) {
            uint t = t0 + e / (S / 4u);
            uint i = 4u * (e % (S / 4u));
            vec4 kv = vec4(0.0), qv = vec4(0.0);
            if (t < N) {
                uint b = (t * HK * S + kh * S + i) >> 1u;
                kv = vec4(unpackHalf2x16(kk[b]), unpackHalf2x16(kk[b + 1u]));
                qv = vec4(unpackHalf2x16(qq[b]), unpackHalf2x16(qq[b + 1u])) * qscale;
            }
            sk_[e] = kv;
            sq_[e] = qv;
        }
        barrier();
        uint tn = min(uint(TB), N - t0);
        for (uint tt = 0u; tt < tn; tt++) {
            uint t = t0 + tt;
            float g = exp(LD(gb, t * 2u * HV + h));
            float beta = LD(gb, t * 2u * HV + HV + h);
            uint vb = (t * HV * S + h * S + j) >> 1u;
            vec4 vj = vec4(unpackHalf2x16(vv[vb]), unpackHalf2x16(vv[vb + 1u]));
            uint b = tt * (S / 4u) + l;
            vec4 k4[R / 4], q4[R / 4];
            for (uint m = 0u; m < R / 4u; m++) {
                k4[m] = sk_[b + LANES * m];
                q4[m] = sq_[b + LANES * m];
            }
            vec4 acc = vec4(0.0);
            for (uint m = 0u; m < R; m++) {
                s[m] *= g;
                acc += s[m] * k4[m >> 2u][m & 3u];
            }
            acc = subgroupClusteredAdd(acc, LANES);
            vec4 d = (vj - acc) * beta;
            vec4 o = vec4(0.0);
            for (uint m = 0u; m < R; m++) {
                s[m] += k4[m >> 2u][m & 3u] * d;
                o += s[m] * q4[m >> 2u][m & 3u];
            }
            o = subgroupClusteredAdd(o, LANES);
            if (l == 0u) {
                oo[vb] = packHalf2x16(o.xy * oscale);
                oo[vb + 1u] = packHalf2x16(o.zw * oscale);
            }
        }
    }
    for (uint m = 0u; m < R; m++) {
        uint r = st0 + (4u * (l + LANES * (m >> 2u)) + (m & 3u)) * S;
        st[r] = s[m].x;
        st[r + 1u] = s[m].y;
        st[r + 2u] = s[m].z;
        st[r + 3u] = s[m].w;
    }
}
