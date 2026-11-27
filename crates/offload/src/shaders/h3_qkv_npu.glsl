#version 450
// MiniMax-H3-Image's raw q / k / v from the QKV matmul's f32 output
// ([row][A + 2 KV], A = the attention width, KV = the width of k and v)
// as f16 rows where the NPU puts them (`gpu/h3blk.rs`): q, k and v at row
// `rq` of their buffers.
#ifndef A
#define A 7168u
#endif
#ifndef KV
#define KV A
#endif
layout(local_size_x = 256) in;

layout(set = 0, binding = 0, std430) writeonly buffer Q { uint qb[]; };
layout(set = 0, binding = 1, std430) readonly buffer Y { vec2 y[]; };
layout(set = 0, binding = 4, std430) writeonly buffer K { uint kb[]; };
layout(set = 0, binding = 5, std430) writeonly buffer Vb { uint vb[]; };
layout(push_constant) uniform Pc {
    uint rq;
};

void main() {
    // one f16 pair: x = pair in the row (q, then k, then v), y = row
    uint p = gl_GlobalInvocationID.x, r = gl_WorkGroupID.y;
    vec2 v = y[r * ((A + 2u * KV) / 2u) + p];
    if (p < A / 2u) {
        qb[(rq + r) * (A / 2u) + p] = packHalf2x16(v);
    } else if (p < A / 2u + KV / 2u) {
        kb[(rq + r) * (KV / 2u) + p - A / 2u] = packHalf2x16(v);
    } else {
        vb[(rq + r) * (KV / 2u) + p - A / 2u - KV / 2u] = packHalf2x16(v);
    }
}
