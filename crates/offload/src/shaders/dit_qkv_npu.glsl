#version 450
// The Qwen-Image transformer's q / k / v from the QKV matmul's f32 output
// ([row][3 D]) as f16 rows where the NPU puts them (`gpu/ditblk.rs`): q at
// rows `rq`, k / v at rows `rk` of the pass' K / V buffers.
#define D 4096u
layout(local_size_x = 256) in;

layout(set = 0, binding = 0, std430) writeonly buffer Q { uint qb[]; };
layout(set = 0, binding = 1, std430) readonly buffer Y { vec2 y[]; };
layout(set = 0, binding = 4, std430) writeonly buffer K { uint kb[]; };
layout(set = 0, binding = 5, std430) writeonly buffer Vb { uint vb[]; };
layout(push_constant) uniform Pc {
    uint rq;
    uint rk;
};

void main() {
    // one f16 pair of q, k and v each: x = pair in the row, y = row
    uint p = gl_GlobalInvocationID.x, r = gl_WorkGroupID.y;
    uint b = r * (3u * D / 2u) + p;
    qb[(rq + r) * (D / 2u) + p] = packHalf2x16(y[b]);
    kb[(rk + r) * (D / 2u) + p] = packHalf2x16(y[b + D / 2u]);
    vb[(rk + r) * (D / 2u) + p] = packHalf2x16(y[b + D]);
}
