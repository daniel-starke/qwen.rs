#version 450
// f32 x f32 matvec/matmul: one output row per thread.
//
// `weights` is declared `uint` and bit-cast rather than `float` so the binding
// is byte-identical to the other kernels' (and to what `upload` writes).
//
// One row per thread, `local_size_x = 64`: see `GpuDevice::grid_x`, which is the
// single source of truth for the dispatch grid, and `rows_per_wg` next to it.

layout(local_size_x = 64) in;

layout(set = 0, binding = 0, std430) readonly buffer Weights { uint weights[]; };
layout(set = 0, binding = 1, std430) readonly buffer Xin { float xin[]; };
layout(set = 0, binding = 2, std430) writeonly buffer Yout { float yout[]; };
layout(set = 0, binding = 3, std140) uniform Params {
    uint out_dim;
    uint in_dim;
    uint seq;
};

void main() {
    uint o = gl_GlobalInvocationID.x;
    uint si = gl_GlobalInvocationID.y;
    if (o >= out_dim || si >= seq) {
        return;
    }
    float acc = 0.0;
    uint row_base = o * in_dim;
    for (uint k = 0u; k < in_dim; k++) {
        acc += uintBitsToFloat(weights[row_base + k]) * xin[si * in_dim + k];
    }
    yout[si * out_dim + o] = acc;
}
