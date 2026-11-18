#version 450
#extension GL_EXT_scalar_block_layout : enable
// Causal attention for one full-attention head (decode, single query). One
// WORKGROUP of 256 invocations per head: the key dot-products and the softmax
// reductions are spread over the workgroup, then each invocation accumulates a
// slice of the output dimension. Appends nothing (k/v are appended separately by
// the block before this runs). Reads the GPU KV cache, softmaxes over [0..pos],
// accumulates v, and gates the result by sigmoid(gate).
//
// The previous version was `local_size_x = 1` with one invocation per head: 24
// threads for the whole dispatch, with a per-thread O(tseen*head_dim) walk. That
// is linear in context length on a single lane and dominated fused full-attn
// decode; here the same work is 256-way parallel per head.
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer Q { float q[]; };
layout(set = 0, binding = 1, std430) readonly buffer Kc { float kc[]; };
layout(set = 0, binding = 2, std430) readonly buffer Vc { float vc[]; };
layout(set = 0, binding = 3, std430) readonly buffer Gate { float gate[]; };
layout(set = 0, binding = 4, std430) writeonly buffer Aout { float aout[]; };
layout(set = 0, binding = 5, std430) buffer Scratch { float scratch[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };
shared float sh[256];
shared float s_max;
shared float s_sum;
void main() {
    uint n_head = p[0];
    uint head_dim = p[1];
    uint n_kv = p[2];
    uint pos = p[3];
    float scale = uintBitsToFloat(p[4]);
    uint max_ctx = p[5];
    uint h = gl_WorkGroupID.x;
    uint lid = gl_LocalInvocationID.x;
    uint g = h / (n_head / n_kv);
    uint tseen = pos + 1u;

    // 1. scores + running max
    float local_max = -1e30;
    for (uint t = lid; t < tseen; t += 256u) {
        float sc = 0.0;
        for (uint d = 0u; d < head_dim; d++) {
            sc += q[h * head_dim + d] * kc[t * n_kv * head_dim + g * head_dim + d];
        }
        sc *= scale;
        scratch[h * max_ctx + t] = sc;
        local_max = max(local_max, sc);
    }
    sh[lid] = local_max;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid < stride) { sh[lid] = max(sh[lid], sh[lid + stride]); }
        barrier();
    }
    if (lid == 0u) { s_max = sh[0]; }
    barrier();

    // 2. sum of exp
    float local_sum = 0.0;
    for (uint t = lid; t < tseen; t += 256u) {
        float e = exp(scratch[h * max_ctx + t] - s_max);
        scratch[h * max_ctx + t] = e;
        local_sum += e;
    }
    sh[lid] = local_sum;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid < stride) { sh[lid] += sh[lid + stride]; }
        barrier();
    }
    if (lid == 0u) { s_sum = sh[0]; }
    barrier();

    // 3. weighted sum over v, one output element per invocation
    float inv_sum = 1.0 / s_sum;
    for (uint d = lid; d < head_dim; d += 256u) {
        float acc = 0.0;
        for (uint t = 0u; t < tseen; t++) {
            acc += scratch[h * max_ctx + t] * vc[t * n_kv * head_dim + g * head_dim + d];
        }
        acc *= inv_sum;
        float sig = 1.0 / (1.0 + exp(-gate[h * head_dim + d]));
        aout[h * head_dim + d] = acc * sig;
    }
}
