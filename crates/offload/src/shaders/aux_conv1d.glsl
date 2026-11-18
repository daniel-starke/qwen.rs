#version 450
#extension GL_EXT_scalar_block_layout : enable
// SSM conv1d: depthwise 1D conv over conv_kernel rows of `conv_dim` channels
// (weights w[f + c*ksz]) applied to the rolling window [state[0..ksz-1]; qkv],
// followed by SiLU. `state` is the persistent (ksz-1)-row window, updated in
// place each call: new_state[f] = old_state[f+1] (f < ksz-2), last row = qkv.
// One invocation per channel `c`.
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) buffer State { float st[]; }; // (ksz-1)*conv_dim, read-write
layout(set = 0, binding = 1, std430) readonly buffer Qkv { float qkv[]; }; // conv_dim
layout(set = 0, binding = 2, std430) readonly buffer Cw { float cw[]; }; // ksz*conv_dim
layout(set = 0, binding = 3, std430) writeonly buffer Cout { float cout[]; }; // conv_dim
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };
void main() {
    uint cd = p[0];
    uint ksz = p[1];
    uint c = gl_GlobalInvocationID.x;
    if (c >= cd) return;
    float acc = 0.0;
    for (uint f = 0u; f < ksz; f++) {
        float xv;
        if (f < ksz - 1u) { xv = st[f * cd + c]; }
        else { xv = qkv[c]; }
        acc += cw[f + c * ksz] * xv;
    }
    float s = acc / (1.0 + exp(-acc)); // silu
    cout[c] = s;
    // update persistent state (ascending f: read st[f+1] before writing st[f])
    for (uint f = 0u; f < ksz - 1u; f++) {
        if (f + 1u < ksz - 1u) { st[f * cd + c] = st[(f + 1u) * cd + c]; }
        else { st[f * cd + c] = qkv[c]; }
    }
}
