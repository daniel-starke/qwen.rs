#version 450
#extension GL_EXT_scalar_block_layout : enable
// Per-head RMSNorm of `o` (length Hv*S) using a shared norm weight `w` of length
// S, repeated per head: for each head h, inv = 1/sqrt(ss/S + eps) where
// ss = sum_{d<S} o[h*S+d]^2; out[h*S+d] = o[h*S+d]*inv*w[d]. Matches the CPU
// SSM `no` computation exactly (rms_norm over each S-length head slice).
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer Oin { float o[]; };
layout(set = 0, binding = 1, std430) readonly buffer Wn { float w[]; };
layout(set = 0, binding = 2, std430) writeonly buffer Nout { float no[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };
void main() {
    uint S = p[0];
    uint Hv = p[1];
    uint eps = p[2];
    uint h = gl_GlobalInvocationID.x;
    if (h >= Hv) return;
    uint base = h * S;
    float ss = 0.0;
    for (uint d = 0u; d < S; d++) { float v = o[base + d]; ss += v * v; }
    float inv = 1.0 / sqrt(ss / float(S) + uintBitsToFloat(eps));
    for (uint d = 0u; d < S; d++) { no[base + d] = o[base + d] * inv * w[d]; }
}
