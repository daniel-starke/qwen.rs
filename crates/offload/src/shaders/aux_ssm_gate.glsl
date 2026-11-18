#version 450
#extension GL_EXT_scalar_block_layout : enable
// SSM gate: gate_h[h] = exp(softplus(alpha_raw[h] + ssm_dt[h]) * ssm_a[h]);
// written into the high half of the combined beta+gate buffer `bg` (length
// 2*Hv) consumed by the delta-net shader. One invocation per head.
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer Ain { float a[]; };
layout(set = 0, binding = 1, std430) readonly buffer Dt { float dt[]; };
layout(set = 0, binding = 2, std430) readonly buffer Sa { float sa[]; };
layout(set = 0, binding = 3, std430) writeonly buffer Bg { float bg[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };
void main() {
    uint Hv = p[0];
    uint h = gl_GlobalInvocationID.x;
    if (h >= Hv) return;
    float x = a[h] + dt[h];
    float sp = (x > 20.0) ? x : log(1.0 + exp(x));
    bg[Hv + h] = exp(sp * sa[h]);
}
