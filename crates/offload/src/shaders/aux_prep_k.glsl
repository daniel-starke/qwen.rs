#version 450
#extension GL_EXT_scalar_block_layout : enable
// Full-attention K preparation: per kv-head, RMSNorm(k) with k_norm, apply MRoPE,
// write roped kn. One workgroup per kv head.
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer K { float k[]; };
layout(set = 0, binding = 1, std430) readonly buffer Kw { float kw[]; };
layout(set = 0, binding = 2, std430) writeonly buffer Kn { float kn[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };
shared float invv;
void main() {
    uint n_kv = p[0];
    uint head_dim = p[1];
    uint pos = p[2];
    uint halfn = p[3];
    float eps = uintBitsToFloat(p[4]);
    float rfb = uintBitsToFloat(p[5]);
    float rfs = uintBitsToFloat(p[6]);
    float raf = uintBitsToFloat(p[7]);
    float sect_dims = uintBitsToFloat(p[9]);
    uint h = gl_WorkGroupID.x;
    uint d = gl_LocalInvocationID.x;
    uint n_rot = 2u * halfn;

    float kc = k[h * head_dim + d];
    if (d == 0u) {
        float ss = 0.0;
        for (uint i = 0u; i < head_dim; i++) { float v = k[h * head_dim + i]; ss += v * v; }
        invv = 1.0 / sqrt(ss / float(head_dim) + eps);
    }
    barrier();
    float knorm = kc * invv * kw[d];

    if (d < n_rot) {
        float x0, x1;
        if (d < halfn) { x0 = knorm; x1 = k[h * head_dim + (d + halfn)] * invv * kw[d + halfn]; }
        else { x0 = k[h * head_dim + (d - halfn)] * invv * kw[d - halfn]; x1 = knorm; }
        uint pidx = (d < halfn) ? d : (d - halfn);
        uint sector = pidx % uint(sect_dims);
        float base = float(pos);
        float theta_scale = pow(rfb, -2.0 / float(n_rot));
        float theta = rfs * base * pow(theta_scale, float(pidx));
        float c = cos(theta) * raf;
        float s = sin(theta) * raf;
        if (d < halfn) {
            kn[h * head_dim + d] = x0 * c - x1 * s;
            kn[h * head_dim + d + halfn] = x0 * s + x1 * c;
        } else {
            kn[h * head_dim + d - halfn] = x0 * c - x1 * s;
            kn[h * head_dim + d] = x0 * s + x1 * c;
        }
    } else {
        kn[h * head_dim + d] = knorm;
    }
}
