#version 450
#extension GL_EXT_scalar_block_layout : enable
// Full-attention Q preparation: per head, extract qcur/gate, RMSNorm(qcur) with
// q_norm, apply MRoPE, write roped qn + gate. One workgroup per head.
// `qfull` is every head's query, then every head's gate (the loader reorders
// `attn_q`'s rows that way, see `Model::q_gate_rows_split`).
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer Qfull { float qfull[]; };
layout(set = 0, binding = 1, std430) readonly buffer Qw { float qw[]; };
layout(set = 0, binding = 2, std430) writeonly buffer Gate { float gate[]; };
layout(set = 0, binding = 3, std430) writeonly buffer Qn { float qn[]; };
layout(set = 0, binding = 8, std430) uniform Params { uint p[16]; };
shared float invv;
void main() {
    uint n_head = p[0];
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

    float qc = qfull[h * head_dim + d];
    float g = qfull[(n_head + h) * head_dim + d];
    if (d == 0u) {
        float ss = 0.0;
        for (uint i = 0u; i < head_dim; i++) { float v = qfull[h * head_dim + i]; ss += v * v; }
        invv = 1.0 / sqrt(ss / float(head_dim) + eps);
    }
    barrier();
    float qnorm = qc * invv * qw[d];

    if (d < n_rot) {
        float x0, x1;
        if (d < halfn) { x0 = qnorm; x1 = qfull[h * head_dim + (d + halfn)] * invv * qw[d + halfn]; }
        else { x0 = qfull[h * head_dim + (d - halfn)] * invv * qw[d - halfn]; x1 = qnorm; }
        uint pidx = (d < halfn) ? d : (d - halfn);
        uint sector = pidx % uint(sect_dims);
        float base = float(pos); // text: t==h==w==pos, e==0
        float theta_scale = pow(rfb, -2.0 / float(n_rot));
        float theta = rfs * base * pow(theta_scale, float(pidx));
        float c = cos(theta) * raf;
        float s = sin(theta) * raf;
        if (d < halfn) {
            qn[h * head_dim + d] = x0 * c - x1 * s;
            qn[h * head_dim + d + halfn] = x0 * s + x1 * c;
        } else {
            qn[h * head_dim + d - halfn] = x0 * c - x1 * s;
            qn[h * head_dim + d] = x0 * s + x1 * c;
        }
    } else {
        qn[h * head_dim + d] = qnorm;
    }
    gate[h * head_dim + d] = g;
}
