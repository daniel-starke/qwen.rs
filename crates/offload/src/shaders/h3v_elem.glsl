#version 450
// MiniMax-H3-Image's VAE decoder elementwise steps (`gpu/h3vae.rs`), one
// thread per element (pair):
//   MODE 0: x f32 = the NPU buffer's f16 token rows (pairs from `ofs`)
//   MODE 1: act [rows][F] = gu[r][c] swish(gu[r][F + c])
//   MODE 2: the NPU buffer's f16 rows (pairs from `ofs`) = y f32
layout(local_size_x = 256) in;

layout(set = 0, binding = 0, std430) buffer Hb { uint hb[]; };
layout(set = 0, binding = 1, std430) readonly buffer A { float a[]; };
layout(set = 0, binding = 2, std430) buffer Y { float y[]; };
layout(push_constant) uniform Pc {
    uint n;     // pairs (MODE 0, 2) or elements (MODE 1)
    uint ofs;   // MODE 0 / 2: first pair in hb; MODE 1: F
};

void main() {
    uint i = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * 65535u * 256u;
    if (i >= n) { return; }
#if MODE == 0
    vec2 v = unpackHalf2x16(hb[ofs + i]);
    y[2u * i] = v.x;
    y[2u * i + 1u] = v.y;
#elif MODE == 1
    uint F = ofs;
    uint r = i / F, c = i % F;
    float g = a[r * 2u * F + F + c];
    y[i] = a[r * 2u * F + c] * (g / (1.0 + exp(-g)));
#else
    hb[ofs + i] = packHalf2x16(vec2(a[2u * i], a[2u * i + 1u]));
#endif
}
