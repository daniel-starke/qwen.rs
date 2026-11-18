#version 450
// A Flash-Next delta layer's mixer for the decode's GEMV chains
// (`gpu/gemv.rs` `Gemv::add_delta`): the conv, the recurrence and the gated
// norm run in the submission of the layer's projections, on states that
// stay in VRAM (`npu::flash::gpudec`'s host forms: `delta_conv_n`,
// `FlashEngine::step_recurrence`, `delta_gate_n`). Positions loop inside
// (push constant `np`, each with its own output region `YS` floats apart).
// Push constant `p0`: the run's first position slot of the aux regions
// (`Gemv::set_wave`: a verify step's second wave continues the first's
// recurrence inputs and saved windows at slot p0 = the first wave's
// positions, and never writes the backup, which keeps the state before
// the first wave).
//   binding 0: y (the projections' output; `a` written to `ya`)
//   binding 1: the layer's state `[H][D][D]` (row = key dim), then its
//              backup (the state before a saving step)
//   binding 2: the layer's aux: conv window `[3][NQKV]` (f16 values), the
//              windows after each position `[9][3][NQKV]` (slot 0 = before),
//              the recurrence inputs `[8][NQKV + 2 H]` (qn, kn, v, g, beta)
//   binding 3: mode 0: conv weights `[NQKV][4]`, dt bias `[H]`, A coef
//              `[H]`; mode 1: the gated norm's weight `[D]`
//   mode 0 (conv):  grid 21 x 512: SiLU(conv) of every channel, l2 norm of
//                   each q / k head, the gates (group 20). flags & 1: keep
//                   the windows.
//   mode 1 (recur): grid H x 512 (4 thread rows of 32 key dims per value
//                   column): the state in registers over the positions,
//                   `a = rms-normed o * w * sigmoid(z)` to y. flags & 1:
//                   write the old state to the backup; flags & 2 (replay):
//                   from the backup, no output; flags & 4: no positions
//                   (with 2: the state = its backup).
//   mode 2 (window): grid 20 x 512: window = the windows' slot `p0 + np` (slot 0
//                   with flags & 4).

#define YS 262144u
#define H 48u
#define HK 16u
#define D 128u
#define NQKV 10240u
#define SD (H * D * D)
#define WIN 0u
#define WSAVE (3u * NQKV)
#define RIN (WSAVE + 9u * 3u * NQKV)
#define RW (NQKV + 2u * H)

layout(local_size_x = 512) in;

layout(set = 0, binding = 0, std430) buffer Yb { float y[]; };
layout(set = 0, binding = 1, std430) buffer Sb { float st[]; };
layout(set = 0, binding = 2, std430) buffer Ab { float ax[]; };
layout(set = 0, binding = 3, std430) readonly buffer Wb { float w[]; };
layout(set = 0, binding = 8, std140) uniform Params {
    uint mode;
    uint flags;
    uint yz;
    uint yab;
    uint ya;
};
layout(push_constant) uniform Pc { uint np; uint p0; };

shared float red[512];
shared float kq[256];

float sigm(float v) { return 1.0 / (1.0 + exp(-v)); }
// `v` rounded to f16 to nearest even, as the host's `f16_bits` (the
// driver's packHalf2x16 is not); overflow is not handled (conv inputs)
float h16(float v) {
    if (abs(v) < 6.103515625e-5) { return roundEven(v * 16777216.0) / 16777216.0; }
    uint b = floatBitsToUint(v);
    b += 0xfffu + ((b >> 13u) & 1u);
    return uintBitsToFloat(b & ~0x1fffu);
}

void conv() {
    uint lid = gl_LocalInvocationID.x, g = gl_WorkGroupID.x;
    if (g == 20u) {
        if (lid < H) {
            for (uint t = 0u; t < np; t++) {
                float a = y[t * YS + yab + lid] + w[NQKV * 4u + lid];
                float sp = a > 20.0 ? a : log(1.0 + exp(a));
                uint ri = RIN + (p0 + t) * RW;
                ax[ri + NQKV + lid] = sp * w[NQKV * 4u + H + lid];
                ax[ri + NQKV + H + lid] = sigm(y[t * YS + yab + H + lid]);
            }
        }
        return;
    }
    bool keep = (flags & 1u) != 0u;
    uint c = g * 512u + lid;
    vec4 cw = vec4(w[c * 4u], w[c * 4u + 1u], w[c * 4u + 2u], w[c * 4u + 3u]);
    float w0 = ax[WIN + c], w1 = ax[WIN + NQKV + c], w2 = ax[WIN + 2u * NQKV + c];
    if (keep) {
        uint o = WSAVE + p0 * 3u * NQKV + c;
        ax[o] = w0;
        ax[o + NQKV] = w1;
        ax[o + 2u * NQKV] = w2;
    }
    // groups 0-7 hold the q / k heads
    bool qk = g < 8u;
    uint seg = lid & ~127u;
    for (uint t = 0u; t < np; t++) {
        float q = y[t * YS + c];
        float a = cw.w * q;
        a += cw.x * w0;
        a += cw.y * w1;
        a += cw.z * w2;
        float v = a * sigm(a);
        w0 = w1;
        w1 = w2;
        w2 = h16(q);
        if (keep) {
            uint o = WSAVE + (p0 + t + 1u) * 3u * NQKV + c;
            ax[o] = w0;
            ax[o + NQKV] = w1;
            ax[o + 2u * NQKV] = w2;
        }
        if (qk) {
            red[lid] = v * v;
            barrier();
            for (uint s = 64u; s > 0u; s >>= 1u) {
                if ((lid & 127u) < s) { red[lid] += red[lid + s]; }
                barrier();
            }
            v *= 1.0 / sqrt(max(red[seg], 1e-12));
            barrier();
        }
        ax[RIN + (p0 + t) * RW + c] = v;
    }
    ax[WIN + c] = w0;
    ax[WIN + NQKV + c] = w1;
    ax[WIN + 2u * NQKV + c] = w2;
}

void recur() {
    uint lid = gl_LocalInvocationID.x, h = gl_WorkGroupID.x;
    uint j = lid & 127u, part = lid >> 7u, kh = h / (H / HK);
    // (a second wave keeps the backup: the state before the first)
    bool save = (flags & 1u) != 0u && p0 == 0u, replay = (flags & 2u) != 0u;
    uint base = h * D * D + part * 32u * D + j;
    uint src = replay ? SD + base : base;
    float s[32];
    for (uint r = 0u; r < 32u; r++) { s[r] = st[src + r * D]; }
    if (save) {
        for (uint r = 0u; r < 32u; r++) { st[SD + base + r * D] = s[r]; }
    }
    float qs = 1.0 / sqrt(float(D));
    uint n = (flags & 4u) != 0u ? 0u : np;
    for (uint t = 0u; t < n; t++) {
        uint ri = RIN + (p0 + t) * RW;
        if (lid < 128u) {
            kq[lid] = ax[ri + HK * D + kh * D + lid];
        } else if (lid < 256u) {
            kq[lid] = ax[ri + kh * D + lid - 128u];
        }
        barrier();
        float pk = 0.0;
        for (uint r = 0u; r < 32u; r++) { pk += kq[part * 32u + r] * s[r]; }
        red[lid] = pk;
        barrier();
        float kv = red[j] + red[128u + j] + red[256u + j] + red[384u + j];
        float eg = exp(ax[ri + NQKV + h]), beta = ax[ri + NQKV + H + h];
        float d = (ax[ri + 2u * HK * D + h * D + j] - eg * kv) * beta;
        float po = 0.0;
        for (uint r = 0u; r < 32u; r++) {
            s[r] = eg * s[r] + kq[part * 32u + r] * d;
            po += kq[128u + part * 32u + r] * s[r];
        }
        barrier();
        red[lid] = po;
        barrier();
        float o = 0.0;
        if (lid < 128u) {
            o = (red[j] + red[128u + j] + red[256u + j] + red[384u + j]) * qs;
        }
        barrier();
        if (!replay) {
            // the gated norm over the head's 128 outputs
            if (lid < 128u) { red[lid] = o * o; }
            barrier();
            for (uint k = 64u; k > 0u; k >>= 1u) {
                if (lid < k) { red[lid] += red[lid + k]; }
                barrier();
            }
            if (lid < 128u) {
                float r = 1.0 / sqrt(red[0] / float(D) + 1e-6);
                float z = y[t * YS + yz + h * D + j];
                y[t * YS + ya + h * D + j] = o * r * w[j] * sigm(z);
            }
            barrier();
        }
    }
    for (uint r = 0u; r < 32u; r++) { st[base + r * D] = s[r]; }
}

void window() {
    uint c = gl_WorkGroupID.x * 512u + gl_LocalInvocationID.x;
    uint o = WSAVE + ((flags & 4u) != 0u ? 0u : p0 + np) * 3u * NQKV + c;
    ax[WIN + c] = ax[o];
    ax[WIN + NQKV + c] = ax[o + NQKV];
    ax[WIN + 2u * NQKV + c] = ax[o + 2u * NQKV];
}

void main() {
    if (mode == 0u) {
        conv();
    } else if (mode == 1u) {
        recur();
    } else {
        window();
    }
}
