#include <metal_stdlib>
using namespace metal;

// Macro to define unary operations
#define DEFINE_UNARY_OP(name, op, type) \
kernel void name##_##type(device type* out [[buffer(0)]], \
                         device const type* in [[buffer(1)]], \
                         constant uint& size [[buffer(2)]], \
                         uint gid [[thread_position_in_grid]]) { \
    if (gid >= size) return; \
    out[gid] = op(in[gid]); \
}

// Negation
kernel void neg_float(device float* out [[buffer(0)]],
                     device const float* in [[buffer(1)]],
                     constant uint& size [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = -in[gid];
}


kernel void neg_int(device int* out [[buffer(0)]],
                   device const int* in [[buffer(1)]],
                   constant uint& size [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = -in[gid];
}

kernel void neg_long(device long* out [[buffer(0)]],
                    device const long* in [[buffer(1)]],
                    constant uint& size [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = -in[gid];
}

// Logical negation for bool (uint8)
kernel void neg_uchar(device uchar* out [[buffer(0)]],
                     device const uchar* in [[buffer(1)]],
                     constant uint& size [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = in[gid] ? 0 : 1;
}

// Logarithm base 2
DEFINE_UNARY_OP(log2, log2, float)

// Exponential base 2
DEFINE_UNARY_OP(exp2, exp2, float)

// Sine
DEFINE_UNARY_OP(sin, sin, float)

// Square root
DEFINE_UNARY_OP(sqrt, sqrt, float)

// Reciprocal
kernel void recip_float(device float* out [[buffer(0)]],
                       device const float* in [[buffer(1)]],
                       constant uint& size [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = 1.0f / in[gid];
}


// Additional math functions that might be needed
DEFINE_UNARY_OP(cos, cos, float)
DEFINE_UNARY_OP(tan, tan, float)
DEFINE_UNARY_OP(exp, exp, float)
DEFINE_UNARY_OP(log, log, float)
DEFINE_UNARY_OP(abs, abs, float)
DEFINE_UNARY_OP(abs, abs, int)
DEFINE_UNARY_OP(abs, abs, long)
