#include <metal_stdlib>
using namespace metal;

// Helper for computing strided index from linear output index
inline uint compute_strided_index(uint out_idx, constant uint* shape, constant int* strides, uint ndim, int offset) {
    int idx = offset;
    uint temp = out_idx;
    
    // Convert linear index to coordinates and compute strided index
    for (int i = int(ndim) - 1; i >= 0; i--) {
        uint coord = temp % shape[i];
        temp /= shape[i];
        idx += int(coord) * strides[i];
    }
    
    return uint(idx);
}

// Macro to define unary operations
#define DEFINE_UNARY_OP(name, op, type) \
kernel void name##_##type(device type* out [[buffer(0)]], \
                         device const type* in [[buffer(1)]], \
                         constant uint* shape [[buffer(2)]], \
                         constant int* strides [[buffer(3)]], \
                         constant uint& ndim [[buffer(4)]], \
                         constant int& offset [[buffer(5)]], \
                         uint gid [[thread_position_in_grid]]) { \
    uint out_idx = gid; \
    uint total_size = 1; \
    for (uint i = 0; i < ndim; i++) { \
        total_size *= shape[i]; \
    } \
    if (out_idx >= total_size) return; \
    \
    uint in_idx = compute_strided_index(out_idx, shape, strides, ndim, offset); \
    out[out_idx] = op(in[in_idx]); \
}

// Negation
kernel void neg_half(device half* out [[buffer(0)]],
                    device const half* in [[buffer(1)]],
                    constant uint* shape [[buffer(2)]],
                    constant int* strides [[buffer(3)]],
                    constant uint& ndim [[buffer(4)]],
                    constant int& offset [[buffer(5)]],
                    uint gid [[thread_position_in_grid]]) {
    uint out_idx = gid;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint in_idx = compute_strided_index(out_idx, shape, strides, ndim, offset);
    out[out_idx] = -in[in_idx];
}

kernel void neg_float(device float* out [[buffer(0)]],
                     device const float* in [[buffer(1)]],
                     constant uint* shape [[buffer(2)]],
                     constant int* strides [[buffer(3)]],
                     constant uint& ndim [[buffer(4)]],
                     constant int& offset [[buffer(5)]],
                     uint gid [[thread_position_in_grid]]) {
    uint out_idx = gid;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint in_idx = compute_strided_index(out_idx, shape, strides, ndim, offset);
    out[out_idx] = -in[in_idx];
}

kernel void neg_char(device char* out [[buffer(0)]],
                    device const char* in [[buffer(1)]],
                    constant uint* shape [[buffer(2)]],
                    constant int* strides [[buffer(3)]],
                    constant uint& ndim [[buffer(4)]],
                    constant int& offset [[buffer(5)]],
                    uint gid [[thread_position_in_grid]]) {
    uint out_idx = gid;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint in_idx = compute_strided_index(out_idx, shape, strides, ndim, offset);
    out[out_idx] = -in[in_idx];
}

kernel void neg_short(device short* out [[buffer(0)]],
                     device const short* in [[buffer(1)]],
                     constant uint* shape [[buffer(2)]],
                     constant int* strides [[buffer(3)]],
                     constant uint& ndim [[buffer(4)]],
                     constant int& offset [[buffer(5)]],
                     uint gid [[thread_position_in_grid]]) {
    uint out_idx = gid;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint in_idx = compute_strided_index(out_idx, shape, strides, ndim, offset);
    out[out_idx] = -in[in_idx];
}

kernel void neg_int(device int* out [[buffer(0)]],
                   device const int* in [[buffer(1)]],
                   constant uint* shape [[buffer(2)]],
                   constant int* strides [[buffer(3)]],
                   constant uint& ndim [[buffer(4)]],
                   constant int& offset [[buffer(5)]],
                   uint gid [[thread_position_in_grid]]) {
    uint out_idx = gid;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint in_idx = compute_strided_index(out_idx, shape, strides, ndim, offset);
    out[out_idx] = -in[in_idx];
}

kernel void neg_long(device long* out [[buffer(0)]],
                    device const long* in [[buffer(1)]],
                    constant uint* shape [[buffer(2)]],
                    constant int* strides [[buffer(3)]],
                    constant uint& ndim [[buffer(4)]],
                    constant int& offset [[buffer(5)]],
                    uint gid [[thread_position_in_grid]]) {
    uint out_idx = gid;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint in_idx = compute_strided_index(out_idx, shape, strides, ndim, offset);
    out[out_idx] = -in[in_idx];
}

// Negation for unsigned types (cast to signed)
kernel void neg_ushort(device ushort* out [[buffer(0)]],
                      device const ushort* in [[buffer(1)]],
                      constant uint* shape [[buffer(2)]],
                      constant int* strides [[buffer(3)]],
                      constant uint& ndim [[buffer(4)]],
                      constant int& offset [[buffer(5)]],
                      uint gid [[thread_position_in_grid]]) {
    uint out_idx = gid;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint in_idx = compute_strided_index(out_idx, shape, strides, ndim, offset);
    out[out_idx] = (ushort)(-(short)in[in_idx]);
}

// Logical negation for bool (uint8)
kernel void neg_uchar(device uchar* out [[buffer(0)]],
                     device const uchar* in [[buffer(1)]],
                     constant uint* shape [[buffer(2)]],
                     constant int* strides [[buffer(3)]],
                     constant uint& ndim [[buffer(4)]],
                     constant int& offset [[buffer(5)]],
                     uint gid [[thread_position_in_grid]]) {
    uint out_idx = gid;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint in_idx = compute_strided_index(out_idx, shape, strides, ndim, offset);
    out[out_idx] = in[in_idx] ? 0 : 1;
}

// Logarithm base 2
DEFINE_UNARY_OP(log2, log2, half)
DEFINE_UNARY_OP(log2, log2, float)

// Exponential base 2
DEFINE_UNARY_OP(exp2, exp2, half)
DEFINE_UNARY_OP(exp2, exp2, float)

// Sine
DEFINE_UNARY_OP(sin, sin, half)
DEFINE_UNARY_OP(sin, sin, float)

// Square root
DEFINE_UNARY_OP(sqrt, sqrt, half)
DEFINE_UNARY_OP(sqrt, sqrt, float)

// Reciprocal (using macro for division)
#define RECIP(x) (1.0f / (x))
#define RECIP_HALF(x) (half(1.0) / (x))

kernel void recip_half(device half* out [[buffer(0)]],
                      device const half* in [[buffer(1)]],
                      constant uint* shape [[buffer(2)]],
                      constant int* strides [[buffer(3)]],
                      constant uint& ndim [[buffer(4)]],
                      constant int& offset [[buffer(5)]],
                      uint gid [[thread_position_in_grid]]) {
    uint out_idx = gid;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint in_idx = compute_strided_index(out_idx, shape, strides, ndim, offset);
    out[out_idx] = RECIP_HALF(in[in_idx]);
}

kernel void recip_float(device float* out [[buffer(0)]],
                       device const float* in [[buffer(1)]],
                       constant uint* shape [[buffer(2)]],
                       constant int* strides [[buffer(3)]],
                       constant uint& ndim [[buffer(4)]],
                       constant int& offset [[buffer(5)]],
                       uint gid [[thread_position_in_grid]]) {
    uint out_idx = gid;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint in_idx = compute_strided_index(out_idx, shape, strides, ndim, offset);
    out[out_idx] = RECIP(in[in_idx]);
}

// Complex unary operations for float2 (Complex32)
kernel void neg_float2(device float2* out [[buffer(0)]],
                      device const float2* in [[buffer(1)]],
                      constant uint* shape [[buffer(2)]],
                      constant int* strides [[buffer(3)]],
                      constant uint& ndim [[buffer(4)]],
                      constant int& offset [[buffer(5)]],
                      uint gid [[thread_position_in_grid]]) {
    uint out_idx = gid;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint in_idx = compute_strided_index(out_idx, shape, strides, ndim, offset);
    out[out_idx] = -in[in_idx];
}

kernel void conj_float2(device float2* out [[buffer(0)]],
                       device const float2* in [[buffer(1)]],
                       constant uint* shape [[buffer(2)]],
                       constant int* strides [[buffer(3)]],
                       constant uint& ndim [[buffer(4)]],
                       constant int& offset [[buffer(5)]],
                       uint gid [[thread_position_in_grid]]) {
    uint out_idx = gid;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint in_idx = compute_strided_index(out_idx, shape, strides, ndim, offset);
    float2 val = in[in_idx];
    out[out_idx] = float2(val.x, -val.y);  // conjugate: real part stays, imaginary part negates
}

kernel void abs_float2(device float* out [[buffer(0)]],
                      device const float2* in [[buffer(1)]],
                      constant uint* shape [[buffer(2)]],
                      constant int* strides [[buffer(3)]],
                      constant uint& ndim [[buffer(4)]],
                      constant int& offset [[buffer(5)]],
                      uint gid [[thread_position_in_grid]]) {
    uint out_idx = gid;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint in_idx = compute_strided_index(out_idx, shape, strides, ndim, offset);
    float2 val = in[in_idx];
    out[out_idx] = sqrt(val.x * val.x + val.y * val.y);  // |a + bi| = sqrt(a^2 + b^2)
}

kernel void sqrt_float2(device float2* out [[buffer(0)]],
                       device const float2* in [[buffer(1)]],
                       constant uint* shape [[buffer(2)]],
                       constant int* strides [[buffer(3)]],
                       constant uint& ndim [[buffer(4)]],
                       constant int& offset [[buffer(5)]],
                       uint gid [[thread_position_in_grid]]) {
    uint out_idx = gid;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint in_idx = compute_strided_index(out_idx, shape, strides, ndim, offset);
    float2 val = in[in_idx];
    
    // Complex sqrt: sqrt(a + bi) = sqrt((r + a)/2) + i*sign(b)*sqrt((r - a)/2)
    // where r = |a + bi| = sqrt(a^2 + b^2)
    float r = sqrt(val.x * val.x + val.y * val.y);
    float sign_y = val.y >= 0.0f ? 1.0f : -1.0f;
    out[out_idx] = float2(sqrt((r + val.x) / 2.0f), sign_y * sqrt((r - val.x) / 2.0f));
}

kernel void exp2_float2(device float2* out [[buffer(0)]],
                       device const float2* in [[buffer(1)]],
                       constant uint* shape [[buffer(2)]],
                       constant int* strides [[buffer(3)]],
                       constant uint& ndim [[buffer(4)]],
                       constant int& offset [[buffer(5)]],
                       uint gid [[thread_position_in_grid]]) {
    uint out_idx = gid;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint in_idx = compute_strided_index(out_idx, shape, strides, ndim, offset);
    float2 val = in[in_idx];
    
    // 2^(a + bi) = 2^a * (cos(b*ln(2)) + i*sin(b*ln(2)))
    float exp_real = exp2(val.x);
    float b_ln2 = val.y * M_LN2_F;
    out[out_idx] = float2(exp_real * cos(b_ln2), exp_real * sin(b_ln2));
}

kernel void log2_float2(device float2* out [[buffer(0)]],
                       device const float2* in [[buffer(1)]],
                       constant uint* shape [[buffer(2)]],
                       constant int* strides [[buffer(3)]],
                       constant uint& ndim [[buffer(4)]],
                       constant int& offset [[buffer(5)]],
                       uint gid [[thread_position_in_grid]]) {
    uint out_idx = gid;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint in_idx = compute_strided_index(out_idx, shape, strides, ndim, offset);
    float2 val = in[in_idx];
    
    // log2(a + bi) = log2(|a + bi|) + i * arg(a + bi) / ln(2)
    float magnitude = sqrt(val.x * val.x + val.y * val.y);
    float phase = atan2(val.y, val.x);
    out[out_idx] = float2(log2(magnitude), phase / M_LN2_F);
}

// Complex unary operations for half2 (Complex16)
kernel void neg_half2(device half2* out [[buffer(0)]],
                     device const half2* in [[buffer(1)]],
                     constant uint* shape [[buffer(2)]],
                     constant int* strides [[buffer(3)]],
                     constant uint& ndim [[buffer(4)]],
                     constant int& offset [[buffer(5)]],
                     uint gid [[thread_position_in_grid]]) {
    uint out_idx = gid;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint in_idx = compute_strided_index(out_idx, shape, strides, ndim, offset);
    out[out_idx] = -in[in_idx];
}

kernel void conj_half2(device half2* out [[buffer(0)]],
                      device const half2* in [[buffer(1)]],
                      constant uint* shape [[buffer(2)]],
                      constant int* strides [[buffer(3)]],
                      constant uint& ndim [[buffer(4)]],
                      constant int& offset [[buffer(5)]],
                      uint gid [[thread_position_in_grid]]) {
    uint out_idx = gid;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint in_idx = compute_strided_index(out_idx, shape, strides, ndim, offset);
    half2 val = in[in_idx];
    out[out_idx] = half2(val.x, -val.y);  // conjugate: real part stays, imaginary part negates
}

kernel void abs_half2(device half* out [[buffer(0)]],
                     device const half2* in [[buffer(1)]],
                     constant uint* shape [[buffer(2)]],
                     constant int* strides [[buffer(3)]],
                     constant uint& ndim [[buffer(4)]],
                     constant int& offset [[buffer(5)]],
                     uint gid [[thread_position_in_grid]]) {
    uint out_idx = gid;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint in_idx = compute_strided_index(out_idx, shape, strides, ndim, offset);
    half2 val = in[in_idx];
    out[out_idx] = sqrt(val.x * val.x + val.y * val.y);  // |a + bi| = sqrt(a^2 + b^2)
}

kernel void sqrt_half2(device half2* out [[buffer(0)]],
                      device const half2* in [[buffer(1)]],
                      constant uint* shape [[buffer(2)]],
                      constant int* strides [[buffer(3)]],
                      constant uint& ndim [[buffer(4)]],
                      constant int& offset [[buffer(5)]],
                      uint gid [[thread_position_in_grid]]) {
    uint out_idx = gid;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint in_idx = compute_strided_index(out_idx, shape, strides, ndim, offset);
    half2 val = in[in_idx];
    
    // Complex sqrt: sqrt(a + bi) = sqrt((r + a)/2) + i*sign(b)*sqrt((r - a)/2)
    // where r = |a + bi| = sqrt(a^2 + b^2)
    half r = sqrt(val.x * val.x + val.y * val.y);
    half sign_y = val.y >= half(0.0) ? half(1.0) : half(-1.0);
    out[out_idx] = half2(sqrt((r + val.x) / half(2.0)), sign_y * sqrt((r - val.x) / half(2.0)));
}

kernel void exp2_half2(device half2* out [[buffer(0)]],
                      device const half2* in [[buffer(1)]],
                      constant uint* shape [[buffer(2)]],
                      constant int* strides [[buffer(3)]],
                      constant uint& ndim [[buffer(4)]],
                      constant int& offset [[buffer(5)]],
                      uint gid [[thread_position_in_grid]]) {
    uint out_idx = gid;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint in_idx = compute_strided_index(out_idx, shape, strides, ndim, offset);
    half2 val = in[in_idx];
    
    // 2^(a + bi) = 2^a * (cos(b*ln(2)) + i*sin(b*ln(2)))
    half exp_real = exp2(val.x);
    half b_ln2 = val.y * half(M_LN2_F);
    out[out_idx] = half2(exp_real * cos(b_ln2), exp_real * sin(b_ln2));
}

kernel void log2_half2(device half2* out [[buffer(0)]],
                      device const half2* in [[buffer(1)]],
                      constant uint* shape [[buffer(2)]],
                      constant int* strides [[buffer(3)]],
                      constant uint& ndim [[buffer(4)]],
                      constant int& offset [[buffer(5)]],
                      uint gid [[thread_position_in_grid]]) {
    uint out_idx = gid;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint in_idx = compute_strided_index(out_idx, shape, strides, ndim, offset);
    half2 val = in[in_idx];
    
    // log2(a + bi) = log2(|a + bi|) + i * arg(a + bi) / ln(2)
    half magnitude = sqrt(val.x * val.x + val.y * val.y);
    half phase = atan2(val.y, val.x);
    out[out_idx] = half2(log2(magnitude), phase / half(M_LN2_F));
}