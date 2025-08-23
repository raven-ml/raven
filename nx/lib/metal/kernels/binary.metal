#include <metal_stdlib>
using namespace metal;

// Helper for broadcasting - compute linear index from position and strides
inline uint compute_index_from_linear(uint linear_idx, constant uint* shape, constant int* strides, uint ndim) {
    int idx = 0;  // Changed to signed to handle negative stride accumulation
    uint temp = linear_idx;
    
    // Convert linear index to coordinates and compute strided index
    for (int i = int(ndim) - 1; i >= 0; i--) {
        uint coord = temp % shape[i];
        temp /= shape[i];
        idx += int(coord) * strides[i];  // Cast coord to int for signed multiplication
    }
    
    // The final index must be non-negative
    return uint(idx);
}

// Macro to define binary operations for all types
#define DEFINE_BINARY_OP(name, op, type) \
kernel void name##_##type(device type* out [[buffer(0)]], \
                         device const type* a [[buffer(1)]], \
                         device const type* b [[buffer(2)]], \
                         constant uint* out_shape [[buffer(3)]], \
                         constant int* a_strides [[buffer(4)]], \
                         constant int* b_strides [[buffer(5)]], \
                         constant uint& ndim [[buffer(6)]], \
                         constant int& a_offset [[buffer(7)]], \
                         constant int& b_offset [[buffer(8)]], \
                         uint3 gid [[thread_position_in_grid]]) { \
    uint out_idx = gid.x; \
    uint total_size = 1; \
    for (uint i = 0; i < ndim; i++) { \
        total_size *= out_shape[i]; \
    } \
    if (out_idx >= total_size) return; \
    \
    uint a_idx = compute_index_from_linear(out_idx, out_shape, a_strides, ndim) + a_offset; \
    uint b_idx = compute_index_from_linear(out_idx, out_shape, b_strides, ndim) + b_offset; \
    \
    out[out_idx] = a[a_idx] op b[b_idx]; \
}

// Macro for comparison operations that return uchar (0 or 1)
#define DEFINE_CMP_OP(name, op, type) \
kernel void name##_##type(device uchar* out [[buffer(0)]], \
                         device const type* a [[buffer(1)]], \
                         device const type* b [[buffer(2)]], \
                         constant uint* out_shape [[buffer(3)]], \
                         constant int* a_strides [[buffer(4)]], \
                         constant int* b_strides [[buffer(5)]], \
                         constant uint& ndim [[buffer(6)]], \
                         constant int& a_offset [[buffer(7)]], \
                         constant int& b_offset [[buffer(8)]], \
                         uint3 gid [[thread_position_in_grid]]) { \
    uint out_idx = gid.x; \
    uint total_size = 1; \
    for (uint i = 0; i < ndim; i++) { \
        total_size *= out_shape[i]; \
    } \
    if (out_idx >= total_size) return; \
    \
    uint a_idx = compute_index_from_linear(out_idx, out_shape, a_strides, ndim) + a_offset; \
    uint b_idx = compute_index_from_linear(out_idx, out_shape, b_strides, ndim) + b_offset; \
    \
    out[out_idx] = a[a_idx] op b[b_idx] ? 1 : 0; \
}

// Macro for function-based binary operations (like max, pow)
#define DEFINE_FUNC_OP(name, func, type) \
kernel void name##_##type(device type* out [[buffer(0)]], \
                         device const type* a [[buffer(1)]], \
                         device const type* b [[buffer(2)]], \
                         constant uint* out_shape [[buffer(3)]], \
                         constant int* a_strides [[buffer(4)]], \
                         constant int* b_strides [[buffer(5)]], \
                         constant uint& ndim [[buffer(6)]], \
                         constant int& a_offset [[buffer(7)]], \
                         constant int& b_offset [[buffer(8)]], \
                         uint3 gid [[thread_position_in_grid]]) { \
    uint out_idx = gid.x; \
    uint total_size = 1; \
    for (uint i = 0; i < ndim; i++) { \
        total_size *= out_shape[i]; \
    } \
    if (out_idx >= total_size) return; \
    \
    uint a_idx = compute_index_from_linear(out_idx, out_shape, a_strides, ndim) + a_offset; \
    uint b_idx = compute_index_from_linear(out_idx, out_shape, b_strides, ndim) + b_offset; \
    \
    out[out_idx] = func(a[a_idx], b[b_idx]); \
}

// Arithmetic operations
DEFINE_BINARY_OP(add, +, half)
DEFINE_BINARY_OP(add, +, float)
DEFINE_BINARY_OP(add, +, char)
DEFINE_BINARY_OP(add, +, uchar)
DEFINE_BINARY_OP(add, +, short)
DEFINE_BINARY_OP(add, +, ushort)
DEFINE_BINARY_OP(add, +, int)
DEFINE_BINARY_OP(add, +, long)
DEFINE_BINARY_OP(add, +, bool)

DEFINE_BINARY_OP(sub, -, half)
DEFINE_BINARY_OP(sub, -, float)
DEFINE_BINARY_OP(sub, -, char)
DEFINE_BINARY_OP(sub, -, uchar)
DEFINE_BINARY_OP(sub, -, short)
DEFINE_BINARY_OP(sub, -, ushort)
DEFINE_BINARY_OP(sub, -, int)
DEFINE_BINARY_OP(sub, -, long)

DEFINE_BINARY_OP(mul, *, half)
DEFINE_BINARY_OP(mul, *, float)
DEFINE_BINARY_OP(mul, *, char)
DEFINE_BINARY_OP(mul, *, uchar)
DEFINE_BINARY_OP(mul, *, short)
DEFINE_BINARY_OP(mul, *, ushort)
DEFINE_BINARY_OP(mul, *, int)
DEFINE_BINARY_OP(mul, *, long)

// Division operations for integer types
kernel void idiv_char(device char* out [[buffer(0)]],
                     device const char* a [[buffer(1)]],
                     device const char* b [[buffer(2)]],
                     constant uint* out_shape [[buffer(3)]],
                     constant int* a_strides [[buffer(4)]],
                     constant int* b_strides [[buffer(5)]],
                     constant uint& ndim [[buffer(6)]],
                    constant int& a_offset [[buffer(7)]],
                    constant int& b_offset [[buffer(8)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= out_shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint a_idx = compute_index_from_linear(out_idx, out_shape, a_strides, ndim) + a_offset;
    uint b_idx = compute_index_from_linear(out_idx, out_shape, b_strides, ndim) + b_offset;
    
    out[out_idx] = a[a_idx] / b[b_idx];
}

kernel void idiv_uchar(device uchar* out [[buffer(0)]],
                      device const uchar* a [[buffer(1)]],
                      device const uchar* b [[buffer(2)]],
                      constant uint* out_shape [[buffer(3)]],
                      constant int* a_strides [[buffer(4)]],
                      constant int* b_strides [[buffer(5)]],
                      constant uint& ndim [[buffer(6)]],
                     constant int& a_offset [[buffer(7)]],
                     constant int& b_offset [[buffer(8)]],
                     uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= out_shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint a_idx = compute_index_from_linear(out_idx, out_shape, a_strides, ndim) + a_offset;
    uint b_idx = compute_index_from_linear(out_idx, out_shape, b_strides, ndim) + b_offset;
    
    out[out_idx] = a[a_idx] / b[b_idx];
}

kernel void idiv_short(device short* out [[buffer(0)]],
                      device const short* a [[buffer(1)]],
                      device const short* b [[buffer(2)]],
                      constant uint* out_shape [[buffer(3)]],
                      constant int* a_strides [[buffer(4)]],
                      constant int* b_strides [[buffer(5)]],
                      constant uint& ndim [[buffer(6)]],
                     constant int& a_offset [[buffer(7)]],
                     constant int& b_offset [[buffer(8)]],
                     uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= out_shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint a_idx = compute_index_from_linear(out_idx, out_shape, a_strides, ndim) + a_offset;
    uint b_idx = compute_index_from_linear(out_idx, out_shape, b_strides, ndim) + b_offset;
    
    out[out_idx] = a[a_idx] / b[b_idx];
}

kernel void idiv_ushort(device ushort* out [[buffer(0)]],
                       device const ushort* a [[buffer(1)]],
                       device const ushort* b [[buffer(2)]],
                       constant uint* out_shape [[buffer(3)]],
                       constant int* a_strides [[buffer(4)]],
                       constant int* b_strides [[buffer(5)]],
                       constant uint& ndim [[buffer(6)]],
                      constant int& a_offset [[buffer(7)]],
                      constant int& b_offset [[buffer(8)]],
                      uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= out_shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint a_idx = compute_index_from_linear(out_idx, out_shape, a_strides, ndim) + a_offset;
    uint b_idx = compute_index_from_linear(out_idx, out_shape, b_strides, ndim) + b_offset;
    
    out[out_idx] = a[a_idx] / b[b_idx];
}

kernel void idiv_int(device int* out [[buffer(0)]],
                    device const int* a [[buffer(1)]],
                    device const int* b [[buffer(2)]],
                    constant uint* out_shape [[buffer(3)]],
                    constant int* a_strides [[buffer(4)]],
                    constant int* b_strides [[buffer(5)]],
                    constant uint& ndim [[buffer(6)]],
                    constant int& a_offset [[buffer(7)]],
                    constant int& b_offset [[buffer(8)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= out_shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint a_idx = compute_index_from_linear(out_idx, out_shape, a_strides, ndim) + a_offset;
    uint b_idx = compute_index_from_linear(out_idx, out_shape, b_strides, ndim) + b_offset;
    
    out[out_idx] = a[a_idx] / b[b_idx];
}

kernel void idiv_long(device long* out [[buffer(0)]],
                     device const long* a [[buffer(1)]],
                     device const long* b [[buffer(2)]],
                     constant uint* out_shape [[buffer(3)]],
                     constant int* a_strides [[buffer(4)]],
                     constant int* b_strides [[buffer(5)]],
                     constant uint& ndim [[buffer(6)]],
                    constant int& a_offset [[buffer(7)]],
                    constant int& b_offset [[buffer(8)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= out_shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint a_idx = compute_index_from_linear(out_idx, out_shape, a_strides, ndim) + a_offset;
    uint b_idx = compute_index_from_linear(out_idx, out_shape, b_strides, ndim) + b_offset;
    
    out[out_idx] = a[a_idx] / b[b_idx];
}

// Float division operations
DEFINE_BINARY_OP(fdiv, /, half)
DEFINE_BINARY_OP(fdiv, /, float)


// Max operations using max function
DEFINE_FUNC_OP(max, max, char)
DEFINE_FUNC_OP(max, max, uchar)
DEFINE_FUNC_OP(max, max, short)
DEFINE_FUNC_OP(max, max, ushort)
DEFINE_FUNC_OP(max, max, int)
DEFINE_FUNC_OP(max, max, long)
DEFINE_FUNC_OP(max, fmax, half)
DEFINE_FUNC_OP(max, fmax, float)

// Mod operations
DEFINE_FUNC_OP(mod, fmod, half)
DEFINE_FUNC_OP(mod, fmod, float)
DEFINE_BINARY_OP(mod, %, char)
DEFINE_BINARY_OP(mod, %, uchar)
DEFINE_BINARY_OP(mod, %, short)
DEFINE_BINARY_OP(mod, %, ushort)
DEFINE_BINARY_OP(mod, %, int)
DEFINE_BINARY_OP(mod, %, long)

// Pow operations (only for floating point)
DEFINE_FUNC_OP(pow, pow, half)
DEFINE_FUNC_OP(pow, pow, float)

// Comparison operations
DEFINE_CMP_OP(cmplt, <, half)
DEFINE_CMP_OP(cmplt, <, float)
DEFINE_CMP_OP(cmplt, <, char)
DEFINE_CMP_OP(cmplt, <, uchar)
DEFINE_CMP_OP(cmplt, <, short)
DEFINE_CMP_OP(cmplt, <, ushort)
DEFINE_CMP_OP(cmplt, <, int)
DEFINE_CMP_OP(cmplt, <, long)

DEFINE_CMP_OP(cmpne, !=, half)
DEFINE_CMP_OP(cmpne, !=, float)
DEFINE_CMP_OP(cmpne, !=, char)
DEFINE_CMP_OP(cmpne, !=, uchar)
DEFINE_CMP_OP(cmpne, !=, short)
DEFINE_CMP_OP(cmpne, !=, ushort)
DEFINE_CMP_OP(cmpne, !=, int)
DEFINE_CMP_OP(cmpne, !=, long)

DEFINE_CMP_OP(cmpeq, ==, half)
DEFINE_CMP_OP(cmpeq, ==, float)
DEFINE_CMP_OP(cmpeq, ==, char)
DEFINE_CMP_OP(cmpeq, ==, uchar)
DEFINE_CMP_OP(cmpeq, ==, short)
DEFINE_CMP_OP(cmpeq, ==, ushort)
DEFINE_CMP_OP(cmpeq, ==, int)
DEFINE_CMP_OP(cmpeq, ==, long)

// Bitwise operations

// Complex number operations for float2
kernel void add_float2(device float2* out [[buffer(0)]],
                      device const float2* a [[buffer(1)]],
                      device const float2* b [[buffer(2)]],
                      constant uint* out_shape [[buffer(3)]],
                      constant int* a_strides [[buffer(4)]],
                      constant int* b_strides [[buffer(5)]],
                      constant uint& ndim [[buffer(6)]],
                      constant int& a_offset [[buffer(7)]],
                      constant int& b_offset [[buffer(8)]],
                      uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= out_shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint a_idx = compute_index_from_linear(out_idx, out_shape, a_strides, ndim) + a_offset;
    uint b_idx = compute_index_from_linear(out_idx, out_shape, b_strides, ndim) + b_offset;
    
    out[out_idx] = a[a_idx] + b[b_idx];
}

kernel void sub_float2(device float2* out [[buffer(0)]],
                      device const float2* a [[buffer(1)]],
                      device const float2* b [[buffer(2)]],
                      constant uint* out_shape [[buffer(3)]],
                      constant int* a_strides [[buffer(4)]],
                      constant int* b_strides [[buffer(5)]],
                      constant uint& ndim [[buffer(6)]],
                      constant int& a_offset [[buffer(7)]],
                      constant int& b_offset [[buffer(8)]],
                      uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= out_shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint a_idx = compute_index_from_linear(out_idx, out_shape, a_strides, ndim) + a_offset;
    uint b_idx = compute_index_from_linear(out_idx, out_shape, b_strides, ndim) + b_offset;
    
    out[out_idx] = a[a_idx] - b[b_idx];
}

kernel void mul_float2(device float2* out [[buffer(0)]],
                      device const float2* a [[buffer(1)]],
                      device const float2* b [[buffer(2)]],
                      constant uint* out_shape [[buffer(3)]],
                      constant int* a_strides [[buffer(4)]],
                      constant int* b_strides [[buffer(5)]],
                      constant uint& ndim [[buffer(6)]],
                      constant int& a_offset [[buffer(7)]],
                      constant int& b_offset [[buffer(8)]],
                      uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= out_shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint a_idx = compute_index_from_linear(out_idx, out_shape, a_strides, ndim) + a_offset;
    uint b_idx = compute_index_from_linear(out_idx, out_shape, b_strides, ndim) + b_offset;
    
    float2 av = a[a_idx];
    float2 bv = b[b_idx];
    
    // Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    out[out_idx] = float2(av.x * bv.x - av.y * bv.y, av.x * bv.y + av.y * bv.x);
}

kernel void fdiv_float2(device float2* out [[buffer(0)]],
                       device const float2* a [[buffer(1)]],
                       device const float2* b [[buffer(2)]],
                       constant uint* out_shape [[buffer(3)]],
                       constant int* a_strides [[buffer(4)]],
                       constant int* b_strides [[buffer(5)]],
                       constant uint& ndim [[buffer(6)]],
                       constant int& a_offset [[buffer(7)]],
                       constant int& b_offset [[buffer(8)]],
                       uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= out_shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint a_idx = compute_index_from_linear(out_idx, out_shape, a_strides, ndim) + a_offset;
    uint b_idx = compute_index_from_linear(out_idx, out_shape, b_strides, ndim) + b_offset;
    
    float2 av = a[a_idx];
    float2 bv = b[b_idx];
    
    // Complex division: (a + bi) / (c + di) = ((ac + bd) + (bc - ad)i) / (c^2 + d^2)
    float denom = bv.x * bv.x + bv.y * bv.y;
    out[out_idx] = float2((av.x * bv.x + av.y * bv.y) / denom,
                         (av.y * bv.x - av.x * bv.y) / denom);
}

// Complex16 (half2) operations
kernel void add_half2(device half2* out [[buffer(0)]],
                     device const half2* a [[buffer(1)]],
                     device const half2* b [[buffer(2)]],
                     constant uint* out_shape [[buffer(3)]],
                     constant int* a_strides [[buffer(4)]],
                     constant int* b_strides [[buffer(5)]],
                     constant uint& ndim [[buffer(6)]],
                     constant int& a_offset [[buffer(7)]],
                     constant int& b_offset [[buffer(8)]],
                     uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= out_shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint a_idx = compute_index_from_linear(out_idx, out_shape, a_strides, ndim) + a_offset;
    uint b_idx = compute_index_from_linear(out_idx, out_shape, b_strides, ndim) + b_offset;
    
    out[out_idx] = a[a_idx] + b[b_idx];
}

kernel void sub_half2(device half2* out [[buffer(0)]],
                     device const half2* a [[buffer(1)]],
                     device const half2* b [[buffer(2)]],
                     constant uint* out_shape [[buffer(3)]],
                     constant int* a_strides [[buffer(4)]],
                     constant int* b_strides [[buffer(5)]],
                     constant uint& ndim [[buffer(6)]],
                     constant int& a_offset [[buffer(7)]],
                     constant int& b_offset [[buffer(8)]],
                     uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= out_shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint a_idx = compute_index_from_linear(out_idx, out_shape, a_strides, ndim) + a_offset;
    uint b_idx = compute_index_from_linear(out_idx, out_shape, b_strides, ndim) + b_offset;
    
    out[out_idx] = a[a_idx] - b[b_idx];
}

kernel void mul_half2(device half2* out [[buffer(0)]],
                     device const half2* a [[buffer(1)]],
                     device const half2* b [[buffer(2)]],
                     constant uint* out_shape [[buffer(3)]],
                     constant int* a_strides [[buffer(4)]],
                     constant int* b_strides [[buffer(5)]],
                     constant uint& ndim [[buffer(6)]],
                     constant int& a_offset [[buffer(7)]],
                     constant int& b_offset [[buffer(8)]],
                     uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= out_shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint a_idx = compute_index_from_linear(out_idx, out_shape, a_strides, ndim) + a_offset;
    uint b_idx = compute_index_from_linear(out_idx, out_shape, b_strides, ndim) + b_offset;
    
    half2 av = a[a_idx];
    half2 bv = b[b_idx];
    
    // Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    out[out_idx] = half2(av.x * bv.x - av.y * bv.y, av.x * bv.y + av.y * bv.x);
}

kernel void fdiv_half2(device half2* out [[buffer(0)]],
                      device const half2* a [[buffer(1)]],
                      device const half2* b [[buffer(2)]],
                      constant uint* out_shape [[buffer(3)]],
                      constant int* a_strides [[buffer(4)]],
                      constant int* b_strides [[buffer(5)]],
                      constant uint& ndim [[buffer(6)]],
                      constant int& a_offset [[buffer(7)]],
                      constant int& b_offset [[buffer(8)]],
                      uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= out_shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint a_idx = compute_index_from_linear(out_idx, out_shape, a_strides, ndim) + a_offset;
    uint b_idx = compute_index_from_linear(out_idx, out_shape, b_strides, ndim) + b_offset;
    
    half2 av = a[a_idx];
    half2 bv = b[b_idx];
    
    // Complex division: (a + bi) / (c + di) = ((ac + bd) + (bc - ad)i) / (c^2 + d^2)
    half denom = bv.x * bv.x + bv.y * bv.y;
    out[out_idx] = half2((av.x * bv.x + av.y * bv.y) / denom,
                        (av.y * bv.x - av.x * bv.y) / denom);
}

// Complex power operations
kernel void pow_float2(device float2* out [[buffer(0)]],
                      device const float2* a [[buffer(1)]],
                      device const float2* b [[buffer(2)]],
                      constant uint* out_shape [[buffer(3)]],
                      constant int* a_strides [[buffer(4)]],
                      constant int* b_strides [[buffer(5)]],
                      constant uint& ndim [[buffer(6)]],
                      constant int& a_offset [[buffer(7)]],
                      constant int& b_offset [[buffer(8)]],
                      uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= out_shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint a_idx = compute_index_from_linear(out_idx, out_shape, a_strides, ndim) + a_offset;
    uint b_idx = compute_index_from_linear(out_idx, out_shape, b_strides, ndim) + b_offset;
    
    float2 av = a[a_idx];
    float2 bv = b[b_idx];
    
    // Complex power: a^b = exp(b * log(a))
    // log(a) = log(|a|) + i*arg(a)
    float mag_a = sqrt(av.x * av.x + av.y * av.y);
    float arg_a = atan2(av.y, av.x);
    float2 log_a = float2(log(mag_a), arg_a);
    
    // b * log(a)
    float2 b_log_a = float2(bv.x * log_a.x - bv.y * log_a.y,
                            bv.x * log_a.y + bv.y * log_a.x);
    
    // exp(b * log(a))
    float exp_real = exp(b_log_a.x);
    out[out_idx] = float2(exp_real * cos(b_log_a.y), exp_real * sin(b_log_a.y));
}

kernel void pow_half2(device half2* out [[buffer(0)]],
                     device const half2* a [[buffer(1)]],
                     device const half2* b [[buffer(2)]],
                     constant uint* out_shape [[buffer(3)]],
                     constant int* a_strides [[buffer(4)]],
                     constant int* b_strides [[buffer(5)]],
                     constant uint& ndim [[buffer(6)]],
                     constant int& a_offset [[buffer(7)]],
                     constant int& b_offset [[buffer(8)]],
                     uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= out_shape[i];
    }
    if (out_idx >= total_size) return;
    
    uint a_idx = compute_index_from_linear(out_idx, out_shape, a_strides, ndim) + a_offset;
    uint b_idx = compute_index_from_linear(out_idx, out_shape, b_strides, ndim) + b_offset;
    
    half2 av = a[a_idx];
    half2 bv = b[b_idx];
    
    // Complex power: a^b = exp(b * log(a))
    // log(a) = log(|a|) + i*arg(a)
    half mag_a = sqrt(av.x * av.x + av.y * av.y);
    half arg_a = atan2(av.y, av.x);
    half2 log_a = half2(log(mag_a), arg_a);
    
    // b * log(a)
    half2 b_log_a = half2(bv.x * log_a.x - bv.y * log_a.y,
                         bv.x * log_a.y + bv.y * log_a.x);
    
    // exp(b * log(a))
    half exp_real = exp(b_log_a.x);
    out[out_idx] = half2(exp_real * cos(b_log_a.y), exp_real * sin(b_log_a.y));
}

// Bitwise operations for integer types
DEFINE_BINARY_OP(xor, ^, char)
DEFINE_BINARY_OP(xor, ^, uchar)
DEFINE_BINARY_OP(xor, ^, short)
DEFINE_BINARY_OP(xor, ^, ushort)
DEFINE_BINARY_OP(xor, ^, int)
DEFINE_BINARY_OP(xor, ^, long)

DEFINE_BINARY_OP(or, |, char)
DEFINE_BINARY_OP(or, |, uchar)
DEFINE_BINARY_OP(or, |, short)
DEFINE_BINARY_OP(or, |, ushort)
DEFINE_BINARY_OP(or, |, int)
DEFINE_BINARY_OP(or, |, long)

DEFINE_BINARY_OP(and, &, char)
DEFINE_BINARY_OP(and, &, uchar)
DEFINE_BINARY_OP(and, &, short)
DEFINE_BINARY_OP(and, &, ushort)
DEFINE_BINARY_OP(and, &, int)
DEFINE_BINARY_OP(and, &, long)
