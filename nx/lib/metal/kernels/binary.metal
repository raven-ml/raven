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

// Arithmetic operations
DEFINE_BINARY_OP(add, +, float)
DEFINE_BINARY_OP(add, +, int)
DEFINE_BINARY_OP(add, +, long)

DEFINE_BINARY_OP(sub, -, float)
DEFINE_BINARY_OP(sub, -, int)
DEFINE_BINARY_OP(sub, -, long)

DEFINE_BINARY_OP(mul, *, float)
DEFINE_BINARY_OP(mul, *, int)
DEFINE_BINARY_OP(mul, *, long)

// Division - separate for int (truncating) and float
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

DEFINE_BINARY_OP(fdiv, /, float)

// Max operation
kernel void max_float(device float* out [[buffer(0)]],
                     device const float* a [[buffer(1)]],
                     device const float* b [[buffer(2)]],
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
    
    out[out_idx] = fmax(a[a_idx], b[b_idx]);
}

kernel void max_int(device int* out [[buffer(0)]],
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
    
    out[out_idx] = max(a[a_idx], b[b_idx]);
}

kernel void max_long(device long* out [[buffer(0)]],
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
    
    out[out_idx] = max(a[a_idx], b[b_idx]);
}

// Modulo
kernel void mod_int(device int* out [[buffer(0)]],
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
    
    out[out_idx] = a[a_idx] % b[b_idx];
}

kernel void mod_long(device long* out [[buffer(0)]],
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
    
    out[out_idx] = a[a_idx] % b[b_idx];
}

// Power
kernel void pow_float(device float* out [[buffer(0)]],
                     device const float* a [[buffer(1)]],
                     device const float* b [[buffer(2)]],
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
    
    out[out_idx] = pow(a[a_idx], b[b_idx]);
}

// Comparison operations - output is uint8 (0 or 1)
kernel void cmplt_float(device uchar* out [[buffer(0)]],
                       device const float* a [[buffer(1)]],
                       device const float* b [[buffer(2)]],
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
    
    out[out_idx] = a[a_idx] < b[b_idx] ? 1 : 0;
}

kernel void cmplt_int(device uchar* out [[buffer(0)]],
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
    
    out[out_idx] = a[a_idx] < b[b_idx] ? 1 : 0;
}

kernel void cmpne_float(device uchar* out [[buffer(0)]],
                       device const float* a [[buffer(1)]],
                       device const float* b [[buffer(2)]],
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
    
    out[out_idx] = a[a_idx] != b[b_idx] ? 1 : 0;
}

kernel void cmpne_int(device uchar* out [[buffer(0)]],
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
    
    out[out_idx] = a[a_idx] != b[b_idx] ? 1 : 0;
}

kernel void cmplt_uchar(device uchar* out [[buffer(0)]],
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
    
    out[out_idx] = a[a_idx] < b[b_idx] ? 1 : 0;
}

kernel void cmpne_uchar(device uchar* out [[buffer(0)]],
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
    
    out[out_idx] = a[a_idx] != b[b_idx] ? 1 : 0;
}

// Bitwise operations
DEFINE_BINARY_OP(xor, ^, uchar)
DEFINE_BINARY_OP(xor, ^, int)
DEFINE_BINARY_OP(xor, ^, long)
DEFINE_BINARY_OP(or, |, uchar)
DEFINE_BINARY_OP(or, |, int)
DEFINE_BINARY_OP(or, |, long)
DEFINE_BINARY_OP(and, &, uchar)
DEFINE_BINARY_OP(and, &, int)
DEFINE_BINARY_OP(and, &, long)
