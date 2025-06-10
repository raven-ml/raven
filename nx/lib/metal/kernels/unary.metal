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
    out[out_idx] = 1.0f / in[in_idx];
}