#include <metal_stdlib>
using namespace metal;

// Helper for broadcasting - compute linear index from position and strides
inline uint compute_index(uint3 pos, constant uint* shape, constant int* strides, uint ndim) {
    uint idx = 0;
    for (uint i = 0; i < ndim; i++) {
        uint coord = i == 0 ? pos.x : (i == 1 ? pos.y : pos.z);
        if (i < ndim) {
            coord = coord % shape[i];
            idx += coord * strides[i];
        }
    }
    return idx;
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
                         uint3 gid [[thread_position_in_grid]]) { \
    uint out_idx = gid.x; \
    if (out_idx >= out_shape[0] * (ndim > 1 ? out_shape[1] : 1) * (ndim > 2 ? out_shape[2] : 1)) return; \
    \
    uint3 pos; \
    uint temp = out_idx; \
    if (ndim > 2) { \
        pos.z = temp % out_shape[2]; \
        temp /= out_shape[2]; \
    } else { \
        pos.z = 0; \
    } \
    if (ndim > 1) { \
        pos.y = temp % out_shape[1]; \
        temp /= out_shape[1]; \
    } else { \
        pos.y = 0; \
    } \
    pos.x = temp; \
    \
    uint a_idx = compute_index(pos, out_shape, a_strides, ndim); \
    uint b_idx = compute_index(pos, out_shape, b_strides, ndim); \
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
                    uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = out_shape[0] * (ndim > 1 ? out_shape[1] : 1) * (ndim > 2 ? out_shape[2] : 1);
    if (out_idx >= total_size) return;
    
    uint3 pos;
    uint temp = out_idx;
    if (ndim > 2) {
        pos.z = temp % out_shape[2];
        temp /= out_shape[2];
    } else {
        pos.z = 0;
    }
    if (ndim > 1) {
        pos.y = temp % out_shape[1];
        temp /= out_shape[1];
    } else {
        pos.y = 0;
    }
    pos.x = temp;
    
    uint a_idx = compute_index(pos, out_shape, a_strides, ndim);
    uint b_idx = compute_index(pos, out_shape, b_strides, ndim);
    
    out[out_idx] = a[a_idx] / b[b_idx];
}

kernel void idiv_long(device long* out [[buffer(0)]],
                     device const long* a [[buffer(1)]],
                     device const long* b [[buffer(2)]],
                     constant uint* out_shape [[buffer(3)]],
                     constant int* a_strides [[buffer(4)]],
                     constant int* b_strides [[buffer(5)]],
                     constant uint& ndim [[buffer(6)]],
                     uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = out_shape[0] * (ndim > 1 ? out_shape[1] : 1) * (ndim > 2 ? out_shape[2] : 1);
    if (out_idx >= total_size) return;
    
    uint3 pos;
    uint temp = out_idx;
    if (ndim > 2) {
        pos.z = temp % out_shape[2];
        temp /= out_shape[2];
    } else {
        pos.z = 0;
    }
    if (ndim > 1) {
        pos.y = temp % out_shape[1];
        temp /= out_shape[1];
    } else {
        pos.y = 0;
    }
    pos.x = temp;
    
    uint a_idx = compute_index(pos, out_shape, a_strides, ndim);
    uint b_idx = compute_index(pos, out_shape, b_strides, ndim);
    
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
                     uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = out_shape[0] * (ndim > 1 ? out_shape[1] : 1) * (ndim > 2 ? out_shape[2] : 1);
    if (out_idx >= total_size) return;
    
    uint3 pos;
    uint temp = out_idx;
    if (ndim > 2) {
        pos.z = temp % out_shape[2];
        temp /= out_shape[2];
    } else {
        pos.z = 0;
    }
    if (ndim > 1) {
        pos.y = temp % out_shape[1];
        temp /= out_shape[1];
    } else {
        pos.y = 0;
    }
    pos.x = temp;
    
    uint a_idx = compute_index(pos, out_shape, a_strides, ndim);
    uint b_idx = compute_index(pos, out_shape, b_strides, ndim);
    
    out[out_idx] = fmax(a[a_idx], b[b_idx]);
}


kernel void max_int(device int* out [[buffer(0)]],
                   device const int* a [[buffer(1)]],
                   device const int* b [[buffer(2)]],
                   constant uint* out_shape [[buffer(3)]],
                   constant int* a_strides [[buffer(4)]],
                   constant int* b_strides [[buffer(5)]],
                   constant uint& ndim [[buffer(6)]],
                   uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = out_shape[0] * (ndim > 1 ? out_shape[1] : 1) * (ndim > 2 ? out_shape[2] : 1);
    if (out_idx >= total_size) return;
    
    uint3 pos;
    uint temp = out_idx;
    if (ndim > 2) {
        pos.z = temp % out_shape[2];
        temp /= out_shape[2];
    } else {
        pos.z = 0;
    }
    if (ndim > 1) {
        pos.y = temp % out_shape[1];
        temp /= out_shape[1];
    } else {
        pos.y = 0;
    }
    pos.x = temp;
    
    uint a_idx = compute_index(pos, out_shape, a_strides, ndim);
    uint b_idx = compute_index(pos, out_shape, b_strides, ndim);
    
    out[out_idx] = max(a[a_idx], b[b_idx]);
}

kernel void max_long(device long* out [[buffer(0)]],
                    device const long* a [[buffer(1)]],
                    device const long* b [[buffer(2)]],
                    constant uint* out_shape [[buffer(3)]],
                    constant int* a_strides [[buffer(4)]],
                    constant int* b_strides [[buffer(5)]],
                    constant uint& ndim [[buffer(6)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = out_shape[0] * (ndim > 1 ? out_shape[1] : 1) * (ndim > 2 ? out_shape[2] : 1);
    if (out_idx >= total_size) return;
    
    uint3 pos;
    uint temp = out_idx;
    if (ndim > 2) {
        pos.z = temp % out_shape[2];
        temp /= out_shape[2];
    } else {
        pos.z = 0;
    }
    if (ndim > 1) {
        pos.y = temp % out_shape[1];
        temp /= out_shape[1];
    } else {
        pos.y = 0;
    }
    pos.x = temp;
    
    uint a_idx = compute_index(pos, out_shape, a_strides, ndim);
    uint b_idx = compute_index(pos, out_shape, b_strides, ndim);
    
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
                   uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = out_shape[0] * (ndim > 1 ? out_shape[1] : 1) * (ndim > 2 ? out_shape[2] : 1);
    if (out_idx >= total_size) return;
    
    uint3 pos;
    uint temp = out_idx;
    if (ndim > 2) {
        pos.z = temp % out_shape[2];
        temp /= out_shape[2];
    } else {
        pos.z = 0;
    }
    if (ndim > 1) {
        pos.y = temp % out_shape[1];
        temp /= out_shape[1];
    } else {
        pos.y = 0;
    }
    pos.x = temp;
    
    uint a_idx = compute_index(pos, out_shape, a_strides, ndim);
    uint b_idx = compute_index(pos, out_shape, b_strides, ndim);
    
    out[out_idx] = a[a_idx] % b[b_idx];
}

kernel void mod_long(device long* out [[buffer(0)]],
                    device const long* a [[buffer(1)]],
                    device const long* b [[buffer(2)]],
                    constant uint* out_shape [[buffer(3)]],
                    constant int* a_strides [[buffer(4)]],
                    constant int* b_strides [[buffer(5)]],
                    constant uint& ndim [[buffer(6)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = out_shape[0] * (ndim > 1 ? out_shape[1] : 1) * (ndim > 2 ? out_shape[2] : 1);
    if (out_idx >= total_size) return;
    
    uint3 pos;
    uint temp = out_idx;
    if (ndim > 2) {
        pos.z = temp % out_shape[2];
        temp /= out_shape[2];
    } else {
        pos.z = 0;
    }
    if (ndim > 1) {
        pos.y = temp % out_shape[1];
        temp /= out_shape[1];
    } else {
        pos.y = 0;
    }
    pos.x = temp;
    
    uint a_idx = compute_index(pos, out_shape, a_strides, ndim);
    uint b_idx = compute_index(pos, out_shape, b_strides, ndim);
    
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
                     uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = out_shape[0] * (ndim > 1 ? out_shape[1] : 1) * (ndim > 2 ? out_shape[2] : 1);
    if (out_idx >= total_size) return;
    
    uint3 pos;
    uint temp = out_idx;
    if (ndim > 2) {
        pos.z = temp % out_shape[2];
        temp /= out_shape[2];
    } else {
        pos.z = 0;
    }
    if (ndim > 1) {
        pos.y = temp % out_shape[1];
        temp /= out_shape[1];
    } else {
        pos.y = 0;
    }
    pos.x = temp;
    
    uint a_idx = compute_index(pos, out_shape, a_strides, ndim);
    uint b_idx = compute_index(pos, out_shape, b_strides, ndim);
    
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
                       uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = out_shape[0] * (ndim > 1 ? out_shape[1] : 1) * (ndim > 2 ? out_shape[2] : 1);
    if (out_idx >= total_size) return;
    
    uint3 pos;
    uint temp = out_idx;
    if (ndim > 2) {
        pos.z = temp % out_shape[2];
        temp /= out_shape[2];
    } else {
        pos.z = 0;
    }
    if (ndim > 1) {
        pos.y = temp % out_shape[1];
        temp /= out_shape[1];
    } else {
        pos.y = 0;
    }
    pos.x = temp;
    
    uint a_idx = compute_index(pos, out_shape, a_strides, ndim);
    uint b_idx = compute_index(pos, out_shape, b_strides, ndim);
    
    out[out_idx] = a[a_idx] < b[b_idx] ? 1 : 0;
}

kernel void cmplt_int(device uchar* out [[buffer(0)]],
                     device const int* a [[buffer(1)]],
                     device const int* b [[buffer(2)]],
                     constant uint* out_shape [[buffer(3)]],
                     constant int* a_strides [[buffer(4)]],
                     constant int* b_strides [[buffer(5)]],
                     constant uint& ndim [[buffer(6)]],
                     uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = out_shape[0] * (ndim > 1 ? out_shape[1] : 1) * (ndim > 2 ? out_shape[2] : 1);
    if (out_idx >= total_size) return;
    
    uint3 pos;
    uint temp = out_idx;
    if (ndim > 2) {
        pos.z = temp % out_shape[2];
        temp /= out_shape[2];
    } else {
        pos.z = 0;
    }
    if (ndim > 1) {
        pos.y = temp % out_shape[1];
        temp /= out_shape[1];
    } else {
        pos.y = 0;
    }
    pos.x = temp;
    
    uint a_idx = compute_index(pos, out_shape, a_strides, ndim);
    uint b_idx = compute_index(pos, out_shape, b_strides, ndim);
    
    out[out_idx] = a[a_idx] < b[b_idx] ? 1 : 0;
}

kernel void cmpne_float(device uchar* out [[buffer(0)]],
                       device const float* a [[buffer(1)]],
                       device const float* b [[buffer(2)]],
                       constant uint* out_shape [[buffer(3)]],
                       constant int* a_strides [[buffer(4)]],
                       constant int* b_strides [[buffer(5)]],
                       constant uint& ndim [[buffer(6)]],
                       uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = out_shape[0] * (ndim > 1 ? out_shape[1] : 1) * (ndim > 2 ? out_shape[2] : 1);
    if (out_idx >= total_size) return;
    
    uint3 pos;
    uint temp = out_idx;
    if (ndim > 2) {
        pos.z = temp % out_shape[2];
        temp /= out_shape[2];
    } else {
        pos.z = 0;
    }
    if (ndim > 1) {
        pos.y = temp % out_shape[1];
        temp /= out_shape[1];
    } else {
        pos.y = 0;
    }
    pos.x = temp;
    
    uint a_idx = compute_index(pos, out_shape, a_strides, ndim);
    uint b_idx = compute_index(pos, out_shape, b_strides, ndim);
    
    out[out_idx] = a[a_idx] != b[b_idx] ? 1 : 0;
}

kernel void cmpne_int(device uchar* out [[buffer(0)]],
                     device const int* a [[buffer(1)]],
                     device const int* b [[buffer(2)]],
                     constant uint* out_shape [[buffer(3)]],
                     constant int* a_strides [[buffer(4)]],
                     constant int* b_strides [[buffer(5)]],
                     constant uint& ndim [[buffer(6)]],
                     uint3 gid [[thread_position_in_grid]]) {
    uint out_idx = gid.x;
    uint total_size = out_shape[0] * (ndim > 1 ? out_shape[1] : 1) * (ndim > 2 ? out_shape[2] : 1);
    if (out_idx >= total_size) return;
    
    uint3 pos;
    uint temp = out_idx;
    if (ndim > 2) {
        pos.z = temp % out_shape[2];
        temp /= out_shape[2];
    } else {
        pos.z = 0;
    }
    if (ndim > 1) {
        pos.y = temp % out_shape[1];
        temp /= out_shape[1];
    } else {
        pos.y = 0;
    }
    pos.x = temp;
    
    uint a_idx = compute_index(pos, out_shape, a_strides, ndim);
    uint b_idx = compute_index(pos, out_shape, b_strides, ndim);
    
    out[out_idx] = a[a_idx] != b[b_idx] ? 1 : 0;
}

// Bitwise operations
DEFINE_BINARY_OP(xor, ^, int)
DEFINE_BINARY_OP(xor, ^, long)
DEFINE_BINARY_OP(or, |, int)
DEFINE_BINARY_OP(or, |, long)
DEFINE_BINARY_OP(and, &, int)
DEFINE_BINARY_OP(and, &, long)
