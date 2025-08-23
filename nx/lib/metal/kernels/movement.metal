#include <metal_stdlib>
using namespace metal;

// ===== COPY OPERATIONS =====
#define DEFINE_COPY(type) \
kernel void copy_##type(device type* out [[buffer(0)]], \
                       device const type* in [[buffer(1)]], \
                       constant uint& size [[buffer(2)]], \
                       uint gid [[thread_position_in_grid]]) { \
    if (gid >= size) return; \
    out[gid] = in[gid]; \
}

// ===== STRIDED COPY OPERATIONS =====
#define DEFINE_STRIDED_COPY(type) \
kernel void strided_copy_##type(device type* out [[buffer(0)]], \
                               device const type* in [[buffer(1)]], \
                               constant uint* shape [[buffer(2)]], \
                               constant int* strides [[buffer(3)]], \
                               constant uint& ndim [[buffer(4)]], \
                               constant uint& size [[buffer(5)]], \
                               constant uint& offset [[buffer(6)]], \
                               uint gid [[thread_position_in_grid]]) { \
    if (gid >= size) return; \
    \
    uint coords[8]; \
    uint temp = gid; \
    \
    for (int i = ndim - 1; i >= 0; i--) { \
        coords[i] = temp % shape[i]; \
        temp /= shape[i]; \
    } \
    \
    uint in_idx = offset; \
    for (uint i = 0; i < ndim; i++) { \
        in_idx += coords[i] * strides[i]; \
    } \
    \
    out[gid] = in[in_idx]; \
}

// ===== PAD OPERATIONS =====
#define DEFINE_PAD(type) \
kernel void pad_##type(device type* out [[buffer(0)]], \
                      device const type* in [[buffer(1)]], \
                      constant uint* out_shape [[buffer(2)]], \
                      constant uint* in_shape [[buffer(3)]], \
                      constant uint* pad_before [[buffer(4)]], \
                      constant type& pad_value [[buffer(5)]], \
                      constant uint& ndim [[buffer(6)]], \
                      constant int* in_strides [[buffer(7)]], \
                      constant uint& in_offset [[buffer(8)]], \
                      uint gid [[thread_position_in_grid]]) { \
    uint out_size = 1; \
    for (uint i = 0; i < ndim; i++) { \
        out_size *= out_shape[i]; \
    } \
    \
    if (gid >= out_size) return; \
    \
    uint out_pos[8]; \
    uint temp = gid; \
    for (int i = ndim - 1; i >= 0; i--) { \
        out_pos[i] = temp % out_shape[i]; \
        temp /= out_shape[i]; \
    } \
    \
    bool in_bounds = true; \
    uint in_pos[8]; \
    \
    for (uint i = 0; i < ndim; i++) { \
        if (out_pos[i] < pad_before[i] || out_pos[i] >= pad_before[i] + in_shape[i]) { \
            in_bounds = false; \
            break; \
        } \
        in_pos[i] = out_pos[i] - pad_before[i]; \
    } \
    \
    if (in_bounds) { \
        uint in_idx = in_offset; \
        for (uint i = 0; i < ndim; i++) { \
            in_idx += in_pos[i] * in_strides[i]; \
        } \
        out[gid] = in[in_idx]; \
    } else { \
        out[gid] = pad_value; \
    } \
}

// ===== CONCAT OPERATIONS =====
#define DEFINE_CONCAT_AXIS(type) \
kernel void concat_axis_##type(device type* out [[buffer(0)]], \
                              device const type* in [[buffer(1)]], \
                              constant uint* out_shape [[buffer(2)]], \
                              constant uint* in_shape [[buffer(3)]], \
                              constant uint& axis [[buffer(4)]], \
                              constant uint& in_offset_along_axis [[buffer(5)]], \
                              constant uint& ndim [[buffer(6)]], \
                              constant int* in_strides [[buffer(7)]], \
                              constant uint& in_offset [[buffer(8)]], \
                              uint gid [[thread_position_in_grid]]) { \
    /* Calculate input size */ \
    uint in_size = 1; \
    for (uint i = 0; i < ndim; i++) { \
        in_size *= in_shape[i]; \
    } \
    \
    if (gid >= in_size) return; \
    \
    /* Convert gid to input coordinates */ \
    uint in_pos[8]; \
    uint temp = gid; \
    for (int i = ndim - 1; i >= 0; i--) { \
        in_pos[i] = temp % in_shape[i]; \
        temp /= in_shape[i]; \
    } \
    \
    /* Calculate input index using strides */ \
    uint in_idx = in_offset; \
    for (uint i = 0; i < ndim; i++) { \
        in_idx += in_pos[i] * in_strides[i]; \
    } \
    \
    /* Calculate output coordinates - same as input except along concat axis */ \
    uint out_pos[8]; \
    for (uint i = 0; i < ndim; i++) { \
        if (i == axis) { \
            out_pos[i] = in_pos[i] + in_offset_along_axis; \
        } else { \
            out_pos[i] = in_pos[i]; \
        } \
    } \
    \
    /* Calculate output index */ \
    uint out_idx = 0; \
    uint out_stride = 1; \
    for (int i = ndim - 1; i >= 0; i--) { \
        out_idx += out_pos[i] * out_stride; \
        out_stride *= out_shape[i]; \
    } \
    \
    out[out_idx] = in[in_idx]; \
}

// ===== PERMUTE OPERATIONS =====
#define DEFINE_PERMUTE(type) \
kernel void permute_##type(device type* out [[buffer(0)]], \
                          device const type* in [[buffer(1)]], \
                          constant uint* shape [[buffer(2)]], \
                          constant uint* axes [[buffer(3)]], \
                          constant int* in_strides [[buffer(4)]], \
                          constant uint& ndim [[buffer(5)]], \
                          constant uint& size [[buffer(6)]], \
                          constant uint& in_offset [[buffer(7)]], \
                          uint gid [[thread_position_in_grid]]) { \
    if (gid >= size) return; \
    \
    uint out_pos[8]; \
    uint temp = gid; \
    \
    for (int i = ndim - 1; i >= 0; i--) { \
        out_pos[i] = temp % shape[axes[i]]; \
        temp /= shape[axes[i]]; \
    } \
    \
    uint in_idx = in_offset; \
    for (uint i = 0; i < ndim; i++) { \
        in_idx += out_pos[axes[i]] * in_strides[i]; \
    } \
    \
    out[gid] = in[in_idx]; \
}

// ===== SLICE OPERATIONS =====  
#define DEFINE_SLICE(type) \
kernel void slice_##type(device type* out [[buffer(0)]], \
                        device const type* in [[buffer(1)]], \
                        constant uint* out_shape [[buffer(2)]], \
                        constant uint* starts [[buffer(3)]], \
                        constant int* in_strides [[buffer(4)]], \
                        constant uint& ndim [[buffer(5)]], \
                        constant uint& size [[buffer(6)]], \
                        constant uint& in_offset [[buffer(7)]], \
                        uint gid [[thread_position_in_grid]]) { \
    if (gid >= size) return; \
    \
    uint out_pos[8]; \
    uint temp = gid; \
    \
    for (int i = ndim - 1; i >= 0; i--) { \
        out_pos[i] = temp % out_shape[i]; \
        temp /= out_shape[i]; \
    } \
    \
    uint in_idx = in_offset; \
    for (uint i = 0; i < ndim; i++) { \
        in_idx += (out_pos[i] + starts[i]) * in_strides[i]; \
    } \
    \
    out[gid] = in[in_idx]; \
}

// Instantiate for all types
DEFINE_COPY(float)
DEFINE_COPY(half)
DEFINE_COPY(int)
DEFINE_COPY(long)
DEFINE_COPY(char)
DEFINE_COPY(uchar)
DEFINE_COPY(short)
DEFINE_COPY(ushort)

DEFINE_STRIDED_COPY(float)
DEFINE_STRIDED_COPY(half)
DEFINE_STRIDED_COPY(int)
DEFINE_STRIDED_COPY(long)
DEFINE_STRIDED_COPY(char)
DEFINE_STRIDED_COPY(uchar)
DEFINE_STRIDED_COPY(short)
DEFINE_STRIDED_COPY(ushort)

DEFINE_PAD(float)
DEFINE_PAD(half)
DEFINE_PAD(int)
DEFINE_PAD(long)
DEFINE_PAD(char)
DEFINE_PAD(uchar)
DEFINE_PAD(short)
DEFINE_PAD(ushort)

DEFINE_CONCAT_AXIS(float)
DEFINE_CONCAT_AXIS(half)
DEFINE_CONCAT_AXIS(int)
DEFINE_CONCAT_AXIS(long)
DEFINE_CONCAT_AXIS(char)
DEFINE_CONCAT_AXIS(uchar)
DEFINE_CONCAT_AXIS(short)
DEFINE_CONCAT_AXIS(ushort)

DEFINE_PERMUTE(float)
DEFINE_PERMUTE(half)
DEFINE_PERMUTE(int)
DEFINE_PERMUTE(long)
DEFINE_PERMUTE(char)
DEFINE_PERMUTE(uchar)
DEFINE_PERMUTE(short)
DEFINE_PERMUTE(ushort)

DEFINE_SLICE(float)
DEFINE_SLICE(half)
DEFINE_SLICE(int)
DEFINE_SLICE(long)
DEFINE_SLICE(char)
DEFINE_SLICE(uchar)
DEFINE_SLICE(short)
DEFINE_SLICE(ushort)

// Complex types
DEFINE_COPY(float2)
DEFINE_COPY(half2)
DEFINE_STRIDED_COPY(float2)
DEFINE_STRIDED_COPY(half2)
DEFINE_PAD(float2)
DEFINE_PAD(half2)
DEFINE_CONCAT_AXIS(float2)
DEFINE_CONCAT_AXIS(half2)
DEFINE_PERMUTE(float2)
DEFINE_PERMUTE(half2)
DEFINE_SLICE(float2)
DEFINE_SLICE(half2)