#include <metal_stdlib>
using namespace metal;

// Helper for strided indexing - compute linear index from position and strides
inline uint compute_strided_index(uint3 pos, constant uint* shape, constant int* strides, uint ndim) {
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

// Where (conditional select) operation with stride support
kernel void where_float(device float* out [[buffer(0)]],
                       device const uchar* cond [[buffer(1)]],
                       device const float* if_true [[buffer(2)]],
                       device const float* if_false [[buffer(3)]],
                       constant uint* shape [[buffer(4)]],
                       constant int* cond_strides [[buffer(5)]],
                       constant int* true_strides [[buffer(6)]],
                       constant int* false_strides [[buffer(7)]],
                       constant uint& ndim [[buffer(8)]],
                       constant uint& cond_offset [[buffer(9)]],
                       constant uint& true_offset [[buffer(10)]],
                       constant uint& false_offset [[buffer(11)]],
                       uint gid [[thread_position_in_grid]]) {
    uint total_size = ndim == 0 ? 1 : shape[0] * (ndim > 1 ? shape[1] : 1) * (ndim > 2 ? shape[2] : 1);
    if (gid >= total_size) return;
    
    // Convert linear index to position
    uint3 pos;
    uint temp = gid;
    if (ndim > 2) {
        pos.z = temp % shape[2];
        temp /= shape[2];
    } else {
        pos.z = 0;
    }
    if (ndim > 1) {
        pos.y = temp % shape[1];
        temp /= shape[1];
    } else {
        pos.y = 0;
    }
    pos.x = temp;
    
    // Compute strided indices with offsets
    uint cond_idx = cond_offset + compute_strided_index(pos, shape, cond_strides, ndim);
    uint true_idx = true_offset + compute_strided_index(pos, shape, true_strides, ndim);
    uint false_idx = false_offset + compute_strided_index(pos, shape, false_strides, ndim);
    
    out[gid] = cond[cond_idx] ? if_true[true_idx] : if_false[false_idx];
}


kernel void where_int(device int* out [[buffer(0)]],
                     device const uchar* cond [[buffer(1)]],
                     device const int* if_true [[buffer(2)]],
                     device const int* if_false [[buffer(3)]],
                     constant uint* shape [[buffer(4)]],
                     constant int* cond_strides [[buffer(5)]],
                     constant int* true_strides [[buffer(6)]],
                     constant int* false_strides [[buffer(7)]],
                     constant uint& ndim [[buffer(8)]],
                     constant uint& cond_offset [[buffer(9)]],
                     constant uint& true_offset [[buffer(10)]],
                     constant uint& false_offset [[buffer(11)]],
                     uint gid [[thread_position_in_grid]]) {
    uint total_size = ndim == 0 ? 1 : shape[0] * (ndim > 1 ? shape[1] : 1) * (ndim > 2 ? shape[2] : 1);
    if (gid >= total_size) return;
    
    // Convert linear index to position
    uint3 pos;
    uint temp = gid;
    if (ndim > 2) {
        pos.z = temp % shape[2];
        temp /= shape[2];
    } else {
        pos.z = 0;
    }
    if (ndim > 1) {
        pos.y = temp % shape[1];
        temp /= shape[1];
    } else {
        pos.y = 0;
    }
    pos.x = temp;
    
    // Compute strided indices with offsets
    uint cond_idx = cond_offset + compute_strided_index(pos, shape, cond_strides, ndim);
    uint true_idx = true_offset + compute_strided_index(pos, shape, true_strides, ndim);
    uint false_idx = false_offset + compute_strided_index(pos, shape, false_strides, ndim);
    
    out[gid] = cond[cond_idx] ? if_true[true_idx] : if_false[false_idx];
}

kernel void where_long(device long* out [[buffer(0)]],
                      device const uchar* cond [[buffer(1)]],
                      device const long* if_true [[buffer(2)]],
                      device const long* if_false [[buffer(3)]],
                      constant uint* shape [[buffer(4)]],
                      constant int* cond_strides [[buffer(5)]],
                      constant int* true_strides [[buffer(6)]],
                      constant int* false_strides [[buffer(7)]],
                      constant uint& ndim [[buffer(8)]],
                      constant uint& cond_offset [[buffer(9)]],
                      constant uint& true_offset [[buffer(10)]],
                      constant uint& false_offset [[buffer(11)]],
                      uint gid [[thread_position_in_grid]]) {
    uint total_size = ndim == 0 ? 1 : shape[0] * (ndim > 1 ? shape[1] : 1) * (ndim > 2 ? shape[2] : 1);
    if (gid >= total_size) return;
    
    // Convert linear index to position
    uint3 pos;
    uint temp = gid;
    if (ndim > 2) {
        pos.z = temp % shape[2];
        temp /= shape[2];
    } else {
        pos.z = 0;
    }
    if (ndim > 1) {
        pos.y = temp % shape[1];
        temp /= shape[1];
    } else {
        pos.y = 0;
    }
    pos.x = temp;
    
    // Compute strided indices with offsets
    uint cond_idx = cond_offset + compute_strided_index(pos, shape, cond_strides, ndim);
    uint true_idx = true_offset + compute_strided_index(pos, shape, true_strides, ndim);
    uint false_idx = false_offset + compute_strided_index(pos, shape, false_strides, ndim);
    
    out[gid] = cond[cond_idx] ? if_true[true_idx] : if_false[false_idx];
}

kernel void where_uchar(device uchar* out [[buffer(0)]],
                       device const uchar* cond [[buffer(1)]],
                       device const uchar* if_true [[buffer(2)]],
                       device const uchar* if_false [[buffer(3)]],
                       constant uint* shape [[buffer(4)]],
                       constant int* cond_strides [[buffer(5)]],
                       constant int* true_strides [[buffer(6)]],
                       constant int* false_strides [[buffer(7)]],
                       constant uint& ndim [[buffer(8)]],
                       constant uint& cond_offset [[buffer(9)]],
                       constant uint& true_offset [[buffer(10)]],
                       constant uint& false_offset [[buffer(11)]],
                       uint gid [[thread_position_in_grid]]) {
    uint total_size = ndim == 0 ? 1 : shape[0] * (ndim > 1 ? shape[1] : 1) * (ndim > 2 ? shape[2] : 1);
    if (gid >= total_size) return;
    
    // Convert linear index to position
    uint3 pos;
    uint temp = gid;
    if (ndim > 2) {
        pos.z = temp % shape[2];
        temp /= shape[2];
    } else {
        pos.z = 0;
    }
    if (ndim > 1) {
        pos.y = temp % shape[1];
        temp /= shape[1];
    } else {
        pos.y = 0;
    }
    pos.x = temp;
    
    // Compute strided indices with offsets
    uint cond_idx = cond_offset + compute_strided_index(pos, shape, cond_strides, ndim);
    uint true_idx = true_offset + compute_strided_index(pos, shape, true_strides, ndim);
    uint false_idx = false_offset + compute_strided_index(pos, shape, false_strides, ndim);
    
    out[gid] = cond[cond_idx] ? if_true[true_idx] : if_false[false_idx];
}

// Type casting kernels

kernel void cast_float_to_int(device int* out [[buffer(0)]],
                             device const float* in [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = int(in[gid]);
}

kernel void cast_float_to_long(device long* out [[buffer(0)]],
                              device const float* in [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = long(in[gid]);
}



kernel void cast_int_to_float(device float* out [[buffer(0)]],
                             device const int* in [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = float(in[gid]);
}


kernel void cast_int_to_long(device long* out [[buffer(0)]],
                            device const int* in [[buffer(1)]],
                            constant uint& size [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = long(in[gid]);
}

kernel void cast_long_to_int(device int* out [[buffer(0)]],
                            device const long* in [[buffer(1)]],
                            constant uint& size [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = int(in[gid]);
}

kernel void cast_long_to_float(device float* out [[buffer(0)]],
                              device const long* in [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = float(in[gid]);
}

kernel void cast_uchar_to_int(device int* out [[buffer(0)]],
                             device const uchar* in [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = int(in[gid]);
}

kernel void cast_int_to_uchar(device uchar* out [[buffer(0)]],
                             device const int* in [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = uchar(in[gid] != 0 ? 1 : 0);
}

kernel void cast_float_to_uchar(device uchar* out [[buffer(0)]],
                               device const float* in [[buffer(1)]],
                               constant uint& size [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = uchar(in[gid]);
}


kernel void cast_float_to_short(device short* out [[buffer(0)]],
                               device const float* in [[buffer(1)]],
                               constant uint& size [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = short(in[gid]);
}

kernel void cast_uchar_to_float(device float* out [[buffer(0)]],
                               device const uchar* in [[buffer(1)]],
                               constant uint& size [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = float(in[gid]);
}

// Comprehensive cast operations for all supported type combinations

// Half to other types
kernel void cast_half_to_float(device float* out [[buffer(0)]],
                              device const half* in [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = float(in[gid]);
}

kernel void cast_half_to_int(device int* out [[buffer(0)]],
                            device const half* in [[buffer(1)]],
                            constant uint& size [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = int(in[gid]);
}

kernel void cast_half_to_long(device long* out [[buffer(0)]],
                             device const half* in [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = long(in[gid]);
}

kernel void cast_half_to_char(device char* out [[buffer(0)]],
                             device const half* in [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = char(in[gid]);
}

kernel void cast_half_to_uchar(device uchar* out [[buffer(0)]],
                              device const half* in [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = uchar(in[gid]);
}

kernel void cast_half_to_short(device short* out [[buffer(0)]],
                              device const half* in [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = short(in[gid]);
}

kernel void cast_half_to_ushort(device ushort* out [[buffer(0)]],
                               device const half* in [[buffer(1)]],
                               constant uint& size [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = ushort(in[gid]);
}

kernel void cast_half_to_half(device half* out [[buffer(0)]],
                             device const half* in [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = in[gid];
}

// Float to other types (add missing ones)
kernel void cast_float_to_half(device half* out [[buffer(0)]],
                              device const float* in [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = half(in[gid]);
}

kernel void cast_float_to_float(device float* out [[buffer(0)]],
                               device const float* in [[buffer(1)]],
                               constant uint& size [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = in[gid];
}

kernel void cast_float_to_char(device char* out [[buffer(0)]],
                              device const float* in [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = char(in[gid]);
}

kernel void cast_float_to_ushort(device ushort* out [[buffer(0)]],
                                device const float* in [[buffer(1)]],
                                constant uint& size [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = ushort(in[gid]);
}

// Char to other types
kernel void cast_char_to_half(device half* out [[buffer(0)]],
                             device const char* in [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = half(in[gid]);
}

kernel void cast_char_to_float(device float* out [[buffer(0)]],
                              device const char* in [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = float(in[gid]);
}

kernel void cast_char_to_int(device int* out [[buffer(0)]],
                            device const char* in [[buffer(1)]],
                            constant uint& size [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = int(in[gid]);
}

kernel void cast_char_to_long(device long* out [[buffer(0)]],
                             device const char* in [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = long(in[gid]);
}

kernel void cast_char_to_char(device char* out [[buffer(0)]],
                             device const char* in [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = in[gid];
}

kernel void cast_char_to_uchar(device uchar* out [[buffer(0)]],
                              device const char* in [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = uchar(in[gid]);
}

kernel void cast_char_to_short(device short* out [[buffer(0)]],
                              device const char* in [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = short(in[gid]);
}

kernel void cast_char_to_ushort(device ushort* out [[buffer(0)]],
                               device const char* in [[buffer(1)]],
                               constant uint& size [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = ushort(in[gid]);
}

// Uchar to other types (add missing ones)
kernel void cast_uchar_to_half(device half* out [[buffer(0)]],
                              device const uchar* in [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = half(in[gid]);
}

kernel void cast_uchar_to_char(device char* out [[buffer(0)]],
                              device const uchar* in [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = char(in[gid]);
}

kernel void cast_uchar_to_short(device short* out [[buffer(0)]],
                               device const uchar* in [[buffer(1)]],
                               constant uint& size [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = short(in[gid]);
}

kernel void cast_uchar_to_ushort(device ushort* out [[buffer(0)]],
                                device const uchar* in [[buffer(1)]],
                                constant uint& size [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = ushort(in[gid]);
}

kernel void cast_uchar_to_long(device long* out [[buffer(0)]],
                              device const uchar* in [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = long(in[gid]);
}

kernel void cast_uchar_to_uchar(device uchar* out [[buffer(0)]],
                               device const uchar* in [[buffer(1)]],
                               constant uint& size [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = in[gid];
}

// Short to other types
kernel void cast_short_to_half(device half* out [[buffer(0)]],
                              device const short* in [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = half(in[gid]);
}

kernel void cast_short_to_float(device float* out [[buffer(0)]],
                               device const short* in [[buffer(1)]],
                               constant uint& size [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = float(in[gid]);
}

kernel void cast_short_to_int(device int* out [[buffer(0)]],
                             device const short* in [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = int(in[gid]);
}

kernel void cast_short_to_long(device long* out [[buffer(0)]],
                              device const short* in [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = long(in[gid]);
}

kernel void cast_short_to_char(device char* out [[buffer(0)]],
                              device const short* in [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = char(in[gid]);
}

kernel void cast_short_to_uchar(device uchar* out [[buffer(0)]],
                               device const short* in [[buffer(1)]],
                               constant uint& size [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = uchar(in[gid]);
}

kernel void cast_short_to_short(device short* out [[buffer(0)]],
                               device const short* in [[buffer(1)]],
                               constant uint& size [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = in[gid];
}

kernel void cast_short_to_ushort(device ushort* out [[buffer(0)]],
                                device const short* in [[buffer(1)]],
                                constant uint& size [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = ushort(in[gid]);
}

// Ushort to other types
kernel void cast_ushort_to_half(device half* out [[buffer(0)]],
                               device const ushort* in [[buffer(1)]],
                               constant uint& size [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = half(in[gid]);
}

kernel void cast_ushort_to_float(device float* out [[buffer(0)]],
                                device const ushort* in [[buffer(1)]],
                                constant uint& size [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = float(in[gid]);
}

kernel void cast_ushort_to_int(device int* out [[buffer(0)]],
                              device const ushort* in [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = int(in[gid]);
}

kernel void cast_ushort_to_long(device long* out [[buffer(0)]],
                               device const ushort* in [[buffer(1)]],
                               constant uint& size [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = long(in[gid]);
}

kernel void cast_ushort_to_char(device char* out [[buffer(0)]],
                               device const ushort* in [[buffer(1)]],
                               constant uint& size [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = char(in[gid]);
}

kernel void cast_ushort_to_uchar(device uchar* out [[buffer(0)]],
                                device const ushort* in [[buffer(1)]],
                                constant uint& size [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = uchar(in[gid]);
}

kernel void cast_ushort_to_short(device short* out [[buffer(0)]],
                                device const ushort* in [[buffer(1)]],
                                constant uint& size [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = short(in[gid]);
}

kernel void cast_ushort_to_ushort(device ushort* out [[buffer(0)]],
                                 device const ushort* in [[buffer(1)]],
                                 constant uint& size [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = in[gid];
}

// Int to other types (add missing ones)
kernel void cast_int_to_half(device half* out [[buffer(0)]],
                            device const int* in [[buffer(1)]],
                            constant uint& size [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = half(in[gid]);
}

kernel void cast_int_to_char(device char* out [[buffer(0)]],
                            device const int* in [[buffer(1)]],
                            constant uint& size [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = char(in[gid]);
}

kernel void cast_int_to_short(device short* out [[buffer(0)]],
                             device const int* in [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = short(in[gid]);
}

kernel void cast_int_to_ushort(device ushort* out [[buffer(0)]],
                              device const int* in [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = ushort(in[gid]);
}

kernel void cast_int_to_int(device int* out [[buffer(0)]],
                           device const int* in [[buffer(1)]],
                           constant uint& size [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = in[gid];
}

// Long to other types (add missing ones)
kernel void cast_long_to_half(device half* out [[buffer(0)]],
                             device const long* in [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = half(in[gid]);
}

kernel void cast_long_to_char(device char* out [[buffer(0)]],
                             device const long* in [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = char(in[gid]);
}

kernel void cast_long_to_uchar(device uchar* out [[buffer(0)]],
                              device const long* in [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = uchar(in[gid]);
}

kernel void cast_long_to_short(device short* out [[buffer(0)]],
                              device const long* in [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = short(in[gid]);
}

kernel void cast_long_to_ushort(device ushort* out [[buffer(0)]],
                               device const long* in [[buffer(1)]],
                               constant uint& size [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = ushort(in[gid]);
}

kernel void cast_long_to_long(device long* out [[buffer(0)]],
                             device const long* in [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = in[gid];
}

// Fill kernel - assigns a constant value
kernel void fill_float(device float* out [[buffer(0)]],
                      constant float& value [[buffer(1)]],
                      constant uint& size [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = value;
}


kernel void fill_int(device int* out [[buffer(0)]],
                    constant int& value [[buffer(1)]],
                    constant uint& size [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = value;
}

kernel void fill_long(device long* out [[buffer(0)]],
                     constant long& value [[buffer(1)]],
                     constant uint& size [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = value;
}

kernel void fill_uchar(device uchar* out [[buffer(0)]],
                      constant uchar& value [[buffer(1)]],
                      constant uint& size [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = value;
}

// Gather operation
kernel void gather_float(device float* out [[buffer(0)]],
                        device const float* data [[buffer(1)]],
                        device const int* indices [[buffer(2)]],
                        constant uint& axis_size [[buffer(3)]],
                        constant uint& inner_size [[buffer(4)]],
                        constant uint& indices_size [[buffer(5)]],
                        constant uint& outer_size [[buffer(6)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= outer_size * indices_size * inner_size) return;
    
    // Decompose global id into outer, index, and inner positions
    uint temp = gid;
    uint inner_pos = temp % inner_size;
    temp /= inner_size;
    uint idx_pos = temp % indices_size;
    uint outer_pos = temp / indices_size;
    
    int idx = indices[idx_pos];
    if (idx < 0) idx += axis_size;
    
    // Compute source index: outer_pos * (axis_size * inner_size) + idx * inner_size + inner_pos
    uint data_idx = outer_pos * axis_size * inner_size + uint(idx) * inner_size + inner_pos;
    out[gid] = data[data_idx];
}

kernel void gather_int(device int* out [[buffer(0)]],
                      device const int* data [[buffer(1)]],
                      device const int* indices [[buffer(2)]],
                      constant uint& axis_size [[buffer(3)]],
                      constant uint& inner_size [[buffer(4)]],
                      constant uint& indices_size [[buffer(5)]],
                      constant uint& outer_size [[buffer(6)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= outer_size * indices_size * inner_size) return;
    
    // Decompose global id into outer, index, and inner positions
    uint temp = gid;
    uint inner_pos = temp % inner_size;
    temp /= inner_size;
    uint idx_pos = temp % indices_size;
    uint outer_pos = temp / indices_size;
    
    int idx = indices[idx_pos];
    if (idx < 0) idx += axis_size;
    
    // Compute source index: outer_pos * (axis_size * inner_size) + idx * inner_size + inner_pos
    uint data_idx = outer_pos * axis_size * inner_size + uint(idx) * inner_size + inner_pos;
    out[gid] = data[data_idx];
}

// Scatter operation - set mode
kernel void scatter_float(device float* out [[buffer(0)]],
                         device const int* indices [[buffer(1)]],
                         device const float* updates [[buffer(2)]],
                         constant uint& axis_size [[buffer(3)]],
                         constant uint& inner_size [[buffer(4)]],
                         constant uint& indices_size [[buffer(5)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= indices_size * inner_size) return;
    
    uint idx_pos = gid / inner_size;
    uint inner_pos = gid % inner_size;
    
    int idx = indices[idx_pos];
    if (idx < 0) idx += axis_size;
    
    uint out_idx = uint(idx) * inner_size + inner_pos;
    out[out_idx] = updates[gid];
}

// Scatter operation - add mode
kernel void scatter_add_float(device float* out [[buffer(0)]],
                             device const int* indices [[buffer(1)]],
                             device const float* updates [[buffer(2)]],
                             constant uint& axis_size [[buffer(3)]],
                             constant uint& inner_size [[buffer(4)]],
                             constant uint& indices_size [[buffer(5)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= indices_size * inner_size) return;
    
    uint idx_pos = gid / inner_size;
    uint inner_pos = gid % inner_size;
    
    int idx = indices[idx_pos];
    if (idx < 0) idx += axis_size;
    
    uint out_idx = uint(idx) * inner_size + inner_pos;
    out[out_idx] += updates[gid];
}

kernel void scatter_int(device int* out [[buffer(0)]],
                       device const int* indices [[buffer(1)]],
                       device const int* updates [[buffer(2)]],
                       constant uint& axis_size [[buffer(3)]],
                       constant uint& inner_size [[buffer(4)]],
                       constant uint& indices_size [[buffer(5)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= indices_size * inner_size) return;
    
    uint idx_pos = gid / inner_size;
    uint inner_pos = gid % inner_size;
    
    int idx = indices[idx_pos];
    if (idx < 0) idx += axis_size;
    
    uint out_idx = uint(idx) * inner_size + inner_pos;
    out[out_idx] = updates[gid];
}

kernel void scatter_add_int(device int* out [[buffer(0)]],
                           device const int* indices [[buffer(1)]],
                           device const int* updates [[buffer(2)]],
                           constant uint& axis_size [[buffer(3)]],
                           constant uint& inner_size [[buffer(4)]],
                           constant uint& indices_size [[buffer(5)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= indices_size * inner_size) return;
    
    uint idx_pos = gid / inner_size;
    uint inner_pos = gid % inner_size;
    
    int idx = indices[idx_pos];
    if (idx < 0) idx += axis_size;
    
    uint out_idx = uint(idx) * inner_size + inner_pos;
    out[out_idx] += updates[gid];
}

kernel void scatter_uchar(device uchar* out [[buffer(0)]],
                         device const int* indices [[buffer(1)]],
                         device const uchar* updates [[buffer(2)]],
                         constant uint& axis_size [[buffer(3)]],
                         constant uint& inner_size [[buffer(4)]],
                         constant uint& indices_size [[buffer(5)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= indices_size * inner_size) return;
    
    uint idx_pos = gid / inner_size;
    uint inner_pos = gid % inner_size;
    
    int idx = indices[idx_pos];
    if (idx < 0) idx += axis_size;
    
    uint out_idx = uint(idx) * inner_size + inner_pos;
    out[out_idx] = updates[gid];
}

kernel void scatter_add_uchar(device uchar* out [[buffer(0)]],
                             device const int* indices [[buffer(1)]],
                             device const uchar* updates [[buffer(2)]],
                             constant uint& axis_size [[buffer(3)]],
                             constant uint& inner_size [[buffer(4)]],
                             constant uint& indices_size [[buffer(5)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= indices_size * inner_size) return;
    
    uint idx_pos = gid / inner_size;
    uint inner_pos = gid % inner_size;
    
    int idx = indices[idx_pos];
    if (idx < 0) idx += axis_size;
    
    uint out_idx = uint(idx) * inner_size + inner_pos;
    out[out_idx] += updates[gid];
}

// Threefry random number generator (simplified version)
kernel void threefry_int32(device int* out [[buffer(0)]],
                          device const int* key [[buffer(1)]],
                          device const int* counter [[buffer(2)]],
                          constant uint& size [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    
    // Simplified Threefry implementation
    // In practice, you'd want a full implementation
    uint k0 = uint(key[gid % 2]);
    uint k1 = uint(key[(gid + 1) % 2]);
    uint c0 = uint(counter[gid % 2]);
    uint c1 = uint(counter[(gid + 1) % 2]);
    
    // Mix operations (simplified)
    uint x0 = c0 + k0;
    uint x1 = c1 + k1;
    
    x0 += x1;
    x1 = (x1 << 13) | (x1 >> 19);
    x1 ^= x0;
    
    x0 += x1;
    x1 = (x1 << 17) | (x1 >> 15);
    x1 ^= x0;
    
    out[gid] = int(x0 ^ x1);
}

// Strided-to-contiguous assign operation (simplified)
kernel void assign_strided_float(device float* dst [[buffer(0)]],
                                device const float* src [[buffer(1)]],
                                constant uint* shape [[buffer(2)]],
                                constant int* src_strides [[buffer(3)]],
                                constant uint& ndim [[buffer(4)]],
                                constant uint& src_offset [[buffer(5)]],
                                uint gid [[thread_position_in_grid]]) {
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    if (gid >= total_size) return;
    
    // Convert linear index to multi-dimensional position
    uint temp = gid;
    uint src_idx = src_offset;
    
    // Calculate source index using strides
    for (int i = int(ndim) - 1; i >= 0; i--) {
        uint coord = temp % shape[i];
        temp /= shape[i];
        src_idx += coord * src_strides[i];
    }
    
    // Write to contiguous destination
    dst[gid] = src[src_idx];
}

kernel void assign_strided_int(device int* dst [[buffer(0)]],
                              device const int* src [[buffer(1)]],
                              constant uint* shape [[buffer(2)]],
                              constant int* src_strides [[buffer(3)]],
                              constant uint& ndim [[buffer(4)]],
                              constant uint& src_offset [[buffer(5)]],
                              uint gid [[thread_position_in_grid]]) {
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    if (gid >= total_size) return;
    
    // Convert linear index to multi-dimensional position
    uint temp = gid;
    uint src_idx = src_offset;
    
    // Calculate source index using strides
    for (int i = int(ndim) - 1; i >= 0; i--) {
        uint coord = temp % shape[i];
        temp /= shape[i];
        src_idx += coord * src_strides[i];
    }
    
    // Write to contiguous destination
    dst[gid] = src[src_idx];
}

kernel void assign_strided_uchar(device uchar* dst [[buffer(0)]],
                                device const uchar* src [[buffer(1)]],
                                constant uint* shape [[buffer(2)]],
                                constant int* src_strides [[buffer(3)]],
                                constant uint& ndim [[buffer(4)]],
                                constant uint& src_offset [[buffer(5)]],
                                uint gid [[thread_position_in_grid]]) {
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    if (gid >= total_size) return;
    
    // Convert linear index to multi-dimensional position
    uint temp = gid;
    uint src_idx = src_offset;
    
    // Calculate source index using strides
    for (int i = int(ndim) - 1; i >= 0; i--) {
        uint coord = temp % shape[i];
        temp /= shape[i];
        src_idx += coord * src_strides[i];
    }
    
    // Write to contiguous destination
    dst[gid] = src[src_idx];
}

// NumPy-style gather operation with multi-dimensional indices
kernel void gather_strided_float(device float* out [[buffer(0)]],
                              device const float* data [[buffer(1)]],
                              device const int* indices [[buffer(2)]],
                              constant uint* out_shape [[buffer(3)]],
                              constant uint* data_shape [[buffer(4)]],
                              constant int* data_strides [[buffer(5)]],
                              constant int* indices_strides [[buffer(6)]],
                              constant uint& ndim [[buffer(7)]],
                              constant uint& axis [[buffer(8)]],
                              constant uint& data_offset [[buffer(9)]],
                              constant uint& indices_offset [[buffer(10)]],
                              uint gid [[thread_position_in_grid]]) {
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= out_shape[i];
    }
    if (gid >= total_size) return;
    
    // Convert linear index to multi-dimensional position
    uint temp = gid;
    uint data_idx = data_offset;
    uint indices_idx = indices_offset;
    
    // Process dimensions from right to left
    for (int i = int(ndim) - 1; i >= 0; i--) {
        uint coord = temp % out_shape[i];
        temp /= out_shape[i];
        
        if (uint(i) == axis) {
            // For the gather axis, use the index from indices tensor
            indices_idx += coord * indices_strides[i];
        } else {
            // For other axes, use the coordinate directly
            data_idx += coord * data_strides[i];
        }
    }
    
    // Read the index value
    int idx = indices[indices_idx];
    
    // Handle negative indices
    int axis_size = int(data_shape[axis]);
    if (idx < 0) idx += axis_size;
    
    // Clamp to valid range
    idx = clamp(idx, 0, axis_size - 1);
    
    // Add the indexed position along the gather axis
    data_idx += uint(idx) * data_strides[axis];
    
    // Copy the value
    out[gid] = data[data_idx];
}

kernel void gather_strided_int(device int* out [[buffer(0)]],
                            device const int* data [[buffer(1)]],
                            device const int* indices [[buffer(2)]],
                            constant uint* out_shape [[buffer(3)]],
                            constant uint* data_shape [[buffer(4)]],
                            constant int* data_strides [[buffer(5)]],
                            constant int* indices_strides [[buffer(6)]],
                            constant uint& ndim [[buffer(7)]],
                            constant uint& axis [[buffer(8)]],
                            constant uint& data_offset [[buffer(9)]],
                            constant uint& indices_offset [[buffer(10)]],
                            uint gid [[thread_position_in_grid]]) {
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= out_shape[i];
    }
    if (gid >= total_size) return;
    
    // Convert linear index to multi-dimensional position
    uint temp = gid;
    uint data_idx = data_offset;
    uint indices_idx = indices_offset;
    
    // Process dimensions from right to left
    for (int i = int(ndim) - 1; i >= 0; i--) {
        uint coord = temp % out_shape[i];
        temp /= out_shape[i];
        
        if (uint(i) == axis) {
            // For the gather axis, use the index from indices tensor
            indices_idx += coord * indices_strides[i];
        } else {
            // For other axes, use the coordinate directly
            data_idx += coord * data_strides[i];
        }
    }
    
    // Read the index value
    int idx = indices[indices_idx];
    
    // Handle negative indices
    int axis_size = int(data_shape[axis]);
    if (idx < 0) idx += axis_size;
    
    // Clamp to valid range
    idx = clamp(idx, 0, axis_size - 1);
    
    // Add the indexed position along the gather axis
    data_idx += uint(idx) * data_strides[axis];
    
    // Copy the value
    out[gid] = data[data_idx];
}

kernel void gather_strided_uchar(device uchar* out [[buffer(0)]],
                              device const uchar* data [[buffer(1)]],
                              device const int* indices [[buffer(2)]],
                              constant uint* out_shape [[buffer(3)]],
                              constant uint* data_shape [[buffer(4)]],
                              constant int* data_strides [[buffer(5)]],
                              constant int* indices_strides [[buffer(6)]],
                              constant uint& ndim [[buffer(7)]],
                              constant uint& axis [[buffer(8)]],
                              constant uint& data_offset [[buffer(9)]],
                              constant uint& indices_offset [[buffer(10)]],
                              uint gid [[thread_position_in_grid]]) {
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= out_shape[i];
    }
    if (gid >= total_size) return;
    
    // Convert linear index to multi-dimensional position
    uint temp = gid;
    uint data_idx = data_offset;
    uint indices_idx = indices_offset;
    
    // Process dimensions from right to left
    for (int i = int(ndim) - 1; i >= 0; i--) {
        uint coord = temp % out_shape[i];
        temp /= out_shape[i];
        
        if (uint(i) == axis) {
            // For the gather axis, use the index from indices tensor
            indices_idx += coord * indices_strides[i];
        } else {
            // For other axes, use the coordinate directly
            data_idx += coord * data_strides[i];
        }
    }
    
    // Read the index value
    int idx = indices[indices_idx];
    
    // Handle negative indices
    int axis_size = int(data_shape[axis]);
    if (idx < 0) idx += axis_size;
    
    // Clamp to valid range
    idx = clamp(idx, 0, axis_size - 1);
    
    // Add the indexed position along the gather axis
    data_idx += uint(idx) * data_strides[axis];
    
    // Copy the value
    out[gid] = data[data_idx];
}
