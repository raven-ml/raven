#include <metal_stdlib>
using namespace metal;

// Simple copy kernel
kernel void copy_float(device float* out [[buffer(0)]],
                      device const float* in [[buffer(1)]],
                      constant uint& size [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = in[gid];
}


kernel void copy_int(device int* out [[buffer(0)]],
                    device const int* in [[buffer(1)]],
                    constant uint& size [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = in[gid];
}

kernel void copy_long(device long* out [[buffer(0)]],
                     device const long* in [[buffer(1)]],
                     constant uint& size [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = in[gid];
}

kernel void copy_uchar(device uchar* out [[buffer(0)]],
                      device const uchar* in [[buffer(1)]],
                      constant uint& size [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = in[gid];
}

// Strided copy for contiguous operation
kernel void strided_copy_float(device float* out [[buffer(0)]],
                              device const float* in [[buffer(1)]],
                              constant uint* shape [[buffer(2)]],
                              constant int* strides [[buffer(3)]],
                              constant uint& ndim [[buffer(4)]],
                              constant uint& size [[buffer(5)]],
                              constant uint& offset [[buffer(6)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    
    // Compute position from linear index
    uint coords[8]; // Support up to 8 dimensions
    uint temp = gid;
    
    // Convert linear index to coordinates
    for (int i = ndim - 1; i >= 0; i--) {
        coords[i] = temp % shape[i];
        temp /= shape[i];
    }
    
    // Compute strided index with offset
    uint in_idx = offset;
    for (uint i = 0; i < ndim; i++) {
        in_idx += coords[i] * strides[i];
    }
    
    out[gid] = in[in_idx];
}


kernel void strided_copy_int(device int* out [[buffer(0)]],
                            device const int* in [[buffer(1)]],
                            constant uint* shape [[buffer(2)]],
                            constant int* strides [[buffer(3)]],
                            constant uint& ndim [[buffer(4)]],
                            constant uint& size [[buffer(5)]],
                            constant uint& offset [[buffer(6)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    
    // Compute position from linear index
    uint coords[8]; // Support up to 8 dimensions
    uint temp = gid;
    
    // Convert linear index to coordinates
    for (int i = ndim - 1; i >= 0; i--) {
        coords[i] = temp % shape[i];
        temp /= shape[i];
    }
    
    // Compute strided index with offset
    uint in_idx = offset;
    for (uint i = 0; i < ndim; i++) {
        in_idx += coords[i] * strides[i];
    }
    
    out[gid] = in[in_idx];
}

kernel void strided_copy_uchar(device uchar* out [[buffer(0)]],
                              device const uchar* in [[buffer(1)]],
                              constant uint* shape [[buffer(2)]],
                              constant int* strides [[buffer(3)]],
                              constant uint& ndim [[buffer(4)]],
                              constant uint& size [[buffer(5)]],
                              constant uint& offset [[buffer(6)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    
    // Compute position from linear index
    uint coords[8]; // Support up to 8 dimensions
    uint temp = gid;
    
    // Convert linear index to coordinates
    for (int i = ndim - 1; i >= 0; i--) {
        coords[i] = temp % shape[i];
        temp /= shape[i];
    }
    
    // Compute strided index with offset
    uint in_idx = offset;
    for (uint i = 0; i < ndim; i++) {
        in_idx += coords[i] * strides[i];
    }
    
    out[gid] = in[in_idx];
}

// Concatenation kernel - copies data with offset
kernel void concat_float(device float* out [[buffer(0)]],
                        device const float* in [[buffer(1)]],
                        constant uint& in_size [[buffer(2)]],
                        constant uint& out_offset [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= in_size) return;
    out[out_offset + gid] = in[gid];
}

// Concatenation kernel that handles axis properly with stride support
kernel void concat_axis_float(device float* out [[buffer(0)]],
                             device const float* in [[buffer(1)]],
                             constant uint* out_shape [[buffer(2)]],
                             constant uint* in_shape [[buffer(3)]],
                             constant int* in_strides [[buffer(4)]],
                             constant uint& axis [[buffer(5)]],
                             constant uint& axis_offset [[buffer(6)]],
                             constant uint& ndim [[buffer(7)]],
                             constant uint& in_offset [[buffer(8)]],
                             uint gid [[thread_position_in_grid]]) {
    uint in_size = 1;
    for (uint i = 0; i < ndim; i++) {
        in_size *= in_shape[i];
    }
    if (gid >= in_size) return;
    
    // Convert linear index to coordinates
    uint coords[8]; // Max 8 dimensions
    uint temp = gid;
    for (int i = ndim - 1; i >= 0; i--) {
        coords[i] = temp % in_shape[i];
        temp /= in_shape[i];
    }
    
    // Compute strided input index
    uint in_idx = in_offset;
    for (uint i = 0; i < ndim; i++) {
        in_idx += coords[i] * in_strides[i];
    }
    
    // Adjust coordinate for concatenation axis
    coords[axis] += axis_offset;
    
    // Convert back to linear index in output
    uint out_idx = 0;
    uint stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        out_idx += coords[i] * stride;
        stride *= out_shape[i];
    }
    
    out[out_idx] = in[in_idx];
}


kernel void concat_int(device int* out [[buffer(0)]],
                      device const int* in [[buffer(1)]],
                      constant uint& in_size [[buffer(2)]],
                      constant uint& out_offset [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= in_size) return;
    out[out_offset + gid] = in[gid];
}

kernel void concat_axis_int(device int* out [[buffer(0)]],
                           device const int* in [[buffer(1)]],
                           constant uint* out_shape [[buffer(2)]],
                           constant uint* in_shape [[buffer(3)]],
                           constant int* in_strides [[buffer(4)]],
                           constant uint& axis [[buffer(5)]],
                           constant uint& axis_offset [[buffer(6)]],
                           constant uint& ndim [[buffer(7)]],
                           constant uint& in_offset [[buffer(8)]],
                           uint gid [[thread_position_in_grid]]) {
    uint in_size = 1;
    for (uint i = 0; i < ndim; i++) {
        in_size *= in_shape[i];
    }
    if (gid >= in_size) return;
    
    // Convert linear index to coordinates
    uint coords[8]; // Max 8 dimensions
    uint temp = gid;
    for (int i = ndim - 1; i >= 0; i--) {
        coords[i] = temp % in_shape[i];
        temp /= in_shape[i];
    }
    
    // Compute strided input index
    uint in_idx = in_offset;
    for (uint i = 0; i < ndim; i++) {
        in_idx += coords[i] * in_strides[i];
    }
    
    // Adjust coordinate for concatenation axis
    coords[axis] += axis_offset;
    
    // Convert back to linear index in output
    uint out_idx = 0;
    uint stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        out_idx += coords[i] * stride;
        stride *= out_shape[i];
    }
    
    out[out_idx] = in[in_idx];
}

kernel void concat_uchar(device uchar* out [[buffer(0)]],
                        device const uchar* in [[buffer(1)]],
                        constant uint& in_size [[buffer(2)]],
                        constant uint& out_offset [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= in_size) return;
    out[out_offset + gid] = in[gid];
}

kernel void concat_axis_uchar(device uchar* out [[buffer(0)]],
                             device const uchar* in [[buffer(1)]],
                             constant uint* out_shape [[buffer(2)]],
                             constant uint* in_shape [[buffer(3)]],
                             constant int* in_strides [[buffer(4)]],
                             constant uint& axis [[buffer(5)]],
                             constant uint& axis_offset [[buffer(6)]],
                             constant uint& ndim [[buffer(7)]],
                             constant uint& in_offset [[buffer(8)]],
                             uint gid [[thread_position_in_grid]]) {
    uint in_size = 1;
    for (uint i = 0; i < ndim; i++) {
        in_size *= in_shape[i];
    }
    if (gid >= in_size) return;
    
    // Convert linear index to coordinates
    uint coords[8]; // Max 8 dimensions
    uint temp = gid;
    for (int i = ndim - 1; i >= 0; i--) {
        coords[i] = temp % in_shape[i];
        temp /= in_shape[i];
    }
    
    // Compute strided input index
    uint in_idx = in_offset;
    for (uint i = 0; i < ndim; i++) {
        in_idx += coords[i] * in_strides[i];
    }
    
    // Adjust coordinate for concatenation axis
    coords[axis] += axis_offset;
    
    // Convert back to linear index in output
    uint out_idx = 0;
    uint stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        out_idx += coords[i] * stride;
        stride *= out_shape[i];
    }
    
    out[out_idx] = in[in_idx];
}

// Padding kernel
kernel void pad_float(device float* out [[buffer(0)]],
                     device const float* in [[buffer(1)]],
                     constant uint* out_shape [[buffer(2)]],
                     constant uint* in_shape [[buffer(3)]],
                     constant uint* pad_before [[buffer(4)]],
                     constant float& pad_value [[buffer(5)]],
                     constant uint& ndim [[buffer(6)]],
                     constant int* in_strides [[buffer(7)]],
                     constant uint& in_offset [[buffer(8)]],
                     uint gid [[thread_position_in_grid]]) {
    // Compute total output size for all dimensions
    uint out_size = 1;
    for (uint i = 0; i < ndim; i++) {
        out_size *= out_shape[i];
    }
    
    if (gid >= out_size) return;
    
    // Compute output position for all dimensions
    uint out_pos[8]; // Max 8 dimensions
    uint temp = gid;
    for (int i = ndim - 1; i >= 0; i--) {
        out_pos[i] = temp % out_shape[i];
        temp /= out_shape[i];
    }
    
    // Check if we're in padding region
    bool in_bounds = true;
    uint in_pos[8];
    
    for (uint i = 0; i < ndim; i++) {
        if (out_pos[i] < pad_before[i] || out_pos[i] >= pad_before[i] + in_shape[i]) {
            in_bounds = false;
            break;
        }
        in_pos[i] = out_pos[i] - pad_before[i];
    }
    
    if (in_bounds) {
        // Compute input index using strides
        uint in_idx = in_offset;
        for (uint i = 0; i < ndim; i++) {
            in_idx += in_pos[i] * in_strides[i];
        }
        
        out[gid] = in[in_idx];
    } else {
        out[gid] = pad_value;
    }
}

