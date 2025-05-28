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
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    
    // Compute position from linear index
    uint temp = gid;
    uint3 pos = uint3(0, 0, 0);
    
    if (ndim > 2) {
        pos.z = temp % shape[2];
        temp /= shape[2];
    }
    if (ndim > 1) {
        pos.y = temp % shape[1];
        temp /= shape[1];
    }
    pos.x = temp;
    
    // Compute strided index
    uint in_idx = 0;
    if (ndim > 0) in_idx += pos.x * strides[0];
    if (ndim > 1) in_idx += pos.y * strides[1];
    if (ndim > 2) in_idx += pos.z * strides[2];
    
    out[gid] = in[in_idx];
}


kernel void strided_copy_int(device int* out [[buffer(0)]],
                            device const int* in [[buffer(1)]],
                            constant uint* shape [[buffer(2)]],
                            constant int* strides [[buffer(3)]],
                            constant uint& ndim [[buffer(4)]],
                            constant uint& size [[buffer(5)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    
    uint temp = gid;
    uint3 pos = uint3(0, 0, 0);
    
    if (ndim > 2) {
        pos.z = temp % shape[2];
        temp /= shape[2];
    }
    if (ndim > 1) {
        pos.y = temp % shape[1];
        temp /= shape[1];
    }
    pos.x = temp;
    
    uint in_idx = 0;
    if (ndim > 0) in_idx += pos.x * strides[0];
    if (ndim > 1) in_idx += pos.y * strides[1];
    if (ndim > 2) in_idx += pos.z * strides[2];
    
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


kernel void concat_int(device int* out [[buffer(0)]],
                      device const int* in [[buffer(1)]],
                      constant uint& in_size [[buffer(2)]],
                      constant uint& out_offset [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= in_size) return;
    out[out_offset + gid] = in[gid];
}

// Padding kernel
kernel void pad_float(device float* out [[buffer(0)]],
                     device const float* in [[buffer(1)]],
                     constant uint* out_shape [[buffer(2)]],
                     constant uint* in_shape [[buffer(3)]],
                     constant uint* pad_before [[buffer(4)]],
                     constant float& pad_value [[buffer(5)]],
                     constant uint& ndim [[buffer(6)]],
                     uint gid [[thread_position_in_grid]]) {
    uint out_size = out_shape[0];
    if (ndim > 1) out_size *= out_shape[1];
    if (ndim > 2) out_size *= out_shape[2];
    
    if (gid >= out_size) return;
    
    // Compute output position
    uint temp = gid;
    uint3 out_pos = uint3(0, 0, 0);
    
    if (ndim > 2) {
        out_pos.z = temp % out_shape[2];
        temp /= out_shape[2];
    }
    if (ndim > 1) {
        out_pos.y = temp % out_shape[1];
        temp /= out_shape[1];
    }
    out_pos.x = temp;
    
    // Check if we're in padding region
    bool in_bounds = true;
    uint3 in_pos = out_pos;
    
    if (ndim > 0) {
        if (out_pos.x < pad_before[0] || out_pos.x >= pad_before[0] + in_shape[0]) {
            in_bounds = false;
        } else {
            in_pos.x = out_pos.x - pad_before[0];
        }
    }
    
    if (ndim > 1 && in_bounds) {
        if (out_pos.y < pad_before[1] || out_pos.y >= pad_before[1] + in_shape[1]) {
            in_bounds = false;
        } else {
            in_pos.y = out_pos.y - pad_before[1];
        }
    }
    
    if (ndim > 2 && in_bounds) {
        if (out_pos.z < pad_before[2] || out_pos.z >= pad_before[2] + in_shape[2]) {
            in_bounds = false;
        } else {
            in_pos.z = out_pos.z - pad_before[2];
        }
    }
    
    if (in_bounds) {
        // Compute input index
        uint in_idx = in_pos.x;
        if (ndim > 1) in_idx = in_idx * in_shape[1] + in_pos.y;
        if (ndim > 2) in_idx = in_idx * in_shape[2] + in_pos.z;
        
        out[gid] = in[in_idx];
    } else {
        out[gid] = pad_value;
    }
}
