#include <metal_stdlib>
using namespace metal;

// Thread group size for cumsum operations
constant uint CUMSUM_THREADS = 256;

// Helper to compute linear index from multi-dimensional coordinates
inline uint compute_linear_index(constant uint* shape, constant int* strides, 
                                uint offset, uint ndim, thread uint* coords) {
    uint linear_idx = offset;
    for (uint i = 0; i < ndim; i++) {
        linear_idx += coords[i] * strides[i];
    }
    return linear_idx;
}

// Helper to compute coordinates from linear index along non-cumsum dimensions
inline void compute_coords_excluding_axis(uint linear_idx, constant uint* shape, 
                                        uint ndim, uint axis, thread uint* coords) {
    for (int i = ndim - 1; i >= 0; i--) {
        if ((uint)i != axis) {
            coords[i] = linear_idx % shape[i];
            linear_idx /= shape[i];
        }
    }
}

// Cumsum kernel for float32
kernel void cumsum_float(device float* out [[buffer(0)]],
                        device const float* in [[buffer(1)]],
                        constant uint& in_size [[buffer(2)]],
                        constant uint& axis [[buffer(3)]],
                        constant uint* shape [[buffer(4)]],
                        constant uint& ndim [[buffer(5)]],
                        constant int* in_strides [[buffer(6)]],
                        constant uint& in_offset [[buffer(7)]],
                        constant int* out_strides [[buffer(8)]],
                        constant uint& out_offset [[buffer(9)]],
                        uint gid [[thread_position_in_grid]]) {
    
    if (gid >= in_size) return;
    
    // Calculate total elements excluding the cumsum axis
    uint total_slices = 1;
    for (uint i = 0; i < ndim; i++) {
        if (i != axis) {
            total_slices *= shape[i];
        }
    }
    
    uint slice_id = gid;
    if (slice_id >= total_slices) return;
    
    // Compute coordinates for this slice (excluding axis dimension)
    uint coords[8]; // Max 8 dimensions
    compute_coords_excluding_axis(slice_id, shape, ndim, axis, coords);
    
    // Perform cumsum along the specified axis
    float cumulative_sum = 0.0f;
    for (uint i = 0; i < shape[axis]; i++) {
        coords[axis] = i;
        
        // Compute input and output indices
        uint in_idx = compute_linear_index(shape, in_strides, in_offset, ndim, coords);
        uint out_idx = compute_linear_index(shape, out_strides, out_offset, ndim, coords);
        
        // Add current element to cumulative sum
        if (in_idx < in_size) {
            cumulative_sum += in[in_idx];
            out[out_idx] = cumulative_sum;
        }
    }
}

// Cumsum kernel for int32
kernel void cumsum_int(device int* out [[buffer(0)]],
                      device const int* in [[buffer(1)]],
                      constant uint& in_size [[buffer(2)]],
                      constant uint& axis [[buffer(3)]],
                      constant uint* shape [[buffer(4)]],
                      constant uint& ndim [[buffer(5)]],
                      constant int* in_strides [[buffer(6)]],
                      constant uint& in_offset [[buffer(7)]],
                      constant int* out_strides [[buffer(8)]],
                      constant uint& out_offset [[buffer(9)]],
                      uint gid [[thread_position_in_grid]]) {
    
    if (gid >= in_size) return;
    
    uint total_slices = 1;
    for (uint i = 0; i < ndim; i++) {
        if (i != axis) {
            total_slices *= shape[i];
        }
    }
    
    uint slice_id = gid;
    if (slice_id >= total_slices) return;
    
    uint coords[8];
    compute_coords_excluding_axis(slice_id, shape, ndim, axis, coords);
    
    int cumulative_sum = 0;
    for (uint i = 0; i < shape[axis]; i++) {
        coords[axis] = i;
        
        uint in_idx = compute_linear_index(shape, in_strides, in_offset, ndim, coords);
        uint out_idx = compute_linear_index(shape, out_strides, out_offset, ndim, coords);
        
        if (in_idx < in_size) {
            cumulative_sum += in[in_idx];
            out[out_idx] = cumulative_sum;
        }
    }
}

// Cumsum kernel for int64 (long)
kernel void cumsum_long(device long* out [[buffer(0)]],
                       device const long* in [[buffer(1)]],
                       constant uint& in_size [[buffer(2)]],
                       constant uint& axis [[buffer(3)]],
                       constant uint* shape [[buffer(4)]],
                       constant uint& ndim [[buffer(5)]],
                       constant int* in_strides [[buffer(6)]],
                       constant uint& in_offset [[buffer(7)]],
                       constant int* out_strides [[buffer(8)]],
                       constant uint& out_offset [[buffer(9)]],
                       uint gid [[thread_position_in_grid]]) {
    
    if (gid >= in_size) return;
    
    uint total_slices = 1;
    for (uint i = 0; i < ndim; i++) {
        if (i != axis) {
            total_slices *= shape[i];
        }
    }
    
    uint slice_id = gid;
    if (slice_id >= total_slices) return;
    
    uint coords[8];
    compute_coords_excluding_axis(slice_id, shape, ndim, axis, coords);
    
    long cumulative_sum = 0;
    for (uint i = 0; i < shape[axis]; i++) {
        coords[axis] = i;
        
        uint in_idx = compute_linear_index(shape, in_strides, in_offset, ndim, coords);
        uint out_idx = compute_linear_index(shape, out_strides, out_offset, ndim, coords);
        
        if (in_idx < in_size) {
            cumulative_sum += in[in_idx];
            out[out_idx] = cumulative_sum;
        }
    }
}

// Cumsum kernel for uint8 (uchar)
kernel void cumsum_uchar(device uchar* out [[buffer(0)]],
                        device const uchar* in [[buffer(1)]],
                        constant uint& in_size [[buffer(2)]],
                        constant uint& axis [[buffer(3)]],
                        constant uint* shape [[buffer(4)]],
                        constant uint& ndim [[buffer(5)]],
                        constant int* in_strides [[buffer(6)]],
                        constant uint& in_offset [[buffer(7)]],
                        constant int* out_strides [[buffer(8)]],
                        constant uint& out_offset [[buffer(9)]],
                        uint gid [[thread_position_in_grid]]) {
    
    if (gid >= in_size) return;
    
    uint total_slices = 1;
    for (uint i = 0; i < ndim; i++) {
        if (i != axis) {
            total_slices *= shape[i];
        }
    }
    
    uint slice_id = gid;
    if (slice_id >= total_slices) return;
    
    uint coords[8];
    compute_coords_excluding_axis(slice_id, shape, ndim, axis, coords);
    
    uchar cumulative_sum = 0;
    for (uint i = 0; i < shape[axis]; i++) {
        coords[axis] = i;
        
        uint in_idx = compute_linear_index(shape, in_strides, in_offset, ndim, coords);
        uint out_idx = compute_linear_index(shape, out_strides, out_offset, ndim, coords);
        
        if (in_idx < in_size) {
            cumulative_sum += in[in_idx];
            out[out_idx] = cumulative_sum;
        }
    }
}


// Cumsum kernel for int16 (short)
kernel void cumsum_short(device short* out [[buffer(0)]],
                        device const short* in [[buffer(1)]],
                        constant uint& in_size [[buffer(2)]],
                        constant uint& axis [[buffer(3)]],
                        constant uint* shape [[buffer(4)]],
                        constant uint& ndim [[buffer(5)]],
                        constant int* in_strides [[buffer(6)]],
                        constant uint& in_offset [[buffer(7)]],
                        constant int* out_strides [[buffer(8)]],
                        constant uint& out_offset [[buffer(9)]],
                        uint gid [[thread_position_in_grid]]) {
    
    if (gid >= in_size) return;
    
    uint total_slices = 1;
    for (uint i = 0; i < ndim; i++) {
        if (i != axis) {
            total_slices *= shape[i];
        }
    }
    
    uint slice_id = gid;
    if (slice_id >= total_slices) return;
    
    uint coords[8];
    compute_coords_excluding_axis(slice_id, shape, ndim, axis, coords);
    
    short cumulative_sum = 0;
    for (uint i = 0; i < shape[axis]; i++) {
        coords[axis] = i;
        
        uint in_idx = compute_linear_index(shape, in_strides, in_offset, ndim, coords);
        uint out_idx = compute_linear_index(shape, out_strides, out_offset, ndim, coords);
        
        if (in_idx < in_size) {
            cumulative_sum += in[in_idx];
            out[out_idx] = cumulative_sum;
        }
    }
}

// Cumsum kernel for uint16 (ushort)
kernel void cumsum_ushort(device ushort* out [[buffer(0)]],
                         device const ushort* in [[buffer(1)]],
                         constant uint& in_size [[buffer(2)]],
                         constant uint& axis [[buffer(3)]],
                         constant uint* shape [[buffer(4)]],
                         constant uint& ndim [[buffer(5)]],
                         constant int* in_strides [[buffer(6)]],
                         constant uint& in_offset [[buffer(7)]],
                         constant int* out_strides [[buffer(8)]],
                         constant uint& out_offset [[buffer(9)]],
                         uint gid [[thread_position_in_grid]]) {
    
    if (gid >= in_size) return;
    
    uint total_slices = 1;
    for (uint i = 0; i < ndim; i++) {
        if (i != axis) {
            total_slices *= shape[i];
        }
    }
    
    uint slice_id = gid;
    if (slice_id >= total_slices) return;
    
    uint coords[8];
    compute_coords_excluding_axis(slice_id, shape, ndim, axis, coords);
    
    ushort cumulative_sum = 0;
    for (uint i = 0; i < shape[axis]; i++) {
        coords[axis] = i;
        
        uint in_idx = compute_linear_index(shape, in_strides, in_offset, ndim, coords);
        uint out_idx = compute_linear_index(shape, out_strides, out_offset, ndim, coords);
        
        if (in_idx < in_size) {
            cumulative_sum += in[in_idx];
            out[out_idx] = cumulative_sum;
        }
    }
}

// Cumsum kernel for int8 (char)
kernel void cumsum_char(device char* out [[buffer(0)]],
                       device const char* in [[buffer(1)]],
                       constant uint& in_size [[buffer(2)]],
                       constant uint& axis [[buffer(3)]],
                       constant uint* shape [[buffer(4)]],
                       constant uint& ndim [[buffer(5)]],
                       constant int* in_strides [[buffer(6)]],
                       constant uint& in_offset [[buffer(7)]],
                       constant int* out_strides [[buffer(8)]],
                       constant uint& out_offset [[buffer(9)]],
                       uint gid [[thread_position_in_grid]]) {
    
    if (gid >= in_size) return;
    
    uint total_slices = 1;
    for (uint i = 0; i < ndim; i++) {
        if (i != axis) {
            total_slices *= shape[i];
        }
    }
    
    uint slice_id = gid;
    if (slice_id >= total_slices) return;
    
    uint coords[8];
    compute_coords_excluding_axis(slice_id, shape, ndim, axis, coords);
    
    char cumulative_sum = 0;
    for (uint i = 0; i < shape[axis]; i++) {
        coords[axis] = i;
        
        uint in_idx = compute_linear_index(shape, in_strides, in_offset, ndim, coords);
        uint out_idx = compute_linear_index(shape, out_strides, out_offset, ndim, coords);
        
        if (in_idx < in_size) {
            cumulative_sum += in[in_idx];
            out[out_idx] = cumulative_sum;
        }
    }
}