#include <metal_stdlib>
using namespace metal;

// Tile size for tiled matrix multiplication
#define TILE_SIZE 16

// Tiled matrix multiplication kernel for float - HIGH PERFORMANCE VERSION
kernel void matmul_tiled_float(device float* out [[buffer(0)]],
                              device const float* a [[buffer(1)]],
                              device const float* b [[buffer(2)]],
                              constant uint* out_shape [[buffer(3)]],
                              constant uint* a_shape [[buffer(4)]],
                              constant uint* b_shape [[buffer(5)]],
                              constant uint& ndim_out [[buffer(6)]],
                              constant uint& ndim_a [[buffer(7)]],
                              constant uint& ndim_b [[buffer(8)]],
                              uint3 gid [[thread_position_in_grid]],
                              uint3 tid [[thread_position_in_threadgroup]],
                              uint3 tgid [[threadgroup_position_in_grid]]) {
    // Shared memory for tiles
    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];
    
    // Get matrix dimensions
    uint m = a_shape[ndim_a - 2];
    uint k = a_shape[ndim_a - 1];
    uint n = b_shape[ndim_b - 1];
    
    // Get batch index and matrix position
    uint batch_idx = gid.z;
    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;
    
    // Local thread indices within tile
    uint ty = tid.y;
    uint tx = tid.x;
    
    // Calculate batch size
    uint batch_size = 1;
    for (uint i = 0; i < ndim_out - 2; i++) {
        batch_size *= out_shape[i];
    }
    
    if (batch_idx >= batch_size) return;
    
    // Calculate batch coordinates
    uint batch_coords[16]; // Max batch dimensions
    uint temp = batch_idx;
    for (int i = ndim_out - 3; i >= 0; i--) {
        batch_coords[i] = temp % out_shape[i];
        temp /= out_shape[i];
    }
    
    // Calculate offsets for A with broadcasting
    uint a_batch_offset = 0;
    uint a_batch_stride = 1;
    for (int i = ndim_a - 3; i >= 0; i--) {
        uint coord_idx = i + (ndim_out - 2) - (ndim_a - 2);
        uint coord = batch_coords[coord_idx];
        uint dim_size = a_shape[i];
        if (dim_size > 1) {
            a_batch_offset += coord * a_batch_stride;
        }
        a_batch_stride *= dim_size;
    }
    a_batch_offset *= (m * k);
    
    // Calculate offsets for B with broadcasting
    uint b_batch_offset = 0;
    uint b_batch_stride = 1;
    for (int i = ndim_b - 3; i >= 0; i--) {
        uint coord_idx = i + (ndim_out - 2) - (ndim_b - 2);
        uint coord = batch_coords[coord_idx];
        uint dim_size = b_shape[i];
        if (dim_size > 1) {
            b_batch_offset += coord * b_batch_stride;
        }
        b_batch_stride *= dim_size;
    }
    b_batch_offset *= (k * n);
    
    // Initialize accumulator
    float sum = 0.0f;
    
    // Loop over tiles
    uint numTiles = (k + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint t = 0; t < numTiles; t++) {
        // Load tile from A
        uint a_row = tgid.y * TILE_SIZE + ty;
        uint a_col = t * TILE_SIZE + tx;
        
        if (a_row < m && a_col < k) {
            tileA[ty][tx] = a[a_batch_offset + a_row * k + a_col];
        } else {
            tileA[ty][tx] = 0.0f;
        }
        
        // Load tile from B
        uint b_row = t * TILE_SIZE + ty;
        uint b_col = tgid.x * TILE_SIZE + tx;
        
        if (b_row < k && b_col < n) {
            tileB[ty][tx] = b[b_batch_offset + b_row * n + b_col];
        } else {
            tileB[ty][tx] = 0.0f;
        }
        
        // Synchronize threads to ensure tiles are loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product for this tile
        for (uint i = 0; i < TILE_SIZE; i++) {
            sum += tileA[ty][i] * tileB[i][tx];
        }
        
        // Synchronize before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result if within bounds
    if (row < m && col < n) {
        uint out_idx = batch_idx * (m * n) + row * n + col;
        out[out_idx] = sum;
    }
}

// Matrix multiplication kernel for float - NAIVE VERSION (for small matrices or debugging)
kernel void matmul_float(device float* out [[buffer(0)]],
                        device const float* a [[buffer(1)]],
                        device const float* b [[buffer(2)]],
                        constant uint* out_shape [[buffer(3)]],
                        constant uint* a_shape [[buffer(4)]],
                        constant uint* b_shape [[buffer(5)]],
                        constant uint& ndim_out [[buffer(6)]],
                        constant uint& ndim_a [[buffer(7)]],
                        constant uint& ndim_b [[buffer(8)]],
                        uint3 gid [[thread_position_in_grid]]) {
    // Get matrix dimensions
    uint m = a_shape[ndim_a - 2];
    uint k = a_shape[ndim_a - 1];
    uint n = b_shape[ndim_b - 1];
    
    // Get output position
    uint batch_idx = gid.z;
    uint row = gid.y;
    uint col = gid.x;
    
    // Calculate batch size
    uint batch_size = 1;
    for (uint i = 0; i < ndim_out - 2; i++) {
        batch_size *= out_shape[i];
    }
    
    if (batch_idx >= batch_size || row >= m || col >= n) return;
    
    // Calculate batch coordinates
    uint batch_coords[16]; // Max batch dimensions
    uint temp = batch_idx;
    for (int i = ndim_out - 3; i >= 0; i--) {
        batch_coords[i] = temp % out_shape[i];
        temp /= out_shape[i];
    }
    
    // Calculate offsets for A with broadcasting
    uint a_batch_offset = 0;
    uint a_batch_stride = 1;
    for (int i = ndim_a - 3; i >= 0; i--) {
        uint coord_idx = i + (ndim_out - 2) - (ndim_a - 2);
        uint coord = batch_coords[coord_idx];
        uint dim_size = a_shape[i];
        if (dim_size > 1) {
            a_batch_offset += coord * a_batch_stride;
        }
        a_batch_stride *= dim_size;
    }
    a_batch_offset *= (m * k);
    
    // Calculate offsets for B with broadcasting
    uint b_batch_offset = 0;
    uint b_batch_stride = 1;
    for (int i = ndim_b - 3; i >= 0; i--) {
        uint coord_idx = i + (ndim_out - 2) - (ndim_b - 2);
        uint coord = batch_coords[coord_idx];
        uint dim_size = b_shape[i];
        if (dim_size > 1) {
            b_batch_offset += coord * b_batch_stride;
        }
        b_batch_stride *= dim_size;
    }
    b_batch_offset *= (k * n);
    
    // Perform matrix multiplication
    float sum = 0.0;
    for (uint i = 0; i < k; i++) {
        uint a_idx = a_batch_offset + row * k + i;
        uint b_idx = b_batch_offset + i * n + col;
        sum += a[a_idx] * b[b_idx];
    }
    
    // Write result
    uint out_idx = batch_idx * (m * n) + row * n + col;
    out[out_idx] = sum;
}

// Matrix multiplication kernel for double
// NOTE: Commented out because Metal doesn't support double on most hardware
/*
kernel void matmul_double(device double* out [[buffer(0)]],
                         device const double* a [[buffer(1)]],
                         device const double* b [[buffer(2)]],
                         constant uint* out_shape [[buffer(3)]],
                         constant uint* a_shape [[buffer(4)]],
                         constant uint* b_shape [[buffer(5)]],
                         constant uint& ndim_out [[buffer(6)]],
                         constant uint& ndim_a [[buffer(7)]],
                         constant uint& ndim_b [[buffer(8)]],
                         uint3 gid [[thread_position_in_grid]]) {
    uint m = a_shape[ndim_a - 2];
    uint k = a_shape[ndim_a - 1];
    uint n = b_shape[ndim_b - 1];
    
    uint batch_idx = gid.z;
    uint row = gid.y;
    uint col = gid.x;
    
    uint batch_size = 1;
    for (uint i = 0; i < ndim_out - 2; i++) {
        batch_size *= out_shape[i];
    }
    
    if (batch_idx >= batch_size || row >= m || col >= n) return;
    
    uint batch_coords[16];
    uint temp = batch_idx;
    for (int i = ndim_out - 3; i >= 0; i--) {
        batch_coords[i] = temp % out_shape[i];
        temp /= out_shape[i];
    }
    
    uint a_batch_offset = 0;
    uint a_batch_stride = 1;
    for (int i = ndim_a - 3; i >= 0; i--) {
        uint coord_idx = i + (ndim_out - 2) - (ndim_a - 2);
        uint coord = batch_coords[coord_idx];
        uint dim_size = a_shape[i];
        if (dim_size > 1) {
            a_batch_offset += coord * a_batch_stride;
        }
        a_batch_stride *= dim_size;
    }
    a_batch_offset *= (m * k);
    
    uint b_batch_offset = 0;
    uint b_batch_stride = 1;
    for (int i = ndim_b - 3; i >= 0; i--) {
        uint coord_idx = i + (ndim_out - 2) - (ndim_b - 2);
        uint coord = batch_coords[coord_idx];
        uint dim_size = b_shape[i];
        if (dim_size > 1) {
            b_batch_offset += coord * b_batch_stride;
        }
        b_batch_stride *= dim_size;
    }
    b_batch_offset *= (k * n);
    
    double sum = 0.0;
    for (uint i = 0; i < k; i++) {
        uint a_idx = a_batch_offset + row * k + i;
        uint b_idx = b_batch_offset + i * n + col;
        sum += a[a_idx] * b[b_idx];
    }
    
    uint out_idx = batch_idx * (m * n) + row * n + col;
    out[out_idx] = sum;
}
*/

// Matrix multiplication kernel for int
kernel void matmul_int(device int* out [[buffer(0)]],
                      device const int* a [[buffer(1)]],
                      device const int* b [[buffer(2)]],
                      constant uint* out_shape [[buffer(3)]],
                      constant uint* a_shape [[buffer(4)]],
                      constant uint* b_shape [[buffer(5)]],
                      constant uint& ndim_out [[buffer(6)]],
                      constant uint& ndim_a [[buffer(7)]],
                      constant uint& ndim_b [[buffer(8)]],
                      uint3 gid [[thread_position_in_grid]]) {
    uint m = a_shape[ndim_a - 2];
    uint k = a_shape[ndim_a - 1];
    uint n = b_shape[ndim_b - 1];
    
    uint batch_idx = gid.z;
    uint row = gid.y;
    uint col = gid.x;
    
    uint batch_size = 1;
    for (uint i = 0; i < ndim_out - 2; i++) {
        batch_size *= out_shape[i];
    }
    
    if (batch_idx >= batch_size || row >= m || col >= n) return;
    
    uint batch_coords[16];
    uint temp = batch_idx;
    for (int i = ndim_out - 3; i >= 0; i--) {
        batch_coords[i] = temp % out_shape[i];
        temp /= out_shape[i];
    }
    
    uint a_batch_offset = 0;
    uint a_batch_stride = 1;
    for (int i = ndim_a - 3; i >= 0; i--) {
        uint coord_idx = i + (ndim_out - 2) - (ndim_a - 2);
        uint coord = batch_coords[coord_idx];
        uint dim_size = a_shape[i];
        if (dim_size > 1) {
            a_batch_offset += coord * a_batch_stride;
        }
        a_batch_stride *= dim_size;
    }
    a_batch_offset *= (m * k);
    
    uint b_batch_offset = 0;
    uint b_batch_stride = 1;
    for (int i = ndim_b - 3; i >= 0; i--) {
        uint coord_idx = i + (ndim_out - 2) - (ndim_b - 2);
        uint coord = batch_coords[coord_idx];
        uint dim_size = b_shape[i];
        if (dim_size > 1) {
            b_batch_offset += coord * b_batch_stride;
        }
        b_batch_stride *= dim_size;
    }
    b_batch_offset *= (k * n);
    
    int sum = 0;
    for (uint i = 0; i < k; i++) {
        uint a_idx = a_batch_offset + row * k + i;
        uint b_idx = b_batch_offset + i * n + col;
        sum += a[a_idx] * b[b_idx];
    }
    
    uint out_idx = batch_idx * (m * n) + row * n + col;
    out[out_idx] = sum;
}

// Matrix multiplication kernel for long
kernel void matmul_long(device long* out [[buffer(0)]],
                       device const long* a [[buffer(1)]],
                       device const long* b [[buffer(2)]],
                       constant uint* out_shape [[buffer(3)]],
                       constant uint* a_shape [[buffer(4)]],
                       constant uint* b_shape [[buffer(5)]],
                       constant uint& ndim_out [[buffer(6)]],
                       constant uint& ndim_a [[buffer(7)]],
                       constant uint& ndim_b [[buffer(8)]],
                       uint3 gid [[thread_position_in_grid]]) {
    uint m = a_shape[ndim_a - 2];
    uint k = a_shape[ndim_a - 1];
    uint n = b_shape[ndim_b - 1];
    
    uint batch_idx = gid.z;
    uint row = gid.y;
    uint col = gid.x;
    
    uint batch_size = 1;
    for (uint i = 0; i < ndim_out - 2; i++) {
        batch_size *= out_shape[i];
    }
    
    if (batch_idx >= batch_size || row >= m || col >= n) return;
    
    uint batch_coords[16];
    uint temp = batch_idx;
    for (int i = ndim_out - 3; i >= 0; i--) {
        batch_coords[i] = temp % out_shape[i];
        temp /= out_shape[i];
    }
    
    uint a_batch_offset = 0;
    uint a_batch_stride = 1;
    for (int i = ndim_a - 3; i >= 0; i--) {
        uint coord_idx = i + (ndim_out - 2) - (ndim_a - 2);
        uint coord = batch_coords[coord_idx];
        uint dim_size = a_shape[i];
        if (dim_size > 1) {
            a_batch_offset += coord * a_batch_stride;
        }
        a_batch_stride *= dim_size;
    }
    a_batch_offset *= (m * k);
    
    uint b_batch_offset = 0;
    uint b_batch_stride = 1;
    for (int i = ndim_b - 3; i >= 0; i--) {
        uint coord_idx = i + (ndim_out - 2) - (ndim_b - 2);
        uint coord = batch_coords[coord_idx];
        uint dim_size = b_shape[i];
        if (dim_size > 1) {
            b_batch_offset += coord * b_batch_stride;
        }
        b_batch_stride *= dim_size;
    }
    b_batch_offset *= (k * n);
    
    long sum = 0;
    for (uint i = 0; i < k; i++) {
        uint a_idx = a_batch_offset + row * k + i;
        uint b_idx = b_batch_offset + i * n + col;
        sum += a[a_idx] * b[b_idx];
    }
    
    uint out_idx = batch_idx * (m * n) + row * n + col;
    out[out_idx] = sum;
}
// Tiled matrix multiplication kernel for int - HIGH PERFORMANCE VERSION
kernel void matmul_tiled_int(device int* out [[buffer(0)]],
                            device const int* a [[buffer(1)]],
                            device const int* b [[buffer(2)]],
                            constant uint* out_shape [[buffer(3)]],
                            constant uint* a_shape [[buffer(4)]],
                            constant uint* b_shape [[buffer(5)]],
                            constant uint& ndim_out [[buffer(6)]],
                            constant uint& ndim_a [[buffer(7)]],
                            constant uint& ndim_b [[buffer(8)]],
                            uint3 gid [[thread_position_in_grid]],
                            uint3 tid [[thread_position_in_threadgroup]],
                            uint3 tgid [[threadgroup_position_in_grid]]) {
    // Shared memory for tiles
    threadgroup int tileA[TILE_SIZE][TILE_SIZE];
    threadgroup int tileB[TILE_SIZE][TILE_SIZE];
    
    // Get matrix dimensions
    uint m = a_shape[ndim_a - 2];
    uint k = a_shape[ndim_a - 1];
    uint n = b_shape[ndim_b - 1];
    
    // Get batch index and matrix position
    uint batch_idx = gid.z;
    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;
    
    // Local thread indices within tile
    uint ty = tid.y;
    uint tx = tid.x;
    
    // Calculate batch size
    uint batch_size = 1;
    for (uint i = 0; i < ndim_out - 2; i++) {
        batch_size *= out_shape[i];
    }
    
    if (batch_idx >= batch_size) return;
    
    // Calculate batch coordinates
    uint batch_coords[16];
    uint temp = batch_idx;
    for (int i = ndim_out - 3; i >= 0; i--) {
        batch_coords[i] = temp % out_shape[i];
        temp /= out_shape[i];
    }
    
    // Calculate offsets for A with broadcasting
    uint a_batch_offset = 0;
    uint a_batch_stride = 1;
    for (int i = ndim_a - 3; i >= 0; i--) {
        uint coord_idx = i + (ndim_out - 2) - (ndim_a - 2);
        uint coord = batch_coords[coord_idx];
        uint dim_size = a_shape[i];
        if (dim_size > 1) {
            a_batch_offset += coord * a_batch_stride;
        }
        a_batch_stride *= dim_size;
    }
    a_batch_offset *= (m * k);
    
    // Calculate offsets for B with broadcasting
    uint b_batch_offset = 0;
    uint b_batch_stride = 1;
    for (int i = ndim_b - 3; i >= 0; i--) {
        uint coord_idx = i + (ndim_out - 2) - (ndim_b - 2);
        uint coord = batch_coords[coord_idx];
        uint dim_size = b_shape[i];
        if (dim_size > 1) {
            b_batch_offset += coord * b_batch_stride;
        }
        b_batch_stride *= dim_size;
    }
    b_batch_offset *= (k * n);
    
    // Initialize accumulator
    int sum = 0;
    
    // Loop over tiles
    uint numTiles = (k + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint t = 0; t < numTiles; t++) {
        // Load tile from A
        uint a_row = tgid.y * TILE_SIZE + ty;
        uint a_col = t * TILE_SIZE + tx;
        
        if (a_row < m && a_col < k) {
            tileA[ty][tx] = a[a_batch_offset + a_row * k + a_col];
        } else {
            tileA[ty][tx] = 0;
        }
        
        // Load tile from B
        uint b_row = t * TILE_SIZE + ty;
        uint b_col = tgid.x * TILE_SIZE + tx;
        
        if (b_row < k && b_col < n) {
            tileB[ty][tx] = b[b_batch_offset + b_row * n + b_col];
        } else {
            tileB[ty][tx] = 0;
        }
        
        // Synchronize threads to ensure tiles are loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product for this tile
        for (uint i = 0; i < TILE_SIZE; i++) {
            sum += tileA[ty][i] * tileB[i][tx];
        }
        
        // Synchronize before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result if within bounds
    if (row < m && col < n) {
        uint out_idx = batch_idx * (m * n) + row * n + col;
        out[out_idx] = sum;
    }
}

// Tiled matrix multiplication kernel for long - HIGH PERFORMANCE VERSION
kernel void matmul_tiled_long(device long* out [[buffer(0)]],
                             device const long* a [[buffer(1)]],
                             device const long* b [[buffer(2)]],
                             constant uint* out_shape [[buffer(3)]],
                             constant uint* a_shape [[buffer(4)]],
                             constant uint* b_shape [[buffer(5)]],
                             constant uint& ndim_out [[buffer(6)]],
                             constant uint& ndim_a [[buffer(7)]],
                             constant uint& ndim_b [[buffer(8)]],
                             uint3 gid [[thread_position_in_grid]],
                             uint3 tid [[thread_position_in_threadgroup]],
                             uint3 tgid [[threadgroup_position_in_grid]]) {
    // Shared memory for tiles
    threadgroup long tileA[TILE_SIZE][TILE_SIZE];
    threadgroup long tileB[TILE_SIZE][TILE_SIZE];
    
    // Get matrix dimensions
    uint m = a_shape[ndim_a - 2];
    uint k = a_shape[ndim_a - 1];
    uint n = b_shape[ndim_b - 1];
    
    // Get batch index and matrix position
    uint batch_idx = gid.z;
    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;
    
    // Local thread indices within tile
    uint ty = tid.y;
    uint tx = tid.x;
    
    // Calculate batch size
    uint batch_size = 1;
    for (uint i = 0; i < ndim_out - 2; i++) {
        batch_size *= out_shape[i];
    }
    
    if (batch_idx >= batch_size) return;
    
    // Calculate batch coordinates
    uint batch_coords[16];
    uint temp = batch_idx;
    for (int i = ndim_out - 3; i >= 0; i--) {
        batch_coords[i] = temp % out_shape[i];
        temp /= out_shape[i];
    }
    
    // Calculate offsets for A with broadcasting
    uint a_batch_offset = 0;
    uint a_batch_stride = 1;
    for (int i = ndim_a - 3; i >= 0; i--) {
        uint coord_idx = i + (ndim_out - 2) - (ndim_a - 2);
        uint coord = batch_coords[coord_idx];
        uint dim_size = a_shape[i];
        if (dim_size > 1) {
            a_batch_offset += coord * a_batch_stride;
        }
        a_batch_stride *= dim_size;
    }
    a_batch_offset *= (m * k);
    
    // Calculate offsets for B with broadcasting
    uint b_batch_offset = 0;
    uint b_batch_stride = 1;
    for (int i = ndim_b - 3; i >= 0; i--) {
        uint coord_idx = i + (ndim_out - 2) - (ndim_b - 2);
        uint coord = batch_coords[coord_idx];
        uint dim_size = b_shape[i];
        if (dim_size > 1) {
            b_batch_offset += coord * b_batch_stride;
        }
        b_batch_stride *= dim_size;
    }
    b_batch_offset *= (k * n);
    
    // Initialize accumulator
    long sum = 0;
    
    // Loop over tiles
    uint numTiles = (k + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint t = 0; t < numTiles; t++) {
        // Load tile from A
        uint a_row = tgid.y * TILE_SIZE + ty;
        uint a_col = t * TILE_SIZE + tx;
        
        if (a_row < m && a_col < k) {
            tileA[ty][tx] = a[a_batch_offset + a_row * k + a_col];
        } else {
            tileA[ty][tx] = 0;
        }
        
        // Load tile from B
        uint b_row = t * TILE_SIZE + ty;
        uint b_col = tgid.x * TILE_SIZE + tx;
        
        if (b_row < k && b_col < n) {
            tileB[ty][tx] = b[b_batch_offset + b_row * n + b_col];
        } else {
            tileB[ty][tx] = 0;
        }
        
        // Synchronize threads to ensure tiles are loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product for this tile
        for (uint i = 0; i < TILE_SIZE; i++) {
            sum += tileA[ty][i] * tileB[i][tx];
        }
        
        // Synchronize before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result if within bounds
    if (row < m && col < n) {
        uint out_idx = batch_idx * (m * n) + row * n + col;
        out[out_idx] = sum;
    }
}
