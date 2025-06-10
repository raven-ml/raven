#include <metal_stdlib>
using namespace metal;

// Thread group size for reductions
constant uint REDUCE_THREADS = 256;

// Helper to compute strided index for reductions
inline uint compute_strided_index(uint reduction_id, uint element_in_reduction,
                                 constant uint* shape, constant uint* axes,
                                 constant int* strides, uint offset,
                                 uint ndim, uint naxes) {
    // Compute the multi-dimensional output index
    uint out_idx[8]; // Max 8 dimensions
    uint temp = reduction_id;
    
    // First, compute indices for non-reduced dimensions
    for (int i = ndim - 1; i >= 0; i--) {
        bool is_reduced = false;
        for (uint j = 0; j < naxes; j++) {
            if (axes[j] == (uint)i) {
                is_reduced = true;
                break;
            }
        }
        if (!is_reduced) {
            out_idx[i] = temp % shape[i];
            temp /= shape[i];
        } else {
            out_idx[i] = 0; // Will be set by element_in_reduction
        }
    }
    
    // Now set indices for reduced dimensions based on element_in_reduction
    temp = element_in_reduction;
    for (int i = naxes - 1; i >= 0; i--) {
        uint axis = axes[i];
        out_idx[axis] = temp % shape[axis];
        temp /= shape[axis];
    }
    
    // Convert multi-dimensional index to linear index using strides
    uint linear_idx = offset;
    for (uint i = 0; i < ndim; i++) {
        linear_idx += out_idx[i] * strides[i];
    }
    
    return uint(linear_idx);
}

// Sum reduction kernels
kernel void reduce_sum_float(device float* out [[buffer(0)]],
                            device const float* in [[buffer(1)]],
                            constant uint& in_size [[buffer(2)]],
                            constant uint& reduction_size [[buffer(3)]],
                            constant uint& num_reductions [[buffer(4)]],
                            constant uint* shape [[buffer(5)]],
                            constant uint* axes [[buffer(6)]],
                            constant uint& ndim [[buffer(7)]],
                            constant uint& naxes [[buffer(8)]],
                            constant int* in_strides [[buffer(9)]],
                            constant uint& in_offset [[buffer(10)]],
                            threadgroup float* shared [[threadgroup(0)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint gid [[thread_position_in_grid]],
                            uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    // Each thread loads elements with strided pattern
    float sum = 0.0f;
    
    // Grid-stride loop for this reduction
    for (uint i = tid; i < reduction_size; i += REDUCE_THREADS) {
        uint idx = compute_strided_index(reduction_id, i, shape, axes, in_strides, in_offset, ndim, naxes);
        if (idx < in_size) {
            sum += in[idx];
        }
    }
    
    // Store to shared memory
    shared[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction in shared memory
    for (uint s = REDUCE_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < REDUCE_THREADS) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (tid == 0) {
        out[reduction_id] = shared[0];
    }
}


kernel void reduce_sum_int(device int* out [[buffer(0)]],
                          device const int* in [[buffer(1)]],
                          constant uint& in_size [[buffer(2)]],
                          constant uint& reduction_size [[buffer(3)]],
                          constant uint& num_reductions [[buffer(4)]],
                          constant uint* shape [[buffer(5)]],
                          constant uint* axes [[buffer(6)]],
                          constant uint& ndim [[buffer(7)]],
                          constant uint& naxes [[buffer(8)]],
                          constant int* in_strides [[buffer(9)]],
                          constant uint& in_offset [[buffer(10)]],
                          threadgroup int* shared [[threadgroup(0)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint gid [[thread_position_in_grid]],
                          uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    int sum = 0;
    
    for (uint i = tid; i < reduction_size; i += REDUCE_THREADS) {
        uint idx = compute_strided_index(reduction_id, i, shape, axes, in_strides, in_offset, ndim, naxes);
        if (idx < in_size) {
            sum += in[idx];
        }
    }
    
    shared[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = REDUCE_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < REDUCE_THREADS) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        out[reduction_id] = shared[0];
    }
}

// Max reduction kernels
kernel void reduce_max_float(device float* out [[buffer(0)]],
                            device const float* in [[buffer(1)]],
                            constant uint& in_size [[buffer(2)]],
                            constant uint& reduction_size [[buffer(3)]],
                            constant uint& num_reductions [[buffer(4)]],
                            constant uint* shape [[buffer(5)]],
                            constant uint* axes [[buffer(6)]],
                            constant uint& ndim [[buffer(7)]],
                            constant uint& naxes [[buffer(8)]],
                            constant int* in_strides [[buffer(9)]],
                            constant uint& in_offset [[buffer(10)]],
                            threadgroup float* shared [[threadgroup(0)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint gid [[thread_position_in_grid]],
                            uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    float max_val = -INFINITY;
    
    for (uint i = tid; i < reduction_size; i += REDUCE_THREADS) {
        uint idx = compute_strided_index(reduction_id, i, shape, axes, in_strides, in_offset, ndim, naxes);
        if (idx < in_size) {
            max_val = fmax(max_val, in[idx]);
        }
    }
    
    shared[tid] = max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = REDUCE_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < REDUCE_THREADS) {
            shared[tid] = fmax(shared[tid], shared[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        out[reduction_id] = shared[0];
    }
}

kernel void reduce_max_int(device int* out [[buffer(0)]],
                          device const int* in [[buffer(1)]],
                          constant uint& in_size [[buffer(2)]],
                          constant uint& reduction_size [[buffer(3)]],
                          constant uint& num_reductions [[buffer(4)]],
                          constant uint* shape [[buffer(5)]],
                          constant uint* axes [[buffer(6)]],
                          constant uint& ndim [[buffer(7)]],
                          constant uint& naxes [[buffer(8)]],
                          constant int* in_strides [[buffer(9)]],
                          constant uint& in_offset [[buffer(10)]],
                          threadgroup int* shared [[threadgroup(0)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint gid [[thread_position_in_grid]],
                          uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    int max_val = INT_MIN;
    
    for (uint i = tid; i < reduction_size; i += REDUCE_THREADS) {
        uint idx = compute_strided_index(reduction_id, i, shape, axes, in_strides, in_offset, ndim, naxes);
        if (idx < in_size) {
            max_val = max(max_val, in[idx]);
        }
    }
    
    shared[tid] = max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = REDUCE_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < REDUCE_THREADS) {
            shared[tid] = max(shared[tid], shared[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        out[reduction_id] = shared[0];
    }
}

// Product reduction kernels
kernel void reduce_prod_float(device float* out [[buffer(0)]],
                             device const float* in [[buffer(1)]],
                             constant uint& in_size [[buffer(2)]],
                             constant uint& reduction_size [[buffer(3)]],
                             constant uint& num_reductions [[buffer(4)]],
                             constant uint* shape [[buffer(5)]],
                             constant uint* axes [[buffer(6)]],
                             constant uint& ndim [[buffer(7)]],
                             constant uint& naxes [[buffer(8)]],
                             constant int* in_strides [[buffer(9)]],
                             constant uint& in_offset [[buffer(10)]],
                             threadgroup float* shared [[threadgroup(0)]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint gid [[thread_position_in_grid]],
                             uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    float prod = 1.0f;
    
    for (uint i = tid; i < reduction_size; i += REDUCE_THREADS) {
        uint idx = compute_strided_index(reduction_id, i, shape, axes, in_strides, in_offset, ndim, naxes);
        if (idx < in_size) {
            prod *= in[idx];
        }
    }
    
    shared[tid] = prod;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = REDUCE_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < REDUCE_THREADS) {
            shared[tid] *= shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        out[reduction_id] = shared[0];
    }
}

kernel void reduce_prod_int(device int* out [[buffer(0)]],
                           device const int* in [[buffer(1)]],
                           constant uint& in_size [[buffer(2)]],
                           constant uint& reduction_size [[buffer(3)]],
                           constant uint& num_reductions [[buffer(4)]],
                           constant uint* shape [[buffer(5)]],
                           constant uint* axes [[buffer(6)]],
                           constant uint& ndim [[buffer(7)]],
                           constant uint& naxes [[buffer(8)]],
                           constant int* in_strides [[buffer(9)]],
                           constant uint& in_offset [[buffer(10)]],
                           threadgroup int* shared [[threadgroup(0)]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint gid [[thread_position_in_grid]],
                           uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    int prod = 1;
    
    for (uint i = tid; i < reduction_size; i += REDUCE_THREADS) {
        uint idx = compute_strided_index(reduction_id, i, shape, axes, in_strides, in_offset, ndim, naxes);
        if (idx < in_size) {
            prod *= in[idx];
        }
    }
    
    shared[tid] = prod;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = REDUCE_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < REDUCE_THREADS) {
            shared[tid] *= shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        out[reduction_id] = shared[0];
    }
}
