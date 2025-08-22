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

kernel void reduce_sum_long(device long* out [[buffer(0)]],
                           device const long* in [[buffer(1)]],
                           constant uint& in_size [[buffer(2)]],
                           constant uint& reduction_size [[buffer(3)]],
                           constant uint& num_reductions [[buffer(4)]],
                           constant uint* shape [[buffer(5)]],
                           constant uint* axes [[buffer(6)]],
                           constant uint& ndim [[buffer(7)]],
                           constant uint& naxes [[buffer(8)]],
                           constant int* in_strides [[buffer(9)]],
                           constant uint& in_offset [[buffer(10)]],
                           threadgroup long* shared [[threadgroup(0)]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint gid [[thread_position_in_grid]],
                           uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    long sum = 0;
    
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
// Add reduce_sum for more types
kernel void reduce_sum_half(device half* out [[buffer(0)]],
                           device const half* in [[buffer(1)]],
                           constant uint& in_size [[buffer(2)]],
                           constant uint& reduction_size [[buffer(3)]],
                           constant uint& num_reductions [[buffer(4)]],
                           constant uint* shape [[buffer(5)]],
                           constant uint* axes [[buffer(6)]],
                           constant uint& ndim [[buffer(7)]],
                           constant uint& naxes [[buffer(8)]],
                           constant int* in_strides [[buffer(9)]],
                           constant uint& in_offset [[buffer(10)]],
                           threadgroup half* shared [[threadgroup(0)]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint gid [[thread_position_in_grid]],
                           uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    half sum = 0.0h;
    
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

kernel void reduce_sum_char(device char* out [[buffer(0)]],
                           device const char* in [[buffer(1)]],
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
        out[reduction_id] = char(shared[0]);
    }
}

kernel void reduce_sum_short(device short* out [[buffer(0)]],
                            device const short* in [[buffer(1)]],
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
        out[reduction_id] = short(shared[0]);
    }
}

kernel void reduce_sum_ushort(device ushort* out [[buffer(0)]],
                             device const ushort* in [[buffer(1)]],
                             constant uint& in_size [[buffer(2)]],
                             constant uint& reduction_size [[buffer(3)]],
                             constant uint& num_reductions [[buffer(4)]],
                             constant uint* shape [[buffer(5)]],
                             constant uint* axes [[buffer(6)]],
                             constant uint& ndim [[buffer(7)]],
                             constant uint& naxes [[buffer(8)]],
                             constant int* in_strides [[buffer(9)]],
                             constant uint& in_offset [[buffer(10)]],
                             threadgroup uint* shared [[threadgroup(0)]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint gid [[thread_position_in_grid]],
                             uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    uint sum = 0;
    
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
        out[reduction_id] = ushort(shared[0]);
    }
}


kernel void reduce_sum_uchar(device uchar* out [[buffer(0)]],
                            device const uchar* in [[buffer(1)]],
                            constant uint& in_size [[buffer(2)]],
                            constant uint& reduction_size [[buffer(3)]],
                            constant uint& num_reductions [[buffer(4)]],
                            constant uint* shape [[buffer(5)]],
                            constant uint* axes [[buffer(6)]],
                            constant uint& ndim [[buffer(7)]],
                            constant uint& naxes [[buffer(8)]],
                            constant int* in_strides [[buffer(9)]],
                            constant uint& in_offset [[buffer(10)]],
                            threadgroup uchar* shared [[threadgroup(0)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint gid [[thread_position_in_grid]],
                            uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    uchar sum = 0;
    
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
            float val = in[idx];
            // Propagate NaN - if either is NaN, result is NaN
            if (isnan(val) || isnan(max_val)) {
                max_val = NAN;
            } else {
                max_val = fmax(max_val, val);
            }
        }
    }
    
    shared[tid] = max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = REDUCE_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < REDUCE_THREADS) {
            float a = shared[tid];
            float b = shared[tid + s];
            // Propagate NaN - if either is NaN, result is NaN
            if (isnan(a) || isnan(b)) {
                shared[tid] = NAN;
            } else {
                shared[tid] = fmax(a, b);
            }
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

kernel void reduce_max_long(device long* out [[buffer(0)]],
                           device const long* in [[buffer(1)]],
                           constant uint& in_size [[buffer(2)]],
                           constant uint& reduction_size [[buffer(3)]],
                           constant uint& num_reductions [[buffer(4)]],
                           constant uint* shape [[buffer(5)]],
                           constant uint* axes [[buffer(6)]],
                           constant uint& ndim [[buffer(7)]],
                           constant uint& naxes [[buffer(8)]],
                           constant int* in_strides [[buffer(9)]],
                           constant uint& in_offset [[buffer(10)]],
                           threadgroup long* shared [[threadgroup(0)]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint gid [[thread_position_in_grid]],
                           uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    long max_val = LONG_MIN;
    
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

kernel void reduce_max_uchar(device uchar* out [[buffer(0)]],
                            device const uchar* in [[buffer(1)]],
                            constant uint& in_size [[buffer(2)]],
                            constant uint& reduction_size [[buffer(3)]],
                            constant uint& num_reductions [[buffer(4)]],
                            constant uint* shape [[buffer(5)]],
                            constant uint* axes [[buffer(6)]],
                            constant uint& ndim [[buffer(7)]],
                            constant uint& naxes [[buffer(8)]],
                            constant int* in_strides [[buffer(9)]],
                            constant uint& in_offset [[buffer(10)]],
                            threadgroup uchar* shared [[threadgroup(0)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint gid [[thread_position_in_grid]],
                            uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    uchar max_val = 0;
    
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

kernel void reduce_prod_long(device long* out [[buffer(0)]],
                            device const long* in [[buffer(1)]],
                            constant uint& in_size [[buffer(2)]],
                            constant uint& reduction_size [[buffer(3)]],
                            constant uint& num_reductions [[buffer(4)]],
                            constant uint* shape [[buffer(5)]],
                            constant uint* axes [[buffer(6)]],
                            constant uint& ndim [[buffer(7)]],
                            constant uint& naxes [[buffer(8)]],
                            constant int* in_strides [[buffer(9)]],
                            constant uint& in_offset [[buffer(10)]],
                            threadgroup long* shared [[threadgroup(0)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint gid [[thread_position_in_grid]],
                            uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    long prod = 1;
    
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

kernel void reduce_prod_uchar(device uchar* out [[buffer(0)]],
                             device const uchar* in [[buffer(1)]],
                             constant uint& in_size [[buffer(2)]],
                             constant uint& reduction_size [[buffer(3)]],
                             constant uint& num_reductions [[buffer(4)]],
                             constant uint* shape [[buffer(5)]],
                             constant uint* axes [[buffer(6)]],
                             constant uint& ndim [[buffer(7)]],
                             constant uint& naxes [[buffer(8)]],
                             constant int* in_strides [[buffer(9)]],
                             constant uint& in_offset [[buffer(10)]],
                             threadgroup uchar* shared [[threadgroup(0)]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint gid [[thread_position_in_grid]],
                             uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    uchar prod = 1;
    
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

// Reduce min operations
kernel void reduce_min_float(device float* out [[buffer(0)]],
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
    
    float min_val = INFINITY;
    
    for (uint i = tid; i < reduction_size; i += REDUCE_THREADS) {
        uint idx = compute_strided_index(reduction_id, i, shape, axes, in_strides, in_offset, ndim, naxes);
        if (idx < in_size) {
            float val = in[idx];
            // Propagate NaN - if either is NaN, result is NaN
            if (isnan(val) || isnan(min_val)) {
                min_val = NAN;
            } else {
                min_val = fmin(min_val, val);
            }
        }
    }
    
    shared[tid] = min_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = REDUCE_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < REDUCE_THREADS) {
            float a = shared[tid];
            float b = shared[tid + s];
            // Propagate NaN - if either is NaN, result is NaN
            if (isnan(a) || isnan(b)) {
                shared[tid] = NAN;
            } else {
                shared[tid] = fmin(a, b);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        out[reduction_id] = shared[0];
    }
}

kernel void reduce_min_int(device int* out [[buffer(0)]],
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
    
    int min_val = INT_MAX;
    
    for (uint i = tid; i < reduction_size; i += REDUCE_THREADS) {
        uint idx = compute_strided_index(reduction_id, i, shape, axes, in_strides, in_offset, ndim, naxes);
        if (idx < in_size) {
            min_val = min(min_val, in[idx]);
        }
    }
    
    shared[tid] = min_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = REDUCE_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < REDUCE_THREADS) {
            shared[tid] = min(shared[tid], shared[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        out[reduction_id] = shared[0];
    }
}

kernel void reduce_min_long(device long* out [[buffer(0)]],
                           device const long* in [[buffer(1)]],
                           constant uint& in_size [[buffer(2)]],
                           constant uint& reduction_size [[buffer(3)]],
                           constant uint& num_reductions [[buffer(4)]],
                           constant uint* shape [[buffer(5)]],
                           constant uint* axes [[buffer(6)]],
                           constant uint& ndim [[buffer(7)]],
                           constant uint& naxes [[buffer(8)]],
                           constant int* in_strides [[buffer(9)]],
                           constant uint& in_offset [[buffer(10)]],
                           threadgroup long* shared [[threadgroup(0)]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint gid [[thread_position_in_grid]],
                           uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    long min_val = LONG_MAX;
    
    for (uint i = tid; i < reduction_size; i += REDUCE_THREADS) {
        uint idx = compute_strided_index(reduction_id, i, shape, axes, in_strides, in_offset, ndim, naxes);
        if (idx < in_size) {
            min_val = min(min_val, in[idx]);
        }
    }
    
    shared[tid] = min_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = REDUCE_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < REDUCE_THREADS) {
            shared[tid] = min(shared[tid], shared[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        out[reduction_id] = shared[0];
    }
}

kernel void reduce_min_uchar(device uchar* out [[buffer(0)]],
                            device const uchar* in [[buffer(1)]],
                            constant uint& in_size [[buffer(2)]],
                            constant uint& reduction_size [[buffer(3)]],
                            constant uint& num_reductions [[buffer(4)]],
                            constant uint* shape [[buffer(5)]],
                            constant uint* axes [[buffer(6)]],
                            constant uint& ndim [[buffer(7)]],
                            constant uint& naxes [[buffer(8)]],
                            constant int* in_strides [[buffer(9)]],
                            constant uint& in_offset [[buffer(10)]],
                            threadgroup uchar* shared [[threadgroup(0)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint gid [[thread_position_in_grid]],
                            uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    uchar min_val = UCHAR_MAX;
    
    for (uint i = tid; i < reduction_size; i += REDUCE_THREADS) {
        uint idx = compute_strided_index(reduction_id, i, shape, axes, in_strides, in_offset, ndim, naxes);
        if (idx < in_size) {
            min_val = min(min_val, in[idx]);
        }
    }
    
    shared[tid] = min_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = REDUCE_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < REDUCE_THREADS) {
            shared[tid] = min(shared[tid], shared[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        out[reduction_id] = shared[0];
    }
}
// Add reduce_max for more types
kernel void reduce_max_half(device half* out [[buffer(0)]],
                           device const half* in [[buffer(1)]],
                           constant uint& in_size [[buffer(2)]],
                           constant uint& reduction_size [[buffer(3)]],
                           constant uint& num_reductions [[buffer(4)]],
                           constant uint* shape [[buffer(5)]],
                           constant uint* axes [[buffer(6)]],
                           constant uint& ndim [[buffer(7)]],
                           constant uint& naxes [[buffer(8)]],
                           constant int* in_strides [[buffer(9)]],
                           constant uint& in_offset [[buffer(10)]],
                           threadgroup half* shared [[threadgroup(0)]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint gid [[thread_position_in_grid]],
                           uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    half max_val = -INFINITY;
    
    for (uint i = tid; i < reduction_size; i += REDUCE_THREADS) {
        uint idx = compute_strided_index(reduction_id, i, shape, axes, in_strides, in_offset, ndim, naxes);
        if (idx < in_size) {
            half val = in[idx];
            // Propagate NaN - if either is NaN, result is NaN
            if (isnan(val) || isnan(max_val)) {
                max_val = NAN;
            } else {
                max_val = fmax(max_val, val);
            }
        }
    }
    
    shared[tid] = max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = REDUCE_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < REDUCE_THREADS) {
            half a = shared[tid];
            half b = shared[tid + s];
            if (isnan(a) || isnan(b)) {
                shared[tid] = NAN;
            } else {
                shared[tid] = fmax(a, b);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        out[reduction_id] = shared[0];
    }
}

kernel void reduce_max_char(device char* out [[buffer(0)]],
                           device const char* in [[buffer(1)]],
                           constant uint& in_size [[buffer(2)]],
                           constant uint& reduction_size [[buffer(3)]],
                           constant uint& num_reductions [[buffer(4)]],
                           constant uint* shape [[buffer(5)]],
                           constant uint* axes [[buffer(6)]],
                           constant uint& ndim [[buffer(7)]],
                           constant uint& naxes [[buffer(8)]],
                           constant int* in_strides [[buffer(9)]],
                           constant uint& in_offset [[buffer(10)]],
                           threadgroup char* shared [[threadgroup(0)]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint gid [[thread_position_in_grid]],
                           uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    char max_val = -128;
    
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

kernel void reduce_max_short(device short* out [[buffer(0)]],
                            device const short* in [[buffer(1)]],
                            constant uint& in_size [[buffer(2)]],
                            constant uint& reduction_size [[buffer(3)]],
                            constant uint& num_reductions [[buffer(4)]],
                            constant uint* shape [[buffer(5)]],
                            constant uint* axes [[buffer(6)]],
                            constant uint& ndim [[buffer(7)]],
                            constant uint& naxes [[buffer(8)]],
                            constant int* in_strides [[buffer(9)]],
                            constant uint& in_offset [[buffer(10)]],
                            threadgroup short* shared [[threadgroup(0)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint gid [[thread_position_in_grid]],
                            uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    short max_val = -32768;
    
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

kernel void reduce_max_ushort(device ushort* out [[buffer(0)]],
                             device const ushort* in [[buffer(1)]],
                             constant uint& in_size [[buffer(2)]],
                             constant uint& reduction_size [[buffer(3)]],
                             constant uint& num_reductions [[buffer(4)]],
                             constant uint* shape [[buffer(5)]],
                             constant uint* axes [[buffer(6)]],
                             constant uint& ndim [[buffer(7)]],
                             constant uint& naxes [[buffer(8)]],
                             constant int* in_strides [[buffer(9)]],
                             constant uint& in_offset [[buffer(10)]],
                             threadgroup ushort* shared [[threadgroup(0)]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint gid [[thread_position_in_grid]],
                             uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    ushort max_val = 0;
    
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

// Add reduce_prod for more types
kernel void reduce_prod_half(device half* out [[buffer(0)]],
                            device const half* in [[buffer(1)]],
                            constant uint& in_size [[buffer(2)]],
                            constant uint& reduction_size [[buffer(3)]],
                            constant uint& num_reductions [[buffer(4)]],
                            constant uint* shape [[buffer(5)]],
                            constant uint* axes [[buffer(6)]],
                            constant uint& ndim [[buffer(7)]],
                            constant uint& naxes [[buffer(8)]],
                            constant int* in_strides [[buffer(9)]],
                            constant uint& in_offset [[buffer(10)]],
                            threadgroup half* shared [[threadgroup(0)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint gid [[thread_position_in_grid]],
                            uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    half prod = 1.0h;
    
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

kernel void reduce_prod_char(device char* out [[buffer(0)]],
                            device const char* in [[buffer(1)]],
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
        out[reduction_id] = char(shared[0]);
    }
}

kernel void reduce_prod_short(device short* out [[buffer(0)]],
                             device const short* in [[buffer(1)]],
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
        out[reduction_id] = short(shared[0]);
    }
}

kernel void reduce_prod_ushort(device ushort* out [[buffer(0)]],
                              device const ushort* in [[buffer(1)]],
                              constant uint& in_size [[buffer(2)]],
                              constant uint& reduction_size [[buffer(3)]],
                              constant uint& num_reductions [[buffer(4)]],
                              constant uint* shape [[buffer(5)]],
                              constant uint* axes [[buffer(6)]],
                              constant uint& ndim [[buffer(7)]],
                              constant uint& naxes [[buffer(8)]],
                              constant int* in_strides [[buffer(9)]],
                              constant uint& in_offset [[buffer(10)]],
                              threadgroup uint* shared [[threadgroup(0)]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint gid [[thread_position_in_grid]],
                              uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    uint prod = 1;
    
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
        out[reduction_id] = ushort(shared[0]);
    }
}