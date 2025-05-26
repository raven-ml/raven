#include <metal_stdlib>
using namespace metal;

// Thread group size for reductions
constant uint REDUCE_THREADS = 256;

// Sum reduction kernels
kernel void reduce_sum_float(device float* out [[buffer(0)]],
                            device const float* in [[buffer(1)]],
                            constant uint& in_size [[buffer(2)]],
                            constant uint& reduction_size [[buffer(3)]],
                            constant uint& num_reductions [[buffer(4)]],
                            threadgroup float* shared [[threadgroup(0)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint gid [[thread_position_in_grid]],
                            uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    // Each thread loads one element
    float sum = 0.0f;
    uint base_idx = reduction_id * reduction_size;
    
    // Grid-stride loop for this reduction
    for (uint i = tid; i < reduction_size; i += REDUCE_THREADS) {
        uint idx = base_idx + i;
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
                          threadgroup int* shared [[threadgroup(0)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint gid [[thread_position_in_grid]],
                          uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    int sum = 0;
    uint base_idx = reduction_id * reduction_size;
    
    for (uint i = tid; i < reduction_size; i += REDUCE_THREADS) {
        uint idx = base_idx + i;
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
                            threadgroup float* shared [[threadgroup(0)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint gid [[thread_position_in_grid]],
                            uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    float max_val = -INFINITY;
    uint base_idx = reduction_id * reduction_size;
    
    for (uint i = tid; i < reduction_size; i += REDUCE_THREADS) {
        uint idx = base_idx + i;
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
                          threadgroup int* shared [[threadgroup(0)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint gid [[thread_position_in_grid]],
                          uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    int max_val = INT_MIN;
    uint base_idx = reduction_id * reduction_size;
    
    for (uint i = tid; i < reduction_size; i += REDUCE_THREADS) {
        uint idx = base_idx + i;
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
                             threadgroup float* shared [[threadgroup(0)]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint gid [[thread_position_in_grid]],
                             uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    float prod = 1.0f;
    uint base_idx = reduction_id * reduction_size;
    
    for (uint i = tid; i < reduction_size; i += REDUCE_THREADS) {
        uint idx = base_idx + i;
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
                           threadgroup int* shared [[threadgroup(0)]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint gid [[thread_position_in_grid]],
                           uint group_id [[threadgroup_position_in_grid]]) {
    
    uint reduction_id = group_id;
    if (reduction_id >= num_reductions) return;
    
    int prod = 1;
    uint base_idx = reduction_id * reduction_size;
    
    for (uint i = tid; i < reduction_size; i += REDUCE_THREADS) {
        uint idx = base_idx + i;
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
