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

// Macro for sum reduction
#define DEFINE_REDUCE_SUM(type, zero_val) \
kernel void reduce_sum_##type(device type* out [[buffer(0)]], \
                            device const type* in [[buffer(1)]], \
                            constant uint& in_size [[buffer(2)]], \
                            constant uint& reduction_size [[buffer(3)]], \
                            constant uint& num_reductions [[buffer(4)]], \
                            constant uint* shape [[buffer(5)]], \
                            constant uint* axes [[buffer(6)]], \
                            constant uint& ndim [[buffer(7)]], \
                            constant uint& naxes [[buffer(8)]], \
                            constant int* in_strides [[buffer(9)]], \
                            constant uint& in_offset [[buffer(10)]], \
                            threadgroup type* shared [[threadgroup(0)]], \
                            uint tid [[thread_index_in_threadgroup]], \
                            uint gid [[thread_position_in_grid]], \
                            uint group_id [[threadgroup_position_in_grid]]) { \
    \
    uint reduction_id = group_id; \
    if (reduction_id >= num_reductions) return; \
    \
    type sum = zero_val; \
    \
    for (uint i = tid; i < reduction_size; i += REDUCE_THREADS) { \
        uint idx = compute_strided_index(reduction_id, i, shape, axes, in_strides, in_offset, ndim, naxes); \
        if (idx < in_size) { \
            sum += in[idx]; \
        } \
    } \
    \
    shared[tid] = sum; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    for (uint s = REDUCE_THREADS / 2; s > 0; s >>= 1) { \
        if (tid < s && tid + s < REDUCE_THREADS) { \
            shared[tid] += shared[tid + s]; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    \
    if (tid == 0) { \
        out[reduction_id] = shared[0]; \
    } \
}

// Helper for NaN-aware max - returns true if a is greater or if b is NaN
template<typename T>
inline T nan_max(T a, T b) {
    // If either is NaN, return NaN
    if (isnan(a)) return a;
    if (isnan(b)) return b;
    return max(a, b);
}

// Specialization for integer types (no NaN)
inline int nan_max(int a, int b) { return max(a, b); }
inline long nan_max(long a, long b) { return max(a, b); }
inline char nan_max(char a, char b) { return max(a, b); }
inline uchar nan_max(uchar a, uchar b) { return max(a, b); }
inline short nan_max(short a, short b) { return max(a, b); }
inline ushort nan_max(ushort a, ushort b) { return max(a, b); }

// Macro for max reduction
#define DEFINE_REDUCE_MAX(type, min_val) \
kernel void reduce_max_##type(device type* out [[buffer(0)]], \
                            device const type* in [[buffer(1)]], \
                            constant uint& in_size [[buffer(2)]], \
                            constant uint& reduction_size [[buffer(3)]], \
                            constant uint& num_reductions [[buffer(4)]], \
                            constant uint* shape [[buffer(5)]], \
                            constant uint* axes [[buffer(6)]], \
                            constant uint& ndim [[buffer(7)]], \
                            constant uint& naxes [[buffer(8)]], \
                            constant int* in_strides [[buffer(9)]], \
                            constant uint& in_offset [[buffer(10)]], \
                            threadgroup type* shared [[threadgroup(0)]], \
                            uint tid [[thread_index_in_threadgroup]], \
                            uint gid [[thread_position_in_grid]], \
                            uint group_id [[threadgroup_position_in_grid]]) { \
    \
    uint reduction_id = group_id; \
    if (reduction_id >= num_reductions) return; \
    \
    type max_val = min_val; \
    \
    for (uint i = tid; i < reduction_size; i += REDUCE_THREADS) { \
        uint idx = compute_strided_index(reduction_id, i, shape, axes, in_strides, in_offset, ndim, naxes); \
        if (idx < in_size) { \
            max_val = nan_max(max_val, in[idx]); \
        } \
    } \
    \
    shared[tid] = max_val; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    for (uint s = REDUCE_THREADS / 2; s > 0; s >>= 1) { \
        if (tid < s && tid + s < REDUCE_THREADS) { \
            shared[tid] = nan_max(shared[tid], shared[tid + s]); \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    \
    if (tid == 0) { \
        out[reduction_id] = shared[0]; \
    } \
}

// Macro for product reduction
#define DEFINE_REDUCE_PROD(type, one_val) \
kernel void reduce_prod_##type(device type* out [[buffer(0)]], \
                             device const type* in [[buffer(1)]], \
                             constant uint& in_size [[buffer(2)]], \
                             constant uint& reduction_size [[buffer(3)]], \
                             constant uint& num_reductions [[buffer(4)]], \
                             constant uint* shape [[buffer(5)]], \
                             constant uint* axes [[buffer(6)]], \
                             constant uint& ndim [[buffer(7)]], \
                             constant uint& naxes [[buffer(8)]], \
                             constant int* in_strides [[buffer(9)]], \
                             constant uint& in_offset [[buffer(10)]], \
                             threadgroup type* shared [[threadgroup(0)]], \
                             uint tid [[thread_index_in_threadgroup]], \
                             uint gid [[thread_position_in_grid]], \
                             uint group_id [[threadgroup_position_in_grid]]) { \
    \
    uint reduction_id = group_id; \
    if (reduction_id >= num_reductions) return; \
    \
    type prod = one_val; \
    \
    for (uint i = tid; i < reduction_size; i += REDUCE_THREADS) { \
        uint idx = compute_strided_index(reduction_id, i, shape, axes, in_strides, in_offset, ndim, naxes); \
        if (idx < in_size) { \
            prod *= in[idx]; \
        } \
    } \
    \
    shared[tid] = prod; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    for (uint s = REDUCE_THREADS / 2; s > 0; s >>= 1) { \
        if (tid < s && tid + s < REDUCE_THREADS) { \
            shared[tid] *= shared[tid + s]; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    \
    if (tid == 0) { \
        out[reduction_id] = shared[0]; \
    } \
}

// Instantiate for all types
DEFINE_REDUCE_SUM(float, 0.0f)
DEFINE_REDUCE_SUM(half, half(0.0f))
DEFINE_REDUCE_SUM(int, 0)
DEFINE_REDUCE_SUM(long, 0L)
DEFINE_REDUCE_SUM(char, char(0))
DEFINE_REDUCE_SUM(uchar, uchar(0))
DEFINE_REDUCE_SUM(short, short(0))
DEFINE_REDUCE_SUM(ushort, ushort(0))

DEFINE_REDUCE_MAX(float, -INFINITY)
DEFINE_REDUCE_MAX(half, half(-65504.0f))  // Minimum half value
DEFINE_REDUCE_MAX(int, INT_MIN)
DEFINE_REDUCE_MAX(long, LONG_MIN)
DEFINE_REDUCE_MAX(char, CHAR_MIN)
DEFINE_REDUCE_MAX(uchar, 0)
DEFINE_REDUCE_MAX(short, SHRT_MIN)
DEFINE_REDUCE_MAX(ushort, 0)

DEFINE_REDUCE_PROD(float, 1.0f)
DEFINE_REDUCE_PROD(half, half(1.0f))
DEFINE_REDUCE_PROD(int, 1)
DEFINE_REDUCE_PROD(long, 1L)
DEFINE_REDUCE_PROD(char, char(1))
DEFINE_REDUCE_PROD(uchar, uchar(1))
DEFINE_REDUCE_PROD(short, short(1))
DEFINE_REDUCE_PROD(ushort, ushort(1))