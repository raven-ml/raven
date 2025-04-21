#include <metal_stdlib>
using namespace metal;

// Sum reduction kernel for float32
kernel void sum_reduction_float32(device const float* input [[buffer(0)]],
                                  device float* output [[buffer(1)]],
                                  constant uint& n [[buffer(2)]],
                                  constant uint& num_thread_groups
                                  [[buffer(3)]],
                                  uint tid [[thread_position_in_threadgroup]],
                                  uint tgid [[threadgroup_position_in_grid]],
                                  uint tgs [[threads_per_threadgroup]]) {
  threadgroup float local_sum[256];  // Fixed size matching thread group size
  float sum = 0.0;
  uint total_threads = tgs * num_thread_groups;
  uint index = tid + tgid * tgs;
  while (index < n) {
    sum += input[index];
    index += total_threads;
  }
  local_sum[tid] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tgs / 2; s > 0; s >>= 1) {
    if (tid < s) {
      local_sum[tid] += local_sum[tid + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    output[tgid] = local_sum[0];
  }
}

// Sum reduction kernel for int32
kernel void sum_reduction_int32(device const int* input [[buffer(0)]],
                                device int* output [[buffer(1)]],
                                constant uint& n [[buffer(2)]],
                                constant uint& num_thread_groups [[buffer(3)]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint tgid [[threadgroup_position_in_grid]],
                                uint tgs [[threads_per_threadgroup]]) {
  threadgroup int local_sum[256];  // Fixed size matching thread group size
  int sum = 0;
  uint total_threads = tgs * num_thread_groups;
  uint index = tid + tgid * tgs;
  while (index < n) {
    sum += input[index];
    index += total_threads;
  }
  local_sum[tid] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tgs / 2; s > 0; s >>= 1) {
    if (tid < s) {
      local_sum[tid] += local_sum[tid + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    output[tgid] = local_sum[0];
  }
}
