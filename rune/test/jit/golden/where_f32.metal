#include <metal_stdlib>
using namespace metal;

kernel void kernel_0(
  device bool* v4 [[buffer(0)]],
  device float* v5 [[buffer(1)]],
  device float* v6 [[buffer(2)]],
  device float* v12 [[buffer(3)]],
  uint3 gtid [[thread_position_in_grid]],
  uint3 lid  [[thread_position_in_threadgroup]],
  uint3 gid  [[threadgroup_position_in_grid]]
) {
  uint v7 = gtid.x;
  bool v8 = v4[v7];
  float v9 = v5[v7];
  float v10 = v6[v7];
  float v11 = v8 ? v9 : v10;
  v12[v7] = v11;
}
