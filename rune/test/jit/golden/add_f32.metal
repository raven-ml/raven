#include <metal_stdlib>
using namespace metal;

kernel void kernel_0(
  device float* v3 [[buffer(0)]],
  device float* v4 [[buffer(1)]],
  device float* v9 [[buffer(2)]],
  uint3 gtid [[thread_position_in_grid]],
  uint3 lid  [[thread_position_in_threadgroup]],
  uint3 gid  [[threadgroup_position_in_grid]]
) {
  uint v5 = gtid.x;
  float v6 = v3[v5];
  float v7 = v4[v5];
  float v8 = v6 + v7;
  v9[v5] = v8;
}
