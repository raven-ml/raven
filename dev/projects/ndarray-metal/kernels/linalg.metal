#include <metal_stdlib>
using namespace metal;

// --- General Stride-Based Kernels (Keep These) ---

// Matrix multiplication kernel for float32 (Handles Strides for A and B)
kernel void matmul_float32(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],  // Assumed Contiguous Output
    constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant long* a_strides [[buffer(6)]],        // Stride for A (elems)
    constant long* b_strides [[buffer(7)]],        // Stride for B (elems)
    uint2 gid [[threadgroup_position_in_grid]],    // Grid ID (tile index)
    uint2 tid [[thread_position_in_threadgroup]],  // Thread ID in group
    uint2 tg_dim [[threads_per_threadgroup]])      // Threadgroup dimensions
{
  // Use a 2D grid where each thread calculates one element of C
  // gid * tg_dim + tid gives the global thread position (output C index)
  uint row = gid.y * tg_dim.y + tid.y;  // Global output row index
  uint col = gid.x * tg_dim.x + tid.x;  // Global output col index

  if (row < M && col < N) {
    float sum = 0.0f;
    long a_stride0 = a_strides[0];  // Stride along dimension M
    long a_stride1 = a_strides[1];  // Stride along dimension K
    long b_stride0 = b_strides[0];  // Stride along dimension K
    long b_stride1 = b_strides[1];  // Stride along dimension N

    for (uint k = 0; k < K; ++k) {
      // A[row, k] = A + row * stride0 + k * stride1
      // B[k, col] = B + k * stride0 + col * stride1
      sum += A[row * a_stride0 + k * a_stride1] *
             B[k * b_stride0 + col * b_stride1];
    }
    // C is assumed contiguous row-major: C[row, col] = C + row * N + col
    C[row * N + col] = sum;
  }
}

// Matrix multiplication kernel for int32 (Handles Strides for A and B)
kernel void matmul_int32(
    device const int* A [[buffer(0)]], device const int* B [[buffer(1)]],
    device int* C [[buffer(2)]],  // Assumed Contiguous Output
    constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant long* a_strides [[buffer(6)]],        // Stride for A (elems)
    constant long* b_strides [[buffer(7)]],        // Stride for B (elems)
    uint2 gid [[threadgroup_position_in_grid]],    // Grid ID (tile index)
    uint2 tid [[thread_position_in_threadgroup]],  // Thread ID in group
    uint2 tg_dim [[threads_per_threadgroup]])      // Threadgroup dimensions
{
  uint row = gid.y * tg_dim.y + tid.y;  // Global output row index
  uint col = gid.x * tg_dim.x + tid.x;  // Global output col index

  if (row < M && col < N) {
    int sum = 0;
    long a_stride0 = a_strides[0];  // Stride along dimension M
    long a_stride1 = a_strides[1];  // Stride along dimension K
    long b_stride0 = b_strides[0];  // Stride along dimension K
    long b_stride1 = b_strides[1];  // Stride along dimension N

    for (uint k = 0; k < K; ++k) {
      // A[row, k] = A + row * stride0 + k * stride1
      // B[k, col] = B + k * stride0 + col * stride1
      sum += A[row * a_stride0 + k * a_stride1] *
             B[k * b_stride0 + col * b_stride1];
    }
    // C is assumed contiguous row-major: C[row, col] = C + row * N + col
    C[row * N + col] = sum;
  }
}

// --- Tiled Contiguous Kernels (New) ---
// Define tile dimension (e.g., 16x16 or 32x32) - Must match threadgroup size
#define TILE_DIM 16

// Tiled matrix multiplication for float32 (Assumes Contiguous A, B, C)
kernel void matmul_float32_tiled_contiguous(
    device const float* A [[buffer(0)]],  // Shape (M, K)
    device const float* B [[buffer(1)]],  // Shape (K, N)
    device float* C [[buffer(2)]],        // Shape (M, N)
    constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],  // Grid ID (tile index)
    uint2 tid
    [[thread_position_in_threadgroup]])  // Thread ID in group (tidy, tidx)
{
  // Thread and tile indices
  uint tidx = tid.x;  // Thread's column index within the tile
  uint tidy = tid.y;  // Thread's row index within the tile

  // Starting global row and column for the tile this threadgroup works on
  uint tile_row_start = gid.y * TILE_DIM;
  uint tile_col_start = gid.x * TILE_DIM;

  // This thread's target global row and column in C
  uint global_c_row = tile_row_start + tidy;
  uint global_c_col = tile_col_start + tidx;

  // Shared memory tiles
  threadgroup float tileA[TILE_DIM][TILE_DIM];
  threadgroup float tileB[TILE_DIM][TILE_DIM];

  // Accumulator for the C element this thread calculates
  float accumulator = 0.0f;

  // Loop over tiles covering the K dimension
  uint num_k_tiles = (K + TILE_DIM - 1) / TILE_DIM;
  for (uint p = 0; p < num_k_tiles; ++p) {
    // Starting column index in A / row index in B for this K-tile
    uint k_start = p * TILE_DIM;

    // --- Load tile A into shared memory ---
    // Calculate the global A index this thread should load
    uint global_a_row = tile_row_start + tidy;
    uint global_a_col = k_start + tidx;

    // Load A[global_a_row, global_a_col] into tileA[tidy][tidx]
    // Handle boundary conditions (padding with 0)
    if (global_a_row < M && global_a_col < K) {
      tileA[tidy][tidx] =
          A[global_a_row * K + global_a_col];  // Assumes row-major A (stride K)
    } else {
      tileA[tidy][tidx] = 0.0f;
    }

    // --- Load tile B into shared memory ---
    // Calculate the global B index this thread should load
    uint global_b_row =
        k_start +
        tidy;  // Row to load from B corresponds to tidy dimension of tileB
    uint global_b_col =
        tile_col_start +
        tidx;  // Col to load from B corresponds to tidx dimension of tileB

    // Load B[global_b_row, global_b_col] into tileB[tidy][tidx]
    // Handle boundary conditions (padding with 0)
    if (global_b_row < K && global_b_col < N) {
      tileB[tidy][tidx] =
          B[global_b_row * N + global_b_col];  // Assumes row-major B (stride N)
    } else {
      tileB[tidy][tidx] = 0.0f;
    }

    // --- Synchronize ---
    // Ensure all threads in the group have finished loading their parts of the
    // tiles
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Compute partial result from tiles ---
    // Each thread computes its C element part using the shared tiles
    for (uint k_tile = 0; k_tile < TILE_DIM; ++k_tile) {
      accumulator += tileA[tidy][k_tile] * tileB[k_tile][tidx];
    }

    // --- Synchronize ---
    // Ensure all threads have finished computation with the current tiles
    // before the next iteration overwrites the shared memory.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // --- Write result to global memory C ---
  // Check if the target C element is within bounds
  if (global_c_row < M && global_c_col < N) {
    C[global_c_row * N + global_c_col] =
        accumulator;  // Assumes row-major C (stride N)
  }
}

// Tiled matrix multiplication for int32 (Assumes Contiguous A, B, C)
kernel void matmul_int32_tiled_contiguous(
    device const int* A [[buffer(0)]],  // Shape (M, K)
    device const int* B [[buffer(1)]],  // Shape (K, N)
    device int* C [[buffer(2)]],        // Shape (M, N)
    constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],  // Grid ID (tile index)
    uint2 tid
    [[thread_position_in_threadgroup]])  // Thread ID in group (tidy, tidx)
{
  uint tidx = tid.x;
  uint tidy = tid.y;
  uint tile_row_start = gid.y * TILE_DIM;
  uint tile_col_start = gid.x * TILE_DIM;
  uint global_c_row = tile_row_start + tidy;
  uint global_c_col = tile_col_start + tidx;

  threadgroup int tileA[TILE_DIM][TILE_DIM];
  threadgroup int tileB[TILE_DIM][TILE_DIM];

  int accumulator = 0;

  uint num_k_tiles = (K + TILE_DIM - 1) / TILE_DIM;
  for (uint p = 0; p < num_k_tiles; ++p) {
    uint k_start = p * TILE_DIM;

    // Load tile A
    uint global_a_row = tile_row_start + tidy;
    uint global_a_col = k_start + tidx;
    if (global_a_row < M && global_a_col < K) {
      tileA[tidy][tidx] = A[global_a_row * K + global_a_col];
    } else {
      tileA[tidy][tidx] = 0;
    }

    // Load tile B
    uint global_b_row = k_start + tidy;
    uint global_b_col = tile_col_start + tidx;
    if (global_b_row < K && global_b_col < N) {
      tileB[tidy][tidx] = B[global_b_row * N + global_b_col];
    } else {
      tileB[tidy][tidx] = 0;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute
    for (uint k_tile = 0; k_tile < TILE_DIM; ++k_tile) {
      accumulator += tileA[tidy][k_tile] * tileB[k_tile][tidx];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write result
  if (global_c_row < M && global_c_col < N) {
    C[global_c_row * N + global_c_col] = accumulator;
  }
}