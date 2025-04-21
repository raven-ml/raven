extern "C" __global__ void add_float32(float* a, float* b, float* c, int& n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) c[idx] = a[idx] + b[idx];
}
