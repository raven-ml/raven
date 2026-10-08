// out[i] = a[i] + b[i] for each thread i of the grid, the three arrays of
// floats named by the argument structure at buffer 0.

struct args {
  device const float *a;
  device const float *b;
  device float *out;
};

kernel void add(constant args &x [[buffer(0)]],
                uint i [[thread_position_in_grid]]) {
  x.out[i] = x.a[i] + x.b[i];
}
