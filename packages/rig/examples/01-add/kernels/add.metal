// out[i] = a[i] + b[i] for each thread i of the grid below n. Metal passes
// the launch's parameters as the structure at buffer 0: the addresses of the
// three arrays, then n.

struct params {
  device const float *a;
  device const float *b;
  device float *out;
  ulong n;
};

kernel void add(constant params &p [[buffer(0)]],
                uint i [[thread_position_in_grid]]) {
  if (i < p.n) p.out[i] = p.a[i] + p.b[i];
}
