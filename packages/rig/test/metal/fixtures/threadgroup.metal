// Two kernels: [small] runs on every GPU, [wide] declares 64 KB of
// threadgroup memory, beyond the 32 KB the Apple families give a
// threadgroup, so Metal compiles its library but makes no pipeline of it.

kernel void small(device uint *out [[buffer(0)]],
                  uint i [[thread_position_in_grid]]) {
  out[i] = i;
}

kernel void wide(device uint *out [[buffer(0)]],
                 uint i [[thread_position_in_grid]]) {
  threadgroup uint t[16384];
  t[i % 16384] = i;
  out[i] = t[(i + 1) % 16384];
}
