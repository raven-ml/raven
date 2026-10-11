#include <metal_compute>

// Kernels launched as parts: each reads its parameters as a constant
// structure at buffer 0, buffers by the addresses in it.

// out[k] = a + b * k + f for each thread, k its index in the grid,
// threadgroup by threadgroup in x, y then z order, and thread by thread in
// the same order within its threadgroup.
struct ids_args {
  device uint *out;
  ulong a;
  uint b;
  float f;
};

kernel void ids(constant ids_args &p [[buffer(0)]],
                uint3 g [[threadgroup_position_in_grid]],
                uint3 ng [[threadgroups_per_grid]],
                uint3 nt [[threads_per_threadgroup]],
                uint t [[thread_index_in_threadgroup]]) {
  uint n = nt.x * nt.y * nt.z;
  uint k = ((g.z * ng.y + g.y) * ng.x + g.x) * n + t;
  p.out[k] = uint(p.a) + p.b * k + uint(p.f);
}

// out[k] = c + the thread index after k's in its threadgroup, read from
// threadgroup memory [0], which holds a word per thread: each thread
// writes its own index there, and a barrier orders the writes before the
// reads.
struct shared_args {
  device uint *out;
  uint c;
};

kernel void neighbours(constant shared_args &p [[buffer(0)]],
                       threadgroup uint *s [[threadgroup(0)]],
                       uint3 g [[threadgroup_position_in_grid]],
                       uint3 ng [[threadgroups_per_grid]],
                       uint3 nt [[threads_per_threadgroup]],
                       uint t [[thread_index_in_threadgroup]]) {
  uint n = nt.x * nt.y * nt.z;
  s[t] = t;
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  uint k = ((g.z * ng.y + g.y) * ng.x + g.x) * n + t;
  p.out[k] = p.c + s[(t + 1) % n];
}

// dst[i] = 2 src[i] + c for each thread i of the grid.
struct twice_args {
  device uint *dst;
  device const uint *src;
  uint c;
};

kernel void twice(constant twice_args &p [[buffer(0)]],
                  uint i [[thread_position_in_grid]]) {
  p.dst[i] = 2 * p.src[i] + p.c;
}

// out[0] = out[0] + 1: a chain of launches counts them only if each runs
// after the one before.
struct step_args {
  device uint *out;
};

kernel void step(constant step_args &p [[buffer(0)]]) {
  p.out[0] = p.out[0] + 1;
}

// out[0] = the sum of the 1,022 words before [out]: 4,096 bytes of
// parameters, the most a launch has, whose last 8 are a ref.
struct last_args {
  uint w[1022];
  device uint *out;
};

kernel void last(constant last_args &p [[buffer(0)]]) {
  uint s = 0;
  for (uint k = 0; k < 1022; k++) s += p.w[k];
  p.out[0] = s;
}

// Nothing: a launch whose cost is the launch's alone.
kernel void empty() {}
