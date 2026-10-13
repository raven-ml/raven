#include <metal_atomic>

// out[i] = 3i + c for each thread i of the grid.

struct args {
  device uint *out;
  uint c;
};

kernel void fill(constant args &a [[buffer(0)]],
                 uint i [[thread_position_in_grid]]) {
  a.out[i] = 3 * i + a.c;
}

// out[0] = out[0] + 1, once per thread: a chain of one-thread dispatches
// counts them only if each runs after the one before.
kernel void step(constant args &a [[buffer(0)]]) {
  a.out[0] = a.out[0] + 1;
}

// out[0] = the c-th state of a linear congruential generator from 0: a
// single thread that runs for a time proportional to c.
kernel void spin(constant args &a [[buffer(0)]]) {
  uint x = 0;
  for (uint k = 0; k < a.c; k++) x = x * 1664525u + 1013904223u;
  a.out[0] = x;
}

// Waits until out[1] is not 0, for at most c rounds, then sets out[0] to 1
// if it saw out[1] set and to 2 if its rounds ran out: work that runs until
// a kernel on another queue opens its gate. A running kernel never sees the
// host's stores, and reads another kernel's only by a read-modify-write.
kernel void gate(constant args &a [[buffer(0)]]) {
  device coherent(device) metal::atomic_uint *open =
      (device coherent(device) metal::atomic_uint *)&a.out[1];
  uint k = 0;
  while (k < a.c &&
         metal::atomic_fetch_or_explicit(open, 0u,
                                         metal::memory_order_relaxed) == 0)
    k++;
  a.out[0] = k < a.c ? 1 : 2;
}

// Sets out[1] to c: work on another queue that opens a gate.
kernel void unlock(constant args &a [[buffer(0)]]) {
  device metal::atomic_uint *open = (device metal::atomic_uint *)&a.out[1];
  metal::atomic_store_explicit(open, a.c, metal::memory_order_relaxed);
}

// Adds 1 to each of the c bytes at out.
kernel void bump(constant args &a [[buffer(0)]],
                 uint i [[thread_position_in_grid]]) {
  device uchar *b = (device uchar *)a.out;
  if (i < a.c) b[i] = b[i] + 1;
}

// dst[i] = src[i] for each thread i of the grid.

struct pair {
  device uint *dst;
  device const uint *src;
};

kernel void copy(constant pair &a [[buffer(0)]],
                 uint i [[thread_position_in_grid]]) {
  a.dst[i] = a.src[i];
}
