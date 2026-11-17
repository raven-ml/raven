#include <metal_compute>

// Linked into constants.metallib beside constants.metal, whose constant 0
// is a uint: here it is a short.

struct args {
  device int *out;
};

constant short s [[function_constant(0)]];

// s, required, sign-extended.
kernel void narrow(constant args &p [[buffer(0)]]) { p.out[0] = int(s); }
