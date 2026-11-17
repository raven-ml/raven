#include <metal_compute>

// Kernels specialised by function constants: each writes the values of its
// constants into out, a word each, a signed one sign-extended. Two kernels
// declare constant 0 at different types.

struct args {
  device int *out;
};

constant uint a [[function_constant(0)]];
constant uint b [[function_constant(1)]];

// a and b, both required.
kernel void pair(constant args &p [[buffer(0)]]) {
  p.out[0] = int(a);
  p.out[1] = int(b);
}

// 42, of no constant.
kernel void fixed(constant args &p [[buffer(0)]]) { p.out[0] = 42; }

// c, or 7 if the image gives none.
constant uint c [[function_constant(2)]];
constant uint c_or = is_function_constant_defined(c) ? c : 7;

kernel void optional(constant args &p [[buffer(0)]]) { p.out[0] = int(c_or); }

constant bool t_bool [[function_constant(3)]];
constant char t_char [[function_constant(4)]];
constant uchar t_uchar [[function_constant(5)]];
constant short t_short [[function_constant(6)]];
constant ushort t_ushort [[function_constant(7)]];
constant int t_int [[function_constant(8)]];

// Constants 3 to 8, one of each type the image's values carry, all
// required.
kernel void typed(constant args &p [[buffer(0)]]) {
  p.out[0] = int(t_bool);
  p.out[1] = int(t_char);
  p.out[2] = int(t_uchar);
  p.out[3] = int(t_short);
  p.out[4] = int(t_ushort);
  p.out[5] = t_int;
}

// The bits of f, or of 1.5 if the image gives none: a float, which the
// image's values do not carry.
constant float f [[function_constant(9)]];
constant float f_or = is_function_constant_defined(f) ? f : 1.5;

kernel void real(constant args &p [[buffer(0)]]) {
  p.out[0] = as_type<int>(f_or);
}
