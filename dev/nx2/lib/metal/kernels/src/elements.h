/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* How nx.metal's kernels read and store elements whose dtype a launch
   names at run time: a float out or init as float32, rounded once on the
   way back; an integer widened by its sign into an accumulator, and an
   accumulator's sum stored as a cast from it does. Every family's source
   includes it. */

#ifndef NX_METAL_ELEMENTS_H
#define NX_METAL_ELEMENTS_H

#include "nx_dtype.h"

/* Loops over registers unroll whole, so no array of them is indexed at run
   time, which would put it in memory. */
#define UNROLL _Pragma("clang loop unroll(full)")

/* x read from the code at [p] of dtype [dt], an out or init dtype. */
static float decode(uint dt, device const void *p, ulong i) {
  switch (dt) {
  case NX_FLOAT16: return float(((device const half *)p)[i]);
  case NX_BFLOAT16: return nx_bf16_to_float(((device const ushort *)p)[i]);
  default: return ((device const float *)p)[i];
  }
}

static void encode(uint dt, device void *p, ulong i, float x) {
  switch (dt) {
  case NX_FLOAT16: ((device half *)p)[i] = half(x); break;
  case NX_BFLOAT16: ((device ushort *)p)[i] = nx_float_to_bf16(x); break;
  default: ((device float *)p)[i] = x;
  }
}

/* The element at [i] of the integer array [p] of dtype [dt], widened to A
   by its sign. */
template <typename A>
static A widen(uint dt, device const uchar *p, ulong i) {
  switch (dt) {
  case NX_INT8: return A(long(((device const char *)p)[i]));
  case NX_UINT8: return A(((device const uchar *)p)[i]);
  case NX_INT16: return A(long(((device const short *)p)[i]));
  case NX_UINT16: return A(((device const ushort *)p)[i]);
  case NX_INT32: return A(long(((device const int *)p)[i]));
  case NX_UINT32: return A(((device const uint *)p)[i]);
  default: return A(((device const ulong *)p)[i]);
  }
}

/* Stores the sum x, wrapped to the accumulator of dtype [acc], at [i] of
   the integer array [p] of dtype [dt], as a cast from the accumulator
   does: its low bits, or, into a wider out, widened by the accumulator's
   sign. */
static void narrow(uint dt, uint acc, device uchar *p, ulong i, ulong x) {
  switch (acc) {
  case NX_INT8: x = ulong(long(char(x))); break;
  case NX_UINT8: x = ulong(uchar(x)); break;
  case NX_INT16: x = ulong(long(short(x))); break;
  case NX_UINT16: x = ulong(ushort(x)); break;
  case NX_INT32: x = ulong(long(int(x))); break;
  case NX_UINT32: x = ulong(uint(x)); break;
  }
  switch (dt) {
  case NX_INT8:
  case NX_UINT8: ((device uchar *)p)[i] = uchar(x); break;
  case NX_INT16:
  case NX_UINT16: ((device ushort *)p)[i] = ushort(x); break;
  case NX_INT32:
  case NX_UINT32: ((device uint *)p)[i] = uint(x); break;
  default: ((device ulong *)p)[i] = x;
  }
}

#endif /* NX_METAL_ELEMENTS_H */
