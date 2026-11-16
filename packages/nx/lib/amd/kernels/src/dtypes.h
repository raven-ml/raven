/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Dtypes: nx_dtype.h's codes and conversions, inlined like device.h's
   functions, and what the units that store any dtype share over them. A
   header apart from device.h, since nx_dtype.h's math.h meets spectral's
   own functions of the same names. */

#ifndef NX_AMD_DTYPES_H
#define NX_AMD_DTYPES_H

#include "device.h"

#pragma clang attribute push(__attribute__((always_inline)), apply_to = function)
#include "nx_dtype.h"
#pragma clang attribute pop

/* Whether the integer dtype [dt] is signed. */
DEVICE bool is_signed(int dt) {
  return dt == NX_INT64 || dt == NX_INT32 || dt == NX_INT16 || dt == NX_INT8 ||
         dt == NX_INT4;
}

/* The code of the narrow float [dt] nearest the float32 [f]. */
DEVICE uint32_t narrow_code(int dt, float f) {
  switch (dt) {
  case NX_FLOAT16: return nx_float_to_f16(f);
  case NX_BFLOAT16: return nx_float_to_bf16(f);
  case NX_FLOAT8_E4M3FN: return nx_float_to_e4m3fn(f);
  case NX_FLOAT8_E5M2: return nx_float_to_e5m2(f);
  default: return nx_float_to_e2m1fn(f);
  }
}

#endif
