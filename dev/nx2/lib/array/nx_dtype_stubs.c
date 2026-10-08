/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <caml/alloc.h>
#include <caml/mlvalues.h>

#include "nx_dtype.h"

/* The value [x] has once stored in the float dtype [dt]. */
double nx_dtype_round(intnat dt, double x) {
  switch (dt) {
    case NX_FLOAT64: return x;
    case NX_FLOAT32: return (float)x;
    default:
      return nx_bits_to_float((int)dt,
                              (uint32_t)nx_double_to_bits((int)dt, x));
  }
}

/* The bits a store of [x] writes into an element of the integer dtype
   [dt]. */
int64_t nx_dtype_store(intnat dt, double x) {
  return nx_double_to_bits((int)dt, x);
}

value nx_dtype_round_byte(value dt, value x) {
  return caml_copy_double(nx_dtype_round(Long_val(dt), Double_val(x)));
}

value nx_dtype_store_byte(value dt, value x) {
  return caml_copy_int64(nx_dtype_store(Long_val(dt), Double_val(x)));
}
