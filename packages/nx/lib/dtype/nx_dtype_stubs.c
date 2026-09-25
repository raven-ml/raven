/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <caml/alloc.h>
#include <caml/mlvalues.h>

#include "nx_dtype.h"

/* [format] is the index of a format in [Nx_dtype.Scalar]'s codec order:
   float16, bfloat16, float8 e4m3, e5m2, e4m3fnuz, e5m2fnuz. The OCaml side
   checks it and the code's range. */

intnat caml_nx_dtype_encode(intnat format, double x) {
  switch (format) {
    case 0: return double_to_half(x);
    case 1: return double_to_bfloat16(x);
    case 2: return double_to_fp8_e4m3(x);
    case 3: return double_to_fp8_e5m2(x);
    case 4: return double_to_fp8_e4m3fnuz(x);
    default: return double_to_fp8_e5m2fnuz(x);
  }
}

double caml_nx_dtype_decode(intnat format, intnat code) {
  switch (format) {
    case 0: return half_to_float((uint16_t)code);
    case 1: return bfloat16_to_float((uint16_t)code);
    case 2: return fp8_e4m3_to_float((uint8_t)code);
    case 3: return fp8_e5m2_to_float((uint8_t)code);
    case 4: return fp8_e4m3fnuz_to_float((uint8_t)code);
    default: return fp8_e5m2fnuz_to_float((uint8_t)code);
  }
}

CAMLprim value caml_nx_dtype_encode_byte(value format, value x) {
  return Val_long(caml_nx_dtype_encode(Long_val(format), Double_val(x)));
}

CAMLprim value caml_nx_dtype_decode_byte(value format, value code) {
  return caml_copy_double(
      caml_nx_dtype_decode(Long_val(format), Long_val(code)));
}
