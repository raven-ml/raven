/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Decimal digits of a float, rounded once. Under mingw-w64, the snprintf of
   msvcrt, which OCaml's Printf calls, keeps 17 correct digits and rounds
   from those, so a value rounds twice: float64 5.5626846462680034e-309
   printed 16 digits as ...004. mingw-w64's own snprintf rounds once. */

#include <stdio.h>

#include <caml/alloc.h>
#include <caml/mlvalues.h>

value caml_nx_format_e(value v_p, value v_x) {
  char buf[64];
#ifdef __MINGW32__
  __mingw_snprintf(buf, sizeof buf, "%.*e", Int_val(v_p), Double_val(v_x));
#else
  snprintf(buf, sizeof buf, "%.*e", Int_val(v_p), Double_val(v_x));
#endif
  return caml_copy_string(buf);
}
