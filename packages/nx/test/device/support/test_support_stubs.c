/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <stdlib.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/mlvalues.h>

/* [n] zeroed bytes the test keeps for good, as mapped memory stands in for. */
value test_support_alloc(value n) {
  void *p = calloc(1, Long_val(n));
  if (p == NULL) caml_raise_out_of_memory();
  return caml_copy_nativeint((intnat)p);
}
