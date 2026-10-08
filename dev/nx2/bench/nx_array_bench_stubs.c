/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The door alone: three operands read and released, with no kernel. */

#include <caml/mlvalues.h>

#include "nx_array.h"

value nx_array_bench_read_3(value z, value x, value y) {
  int dt = nx_array_dtype(z);
  nx_operand in[3] = {{z, dt, 1}, {x, dt, 0}, {y, dt, 0}};
  nx_array a[3];
  int e = nx_read(3, in, a);
  if (!e) nx_done(3, a);
  return Val_int(e);
}
