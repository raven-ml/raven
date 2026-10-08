/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Takes the address of cbrt, which the process defines, so the object
   reads it from a word of a global offset table: out[0] = cbrt in[0]. */

#include <stdint.h>

double cbrt(double);

void got(void **b, const int64_t *v) {
  (void)v;
  double *out = b[0];
  const double *in = b[1];
  double (*volatile f)(double) = cbrt;
  out[0] = f(in[0]);
}
