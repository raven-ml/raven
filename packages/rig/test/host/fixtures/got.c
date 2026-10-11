/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Takes the address of cbrt, which the process defines, so the object
   reads it from a word of a global offset table: out[0] = cbrt in[0]. Its
   parameters are out and in. */

#include <stdint.h>

double cbrt(double);

struct params {
  double *out;
  const double *in;
};

void got(const struct params *p, int64_t first, int64_t end) {
  (void)first;
  (void)end;
  double (*volatile f)(double) = cbrt;
  p->out[0] = f(p->in[0]);
}
