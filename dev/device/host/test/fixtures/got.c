/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Takes the address of cbrt, which the process defines, so the object
   reads it from a word the loader fills: out[0] = cbrt in[0]. */

#include <stdint.h>

/* On x86_64 Windows the process's functions follow its convention. */
#ifdef WINDOWS
#define PROCESS_ABI __attribute__((ms_abi))
#else
#define PROCESS_ABI
#endif

PROCESS_ABI double cbrt(double);

void got(void **b, const int64_t *v) {
  (void)v;
  double *out = b[0];
  const double *in = b[1];
  double(PROCESS_ABI *volatile f)(double) = cbrt;
  out[0] = f(in[0]);
}
