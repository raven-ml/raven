/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A function that needs linking: a constant of .rodata, a call of a
   function of its own, and of cbrt, which the process defines. Built with
   unwind tables, whose .eh_frame relocates to the code. With buffers out
   and in, and value i: out = {cbrt in[0], table[i], twice in[1]}. */

#include <stdint.h>

/* On x86_64 Windows the process's functions follow its convention. */
#ifdef WINDOWS
#define PROCESS_ABI __attribute__((ms_abi))
#else
#define PROCESS_ABI
#endif

PROCESS_ABI double cbrt(double);

static const double table[4] = {1.5, 2.25, 3.125, 4.0625};

__attribute__((noinline)) double twice(double x) { return 2 * x; }

void linked(void **b, const int64_t *v) {
  double *out = b[0];
  const double *in = b[1];
  out[0] = cbrt(in[0]);
  out[1] = table[v[0] & 3];
  out[2] = twice(in[1]);
}
