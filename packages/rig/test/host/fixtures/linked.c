/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A function that needs linking: a constant of .rodata, a call of a
   function of its own, and of cbrt, which the process defines. With the
   parameters out, in and i: out = {cbrt in[0], table[i], twice in[1]}. */

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

struct params {
  double *out;
  const double *in;
  int64_t i;
};

void linked(const struct params *p, int64_t first, int64_t end) {
  (void)first;
  (void)end;
  p->out[0] = cbrt(p->in[0]);
  p->out[1] = table[p->i & 3];
  p->out[2] = twice(p->in[1]);
}
