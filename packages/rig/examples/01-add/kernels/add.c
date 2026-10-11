/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* out[i] = a[i] + b[i] for each thread i of the launch below n, on the
   host's cores. Its parameters are those of every GPU's add: the addresses
   of the three arrays, then n. */

#include <stdint.h>

struct add {
  const float *a, *b;
  float *out;
  int64_t n;
};

void add(const struct add *p, int64_t first, int64_t end) {
  for (int64_t i = first; i < end && i < p->n; i++)
    p->out[i] = p->a[i] + p->b[i];
}
