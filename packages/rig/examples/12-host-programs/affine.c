/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* out[i] = a * in[i] + c for each thread i from first to end - 1 that is
   below n. Its parameters are out, in, n, a and c. */

#include <stdint.h>

struct params {
  int64_t *out;
  const int64_t *in;
  int64_t n, a, c;
};

void affine(const struct params *p, int64_t first, int64_t end) {
  for (int64_t i = first; i < end && i < p->n; i++)
    p->out[i] = p->a * p->in[i] + p->c;
}
