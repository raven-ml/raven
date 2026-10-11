/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A compute-bound kernel over the float32 array x for each thread i of the
   range below n, its parameters: 64 dependent multiply-adds per element. */

#include <stdint.h>

struct params {
  float *x;
  int64_t n;
};

void scale(const struct params *p, int64_t first, int64_t end) {
  for (int64_t i = first; i < end && i < p->n; i++) {
    float y = p->x[i];
    for (int k = 0; k < 64; k++) y = y * 0.999f + 0.5f;
    p->x[i] = y;
  }
}
