/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A compute-bound kernel over the float32 buffer x, for its iterations
   from value 0 to the one before value 1: 64 dependent multiply-adds per
   element. */

#include <stdint.h>

void scale(void **b, const int64_t *v) {
  float *x = b[0];
  for (int64_t i = v[0]; i < v[1]; i++) {
    float y = x[i];
    for (int k = 0; k < 64; k++) y = y * 0.999f + 0.5f;
    x[i] = y;
  }
}
