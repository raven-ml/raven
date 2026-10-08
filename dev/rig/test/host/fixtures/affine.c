/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* out[i] = a * in[i] + c for i below n, with buffers out and in and values
   n, a and c. */

#include <stdint.h>

void affine(void **b, const int64_t *v) {
  int64_t *out = b[0];
  const int64_t *in = b[1];
  for (int64_t i = 0; i < v[0]; i++) out[i] = v[1] * in[i] + v[2];
}
