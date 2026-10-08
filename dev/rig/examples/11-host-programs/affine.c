/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* out[i] = a * in[i] + c for the iterations i from value 0 to the one
   before value 1, with a and c values 2 and 3. Buffers out and in. */

#include <stdint.h>

void affine(void **b, const int64_t *v) {
  int64_t *out = b[0];
  const int64_t *in = b[1];
  for (int64_t i = v[0]; i < v[1]; i++) out[i] = v[2] * in[i] + v[3];
}
