/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Calls the function at its first parameter, of another image, over its
   range, with the parameters that follow. */

#include <stdint.h>

typedef void (*entry)(const void *, int64_t, int64_t);

struct params {
  entry f;
  int64_t rest[];
};

void calling(const struct params *p, int64_t first, int64_t end) {
  p->f(p->rest, first, end);
}
