/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Layouts as C reads them: the bytes of an OCaml Layout.t, in native order.
   Only nx_array's stubs read them; kernels read descriptors (nx_array.h).

   A layout is in canonical form: an axis of extent 1 has stride 0, and a
   layout with no element has offset 0 and every stride 0. Its positions are
   non-negative. */

#ifndef NX_LAYOUT_H
#define NX_LAYOUT_H

#include <stdint.h>

typedef struct {
  int64_t rank, flags; /* flags: NX_CONTIGUOUS, NX_DISTINCT, NX_EMPTY */
  int64_t offset;      /* elements */
  int64_t lo, hi;      /* every position lies in [lo, hi); (0, 0) if none */
  int64_t dim[];       /* rank extents, then rank strides */
} nx_layout;

#endif /* NX_LAYOUT_H */
