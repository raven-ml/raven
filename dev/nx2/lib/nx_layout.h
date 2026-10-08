/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Layouts as C reads them: the bytes of a Layout.t, an OCaml string, in
   native order. layout.ml writes them and nx_array's stubs read them; the
   two change together. Kernels read descriptors (nx_array.h).

   The string lives in the OCaml heap, which a collection moves: a stub
   copies what it needs before it allocates or releases the domain lock. */

#ifndef NX_LAYOUT_H
#define NX_LAYOUT_H

#include <stdint.h>

typedef struct {
  int64_t rank, flags; /* flags: NX_CONTIGUOUS, NX_DISTINCT, NX_EMPTY */
  int64_t offset;      /* elements */
  int64_t lo, hi;      /* Layout.span */
  int64_t dim[];       /* rank extents, then rank strides */
} nx_layout;

#endif /* NX_LAYOUT_H */
