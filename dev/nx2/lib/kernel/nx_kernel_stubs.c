/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <caml/mlvalues.h>

#include "nx_array.h"
#include "nx_spec.h"

_Static_assert(NX_SPEC_MAX_RANK == NX_MAX_RANK,
               "nx_spec.h's rank bound is Layout.max_rank");

/* Spec.Contract_view's grouping: nx_coalesce over [n] operands of the [r]
   extents [ext], operand k's strides from [st] + k·NX_MAX_RANK, rewritten in
   place. Answers the coalesced rank. The caller passes 1 to NX_MAX_OPERANDS
   operands of at most NX_MAX_RANK axes. */
intnat nx_kernel_coalesce(intnat n, intnat r, value ext, value st) {
  nx_array a[NX_MAX_OPERANDS];
  nx_loop l;
  for (int k = 0; k < n; k++) {
    a[k].rank = (int)r;
    a[k].flags = 0;
    a[k].offset = 0;
    for (int i = 0; i < r; i++) {
      a[k].dim[i] = Long_val(Field(ext, i));
      a[k].dim[r + i] = Long_val(Field(st, k * NX_MAX_RANK + i));
    }
  }
  (void)nx_coalesce((int)n, a, &l);
  /* Int arrays: immediates need no write barrier. */
  for (int i = 0; i < l.rank; i++) {
    Field(ext, i) = Val_long(l.extent[i]);
    for (int k = 0; k < n; k++)
      Field(st, k * NX_MAX_RANK + i) = Val_long(l.step[k][i]);
  }
  return l.rank;
}

value nx_kernel_coalesce_byte(value n, value r, value ext, value st) {
  return Val_long(nx_kernel_coalesce(Long_val(n), Long_val(r), ext, st));
}
