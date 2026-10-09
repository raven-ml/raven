/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <stddef.h>

#include <caml/mlvalues.h>

#include "nx_array.h"
#include "nx_spec.h"

/* spec.ml writes nx_spec_contract at these byte offsets. */
_Static_assert(offsetof(nx_spec_contract, family) == 0, "at_family");
_Static_assert(offsetof(nx_spec_contract, acc) == 4, "at_acc");
_Static_assert(offsetof(nx_spec_contract, out) == 8, "at_out");
_Static_assert(offsetof(nx_spec_contract, init) == 12, "at_init");
_Static_assert(offsetof(nx_spec_contract, nbatch) == 16, "at_nbatch");
_Static_assert(offsetof(nx_spec_contract, ncontracting) == 20,
               "at_ncontracting");
_Static_assert(offsetof(nx_spec_contract, pairs) == 24, "at_pairs");
_Static_assert(sizeof(((nx_spec_contract *)0)->pairs[0]) == 8,
               "two int32 per pair");

/* Spec.Contract_view's grouping: nx_coalesce over [n] operands of the [r]
   extents [ext], operand k's strides from [st] + k·NX_MAX_RANK, rewritten in
   place. Answers the coalesced rank, or 0 if nx_coalesce refuses, which it
   cannot: the caller passes 2 to NX_MAX_OPERANDS operands, every one of the
   extents [ext]. */
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
  if (nx_coalesce((int)n, a, &l) != NX_OK) return 0;
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
