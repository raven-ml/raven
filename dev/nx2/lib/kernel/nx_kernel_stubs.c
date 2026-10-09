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

/* prog.ml writes nx_prog at these byte offsets. */
_Static_assert(offsetof(nx_prog, nodes) == 16, "header");
_Static_assert(sizeof(nx_prog_node) == 40, "record");
_Static_assert(offsetof(nx_prog_node, bits) == 24, "at_bits");
_Static_assert(NX_OP1_COUNT == 28 && NX_OP2_COUNT == 18 && NX_OP3_COUNT == 2,
               "one code per kind");

/* The bits nx_dtype.h's store of [x] writes into an element of the float
   dtype [dt] of at most 16 bits. */
intnat nx_kernel_narrow_bits(intnat dt, double x) {
  return (intnat)nx_double_to_bits((int)dt, x);
}

value nx_kernel_narrow_bits_byte(value dt, value x) {
  return Val_long(nx_kernel_narrow_bits(Long_val(dt), Double_val(x)));
}

/* spec.ml writes nx_contract_view at these byte offsets. */
_Static_assert(offsetof(nx_contract_view, extent) == 0, "at_extent");
_Static_assert(offsetof(nx_contract_view, offset) == 32, "at_offset");
_Static_assert(offsetof(nx_contract_view, stride) == 64, "at_stride");
_Static_assert(NX_VIEW_DST == 3 && NX_VIEW_CONTRACTED == 3,
               "operand and axis indices");
_Static_assert(sizeof(nx_contract_view) == 192, "view_bytes");

/* Spec.Contract_view's grouping: nx_coalesce over [n] operands of the [r]
   int64 extents at byte [at_ext] of the view [v], operand k's strides from
   byte [at_st] + 8·k·NX_MAX_RANK, rewritten in place. Answers the coalesced
   rank, or 0 if nx_coalesce refuses, which it cannot: the caller passes 2
   to NX_MAX_OPERANDS operands, every one of the extents [ext]. */
intnat nx_kernel_coalesce(intnat n, intnat r, value v, intnat at_ext,
                          intnat at_st) {
  int64_t *ext = (int64_t *)(Bytes_val(v) + at_ext);
  int64_t *st = (int64_t *)(Bytes_val(v) + at_st);
  nx_array a[NX_MAX_OPERANDS];
  nx_loop l;
  for (int k = 0; k < n; k++) {
    a[k].rank = (int)r;
    a[k].flags = 0;
    a[k].offset = 0;
    for (int i = 0; i < r; i++) {
      a[k].dim[i] = ext[i];
      a[k].dim[r + i] = st[k * NX_MAX_RANK + i];
    }
  }
  if (nx_coalesce((int)n, a, &l) != NX_OK) return 0;
  for (int i = 0; i < l.rank; i++) {
    ext[i] = l.extent[i];
    for (int k = 0; k < n; k++) st[k * NX_MAX_RANK + i] = l.step[k][i];
  }
  return l.rank;
}

value nx_kernel_coalesce_byte(value n, value r, value v, value at_ext,
                              value at_st) {
  return Val_long(nx_kernel_coalesce(Long_val(n), Long_val(r), v,
                                     Long_val(at_ext), Long_val(at_st)));
}
