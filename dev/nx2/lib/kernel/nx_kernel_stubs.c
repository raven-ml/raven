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

/* spec.ml writes nx_spec_map and nx_spec_pad at these byte offsets. */
_Static_assert(offsetof(nx_spec_map, nloads) == 4, "at_nloads");
_Static_assert(offsetof(nx_spec_map, at_prog) == 8, "at_prog");
_Static_assert(offsetof(nx_spec_map, prog_len) == 12, "at_prog_len");
_Static_assert(offsetof(nx_spec_map, loads) == 16, "at_loads");
_Static_assert(offsetof(nx_spec_pad, fill) == 8, "at_fill");
_Static_assert(offsetof(nx_spec_pad, geometry) == 24, "at_geometry");

/* The most operands plus outputs of a loop, which bounds a program's. */
intnat nx_kernel_max_operands(value unit) {
  (void)unit;
  return NX_MAX_OPERANDS;
}

value nx_kernel_max_operands_byte(value unit) {
  return Val_long(nx_kernel_max_operands(unit));
}

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

/* Spec.Contract_view's grouping: nx_coalesce_dims over [n] operands of the
   [r] int64 extents at byte [at_ext] of the view [v], operand k's strides
   from byte [at_st] + 8·k·NX_MAX_RANK, in place. Answers the merged rank. */
intnat nx_kernel_coalesce(intnat n, intnat r, value v, intnat at_ext,
                          intnat at_st) {
  return nx_coalesce_dims((int)n, (int)r, (int64_t *)(Bytes_val(v) + at_ext),
                          (int64_t (*)[NX_MAX_RANK])(Bytes_val(v) + at_st));
}

value nx_kernel_coalesce_byte(value n, value r, value v, value at_ext,
                              value at_st) {
  return Val_long(nx_kernel_coalesce(Long_val(n), Long_val(r), v,
                                     Long_val(at_ext), Long_val(at_st)));
}
