/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <string.h>

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "nx_spec.h"

/* The fields of the contraction descriptor [s], read through
   nx_spec_contract: family, acc, out, init, nbatch, ncontracting, then each
   batch pair and each contracting pair. */
value nx_kernel_support_contract(value s) {
  CAMLparam1(s);
  CAMLlocal1(r);
  if (caml_string_length(s) < sizeof(nx_spec_contract))
    caml_invalid_argument("not an nx_spec_contract");
  const nx_spec_contract *c = (const nx_spec_contract *)String_val(s);
  int nb = c->nbatch, nc = c->ncontracting;
  if (caml_string_length(s) !=
      sizeof(nx_spec_contract) + sizeof c->pairs[0] * (size_t)(nb + nc))
    caml_invalid_argument("not an nx_spec_contract");
  r = caml_alloc_tuple(6 + 2 * (nb + nc));
  int at = 0;
  Store_field(r, at++, Val_int(c->family));
  Store_field(r, at++, Val_int(c->acc));
  Store_field(r, at++, Val_int(c->out));
  Store_field(r, at++, Val_int(c->init));
  Store_field(r, at++, Val_int(nb));
  Store_field(r, at++, Val_int(nc));
  for (int k = 0; k < nb + nc; k++)
    for (int j = 0; j < 2; j++) Store_field(r, at++, Val_int(c->pairs[k][j]));
  CAMLreturn(r);
}

/* The view [v] read through nx_contract_view, copied first as a kernel
   does: its 4 extents, 4 offsets, then 16 strides by operand then axis. */
value nx_kernel_support_view(value v) {
  CAMLparam1(v);
  CAMLlocal1(r);
  if (caml_string_length(v) < sizeof(nx_contract_view))
    caml_invalid_argument("not an nx_contract_view");
  nx_contract_view c;
  memcpy(&c, Bytes_val(v), sizeof c);
  r = caml_alloc_tuple(24);
  for (int i = 0; i < 4; i++) {
    Store_field(r, i, Val_long(c.extent[i]));
    Store_field(r, 4 + i, Val_long(c.offset[i]));
    for (int x = 0; x < 4; x++)
      Store_field(r, 8 + 4 * i + x, Val_long(c.stride[i][x]));
  }
  CAMLreturn(r);
}
