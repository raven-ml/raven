/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
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
  /* The allocation may have moved [s]. */
  c = (const nx_spec_contract *)String_val(s);
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

static const char *const tags[] = {"in",  "coord", "const",
                                   "op1", "op2",   "op3"};

static const char *const op1s[NX_OP1_COUNT] = {
    "copy",  "cast", "bitcast", "neg",  "recip", "abs",   "sign",
    "sqrt",  "exp",  "exp2",    "log",  "log2",  "log1p", "expm1",
    "sin",   "cos",  "tan",     "asin", "acos",  "atan",  "sinh",
    "cosh",  "tanh", "erf",     "floor", "ceil", "round", "trunc"};

static const char *const op2s[NX_OP2_COUNT] = {
    "add",     "sub",     "mul", "fdiv", "idiv",     "mod",
    "pow",     "atan2",   "maximum", "minimum", "and", "or",
    "xor",     "threefry", "equal", "not_equal", "less", "less_equal"};

static const char *const op3s[NX_OP3_COUNT] = {"where", "fma"};

/* The program [p] read through nx_prog, copied first as a kernel copies it,
   one line per node: its tag, kind (or -), dtype code, three operands and
   sixteen bytes of constant bits in hex; then its operands' dtypes and its
   outputs. */
value nx_kernel_support_prog(value p) {
  CAMLparam1(p);
  size_t n = caml_string_length(p);
  nx_prog *c = malloc(n);
  if (c == NULL) caml_raise_out_of_memory();
  memcpy(c, String_val(p), n);
  size_t cap = 128 + 160 * (size_t)c->nnodes + 16 * (size_t)(c->nins + c->nouts);
  char *s = malloc(cap);
  if (s == NULL) {
    free(c);
    caml_raise_out_of_memory();
  }
  size_t at = 0;
  for (int i = 0; i < c->nnodes; i++) {
    const nx_prog_node *d = &c->nodes[i];
    const char *kind = d->tag == NX_NODE_OP1   ? op1s[d->kind]
                       : d->tag == NX_NODE_OP2 ? op2s[d->kind]
                       : d->tag == NX_NODE_OP3 ? op3s[d->kind]
                                               : "-";
    at += snprintf(s + at, cap - at, "%s %s %d %d %d %d ", tags[d->tag], kind,
                   d->dtype, d->a, d->b, d->c);
    for (int k = 0; k < 16; k++)
      at += snprintf(s + at, cap - at, "%02x", d->bits[k]);
    at += snprintf(s + at, cap - at, "\n");
  }
  at += snprintf(s + at, cap - at, "ins");
  for (int k = 0; k < c->nins; k++)
    at += snprintf(s + at, cap - at, " %d", nx_prog_ins(c)[k]);
  at += snprintf(s + at, cap - at, "\nouts");
  for (int k = 0; k < c->nouts; k++)
    at += snprintf(s + at, cap - at, " %d", nx_prog_outs(c)[k]);
  at += snprintf(s + at, cap - at, "\n");
  free(c);
  value r = caml_copy_string(s);
  free(s);
  CAMLreturn(r);
}

/* The map descriptor [s] read through nx_spec_map and nx_spec_pad, copied
   first: its program's bytes in hex, then per load "plain", or "padded",
   its rank, window count, fill in hex, lo, hi, interior and each window's
   axis, size, step and dilation. */
value nx_kernel_support_map(value s) {
  CAMLparam1(s);
  size_t n = caml_string_length(s);
  nx_spec_map *m = malloc(n);
  if (m == NULL) caml_raise_out_of_memory();
  memcpy(m, String_val(s), n);
  size_t cap = 64 + 2 * (size_t)m->prog_len + 8192 * (size_t)m->nloads;
  char *r = malloc(cap);
  if (r == NULL) {
    free(m);
    caml_raise_out_of_memory();
  }
  size_t at = snprintf(r, cap, "family %d prog ", m->family);
  const uint8_t *p = (const uint8_t *)nx_spec_map_prog(m);
  for (int i = 0; i < m->prog_len; i++)
    at += snprintf(r + at, cap - at, "%02x", p[i]);
  for (int k = 0; k < m->nloads; k++) {
    const nx_spec_pad *d = nx_spec_map_pad(m, k);
    if (d == NULL) {
      at += snprintf(r + at, cap - at, "\nplain");
      continue;
    }
    at += snprintf(r + at, cap - at, "\npadded %d %d ", d->rank, d->nwindows);
    for (int i = 0; i < 16; i++)
      at += snprintf(r + at, cap - at, "%02x", d->fill[i]);
    for (int i = 0; i < 3 * d->rank + 4 * d->nwindows; i++)
      at += snprintf(r + at, cap - at, " %lld", (long long)d->geometry[i]);
  }
  free(m);
  value v = caml_copy_string(r);
  free(r);
  CAMLreturn(v);
}
