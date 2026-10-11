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

#include "nx_contract_view.h"
#include "nx_kernel.h"
#include "nx_prog.h"

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

/* The tag C reads of a constructor [v]: Long_val of a constant one,
   Tag_val of a block. */
value nx_kernel_support_tag(value v) {
  return Is_long(v) ? Val_long(Long_val(v)) : Val_int(Tag_val(v));
}

/* nx_kernel.h's codes by name. */
static const struct {
  const char *name;
  int code;
} codes[] = {
#define CODE(c) {#c, c}
    CODE(NX_LOAD_PLAIN),        CODE(NX_LOAD_PADDED),
    CODE(NX_MONOID_SUM),        CODE(NX_MONOID_PROD),
    CODE(NX_MONOID_MAX),        CODE(NX_MONOID_MIN),
    CODE(NX_MONOID_LOGSUMEXP),  CODE(NX_REDUCTION_MONOID),
    CODE(NX_REDUCTION_ARG),     CODE(NX_REDUCTION_MOMENTS),
    CODE(NX_COMBINE_SET),       CODE(NX_COMBINE_ADD),
    CODE(NX_COMBINE_MAX),       CODE(NX_COMBINE_MIN),
    CODE(NX_TRIANGLE_LOWER),    CODE(NX_TRIANGLE_UPPER),
    CODE(NX_FACTORS_REDUCED),   CODE(NX_FACTORS_COMPLETE),
#undef CODE
};

value nx_kernel_support_code(value name) {
  for (size_t i = 0; i < sizeof codes / sizeof codes[0]; i++)
    if (strcmp(codes[i].name, String_val(name)) == 0)
      return Val_int(codes[i].code);
  caml_invalid_argument("Nx_kernel_support.code: not in nx_kernel.h");
}
