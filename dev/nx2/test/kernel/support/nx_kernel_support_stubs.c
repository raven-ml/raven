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

#include "nx_array.h"
#include "nx_kernel.h"
#include "nx_prog.h"

static const char *const tags[] = {"in",  "coord", "const",
                                   "op1", "op2",   "op3"};

static const char *const op1s[NX_OP1_COUNT] = {
    "copy",  "cast", "bitcast", "neg",  "recip", "abs",   "sign",
    "sqrt",  "exp",  "exp2",    "log",  "log2",  "log1p", "expm1",
    "sin",   "cos",  "tan",     "asin", "acos",  "atan",  "sinh",
    "cosh",  "tanh", "asinh",   "acosh", "atanh", "erf",  "floor",
    "ceil",  "round", "trunc"};

static const char *const op2s[NX_OP2_COUNT] = {
    "add",     "sub",     "mul", "fdiv", "idiv",     "mod",
    "pow",     "atan2",   "maximum", "minimum", "and", "or",
    "xor",     "threefry", "equal", "not_equal", "less", "less_equal"};

static const char *const op3s[NX_OP3_COUNT] = {"where", "fma"};

/* The kind of the node [d] of tag [tag] read through nx_prog.h's codes. */
static const char *kind(int tag, value d) {
  switch (tag) {
    case NX_NODE_OP1: return op1s[NX_OP1_CODE(Field(d, NX_OP1_KIND))];
    case NX_NODE_OP2: return op2s[NX_OP2_CODE(Field(d, NX_OP2_KIND))];
    default: return op3s[Long_val(Field(d, NX_OP3_KIND))];
  }
}

value nx_kernel_support_op1(value k) { return Val_int(NX_OP1_CODE(k)); }
value nx_kernel_support_op2(value k) { return Val_int(NX_OP2_CODE(k)); }

/* The program [p] read through nx_prog.h's codes, one line per node: its
   tag, then its fields; then its operands' dtypes and its outputs. */
value nx_kernel_support_prog(value p) {
  CAMLparam1(p);
  value ins = Field(p, NX_PROG_INS), nodes = Field(p, NX_PROG_NODES);
  value outs = Field(p, NX_PROG_OUTS);
  size_t cap = 128 + 96 * Wosize_val(nodes) +
               16 * (Wosize_val(ins) + Wosize_val(outs));
  char *s = malloc(cap);
  if (s == NULL) caml_raise_out_of_memory();
  size_t at = 0;
  for (mlsize_t i = 0; i < Wosize_val(nodes); i++) {
    value d = Field(nodes, i);
    int tag = Tag_val(d);
    at += snprintf(s + at, cap - at, "%s", tags[tag]);
    switch (tag) {
      case NX_NODE_IN:
        at += snprintf(s + at, cap - at, " %ld",
                       Long_val(Field(d, NX_IN_OPERAND)));
        break;
      case NX_NODE_COORD:
        at += snprintf(s + at, cap - at, " %ld",
                       Long_val(Field(d, NX_COORD_AXIS)));
        break;
      case NX_NODE_CONST: {
        value b = Field(d, NX_CONST_BITS);
        at += snprintf(s + at, cap - at, " %ld ",
                       Long_val(Field(d, NX_CONST_DTYPE)));
        for (mlsize_t k = 0; k < caml_string_length(b); k++)
          at += snprintf(s + at, cap - at, "%02x", Byte_u(b, k));
        break;
      }
      case NX_NODE_OP1:
        at += snprintf(s + at, cap - at, " %s %ld %ld", kind(tag, d),
                       Long_val(Field(d, NX_OP1_DTYPE)),
                       Long_val(Field(d, NX_OP1_OPERAND)));
        break;
      case NX_NODE_OP2:
        at += snprintf(s + at, cap - at, " %s", kind(tag, d));
        for (int k = 0; k < 2; k++)
          at += snprintf(s + at, cap - at, " %ld",
                         Long_val(Field(d, NX_OP2_OPERANDS + k)));
        break;
      default:
        at += snprintf(s + at, cap - at, " %s", kind(tag, d));
        for (int k = 0; k < 3; k++)
          at += snprintf(s + at, cap - at, " %ld",
                         Long_val(Field(d, NX_OP3_OPERANDS + k)));
    }
    at += snprintf(s + at, cap - at, "\n");
  }
  at += snprintf(s + at, cap - at, "ins");
  for (mlsize_t k = 0; k < Wosize_val(ins); k++)
    at += snprintf(s + at, cap - at, " %ld", Long_val(Field(ins, k)));
  at += snprintf(s + at, cap - at, "\nouts");
  for (mlsize_t k = 0; k < Wosize_val(outs); k++)
    at += snprintf(s + at, cap - at, " %ld", Long_val(Field(outs, k)));
  at += snprintf(s + at, cap - at, "\n");
  value r = caml_copy_string(s);
  free(s);
  CAMLreturn(r);
}

/* The tag C reads of a constructor [v]: Long_val of a constant one,
   Tag_val of a block. */
value nx_kernel_support_tag(value v) {
  return Is_long(v) ? Val_long(Long_val(v)) : Val_int(Tag_val(v));
}

/* nx_kernel.h's and nx_prog.h's codes by name. */
static const struct {
  const char *name;
  int code;
} codes[] = {
#define CODE(c) {#c, c}
    CODE(NX_LOAD_PLAIN),        CODE(NX_LOAD_PADDED),
    CODE(NX_MONOID_SUM),        CODE(NX_MONOID_PROD),
    CODE(NX_MONOID_MAX),        CODE(NX_MONOID_MIN),
    CODE(NX_REDUCTION_MONOID),  CODE(NX_REDUCTION_ARG),
    CODE(NX_REDUCTION_LOGSUMEXP), CODE(NX_REDUCTION_MOMENTS),
    CODE(NX_COMBINE_SET),       CODE(NX_COMBINE_ADD),
    CODE(NX_COMBINE_MAX),       CODE(NX_COMBINE_MIN),
    CODE(NX_TRIANGLE_LOWER),    CODE(NX_TRIANGLE_UPPER),
    CODE(NX_PROG_INS),          CODE(NX_PROG_NODES),
    CODE(NX_PROG_OUTS),         CODE(NX_NODE_IN),
    CODE(NX_NODE_COORD),        CODE(NX_NODE_CONST),
    CODE(NX_NODE_OP1),          CODE(NX_NODE_OP2),
    CODE(NX_NODE_OP3),          CODE(NX_IN_OPERAND),
    CODE(NX_COORD_AXIS),        CODE(NX_CONST_DTYPE),
    CODE(NX_CONST_BITS),        CODE(NX_OP1_KIND),
    CODE(NX_OP1_DTYPE),         CODE(NX_OP1_OPERAND),
    CODE(NX_OP2_KIND),          CODE(NX_OP2_OPERANDS),
    CODE(NX_OP3_KIND),          CODE(NX_OP3_OPERANDS),
    CODE(NX_OP2_BINARY),        CODE(NX_OP2_COMPARE),
    CODE(NX_OP1_COPY),          CODE(NX_OP1_CAST),
    CODE(NX_OP1_BITCAST),       CODE(NX_OP1_NEG),
    CODE(NX_OP1_RECIP),         CODE(NX_OP1_ABS),
    CODE(NX_OP1_SIGN),          CODE(NX_OP1_SQRT),
    CODE(NX_OP1_EXP),           CODE(NX_OP1_EXP2),
    CODE(NX_OP1_LOG),           CODE(NX_OP1_LOG2),
    CODE(NX_OP1_LOG1P),         CODE(NX_OP1_EXPM1),
    CODE(NX_OP1_SIN),           CODE(NX_OP1_COS),
    CODE(NX_OP1_TAN),           CODE(NX_OP1_ASIN),
    CODE(NX_OP1_ACOS),          CODE(NX_OP1_ATAN),
    CODE(NX_OP1_SINH),          CODE(NX_OP1_COSH),
    CODE(NX_OP1_TANH),          CODE(NX_OP1_ASINH),
    CODE(NX_OP1_ACOSH),         CODE(NX_OP1_ATANH),
    CODE(NX_OP1_ERF),
    CODE(NX_OP1_FLOOR),         CODE(NX_OP1_CEIL),
    CODE(NX_OP1_ROUND),         CODE(NX_OP1_TRUNC),
    CODE(NX_OP2_ADD),           CODE(NX_OP2_SUB),
    CODE(NX_OP2_MUL),           CODE(NX_OP2_FDIV),
    CODE(NX_OP2_IDIV),          CODE(NX_OP2_MOD),
    CODE(NX_OP2_POW),           CODE(NX_OP2_ATAN2),
    CODE(NX_OP2_MAXIMUM),       CODE(NX_OP2_MINIMUM),
    CODE(NX_OP2_AND),           CODE(NX_OP2_OR),
    CODE(NX_OP2_XOR),           CODE(NX_OP2_THREEFRY),
    CODE(NX_OP2_EQUAL),         CODE(NX_OP2_NOT_EQUAL),
    CODE(NX_OP2_LESS),          CODE(NX_OP2_LESS_EQUAL),
    CODE(NX_OP3_WHERE),         CODE(NX_OP3_FMA),
    CODE(NX_OP1_COUNT),         CODE(NX_OP2_COUNT),
    CODE(NX_OP3_COUNT),
#undef CODE
};

value nx_kernel_support_code(value name) {
  for (size_t i = 0; i < sizeof codes / sizeof codes[0]; i++)
    if (strcmp(codes[i].name, String_val(name)) == 0)
      return Val_int(codes[i].code);
  caml_invalid_argument("Nx_kernel_support.code: in no header");
}
