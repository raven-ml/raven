/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Programs (Nx_kernel.Prog) as C kernels read them. A program is the start
   of an OCaml string, which the collector may move: a kernel copies what
   it reads into its own memory before anything that can run OCaml code,
   nx_read included. Free of OCaml's headers, so device code includes it. */

#ifndef NX_PROG_H
#define NX_PROG_H

#include <stdint.h>

/* Prog.t: a scalar program. Counts of operands, nodes and outputs, then a
   record per node, then the operands' dtype codes (nx_dtype.h) and the
   outputs' node indices; read through nx_prog_ins and nx_prog_outs. A
   node's [dtype] is its result's; [a], [b] and [c] are its operand, axis or
   earlier nodes; a constant's bits are in [bits], zero-padded. Kinds are
   the codes below, in the order of Prog's constructors. */
enum { NX_NODE_IN, NX_NODE_COORD, NX_NODE_CONST, NX_NODE_OP1, NX_NODE_OP2,
       NX_NODE_OP3 };

enum {
  NX_OP1_COPY, NX_OP1_CAST, NX_OP1_BITCAST, NX_OP1_NEG, NX_OP1_RECIP,
  NX_OP1_ABS, NX_OP1_SIGN, NX_OP1_SQRT, NX_OP1_EXP, NX_OP1_EXP2, NX_OP1_LOG,
  NX_OP1_LOG2, NX_OP1_LOG1P, NX_OP1_EXPM1, NX_OP1_SIN, NX_OP1_COS,
  NX_OP1_TAN, NX_OP1_ASIN, NX_OP1_ACOS, NX_OP1_ATAN, NX_OP1_SINH,
  NX_OP1_COSH, NX_OP1_TANH, NX_OP1_ERF, NX_OP1_FLOOR, NX_OP1_CEIL,
  NX_OP1_ROUND, NX_OP1_TRUNC, NX_OP1_COUNT
};

enum {
  NX_OP2_ADD, NX_OP2_SUB, NX_OP2_MUL, NX_OP2_FDIV, NX_OP2_IDIV, NX_OP2_MOD,
  NX_OP2_POW, NX_OP2_ATAN2, NX_OP2_MAXIMUM, NX_OP2_MINIMUM, NX_OP2_AND,
  NX_OP2_OR, NX_OP2_XOR, NX_OP2_THREEFRY, NX_OP2_EQUAL, NX_OP2_NOT_EQUAL,
  NX_OP2_LESS, NX_OP2_LESS_EQUAL, NX_OP2_COUNT
};

enum { NX_OP3_WHERE, NX_OP3_FMA, NX_OP3_COUNT };

typedef struct {
  int32_t tag, kind, dtype;
  int32_t a, b, c;
  uint8_t bits[16];
} nx_prog_node;

typedef struct {
  int32_t nins, nnodes, nouts, zero;
  nx_prog_node nodes[];
} nx_prog;

static inline const int32_t *nx_prog_ins(const nx_prog *p) {
  return (const int32_t *)(p->nodes + p->nnodes);
}

static inline const int32_t *nx_prog_outs(const nx_prog *p) {
  return nx_prog_ins(p) + p->nins;
}

/* Whether [p] has one operand and one output, the operand unchanged: its
   output node is In 0, whatever nodes come before it, as in both [In 0]
   and Prog.of_node's [In 0; In 0]. */
static inline int nx_prog_is_operand(const nx_prog *p) {
  if (p->nins != 1 || p->nouts != 1) return 0;
  const nx_prog_node *out = &p->nodes[nx_prog_outs(p)[0]];
  return out->tag == NX_NODE_IN && out->a == 0;
}

#endif
