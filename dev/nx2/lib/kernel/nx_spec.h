/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Descriptors (Nx_kernel.Spec) and contraction views as C kernels read
   them. A descriptor is its family's struct, int32 fields in the host's byte
   order, a trailing array as long as the counts before it say; a view is
   int64. Each is the start of an OCaml string or bytes, which the collector
   may move: a kernel copies what it reads into its own memory before
   anything that can run OCaml code, nx_read included. Free of OCaml's
   headers, so device code includes it. */

#ifndef NX_SPEC_H
#define NX_SPEC_H

#include <stdint.h>

/* A descriptor's family, its first field. */
enum { NX_SPEC_CONTRACT = 1 };

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

/* Spec.contract: the dtypes the sum runs in and its result has (nx_dtype.h's
   codes), whether an init operand is given, and the pairs of an axis of a and
   an axis of b: [nbatch] batch pairs, then [ncontracting] contracting
   pairs. */
typedef struct {
  int32_t family; /* NX_SPEC_CONTRACT */
  int32_t acc, out;
  int32_t init;
  int32_t nbatch, ncontracting;
  int32_t pairs[][2];
} nx_spec_contract;

/* Spec.Contract_view: a contraction's operands and result grouped into four
   axes, as Contract_view.fill leaves them, its arrays indexed by the
   operands and axes below. [offset] is each operand's first element and
   [stride] its stride along each axis, in elements; an axis an operand
   lacks, and an absent Init, read 0. Kernels write
   v.stride[NX_VIEW_A][NX_VIEW_CONTRACTED]. */
enum { NX_VIEW_A, NX_VIEW_B, NX_VIEW_INIT, NX_VIEW_DST };
enum { NX_VIEW_BATCH, NX_VIEW_ROW, NX_VIEW_COLUMN, NX_VIEW_CONTRACTED };

typedef struct {
  int64_t extent[4];    /* by axis */
  int64_t offset[4];    /* by operand */
  int64_t stride[4][4]; /* by operand, then axis */
} nx_contract_view;

#endif
