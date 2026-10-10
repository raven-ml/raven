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
enum {
  NX_SPEC_CONTRACT = 1,
  NX_SPEC_MAP = 2,
  NX_SPEC_REDUCE = 3,
  NX_SPEC_SCAN = 4,
  NX_SPEC_GATHER = 5,
  NX_SPEC_SCATTER = 6,
  NX_SPEC_SORT = 7,
  NX_SPEC_ASSEMBLE = 8,
  NX_SPEC_FOLD = 9
};

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

/* A reduction's kind, in the order of Spec.reduction's cases: the
   monoids, Moments, then Arg Max and Arg Min. */
enum { NX_SUM, NX_PROD, NX_MAX, NX_MIN, NX_LOGSUMEXP, NX_MOMENTS, NX_ARGMAX,
       NX_ARGMIN };

/* A reduction: its kind, the program output it reduces and its result's
   dtype code. */
typedef struct {
  int32_t kind, output, dtype;
} nx_spec_reduction;

/* Spec.map, Spec.reduce and Spec.scan: the program, at byte [at_prog] and
   [prog_len] bytes long; per load the byte offset of its padding, or 0 for
   a plain load; then the [naxes] axes reduced and the [nreductions]
   reductions, none for a map, one each for a scan. A padding holds its
   operand's rank and window count, the fill's bits, zero-padded, then int64
   [lo], [hi] and [interior] by axis and each window's axis, size, step and
   dilation. */
typedef struct {
  int32_t family; /* NX_SPEC_MAP, NX_SPEC_REDUCE or NX_SPEC_SCAN */
  int32_t nloads, naxes, nreductions;
  int32_t at_prog, prog_len;
  int32_t loads[];
} nx_spec_loop;

typedef struct {
  int32_t rank, nwindows;
  uint8_t fill[16];
  int64_t geometry[]; /* lo, hi, interior, then windows[nwindows][4] */
} nx_spec_pad;

static inline const nx_prog *nx_spec_loop_prog(const nx_spec_loop *m) {
  return (const nx_prog *)((const uint8_t *)m + m->at_prog);
}

/* Load [k]'s padding, or NULL for a plain load. */
static inline const nx_spec_pad *nx_spec_loop_pad(const nx_spec_loop *m,
                                                  int k) {
  if (m->loads[k] == 0) return 0;
  return (const nx_spec_pad *)((const uint8_t *)m + m->loads[k]);
}

static inline const int32_t *nx_spec_loop_axes(const nx_spec_loop *m) {
  return m->loads + m->nloads;
}

static inline const nx_spec_reduction *nx_spec_loop_reductions(
    const nx_spec_loop *m) {
  return (const nx_spec_reduction *)(nx_spec_loop_axes(m) + m->naxes);
}

/* A scatter's combine, in the order of Spec.combine's cases. */
enum { NX_SCATTER_SET, NX_SCATTER_ADD, NX_SCATTER_MAX, NX_SCATTER_MIN };

/* Spec.gather, Spec.scatter and Spec.sort: the axis; a scatter's combine
   and 1 where its targets are unique; a sort's direction, 1 descending, and
   the elements it keeps along its axis, -1 for all. */
typedef struct {
  int32_t family; /* NX_SPEC_GATHER, NX_SPEC_SCATTER or NX_SPEC_SORT */
  int32_t axis;
  int32_t combine; /* a sort's: descending */
  int32_t unique;
  int64_t k;
} nx_spec_axis;

/* Spec.assemble and Spec.fold: the result's rank and shape. An assembly
   holds its fill's [nfill] bytes and per piece a range per axis: start,
   count and step. A fold holds its padding at byte [at_pad], as a loop's
   load holds one; its fill is unused. */
typedef struct {
  int32_t family; /* NX_SPEC_ASSEMBLE or NX_SPEC_FOLD */
  int32_t rank, npieces, nfill, at_pad, zero;
  uint8_t fill[16];
  int64_t shape[]; /* then ranges[npieces][rank][3] */
} nx_spec_shaped;

/* Piece [j]'s range along axis [i]: its start, count and step. */
static inline const int64_t *nx_spec_shaped_range(const nx_spec_shaped *s,
                                                  int j, int i) {
  return s->shape + s->rank + 3 * ((int64_t)j * s->rank + i);
}

static inline const nx_spec_pad *nx_spec_shaped_pad(const nx_spec_shaped *s) {
  return (const nx_spec_pad *)((const uint8_t *)s + s->at_pad);
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
