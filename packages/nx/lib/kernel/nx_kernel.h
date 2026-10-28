/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Nx_kernel's values as C reads them: each constructor's tag and each
   field's position, in the order of their declarations. A constant
   constructor is its index among its type's constant constructors, read
   by Long_val; one with an argument is a block whose tag, read by Tag_val,
   is its index among those with one. A dtype is its code (nx_dtype.h).
   Free of OCaml's headers, so device code includes it; the macros expand
   where they are. */

#ifndef NX_KERNEL_H
#define NX_KERNEL_H

/* Programs (Nx_kernel.Prog) */

/* Prog.t's fields: the operands' dtypes, the nodes and the outputs, each an
   immutable array. */
enum { NX_PROG_INS, NX_PROG_NODES, NX_PROG_OUTS };

/* Prog.node: blocks, by tag. */
enum { NX_NODE_IN, NX_NODE_COORD, NX_NODE_CONST, NX_NODE_OP1, NX_NODE_OP2,
       NX_NODE_OP3 };

/* Each node's fields: In's operand, Coord's axis, Const's dtype and bits;
   Op1's kind, dtype and operand; Op2's and Op3's kind, then their operands
   in order from NX_OP2_OPERANDS and NX_OP3_OPERANDS. */
enum { NX_IN_OPERAND };
enum { NX_COORD_AXIS };
enum { NX_CONST_DTYPE, NX_CONST_BITS };
enum { NX_OP1_KIND, NX_OP1_DTYPE, NX_OP1_OPERAND };
enum { NX_OP2_KIND, NX_OP2_OPERANDS };
enum { NX_OP3_KIND, NX_OP3_OPERANDS };

/* The kinds, one code per arity, in the order of Prog's constructors:
   Copy, Cast and Bitcast, then the unary kinds; the binary kinds, then the
   comparisons; Where and Fma. */
enum {
  NX_OP1_COPY, NX_OP1_CAST, NX_OP1_BITCAST, NX_OP1_NEG, NX_OP1_RECIP,
  NX_OP1_ABS, NX_OP1_SIGN, NX_OP1_SQRT, NX_OP1_EXP, NX_OP1_EXP2, NX_OP1_LOG,
  NX_OP1_LOG2, NX_OP1_LOG1P, NX_OP1_EXPM1, NX_OP1_SIN, NX_OP1_COS,
  NX_OP1_TAN, NX_OP1_ASIN, NX_OP1_ACOS, NX_OP1_ATAN, NX_OP1_SINH,
  NX_OP1_COSH, NX_OP1_TANH, NX_OP1_ASINH, NX_OP1_ACOSH, NX_OP1_ATANH,
  NX_OP1_ERF, NX_OP1_FLOOR, NX_OP1_CEIL, NX_OP1_ROUND, NX_OP1_TRUNC,
  NX_OP1_COUNT
};

enum {
  NX_OP2_ADD, NX_OP2_SUB, NX_OP2_MUL, NX_OP2_FDIV, NX_OP2_IDIV, NX_OP2_MOD,
  NX_OP2_POW, NX_OP2_ATAN2, NX_OP2_MAXIMUM, NX_OP2_MINIMUM, NX_OP2_AND,
  NX_OP2_OR, NX_OP2_XOR, NX_OP2_THREEFRY, NX_OP2_EQUAL, NX_OP2_NOT_EQUAL,
  NX_OP2_LESS, NX_OP2_LESS_EQUAL, NX_OP2_COUNT
};

enum { NX_OP3_WHERE, NX_OP3_FMA, NX_OP3_COUNT };

/* Prog.op2: blocks of one field, by tag. */
enum { NX_OP2_BINARY, NX_OP2_COMPARE };

/* The code of [k], a Prog.op1 value: Copy, Cast and Bitcast are constants
   of their codes; Unary is a block whose field is its kind's index from
   NX_OP1_NEG. */
#define NX_OP1_CODE(k)                                                     \
  (Is_long(k) ? (int)Long_val(k) : NX_OP1_NEG + (int)Long_val(Field(k, 0)))

/* The code of [k], a Prog.op2 value: its field is its kind's index from
   NX_OP2_ADD for Binary, from NX_OP2_EQUAL for Compare. An op3 is a
   constant of its code. */
#define NX_OP2_CODE(k)                                                     \
  ((Tag_val(k) == NX_OP2_COMPARE ? NX_OP2_EQUAL : NX_OP2_ADD) +            \
   (int)Long_val(Field(k, 0)))

/* Loads, reductions, scatters and matrices */

/* Nx_kernel.load: blocks. */
enum { NX_LOAD_PLAIN, NX_LOAD_PADDED, NX_LOAD_COUNT };

/* Nx_kernel.monoid. */
enum {
  NX_MONOID_SUM, NX_MONOID_PROD, NX_MONOID_MAX, NX_MONOID_MIN,
  NX_MONOID_COUNT
};

/* Nx_kernel.reduction: Monoid and Arg are blocks, Logsumexp and Moments
   constants. */
enum { NX_REDUCTION_MONOID, NX_REDUCTION_ARG, NX_REDUCTION_BLOCKS };
enum { NX_REDUCTION_LOGSUMEXP, NX_REDUCTION_MOMENTS, NX_REDUCTION_CONSTANTS };

/* Nx_kernel.combine. */
enum {
  NX_COMBINE_SET, NX_COMBINE_ADD, NX_COMBINE_MAX, NX_COMBINE_MIN,
  NX_COMBINE_COUNT
};

/* Nx_kernel.triangle. */
enum { NX_TRIANGLE_LOWER, NX_TRIANGLE_UPPER, NX_TRIANGLE_COUNT };

#endif
