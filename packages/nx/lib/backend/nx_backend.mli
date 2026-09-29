(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What computes on arrays.

    The kinds of nx's operations. Operations of one kind share their operands'
    and results' types, and nx routes and batches them by one rule; the kind
    names the mathematical function, after the [Nx] function it implements. *)

(** {1:kinds Kinds} *)

(** The type for elementwise functions of one operand, of its dtype. [Trunc],
    [Ceil], [Floor] and [Round] (half away from zero) are the identity on
    integers. *)
type unary =
  | Neg
  | Recip
  | Abs
  | Sqrt
  | Sign
  | Exp
  | Log
  | Sin
  | Cos
  | Tan
  | Asin
  | Acos
  | Atan
  | Sinh
  | Cosh
  | Tanh
  | Trunc
  | Ceil
  | Floor
  | Round
  | Erf

(** The type for elementwise functions of two operands of one shape and dtype,
    of that dtype. [Fdiv] divides floats and [Idiv] integers, truncating: the
    surface picks one by dtype. [And], [Or] and [Xor] are bitwise. *)
type binary =
  | Add
  | Sub
  | Mul
  | Fdiv
  | Idiv
  | Mod
  | Pow
  | Atan2
  | Maximum
  | Minimum
  | And
  | Or
  | Xor

(** The type for elementwise comparisons of two operands of one shape and dtype,
    to booleans. *)
type compare = Equal | Not_equal | Less | Less_equal

(** The type for associative reductions, over axes ([Nx.sum]) or as running
    values along one axis ([Nx.cumsum]). *)
type reduce = Sum | Prod | Max | Min

(** The type for the position of an extreme along one axis, as [int32]. The
    first of equal extremes is taken. *)
type arg_reduce = Argmax | Argmin

(** The type for dtype conversions: [Cast] converts values, [Bitcast]
    reinterprets the bits of elements of the same width. *)
type conversion = Cast | Bitcast
