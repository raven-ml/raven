(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Random compositions of nx functions of two arrays of one shape. *)

(** {1:exprs Expressions} *)

(** The type for unary functions. An axis function keeps its axis. *)
type un =
  | Neg
  | Abs
  | Recip
  | Sin
  | Exp
  | Sqrt
  | Tanh
  | Square
  | Log
  | Sigmoid
  | Cumsum of int
  | Flip of int
  | Sum of int
  | Max of int
  | Mean of int

(** The type for binary functions. [Where_less] is [where (x < y) x y]. *)
type bin = Add | Sub | Mul | Div | Maximum | Minimum | Where_less | Pow

(** The type for expressions. [K c] is [c] at the operands' dtype. *)
type t = X | Y | K of Operand.value | Un of un * t | Bin of bin * t * t

val pp : Format.formatter -> t -> unit
(** [pp] formats an expression as calls, [mul(x, sin(y))]. *)

val size : t -> int
(** [size e] is the number of nodes of [e]. *)

val gen : rank:int -> t Windtrap.Gen.t
(** [gen ~rank] is an expression of depth at most 5 over operands of [rank]
    axes. Its constants are [0], [-0], [0.5], [1], [2], [-3], infinity and the
    dtype's least and greatest finite values. *)

val reads_x : t -> bool
(** [reads_x e] is [true] iff [e] reads [X]. *)

val reads_y : t -> bool
(** [reads_y e] is [true] iff [e] reads [Y]. *)

val constant : t -> bool
(** [constant e] is [true] iff [e] reads neither [X] nor [Y]. *)

(** {1:eval Evaluation} *)

val eval :
  (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t -> t -> (float, 's, 'd) Nx.t
(** [eval x y e] is [e] at [x] and [y], by nx. *)

val generic : t -> t
(** [generic e] is [e] with the functions only floats have replaced by ones
    every dtype has: [sin], [exp], [sqrt], [log] by [abs]; [recip] by [neg];
    [tanh], [sigmoid] by [square]; [mean] by [sum]; [div], [pow] by [mul]. *)

val eval_any : ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t -> t -> ('v, 's, 'd) Nx.t
(** [eval_any x y e] is [eval x y e] at any dtype.

    Raises [Invalid_argument] if [e] is not {!generic}. *)

(** {1:subsets Subsets}

    A compiled call gives eager's bits for the functions both compute exactly:
    movements, extremes, comparisons, selections and IEEE 754's correctly
    rounded arithmetic. A law that compares bits draws from these subsets and
    says so in its name. *)

val exact : t -> t
(** [exact e] keeps [e]'s exact functions and replaces the others: [sin] and
    [log] by [neg]; [exp], [tanh] and [sigmoid] by [abs]; sums, cumulative sums
    and means by [max]; [pow] by [mul]. *)

val exact_at : ('v, 's) Nx.dtype -> t -> t
(** [exact_at dt e] keeps the {!generic} [e]'s functions exact at [dt] and
    replaces the others: at a complex dtype, [abs] and [square] by [neg], sums,
    cumulative sums and maxima by [flip], [mul] by [add], since these leave a
    NaN's parts unspecified; at a float dtype, sums and cumulative sums by
    [max]. *)

val unsummed : t -> t
(** [unsummed e] replaces each [max] by [flip], so that no derivative of [e]
    sums terms along an axis, as a reduction's transpose or its ties' share
    does. A derivative still sums the contributions of the operands [e] reads
    more than once. *)
