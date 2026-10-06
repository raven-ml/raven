(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Double-word numbers.

    A value of type ['b t] is a tensor of real numbers, each held as the sum
    [hi + lo] of two floats of dtype ['b], with [hi] that sum rounded to
    nearest. It carries 106 significand bits at float64 and 48 at float32.
    Bounds use [u], the dtype's unit roundoff, and hold where operands and
    results are zero or of magnitude between [2^-969] and [2^969] at float64
    ([2^-102] and [2^102] at float32), where every error term is a normal float.
    A number whose sum is infinite or NaN is held as that sum with a zero [lo];
    an operation on it gives the float result of the [hi] words with a zero
    [lo], and comparisons compare [hi], so NaN compares false.

    Operations broadcast as nx's arithmetic does. Each is a fixed program of
    nx's elementwise operations, with every sum and product rounded once as
    written. *)

type 'b t
(** The type for tensors of double-word numbers of float dtype ['b]. *)

(** {1:constructors Constructing} *)

val v : ?lo:(float, 'b) Nx.t -> (float, 'b) Nx.t -> 'b t
(** [v ~lo hi] is the number [hi + lo], exactly. [lo] defaults to zero. The two
    broadcast against each other. A pair already normalised is returned
    unchanged.

    Raises [Invalid_argument] unless the dtype is float32 or float64, or if the
    shapes do not broadcast. *)

val hi : 'b t -> (float, 'b) Nx.t
(** [hi w] is [w] rounded to nearest. *)

val lo : 'b t -> (float, 'b) Nx.t
(** [lo w] is [w - hi w], exactly. *)

(** {1:arithmetic Arithmetic} *)

val add : 'b t -> 'b t -> 'b t
(** [add a b] is [a + b] within [3u² / (1 - 4u)] relative. *)

val sub : 'b t -> 'b t -> 'b t
(** [sub a b] is [a - b] within [3u² / (1 - 4u)] relative, and exactly zero when
    [a = b]. *)

val mul : 'b t -> 'b t -> 'b t
(** [mul a b] is [a * b] within [4u²] relative, exactly when both have a zero
    [lo]. *)

val div : 'b t -> 'b t -> 'b t
(** [div a b] is [a / b] within [10u²] relative. *)

val floor : 'b t -> 'b t
(** [floor w] is the greatest integer at most [w], exactly. *)

(** {1:comparisons Comparisons} *)

val less : 'b t -> 'b t -> Nx.bool_t
(** [less a b] is [a < b], exactly. *)

val equal : 'b t -> 'b t -> Nx.bool_t
(** [equal a b] is [a = b], exactly. *)

(** {1:sums Sums} *)

val sum : ?axes:int list -> 'b t -> 'b t
(** [sum ?axes w] adds [w] along [axes] (default all, negative axes counting
    from the end) in a balanced tree of {!add}s, [⌈log₂ n⌉] deep for [n]
    summands, whose association depends only on the shape: within
    [((1 + 3u² / (1 - 4u))^⌈log₂ n⌉ - 1) · Σ|wᵢ|]. A sum of no summand is zero,
    and one that is not finite is the float sum of the high words.

    Raises [Invalid_argument] if an axis is out of bounds. *)

(** {1:structures Structures} *)

val ptree : (float, 'b) Nx.dtype -> 'b t Nx.Ptree.t
(** [ptree dtype] describes a value of [dtype] as two words at paths ["hi"] and
    ["lo"]: {!hi} and {!lo} for a value an operation or {!v} made, and the words
    it was rebuilt from for one a walk rebuilt. A walk that returns both words
    unchanged keeps the value. One that returns other words, as leafwise
    arithmetic does ({!Nx.Ptree.map2}, {!Nx.Ptree.axpy}), rebuilds the number
    their sum makes, which an operation, {!hi} and {!lo} read normalised through
    {!v}.

    Raises [Invalid_argument] unless [dtype] is float32 or float64. *)
