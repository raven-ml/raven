(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Double-word numbers.

    A value of type [('s, 'd) t] is an array of real numbers on the device set
    ['d], each held as the sum [hi + lo] of two floats of the format ['s], with
    [hi] that sum rounded to nearest. It carries 106 significand bits at float64
    and 48 at float32. Bounds use [u], the format's unit roundoff, and hold
    where operands and results are zero or of magnitude between [2^-969] and
    [2^969] at float64 ([2^-102] and [2^102] at float32), where every error term
    is a normal float. A number whose sum is infinite or NaN is held as that sum
    with a zero [lo]; an operation on it gives the float result of the [hi]
    words with a zero [lo], and comparisons compare [hi], so NaN compares false.

    Operations broadcast as nx's arithmetic does and compute where their
    operands lie. Each is a fixed program of nx's elementwise operations, with
    every sum and product rounded once as written. *)

type ('s, 'd) t
(** The type for arrays of double-word numbers of the float format ['s] on the
    set ['d]. *)

(** {1:constructors Constructing} *)

val v : ?lo:(float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t -> ('s, 'd) t
(** [v ~lo hi] is the number [hi + lo], exactly. [lo] defaults to zero. The two
    broadcast against each other. A pair already normalised is returned
    unchanged.

    Raises [Invalid_argument] unless the dtype is float32 or float64, or if the
    shapes do not broadcast. *)

val hi : ('s, 'd) t -> (float, 's, 'd) Nx.t
(** [hi w] is [w] rounded to nearest. *)

val lo : ('s, 'd) t -> (float, 's, 'd) Nx.t
(** [lo w] is [w - hi w], exactly. *)

(** {1:arithmetic Arithmetic} *)

val add : ('s, 'd) t -> ('s, 'd) t -> ('s, 'd) t
(** [add a b] is [a + b] within [3u² / (1 - 4u)] relative. *)

val sub : ('s, 'd) t -> ('s, 'd) t -> ('s, 'd) t
(** [sub a b] is [a - b] within [3u² / (1 - 4u)] relative, and exactly zero when
    [a = b]. *)

val mul : ('s, 'd) t -> ('s, 'd) t -> ('s, 'd) t
(** [mul a b] is [a * b] within [4u²] relative, exactly when both have a zero
    [lo]. *)

val div : ('s, 'd) t -> ('s, 'd) t -> ('s, 'd) t
(** [div a b] is [a / b] within [10u²] relative. *)

val floor : ('s, 'd) t -> ('s, 'd) t
(** [floor w] is the greatest integer at most [w], exactly. *)

(** {1:comparisons Comparisons} *)

val less : ('s, 'd) t -> ('s, 'd) t -> 'd Nx.bool_t
(** [less a b] is [a < b], exactly. *)

val equal : ('s, 'd) t -> ('s, 'd) t -> 'd Nx.bool_t
(** [equal a b] is [a = b], exactly. *)

(** {1:sums Sums} *)

val sum : ?axes:int list -> ('s, 'd) t -> ('s, 'd) t
(** [sum ~axes w] adds [w] along [axes] (default all, a negative axis counting
    from the end) in a balanced tree of {!add}s, [⌈log₂ n⌉] deep for [n]
    summands, whose association depends only on the shape: within
    [((1 + 3u² / (1 - 4u))^⌈log₂ n⌉ - 1) · Σ|wᵢ|]. It sums from [+0], so a sum
    of [-0] terms is [+0], as is a sum of no summand; one that is not finite is
    the float sum of the high words in the same tree.

    Raises [Invalid_argument] if an axis is outside [w]'s rank or repeated. *)

(** {1:structures Structures} *)

val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> ('s, 'd) t -> ('s, 'e) t
(** [walk c w] walks [w] as two fixed arrays ({!Nx.Ptree.Walk.tensor}) at [c]'s
    path then ["hi"] and ["lo"]: {!hi} and {!lo} for a value an operation or
    {!v} made, and the words it was rebuilt from for one a walk rebuilt. It
    rebuilds the number the words it returns sum to, which an operation, {!hi}
    and {!lo} read normalised through {!v}: words returned unchanged rebuild
    [w], and leafwise arithmetic ({!Nx.Ptree.map2}, {!Nx.Ptree.axpy}) rebuilds
    the number its words make. A structure holding a number walks it with
    [field c "x" Nx_wide.walk x]; [Nx.Ptree.v Nx_wide.walk] describes a number
    alone. *)
