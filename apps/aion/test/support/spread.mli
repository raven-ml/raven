(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Parity against a float64 reference within its rounding ensemble's spread.

    The {e reference} is upstream's code run in float64. Its {e ensemble} is [n]
    more float64 runs, each moving every rounding operation's result by a normal
    draw at the operation's local rounding scale, and a value's {e spread} [ŝ]
    is the root mean square of the [n] runs' distances from the reference. To
    first order a computation of that rounding scale differs from the reference
    by a normal draw of the spread's variance, independent of the ensemble. Over
    [N] values, each within [τ ŝ] of the reference, with

    {v τ = √(n ((2N/α)^(2/n) - 1)) v}

    a correct computation fails with probability at most [α] (Hoeffding's
    inequality, the χ² moment generating function of the [n]-sample spread, and
    the union bound over the [N] values). [α] is [10{^-3}].

    A {e decision} compares a value with a boundary, as a token's bit compares
    the LFQ's input with [0]. A decision may differ from the reference's only
    where the reference's value lies within [τ ŝ] of the boundary, and then only
    to the outcome across that boundary. *)

val alpha : float
(** [alpha] is [10{^-3}], the probability with which a check of a correct
    computation may fail. *)

val tau : n:int -> int -> float
(** [tau ~n count] is [τ] for an ensemble of [n] runs and [count] values. *)

type check = {
  count : int;  (** The values checked. *)
  tau : float;  (** Their [τ]. *)
  worst : float;
      (** The largest distance from the reference over [τ ŝ]: [0] where every
          value equals the reference's, [infinity] where a value with [ŝ = 0]
          differs. *)
  failures : (int * float * float * float) list;
      (** Each value or decision that fails, as its index, the value, the
          reference's and [τ ŝ], in index order. *)
}
(** The type for the outcomes of checks. *)

val values :
  n:int -> reference:float array -> spread:float array -> float array -> check
(** [values ~n ~reference ~spread x] checks that each [x.(i)] lies within
    [τ spread.(i)] of [reference.(i)].

    Raises [Invalid_argument] if the arrays differ in length. *)

val signs :
  n:int -> reference:float array -> spread:float array -> float array -> check
(** [signs ~n ~reference ~spread x] checks each decision [x.(i) > 0] against
    [reference.(i) > 0], with [0] as the boundary.

    Raises [Invalid_argument] if the arrays differ in length. *)

val levels :
  n:int ->
  edges:float array ->
  reference:float array ->
  spread:float array ->
  int array ->
  check
(** [levels ~n ~edges ~reference ~spread k] checks each token [k.(i)] of a value
    whose reference is [reference.(i)], tokens counting the [edges] below it:
    [k.(i)] must be the reference's token, or the token next to it across an
    edge within [τ spread.(i)] of the reference. The check's value is the token,
    as a float.

    Raises [Invalid_argument] if the arrays differ in length. *)

val pp : string -> Format.formatter -> check -> unit
(** [pp name] prints a check's count, [τ], worst ratio and first failures. *)
