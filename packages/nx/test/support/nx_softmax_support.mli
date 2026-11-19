(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** {!Nx.softmax}'s definition and the bound its doc states, for nx's and
    rune's suites.

    The reference of an element is [exp (y - M) / Σ exp (y - M)] for
    [y = scale x] and the lane's greatest [M], with [y] and [y - M] exact as
    pairs of float64, the [exp] of a pair libm's of its high part, within an
    ulp, times one plus its low part, and the sum and the quotient carried
    as pairs. It is within a relative [2{^-51}] of the exact value: the
    numerator's [exp] and the sum's, a weighted mean of its terms', err by
    an ulp of float64 each. *)

type lane = {
  p : (float * float) array;
      (** Each element's reference, as a pair [hi + lo]. *)
  y : float array;  (** Each element's [scale x], rounded to float64. *)
  d : float array;  (** Each element's [y - M], rounded to float64. *)
  a : float;  (** The greatest finite [|y|]. *)
}
(** The type for what the bound reads of a lane. *)

val lane : scale:float -> float array -> lane option
(** [lane ~scale xs] is the lane [xs] by the definition, or [None] where
    {!Nx.softmax} makes it NaN: where it holds a NaN or its greatest [y] is
    infinite. *)

val beta : int -> int
(** [beta n] is [β(n)] of {!Nx_kernel.section-bounds}. *)

val bound :
  float -> exp_ulps:int -> b:int -> n:int -> scale:float -> lane -> int -> float
(** [bound u ~exp_ulps ~b ~n ~scale l j] is the relative bound at element
    [j] of [l], in a lane of [n] elements whose sum takes [b] roundings:
    [k u] for [k = b + |d| + ln n + 4E + 3], [E] being [exp_ulps], and
    [2 (|y| + A)] more where [|scale|] is not a power of two, as
    [k u / (1 - k u)], past every order of [k u] that [exp] makes of an
    error [k u] in its exponent; [infinity] where [k u >= 1]. *)

val check :
  float ->
  ?narrow:float ->
  normal:float ->
  exp_ulps:int ->
  b:int ->
  n:int ->
  scale:float ->
  lane ->
  at:(int -> int) ->
  float array ->
  unit
(** [check u ~narrow ~normal ~exp_ulps ~b ~n ~scale l ~at got] fails unless
    each element [j] of [l], [got.(at j)], is within {!bound} of its
    reference plus the reference's error, and a relative [narrow] more,
    with its product with the bound, for a dtype's rounding after the
    computation; [0.] by default. It checks the element only where the
    reference is at least [normal], and only that it lies in [[0, 1]] where
    the bound is [infinity]. An element whose [y] is [-inf] must be [+0.]. *)

val covers : normal:float -> lane -> unit
(** [covers ~normal l] covers a [-inf] element, an element past
    [|d| = 40] whose reference is at least [normal], and one whose reference
    is below it. *)
