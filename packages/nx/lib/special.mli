(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Special functions: the implementation of {!Nx}'s, each a fixed program
    of nx's operations over the tables of {!Special_tables}.

    Every region of a function runs on every element, over its inputs
    clamped into the region by {!Frontend.where}, which then selects the
    region's result. So every intermediate is finite where its region is not
    selected, and the zero partial derivative that [where] gives it stays
    zero. *)

type 'a real := (float, 'a) Frontend.t

val erfinv : 'a real -> 'a real
val erfc : 'a real -> 'a real
val ndtr : 'a real -> 'a real
val log_ndtr : 'a real -> 'a real
val ndtri : 'a real -> 'a real

val ndtr_halves :
  'a real ->
  (bool, Nx_dtype.bool_elt) Frontend.t * 'a real * 'a real * 'a real
(** [ndtr_halves x] is [(positive, n, log_n, log_m)]: whether each element of
    [x] is positive, the smaller normal mass [n = ndtr (-|x|)], its logarithm,
    and the logarithm of the larger mass [1 - n], all from one {!ndtr}. [log_n]
    is [-infinity] where [x^2/2] overflows. *)

val ndtri_mass : 'a real -> 'a real -> 'a real
(** [ndtri_mass dc log_q] is the [x <= 0] with [ndtr x = q], for a normal
    mass [q] of at most [1/2] given twice: as [dc = q - 1/2] and as its finite
    logarithm [log_q]. Near the centre it solves on [dc], in the tail on
    [log_q], so that it keeps its accuracy however small [q] is. *)

val lgamma : 'a real -> 'a real
val digamma : 'a real -> 'a real
val lbeta : 'a real -> 'a real -> 'a real
val i0e : 'a real -> 'a real
val i1e : 'a real -> 'a real
val gammainc : 'a real -> 'a real -> 'a real
val gammaincc : 'a real -> 'a real -> 'a real
val log_gammainc : 'a real -> 'a real -> 'a real
val log_gammaincc : 'a real -> 'a real -> 'a real
val gammaincinv : 'a real -> 'a real -> 'a real
val gammainccinv : 'a real -> 'a real -> 'a real
val betainc : 'a real -> 'a real -> 'a real -> 'a real
val betaincc : 'a real -> 'a real -> 'a real -> 'a real
val log_betainc : 'a real -> 'a real -> 'a real -> 'a real
val log_betaincc : 'a real -> 'a real -> 'a real -> 'a real
