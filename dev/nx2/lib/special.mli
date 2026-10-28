(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Special functions as expressions.

    Each function is the expression of the function {!Nx} names alike, over
    expressions of one dtype, which must be float32 or float64. Its contract,
    the domain, edges and bound, is {!Nx}'s. Its arguments are in {!Nx}'s order,
    parameters first and the variable last. The module also holds the constants
    its callers share with it ({!section-constants}).

    This module lies above [Special_tables], which holds its coefficients;
    {!Rng} and {!Nx} build programs from it. *)

type 'w e := (float, 'w) Nx_kernel.Prog.E.t

val erfc : 'w e -> 'w e
val erfinv : 'w e -> 'w e
val ndtr : 'w e -> 'w e
val log_ndtr : 'w e -> 'w e
val ndtri : 'w e -> 'w e
val lgamma : 'w e -> 'w e
val digamma : 'w e -> 'w e
val lbeta : 'w e -> 'w e -> 'w e
val gammainc : 'w e -> 'w e -> 'w e
val gammaincc : 'w e -> 'w e -> 'w e
val log_gammainc : 'w e -> 'w e -> 'w e
val log_gammaincc : 'w e -> 'w e -> 'w e
val gammaincinv : 'w e -> 'w e -> 'w e
val gammainccinv : 'w e -> 'w e -> 'w e
val betainc : 'w e -> 'w e -> 'w e -> 'w e
val betaincc : 'w e -> 'w e -> 'w e -> 'w e
val log_betainc : 'w e -> 'w e -> 'w e -> 'w e
val log_betaincc : 'w e -> 'w e -> 'w e -> 'w e
val i0e : 'w e -> 'w e
val i1e : 'w e -> 'w e

(** {1:parts Parts}

    Parts of the functions above, for samplers that invert the normal
    distribution from masses they hold as logarithms. *)

val ndtr_halves :
  'w e ->
  (bool, Nx_array.Dtype.bool_elt) Nx_kernel.Prog.E.t * 'w e * 'w e * 'w e
(** [ndtr_halves x] is whether [x] is positive, [ndtr (-|x|)], its logarithm,
    and [log (ndtr |x|)], each to the relative precision of {!ndtr} and
    {!log_ndtr}, an infinite [x]'s included. *)

val ndtri_mass : 'w e -> 'w e -> 'w e
(** [ndtri_mass d log_q] is [x] with [ndtr x = 1/2 + d], from [d] and the
    logarithm [log_q] of the nearer tail's mass [1/2 - |d|], finite, which keeps
    the tails' digits where that mass underflows. *)

(** {1:constants Constants}

    Each constant is the double nearest its value, written out, so that it has
    the same bits on every system. *)

val ln2 : float
(** [ln2] is the logarithm of 2. *)

val log_sqrt_2pi : float
(** [log_sqrt_2pi] is the logarithm of [sqrt (2 pi)]. *)
