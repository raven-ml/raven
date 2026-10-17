(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Special functions as expressions.

    Each value is the expression of the function {!Nx} names alike, over
    expressions of one dtype, which must be float32 or float64: another
    dtype's expression computes with float32's tables at its own precision.
    Its contract, the domain, edges and bound, is {!Nx}'s. Its arguments are
    in {!Nx}'s order, parameters first and the variable last. *)

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
