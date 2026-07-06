(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Pointwise derivative formulas shared by the reverse and forward engines.
   Reverse multiplies the cotangent by these; forward multiplies the tangent. *)

module T = Nx

let ln2 = 0.693147180559945309417
let two_over_sqrt_pi = 1.12837916709551257390

let float_scalar_like (type a b) (x : (a, b) T.t) (v : float) : (a, b) T.t =
  T.full (T.dtype x) [||] (Nx_core.Dtype.of_float (T.dtype x) v)

(* d/dx sqrt(x) = 1 / (2 * sqrt(x)), expressed with the primal output. *)
let sqrt' sqrt_x =
  T.div (T.ones_like sqrt_x) (T.mul (float_scalar_like sqrt_x 2.0) sqrt_x)

(* d/dx (1/x) = -1/x^2 *)
let recip' x = T.neg (T.recip (T.mul x x))

(* d/dx tan(x) = 1/cos^2(x) *)
let tan' x =
  let cos_x = T.cos x in
  T.recip (T.mul cos_x cos_x)

(* d/dx asin(x) = 1/sqrt(1 - x^2) *)
let asin' x =
  let one = T.ones_like x in
  T.recip (T.sqrt (T.sub one (T.mul x x)))

(* d/dx atan(x) = 1/(1 + x^2) *)
let atan' x =
  let one = T.ones_like x in
  T.recip (T.add one (T.mul x x))

(* d/dx erf(x) = (2/sqrt(pi)) * exp(-x^2) *)
let erf' x =
  let coeff = float_scalar_like x two_over_sqrt_pi in
  T.mul coeff (T.exp (T.neg (T.mul x x)))

(* d/dx tanh(x) = 1 - tanh(x)^2, expressed with the primal output. *)
let tanh' tanh_x = T.sub (T.ones_like tanh_x) (T.mul tanh_x tanh_x)

(* d/da (a^b) = b * a^(b-1) *)
let pow_wrt_base base exp = T.mul exp (T.pow base (T.sub exp (T.ones_like exp)))

(* d/db (a^b) = a^b * ln(a) = a^b * log2(a) * ln(2) *)
let pow_wrt_exp base result =
  let ln_base = T.mul (T.log2 base) (float_scalar_like base ln2) in
  T.mul result ln_base
