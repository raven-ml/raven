(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Every function lowers to Nx operations with reverse- and forward-mode rules
   in Rune (add, mul, div, exp, log, tanh, erf, maximum, where, reduce_max,
   reduce_sum), so they all differentiate. *)

(* A float constant in [x]'s element type, for the [_s] scalar operations. *)
let const x v = Nx_core.Dtype.of_float (Nx.dtype x) v
let relu = Nx.relu
let sigmoid = Nx.sigmoid
let tanh = Nx.tanh

let leaky_relu ?(negative_slope = 0.01) x =
  Nx.where
    (Nx.greater x (Nx.zeros_like x))
    x
    (Nx.mul_s x (const x negative_slope))

let inv_sqrt2 = 0.7071067811865476
let sqrt_2_over_pi = 0.7978845608028654

let gelu x =
  let gauss_cdf =
    Nx.add_s (Nx.erf (Nx.mul_s x (const x inv_sqrt2))) (const x 1.0)
  in
  Nx.mul_s (Nx.mul x gauss_cdf) (const x 0.5)

let gelu_approx x =
  let x3 = Nx.mul x (Nx.mul x x) in
  let inner =
    Nx.mul_s
      (Nx.add x (Nx.mul_s x3 (const x 0.044715)))
      (const x sqrt_2_over_pi)
  in
  Nx.mul_s (Nx.mul x (Nx.add_s (Nx.tanh inner) (const x 1.0))) (const x 0.5)

let silu x = Nx.mul x (Nx.sigmoid x)

(* softplus(x) = max(x, 0) + log(1 + exp(-|x|)): [exp] sees a non-positive
   argument on both sides of 0, so large inputs cannot overflow. *)
let softplus x =
  Nx.add (Nx.relu x)
    (Nx.log (Nx.add_s (Nx.exp (Nx.neg (Nx.abs x))) (const x 1.0)))

let softmax ?(axis = -1) x = Nx.softmax ~axes:[ axis ] x
let log_softmax ?(axis = -1) x = Nx.log_softmax ~axes:[ axis ] x
