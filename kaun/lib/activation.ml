(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Re-exports *)

let relu x = Rune.relu x
let sigmoid x = Rune.sigmoid x
let tanh x = Rune.tanh x

(* Activations *)

let relu6 x =
  let zero = Rune.scalar_like x 0.0 in
  let six = Rune.scalar_like x 6.0 in
  Rune.minimum (Rune.maximum x zero) six

let hard_sigmoid ?(alpha = 1.0 /. 6.0) ?(beta = 0.5) x =
  let linear =
    Rune.add (Rune.mul (Rune.scalar_like x alpha) x) (Rune.scalar_like x beta)
  in
  let zero = Rune.scalar_like x 0. in
  let one = Rune.scalar_like x 1. in
  Rune.minimum one (Rune.maximum zero linear)

let softplus x = Rune.log (Rune.add (Rune.scalar_like x 1.) (Rune.exp x))
let silu x = Rune.mul x (Rune.sigmoid x)
let swish x = silu x
let hard_silu x = Rune.mul x (hard_sigmoid x)
let hard_swish x = hard_silu x

let prelu ~alpha x =
  let zero = Rune.zeros_like x in
  Rune.add (Rune.maximum zero x) (Rune.mul alpha (Rune.minimum zero x))

let log_sigmoid x =
  (* Numerically stable: branch on sign to avoid overflow *)
  let zero = Rune.scalar_like x 0.0 in
  let one = Rune.scalar_like x 1.0 in
  let is_positive = Rune.greater x zero in
  let branch_pos = Rune.neg (Rune.log (Rune.add one (Rune.exp (Rune.neg x)))) in
  let branch_neg = Rune.sub x (Rune.log (Rune.add one (Rune.exp x))) in
  Rune.where is_positive branch_pos branch_neg

let leaky_relu ?(negative_slope = 0.01) x =
  Rune.maximum x (Rune.mul (Rune.scalar_like x negative_slope) x)

let hard_tanh x =
  let one = Rune.scalar_like x 1. in
  let neg_one = Rune.scalar_like x (-1.0) in
  Rune.maximum neg_one (Rune.minimum x one)

let elu ?(alpha = 1.0) x =
  let zero = Rune.scalar_like x 0.0 in
  let one = Rune.scalar_like x 1. in
  let alpha_s = Rune.scalar_like x alpha in
  let exp_minus_one = Rune.sub (Rune.exp x) one in
  Rune.add (Rune.maximum x zero)
    (Rune.mul alpha_s (Rune.minimum zero exp_minus_one))

let selu x =
  let alpha = 1.6732632423543772848170429916717 in
  let lambda = 1.0507009873554804934193349852946 in
  Rune.mul (Rune.scalar_like x lambda) (elu ~alpha x)

let celu ?(alpha = 1.0) x =
  let zero = Rune.zeros_like x in
  let alpha_s = Rune.scalar_like x alpha in
  let one = Rune.scalar_like x 1. in
  let neg_term =
    Rune.mul alpha_s
      (Rune.sub (Rune.exp (Rune.div (Rune.minimum zero x) alpha_s)) one)
  in
  Rune.add (Rune.maximum zero x) neg_term

let squareplus ?(b = 4.0) x =
  let half = Rune.scalar_like x 0.5 in
  let inside = Rune.add (Rune.square x) (Rune.scalar_like x b) in
  Rune.mul half (Rune.add x (Rune.sqrt inside))

let glu ?(axis = -1) x =
  match Rune.split ~axis 2 x with
  | [ left; right ] -> Rune.mul left (Rune.sigmoid right)
  | _ -> invalid_arg "Activation.glu: split did not produce two partitions"

let sparse_plus x =
  let zero = Rune.zeros_like x in
  let one = Rune.scalar_like x 1. in
  let neg_one = Rune.scalar_like x (-1.) in
  let quadratic =
    Rune.mul (Rune.scalar_like x 0.25) (Rune.square (Rune.add x one))
  in
  let res = Rune.where (Rune.greater_equal x one) x quadratic in
  Rune.where (Rune.less_equal x neg_one) zero res

let sparse_sigmoid x =
  let zero = Rune.zeros_like x in
  let one = Rune.scalar_like x 1. in
  let neg_one = Rune.scalar_like x (-1.) in
  let half = Rune.scalar_like x 0.5 in
  let linear = Rune.mul half (Rune.add x one) in
  let res = Rune.where (Rune.greater_equal x one) one linear in
  Rune.where (Rune.less_equal x neg_one) zero res

let gelu_approx x =
  let one = Rune.scalar_like x 1.0 in
  let half = Rune.scalar_like x 0.5 in
  let sqrt2_pi = Rune.scalar_like x 0.7978845608 in
  let coeff = Rune.scalar_like x 0.044715 in
  let x2 = Rune.mul x x in
  let inner = Rune.add one (Rune.mul coeff x2) in
  let arg = Rune.mul (Rune.mul x sqrt2_pi) inner in
  Rune.mul half (Rune.mul x (Rune.add one (Rune.tanh arg)))

let gelu x =
  let half = Rune.scalar_like x 0.5 in
  let one = Rune.scalar_like x 1.0 in
  let sqrt2 = Rune.scalar_like x 1.4142135623730951 in
  Rune.mul (Rune.mul half x) (Rune.add one (Rune.erf (Rune.div x sqrt2)))

let softsign x =
  let one = Rune.scalar_like x 1.0 in
  Rune.div x (Rune.add one (Rune.abs x))

let mish x = Rune.mul x (Rune.tanh (softplus x))
