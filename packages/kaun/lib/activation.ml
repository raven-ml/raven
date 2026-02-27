(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Re-exports *)

let relu x = Nx.relu x
let sigmoid x = Nx.sigmoid x
let tanh x = Nx.tanh x

(* Activations *)

let relu6 x =
  let zero = Nx.scalar_like x 0.0 in
  let six = Nx.scalar_like x 6.0 in
  Nx.minimum (Nx.maximum x zero) six

let hard_sigmoid ?(alpha = 1.0 /. 6.0) ?(beta = 0.5) x =
  let linear =
    Nx.add (Nx.mul (Nx.scalar_like x alpha) x) (Nx.scalar_like x beta)
  in
  let zero = Nx.scalar_like x 0. in
  let one = Nx.scalar_like x 1. in
  Nx.minimum one (Nx.maximum zero linear)

let softplus x = Nx.log (Nx.add (Nx.scalar_like x 1.) (Nx.exp x))
let silu x = Nx.mul x (Nx.sigmoid x)
let swish x = silu x
let hard_silu x = Nx.mul x (hard_sigmoid x)
let hard_swish x = hard_silu x

let prelu ~alpha x =
  let zero = Nx.zeros_like x in
  Nx.add (Nx.maximum zero x) (Nx.mul alpha (Nx.minimum zero x))

let log_sigmoid x =
  (* Numerically stable: branch on sign to avoid overflow *)
  let zero = Nx.scalar_like x 0.0 in
  let one = Nx.scalar_like x 1.0 in
  let is_positive = Nx.greater x zero in
  let branch_pos = Nx.neg (Nx.log (Nx.add one (Nx.exp (Nx.neg x)))) in
  let branch_neg = Nx.sub x (Nx.log (Nx.add one (Nx.exp x))) in
  Nx.where is_positive branch_pos branch_neg

let leaky_relu ?(negative_slope = 0.01) x =
  Nx.maximum x (Nx.mul (Nx.scalar_like x negative_slope) x)

let hard_tanh x =
  let one = Nx.scalar_like x 1. in
  let neg_one = Nx.scalar_like x (-1.0) in
  Nx.maximum neg_one (Nx.minimum x one)

let elu ?(alpha = 1.0) x =
  let zero = Nx.scalar_like x 0.0 in
  let one = Nx.scalar_like x 1. in
  let alpha_s = Nx.scalar_like x alpha in
  let exp_minus_one = Nx.sub (Nx.exp x) one in
  Nx.add (Nx.maximum x zero) (Nx.mul alpha_s (Nx.minimum zero exp_minus_one))

let selu x =
  let alpha = 1.6732632423543772848170429916717 in
  let lambda = 1.0507009873554804934193349852946 in
  Nx.mul (Nx.scalar_like x lambda) (elu ~alpha x)

let celu ?(alpha = 1.0) x =
  let zero = Nx.zeros_like x in
  let alpha_s = Nx.scalar_like x alpha in
  let one = Nx.scalar_like x 1. in
  let neg_term =
    Nx.mul alpha_s (Nx.sub (Nx.exp (Nx.div (Nx.minimum zero x) alpha_s)) one)
  in
  Nx.add (Nx.maximum zero x) neg_term

let squareplus ?(b = 4.0) x =
  let half = Nx.scalar_like x 0.5 in
  let inside = Nx.add (Nx.square x) (Nx.scalar_like x b) in
  Nx.mul half (Nx.add x (Nx.sqrt inside))

let glu ?(axis = -1) x =
  match Nx.split ~axis 2 x with
  | [ left; right ] -> Nx.mul left (Nx.sigmoid right)
  | _ -> invalid_arg "Activation.glu: split did not produce two partitions"

let sparse_plus x =
  let zero = Nx.zeros_like x in
  let one = Nx.scalar_like x 1. in
  let neg_one = Nx.scalar_like x (-1.) in
  let quadratic = Nx.mul (Nx.scalar_like x 0.25) (Nx.square (Nx.add x one)) in
  let res = Nx.where (Nx.greater_equal x one) x quadratic in
  Nx.where (Nx.less_equal x neg_one) zero res

let sparse_sigmoid x =
  let zero = Nx.zeros_like x in
  let one = Nx.scalar_like x 1. in
  let neg_one = Nx.scalar_like x (-1.) in
  let half = Nx.scalar_like x 0.5 in
  let linear = Nx.mul half (Nx.add x one) in
  let res = Nx.where (Nx.greater_equal x one) one linear in
  Nx.where (Nx.less_equal x neg_one) zero res

let gelu_approx x =
  let one = Nx.scalar_like x 1.0 in
  let half = Nx.scalar_like x 0.5 in
  let sqrt2_pi = Nx.scalar_like x 0.7978845608 in
  let coeff = Nx.scalar_like x 0.044715 in
  let x2 = Nx.mul x x in
  let inner = Nx.add one (Nx.mul coeff x2) in
  let arg = Nx.mul (Nx.mul x sqrt2_pi) inner in
  Nx.mul half (Nx.mul x (Nx.add one (Nx.tanh arg)))

let gelu x =
  let half = Nx.scalar_like x 0.5 in
  let one = Nx.scalar_like x 1.0 in
  let sqrt2 = Nx.scalar_like x 1.4142135623730951 in
  Nx.mul (Nx.mul half x) (Nx.add one (Nx.erf (Nx.div x sqrt2)))

let softsign x =
  let one = Nx.scalar_like x 1.0 in
  Nx.div x (Nx.add one (Nx.abs x))

let mish x = Nx.mul x (Nx.tanh (softplus x))
