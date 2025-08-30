open Rune

(* Standard Activations *)

let relu = relu
let relu6 = relu6
let sigmoid = sigmoid
let tanh = tanh
let softmax = softmax

(* Modern Activations *)

let gelu = gelu
let silu = silu
let swish = silu (* Alias for silu *)
let mish = mish

(* Parametric Activations *)

let leaky_relu = leaky_relu
let elu = elu
let selu = selu

let prelu alpha x =
  (* max(0, x) + alpha * min(0, x) *)
  let zero = zeros_like x in
  add (maximum zero x) (mul alpha (minimum zero x))

(* Gated Linear Units (GLUs) *)

let glu x gate =
  (* x * sigmoid(gate) *)
  mul x (sigmoid gate)

let swiglu x =
  (* x * silu(x) *)
  mul x (silu x)

let geglu x gate =
  (* x * gelu(gate) *)
  mul x (gelu gate)

let reglu x gate =
  (* x * relu(gate) *)
  mul x (relu gate)

(* Other Activations *)

let softplus = softplus
let softsign = softsign
let hard_sigmoid = hard_sigmoid
let hard_tanh = hard_tanh

let hard_swish x =
  (* x * relu6(x + 3) / 6 - Not in Rune, but hard_silu is *)
  hard_silu x (* hard_silu is essentially the same as hard_swish *)
