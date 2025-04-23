open Internal
open Tensor

(* one-hot encoding: converts integer labels to one-hot vectors *)
let one_hot (type a b c d dev) (dtype : (a, b) dtype) (labels : (c, d, dev) t)
    depth : (a, b, dev) t =
  let input_shape = shape labels in
  let n = size labels in
  let labels_flat = reshape [| n |] labels in
  let oh_flat = empty_on_device (device labels) dtype [| n; depth |] in
  Dispatch.fill (Ndarray_core.zero dtype) oh_flat.data;
  let lbl_dtype = Dispatch.dtype labels.data in
  for i = 0 to n - 1 do
    let idx : int =
      match lbl_dtype with
      | Int8 -> get [| i |] labels_flat
      | UInt8 -> get [| i |] labels_flat
      | Int16 -> get [| i |] labels_flat
      | UInt16 -> get [| i |] labels_flat
      | Int32 -> Int32.to_int (get [| i |] labels_flat)
      | Int64 -> Int64.to_int (get [| i |] labels_flat)
      | _ -> failwith "one_hot: labels must have integer dtype"
    in
    let one : a = Ndarray_core.one dtype in
    set [| i; idx |] one oh_flat
  done;
  reshape (Array.append input_shape [| depth |]) oh_flat

(* Activation functions *)

(** Rectified Linear Unit: max(x, 0) *)
let relu x =
  let zero = scalar_like x 0.0 in
  maximum x zero

(** ReLU6: min(max(x, 0), 6) *)
let relu6 x =
  let zero = scalar_like x 0.0 in
  let six = scalar_like x 6.0 in
  let max_x = maximum x zero in
  minimum max_x six

(** Sigmoid: 1 / (1 + exp(-x)) *)
let sigmoid x =
  let one = scalar_like x 1. in
  let neg_x = neg x in
  let exp_neg_x = exp neg_x in
  let denom = add one exp_neg_x in
  div one denom

(** Hard Sigmoid: relu6(x + 3) / 6 *)
let hard_sigmoid x =
  let three = scalar_like x 3.0 in
  let six = scalar_like x 6.0 in
  let y = relu6 (add x three) in
  div y six

(** Softplus: log(1 + exp(x)) *)
let softplus x =
  let one = scalar_like x 1. in
  let exp_x = exp x in
  let sum = add one exp_x in
  log sum

(** SiLU (Swish): x * sigmoid(x) *)
let silu x =
  let sig_x = sigmoid x in
  mul x sig_x

(** Hard SiLU: x * hard_sigmoid(x) *)
let hard_silu x =
  let y = hard_sigmoid x in
  mul x y

(** Log-Sigmoid: log(sigmoid(x)) *)
let log_sigmoid x =
  let sig_x = sigmoid x in
  log sig_x

(** Leaky ReLU: max(x, negative_slope * x) *)
let leaky_relu ?(negative_slope = 0.01) x =
  let slope = scalar_like x negative_slope in
  let slope_x = mul slope x in
  maximum x slope_x

let tanh x =
  let exp_x = exp x in
  let exp_neg_x = exp (neg x) in
  let num = sub exp_x exp_neg_x in
  let den = add exp_x exp_neg_x in
  div num den

(** Hard Tanh: max(-1, min(1, x)) *)
let hard_tanh x =
  let one = scalar_like x 1. in
  let neg_one = scalar_like x (-1.0) in
  let min_x = minimum x one in
  maximum neg_one min_x

let elu ?(alpha = 1.0) x =
  let zero = scalar_like x 0.0 in
  let one = scalar_like x 1. in
  let alpha_scalar = scalar_like x alpha in
  let exp_x = exp x in
  let exp_minus_one = sub exp_x one in
  let min_part = minimum zero exp_minus_one in
  let alpha_min = mul alpha_scalar min_part in
  let max_x = maximum x zero in
  add max_x alpha_min

let selu x =
  let alpha = 1.6732632423543772848170429916717 in
  let lambda = 1.0507009873554804934193349852946 in
  let elu_x = elu ~alpha x in
  let lambda_scalar = scalar_like x lambda in
  mul lambda_scalar elu_x

let softmax ?(axes = [| -1 |]) x =
  let ndim = Array.length (shape x) in
  let axes = Array.map (fun ax -> if ax < 0 then ndim + ax else ax) axes in
  let max_x = max x ~axes ~keepdims:true in
  let x_shifted = sub x max_x in
  let exp_x = exp x_shifted in
  let sum_exp = sum exp_x ~axes ~keepdims:true in
  div exp_x sum_exp

(** Approximated Gaussian Error Linear Unit: 0.5 * x * (1 + tanh(x *
    0.7978845608 * (1 + 0.044715 * x * x))) *)
let gelu_approx x =
  let one = scalar_like x 1.0 in
  let half = scalar_like x 0.5 in
  let sqrt2_pi = scalar_like x 0.7978845608 in
  let coeff = scalar_like x 0.044715 in
  let x2 = mul x x in
  let inner = add one (mul coeff x2) in
  let arg = mul (mul x sqrt2_pi) inner in
  let y = tanh arg in
  mul half (mul x (add one y))

(** Soft-sign: x / (|x| + 1)*)
let softsign x =
  let one = scalar_like x 1.0 in
  let abs_x = maximum x (neg x) in
  div x (add one abs_x)

(** Mish: x * tanh(softplus(x)) *)
let mish x =
  let arg = softplus x in
  let y = tanh arg in
  mul x y
