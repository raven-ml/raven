(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop
module D = Dtype
module T = Tensor

let alu_unary = T.alu_unary
let alu_binary = T.alu_binary
let alu_ternary = T.alu_ternary
let contiguous t = T.of_uop (Uop.contiguous ~src:(T.uop t) ())

(* Broadcasting and promotion are provided by [Tensor.broadcasted], whose
   concrete implementation lives in [Op] (mirroring tinygrad, where
   [_broadcasted] is abstract in the element-wise mixin and concrete in the
   composed-op mixin). *)

let binop op ?(reverse = false) a b =
  let x, y = T.broadcasted ~reverse a b in
  alu_binary op x y

(* Arithmetic *)

let add a b = binop Ops.Add a b
let mul a b = binop Ops.Mul a b

let logical_not t = binop Ops.Cmpne (Dtype_ops.bool t) (T.b true)

(* Negation multiplies by -1 at [t]'s own dtype (matching a weakly typed
   scalar): on an unsigned dtype the constant wraps and the product is the
   two's-complement negation, with no promotion to a wider type. *)
let neg t =
  if D.is_bool (T.dtype t) then logical_not t
  else
    mul t (T.of_uop (Uop.const (T.scalar_const (T.val_dtype t) (T.Sint (-1)))))

let sub a b =
  let x, y = T.broadcasted a b in
  add x (neg y)

(* Comparisons *)

let lt a b = binop Ops.Cmplt a b
let gt a b = binop ~reverse:true Ops.Cmplt a b
let ne a b = binop Ops.Cmpne a b
let ge a b = logical_not (lt a b)
let le a b = logical_not (gt a b)
let eq a b = logical_not (ne a b)

(* Bitwise *)

let bitwise_and a b = binop Ops.And a b
let bitwise_or a b = binop Ops.Or a b
let bitwise_xor a b = binop Ops.Xor a b

let bitwise_not t =
  if D.is_bool (T.dtype t) then logical_not t
  else bitwise_xor t (Creation.const_like t (T.Sint (-1)))

(* Selection *)

let where cond x y =
  let x, y = T.broadcasted x y in
  let out = T.broadcast_shape [ T.shape cond; T.shape x ] in
  let cond = Movement.broadcast_to (Dtype_ops.bool cond) out in
  alu_ternary Ops.Where cond (Movement.broadcast_to x out)
    (Movement.broadcast_to y out)

let maximum a b = binop Ops.Max a b
let masked_fill t mask value = where mask value t
let threefry t seed = alu_binary Ops.Threefry t seed
let inverse t = if Dtype_ops.is_floating_point t then neg t else bitwise_not t

(* Variadic folds. On a boolean first operand these fold with logical or / and;
   otherwise with sum / product. Used to combine index masks and terms. *)

let usum t ts =
  List.fold_left (if D.is_bool (T.dtype t) then bitwise_or else add) t ts

let uprod t ts =
  List.fold_left (if D.is_bool (T.dtype t) then bitwise_and else mul) t ts

let minimum a b =
  let x, y = T.broadcasted a b in
  inverse (maximum (inverse x) (inverse y))

(* Unary math *)

let ensure_float t =
  if Dtype_ops.is_floating_point t then t
  else Dtype_ops.cast t (D.least_upper_float (T.dtype t))

let reciprocal t = alu_unary Ops.Reciprocal (ensure_float t)
let sqrt t = alu_unary Ops.Sqrt (ensure_float t)
let rsqrt t = reciprocal (sqrt t)
let sin t = alu_unary Ops.Sin (ensure_float t)
let exp2 t = alu_unary Ops.Exp2 (ensure_float t)
let log2 t = alu_unary Ops.Log2 (ensure_float t)
let trunc t = alu_unary Ops.Trunc t

let cos t =
  if Dtype_ops.is_floating_point t then
    let up = Dtype_ops.cast t (D.least_upper_dtype [ T.dtype t; D.float32 ]) in
    Dtype_ops.cast (sin (sub (T.f (Float.pi /. 2.)) up)) (T.dtype t)
  else sin (sub (T.f (Float.pi /. 2.)) t)

let exp t =
  let c = T.f (1. /. Float.log 2.) in
  if Dtype_ops.is_floating_point t then
    let up = Dtype_ops.cast t (D.least_upper_dtype [ T.dtype t; D.float32 ]) in
    Dtype_ops.cast (exp2 (mul up c)) (T.dtype t)
  else exp2 (mul t c)

let log t = mul (log2 t) (T.f (Float.log 2.))

(* Rounding and division *)

let floor t =
  let bt = trunc t in
  where (lt t bt) (sub bt (T.i 1)) bt

let ceil t =
  let bt = trunc t in
  where (gt t bt) (add bt (T.i 1)) bt

let div a b =
  let x, y = T.broadcasted a b in
  mul x (reciprocal y)

let floordiv a b =
  let x, y = T.broadcasted a b in
  if D.is_int (T.dtype x) then alu_binary Ops.Floordiv x y else floor (div x y)

let mod_ a b =
  let x, y = T.broadcasted a b in
  if D.is_int (T.dtype x) then alu_binary Ops.Floormod x y
  else sub x (mul (floor (div x y)) y)

(* Shape-preserving nonlinearities *)

let square t = mul t t

let sign t =
  where (ne t (T.i 0))
    (where (lt t (T.i 0))
       (Creation.const_like t (T.Sint (-1)))
       (Creation.const_like t (T.Sint 1)))
    (Creation.const_like t (T.Sint 0))

let abs t = mul t (sign t)
let relu t = where (gt t (T.i 0)) t (Creation.const_like t (T.Sint 0))

let sigmoid t =
  reciprocal (add (T.f 1.0) (exp2 (mul t (T.f (-1. /. Float.log 2.)))))

let tanh t = sub (mul (T.f 2.0) (sigmoid (mul (T.f 2.0) t))) (T.f 1.0)

(* Horner evaluation of a polynomial with float coefficients, high order
   first: [polyn x [c0; ...; cn]] is [c0*x^n + ... + cn]. *)
let polyn x coeffs =
  List.fold_left (fun acc c -> add (mul acc x) (T.f c)) (T.f 0.0) coeffs

(* Shifts and truncating division *)

let lshift a b = binop Ops.Shl a b
let rshift a b = binop Ops.Shr a b

let cdiv a b =
  let x, y = T.broadcasted a b in
  if D.is_int (T.dtype x) then alu_binary Ops.Cdiv x y else trunc (mul x (reciprocal y))

let fmod a b =
  let x, y = T.broadcasted a b in
  if D.is_int (T.dtype x) then alu_binary Ops.Cmod x y else sub x (mul (cdiv x y) y)

(* Rounding to nearest, half to even *)

let round t =
  let b = div (trunc t) (T.f 2.0) in
  where
    (eq (gt t (T.i 0)) (eq (trunc b) b))
    (ceil (sub t (T.f 0.5)))
    (floor (add t (T.f 0.5)))

(* Power *)

let pow a b =
  let self_dt = T.dtype a in
  let base, exponent = T.broadcasted a b in
  let ret = alu_binary Ops.Pow base exponent in
  if (not (D.is_float self_dt)) && D.is_float (T.dtype exponent) then
    Dtype_ops.cast (round ret) self_dt
  else ret

(* Clamping *)

let clamp ?min ?max t =
  if min = None && max = None then
    invalid_arg "Elementwise.clamp: at least one bound must be given";
  let ret = match min with Some m -> where (lt t m) m t | None -> t in
  match max with Some m -> where (gt ret m) m ret | None -> ret

let clip = clamp

(* Sign transfer and log-space helpers *)

let copysign a b =
  let a, b = T.broadcasted a b in
  mul (abs a)
    (where
       (bitwise_or (lt b (T.i 0)) (lt (reciprocal b) (T.i 0)))
       (T.i (-1)) (T.i 1))

let logaddexp a b =
  let a, b = T.broadcasted a b in
  let m = maximum a b in
  add (log (add (exp (sub a m)) (exp (sub b m)))) m

let lerp t end_ weight = add t (mul (sub end_ t) weight)

(* Floating-point classification *)

let isnan t = ne t t

let isinf ?(detect_positive = true) ?(detect_negative = true) t =
  add
    (mul (eq t (T.f infinity)) (T.b detect_positive))
    (mul (eq t (T.f neg_infinity)) (T.b detect_negative))

let isfinite t = logical_not (bitwise_or (isinf t) (isnan t))

let isclose ?(rtol = 1e-05) ?(atol = 1e-08) ?(equal_nan = false) t other =
  let finite =
    bitwise_and
      (bitwise_and (isfinite t) (isfinite other))
      (le (abs (sub t other)) (add (T.f atol) (mul (T.f rtol) (abs other))))
  in
  let infinite = bitwise_and (bitwise_or (isinf t) (isinf other)) (eq t other) in
  let nan = bitwise_and (bitwise_and (isnan t) (isnan other)) (T.b equal_nan) in
  bitwise_or (bitwise_or finite infinite) nan

(* Error function and other logarithms *)

let erf t =
  let s = reciprocal (add (T.f 1.0) (mul (T.f 0.3275911) (abs t))) in
  mul (sign t)
    (sub (T.f 1.0)
       (mul
          (mul s
             (polyn s
                [ 1.061405429; -1.453152027; 1.421413741; -0.284496736; 0.254829592 ]))
          (exp (neg (square t)))))

let log10 t = mul (log2 t) (T.f (Float.log10 2.0))

(* Trigonometric inverses *)

let tan t = div (sin t) (cos t)

let asin t =
  let coeffs =
    [
      -0.0012624911; 0.0066700901; -0.0170881256; 0.0308918810; -0.0501743046;
      0.0889789874; -0.2145988016; 1.5707963050;
    ]
  in
  mul (sign t)
    (sub
       (T.f (Float.pi /. 2.))
       (mul (sqrt (sub (T.f 1.0) (abs t))) (polyn (abs t) coeffs)))

let acos t = sub (T.f (Float.pi /. 2.)) (asin t)
let atan t = asin (div t (sqrt (add (T.f 1.0) (mul t t))))

(* Hyperbolic functions and their inverses *)

let sinh t = div (sub (exp t) (exp (neg t))) (T.f 2.0)
let cosh t = div (add (exp t) (exp (neg t))) (T.f 2.0)
let atanh t = div (log (div (add (T.f 1.0) t) (sub (T.f 1.0) t))) (T.f 2.0)
let asinh t = log (add t (sqrt (add (square t) (T.f 1.0))))
let acosh t = log (add t (sqrt (sub (square t) (T.f 1.0))))

(* Activations *)

let relu6 t = sub (relu t) (relu (sub t (T.f 6.0)))
let hardswish t = mul (mul t (relu6 (add t (T.f 3.0)))) (T.f (1. /. 6.))

let hardsigmoid ?(alpha = 1. /. 6.) ?(beta = 0.5) t =
  sub
    (relu (add (mul (T.f alpha) t) (T.f beta)))
    (relu (sub (add (mul (T.f alpha) t) (T.f beta)) (T.f 1.0)))

let hardtanh ?(min_val = -1.0) ?(max_val = 1.0) t =
  clamp ~min:(T.f min_val) ~max:(T.f max_val) t

let leaky_relu ?(neg_slope = 0.01) t = where (lt t (T.i 0)) (mul (T.f neg_slope) t) t
let quick_gelu t = mul t (sigmoid (mul t (T.f 1.702)))

let gelu t =
  mul (mul (T.f 0.5) t)
    (add (T.f 1.0)
       (tanh
          (mul
             (T.f (Float.sqrt (2. /. Float.pi)))
             (add t (mul (T.f 0.044715) (pow t (T.i 3)))))))

let swish t = mul t (sigmoid t)
let silu = swish

let elu ?(alpha = 1.0) t =
  sub (relu t) (mul (T.f alpha) (relu (sub (T.f 1.0) (exp t))))

let celu ?(alpha = 1.0) t =
  add (maximum t (T.i 0))
    (minimum (mul (T.f alpha) (sub (exp (div t (T.f alpha))) (T.f 1.0))) (T.i 0))

let selu ?(alpha = 1.67326) ?(gamma = 1.0507) t =
  mul (T.f gamma) (where (ge t (T.i 0)) t (mul (T.f alpha) (sub (exp t) (T.f 1.0))))

let softplus ?(beta = 1.0) t =
  mul (T.f (1. /. beta)) (logaddexp (mul t (T.f beta)) (T.f 0.0))

let mish t = mul t (tanh (softplus t))
let logsigmoid t = neg (softplus (neg t))
let softsign t = div t (add (T.f 1.0) (abs t))
