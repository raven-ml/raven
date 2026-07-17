(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Capture before [open Tolk_uop] shadows it with the uop movement module. *)
module Movement_ops = Movement
open Tolk_uop
module Movement = Movement_ops
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

(* A scalar literal used as an operand takes the dtype of the tensor it is
   paired with: a float tensor absorbs any scalar at its own dtype, an integer
   tensor absorbs an integer scalar at its own dtype, and every other pairing
   falls back to the scalar's own default dtype. This keeps mixed-precision
   arithmetic at the operand's precision (a half tensor plus [1.0] stays half)
   instead of widening through a default-typed constant. [~like] is the paired
   tensor. Only scalars paired with a tensor go through this; a scalar paired
   with another scalar keeps its default dtype. *)
let uf ~like s =
  let dt = T.dtype like in
  let sdt =
    if D.is_float dt then dt
    else
      match s with
      | T.Sint _ when D.is_int dt -> dt
      | T.Sint _ -> D.default_int
      | T.Sfloat _ -> D.default_float
      | T.Sbool _ -> D.bool
  in
  T.of_uop (Uop.const (T.scalar_const sdt s))

let sf ~like x = uf ~like (T.Sfloat x)
let si ~like n = uf ~like (T.Sint n)

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

(* Unary math. The transcendental ALU ops derive a float output dtype at the
   IR level (an integer or weak-integer operand promotes to its least-upper
   float), so no cast is needed here. *)

let reciprocal t = alu_unary Ops.Reciprocal t
let sqrt t = alu_unary Ops.Sqrt t
let rsqrt t = reciprocal (sqrt t)
let sin t = alu_unary Ops.Sin t
let exp2 t = alu_unary Ops.Exp2 t
let log2 t = alu_unary Ops.Log2 t
let trunc t = alu_unary Ops.Trunc t

let cos t =
  if Dtype_ops.is_floating_point t then
    let up = Dtype_ops.cast t (D.least_upper_dtype [ T.dtype t; D.float32 ]) in
    Dtype_ops.cast (sin (sub (sf ~like:up (Float.pi /. 2.)) up)) (T.dtype t)
  else sin (sub (sf ~like:t (Float.pi /. 2.)) t)

let exp t =
  if Dtype_ops.is_floating_point t then
    let up = Dtype_ops.cast t (D.least_upper_dtype [ T.dtype t; D.float32 ]) in
    Dtype_ops.cast (exp2 (mul up (sf ~like:up (1. /. Float.log 2.)))) (T.dtype t)
  else exp2 (mul t (sf ~like:t (1. /. Float.log 2.)))

let log t =
  let l = log2 t in
  mul l (sf ~like:l (Float.log 2.))

(* Rounding and division *)

let floor t =
  let bt = trunc t in
  where (lt t bt) (sub bt (si ~like:bt 1)) bt

let ceil t =
  let bt = trunc t in
  where (gt t bt) (add bt (si ~like:bt 1)) bt

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
  where (ne t (si ~like:t 0))
    (where (lt t (si ~like:t 0))
       (Creation.const_like t (T.Sint (-1)))
       (Creation.const_like t (T.Sint 1)))
    (Creation.const_like t (T.Sint 0))

let abs t = mul t (sign t)
let relu t = where (gt t (si ~like:t 0)) t (Creation.const_like t (T.Sint 0))

let sigmoid t =
  reciprocal
    (add (sf ~like:t 1.0) (exp2 (mul t (sf ~like:t (-1. /. Float.log 2.)))))

let tanh t =
  sub (mul (sf ~like:t 2.0) (sigmoid (mul (sf ~like:t 2.0) t))) (sf ~like:t 1.0)

(* Horner evaluation of a polynomial with float coefficients, high order
   first: [polyn x [c0; ...; cn]] is [c0*x^n + ... + cn]. *)
let polyn x coeffs =
  List.fold_left
    (fun acc c -> add (mul acc x) (sf ~like:x c))
    (sf ~like:x 0.0) coeffs

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
  let b = div (trunc t) (sf ~like:t 2.0) in
  where
    (eq (gt t (si ~like:t 0)) (eq (trunc b) b))
    (ceil (sub t (sf ~like:t 0.5)))
    (floor (add t (sf ~like:t 0.5)))

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
       (bitwise_or (lt b (si ~like:b 0)) (lt (reciprocal b) (si ~like:b 0)))
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
    (mul (eq t (sf ~like:t infinity)) (T.b detect_positive))
    (mul (eq t (sf ~like:t neg_infinity)) (T.b detect_negative))

let isfinite t = logical_not (bitwise_or (isinf t) (isnan t))

let isclose ?(rtol = 1e-05) ?(atol = 1e-08) ?(equal_nan = false) t other =
  let finite =
    bitwise_and
      (bitwise_and (isfinite t) (isfinite other))
      (le (abs (sub t other))
         (add (sf ~like:other atol) (mul (sf ~like:other rtol) (abs other))))
  in
  let infinite = bitwise_and (bitwise_or (isinf t) (isinf other)) (eq t other) in
  let nan = bitwise_and (bitwise_and (isnan t) (isnan other)) (T.b equal_nan) in
  bitwise_or (bitwise_or finite infinite) nan

(* Error function and other logarithms *)

let erf t =
  let s = reciprocal (add (sf ~like:t 1.0) (mul (sf ~like:t 0.3275911) (abs t))) in
  mul (sign t)
    (sub (sf ~like:t 1.0)
       (mul
          (mul s
             (polyn s
                [ 1.061405429; -1.453152027; 1.421413741; -0.284496736; 0.254829592 ]))
          (exp (neg (square t)))))

let log10 t =
  let l = log2 t in
  mul l (sf ~like:l (Float.log10 2.0))

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
       (sf ~like:t (Float.pi /. 2.))
       (mul (sqrt (sub (sf ~like:t 1.0) (abs t))) (polyn (abs t) coeffs)))

let acos t = sub (sf ~like:t (Float.pi /. 2.)) (asin t)
let atan t = asin (div t (sqrt (add (sf ~like:t 1.0) (mul t t))))

(* Hyperbolic functions and their inverses *)

let sinh t = div (sub (exp t) (exp (neg t))) (sf ~like:t 2.0)
let cosh t = div (add (exp t) (exp (neg t))) (sf ~like:t 2.0)

let atanh t =
  div (log (div (add (sf ~like:t 1.0) t) (sub (sf ~like:t 1.0) t))) (sf ~like:t 2.0)

let asinh t = log (add t (sqrt (add (square t) (sf ~like:t 1.0))))
let acosh t = log (add t (sqrt (sub (square t) (sf ~like:t 1.0))))

(* Activations *)

let relu6 t = sub (relu t) (relu (sub t (sf ~like:t 6.0)))

let hardswish t =
  mul (mul t (relu6 (add t (sf ~like:t 3.0)))) (sf ~like:t (1. /. 6.))

let hardsigmoid ?(alpha = 1. /. 6.) ?(beta = 0.5) t =
  sub
    (relu (add (mul (sf ~like:t alpha) t) (sf ~like:t beta)))
    (relu (sub (add (mul (sf ~like:t alpha) t) (sf ~like:t beta)) (sf ~like:t 1.0)))

let hardtanh ?(min_val = -1.0) ?(max_val = 1.0) t =
  clamp ~min:(sf ~like:t min_val) ~max:(sf ~like:t max_val) t

let leaky_relu ?(neg_slope = 0.01) t =
  where (lt t (si ~like:t 0)) (mul (sf ~like:t neg_slope) t) t

let quick_gelu t = mul t (sigmoid (mul t (sf ~like:t 1.702)))

let gelu t =
  mul (mul (sf ~like:t 0.5) t)
    (add (sf ~like:t 1.0)
       (tanh
          (mul
             (sf ~like:t (Float.sqrt (2. /. Float.pi)))
             (add t (mul (sf ~like:t 0.044715) (pow t (si ~like:t 3)))))))

let swish t = mul t (sigmoid t)
let silu = swish

let elu ?(alpha = 1.0) t =
  sub (relu t) (mul (sf ~like:t alpha) (relu (sub (sf ~like:t 1.0) (exp t))))

let celu ?(alpha = 1.0) t =
  add (maximum t (si ~like:t 0))
    (minimum
       (mul (sf ~like:t alpha)
          (sub (exp (div t (sf ~like:t alpha))) (sf ~like:t 1.0)))
       (si ~like:t 0))

let selu ?(alpha = 1.67326) ?(gamma = 1.0507) t =
  mul (sf ~like:t gamma)
    (where (ge t (si ~like:t 0)) t
       (mul (sf ~like:t alpha) (sub (exp t) (sf ~like:t 1.0))))

let softplus ?(beta = 1.0) t =
  mul (sf ~like:t (1. /. beta))
    (logaddexp (mul t (sf ~like:t beta)) (sf ~like:t 0.0))

let mish t = mul t (tanh (softplus t))
let logsigmoid t = neg (softplus (neg t))
let softsign t = div t (add (sf ~like:t 1.0) (abs t))
