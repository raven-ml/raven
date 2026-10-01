(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

type unary = Sin | Tanh | Exp_tanh | Neg | Log1p_sq
type binary = Add | Sub | Mul | Div_safe | Maximum

type t =
  | X
  | Const of float array
  | Un of unary * t
  | Bin of binary * t * t
  | Row_sum of t
  | Transposed of t
  | Matmul_const of t

let shape = [| 2; 3 |]

let matrix () =
  Nx.create Nx.float64 [| 3; 3 |]
    [| 0.5; -1.25; 0.75; 1.5; 0.25; -0.5; -0.75; 1.; 2. |]

let unary op x =
  match op with
  | Sin -> Nx.sin x
  | Tanh -> Nx.tanh x
  | Exp_tanh -> Nx.exp (Nx.tanh x)
  | Neg -> Nx.neg x
  | Log1p_sq -> Nx.log (Nx.add_s (Nx.mul x x) 1.)

let binary op a b =
  match op with
  | Add -> Nx.add a b
  | Sub -> Nx.sub a b
  | Mul -> Nx.mul a b
  | Div_safe -> Nx.div a (Nx.add_s (Nx.mul b b) 1.)
  | Maximum -> Nx.maximum a b

let rec eval p x =
  match p with
  | X -> x
  | Const c -> Nx.create Nx.float64 shape c
  | Un (op, p) -> unary op (eval p x)
  | Bin (op, p, q) -> binary op (eval p x) (eval q x)
  | Row_sum p ->
      Nx.broadcast_to shape (Nx.sum ~axes:[ 1 ] ~keepdims:true (eval p x))
  | Transposed p -> Nx.transpose (Nx.transpose (eval p x))
  | Matmul_const p -> Nx.matmul (eval p x) (matrix ())

let objective p x = Nx.sum (eval p x)

let unary_name = function
  | Sin -> "sin"
  | Tanh -> "tanh"
  | Exp_tanh -> "exp∘tanh"
  | Neg -> "neg"
  | Log1p_sq -> "log1p_sq"

let binary_name = function
  | Add -> "+"
  | Sub -> "-"
  | Mul -> "*"
  | Div_safe -> "/(1+_²)"
  | Maximum -> "max"

let rec pp ppf = function
  | X -> Format.pp_print_string ppf "x"
  | Const c ->
      Format.fprintf ppf "[%a]"
        (Format.pp_print_array
           ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
           Format.pp_print_float)
        c
  | Un (op, p) -> Format.fprintf ppf "%s(%a)" (unary_name op) pp p
  | Bin (op, p, q) -> Format.fprintf ppf "(%a %s %a)" pp p (binary_name op) pp q
  | Row_sum p -> Format.fprintf ppf "row_sum(%a)" pp p
  | Transposed p -> Format.fprintf ppf "transposed(%a)" pp p
  | Matmul_const p -> Format.fprintf ppf "(%a @ C)" pp p

let rec reads_x = function
  | X -> true
  | Const _ -> false
  | Un (_, p) | Row_sum p | Transposed p | Matmul_const p -> reads_x p
  | Bin (_, p, q) -> reads_x p || reads_x q

let element = Gen.float_range (-2.) 2.
let unary_gen = Gen.of_list [ Sin; Tanh; Exp_tanh; Neg; Log1p_sq ]

let binary_gen ~kinks =
  Gen.of_list
    (if kinks then [ Add; Sub; Mul; Div_safe; Maximum ]
     else [ Add; Sub; Mul; Div_safe ])

let map2' f a b = Gen.map (fun (a, b) -> f a b) (Gen.pair a b)
let map3' f a b c = Gen.map (fun (a, b, c) -> f a b c) (Gen.triple a b c)

let rec sized ~kinks depth =
  let open Gen in
  let leaf =
    frequency
      [
        (3, constant X);
        (1, map (fun c -> Const c) (array ~size:(constant 6) element));
      ]
  in
  if depth = 0 then leaf
  else
    let sub = sized ~kinks (depth - 1) in
    frequency
      [
        (2, leaf);
        (3, map2' (fun op p -> Un (op, p)) unary_gen sub);
        (4, map3' (fun op p q -> Bin (op, p, q)) (binary_gen ~kinks) sub sub);
        (1, map (fun p -> Row_sum p) sub);
        (1, map (fun p -> Transposed p) sub);
        (1, map (fun p -> Matmul_const p) sub);
      ]

let gen = Gen.with_pp pp (Gen.such_that reads_x (sized ~kinks:true 4))
let smooth = Gen.with_pp pp (Gen.such_that reads_x (sized ~kinks:false 4))
let pp_point ppf x = Nx.pp ppf x

let point =
  Gen.with_pp pp_point
    (Gen.map
       (fun a -> Nx.create Nx.float64 shape a)
       (Gen.array ~size:(Gen.constant 6) element))

let points n =
  Gen.with_pp pp_point
    (Gen.map
       (fun a -> Nx.create Nx.float64 (Array.append [| n |] shape) a)
       (Gen.array ~size:(Gen.constant (6 * n)) element))
