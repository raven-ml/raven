(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

type un =
  | Neg
  | Abs
  | Recip
  | Sin
  | Exp
  | Sqrt
  | Tanh
  | Square
  | Log
  | Sigmoid
  | Cumsum of int
  | Flip of int
  | Sum of int
  | Max of int
  | Mean of int

type bin = Add | Sub | Mul | Div | Maximum | Minimum | Where_less | Pow
type t = X | Y | K of Operand.value | Un of un * t | Bin of bin * t * t

let pp_un ppf = function
  | Neg -> Format.pp_print_string ppf "neg"
  | Abs -> Format.pp_print_string ppf "abs"
  | Recip -> Format.pp_print_string ppf "recip"
  | Sin -> Format.pp_print_string ppf "sin"
  | Exp -> Format.pp_print_string ppf "exp"
  | Sqrt -> Format.pp_print_string ppf "sqrt"
  | Tanh -> Format.pp_print_string ppf "tanh"
  | Square -> Format.pp_print_string ppf "square"
  | Log -> Format.pp_print_string ppf "log"
  | Sigmoid -> Format.pp_print_string ppf "sigmoid"
  | Cumsum a -> Format.fprintf ppf "cumsum~%d" a
  | Flip a -> Format.fprintf ppf "flip~%d" a
  | Sum a -> Format.fprintf ppf "sum~%d" a
  | Max a -> Format.fprintf ppf "max~%d" a
  | Mean a -> Format.fprintf ppf "mean~%d" a

let pp_bin ppf b =
  Format.pp_print_string ppf
    (match b with
    | Add -> "add"
    | Sub -> "sub"
    | Mul -> "mul"
    | Div -> "div"
    | Maximum -> "maximum"
    | Minimum -> "minimum"
    | Where_less -> "where_less"
    | Pow -> "pow")

let rec pp ppf = function
  | X -> Format.pp_print_string ppf "x"
  | Y -> Format.pp_print_string ppf "y"
  | K c -> Operand.pp_value ppf c
  | Un (u, e) -> Format.fprintf ppf "%a(%a)" pp_un u pp e
  | Bin (b, e, e') -> Format.fprintf ppf "%a(%a, %a)" pp_bin b pp e pp e'

let rec size = function
  | X | Y | K _ -> 1
  | Un (_, e) -> 1 + size e
  | Bin (_, e, e') -> 1 + size e + size e'

let constants =
  Operand.
    [
      F 0.;
      F (-0.);
      F 0.5;
      F 1.;
      F 2.;
      F (-3.);
      F Float.infinity;
      Least;
      Greatest;
    ]

let gen ~rank =
  let open Gen in
  let leaf =
    frequency
      [
        (3, constant X);
        (3, constant Y);
        (1, map (fun c -> K c) (of_list constants));
      ]
  in
  let axis = int_range 0 (Int.max 0 (rank - 1)) in
  let unary =
    let plain =
      [ Neg; Abs; Recip; Sin; Exp; Sqrt; Tanh; Square; Log; Sigmoid ]
    in
    if rank = 0 then of_list plain
    else
      frequency
        [
          (3, of_list plain);
          ( 2,
            let* a = axis in
            of_list [ Cumsum a; Flip a; Sum a; Max a; Mean a ] );
        ]
  in
  let binary =
    of_list [ Add; Sub; Mul; Div; Maximum; Minimum; Where_less; Pow ]
  in
  let rec go depth =
    if depth = 0 then leaf
    else
      frequency
        [
          (1, leaf);
          ( 2,
            let* u = unary in
            let+ e = go (depth - 1) in
            Un (u, e) );
          ( 3,
            let* b = binary in
            let* e = go (depth - 1) in
            let+ e' = go (depth - 1) in
            Bin (b, e, e') );
        ]
  in
  go 5

let rec reads_x = function
  | X -> true
  | Y | K _ -> false
  | Un (_, e) -> reads_x e
  | Bin (_, e, e') -> reads_x e || reads_x e'

let rec reads_y = function
  | Y -> true
  | X | K _ -> false
  | Un (_, e) -> reads_y e
  | Bin (_, e, e') -> reads_y e || reads_y e'

let constant e = not (reads_x e || reads_y e)

(* Evaluation *)

let un (type s d) u (x : (float, s, d) Nx.t) : (float, s, d) Nx.t =
  match u with
  | Neg -> Nx.neg x
  | Abs -> Nx.abs x
  | Recip -> Nx.recip x
  | Sin -> Nx.sin x
  | Exp -> Nx.exp x
  | Sqrt -> Nx.sqrt x
  | Tanh -> Nx.tanh x
  | Square -> Nx.square x
  | Log -> Nx.log x
  | Sigmoid -> Nx.sigmoid x
  | Cumsum a -> Nx.cumsum ~axis:a x
  | Flip a -> Nx.flip ~axes:[ a ] x
  | Sum a -> Nx.sum ~axes:[ a ] ~keepdims:true x
  | Max a -> Nx.max ~axes:[ a ] ~keepdims:true x
  | Mean a -> Nx.mean ~axes:[ a ] ~keepdims:true x

let bin (type v s d) b (x : (v, s, d) Nx.t) y : (v, s, d) Nx.t =
  match b with
  | Add -> Nx.add x y
  | Sub -> Nx.sub x y
  | Mul -> Nx.mul x y
  | Div -> Nx.div x y
  | Maximum -> Nx.maximum x y
  | Minimum -> Nx.minimum x y
  | Where_less -> Nx.where (Nx.less x y) x y
  | Pow -> Nx.pow x y

let scalar (type v s d) (x : (v, s, d) Nx.t) c =
  Nx.scalar_like x (Operand.value (Nx.dtype x) c)

let rec eval : type s d.
    (float, s, d) Nx.t -> (float, s, d) Nx.t -> t -> (float, s, d) Nx.t =
 fun x y -> function
  | X -> x
  | Y -> y
  | K c -> scalar x c
  | Un (u, e) -> un u (eval x y e)
  | Bin (b, e, e') -> bin b (eval x y e) (eval x y e')

let rec generic = function
  | (X | Y | K _) as e -> e
  | Un (u, e) ->
      let u =
        match u with
        | Sin | Exp | Sqrt | Log -> Abs
        | Recip -> Neg
        | Tanh | Sigmoid -> Square
        | Mean a -> Sum a
        | u -> u
      in
      Un (u, generic e)
  | Bin (b, e, e') ->
      let b = match b with Div | Pow -> Mul | b -> b in
      Bin (b, generic e, generic e')

let un_any (type v s d) u (x : (v, s, d) Nx.t) : (v, s, d) Nx.t =
  match u with
  | Neg -> Nx.neg x
  | Abs -> Nx.abs x
  | Square -> Nx.square x
  | Cumsum a -> Nx.cumsum ~axis:a x
  | Flip a -> Nx.flip ~axes:[ a ] x
  | Sum a -> Nx.sum ~axes:[ a ] ~keepdims:true x
  | Max a -> Nx.max ~axes:[ a ] ~keepdims:true x
  | Recip | Sin | Exp | Sqrt | Tanh | Log | Sigmoid | Mean _ ->
      invalid_arg "Expr.eval_any: a float function"

let rec eval_any : type v s d.
    (v, s, d) Nx.t -> (v, s, d) Nx.t -> t -> (v, s, d) Nx.t =
 fun x y -> function
  | X -> x
  | Y -> y
  | K c -> scalar x c
  | Un (u, e) -> un_any u (eval_any x y e)
  | Bin ((Div | Pow), _, _) -> invalid_arg "Expr.eval_any: a float function"
  | Bin (b, e, e') -> bin b (eval_any x y e) (eval_any x y e')

(* Subsets *)

let rec exact = function
  | (X | Y | K _) as e -> e
  | Un (u, e) ->
      let u =
        match u with
        | Sin | Log -> Neg
        | Exp | Tanh | Sigmoid -> Abs
        | Cumsum a | Sum a | Mean a -> Max a
        | (Neg | Abs | Recip | Sqrt | Square | Flip _ | Max _) as u -> u
      in
      Un (u, exact e)
  | Bin (b, e, e') ->
      let b = match b with Pow -> Mul | b -> b in
      Bin (b, exact e, exact e')

let rec exact_at : type v s. (v, s) Nx.dtype -> t -> t =
 fun dt e ->
  let complex = Nx.Dtype.is Complex dt and float = Nx.Dtype.is Float dt in
  match e with
  | (X | Y | K _) as e -> e
  | Un (u, e) ->
      let u =
        match u with
        | (Abs | Square) when complex -> Neg
        | (Sum a | Cumsum a | Max a) when complex -> Flip a
        | (Sum a | Cumsum a) when float -> Max a
        | u -> u
      in
      Un (u, exact_at dt e)
  | Bin (Mul, e, e') when complex -> Bin (Add, exact_at dt e, exact_at dt e')
  | Bin (b, e, e') -> Bin (b, exact_at dt e, exact_at dt e')

let rec unsummed = function
  | (X | Y | K _) as e -> e
  | Un (Max a, e) -> Un (Flip a, unsummed e)
  | Un (u, e) -> Un (u, unsummed e)
  | Bin (b, e, e') -> Bin (b, unsummed e, unsummed e')
