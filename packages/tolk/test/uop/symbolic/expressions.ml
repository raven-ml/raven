(* Random integer expressions over bounded variables: index arithmetic as
   kernels compute it, weak or at a committed width, with divisions by constants
   and by variables whose bounds exclude 0; and random float32 expressions over
   scalar parameters and IEEE's special values. *)

open Windtrap
open Tolk

type t =
  | Var of int  (** The [i]th variable. *)
  | Divisor of int  (** The [i]th divisor variable. *)
  | Const of int
  | Neg of t
  | Add of t * t
  | Mul of t * int
  | Mul_var of t * t
  | Div of t * int
  | Mod of t * int
  | Div_var of t * int
  | Mod_var of t * int
  | Max of t * t
  | Select of t * t * t  (** [Select (a, c, b)] is [where (a < c) b a]. *)

(* An expression with the type and the bounds of its variables and divisors. *)
type scenario = {
  dtype : Dtype.t;
  vars : (int * int) list;
  divisors : (int * int) list;
  expr : t;
}

let rec pp ppf = function
  | Var i -> Format.fprintf ppf "%c" "abc".[i]
  | Divisor i -> Format.fprintf ppf "%c" "de".[i]
  | Const n -> Format.pp_print_int ppf n
  | Neg a -> Format.fprintf ppf "-%a" pp a
  | Add (a, b) -> Format.fprintf ppf "(%a + %a)" pp a pp b
  | Mul (a, c) -> Format.fprintf ppf "(%a * %d)" pp a c
  | Mul_var (a, b) -> Format.fprintf ppf "(%a * %a)" pp a pp b
  | Div (a, c) -> Format.fprintf ppf "(%a // %d)" pp a c
  | Mod (a, c) -> Format.fprintf ppf "(%a %% %d)" pp a c
  | Div_var (a, i) -> Format.fprintf ppf "(%a // %c)" pp a "de".[i]
  | Mod_var (a, i) -> Format.fprintf ppf "(%a %% %c)" pp a "de".[i]
  | Max (a, b) -> Format.fprintf ppf "max(%a, %a)" pp a pp b
  | Select (a, c, b) ->
      Format.fprintf ppf "where(%a < %a, %a, %a)" pp a pp c pp b pp a

let pp_scenario ppf s =
  let pp_bounds names ppf l =
    List.iteri
      (fun i (lo, hi) -> Format.fprintf ppf "%c in [%d, %d]; " names.[i] lo hi)
      l
  in
  Format.fprintf ppf "%a: %a%a%a" Dtype.pp s.dtype (pp_bounds "abc") s.vars
    (pp_bounds "de") s.divisors pp s.expr

(* Nodes *)

(* [build leaf s] is [s]'s expression, its variables and divisors made by [leaf
   name bounds]. *)
let build leaf s =
  let vars = List.mapi (fun i b -> leaf (String.make 1 "abc".[i]) b) s.vars in
  let divisors =
    List.mapi (fun i b -> leaf (String.make 1 "de".[i]) b) s.divisors
  in
  let rec node e =
    Ops.O.(
      match e with
      | Var i -> List.nth vars i
      | Divisor i -> List.nth divisors i
      | Const n -> int n
      | Neg a -> ~-(node a)
      | Add (a, b) -> node a + node b
      | Mul (a, c) -> node a * int c
      | Mul_var (a, b) -> node a * node b
      | Div (a, c) -> node a // int c
      | Mod (a, c) -> node a % int c
      | Div_var (a, i) -> node a // List.nth divisors i
      | Mod_var (a, i) -> node a % List.nth divisors i
      | Max (a, b) -> Ops.maximum (node a) (node b)
      | Select (a, c, b) -> Ops.where (node a < node c) (node b) (node a))
  in
  node s.expr

let node s = build (fun name (lo, hi) -> Common.var ~dtype:s.dtype name lo hi) s

(* [constants s] is [s]'s expression over committed constants of its type in
   place of its variables: a variable's least value, a divisor's greatest. *)
let constants s =
  build
    (fun name (lo, hi) ->
      Ops.int ~dtype:s.dtype (if name = "d" || name = "e" then hi else lo))
    s

(* Generators *)

let leaf =
  Gen.frequency
    [
      (7, Gen.map (fun i -> Var i) (Gen.int_range 0 2));
      ( 3,
        Gen.map
          (fun n -> Const n)
          (Gen.of_list [ 0; 1; -1; 2; 3; 4; 5; 7; 8; -3; 16 ]) );
    ]

let rec expr depth =
  if depth = 0 then leaf
  else
    let open Gen in
    let sub = expr (depth - 1) in
    let factor = of_list [ 0; 1; -1; 2; 3; 4; -2; 5 ] in
    let divisor = of_list [ 1; 2; 3; 4; 5; 8; -2; -3; 7; 12 ] in
    let which = int_range 0 1 in
    frequency
      [
        (2, leaf);
        (1, map (fun a -> Neg a) sub);
        ( 2,
          let+ a = sub and+ b = sub in
          Add (a, b) );
        ( 1,
          let+ a = sub and+ c = factor in
          Mul (a, c) );
        ( 1,
          let+ a = sub and+ b = sub in
          Mul_var (a, b) );
        ( 1,
          let+ a = sub and+ c = divisor in
          Div (a, c) );
        ( 1,
          let+ a = sub and+ c = divisor in
          Mod (a, c) );
        ( 1,
          let+ a = sub and+ i = which in
          Div_var (a, i) );
        ( 1,
          let+ a = sub and+ i = which in
          Mod_var (a, i) );
        ( 1,
          let+ a = sub and+ b = sub in
          Max (a, b) );
        ( 1,
          let+ a = sub and+ c = sub and+ b = sub in
          Select (a, c, b) );
      ]

let dtypes = Dtype.[ Weak_int; Int8; Uint8; Int16; Int32; Uint32 ]

(* Bounds within [dtype]'s: an unsigned variable is never negative. *)
let bounds dtype =
  let open Gen in
  let low = if Dtype.is_unsigned dtype then 0 else -10 in
  let+ lo = int_range low 10 and+ span = int_range 0 20 in
  (lo, lo + span)

(* A divisor's bounds exclude 0: division by 0 is undefined. *)
let divisor_bounds dtype =
  let open Gen in
  let+ lo = int_range 1 6 and+ span = int_range 0 6 and+ negative = bool in
  if negative && not (Dtype.is_unsigned dtype) then (-(lo + span), -lo)
  else (lo, lo + span)

(* [scenario_of dtype] draws an expression over variables of type [dtype]. *)
let scenario_of dtype =
  let open Gen in
  (let* dtype = dtype in
   let+ vars = list ~size:(constant 3) (bounds dtype)
   and+ divisors = list ~size:(constant 2) (divisor_bounds dtype)
   and+ expr = expr 4 in
   { dtype; vars; divisors; expr })
  |> with_pp pp_scenario

let scenario =
  scenario_of
    Gen.(frequency [ (3, constant Dtype.Weak_int); (2, of_list dtypes) ])

let weak_scenario = scenario_of (Gen.constant Dtype.Weak_int)
let committed_scenario = scenario_of (Gen.of_list (List.tl dtypes))

(* Floats *)

type float_expr =
  | Param of int  (** The scalar float32 parameter of slot [i]. *)
  | Special of int  (** The [i]th of {!Common.specials}, as a constant. *)
  | Fneg of float_expr
  | Fadd of float_expr * float_expr
  | Fsub of float_expr * float_expr
  | Fmul of float_expr * float_expr
  | Fdiv of float_expr * float_expr
  | Fmax of float_expr * float_expr
  | Fselect of float_expr * float_expr * float_expr
      (** [Fselect (a, c, b)] is [where (a < c) b a]. *)

let rec pp_float ppf = function
  | Param i -> Format.fprintf ppf "p%d" i
  | Special i -> Dtype.pp_const ppf (List.nth Common.specials i)
  | Fneg a -> Format.fprintf ppf "-%a" pp_float a
  | Fadd (a, b) -> Format.fprintf ppf "(%a + %a)" pp_float a pp_float b
  | Fsub (a, b) -> Format.fprintf ppf "(%a - %a)" pp_float a pp_float b
  | Fmul (a, b) -> Format.fprintf ppf "(%a * %a)" pp_float a pp_float b
  | Fdiv (a, b) -> Format.fprintf ppf "(%a / %a)" pp_float a pp_float b
  | Fmax (a, b) -> Format.fprintf ppf "max(%a, %a)" pp_float a pp_float b
  | Fselect (a, c, b) ->
      Format.fprintf ppf "where(%a < %a, %a, %a)" pp_float a pp_float c pp_float
        b pp_float a

let rec float_node e =
  Ops.O.(
    match e with
    | Param i -> Shape.param i Float32
    | Special i ->
        Ops.const ~dtype:Float32 (List.nth Common.specials i :> Dtype.const)
    | Fneg a -> ~-(float_node a)
    | Fadd (a, b) -> float_node a + float_node b
    | Fsub (a, b) -> float_node a - float_node b
    | Fmul (a, b) -> float_node a * float_node b
    | Fdiv (a, b) -> float_node a / float_node b
    | Fmax (a, b) -> Ops.maximum (float_node a) (float_node b)
    | Fselect (a, c, b) ->
        Ops.where (float_node a < float_node c) (float_node b) (float_node a))

let rec float_expr depth =
  let open Gen in
  let leaf =
    frequency
      [
        (3, map (fun i -> Param i) (int_range 0 2));
        ( 2,
          map
            (fun i -> Special i)
            (int_range 0 (List.length Common.specials - 1)) );
      ]
  in
  if depth = 0 then leaf
  else
    let sub = float_expr (depth - 1) in
    let binary f =
      let+ a = sub and+ b = sub in
      f a b
    in
    frequency
      [
        (2, leaf);
        (1, map (fun a -> Fneg a) sub);
        (2, binary (fun a b -> Fadd (a, b)));
        (1, binary (fun a b -> Fsub (a, b)));
        (2, binary (fun a b -> Fmul (a, b)));
        (1, binary (fun a b -> Fdiv (a, b)));
        (1, binary (fun a b -> Fmax (a, b)));
        ( 1,
          let+ a = sub and+ c = sub and+ b = sub in
          Fselect (a, c, b) );
      ]

let float_scenario = Gen.with_pp pp_float (float_expr 4)
