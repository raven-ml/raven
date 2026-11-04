(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Witness

type op = Add | Sub | Mul | Neg | Abs | Lshift of int | Div | Mod

type t =
  | Case : {
      dt : ('v, 's) Nx.dtype;
      op : op;
      xs : int64 array;
      ys : int64 array;
    }
      -> t

let name = function
  | Add -> "add"
  | Sub -> "sub"
  | Mul -> "mul"
  | Neg -> "neg"
  | Abs -> "abs"
  | Lshift _ -> "lshift"
  | Div -> "div"
  | Mod -> "mod"

let bits (Nx.Dtype.Any dt) = Nx.Dtype.bits dt

(* The operations of [d]: a shift by one, and by one short of its width, which
   moves the lowest bit into the sign. *)
let ops d =
  [ Add; Sub; Mul; Neg; Abs; Lshift 1; Lshift (bits d - 1); Div; Mod ]

let pp ppf (Case c) =
  let values ppf a =
    Format.pp_print_list ~pp_sep:Format.pp_print_space
      (fun ppf v -> Format.fprintf ppf "%Ld" v)
      ppf (Array.to_list a)
  in
  let op ppf = function
    | Lshift n -> Format.fprintf ppf "lshift %d" n
    | o -> Format.pp_print_string ppf (name o)
  in
  Format.fprintf ppf "%a %a of [%a] and [%a]" op c.op Nx.pp_dtype c.dt
    values c.xs values c.ys

let integers =
  List.filter
    (fun (Nx.Dtype.Any dt) ->
      match Nx.Dtype.kind dt with Signed | Unsigned -> true | _ -> false)
    Nx.Dtype.all

let signed (Nx.Dtype.Any dt) =
  match Nx.Dtype.kind dt with Signed -> true | _ -> false

(* A dtype's least and greatest values, as int64 taken modulo its width: the
   greatest uint64 is [-1L]. *)
let bounds (Nx.Dtype.Any dt) =
  let b = Nx.Dtype.bits dt in
  let power k = Int64.shift_left 1L k in
  if signed (Any dt) then
    (Int64.neg (power (b - 1)), Int64.pred (power (b - 1)))
  else (0L, if b = 64 then -1L else Int64.pred (power b))

(* The values whose sums, differences and products leave the range: the
   extremes, their neighbours, [-2] to [2], and either side of half the
   greatest, whose double is past it. *)
let edges d =
  let lo, hi = bounds d in
  let half = Int64.shift_right_logical hi 1 in
  [
    lo; Int64.succ lo; Int64.pred hi; hi; -2L; -1L; 0L; 1L; 2L; half;
    Int64.succ half;
  ]

let case (Nx.Dtype.Any dt) op xs ys = Case { dt; op; xs; ys }

let operand d =
  let open Gen in
  frequency [ (3, of_list (edges d)); (1, int64) ]

let pairs = 64

let gen =
  let open Gen in
  let* d = of_list integers in
  let* op =
    frequency
      [
        (5, of_list (ops d));
        (1, map (fun n -> Lshift n) (int_range 0 (bits d)));
      ]
  in
  let+ xs = array ~size:(constant pairs) (operand d)
  and+ ys = array ~size:(constant pairs) (operand d) in
  case d op xs ys

let examples =
  List.concat_map
    (fun d ->
      let e = edges d in
      let each f = Array.of_list (List.concat_map f e) in
      let xs = each (fun x -> List.map (fun _ -> x) e)
      and ys = each (fun _ -> e) in
      List.map (fun op -> case d op xs ys) (ops d))
    integers

(* [v] modulo [d]'s width as a float, near enough to tell a result past the
   range. *)
let value d v =
  let b = bits d in
  let f = Int64.to_float v in
  if b = 64 then if signed d || v >= 0L then f else f +. 0x1p64
  else
    let m = Int64.logand v (Int64.pred (Int64.shift_left 1L b)) in
    let m = Int64.to_float m in
    if signed d && m >= Float.ldexp 1. (b - 1) then m -. Float.ldexp 1. b
    else m

let exact op x y =
  match op with
  | Add -> x +. y
  | Sub -> x -. y
  | Mul -> x *. y
  | Neg -> -.x
  | Abs -> Float.abs x
  | Lshift n -> Float.ldexp x n
  | Div -> if y = 0. then 0. else Float.trunc (x /. y)
  | Mod -> if y = 0. then x else Float.rem x y

(* [op] and its comparisons, at every device brand. *)
let run (type v s) (module S : Nx.Devices) (dt : (v, s) Nx.dtype) op xs ys =
  let r : type d. (v, s, d) Nx.t -> (v, s, d) Nx.t -> (v, s, d) Nx.t =
   fun x y ->
    match op with
    | Add -> Nx.add x y
    | Sub -> Nx.sub x y
    | Mul -> Nx.mul x y
    | Neg -> Nx.neg x
    | Abs -> Nx.abs x
    | Lshift n -> Nx.lshift x n
    | Div -> Nx.div x y
    | Mod -> Nx.mod_ x y
  in
  let above : type d. (v, s, d) Nx.t -> (v, s, d) Nx.t -> d Nx.bool_t =
   fun x y -> Nx.greater (r x y) x
  and below : type d. (v, s, d) Nx.t -> (v, s, d) Nx.t -> d Nx.bool_t =
   fun x y -> Nx.less (r x y) (Nx.zeros_like x)
  in
  let n = Array.length xs in
  let x = Nx.cast dt (Nx.create Nx.int64 [| n |] xs)
  and y = Nx.cast dt (Nx.create Nx.int64 [| n |] ys) in
  let jit f =
    Rune.jit Nx.Ptree.(tensor @-> tensor @-> returns tensor) f
      (Nx.place S.on x) (Nx.place S.on y)
  in
  let check what eager compiled =
    equal ~msg:what (outcome_t codes_t)
      (outcome (fun () -> codes (eager x y)))
      (outcome (fun () -> codes (Nx.place Nx.Host.on (jit compiled))))
  in
  let before = Rune.evaluations () in
  check (name op) r r;
  check "greater than the first operand" above above;
  check "below zero" below below;
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before)

let law s (Case { dt; op; xs; ys }) =
  let d = Nx.Dtype.Any dt in
  List.iter (fun o -> cover (name o) (name op = name o)) (ops d);
  List.iter
    (fun (Nx.Dtype.Any e as a) ->
      if signed a then
        cover (Format.asprintf "%a" Nx.pp_dtype e) (Nx.Dtype.equal e dt))
    integers;
  let lo, hi = bounds d in
  let lo = value d lo and hi = value d hi in
  let past =
    Array.exists2
      (fun x y ->
        let r = exact op (value d x) (value d y) in
        r < lo || r > hi)
      xs ys
  in
  cover "a result past the range" past;
  let divides = match op with Div | Mod -> true | _ -> false in
  cover "a division by 0"
    (divides && Array.exists (fun y -> value d y = 0.) ys);
  run s dt op xs ys
