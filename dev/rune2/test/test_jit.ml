(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.jit: a function and the same function under jit give the same bits,
   every NaN read as one; inside it reads and keyless draws raise, naming
   Rune.jit. *)

open Windtrap
open Compose
module Rune = Rune2

let invalid ~sub f = raises_match (Exn.invalid_arg ~substring:sub) f

type case =
  | Case : (float, 's) Nx.dtype * int array * operand * operand * expr -> case

let cases =
  let open Gen in
  (let* s = shape in
   let* e = expr ~rank:(Array.length s) in
   let* x = operand s and+ y = operand s in
   let+ (Nx.Dtype.Any dt) =
     of_list [ Nx.Dtype.Any Nx.float32; Any Nx.float64 ]
   in
   match Nx.Dtype.kind dt with
   | Float -> Case (dt, s, x, y, e)
   | _ -> assert false)
  |> with_pp (fun ppf (Case (dt, s, x, y, e)) ->
      Format.fprintf ppf "%a %a, x %a, y %a: %a" Nx.pp_dtype dt pp_shape s
        pp_view x.view pp_view y.view pp e)

let rec reduces = function
  | X | Y | K _ -> false
  | Un ((Sum _ | Max _ | Mean _ | Cumsum _), _) -> true
  | Un (_, e) -> reduces e
  | Bin (_, e, e') -> reduces e || reduces e'

let law (Case (dt, s, xo, yo, e)) =
  let x = make dt s xo and y = make dt s yo in
  cover "a zero-size shape" (numel s = 0);
  cover "a one-element shape" (numel s = 1 && Array.length s > 0);
  cover "a 0-d shape" (Array.length s = 0);
  cover "a view operand" (xo.view <> Plain || yo.view <> Plain);
  cover "a reduction or scan" (reduces e);
  cover "a composition of over 6 functions" (size e > 6);
  let f x y = eval x y e in
  let eager = outcome (fun () -> bits (f x y)) in
  let pair = Nx.Ptree.(tensor @-> tensor @-> returns tensor) in
  let jitted = outcome (fun () -> bits (Rune.jit pair f x y)) in
  cover "a refused operation" (match eager with Raised _ -> true | _ -> false);
  equal (outcome_t bits_t) eager jitted

(* Every dtype, through the functions every dtype has. *)
type any_case =
  | Any_case :
      ('v, 's) Nx.dtype * int array * operand * operand * expr
      -> any_case

let any_cases =
  let open Gen in
  (let* s = shape in
   let* e = expr ~rank:(Array.length s) in
   let* x = operand s and+ y = operand s in
   let+ (Nx.Dtype.Any dt) = of_list Nx.Dtype.all in
   Any_case (dt, s, x, y, generic e))
  |> with_pp (fun ppf (Any_case (dt, s, x, y, e)) ->
      Format.fprintf ppf "%a %a, x %a, y %a: %a" Nx.pp_dtype dt pp_shape s
        pp_view x.view pp_view y.view pp e)

let law_any (Any_case (dt, s, xo, yo, e)) =
  let x = make_any dt s xo and y = make_any dt s yo in
  let kind : type v. v Nx.Dtype.kind -> string = function
    | Float ->
        if Nx.Dtype.equal dt Nx.float64 || Nx.Dtype.equal dt Nx.float32 then
          "wide float"
        else "narrow float"
    | Complex -> "complex"
    | Signed | Unsigned -> "integer"
    | Boolean -> "boolean"
  in
  let k = kind (Nx.Dtype.kind dt) in
  List.iter
    (fun c -> cover ("a " ^ c ^ " dtype") (k = c))
    [ "narrow float"; "complex"; "integer"; "boolean" ];
  let f x y = eval_any x y e in
  let eager = outcome (fun () -> codes (f x y)) in
  let pair = Nx.Ptree.(tensor @-> tensor @-> returns tensor) in
  let jitted = outcome (fun () -> codes (Rune.jit pair f x y)) in
  equal (outcome_t codes_t) eager jitted

let f32 = Nx.float32
let v xs = Nx.create f32 [| Array.length xs |] xs

let test_read () =
  invalid ~sub:"Nx.item: the value is traced by Rune.jit" (fun () ->
      Rune.jit' (fun x -> Nx.scalar f32 (Nx.item [ 0 ] x)) (v [| 1. |]));
  (* A value that depends on no argument is staged too. *)
  let c = v [| 2. |] in
  invalid ~sub:"traced by Rune.jit" (fun () ->
      Rune.jit'
        (fun x -> Nx.add x (Nx.scalar f32 (Nx.item [ 0 ] (Nx.mul c c))))
        c)

let test_keyless () =
  invalid ~sub:"no key inside Rune.jit" (fun () ->
      Rune.jit'
        (fun x -> Nx.add x (Nx.Rng.uniform f32 [| 2 |]))
        (v [| 1.; 2. |]))

(* A key that depends on no argument draws the same values on every call. *)
let test_constant_key () =
  let f x = Nx.add x (Nx.Rng.normal ~key:(Nx.Rng.key 7) f32 [| 3 |]) in
  let x = v [| 0.; 1.; 2. |] in
  let a = bits (Rune.jit' f x) and b = bits (Rune.jit' f x) in
  equal bits_t (bits (f x)) a;
  equal bits_t a b

(* The function's code runs at each call, once. *)
let test_runs_once () =
  let n = ref 0 in
  let f x =
    incr n;
    Nx.sin x
  in
  ignore (Rune.jit' f (v [| 1. |]));
  equal int 1 !n

(* Arguments rebuild their structure; a result of several arrays. *)
let test_structures () =
  let s =
    Nx.Ptree.(
      pair tensor (list tensor) @-> tensor @-> returns (pair tensor tensor))
  in
  let f (a, bs) c = (Nx.add a c, List.fold_left Nx.mul c bs) in
  let a = v [| 1.; 2. |] and bs = [ v [| 3.; 4. |]; v [| -0.; 5. |] ] in
  let c = v [| 0.5; -1. |] in
  let p, q = Rune.jit s f (a, bs) c and p', q' = f (a, bs) c in
  equal bits_t (bits p') (bits p);
  equal bits_t (bits q') (bits q)

let () =
  exit
    (run "rune2 jit"
       [
         group "jit"
           [
             prop "jit gives eager's bits" cases law;
             prop "jit gives eager's bits at every dtype" any_cases law_any;
             test "a read raises naming Rune.jit" test_read;
             test "a keyless draw raises naming Rune.jit" test_keyless;
             test "a constant key draws the same each call" test_constant_key;
             test "the function runs once per call" test_runs_once;
             test "structures in and out" test_structures;
           ];
       ])
