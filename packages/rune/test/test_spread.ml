(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* {!Nx.var} and {!Nx.std} over rune: a compiled call equals an eager one at
   both ends of the range, a captured constant compiles, and gradients agree
   with finite differences where the .mli promises them. *)

open Windtrap

let f64 l = Nx.create Nx.float64 [| Array.length l |] l
let f32 l = Nx.create Nx.float32 [| Array.length l |] l

let bits a = Int64.bits_of_float a

let number v = Nx.item [] (Nx.cast Nx.float64 v)

(* Lanes *)

type lane =
  | D64 of string * float array * int
  | D32 of string * float array * int

let name = function D64 (n, _, _) | D32 (n, _, _) -> n

let lanes =
  [
    D64 ("ordinary", [| 1.; 2.; 4.; -3. |], 1);
    D64 ("ordinary ddof 0", [| 0.1; 7.5; -2.25 |], 0);
    D64 ("[1e154; -1e154; 0]", [| 1e154; -1e154; 0. |], 0);
    D64 ("[1e200; -1e200]", [| 1e200; -1e200 |], 0);
    D64 ("[1e200; -1e200] ddof 1", [| 1e200; -1e200 |], 1);
    D64 ("two at the top", [| 1e308; 1e308 |], 0);
    D64 ("greatest, constant", [| max_float; max_float; max_float |], 0);
    D64 ("alternating 1e308", Array.init 64 (fun i -> if i mod 2 = 0 then 1e308 else -1e308), 0);
    D64 ("[1e-170; -1e-170]", [| 1e-170; -1e-170 |], 0);
    D64 ("[1e-170; -1e-170] ddof 1", [| 1e-170; -1e-170 |], 1);
    D64 ("least subnormal", [| 0x1p-1074; -0x1p-1074 |], 0);
    D64 ("near 2^540", Array.init 4 (fun k -> Float.ldexp (1. +. Float.ldexp (Float.of_int k) (-41)) 540), 0);
    D64 ("near-constant, low", [| 1e-300; 1e-300 *. (1. +. 1e-10) |], 0);
    D64 ("mixed magnitudes", [| 1e300; -1e-300; 2.5; -1e150 |], 1);
    D64 ("a NaN term", [| Float.nan; 1e200 |], 0);
    D64 ("an infinity", [| Float.infinity; 1. |], 0);
    D64 ("-0.", [| -0.; -0. |], 0);
    D64 ("one element", [| 1e300 |], 0);
    D32 ("float32 [1.5e19; -1.5e19; 0]", [| 1.5e19; -1.5e19; 0. |], 0);
    D32 ("float32 [1e25; -1e25]", [| 1e25; -1e25 |], 0);
    D32 ("float32 [1e-25; -1e-25]", [| 1e-25; -1e-25 |], 1);
    D32 ("float32 [1e37; 1e37(1+2^-20)]", [| 1e37; 1e37 *. (1. +. 0x1p-20) |], 0);
    D32 ("float32 ordinary", [| 0.5; -2.; 3.5 |], 1);
  ]

let u_of = function D64 _ -> epsilon_float /. 2. | D32 _ -> 0x1p-24

(* A compiled result equals an eager one, or within a compiled sum's
   association; both are NaN, or both the same infinity or zero, alike. *)
let close ~msg ~rel a b =
  let special v = Float.is_nan v || not (Float.is_finite v) || v = 0. in
  if special a || special b then
    equal ~msg int64 (if Float.is_nan a then 0L else bits a) (if Float.is_nan b then 0L else bits b)
  else if Float.abs (a -. b) > rel *. Float.abs a then
    failf "%s: compiled %h, eager %h" msg b a

let check_jit l =
  let run (type s) (x : (float, s, Nx.host) Nx.t) ddof =
    let u = u_of l in
    let var y = Nx.var ~ddof y and std y = Nx.std ~ddof y in
    close ~msg:(name l ^ " var") ~rel:(256. *. u) (number (var x))
      (number (Rune.jit' var x));
    close ~msg:(name l ^ " std") ~rel:(256. *. u) (number (std x))
      (number (Rune.jit' std x))
  in
  match l with D64 (_, a, ddof) -> run (f64 a) ddof | D32 (_, a, ddof) -> run (f32 a) ddof

let test_lanes () =
  let x =
    Nx.create Nx.float64 [| 7; 2 |]
      [|
        1e200; -1e200; 1.; 2.; 0.; 0.; 0x1p-1074; -0x1p-1074; Float.nan; 1.;
        Float.infinity; 1.; 1e-200; -1e-200;
      |]
  in
  let eager = Nx.to_array (Nx.std ~axes:[ 1 ] x) in
  let compiled = Nx.to_array (Rune.jit' (Nx.std ~axes:[ 1 ]) x) in
  Array.iteri
    (fun i a ->
      close ~msg:(Printf.sprintf "lane %d" i) ~rel:1e-13 a compiled.(i))
    eager;
  equal ~msg:"the first lane" (float 1e186) 1e200 compiled.(0)

let test_captured () =
  (* A compiled body over a captured constant compiles. *)
  let c = f64 [| 1e200; -1e200 |] in
  let y = Rune.jit' (fun y -> Nx.add (Nx.std c) y) (Nx.scalar Nx.float64 1.) in
  let want = Nx.item [] (Nx.std c) in
  close ~msg:"captured std" ~rel:1e-14 want (number y);
  let v = Rune.jit' (fun y -> Nx.add (Nx.var (f64 [| 1.; 3. |])) y) (Nx.scalar Nx.float64 1.) in
  equal ~msg:"captured var" float_exact 2. (number v)

let test_empty () =
  let x = Nx.zeros Nx.float32 [| 0 |] in
  let nan v = equal ~msg:"NaN" bool true (Float.is_nan (number v)) in
  nan (Rune.jit' Nx.var x);
  nan (Rune.jit' Nx.std x);
  let wide = Nx.zeros Nx.float64 [| 3; 0 |] in
  let y = Rune.jit' (Nx.std ~axes:[ 1 ]) wide in
  equal ~msg:"shape" (array int) [| 3 |] (Nx.shape y);
  equal ~msg:"NaN lanes" bool true
    (Array.for_all Float.is_nan (Nx.to_array (Nx.cast Nx.float64 y)))

(* Gradients *)

(* The elements scaled by [2^-e], their greatest magnitude near 1. *)
let exponent_of x =
  let s = Array.fold_left (fun a v -> Float.max a (Float.abs v)) 0. x in
  snd (Float.frexp s)

let var_grad_exact x =
  let n = Float.of_int (Array.length x) in
  let e = exponent_of x in
  let z = Array.map (fun v -> Float.ldexp v (-e)) x in
  let m = Array.fold_left ( +. ) 0. z /. n in
  Array.map (fun z -> Float.ldexp (2. *. (z -. m) /. n) e) z

let std_grad_exact x ddof =
  let n = Array.length x in
  let e = exponent_of x in
  let z = Array.map (fun v -> Float.ldexp v (-e)) x in
  let nf = Float.of_int n in
  let m = Array.fold_left ( +. ) 0. z /. nf in
  let s2 = Array.fold_left (fun a v -> a +. ((v -. m) *. (v -. m))) 0. z in
  let sd = sqrt (s2 /. Float.of_int (n - ddof)) in
  Array.map (fun v -> (v -. m) /. (Float.of_int (n - ddof) *. sd)) z

(* Each element within [rel] of the largest expected magnitude. *)
let closeness ~msg ~rel want got =
  let top = Array.fold_left (fun a w -> Float.max a (Float.abs w)) 0. want in
  Array.iteri
    (fun i w ->
      let g = got.(i) in
      if Float.abs (g -. w) > rel *. top then
        failf "%s[%d]: %h, expected %h" msg i g w)
    want

let var_grad_cases =
  [
    ("ordinary", [| 1.; 2.; 4.; -3. |]);
    ("[1e154; -1e154; 0]", [| 1e154; -1e154; 0. |]);
    ("near 2^540", Array.init 4 (fun k -> Float.ldexp (1. +. Float.ldexp (Float.of_int k) (-41)) 540));
    ("[1e-170; -1e-170]", [| 1e-170; -1e-170 |]);
  ]

let check_var_grad (_, a) =
  let x = f64 a in
  let g = Nx.to_array (Rune.grad' Nx.var x) in
  closeness ~msg:"grad var" ~rel:1e-13 (var_grad_exact a) g;
  (* Central differences: exact on a quadratic, at a step within reach of
     the deviations and their squares. *)
  let n = Array.length a in
  let e = exponent_of a in
  let sd = number (Nx.std (f64 (Array.map (fun v -> Float.ldexp v (-e)) a))) in
  (* A power of two near the deviations, so each perturbed element is exact. *)
  let h = Float.ldexp 1. (e + snd (Float.frexp sd) - 1) in
  let perturbed i s =
    Array.mapi (fun j v -> if j = i then v +. (s *. h) else v) a
  in
  let f l = number (Nx.var (f64 l)) in
  let fd =
    Array.init n (fun i ->
        let p = f (perturbed i 1.) and q = f (perturbed i (-1.)) in
        (p -. q) /. (2. *. h))
  in
  (* A variance rounding to zero below the range has no differences to read;
     its gradient was checked against the formula. *)
  if f a > 0. && Array.for_all Float.is_finite fd then
    closeness ~msg:"finite differences" ~rel:1e-5 fd g

let std_grad_cases =
  [
    ("ordinary", [| 1.; 2.; 4.; -3. |], 1);
    ("[1e200; -1e200]", [| 1e200; -1e200 |], 0);
    ("[1e154; -1e154; 0]", [| 1e154; -1e154; 0. |], 0);
    ("[1e-170; -1e-170]", [| 1e-170; -1e-170 |], 0);
    ("[1e-170; -1e-170] ddof 1", [| 1e-170; -1e-170 |], 1);
    ("mixed", [| 1e200; 1e199; -3e199; 1e198 |], 1);
  ]

let check_std_grad (_, a, ddof) =
  let x = f64 a in
  let g = Nx.to_array (Rune.grad' (Nx.std ~ddof) x) in
  let want = std_grad_exact a ddof in
  closeness ~msg:"grad std" ~rel:1e-13 want g;
  let e = exponent_of a in
  let sd =
    let z = Array.map (fun v -> Float.ldexp v (-e)) a in
    number (Nx.std ~ddof (f64 z))
  in
  let h = Float.ldexp (1e-6 *. sd) e in
  let n = Array.length a in
  let f l = number (Nx.std ~ddof (f64 l)) in
  let perturbed i s = Array.mapi (fun j v -> if j = i then v +. (s *. h) else v) a in
  let fd =
    Array.init n (fun i ->
        let p = f (perturbed i 1.) and q = f (perturbed i (-1.)) in
        (p -. q) /. (2. *. h))
  in
  closeness ~msg:"finite differences" ~rel:1e-5 fd g

let test_grad_past () =
  (* [1e200; -1e200]: the variance passes the range, its gradient may be
     an infinity or a NaN; it is defined by the value being [+∞]. *)
  let g = Nx.to_array (Rune.grad' Nx.var (f64 [| 1e200; -1e200 |])) in
  equal ~msg:"not finite at +∞" bool false (Array.for_all Float.is_finite g);
  (* Below the range the value rounds to zero, the derivative is ordinary. *)
  let g = Nx.to_array (Rune.grad' Nx.var (f64 [| 1e-170; -1e-170 |])) in
  closeness ~msg:"grad var below the range" ~rel:1e-13 [| 1e-170; -1e-170 |] g;
  let g32 = Nx.to_array (Nx.cast Nx.float64 (Rune.grad' Nx.std (f32 [| 1e25; -1e25 |]))) in
  closeness ~msg:"f32 grad std" ~rel:1e-5 [| 0.5; -0.5 |] g32;
  let g32 = Nx.to_array (Nx.cast Nx.float64 (Rune.grad' Nx.var (f32 [| 1.5e19; -1.5e19; 0. |]))) in
  closeness ~msg:"f32 grad var" ~rel:1e-5 [| 2. *. 1.5e19 /. 3.; -2. *. 1.5e19 /. 3.; 0. |] g32

let test_check_grads () =
  let x = f64 [| 0.5; 1.25; -3.; 2.; 0.75 |] in
  let t = Nx.Ptree.tensor in
  let g f =
    equal ~msg:"check_grads" (result unit string) (Ok ())
      (Rune.check_grads t f x)
  in
  g Nx.var;
  g Nx.std;
  g (Nx.std ~ddof:1);
  g (Nx.var ~ddof:2)

let () =
  exit
    (run "rune var and std"
       [
         cases "a compiled call equals an eager one" ~id:name lanes check_jit;
         test "a compiled call, lane by lane" test_lanes;
         test "a captured constant compiles" test_captured;
         test "empty lanes compile" test_empty;
         cases "the gradient of var" ~id:fst var_grad_cases check_var_grad;
         cases "the gradient of std"
           ~id:(fun (n, _, d) -> n ^ " " ^ string_of_int d)
           std_grad_cases check_std_grad;
         test "gradients at the edges" test_grad_past;
         test "check_grads at ordinary points" test_check_grads;
       ])
