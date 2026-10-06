(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Double-word numbers compiled and differentiated. A compiled program rounds
   every sum and product as written, so its results equal eager ones bit for
   bit; a derivative through the high word is the float derivative. *)

open Windtrap

(* A word's bits, every NaN as one: arithmetic leaves a NaN's sign and payload
   to the target. *)
let bits w =
  let bits x =
    if Float.is_nan x then 0x7ff8_0000_0000_0000L else Int64.bits_of_float x
  in
  ( Array.map bits (Nx.to_array (Nx_wide.hi w)),
    Array.map bits (Nx.to_array (Nx_wide.lo w)) )

let same_bits = pair (array int64) (array int64)

(* Compiled equals eager *)

type op = {
  name : string;
  f : 'b. 'b Nx_wide.t -> 'b Nx_wide.t -> 'b Nx_wide.t;
}

let ops =
  [
    { name = "add"; f = Nx_wide.add };
    { name = "sub"; f = Nx_wide.sub };
    { name = "mul"; f = Nx_wide.mul };
    { name = "div"; f = Nx_wide.div };
    { name = "floor"; f = (fun a _ -> Nx_wide.floor a) };
    { name = "sum"; f = (fun a b -> Nx_wide.sum (Nx_wide.add a b)) };
  ]

let width = 16

(* Words of [width] numbers: [hi] over many binades, [lo] a fraction of it
   [2^-k] down, with infinities, NaN and signed zeros among them. *)
let numbers =
  let open Gen in
  let special =
    of_list ~pp:Format.pp_print_float
      [ 0.; -0.; 1.; -1.; Float.infinity; Float.nan; 0.5; 3. ]
  in
  let hi =
    one_of
      [
        special;
        map
          (fun (m, e) -> Float.ldexp m e)
          (pair (float_range (-2.) 2.) (int_range (-60) 60));
      ]
  in
  let lo = pair (float_range (-1.) 1.) (int_range 50 70) in
  array ~size:(int_range width width) (pair hi lo)

let words (type b) (dt : (float, b) Nx.dtype) xs =
  let hi = Array.map fst xs in
  let lo = Array.map (fun (h, (m, k)) -> Float.ldexp (m *. h) (-k)) xs in
  let lo = Array.map (fun l -> if Float.is_finite l then l else 0.) lo in
  let t a = Nx.cast dt (Nx.create Nx.float64 [| width |] a) in
  Nx_wide.v ~lo:(t lo) (t hi)

let compiled (type b) (dt : (float, b) Nx.dtype) tag =
  List.map
    (fun op ->
      let p = Nx_wide.ptree dt in
      let g = Rune.jit Nx.Ptree.(p @-> p @-> returns p) op.f in
      prop
        (Printf.sprintf "%s at %s, compiled, equals it eagerly bit for bit"
           op.name tag) (Gen.pair numbers numbers) (fun (a, b) ->
          let a = words dt a and b = words dt b in
          equal same_bits (bits (op.f a b)) (bits (g a b))))
    ops

let f64 x = Nx.scalar Nx.float64 x
let words_of w = (Nx.item [] (Nx_wide.hi w), Nx.item [] (Nx_wide.lo w))
let p64 = Nx_wide.ptree Nx.float64

(* [exact name f a b expected]: [f a b] is [expected]'s words, eagerly and
   compiled. A fold of [(a + b) - a] into [b] would make each low word zero. *)
let exact name f a b expected =
  test name (fun () ->
      let g = Rune.jit Nx.Ptree.(p64 @-> p64 @-> returns p64) f in
      equal ~msg:"eager"
        (pair float_exact float_exact)
        expected
        (words_of (f a b));
      equal ~msg:"compiled"
        (pair float_exact float_exact)
        expected
        (words_of (g a b)))

let one = Nx_wide.v (f64 1.)
let tiny = Nx_wide.v (f64 (Float.ldexp 1. (-60)))
let e k = Float.ldexp 1. k

let error_terms =
  group "error terms"
    [
      exact "1 + 2^-60 keeps its low word" Nx_wide.add one tiny (1., e (-60));
      exact "1 - 2^-60 keeps its low word" Nx_wide.sub one tiny (1., -.e (-60));
      exact "(1 + 2^-30)² keeps its product's error"
        (fun a _ -> Nx_wide.mul a a)
        (Nx_wide.v (f64 (1. +. e (-30))))
        one
        (1. +. e (-29), e (-60));
      test "a quotient by 1/yh compiled is the eager one, its low word kept"
        (fun () ->
          (* (3 + 3 2^-60) / 3 = 1 + 2^-60. *)
          let x = Nx_wide.v ~lo:(f64 (3. *. e (-60))) (f64 3.)
          and y = Nx_wide.v (f64 3.) in
          let g = Rune.jit Nx.Ptree.(p64 @-> p64 @-> returns p64) Nx_wide.div in
          let eager = Nx_wide.div x y in
          equal (pair int64 int64)
            (let h, l = words_of eager in
             (Int64.bits_of_float h, Int64.bits_of_float l))
            (let h, l = words_of (g x y) in
             (Int64.bits_of_float h, Int64.bits_of_float l));
          equal ~msg:"high word" float_exact 1. (fst (words_of eager));
          at_most ~msg:"the low word's error" float_exact
            ~than:(10. *. e (-106))
            (Float.abs (snd (words_of eager) -. e (-60))));
      exact "floor (1 - 2^-60) is 0"
        (fun a b -> Nx_wide.floor (Nx_wide.sub a b))
        one tiny (0., 0.);
    ]

(* Derivatives through the high word *)

(* A phase [f0 t + f1 t² / 2] of a double-word time. *)
let phase t =
  let c x = Nx_wide.v (Nx.full_like (Nx_wide.hi t) x) in
  Nx_wide.(add (mul (c 29.946923) t) (mul (c (-3.77535e-10)) (mul t t)))

let phase_f t =
  Nx.add (Nx.mul_s t 29.946923) (Nx.mul_s (Nx.mul t t) (-3.77535e-10))

let close = Oracle.tensor ~rel:1e-12 ~abs:0. ()

let derivatives =
  let t = Nx.create Nx.float64 [| 3 |] [| 1e8; -2.5e9; 4.1e7 |] in
  let lo = Nx.create Nx.float64 [| 3 |] [| 3e-9; -1e-8; 2e-10 |] in
  let through_hi h = Nx.sum (Nx_wide.hi (phase (Nx_wide.v ~lo h))) in
  let float_derivative =
    Rune.grad Nx.Ptree.tensor (fun t -> Nx.sum (phase_f t)) t
  in
  group "derivatives"
    [
      test "a derivative through the high word is the float derivative"
        (fun () ->
          equal close float_derivative (Rune.grad Nx.Ptree.tensor through_hi t));
      test "compiled, it is the same" (fun () ->
          equal close float_derivative
            (Rune.jit' (Rune.grad Nx.Ptree.tensor through_hi) t));
      test "a derivative in both words, rebuilt as one number, is twice it"
        (fun () ->
          let p = Nx_wide.ptree Nx.float64 in
          let g =
            Rune.grad p
              (fun w -> Nx.sum (Nx_wide.hi (phase w)))
              (Nx_wide.v ~lo t)
          in
          equal close (Nx.mul_s float_derivative 2.) (Nx_wide.hi g));
    ]

let () =
  exit
    (run "Rune nx.wide"
       [
         group "compiled"
           (compiled Nx.float64 "float64" @ compiled Nx.float32 "float32");
         error_terms;
         derivatives;
       ])
