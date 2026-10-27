(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.program and Rune.eval_shape: what a function traces, printed, and its
   result's structure without computing it. *)

open Windtrap
module Rune = Rune2
module T = Nx.Ptree

let f32 = Nx.float32
let v xs = Nx.create f32 [| Array.length xs |] xs
let invalid ~sub f = raises_match (Exn.invalid_arg ~substring:sub) f
let print p = Format.asprintf "%a" Rune.pp_program p

(* The tangent of each map is computed with its primal in one map of two
   outputs. *)
let test_jvp_fused () =
  let f x = Rune.jvp' (fun x -> Nx.sin (Nx.mul x x)) x x in
  let p = Rune.program T.tensor T.(pair tensor tensor) f (v [| 1.; 2. |]) in
  print_string (print p);
  expect (output ()) @@ __POS_OF__ {|
    program (%0)
      %1 : float32 [2] at CPU, %2 : float32 [2] at CPU = Map [mul(x0, x1); add(mul(x2, x1), mul(x0, x3))] (x0: float32 [2] at CPU) (x1: float32 [2] at CPU) (x2: float32 [2] at CPU) (x3: float32 [2] at CPU)
        with x0 = %0, x1 = %0, x2 = %0, x3 = %0
      %3 : float32 [2] at CPU, %4 : float32 [2] at CPU = Map [sin(x0); mul(x1, cos(x0))] (x0: float32 [2] at CPU) (x1: float32 [2] at CPU)
        with x0 = %1, x1 = %2
      return %3, %4
    |}

(* A scan carries its step's program. *)
let test_scan () =
  let f x =
    fst (Rune.scan' ~f:(fun c r -> (Nx.add c r, r)) ~init:x (Nx.reshape [| 2; 1 |] x))
  in
  print_string (print (Rune.program T.tensor T.tensor f (v [| 1.; 2. |])));
  expect (output ()) @@ __POS_OF__ {|
    program (%0)
      %1 : float32 [2; 1] at CPU = Move (x0: float32 [2] at CPU)
        with x0 = %0
      %2 : float32 [2] at CPU, %3 : float32 [2; 1] at CPU = scan 2 rows, init (%0), xs (%1), captures ()
        step
          program (%4, %5)
            %6 : float32 [2] at CPU = Move (x0: float32 [1] at CPU)
              with x0 = %5
            %7 : float32 [2] at CPU = Map [add(x0, x1)] (x0: float32 [2] at CPU) (x1: float32 [2] at CPU)
              with x0 = %4, x1 = %6
            return %7, %5
      return %2
    |}

(* A constant, a detach, a total's addition and a grad's tape, each
   printed. *)
let test_constructs () =
  let w = v [| 0.5; -0. |] in
  let total = Rune.Total.make () in
  let f x =
    Rune.Total.add total x;
    Nx.add (Rune.detach x) (Rune.grad' (fun x -> Nx.sum (Nx.mul x w)) x)
  in
  let p, _ =
    Rune.Total.collect total ~zero:(Nx.zeros f32 [| 2 |]) (fun () ->
        Rune.program T.tensor T.tensor f (v [| 1.; 2. |]))
  in
  print_string (print p);
  expect (output ()) @@ __POS_OF__ {|
    c0 : float32 [2] at CPU
    program (%0)
      %1 : float32 [2] at CPU = Map [mul(x0, x1)] (x0: float32 [2] at CPU) (x1: float32 [2] at CPU)
        with x0 = %0, x1 = c0
      %2 : float32 [] at CPU = Reduce [Sum x0] over [0] (x0: float32 [2] at CPU)
        with x0 = %1
      %3 : float32 [] of every set = Map [float32:0x0000803f]
      %4 : float32 [] at CPU = Place (x0: float32 [] of every set)
        with x0 = %3
      %5 : float32 [1] at CPU = Move (x0: float32 [] at CPU)
        with x0 = %4
      %6 : float32 [2] at CPU = Move (x0: float32 [1] at CPU)
        with x0 = %5
      %7 : float32 [2] at CPU = Map [x0] (x0: float32 [2] at CPU)
        with x0 = %6
      %8 : float32 [2] at CPU = Map [mul(x1, x0)] (x0: float32 [2] at CPU) (x1: float32 [2] at CPU)
        with x0 = c0, x1 = %7
      %9 : float32 [2] at CPU = detach %0
      %10 : float32 [2] at CPU = Map [add(x0, x1)] (x0: float32 [2] at CPU) (x1: float32 [2] at CPU)
        with x0 = %9, x1 = %8
      return %10
      adds %0
    |}

(* An iterate carries its test and its step; nested programs number their
   variables after their parent's. *)
let test_iterate () =
  let f x =
    Rune.iterate' ~max:4
      ~until:(fun c -> Nx.greater_s (Nx.sum c) 10.)
      ~f:(fun c -> Nx.mul c (Nx.scalar_like c 2.))
      x
  in
  print_string (print (Rune.program T.tensor T.tensor f (v [| 1.; 2. |])));
  expect (output ()) @@ __POS_OF__ {|
    program (%0)
      %1 : float32 [] of every set = Map [float32:0x00000040]
      %2 : float32 [2] of every set = Move (x0: float32 [] of every set)
        with x0 = %1
      %3 : float32 [] of every set = Map [float32:0x00002041]
      %4 : float32 [2] at CPU = iterate at most 4, init (%0), captures (%3) and (%2)
        until
          program (%5, %7)
            %6 : float32 [] at CPU = Reduce [Sum x0] over [0] (x0: float32 [2] at CPU)
              with x0 = %5
            %8 : bool [] at CPU = Map [less(x0, x1)] (x0: float32 [] of every set) (x1: float32 [] at CPU)
              with x0 = %7, x1 = %6
            return %8
        step
          program (%9, %10)
            %11 : float32 [2] at CPU = Map [mul(x0, x1)] (x0: float32 [2] at CPU) (x1: float32 [2] of every set)
              with x0 = %9, x1 = %10
            return %11
      return %4
    |}

(* Two traces of one function print alike. *)
let test_stable () =
  let f (x, y) = Nx.add (Nx.mul x y) (Nx.scalar_like x 2.) in
  let s = T.(pair tensor tensor) in
  let trace () = print (Rune.program s T.tensor f (v [| 1. |], v [| 3. |])) in
  equal string (trace ()) (trace ())

(* The result's structure, dtypes, shapes and placements, without computing;
   an operation on a stand-in raises. *)
let test_eval_shape () =
  let n = ref 0 in
  let f x =
    incr n;
    (Nx.sum ~axes:[ 0 ] x, Nx.cast Nx.int32 x)
  in
  let x = Nx.create f32 [| 2; 3 |] (Array.make 6 1.) in
  let s, c = Rune.eval_shape T.tensor T.(pair tensor tensor) f x in
  equal ~msg:"f runs once" int 1 !n;
  equal (array int) [| 3 |] (Nx.shape s);
  equal (array int) [| 2; 3 |] (Nx.shape c);
  equal ~msg:"dtype" string "int32" (Nx.Dtype.name (Nx.dtype c));
  let at y = Format.asprintf "%a" Nx.Placement.pp (Option.get (Nx.placement y)) in
  equal ~msg:"placement" string (at x) (at s);
  invalid ~sub:"Rune.eval_shape" (fun () -> Nx.to_array s)

let () =
  exit
    (run "rune2 program"
       [
         group "program"
           [
             test "a map's tangent is in its map" test_jvp_fused;
             test "a scan carries its step" test_scan;
             test "constants, detach, totals and grad" test_constructs;
             test "an iterate carries its test and step" test_iterate;
             test "two traces print alike" test_stable;
             test "eval_shape" test_eval_shape;
           ];
       ])
