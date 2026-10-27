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
      %1, %2 = Map [mul(x0, x1); add(mul(x2, x1), mul(x0, x3))] (x0: float32 [2] at CPU) (x1: float32 [2] at CPU) (x2: float32 [2] at CPU) (x3: float32 [2] at CPU)
        with x0 = %0, x1 = %0, x2 = %0, x3 = %0
      %3, %4 = Map [sin(x0); mul(x1, cos(x0))] (x0: float32 [2] at CPU) (x1: float32 [2] at CPU)
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
      %1 = Move (x0: float32 [2] at CPU)
        with x0 = %0
      %2, %3 = scan 2 rows, init (%0), xs (%1), captures ()
        program (%0, %1)
          %2 = Move (x0: float32 [1] at CPU)
            with x0 = %1
          %3 = Map [add(x0, x1)] (x0: float32 [2] at CPU) (x1: float32 [2] at CPU)
            with x0 = %0, x1 = %2
          return %3, %1
      return %2
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
  equal bool true (Nx.Dtype.equal Nx.int32 (Nx.dtype c));
  equal bool true
    (Nx.Placement.equal
       (Option.get (Nx.placement x))
       (Option.get (Nx.placement s)));
  invalid ~sub:"Rune.eval_shape" (fun () -> Nx.to_array s)

let () =
  exit
    (run "rune2 program"
       [
         group "program"
           [
             test "a map's tangent is in its map" test_jvp_fused;
             test "a scan carries its step" test_scan;
             test "two traces print alike" test_stable;
             test "eval_shape" test_eval_shape;
           ];
       ])
