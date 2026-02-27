(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx
open Rune

(* Example *)
let () =
  let f x = mul x (mul x (mul x x)) in
  let x = create Float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let result = f x in
  Printf.printf "Result: %s\n" (to_string result);

  (* eager *)
  let result = add x x in
  Printf.printf "Eager result: %s\n" (to_string result);

  (* gradient *)
  let gradient = (grad f) x in
  Printf.printf "First order derivative: %s\n" (to_string gradient);

  let gradient = (grad (grad f)) x in
  Printf.printf "Second order derivative: %s\n" (to_string gradient);

  let gradient = (grad (grad (grad f))) x in
  Printf.printf "Third order derivative: %s\n" (to_string gradient)
