(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun

let () =
  let rngs = Rune.Rng.key 42 in
  let dtype = Rune.float32 in

  (* XOR dataset *)
  let x = Rune.create dtype [| 4; 2 |] [| 0.; 0.; 0.; 1.; 1.; 0.; 1.; 1. |] in
  let y = Rune.create dtype [| 4; 1 |] [| 0.; 1.; 1.; 0. |] in

  (* Model *)
  let model =
    Layer.sequential
      [
        Layer.linear ~in_features:2 ~out_features:4 ();
        Layer.tanh ();
        Layer.linear ~in_features:4 ~out_features:1 ();
      ]
  in

  (* Trainer = model + optimizer *)
  let trainer =
    Train.make ~model
      ~optimizer:(Optim.adam ~lr:(Optim.Schedule.constant 0.01) ())
  in

  (* Initialize train state (vars + optimizer state) *)
  let st = Train.init trainer ~rngs ~dtype in

  (* Fit *)
  let st =
    Train.fit trainer st ~rngs
      ~report:(fun ~step ~loss _st ->
        if step mod 200 = 0 then Printf.printf "step %4d  loss %.6f\n" step loss)
      (Data.repeat 1000 (x, fun pred -> Loss.binary_cross_entropy pred y))
  in

  (* Evaluate *)
  let pred = Train.predict trainer st x |> Rune.sigmoid in
  Printf.printf "\npredictions (expected 0 1 1 0):\n";
  for i = 0 to 3 do
    Printf.printf "  [%.0f, %.0f] -> %.3f\n"
      (Rune.item [ i; 0 ] x)
      (Rune.item [ i; 1 ] x)
      (Rune.item [ i; 0 ] pred)
  done
