(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Build custom optimizers by composing gradient transformation primitives.

   Vega's core abstraction is the composable gradient transformation. Optimizer
   aliases like [adam] are just shorthand for [chain]. This example shows how
   to:

   1. Recreate AdamW from primitives 2. Add gradient clipping to any optimizer
   3. Use update + apply_updates for explicit two-step control *)

let dt = Nx.float32
let x0 () = Nx.create dt [| 2 |] [| 5.0; -3.0 |]

let run name tx steps =
  let param = ref (x0 ()) in
  let st = ref (Vega.init tx !param) in
  for _ = 1 to steps do
    let p, s = Vega.step !st ~grad:!param ~param:!param in
    param := p;
    st := s
  done;
  Printf.printf "  %-40s x = %s\n" name (Nx.data_to_string !param)

let () =
  let lr = Vega.Schedule.constant 0.01 in

  (* 1. AdamW is just a chain of primitives *)
  Printf.printf "--- AdamW: alias vs primitives (50 steps) ---\n";

  run "adamw (alias)" (Vega.adamw ~weight_decay:0.01 lr) 50;

  run "chain [adam; decay; lr] (manual)"
    (Vega.chain
       [
         Vega.scale_by_adam ();
         Vega.add_decayed_weights ~rate:(Vega.Schedule.constant 0.01) ();
         Vega.scale_by_learning_rate lr;
       ])
    50;

  Printf.printf "\n";

  (* 2. Gradient clipping composes with any optimizer *)
  Printf.printf "--- Adding gradient clipping (50 steps) ---\n";

  run "adam (no clipping)" (Vega.adam lr) 50;

  run "clip_by_norm 1.0 + adam"
    (Vega.chain [ Vega.clip_by_norm 1.0; Vega.adam lr ])
    50;

  run "clip_by_value 0.5 + adam"
    (Vega.chain [ Vega.clip_by_value 0.5; Vega.adam lr ])
    50;

  Printf.printf "\n";

  (* 3. chain is associative: nesting doesn't change behavior *)
  Printf.printf "--- chain is associative (50 steps) ---\n";

  let a = Vega.scale_by_adam () in
  let b = Vega.add_decayed_weights ~rate:(Vega.Schedule.constant 0.01) () in
  let c = Vega.scale_by_learning_rate lr in

  run "chain [a; b; c]" (Vega.chain [ a; b; c ]) 50;
  run "chain [chain [a; b]; c]" (Vega.chain [ Vega.chain [ a; b ]; c ]) 50;

  Printf.printf "\n";

  (* 4. update + apply_updates: the explicit two-step API *)
  Printf.printf "--- update + apply_updates (explicit) ---\n";
  let tx = Vega.adam lr in
  let param = ref (x0 ()) in
  let st = ref (Vega.init tx !param) in
  for i = 1 to 50 do
    let updates, s = Vega.update !st ~grad:!param ~param:!param in
    param := Vega.apply_updates ~param:!param ~updates;
    st := s;
    if i mod 10 = 0 then
      Printf.printf "  step %2d  x = %s\n" i (Nx.data_to_string !param)
  done
