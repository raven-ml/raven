(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Minimize f(x) = 0.5 * ||x||^2 using different optimizers.

   The gradient is simply x, so this is a clean testbed for comparing
   convergence behavior. Each optimizer starts from the same point x = [5.0;
   -3.0] and runs 50 steps. *)

let dt = Nx.float32
let x0 () = Nx.create dt [| 2 |] [| 5.0; -3.0 |]

let run name tx =
  Printf.printf "--- %s ---\n" name;
  let param = ref (x0 ()) in
  let st = ref (Vega.init tx !param) in
  for i = 1 to 50 do
    let p, s = Vega.step !st ~grad:!param ~param:!param in
    param := p;
    st := s;
    if i mod 10 = 0 then
      Printf.printf "  step %2d  x = %s\n" i (Nx.data_to_string !param)
  done;
  Printf.printf "\n"

let () =
  let lr = Vega.Schedule.constant 0.1 in
  run "SGD (lr=0.1)" (Vega.sgd lr);

  let lr = Vega.Schedule.constant 0.01 in
  run "Adam (lr=0.01)" (Vega.adam lr);

  let lr = Vega.Schedule.constant 0.01 in
  run "AdamW (lr=0.01, wd=0.01)" (Vega.adamw ~weight_decay:0.01 lr)
