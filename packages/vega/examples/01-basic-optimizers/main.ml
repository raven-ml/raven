(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Minimize f(x) = 0.5 * ||x||^2 using different optimizers.

   The gradient is simply x, so this is a clean testbed for comparing
   convergence behavior. The parameters are a single tensor, whose structure is
   [Nx.Ptree.tensor]. Each optimizer starts from the same point x = [5.0; -3.0]
   and runs 50 steps. *)

let p = Nx.Ptree.tensor
let x0 () = Nx.create Nx.float32 [| 2 |] [| 5.0; -3.0 |]

let run name init step =
  Printf.printf "--- %s ---\n" name;
  let params = ref (x0 ()) in
  let st = ref (init p !params) in
  for i = 1 to 50 do
    let params', st' = step !st ~params:!params ~grads:!params in
    params := params';
    st := st';
    if i mod 10 = 0 then
      Printf.printf "  step %2d  x = %s\n" i (Nx.to_string !params)
  done;
  Printf.printf "\n"

let () =
  run "SGD (lr=0.1)" Vega.sgd_init (fun st ->
      Vega.sgd_step p ~lr:(Vega.lr 0.1) st);
  run "Adam (lr=0.01)" Vega.adam_init (fun st ->
      Vega.adam_step p ~lr:(Vega.lr 0.01) st);
  run "AdamW (lr=0.01, wd=0.01)" Vega.adamw_init (fun st ->
      Vega.adamw_step p ~lr:(Vega.lr 0.01) ~weight_decay:0.01 st)
