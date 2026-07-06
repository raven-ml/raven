(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* XOR with kaun-next: a model is a plain record, the training step is
   value_and_grad + Vega, and the loop is a plain for loop — no trainer, no
   layer type. *)

open Kaun_next

module Mlp = struct
  type t = { l1 : Linear.t; l2 : Linear.t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { l1; l2 } =
    { l1 = Linear.map f l1; l2 = Linear.map f l2 }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { l1 = Linear.map2 f p.l1 q.l1; l2 = Linear.map2 f p.l2 q.l2 }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { l1; l2 } =
    Linear.iter f l1;
    Linear.iter f l2

  let apply p x = Linear.apply p.l2 (Nx.tanh (Linear.apply p.l1 x))
end

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  (* XOR dataset *)
  let x =
    Nx.create Nx.float32 [| 4; 2 |] [| 0.; 0.; 0.; 1.; 1.; 0.; 1.; 1. |]
  in
  let y = Nx.create Nx.float32 [| 4; 1 |] [| 0.; 1.; 1.; 0. |] in

  (* Model parameters *)
  let params =
    {
      Mlp.l1 = Linear.init ~inputs:2 ~outputs:8;
      l2 = Linear.init ~inputs:8 ~outputs:1;
    }
  in

  (* Training step: value_and_grad + one Adam update *)
  let loss p = Loss.sigmoid_bce (Mlp.apply p x) y in
  let step (params, ostate) =
    let l, grads = Rune_next.value_and_grad (module Mlp) loss params in
    let params, ostate =
      Vega.adam_step (module Mlp) ~lr:0.05 ostate ~params ~grads
    in
    ((params, ostate), Nx.item [] l)
  in

  (* Fit *)
  let state = ref (params, Vega.adam_init (module Mlp) params) in
  for i = 1 to 500 do
    let s, l = step !state in
    state := s;
    if i mod 100 = 0 then Printf.printf "step %4d  loss %.6f\n%!" i l
  done;

  (* Evaluate *)
  let pred = Fn.sigmoid (Mlp.apply (fst !state) x) in
  Printf.printf "\npredictions (expected 0 1 1 0):\n";
  for i = 0 to 3 do
    Printf.printf "  [%.0f, %.0f] -> %.3f\n"
      (Nx.item [ i; 0 ] x)
      (Nx.item [ i; 1 ] x)
      (Nx.item [ i; 0 ] pred)
  done
