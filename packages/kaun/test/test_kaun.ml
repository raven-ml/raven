(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The north-star test: a model is a plain record, the training step is
   value_and_grad + Vega, and the loop is a fold — no trainer, no layer type. *)

open Windtrap
open Kaun

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

let xor_x =
  lazy (Nx.create Nx.float32 [| 4; 2 |] [| 0.; 0.; 0.; 1.; 1.; 0.; 1.; 1. |])

let xor_y = lazy (Nx.create Nx.float32 [| 4; 1 |] [| 0.; 1.; 1.; 0. |])

let test_xor_trains () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let x = Lazy.force xor_x and y = Lazy.force xor_y in
  let params =
    {
      Mlp.l1 = Linear.init ~inputs:2 ~outputs:8;
      l2 = Linear.init ~inputs:8 ~outputs:1;
    }
  in
  let loss p = Loss.sigmoid_bce (Mlp.apply p x) y in
  let step (params, ostate) =
    let l, grads = Rune.value_and_grad (module Mlp) loss params in
    let params, ostate =
      Vega.adam_step (module Mlp) ~lr:0.05 ostate ~params ~grads
    in
    ((params, ostate), Nx.item [] l)
  in
  let state = ref (params, Vega.adam_init (module Mlp) params) in
  let last = ref Float.infinity in
  for _ = 1 to 500 do
    let s, l = step !state in
    state := s;
    last := l
  done;
  is_true ~msg:"loss converged" (!last < 0.05);
  let pred = Nx.to_array (Nx.reshape [| 4 |] (Mlp.apply (fst !state) x)) in
  let expect = [| 0.; 1.; 1.; 0. |] in
  Array.iteri
    (fun i e ->
      is_true
        ~msg:(Printf.sprintf "prediction %d" i)
        (if e > 0.5 then pred.(i) > 0.0 else pred.(i) < 0.0))
    expect

let test_sgd_decreases_loss () =
  Nx.Rng.run ~seed:7 @@ fun () ->
  let x = Lazy.force xor_x and y = Lazy.force xor_y in
  let params =
    {
      Mlp.l1 = Linear.init ~inputs:2 ~outputs:8;
      l2 = Linear.init ~inputs:8 ~outputs:1;
    }
  in
  let loss p = Loss.mse (Mlp.apply p x) y in
  let l0 = Nx.item [] (loss params) in
  let state = ref (params, Vega.sgd_init (module Mlp) params) in
  for _ = 1 to 100 do
    let params, ostate = !state in
    let _, grads = Rune.value_and_grad (module Mlp) loss params in
    state :=
      Vega.sgd_step (module Mlp) ~lr:0.1 ~momentum:0.9 ostate ~params ~grads
  done;
  let l1 = Nx.item [] (loss (fst !state)) in
  is_true ~msg:"sgd decreases the loss" (l1 < l0 *. 0.5)

let tests =
  [
    group "end to end"
      [
        test "adam trains xor" test_xor_trains;
        test "sgd with momentum decreases the loss" test_sgd_decreases_loss;
      ];
  ]

let () = run "kaun" tests
