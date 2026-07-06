(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Kaun (typed parameter records + composable training step) on a 3-layer MLP,
   784 -> 256 -> 128 -> 10, relu activations, batch 128, float32, softmax
   cross-entropy loss, Adam.

   The training-step case measures one full step from the same fixed initial
   state: forward, loss, backward, and an Adam update. *)

let batch = 128
let d_in = 784
let d_h1 = 256
let d_h2 = 128
let d_out = 10
let lr = 1e-3

(* Model: a plain record of Linear layers. *)

type mlp = { l1 : Kaun.Linear.t; l2 : Kaun.Linear.t; l3 : Kaun.Linear.t }

module Mlp = struct
  type t = mlp

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { l1; l2; l3 } =
    {
      l1 = Kaun.Linear.map f l1;
      l2 = Kaun.Linear.map f l2;
      l3 = Kaun.Linear.map f l3;
    }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    {
      l1 = Kaun.Linear.map2 f p.l1 q.l1;
      l2 = Kaun.Linear.map2 f p.l2 q.l2;
      l3 = Kaun.Linear.map2 f p.l3 q.l3;
    }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { l1; l2; l3 } =
    Kaun.Linear.iter f l1;
    Kaun.Linear.iter f l2;
    Kaun.Linear.iter f l3

  let apply p x =
    let open Kaun in
    Linear.apply p.l3
      (Fn.relu (Linear.apply p.l2 (Fn.relu (Linear.apply p.l1 x))))
end

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let x = Nx.randn Nx.float32 [| batch; d_in |] in
  let y =
    (* One-hot labels. *)
    let labels = Array.init batch (fun i -> i mod d_out) in
    let t = Nx.zeros Nx.float32 [| batch; d_out |] in
    Array.iteri (fun i c -> Nx.set_item [ i; c ] 1.0 t) labels;
    t
  in

  (* value_and_grad + Vega.adam_step over the parameter record. *)
  let params =
    {
      l1 = Kaun.Linear.init ~inputs:d_in ~outputs:d_h1;
      l2 = Kaun.Linear.init ~inputs:d_h1 ~outputs:d_h2;
      l3 = Kaun.Linear.init ~inputs:d_h2 ~outputs:d_out;
    }
  in
  let ostate = Vega.adam_init (module Mlp) params in
  let loss p = Kaun.Loss.softmax_cross_entropy (Mlp.apply p x) y in
  let step () =
    let l, grads = Rune.value_and_grad (module Mlp) loss params in
    let params', ostate' =
      Vega.adam_step (module Mlp) ~lr ostate ~params ~grads
    in
    (l, params', ostate')
  in

  Thumper.run "kaun"
    [
      Thumper.group "TrainStep"
        [ Thumper.bench "train step" (fun () -> step ()) ];
      Thumper.group "Forward"
        [ Thumper.bench "apply" (fun () -> Mlp.apply params x) ];
    ]
