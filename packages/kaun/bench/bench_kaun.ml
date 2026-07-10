(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Kaun (typed parameter records + composable training step) across a few
   representative workloads:

   - an MLP (784 -> 256 -> 128 -> 10, relu, batch 128, float32, softmax
     cross-entropy) trained one full step with Adam, and separately with SGD;
   - the same MLP's forward pass in isolation;
   - a small CNN (two conv + max-pool blocks then a dense head) trained one full
     step with Adam;
   - a single Linear layer (512 -> 512, batch 128) forward, and forward+backward.

   Every case measures only the operation: inputs, parameters and optimizer
   state are built once in setup, outside the timed region. The train-step cases
   run one full step (forward, loss, backward, optimizer update) from the same
   fixed initial state each call. The Linear and MLP-forward cases isolate
   kaun/lib layer code from the optimizer in vega. *)

(* MLP: 3 layers, 784 -> 256 -> 128 -> 10. *)

let batch = 128
let d_in = 784
let d_h1 = 256
let d_h2 = 128
let d_out = 10
let lr = 1e-3

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

(* CNN: two conv + max-pool blocks then a dense head, NCHW, single-channel
   28x28 inputs, batch 32, float32. Spatial size shrinks
   28 -conv3-> 26 -pool2-> 13 -conv3-> 11 -pool2-> 5, so the head sees
   16 * 5 * 5 features. *)

let cnn_batch = 32
let cnn_img = 28
let cnn_feats = 16 * 5 * 5

type cnn = { c1 : Kaun.Conv.t; c2 : Kaun.Conv.t; head : Kaun.Linear.t }

module Cnn = struct
  type t = cnn

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { c1; c2; head } =
    {
      c1 = Kaun.Conv.map f c1;
      c2 = Kaun.Conv.map f c2;
      head = Kaun.Linear.map f head;
    }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    {
      c1 = Kaun.Conv.map2 f p.c1 q.c1;
      c2 = Kaun.Conv.map2 f p.c2 q.c2;
      head = Kaun.Linear.map2 f p.head q.head;
    }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { c1; c2; head } =
    Kaun.Conv.iter f c1;
    Kaun.Conv.iter f c2;
    Kaun.Linear.iter f head

  let apply p x =
    let open Kaun in
    let h = Pool.max_pool2d ~kernel_size:(2, 2) (Fn.relu (Conv.apply p.c1 x)) in
    let h = Pool.max_pool2d ~kernel_size:(2, 2) (Fn.relu (Conv.apply p.c2 h)) in
    Linear.apply p.head (Nx.reshape [| cnn_batch; -1 |] h)
end

(* One-hot labels [0; 1; ...; d_out-1] cycled over [n] rows. *)
let one_hot n =
  let labels = Array.init n (fun i -> i mod d_out) in
  let t = Nx.zeros Nx.float32 [| n; d_out |] in
  Array.iteri (fun i c -> Nx.set_item [ i; c ] 1.0 t) labels;
  t

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  (* MLP: inputs, parameters, and both optimizer states. *)
  let x = Nx.randn Nx.float32 [| batch; d_in |] in
  let y = one_hot batch in
  let params =
    {
      l1 = Kaun.Linear.init ~inputs:d_in ~outputs:d_h1;
      l2 = Kaun.Linear.init ~inputs:d_h1 ~outputs:d_h2;
      l3 = Kaun.Linear.init ~inputs:d_h2 ~outputs:d_out;
    }
  in
  let loss p = Kaun.Loss.softmax_cross_entropy (Mlp.apply p x) y in
  let adam_state = Vega.adam_init (module Mlp) params in
  let sgd_state = Vega.sgd_init (module Mlp) params in
  let adam_step () =
    let l, grads = Rune.value_and_grad (module Mlp) loss params in
    let params', state' = Vega.adam_step (module Mlp) ~lr adam_state ~params ~grads in
    (l, params', state')
  in
  let sgd_step () =
    let l, grads = Rune.value_and_grad (module Mlp) loss params in
    let params', state' = Vega.sgd_step (module Mlp) ~lr sgd_state ~params ~grads in
    (l, params', state')
  in

  (* CNN: inputs, parameters, and Adam state. *)
  let cx = Nx.randn Nx.float32 [| cnn_batch; 1; cnn_img; cnn_img |] in
  let cy = one_hot cnn_batch in
  let cnn_params =
    {
      c1 = Kaun.Conv.init ~in_channels:1 ~out_channels:8 ~kernel_size:(3, 3);
      c2 = Kaun.Conv.init ~in_channels:8 ~out_channels:16 ~kernel_size:(3, 3);
      head = Kaun.Linear.init ~inputs:cnn_feats ~outputs:d_out;
    }
  in
  let cnn_loss p = Kaun.Loss.softmax_cross_entropy (Cnn.apply p cx) cy in
  let cnn_state = Vega.adam_init (module Cnn) cnn_params in
  let cnn_step () =
    let l, grads = Rune.value_and_grad (module Cnn) cnn_loss cnn_params in
    let params', state' =
      Vega.adam_step (module Cnn) ~lr cnn_state ~params:cnn_params ~grads
    in
    (l, params', state')
  in

  (* A single Linear layer in isolation: forward, and value_and_grad through it. *)
  let lin_in = 512 and lin_out = 512 and lin_batch = 128 in
  let lin = Kaun.Linear.init ~inputs:lin_in ~outputs:lin_out in
  let lx = Nx.randn Nx.float32 [| lin_batch; lin_in |] in
  let lin_loss p = Nx.sum (Kaun.Linear.apply p lx) in

  let budgets =
    [
      Thumper.Budget.no_slower_than ~metric:Thumper.Metric.wall_time 0.05;
      Thumper.Budget.no_more_alloc_than 0.01;
    ]
  in
  Thumper.run "kaun" ~budgets
    [
      Thumper.group "TrainStep"
        [
          Thumper.bench "train step" ~tags:[ "lab" ] (fun () -> adam_step ());
          Thumper.bench "sgd train step" (fun () -> sgd_step ());
        ];
      Thumper.group "Forward"
        [ Thumper.bench "apply" (fun () -> Mlp.apply params x) ];
      Thumper.group "Conv"
        [
          Thumper.bench "conv train step" ~tags:[ "lab" ] (fun () -> cnn_step ());
        ];
      Thumper.group "Linear"
        [
          Thumper.bench "linear fwd" (fun () -> Kaun.Linear.apply lin lx);
          Thumper.bench "linear fwd+bwd" (fun () ->
              Rune.value_and_grad (module Kaun.Linear) lin_loss lin);
        ];
    ]
