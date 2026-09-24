(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Kaun (typed parameter records + composable training step) across a few
   representative workloads:

   - an MLP (784 -> 256 -> 128 -> 10, relu, batch 128, float32, softmax
   cross-entropy) trained one full step with Adam, and separately with SGD; -
   the same MLP's forward pass in isolation; - a small CNN (two conv + max-pool
   blocks then a dense head) trained one full step with Adam; - a single Linear
   layer (512 -> 512, batch 128) forward, and forward+backward.

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

module Mlp = struct
  type 'a t = {
    l1 : 'a Kaun.Linear.t;
    l2 : 'a Kaun.Linear.t;
    l3 : 'a Kaun.Linear.t;
  }

  let walk c { l1; l2; l3 } =
    let open Nx.Ptree.Walk in
    let l1 = field c "l1" Kaun.Linear.walk l1 in
    let l2 = field c "l2" Kaun.Linear.walk l2 in
    let l3 = field c "l3" Kaun.Linear.walk l3 in
    { l1; l2; l3 }

  let apply p x =
    let open Kaun in
    Linear.apply p.l3
      (Fn.relu (Linear.apply p.l2 (Fn.relu (Linear.apply p.l1 x))))
end

(* CNN: two conv + max-pool blocks then a dense head, NCHW, single-channel 28x28
   inputs, batch 32, float32. Spatial size shrinks 28 -conv3-> 26 -pool2-> 13
   -conv3-> 11 -pool2-> 5, so the head sees 16 * 5 * 5 features. *)

let cnn_batch = 32
let cnn_img = 28
let cnn_feats = 16 * 5 * 5

module Cnn = struct
  type 'a t = {
    c1 : 'a Kaun.Conv.t;
    c2 : 'a Kaun.Conv.t;
    head : 'a Kaun.Linear.t;
  }

  let walk c { c1; c2; head } =
    let open Nx.Ptree.Walk in
    let c1 = field c "c1" Kaun.Conv.walk c1 in
    let c2 = field c "c2" Kaun.Conv.walk c2 in
    let head = field c "head" Kaun.Linear.walk head in
    { c1; c2; head }

  let apply p x =
    let open Kaun in
    let h = Pool.max_pool2d ~kernel_size:(2, 2) (Fn.relu (Conv.apply p.c1 x)) in
    let h = Pool.max_pool2d ~kernel_size:(2, 2) (Fn.relu (Conv.apply p.c2 h)) in
    Linear.apply p.head (Nx.reshape [| cnn_batch; -1 |] h)
end

(* One-hot labels [0; 1; ...; d_out-1] cycled over [n] rows. *)
let one_hot n =
  let labels =
    Nx.init Nx.int32 [| n |] (fun i -> Int32.of_int (i.(0) mod d_out))
  in
  Nx.cast Nx.float32 (Nx.one_hot ~num_classes:d_out labels)

let mlp = Nx.Ptree.instantiate (module Mlp)
let cnn = Nx.Ptree.instantiate (module Cnn)

let () =
  Nx.Rng.with_key (Nx.Rng.key 42) @@ fun () ->
  (* MLP: inputs, parameters, and both optimizer states. *)
  let x = Nx.randn Nx.float32 [| batch; d_in |] in
  let y = one_hot batch in
  let params =
    {
      Mlp.l1 = Kaun.Linear.init ~inputs:d_in ~outputs:d_h1;
      l2 = Kaun.Linear.init ~inputs:d_h1 ~outputs:d_h2;
      l3 = Kaun.Linear.init ~inputs:d_h2 ~outputs:d_out;
    }
  in
  let loss p = Kaun.Loss.softmax_cross_entropy (Mlp.apply p x) y in
  let adam_state = Vega.adam_init mlp params in
  let sgd_state = Vega.sgd_init mlp params in
  let adam_step () =
    let l, grads = Rune.value_and_grad mlp loss params in
    let params', state' =
      Vega.adam_step mlp ~lr:(Vega.lr lr) adam_state ~params ~grads
    in
    (l, params', state')
  in
  let sgd_step () =
    let l, grads = Rune.value_and_grad mlp loss params in
    let params', state' =
      Vega.sgd_step mlp ~lr:(Vega.lr lr) sgd_state ~params ~grads
    in
    (l, params', state')
  in

  (* CNN: inputs, parameters, and Adam state. *)
  let cx = Nx.randn Nx.float32 [| cnn_batch; 1; cnn_img; cnn_img |] in
  let cy = one_hot cnn_batch in
  let cnn_params =
    {
      Cnn.c1 = Kaun.Conv.init ~in_channels:1 ~out_channels:8 ~kernel_size:(3, 3);
      c2 = Kaun.Conv.init ~in_channels:8 ~out_channels:16 ~kernel_size:(3, 3);
      head = Kaun.Linear.init ~inputs:cnn_feats ~outputs:d_out;
    }
  in
  let cnn_loss p = Kaun.Loss.softmax_cross_entropy (Cnn.apply p cx) cy in
  let cnn_state = Vega.adam_init cnn cnn_params in
  let cnn_step () =
    let l, grads = Rune.value_and_grad cnn cnn_loss cnn_params in
    let params', state' =
      Vega.adam_step cnn ~lr:(Vega.lr lr) cnn_state ~params:cnn_params ~grads
    in
    (l, params', state')
  in

  (* A single Linear layer in isolation: forward, and value_and_grad through
     it. *)
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
          Thumper.bench "conv train step" ~tags:[ "lab" ] (fun () ->
              cnn_step ());
        ];
      Thumper.group "Linear"
        [
          Thumper.bench "linear fwd" (fun () -> Kaun.Linear.apply lin lx);
          Thumper.bench "linear fwd+bwd" (fun () ->
              Rune.value_and_grad
                (Nx.Ptree.instantiate (module Kaun.Linear))
                lin_loss lin);
        ];
    ]
