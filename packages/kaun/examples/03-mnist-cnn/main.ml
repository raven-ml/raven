(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A small CNN on MNIST: Conv + Pool + Dropout in the forward pass, and
   Checkpoint to save and restore the model and AdamW optimizer state.

   Trains on a subset of MNIST to keep the run short: expect ~93% test accuracy
   after three epochs, in well under a minute on CPU. *)

open Kaun

let batch_size = 128
let epochs = 3
let train_examples = 6_000
let test_examples = 2_000
let lr = 3e-3

(* conv(1->8) -> relu -> pool -> conv(8->16) -> relu -> pool -> dropout ->
   linear(400->10). Images are NCHW [n; 1; 28; 28]. *)
module Cnn = struct
  type t = { c1 : Conv.t; c2 : Conv.t; fc : Linear.t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { c1; c2; fc } =
    { c1 = Conv.map f c1; c2 = Conv.map f c2; fc = Linear.map f fc }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    {
      c1 = Conv.map2 f p.c1 q.c1;
      c2 = Conv.map2 f p.c2 q.c2;
      fc = Linear.map2 f p.fc q.fc;
    }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { c1; c2; fc } =
    Conv.iter f c1;
    Conv.iter f c2;
    Linear.iter f fc

  let names { c1; c2; fc } =
    List.concat
      [
        List.map (( ^ ) "c1.") (Conv.names c1);
        List.map (( ^ ) "c2.") (Conv.names c2);
        List.map (( ^ ) "fc.") (Linear.names fc);
      ]

  let init () =
    {
      c1 = Conv.init ~in_channels:1 ~out_channels:8 ~kernel_size:(3, 3);
      c2 = Conv.init ~in_channels:8 ~out_channels:16 ~kernel_size:(3, 3);
      fc = Linear.init ~inputs:(16 * 5 * 5) ~outputs:10;
    }

  let apply p ~training x =
    let n = (Nx.shape x).(0) in
    Conv.apply p.c1 x |> Fn.relu
    |> Pool.max_pool2d ~kernel_size:(2, 2)
    |> Conv.apply p.c2 |> Fn.relu
    |> Pool.max_pool2d ~kernel_size:(2, 2)
    |> Nx.reshape [| n; 16 * 5 * 5 |]
    |> Dropout.apply ~rate:0.25 ~training
    |> Linear.apply p.fc
end

let accuracy params (x, y) =
  Metric.accuracy (Cnn.apply params ~training:false x) y

let save_training_state path (params, (ostate : Cnn.t Vega.adam_state)) =
  Checkpoint.save path
    (Checkpoint.concat
       [
         Checkpoint.of_params (module Cnn) ~prefix:"model" params;
         Checkpoint.of_params (module Cnn) ~prefix:"optim.mu" ostate.mu;
         Checkpoint.of_params (module Cnn) ~prefix:"optim.nu" ostate.nu;
         Checkpoint.of_int "optim.step" ostate.step;
       ])

let load_training_state path =
  let ckpt = Checkpoint.load path in
  let like = Cnn.init () in
  let like_ostate = Vega.adamw_init (module Cnn) like in
  let params = Checkpoint.to_params (module Cnn) ~prefix:"model" ~like ckpt in
  let ostate =
    {
      Vega.mu =
        Checkpoint.to_params
          (module Cnn)
          ~prefix:"optim.mu" ~like:like_ostate.mu ckpt;
      nu =
        Checkpoint.to_params
          (module Cnn)
          ~prefix:"optim.nu" ~like:like_ostate.nu ckpt;
      step = Checkpoint.to_int "optim.step" ckpt;
    }
  in
  (params, ostate)

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  Printf.printf "Loading MNIST...\n%!";
  match Kaun_datasets.mnist () with
  | exception Failure msg ->
      Printf.printf "MNIST unavailable (%s); skipping.\n" msg
  | train_x, train_y, test_x, test_y ->
      let take n t = Nx.slice [ Nx.R (0, n) ] t in
      let train_x = take train_examples train_x
      and train_y = take train_examples train_y
      and test = (take test_examples test_x, take test_examples test_y) in

      let params = Cnn.init () in

      (* Training step: value_and_grad + one AdamW update. *)
      let step (params, ostate) (x, y) =
        let loss p =
          Loss.softmax_cross_entropy_sparse (Cnn.apply p ~training:true x) y
        in
        let l, grads = Rune.value_and_grad (module Cnn) loss params in
        let params, ostate =
          Vega.adamw_step (module Cnn) ~lr ostate ~params ~grads
        in
        ((params, ostate), Nx.item [] l)
      in

      (* Iterating the shuffled sequence once per epoch reshuffles each
         epoch. *)
      let batches =
        Data.batches2 ~shuffle:true ~batch_size (train_x, train_y)
      in
      let state = ref (params, Vega.adamw_init (module Cnn) params) in
      for epoch = 1 to epochs do
        let losses = ref 0.0 and n = ref 0 in
        batches
        |> Seq.iter (fun batch ->
            let s, l = step !state batch in
            state := s;
            losses := !losses +. l;
            incr n);
        Printf.printf "  epoch %d/%d  mean loss %.4f\n%!" epoch epochs
          (!losses /. float_of_int !n)
      done;
      let params, ostate = !state in
      Printf.printf "test accuracy: %.2f%%\n\n" (100. *. accuracy params test);

      (* The optimizer moments and step counter belong in the same checkpoint:
         restoring parameters alone would restart AdamW's history. *)
      let path = Filename.temp_file "mnist-cnn" ".safetensors" in
      save_training_state path (params, ostate);
      Printf.printf "saved checkpoint to %s\n" path;
      let restored_params, restored_ostate = load_training_state path in
      Printf.printf "restored accuracy: %.2f%% (optimizer step %d)\n"
        (100. *. accuracy restored_params test)
        restored_ostate.step;
      Sys.remove path
