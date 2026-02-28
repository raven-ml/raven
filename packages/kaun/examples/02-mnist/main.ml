(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun

let batch_size = 64
let epochs = 3
let lr = 0.001

let model =
  Layer.sequential
    [
      Layer.conv2d ~in_channels:1 ~out_channels:16 ();
      Layer.relu ();
      Layer.max_pool2d ~kernel_size:(2, 2) ();
      Layer.conv2d ~in_channels:16 ~out_channels:32 ();
      Layer.relu ();
      Layer.max_pool2d ~kernel_size:(2, 2) ();
      Layer.flatten ();
      Layer.linear ~in_features:(32 * 7 * 7) ~out_features:128 ();
      Layer.relu ();
      Layer.linear ~in_features:128 ~out_features:10 ();
    ]

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let dtype = Nx.float32 in

  Printf.printf "Loading MNIST...\n%!";
  let (x_train, y_train), (x_test, y_test) = Kaun_datasets.mnist () in
  let n_train = (Nx.shape x_train).(0) in
  Printf.printf "  train: %d  test: %d\n%!" n_train (Nx.shape x_test).(0);

  (* Test batches (fixed order, no shuffle) *)
  let test_batches = Data.prepare ~batch_size (x_test, y_test) in

  (* Trainer *)
  let trainer =
    Train.make ~model
      ~optimizer:(Optim.adam ~lr:(Optim.Schedule.constant lr) ())
  in
  let st = ref (Train.init trainer ~dtype) in

  for epoch = 1 to epochs do
    let train_data =
      Data.prepare ~shuffle:true ~batch_size (x_train, y_train)
      |> Data.map (fun (x, y) ->
          (x, fun logits -> Loss.cross_entropy_sparse logits y))
    in
    let num_batches = n_train / batch_size in
    let tracker = Metric.tracker () in
    st :=
      Train.fit trainer !st
        ~report:(fun ~step ~loss _st ->
          Metric.observe tracker "loss" loss;
          Printf.printf "\r  batch %d/%d  loss: %.4f%!" step num_batches loss)
        train_data;
    Printf.printf "\n%!";

    (* Evaluate *)
    Data.reset test_batches;
    let test_acc =
      Metric.eval
        (fun (x, y) ->
          let logits = Train.predict trainer !st x in
          Metric.accuracy logits y)
        test_batches
    in

    Printf.printf "epoch %d  train_loss: %.4f  test_acc: %.2f%%\n%!" epoch
      (Metric.mean tracker "loss")
      (test_acc *. 100.)
  done;

  Printf.printf "\nDone.\n"
