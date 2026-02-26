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

(* Collect Data.t of (image, label) pairs into full (x, y) tensors *)
let collect ds =
  let xs = ref [] and ys = ref [] in
  Data.iter
    (fun (x, y) ->
      xs := x :: !xs;
      ys := y :: !ys)
    ds;
  ( Rune.stack ~axis:0 (List.rev !xs),
    Rune.cast Rune.int32 (Rune.stack ~axis:0 (List.rev !ys)) )

let () =
  let rngs = Rune.Rng.key 42 in
  let dtype = Rune.float32 in

  (* Load MNIST into full tensors (once) *)
  Printf.printf "Loading MNIST...\n%!";
  let train_ds, test_ds = Kaun_datasets.mnist () in
  let x_train, y_train = collect train_ds in
  let x_test, y_test = collect test_ds in
  let n_train = (Rune.shape x_train).(0) in
  Printf.printf "  train: %d  test: %d\n%!" n_train (Rune.shape x_test).(0);

  (* Test batches (fixed order, no shuffle) *)
  let test_batches = Data.prepare ~batch_size (x_test, y_test) in

  (* Trainer *)
  let trainer =
    Train.make ~model
      ~optimizer:(Optim.adam ~lr:(Optim.Schedule.constant lr) ())
  in
  let st = ref (Train.init trainer ~rngs ~dtype) in

  for epoch = 1 to epochs do
    let epoch_key = Rune.Rng.fold_in rngs epoch in
    let train_data =
      Data.prepare ~shuffle:epoch_key ~batch_size (x_train, y_train)
      |> Data.map (fun (x, y) ->
             (x, fun logits -> Loss.cross_entropy_sparse logits y))
    in
    let num_batches = n_train / batch_size in
    let tracker = Metric.tracker () in
    st :=
      Train.fit trainer !st ~rngs:epoch_key
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
