(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun

let batch_size = 64
let epochs = 3
let lr = 0.001

(* Yield (x_batch, y_batch) slices from full tensors *)
let batches x y =
  let n = (Nx.shape x).(0) in
  let num_batches = n / batch_size in
  Data.of_fn num_batches (fun i ->
      let s = i * batch_size in
      let e = s + batch_size in
      (Nx.slice [ R (s, e) ] x, Nx.slice [ R (s, e) ] y))

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
  let dtype = Nx.float32 in

  (* Set up kaun-board logger *)
  let logger =
    Kaun_board.Log.create ~experiment:"mnist"
      ~config:
        [
          ("lr", Jsont.Json.number lr);
          ("batch_size", Jsont.Json.int batch_size);
          ("epochs", Jsont.Json.int epochs);
        ]
      ()
  in
  Printf.printf "Run ID: %s\n%!" (Kaun_board.Log.run_id logger);
  Printf.printf "To monitor: kaun-board %s\n\n%!" (Kaun_board.Log.run_id logger);

  Printf.printf "Loading MNIST...\n%!";
  let (x_train, y_train), (x_test, y_test) = Kaun_datasets.mnist () in
  let n_train = (Nx.shape x_train).(0) in
  Printf.printf "  train: %d  test: %d\n%!" n_train (Nx.shape x_test).(0);

  (* Test batches (fixed) *)
  let test_batches = batches x_test y_test in

  (* Trainer *)
  let trainer =
    Train.make ~model
      ~optimizer:(Optim.adam ~lr:(Optim.Schedule.constant lr) ())
  in
  let st = ref (Train.init trainer ~dtype) in
  let global_step = ref 0 in

  for epoch = 1 to epochs do
    (* Shuffle training data at the tensor level *)
    let perm = Nx.permutation n_train in
    let x_shuf = Nx.take ~axis:0 perm x_train in
    let y_shuf = Nx.take ~axis:0 perm y_train in
    let train_batches = batches x_shuf y_shuf in

    (* Train *)
    let num_batches = n_train / batch_size in
    let tracker = Metric.tracker () in
    let batch_i = ref 0 in
    Data.iter
      (fun (x, y) ->
        incr batch_i;
        incr global_step;
        let loss_val, st' =
          Train.step trainer !st ~training:true
            ~loss:(fun logits -> Loss.cross_entropy_sparse logits y)
            x
        in
        st := st';
        let loss = Nx.item [] loss_val in
        Metric.observe tracker "loss" loss;
        Kaun_board.Log.log_scalar logger ~step:!global_step ~epoch
          ~tag:"train/loss" loss;
        Printf.printf "\r  batch %d/%d  loss: %.4f%!" !batch_i num_batches loss)
      train_batches;
    Printf.printf "\n%!";

    Data.reset train_batches;
    let train_acc =
      Metric.eval
        (fun (x, y) ->
          let logits = Train.predict trainer !st x in
          Metric.accuracy logits y)
        train_batches
    in
    (* Evaluate *)
    Data.reset test_batches;
    let test_acc =
      Metric.eval
        (fun (x, y) ->
          let logits = Train.predict trainer !st x in
          Metric.accuracy logits y)
        test_batches
    in

    let avg_loss = Metric.mean tracker "loss" in
    Kaun_board.Log.log_scalars logger ~step:!global_step ~epoch
      [
        ("train/loss_avg", avg_loss);
        ("train/accuracy", train_acc);
        ("test/accuracy", test_acc);
      ];

    Printf.printf "epoch %d  train_loss: %.4f  train_acc: %.2f%%  test_acc: %.2f%%\n%!"
      epoch avg_loss (train_acc *. 100.) (test_acc *. 100.)
  done;

  Kaun_board.Log.close logger;
  Printf.printf "\nDone. Run logged to %s\n" (Kaun_board.Log.run_dir logger)
