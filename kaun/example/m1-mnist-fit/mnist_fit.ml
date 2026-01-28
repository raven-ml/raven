(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* MNIST example using improved Kaun APIs *)
open Kaun

let () =
  (* Configuration *)
  let epochs = 10 in
  let batch_size = 32 in
  let learning_rate = 0.001 in

  let dtype = Rune.float32 in
  let rngs = Rune.Rng.key 0 in

  (* Create logger for monitoring *)
  let logger =
    Log.create ~experiment:"mnist"
      ~config:
        [
          ("epochs", `Int epochs);
          ("batch_size", `Int batch_size);
          ("learning_rate", `Float learning_rate);
        ]
      ()
  in

  (* Define model using explicit dimensions *)
  let model =
    Layer.sequential
      [
        Layer.conv2d ~in_channels:1 ~out_channels:8 ();
        Layer.relu ();
        Layer.avg_pool2d ~kernel_size:(2, 2) ();
        Layer.conv2d ~in_channels:8 ~out_channels:16 ();
        Layer.relu ();
        Layer.avg_pool2d ~kernel_size:(2, 2) ();
        Layer.flatten ();
        Layer.linear ~in_features:784 ~out_features:128 ();
        Layer.relu ();
        Layer.linear ~in_features:128 ~out_features:10 ();
      ]
  in

  (* Create metrics including loss tracking *)
  let metrics =
    Metrics.Collection.create
      [ ("loss", Metrics.loss ()); ("accuracy", Metrics.accuracy ()) ]
  in

  (* Prepare datasets *)
  let train_data =
    Kaun_datasets.mnist ~train:true ~flatten:false ()
    |> Dataset.prepare ~shuffle_buffer:60000 ~batch_size ~prefetch:2
  in

  let test_data =
    Kaun_datasets.mnist ~train:false ~flatten:false ()
    |> Dataset.prepare ~batch_size:100 ~prefetch:2
  in

  (* Info to guide the user, can comment out *)
  Printf.printf "Starting training...\n";
  Printf.printf "Run ID: %s\n" (Log.run_id logger);
  Printf.printf "Run directory: %s\n" (Log.run_dir logger);
  Printf.printf "\n";
  Printf.printf "To monitor this run, open another terminal and run:\n";
  Printf.printf "  dune exec kaun-console\n";
  Printf.printf "\n%!";

  (* Launch dashboard in a separate thread *)
  (* COMMENTED OUT: Console is now a standalone executable *)
  (* let _dashboard_thread =
   *     Thread.create
   *       (fun () -> Kaun_console.run ~runs:[ Log.run_id logger ] ())
   *       ()
   *   in *)

  (* Training with new high-level API *)
  let _state, _history =
    let lr = Optimizer.Schedule.constant learning_rate in
    Training.fit ~model ~optimizer:(Optimizer.adam ~lr ())
      ~loss_fn:Loss.softmax_cross_entropy_with_indices ~metrics ~train_data
      ~val_data:test_data ~epochs ~progress:false
      ~callbacks:[ Training.Callbacks.logging logger ]
      ~rngs ~dtype ()
  in

  (* Close logger - TUI will continue showing final metrics *)
  Log.close logger
