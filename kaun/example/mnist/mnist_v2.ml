(* MNIST example using improved Kaun APIs *)
open Kaun

let () =
  (* Configuration *)
  let epochs = 10 in
  let batch_size = 32 in
  let learning_rate = 0.001 in

  let dtype = Rune.float32 in
  let rngs = Rune.Rng.key 0 in

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
  Printf.printf "Loading datasets...\n%!";

  let train_data =
    Kaun_datasets.mnist ~train:true ~flatten:false ()
    |> Dataset.prepare ~shuffle_buffer:60000 ~batch_size ~prefetch:2
  in

  let test_data =
    Kaun_datasets.mnist ~train:false ~flatten:false ()
    |> Dataset.prepare ~batch_size:100 ~prefetch:2
  in

  Printf.printf "Datasets ready!\n\n%!";

  (* Training with new high-level API *)
  let state, history =
    Training.fit ~model
      ~optimizer:(Optimizer.adam ~lr:learning_rate ())
      ~loss_fn:Loss.softmax_cross_entropy_with_indices ~metrics ~train_data
      ~val_data:test_data ~epochs ~progress:true ~rngs ~dtype ()
  in

  (* Print final results using new History API *)
  Printf.printf "\n=== Training Complete ===\n";

  (match Training.History.final_train_loss history with
  | Some loss -> Printf.printf "Final train loss: %.4f\n" loss
  | None -> ());

  List.iter
    (fun (name, value) -> Printf.printf "Final train %s: %.4f\n" name value)
    (Training.History.final_train_metrics history);

  (match Training.History.final_val_loss history with
  | Some loss -> Printf.printf "Final val loss: %.4f\n" loss
  | None -> ());

  (match Training.History.best_val_loss history with
  | Some loss -> Printf.printf "Best val loss: %.4f\n" loss
  | None -> ());

  (* State contains final parameters *)
  let _ = state in
  ()
