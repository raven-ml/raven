open Kaun

let rngs = Rune.Rng.key 0

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

let metrics = Metrics.create [ Metrics.avg "loss"; Metrics.accuracy "accuracy" ]
let device = Rune.c

let train () =
  (* Datasets *)
  Printf.printf "Creating datasets...\n%!";
  let start = Unix.gettimeofday () in
  let train_data = Kaun_datasets.mnist ~train:true ~flatten:false ~device () in
  Printf.printf "  MNIST train data loaded in %.2fs\n%!"
    (Unix.gettimeofday () -. start);

  let shuffle_start = Unix.gettimeofday () in
  let train_data_shuffled = Dataset.shuffle ~seed:42 train_data in
  Printf.printf "  Shuffle done in %.2fs\n%!"
    (Unix.gettimeofday () -. shuffle_start);

  let batch_start = Unix.gettimeofday () in
  let train_ds = Dataset.batch_xy 32 train_data_shuffled in
  Printf.printf "  Batching done in %.2fs\n%!"
    (Unix.gettimeofday () -. batch_start);

  Printf.printf "Train dataset created in %.2fs\n%!"
    (Unix.gettimeofday () -. start);

  let start = Unix.gettimeofday () in
  let test_ds =
    Kaun_datasets.mnist ~train:false ~flatten:false ~device ()
    |> Dataset.batch_xy 100
  in
  Printf.printf "Test dataset created in %.2fs\n%!"
    (Unix.gettimeofday () -. start);

  (* Initialize model with dummy input to get params *)
  Printf.printf "Initializing model...\n%!";
  let start = Unix.gettimeofday () in
  let dummy_input = Rune.zeros device Rune.float32 [| 1; 1; 28; 28 |] in
  let params = init ~rngs model dummy_input in
  let optimizer = Optimizer.adam ~lr:0.001 () in
  let opt_state = ref (optimizer.init params) in
  Printf.printf "Model initialized in %.2fs\n%!" (Unix.gettimeofday () -. start);

  (* Training loop *)
  for epoch = 1 to 10 do
    Printf.printf "\nEpoch %d/10\n" epoch;
    let epoch_start = Unix.gettimeofday () in
    Metrics.reset metrics;
    let batch_count = ref 0 in

    (* Training *)
    Printf.printf "Starting training iteration...\n%!";
    Dataset.iter
      (fun (x_batch, y_batch) ->
        incr batch_count;
        if !batch_count = 1 then
          Printf.printf "Got first batch! Shape: X=[%s], Y=[%s]\n%!"
            (String.concat "; "
               (Array.to_list (Array.map string_of_int (Rune.shape x_batch))))
            (String.concat "; "
               (Array.to_list (Array.map string_of_int (Rune.shape y_batch))));
        let batch_start = Unix.gettimeofday () in

        (* Forward and backward pass *)
        let fwd_bwd_start = Unix.gettimeofday () in
        let loss, grads =
          let loss, grads =
            value_and_grad
              (fun params ->
                let logits = apply model params ~training:true x_batch in
                Loss.softmax_cross_entropy_with_indices logits y_batch)
              params
          in
          (loss, grads)
        in
        let fwd_bwd_time = Unix.gettimeofday () -. fwd_bwd_start in

        (* Update weights *)
        let opt_start = Unix.gettimeofday () in
        let updates, new_state = optimizer.update !opt_state params grads in
        opt_state := new_state;
        Optimizer.apply_updates_inplace params updates;
        let opt_time = Unix.gettimeofday () -. opt_start in

        (* Track metrics *)
        let metric_start = Unix.gettimeofday () in
        let logits = apply model params ~training:false x_batch in
        Metrics.update metrics ~loss ~logits ~labels:y_batch ();
        let metric_time = Unix.gettimeofday () -. metric_start in

        let batch_time = Unix.gettimeofday () -. batch_start in

        Printf.printf
          "  Batch %d: %.3fs (fwd+bwd: %.3fs, opt: %.3fs, metric: %.3fs) - \
           Loss: %.4f\n\
           %!"
          !batch_count batch_time fwd_bwd_time opt_time metric_time
          (Rune.unsafe_get [] loss))
      train_ds;

    (* Print training metrics *)
    let train_metrics = Metrics.compute metrics in
    List.iter
      (fun (name, value) -> Printf.printf "  %s: %.4f\n" name value)
      train_metrics;
    Printf.printf "  Epoch time: %.2fs\n" (Unix.gettimeofday () -. epoch_start);

    (* Evaluation *)
    Printf.printf "  Evaluating...\n%!";
    let eval_start = Unix.gettimeofday () in
    Metrics.reset metrics;
    Dataset.iter
      (fun (x_batch, y_batch) ->
        let logits = apply model params ~training:false x_batch in
        let loss = Loss.softmax_cross_entropy_with_indices logits y_batch in
        Metrics.update metrics ~loss ~logits ~labels:y_batch ())
      test_ds;

    (* Print test metrics *)
    Printf.printf "  Test: ";
    List.iter
      (fun (name, value) -> Printf.printf "%s=%.4f " name value)
      (Metrics.compute metrics);
    Printf.printf " (eval time: %.2fs)\n\n%!"
      (Unix.gettimeofday () -. eval_start)
  done

let () = train ()
