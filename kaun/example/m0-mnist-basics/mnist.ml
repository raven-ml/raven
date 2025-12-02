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

let metrics =
  Metrics.Collection.create
    [ ("loss", Metrics.loss ()); ("accuracy", Metrics.accuracy ()) ]

let train () =
  (* Datasets *)
  Printf.printf "Creating datasets...\n%!";
  let start = Unix.gettimeofday () in
  let train_data = Kaun_datasets.mnist ~train:true ~flatten:false () in
  Printf.printf "  MNIST train data loaded in %.2fs\n%!"
    (Unix.gettimeofday () -. start);
  let test_load_start = Unix.gettimeofday () in
  let test_data = Kaun_datasets.mnist ~train:false ~flatten:false () in
  Printf.printf "Test data loaded in %.2fs\n%!"
    (Unix.gettimeofday () -. test_load_start);
  let test_ds =
    let start = Unix.gettimeofday () in
    let ds =
      Kaun.Dataset.batch_map 100 (fun batch ->
          let images, labels = Array.split batch in
          let batched_images = Rune.stack ~axis:0 (Array.to_list images) in
          let batched_labels = Rune.stack ~axis:0 (Array.to_list labels) in
          (batched_images, batched_labels))
        test_data
    in
    Printf.printf "Test dataset created in %.2fs\n%!"
      (Unix.gettimeofday () -. start);
    ds
  in

  let make_train_ds ~epoch =
    Kaun.Dataset.reset train_data;
    let ds_start = Unix.gettimeofday () in
    let shuffle_key = Rune.Rng.fold_in rngs epoch in
    let shuffle_start = Unix.gettimeofday () in
    let train_data_shuffled =
      Kaun.Dataset.shuffle ~rng:shuffle_key ~buffer_size:60000 train_data
    in
    Printf.printf "  Shuffle done in %.2fs\n%!"
      (Unix.gettimeofday () -. shuffle_start);

    let batch_start = Unix.gettimeofday () in
    let train_ds =
      Kaun.Dataset.batch_map 32
        (fun batch ->
          let images, labels = Array.split batch in
          let batched_images = Rune.stack ~axis:0 (Array.to_list images) in
          let batched_labels = Rune.stack ~axis:0 (Array.to_list labels) in
          (batched_images, batched_labels))
        train_data_shuffled
    in
    Printf.printf "  Batching done in %.2fs\n%!"
      (Unix.gettimeofday () -. batch_start);
    Printf.printf "  Train dataset created in %.2fs\n%!"
      (Unix.gettimeofday () -. ds_start);
    train_ds
  in

  (* Initialize model with dummy input to get params *)
  Printf.printf "Initializing model...\n%!";
  let start = Unix.gettimeofday () in
  let params = Kaun.init model ~rngs ~dtype:Rune.float32 in
  let lr = Optimizer.Schedule.constant 0.001 in
  let optimizer = Optimizer.adam ~lr () in
  let opt_state = ref (Optimizer.init optimizer params) in
  Printf.printf "Model initialized in %.2fs\n%!" (Unix.gettimeofday () -. start);

  (* Training loop *)
  for epoch = 1 to 10 do
    Printf.printf "\nEpoch %d/10\n" epoch;
    let epoch_start = Unix.gettimeofday () in
    Metrics.Collection.reset metrics;
    let batch_count = ref 0 in

    (* Training *)
    Printf.printf "Starting training iteration...\n%!";
    let train_ds = make_train_ds ~epoch in
    Kaun.Dataset.iter
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
                let logits = Kaun.apply model params ~training:true x_batch in
                Loss.softmax_cross_entropy_with_indices logits y_batch)
              params
          in
          (loss, grads)
        in
        let fwd_bwd_time = Unix.gettimeofday () -. fwd_bwd_start in

        (* Update weights *)
        let opt_start = Unix.gettimeofday () in
        let updates, new_state =
          Optimizer.step optimizer !opt_state params grads
        in
        opt_state := new_state;
        Optimizer.apply_updates_inplace params updates;
        let opt_time = Unix.gettimeofday () -. opt_start in

        (* Track metrics *)
        let metric_start = Unix.gettimeofday () in
        let logits = Kaun.apply model params ~training:false x_batch in
        (* Update metrics - need to compute predictions from logits *)
        let predictions = Rune.softmax logits ~axes:[ -1 ] in
        Metrics.Collection.update metrics ~loss ~predictions ~targets:y_batch ();
        let metric_time = Unix.gettimeofday () -. metric_start in

        let batch_time = Unix.gettimeofday () -. batch_start in

        Printf.printf
          "  Batch %d: %.3fs (fwd+bwd: %.3fs, opt: %.3fs, metric: %.3fs) - \
           Loss: %.4f\n\
           %!"
          !batch_count batch_time fwd_bwd_time opt_time metric_time
          (Rune.item [] loss))
      train_ds;

    (* Print training metrics *)
    let train_metrics = Metrics.Collection.compute metrics in
    List.iter
      (fun (name, value) -> Printf.printf "  %s: %.4f\n" name value)
      train_metrics;
    Printf.printf "  Epoch time: %.2fs\n" (Unix.gettimeofday () -. epoch_start);

    (* Evaluation *)
    Printf.printf "  Evaluating...\n%!";
    Kaun.Dataset.reset test_ds;
    let eval_start = Unix.gettimeofday () in
    Metrics.Collection.reset metrics;
    Kaun.Dataset.iter
      (fun (x_batch, y_batch) ->
        let logits = Kaun.apply model params ~training:false x_batch in
        (* Update metrics with predictions instead of loss/logits/labels *)
        let predictions = Rune.softmax logits ~axes:[ -1 ] in
        let loss = Loss.softmax_cross_entropy_with_indices logits y_batch in
        Metrics.Collection.update metrics ~loss ~predictions ~targets:y_batch ())
      test_ds;

    (* Print test metrics *)
    Printf.printf "  Test: ";
    List.iter
      (fun (name, value) -> Printf.printf "%s=%.4f " name value)
      (Metrics.Collection.compute metrics);
    Printf.printf " (eval time: %.2fs)\n\n%!"
      (Unix.gettimeofday () -. eval_start)
  done

let () = train ()
