open Kaun

let rngs = Rngs.create ~seed:0 ()

let model =
  Layer.sequential
    [
      Layer.conv2d ~in_channels:1 ~out_channels:8 ~rngs ();
      Layer.relu ();
      Layer.avg_pool2d ~kernel_size:(2, 2) ();
      Layer.conv2d ~in_channels:8 ~out_channels:16 ~rngs ();
      Layer.relu ();
      Layer.avg_pool2d ~kernel_size:(2, 2) ();
      Layer.flatten ();
      Layer.linear ~in_features:784 ~out_features:128 ~rngs ();
      Layer.relu ();
      Layer.linear ~in_features:128 ~out_features:10 ~rngs ();
    ]

let metrics = Metrics.create [ Metrics.avg "loss"; Metrics.accuracy "accuracy" ]

let train () =
  (* Datasets *)
  let train_ds =
    Kaun_datasets.mnist ~train:true ~flatten:false ()
    |> Dataset.shuffle ~seed:42 |> Dataset.batch_xy 32
  in

  let test_ds =
    Kaun_datasets.mnist ~train:false ~flatten:false () |> Dataset.batch_xy 100
  in

  (* Initialize model with dummy input to get params *)
  let dummy_input = Rune.zeros Rune.cpu Rune.float32 [| 1; 1; 28; 28 |] in
  let params = init model ~rngs dummy_input in
  let optimizer = Optimizer.create (Optimizer.adam ~lr:0.001 ()) in

  (* Training loop *)
  for epoch = 1 to 10 do
    Printf.printf "Epoch %d/10\n" epoch;
    Metrics.reset metrics;

    (* Training *)
    Dataset.iter
      (fun (x_batch, y_batch) ->
        (* Forward and backward pass *)
        let loss, grads =
          value_and_grad
            (fun params ->
              let logits = apply model params ~training:true x_batch in
              Loss.softmax_cross_entropy_with_indices logits y_batch)
            params
        in

        (* Update weights *)
        Optimizer.update optimizer params grads;

        (* Track metrics *)
        let logits = apply model params ~training:false x_batch in
        Metrics.update metrics ~loss ~logits ~labels:y_batch ())
      train_ds;

    (* Print training metrics *)
    let train_metrics = Metrics.compute metrics in
    List.iter
      (fun (name, value) -> Printf.printf "  %s: %.4f\n" name value)
      train_metrics;

    (* Evaluation *)
    Metrics.reset metrics;
    Dataset.iter
      (fun (x_batch, y_batch) ->
        let logits = apply model params ~training:false x_batch in
        let loss = Loss.softmax_cross_entropy logits y_batch in
        Metrics.update metrics ~loss ~logits ~labels:y_batch ())
      test_ds;

    (* Print test metrics *)
    Printf.printf "  Test: ";
    List.iter
      (fun (name, value) -> Printf.printf "%s=%.4f " name value)
      (Metrics.compute metrics);
    Printf.printf "\n\n"
  done

let () = train ()
