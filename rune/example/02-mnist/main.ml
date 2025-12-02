open Rune

let () = Printexc.record_backtrace true

(* Get a batch of data *)
let get_batch x y batch_size batch_idx =
  let num_samples = dim 0 x in
  let start = batch_idx * batch_size in
  let end_ = Stdlib.min (start + batch_size) num_samples in
  let batch_size_actual = end_ - start in
  if batch_size_actual <= 0 then failwith "Empty batch encountered"
  else
    let x_batch =
      slice
        [ R (start, start + batch_size_actual); R (0, 1); R (0, 28); R (0, 28) ]
        x
    in
    let y_batch =
      slice [ R (start, start + batch_size_actual); R (0, dim 1 y) ] y
    in
    (x_batch, y_batch)

(* Initialize parameters for LeNet *)
let init_lenet_params =
  let keys = Rng.split ~n:5 (Rng.key 42) in
  (* Conv1: 1 input channel, 6 output channels, 5x5 kernel *)
  let conv1_w =
    div
      (randn Float32 ~key:keys.(0) [| 6; 1; 5; 5 |])
      (scalar Float32 (Stdlib.sqrt 25.0))
  in
  let conv1_b = zeros Float32 [| 6 |] in

  (* Conv2: 6 input channels, 16 output channels, 5x5 kernel *)
  let conv2_w =
    div
      (randn Float32 ~key:keys.(1) [| 16; 6; 5; 5 |])
      (scalar Float32 (Stdlib.sqrt (6.0 *. 25.0)))
  in
  let conv2_b = zeros Float32 [| 16 |] in

  (* FC1: 16*4*4 = 256 inputs, 120 outputs *)
  let fc1_w =
    div
      (randn Float32 ~key:keys.(2) [| 256; 120 |])
      (scalar Float32 (Stdlib.sqrt 256.0))
  in
  let fc1_b = zeros Float32 [| 120 |] in

  (* FC2: 120 inputs, 84 outputs *)
  let fc2_w =
    div
      (randn Float32 ~key:keys.(3) [| 120; 84 |])
      (scalar Float32 (Stdlib.sqrt 120.0))
  in
  let fc2_b = zeros Float32 [| 84 |] in

  (* FC3: 84 inputs, 10 outputs *)
  let fc3_w =
    div
      (randn Float32 ~key:keys.(4) [| 84; 10 |])
      (scalar Float32 (Stdlib.sqrt 84.0))
  in
  let fc3_b = zeros Float32 [| 10 |] in

  [
    conv1_w; conv1_b; conv2_w; conv2_b; fc1_w; fc1_b; fc2_w; fc2_b; fc3_w; fc3_b;
  ]

(* Forward pass for LeNet *)
let forward_lenet params inputs =
  match params with
  | [
   conv1_w; conv1_b; conv2_w; conv2_b; fc1_w; fc1_b; fc2_w; fc2_b; fc3_w; fc3_b;
  ] ->
      (* Conv1 + Tanh + MaxPool *)
      let conv1_out = convolve2d inputs conv1_w ~padding_mode:`Valid in
      let conv1_out = add conv1_out (reshape [| 1; 6; 1; 1 |] conv1_b) in
      let conv1_out = tanh conv1_out in
      let pool1_out, _ =
        max_pool2d conv1_out ~kernel_size:(2, 2) ~stride:(2, 2)
      in

      (* Conv2 + Tanh + MaxPool *)
      let conv2_out = convolve2d pool1_out conv2_w ~padding_mode:`Valid in
      let conv2_out = add conv2_out (reshape [| 1; 16; 1; 1 |] conv2_b) in
      let conv2_out = tanh conv2_out in
      let pool2_out, _ =
        max_pool2d conv2_out ~kernel_size:(2, 2) ~stride:(2, 2)
      in

      (* Flatten for FC layers *)
      let batch_size = dim 0 pool2_out in
      let flattened = reshape [| batch_size; 256 |] pool2_out in

      (* FC1 + Tanh *)
      let fc1_out = add (matmul flattened fc1_w) fc1_b in
      let fc1_out = tanh fc1_out in

      (* FC2 + Tanh *)
      let fc2_out = add (matmul fc1_out fc2_w) fc2_b in
      let fc2_out = tanh fc2_out in

      (* FC3 (logits) *)
      let logits = add (matmul fc2_out fc3_w) fc3_b in
      logits
  | _ -> failwith "Invalid parameters"

(* Cross-entropy loss *)
let cross_entropy_loss logits y_true =
  let epsilon = 1e-10 in
  let probs = softmax logits ~axes:[ 1 ] in
  let log_probs = log (add probs (scalar Float32 epsilon)) in
  let mul_term = mul y_true log_probs in
  let sum_term = sum mul_term ~axes:[ 1 ] in
  let neg_term = neg sum_term in
  let loss = mean neg_term in
  loss

(* Calculate accuracy *)
let accuracy predictions labels =
  let pred_classes = argmax predictions ~axis:1 ~keepdims:false in
  let correct = equal pred_classes labels in
  let correct_float = astype Float32 correct in
  let acc = mean correct_float in
  item [] acc

(* Training function *)
let train_lenet x_train y_train_onehot y_train_labels x_test y_test_onehot
    y_test_labels batch_size learning_rate epochs =
  (* Initialize parameters *)
  let params = init_lenet_params in

  (* Training loop *)
  let num_samples = dim 0 x_train in
  let num_batches = (num_samples + batch_size - 1) / batch_size in

  for epoch = 1 to epochs do
    let epoch_loss = ref 0.0 in
    let epoch_correct = ref 0 in
    let epoch_samples = ref 0 in

    for batch_idx = 0 to Stdlib.min 20 (num_batches - 1) do
      (* Just train on first 20 batches for testing *)
      let x_batch, y_batch =
        get_batch x_train y_train_onehot batch_size batch_idx
      in
      let y_labels_batch =
        slice
          [
            R
              ( batch_idx * batch_size,
                Stdlib.min ((batch_idx + 1) * batch_size) num_samples );
          ]
          y_train_labels
      in

      (* Forward and backward pass *)
      let loss_fn params =
        let logits = forward_lenet params x_batch in
        cross_entropy_loss logits y_batch
      in

      let loss, grad_params = value_and_grads loss_fn params in

      (* Track metrics *)
      epoch_loss := !epoch_loss +. item [] loss;
      let logits = forward_lenet params x_batch in
      let pred_classes = argmax logits ~axis:1 ~keepdims:false in
      let correct = equal pred_classes y_labels_batch in
      let correct_count =
        item [] (sum (astype Float32 correct)) |> int_of_float
      in
      epoch_correct := !epoch_correct + correct_count;
      epoch_samples := !epoch_samples + dim 0 x_batch;

      (* Update parameters with gradient clipping *)
      List.combine params grad_params
      |> List.iter (fun (param, grad) ->
             (* Clip gradients to prevent NaN *)
             let grad_clipped = clip grad ~min:(-1.0) ~max:1.0 in
             isub param (mul (scalar Float32 learning_rate) grad_clipped)
             |> ignore);

      (* Print progress *)
      Printf.printf "Epoch %d, Batch %d/%d: Loss = %.4f\n%!" epoch batch_idx
        num_batches (item [] loss)
    done;

    (* Evaluate on test set *)
    let test_logits = forward_lenet params x_test in
    let test_loss = cross_entropy_loss test_logits y_test_onehot in
    let test_acc = accuracy test_logits y_test_labels in

    Printf.printf
      "Epoch %d: Train Loss = %.4f, Train Acc = %.2f%%, Test Loss = %.4f, Test \
       Acc = %.2f%%\n\
       %!"
      epoch
      (!epoch_loss /. float num_batches)
      (100.0 *. float !epoch_correct /. float !epoch_samples)
      (item [] test_loss) (100.0 *. test_acc)
  done;
  params

let () =
  (* Load MNIST dataset *)
  let (x_train, y_train), (x_test, y_test) = Nx_datasets.load_mnist () in

  (* Convert to Rune tensors and preprocess *)
  let x_train = of_bigarray (Nx.to_bigarray x_train) in
  let y_train = of_bigarray (Nx.to_bigarray y_train) in
  let x_test = of_bigarray (Nx.to_bigarray x_test) in
  let y_test = of_bigarray (Nx.to_bigarray y_test) in

  (* Normalize and reshape to NCHW format (batch, channels, height, width) *)
  let x_train = div (astype Float32 x_train) (scalar Float32 255.0) in
  let x_train = reshape [| 60000; 1; 28; 28 |] x_train in

  let x_test = div (astype Float32 x_test) (scalar Float32 255.0) in
  let x_test = reshape [| 10000; 1; 28; 28 |] x_test in

  (* Prepare labels *)
  let y_train_labels = reshape [| dim 0 y_train |] y_train in
  let y_train_labels = astype Int32 y_train_labels in
  let y_train_onehot = one_hot ~num_classes:10 y_train_labels in
  let y_train_onehot = astype Float32 y_train_onehot in

  let y_test_labels = reshape [| dim 0 y_test |] y_test in
  let y_test_labels = astype Int32 y_test_labels in
  let y_test_onehot = one_hot ~num_classes:10 y_test_labels in
  let y_test_onehot = astype Float32 y_test_onehot in

  (* Training parameters *)
  let batch_size = 64 in
  let learning_rate = 0.001 in
  (* Lower learning rate for CNN *)
  let epochs = 1 in

  Printf.printf "Starting LeNet training on MNIST...\n";
  Printf.printf "Training samples: %d, Test samples: %d\n%!" (dim 0 x_train)
    (dim 0 x_test);

  (* Train the model *)
  let _trained_params =
    train_lenet x_train y_train_onehot y_train_labels x_test y_test_onehot
      y_test_labels batch_size learning_rate epochs
  in

  Printf.printf "\nTraining completed!\n"
