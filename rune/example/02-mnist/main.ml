open Rune

(* Get a batch of data *)
let get_batch x y batch_size batch_idx =
  let num_samples = dim 0 x in
  let start = batch_idx * batch_size in
  let end_ = min (start + batch_size) num_samples in
  let x_batch = slice [| start; 0 |] [| end_ - start; dim 1 x |] x in
  let y_batch = slice [| start; 0 |] [| end_ - start; dim 1 y |] y in
  (x_batch, y_batch)

(* Forward pass: computes the MLP output (logits) *)
let forward params inputs =
  match params with
  | [ w1; b1; w2; b2 ] ->
      let z1 = add (matmul inputs w1) b1 in
      let a1 = maximum (scalar Float32 0.0) z1 in
      let z2 = add (matmul a1 w2) b2 in
      z2
  | _ -> failwith "Invalid parameters"

(* Softmax function: converts logits to probabilities *)
let softmax logits =
  let exp_logits = exp logits in
  let sum_exp = sum exp_logits ~axes:[| 1 |] in
  let sum_exp_broadcast = reshape sum_exp [| dim 0 logits; 1 |] in
  div exp_logits sum_exp_broadcast

(* Cross-entropy loss: computes loss between predicted probabilities and true
   labels *)
let cross_entropy_loss logits y_true =
  let epsilon = 1e-10 in
  let probs = softmax logits in
  let log_probs = log (add probs (scalar Float32 epsilon)) in
  let loss = neg (sum (mul y_true log_probs) ~axes:[| 1 |]) in
  mean loss

(* Training function *)
let train_mlp x_train y_train_onehot batch_size learning_rate epochs =
  (* MLP architecture dimensions *)
  (* Input: 28*28 *)
  let d = 784 in
  (* Hidden layer size *)
  let h = 128 in
  (* Output: 10 classes *)
  let c = 10 in

  (* Initialize parameters *)
  let w1 = rand Float32 [| d; h |] in
  let b1 = zeros Float32 [| h |] in
  let w2 = rand Float32 [| h; c |] in
  let b2 = zeros Float32 [| c |] in
  let params = [ w1; b1; w2; b2 ] in

  (* Training loop with mini-batches *)
  let num_samples = dim 0 x_train in
  let num_batches = (num_samples + batch_size - 1) / batch_size in
  for epoch = 1 to epochs do
    for batch_idx = 0 to num_batches - 1 do
      let x_batch, y_batch =
        get_batch x_train y_train_onehot batch_size batch_idx
      in
      let loss_fn params =
        let logits = forward params x_batch in
        cross_entropy_loss logits y_batch
      in
      let loss, grad_params = value_and_grads loss_fn params in
      Printf.printf "Epoch %d, Batch %d: Loss = %f\n" epoch batch_idx
        (get [||] loss);
      List.combine params grad_params
      |> List.iter (fun (param, grad) ->
             sub_inplace param (mul (scalar Float32 learning_rate) grad)
             |> ignore)
    done
  done;
  params

(* Main function *)
let () =
  (* Load MNIST dataset *)
  let (x_train, y_train), (x_test, y_test) = Ndarray_datasets.load_mnist () in
  let x_train = ndarray x_train in
  let y_train = ndarray y_train in
  let x_test = ndarray x_test in
  let _y_test = ndarray y_test in

  (* Preprocess training data *)
  let x_train = div (cast x_train Float32) (scalar Float32 255.0) in
  let x_train = reshape x_train [| 60000; 784 |] in
  let y_train_onehot = one_hot Float32 y_train 10 in

  (* Preprocess test data *)
  let x_test = div (cast x_test Float32) (scalar Float32 255.0) in
  let x_test = reshape x_test [| 10000; 784 |] in
  let _y_test_onehot = one_hot Float32 _y_test 10 in

  (* Training parameters *)
  let batch_size = 64 in
  let learning_rate = 0.01 in
  let epochs = 10 in

  (* Train the MLP *)
  let trained_params =
    train_mlp x_train y_train_onehot batch_size learning_rate epochs
  in

  (* Make predictions on test set *)
  let logits = forward trained_params x_test in
  let probs = softmax logits in
  print_endline "Predicted probabilities on test set after training:";
  print probs
