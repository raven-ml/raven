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
      slice [| start; 0 |] [| start + batch_size_actual; dim 1 x |] x
    in
    let y_batch =
      slice [| start; 0 |] [| start + batch_size_actual; dim 1 y |] y
    in
    (x_batch, y_batch)

(* Forward pass: computes the MLP output (logits) with shape logging *)
let forward params inputs =
  match params with
  | [ w1; b1; w2; b2 ] ->
      let z1 = add (matmul inputs w1) b1 in
      let a1 = maximum (scalar Float32 0.0) z1 in
      let z2 = add (matmul a1 w2) b2 in
      z2
  | _ -> failwith "Invalid parameters"

(* Cross-entropy loss: computes loss with shape logging *)
let cross_entropy_loss logits y_true =
  let epsilon = 1e-10 in
  let probs = softmax logits in
  let log_probs = log (add probs (scalar Float32 epsilon)) in
  let mul_term = mul y_true log_probs in
  let sum_term = sum mul_term ~axes:[| 1 |] in
  let neg_term = neg sum_term in
  let loss = mean neg_term in
  loss

(* Training function *)
let train_mlp x_train y_train_onehot batch_size learning_rate epochs =
  (* MLP architecture dimensions *)
  let d = 784 in
  (* Input: 28*28 *)
  let h = 128 in
  (* Hidden layer size *)
  let c = 10 in
  (* Output: 10 classes *)

  (* Initialize parameters *)
  let w1 = div (randn Float32 [| d; h |]) (scalar Float32 (sqrt (float d))) in
  let b1 = zeros Float32 [| h |] in
  let w2 = div (randn Float32 [| h; c |]) (scalar Float32 (sqrt (float h))) in
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
      print_endline
      @@ Printf.sprintf "Epoch %d, Batch %d: Loss = %f\n" epoch batch_idx
           (get [||] loss);
      List.combine params grad_params
      |> List.iter (fun (param, grad) ->
             sub_inplace param (mul (scalar Float32 learning_rate) grad)
             |> ignore)
    done
  done;
  params

let () =
  (* Load MNIST dataset *)
  let (x_train, y_train), (x_test, y_test) = Ndarray_datasets.load_mnist () in
  let x_train = ndarray x_train in
  let y_train = ndarray y_train in
  let x_test = ndarray x_test in
  let _y_test = ndarray y_test in

  (* Preprocess training data *)
  let x_train = div (astype Float32 x_train) (scalar Float32 255.0) in
  let x_train = reshape [| 60000; 784 |] x_train in
  let y_train = reshape [| dim 0 y_train |] y_train in
  let y_train_onehot = one_hot Float32 y_train 10 in

  (* Preprocess test data *)
  let x_test = div (astype Float32 x_test) (scalar Float32 255.0) in
  let x_test = reshape [| 10000; 784 |] x_test in
  let _y_test = reshape [| dim 0 _y_test |] _y_test in
  let _y_test_onehot = one_hot Float32 _y_test 10 in

  (* Training parameters *)
  let batch_size = 64 in
  let learning_rate = 0.1 in
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
