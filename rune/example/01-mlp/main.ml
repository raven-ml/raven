(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rune

(* Forward pass: computes the MLP output *)
let forward params inputs =
  match params with
  | [ w1; b1; w2; b2 ] ->
      (* Input layer to hidden layer *)
      let z1 = add (matmul inputs w1) b1 in
      (* Hidden layer activation *)
      let a1 = maximum (scalar Float32 0.0) z1 in
      (* Hidden layer to output layer *)
      let z2 = add (matmul a1 w2) b2 in
      (* Output layer *)
      z2
  | _ -> failwith "Invalid parameters"

(* Mean Squared Error loss *)
let mse_loss y_pred y_true =
  let diff = sub y_pred y_true in
  let squared_diff = mul diff diff in
  mean squared_diff

(* Training function *)
let train_mlp inputs y_true learning_rate epochs =
  (* Initialize MLP parameters *)
  let d = dim 1 inputs in
  (* Number of input features *)
  let h = 3 in
  (* Hidden layer size *)
  let c = dim 1 y_true in
  (* Number of outputs *)
  let keys = Rng.split (Rng.key 42) in
  let w1 = rand Float32 ~key:keys.(0) [| d; h |] in
  let b1 = zeros Float32 [| h |] in
  let w2 = rand Float32 ~key:keys.(1) [| h; c |] in
  let b2 = zeros Float32 [| c |] in
  let params = [ w1; b1; w2; b2 ] in

  (* Define the loss as a function of parameters *)
  let loss_fn params =
    let y_pred = forward params inputs in
    mse_loss y_pred y_true
  in

  (* Training loop *)
  for epoch = 1 to epochs do
    (* Compute gradients using the provided grad function *)
    let loss, grad_params = value_and_grads loss_fn params in

    Printf.printf "Epoch %d: Loss = %f\n" epoch (item [] loss);

    List.combine params grad_params
    |> List.iter (fun (param, grad) ->
        isub param (mul (scalar Float32 learning_rate) grad) |> ignore)
  done;
  params

(* Example usage *)
let () =
  (* Dummy input data: 4 samples with 2 features *)
  let inputs =
    create Float32 [| 4; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0 |]
  in
  (* Dummy target data: 4 samples with 1 output *)
  let y_true = create Float32 [| 4; 1 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let learning_rate = 0.01 in
  let epochs = 100 in

  (* Train the MLP *)
  let trained_params = train_mlp inputs y_true learning_rate epochs in

  (* Make predictions with trained parameters *)
  let y_pred = forward trained_params inputs in
  print_endline "Predictions after training:";
  print y_pred
