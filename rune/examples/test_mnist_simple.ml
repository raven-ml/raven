(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rune

let () =
  (* Simulate MNIST-like input: batch=2, channels=1, 28x28 *)
  let keys = Rng.split (Rng.key 42) in
  let x = randn Float32 ~key:keys.(0) [| 2; 1; 28; 28 |] in

  (* Conv1: 1 -> 6 channels, 5x5 kernel *)
  let conv1_w = randn Float32 ~key:keys.(1) [| 6; 1; 5; 5 |] in
  let conv1_b = zeros Float32 [| 6 |] in

  Printf.printf "Input shape: %s\n"
    (Array.to_list (shape x) |> List.map string_of_int |> String.concat ", ");

  (* Conv1 + bias + tanh *)
  let conv1_out = convolve2d x conv1_w ~padding_mode:`Valid in
  Printf.printf "After conv1: %s\n"
    (Array.to_list (shape conv1_out)
    |> List.map string_of_int |> String.concat ", ");

  let conv1_out = add conv1_out (reshape [| 1; 6; 1; 1 |] conv1_b) in
  let conv1_out = tanh conv1_out in

  (* MaxPool 2x2 *)
  let pool1_out, _ = max_pool2d conv1_out ~kernel_size:(2, 2) ~stride:(2, 2) in
  Printf.printf "After pool1: %s\n"
    (Array.to_list (shape pool1_out)
    |> List.map string_of_int |> String.concat ", ");

  (* Test gradient computation *)
  let loss_fn params =
    match params with
    | [ conv1_w; conv1_b ] ->
        let conv1_out = convolve2d x conv1_w ~padding_mode:`Valid in
        let conv1_out = add conv1_out (reshape [| 1; 6; 1; 1 |] conv1_b) in
        let conv1_out = tanh conv1_out in
        let pool1_out, _ =
          max_pool2d conv1_out ~kernel_size:(2, 2) ~stride:(2, 2)
        in
        sum pool1_out
    | _ -> failwith "wrong params"
  in

  Printf.printf "\nTesting gradients...\n";
  let loss, grads = value_and_grads loss_fn [ conv1_w; conv1_b ] in
  Printf.printf "Loss: %f\n" (item [] loss);
  Printf.printf "Conv1_w gradient shape: %s\n"
    (Array.to_list (shape (List.nth grads 0))
    |> List.map string_of_int |> String.concat ", ");
  Printf.printf "Conv1_b gradient shape: %s\n"
    (Array.to_list (shape (List.nth grads 1))
    |> List.map string_of_int |> String.concat ", ")
