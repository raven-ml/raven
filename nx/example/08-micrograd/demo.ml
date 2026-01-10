(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Demo.ml - Micrograd demo with nx-datasets *)

open Nx
module Engine = Micrograd.Engine
module Nn = Micrograd.Nn

(** Convert nx tensor to engine values *)
let tensor_to_values tensor =
  let shape = Nx.shape tensor in
  let n_samples = shape.(0) in
  let n_features = shape.(1) in
  let data = Nx.to_array tensor in

  Array.init n_samples (fun i ->
      List.init n_features (fun j ->
          let idx = (i * n_features) + j in
          Engine.scalar float32 data.(idx)))

(** Convert labels to -1 or 1 for SVM loss *)
let prepare_labels labels =
  let labels_arr = Nx.to_array labels |> Array.map Int32.to_int in
  Array.map (fun y -> if y = 0 then -1.0 else 1.0) labels_arr

(** SVM max-margin loss function *)
let loss model x_batch y_batch =
  let batch_size = Array.length x_batch in

  (* Forward pass to get scores *)
  let scores = Array.map (fun x -> Nn.MLP.call model x) x_batch in

  (* Extract scalar from single-output network *)
  let scores_flat =
    Array.map
      (fun score_list ->
        match score_list with
        | [ score ] -> score
        | _ -> failwith "Expected single output from model")
      scores
  in

  (* SVM loss: max(0, 1 - y * score) *)
  let losses =
    Array.map2
      (fun y score ->
        let y_val = Engine.scalar float32 y in
        let one = Engine.scalar float32 1.0 in
        let margin = Engine.(one - (y_val * score)) in
        Engine.relu margin)
      y_batch scores_flat
  in

  (* Average loss *)
  let data_loss =
    let sum = Array.fold_left Engine.( + ) (Engine.scalar float32 0.0) losses in
    let batch_scalar = Engine.scalar float32 (float_of_int batch_size) in
    Engine.(sum / batch_scalar)
  in

  (* L2 regularization *)
  let alpha = 1e-4 in
  let reg_loss =
    let params = Nn.MLP.parameters model in
    let squared_params = List.map (fun p -> Engine.(p * p)) params in
    let sum_sq =
      List.fold_left Engine.( + ) (Engine.scalar float32 0.0) squared_params
    in
    let alpha_scalar = Engine.scalar float32 alpha in
    Engine.(alpha_scalar * sum_sq)
  in

  (* Total loss *)
  let total_loss = Engine.(data_loss + reg_loss) in

  (* Compute accuracy *)
  let predictions =
    Array.map (fun score -> Engine.item score > 0.0) scores_flat
  in
  let correct = Array.map2 (fun y pred -> y > 0.0 = pred) y_batch predictions in
  let accuracy =
    let correct_count =
      Array.fold_left (fun acc c -> if c then acc + 1 else acc) 0 correct
    in
    float_of_int correct_count /. float_of_int batch_size
  in

  (total_loss, accuracy)

(** Create mini-batch *)
let get_batch x_data y_data batch_size =
  let n_samples = Array.length x_data in

  if batch_size >= n_samples then (x_data, y_data)
  else
    let indices = Array.init batch_size (fun _ -> Random.int n_samples) in
    let x_batch = Array.map (fun i -> x_data.(i)) indices in
    let y_batch = Array.map (fun i -> y_data.(i)) indices in
    (x_batch, y_batch)

(** Evaluate model on test data *)
let evaluate_model model x_data y_data =
  (* Make predictions *)
  let predictions =
    Array.mapi
      (fun i x ->
        let output = Nn.MLP.call model x in
        let score =
          match output with
          | [ s ] -> Engine.item s
          | _ -> failwith "Expected single output"
        in
        Printf.printf
          "Sample %2d: features = [%.3f, %.3f], true label = %2.0f, score = \
           %6.3f, predicted = %2d\n"
          i
          (Engine.item (List.nth x 0))
          (Engine.item (List.nth x 1))
          y_data.(i) score
          (if score > 0.0 then 1 else -1))
      x_data
  in

  predictions

(** Main demo *)
let () =
  Random.self_init ();

  (* Generate moon dataset *)
  Printf.printf
    "=== Micrograd Demo: Binary Classification with Moon Dataset ===\n\n%!";
  Printf.printf "Generating moon dataset...\n%!";
  let x, y = Nx_datasets.make_moons ~n_samples:100 ~noise:0.1 () in
  Printf.printf "Done generating dataset!\n%!";

  (* Convert to appropriate format *)
  let x_data = tensor_to_values x in
  let y_labels = prepare_labels y in

  (* Initialize model: 2 inputs -> 16 hidden -> 16 hidden -> 1 output *)
  Printf.printf "\nInitializing MLP model...\n";
  let model = Nn.MLP.create 2 [ 16; 16; 1 ] in
  Printf.printf "Model architecture: %s\n" (Nn.MLP.to_string model);
  Printf.printf "Number of parameters: %d\n\n"
    (List.length (Nn.MLP.parameters model));

  (* Training loop *)
  Printf.printf "Training for 100 epochs...\n";
  Printf.printf "%-10s %-10s %-10s\n" "Epoch" "Loss" "Accuracy";
  Printf.printf "%s\n" (String.make 35 '-');

  let n_epochs = 100 in
  let batch_size = None in
  (* Use full batch *)

  for epoch = 0 to n_epochs - 1 do
    (* Compute learning rate with decay *)
    let learning_rate =
      1.0 -. (0.9 *. float_of_int epoch /. float_of_int n_epochs)
    in

    (* Get batch (full dataset if batch_size is None) *)
    let x_batch, y_batch =
      match batch_size with
      | None -> (x_data, y_labels)
      | Some bs -> get_batch x_data y_labels bs
    in

    (* Train step *)
    let loss_val, acc = loss model x_batch y_batch in
    Nn.MLP.zero_grad model;
    Engine.backward loss_val;

    (* Update parameters *)
    let params = Nn.MLP.parameters model in
    List.iter
      (fun p ->
        let grad = !(p.Engine.grad) in
        let lr_scalar = Nx.scalar (Nx.dtype p.Engine.data) learning_rate in
        let update = Nx.mul lr_scalar grad in
        p.Engine.data <- Nx.sub p.Engine.data update)
      params;

    (* Print progress *)
    if epoch mod 10 = 0 || epoch = n_epochs - 1 then
      Printf.printf "%-10d %-10.4f %-10.1f%%\n" epoch (Engine.item loss_val)
        (acc *. 100.0)
  done;

  Printf.printf "\nTraining complete!\n\n";

  (* Show some sample predictions *)
  Printf.printf "Sample predictions (first 10 examples):\n";
  Printf.printf "%s\n" (String.make 80 '-');
  let sample_x = Array.sub x_data 0 (Int.min 10 (Array.length x_data)) in
  let sample_y = Array.sub y_labels 0 (Int.min 10 (Array.length y_labels)) in
  let _ = evaluate_model model sample_x sample_y in

  Printf.printf "\n=== Demo complete! ===\n";
  Printf.printf
    "\nThe model has successfully learned to classify the moon dataset.\n";
  Printf.printf
    "This demonstrates automatic differentiation and gradient descent in OCaml!\n";

  (* Visualization with Hugin *)
  Printf.printf "\nGenerating visualization...\n";

  (* Create figure *)
  let fig = Hugin.Figure.create ~width:800 ~height:600 () in
  let ax = Hugin.Figure.add_subplot fig in

  (* Plot data points *)
  let x_coords = Nx.slice [ Nx.A; Nx.I 0 ] x in
  let y_coords = Nx.slice [ Nx.A; Nx.I 1 ] x in

  (* Separate positive and negative classes *)
  let pos_mask = Array.mapi (fun _i label -> label > 0.0) y_labels in
  let neg_mask = Array.mapi (fun _i label -> label <= 0.0) y_labels in

  let pos_indices =
    Array.to_list @@ Array.mapi (fun i b -> if b then Some i else None) pos_mask
    |> List.filter_map (fun x -> x)
  in
  let neg_indices =
    Array.to_list @@ Array.mapi (fun i b -> if b then Some i else None) neg_mask
    |> List.filter_map (fun x -> x)
  in

  (* Create arrays for positive and negative samples *)
  let filter_by_indices indices arr =
    let values =
      List.map
        (fun idx ->
          (* Get the value at index idx from the 1D array *)
          let scalar_tensor = Nx.get [ idx ] arr in
          Nx.to_array scalar_tensor |> fun a -> a.(0))
        indices
    in
    Nx.create float32 [| List.length indices |] (Array.of_list values)
  in

  let pos_x = filter_by_indices pos_indices x_coords in
  let pos_y = filter_by_indices pos_indices y_coords in
  let neg_x = filter_by_indices neg_indices x_coords in
  let neg_y = filter_by_indices neg_indices y_coords in

  (* Plot scatter points *)
  let ax =
    Hugin.Plotting.scatter ~x:pos_x ~y:pos_y ~c:Hugin.Artist.Color.blue
      ~marker:Hugin.Artist.Circle ~label:"Class 1" ~s:8.0 ax
  in
  let ax =
    Hugin.Plotting.scatter ~x:neg_x ~y:neg_y ~c:Hugin.Artist.Color.red
      ~marker:Hugin.Artist.Circle ~label:"Class -1" ~s:8.0 ax
  in

  (* Add labels and title *)
  let ax = Hugin.Axes.set_xlabel "Feature 1" ax in
  let ax = Hugin.Axes.set_ylabel "Feature 2" ax in
  let ax =
    Hugin.Axes.set_title "Micrograd Binary Classification - Moon Dataset" ax
  in
  let _ax = Hugin.Axes.grid true ax in

  (* Save the figure *)
  Hugin.savefig "micrograd_moon_dataset.png" fig;
  Printf.printf "Visualization saved to micrograd_moon_dataset.png\n";

  Printf.printf
    "\n\
     NOTE: Hugin doesn't currently support contour plots, so the decision \
     boundary\n";
  Printf.printf
    "visualization is limited to showing the data points only. To add decision\n";
  Printf.printf
    "boundary visualization, Hugin would need contour/contourf functionality.\n"
