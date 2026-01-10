(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rune

(* Configuration *)

type config = {
  num_classes : int;
  input_channels : int;
  input_size : int * int;
  activation : [ `tanh | `relu | `sigmoid ];
  dropout_rate : float option;
}

let default_config =
  {
    num_classes = 10;
    input_channels = 1;
    input_size = (32, 32);
    (* Original LeNet-5 uses 32x32 *)
    activation = `tanh;
    (* Original used tanh *)
    dropout_rate = None;
  }

let mnist_config =
  {
    default_config with
    input_size = (28, 28);
    (* MNIST is 28x28, will be padded *)
  }

let cifar10_config =
  {
    num_classes = 10;
    input_channels = 3;
    (* RGB *)
    input_size = (32, 32);
    activation = `relu;
    (* Modern choice *)
    dropout_rate = Some 0.5;
  }

(* Model Definition *)

type t = Kaun.module_

let create ?(config = default_config) () =
  let open Kaun.Layer in
  (* Select activation function *)
  let activation_fn =
    match config.activation with
    | `tanh -> tanh ()
    | `relu -> relu ()
    | `sigmoid -> sigmoid ()
  in

  (* Build layers *)
  let layers =
    [
      (* First convolutional block *)
      (* Conv1: 6 filters of 5x5 *)
      conv2d ~in_channels:config.input_channels ~out_channels:6
        ~kernel_size:(5, 5) ();
      activation_fn;
      (* Pool1: 2x2 average pooling (original used average, modern uses max) *)
      avg_pool2d ~kernel_size:(2, 2) ~stride:(2, 2) ();
      (* Second convolutional block *)
      (* Conv2: 16 filters of 5x5 *)
      conv2d ~in_channels:6 ~out_channels:16 ~kernel_size:(5, 5) ();
      activation_fn;
      (* Pool2: 2x2 average pooling *)
      avg_pool2d ~kernel_size:(2, 2) ~stride:(2, 2) ();
      (* Flatten for fully connected layers *)
      flatten ();
      (* Fully connected layers *)
      (* The size after conv2 and pool2 depends on input size *)
      (* For 32x32 input: after conv1+pool1: 14x14, after conv2+pool2: 5x5 *)
      (* So flattened size is 16 * 5 * 5 = 400 *)
      (* For 28x28 input: after conv1+pool1: 12x12, after conv2+pool2: 4x4 *)
      (* So flattened size is 16 * 4 * 4 = 256 *)

      (* FC1: 120 units *)
      linear ~in_features:400 ~out_features:120 ();
      (* Assuming 32x32 input *)
      activation_fn;
    ]
    (* Add optional dropout *)
    @ (match config.dropout_rate with
      | Some rate -> [ dropout ~rate () ]
      | None -> [])
    @ [
        (* FC2: 84 units *)
        linear ~in_features:120 ~out_features:84 ();
        activation_fn;
      ]
    (* Add optional dropout *)
    @ (match config.dropout_rate with
      | Some rate -> [ dropout ~rate () ]
      | None -> [])
    @ [
        (* Output layer *)
        linear ~in_features:84 ~out_features:config.num_classes ();
        (* No activation for logits output *)
      ]
  in

  sequential layers

let for_mnist () = create ~config:mnist_config ()
let for_cifar10 () = create ~config:cifar10_config ()

(* Forward Pass *)

let forward ~model ~params ~training ~input =
  Kaun.apply model params ~training input

let extract_features ~model:_ ~params:_ ~input:_ =
  (* TODO: would need to modify model to extract intermediate features *)
  (* For now, just return a dummy tensor *)
  failwith "extract_features not implemented yet"

(* Model Statistics *)

let num_parameters params =
  let tensors = Kaun.Ptree.flatten_with_paths params in
  List.fold_left
    (fun acc (_, tensor) -> acc + Kaun.Ptree.Tensor.numel tensor)
    0 tensors

let parameter_breakdown params =
  let tensors = Kaun.Ptree.flatten_with_paths params in
  let breakdown = Buffer.create 256 in
  Buffer.add_string breakdown "LeNet-5 Parameter Breakdown:\n";
  Buffer.add_string breakdown "============================\n";

  let layer_params = Hashtbl.create 10 in

  (* Group parameters by layer *)
  List.iter
    (fun (path, tensor) ->
      let name = Kaun.Ptree.Path.to_string path in
      let layer_name =
        (* Extract layer name from parameter path *)
        try
          let idx = String.index name '.' in
          String.sub name 0 idx
        with Not_found -> name
      in
      let size = Kaun.Ptree.Tensor.numel tensor in
      let current =
        try Hashtbl.find layer_params layer_name with Not_found -> 0
      in
      Hashtbl.replace layer_params layer_name (current + size))
    tensors;

  (* Print breakdown *)
  Hashtbl.iter
    (fun layer count ->
      Buffer.add_string breakdown
        (Printf.sprintf "  %s: %d parameters\n" layer count))
    layer_params;

  let total = num_parameters params in
  Buffer.add_string breakdown
    (Printf.sprintf "\nTotal: %d parameters (%.2f MB with float32)\n" total
       (float_of_int (total * 4) /. 1024. /. 1024.));

  Buffer.contents breakdown

(* Training Helpers *)

type train_config = {
  learning_rate : float;
  batch_size : int;
  num_epochs : int;
  weight_decay : float option;
  momentum : float option;
}

let default_train_config =
  {
    learning_rate = 0.01;
    batch_size = 64;
    num_epochs = 10;
    weight_decay = Some 0.0001;
    momentum = Some 0.9;
  }

let accuracy ~predictions ~labels =
  (* Get predicted classes *)
  let pred_classes = argmax predictions ~axis:1 in
  (* Cast labels to same type for comparison *)
  let labels_int32 = cast Int32 labels in
  (* Compute accuracy *)
  let correct = equal pred_classes labels_int32 in
  let correct_float = cast Float32 correct in
  let total = float_of_int (Array.get (shape labels) 0) in
  (* Sum and convert to float *)
  let num_correct = sum correct_float in
  let num_correct_scalar =
    (* Extract scalar value - simplified version *)
    match shape num_correct with
    | [||] ->
        (* It's already a scalar, extract the value *)
        let arr = to_array num_correct in
        arr.(0)
    | _ -> failwith "Expected scalar result from sum"
  in
  num_correct_scalar /. total
