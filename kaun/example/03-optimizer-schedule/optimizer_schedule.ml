(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun

let dtype = Rune.float32

let build_dataset () =
  let sample_count = 16 in
  let xs = Array.init sample_count (fun i -> float_of_int i /. 5.) in
  let ys = Array.map (fun x -> (3. *. x) +. 1.) xs in
  let inputs = Rune.create dtype [| sample_count; 1 |] xs in
  let targets = Rune.create dtype [| sample_count; 1 |] ys in
  (inputs, targets)

let model = Layer.sequential [ Layer.linear ~in_features:1 ~out_features:1 () ]

let evaluate params inputs targets =
  let preds = Kaun.apply model params ~training:false inputs in
  let loss = Loss.mse preds targets in
  Rune.item [] loss

let extract_weight_bias params =
  match Kaun.Ptree.List.items_exn params with
  | layer :: _ ->
      let fields = Kaun.Ptree.Dict.fields_exn layer in
      let weight =
        Kaun.Ptree.Dict.get_tensor_exn fields ~name:"weight" dtype
        |> Rune.item [ 0; 0 ]
      in
      let bias =
        Kaun.Ptree.Dict.get_tensor_exn fields ~name:"bias" dtype
        |> Rune.item [ 0 ]
      in
      (weight, bias)
  | _ -> failwith "Unexpected parameter tree layout"

let train schedule_name schedule =
  let inputs, targets = build_dataset () in
  let params = Kaun.init model ~rngs:(Rune.Rng.key 0) ~dtype in
  let optimizer = Optimizer.adam ~lr:schedule () in
  let opt_state = ref (Optimizer.init optimizer params) in
  let steps = 200 in
  for step = 1 to steps do
    let loss, new_state =
      let loss, grads =
        value_and_grad
          (fun params ->
            let preds = Kaun.apply model params ~training:true inputs in
            Loss.mse preds targets)
          params
      in
      let updates, new_state =
        Optimizer.step optimizer !opt_state params grads
      in
      Optimizer.apply_updates_inplace params updates;
      (Rune.item [] loss, new_state)
    in
    opt_state := new_state;
    if step mod 40 = 0 then
      Printf.printf "%s step %3d lr=%.4f loss=%.5f\n%!" schedule_name step
        (schedule step) loss
  done;
  let final_loss = evaluate params inputs targets in
  let weight, bias = extract_weight_bias params in
  (final_loss, weight, bias)

let () =
  let schedules =
    [
      ("constant", Optimizer.Schedule.constant 0.05, "Fixed learning rate");
      ( "exp_decay",
        Optimizer.Schedule.exponential_decay ~init_value:0.1 ~decay_rate:0.6
          ~decay_steps:60,
        "Exponential decay" );
      ( "cosine",
        Optimizer.Schedule.cosine_decay ~init_value:0.1 ~decay_steps:120
          ~alpha:0.1 (),
        "Cosine decay with floor" );
    ]
  in
  List.iter
    (fun (name, schedule, description) ->
      Printf.printf "\n=== %s (%s) ===\n" name description;
      Printf.printf "lr[1..5] = %s\n"
        (List.init 5 (fun i -> schedule (i + 1))
        |> List.map (fun lr -> Printf.sprintf "%.4f" lr)
        |> String.concat ", ");
      let final_loss, weight, bias = train name schedule in
      Printf.printf
        "Final loss: %.6f\nEstimated linear model y ≈ %.3fx + %.3f\n" final_loss
        weight bias)
    schedules
