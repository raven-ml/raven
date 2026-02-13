(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun

module Dataset = struct
  let inputs =
    Rune.create Rune.float32 [| 4; 2 |] [| 0.; 0.; 0.; 1.; 1.; 0.; 1.; 1. |]

  let labels = Rune.create Rune.float32 [| 4; 1 |] [| 0.; 1.; 1.; 0. |]
end

let model =
  Layer.sequential
    [
      Layer.linear ~in_features:2 ~out_features:4 ();
      Layer.tanh ();
      Layer.linear ~in_features:4 ~out_features:1 ();
      Layer.sigmoid ();
    ]

let forward params x = Kaun.apply model params ~training:false x

let train ?(epochs = 500) ?(learning_rate = 0.1) () =
  let rngs = Rune.Rng.key 42 in
  let params = Kaun.init model ~rngs ~dtype:Rune.float32 in
  let schedule = Optimizer.Schedule.constant learning_rate in
  let optimizer = Optimizer.adam ~lr:schedule () in
  let opt_state = ref (Optimizer.init optimizer params) in

  for epoch = 1 to epochs do
    let loss, grads =
      value_and_grad
        (fun params ->
          let predictions =
            Kaun.apply model params ~training:true Dataset.inputs
          in
          Loss.binary_cross_entropy predictions Dataset.labels)
        params
    in

    let updates, next_state =
      Optimizer.step optimizer !opt_state params grads
    in
    opt_state := next_state;
    Optimizer.apply_updates_inplace params updates;

    if epoch mod 100 = 0 then
      Printf.printf "Epoch %d: loss=%.6f\n%!" epoch (Rune.item [] loss)
  done;

  params

let predict probs_threshold params inputs =
  let logits = forward params inputs in
  let arr = Rune.to_bigarray logits in
  let shape = Bigarray.Genarray.dims arr in
  let batch = shape.(0) in
  Array.init batch (fun i ->
      let prob = Bigarray.Genarray.get arr [| i; 0 |] in
      if prob >= probs_threshold then 1 else 0)

let accuracy predictions =
  let expected = [| 0; 1; 1; 0 |] in
  let correct = ref 0 in
  for i = 0 to Array.length expected - 1 do
    if predictions.(i) = expected.(i) then incr correct
  done;
  Float.of_int !correct /. Float.of_int (Array.length expected)

let () =
  let params = train () in
  let preds = predict 0.5 params Dataset.inputs in
  Printf.printf "\nPredictions: [%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int preds)));
  Printf.printf "Accuracy: %.2f\n" (accuracy preds)
