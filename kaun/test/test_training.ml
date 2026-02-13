(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Kaun

let dtype = Rune.float32
let constant_loss _ _ = Rune.scalar dtype 0.

let expect_invalid_argument ~expected f =
  match
    try
      f ();
      None
    with exn -> Some exn
  with
  | Some (Invalid_argument msg) ->
      equal ~msg:"invalid argument message" string expected msg
  | Some exn ->
      failf "Expected Invalid_argument but caught %s"
        (Printexc.to_string exn)
  | None -> fail "Expected Invalid_argument to be raised"

let test_fit_empty_dataset_raises () =
  let model = Layer.relu () in
  let optimizer = Optimizer.identity () in
  let rngs = Rune.Rng.key 0 in
  let train_data = Dataset.from_list [] in
  let run () =
    ignore
      (Training.fit ~model ~optimizer ~loss_fn:constant_loss ~train_data
         ~epochs:1 ~progress:false ~rngs ~dtype ())
  in
  expect_invalid_argument
    ~expected:
      "Training.train_epoch: dataset produced no batches. Ensure your dataset \
       yields at least one batch per epoch."
    run

let test_evaluate_empty_dataset_raises () =
  let model = Layer.relu () in
  let optimizer = Optimizer.identity () in
  let rngs = Rune.Rng.key 1 in
  let metrics = Metrics.Collection.create [] in
  let state = Train_state.init ~model ~optimizer ~metrics ~rngs ~dtype () in
  let dataset = Dataset.from_list [] in
  let run () =
    ignore (Training.evaluate ~model ~state ~dataset ~loss_fn:constant_loss ())
  in
  expect_invalid_argument
    ~expected:
      "Training.evaluate: dataset produced no batches. Ensure your validation \
       dataset yields at least one batch."
    run

let make_counter_metric name =
  let init () = [] in
  let update state ~predictions ~targets:_ ?weights:_ () =
    let dtype = Rune.dtype predictions in
    let current =
      match state with
      | [] -> Rune.scalar dtype 0.
      | [ tensor ] -> tensor
      | _ -> failwith "Unexpected metric state"
    in
    let one = Rune.scalar dtype 1. in
    [ Rune.add current one ]
  in
  let compute = function
    | [ tensor ] -> tensor
    | _ -> failwith "Unexpected metric state"
  in
  let reset = function
    | [ tensor ] -> [ Rune.zeros_like tensor ]
    | _ -> init ()
  in
  Metrics.create_custom ~dtype:Rune.float32 ~name ~init ~update ~compute ~reset

let find_metric_values name metrics =
  match List.assoc_opt name metrics with
  | Some values -> values
  | None -> failf "Metric %s not found in history" name

let test_metric_history_handles_dynamic_metrics () =
  let rngs = Rune.Rng.key 42 in
  let model = Layer.linear ~in_features:1 ~out_features:1 () in
  let optimizer = Optimizer.identity () in
  let metric_a = make_counter_metric "metric_a" in
  let metric_b = make_counter_metric "metric_b" in
  let metrics_collection =
    Metrics.Collection.create [ ("metric_a", metric_a) ]
  in
  let x = Rune.create Rune.float32 [| 1; 1 |] [| 0.5 |] in
  let y = Rune.create Rune.float32 [| 1; 1 |] [| 0.5 |] in
  let train_data = Dataset.from_list [ (x, y) ] in
  let loss_fn predictions targets = Loss.mse predictions targets in
  let callback =
    Training.Callbacks.custom
      ~on_epoch_end:(fun ctx ->
        if ctx.epoch = 1 then
          match ctx.state.metrics with
          | Some collection ->
              Metrics.Collection.add collection "metric_b" metric_b;
              true
          | None -> true
        else true)
      ()
  in
  let _, history =
    Training.fit ~model ~optimizer ~loss_fn ~metrics:metrics_collection
      ~train_data ~epochs:2 ~callbacks:[ callback ] ~progress:false ~rngs ~dtype
      ()
  in
  let open Training.History in
  let metric_a_values = find_metric_values "metric_a" history.train_metrics in
  equal ~msg:"metric_a tracked both epochs" int 2
    (List.length metric_a_values);
  let metric_b_values = find_metric_values "metric_b" history.train_metrics in
  equal ~msg:"metric_b tracked from second epoch" int 1
    (List.length metric_b_values)

let () =
  run "Training tests"
    [
      group "Error handling"
        [
          test "fit raises on empty training dataset"
            test_fit_empty_dataset_raises;
          test "evaluate raises on empty validation dataset"
            test_evaluate_empty_dataset_raises;
        ];
      group "Metric history"
        [
          test "history tolerates dynamic metric set"
            test_metric_history_handles_dynamic_metrics;
        ];
    ]
