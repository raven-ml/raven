(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Metric = Kaun.Metric

(* Tracker *)

let test_tracker_observe_and_mean () =
  let t = Metric.tracker () in
  Metric.observe t "loss" 1.0;
  Metric.observe t "loss" 3.0;
  equal ~msg:"mean of two" (float 1e-10) 2.0 (Metric.mean t "loss")

let test_tracker_count () =
  let t = Metric.tracker () in
  Metric.observe t "acc" 0.9;
  Metric.observe t "acc" 0.8;
  Metric.observe t "acc" 0.7;
  equal ~msg:"count" int 3 (Metric.count t "acc")

let test_tracker_not_found () =
  let t = Metric.tracker () in
  raises Not_found (fun () -> ignore (Metric.mean t "missing"));
  raises Not_found (fun () -> ignore (Metric.count t "missing"))

let test_tracker_reset () =
  let t = Metric.tracker () in
  Metric.observe t "x" 1.0;
  Metric.reset t;
  equal ~msg:"empty after reset"
    (list (pair string (float 1e-10)))
    [] (Metric.to_list t)

let test_tracker_to_list_sorted () =
  let t = Metric.tracker () in
  Metric.observe t "loss" 0.5;
  Metric.observe t "accuracy" 0.9;
  Metric.observe t "lr" 0.001;
  let names = List.map fst (Metric.to_list t) in
  equal ~msg:"sorted by name" (list string) [ "accuracy"; "loss"; "lr" ] names

let test_tracker_summary () =
  let t = Metric.tracker () in
  Metric.observe t "loss" 0.4;
  Metric.observe t "accuracy" 0.9;
  let s = Metric.summary t in
  (* sorted: accuracy before loss *)
  equal ~msg:"summary format" string "accuracy: 0.9000  loss: 0.4000" s

(* Dataset evaluation *)

let test_eval_mean () =
  let data = Kaun.Data.of_array [| 2.0; 4.0; 6.0 |] in
  let result = Metric.eval Fun.id data in
  equal ~msg:"eval mean" (float 1e-10) 4.0 result

let test_eval_empty_raises () =
  let data = Kaun.Data.of_array [||] in
  raises_invalid_arg "Metric.eval: empty dataset" (fun () ->
      ignore (Metric.eval Fun.id data))

let test_eval_many () =
  let data = Kaun.Data.of_array [| 1.0; 3.0 |] in
  let result =
    Metric.eval_many
      (fun x -> [ ("double", x *. 2.0); ("half", x /. 2.0) ])
      data
  in
  equal ~msg:"double" (float 1e-10) 4.0 (List.assoc "double" result);
  equal ~msg:"half" (float 1e-10) 1.0 (List.assoc "half" result)

let test_eval_many_empty_raises () =
  let data = Kaun.Data.of_array [||] in
  raises_invalid_arg "Metric.eval_many: empty dataset" (fun () ->
      ignore (Metric.eval_many (fun x -> [ ("v", x) ]) data))

(* Accuracy *)

let test_accuracy_multiclass () =
  (* logits: batch=4, classes=3 *)
  let predictions =
    Rune.create Rune.float32 [| 4; 3 |]
      [|
        (* predicted class 2 *) 0.1;
        0.2;
        0.7;
        (* predicted class 0 *) 0.9;
        0.05;
        0.05;
        (* predicted class 1 *) 0.1;
        0.8;
        0.1;
        (* predicted class 0 *) 0.6;
        0.2;
        0.2;
      |]
  in
  (* targets: class indices *)
  let targets = Rune.create Rune.int32 [| 4 |] [| 2l; 0l; 0l; 0l |] in
  (* correct: sample 0 (2=2), sample 1 (0=0), sample 3 (0=0) = 3/4 *)
  equal ~msg:"multiclass accuracy" (float 1e-6) 0.75
    (Metric.accuracy predictions targets)

let test_accuracy_binary () =
  let predictions = Rune.create Rune.float32 [| 4 |] [| 0.8; 0.3; 0.6; 0.1 |] in
  let targets = Rune.create Rune.int32 [| 4 |] [| 1l; 0l; 1l; 1l |] in
  (* predicted: 1, 0, 1, 0; targets: 1, 0, 1, 1 => 3/4 correct *)
  equal ~msg:"binary accuracy" (float 1e-6) 0.75
    (Metric.accuracy predictions targets)

let test_binary_accuracy_default_threshold () =
  let predictions = Rune.create Rune.float32 [| 4 |] [| 0.8; 0.3; 0.6; 0.1 |] in
  let targets = Rune.create Rune.float32 [| 4 |] [| 1.0; 0.0; 1.0; 1.0 |] in
  equal ~msg:"binary_accuracy default" (float 1e-6) 0.75
    (Metric.binary_accuracy predictions targets)

let test_binary_accuracy_custom_threshold () =
  let predictions = Rune.create Rune.float32 [| 4 |] [| 0.8; 0.3; 0.6; 0.1 |] in
  let targets = Rune.create Rune.float32 [| 4 |] [| 1.0; 1.0; 1.0; 0.0 |] in
  (* threshold=0.25: predicted 1, 1, 1, 0; targets: 1, 1, 1, 0 => 4/4 *)
  equal ~msg:"binary_accuracy threshold=0.25" (float 1e-6) 1.0
    (Metric.binary_accuracy ~threshold:0.25 predictions targets)

(* Precision / Recall / F1 *)

(* Test scenario: 3 classes, 6 samples. predictions (logits): argmax gives [0;
   1; 0; 2; 1; 0] targets: [0; 1; 1; 2; 0; 0]

   Confusion per class: class 0: TP=2, FP=1, FN=1, support=3 class 1: TP=1,
   FP=1, FN=1, support=2 class 2: TP=1, FP=0, FN=0, support=1

   Per-class precision: [2/3; 1/2; 1/1] Per-class recall: [2/3; 1/2; 1/1]
   Per-class f1: [2/3; 1/2; 1/1] *)

let prf_predictions () =
  Rune.create Rune.float32 [| 6; 3 |]
    [|
      (* pred 0 *) 0.8;
      0.1;
      0.1;
      (* pred 1 *) 0.1;
      0.7;
      0.2;
      (* pred 0 *) 0.6;
      0.3;
      0.1;
      (* pred 2 *) 0.1;
      0.2;
      0.7;
      (* pred 1 *) 0.2;
      0.6;
      0.2;
      (* pred 0 *) 0.5;
      0.3;
      0.2;
    |]

let prf_targets () = Rune.create Rune.int32 [| 6 |] [| 0l; 1l; 1l; 2l; 0l; 0l |]

let test_precision_macro () =
  let predictions = prf_predictions () in
  let targets = prf_targets () in
  (* macro = mean(2/3, 1/2, 1/1) = (2/3 + 1/2 + 1) / 3 *)
  let expected = ((2.0 /. 3.0) +. (1.0 /. 2.0) +. 1.0) /. 3.0 in
  equal ~msg:"precision macro" (float 1e-6) expected
    (Metric.precision Macro predictions targets)

let test_precision_micro () =
  let predictions = prf_predictions () in
  let targets = prf_targets () in
  (* micro = sum(TP) / (sum(TP) + sum(FP)) = 4 / (4 + 2) = 2/3 *)
  equal ~msg:"precision micro" (float 1e-6) (4.0 /. 6.0)
    (Metric.precision Micro predictions targets)

let test_precision_weighted () =
  let predictions = prf_predictions () in
  let targets = prf_targets () in
  (* weighted = (3 * 2/3 + 2 * 1/2 + 1 * 1) / 6 = (2 + 1 + 1) / 6 = 2/3 *)
  let expected =
    ((3.0 *. 2.0 /. 3.0) +. (2.0 *. 1.0 /. 2.0) +. (1.0 *. 1.0)) /. 6.0
  in
  equal ~msg:"precision weighted" (float 1e-6) expected
    (Metric.precision Weighted predictions targets)

let test_recall_macro () =
  let predictions = prf_predictions () in
  let targets = prf_targets () in
  let expected = ((2.0 /. 3.0) +. (1.0 /. 2.0) +. 1.0) /. 3.0 in
  equal ~msg:"recall macro" (float 1e-6) expected
    (Metric.recall Macro predictions targets)

let test_recall_micro () =
  let predictions = prf_predictions () in
  let targets = prf_targets () in
  (* micro recall = sum(TP) / (sum(TP) + sum(FN)) = 4 / (4 + 2) = 2/3 *)
  equal ~msg:"recall micro" (float 1e-6) (4.0 /. 6.0)
    (Metric.recall Micro predictions targets)

let test_f1_macro () =
  let predictions = prf_predictions () in
  let targets = prf_targets () in
  (* per-class f1 = [2/3; 1/2; 1] *)
  let expected = ((2.0 /. 3.0) +. (1.0 /. 2.0) +. 1.0) /. 3.0 in
  equal ~msg:"f1 macro" (float 1e-6) expected
    (Metric.f1 Macro predictions targets)

let test_f1_micro () =
  let predictions = prf_predictions () in
  let targets = prf_targets () in
  (* micro f1 = 2*sum(TP) / (2*sum(TP) + sum(FP) + sum(FN)) = 2*4 / (2*4 + 2 +
     2) = 8/12 = 2/3 *)
  equal ~msg:"f1 micro" (float 1e-6) (8.0 /. 12.0)
    (Metric.f1 Micro predictions targets)

let test_micro_equals_accuracy () =
  (* For multiclass single-label, micro P = micro R = micro F1 = accuracy *)
  let predictions = prf_predictions () in
  let targets = prf_targets () in
  let acc = Metric.accuracy predictions targets in
  equal ~msg:"micro precision = accuracy" (float 1e-6) acc
    (Metric.precision Micro predictions targets);
  equal ~msg:"micro recall = accuracy" (float 1e-6) acc
    (Metric.recall Micro predictions targets);
  equal ~msg:"micro f1 = accuracy" (float 1e-6) acc
    (Metric.f1 Micro predictions targets)

let test_precision_zero_predictions () =
  (* class 2 has no predictions: pred=[0,1,0], targets=[0,1,2] *)
  let predictions =
    Rune.create Rune.float32 [| 3; 3 |]
      [| 0.8; 0.2; 0.0; 0.1; 0.9; 0.0; 0.6; 0.4; 0.0 |]
  in
  let targets = Rune.create Rune.int32 [| 3 |] [| 0l; 1l; 2l |] in
  (* class 0: TP=1, FP=1 => P=1/2 class 1: TP=1, FP=0 => P=1 class 2: TP=0, FP=0
     => P=0.0 (zero-div) macro = (1/2 + 1 + 0) / 3 = 0.5 *)
  equal ~msg:"precision with missing class" (float 1e-6) 0.5
    (Metric.precision Macro predictions targets)

let test_binary_f1 () =
  (* 2-class problem *)
  let predictions =
    Rune.create Rune.float32 [| 4; 2 |]
      [|
        0.9;
        0.1;
        (* pred 0 *)
        0.3;
        0.7;
        (* pred 1 *)
        0.4;
        0.6;
        (* pred 1 *)
        0.8;
        0.2;
        (* pred 0 *)
      |]
  in
  let targets = Rune.create Rune.int32 [| 4 |] [| 0l; 1l; 0l; 0l |] in
  (* class 0: TP=2, FP=0, FN=1 => P=1.0, R=2/3, F1=2*1*(2/3)/(1+2/3)=4/5 *)
  (* class 1: TP=1, FP=1, FN=0 => P=1/2, R=1.0, F1=2*(1/2)*1/(1/2+1)=2/3 *)
  let expected_macro = ((4.0 /. 5.0) +. (2.0 /. 3.0)) /. 2.0 in
  equal ~msg:"binary f1 macro" (float 1e-6) expected_macro
    (Metric.f1 Macro predictions targets)

let () =
  run "Kaun.Metric"
    [
      group "tracker"
        [
          test "observe and mean" test_tracker_observe_and_mean;
          test "count" test_tracker_count;
          test "not found raises" test_tracker_not_found;
          test "reset" test_tracker_reset;
          test "to_list sorted" test_tracker_to_list_sorted;
          test "summary" test_tracker_summary;
        ];
      group "eval"
        [
          test "eval mean" test_eval_mean;
          test "eval empty raises" test_eval_empty_raises;
          test "eval_many" test_eval_many;
          test "eval_many empty raises" test_eval_many_empty_raises;
        ];
      group "accuracy"
        [
          test "multiclass" test_accuracy_multiclass;
          test "binary" test_accuracy_binary;
          test "binary_accuracy default" test_binary_accuracy_default_threshold;
          test "binary_accuracy custom threshold"
            test_binary_accuracy_custom_threshold;
        ];
      group "precision/recall/f1"
        [
          test "precision macro" test_precision_macro;
          test "precision micro" test_precision_micro;
          test "precision weighted" test_precision_weighted;
          test "recall macro" test_recall_macro;
          test "recall micro" test_recall_micro;
          test "f1 macro" test_f1_macro;
          test "f1 micro" test_f1_micro;
          test "micro = accuracy" test_micro_equals_accuracy;
          test "precision zero predictions" test_precision_zero_predictions;
          test "binary f1" test_binary_f1;
        ];
    ]
