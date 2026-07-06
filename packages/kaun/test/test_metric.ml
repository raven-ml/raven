(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Kaun

let vec xs = Nx.create Nx.float64 [| Array.length xs |] xs
let mat rows cols xs = Nx.create Nx.float64 [| rows; cols |] xs
let labels ls = Nx.create Nx.int32 [| Array.length ls |] ls
let close ?(eps = 1e-12) expected actual = equal (float eps) expected actual

(* Predictions over [classes] classes whose argmax row [i] is [preds.(i)]. *)
let predicting classes preds =
  let n = Array.length preds in
  let xs = Array.make (n * classes) 0.0 in
  Array.iteri (fun i p -> xs.((i * classes) + p) <- 1.0) preds;
  mat n classes xs

(* Accuracy *)

let accuracy_tests =
  [
    test "matches the hand-computed fraction" (fun () ->
        (* Argmaxes 0, 1, 2, 0 against labels 0, 1, 2, 2. *)
        close 0.75
          (Metric.accuracy
             (mat 4 3 [| 5.; 1.; 0.; 0.; 2.; 1.; 0.; 1.; 9.; 3.; 2.; 1. |])
             (labels [| 0l; 1l; 2l; 2l |])));
    test "is 1 when every argmax matches" (fun () ->
        close 1.
          (Metric.accuracy
             (predicting 3 [| 2; 0; 1 |])
             (labels [| 2l; 0l; 1l |])));
    test "resolves argmax ties to the lowest class index" (fun () ->
        let tied = mat 1 3 [| 1.; 1.; 0. |] in
        close 1. (Metric.accuracy tied (labels [| 0l |]));
        close 0. (Metric.accuracy tied (labels [| 1l |])));
    test "accepts an unbatched example" (fun () ->
        close 1.
          (Metric.accuracy (vec [| 0.; 2.; 1. |]) (Nx.scalar Nx.int32 1l)));
    test "rejects mismatched label shapes" (fun () ->
        raises_invalid_arg
          "Metric.accuracy: labels shape [2] does not match predictions batch \
           shape [4]" (fun () ->
            Metric.accuracy (mat 4 3 (Array.make 12 0.)) (labels [| 0l; 0l |])));
    test "rejects an empty batch" (fun () ->
        raises_invalid_arg "Metric.accuracy: there are no examples" (fun () ->
            Metric.accuracy (mat 0 3 [||]) (labels [||])));
    test "rejects out-of-range labels" (fun () ->
        raises_invalid_arg "Metric.accuracy: label 3 is out of range [0;2]"
          (fun () -> Metric.accuracy (predicting 3 [| 0 |]) (labels [| 3l |])));
  ]

(* Top-k accuracy *)

let top_k_tests =
  [
    test "matches the hand-computed fraction" (fun () ->
        (* Label ranks per row: 2nd, 4th, 1st. *)
        let p = mat 3 4 [| 9.; 3.; 2.; 1.; 4.; 3.; 2.; 1.; 5.; 6.; 7.; 8. |] in
        let l = labels [| 1l; 3l; 3l |] in
        close (2. /. 3.) (Metric.top_k_accuracy ~k:2 p l);
        close (1. /. 3.) (Metric.top_k_accuracy ~k:1 p l);
        close 1. (Metric.top_k_accuracy ~k:4 p l));
    test "agrees with accuracy at k = 1 on tie-free predictions" (fun () ->
        let p = mat 2 3 [| 0.5; 0.1; 0.9; 3.; 2.; 1. |] in
        let l = labels [| 2l; 1l |] in
        close (Metric.accuracy p l) (Metric.top_k_accuracy ~k:1 p l));
    test "gives the label the benefit of ties" (fun () ->
        (* Classes 0 and 1 tie for the maximum; the label still counts. *)
        let tied = mat 1 3 [| 2.; 2.; 1. |] in
        close 1. (Metric.top_k_accuracy ~k:1 tied (labels [| 1l |]));
        close 0. (Metric.accuracy tied (labels [| 1l |])));
    test "rejects k below 1" (fun () ->
        raises_invalid_arg "Metric.top_k_accuracy: k must be in [1;3] (got 0)"
          (fun () ->
            Metric.top_k_accuracy ~k:0 (predicting 3 [| 0 |]) (labels [| 0l |])));
    test "rejects k above the class count" (fun () ->
        raises_invalid_arg "Metric.top_k_accuracy: k must be in [1;3] (got 4)"
          (fun () ->
            Metric.top_k_accuracy ~k:4 (predicting 3 [| 0 |]) (labels [| 0l |])));
  ]

(* Confusion matrix.

   Shared example: predicted classes [0; 0; 2; 2; 0; 2] against labels [2; 0; 2;
   2; 0; 1]. The confusion matrix rows (for labels 0, 1, 2) are [2 0 0], [0 0 1]
   and [1 0 2], so tp = [2; 0; 2], true instances (row sums) = [2; 1; 3] and
   predicted counts (column sums) = [3; 0; 3]. *)

let example_predictions = predicting 3 [| 0; 0; 2; 2; 0; 2 |]
let example_labels = labels [| 2l; 0l; 2l; 2l; 0l; 1l |]

let confusion_tests =
  [
    test "matches the hand-computed matrix" (fun () ->
        let m = Metric.confusion_matrix example_predictions example_labels in
        equal (array int) [| 3; 3 |] (Nx.shape m);
        equal (array int)
          [| 2; 0; 0; 0; 0; 1; 1; 0; 2 |]
          (Array.map Int32.to_int (Nx.to_array m)));
    test "puts a single observed class on the diagonal" (fun () ->
        let m =
          Metric.confusion_matrix
            (predicting 3 [| 0; 0 |])
            (labels [| 0l; 0l |])
        in
        equal (array int)
          [| 2; 0; 0; 0; 0; 0; 0; 0; 0 |]
          (Array.map Int32.to_int (Nx.to_array m)));
    test "rejects out-of-range labels" (fun () ->
        raises_invalid_arg
          "Metric.confusion_matrix: label 3 is out of range [0;2]" (fun () ->
            Metric.confusion_matrix (predicting 3 [| 0 |]) (labels [| 3l |])));
  ]

(* Precision, recall, F1 *)

let predictions_and_labels =
  Testable.with_gen
    Gen.(
      pair
        (list_size (pure 12) (float_range (-5.) 5.))
        (list_size (pure 4) (int_range 0 2)))
    (pair (list (float 1e-12)) (list int))

let prf_tests =
  [
    test "averages per-class precision by default" (fun () ->
        (* (2/3 + 0 + 2/3) / 3; class 1 is never predicted. *)
        close (4. /. 9.) (Metric.precision example_predictions example_labels));
    test "averages per-class recall under macro" (fun () ->
        (* (1 + 0 + 2/3) / 3 *)
        close (5. /. 9.)
          (Metric.recall ~average:`Macro example_predictions example_labels));
    test "averages per-class F1 by default" (fun () ->
        (* (4/5 + 0 + 2/3) / 3 *)
        close (22. /. 45.) (Metric.f1 example_predictions example_labels));
    test "micro scores pool counts across classes" (fun () ->
        (* 4 of 6 examples are correct. *)
        close (2. /. 3.)
          (Metric.precision ~average:`Micro example_predictions example_labels);
        close (2. /. 3.)
          (Metric.recall ~average:`Micro example_predictions example_labels);
        close (2. /. 3.)
          (Metric.f1 ~average:`Micro example_predictions example_labels));
    test "counts absent classes as zero in the macro mean" (fun () ->
        (* Only class 0 appears: class 1 scores 0 and still divides the mean. *)
        let p = predicting 2 [| 0; 0; 0 |] and l = labels [| 0l; 0l; 0l |] in
        close 0.5 (Metric.precision p l);
        close 0.5 (Metric.recall p l);
        close 0.5 (Metric.f1 p l));
    prop' "micro F1 equals accuracy" predictions_and_labels (fun (xs, ls) ->
        let p = mat 4 3 (Array.of_list xs) in
        let l = labels (Array.of_list (List.map Int32.of_int ls)) in
        close (Metric.accuracy p l) (Metric.f1 ~average:`Micro p l));
  ]

(* AUC-ROC *)

(* Scores drawn from four integer values, so ties are frequent. *)
let tied_scores =
  Testable.with_gen Gen.(list_size (pure 6) (int_range 0 3)) (list int)

let auc_tests =
  [
    test "matches the hand-computed value" (fun () ->
        (* Positives 0.35 and 0.8 beat negatives 0.1 and 0.4 in 3 of 4 pairs. *)
        close 0.75
          (Metric.auc_roc
             (vec [| 0.1; 0.4; 0.35; 0.8 |])
             (labels [| 0l; 0l; 1l; 1l |])));
    test "is 1 for a perfect ranking and 0 for a reversed one" (fun () ->
        let s = vec [| 0.1; 0.9 |] in
        close 1. (Metric.auc_roc s (labels [| 0l; 1l |]));
        close 0. (Metric.auc_roc s (labels [| 1l; 0l |])));
    test "gives tied pairs half credit" (fun () ->
        (* The positive ties one negative (1/2) and beats the other (1). *)
        close 0.75
          (Metric.auc_roc (vec [| 1.; 1.; 0. |]) (labels [| 1l; 0l; 0l |])));
    test "is a half when all scores tie" (fun () ->
        close 0.5
          (Metric.auc_roc
             (vec [| 0.3; 0.3; 0.3; 0.3 |])
             (labels [| 1l; 1l; 0l; 0l |])));
    prop' "agrees with exhaustive pair counting" tied_scores (fun xs ->
        let scores = Array.of_list (List.map float_of_int xs) in
        let positive = [| true; false; true; false; true; false |] in
        let wins = ref 0.0 and pairs = ref 0 in
        Array.iteri
          (fun i pi ->
            if pi then
              Array.iteri
                (fun j pj ->
                  if not pj then begin
                    incr pairs;
                    if scores.(i) > scores.(j) then wins := !wins +. 1.
                    else if scores.(i) = scores.(j) then wins := !wins +. 0.5
                  end)
                positive)
          positive;
        close
          (!wins /. float_of_int !pairs)
          (Metric.auc_roc (vec scores) (labels [| 1l; 0l; 1l; 0l; 1l; 0l |])));
    test "rejects labels other than 0 and 1" (fun () ->
        raises_invalid_arg "Metric.auc_roc: label 2 is neither 0 nor 1"
          (fun () -> Metric.auc_roc (vec [| 0.1; 0.9 |]) (labels [| 0l; 2l |])));
    test "rejects a single-class batch" (fun () ->
        raises_invalid_arg "Metric.auc_roc: labels must contain both classes"
          (fun () -> Metric.auc_roc (vec [| 0.1; 0.9 |]) (labels [| 1l; 1l |])));
    test "rejects mismatched shapes" (fun () ->
        raises_invalid_arg
          "Metric.auc_roc: labels shape [2] does not match scores shape [3]"
          (fun () ->
            Metric.auc_roc (vec [| 0.1; 0.5; 0.9 |]) (labels [| 0l; 1l |])));
  ]

let tests =
  [
    group "accuracy" accuracy_tests;
    group "top_k_accuracy" top_k_tests;
    group "confusion_matrix" confusion_tests;
    group "precision, recall, f1" prf_tests;
    group "auc_roc" auc_tests;
  ]

let () = run "kaun metric" tests
