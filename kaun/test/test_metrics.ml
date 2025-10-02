open Alcotest
open Kaun

let tensor_testable eps =
  testable
    (fun fmt t ->
      Format.fprintf fmt "tensor[%s]" (Rune.shape_to_string (Rune.shape t)))
    (fun a b ->
      if Rune.shape a <> Rune.shape b then false
      else
        let diff = Rune.abs (Rune.sub a b) in
        let max_diff = Rune.item [] (Rune.max diff) in
        max_diff < eps)

let test_accuracy () =
  let dtype = Rune.float32 in

  (* Binary classification test *)
  let predictions = Rune.create dtype [| 4 |] [| 0.2; 0.8; 0.6; 0.3 |] in
  let targets = Rune.create dtype [| 4 |] [| 0.; 1.; 1.; 0. |] in

  let acc = Metrics.accuracy () in
  Metrics.update acc ~predictions ~targets ();
  let result = Metrics.compute acc in
  let expected = Rune.scalar dtype 1.0 in
  (* All correct with 0.5 threshold *)
  check (tensor_testable 1e-5) "binary accuracy" expected result;

  (* Multi-class classification test *)
  let predictions =
    Rune.create dtype [| 3; 3 |]
      [| 0.9; 0.05; 0.05; 0.1; 0.8; 0.1; 0.2; 0.2; 0.6 |]
  in
  let targets_int = Rune.create Rune.int32 [| 3 |] [| 0l; 1l; 2l |] in
  let targets = Rune.cast dtype targets_int in

  let acc = Metrics.accuracy () in
  Metrics.reset acc;
  Metrics.update acc ~predictions ~targets ();
  let result = Metrics.compute acc in
  let expected = Rune.scalar dtype 1.0 in
  (* All correct *)
  check (tensor_testable 1e-5) "multi-class accuracy" expected result

let test_precision_recall () =
  let dtype = Rune.float32 in

  (* Binary classification: TP=2, FP=1, TN=1, FN=0 *)
  let predictions = Rune.create dtype [| 4 |] [| 0.8; 0.7; 0.6; 0.3 |] in
  let targets = Rune.create dtype [| 4 |] [| 1.; 1.; 0.; 0. |] in

  let prec = Metrics.precision () in
  Metrics.update prec ~predictions ~targets ();
  let result = Metrics.compute prec in
  (* Precision = TP/(TP+FP) = 2/(2+1) = 0.667 *)
  let expected = Rune.scalar dtype (2. /. 3.) in
  check (tensor_testable 1e-5) "precision" expected result;

  let rec_metric = Metrics.recall () in
  Metrics.update rec_metric ~predictions ~targets ();
  let result = Metrics.compute rec_metric in
  (* Recall = TP/(TP+FN) = 2/(2+0) = 1.0 *)
  let expected = Rune.scalar dtype 1.0 in
  check (tensor_testable 1e-5) "recall" expected result

let test_f1_score () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 4 |] [| 0.8; 0.7; 0.6; 0.3 |] in
  let targets = Rune.create dtype [| 4 |] [| 1.; 1.; 0.; 0. |] in

  let f1 = Metrics.f1_score () in
  Metrics.update f1 ~predictions ~targets ();
  let result = Metrics.compute f1 in
  (* F1 = 2 * (precision * recall) / (precision + recall) *)
  (* precision = 2/3, recall = 1.0 *)
  (* F1 = 2 * (2/3 * 1) / (2/3 + 1) = 2 * (2/3) / (5/3) = 4/5 = 0.8 *)
  let expected = Rune.scalar dtype 0.8 in
  check (tensor_testable 1e-5) "f1 score" expected result

let test_confusion_matrix () =
  let dtype = Rune.float32 in

  (* 3-class problem *)
  let predictions_int =
    Rune.create Rune.int32 [| 6 |] [| 0l; 1l; 2l; 0l; 1l; 2l |]
  in
  let targets_int =
    Rune.create Rune.int32 [| 6 |] [| 0l; 2l; 2l; 0l; 1l; 1l |]
  in
  let predictions = Rune.cast dtype predictions_int in
  let targets = Rune.cast dtype targets_int in

  let cm = Metrics.confusion_matrix ~num_classes:3 () in
  Metrics.update cm ~predictions ~targets ();
  let result = Metrics.compute cm in

  (* Expected confusion matrix: [[2, 0, 0], (true 0: predicted as 0,1,2) [0, 1,
     1], (true 1: predicted as 0,1,2) [0, 1, 1]] (true 2: predicted as 0,1,2) *)
  let expected =
    Rune.create dtype [| 3; 3 |] [| 2.; 0.; 0.; 0.; 1.; 1.; 0.; 1.; 1. |]
  in
  check (tensor_testable 1e-5) "confusion matrix" expected result

let test_mse_rmse () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let targets = Rune.create dtype [| 4 |] [| 1.5; 2.5; 2.5; 3.5 |] in

  (* MSE = mean((predictions - targets)^2) = mean([0.25, 0.25, 0.25, 0.25]) =
     0.25 *)
  let mse = Metrics.mse () in
  Metrics.update mse ~predictions ~targets ();
  let result = Metrics.compute mse in
  let expected = Rune.scalar dtype 0.25 in
  check (tensor_testable 1e-5) "mse" expected result;

  (* RMSE = sqrt(MSE) = sqrt(0.25) = 0.5 *)
  let rmse = Metrics.rmse () in
  Metrics.update rmse ~predictions ~targets ();
  let result = Metrics.compute rmse in
  let expected = Rune.scalar dtype 0.5 in
  check (tensor_testable 1e-5) "rmse" expected result

let test_mae () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let targets = Rune.create dtype [| 4 |] [| 1.5; 2.5; 2.5; 3.5 |] in

  (* MAE = mean(|predictions - targets|) = mean([0.5, 0.5, 0.5, 0.5]) = 0.5 *)
  let mae = Metrics.mae () in
  Metrics.update mae ~predictions ~targets ();
  let result = Metrics.compute mae in
  let expected = Rune.scalar dtype 0.5 in
  check (tensor_testable 1e-5) "mae" expected result

let test_r2_score () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let targets = Rune.create dtype [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in

  (* Perfect prediction: R² = 1.0 *)
  let r2 = Metrics.r2_score () in
  Metrics.update r2 ~predictions ~targets ();
  let result = Metrics.compute r2 in
  let expected = Rune.scalar dtype 1.0 in
  check (tensor_testable 1e-5) "r2 perfect" expected result;

  (* Test with some error *)
  let predictions = Rune.create dtype [| 4 |] [| 1.5; 2.5; 3.5; 4.5 |] in
  let r2 = Metrics.r2_score () in
  Metrics.reset r2;
  Metrics.update r2 ~predictions ~targets ();
  let result = Metrics.compute r2 in
  (* R² should be high but < 1.0 *)
  let result_val = Rune.item [] result in
  check bool "r2 with error" true (result_val > 0.8 && result_val < 1.0)

let test_cross_entropy () =
  let dtype = Rune.float32 in

  (* Test with logits *)
  let predictions =
    Rune.create dtype [| 2; 3 |] [| 2.0; -1.0; 0.5; -1.0; 3.0; 0.0 |]
  in
  let targets_int = Rune.create Rune.int32 [| 2 |] [| 0l; 1l |] in
  let targets = Rune.cast dtype targets_int in

  let ce = Metrics.cross_entropy ~from_logits:true () in
  Metrics.update ce ~predictions ~targets ();
  let result = Metrics.compute ce in
  (* Result should be positive *)
  let result_val = Rune.item [] result in
  check bool "cross entropy positive" true (result_val > 0.0)

let test_binary_cross_entropy () =
  let dtype = Rune.float32 in

  (* Perfect predictions *)
  let predictions = Rune.create dtype [| 4 |] [| 0.0; 1.0; 0.0; 1.0 |] in
  let targets = Rune.create dtype [| 4 |] [| 0.; 1.; 0.; 1. |] in

  let bce = Metrics.binary_cross_entropy ~from_logits:false () in
  Metrics.update bce ~predictions ~targets ();
  let result = Metrics.compute bce in
  (* Perfect predictions should give very low loss *)
  let result_val = Rune.item [] result in
  check bool "binary cross entropy perfect" true (result_val < 0.01)

let test_metric_collection () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 4 |] [| 0.8; 0.7; 0.6; 0.3 |] in
  let targets = Rune.create dtype [| 4 |] [| 1.; 1.; 0.; 0. |] in

  let collection =
    Metrics.Collection.create
      [
        ("accuracy", Metrics.accuracy ());
        ("precision", Metrics.precision ());
        ("recall", Metrics.recall ());
        ("f1", Metrics.f1_score ());
      ]
  in

  Metrics.Collection.update collection ~predictions ~targets ();
  let results = Metrics.Collection.compute collection in

  (* Check we got all metrics *)
  check int "collection size" 4 (List.length results);

  (* Check metric names *)
  let names = List.map fst results in
  check (list string) "metric names"
    [ "accuracy"; "precision"; "recall"; "f1" ]
    names;

  (* Test add/remove *)
  Metrics.Collection.add collection "mae" (Metrics.mae ());
  let results = Metrics.Collection.compute collection in
  check int "collection size after add" 5 (List.length results);

  Metrics.Collection.remove collection "mae";
  let results = Metrics.Collection.compute collection in
  check int "collection size after remove" 4 (List.length results)

let test_weighted_metrics () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let targets = Rune.create dtype [| 4 |] [| 1.5; 2.5; 2.5; 3.5 |] in
  let weights = Rune.create dtype [| 4 |] [| 1.0; 2.0; 2.0; 1.0 |] in

  (* Weighted MSE *)
  let mse = Metrics.mse () in
  Metrics.update mse ~predictions ~targets ~weights ();
  let result = Metrics.compute mse in
  (* Weighted MSE = sum(weights * (pred - target)^2) / sum(weights) *)
  (* = (1*0.25 + 2*0.25 + 2*0.25 + 1*0.25) / 6 = 1.5/6 = 0.25 *)
  let expected = Rune.scalar dtype 0.25 in
  check (tensor_testable 1e-5) "weighted mse" expected result

let test_metric_reset () =
  let dtype = Rune.float32 in

  let acc = Metrics.accuracy () in

  (* First batch *)
  let predictions1 = Rune.create dtype [| 2 |] [| 0.8; 0.2 |] in
  let targets1 = Rune.create dtype [| 2 |] [| 1.; 0. |] in
  Metrics.update acc ~predictions:predictions1 ~targets:targets1 ();

  (* Second batch *)
  let predictions2 = Rune.create dtype [| 2 |] [| 0.3; 0.7 |] in
  let targets2 = Rune.create dtype [| 2 |] [| 0.; 1. |] in
  Metrics.update acc ~predictions:predictions2 ~targets:targets2 ();

  let result = Metrics.compute acc in
  let expected = Rune.scalar dtype 1.0 in
  (* All 4 correct *)
  check (tensor_testable 1e-5) "accumulated accuracy" expected result;

  (* Reset and compute again *)
  Metrics.reset acc;
  Metrics.update acc ~predictions:predictions2 ~targets:targets2 ();
  let result = Metrics.compute acc in
  let expected = Rune.scalar dtype 1.0 in
  (* Only last 2 *)
  check (tensor_testable 1e-5) "accuracy after reset" expected result

let test_custom_metric () =
  let dtype = Rune.float32 in

  (* Create a custom metric that computes mean absolute percentage error *)
  let custom_mape =
    Metrics.create_custom ~name:"custom_mape"
      ~init:(fun () ->
        [
          Rune.zeros dtype [||];
          (* sum of absolute percentage errors *)
          Rune.zeros dtype [||];
          (* count *)
        ])
      ~update:(fun state ~predictions ~targets ?weights:_ () ->
        match state with
        | [ sum_err; count ] ->
            let abs_err = Rune.abs (Rune.sub predictions targets) in
            let percentage_err =
              Rune.div abs_err (Rune.add targets (Rune.scalar dtype 1e-7))
            in
            let batch_sum = Rune.sum percentage_err in
            let batch_count =
              Rune.scalar dtype (float_of_int (Rune.numel predictions))
            in
            [ Rune.add sum_err batch_sum; Rune.add count batch_count ]
        | _ -> failwith "Invalid state")
      ~compute:(fun state ->
        match state with
        | [ sum_err; count ] ->
            Rune.mul (Rune.div sum_err count) (Rune.scalar dtype 100.0)
        | _ -> failwith "Invalid state")
      ~reset:(fun _ -> [ Rune.zeros dtype [||]; Rune.zeros dtype [||] ])
  in

  let predictions = Rune.create dtype [| 3 |] [| 110.; 210.; 310. |] in
  let targets = Rune.create dtype [| 3 |] [| 100.; 200.; 300. |] in

  Metrics.update custom_mape ~predictions ~targets ();
  let result = Metrics.compute custom_mape in
  (* MAPE = mean(|110-100|/100, |210-200|/200, |310-300|/300) * 100 *)
  (* = mean(0.1, 0.05, 0.033) * 100 ≈ 6.1% *)
  let result_val = Rune.item [] result in
  check bool "custom MAPE in range" true (result_val > 5.5 && result_val < 6.5)

let test_metric_utilities () =
  let dtype = Rune.float32 in

  let acc = Metrics.accuracy () in

  (* Test name *)
  let name = Metrics.name acc in
  check string "metric name" "accuracy" name;

  (* Test is_better *)
  let is_better =
    Metrics.is_better acc ~higher_better:true ~old_val:0.8 ~new_val:0.9
  in
  check bool "is_better higher" true is_better;

  let is_better =
    Metrics.is_better acc ~higher_better:true ~old_val:0.9 ~new_val:0.8
  in
  check bool "is_better lower" false is_better;

  (* Test format *)
  let value = Rune.scalar dtype 0.8567 in
  let formatted = Metrics.format acc value in
  check bool "formatted contains value" true
    (String.contains formatted '8' || String.contains formatted '0')

let test_clone_metric () =
  let dtype = Rune.float32 in

  let acc1 = Metrics.accuracy () in
  let predictions = Rune.create dtype [| 2 |] [| 0.8; 0.2 |] in
  let targets = Rune.create dtype [| 2 |] [| 1.; 0. |] in

  (* Update original *)
  Metrics.update acc1 ~predictions ~targets ();

  (* Clone and verify independence *)
  let acc2 = Metrics.clone acc1 in

  (* Reset clone *)
  Metrics.reset acc2;

  (* Original should still have its state *)
  let result1 = Metrics.compute acc1 in
  let expected = Rune.scalar dtype 1.0 in
  check (tensor_testable 1e-5) "original after clone" expected result1;

  (* Clone should be reset *)
  Metrics.update acc2 ~predictions ~targets ();
  let result2 = Metrics.compute acc2 in
  check (tensor_testable 1e-5) "clone after reset" expected result2

let () =
  let open Alcotest in
  run "Metrics"
    [
      ( "classification",
        [
          test_case "accuracy" `Quick test_accuracy;
          test_case "precision_recall" `Quick test_precision_recall;
          test_case "f1_score" `Quick test_f1_score;
          test_case "confusion_matrix" `Quick test_confusion_matrix;
        ] );
      ( "regression",
        [
          test_case "mse_rmse" `Quick test_mse_rmse;
          test_case "mae" `Quick test_mae;
          test_case "r2_score" `Quick test_r2_score;
        ] );
      ( "probabilistic",
        [
          test_case "cross_entropy" `Quick test_cross_entropy;
          test_case "binary_cross_entropy" `Quick test_binary_cross_entropy;
        ] );
      ( "collections",
        [
          test_case "metric_collection" `Quick test_metric_collection;
          test_case "weighted_metrics" `Quick test_weighted_metrics;
        ] );
      ( "utilities",
        [
          test_case "metric_reset" `Quick test_metric_reset;
          test_case "custom_metric" `Quick test_custom_metric;
          test_case "metric_utilities" `Quick test_metric_utilities;
          test_case "clone_metric" `Quick test_clone_metric;
        ] );
    ]
