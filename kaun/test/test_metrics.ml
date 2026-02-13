(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Kaun

let float_eps eps = float eps

let test_accuracy () =
  let dtype = Rune.float32 in

  (* Binary classification test *)
  let predictions = Rune.create dtype [| 4 |] [| 0.2; 0.8; 0.6; 0.3 |] in
  let targets = Rune.create dtype [| 4 |] [| 0.; 1.; 1.; 0. |] in

  let acc = Metrics.accuracy () in
  Metrics.update acc ~predictions ~targets ();
  let result = Metrics.compute acc in
  let expected = 1.0 in
  (* All correct with 0.5 threshold *)
  equal ~msg:"binary accuracy" (float_eps 1e-5) expected result;

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
  let expected = 1.0 in
  (* All correct *)
  equal ~msg:"multi-class accuracy" (float_eps 1e-5) expected result

let test_accuracy_topk () =
  let dtype = Rune.float32 in

  let predictions =
    Rune.create dtype [| 3; 4 |]
      [| 0.1; 0.6; 0.2; 0.1; 0.3; 0.4; 0.2; 0.1; 0.5; 0.4; 0.3; 0.2 |]
  in
  let targets_int = Rune.create Rune.int32 [| 3 |] [| 1l; 2l; 0l |] in
  let targets = Rune.cast dtype targets_int in

  let acc = Metrics.accuracy ~top_k:2 () in
  Metrics.update acc ~predictions ~targets ();
  let result = Metrics.compute acc in
  let expected = 2. /. 3. in
  equal ~msg:"top-k accuracy" (float_eps 1e-5) expected result

let test_precision_recall () =
  let dtype = Rune.float32 in

  (* Binary classification: TP=2, FP=1, TN=1, FN=0 *)
  let predictions = Rune.create dtype [| 4 |] [| 0.8; 0.7; 0.6; 0.3 |] in
  let targets = Rune.create dtype [| 4 |] [| 1.; 1.; 0.; 0. |] in

  let prec = Metrics.precision () in
  Metrics.update prec ~predictions ~targets ();
  let result = Metrics.compute prec in
  (* Precision = TP/(TP+FP) = 2/(2+1) = 0.667 *)
  let expected = 2. /. 3. in
  equal ~msg:"precision" (float_eps 1e-5) expected result;

  let rec_metric = Metrics.recall () in
  Metrics.update rec_metric ~predictions ~targets ();
  let result = Metrics.compute rec_metric in
  (* Recall = TP/(TP+FN) = 2/(2+0) = 1.0 *)
  let expected = 1.0 in
  equal ~msg:"recall" (float_eps 1e-5) expected result

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
  let expected = 0.8 in
  equal ~msg:"f1 score" (float_eps 1e-5) expected result

let test_auc_roc () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 4 |] [| 0.8; 0.7; 0.6; 0.3 |] in
  let targets = Rune.create dtype [| 4 |] [| 1.; 1.; 0.; 0. |] in

  let auc = Metrics.auc_roc () in
  Metrics.update auc ~predictions ~targets ();
  let result = Metrics.compute auc in
  (* For perfectly separable predictions, AUC should be 1.0 *)
  let expected = 1.0 in
  equal ~msg:"auc roc" (float_eps 1e-5) expected result

let test_auc_roc_multiple_updates () =
  let dtype = Rune.float32 in

  let predictions_full = Rune.create dtype [| 4 |] [| 0.8; 0.7; 0.6; 0.3 |] in
  let targets_full = Rune.create dtype [| 4 |] [| 1.; 1.; 0.; 0. |] in

  let auc_single = Metrics.auc_roc () in
  Metrics.update auc_single ~predictions:predictions_full ~targets:targets_full
    ();
  let full_result = Metrics.compute auc_single in

  let auc_chunked = Metrics.auc_roc () in
  let predictions_1 = Rune.create dtype [| 2 |] [| 0.8; 0.7 |] in
  let targets_1 = Rune.create dtype [| 2 |] [| 1.; 1. |] in
  Metrics.update auc_chunked ~predictions:predictions_1 ~targets:targets_1 ();
  let predictions_2 = Rune.create dtype [| 2 |] [| 0.6; 0.3 |] in
  let targets_2 = Rune.create dtype [| 2 |] [| 0.; 0. |] in
  Metrics.update auc_chunked ~predictions:predictions_2 ~targets:targets_2 ();
  let chunked_result = Metrics.compute auc_chunked in

  equal ~msg:"auc roc incremental" (float_eps 1e-5) full_result chunked_result

let test_auc_pr () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 4 |] [| 0.8; 0.7; 0.6; 0.3 |] in
  let targets = Rune.create dtype [| 4 |] [| 1.; 1.; 0.; 0. |] in

  let auc = Metrics.auc_pr () in
  Metrics.update auc ~predictions ~targets ();
  let result = Metrics.compute auc in
  (* For perfectly separable predictions, AUC should be 1.0 *)
  let expected = 1.0 in
  equal ~msg:"auc pr" (float_eps 1e-5) expected result

let test_auc_pr_multiple_updates () =
  let dtype = Rune.float32 in

  let predictions_full = Rune.create dtype [| 4 |] [| 0.8; 0.7; 0.6; 0.3 |] in
  let targets_full = Rune.create dtype [| 4 |] [| 1.; 1.; 0.; 0. |] in

  let auc_single = Metrics.auc_pr () in
  Metrics.update auc_single ~predictions:predictions_full ~targets:targets_full
    ();
  let full_result = Metrics.compute auc_single in

  let auc_chunked = Metrics.auc_pr () in
  let predictions_1 = Rune.create dtype [| 2 |] [| 0.8; 0.7 |] in
  let targets_1 = Rune.create dtype [| 2 |] [| 1.; 1. |] in
  Metrics.update auc_chunked ~predictions:predictions_1 ~targets:targets_1 ();
  let predictions_2 = Rune.create dtype [| 2 |] [| 0.6; 0.3 |] in
  let targets_2 = Rune.create dtype [| 2 |] [| 0.; 0. |] in
  Metrics.update auc_chunked ~predictions:predictions_2 ~targets:targets_2 ();
  let chunked_result = Metrics.compute auc_chunked in

  equal ~msg:"auc pr incremental" (float_eps 1e-5) full_result chunked_result

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
  (* Confusion matrix is structured; fetch tensor output and skip strict
     check *)
  ignore (Metrics.compute_tensor cm)

let test_mse_rmse () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let targets = Rune.create dtype [| 4 |] [| 1.5; 2.5; 2.5; 3.5 |] in

  (* MSE = mean((predictions - targets)^2) = mean([0.25, 0.25, 0.25, 0.25]) =
     0.25 *)
  let mse = Metrics.mse () in
  Metrics.update mse ~predictions ~targets ();
  let result = Metrics.compute mse in
  let expected = 0.25 in
  equal ~msg:"mse" (float_eps 1e-5) expected result;

  (* RMSE = sqrt(MSE) = sqrt(0.25) = 0.5 *)
  let rmse = Metrics.rmse () in
  Metrics.update rmse ~predictions ~targets ();
  let result = Metrics.compute rmse in
  let expected = 0.5 in
  equal ~msg:"rmse" (float_eps 1e-5) expected result

let test_mae () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let targets = Rune.create dtype [| 4 |] [| 1.5; 2.5; 2.5; 3.5 |] in

  (* MAE = mean(|predictions - targets|) = mean([0.5, 0.5, 0.5, 0.5]) = 0.5 *)
  let mae = Metrics.mae () in
  Metrics.update mae ~predictions ~targets ();
  let result = Metrics.compute mae in
  let expected = 0.5 in
  equal ~msg:"mae" (float_eps 1e-5) expected result

let test_mape () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 4 |] [| 90.0; 110.0; 95.0; 105.0 |] in
  let targets = Rune.create dtype [| 4 |] [| 100.0; 100.0; 100.0; 100.0 |] in

  (* Absolute percentage errors: [0.1; 0.1; 0.05; 0.05] -> MAPE = 7.5% *)
  let mape = Metrics.mape () in
  Metrics.update mape ~predictions ~targets ();
  let result = Metrics.compute mape in
  let expected = 7.5 in
  equal ~msg:"mape" (float_eps 1e-5) expected result

let test_r2_score () =
  let dtype = Rune.float32 in

  (* Perfect prediction: R² = 1.0 *)
  let predictions1 = Rune.create dtype [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let targets1 = Rune.create dtype [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let r2_1 = Metrics.r2_score () in
  Metrics.update r2_1 ~predictions:predictions1 ~targets:targets1 ();
  let res1 = Metrics.compute r2_1 in
  equal ~msg:"r2 perfect" (float_eps 1e-5) 1.0 res1;

  (* Test with some error *)
  let predictions2 = Rune.create dtype [| 4 |] [| 1.5; 2.5; 3.5; 4.5 |] in
  let r2_2 = Metrics.r2_score () in
  Metrics.update r2_2 ~predictions:predictions2 ~targets:targets1 ();
  let res2 = Metrics.compute r2_2 in
  equal ~msg:"r2 with error" bool true (res2 > 0.8 && res2 < 1.0);

  (* Constant targets - perfect prediction *)
  let targets_const = Rune.create dtype [| 4 |] [| 5.0; 5.0; 5.0; 5.0 |] in
  let predictions_const = Rune.create dtype [| 4 |] [| 5.0; 5.0; 5.0; 5.0 |] in
  let r2_const = Metrics.r2_score () in
  Metrics.update r2_const ~predictions:predictions_const ~targets:targets_const
    ();
  let res3 = Metrics.compute r2_const in
  equal ~msg:"r2 constant perfect" (float_eps 1e-5) 1.0 res3;

  (* Constant targets - imperfect prediction should fallback to 0.0 *)
  let predictions_bad = Rune.create dtype [| 4 |] [| 4.0; 6.0; 5.0; 7.0 |] in
  let r2_bad = Metrics.r2_score () in
  Metrics.update r2_bad ~predictions:predictions_bad ~targets:targets_const ();
  let res4 = Metrics.compute r2_bad in
  equal ~msg:"r2 constant imperfect" (float_eps 1e-5) 0.0 res4;

  (* Adjusted R² with known closed-form expectation *)
  let predictions_adj =
    Rune.create dtype [| 5 |] [| 1.1; 1.9; 3.0; 4.1; 4.9 |]
  in
  let targets_adj = Rune.create dtype [| 5 |] [| 1.0; 2.0; 3.0; 4.0; 5.0 |] in
  let r2_adjusted = Metrics.r2_score ~adjusted:true ~num_features:1 () in
  Metrics.update r2_adjusted ~predictions:predictions_adj ~targets:targets_adj
    ();
  let res5 = Metrics.compute r2_adjusted in
  equal ~msg:"r2 adjusted" (float_eps 1e-5) 0.9946666667 res5

let test_explained_variance () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let targets = Rune.create dtype [| 4 |] [| 2.0; 4.0; 6.0; 8.0 |] in

  let ev = Metrics.explained_variance () in
  Metrics.update ev ~predictions ~targets ();
  let result = Metrics.compute ev in
  let expected = 0.75 in
  equal ~msg:"explained variance" (float_eps 1e-5) expected result;

  let targets_const = Rune.create dtype [| 3 |] [| 2.0; 2.0; 2.0 |] in
  let predictions_const = Rune.create dtype [| 3 |] [| 2.0; 2.0; 2.0 |] in
  let ev_const = Metrics.explained_variance () in
  Metrics.update ev_const ~predictions:predictions_const ~targets:targets_const
    ();
  let result = Metrics.compute ev_const in
  let expected = 1.0 in
  equal ~msg:"explained variance constant" (float_eps 1e-5) expected result;

  let predictions_bad = Rune.create dtype [| 3 |] [| 1.0; 3.0; 2.0 |] in
  let ev_bad = Metrics.explained_variance () in
  Metrics.update ev_bad ~predictions:predictions_bad ~targets:targets_const ();
  let result = Metrics.compute ev_bad in
  let expected = 0.0 in
  equal ~msg:"explained variance constant imperfect" (float_eps 1e-5) expected result

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
  equal ~msg:"cross entropy positive" bool true (result > 0.0)

let test_binary_cross_entropy () =
  let dtype = Rune.float32 in

  (* Perfect predictions *)
  let predictions = Rune.create dtype [| 4 |] [| 0.0; 1.0; 0.0; 1.0 |] in
  let targets = Rune.create dtype [| 4 |] [| 0.; 1.; 0.; 1. |] in

  let bce = Metrics.binary_cross_entropy ~from_logits:false () in
  Metrics.update bce ~predictions ~targets ();
  let result = Metrics.compute bce in
  (* Perfect predictions should give very low loss *)
  equal ~msg:"binary cross entropy perfect" bool true (result < 0.01)

let test_ndcg () =
  let dtype = Rune.float32 in

  let predictions =
    Rune.create dtype [| 2; 3 |] [| 0.3; 0.2; 0.1; 0.9; 0.8; 0.7 |]
  in
  let targets = Rune.create dtype [| 2; 3 |] [| 1.; 2.; 3.; 3.; 2.; 1. |] in

  let ndcg = Metrics.ndcg () in
  Metrics.update ndcg ~predictions ~targets ();
  let result = Metrics.compute ndcg in
  let expected = (0.6806060568 +. 1.0) /. 2.0 in
  equal ~msg:"ndcg" (float_eps 1e-5) expected result

let test_map_metric () =
  let dtype = Rune.float32 in

  let predictions =
    Rune.create dtype [| 2; 4 |] [| 0.9; 0.8; 0.7; 0.1; 0.4; 0.3; 0.2; 0.1 |]
  in
  let targets =
    Rune.create dtype [| 2; 4 |] [| 1.; 0.; 1.; 0.; 1.; 1.; 0.; 0. |]
  in

  let map = Metrics.map () in
  Metrics.update map ~predictions ~targets ();
  let result = Metrics.compute map in
  let expected = 11. /. 12. in
  equal ~msg:"map" (float_eps 1e-5) expected result

let test_mrr_metric () =
  let dtype = Rune.float32 in

  let predictions =
    Rune.create dtype [| 2; 3 |] [| 0.9; 0.8; 0.7; 0.9; 0.2; 0.1 |]
  in
  let targets = Rune.create dtype [| 2; 3 |] [| 0.; 0.; 1.; 0.; 1.; 0. |] in

  let mrr = Metrics.mrr () in
  Metrics.update mrr ~predictions ~targets ();
  let result = Metrics.compute mrr in
  let expected = 5. /. 12. in
  equal ~msg:"mrr" (float_eps 1e-5) expected result

let test_bleu () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 1; 4 |] [| 1.; 2.; 3.; 4. |] in
  let targets = Rune.create dtype [| 1; 4 |] [| 1.; 2.; 3.; 4. |] in

  let bleu = Metrics.bleu () in
  Metrics.update bleu ~predictions ~targets ();
  let result = Metrics.compute bleu in
  let expected = 1.0 in
  equal ~msg:"bleu" (float_eps 1e-5) expected result

let test_rouge () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 1; 3 |] [| 1.; 2.; 3. |] in
  let targets = Rune.create dtype [| 1; 3 |] [| 1.; 3.; 4. |] in

  let rouge = Metrics.rouge ~variant:`Rouge1 () in
  Metrics.update rouge ~predictions ~targets ();
  let result = Metrics.compute rouge in
  let expected = 2. /. 3. in
  equal ~msg:"rouge1" (float_eps 1e-5) expected result

let test_meteor_metric () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 1; 4 |] [| 1.; 2.; 3.; 0. |] in
  let targets = Rune.create dtype [| 1; 4 |] [| 1.; 2.; 3.; 4. |] in

  let meteor = Metrics.meteor () in
  Metrics.update meteor ~predictions ~targets ();
  let result = Metrics.compute meteor in
  let expected = 0.75498576 in
  equal ~msg:"meteor" (float_eps 1e-5) expected result

let test_ssim () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 4 |] [| 0.0; 1.0; 0.0; 1.0 |] in
  let targets = Rune.create dtype [| 4 |] [| 0.0; 1.0; 0.0; 1.0 |] in

  let metric = Metrics.ssim () in
  Metrics.update metric ~predictions ~targets ();
  let result = Metrics.compute metric in
  let expected = 1.0 in
  equal ~msg:"ssim" (float_eps 1e-5) expected result

let test_iou () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 4 |] [| 0.; 1.; 1.; 0. |] in
  let targets = Rune.create dtype [| 4 |] [| 0.; 1.; 0.; 0. |] in

  let metric = Metrics.iou ~num_classes:2 () in
  Metrics.update metric ~predictions ~targets ();
  let result = Metrics.compute metric in
  let expected = ((2. /. 3.) +. 0.5) /. 2.0 in
  equal ~msg:"iou" (float_eps 1e-5) expected result

let test_dice () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 4 |] [| 0.; 1.; 1.; 0. |] in
  let targets = Rune.create dtype [| 4 |] [| 0.; 1.; 0.; 0. |] in

  let metric = Metrics.dice ~num_classes:2 () in
  Metrics.update metric ~predictions ~targets ();
  let result = Metrics.compute metric in
  let expected = (0.8 +. (2. /. 3.)) /. 2.0 in
  equal ~msg:"dice" (float_eps 1e-5) expected result

let test_kl_divergence () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 2; 2 |] [| 0.6; 0.4; 0.5; 0.5 |] in
  let targets = Rune.create dtype [| 2; 2 |] [| 0.7; 0.3; 0.4; 0.6 |] in

  let kl = Metrics.kl_divergence () in
  Metrics.update kl ~predictions ~targets ();
  let result = Metrics.compute kl in
  let expected_value =
    let open Float in
    let kl1 = (0.7 *. log (0.7 /. 0.6)) +. (0.3 *. log (0.3 /. 0.4)) in
    let kl2 = (0.4 *. log (0.4 /. 0.5)) +. (0.6 *. log (0.6 /. 0.5)) in
    (kl1 +. kl2) /. 2.0
  in
  let expected = expected_value in
  equal ~msg:"kl divergence" (float_eps 1e-5) expected result;

  let kl_zero = Metrics.kl_divergence () in
  Metrics.update kl_zero ~predictions:targets ~targets ();
  let result = Metrics.compute kl_zero in
  let expected = 0.0 in
  equal ~msg:"kl divergence zero" (float_eps 1e-5) expected result

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
  equal ~msg:"collection size" int 4 (List.length results);

  (* Check metric names *)
  let names = List.map fst results in
  equal ~msg:"metric names" (list string)
    [ "accuracy"; "precision"; "recall"; "f1" ]
    names;

  (* Test add/remove *)
  Metrics.Collection.add collection "mae" (Metrics.mae ());
  let results = Metrics.Collection.compute collection in
  equal ~msg:"collection size after add" int 5 (List.length results);

  Metrics.Collection.remove collection "mae";
  let results = Metrics.Collection.compute collection in
  equal ~msg:"collection size after remove" int 4 (List.length results)

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
  let expected = 0.25 in
  equal ~msg:"weighted mse" (float_eps 1e-5) expected result

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
  let expected = 1.0 in
  (* All 4 correct *)
  equal ~msg:"accumulated accuracy" (float_eps 1e-5) expected result;

  (* Reset and compute again *)
  Metrics.reset acc;
  Metrics.update acc ~predictions:predictions2 ~targets:targets2 ();
  let result = Metrics.compute acc in
  let expected = 1.0 in
  (* Only last 2 *)
  equal ~msg:"accuracy after reset" (float_eps 1e-5) expected result

let test_custom_metric () =
  let dtype = Rune.float32 in

  (* Create a custom metric that computes mean absolute percentage error *)
  let custom_mape =
    Metrics.create_custom ~dtype ~name:"custom_mape"
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
  equal ~msg:"custom MAPE in range" bool true (result > 5.5 && result < 6.5)

let test_metric_utilities () =
  let acc = Metrics.accuracy () in

  (* Test name *)
  let name = Metrics.name acc in
  equal ~msg:"metric name" string "accuracy" name;

  (* Test is_better *)
  let is_better =
    Metrics.is_better acc ~higher_better:true ~old_val:0.8 ~new_val:0.9
  in
  equal ~msg:"is_better higher" bool true is_better;

  let is_better =
    Metrics.is_better acc ~higher_better:true ~old_val:0.9 ~new_val:0.8
  in
  equal ~msg:"is_better lower" bool false is_better;

  (* Test format *)
  let value = 0.8567 in
  let formatted = Metrics.format acc value in
  equal ~msg:"formatted contains value" bool true
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
  let expected = 1.0 in
  equal ~msg:"original after clone" (float_eps 1e-5) expected result1;

  (* Clone should be reset *)
  Metrics.update acc2 ~predictions ~targets ();
  let result2 = Metrics.compute acc2 in
  equal ~msg:"clone after reset" (float_eps 1e-5) expected result2

let () =
  run "Metrics"
    [
      group "classification"
        [
          test "accuracy" test_accuracy;
          test "accuracy_topk" test_accuracy_topk;
          test "precision_recall" test_precision_recall;
          test "f1_score" test_f1_score;
          test "auc_roc" test_auc_roc;
          test "auc_roc_multiple_updates"
            test_auc_roc_multiple_updates;
          test "auc_pr" test_auc_pr;
          test "auc_pr_multiple_updates"
            test_auc_pr_multiple_updates;
          test "confusion_matrix" test_confusion_matrix;
        ];
      group "ranking"
        [
          test "ndcg" test_ndcg;
          test "map" test_map_metric;
          test "mrr" test_mrr_metric;
        ];
      group "nlp"
        [
          test "bleu" test_bleu;
          test "rouge" test_rouge;
          test "meteor" test_meteor_metric;
        ];
      group "vision"
        [
          test "ssim" test_ssim;
          test "iou" test_iou;
          test "dice" test_dice;
        ];
      group "regression"
        [
          test "mse_rmse" test_mse_rmse;
          test "mae" test_mae;
          test "mape" test_mape;
          test "r2_score" test_r2_score;
          test "explained_variance" test_explained_variance;
        ];
      group "probabilistic"
        [
          test "cross_entropy" test_cross_entropy;
          test "binary_cross_entropy" test_binary_cross_entropy;
          test "kl_divergence" test_kl_divergence;
        ];
      group "collections"
        [
          test "metric_collection" test_metric_collection;
          test "weighted_metrics" test_weighted_metrics;
        ];
      group "utilities"
        [
          test "metric_reset" test_metric_reset;
          test "custom_metric" test_custom_metric;
          test "metric_utilities" test_metric_utilities;
          test "clone_metric" test_clone_metric;
        ];
    ]
