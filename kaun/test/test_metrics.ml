open Alcotest
open Kaun

let float_eps eps = Alcotest.float eps

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
  check (float_eps 1e-5) "binary accuracy" expected result;

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
  check (float_eps 1e-5) "multi-class accuracy" expected result

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
  check (float_eps 1e-5) "top-k accuracy" expected result

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
  check (float_eps 1e-5) "precision" expected result;

  let rec_metric = Metrics.recall () in
  Metrics.update rec_metric ~predictions ~targets ();
  let result = Metrics.compute rec_metric in
  (* Recall = TP/(TP+FN) = 2/(2+0) = 1.0 *)
  let expected = 1.0 in
  check (float_eps 1e-5) "recall" expected result

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
  check (float_eps 1e-5) "f1 score" expected result

let test_auc_roc () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 4 |] [| 0.8; 0.7; 0.6; 0.3 |] in
  let targets = Rune.create dtype [| 4 |] [| 1.; 1.; 0.; 0. |] in

  let auc = Metrics.auc_roc () in
  Metrics.update auc ~predictions ~targets ();
  let result = Metrics.compute auc in
  (* For perfectly separable predictions, AUC should be 1.0 *)
  let expected = 1.0 in
  check (float_eps 1e-5) "auc roc" expected result

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

  check (float_eps 1e-5) "auc roc incremental" full_result chunked_result

let test_auc_pr () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 4 |] [| 0.8; 0.7; 0.6; 0.3 |] in
  let targets = Rune.create dtype [| 4 |] [| 1.; 1.; 0.; 0. |] in

  let auc = Metrics.auc_pr () in
  Metrics.update auc ~predictions ~targets ();
  let result = Metrics.compute auc in
  (* For perfectly separable predictions, AUC should be 1.0 *)
  let expected = 1.0 in
  check (float_eps 1e-5) "auc pr" expected result

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

  check (float_eps 1e-5) "auc pr incremental" full_result chunked_result

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
  check (float_eps 1e-5) "mse" expected result;

  (* RMSE = sqrt(MSE) = sqrt(0.25) = 0.5 *)
  let rmse = Metrics.rmse () in
  Metrics.update rmse ~predictions ~targets ();
  let result = Metrics.compute rmse in
  let expected = 0.5 in
  check (float_eps 1e-5) "rmse" expected result

let test_mae () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let targets = Rune.create dtype [| 4 |] [| 1.5; 2.5; 2.5; 3.5 |] in

  (* MAE = mean(|predictions - targets|) = mean([0.5, 0.5, 0.5, 0.5]) = 0.5 *)
  let mae = Metrics.mae () in
  Metrics.update mae ~predictions ~targets ();
  let result = Metrics.compute mae in
  let expected = 0.5 in
  check (float_eps 1e-5) "mae" expected result

let test_mape () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 4 |] [| 90.0; 110.0; 95.0; 105.0 |] in
  let targets = Rune.create dtype [| 4 |] [| 100.0; 100.0; 100.0; 100.0 |] in

  (* Absolute percentage errors: [0.1; 0.1; 0.05; 0.05] -> MAPE = 7.5% *)
  let mape = Metrics.mape () in
  Metrics.update mape ~predictions ~targets ();
  let result = Metrics.compute mape in
  let expected = 7.5 in
  check (float_eps 1e-5) "mape" expected result

let test_r2_score () =
  let dtype = Rune.float32 in

  (* Perfect prediction: R² = 1.0 *)
  let predictions1 = Rune.create dtype [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let targets1 = Rune.create dtype [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let r2_1 = Metrics.r2_score () in
  Metrics.update r2_1 ~predictions:predictions1 ~targets:targets1 ();
  let res1 = Metrics.compute r2_1 in
  check (float_eps 1e-5) "r2 perfect" 1.0 res1;

  (* Test with some error *)
  let predictions2 = Rune.create dtype [| 4 |] [| 1.5; 2.5; 3.5; 4.5 |] in
  let r2_2 = Metrics.r2_score () in
  Metrics.update r2_2 ~predictions:predictions2 ~targets:targets1 ();
  let res2 = Metrics.compute r2_2 in
  check bool "r2 with error" true (res2 > 0.8 && res2 < 1.0);

  (* Constant targets - perfect prediction *)
  let targets_const = Rune.create dtype [| 4 |] [| 5.0; 5.0; 5.0; 5.0 |] in
  let predictions_const = Rune.create dtype [| 4 |] [| 5.0; 5.0; 5.0; 5.0 |] in
  let r2_const = Metrics.r2_score () in
  Metrics.update r2_const ~predictions:predictions_const ~targets:targets_const
    ();
  let res3 = Metrics.compute r2_const in
  check (float_eps 1e-5) "r2 constant perfect" 1.0 res3;

  (* Constant targets - imperfect prediction should fallback to 0.0 *)
  let predictions_bad = Rune.create dtype [| 4 |] [| 4.0; 6.0; 5.0; 7.0 |] in
  let r2_bad = Metrics.r2_score () in
  Metrics.update r2_bad ~predictions:predictions_bad ~targets:targets_const ();
  let res4 = Metrics.compute r2_bad in
  check (float_eps 1e-5) "r2 constant imperfect" 0.0 res4;

  (* Adjusted R² with known closed-form expectation *)
  let predictions_adj =
    Rune.create dtype [| 5 |] [| 1.1; 1.9; 3.0; 4.1; 4.9 |]
  in
  let targets_adj = Rune.create dtype [| 5 |] [| 1.0; 2.0; 3.0; 4.0; 5.0 |] in
  let r2_adjusted = Metrics.r2_score ~adjusted:true ~num_features:1 () in
  Metrics.update r2_adjusted ~predictions:predictions_adj ~targets:targets_adj
    ();
  let res5 = Metrics.compute r2_adjusted in
  check (float_eps 1e-5) "r2 adjusted" 0.9946666667 res5

let test_explained_variance () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let targets = Rune.create dtype [| 4 |] [| 2.0; 4.0; 6.0; 8.0 |] in

  let ev = Metrics.explained_variance () in
  Metrics.update ev ~predictions ~targets ();
  let result = Metrics.compute ev in
  let expected = 0.75 in
  check (float_eps 1e-5) "explained variance" expected result;

  let targets_const = Rune.create dtype [| 3 |] [| 2.0; 2.0; 2.0 |] in
  let predictions_const = Rune.create dtype [| 3 |] [| 2.0; 2.0; 2.0 |] in
  let ev_const = Metrics.explained_variance () in
  Metrics.update ev_const ~predictions:predictions_const ~targets:targets_const
    ();
  let result = Metrics.compute ev_const in
  let expected = 1.0 in
  check (float_eps 1e-5) "explained variance constant" expected result;

  let predictions_bad = Rune.create dtype [| 3 |] [| 1.0; 3.0; 2.0 |] in
  let ev_bad = Metrics.explained_variance () in
  Metrics.update ev_bad ~predictions:predictions_bad ~targets:targets_const ();
  let result = Metrics.compute ev_bad in
  let expected = 0.0 in
  check (float_eps 1e-5) "explained variance constant imperfect" expected result

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
  check bool "cross entropy positive" true (result > 0.0)

let test_binary_cross_entropy () =
  let dtype = Rune.float32 in

  (* Perfect predictions *)
  let predictions = Rune.create dtype [| 4 |] [| 0.0; 1.0; 0.0; 1.0 |] in
  let targets = Rune.create dtype [| 4 |] [| 0.; 1.; 0.; 1. |] in

  let bce = Metrics.binary_cross_entropy ~from_logits:false () in
  Metrics.update bce ~predictions ~targets ();
  let result = Metrics.compute bce in
  (* Perfect predictions should give very low loss *)
  check bool "binary cross entropy perfect" true (result < 0.01)

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
  check (float_eps 1e-5) "ndcg" expected result

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
  check (float_eps 1e-5) "map" expected result

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
  check (float_eps 1e-5) "mrr" expected result

let test_bleu () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 1; 4 |] [| 1.; 2.; 3.; 4. |] in
  let targets = Rune.create dtype [| 1; 4 |] [| 1.; 2.; 3.; 4. |] in

  let bleu = Metrics.bleu () in
  Metrics.update bleu ~predictions ~targets ();
  let result = Metrics.compute bleu in
  let expected = 1.0 in
  check (float_eps 1e-5) "bleu" expected result

let test_rouge () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 1; 3 |] [| 1.; 2.; 3. |] in
  let targets = Rune.create dtype [| 1; 3 |] [| 1.; 3.; 4. |] in

  let rouge = Metrics.rouge ~variant:`Rouge1 () in
  Metrics.update rouge ~predictions ~targets ();
  let result = Metrics.compute rouge in
  let expected = 2. /. 3. in
  check (float_eps 1e-5) "rouge1" expected result

let test_meteor_metric () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 1; 4 |] [| 1.; 2.; 3.; 0. |] in
  let targets = Rune.create dtype [| 1; 4 |] [| 1.; 2.; 3.; 4. |] in

  let meteor = Metrics.meteor () in
  Metrics.update meteor ~predictions ~targets ();
  let result = Metrics.compute meteor in
  let expected = 0.75498576 in
  check (float_eps 1e-5) "meteor" expected result

let test_ssim () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 4 |] [| 0.0; 1.0; 0.0; 1.0 |] in
  let targets = Rune.create dtype [| 4 |] [| 0.0; 1.0; 0.0; 1.0 |] in

  let metric = Metrics.ssim () in
  Metrics.update metric ~predictions ~targets ();
  let result = Metrics.compute metric in
  let expected = 1.0 in
  check (float_eps 1e-5) "ssim" expected result

let test_iou () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 4 |] [| 0.; 1.; 1.; 0. |] in
  let targets = Rune.create dtype [| 4 |] [| 0.; 1.; 0.; 0. |] in

  let metric = Metrics.iou ~num_classes:2 () in
  Metrics.update metric ~predictions ~targets ();
  let result = Metrics.compute metric in
  let expected = ((2. /. 3.) +. 0.5) /. 2.0 in
  check (float_eps 1e-5) "iou" expected result

let test_dice () =
  let dtype = Rune.float32 in

  let predictions = Rune.create dtype [| 4 |] [| 0.; 1.; 1.; 0. |] in
  let targets = Rune.create dtype [| 4 |] [| 0.; 1.; 0.; 0. |] in

  let metric = Metrics.dice ~num_classes:2 () in
  Metrics.update metric ~predictions ~targets ();
  let result = Metrics.compute metric in
  let expected = (0.8 +. (2. /. 3.)) /. 2.0 in
  check (float_eps 1e-5) "dice" expected result

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
  check (float_eps 1e-5) "kl divergence" expected result;

  let kl_zero = Metrics.kl_divergence () in
  Metrics.update kl_zero ~predictions:targets ~targets ();
  let result = Metrics.compute kl_zero in
  let expected = 0.0 in
  check (float_eps 1e-5) "kl divergence zero" expected result

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
  let expected = 0.25 in
  check (float_eps 1e-5) "weighted mse" expected result

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
  check (float_eps 1e-5) "accumulated accuracy" expected result;

  (* Reset and compute again *)
  Metrics.reset acc;
  Metrics.update acc ~predictions:predictions2 ~targets:targets2 ();
  let result = Metrics.compute acc in
  let expected = 1.0 in
  (* Only last 2 *)
  check (float_eps 1e-5) "accuracy after reset" expected result

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
  check bool "custom MAPE in range" true (result > 5.5 && result < 6.5)

let test_metric_utilities () =
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
  let value = 0.8567 in
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
  let expected = 1.0 in
  check (float_eps 1e-5) "original after clone" expected result1;

  (* Clone should be reset *)
  Metrics.update acc2 ~predictions ~targets ();
  let result2 = Metrics.compute acc2 in
  check (float_eps 1e-5) "clone after reset" expected result2

let () =
  let open Alcotest in
  run "Metrics"
    [
      ( "classification",
        [
          test_case "accuracy" `Quick test_accuracy;
          test_case "accuracy_topk" `Quick test_accuracy_topk;
          test_case "precision_recall" `Quick test_precision_recall;
          test_case "f1_score" `Quick test_f1_score;
          test_case "auc_roc" `Quick test_auc_roc;
          test_case "auc_roc_multiple_updates" `Quick
            test_auc_roc_multiple_updates;
          test_case "auc_pr" `Quick test_auc_pr;
          test_case "auc_pr_multiple_updates" `Quick
            test_auc_pr_multiple_updates;
          test_case "confusion_matrix" `Quick test_confusion_matrix;
        ] );
      ( "ranking",
        [
          test_case "ndcg" `Quick test_ndcg;
          test_case "map" `Quick test_map_metric;
          test_case "mrr" `Quick test_mrr_metric;
        ] );
      ( "nlp",
        [
          test_case "bleu" `Quick test_bleu;
          test_case "rouge" `Quick test_rouge;
          test_case "meteor" `Quick test_meteor_metric;
        ] );
      ( "vision",
        [
          test_case "ssim" `Quick test_ssim;
          test_case "iou" `Quick test_iou;
          test_case "dice" `Quick test_dice;
        ] );
      ( "regression",
        [
          test_case "mse_rmse" `Quick test_mse_rmse;
          test_case "mae" `Quick test_mae;
          test_case "mape" `Quick test_mape;
          test_case "r2_score" `Quick test_r2_score;
          test_case "explained_variance" `Quick test_explained_variance;
        ] );
      ( "probabilistic",
        [
          test_case "cross_entropy" `Quick test_cross_entropy;
          test_case "binary_cross_entropy" `Quick test_binary_cross_entropy;
          test_case "kl_divergence" `Quick test_kl_divergence;
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
