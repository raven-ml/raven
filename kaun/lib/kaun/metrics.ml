(** Core types *)

type reduction = Mean | Sum | None
type averaging = Micro | Macro | Weighted | Samples

type 'layout metric_state = {
  mutable state_tensors : (float, 'layout) Rune.t list;
  name : string;
  init_fn : unit -> (float, 'layout) Rune.t list;
  update_fn :
    (float, 'layout) Rune.t list ->
    predictions:(float, 'layout) Rune.t ->
    targets:(float, 'layout) Rune.t ->
    ?weights:(float, 'layout) Rune.t ->
    unit ->
    (float, 'layout) Rune.t list;
  compute_fn : (float, 'layout) Rune.t list -> (float, 'layout) Rune.t;
  reset_fn : (float, 'layout) Rune.t list -> (float, 'layout) Rune.t list;
}

type 'layout t = 'layout metric_state

type 'layout metric_fn =
  predictions:(float, 'layout) Rune.t ->
  targets:(float, 'layout) Rune.t ->
  ?weights:(float, 'layout) Rune.t ->
  unit ->
  (float, 'layout) Rune.t

(** Helper functions *)

let scalar_tensor dtype value = Rune.scalar dtype value
let ones_like t = Rune.ones (Rune.dtype t) (Rune.shape t)

(** Core metric operations *)

let update metric ~predictions ~targets ?weights () =
  metric.state_tensors <-
    metric.update_fn metric.state_tensors ~predictions ~targets ?weights ()

let compute metric = metric.compute_fn metric.state_tensors
let reset metric = metric.state_tensors <- metric.reset_fn metric.state_tensors
let clone metric = { metric with state_tensors = metric.init_fn () }
let name metric = metric.name

(** Custom metric creation *)

let create_custom ~name ~init ~update ~compute ~reset =
  {
    state_tensors = init ();
    name;
    init_fn = init;
    update_fn = update;
    compute_fn = compute;
    reset_fn = reset;
  }

(** Classification Metrics *)

let accuracy ?(threshold = 0.5) ?top_k ?(averaging = Micro) () =
  let _ = averaging in
  let name =
    match top_k with
    | Some k -> Printf.sprintf "accuracy@%d" k
    | None -> "accuracy"
  in
  create_custom ~name
    ~init:(fun () ->
      (* We need to create these with a device and dtype, but we don't have them
         yet So we'll initialize with dummy values and replace on first
         update *)
      [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let dtype = Rune.dtype predictions in

      let correct_count, total_count =
        match state with
        | [ c; t ] -> (c, t)
        | _ ->
            (* First call - initialize *)
            (scalar_tensor dtype 0.0, scalar_tensor dtype 0.0)
      in

      let batch_correct, batch_total =
        match top_k with
        | Some _k ->
            (* Top-k accuracy - placeholder until topk is available *)
            failwith
              "Top-k accuracy not yet implemented - requires topk operation"
        | None ->
            (* Standard accuracy *)
            let correct =
              if
                Array.length (Rune.shape predictions) > 1
                && (Rune.shape predictions).(Array.length
                                               (Rune.shape predictions)
                                             - 1)
                   > 1
              then
                (* Multi-class: use argmax *)
                let preds =
                  Rune.argmax predictions ~axis:(-1) ~keepdims:false
                in
                let targets_int32 = Rune.cast Rune.int32 targets in
                Rune.equal preds targets_int32
              else
                (* Binary: threshold *)
                let threshold_t = scalar_tensor dtype threshold in
                let preds_binary = Rune.greater predictions threshold_t in
                let targets_binary = Rune.greater targets threshold_t in
                Rune.equal preds_binary targets_binary
            in
            let correct_float = Rune.cast dtype correct in
            (correct_float, ones_like correct_float)
      in

      (* Apply weights if provided *)
      let batch_correct, batch_total =
        match weights with
        | Some w -> (Rune.mul batch_correct w, Rune.mul batch_total w)
        | None -> (batch_correct, batch_total)
      in

      let new_correct = Rune.add correct_count (Rune.sum batch_correct) in
      let new_total = Rune.add total_count (Rune.sum batch_total) in
      [ new_correct; new_total ])
    ~compute:(fun state ->
      match state with
      | [ correct; total ] -> Rune.div correct total
      | _ -> failwith "Invalid accuracy state")
    ~reset:(fun _ -> [])

let precision ?(threshold = 0.5) ?(averaging = Micro) ?(zero_division = 0.0) ()
    =
  let _ = averaging in
  create_custom ~name:"precision"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let dtype = Rune.dtype predictions in

      let tp, fp =
        match state with
        | [ tp; fp ] -> (tp, fp)
        | _ -> (scalar_tensor dtype 0.0, scalar_tensor dtype 0.0)
      in

      (* Binary predictions *)
      let threshold_t = scalar_tensor dtype threshold in
      let preds = Rune.greater predictions threshold_t in
      let preds_float = Rune.cast dtype preds in
      let targets_float = Rune.cast dtype targets in

      (* True positives: predicted positive and actually positive *)
      let batch_tp = Rune.mul preds_float targets_float in

      (* False positives: predicted positive but actually negative *)
      let neg_targets = Rune.sub (ones_like targets_float) targets_float in
      let batch_fp = Rune.mul preds_float neg_targets in

      (* Apply weights if provided *)
      let batch_tp, batch_fp =
        match weights with
        | Some w -> (Rune.mul batch_tp w, Rune.mul batch_fp w)
        | None -> (batch_tp, batch_fp)
      in

      let new_tp = Rune.add tp (Rune.sum batch_tp) in
      let new_fp = Rune.add fp (Rune.sum batch_fp) in
      [ new_tp; new_fp ])
    ~compute:(fun state ->
      match state with
      | [ tp; fp ] ->
          let dtype = Rune.dtype tp in
          let denominator = Rune.add tp fp in
          let eps = scalar_tensor dtype 1e-7 in
          let is_zero = Rune.less denominator eps in
          let zero_val = scalar_tensor dtype zero_division in
          let precision_val = Rune.div tp (Rune.add denominator eps) in
          Rune.where is_zero zero_val precision_val
      | _ -> failwith "Invalid precision state")
    ~reset:(fun _ -> [])

let recall ?(threshold = 0.5) ?(averaging = Micro) ?(zero_division = 0.0) () =
  let _ = averaging in
  create_custom ~name:"recall"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let dtype = Rune.dtype predictions in

      let tp, fn =
        match state with
        | [ tp; fn ] -> (tp, fn)
        | _ -> (scalar_tensor dtype 0.0, scalar_tensor dtype 0.0)
      in

      (* Binary predictions *)
      let threshold_t = scalar_tensor dtype threshold in
      let preds = Rune.greater predictions threshold_t in
      let preds_float = Rune.cast dtype preds in
      let targets_float = Rune.cast dtype targets in

      (* True positives: predicted positive and actually positive *)
      let batch_tp = Rune.mul preds_float targets_float in

      (* False negatives: predicted negative but actually positive *)
      let neg_preds = Rune.sub (ones_like preds_float) preds_float in
      let batch_fn = Rune.mul neg_preds targets_float in

      (* Apply weights if provided *)
      let batch_tp, batch_fn =
        match weights with
        | Some w -> (Rune.mul batch_tp w, Rune.mul batch_fn w)
        | None -> (batch_tp, batch_fn)
      in

      let new_tp = Rune.add tp (Rune.sum batch_tp) in
      let new_fn = Rune.add fn (Rune.sum batch_fn) in
      [ new_tp; new_fn ])
    ~compute:(fun state ->
      match state with
      | [ tp; fn ] ->
          let dtype = Rune.dtype tp in
          let denominator = Rune.add tp fn in
          let eps = scalar_tensor dtype 1e-7 in
          let is_zero = Rune.less denominator eps in
          let zero_val = scalar_tensor dtype zero_division in
          let recall_val = Rune.div tp (Rune.add denominator eps) in
          Rune.where is_zero zero_val recall_val
      | _ -> failwith "Invalid recall state")
    ~reset:(fun _ -> [])

let f1_score ?(threshold = 0.5) ?(averaging = Micro) ?(beta = 1.0) () =
  let _ = averaging in
  create_custom
    ~name:(Printf.sprintf "f%.1f_score" beta)
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let dtype = Rune.dtype predictions in

      let tp, fp, fn =
        match state with
        | [ tp; fp; fn ] -> (tp, fp, fn)
        | _ ->
            ( scalar_tensor dtype 0.0,
              scalar_tensor dtype 0.0,
              scalar_tensor dtype 0.0 )
      in

      (* Binary predictions *)
      let threshold_t = scalar_tensor dtype threshold in
      let preds = Rune.greater predictions threshold_t in
      let preds_float = Rune.cast dtype preds in
      let targets_float = Rune.cast dtype targets in

      (* Compute TP, FP, FN *)
      let batch_tp = Rune.mul preds_float targets_float in
      let neg_targets = Rune.sub (ones_like targets_float) targets_float in
      let batch_fp = Rune.mul preds_float neg_targets in
      let neg_preds = Rune.sub (ones_like preds_float) preds_float in
      let batch_fn = Rune.mul neg_preds targets_float in

      (* Apply weights if provided *)
      let batch_tp, batch_fp, batch_fn =
        match weights with
        | Some w ->
            (Rune.mul batch_tp w, Rune.mul batch_fp w, Rune.mul batch_fn w)
        | None -> (batch_tp, batch_fp, batch_fn)
      in

      let new_tp = Rune.add tp (Rune.sum batch_tp) in
      let new_fp = Rune.add fp (Rune.sum batch_fp) in
      let new_fn = Rune.add fn (Rune.sum batch_fn) in
      [ new_tp; new_fp; new_fn ])
    ~compute:(fun state ->
      match state with
      | [ tp; fp; fn ] ->
          let dtype = Rune.dtype tp in
          let beta_sq = beta *. beta in
          let beta_sq_t = scalar_tensor dtype beta_sq in
          let one_plus_beta_sq = scalar_tensor dtype (1.0 +. beta_sq) in

          let precision_denom = Rune.add tp fp in
          let recall_denom = Rune.add tp fn in
          let eps = scalar_tensor dtype 1e-7 in

          let precision = Rune.div tp (Rune.add precision_denom eps) in
          let recall = Rune.div tp (Rune.add recall_denom eps) in

          let numerator =
            Rune.mul one_plus_beta_sq (Rune.mul precision recall)
          in
          let denominator = Rune.add (Rune.mul beta_sq_t precision) recall in

          Rune.div numerator (Rune.add denominator eps)
      | _ -> failwith "Invalid f1 state")
    ~reset:(fun _ -> [])

(* Placeholder implementations for complex metrics *)
let auc_roc ?(num_thresholds = 200) ?(curve = false) () =
  let _ = num_thresholds in
  let _ = curve in
  create_custom ~name:"auc_roc"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let predictions = Rune.reshape [| -1 |] predictions in
      let targets = Rune.reshape [| -1 |] targets in
      let dtype =
        match state with
        | [ preds_acc; _; _ ] -> Rune.dtype preds_acc
        | _ -> Rune.dtype predictions
      in
      let predictions = Rune.cast dtype predictions in
      let targets = Rune.cast dtype targets in
      let weights =
        match weights with
        | Some w -> Rune.cast dtype (Rune.reshape [| -1 |] w)
        | None -> Rune.ones dtype (Rune.shape predictions)
      in
      match state with
      | [] -> [ predictions; targets; weights ]
      | [ preds_acc; targets_acc; weights_acc ] ->
          let preds_acc =
            Rune.concatenate ~axis:0 [ preds_acc; predictions ]
          in
          let targets_acc =
            Rune.concatenate ~axis:0 [ targets_acc; targets ]
          in
          let weights_acc =
            Rune.concatenate ~axis:0 [ weights_acc; weights ]
          in
          [ preds_acc; targets_acc; weights_acc ]
      | _ -> failwith "Invalid auc_roc state")
    ~compute:(fun state ->
      match state with
      | [ preds; targets; weights ] ->
          let dtype = Rune.dtype preds in
          let ones = Rune.ones dtype (Rune.shape targets) in
          let sorted_idx =
            Rune.argsort ~axis:0 ~descending:true preds
          in
          let sorted_targets =
            Rune.take_along_axis ~axis:0 sorted_idx targets
          in
          let sorted_weights =
            Rune.take_along_axis ~axis:0 sorted_idx weights
          in

          let positives = Rune.mul sorted_targets sorted_weights in
          let negatives =
            Rune.mul (Rune.sub ones sorted_targets) sorted_weights
          in

          let cum_tp = Rune.cumsum ~axis:0 positives in
          let cum_fp = Rune.cumsum ~axis:0 negatives in
          let zero = scalar_tensor dtype 0.0 in
          let cum_tp =
            Rune.concatenate ~axis:0 [ Rune.reshape [| 1 |] zero; cum_tp ]
          in
          let cum_fp =
            Rune.concatenate ~axis:0 [ Rune.reshape [| 1 |] zero; cum_fp ]
          in

          let total_pos = Rune.item [] (Rune.sum positives) in
          let total_neg = Rune.item [] (Rune.sum negatives) in

          let ratio cumulative total =
            if Float.abs total < 1e-12 then
              Rune.zeros dtype (Rune.shape cumulative)
            else
              let total_tensor = scalar_tensor dtype total in
              Rune.div cumulative total_tensor
          in

          let tpr = ratio cum_tp total_pos in
          let fpr = ratio cum_fp total_neg in

          let n = Rune.size tpr in
          if n < 2 then scalar_tensor dtype 0.0
          else
            let tail_fpr = Rune.slice [ Rune.R (1, n) ] fpr in
            let head_fpr = Rune.slice [ Rune.R (0, n - 1) ] fpr in
            let dx = Rune.sub tail_fpr head_fpr in

            let tail_tpr = Rune.slice [ Rune.R (1, n) ] tpr in
            let head_tpr = Rune.slice [ Rune.R (0, n - 1) ] tpr in
            let avg_tpr =
              Rune.mul (scalar_tensor dtype 0.5)
                (Rune.add tail_tpr head_tpr)
            in
            Rune.sum (Rune.mul dx avg_tpr)
      | _ -> failwith "Invalid auc_roc state")
    ~reset:(fun _ -> [])


let auc_pr ?(num_thresholds = 200) ?(curve = false) () =
  let _ = num_thresholds in
  let _ = curve in
  create_custom ~name:"auc_pr"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions:_ ~targets:_ ?weights:_ () -> state)
    ~compute:(fun _ -> failwith "AUC-PR not yet implemented")
    ~reset:(fun _ -> [])

let confusion_matrix ~num_classes ?(normalize = `None) () =
  let _ = normalize in
  create_custom ~name:"confusion_matrix"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let dtype = Rune.dtype predictions in

      let matrix =
        match state with
        | [ m ] -> m
        | _ ->
            (* Initialize matrix with zeros *)
            Rune.zeros dtype [| num_classes; num_classes |]
      in

      (* Convert predictions and targets to int32 for indexing *)
      let preds_int = Rune.cast Rune.int32 predictions in
      let targets_int = Rune.cast Rune.int32 targets in

      (* Flatten to 1D *)
      let preds_flat = Rune.reshape [| -1 |] preds_int in
      let targets_flat = Rune.reshape [| -1 |] targets_int in

      (* Compute linear indices for the confusion matrix *)
      (* index = target * num_classes + prediction *)
      let num_classes_t = Rune.scalar Rune.int32 (Int32.of_int num_classes) in
      let linear_indices =
        Rune.add (Rune.mul targets_flat num_classes_t) preds_flat
      in

      (* Create one-hot encoding of the linear indices *)
      let total_bins = num_classes * num_classes in

      (* Use scatter to accumulate counts *)
      (* For each sample, increment the corresponding bin *)
      let ones = Rune.ones dtype (Rune.shape preds_flat) in
      let weighted_ones =
        match weights with
        | Some w -> Rune.mul ones (Rune.reshape [| -1 |] w)
        | None -> ones
      in

      (* Manual bincount implementation using a loop *)
      (* Since we don't have a scatter operation, we'll use item access *)
      let counts_array = Array.make total_bins 0.0 in
      let n_samples = (Rune.shape preds_flat).(0) in
      for i = 0 to n_samples - 1 do
        let idx = Rune.item [ i ] linear_indices |> Int32.to_int in
        let weight = Rune.item [ i ] weighted_ones in
        counts_array.(idx) <- counts_array.(idx) +. weight
      done;

      let counts = Rune.create dtype [| total_bins |] counts_array in
      let counts_2d = Rune.reshape [| num_classes; num_classes |] counts in

      let new_matrix = Rune.add matrix counts_2d in
      [ new_matrix ])
    ~compute:(fun state ->
      match state with
      | [ matrix ] -> matrix
      | _ -> failwith "Invalid confusion_matrix state")
    ~reset:(fun _ -> [])

(** Regression Metrics *)

let mse ?(reduction = Mean) () =
  let _ = reduction in
  create_custom ~name:"mse"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let dtype = Rune.dtype predictions in

      let sse, count =
        match state with
        | [ s; c ] -> (s, c)
        | _ -> (scalar_tensor dtype 0.0, scalar_tensor dtype 0.0)
      in

      let diff = Rune.sub predictions targets in
      let squared_diff = Rune.mul diff diff in

      (* Apply weights if provided *)
      let squared_diff, batch_count =
        match weights with
        | Some w -> (Rune.mul squared_diff w, Rune.sum w)
        | None ->
            let n = float_of_int (Rune.numel squared_diff) in
            (squared_diff, scalar_tensor dtype n)
      in

      let new_sse = Rune.add sse (Rune.sum squared_diff) in
      let new_count = Rune.add count batch_count in
      [ new_sse; new_count ])
    ~compute:(fun state ->
      match state with
      | [ sse; count ] -> Rune.div sse count
      | _ -> failwith "Invalid mse state")
    ~reset:(fun _ -> [])

let rmse ?(reduction = Mean) () =
  let mse_metric = mse ~reduction () in
  create_custom ~name:"rmse" ~init:mse_metric.init_fn
    ~update:mse_metric.update_fn
    ~compute:(fun state ->
      let mse_val = mse_metric.compute_fn state in
      Rune.sqrt mse_val)
    ~reset:mse_metric.reset_fn

let mae ?(reduction = Mean) () =
  let _ = reduction in
  create_custom ~name:"mae"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let dtype = Rune.dtype predictions in

      let sae, count =
        match state with
        | [ s; c ] -> (s, c)
        | _ -> (scalar_tensor dtype 0.0, scalar_tensor dtype 0.0)
      in

      let diff = Rune.sub predictions targets in
      let abs_diff = Rune.abs diff in

      (* Apply weights if provided *)
      let abs_diff, batch_count =
        match weights with
        | Some w -> (Rune.mul abs_diff w, Rune.sum w)
        | None ->
            let n = float_of_int (Rune.numel abs_diff) in
            (abs_diff, scalar_tensor dtype n)
      in

      let new_sae = Rune.add sae (Rune.sum abs_diff) in
      let new_count = Rune.add count batch_count in
      [ new_sae; new_count ])
    ~compute:(fun state ->
      match state with
      | [ sae; count ] -> Rune.div sae count
      | [] -> Rune.scalar Rune.float32 Float.nan
      | _ -> failwith "Invalid mae state")
    ~reset:(fun _ -> [])

(** Loss Metric - tracks running average of loss values *)

let loss () =
  create_custom ~name:"loss"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions:_ ~targets:_ ?weights () ->
      (* The loss value should be passed via weights parameter *)
      let dtype =
        match weights with
        | Some w -> Rune.dtype w
        | None ->
            failwith "loss metric requires loss value in weights parameter"
      in

      let sum_loss, count =
        match state with
        | [ s; c ] -> (s, c)
        | _ -> (scalar_tensor dtype 0.0, scalar_tensor dtype 0.0)
      in

      match weights with
      | Some loss_value ->
          (* Accumulate loss value *)
          let new_sum = Rune.add sum_loss loss_value in
          let new_count = Rune.add count (scalar_tensor dtype 1.0) in
          [ new_sum; new_count ]
      | None -> state)
    ~compute:(fun state ->
      match state with
      | [ sum_loss; count ] -> Rune.div sum_loss count
      | _ -> failwith "Invalid loss state")
    ~reset:(fun _ -> [])

(* Placeholder implementations for remaining metrics *)

let mape ?(eps = 1e-7) () =
  let _ = eps in
  create_custom ~name:"mape"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions:_ ~targets:_ ?weights:_ () -> state)
    ~compute:(fun _ -> failwith "MAPE not yet implemented")
    ~reset:(fun _ -> [])

let r2_score ?(adjusted = false) ?num_features () =
  let _ = adjusted in
  let _ = num_features in
  create_custom ~name:"r2_score"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let dtype = Rune.dtype predictions in

      let ss_res, sum_targets, sum_sq_targets, count =
        match state with
        | [ ssr; st; sst; c ] -> (ssr, st, sst, c)
        | _ ->
            ( scalar_tensor dtype 0.0,
              scalar_tensor dtype 0.0,
              scalar_tensor dtype 0.0,
              scalar_tensor dtype 0.0 )
      in

      (* Compute residuals *)
      let residuals = Rune.sub targets predictions in
      let squared_residuals = Rune.mul residuals residuals in

      (* Compute sum of squared targets for SS_tot *)
      let squared_targets = Rune.mul targets targets in

      (* Apply weights if provided *)
      let squared_residuals, targets_weighted, squared_targets, batch_count =
        match weights with
        | Some w ->
            ( Rune.mul squared_residuals w,
              Rune.mul targets w,
              Rune.mul squared_targets w,
              Rune.sum w )
        | None ->
            let n = float_of_int (Rune.numel targets) in
            (squared_residuals, targets, squared_targets, scalar_tensor dtype n)
      in

      let new_ss_res = Rune.add ss_res (Rune.sum squared_residuals) in
      let new_sum_targets = Rune.add sum_targets (Rune.sum targets_weighted) in
      let new_sum_sq_targets =
        Rune.add sum_sq_targets (Rune.sum squared_targets)
      in
      let new_count = Rune.add count batch_count in
      [ new_ss_res; new_sum_targets; new_sum_sq_targets; new_count ])
    ~compute:(fun state ->
      match state with
      | [ ss_res; sum_targets; sum_sq_targets; count ] ->
          (* mean = sum_targets / count *)
          let mean_targets = Rune.div sum_targets count in
          (* SS_tot = sum(y^2) - n * mean^2 *)
          let mean_sq = Rune.mul mean_targets mean_targets in
          let ss_tot = Rune.sub sum_sq_targets (Rune.mul count mean_sq) in
          (* R² = 1 - SS_res / SS_tot *)
          let dtype = Rune.dtype ss_res in
          let one = scalar_tensor dtype 1.0 in
          let eps = scalar_tensor dtype 1e-7 in
          let r2 = Rune.sub one (Rune.div ss_res (Rune.add ss_tot eps)) in
          r2
      | _ -> failwith "Invalid r2_score state")
    ~reset:(fun _ -> [])

let explained_variance () =
  create_custom ~name:"explained_variance"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions:_ ~targets:_ ?weights:_ () -> state)
    ~compute:(fun _ -> failwith "Explained variance not yet implemented")
    ~reset:(fun _ -> [])

(** Probabilistic Metrics *)

let cross_entropy ?(from_logits = true) () =
  create_custom ~name:"cross_entropy"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let dtype = Rune.dtype predictions in

      let sum_ce, count =
        match state with
        | [ s; c ] -> (s, c)
        | _ -> (scalar_tensor dtype 0.0, scalar_tensor dtype 0.0)
      in

      let ce =
        if from_logits then
          (* Check if targets are indices or one-hot encoded *)
          let target_shape = Rune.shape targets in
          if
            Array.length target_shape = 1
            || Array.length target_shape > 0
               && target_shape.(Array.length target_shape - 1) = 1
          then
            (* Targets are class indices *)
            Loss.softmax_cross_entropy_with_indices predictions targets
          else
            (* Targets are one-hot encoded *)
            Loss.softmax_cross_entropy predictions targets
        else
          (* Assume predictions are probabilities *)
          let eps = scalar_tensor dtype 1e-7 in
          let safe_preds = Rune.add predictions eps in
          let log_preds = Rune.log safe_preds in
          Rune.neg
            (Rune.mean (Rune.sum (Rune.mul targets log_preds) ~axes:[ -1 ]))
      in

      (* For batch accumulation *)
      let batch_count = scalar_tensor dtype 1.0 in

      (* Apply weights if provided *)
      let ce, batch_count =
        match weights with
        | Some w ->
            let weighted_ce = Rune.mul ce w in
            (weighted_ce, Rune.sum w)
        | None -> (ce, batch_count)
      in

      let new_sum = Rune.add sum_ce ce in
      let new_count = Rune.add count batch_count in
      [ new_sum; new_count ])
    ~compute:(fun state ->
      match state with
      | [ sum_ce; count ] -> Rune.div sum_ce count
      | _ -> failwith "Invalid cross_entropy state")
    ~reset:(fun _ -> [])

let binary_cross_entropy ?(from_logits = true) () =
  create_custom ~name:"binary_cross_entropy"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let dtype = Rune.dtype predictions in

      let sum_bce, count =
        match state with
        | [ s; c ] -> (s, c)
        | _ -> (scalar_tensor dtype 0.0, scalar_tensor dtype 0.0)
      in

      let bce =
        if from_logits then
          let per_sample =
            Loss.sigmoid_binary_cross_entropy predictions targets
          in
          Rune.mean per_sample
        else Loss.binary_cross_entropy predictions targets
      in

      let batch_count = scalar_tensor dtype 1.0 in

      (* Apply weights if provided *)
      let bce, batch_count =
        match weights with
        | Some w -> (Rune.mul bce (Rune.mean w), Rune.sum w)
        | None -> (bce, batch_count)
      in

      let new_sum = Rune.add sum_bce bce in
      let new_count = Rune.add count batch_count in
      [ new_sum; new_count ])
    ~compute:(fun state ->
      match state with
      | [ sum_bce; count ] -> Rune.div sum_bce count
      | _ -> failwith "Invalid binary_cross_entropy state")
    ~reset:(fun _ -> [])

let kl_divergence ?(eps = 1e-7) () =
  let _ = eps in
  create_custom ~name:"kl_divergence"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions:_ ~targets:_ ?weights:_ () -> state)
    ~compute:(fun _ -> failwith "KL divergence not yet implemented")
    ~reset:(fun _ -> [])

let perplexity ?(base = 2.71828) () =
  let ce_metric = cross_entropy ~from_logits:true () in
  create_custom ~name:"perplexity" ~init:ce_metric.init_fn
    ~update:ce_metric.update_fn
    ~compute:(fun state ->
      let ce = ce_metric.compute_fn state in
      let dtype = Rune.dtype ce in
      let base_t = scalar_tensor dtype base in
      Rune.pow base_t ce)
    ~reset:ce_metric.reset_fn

(* Placeholder implementations for remaining metrics *)

let ndcg ?k () =
  let _ = k in
  create_custom ~name:"ndcg"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions:_ ~targets:_ ?weights:_ () -> state)
    ~compute:(fun _ -> failwith "NDCG not yet implemented")
    ~reset:(fun _ -> [])

let map ?k () =
  let _ = k in
  create_custom ~name:"map"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions:_ ~targets:_ ?weights:_ () -> state)
    ~compute:(fun _ -> failwith "MAP not yet implemented")
    ~reset:(fun _ -> [])

let mrr () =
  create_custom ~name:"mrr"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions:_ ~targets:_ ?weights:_ () -> state)
    ~compute:(fun _ -> failwith "MRR not yet implemented")
    ~reset:(fun _ -> [])

let bleu ?(max_n = 4) ?weights ?(smoothing = true) ~tokenizer:_ () =
  let _ = max_n in
  let _ = weights in
  let _ = smoothing in
  create_custom ~name:"bleu"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions:_ ~targets:_ ?weights:_ () -> state)
    ~compute:(fun _ -> failwith "BLEU not yet implemented")
    ~reset:(fun _ -> [])

let rouge ~variant ?use_stemmer ~tokenizer:_ () =
  let _ = variant in
  let _ = use_stemmer in
  create_custom ~name:"rouge"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions:_ ~targets:_ ?weights:_ () -> state)
    ~compute:(fun _ -> failwith "ROUGE not yet implemented")
    ~reset:(fun _ -> [])

let meteor ?(alpha = 0.9) ?(beta = 3.0) ?(gamma = 0.5) ~tokenizer:_ () =
  let _ = alpha in
  let _ = beta in
  let _ = gamma in
  create_custom ~name:"meteor"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions:_ ~targets:_ ?weights:_ () -> state)
    ~compute:(fun _ -> failwith "METEOR not yet implemented")
    ~reset:(fun _ -> [])

let psnr ?(max_val = 1.0) () =
  let mse_metric = mse () in
  create_custom ~name:"psnr" ~init:mse_metric.init_fn
    ~update:mse_metric.update_fn
    ~compute:(fun state ->
      let mse_val = mse_metric.compute_fn state in
      let dtype = Rune.dtype mse_val in
      let max_val_sq = max_val *. max_val in
      let max_val_sq_t = scalar_tensor dtype max_val_sq in
      let ratio = Rune.div max_val_sq_t mse_val in
      let ten = scalar_tensor dtype 10.0 in
      (* log10(x) = log(x) / log(10) *)
      let log_ratio = Rune.log ratio in
      let log10_val = 2.302585093 in
      (* log(10) *)
      let log10_t = scalar_tensor dtype log10_val in
      Rune.mul ten (Rune.div log_ratio log10_t))
    ~reset:mse_metric.reset_fn

let ssim ?(window_size = 11) ?(k1 = 0.01) ?(k2 = 0.03) () =
  let _ = window_size in
  let _ = k1 in
  let _ = k2 in
  create_custom ~name:"ssim"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions:_ ~targets:_ ?weights:_ () -> state)
    ~compute:(fun _ -> failwith "SSIM not yet implemented")
    ~reset:(fun _ -> [])

let iou ?(threshold = 0.5) ?(per_class = false) ~num_classes () =
  let _ = threshold in
  let _ = per_class in
  let _ = num_classes in
  create_custom ~name:"iou"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions:_ ~targets:_ ?weights:_ () -> state)
    ~compute:(fun _ -> failwith "IoU not yet implemented")
    ~reset:(fun _ -> [])

let dice ?(threshold = 0.5) ?(per_class = false) ~num_classes () =
  let _ = threshold in
  let _ = per_class in
  let _ = num_classes in
  create_custom ~name:"dice"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions:_ ~targets:_ ?weights:_ () -> state)
    ~compute:(fun _ -> failwith "Dice coefficient not yet implemented")
    ~reset:(fun _ -> [])

(** Metric Collections *)

(* Capture outer module functions to avoid shadowing *)
let outer_update = update
let outer_compute = compute
let outer_reset = reset

module Collection = struct
  type 'layout metric = 'layout t
  type 'layout t = { mutable metrics : (string * 'layout metric) list }

  let create metrics = { metrics }

  let update collection ~predictions ~targets ?weights () =
    List.iter
      (fun (_, m) -> outer_update m ~predictions ~targets ?weights ())
      collection.metrics

  let update_with_loss collection ~loss ~predictions ~targets () =
    List.iter
      (fun (name, m) ->
        if name = "loss" then
          (* Pass loss value as weights for the loss metric *)
          outer_update m ~predictions ~targets ~weights:loss ()
        else outer_update m ~predictions ~targets ())
      collection.metrics

  let compute collection =
    List.map (fun (name, m) -> (name, outer_compute m)) collection.metrics

  let compute_dict collection =
    let tbl = Hashtbl.create (List.length collection.metrics) in
    List.iter
      (fun (name, m) -> Hashtbl.add tbl name (outer_compute m))
      collection.metrics;
    tbl

  let reset collection =
    List.iter (fun (_, m) -> outer_reset m) collection.metrics

  let add collection name m =
    collection.metrics <- (name, m) :: collection.metrics

  let remove collection name =
    collection.metrics <-
      List.filter (fun (n, _) -> n <> name) collection.metrics
end

(** Utilities *)

let is_better _metric ~higher_better ~old_val ~new_val =
  if higher_better then new_val > old_val else new_val < old_val

let format metric value =
  (* Extract scalar value *)
  let scalar_val = Rune.item [] value in
  Printf.sprintf "%s: %.4f" (name metric) scalar_val
