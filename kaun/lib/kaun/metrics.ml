(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Core types *)

type reduction = Mean | Sum

type metric =
  | Metric : {
      mutable state_tensors : (float, 'layout) Rune.t list;
      dtype : (float, 'layout) Rune.dtype;
      name : string;
      init_fn : unit -> (float, 'layout) Rune.t list;
      update_fn :
        (float, 'layout) Rune.t list ->
        predictions:(float, 'layout) Rune.t ->
        targets:(float, 'layout) Rune.t ->
        ?loss:(float, 'layout) Rune.t ->
        ?weights:(float, 'layout) Rune.t ->
        unit ->
        (float, 'layout) Rune.t list;
      compute_fn : (float, 'layout) Rune.t list -> (float, 'layout) Rune.t;
      reset_fn : (float, 'layout) Rune.t list -> (float, 'layout) Rune.t list;
    }
      -> metric

(** Helper functions *)

let scalar_tensor dtype value = Rune.scalar dtype value
let ones_like t = Rune.ones (Rune.dtype t) (Rune.shape t)

let reshape_to_vector tensor =
  let shape = Rune.shape tensor in
  match Array.length shape with
  | 0 -> Rune.reshape [| 1 |] tensor
  | 1 -> tensor
  | _ -> Rune.reshape [| -1 |] tensor

let sum_last_axes tensor =
  let shape = Rune.shape tensor in
  match Array.length shape with
  | 0 | 1 -> tensor
  | rank ->
      let axes = List.init (rank - 1) (fun i -> i + 1) in
      Rune.sum ~axes tensor

let weights_to_vector ?weights values =
  match weights with
  | None -> None
  | Some w -> Some (reshape_to_vector w |> Rune.cast (Rune.dtype values))

let weighted_total ?weights dtype values =
  let values = reshape_to_vector values in
  match weights_to_vector ?weights values with
  | Some w ->
      let weights = Rune.cast dtype w in
      let weighted = Rune.mul values weights in
      (Rune.sum weighted, Rune.sum weights)
  | None ->
      let n = float_of_int (Rune.numel values) in
      (Rune.sum values, scalar_tensor dtype n)

let tensor_length1d tensor =
  let shape = Rune.shape tensor in
  match Array.length shape with
  | 0 -> 1
  | 1 -> shape.(0)
  | _ -> invalid_arg "Expected a 1D tensor for classification reductions"

let class_scalar tensor idx =
  let slice = Rune.slice [ Rune.R (idx, idx + 1) ] tensor in
  Rune.reshape [||] slice

let micro_counts tp pred support =
  let num_classes = tensor_length1d tp in
  if num_classes = 2 then
    let idx = num_classes - 1 in
    (class_scalar tp idx, class_scalar pred idx, class_scalar support idx)
  else (Rune.sum tp, Rune.sum pred, Rune.sum support)

let reshape_weights dtype num_samples weights =
  match weights with
  | None -> None
  | Some w -> Some (Rune.cast dtype (Rune.reshape [| num_samples |] w))

let expand_sample_weights num_samples weights =
  Option.map (fun w -> Rune.reshape [| num_samples; 1 |] w) weights

let prepare_class_indices ~metric_name ~threshold predictions targets =
  let dtype = Rune.dtype predictions in
  let shape = Rune.shape predictions in
  let rank = Array.length shape in
  if rank = 0 then
    invalid_arg (Printf.sprintf "%s expects at least one dimension" metric_name);
  let last_dim = shape.(rank - 1) in
  if rank = 1 || last_dim = 1 then
    let preds_flat = Rune.reshape [| -1 |] predictions in
    let samples_shape = Rune.shape preds_flat in
    let num_samples =
      match Array.length samples_shape with
      | 0 -> 1
      | 1 -> samples_shape.(0)
      | _ ->
          invalid_arg
            (Printf.sprintf "%s: unexpected prediction shape" metric_name)
    in
    let threshold_t = scalar_tensor dtype threshold in
    let preds_binary = Rune.greater_equal preds_flat threshold_t in
    let pred_indices = Rune.cast Rune.int32 preds_binary in
    let targets_flat = Rune.reshape [| num_samples |] targets in
    let target_thresh = scalar_tensor dtype 0.5 in
    let targets_binary = Rune.greater_equal targets_flat target_thresh in
    let target_indices = Rune.cast Rune.int32 targets_binary in
    (num_samples, 2, pred_indices, target_indices)
  else
    let num_classes = last_dim in
    let num_samples = Rune.numel predictions / num_classes in
    let logits = Rune.reshape [| num_samples; num_classes |] predictions in
    let pred_indices = Rune.argmax logits ~axis:(-1) ~keepdims:false in
    let target_elements = Rune.numel targets in
    let target_indices =
      if target_elements = num_samples * num_classes then
        let reshaped = Rune.reshape [| num_samples; num_classes |] targets in
        Rune.argmax reshaped ~axis:(-1) ~keepdims:false
      else if target_elements = num_samples then
        let flat = Rune.reshape [| num_samples |] targets in
        Rune.cast Rune.int32 flat
      else
        invalid_arg
          (Printf.sprintf "%s expects targets with %d or %d elements, got %d"
             metric_name num_samples
             (num_samples * num_classes)
             target_elements)
    in
    (num_samples, num_classes, pred_indices, target_indices)

let classification_counts ?(threshold = 0.5) ~metric_name predictions targets
    ?weights () =
  let dtype = Rune.dtype predictions in
  let num_samples, num_classes, pred_indices, target_indices =
    prepare_class_indices ~metric_name ~threshold predictions targets
  in
  let pred_one_hot =
    Rune.one_hot ~num_classes pred_indices |> Rune.cast dtype
  in
  let target_one_hot =
    Rune.one_hot ~num_classes target_indices |> Rune.cast dtype
  in
  let pred_one_hot = Rune.reshape [| num_samples; num_classes |] pred_one_hot in
  let target_one_hot =
    Rune.reshape [| num_samples; num_classes |] target_one_hot
  in
  let weights_vec = reshape_weights dtype num_samples weights in
  let weights_expanded = expand_sample_weights num_samples weights_vec in
  let apply_weights tensor =
    match weights_expanded with Some w -> Rune.mul tensor w | None -> tensor
  in
  let pred_weighted = apply_weights pred_one_hot in
  let target_weighted = apply_weights target_one_hot in
  let tp_matrix = Rune.mul pred_one_hot target_one_hot in
  let tp_weighted = apply_weights tp_matrix in
  let sum_axis0 tensor = Rune.sum ~axes:[ 0 ] tensor in
  let tp_per_class = sum_axis0 tp_weighted in
  let pred_per_class = sum_axis0 pred_weighted in
  let target_per_class = sum_axis0 target_weighted in
  (tp_per_class, pred_per_class, target_per_class)

let zeros_of_shape dtype shape =
  if Array.length shape = 0 then scalar_tensor dtype 0.0
  else Rune.zeros dtype shape

let prepare_ranking_inputs ?k predictions targets =
  let dtype = Rune.dtype predictions in
  let shape = Rune.shape predictions in
  let rank = Array.length shape in
  if rank = 0 then failwith "Ranking metrics require at least one dimension";
  let axis = rank - 1 in
  let num_items = shape.(axis) in
  let depth =
    match k with
    | None -> num_items
    | Some value when value <= 0 -> 0
    | Some value -> Int.min value num_items
  in
  let leading_shape = if axis = 0 then [||] else Array.sub shape 0 axis in
  let sorted_idx = Rune.argsort ~axis ~descending:true predictions in
  let sorted_targets = Rune.take_along_axis ~axis sorted_idx targets in
  (dtype, axis, depth, leading_shape, sorted_idx, sorted_targets)

let slice_top_k axis k tensor =
  let rank = Array.length (Rune.shape tensor) in
  let selectors =
    let rec build i acc =
      if i = rank then List.rev acc
      else
        let sel = if i = axis then Rune.R (0, k) else Rune.A in
        build (i + 1) (sel :: acc)
    in
    build 0 []
  in
  Rune.slice selectors tensor

let accumulate_metric_mean metric_name state dtype ?weights values =
  let sum, count =
    match state with
    | [ total; weight ] -> (total, weight)
    | [] -> (scalar_tensor dtype 0.0, scalar_tensor dtype 0.0)
    | _ -> failwith (Printf.sprintf "Invalid %s state" metric_name)
  in
  let batch_sum, batch_count = weighted_total ?weights dtype values in
  let new_sum = Rune.add sum batch_sum in
  let new_count = Rune.add count batch_count in
  [ new_sum; new_count ]

let compute_metric_mean metric_name state =
  match state with
  | [ sum; count ] ->
      let dtype = Rune.dtype sum in
      let eps = scalar_tensor dtype 1e-7 in
      Rune.div sum (Rune.add count eps)
  | _ -> failwith (Printf.sprintf "Invalid %s state" metric_name)

let apply_reduction reduction sum count =
  match reduction with
  | Mean ->
      let dtype = Rune.dtype sum in
      let eps = scalar_tensor dtype 1e-7 in
      Rune.div sum (Rune.add count eps)
  | Sum -> sum

let tensor_to_sequence_batch metric_name tensor =
  let shape = Rune.shape tensor in
  let dims = Array.length shape in
  let batch, seq_len =
    match dims with
    | 1 -> (1, shape.(0))
    | 2 -> (shape.(0), shape.(1))
    | _ ->
        invalid_arg
          (Printf.sprintf "%s expects tensors shaped [batch, seq_len]"
             metric_name)
  in
  let data = Rune.to_array tensor in
  let to_token v = int_of_float (Float.floor (v +. 0.5)) in
  let sequences =
    Array.init batch (fun b ->
        let offset = b * seq_len in
        Array.init seq_len (fun i -> to_token data.(offset + i)))
  in
  let trim tokens =
    let len = Array.length tokens in
    let buf = Array.make len 0 in
    let count = ref 0 in
    for i = 0 to len - 1 do
      let token = tokens.(i) in
      if token <> 0 then (
        buf.(!count) <- token;
        incr count)
    done;
    Array.sub buf 0 !count
  in
  Array.map trim sequences

let tensor_to_flat_ints tensor =
  let data = Rune.to_array tensor in
  Array.init (Array.length data) (fun i ->
      int_of_float (Float.floor (data.(i) +. 0.5)))

let flatten_predictions_for_classes ~num_classes ~threshold tensor =
  let data = Rune.to_array tensor in
  let len = Array.length data in
  let result = Array.make len 0 in
  match threshold with
  | Some th when num_classes = 2 ->
      for i = 0 to len - 1 do
        result.(i) <- (if data.(i) >= th then 1 else 0)
      done;
      result
  | _ -> Array.init len (fun i -> int_of_float (Float.floor (data.(i) +. 0.5)))

let ngram_key tokens start n =
  let buf = Buffer.create (n * 4) in
  for i = 0 to n - 1 do
    if i > 0 then Buffer.add_char buf ',';
    Buffer.add_string buf (string_of_int tokens.(start + i))
  done;
  Buffer.contents buf

let count_ngrams tokens n =
  let len = Array.length tokens in
  let tbl = Hashtbl.create (max 1 (len - n + 1)) in
  for i = 0 to len - n do
    let key = ngram_key tokens i n in
    let count = match Hashtbl.find_opt tbl key with Some c -> c | None -> 0 in
    Hashtbl.replace tbl key (count + 1)
  done;
  tbl

let clip_counts cand_counts ref_counts =
  Hashtbl.fold
    (fun key count acc ->
      let reference =
        match Hashtbl.find_opt ref_counts key with Some c -> c | None -> 0
      in
      acc + Int.min count reference)
    cand_counts 0

let bleu_precision cand ref n =
  let cand_counts = count_ngrams cand n in
  let ref_counts = count_ngrams ref n in
  let overlap = clip_counts cand_counts ref_counts in
  let total =
    let len = Array.length cand in
    max 0 (len - n + 1)
  in
  (overlap, total)

let lcs_length a b =
  let m = Array.length a in
  let n = Array.length b in
  let dp = Array.make_matrix (m + 1) (n + 1) 0 in
  for i = 1 to m do
    for j = 1 to n do
      if a.(i - 1) = b.(j - 1) then dp.(i).(j) <- dp.(i - 1).(j - 1) + 1
      else dp.(i).(j) <- Int.max dp.(i - 1).(j) dp.(i).(j - 1)
    done
  done;
  dp.(m).(n)

let meteor_alignment candidate reference =
  let m = Array.length candidate in
  let n = Array.length reference in
  let used = Array.make n false in
  let matches = ref [] in
  for i = 0 to m - 1 do
    let token = candidate.(i) in
    let rec find j =
      if j = n then Stdlib.Option.none
      else if (not used.(j)) && reference.(j) = token then Stdlib.Option.some j
      else find (j + 1)
    in
    match find 0 with
    | Stdlib.Option.None -> ()
    | Stdlib.Option.Some j ->
        used.(j) <- true;
        matches := (i, j) :: !matches
  done;
  let matches = List.rev !matches in
  let chunks =
    match matches with
    | [] -> 0
    | (i0, j0) :: rest ->
        let _, chunk_count =
          List.fold_left
            (fun ((prev_i, prev_j), count) (ci, cj) ->
              if ci = prev_i + 1 && cj = prev_j + 1 then ((ci, cj), count)
              else ((ci, cj), count + 1))
            ((i0, j0), 1)
            rest
        in
        chunk_count
  in
  (List.length matches, chunks)

type confusion_stats = {
  counts : int array array;
  row_sums : int array;
  col_sums : int array;
}

let compute_confusion_matrix ~metric_name ~num_classes preds targets =
  if Array.length preds <> Array.length targets then
    invalid_arg
      (Printf.sprintf
         "%s expects predictions and targets with matching element counts"
         metric_name);
  let matrix = Array.make_matrix num_classes num_classes 0 in
  for idx = 0 to Array.length preds - 1 do
    let pred = preds.(idx) in
    let target = targets.(idx) in
    if target < 0 || target >= num_classes then
      invalid_arg
        (Printf.sprintf "%s: target class %d outside [0, %d)" metric_name target
           num_classes);
    if pred < 0 || pred >= num_classes then
      invalid_arg
        (Printf.sprintf "%s: prediction class %d outside [0, %d)" metric_name
           pred num_classes);
    matrix.(target).(pred) <- matrix.(target).(pred) + 1
  done;
  let row_sums =
    Array.init num_classes (fun i -> Array.fold_left ( + ) 0 matrix.(i))
  in
  let col_sums =
    Array.init num_classes (fun j ->
        let sum = ref 0 in
        for i = 0 to num_classes - 1 do
          sum := !sum + matrix.(i).(j)
        done;
        !sum)
  in
  { counts = matrix; row_sums; col_sums }

let float_array arr = Array.map float arr

let update_pair_state metric_name state dtype left_values right_values =
  let len = Array.length left_values in
  if Array.length right_values <> len then
    invalid_arg
      (Printf.sprintf "%s internal error: mismatched state lengths" metric_name);
  let shape = [| len |] in
  let left_tensor = Rune.create dtype shape (float_array left_values) in
  let right_tensor = Rune.create dtype shape (float_array right_values) in
  match state with
  | [] -> [ left_tensor; right_tensor ]
  | [ left_prev; right_prev ] ->
      [ Rune.add left_prev left_tensor; Rune.add right_prev right_tensor ]
  | _ -> failwith (Printf.sprintf "Invalid %s state" metric_name)

let prepare_rank_curve_inputs preds targets weights =
  let dtype = Rune.dtype preds in
  let sorted_idx = Rune.argsort ~axis:0 ~descending:true preds in
  let sorted_targets = Rune.take_along_axis ~axis:0 sorted_idx targets in
  let sorted_weights = Rune.take_along_axis ~axis:0 sorted_idx weights in
  let positives = Rune.mul sorted_targets sorted_weights in
  let negatives =
    let ones = Rune.ones dtype (Rune.shape sorted_targets) in
    Rune.mul (Rune.sub ones sorted_targets) sorted_weights
  in
  (positives, negatives)

(** Core metric operations *)

let update : type layout.
    metric ->
    predictions:(float, layout) Rune.t ->
    targets:(_, layout) Rune.t ->
    ?loss:(float, layout) Rune.t ->
    ?weights:(float, layout) Rune.t ->
    unit ->
    unit =
 fun metric ~predictions ~targets ?loss ?weights () ->
  match metric with
  | Metric record ->
      let predictions' = Rune.cast record.dtype predictions in
      let targets' = Rune.cast record.dtype targets in
      let weights' = Option.map (Rune.cast record.dtype) weights in
      let loss' = Option.map (Rune.cast record.dtype) loss in
      record.state_tensors <-
        record.update_fn record.state_tensors ~predictions:predictions'
          ~targets:targets' ?loss:loss' ?weights:weights' ()

let compute (Metric metric) =
  let tensor = metric.compute_fn metric.state_tensors in
  Rune.item [] tensor

let compute_tensor (Metric metric) =
  let t = metric.compute_fn metric.state_tensors in
  Ptree.P t

let reset (Metric metric) =
  metric.state_tensors <- metric.reset_fn metric.state_tensors

let clone (Metric metric) =
  Metric { metric with state_tensors = metric.init_fn () }

let name (Metric metric) = metric.name

(** Custom metric creation *)

let create_custom ~dtype ~name ~init ~update ~compute ~reset =
  Metric
    {
      state_tensors = init ();
      dtype;
      name;
      init_fn = init;
      update_fn =
        (fun state ~predictions ~targets ?loss:_ ?weights () ->
          update state ~predictions ~targets ?weights ());
      compute_fn = compute;
      reset_fn = reset;
    }

(** Classification Metrics *)

let accuracy ?(threshold = 0.5) ?top_k () =
  let name =
    match top_k with
    | Some k -> Printf.sprintf "accuracy@%d" k
    | None -> "accuracy"
  in
  create_custom ~dtype:Rune.float32 ~name
    ~init:(fun () ->
      (* We need to create these with a device and dtype, but we don't have them
         yet So we'll initialize with dummy values and replace on first
         update *)
      [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let dtype = Rune.dtype predictions in
      let correct_acc, total_acc =
        match state with
        | [ c; t ] -> (c, t)
        | [] -> (scalar_tensor dtype 0.0, scalar_tensor dtype 0.0)
        | _ -> failwith "Invalid accuracy state"
      in
      let batch_correct, batch_total =
        match top_k with
        | Some k ->
            let shape = Rune.shape predictions in
            let rank = Array.length shape in
            if rank = 0 then failwith "Accuracy requires at least one dimension";
            let axis = rank - 1 in
            let num_classes = shape.(axis) in
            let depth =
              match k with
              | value when value <= 0 -> 0
              | value -> Int.min value num_classes
            in
            if depth = 0 then
              invalid_arg "Metrics.accuracy: top_k must be positive"
            else
              let sorted_idx =
                Rune.argsort ~axis ~descending:true predictions
              in
              let top_indices = slice_top_k axis depth sorted_idx in
              let targets_int32 = Rune.cast Rune.int32 targets in
              let targets_expanded = Rune.expand_dims [ -1 ] targets_int32 in
              let hits = Rune.equal top_indices targets_expanded in
              let hits_float = Rune.cast dtype hits in
              let per_example = Rune.sum ~axes:[ -1 ] hits_float in
              let clamped =
                let ones = ones_like per_example in
                Rune.minimum per_example ones
              in
              weighted_total ?weights dtype clamped
        | None ->
            let tp_per_class, _, support_per_class =
              classification_counts ~metric_name:name ~threshold predictions
                targets ?weights ()
            in
            (Rune.sum tp_per_class, Rune.sum support_per_class)
      in
      let new_correct = Rune.add correct_acc batch_correct in
      let new_total = Rune.add total_acc batch_total in
      [ new_correct; new_total ])
    ~compute:(fun state ->
      match state with
      | [ correct; total ] ->
          let dtype = Rune.dtype correct in
          let eps = scalar_tensor dtype 1e-7 in
          Rune.div correct (Rune.add total eps)
      | _ -> failwith "Invalid accuracy state")
    ~reset:(fun _ -> [])

let precision ?(threshold = 0.5) ?(zero_division = 0.0) () =
  create_custom ~dtype:Rune.float32 ~name:"precision"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let dtype = Rune.dtype predictions in
      let tp_per_class, pred_per_class, support_per_class =
        classification_counts ~metric_name:"precision" ~threshold predictions
          targets ?weights ()
      in
      let tp_scalar, pred_scalar, _ =
        micro_counts tp_per_class pred_per_class support_per_class
      in
      let tp_acc, pred_acc =
        match state with
        | [ tp; pred ] -> (tp, pred)
        | [] ->
            let zero = scalar_tensor dtype 0.0 in
            (zero, zero)
        | _ -> failwith "Invalid precision state"
      in
      let tp_total = Rune.add tp_acc tp_scalar in
      let pred_total = Rune.add pred_acc pred_scalar in
      [ tp_total; pred_total ])
    ~compute:(fun state ->
      match state with
      | [ tp_total; pred_total ] ->
          let dtype = Rune.dtype tp_total in
          let eps = scalar_tensor dtype 1e-7 in
          let zero_val = scalar_tensor dtype zero_division in
          let zero_mask = Rune.less pred_total eps in
          let value = Rune.div tp_total (Rune.add pred_total eps) in
          Rune.where zero_mask zero_val value
      | _ -> failwith "Invalid precision state")
    ~reset:(fun _ -> [])

let recall ?(threshold = 0.5) ?(zero_division = 0.0) () =
  create_custom ~dtype:Rune.float32 ~name:"recall"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let dtype = Rune.dtype predictions in
      let tp_per_class, pred_per_class, support_per_class =
        classification_counts ~metric_name:"recall" ~threshold predictions
          targets ?weights ()
      in
      let tp_scalar, _, support_scalar =
        micro_counts tp_per_class pred_per_class support_per_class
      in
      let tp_acc, support_acc =
        match state with
        | [ tp; support ] -> (tp, support)
        | [] ->
            let zero = scalar_tensor dtype 0.0 in
            (zero, zero)
        | _ -> failwith "Invalid recall state"
      in
      [ Rune.add tp_acc tp_scalar; Rune.add support_acc support_scalar ])
    ~compute:(fun state ->
      match state with
      | [ tp_total; support_total ] ->
          let dtype = Rune.dtype tp_total in
          let eps = scalar_tensor dtype 1e-7 in
          let zero_val = scalar_tensor dtype zero_division in
          let zero_mask = Rune.less support_total eps in
          let value = Rune.div tp_total (Rune.add support_total eps) in
          Rune.where zero_mask zero_val value
      | _ -> failwith "Invalid recall state")
    ~reset:(fun _ -> [])

let f1_score ?(threshold = 0.5) ?(beta = 1.0) () =
  create_custom ~dtype:Rune.float32
    ~name:(Printf.sprintf "f%.1f_score" beta)
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let dtype = Rune.dtype predictions in
      let tp_per_class, pred_per_class, support_per_class =
        classification_counts ~metric_name:"f1_score" ~threshold predictions
          targets ?weights ()
      in
      let tp_scalar, pred_scalar, support_scalar =
        micro_counts tp_per_class pred_per_class support_per_class
      in
      let tp_acc, pred_acc, support_acc =
        match state with
        | [ tp; pred; support ] -> (tp, pred, support)
        | [] ->
            let zero = scalar_tensor dtype 0.0 in
            (zero, zero, zero)
        | _ -> failwith "Invalid f1 state"
      in
      [
        Rune.add tp_acc tp_scalar;
        Rune.add pred_acc pred_scalar;
        Rune.add support_acc support_scalar;
      ])
    ~compute:(fun state ->
      match state with
      | [ tp_total; pred_total; support_total ] ->
          let dtype = Rune.dtype tp_total in
          let eps = scalar_tensor dtype 1e-7 in
          let zero = scalar_tensor dtype 0.0 in
          let beta_sq = beta *. beta in
          let beta_sq_t = scalar_tensor dtype beta_sq in
          let one_plus_beta_sq = scalar_tensor dtype (1.0 +. beta_sq) in
          let precision =
            let denom = Rune.add pred_total eps in
            let value = Rune.div tp_total denom in
            let zero_mask = Rune.less pred_total eps in
            Rune.where zero_mask zero value
          in
          let recall =
            let denom = Rune.add support_total eps in
            let value = Rune.div tp_total denom in
            let zero_mask = Rune.less support_total eps in
            Rune.where zero_mask zero value
          in
          let numerator =
            Rune.mul one_plus_beta_sq (Rune.mul precision recall)
          in
          let denominator = Rune.add (Rune.mul beta_sq_t precision) recall in
          let zero_mask = Rune.less denominator eps in
          let value = Rune.div numerator (Rune.add denominator eps) in
          Rune.where zero_mask zero value
      | _ -> failwith "Invalid f1 state")
    ~reset:(fun _ -> [])

let auc_roc () =
  let preds_chunks = ref [] in
  let targets_chunks = ref [] in
  let weights_chunks = ref [] in
  let reset_refs () =
    preds_chunks := [];
    targets_chunks := [];
    weights_chunks := []
  in
  let concat axis dtype tensors =
    match List.rev tensors with
    | [] -> Rune.zeros dtype [| 0 |]
    | [ x ] -> x
    | xs -> Rune.concatenate ~axis xs
  in
  create_custom ~dtype:Rune.float32 ~name:"auc_roc"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let _ = state in
      let preds = Rune.reshape [| -1 |] predictions in
      let dtype = Rune.dtype preds in
      let preds = Rune.cast dtype preds in
      let targets = Rune.cast dtype (Rune.reshape [| -1 |] targets) in
      let weights_tensor =
        match weights with
        | Some w -> Rune.cast dtype (Rune.reshape [| -1 |] w)
        | None -> Rune.ones dtype (Rune.shape preds)
      in
      preds_chunks := preds :: !preds_chunks;
      targets_chunks := targets :: !targets_chunks;
      weights_chunks := weights_tensor :: !weights_chunks;
      [ scalar_tensor dtype 0.0 ])
    ~compute:(fun _ ->
      match !preds_chunks with
      | [] -> failwith "auc_roc: metric has no data"
      | _ ->
          let dtype =
            match !weights_chunks with
            | [] -> failwith "auc_roc: metric has no data"
            | tensor :: _ -> Rune.dtype tensor
          in
          let preds = concat 0 dtype !preds_chunks in
          let targets = concat 0 dtype !targets_chunks in
          let weights = concat 0 dtype !weights_chunks in
          let positives, negatives =
            prepare_rank_curve_inputs preds targets weights
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
              Rune.mul (scalar_tensor dtype 0.5) (Rune.add tail_tpr head_tpr)
            in
            Rune.sum (Rune.mul dx avg_tpr))
    ~reset:(fun _ ->
      reset_refs ();
      [])

let auc_pr () =
  let preds_chunks = ref [] in
  let targets_chunks = ref [] in
  let weights_chunks = ref [] in
  let reset_refs () =
    preds_chunks := [];
    targets_chunks := [];
    weights_chunks := []
  in
  let concat axis dtype tensors =
    match List.rev tensors with
    | [] -> Rune.zeros dtype [| 0 |]
    | [ x ] -> x
    | xs -> Rune.concatenate ~axis xs
  in
  create_custom ~dtype:Rune.float32 ~name:"auc_pr"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let _ = state in
      let preds = Rune.reshape [| -1 |] predictions in
      let dtype = Rune.dtype preds in
      let preds = Rune.cast dtype preds in
      let targets = Rune.cast dtype (Rune.reshape [| -1 |] targets) in
      let weights_tensor =
        match weights with
        | Some w -> Rune.cast dtype (Rune.reshape [| -1 |] w)
        | None -> Rune.ones dtype (Rune.shape preds)
      in
      preds_chunks := preds :: !preds_chunks;
      targets_chunks := targets :: !targets_chunks;
      weights_chunks := weights_tensor :: !weights_chunks;
      [ scalar_tensor dtype 0.0 ])
    ~compute:(fun _ ->
      match !preds_chunks with
      | [] -> failwith "auc_pr: metric has no data"
      | _ ->
          let dtype =
            match !weights_chunks with
            | [] -> failwith "auc_pr: metric has no data"
            | tensor :: _ -> Rune.dtype tensor
          in
          let preds = concat 0 dtype !preds_chunks in
          let targets = concat 0 dtype !targets_chunks in
          let weights = concat 0 dtype !weights_chunks in
          let positives, negatives =
            prepare_rank_curve_inputs preds targets weights
          in
          let cum_tp = Rune.cumsum ~axis:0 positives in
          let cum_fp = Rune.cumsum ~axis:0 negatives in
          let cum_fn = Rune.sub (Rune.sum positives) cum_tp in
          let zero = scalar_tensor dtype 0.0 in
          let cum_tp =
            Rune.concatenate ~axis:0 [ Rune.reshape [| 1 |] zero; cum_tp ]
          in
          let cum_fp =
            Rune.concatenate ~axis:0 [ Rune.reshape [| 1 |] zero; cum_fp ]
          in
          let cum_fn =
            Rune.concatenate ~axis:0 [ Rune.reshape [| 1 |] zero; cum_fn ]
          in
          let precision_denom = Rune.add cum_tp cum_fp in
          let recall_denom = Rune.add cum_tp cum_fn in
          let eps = scalar_tensor dtype 1e-7 in
          let precision = Rune.div cum_tp (Rune.add precision_denom eps) in
          let recall = Rune.div cum_tp (Rune.add recall_denom eps) in
          let n = Rune.size precision in
          if n < 2 then scalar_tensor dtype 0.0
          else
            let tail_recall = Rune.slice [ Rune.R (1, n) ] recall in
            let head_recall = Rune.slice [ Rune.R (0, n - 1) ] recall in
            let dx = Rune.sub tail_recall head_recall in
            let precision_k = Rune.slice [ Rune.R (1, n) ] precision in
            Rune.sum (Rune.mul dx precision_k))
    ~reset:(fun _ ->
      reset_refs ();
      [])

let confusion_matrix ~num_classes ?(normalize = `None) () =
  create_custom ~dtype:Rune.float32 ~name:"confusion_matrix"
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
      let weights_flat =
        match weights with
        | Some w -> Rune.cast dtype (Rune.reshape [| -1 |] w)
        | None -> ones
      in

      (* Manual bincount implementation using a loop *)
      (* Since we don't have a scatter operation, we'll use item access *)
      let counts_array = Array.make total_bins 0.0 in
      let n_samples = (Rune.shape preds_flat).(0) in
      for i = 0 to n_samples - 1 do
        let idx = Rune.item [ i ] linear_indices |> Int32.to_int in
        let weight = Rune.item [ i ] weights_flat in
        counts_array.(idx) <- counts_array.(idx) +. weight
      done;

      let counts = Rune.create dtype [| total_bins |] counts_array in
      let counts_2d = Rune.reshape [| num_classes; num_classes |] counts in

      let new_matrix = Rune.add matrix counts_2d in
      [ new_matrix ])
    ~compute:(fun state ->
      match state with
      | [ matrix ] -> (
          let dtype = Rune.dtype matrix in
          let eps = scalar_tensor dtype 1e-7 in
          match normalize with
          | `None -> matrix
          | `All -> Rune.div matrix (Rune.add (Rune.sum matrix) eps)
          | `True ->
              let row_sums = Rune.sum ~axes:[ 1 ] matrix in
              let denom = Rune.add (Rune.expand_dims [ 1 ] row_sums) eps in
              Rune.div matrix denom
          | `Pred ->
              let col_sums = Rune.sum ~axes:[ 0 ] matrix in
              let denom = Rune.add (Rune.expand_dims [ 0 ] col_sums) eps in
              Rune.div matrix denom)
      | _ -> failwith "Invalid confusion_matrix state")
    ~reset:(fun _ -> [])

(** Regression Metrics *)

let mse ?(reduction = Mean) () =
  create_custom ~dtype:Rune.float32 ~name:"mse"
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

      let batch_sse, batch_count = weighted_total ?weights dtype squared_diff in
      let new_sse = Rune.add sse batch_sse in
      let new_count = Rune.add count batch_count in
      [ new_sse; new_count ])
    ~compute:(fun state ->
      match state with
      | [ sse; count ] -> apply_reduction reduction sse count
      | _ -> failwith "Invalid mse state")
    ~reset:(fun _ -> [])

let rmse ?(reduction = Mean) () =
  let _ = reduction in
  create_custom ~dtype:Rune.float32 ~name:"rmse"
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
      | [ sse; count ] ->
          let mse_val = Rune.div sse count in
          Rune.sqrt mse_val
      | _ -> failwith "Invalid rmse state")
    ~reset:(fun _ -> [])

let mae ?(reduction = Mean) () =
  create_custom ~dtype:Rune.float32 ~name:"mae"
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

      let batch_sae, batch_count = weighted_total ?weights dtype abs_diff in
      let new_sae = Rune.add sae batch_sae in
      let new_count = Rune.add count batch_count in
      [ new_sae; new_count ])
    ~compute:(fun state ->
      match state with
      | [ sae; count ] -> apply_reduction reduction sae count
      | [] -> failwith "mae: metric has no data"
      | _ -> failwith "Invalid mae state")
    ~reset:(fun _ -> [])

(** Loss Metric - tracks running average of loss values *)

let loss () =
  let dtype = Rune.float32 in
  let name = "loss" in
  let init () = [] in
  Metric
    {
      state_tensors = init ();
      dtype;
      name;
      init_fn = init;
      update_fn =
        (fun state ~predictions:_ ~targets:_ ?loss ?weights:_ () ->
          let loss_value =
            match loss with
            | Some l -> l
            | None -> failwith "loss metric requires loss value"
          in
          let dtype = Rune.dtype loss_value in
          let sum_loss, count =
            match state with
            | [ s; c ] -> (s, c)
            | [] -> (scalar_tensor dtype 0.0, scalar_tensor dtype 0.0)
            | _ -> failwith "Invalid loss state"
          in
          let new_sum = Rune.add sum_loss loss_value in
          let new_count = Rune.add count (scalar_tensor dtype 1.0) in
          [ new_sum; new_count ]);
      compute_fn =
        (fun state ->
          match state with
          | [ sum_loss; count ] -> Rune.div sum_loss count
          | [] -> failwith "loss: metric has no data"
          | _ -> failwith "Invalid loss state");
      reset_fn = (fun _ -> []);
    }

let mape ?(eps = 1e-7) () =
  create_custom ~dtype:Rune.float32 ~name:"mape"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let dtype = Rune.dtype predictions in
      let total_error, count =
        match state with
        | [ err; c ] -> (err, c)
        | _ -> (scalar_tensor dtype 0.0, scalar_tensor dtype 0.0)
      in

      let abs_diff = Rune.abs (Rune.sub predictions targets) in
      let denom = Rune.maximum (scalar_tensor dtype eps) (Rune.abs targets) in
      let abs_pct = Rune.div abs_diff denom in
      let sum_error, batch_count = weighted_total ?weights dtype abs_pct in
      let new_error = Rune.add total_error sum_error in
      let new_count = Rune.add count batch_count in
      [ new_error; new_count ])
    ~compute:(fun state ->
      match state with
      | [ total_error; count ] ->
          let dtype = Rune.dtype total_error in
          let eps_count = scalar_tensor dtype 1e-7 in
          let mean_error = Rune.div total_error (Rune.add count eps_count) in
          let hundred = scalar_tensor dtype 100.0 in
          Rune.mul hundred mean_error
      | _ -> failwith "Invalid mape state")
    ~reset:(fun _ -> [])

let r2_score ?(adjusted = false) ?num_features () =
  create_custom ~dtype:Rune.float32 ~name:"r2_score"
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

      let ss_res_inc, batch_count =
        weighted_total ?weights dtype squared_residuals
      in
      let sum_targets_inc, _ = weighted_total ?weights dtype targets in
      let sum_sq_targets_inc, _ =
        weighted_total ?weights dtype squared_targets
      in

      let new_ss_res = Rune.add ss_res ss_res_inc in
      let new_sum_targets = Rune.add sum_targets sum_targets_inc in
      let new_sum_sq_targets = Rune.add sum_sq_targets sum_sq_targets_inc in
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
          let zero = scalar_tensor dtype 0.0 in
          let eps = scalar_tensor dtype 1e-7 in
          let base_r2 = Rune.sub one (Rune.div ss_res (Rune.add ss_tot eps)) in
          let near_zero_tot = Rune.less (Rune.abs ss_tot) eps in
          let near_zero_res = Rune.less (Rune.abs ss_res) eps in
          let fallback = Rune.where near_zero_res one zero in
          let base_r2 = Rune.where near_zero_tot fallback base_r2 in
          if adjusted then
            match num_features with
            | None ->
                failwith "Adjusted R² requires [num_features] to be provided"
            | Some p ->
                let p_t = scalar_tensor dtype (float_of_int p) in
                let denom = Rune.sub count (Rune.add p_t one) in
                let denom_safe = Rune.add denom eps in
                let adjust_factor = Rune.div (Rune.sub count one) denom_safe in
                let adjusted_r2 =
                  Rune.sub one (Rune.mul (Rune.sub one base_r2) adjust_factor)
                in
                let invalid = Rune.less_equal denom eps in
                Rune.where invalid base_r2 adjusted_r2
          else base_r2
      | _ -> failwith "Invalid r2_score state")
    ~reset:(fun _ -> [])

let explained_variance () =
  create_custom ~dtype:Rune.float32 ~name:"explained_variance"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let dtype = Rune.dtype predictions in

      let sum_res, sum_sq_res, sum_targets, sum_sq_targets, count =
        match state with
        | [ sr; ssr; st; sst; c ] -> (sr, ssr, st, sst, c)
        | _ ->
            let zero = scalar_tensor dtype 0.0 in
            (zero, zero, zero, zero, zero)
      in

      let residuals = Rune.sub predictions targets in
      let squared_residuals = Rune.mul residuals residuals in
      let squared_targets = Rune.mul targets targets in

      let sum_res_inc, batch_count = weighted_total ?weights dtype residuals in
      let sum_sq_res_inc, _ = weighted_total ?weights dtype squared_residuals in
      let sum_targets_inc, _ = weighted_total ?weights dtype targets in
      let sum_sq_targets_inc, _ =
        weighted_total ?weights dtype squared_targets
      in

      let new_sum_res = Rune.add sum_res sum_res_inc in
      let new_sum_sq_res = Rune.add sum_sq_res sum_sq_res_inc in
      let new_sum_targets = Rune.add sum_targets sum_targets_inc in
      let new_sum_sq_targets = Rune.add sum_sq_targets sum_sq_targets_inc in
      let new_count = Rune.add count batch_count in
      [
        new_sum_res;
        new_sum_sq_res;
        new_sum_targets;
        new_sum_sq_targets;
        new_count;
      ])
    ~compute:(fun state ->
      match state with
      | [ sum_res; sum_sq_res; sum_targets; sum_sq_targets; count ] ->
          let dtype = Rune.dtype sum_res in
          let one = scalar_tensor dtype 1.0 in
          let zero = scalar_tensor dtype 0.0 in
          let eps = scalar_tensor dtype 1e-7 in

          let mean_res = Rune.div sum_res count in
          let mean_res_sq = Rune.mul mean_res mean_res in
          let var_res =
            Rune.maximum zero (Rune.sub (Rune.div sum_sq_res count) mean_res_sq)
          in

          let mean_targets = Rune.div sum_targets count in
          let mean_targets_sq = Rune.mul mean_targets mean_targets in
          let var_targets =
            Rune.maximum zero
              (Rune.sub (Rune.div sum_sq_targets count) mean_targets_sq)
          in

          let base_ev =
            Rune.sub one (Rune.div var_res (Rune.add var_targets eps))
          in
          let near_zero_var_targets = Rune.less (Rune.abs var_targets) eps in
          let near_zero_var_res = Rune.less (Rune.abs var_res) eps in
          let fallback = Rune.where near_zero_var_res one zero in
          Rune.where near_zero_var_targets fallback base_ev
      | _ -> failwith "Invalid explained_variance state")
    ~reset:(fun _ -> [])

(** Probabilistic Metrics *)

let cross_entropy ?(from_logits = true) () =
  let dtype_ref = ref None in
  create_custom ~dtype:Rune.float32 ~name:"cross_entropy"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let dtype = Rune.dtype predictions in
      (match !dtype_ref with None -> dtype_ref := Some dtype | Some _ -> ());

      let shape = Rune.shape predictions in
      let rank = Array.length shape in
      if rank = 0 then
        invalid_arg "cross_entropy expects predictions with at least one axis";
      let num_classes = shape.(rank - 1) in
      if num_classes <= 1 then
        invalid_arg "cross_entropy expects class dimension > 1";
      let num_samples = Rune.numel predictions / num_classes in
      let logits = Rune.reshape [| num_samples; num_classes |] predictions in
      let per_sample =
        if from_logits then
          let max_logits = Rune.max logits ~axes:[ -1 ] ~keepdims:true in
          let shifted = Rune.sub logits max_logits in
          let logsumexp =
            Rune.log (Rune.sum (Rune.exp shifted) ~axes:[ -1 ] ~keepdims:true)
          in
          let log_probs = Rune.sub shifted logsumexp in
          let target_shape = Rune.shape targets in
          if
            Array.length target_shape = 1
            || Array.length target_shape > 0
               && target_shape.(Array.length target_shape - 1) = 1
          then
            let indices =
              Rune.cast Rune.int32 (Rune.reshape [| num_samples |] targets)
            in
            let one_hot =
              Rune.one_hot ~num_classes indices |> Rune.cast dtype
            in
            let losses =
              Rune.neg (Rune.sum (Rune.mul one_hot log_probs) ~axes:[ 1 ])
            in
            Rune.reshape [| num_samples |] losses
          else
            let flat_targets =
              Rune.reshape [| num_samples; num_classes |] targets
              |> Rune.cast dtype
            in
            let losses =
              Rune.neg (Rune.sum (Rune.mul flat_targets log_probs) ~axes:[ 1 ])
            in
            Rune.reshape [| num_samples |] losses
        else
          let probs =
            let eps = scalar_tensor dtype 1e-7 in
            let reshaped =
              Rune.reshape [| num_samples; num_classes |] predictions
            in
            Rune.maximum eps reshaped
          in
          let log_probs = Rune.log probs in
          let target_shape = Rune.shape targets in
          if
            Array.length target_shape = 1
            || Array.length target_shape > 0
               && target_shape.(Array.length target_shape - 1) = 1
          then
            let indices =
              Rune.cast Rune.int32 (Rune.reshape [| num_samples |] targets)
            in
            let one_hot =
              Rune.one_hot ~num_classes indices |> Rune.cast dtype
            in
            let losses =
              Rune.neg (Rune.sum (Rune.mul one_hot log_probs) ~axes:[ 1 ])
            in
            Rune.reshape [| num_samples |] losses
          else
            let flat_targets =
              Rune.reshape [| num_samples; num_classes |] targets
              |> Rune.cast dtype
            in
            let losses =
              Rune.neg (Rune.sum (Rune.mul flat_targets log_probs) ~axes:[ 1 ])
            in
            Rune.reshape [| num_samples |] losses
      in
      accumulate_metric_mean "cross_entropy" state dtype ?weights per_sample)
    ~compute:(compute_metric_mean "cross_entropy")
    ~reset:(fun _ ->
      dtype_ref := None;
      [])

let binary_cross_entropy ?(from_logits = true) () =
  let dtype_ref = ref None in
  create_custom ~dtype:Rune.float32 ~name:"binary_cross_entropy"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let dtype = Rune.dtype predictions in
      (match !dtype_ref with None -> dtype_ref := Some dtype | Some _ -> ());
      let preds = Rune.cast dtype predictions in
      let targets = Rune.cast dtype targets in
      let one = scalar_tensor dtype 1.0 in
      let per_element =
        if from_logits then
          let log_sig = Rune.log_sigmoid preds in
          let log_sig_neg = Rune.log_sigmoid (Rune.neg preds) in
          let term1 = Rune.mul targets log_sig in
          let term2 = Rune.mul (Rune.sub one targets) log_sig_neg in
          Rune.neg (Rune.add term1 term2)
        else
          let eps = scalar_tensor dtype 1e-7 in
          let preds_clipped =
            Rune.maximum eps (Rune.minimum (Rune.sub one eps) preds)
          in
          let term1 = Rune.mul targets (Rune.log preds_clipped) in
          let term2 =
            Rune.mul (Rune.sub one targets)
              (Rune.log (Rune.sub one preds_clipped))
          in
          Rune.neg (Rune.add term1 term2)
      in
      let per_sample = sum_last_axes per_element in
      accumulate_metric_mean "binary_cross_entropy" state dtype ?weights
        per_sample)
    ~compute:(compute_metric_mean "binary_cross_entropy")
    ~reset:(fun _ ->
      dtype_ref := None;
      [])

let kl_divergence ?(eps = 1e-7) () =
  create_custom ~dtype:Rune.float32 ~name:"kl_divergence"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let dtype = Rune.dtype predictions in
      let sum_kl, count =
        match state with
        | [ s; c ] -> (s, c)
        | _ -> (scalar_tensor dtype 0.0, scalar_tensor dtype 0.0)
      in

      let eps_t = scalar_tensor dtype eps in
      let safe_preds = Rune.maximum eps_t predictions in
      let safe_targets = Rune.maximum eps_t targets in
      let log_ratio = Rune.sub (Rune.log safe_targets) (Rune.log safe_preds) in
      let element_kl = Rune.mul safe_targets log_ratio in
      let dims = Array.length (Rune.shape element_kl) in
      let per_example =
        if dims = 0 then element_kl else Rune.sum ~axes:[ dims - 1 ] element_kl
      in
      let per_example = reshape_to_vector per_example in
      let weights_opt =
        Option.map
          (fun w ->
            let w_dims = Array.length (Rune.shape w) in
            let reduced =
              if w_dims = dims then Rune.sum ~axes:[ w_dims - 1 ] w else w
            in
            reshape_to_vector reduced)
          weights
      in
      let sum_kl_inc, batch_count =
        weighted_total ?weights:weights_opt dtype per_example
      in
      let new_sum = Rune.add sum_kl sum_kl_inc in
      let new_count = Rune.add count batch_count in
      [ new_sum; new_count ])
    ~compute:(fun state ->
      match state with
      | [ sum_kl; count ] ->
          let dtype = Rune.dtype sum_kl in
          let eps_count = scalar_tensor dtype 1e-7 in
          Rune.div sum_kl (Rune.add count eps_count)
      | _ -> failwith "Invalid kl_divergence state")
    ~reset:(fun _ -> [])

let perplexity ?(base = Float.exp 1.) () =
  match cross_entropy ~from_logits:true () with
  | Metric ce ->
      create_custom ~dtype:ce.dtype ~name:"perplexity" ~init:ce.init_fn
        ~update:(fun state ~predictions ~targets ?weights () ->
          ce.update_fn state ~predictions ~targets ?weights ())
        ~compute:(fun state ->
          let ce_t = ce.compute_fn state in
          let dtype = Rune.dtype ce_t in
          let base_t = scalar_tensor dtype base in
          Rune.pow base_t ce_t)
        ~reset:ce.reset_fn

let ndcg ?k () =
  create_custom ~dtype:Rune.float32 ~name:"ndcg"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let dtype, axis, depth, leading_shape, _, sorted_targets =
        prepare_ranking_inputs ?k predictions targets
      in
      let ndcg_values =
        if depth = 0 then zeros_of_shape dtype leading_shape
        else
          let top_targets = slice_top_k axis depth sorted_targets in
          let ideal_idx = Rune.argsort ~axis ~descending:true targets in
          let ideal_sorted = Rune.take_along_axis ~axis ideal_idx targets in
          let ideal_top = slice_top_k axis depth ideal_sorted in
          let two = scalar_tensor dtype 2.0 in
          let log_two = scalar_tensor dtype (Float.log 2.0) in
          let positions = Rune.arange_f dtype 0.0 (float depth) 1.0 in
          let discounts =
            let denom =
              Rune.log (Rune.add positions (scalar_tensor dtype 2.0))
            in
            let discount = Rune.div log_two denom in
            let discount_shape =
              Array.init (axis + 1) (fun i -> if i = axis then depth else 1)
            in
            Rune.reshape discount_shape discount
          in
          let gains targets =
            let gains =
              Rune.sub (Rune.pow two targets) (scalar_tensor dtype 1.0)
            in
            Rune.mul gains discounts
          in
          let dcg = Rune.sum ~axes:[ -1 ] (gains top_targets) in
          let ideal_dcg = Rune.sum ~axes:[ -1 ] (gains ideal_top) in
          let zero = scalar_tensor dtype 0.0 in
          let eps = scalar_tensor dtype 1e-7 in
          let raw = Rune.div dcg (Rune.add ideal_dcg eps) in
          let mask = Rune.less ideal_dcg eps in
          Rune.where mask zero raw
      in
      accumulate_metric_mean "ndcg" state dtype ?weights ndcg_values)
    ~compute:(compute_metric_mean "ndcg")
    ~reset:(fun _ -> [])

let map ?k () =
  create_custom ~dtype:Rune.float32 ~name:"map"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let dtype, axis, depth, leading_shape, _, sorted_targets =
        prepare_ranking_inputs ?k predictions targets
      in
      let map_values =
        if depth = 0 then zeros_of_shape dtype leading_shape
        else
          let top_targets = slice_top_k axis depth sorted_targets in
          let zero = scalar_tensor dtype 0.0 in
          let rel_mask = Rune.greater top_targets zero in
          let rel_float = Rune.cast dtype rel_mask in
          let cumsum_rel = Rune.cumsum ~axis:(-1) rel_float in
          let positions = Rune.arange_f dtype 1.0 (float depth +. 1.0) 1.0 in
          let position_shape =
            Array.init (axis + 1) (fun i -> if i = axis then depth else 1)
          in
          let positions = Rune.reshape position_shape positions in
          let precision = Rune.div cumsum_rel positions in
          let precision_on_rel = Rune.mul precision rel_float in
          let sum_precision = Rune.sum ~axes:[ -1 ] precision_on_rel in
          let total_rel = Rune.sum ~axes:[ -1 ] rel_float in
          let eps = scalar_tensor dtype 1e-7 in
          let raw = Rune.div sum_precision (Rune.add total_rel eps) in
          let mask = Rune.less total_rel eps in
          Rune.where mask zero raw
      in
      accumulate_metric_mean "map" state dtype ?weights map_values)
    ~compute:(compute_metric_mean "map")
    ~reset:(fun _ -> [])

let mrr ?k () =
  create_custom ~dtype:Rune.float32 ~name:"mrr"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let dtype, axis, depth, leading_shape, _, sorted_targets =
        prepare_ranking_inputs ?k predictions targets
      in
      let mrr_values =
        if depth = 0 then zeros_of_shape dtype leading_shape
        else
          let top_targets = slice_top_k axis depth sorted_targets in
          let zero = scalar_tensor dtype 0.0 in
          let rel_mask = Rune.greater top_targets zero in
          let rel_float = Rune.cast dtype rel_mask in
          let cumsum_rel = Rune.cumsum ~axis:(-1) rel_float in
          let one = scalar_tensor dtype 1.0 in
          let first_hits =
            Rune.logical_and rel_mask (Rune.equal cumsum_rel one)
          in
          let first_float = Rune.cast dtype first_hits in
          let positions = Rune.arange_f dtype 1.0 (float depth +. 1.0) 1.0 in
          let position_shape =
            Array.init (axis + 1) (fun i -> if i = axis then depth else 1)
          in
          let positions = Rune.reshape position_shape positions in
          let reciprocal = Rune.div one positions in
          let rr_candidates = Rune.mul reciprocal first_float in
          Rune.sum ~axes:[ -1 ] rr_candidates
      in
      accumulate_metric_mean "mrr" state dtype ?weights mrr_values)
    ~compute:(compute_metric_mean "mrr")
    ~reset:(fun _ -> [])

let bleu ?(max_n = 4) ?weights ?(smoothing = true) () =
  if max_n <= 0 then invalid_arg "Metrics.bleu: max_n must be positive";
  let weight_vector =
    match weights with
    | None ->
        let w = 1. /. float max_n in
        Array.init max_n (fun _ -> w)
    | Some w ->
        if Array.length w <> max_n then
          invalid_arg "Metrics.bleu: weights must have length max_n";
        let sum = Array.fold_left ( +. ) 0.0 w in
        if sum = 0.0 then
          invalid_arg "Metrics.bleu: weights must sum to a positive value";
        Array.map (fun v -> v /. sum) w
  in
  create_custom ~dtype:Rune.float32 ~name:"bleu"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights:sample_weights () ->
      let candidates = tensor_to_sequence_batch "bleu" predictions in
      let references = tensor_to_sequence_batch "bleu" targets in
      if Array.length candidates <> Array.length references then
        invalid_arg
          "Metrics.bleu: predictions and targets must share the same batch size";
      let batch = Array.length candidates in
      let scores =
        Array.init batch (fun idx ->
            let cand = candidates.(idx) in
            let reference = references.(idx) in
            let cand_len = Array.length cand in
            let ref_len = Array.length reference in
            if cand_len = 0 then 0.0
            else
              let brevity =
                if ref_len = 0 then 0.0
                else if cand_len > ref_len then 1.0
                else Float.exp (1.0 -. (float ref_len /. float cand_len))
              in
              let zero_precision = ref false in
              let log_precision =
                let acc = ref 0.0 in
                for n = 1 to max_n do
                  let weight = weight_vector.(n - 1) in
                  if weight <> 0.0 then
                    let overlap, total = bleu_precision cand reference n in
                    let num, den =
                      if total = 0 then
                        if smoothing then (1.0, 1.0) else (0.0, 1.0)
                      else
                        let adjust = if smoothing then 1.0 else 0.0 in
                        (float overlap +. adjust, float total +. adjust)
                    in
                    if den = 0.0 || num = 0.0 then zero_precision := true
                    else acc := !acc +. (weight *. Float.log (num /. den))
                done;
                !acc
              in
              if !zero_precision then 0.0
              else brevity *. Float.exp log_precision)
      in
      let dtype = Rune.dtype predictions in
      let values = Rune.create dtype [| batch |] scores in
      accumulate_metric_mean "bleu" state dtype ?weights:sample_weights values)
    ~compute:(compute_metric_mean "bleu")
    ~reset:(fun _ -> [])

let rouge ~variant ?use_stemmer () =
  (match use_stemmer with
  | Some true -> invalid_arg "Metrics.rouge: stemming is not supported yet"
  | _ -> ());
  let score_fn =
    match variant with
    | `Rouge1 ->
        fun cand reference ->
          let overlap, _ = bleu_precision cand reference 1 in
          let cand_total = Array.length cand in
          let ref_total = Array.length reference in
          if cand_total = 0 || ref_total = 0 then 0.0
          else
            let recall = float overlap /. float ref_total in
            let precision = float overlap /. float cand_total in
            if recall = 0.0 && precision = 0.0 then 0.0
            else 2.0 *. precision *. recall /. (precision +. recall)
    | `Rouge2 ->
        fun cand reference ->
          let overlap, _ = bleu_precision cand reference 2 in
          let cand_total = max 0 (Array.length cand - 1) in
          let ref_total = max 0 (Array.length reference - 1) in
          if cand_total = 0 || ref_total = 0 then 0.0
          else
            let recall = float overlap /. float ref_total in
            let precision = float overlap /. float cand_total in
            if recall = 0.0 && precision = 0.0 then 0.0
            else 2.0 *. precision *. recall /. (precision +. recall)
    | `RougeL ->
        fun cand reference ->
          let cand_len = Array.length cand in
          let ref_len = Array.length reference in
          if cand_len = 0 || ref_len = 0 then 0.0
          else
            let lcs = float (lcs_length cand reference) in
            let recall = lcs /. float ref_len in
            let precision = lcs /. float cand_len in
            if recall = 0.0 && precision = 0.0 then 0.0
            else 2.0 *. precision *. recall /. (precision +. recall)
  in
  create_custom ~dtype:Rune.float32 ~name:"rouge"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let candidates = tensor_to_sequence_batch "rouge" predictions in
      let references = tensor_to_sequence_batch "rouge" targets in
      if Array.length candidates <> Array.length references then
        invalid_arg
          "Metrics.rouge: predictions and targets must share the same batch \
           size";
      let batch = Array.length candidates in
      let values =
        Array.init batch (fun idx -> score_fn candidates.(idx) references.(idx))
      in
      let dtype = Rune.dtype predictions in
      let tensor = Rune.create dtype [| batch |] values in
      accumulate_metric_mean "rouge" state dtype ?weights tensor)
    ~compute:(compute_metric_mean "rouge")
    ~reset:(fun _ -> [])

let meteor ?(alpha = 0.9) ?(beta = 3.0) ?(gamma = 0.5) () =
  if alpha <= 0.0 || alpha >= 1.0 then
    invalid_arg "Metrics.meteor: alpha must be in (0, 1)";
  if beta <= 0.0 then invalid_arg "Metrics.meteor: beta must be positive";
  if gamma < 0.0 then invalid_arg "Metrics.meteor: gamma must be non-negative";
  create_custom ~dtype:Rune.float32 ~name:"meteor"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let candidates = tensor_to_sequence_batch "meteor" predictions in
      let references = tensor_to_sequence_batch "meteor" targets in
      if Array.length candidates <> Array.length references then
        invalid_arg
          "Metrics.meteor: predictions and targets must share the same batch \
           size";
      let batch = Array.length candidates in
      let scores =
        Array.init batch (fun idx ->
            let cand = candidates.(idx) in
            let reference = references.(idx) in
            let cand_len = Array.length cand in
            let ref_len = Array.length reference in
            if cand_len = 0 || ref_len = 0 then 0.0
            else
              let matches, chunks = meteor_alignment cand reference in
              if matches = 0 then 0.0
              else
                let precision = float matches /. float cand_len in
                let recall = float matches /. float ref_len in
                let f_mean =
                  precision *. recall
                  /. ((alpha *. precision) +. ((1.0 -. alpha) *. recall))
                in
                let penalty =
                  gamma *. ((float chunks /. float matches) ** beta)
                in
                (1.0 -. penalty) *. f_mean)
      in
      let dtype = Rune.dtype predictions in
      let tensor = Rune.create dtype [| batch |] scores in
      accumulate_metric_mean "meteor" state dtype ?weights tensor)
    ~compute:(compute_metric_mean "meteor")
    ~reset:(fun _ -> [])

let psnr ?(max_val = 1.0) () =
  let mse_metric = mse () in
  match mse_metric with
  | Metric m ->
      create_custom ~dtype:m.dtype ~name:"psnr" ~init:m.init_fn
        ~update:(fun state ~predictions ~targets ?weights () ->
          m.update_fn state ~predictions ~targets ?weights ())
        ~compute:(fun state ->
          let mse_val = m.compute_fn state in
          let dtype = Rune.dtype mse_val in
          let max_val_sq = max_val *. max_val in
          let max_val_sq_t = scalar_tensor dtype max_val_sq in
          let eps = scalar_tensor dtype 1e-12 in
          let mse_clamped = Rune.maximum eps mse_val in
          let ratio = Rune.div max_val_sq_t mse_clamped in
          let ten = scalar_tensor dtype 10.0 in
          (* log10(x) = log(x) / log(10) *)
          let log_ratio = Rune.log ratio in
          let log10_val = 2.302585093 in
          (* log(10) *)
          let log10_t = scalar_tensor dtype log10_val in
          Rune.mul ten (Rune.div log_ratio log10_t))
        ~reset:m.reset_fn

let ssim ?(window_size = 11) ?(k1 = 0.01) ?(k2 = 0.03) () =
  if window_size <= 0 then
    invalid_arg "Metrics.ssim: window_size must be positive";
  if k1 < 0.0 || k2 < 0.0 then
    invalid_arg "Metrics.ssim: k1 and k2 must be non-negative";
  create_custom ~dtype:Rune.float32 ~name:"ssim"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      let dtype = Rune.dtype predictions in
      let preds = Rune.cast dtype predictions in
      let refs = Rune.cast dtype targets in
      if Rune.numel preds <> Rune.numel refs then
        invalid_arg
          "Metrics.ssim: predictions and targets must share the same shape";
      let numel = Rune.numel preds in
      if numel = 0 then state
      else
        let flat_preds = Rune.reshape [| numel |] preds in
        let flat_refs = Rune.reshape [| numel |] refs in
        let mu_x = Rune.mean flat_preds in
        let mu_y = Rune.mean flat_refs in
        let diff_x = Rune.sub flat_preds mu_x in
        let diff_y = Rune.sub flat_refs mu_y in
        let var_x = Rune.mean (Rune.mul diff_x diff_x) in
        let var_y = Rune.mean (Rune.mul diff_y diff_y) in
        let cov_xy = Rune.mean (Rune.mul diff_x diff_y) in
        let max_pred = Rune.item [] (Rune.max flat_preds) in
        let max_ref = Rune.item [] (Rune.max flat_refs) in
        let min_pred = Rune.item [] (Rune.min flat_preds) in
        let min_ref = Rune.item [] (Rune.min flat_refs) in
        let max_val = Float.max max_pred max_ref in
        let min_val = Float.min min_pred min_ref in
        let dynamic_range = Float.max 1e-6 (max_val -. min_val) in
        let c1 = scalar_tensor dtype ((k1 *. dynamic_range) ** 2.) in
        let c2 = scalar_tensor dtype ((k2 *. dynamic_range) ** 2.) in
        let two = scalar_tensor dtype 2.0 in
        let numerator =
          let mu_term = Rune.mul two (Rune.mul mu_x mu_y) in
          let cov_term = Rune.mul two cov_xy in
          Rune.mul (Rune.add mu_term c1) (Rune.add cov_term c2)
        in
        let denominator =
          let mu_sq = Rune.add (Rune.mul mu_x mu_x) (Rune.mul mu_y mu_y) in
          let var_sum = Rune.add var_x var_y in
          Rune.mul (Rune.add mu_sq c1) (Rune.add var_sum c2)
        in
        let eps = scalar_tensor dtype 1e-7 in
        let ssim_value = Rune.div numerator (Rune.add denominator eps) in
        accumulate_metric_mean "ssim" state dtype ?weights ssim_value)
    ~compute:(compute_metric_mean "ssim")
    ~reset:(fun _ -> [])

let iou ?(threshold = 0.5) ?(per_class = false) ~num_classes () =
  if num_classes <= 0 then
    invalid_arg "Metrics.iou: num_classes must be positive";
  create_custom ~dtype:Rune.float32 ~name:"iou"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      (match weights with
      | Some _ -> invalid_arg "Metrics.iou: sample weights are not supported"
      | None -> ());
      let dtype = Rune.dtype predictions in
      let preds =
        flatten_predictions_for_classes ~num_classes ~threshold:(Some threshold)
          predictions
      in
      let refs = tensor_to_flat_ints targets in
      let stats =
        compute_confusion_matrix ~metric_name:"iou" ~num_classes preds refs
      in
      let intersections =
        Array.init num_classes (fun c -> stats.counts.(c).(c))
      in
      let unions =
        Array.init num_classes (fun c ->
            stats.row_sums.(c) + stats.col_sums.(c) - stats.counts.(c).(c))
      in
      update_pair_state "iou" state dtype intersections unions)
    ~compute:(fun state ->
      match state with
      | [ intersections; unions ] ->
          let dtype = Rune.dtype intersections in
          let eps = scalar_tensor dtype 1e-7 in
          let per_class_iou = Rune.div intersections (Rune.add unions eps) in
          if per_class then per_class_iou
          else
            let valid_mask = Rune.greater unions eps in
            let valid = Rune.cast dtype valid_mask in
            let weighted = Rune.mul per_class_iou valid in
            let total = Rune.sum weighted in
            let count = Rune.sum valid in
            Rune.div total (Rune.add count eps)
      | _ -> failwith "Invalid iou state")
    ~reset:(fun _ -> [])

let dice ?(threshold = 0.5) ?(per_class = false) ~num_classes () =
  if num_classes <= 0 then
    invalid_arg "Metrics.dice: num_classes must be positive";
  create_custom ~dtype:Rune.float32 ~name:"dice"
    ~init:(fun () -> [])
    ~update:(fun state ~predictions ~targets ?weights () ->
      (match weights with
      | Some _ -> invalid_arg "Metrics.dice: sample weights are not supported"
      | None -> ());
      let dtype = Rune.dtype predictions in
      let preds =
        flatten_predictions_for_classes ~num_classes ~threshold:(Some threshold)
          predictions
      in
      let refs = tensor_to_flat_ints targets in
      let stats =
        compute_confusion_matrix ~metric_name:"dice" ~num_classes preds refs
      in
      let intersections =
        Array.init num_classes (fun c -> stats.counts.(c).(c))
      in
      let denominators =
        Array.init num_classes (fun c ->
            let tp = stats.counts.(c).(c) in
            let fp = stats.col_sums.(c) - tp in
            let fn = stats.row_sums.(c) - tp in
            (2 * tp) + fp + fn)
      in
      update_pair_state "dice" state dtype intersections denominators)
    ~compute:(fun state ->
      match state with
      | [ intersections; denominators ] ->
          let dtype = Rune.dtype intersections in
          let eps = scalar_tensor dtype 1e-7 in
          let per_class_dice =
            let twice_tp = Rune.mul (scalar_tensor dtype 2.0) intersections in
            Rune.div twice_tp (Rune.add denominators eps)
          in
          if per_class then per_class_dice
          else
            let valid_mask = Rune.greater denominators eps in
            let valid = Rune.cast dtype valid_mask in
            let weighted = Rune.mul per_class_dice valid in
            let total = Rune.sum weighted in
            let count = Rune.sum valid in
            Rune.div total (Rune.add count eps)
      | _ -> failwith "Invalid dice state")
    ~reset:(fun _ -> [])

(** Metric Collections *)

(* Capture outer module functions to avoid shadowing *)
let compute_metric = compute

module Collection = struct
  type t = { mutable metrics : (string * metric) list }

  let empty () = { metrics = [] }
  let of_list metrics = { metrics }
  let create metrics = of_list metrics

  let add collection name metric =
    collection.metrics <- (name, metric) :: collection.metrics

  let remove collection name =
    collection.metrics <-
      List.filter (fun (n, _) -> not (String.equal n name)) collection.metrics

  let reset collection =
    List.iter (fun (_, metric) -> reset metric) collection.metrics

  let update collection ~predictions ~targets ?loss ?weights () =
    List.iter
      (fun (_, metric) -> update metric ~predictions ~targets ?loss ?weights ())
      collection.metrics

  let compute collection =
    let handle_compute m =
      try compute m
      with Failure msg ->
        let suffix = "metric has no data" in
        let len_msg = String.length msg in
        let len_suffix = String.length suffix in
        if
          len_msg >= len_suffix
          && String.sub msg (len_msg - len_suffix) len_suffix = suffix
        then 0.0
        else raise (Failure msg)
    in
    List.map (fun (name, m) -> (name, handle_compute m)) collection.metrics

  let compute_tensors collection =
    List.map
      (fun (name, metric) -> (name, compute_tensor metric))
      collection.metrics

  let compute_dict (collection : t) =
    let tbl : (string, float) Hashtbl.t =
      Hashtbl.create (List.length collection.metrics)
    in
    List.iter
      (fun (name, metric) -> Hashtbl.add tbl name (compute_metric metric))
      collection.metrics;
    tbl
end

(** Utilities *)

let is_better _metric ~higher_better ~old_val ~new_val =
  if higher_better then new_val > old_val else new_val < old_val

let format metric value = Printf.sprintf "%s: %.4f" (name metric) value
