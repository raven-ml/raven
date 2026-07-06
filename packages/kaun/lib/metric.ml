(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type average = [ `Macro | `Micro ]

let invalid_argf fn fmt =
  Printf.ksprintf (fun msg -> invalid_arg ("Metric." ^ fn ^ ": " ^ msg)) fmt

let shape_str s =
  "[" ^ String.concat "; " (Array.to_list (Array.map string_of_int s)) ^ "]"

(* Validates the [[...; classes]] predictions / class-index labels convention
   and returns the class count with the labels as a flat row-major array. *)
let check_predictions ~fn predictions labels =
  let shape = Nx.shape predictions in
  let rank = Array.length shape in
  if rank < 1 then invalid_argf fn "predictions must have rank >= 1";
  let classes = shape.(rank - 1) in
  if classes <= 0 then
    invalid_argf fn "predictions class dimension must be positive (got %d)"
      classes;
  let batch_shape = Array.sub shape 0 (rank - 1) in
  let labels_shape = Nx.shape labels in
  if labels_shape <> batch_shape then
    invalid_argf fn "labels shape %s does not match predictions batch shape %s"
      (shape_str labels_shape) (shape_str batch_shape);
  let flat = Nx.to_array labels in
  if Array.length flat = 0 then invalid_argf fn "there are no examples";
  Array.iter
    (fun l ->
      let l = Int32.to_int l in
      if l < 0 || l >= classes then
        invalid_argf fn "label %d is out of range [0;%d]" l (classes - 1))
    flat;
  (classes, flat)

let fraction correct = Nx.item [] (Nx.mean (Nx.cast Nx.float64 correct))

(* Accuracy *)

let accuracy predictions labels =
  let _ = check_predictions ~fn:"accuracy" predictions labels in
  fraction (Nx.equal (Nx.argmax ~axis:(-1) predictions) labels)

let top_k_accuracy ~k predictions labels =
  let fn = "top_k_accuracy" in
  let classes, _ = check_predictions ~fn predictions labels in
  if k < 1 || k > classes then
    invalid_argf fn "k must be in [1;%d] (got %d)" classes k;
  (* The label is in the top k iff fewer than [k] classes score strictly higher,
     which needs no sort and resolves ties in the label's favor. *)
  let label_score =
    Nx.take_along_axis ~axis:(-1) (Nx.expand_dims [ -1 ] labels) predictions
  in
  let higher =
    Nx.sum ~axes:[ -1 ] (Nx.cast Nx.int32 (Nx.greater predictions label_score))
  in
  fraction (Nx.less_s higher (Int32.of_int k))

(* Confusion-matrix metrics *)

(* Flat row-major [classes * classes] counts; row = label, column = predicted
   class. *)
let confusion_counts ~fn predictions labels =
  let classes, labels = check_predictions ~fn predictions labels in
  let predicted = Nx.to_array (Nx.argmax ~axis:(-1) predictions) in
  let counts = Array.make (classes * classes) 0 in
  Array.iteri
    (fun i l ->
      let cell = (Int32.to_int l * classes) + Int32.to_int predicted.(i) in
      counts.(cell) <- counts.(cell) + 1)
    labels;
  (classes, counts)

let confusion_matrix predictions labels =
  let classes, counts =
    confusion_counts ~fn:"confusion_matrix" predictions labels
  in
  Nx.create Nx.int32 [| classes; classes |] (Array.map Int32.of_int counts)

(* Per-class true positives, true instances (row sums) and predicted instances
   (column sums). *)
let class_counts ~fn predictions labels =
  let classes, counts = confusion_counts ~fn predictions labels in
  let tp = Array.make classes 0 in
  let actual = Array.make classes 0 in
  let predicted = Array.make classes 0 in
  for l = 0 to classes - 1 do
    for p = 0 to classes - 1 do
      let c = counts.((l * classes) + p) in
      actual.(l) <- actual.(l) + c;
      predicted.(p) <- predicted.(p) + c;
      if l = p then tp.(l) <- tp.(l) + c
    done
  done;
  (tp, actual, predicted)

let ratio num den =
  if den = 0 then 0.0 else float_of_int num /. float_of_int den

(* Combines per-class scores [nums.(c) / dens.(c)], zero on an empty
   denominator. *)
let averaged average nums dens =
  match average with
  | `Micro ->
      ratio (Array.fold_left ( + ) 0 nums) (Array.fold_left ( + ) 0 dens)
  | `Macro ->
      let classes = Array.length nums in
      let sum = ref 0.0 in
      for c = 0 to classes - 1 do
        sum := !sum +. ratio nums.(c) dens.(c)
      done;
      !sum /. float_of_int classes

let precision ?(average = `Macro) predictions labels =
  let tp, _, predicted = class_counts ~fn:"precision" predictions labels in
  averaged average tp predicted

let recall ?(average = `Macro) predictions labels =
  let tp, actual, _ = class_counts ~fn:"recall" predictions labels in
  averaged average tp actual

let f1 ?(average = `Macro) predictions labels =
  let tp, actual, predicted = class_counts ~fn:"f1" predictions labels in
  averaged average
    (Array.map (fun n -> 2 * n) tp)
    (Array.map2 ( + ) actual predicted)

(* Ranking *)

let auc_roc scores labels =
  let fn = "auc_roc" in
  let scores_shape = Nx.shape scores and labels_shape = Nx.shape labels in
  if labels_shape <> scores_shape then
    invalid_argf fn "labels shape %s does not match scores shape %s"
      (shape_str labels_shape) (shape_str scores_shape);
  let s = Nx.to_array scores in
  let positive =
    Array.map
      (fun l ->
        if l <> 0l && l <> 1l then
          invalid_argf fn "label %ld is neither 0 nor 1" l;
        l = 1l)
      (Nx.to_array labels)
  in
  let n = Array.length s in
  let n_pos =
    Array.fold_left (fun acc p -> if p then acc + 1 else acc) 0 positive
  in
  let n_neg = n - n_pos in
  if n_pos = 0 || n_neg = 0 then
    invalid_argf fn "labels must contain both classes";
  let order = Array.init n Fun.id in
  Array.sort (fun i j -> Float.compare s.(i) s.(j)) order;
  (* Mann-Whitney form: AUC = (R - p (p + 1) / 2) / (p n), where [R] sums the
     ascending 1-based ranks of the positives, tied scores taking their midrank.
     Equals trapezoidal integration of the ROC curve. *)
  let rank_sum = ref 0.0 in
  let i = ref 0 in
  while !i < n do
    let j = ref (!i + 1) in
    while !j < n && s.(order.(!j)) = s.(order.(!i)) do
      incr j
    done;
    (* Positions [!i, !j) hold ranks [!i + 1, !j]. *)
    let midrank = float_of_int (!i + !j + 1) /. 2.0 in
    for k = !i to !j - 1 do
      if positive.(order.(k)) then rank_sum := !rank_sum +. midrank
    done;
    i := !j
  done;
  let n_pos = float_of_int n_pos and n_neg = float_of_int n_neg in
  (!rank_sum -. (n_pos *. (n_pos +. 1.) /. 2.)) /. (n_pos *. n_neg)
