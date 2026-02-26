(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* Tracker *)

type entry = { mutable sum : float; mutable n : int }
type tracker = (string, entry) Hashtbl.t

let tracker () : tracker = Hashtbl.create 16

let observe (t : tracker) name value =
  match Hashtbl.find_opt t name with
  | Some e ->
      e.sum <- e.sum +. value;
      e.n <- e.n + 1
  | None -> Hashtbl.replace t name { sum = value; n = 1 }

let find_entry t name = Hashtbl.find t name

let mean t name =
  let e = find_entry t name in
  e.sum /. float_of_int e.n

let count t name =
  let e = find_entry t name in
  e.n

let reset t = Hashtbl.reset t

let to_list t =
  let pairs =
    Hashtbl.fold (fun k e acc -> (k, e.sum /. float_of_int e.n) :: acc) t []
  in
  List.sort (fun (a, _) (b, _) -> String.compare a b) pairs

let summary t =
  let pairs = to_list t in
  String.concat "  " (List.map (fun (k, v) -> strf "%s: %.4f" k v) pairs)

(* Dataset evaluation *)

let eval f data =
  let sum = ref 0.0 in
  let n = ref 0 in
  Data.iter
    (fun x ->
      sum := !sum +. f x;
      incr n)
    data;
  if !n = 0 then invalid_arg "Metric.eval: empty dataset";
  !sum /. float_of_int !n

let eval_many f data =
  let tbl = Hashtbl.create 8 in
  let n = ref 0 in
  Data.iter
    (fun x ->
      let pairs = f x in
      List.iter
        (fun (k, v) ->
          match Hashtbl.find_opt tbl k with
          | Some e ->
              e.sum <- e.sum +. v;
              e.n <- e.n + 1
          | None -> Hashtbl.replace tbl k { sum = v; n = 1 })
        pairs;
      incr n)
    data;
  if !n = 0 then invalid_arg "Metric.eval_many: empty dataset";
  let pairs =
    Hashtbl.fold (fun k e acc -> (k, e.sum /. float_of_int e.n) :: acc) tbl []
  in
  List.sort (fun (a, _) (b, _) -> String.compare a b) pairs

type average = Macro | Micro | Weighted

(* Metric functions *)

let accuracy (type a b c) (predictions : (float, a) Rune.t)
    (targets : (b, c) Rune.t) =
  let pred_shape = Rune.shape predictions in
  let rank = Array.length pred_shape in
  let predicted =
    if rank >= 2 then
      (* Multi-class: argmax along last axis *)
      Rune.argmax ~axis:(-1) predictions
    else
      (* Binary: threshold at 0.5 *)
      let half = Rune.scalar (Rune.dtype predictions) 0.5 in
      Rune.cast Rune.int32 (Rune.greater predictions half)
  in
  let targets_i32 = Rune.cast Rune.int32 targets in
  let correct = Rune.equal predicted targets_i32 in
  let correct_f = Rune.cast Rune.float32 correct in
  Rune.item [] (Rune.mean correct_f)

let binary_accuracy ?(threshold = 0.5) predictions targets =
  let dtype = Rune.dtype predictions in
  let thresh = Rune.scalar dtype threshold in
  let predicted = Rune.cast Rune.float32 (Rune.greater predictions thresh) in
  let targets_f = Rune.cast Rune.float32 targets in
  let correct = Rune.equal predicted targets_f in
  let correct_f = Rune.cast Rune.float32 correct in
  Rune.item [] (Rune.mean correct_f)

(* Classification metrics *)

let confusion_counts (type a b c) (predictions : (float, a) Rune.t)
    (targets : (b, c) Rune.t) =
  let pred_shape = Rune.shape predictions in
  let num_classes = pred_shape.(Array.length pred_shape - 1) in
  let predicted = Rune.argmax ~axis:(-1) predictions in
  let targets_i32 = Rune.cast Rune.int32 targets in
  let pred_oh = Rune.cast Rune.float32 (Rune.one_hot ~num_classes predicted) in
  let tgt_oh = Rune.cast Rune.float32 (Rune.one_hot ~num_classes targets_i32) in
  let tp = Rune.sum (Rune.mul pred_oh tgt_oh) ~axes:[ 0 ] in
  let pred_sum = Rune.sum pred_oh ~axes:[ 0 ] in
  let tgt_sum = Rune.sum tgt_oh ~axes:[ 0 ] in
  let fp = Rune.sub pred_sum tp in
  let fn = Rune.sub tgt_sum tp in
  (tp, fp, fn, num_classes)

let safe_div a b = if b = 0.0 then 0.0 else a /. b

let precision avg predictions targets =
  let tp, fp, fn, num_classes = confusion_counts predictions targets in
  let tp = Rune.to_array tp in
  let fp = Rune.to_array fp in
  match avg with
  | Micro ->
      let tp_sum = Array.fold_left ( +. ) 0.0 tp in
      let fp_sum = Array.fold_left ( +. ) 0.0 fp in
      safe_div tp_sum (tp_sum +. fp_sum)
  | Macro ->
      let sum = ref 0.0 in
      for c = 0 to num_classes - 1 do
        sum := !sum +. safe_div tp.(c) (tp.(c) +. fp.(c))
      done;
      !sum /. float_of_int num_classes
  | Weighted ->
      let fn = Rune.to_array fn in
      let w_sum = ref 0.0 in
      let total = ref 0.0 in
      for c = 0 to num_classes - 1 do
        let support = tp.(c) +. fn.(c) in
        w_sum := !w_sum +. (support *. safe_div tp.(c) (tp.(c) +. fp.(c)));
        total := !total +. support
      done;
      safe_div !w_sum !total

let recall avg predictions targets =
  let tp, _fp, fn, num_classes = confusion_counts predictions targets in
  let tp = Rune.to_array tp in
  let fn = Rune.to_array fn in
  match avg with
  | Micro ->
      let tp_sum = Array.fold_left ( +. ) 0.0 tp in
      let fn_sum = Array.fold_left ( +. ) 0.0 fn in
      safe_div tp_sum (tp_sum +. fn_sum)
  | Macro ->
      let sum = ref 0.0 in
      for c = 0 to num_classes - 1 do
        sum := !sum +. safe_div tp.(c) (tp.(c) +. fn.(c))
      done;
      !sum /. float_of_int num_classes
  | Weighted ->
      let w_sum = ref 0.0 in
      let total = ref 0.0 in
      for c = 0 to num_classes - 1 do
        let support = tp.(c) +. fn.(c) in
        w_sum := !w_sum +. (support *. safe_div tp.(c) (tp.(c) +. fn.(c)));
        total := !total +. support
      done;
      safe_div !w_sum !total

let f1 avg predictions targets =
  let tp, fp, fn, num_classes = confusion_counts predictions targets in
  let tp = Rune.to_array tp in
  let fp = Rune.to_array fp in
  let fn = Rune.to_array fn in
  match avg with
  | Micro ->
      let tp_sum = Array.fold_left ( +. ) 0.0 tp in
      let fp_sum = Array.fold_left ( +. ) 0.0 fp in
      let fn_sum = Array.fold_left ( +. ) 0.0 fn in
      safe_div (2.0 *. tp_sum) ((2.0 *. tp_sum) +. fp_sum +. fn_sum)
  | Macro ->
      let sum = ref 0.0 in
      for c = 0 to num_classes - 1 do
        sum :=
          !sum +. safe_div (2.0 *. tp.(c)) ((2.0 *. tp.(c)) +. fp.(c) +. fn.(c))
      done;
      !sum /. float_of_int num_classes
  | Weighted ->
      let w_sum = ref 0.0 in
      let total = ref 0.0 in
      for c = 0 to num_classes - 1 do
        let support = tp.(c) +. fn.(c) in
        w_sum :=
          !w_sum
          +. support
             *. safe_div (2.0 *. tp.(c)) ((2.0 *. tp.(c)) +. fp.(c) +. fn.(c));
        total := !total +. support
      done;
      safe_div !w_sum !total
