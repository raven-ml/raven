(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Metric = Kaun.Metric

(* The unit roundoff of float64. *)
let u = Float.ldexp 1. (-53)

(* A batch: predictions of shape [lead; classes], small integers so that rows
   tie, and one label per row. *)
type batch = {
  lead : int array;
  classes : int;
  scores : float array;
  labels : int array;
}

let rows b = Array.fold_left ( * ) 1 b.lead
let row b i = Array.sub b.scores (i * b.classes) b.classes

let predictions b =
  Nx.create Nx.float64 (Array.append b.lead [| b.classes |]) b.scores

let labels b = Nx.create Nx.int64 b.lead (Array.map Int64.of_int b.labels)

let batch_gen =
  let open Gen in
  let+ (lead, classes), (scores, labels) =
    pair
      (pair (array ~size:(int_range 0 2) (int_range 1 3)) (int_range 1 4))
      (pair
         (array ~size:(constant 36) (int_range 0 3))
         (array ~size:(constant 9) (int_range 0 3)))
  in
  let n = Array.fold_left ( * ) 1 lead in
  {
    lead;
    classes;
    scores = Array.map float_of_int (Array.sub scores 0 (n * classes));
    labels = Array.map (fun l -> l mod classes) (Array.sub labels 0 n);
  }

(* The predicted class of a row: the first of its maxima. *)
let argmax row =
  let best = ref 0 in
  Array.iteri (fun j v -> if v > row.(!best) then best := j) row;
  !best

let predicted b = Array.init (rows b) (fun i -> argmax (row b i))
let count p a = Array.fold_left (fun n x -> if p x then n + 1 else n) 0 a
let fraction n b = float_of_int n /. float_of_int (rows b)

let correct b =
  let p = predicted b in
  count Fun.id (Array.mapi (fun i l -> p.(i) = l) b.labels)

(* Fewer than [k] classes score strictly higher than the label's. *)
let in_top_k k b =
  count Fun.id
    (Array.mapi
       (fun i l ->
         let r = row b i in
         count (fun v -> v > r.(l)) r < k)
       b.labels)

let confusion b =
  let m = Array.make (b.classes * b.classes) 0 in
  let p = predicted b in
  Array.iteri
    (fun i l -> m.((l * b.classes) + p.(i)) <- m.((l * b.classes) + p.(i)) + 1)
    b.labels;
  m

let tied b =
  List.exists
    (fun i ->
      let r = row b i in
      count (fun v -> v = r.(argmax r)) r > 1)
    (List.init (rows b) Fun.id)

(* [tp], true instances (row sums) and predicted instances (column sums) per
   class. *)
let class_counts b =
  let m = confusion b and c = b.classes in
  let tp = Array.init c (fun k -> m.((k * c) + k)) in
  let actual =
    Array.init c (fun k -> Array.fold_left ( + ) 0 (Array.sub m (k * c) c))
  in
  let predicted =
    Array.init c (fun k ->
        Array.fold_left ( + ) 0 (Array.init c (fun l -> m.((l * c) + k))))
  in
  (tp, actual, predicted)

let ratio n d = if d = 0 then 0. else float_of_int n /. float_of_int d

let average a nums dens =
  match a with
  | `Micro ->
      ratio (Array.fold_left ( + ) 0 nums) (Array.fold_left ( + ) 0 dens)
  | `Macro ->
      Array.fold_left ( +. ) 0. (Array.map2 ratio nums dens)
      /. float_of_int (Array.length nums)

(* A macro mean of at most four ratios, each rounded once, summed and divided:
   within [8u] relative of another order of the same sum. *)
let score = float_rel ~rel:(8. *. u) ~abs:0.

let averages =
  Gen.(
    option
      (of_list
         ~pp:(fun ppf a ->
           Format.pp_print_string ppf
             (match a with `Macro -> "`Macro" | `Micro -> "`Micro"))
         [ `Macro; `Micro ]))

let covers b =
  cover "a tie for a row's maximum" (tied b);
  cover "0-d labels" (b.lead = [||]);
  cover "predictions of rank 3" (Array.length b.lead = 2)

(* Accuracy *)

let accuracy_law b =
  covers b;
  equal float_exact
    (fraction (correct b) b)
    (Metric.accuracy (predictions b) (labels b))

let top_k_law (k, b) =
  let k = 1 + ((k - 1) mod b.classes) in
  covers b;
  cover "k = classes" (k = b.classes && b.classes > 1);
  cover "top 1 above accuracy on a tie" (k = 1 && in_top_k 1 b > correct b);
  equal float_exact
    (fraction (in_top_k k b) b)
    (Metric.top_k_accuracy ~k (predictions b) (labels b))

(* Confusion-matrix metrics *)

let confusion_law b =
  covers b;
  let m = Metric.confusion_matrix (predictions b) (labels b) in
  equal (array int) [| b.classes; b.classes |] (Nx.shape m);
  equal (array int) (confusion b) (Array.map Int32.to_int (Nx.to_array m))

let absent b =
  let _, actual, predicted = class_counts b in
  Array.exists2 (fun a p -> a = 0 && p = 0) actual predicted

let precision_law (a, b) =
  covers b;
  cover "a class absent from labels and predictions" (absent b);
  let tp, _, predicted = class_counts b in
  equal score
    (average (Option.value a ~default:`Macro) tp predicted)
    (Metric.precision ?average:a (predictions b) (labels b))

let recall_law (a, b) =
  covers b;
  cover "a class absent from labels and predictions" (absent b);
  let tp, actual, _ = class_counts b in
  equal score
    (average (Option.value a ~default:`Macro) tp actual)
    (Metric.recall ?average:a (predictions b) (labels b))

let f1_law (a, b) =
  covers b;
  cover "a class absent from labels and predictions" (absent b);
  let tp, actual, predicted = class_counts b in
  equal score
    (average
       (Option.value a ~default:`Macro)
       (Array.map (( * ) 2) tp)
       (Array.map2 ( + ) actual predicted))
    (Metric.f1 ?average:a (predictions b) (labels b))

let micro_is_accuracy b =
  let p = predictions b and l = labels b in
  let accuracy = Metric.accuracy p l in
  equal ~msg:"precision" float_exact accuracy
    (Metric.precision ~average:`Micro p l);
  equal ~msg:"recall" float_exact accuracy (Metric.recall ~average:`Micro p l);
  equal ~msg:"f1" float_exact accuracy (Metric.f1 ~average:`Micro p l)

(* Batches *)

(* A dataset of [n] rows cut after row [cut]: the two batches. *)
let split_gen =
  let open Gen in
  let+ (classes, cut), (scores, labels) =
    pair
      (pair (int_range 1 4) (int_range 1 8))
      (pair
         (array ~size:(constant 36) (int_range 0 3))
         (array ~size:(int_range 2 9) (int_range 0 3)))
  in
  let n = Array.length labels in
  let cut = 1 + ((cut - 1) mod (n - 1)) in
  let batch lo hi =
    {
      lead = [| hi - lo |];
      classes;
      scores =
        Array.map float_of_int
          (Array.sub scores (lo * classes) ((hi - lo) * classes));
      labels =
        Array.map (fun l -> l mod classes) (Array.sub labels lo (hi - lo));
    }
  in
  (batch 0 n, batch 0 cut, batch cut n)

(* A batch's count of correct examples, from its value: [n] times a fraction of
   [n] rounds back to the count. *)
let counted b value =
  Float.to_int (Float.round (value *. float_of_int (rows b)))

let decomposes (whole, first, second) =
  let check msg f =
    let c b = counted b (f (predictions b) (labels b)) in
    equal ~msg int (c whole) (c first + c second)
  in
  check "accuracy" Metric.accuracy;
  check "top_k_accuracy" (Metric.top_k_accuracy ~k:1);
  check "micro precision" (Metric.precision ~average:`Micro);
  check "micro recall" (Metric.recall ~average:`Micro);
  check "micro f1" (Metric.f1 ~average:`Micro)

let confusion_sums (whole, first, second) =
  let m b = Nx.to_array (Metric.confusion_matrix (predictions b) (labels b)) in
  equal ~msg:"confusion_matrix" (array int32) (m whole)
    (Array.map2 Int32.add (m first) (m second))

(* Ranking *)

(* Examples [(score, positive)], scores from four integers so that pairs tie;
   the first two are a positive and a negative, so both classes are present. *)
let auc_gen =
  Gen.(
    map
      (fun ((p, n), rest) -> Array.append [| (p, true); (n, false) |] rest)
      (pair
         (pair (int_range 0 3) (int_range 0 3))
         (array ~size:(int_range 0 6) (pair (int_range 0 3) bool))))

let auc_examples =
  [
    (* Perfect, reversed and all tied. *)
    [| (3, true); (0, false) |];
    [| (0, true); (3, false) |];
    [| (1, true); (1, false); (1, true); (1, false) |];
  ]

let auc_law examples =
  let scores = Array.map (fun (s, _) -> float_of_int s) examples in
  let positive = Array.map snd examples in
  let wins = ref 0. and pairs = ref 0 and ties = ref 0 in
  Array.iteri
    (fun i pi ->
      Array.iteri
        (fun j pj ->
          if pi && not pj then begin
            incr pairs;
            if scores.(i) > scores.(j) then wins := !wins +. 1.
            else if scores.(i) = scores.(j) then (
              incr ties;
              wins := !wins +. 0.5)
          end)
        positive)
    positive;
  cover "a tied positive and negative" (!ties > 0);
  let n = Array.length examples in
  equal float_exact
    (!wins /. float_of_int !pairs)
    (Metric.auc_roc
       (Nx.create Nx.float64 [| n |] scores)
       (Nx.create Nx.int64 [| n |]
          (Array.map (fun p -> if p then 1L else 0L) positive)))

(* Placement *)

let software =
  fixture (fun () -> Result.get_ok (Rig.software "kaun-test-metric"))

(* [f ()] and the bytes copied into [device]'s memory while it ran. *)
let received device f =
  let y, events = Rig.Profile.take f in
  let count n = function
    | Rig.Profile.Copy { dst; bytes; _ } when Rig.equal dst device -> n + bytes
    | _ -> n
  in
  (y, List.fold_left count 0 events)

(* A metric at any placement. *)
type metric = {
  name : string;
  f : 'd. (float, Nx.float64_elt, 'd) Nx.t -> 'd Nx.int64_t -> float;
}

let metrics =
  [
    { name = "accuracy"; f = (fun p l -> Metric.accuracy p l) };
    { name = "top_k_accuracy"; f = (fun p l -> Metric.top_k_accuracy ~k:2 p l) };
    { name = "precision"; f = (fun p l -> Metric.precision p l) };
    { name = "recall"; f = (fun p l -> Metric.recall p l) };
    { name = "f1"; f = (fun p l -> Metric.f1 p l) };
    {
      name = "confusion_matrix";
      f =
        (fun p l ->
          Int32.to_float (Nx.item [ 2; 2 ] (Metric.confusion_matrix p l)));
    };
  ]

let computes_on_host () =
  let device = software () in
  let module S = (val Nx.devices [ device ]) in
  let b =
    {
      lead = [| 4 |];
      classes = 3;
      scores = [| 1.; 0.; 0.; 0.; 0.; 1.; 0.; 1.; 0.; 0.; 0.; 1. |];
      labels = [| 0; 2; 2; 2 |];
    }
  in
  List.iter
    (fun m ->
      let p = Nx.place S.on (predictions b) and l = Nx.place S.on (labels b) in
      let value, uploaded = received device (fun () -> m.f p l) in
      equal ~msg:m.name float_exact (m.f (predictions b) (labels b)) value;
      equal ~msg:(m.name ^ ": bytes placed on the device") int 0 uploaded)
    metrics;
  let scores = Nx.create Nx.float64 [| 4 |] [| 0.1; 0.4; 0.35; 0.8 |] in
  let binary = Nx.create Nx.int64 [| 4 |] [| 0L; 0L; 1L; 1L |] in
  let s = Nx.place S.on scores and l = Nx.place S.on binary in
  let value, uploaded = received device (fun () -> Metric.auc_roc s l) in
  equal ~msg:"auc_roc" float_exact (Metric.auc_roc scores binary) value;
  equal ~msg:"auc_roc: bytes placed on the device" int 0 uploaded

(* Refusals *)

let raises_prefixed fn f =
  let prefix = "Metric." ^ fn ^ ":" in
  raises_match
    (function Invalid_argument m -> String.starts_with ~prefix m | _ -> false)
    f

let classifiers =
  [
    ("accuracy", fun p l -> ignore (Metric.accuracy p l));
    ("top_k_accuracy", fun p l -> ignore (Metric.top_k_accuracy ~k:1 p l));
    ("confusion_matrix", fun p l -> ignore (Metric.confusion_matrix p l));
    ("precision", fun p l -> ignore (Metric.precision p l));
    ("recall", fun p l -> ignore (Metric.recall p l));
    ("f1", fun p l -> ignore (Metric.f1 p l));
  ]

(* Predictions' shape and the labels. *)
let bad_inputs =
  [
    ("labels of another shape", [| 4; 3 |], [| 0L; 0L |]);
    ("a label out of range", [| 1; 3 |], [| 3L |]);
    ("a negative label", [| 1; 3 |], [| -1L |]);
    ("no examples", [| 0; 3 |], [||]);
  ]

let refusal_rows =
  List.concat_map
    (fun (fn, f) ->
      List.map
        (fun (why, shape, ls) ->
          let labels = Nx.create Nx.int64 [| Array.length ls |] in
          ( fn ^ ", " ^ why,
            fn,
            fun () -> f (Nx.zeros Nx.float64 shape) (labels ls) ))
        bad_inputs)
    classifiers

let auc_refusals =
  let s xs = Nx.create Nx.float64 [| Array.length xs |] xs in
  let l ls = Nx.create Nx.int64 [| Array.length ls |] ls in
  [
    ( "shapes differ",
      fun () -> Metric.auc_roc (s [| 0.1; 0.5; 0.9 |]) (l [| 0L; 1L |]) );
    ( "a label of 2",
      fun () -> Metric.auc_roc (s [| 0.1; 0.9 |]) (l [| 0L; 2L |]) );
    ( "only positives",
      fun () -> Metric.auc_roc (s [| 0.1; 0.9 |]) (l [| 1L; 1L |]) );
    ("no examples", fun () -> Metric.auc_roc (s [||]) (l [||]));
  ]

let () =
  let one_hot =
    { lead = [| 1 |]; classes = 3; scores = [| 0.; 0.; 1. |]; labels = [| 2 |] }
  in
  exit
    (run "kaun metric"
       [
         group "metrics"
           [
             test "compute a placed value's metric on the host, placing nothing"
               computes_on_host;
             prop
               "batch means of accuracy, top_k_accuracy and micro scores \
                weighted by size equal the dataset's"
               split_gen decomposes;
             prop "confusion matrices sum over batches" split_gen confusion_sums;
             cases
               "raise with the function's prefix on bad shapes, labels or an \
                empty batch"
               ~id:(fun (id, _, _) -> id)
               refusal_rows
               (fun (_, fn, f) -> raises_prefixed fn f);
             (* A label whose top bit a conversion to int would drop. *)
             test "name a label that int64 holds and int does not" (fun () ->
                 raises
                   (Invalid_argument
                      "Metric.accuracy: label -9223372036854775807 is out of \
                       range [0;2]") (fun () ->
                     Metric.accuracy (predictions one_hot)
                       (Nx.create Nx.int64 [| 1 |]
                          [| Int64.add Int64.min_int 1L |])));
           ];
         group "accuracy"
           [
             prop "is the fraction whose first maximum is the label" batch_gen
               accuracy_law;
           ];
         group "top_k_accuracy"
           [
             prop
               "is the fraction with fewer than k classes scoring above the \
                label"
               ~examples:
                 [
                   (* The label ties the first maximum: top 1 counts it,
                      accuracy does not. *)
                   ( 1,
                     {
                       lead = [| 1 |];
                       classes = 3;
                       scores = [| 2.; 2.; 1. |];
                       labels = [| 1 |];
                     } );
                 ]
               Gen.(pair (int_range 1 4) batch_gen)
               top_k_law;
             cases "raises on a k outside [1, classes]" ~id:string_of_int
               [ 0; 4; -1 ] (fun k ->
                 raises_prefixed "top_k_accuracy" (fun () ->
                     Metric.top_k_accuracy ~k (predictions one_hot)
                       (labels one_hot)));
           ];
         group "confusion_matrix"
           [
             prop "counts each label's examples by predicted class" batch_gen
               confusion_law;
           ];
         group "precision"
           [
             prop "combines tp / (tp + fp) per class"
               Gen.(pair averages batch_gen)
               precision_law;
           ];
         group "recall"
           [
             prop "combines tp / (tp + fn) per class"
               Gen.(pair averages batch_gen)
               recall_law;
           ];
         group "f1"
           [
             prop "combines 2 tp / (2 tp + fp + fn) per class"
               Gen.(pair averages batch_gen)
               f1_law;
           ];
         group "average"
           [
             prop "`Micro precision, recall and f1 equal accuracy" batch_gen
               micro_is_accuracy;
           ];
         group "auc_roc"
           [
             prop
               "is the share of positive-negative pairs the positive wins, \
                ties half"
               ~examples:auc_examples auc_gen auc_law;
             cases
               "raises on mismatched shapes, a label other than 0 and 1, or \
                one class"
               ~id:fst auc_refusals (fun (_, f) -> raises_prefixed "auc_roc" f);
           ];
       ])
