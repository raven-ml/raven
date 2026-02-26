(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Training metrics.

    {!Metric} provides running scalar tracking and dataset evaluation.

    A {!type:tracker} accumulates named running means during training. For
    dataset evaluation, {!eval} and {!eval_many} fold user-supplied functions
    over a {!Data.t} pipeline and return averaged results.

    Metric functions such as {!accuracy} are plain tensor-to-scalar functions
    that compose freely with {!eval}. *)

(** {1:tracker Running Tracker} *)

type tracker
(** A mutable set of named running-mean accumulators. *)

val tracker : unit -> tracker
(** [tracker ()] is a fresh tracker with no observations. *)

val observe : tracker -> string -> float -> unit
(** [observe t name value] records [value] under [name]. *)

val mean : tracker -> string -> float
(** [mean t name] is the running mean of observations under [name].

    Raises [Not_found] if [name] was never observed. *)

val count : tracker -> string -> int
(** [count t name] is the number of observations under [name].

    Raises [Not_found] if [name] was never observed. *)

val reset : tracker -> unit
(** [reset t] clears all observations. *)

val to_list : tracker -> (string * float) list
(** [to_list t] is the current means as [(name, mean)] pairs, sorted by name. *)

val summary : tracker -> string
(** [summary t] is a human-readable one-liner of all current means, e.g.
    ["accuracy: 0.9150  loss: 0.4231"]. *)

(** {1:eval Dataset Evaluation} *)

val eval : ('a -> float) -> 'a Data.t -> float
(** [eval f data] is the mean of [f batch] over all elements of [data].

    Raises [Invalid_argument] if [data] yields no elements. *)

val eval_many :
  ('a -> (string * float) list) -> 'a Data.t -> (string * float) list
(** [eval_many f data] is the per-name mean of [f batch] over all elements of
    [data]. Returns [(name, mean)] pairs sorted by name.

    Raises [Invalid_argument] if [data] yields no elements. *)

(** {1:average Averaging} *)

type average =
  | Macro
  | Micro
  | Weighted
      (** The type for multi-class averaging modes.
          - [Macro] is the unweighted mean of per-class scores.
          - [Micro] aggregates TP, FP, FN globally before computing.
          - [Weighted] is the mean of per-class scores weighted by class support
            (number of true instances). *)

(** {1:compute Common Metric Functions} *)

val accuracy : (float, 'a) Rune.t -> ('b, 'c) Rune.t -> float
(** [accuracy predictions targets] is the fraction of correct predictions.

    Multi-class: [predictions] has shape [[batch; num_classes]] (logits or
    probabilities), [targets] has shape [[batch]] (integer class indices).
    Predicted class is [argmax] along the last axis.

    Binary: both tensors have shape [[batch]] or [[batch; 1]]. Predictions above
    [0.5] count as class [1]. *)

val binary_accuracy :
  ?threshold:float -> (float, 'a) Rune.t -> (float, 'a) Rune.t -> float
(** [binary_accuracy ?threshold predictions targets] is the fraction of correct
    binary predictions.

    [threshold] defaults to [0.5]. Predictions above [threshold] count as class
    [1], targets are expected in \[[0];[1]\]. *)

(** {1:classification Classification} *)

val precision : average -> (float, 'a) Rune.t -> ('b, 'c) Rune.t -> float
(** [precision avg predictions targets] is the precision score.

    [predictions] has shape [[batch; num_classes]] (logits or probabilities).
    [targets] has shape [[batch]] (integer class indices). Predicted class is
    [argmax] along the last axis.

    When a class has no predicted instances, its precision is [0.0]. *)

val recall : average -> (float, 'a) Rune.t -> ('b, 'c) Rune.t -> float
(** [recall avg predictions targets] is the recall score.

    Input convention is the same as {!precision}.

    When a class has no true instances, its recall is [0.0]. *)

val f1 : average -> (float, 'a) Rune.t -> ('b, 'c) Rune.t -> float
(** [f1 avg predictions targets] is the F1 score (harmonic mean of {!precision}
    and {!recall}).

    Input convention is the same as {!precision}.

    When both precision and recall are [0.0] for a class, its F1 is [0.0]. *)
