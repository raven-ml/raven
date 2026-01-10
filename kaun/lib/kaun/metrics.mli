(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Performance metrics for neural network training and evaluation.

    This module provides a comprehensive set of metrics for monitoring model
    performance during training and evaluation. Metrics are designed to be
    composable, efficient, and stateful for accumulation across batches while
    remaining layout-agnostic at the type level. *)

(** {1 Core Types} *)

type metric
(** Layout-independent metric accumulator that produces host [float] values when
    computed. *)

type reduction =
  | Mean
  | Sum  (** How to reduce metric values across batch dimensions *)

(** {1 Metric Creation} *)

(** {2 Classification Metrics} *)

val accuracy : ?threshold:float -> ?top_k:int -> unit -> metric
(** [accuracy ?threshold ?top_k ()] creates an accuracy metric.

    @param threshold Threshold for binary classification (default: 0.5)
    @param top_k
      For multi-class, count as correct if true label is in top-k predictions

    For per-class or aggregated variants, combine [Metrics.confusion_matrix]
    with custom post-processing.

    {4 Example}
    {[
      let acc = Metrics.accuracy () in
      let top5_acc = Metrics.accuracy ~top_k:5 ()
    ]} *)

val precision : ?threshold:float -> ?zero_division:float -> unit -> metric
(** [precision ?threshold ?zero_division ()] creates a precision metric.

    Precision = True Positives / (True Positives + False Positives)

    @param threshold Binary classification threshold (default: 0.5)
    @param zero_division
      Value to return when there are no positive predictions (default: 0.0)

    {4 Example}
    {[
      let prec = Metrics.precision ()
    ]} *)

val recall : ?threshold:float -> ?zero_division:float -> unit -> metric
(** [recall ?threshold ?zero_division ()] creates a recall metric.

    Recall = True Positives / (True Positives + False Negatives)

    @param threshold Binary classification threshold (default: 0.5)
    @param zero_division
      Value to return when there are no actual positives (default: 0.0) *)

val f1_score : ?threshold:float -> ?beta:float -> unit -> metric
(** [f1_score ?threshold ?beta ()] creates an F-score metric.

    F-score = (1 + β²) * (Precision * Recall) / (β² * Precision + Recall)

    @param threshold Binary classification threshold (default: 0.5)
    @param beta Weight of recall vs precision (default: 1.0 for F1) *)

val auc_roc : unit -> metric
(** [auc_roc ()] creates an AUC-ROC (Area Under the Receiver Operating
    Characteristic) metric that integrates true/false positive rates observed
    across batches. *)

val auc_pr : unit -> metric
(** [auc_pr ()] creates an AUC-PR (Area Under the Precision–Recall) metric.
    Computes the exact precision–recall integral by sorting predictions and
    accumulating precision/recall scores across all seen batches. *)

val confusion_matrix :
  num_classes:int ->
  ?normalize:[ `None | `True | `Pred | `All ] ->
  unit ->
  metric
(** [confusion_matrix ~num_classes ?normalize ()] accumulates a confusion matrix
    for classification tasks.

    @param num_classes Number of classes
    @param normalize Normalisation mode (default: [`None]) *)

(** {2 Regression Metrics} *)

val mse : ?reduction:reduction -> unit -> metric
(** [mse ?reduction ()] creates a Mean Squared Error metric.

    MSE = mean((predictions - targets)²) *)

val rmse : ?reduction:reduction -> unit -> metric
(** [rmse ?reduction ()] creates a Root Mean Squared Error metric.

    RMSE = sqrt(mean((predictions - targets)²)) *)

val mae : ?reduction:reduction -> unit -> metric
(** [mae ?reduction ()] creates a Mean Absolute Error metric.

    MAE = mean(|predictions - targets|) *)

val loss : unit -> metric
(** [loss ()] tracks the running mean of loss values. Pass batch losses through
    [update ~loss] to accumulate them. *)

val mape : ?eps:float -> unit -> metric
(** [mape ?eps ()] creates a Mean Absolute Percentage Error metric.

    MAPE = mean(|predictions - targets| / (|targets| + eps)) * 100

    @param eps Small value to avoid division by zero (default: 1e-7) *)

val r2_score : ?adjusted:bool -> ?num_features:int -> unit -> metric
(** [r2_score ?adjusted ?num_features ()] creates an R² coefficient of
    determination metric.

    R² = 1 - (SS_res / SS_tot)

    @param adjusted If true, compute adjusted R² (requires [num_features])
    @param num_features Number of features (needed for adjusted R²) *)

val explained_variance : unit -> metric
(** [explained_variance ()] creates an explained variance metric.

    EV = 1 - Var(targets - predictions) / Var(targets) *)

(** {2 Probabilistic Metrics} *)

val cross_entropy : ?from_logits:bool -> unit -> metric
(** [cross_entropy ?from_logits ()] creates a cross-entropy metric.

    @param from_logits If true, apply softmax to predictions (default: true) *)

val binary_cross_entropy : ?from_logits:bool -> unit -> metric
(** [binary_cross_entropy ?from_logits ()] creates a binary cross-entropy
    metric.

    @param from_logits If true, apply sigmoid to predictions (default: true) *)

val kl_divergence : ?eps:float -> unit -> metric
(** [kl_divergence ?eps ()] creates a Kullback–Leibler divergence metric.

    KL(P||Q) = Σ P log(P / Q)

    @param eps Small value for numerical stability (default: 1e-7) *)

val perplexity : ?base:float -> unit -> metric
(** [perplexity ?base ()] creates a perplexity metric for language models.

    Perplexity = base^(cross_entropy)

    @param base Base for exponentiation (default: e) *)

(** {2 Ranking Metrics} *)

val ndcg : ?k:int -> unit -> metric
(** [ndcg ?k ()] creates a Normalised Discounted Cumulative Gain metric.

    @param k Consider only the top-k ranked items (default: all) *)

val map : ?k:int -> unit -> metric
(** [map ?k ()] creates a Mean Average Precision metric for ranking. *)

val mrr : ?k:int -> unit -> metric
(** [mrr ?k ()] creates a Mean Reciprocal Rank metric.

    MRR = mean(1 / rank_of_first_relevant_item)

    @param k
      Consider only top-k items when computing the reciprocal rank (default:
      all) *)

(** {2 Natural Language Metrics} *)

val bleu :
  ?max_n:int -> ?weights:float array -> ?smoothing:bool -> unit -> metric
(** [bleu ?max_n ?weights ?smoothing ()] creates a BLEU score metric for
    pre-tokenized integer sequences.

    @param max_n Maximum n-gram order (default: 4)
    @param weights Weights for each n-gram order (default: uniform)
    @param smoothing Apply smoothing for zero counts (default: true)

    Predictions and targets must be shaped [batch, seq_len] with integer token
    identifiers. Zero values are treated as padding and ignored. *)

val rouge :
  variant:[ `Rouge1 | `Rouge2 | `RougeL ] -> ?use_stemmer:bool -> unit -> metric
(** [rouge ~variant ?use_stemmer ()] creates a ROUGE score metric for
    pre-tokenized integer sequences.

    @param variant Which ROUGE variant to compute
    @param use_stemmer Enable stemming (currently unsupported; raises when set)

    Predictions and targets must be shaped [batch, seq_len] with integer token
    identifiers. Zero values are treated as padding and ignored. *)

val meteor : ?alpha:float -> ?beta:float -> ?gamma:float -> unit -> metric
(** [meteor ?alpha ?beta ?gamma ()] creates a METEOR score metric for
    pre-tokenized integer sequences.

    @param alpha
      Parameter controlling precision vs recall balance (default: 0.9)
    @param beta Exponent for the chunk penalty (default: 3.0)
    @param gamma Weight of the chunk penalty (default: 0.5)

    Predictions and targets must be shaped [batch, seq_len] with integer token
    identifiers. Zero values are treated as padding and ignored. *)

(** {2 Image Metrics} *)

val psnr : ?max_val:float -> unit -> metric
(** [psnr ?max_val ()] creates a Peak Signal-to-Noise Ratio metric.

    PSNR = 10 * log10(max_val² / MSE)

    @param max_val Maximum possible pixel value (default: 1.0) *)

val ssim : ?window_size:int -> ?k1:float -> ?k2:float -> unit -> metric
(** [ssim ?window_size ?k1 ?k2 ()] creates a Structural Similarity Index metric.

    The implementation evaluates the global SSIM across the full prediction and
    target tensors using scalar statistics derived from [window_size], [k1], and
    [k2]. *)

val iou :
  ?threshold:float -> ?per_class:bool -> num_classes:int -> unit -> metric
(** [iou ?threshold ?per_class ~num_classes ()] creates an Intersection over
    Union metric.

    Inputs must contain integer class indices in \[0, num_classes). When
    [num_classes = 2], [threshold] binarises predictions before computing IoU.
    When [per_class = true], the metric reports one IoU per class; otherwise it
    returns the mean over classes with non-zero support. *)

val dice :
  ?threshold:float -> ?per_class:bool -> num_classes:int -> unit -> metric
(** [dice ?threshold ?per_class ~num_classes ()] creates a Sørenson Dice
    coefficient metric with the same input conventions as {!iou}. *)

(** {1 Metric Operations} *)

val update :
  metric ->
  predictions:(float, 'layout) Rune.t ->
  targets:(_, 'layout) Rune.t ->
  ?loss:(float, 'layout) Rune.t ->
  ?weights:(float, 'layout) Rune.t ->
  unit ->
  unit
(** [update metric ~predictions ~targets ?loss ?weights ()] updates the metric
    state. All tensors must share the same (hidden) layout. When supplied, the
    [loss] tensor is treated as an auxiliary scalar for metrics that track
    losses. *)

val compute : metric -> float
(** [compute metric] returns the aggregated metric value as a host float. *)

val compute_tensor : metric -> Ptree.tensor
(** [compute_tensor metric] returns the aggregated metric value as a device
    tensor. *)

val reset : metric -> unit
(** [reset metric] clears internal accumulators for a fresh run. *)

val clone : metric -> metric
(** [clone metric] creates a new metric with the same configuration but fresh
    state. *)

val name : metric -> string
(** [name metric] returns the metric's descriptive name. *)

val create_custom :
  dtype:(float, 'layout) Rune.dtype ->
  name:string ->
  init:(unit -> (float, 'layout) Rune.t list) ->
  update:
    ((float, 'layout) Rune.t list ->
    predictions:(float, 'layout) Rune.t ->
    targets:(float, 'layout) Rune.t ->
    ?weights:(float, 'layout) Rune.t ->
    unit ->
    (float, 'layout) Rune.t list) ->
  compute:((float, 'layout) Rune.t list -> (float, 'layout) Rune.t) ->
  reset:((float, 'layout) Rune.t list -> (float, 'layout) Rune.t list) ->
  metric
(** [create_custom ~dtype ~name ~init ~update ~compute ~reset] constructs a
    custom metric from user-provided accumulator functions. *)

val is_better :
  metric -> higher_better:bool -> old_val:float -> new_val:float -> bool
(** [is_better metric ~higher_better ~old_val ~new_val] determines whether the
    new metric value improves upon the previous one. *)

val format : metric -> float -> string
(** [format metric value] pretty-prints a metric value for logging. *)

(** {1 Metric Collections} *)

module Collection : sig
  type t
  (** Layout-agnostic collection of named metrics. *)

  val empty : unit -> t
  val of_list : (string * metric) list -> t
  val create : (string * metric) list -> t
  val add : t -> string -> metric -> unit
  val remove : t -> string -> unit
  val reset : t -> unit

  val update :
    t ->
    predictions:(float, 'layout) Rune.t ->
    targets:(_, 'layout) Rune.t ->
    ?loss:(float, 'layout) Rune.t ->
    ?weights:(float, 'layout) Rune.t ->
    unit ->
    unit

  val compute : t -> (string * float) list
  val compute_tensors : t -> (string * Ptree.tensor) list
  val compute_dict : t -> (string, float) Hashtbl.t
end
