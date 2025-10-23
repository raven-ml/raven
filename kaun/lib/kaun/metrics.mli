(** Performance metrics for neural network training and evaluation.

    This module provides a comprehensive set of metrics for monitoring model
    performance during training and evaluation. Metrics are designed to be
    composable, efficient, and stateful for accumulation across batches.

    {2 Design Philosophy}

    Metrics in this module follow a stateful accumulator pattern:
    - Create metrics with specific configurations
    - Update them incrementally with batches of data
    - Compute final values when needed
    - Reset for new epochs or evaluation runs

    All metrics support:
    - Batch-wise updates for efficient computation
    - Weighted samples for imbalanced datasets
    - Multi-task learning with metric collections
    - Distributed training aggregation *)

(** {1 Core Types} *)

type 'layout t
(** A stateful metric accumulator *)

type 'layout metric_fn =
  predictions:(float, 'layout) Rune.t ->
  targets:(float, 'layout) Rune.t ->
  ?weights:(float, 'layout) Rune.t ->
  unit ->
  (float, 'layout) Rune.t
(** Function signature for metric computations *)

type reduction =
  | Mean
  | Sum
  | None  (** How to reduce metric values across batch dimensions *)

type averaging =
  | Micro
  | Macro
  | Weighted
  | Samples
      (** Averaging strategy for multi-class/multi-label metrics:
          - Micro: Calculate metric globally across all samples
          - Macro: Calculate metric for each class and average
          - Weighted: Like macro but weighted by class support
          - Samples: Average across samples (for multi-label) *)

(** {1 Metric Creation} *)

(** {2 Classification Metrics} *)

val accuracy :
  ?threshold:float -> ?top_k:int -> ?averaging:averaging -> unit -> 'layout t
(** [accuracy ?threshold ?top_k ?averaging ()] creates an accuracy metric.

    @param threshold
      For binary classification, threshold for positive class (default: 0.5)
    @param top_k
      For multi-class, count as correct if true label is in top-k predictions
    @param averaging
      Averaging strategy for multi-class problems (default: Micro)

    {4 Example}
    {[
      let acc = Metrics.accuracy () in
      let top5_acc = Metrics.accuracy ~top_k:5 ()
    ]} *)

val precision :
  ?threshold:float ->
  ?averaging:averaging ->
  ?zero_division:float ->
  unit ->
  'layout t
(** [precision ?threshold ?averaging ?zero_division ()] creates a precision
    metric.

    Precision = True Positives / (True Positives + False Positives)

    @param threshold Binary classification threshold (default: 0.5)
    @param averaging Multi-class averaging strategy (default: Micro)
    @param zero_division
      Value to return when there are no positive predictions (default: 0.0)

    {4 Example}
    {[
      let prec = Metrics.precision ~averaging:Macro ()
    ]} *)

val recall :
  ?threshold:float ->
  ?averaging:averaging ->
  ?zero_division:float ->
  unit ->
  'layout t
(** [recall ?threshold ?averaging ?zero_division ()] creates a recall metric.

    Recall = True Positives / (True Positives + False Negatives)

    @param threshold Binary classification threshold (default: 0.5)
    @param averaging Multi-class averaging strategy (default: Micro)
    @param zero_division
      Value to return when there are no actual positives (default: 0.0) *)

val f1_score :
  ?threshold:float -> ?averaging:averaging -> ?beta:float -> unit -> 'layout t
(** [f1_score ?threshold ?averaging ?beta ()] creates an F-score metric.

    F-score = (1 + β²) * (Precision * Recall) / (β² * Precision + Recall)

    @param threshold Binary classification threshold (default: 0.5)
    @param averaging Multi-class averaging strategy (default: Micro)
    @param beta Weight of recall vs precision (default: 1.0 for F1) *)

val auc_roc : unit -> 'layout t
(** [auc_roc ()] creates an AUC-ROC metric.

    Area Under the Receiver Operating Characteristic Curve.

    Computes the exact ROC integral by sorting predictions and accumulating
    true/false positive rates across all seen batches. *)

val auc_pr : ?num_thresholds:int -> ?curve:bool -> unit -> 'layout t
(** [auc_pr ?num_thresholds ?curve ()] creates an AUC-PR metric.

    Area Under the Precision-Recall Curve. *)

val confusion_matrix :
  num_classes:int ->
  ?normalize:[ `None | `True | `Pred | `All ] ->
  unit ->
  'layout t
(** [confusion_matrix ~num_classes ?normalize ()] creates a confusion matrix
    accumulator.

    @param num_classes Number of classes
    @param normalize Normalization mode (default: `None) *)

(** {2 Regression Metrics} *)

val mse : ?reduction:reduction -> unit -> 'layout t
(** [mse ?reduction ()] creates a Mean Squared Error metric.

    MSE = mean((predictions - targets)²) *)

val rmse : ?reduction:reduction -> unit -> 'layout t
(** [rmse ?reduction ()] creates a Root Mean Squared Error metric.

    RMSE = sqrt(mean((predictions - targets)²)) *)

val mae : ?reduction:reduction -> unit -> Bigarray.float32_elt t
(** [mae ?reduction ()] creates a Mean Absolute Error metric.

    MAE = mean(|predictions - targets|) *)

val loss : unit -> 'layout t
(** [loss ()] creates a loss metric that tracks the running average of loss
    values.

    This metric is designed to track training/validation loss alongside other
    metrics. The loss value should be passed via the weights parameter in
    update. *)

val mape : ?eps:float -> unit -> 'layout t
(** [mape ?eps ()] creates a Mean Absolute Percentage Error metric.

    MAPE = mean(|predictions - targets| / (|targets| + eps)) * 100

    @param eps Small value to avoid division by zero (default: 1e-7) *)

val r2_score : ?adjusted:bool -> ?num_features:int -> unit -> 'layout t
(** [r2_score ?adjusted ?num_features ()] creates an R² coefficient of
    determination.

    R² = 1 - (SS_res / SS_tot)

    @param adjusted If true, compute adjusted R² (requires num_features)
    @param num_features Number of features (required for adjusted R²) *)

val explained_variance : unit -> 'layout t
(** [explained_variance ()] creates an explained variance metric.

    EV = 1 - Var(targets - predictions) / Var(targets) *)

(** {2 Probabilistic Metrics} *)

val cross_entropy : ?from_logits:bool -> unit -> 'layout t
(** [cross_entropy ?from_logits ()] creates a cross-entropy metric.

    @param from_logits If true, apply softmax to predictions (default: true) *)

val binary_cross_entropy : ?from_logits:bool -> unit -> 'layout t
(** [binary_cross_entropy ?from_logits ()] creates a binary cross-entropy
    metric.

    @param from_logits If true, apply sigmoid to predictions (default: true) *)

val kl_divergence : ?eps:float -> unit -> 'layout t
(** [kl_divergence ?eps ()] creates a KL divergence metric.

    KL(P||Q) = sum(P * log(P / Q))

    @param eps Small value for numerical stability (default: 1e-7) *)

val perplexity : ?base:float -> unit -> 'layout t
(** [perplexity ?base ()] creates a perplexity metric for language models.

    Perplexity = base^(cross_entropy)

    @param base Base for exponentiation (default: e ≈ 2.718) *)

(** {2 Ranking Metrics} *)

val ndcg : ?k:int -> unit -> 'layout t
(** [ndcg ?k ()] creates a Normalized Discounted Cumulative Gain metric.

    @param k Consider only top-k items (default: all) *)

val map : ?k:int -> unit -> 'layout t
(** [map ?k ()] creates a Mean Average Precision metric for ranking.

    @param k Consider only top-k items (default: all) *)

val mrr : unit -> 'layout t
(** [mrr ()] creates a Mean Reciprocal Rank metric.

    MRR = mean(1 / rank_of_first_relevant_item) *)

(** {2 Natural Language Metrics} *)

val bleu :
  ?max_n:int ->
  ?weights:float array ->
  ?smoothing:bool ->
  tokenizer:(string -> int array) ->
  unit ->
  'layout t
(** [bleu ?max_n ?weights ?smoothing ~tokenizer ()] creates a BLEU score metric.

    @param max_n Maximum n-gram order (default: 4)
    @param weights Weights for each n-gram order (default: uniform)
    @param smoothing Apply smoothing for zero counts (default: true)
    @param tokenizer Function to tokenize text into token IDs *)

val rouge :
  variant:[ `Rouge1 | `Rouge2 | `RougeL ] ->
  ?use_stemmer:bool ->
  tokenizer:(string -> int array) ->
  unit ->
  'layout t
(** [rouge ~variant ?use_stemmer ~tokenizer ()] creates a ROUGE score metric.

    @param variant Which ROUGE variant to compute
    @param use_stemmer Apply stemming before comparison (default: false)
    @param tokenizer Function to tokenize text *)

val meteor :
  ?alpha:float ->
  ?beta:float ->
  ?gamma:float ->
  tokenizer:(string -> int array) ->
  unit ->
  'layout t
(** [meteor ?alpha ?beta ?gamma ~tokenizer ()] creates a METEOR score metric.

    @param alpha Parameter for recall weight (default: 0.9)
    @param beta Parameter for precision weight (default: 3.0)
    @param gamma Penalty for fragmentation (default: 0.5) *)

(** {2 Image Metrics} *)

val psnr : ?max_val:float -> unit -> 'layout t
(** [psnr ?max_val ()] creates a Peak Signal-to-Noise Ratio metric.

    PSNR = 10 * log10(max_val² / MSE)

    @param max_val Maximum possible pixel value (default: 1.0) *)

val ssim : ?window_size:int -> ?k1:float -> ?k2:float -> unit -> 'layout t
(** [ssim ?window_size ?k1 ?k2 ()] creates a Structural Similarity Index metric.

    @param window_size Size of the gaussian window (default: 11)
    @param k1 Constant for luminance (default: 0.01)
    @param k2 Constant for contrast (default: 0.03) *)

val iou :
  ?threshold:float -> ?per_class:bool -> num_classes:int -> unit -> 'layout t
(** [iou ?threshold ?per_class ~num_classes ()] creates an Intersection over
    Union metric.

    @param threshold Threshold for binary segmentation (default: 0.5)
    @param per_class Return IoU per class (default: false)
    @param num_classes Number of segmentation classes *)

val dice :
  ?threshold:float -> ?per_class:bool -> num_classes:int -> unit -> 'layout t
(** [dice ?threshold ?per_class ~num_classes ()] creates a Dice coefficient
    metric.

    Dice = 2 * |A ∩ B| / (|A| + |B|) *)

(** {1 Metric Operations} *)

val update :
  'layout t ->
  predictions:(float, 'layout) Rune.t ->
  targets:(float, 'layout) Rune.t ->
  ?weights:(float, 'layout) Rune.t ->
  unit ->
  unit
(** [update metric ~predictions ~targets ?weights ()] updates the metric state.

    @param predictions Model predictions
    @param targets Ground truth targets
    @param weights Optional sample weights for weighted metrics *)

val compute : 'layout t -> (float, 'layout) Rune.t
(** [compute metric] computes the final metric value from accumulated state.

    @return Scalar tensor with the metric value *)

val reset : 'layout t -> unit
(** [reset metric] resets the metric state for a new epoch or evaluation. *)

val clone : 'layout t -> 'layout t
(** [clone metric] creates a copy of the metric with fresh state. *)

(** {1 Metric Collections} *)

module Collection : sig
  type 'layout metric = 'layout t

  type 'layout t
  (** A collection of metrics computed together *)

  val create : (string * 'layout metric) list -> 'layout t
  (** [create metrics] creates a metric collection from named metrics. *)

  val update :
    'layout t ->
    predictions:(float, 'layout) Rune.t ->
    targets:(float, 'layout) Rune.t ->
    ?weights:(float, 'layout) Rune.t ->
    unit ->
    unit
  (** [update collection ~predictions ~targets ?weights ()] updates all metrics.
  *)

  val update_with_loss :
    'layout t ->
    loss:(float, 'layout) Rune.t ->
    predictions:(float, 'layout) Rune.t ->
    targets:(float, 'layout) Rune.t ->
    unit ->
    unit
  (** [update_with_loss collection ~loss ~predictions ~targets ()] updates all
      metrics including loss tracking. The loss value is automatically passed to
      the loss metric if present in the collection. *)

  val compute : 'layout t -> (string * (float, 'layout) Rune.t) list
  (** [compute collection] computes all metric values. *)

  val compute_dict : 'layout t -> (string, (float, 'layout) Rune.t) Hashtbl.t
  (** [compute_dict collection] computes metrics as a hash table. *)

  val reset : 'layout t -> unit
  (** [reset collection] resets all metrics. *)

  val add : 'layout t -> string -> 'layout metric -> unit
  (** [add collection name metric] adds a new metric to the collection. *)

  val remove : 'layout t -> string -> unit
  (** [remove collection name] removes a metric from the collection. *)
end

(** {1 Custom Metrics} *)

val create_custom :
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
  'layout t
(** [create_custom ~name ~init ~update ~compute ~reset] creates a custom metric.

    @param name Metric name for debugging
    @param init Function to initialize state tensors
    @param update Function to update state with new batch
    @param compute Function to compute final metric from state
    @param reset Function to reset state

    {4 Example}
    {[
      let harmonic_mean =
        create_custom ~name:"harmonic_mean"
          ~init:(fun () ->
            [
              Rune.zeros device float32 [||];
              (* sum of reciprocals *)
              Rune.zeros device float32 [||];
              (* count *)
            ])
          ~update:(fun state ~predictions ~targets ?weights () ->
            let sum_recip, count =
              match state with
              | [ s; c ] -> (s, c)
              | _ -> failwith "Invalid state"
            in
            let diff = Rune.abs (Rune.sub predictions targets) in
            let reciprocals =
              Rune.reciprocal (Rune.add diff (Rune.scalar device float32 1e-7))
            in
            let batch_sum = Rune.sum reciprocals in
            let batch_count =
              Rune.scalar device float32 (float_of_int (Rune.numel predictions))
            in
            [ Rune.add sum_recip batch_sum; Rune.add count batch_count ])
          ~compute:(fun state ->
            match state with
            | [ sum_recip; count ] -> Rune.div count sum_recip
            | _ -> failwith "Invalid state")
          ~reset:(fun _ ->
            [ Rune.zeros device float32 [||]; Rune.zeros device float32 [||] ])
    ]} *)

(** {1 Utilities} *)

val name : 'layout t -> string
(** [name metric] returns the metric's name for logging. *)

val is_better :
  'layout t -> higher_better:bool -> old_val:float -> new_val:float -> bool
(** [is_better metric ~higher_better ~old_val ~new_val] checks if new value is
    better.

    @param higher_better If true, higher values are better
    @param old_val Previous metric value
    @param new_val Current metric value *)

val format : 'layout t -> (float, 'layout) Rune.t -> string
(** [format metric value] formats the metric value for display. *)
