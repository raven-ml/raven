(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Classification metrics.

    A metric maps a batch of predictions and integer labels to a plain [float].
    Metrics are evaluation summaries, never differentiated — train against
    {!Loss} — so they return bare floats that drop straight into logs and
    comparisons. The exception is {!confusion_matrix}, which returns a count
    matrix.

    Metrics are pure and hold no state; running tracking belongs to the caller
    (see munin) and aggregation across batches is the caller's fold. Batch means
    of {!accuracy}, {!top_k_accuracy} and micro-averaged scores weighted by
    batch size equal the dataset value, and {!confusion_matrix} sums over
    batches. {b Warning.} {!auc_roc} and macro-averaged scores do not decompose
    over batches: averaging per-batch values is not the dataset-level metric.
    Compute them once over the full evaluation set.

    Multiclass metrics share the sparse-label convention of
    {!Loss.softmax_cross_entropy_sparse}: [predictions] is a float tensor of
    shape [[...; classes]] holding logits or probabilities — only the order
    within each row matters — and [labels] holds [int32] class indices in
    \[[0];[classes - 1]\], shaped like [predictions] without the last axis. The
    predicted class is the argmax over the last axis, ties resolving to the
    lowest class index.

    All functions raise [Invalid_argument] — with messages prefixed by
    [Metric.<function>:] — if shapes disagree, a label is out of range, or there
    are no examples. *)

(** {1:accuracy Accuracy} *)

val accuracy : (float, 'a) Nx.t -> (int32, Nx.int32_elt) Nx.t -> float
(** [accuracy predictions labels] is the fraction of examples whose predicted
    class equals their label. *)

val top_k_accuracy :
  k:int -> (float, 'a) Nx.t -> (int32, Nx.int32_elt) Nx.t -> float
(** [top_k_accuracy ~k predictions labels] is the fraction of examples whose
    label ranks among the [k] highest-scoring classes. An example counts as
    correct when fewer than [k] classes score strictly higher than its label's
    class, so score ties resolve in the label's favor and [top_k_accuracy ~k:1]
    can exceed {!accuracy} on exact ties.

    Raises [Invalid_argument] if [k] is not in \[[1];[classes]\]. *)

(** {1:confusion Confusion-matrix metrics} *)

val confusion_matrix :
  (float, 'a) Nx.t -> (int32, Nx.int32_elt) Nx.t -> (int32, Nx.int32_elt) Nx.t
(** [confusion_matrix predictions labels] is the [[classes; classes]] matrix
    whose row [i], column [j] entry counts the examples with label [i] and
    predicted class [j]. Correct predictions lie on the diagonal. *)

type average = [ `Macro | `Micro ]
(** The type for combining per-class scores. [`Macro] is the unweighted mean
    over all [classes] classes, so rare classes weigh as much as common ones and
    classes absent from the batch contribute [0]. [`Micro] pools true-positive,
    false-positive and false-negative counts over all classes before scoring;
    for single-label classification micro precision, recall and F1 all equal
    {!accuracy}. *)

val precision :
  ?average:average -> (float, 'a) Nx.t -> (int32, Nx.int32_elt) Nx.t -> float
(** [precision ?average predictions labels] is the fraction of each class's
    predicted instances that are correct — [tp / (tp + fp)] — combined according
    to [average], which defaults to [`Macro]. A class never predicted has
    precision [0]. *)

val recall :
  ?average:average -> (float, 'a) Nx.t -> (int32, Nx.int32_elt) Nx.t -> float
(** [recall ?average predictions labels] is the fraction of each class's true
    instances that are predicted — [tp / (tp + fn)] — combined according to
    [average], which defaults to [`Macro]. A class with no true instances has
    recall [0]. *)

val f1 :
  ?average:average -> (float, 'a) Nx.t -> (int32, Nx.int32_elt) Nx.t -> float
(** [f1 ?average predictions labels] is the harmonic mean of precision and
    recall per class — [2 tp / (2 tp + fp + fn)] — combined according to
    [average], which defaults to [`Macro]. A class with no true and no predicted
    instances has F1 [0]. Macro F1 averages per-class F1 scores; it is not the
    harmonic mean of macro {!precision} and macro {!recall}. *)

(** {1:ranking Ranking} *)

val auc_roc : (float, 'a) Nx.t -> (int32, Nx.int32_elt) Nx.t -> float
(** [auc_roc scores labels] is the area under the ROC curve of a binary
    classifier: the probability that a uniformly drawn positive example
    outscores a uniformly drawn negative one, tied pairs counting half —
    equivalently, trapezoidal integration of the ROC curve. [scores] ranks the
    examples (logits or probabilities, higher meaning more positive; only their
    order matters) and [labels], of [scores]' shape, marks each example [0]
    (negative) or [1] (positive). Computed by sorting, in O(n log n).

    Raises [Invalid_argument] if the shapes differ, if a label is neither [0]
    nor [1], or if either class is absent — the curve is then undefined. *)
