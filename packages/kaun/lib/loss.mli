(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Loss functions.

    A loss maps predictions and targets to a scalar tensor — the shape
    {!Rune.grad} requires of a training objective. Every loss is differentiable
    with respect to its predictions (or logits); targets are data and receive no
    gradients.

    Classification losses take raw logits, not probabilities, and evaluate in
    log space: they are finite and accurate at any logit magnitude.

    [Invalid_argument] messages are prefixed with [Loss.<function>:]. *)

(** {1:reduction Reduction} *)

type reduction = [ `Mean | `Sum ]
(** The type for reducing per-element (or per-example) losses to a scalar.
    [`Mean] averages and [`Sum] adds. Losses default to [`Mean], which keeps the
    objective's scale independent of the batch size. *)

(** {1:regression Regression} *)

val mse :
  ?reduction:reduction ->
  (float, 'a) Nx.t ->
  (float, 'a) Nx.t ->
  (float, 'a) Nx.t
(** [mse predictions targets] is the reduction over all elements of the squared
    error [(predictions - targets)²]. Shape compatibility follows Nx
    broadcasting semantics. *)

val mae :
  ?reduction:reduction ->
  (float, 'a) Nx.t ->
  (float, 'a) Nx.t ->
  (float, 'a) Nx.t
(** [mae predictions targets] is the reduction over all elements of the absolute
    error [|predictions - targets|]. Shape compatibility follows Nx broadcasting
    semantics. Its gradient is discontinuous where an element of [predictions]
    equals its target; {!huber} is smooth there. *)

val huber :
  ?delta:float ->
  ?reduction:reduction ->
  (float, 'a) Nx.t ->
  (float, 'a) Nx.t ->
  (float, 'a) Nx.t
(** [huber ?delta predictions targets] is the reduction over all elements of the
    Huber loss of the error [d = predictions - targets]: [d²/2] where
    [|d| <= delta] and [delta * (|d| - delta/2)] beyond. Quadratic near zero
    like {!mse}, linear in the tails like {!mae}, with a continuous gradient at
    the crossover. Shape compatibility follows Nx broadcasting semantics.
    [delta] defaults to [1.0].

    Raises [Invalid_argument] if [delta] is not positive. *)

(** {1:classification Classification} *)

val sigmoid_bce :
  ?reduction:reduction ->
  (float, 'a) Nx.t ->
  (float, 'a) Nx.t ->
  (float, 'a) Nx.t
(** [sigmoid_bce logits targets] is the reduction over all elements of the
    binary cross-entropy between [sigmoid logits] and [targets]:
    [-(y log p + (1 - y) log (1 - p))] with [p = sigmoid logits] and
    [y = targets]. [targets] are probabilities in \[[0];[1]\] — hard [0]/[1]
    labels or soft targets. Shape compatibility follows Nx broadcasting
    semantics.

    Evaluated as [max logits 0 - logits * y + log (1 + exp (-|logits|))], which
    never forms [sigmoid logits] and stays finite at extreme logits. *)

val softmax_cross_entropy :
  ?reduction:reduction ->
  (float, 'a) Nx.t ->
  (float, 'a) Nx.t ->
  (float, 'a) Nx.t
(** [softmax_cross_entropy logits targets] is the reduction over examples of the
    cross-entropy between [softmax logits] and [targets] along the last axis:
    [-Σ targets * log_softmax logits]. [logits] has shape [[...; classes]];
    [targets] has the same shape, each slice along the last axis a probability
    distribution over classes — one-hot labels or soft targets. Computed via the
    log-sum-exp trick, so it is stable at extreme logits. Use
    {!softmax_cross_entropy_sparse} when targets are class indices.

    Raises [Invalid_argument] if [logits] has rank [0], if its class dimension
    is empty, or if [targets]' shape differs from [logits]'. *)

val softmax_cross_entropy_sparse :
  ?reduction:reduction ->
  (float, 'a) Nx.t ->
  (int32, Nx.int32_elt) Nx.t ->
  (float, 'a) Nx.t
(** [softmax_cross_entropy_sparse logits labels] is {!softmax_cross_entropy}
    with integer class labels: [labels] has [logits]' shape without the last
    axis, and each label is the index of the true class, in
    \[[0];[classes - 1]\]. Equivalent to one-hot targets without materializing
    them.

    Raises [Invalid_argument] if [logits] has rank [0], if its class dimension
    is empty, or if [labels]' shape is not [logits]' shape without the last
    axis. *)
