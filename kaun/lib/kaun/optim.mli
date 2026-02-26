(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Optimizers and learning-rate schedules.

    An {!algorithm} combines a learning-rate schedule and an update rule.
    {!state} stores optimizer-specific accumulators and the step count. *)

(** {1:types Types} *)

type state
(** The type for optimizer states. *)

type algorithm
(** The type for optimization algorithms. *)

(** {1:core Core} *)

val init : algorithm -> Ptree.t -> state
(** [init algo params] is the initial state of [algo] for [params]. *)

val step : algorithm -> state -> Ptree.t -> Ptree.t -> Ptree.t * state
(** [step algo state params grads] is [(updates, state')] where [updates] are
    additive parameter deltas.

    The step count is incremented before the learning-rate schedule is
    evaluated. Use {!apply_updates} to apply [updates] to [params]. *)

val apply_updates : Ptree.t -> Ptree.t -> Ptree.t
(** [apply_updates params updates] is [params + updates] element-wise. *)

val update : algorithm -> state -> Ptree.t -> Ptree.t -> Ptree.t * state
(** [update algo st params grads] is
    [let u, st' = step algo st params grads in (apply_updates params u, st')].

    Convenience for the common case where you want updated parameters directly
    rather than additive deltas. *)

(** {1:schedules Learning-Rate Schedules} *)

module Schedule : sig
  type t = int -> float
  (** The type for learning-rate schedules.

      [s step] is the learning rate for 1-based [step]. *)

  val constant : float -> t
  (** [constant lr] is the schedule that always returns [lr]. *)

  val cosine_decay :
    init_value:float -> decay_steps:int -> ?alpha:float -> unit -> t
  (** [cosine_decay ~init_value ~decay_steps ?alpha ()] is cosine decay from
      [init_value] to [alpha * init_value] over [decay_steps].

      [alpha] defaults to [0.]. *)

  val warmup_cosine :
    init_value:float -> peak_value:float -> warmup_steps:int -> t
  (** [warmup_cosine ~init_value ~peak_value ~warmup_steps] is cosine warmup
      from [init_value] to [peak_value] over [warmup_steps]. *)

  val exponential_decay :
    init_value:float -> decay_rate:float -> decay_steps:int -> t
  (** [exponential_decay ~init_value ~decay_rate ~decay_steps] is
      [init_value * decay_rate{^ (step / decay_steps)}]. *)

  val warmup_linear :
    init_value:float -> peak_value:float -> warmup_steps:int -> t
  (** [warmup_linear ~init_value ~peak_value ~warmup_steps] is linear warmup
      from [init_value] to [peak_value] over [warmup_steps]. *)
end

(** {1:optimizers Optimizers} *)

val sgd :
  lr:Schedule.t -> ?momentum:float -> ?nesterov:bool -> unit -> algorithm
(** [sgd ~lr ?momentum ?nesterov ()] is stochastic gradient descent.

    [momentum] defaults to [0.]. [nesterov] defaults to [false]. Nesterov mode
    is ignored when [momentum = 0.].

    Raises [Invalid_argument] if [momentum] is not in [0.0 <= momentum < 1.0].
*)

val adam :
  lr:Schedule.t -> ?b1:float -> ?b2:float -> ?eps:float -> unit -> algorithm
(** [adam ~lr ?b1 ?b2 ?eps ()] is Adam with bias correction.

    [b1] defaults to [0.9]. [b2] defaults to [0.999]. [eps] defaults to [1e-8].

    Raises [Invalid_argument] if [b1] or [b2] is not in [0.0 <= b < 1.0], or if
    [eps <= 0.0]. *)

val adamw :
  lr:Schedule.t ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  ?weight_decay:float ->
  unit ->
  algorithm
(** [adamw ~lr ?b1 ?b2 ?eps ?weight_decay ()] is AdamW.

    [b1] defaults to [0.9]. [b2] defaults to [0.999]. [eps] defaults to [1e-8].
    [weight_decay] defaults to [0.01].

    Weight decay is decoupled from the Adam moment estimates.

    Raises [Invalid_argument] if [b1] or [b2] is not in [0.0 <= b < 1.0], if
    [eps <= 0.0], or if [weight_decay < 0.0]. *)

val rmsprop :
  lr:Schedule.t ->
  ?decay:float ->
  ?eps:float ->
  ?momentum:float ->
  unit ->
  algorithm
(** [rmsprop ~lr ?decay ?eps ?momentum ()] is RMSprop.

    [decay] defaults to [0.9]. [eps] defaults to [1e-8]. [momentum] defaults to
    [0.] (no momentum).

    Raises [Invalid_argument] if [decay] or [momentum] is not in
    [0.0 <= x < 1.0], or if [eps <= 0.0]. *)

val adagrad : lr:Schedule.t -> ?eps:float -> unit -> algorithm
(** [adagrad ~lr ?eps ()] is Adagrad.

    [eps] defaults to [1e-8].

    Raises [Invalid_argument] if [eps <= 0.0]. *)

(** {1:serialization Serialization} *)

val state_to_trees : state -> int * Ptree.t list
(** [state_to_trees st] is [(count, trees)] where [count] is the optimizer step
    count and [trees] are the internal state as parameter trees.

    SGD without momentum returns an empty list. Adam returns [[mu; nu]]. *)

val state_of_trees : algorithm -> count:int -> Ptree.t list -> state
(** [state_of_trees algo ~count trees] reconstructs optimizer state from an
    algorithm, step count, and serialized trees.

    Raises [Invalid_argument] if the number of trees does not match the
    algorithm's expectation. *)

(** {1:grad Gradient Utilities} *)

val clip_by_global_norm : float -> Ptree.t -> Ptree.t
(** [clip_by_global_norm max_norm grads] rescales [grads] so their global L2
    norm does not exceed [max_norm]. Returns [grads] unchanged if the norm is
    already within bounds.

    Raises [Invalid_argument] if a leaf tensor is not floating point. *)

val global_norm : Ptree.t -> float
(** [global_norm t] is the L2 norm across all leaf tensors of [t].

    Raises [Invalid_argument] if a leaf tensor is not floating point. *)
