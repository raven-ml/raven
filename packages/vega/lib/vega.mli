(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Composable gradient-based optimizers.

    Vega provides typed, per-parameter optimizer primitives that compose via
    {!chain}. Each primitive is a gradient transformation: it takes updates
    (gradients) and returns modified updates. Primitives are chained to build
    optimizers:

    {[
      let tx =
        Vega.chain
          [
            Vega.scale_by_adam ();
            Vega.add_decayed_weights ~rate:(Schedule.constant 0.01) ();
            Vega.scale_by_learning_rate lr;
          ]
    ]}

    Common optimizers are provided as aliases: {!adam}, {!sgd}, {!adamw}, etc.

    {b Narrow waist.} The core abstraction is [t]: a composable gradient
    transformation. The per-parameter {!state} is fully self-contained and
    tracks moments, step count, and the update rule. *)

(** {1:schedules Learning-Rate Schedules} *)

module Schedule = Schedule

(** {1:types Types} *)

type t
(** A composable gradient transformation. Constructed via primitives like
    {!scale_by_adam}, {!trace}, etc., and composed via {!chain}. *)

type ('a, 'b) state
(** Per-parameter optimizer state. Typed to match the parameter tensor. Tracks
    moments, step count, and the transformation chain. Created via {!init},
    advanced via {!update} or {!step}. *)

(** {1:core Core} *)

val chain : t list -> t
(** [chain transforms] composes transforms sequentially. {!update} applies each
    transform in order, threading the modified updates through.

    {!chain} is associative: [chain [chain [a; b]; c]] is equivalent to
    [chain [a; b; c]]. *)

val init : t -> ('a, 'b) Nx.t -> ('a, 'b) state
(** [init tx param] creates initial optimizer state matching [param]'s shape and
    dtype. Step count starts at [0]. *)

val update :
  ('a, 'b) state ->
  grad:('a, 'b) Nx.t ->
  param:('a, 'b) Nx.t ->
  ('a, 'b) Nx.t * ('a, 'b) state
(** [update state ~grad ~param] returns [(updates, new_state)].

    The returned [updates] are gradient-scale values that include the
    learning-rate sign. Apply them via {!apply_updates}. *)

val apply_updates :
  param:('a, 'b) Nx.t -> updates:('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [apply_updates ~param ~updates] is [Nx.add param updates]. *)

val step :
  ('a, 'b) state ->
  grad:('a, 'b) Nx.t ->
  param:('a, 'b) Nx.t ->
  ('a, 'b) Nx.t * ('a, 'b) state
(** [step state ~grad ~param] returns [(new_param, new_state)].

    Convenience for:
    {[
      let updates, state = update state ~grad ~param in
      (apply_updates ~param ~updates, state)
    ]} *)

(** {1:scaling Scaling Transforms} *)

val scale : float -> t
(** [scale s] multiplies updates by [s]. Stateless. *)

val scale_by_schedule : Schedule.t -> t
(** [scale_by_schedule f] multiplies updates by [f step]. *)

val scale_by_learning_rate : Schedule.t -> t
(** [scale_by_learning_rate lr] multiplies updates by [-lr step]. Negates the
    learning rate so that {!apply_updates} performs gradient descent. *)

(** {1:adaptive Adaptive Scaling Transforms} *)

val scale_by_adam :
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  ?nesterov:bool ->
  ?amsgrad:bool ->
  unit ->
  t
(** [scale_by_adam ?b1 ?b2 ?eps ?nesterov ?amsgrad ()] scales updates by Adam's
    bias-corrected first and second moment estimates.

    When [amsgrad] is [true], the denominator uses the running maximum of past
    second moments, preventing the adaptive learning rate from increasing.

    [b1] defaults to [0.9]. [b2] defaults to [0.999]. [eps] defaults to [1e-8].
    [nesterov] defaults to [false]. [amsgrad] defaults to [false].

    State: 2 tensors when [amsgrad] is [false], 3 when [true] (first moment,
    second moment, max second moment). *)

val scale_by_rms : ?decay:float -> ?eps:float -> unit -> t
(** [scale_by_rms ?decay ?eps ()] scales updates by the inverse root mean square
    of past gradients (the core of RMSprop).

    [decay] defaults to [0.9]. [eps] defaults to [1e-8].

    State: 1 tensor (second moment EMA). *)

val scale_by_adagrad : ?eps:float -> unit -> t
(** [scale_by_adagrad ?eps ()] scales updates by the inverse root of accumulated
    squared gradients.

    [eps] defaults to [1e-8].

    State: 1 tensor (accumulated squared gradients). *)

val scale_by_lion : ?b1:float -> ?b2:float -> unit -> t
(** [scale_by_lion ?b1 ?b2 ()] produces sign-based updates using two momentum
    rates: [b1] for the update direction, [b2] for the momentum state.

    [b1] defaults to [0.9]. [b2] defaults to [0.99].

    State: 1 tensor (momentum). *)

val scale_by_radam : ?b1:float -> ?b2:float -> ?eps:float -> unit -> t
(** [scale_by_radam ?b1 ?b2 ?eps ()] scales by rectified Adam. Uses the length
    of the approximated SMA to decide between adaptive and momentum-only
    updates, avoiding unstable variance in early steps.

    [b1] defaults to [0.9]. [b2] defaults to [0.999]. [eps] defaults to [1e-8].

    State: 2 tensors (first moment, second moment). *)

val scale_by_trust_ratio : ?eps:float -> unit -> t
(** [scale_by_trust_ratio ?eps ()] scales updates by the ratio
    [||param|| / (||updates|| + eps)] (the LAMB/LARS trust ratio).

    [eps] defaults to [1e-6].

    State: 0 tensors. *)

val scale_by_adafactor :
  ?b2_decay:[ `Constant of float | `Rms ] ->
  ?eps:float ->
  ?eps_scale:float ->
  ?factored:bool ->
  ?clipping_threshold:float ->
  unit ->
  t
(** [scale_by_adafactor ?b2_decay ?eps ?eps_scale ?factored ?clipping_threshold
     ()] scales updates using Adafactor's factored second-moment estimation. For
    2D+ parameters, row and column factors are maintained instead of the full
    second moment matrix, reducing memory from O(mn) to O(m+n).

    [b2_decay] controls second moment decay. [`Rms] (default) uses
    [1 - step{^-0.8}]. [`Constant rho] uses fixed decay [rho]. [eps] defaults to
    [1e-30]. [eps_scale] defaults to [1e-3]. [factored] defaults to [true]; when
    [false], uses a full second moment. [clipping_threshold] defaults to [1.0];
    set to [infinity] to disable.

    State: 2 tensors (row factor, col factor for factored 2D+; full second
    moment + dummy for 1D or unfactored). *)

val scale_by_adan :
  ?b1:float -> ?b2:float -> ?b3:float -> ?eps:float -> unit -> t
(** [scale_by_adan ?b1 ?b2 ?b3 ?eps ()] scales updates using Adan's adaptive
    Nesterov momentum estimation. Maintains first moment, gradient difference
    moment, and second moment.

    [b1] defaults to [0.98]. [b2] defaults to [0.92]. [b3] defaults to [0.99].
    [eps] defaults to [1e-8].

    State: 4 tensors (first moment, gradient difference moment, second moment,
    previous gradient). *)

(** {1:accumulation Accumulation Transforms} *)

val trace : ?decay:float -> ?nesterov:bool -> unit -> t
(** [trace ?decay ?nesterov ()] accumulates a trace (momentum) of updates.

    [decay] defaults to [0.9]. [nesterov] defaults to [false].

    State: 1 tensor (trace/velocity). *)

(** {1:regularization Regularization Transforms} *)

val add_decayed_weights : ?rate:Schedule.t -> unit -> t
(** [add_decayed_weights ?rate ()] adds [rate step * param] to updates. When
    placed before {!scale_by_learning_rate}, this implements decoupled weight
    decay.

    [rate] defaults to [Schedule.constant 0.01].

    State: 0 tensors. *)

(** {1:clipping Clipping Transforms} *)

val clip_by_value : float -> t
(** [clip_by_value delta] clips updates element-wise to [[-delta, +delta]].

    State: 0 tensors. *)

val clip_by_norm : float -> t
(** [clip_by_norm max_norm] rescales updates so their L2 norm does not exceed
    [max_norm]. Returns updates unchanged if the norm is already within bounds.

    State: 0 tensors. *)

(** {1:gradient_processing Gradient Processing} *)

val centralize : t
(** [centralize] subtracts the mean from each gradient tensor. For tensors with
    2+ dimensions, the mean is computed over all axes except the first (output
    features). Scalars and 1D tensors are left unchanged.

    State: 0 tensors. *)

val add_noise : eta:Schedule.t -> ?gamma:float -> unit -> t
(** [add_noise ~eta ?gamma ()] adds Gaussian noise with variance
    [eta step / (1 + step){^ gamma}] to updates. The annealing ensures noise
    decreases over training.

    [gamma] defaults to [0.55].

    State: 0 tensors. *)

(** {1:robustness Robustness} *)

val apply_if_finite : t -> t
(** [apply_if_finite tx] wraps [tx] so that if any update produced by [tx]
    contains non-finite values (NaN or Inf), the update is skipped: zero updates
    are returned and the inner state is not advanced.

    State: inner state + 1 tensor (count of consecutive non-finite steps). *)

(** {1:aliases Optimizer Aliases} *)

val sgd : ?momentum:float -> ?nesterov:bool -> Schedule.t -> t
(** [sgd lr] is stochastic gradient descent.

    Without momentum: [chain [scale_by_learning_rate lr]]. With momentum:
    [chain [trace ~decay:momentum ~nesterov (); scale_by_learning_rate lr]].

    [momentum] defaults to [0.]. [nesterov] defaults to [false]. *)

val adam : ?b1:float -> ?b2:float -> ?eps:float -> Schedule.t -> t
(** [adam lr] is Adam with bias correction.

    Equivalent to
    [chain [scale_by_adam ~b1 ~b2 ~eps (); scale_by_learning_rate lr]]. *)

val adamw :
  ?b1:float -> ?b2:float -> ?eps:float -> ?weight_decay:float -> Schedule.t -> t
(** [adamw lr] is AdamW with decoupled weight decay.

    Equivalent to
    [chain [scale_by_adam ~b1 ~b2 ~eps (); add_decayed_weights
     ~rate:(Schedule.constant weight_decay) (); scale_by_learning_rate lr]]. *)

val rmsprop : ?decay:float -> ?eps:float -> ?momentum:float -> Schedule.t -> t
(** [rmsprop lr] is RMSprop.

    Equivalent to
    [chain [scale_by_rms ~decay ~eps (); (* trace if momentum > 0 *)
     scale_by_learning_rate lr]]. *)

val adagrad : ?eps:float -> Schedule.t -> t
(** [adagrad lr] is Adagrad.

    Equivalent to [chain [scale_by_adagrad ~eps (); scale_by_learning_rate lr]].
*)

val lamb :
  ?b1:float -> ?b2:float -> ?eps:float -> ?weight_decay:float -> Schedule.t -> t
(** [lamb lr] is LAMB (Layer-wise Adaptive Moments) for large-batch training.

    Equivalent to
    [chain [scale_by_adam ~b1 ~b2 ~eps (); add_decayed_weights
     ~rate:(Schedule.constant weight_decay) (); scale_by_trust_ratio ();
     scale_by_learning_rate lr]]. *)

val lion : ?b1:float -> ?b2:float -> Schedule.t -> t
(** [lion lr] is Lion (Evolved Sign Momentum).

    Equivalent to [chain [scale_by_lion ~b1 ~b2 (); scale_by_learning_rate lr]].
*)

val radam : ?b1:float -> ?b2:float -> ?eps:float -> Schedule.t -> t
(** [radam lr] is Rectified Adam.

    Equivalent to
    [chain [scale_by_radam ~b1 ~b2 ~eps (); scale_by_learning_rate lr]]. *)

val lars :
  ?momentum:float -> ?weight_decay:float -> ?nesterov:bool -> Schedule.t -> t
(** [lars lr] is LARS (Layer-wise Adaptive Rate Scaling) for large-batch SGD
    training.

    Equivalent to
    [chain [trace ~decay:momentum ~nesterov (); add_decayed_weights
     ~rate:(Schedule.constant weight_decay) (); scale_by_trust_ratio ();
     scale_by_learning_rate lr]].

    [momentum] defaults to [0.9]. [weight_decay] defaults to [0.01]. [nesterov]
    defaults to [false]. *)

val adan :
  ?b1:float ->
  ?b2:float ->
  ?b3:float ->
  ?eps:float ->
  ?weight_decay:float ->
  Schedule.t ->
  t
(** [adan lr] is Adan with decoupled weight decay.

    Equivalent to
    [chain [scale_by_adan ~b1 ~b2 ~b3 ~eps (); add_decayed_weights
     ~rate:(Schedule.constant weight_decay) (); scale_by_learning_rate lr]].

    [weight_decay] defaults to [0.02]. *)

val adafactor : ?b2_decay:[ `Constant of float | `Rms ] -> unit -> t
(** [adafactor ?b2_decay ()] is Adafactor with default parameters.

    Equivalent to [chain [scale_by_adafactor ?b2_decay ()]].

    Adafactor includes its own learning rate schedule (inverse root of step) so
    no separate {!scale_by_learning_rate} is needed. *)

(** {1:serialization Serialization} *)

val n_tensors : t -> int
(** [n_tensors tx] is the total number of state tensors across all primitives in
    the chain. *)

val state_to_tensors : ('a, 'b) state -> int * ('a, 'b) Nx.t array
(** [state_to_tensors state] is [(count, tensors)] where [count] is the current
    step count and [tensors] are the internal state tensors (flat array, ordered
    by primitive in the chain). *)

val state_of_tensors : t -> count:int -> ('a, 'b) Nx.t array -> ('a, 'b) state
(** [state_of_tensors tx ~count tensors] reconstructs state from a
    transformation, step count, and previously serialized tensors.

    Raises [Invalid_argument] if [Array.length tensors <> n_tensors tx]. *)
