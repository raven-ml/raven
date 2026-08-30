(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Gradient-descent optimizers.

    Vega is the optimizer layer of the Raven ecosystem. Its primary surface is
    {e structural}: optimizers step whole parameter structures — any type
    implementing {!Nx.Ptree.S}. Optimizer state has the shape of the parameters
    themselves: each algorithm keeps its per-parameter accumulators as values of
    the user's own structure type, in a small record the training loop threads
    explicitly ({!type:sgd_state}, {!type:adam_state}). Steps are pure
    traversals — a step consumes a state and returns the next one — so a
    training step is an ordinary function of [(params, state)], and
    checkpointing an optimizer means saving a record of parameter-shaped values.

    There is no optimizer object; composition is function application. Transform
    gradients before the step (for example {!clip_by_global_norm}) and derive
    the step's learning rate from the state's step counter with a schedule
    ({!Schedule}). Because optimizer state is itself a parameter tree
    ({!Adam_state}, {!Sgd_state}), a whole training step — forward, backward and
    update — is an ordinary function of [(params, state)] that threads both
    through one {!Rune.val-jit} call, so the step compiles into a single program
    on any device and the state rides it as ordinary input and output leaves:

    {[
    module Opt = Vega.Adam_state (Model)

    module Step_in = struct
      type t = {
        params : Model.t;
        opt : Opt.t;
        inputs : Nx.float32_t;
        targets : (int32, Nx.int32_elt) Nx.t;
      }

      (* map/map2/iter: one-line delegations to [Model] and [Opt] over the
         fields — or [@@deriving ptree] with ppx_ptree. [Step_out] carries
         [params], [opt] and the loss the same way. *)
    end

    let sched = Vega.Schedule.cosine_decay ~init_value:1e-3 ~decay_steps:1000 ()

    let train_step { Step_in.params; opt; inputs; targets } =
      let loss, grads =
        Rune.value_and_grad model (objective inputs targets) params
      in
      let grads = Vega.clip_by_global_norm model ~max_norm:1.0 grads in
      let params, opt =
        Vega.adamw_step model ~lr:(sched opt.step) opt ~params ~grads
      in
      { Step_out.params; opt; loss }

    (* ~donate:true hands the previous generation's device buffers back to the
       allocator once the call completes; the loop never reads the pre-step
       state, so they are safe to release. *)
    let step =
      Rune.jit2 ~donate:true (module Step_in) (module Step_out) train_step
    ]}

    Hyperparameters that do not change across steps ([b1], [b2], [eps],
    [weight_decay], [max_norm]) are compile-time constants. Everything that does
    — the moments, the step counter, the learning rate — is a tensor leaf or
    derived from one, so the compiled program replays correctly on every call:
    no data transfers, no retracing. [~donate:true] keeps the state-to-state
    loop at about two generations of device buffers instead of one per call
    awaiting collection; it consumes the handles it frees — reading the pre-step
    state after the call raises — so leave it off while a loop still inspects
    the state it feeds in. On the CPU device it changes nothing: outputs are
    host tensors there. Steps are pure traversals — a step consumes a state and
    returns the next one — so checkpointing an optimizer means saving a record
    of parameter-shaped values plus a step counter leaf.

    {b Non-parameter leaves.} A structure may carry leaves that are not
    parameters — an {!Nx.Rng.key} threaded through a compiled step, a counter, a
    batch of indices. {!Rune.val-grad} does not differentiate them, and the
    structural optimizers here do not update them: every step passes a non-float
    leaf through unchanged. So one structure can serve the objective, the
    gradient and the update without splitting the values that must reach a
    compiled step from the values being trained. *)

(** {1:schedules Learning-Rate Schedules}

    A schedule maps a step counter to a learning rate; it is a plain function
    from a scalar [int32] step tensor to a scalar [float32] rate tensor, shared
    by both tiers. Structural training loops apply a schedule to the state's
    step counter and pass the result as [~lr] — tensor arithmetic, so the same
    code runs eagerly and inside a compiled step; the per-tensor
    {!scale_by_learning_rate} and {!scale_by_schedule} evaluate a schedule at
    the chain's own update count. *)

module Schedule = Schedule

(** {1:gradients Gradient Transformations}

    Pure functions on gradient structures, applied between the backward pass and
    the optimizer step. *)

val global_norm : (module Nx.Ptree.S with type t = 'p) -> 'p -> float
(** [global_norm (module P) grads] is the L2 norm of all leaves of [grads] taken
    together: [sqrt (sum of every element squared)]. *)

val clip_by_global_norm :
  (module Nx.Ptree.S with type t = 'p) -> max_norm:float -> 'p -> 'p
(** [clip_by_global_norm (module P) ~max_norm grads] scales [grads] so that its
    {!global_norm} does not exceed [max_norm]. Gradients within the bound
    (including all-zero gradients) are returned unchanged; larger ones are
    scaled by [max_norm /. norm], preserving their direction.

    The scale factor is computed in float32 tensor arithmetic and selected with
    {!Nx.where} — no host read — so the transform traces under {!Rune.val-jit}
    on any device and can sit between a jitted backward pass and a jitted
    optimizer step. {!global_norm} remains the float64 host read for
    reporting.

    Raises [Invalid_argument] if [max_norm <= 0.]. *)

val clip_by_value :
  (module Nx.Ptree.S with type t = 'p) -> max:float -> 'p -> 'p
(** [clip_by_value (module P) ~max grads] clips every gradient element to the
    interval \[[-. max];[max]\].

    Raises [Invalid_argument] if [max <= 0.]. *)

(** {1:loss_scaling Loss Scaling}

    Float16 gradients underflow: activations and gradients that fit float16
    still produce per-element gradient contributions below [2^-24], which round
    to zero. Loss scaling multiplies the loss by a large factor before the
    backward pass — scaling every gradient with it — and divides the gradients
    back down before the optimizer step. A {!Loss_scale.dynamic} scale also
    adapts itself: overflowed steps (non-finite gradients) are skipped and the
    scale backs off; long runs of finite steps grow it back.

    {[
      let step (params, ls) =
        let objective p = Vega.Loss_scale.scale ls (loss p) in
        let sloss, grads = value_and_grad (module Model) objective params in
        let grads = Vega.Loss_scale.unscale (module Model) ls grads in
        let finite = Vega.Loss_scale.grads_finite (module Model) grads in
        let params' = (* optimizer step on [grads] *) in
        let params =
          Model.map2 (fun p p' -> Nx.where finite p' p) params params'
        in
        ((params, Vega.Loss_scale.adjust ls ~finite), sloss)
    ]}

    Bfloat16 shares float32's exponent range and needs none of this — loss
    scaling is for float16 training. *)

(** Loss scales for float16 training, after JAX's [jmp]. *)
module Loss_scale : sig
  type t = { scale : Nx.float32_t; good_steps : Nx.int32_t }
  (** The type for loss scales: the current scale factor and the number of
      consecutive finite steps since it last changed, both scalar tensors.
      Tensors, not floats — threaded through a {!Rune.jit2} (or pmap) step as
      ordinary input and output leaves, the state updates across compiled calls,
      whereas a captured float would be burned into the trace as a constant.
      [good_steps] is [-1] for a {!static} scale. *)

  val static : float -> t
  (** [static s] is the fixed scale [s]: {!adjust} returns it unchanged.
      [static 1.0] makes the loss-scaling plumbing the identity.

      Raises [Invalid_argument] if [s] is not positive. *)

  val dynamic : ?init:float -> unit -> t
  (** [dynamic ()] is a fresh adaptive scale, adjusted by {!adjust}. [init]
      defaults to [32768.] ([2^15]).

      Raises [Invalid_argument] if [init] is not positive. *)

  val scale : t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
  (** [scale ls x] is [x] times the current scale, at [x]'s dtype. Apply it to
      the loss, inside the differentiated objective. *)

  val unscale : (module Nx.Ptree.S with type t = 'p) -> t -> 'p -> 'p
  (** [unscale (module P) ls grads] divides every leaf of [grads] by the current
      scale, at the leaf's dtype. Apply it to the gradients before any gradient
      transformation or optimizer step. *)

  val grads_finite :
    (module Nx.Ptree.S with type t = 'p) -> 'p -> (bool, Nx.bool_elt) Nx.t
  (** [grads_finite (module P) grads] is a scalar boolean tensor: [true] iff
      every element of every leaf of [grads] is finite (no NaN or infinity).
      Feed it to {!adjust} and use it to skip the parameter update of an
      overflowed step (select between updated and previous parameters with
      {!Nx.where}, as in the module preamble — tensor arithmetic, so the step
      still traces under jit). *)

  val adjust :
    ?growth_interval:int ->
    ?growth_factor:float ->
    ?backoff_factor:float ->
    t ->
    finite:(bool, Nx.bool_elt) Nx.t ->
    t
  (** [adjust ls ~finite] is the scale for the next step. For a {!dynamic}
      scale: if [finite] is [false] the scale is multiplied by [backoff_factor]
      (default [0.5]) and the finite-step counter resets; if [finite] is [true]
      the counter advances, and on reaching [growth_interval] (default [2000])
      the scale is multiplied by [growth_factor] (default [2.]) and the counter
      resets. For a {!static} scale, [adjust] is the identity. Pure [Nx.where]
      arithmetic on the state tensors — safe inside a jitted step.

      Raises [Invalid_argument] if [growth_interval], [growth_factor] or
      [backoff_factor] is not positive. *)

  (** {2:traversals Traversals}

      Plain traversals over the two state tensors, in the order [scale] then
      [good_steps]; with them a training step's input and output structures can
      carry the loss scale as leaves. They satisfy the {!Nx.Ptree.S} contract.
  *)

  val map : ('a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) -> t -> t
  (** [map f ls] is [ls] with [f] applied to both state tensors. *)

  val map2 :
    ('a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) -> t -> t -> t
  (** [map2 f ls ls'] combines [ls] and [ls'] leafwise with [f]. *)

  val iter : ('a 'b. ('a, 'b) Nx.t -> unit) -> t -> unit
  (** [iter f ls] applies [f] to both state tensors. *)
end

(** {1:lr Learning Rates}

    A structural step applies its learning rate as a scalar tensor, cast to each
    leaf's dtype. A constant rate is one call to {!lr}; a scheduled one is a
    {!Schedule} applied to the state's step counter — tensor arithmetic either
    way, which is what makes the rate track correctly under {!Rune.val-jit}. *)

val lr : float -> Nx.float32_t
(** [lr v] is the learning rate [v] as a scalar tensor, the form the step
    functions' [~lr] argument takes. The value is cast to each leaf's dtype when
    the step applies it, so one value serves any parameter dtype: [Vega.lr 1e-3]
    is exactly [Nx.scalar Nx.float32 1e-3]. *)

(** {1:sgd Stochastic Gradient Descent} *)

type 'p sgd_state = { velocity : 'p; step : Nx.int32_t }
(** The state for {!sgd_step}: the momentum velocity, with the shape of the
    parameters, and the number of completed steps as a scalar tensor. Every
    structural state carries its counter, so a {!Schedule} applies to
    [st.step] whichever optimizer is stepping. *)

(** [Sgd_state (P)] is the state over the parameter tree [P] as a parameter
    tree itself: its [t] is [P.t sgd_state] and its traversals walk the state's
    leaves through [P]. Bind it once per model and embed it in a jitted step's
    input/output records, whose traversals delegate to it field by field
    (by hand, or with [ppx_ptree]'s [@@deriving ptree]):

    {[
      module Opt = Vega.Sgd_state (Model)

      module Step_in = struct
        type t = { params : Model.t; opt : Opt.t; x : Nx.float32_t }

        let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) s =
          { params = Model.map f s.params; opt = Opt.map f s.opt; x = f s.x }

        (* map2 and iter: the same one-liners. *)
      end
    ]}

    The resulting leaf order — every leaf of [velocity], in [P]'s order, then
    [step] — is part of a compiled step's leaf signature and is fixed for
    good. *)
module Sgd_state (P : Nx.Ptree.S) : Nx.Ptree.S with type t = P.t sgd_state

val sgd_init : (module Nx.Ptree.S with type t = 'p) -> 'p -> 'p sgd_state
(** [sgd_init (module P) params] is the initial state for optimizing [params]:
    an all-zero velocity of [params]' shape and [step = 0]. *)

val sgd_step :
  (module Nx.Ptree.S with type t = 'p) ->
  lr:(float, 'b) Nx.t ->
  ?momentum:float ->
  'p sgd_state ->
  params:'p ->
  grads:'p ->
  'p * 'p sgd_state
(** [sgd_step (module P) ~lr st ~params ~grads] is [(params', st')] after one
    step of gradient descent with heavy-ball momentum. Per element:

    {v
    v' = momentum * v + g
    p' = p - lr * v'
    v}

    [lr] is a scalar tensor ({!lr}); [momentum] defaults to [0.], plain gradient
    descent: the velocity is then the last gradient, and the input velocity is
    not read at all. The counter advances by one. The whole step is tensor
    arithmetic over [(params, st)] — it traces under {!Rune.val-jit}. *)

(** {1:adam Adam and AdamW} *)

type 'p adam_state = {
  mu : 'p;
      (** Exponential moving average of gradients (biased; the correction
          applies when computing the update), with the parameters' shape. *)
  nu : 'p;
      (** Exponential moving average of squared gradients, with the parameters'
          shape. *)
  step : Nx.int32_t;
      (** Completed steps, a scalar tensor — not a host [int], which would be
          burned into a compiled trace as a constant and replayed stale. The
          steps derive their bias corrections from it, and it is the counter
          schedules take. *)
}

(** [Adam_state (P)] is the state over the parameter tree [P] as a parameter
    tree itself — see {!Sgd_state}. The leaf order — every leaf of [mu] in
    [P]'s order, every leaf of [nu], then [step] — is part of a compiled step's
    leaf signature and is fixed for good. *)
module Adam_state (P : Nx.Ptree.S) : Nx.Ptree.S with type t = P.t adam_state

val adam_init : (module Nx.Ptree.S with type t = 'p) -> 'p -> 'p adam_state
(** [adam_init (module P) params] is the initial state for optimizing [params]:
    all-zero moments and [step = 0]. *)

val adam_step :
  (module Nx.Ptree.S with type t = 'p) ->
  lr:(float, 'b) Nx.t ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  'p adam_state ->
  params:'p ->
  grads:'p ->
  'p * 'p adam_state
(** [adam_step (module P) ~lr st ~params ~grads] is [(params', st')] after one
    Adam step (Kingma and Ba, 2015). Per element, with [t = st.step + 1]:

    {v
    mu' = b1 * mu + (1 - b1) * g
    nu' = b2 * nu + (1 - b2) * g^2
    d   = (mu' / (1 - b1^t)) / (sqrt (nu' / (1 - b2^t)) + eps)
    p'  = p - lr * d
    v}

    [lr] is a scalar tensor ({!lr}). [b1] defaults to [0.9], [b2] to [0.999],
    [eps] to [1e-8]; they are compile-time constants, safe captures under
    {!Rune.val-jit}. The bias corrections are derived from the state's counter
    per leaf, at the leaf's dtype, in tensor arithmetic — so the whole step
    traces, and the returned state feeds the next call. *)

val adamw_init : (module Nx.Ptree.S with type t = 'p) -> 'p -> 'p adam_state
(** [adamw_init] is {!adam_init}: AdamW shares Adam's state. *)

val adamw_step :
  (module Nx.Ptree.S with type t = 'p) ->
  lr:(float, 'b) Nx.t ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  ?weight_decay:float ->
  'p adam_state ->
  params:'p ->
  grads:'p ->
  'p * 'p adam_state
(** [adamw_step (module P) ~lr st ~params ~grads] is like {!adam_step} with
    decoupled weight decay (Loshchilov and Hutter, 2019): with [d] Adam's
    bias-corrected direction, the parameter update becomes

    {v p' = p - lr * (d + weight_decay * p) v}

    The decay applies to the parameters directly rather than through the
    adaptive scaling, so its effective strength does not depend on the gradient
    history. [weight_decay] defaults to [0.01]; with [weight_decay = 0.] the
    step is exactly {!adam_step}. *)

(** {1:chains Per-Tensor Transformation Chains}

    An Optax-style tier below the structural API. A {!type:t} is a composable
    gradient transformation on a single tensor: it takes updates (gradients) and
    returns modified updates. Primitives are chained to build optimizers:

    {[
    let tx =
      Vega.chain
        [
          Vega.scale_by_adam ();
          Vega.add_decayed_weights ~rate:(Vega.Schedule.constant 0.01) ();
          Vega.scale_by_learning_rate lr;
        ]
    ]}

    Common optimizers are provided as aliases: {!adam}, {!sgd}, {!adamw}, etc.
    The core abstraction is [t]; the per-parameter {!type:state} is fully
    self-contained — it tracks moments, step count, and the update rule — and
    serializes via {!state_to_tensors}. *)

(** {2:types Types} *)

type t
(** A composable gradient transformation. Constructed via primitives like
    {!scale_by_adam}, {!trace}, etc., and composed via {!chain}. *)

type ('a, 'b) state
(** Per-parameter optimizer state. Typed to match the parameter tensor. Tracks
    moments, step count, and the transformation chain. Created via {!init},
    advanced via {!update} or {!step}. *)

(** {2:core Core} *)

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

(** {2:scaling Scaling Transforms} *)

val scale : float -> t
(** [scale s] multiplies updates by [s]. Stateless. *)

val scale_by_schedule : Schedule.t -> t
(** [scale_by_schedule f] multiplies updates by [f step]. *)

val scale_by_learning_rate : Schedule.t -> t
(** [scale_by_learning_rate lr] multiplies updates by [-lr step]. Negates the
    learning rate so that {!apply_updates} performs gradient descent. *)

(** {2:adaptive Adaptive Scaling Transforms} *)

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

(** {2:accumulation Accumulation Transforms} *)

val trace : ?decay:float -> ?nesterov:bool -> unit -> t
(** [trace ?decay ?nesterov ()] accumulates a trace (momentum) of updates.

    [decay] defaults to [0.9]. [nesterov] defaults to [false].

    State: 1 tensor (trace/velocity). *)

(** {2:regularization Regularization Transforms} *)

val add_decayed_weights : ?rate:Schedule.t -> unit -> t
(** [add_decayed_weights ?rate ()] adds [rate step * param] to updates. When
    placed before {!scale_by_learning_rate}, this implements decoupled weight
    decay.

    [rate] defaults to [Schedule.constant 0.01].

    State: 0 tensors. *)

(** {2:clipping Clipping Transforms} *)

val clip : float -> t
(** [clip delta] clips updates element-wise to [[-delta, +delta]] (Optax's
    [clip]). The structural counterpart is {!clip_by_value}.

    State: 0 tensors. *)

val clip_by_norm : float -> t
(** [clip_by_norm max_norm] rescales updates so their L2 norm does not exceed
    [max_norm]. Returns updates unchanged if the norm is already within bounds.

    State: 0 tensors. *)

(** {2:gradient_processing Gradient Processing} *)

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

(** {2:robustness Robustness} *)

val apply_if_finite : t -> t
(** [apply_if_finite tx] wraps [tx] so that if any update produced by [tx]
    contains non-finite values (NaN or Inf), the update is skipped: zero updates
    are returned and the inner state is not advanced.

    State: inner state + 1 tensor (count of consecutive non-finite steps). *)

(** {2:aliases Optimizer Aliases} *)

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

(** {2:serialization Serialization} *)

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
