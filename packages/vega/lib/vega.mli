(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Gradient-descent optimizers.

    Vega is the optimizer layer of the Raven ecosystem. Its optimizers step
    whole parameter structures, any value with an {!Nx.Ptree.t}. Optimizer state
    has the shape of the parameters themselves: each algorithm keeps its
    per-parameter accumulators as values of the user's own structure type, in a
    small record the training loop threads explicitly ({!type:sgd_state},
    {!type:adam_state}). Steps are pure: a step takes a state and returns the
    next one, so a training step is an ordinary function of [(params, state)],
    and checkpointing an optimizer means saving a record of parameter-shaped
    values.

    There is no optimizer object; composition is function application. Transform
    gradients before the step (for example {!clip_by_global_norm}) and derive
    the step's learning rate from the state's step counter with a schedule
    ({!Schedule}). Each state record is a structure itself ({!adam_ptree},
    {!sgd_ptree}), so a whole training step (forward, backward and update) is
    one function of [(params, state)] that one {!Rune.val-jit} call compiles
    into a single program on any device, the state riding it as ordinary
    argument and result leaves:

    {[
    let model = Nx.Ptree.instantiate (module Model)
    let state = Nx.Ptree.pair model (Vega.adam_ptree model)
    let sched = Vega.Schedule.cosine_decay ~init_value:1e-3 ~decay_steps:1000 ()

    let step =
      Rune.jit
        Nx.Ptree.(
          tensor @-> tensor @-> consumes state @@ returns (pair tensor state))
        (fun inputs targets (params, opt) ->
          let loss, grads =
            Rune.value_and_grad model (objective inputs targets) params
          in
          let grads = Vega.clip_by_global_norm model ~max_norm:1.0 grads in
          let params, opt =
            Vega.adamw_step model ~lr:(sched opt.step) opt ~params ~grads
          in
          (loss, (params, opt)))
    ]}

    Hyperparameters that do not change across steps ([b1], [b2], [eps],
    [weight_decay], [max_norm]) are compile-time constants. Everything that does
    (the moments, the step counter, the learning rate) is a tensor leaf or
    derived from one, so the compiled program replays correctly on every call:
    no data transfers, no retracing. The step consumes the state it is given and
    returns the next one beside the loss, which stays readable after later
    calls.

    Every function over parameters takes their structure first, a
    ['p Nx.Ptree.t] built with [Nx.Ptree.instantiate]; see {!section-structures}
    for what it checks.

    The optimizers are {!sgd_step}, {!lars_step}, {!adam_step}, {!adamw_step},
    {!radam_step}, {!lamb_step}, {!rmsprop_step}, {!adagrad_step}, {!adan_step},
    {!lion_step}, {!adafactor_step} and {!lbfgs_step}, each with its [*_init].
*)

(** {1:structures Parameters and states}

    A function over parameters walks its values with their structure [p]. Four
    rules hold:

    - {b One skeleton.} The parameters, the gradients and each part of a state
      that has the parameters' shape visit the same leaves and reports under [p]
      (the same {!Nx.Ptree.visits}) and have equal dtypes leaf by leaf. Every
      step but {!lbfgs_step} raises [Invalid_argument] naming itself, the first
      path at which a value differs from the parameters, and what the value and
      the parameters hold there, for example
      ["Vega.adam_step: the root: length 1 in the gradients, length 2 in the
       parameters"] or
      ["Vega.adam_step: w: float32 in the gradients, float64 in the parameters"].
      A skeleton mismatch raises before any tensor is computed; a dtype mismatch
      raises at its leaf. The other functions that take two values raise as
      {!Nx.Ptree.map2} does.
    - {b Non-parameter leaves.} A structure may carry tensors that are not
      parameters: an {!Nx.Rng.key} threaded through a compiled step, a counter,
      a batch of indices. {!Rune.val-grad} does not differentiate them, and a
      step passes every leaf whose dtype is not a float through unchanged, in
      the parameters and in the state. So one structure serves the objective,
      the gradient and the update.
    - {b States are structures.} {!sgd_ptree}, {!adam_ptree}, {!rmsprop_ptree},
      {!adagrad_ptree}, {!adan_ptree}, {!lion_ptree}, {!adafactor_ptree} and
      {!lbfgs_ptree} are a state's structure over [p]. A state's leaf paths are
      its field name followed by the parameter's path ([mu.blocks.0.w]), and
      [step] for the counter, so a state is named in a compiled step's signature
      and saved with its paths as checkpoint names.
    - {b Precision.} A step computes each leaf at float32, or at float64 for a
      float64 leaf, and returns the parameters and the state at their own
      dtypes, so a float16 or bfloat16 leaf is rounded once, when stored.
      Constants that depend only on hyperparameters are computed on the host in
      float64; scalars of the step counter, once per step at the leaf's compute
      dtype, [1 - b^t] without cancellation. A low-precision state still rounds
      what it stores: a bfloat16 moving average whose decay is below half its
      spacing does not decay. {!lbfgs_step} keeps its scalars at the objective's
      dtype instead. *)

(** {1:schedules Learning-Rate Schedules}

    A schedule maps a step counter to a learning rate; it is a plain function
    from a scalar [int32] step tensor to a scalar [float32] rate tensor.
    Training loops apply a schedule to the state's step counter and pass the
    result as [~lr] — tensor arithmetic, so the same code runs eagerly and
    inside a compiled step. *)

module Schedule = Schedule

(** {1:gradients Gradient Transformations}

    Pure functions on gradient structures, applied between the backward pass and
    the optimizer step. *)

val global_norm : 'p Nx.Ptree.t -> 'p -> float
(** [global_norm p grads] is the L2 norm of all leaves of [grads] taken
    together: [sqrt (sum of every element squared)]. *)

val clip_by_global_norm : 'p Nx.Ptree.t -> max_norm:float -> 'p -> 'p
(** [clip_by_global_norm p ~max_norm grads] scales [grads] so that its
    {!global_norm} does not exceed [max_norm]. Gradients within the bound
    (including all-zero gradients) are returned unchanged; larger ones are
    scaled by [max_norm /. norm], preserving their direction.

    The scale factor is computed in float32 tensor arithmetic and selected with
    {!Nx.where} — no host read — so the transform traces under {!Rune.val-jit}
    on any device and can sit between a jitted backward pass and a jitted
    optimizer step. {!global_norm} remains the float64 host read for reporting.

    Raises [Invalid_argument] if [max_norm <= 0.]. *)

val clip_by_value : 'p Nx.Ptree.t -> max:float -> 'p -> 'p
(** [clip_by_value p ~max grads] clips every gradient element to the interval
    \[[-. max];[max]\].

    Raises [Invalid_argument] if [max <= 0.]. *)

val global_dot :
  'p Nx.Ptree.t -> (float, 'v) Nx.dtype -> 'p -> 'p -> (float, 'v) Nx.t
(** [global_dot p dt a b] is the inner product of [a] and [b] over all their
    float leaves taken together, as a scalar tensor: every leaf's product is
    summed at the leaf's dtype, then cast to [dt] and accumulated. Non-float
    leaves contribute nothing. Tensor arithmetic with no host read, so it traces
    under {!Rune.val-jit}; [dt] sets the precision of the accumulation.

    Raises [Invalid_argument], naming the first path at which they differ, if
    [a] and [b] differ in their paths, reports or dtypes. *)

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
        let sloss, grads = Rune.value_and_grad model objective params in
        let grads = Vega.Loss_scale.unscale model ls grads in
        let finite = Vega.Loss_scale.grads_finite model grads in
        let params' = (* optimizer step on [grads] *) in
        let params =
          Nx.Ptree.map2 model (fun _ p p' -> Nx.where finite p' p) params params'
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
      [good_steps] is [-1] for a {!static} scale. Both are tensors so that a
      compiled step takes them as arguments and returns them ({!ptree}); a
      captured float would be a constant of the program. *)

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

  val unscale : 'p Nx.Ptree.t -> t -> 'p -> 'p
  (** [unscale p ls grads] divides every leaf of [grads] by the current scale,
      at the leaf's dtype. Apply it to the gradients before any gradient
      transformation or optimizer step. *)

  val grads_finite : 'p Nx.Ptree.t -> 'p -> (bool, Nx.bool_elt) Nx.t
  (** [grads_finite p grads] is a scalar boolean tensor: [true] iff every
      element of every leaf of [grads] is finite (no NaN or infinity). Feed it
      to {!adjust} and use it to skip the parameter update of an overflowed step
      (select between updated and previous parameters with {!Nx.where}, as in
      the module preamble — tensor arithmetic, so the step still traces under
      jit). *)

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

  val ptree : t Nx.Ptree.t
  (** [ptree] is the structure of a loss scale. It visits the fixed tensors
      [scale] then [good_steps], at those paths, and reports nothing. *)
end

(** {1:lr Learning Rates}

    A step applies its learning rate as a scalar tensor, cast to each leaf's
    dtype. A constant rate is one call to {!lr}; a scheduled one is a
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
    state carries its counter, so a {!Schedule} applies to [st.step] whichever
    optimizer is stepping. *)

module Sgd_state : Nx.Ptree.S with type 'p t = 'p sgd_state
(** The structure of SGD states. Its [walk] visits [velocity] at the field
    ["velocity"] as a position of the parameter, then [step] at ["step"] as a
    fixed [int32] tensor. Use it with {!Nx.Ptree.Payload} and {!Nx.Ptree.cast};
    transformations take {!sgd_ptree}. *)

val sgd_ptree : 'p Nx.Ptree.t -> 'p sgd_state Nx.Ptree.t
(** [sgd_ptree p] is [Nx.Ptree.nest (module Sgd_state) p], the structure of an
    SGD state over parameters of structure [p]. It visits [velocity.]{e path}
    for each of [p]'s visits, in [p]'s order, then [step]. A compiled step names
    it beside the parameters:

    {[
    let state = Nx.Ptree.pair model (Vega.sgd_ptree model)
    ]} *)

val sgd_init : 'p Nx.Ptree.t -> 'p -> 'p sgd_state
(** [sgd_init p params] is the initial state for optimizing [params]: an
    all-zero velocity of [params]' shape and [step = 0]. *)

val sgd_step :
  'p Nx.Ptree.t ->
  lr:(float, 'b) Nx.t ->
  ?momentum:float ->
  'p sgd_state ->
  params:'p ->
  grads:'p ->
  'p * 'p sgd_state
(** [sgd_step p ~lr st ~params ~grads] is [(params', st')] after one step of
    gradient descent with heavy-ball momentum. Per element:

    {v
    v' = momentum * v + g
    p' = p - lr * v'
    v}

    [lr] is a scalar tensor ({!lr}); [momentum] defaults to [0.], plain gradient
    descent: the velocity is then the last gradient, and the input velocity is
    not read at all. The counter advances by one. The whole step is tensor
    arithmetic over [(params, st)], so it traces under {!Rune.val-jit}.

    Raises [Invalid_argument] if [momentum] is outside \[[0];[1]\), or as
    {!section-structures} states if [grads], or [st.velocity] when [momentum] is
    not [0.], does not have [params]' skeleton. *)

(** {1:lars LARS} *)

val lars_init : 'p Nx.Ptree.t -> 'p -> 'p sgd_state
(** [lars_init] is {!sgd_init}: LARS keeps SGD's momentum velocity. *)

val lars_step :
  'p Nx.Ptree.t ->
  lr:(float, 'b) Nx.t ->
  ?momentum:float ->
  ?weight_decay:float ->
  ?nesterov:bool ->
  'p sgd_state ->
  params:'p ->
  grads:'p ->
  'p * 'p sgd_state
(** [lars_step p ~lr st ~params ~grads] is [(params', st')] after one LARS step
    (You, Gitman and Ginsburg, 2017), for large-batch SGD: each leaf is a layer
    whose step is scaled by the ratio of its weights' norm to its update's norm.
    Per leaf, with [|x|] the L2 norm over the leaf's elements:

    {v
    u  = g + weight_decay * p
    r  = |p| / (|u| + 1e-6), or 1 if |p| = 0 or |u| = 0
    v' = momentum * v + r * u
    p' = p - lr * v'
    v}

    With [nesterov], the last line is [p' = p - lr * (r * u + momentum * v')].
    [momentum] defaults to [0.9], [weight_decay] to [0.01], [nesterov] to
    [false]. The learning rate scales the velocity rather than entering it, as
    in {!sgd_step}, and the paper's trust coefficient is folded into [lr].

    Raises [Invalid_argument] if [momentum] is outside \[[0];[1]\) or
    [weight_decay] is negative, or as {!section-structures} states if [grads] or
    [st.velocity] does not have [params]' skeleton. *)

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

module Adam_state : Nx.Ptree.S with type 'p t = 'p adam_state
(** The structure of Adam states. Its [walk] visits [mu] then [nu] at those
    fields as positions of the parameter, then [step] at ["step"] as a fixed
    [int32] tensor. Transformations take {!adam_ptree}. *)

val adam_ptree : 'p Nx.Ptree.t -> 'p adam_state Nx.Ptree.t
(** [adam_ptree p] is [Nx.Ptree.nest (module Adam_state) p], the structure of an
    Adam state over parameters of structure [p]. It visits [mu.]{e path} for
    each of [p]'s visits, in [p]'s order, then [nu.]{e path} likewise, then
    [step]. *)

val adam_init : 'p Nx.Ptree.t -> 'p -> 'p adam_state
(** [adam_init p params] is the initial state for optimizing [params]: all-zero
    moments and [step = 0]. *)

val adam_step :
  'p Nx.Ptree.t ->
  lr:(float, 'b) Nx.t ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  'p adam_state ->
  params:'p ->
  grads:'p ->
  'p * 'p adam_state
(** [adam_step p ~lr st ~params ~grads] is [(params', st')] after one Adam step
    (Kingma and Ba, 2015). Per element, with [t = st.step + 1]:

    {v
    mu' = b1 * mu + (1 - b1) * g
    nu' = b2 * nu + (1 - b2) * g^2
    d   = (mu' / (1 - b1^t)) / (sqrt (nu' / (1 - b2^t)) + eps)
    p'  = p - lr * d
    v}

    [lr] is a scalar tensor ({!lr}). [b1] defaults to [0.9], [b2] to [0.999],
    [eps] to [1e-8]; they are compile-time constants, safe captures under
    {!Rune.val-jit}. The bias corrections are derived from the state's counter
    in tensor arithmetic (see {!section-structures} for their precision), so the
    whole step traces and the returned state feeds the next call.

    Raises [Invalid_argument] if [b1] or [b2] is outside \[[0];[1]\) or [eps] is
    not positive, or as {!section-structures} states if [grads], [st.mu] or
    [st.nu] does not have [params]' skeleton. *)

val adamw_init : 'p Nx.Ptree.t -> 'p -> 'p adam_state
(** [adamw_init] is {!adam_init}: AdamW shares Adam's state. *)

val adamw_step :
  'p Nx.Ptree.t ->
  lr:(float, 'b) Nx.t ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  ?weight_decay:float ->
  'p adam_state ->
  params:'p ->
  grads:'p ->
  'p * 'p adam_state
(** [adamw_step p ~lr st ~params ~grads] is like {!adam_step} with decoupled
    weight decay (Loshchilov and Hutter, 2019): with [d] Adam's bias-corrected
    direction, the parameter update becomes

    {v p' = p - lr * (d + weight_decay * p) v}

    The decay applies to the parameters directly rather than through the
    adaptive scaling, so its effective strength does not depend on the gradient
    history. [weight_decay] defaults to [0.01]; with [weight_decay = 0.] the
    step is exactly {!adam_step}.

    Raises [Invalid_argument] if [weight_decay] is negative, or as {!adam_step}
    does. *)

(** {1:radam RAdam} *)

val radam_init : 'p Nx.Ptree.t -> 'p -> 'p adam_state
(** [radam_init] is {!adam_init}: RAdam shares Adam's state. *)

val radam_step :
  'p Nx.Ptree.t ->
  lr:(float, 'b) Nx.t ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  'p adam_state ->
  params:'p ->
  grads:'p ->
  'p * 'p adam_state
(** [radam_step p ~lr st ~params ~grads] is [(params', st')] after one rectified
    Adam step (Liu et al., 2020). Early on, too few squared gradients have been
    averaged for Adam's adaptive scaling to be reliable; RAdam takes plain
    momentum steps until the variance of that scaling is bounded, then Adam's
    steps scaled by a rectification factor that rises towards [1]. Per element,
    with [t = st.step + 1] and Adam's [mu'], [nu'] and bias-corrected [mu_hat],
    [nu_hat] ({!adam_step}):

    {v
    rho_inf = 2 / (1 - b2) - 1
    rho     = rho_inf - 2 t b2^t / (1 - b2^t)
    r       = sqrt ((rho - 4) (rho - 2) rho_inf
                    / ((rho_inf - 4) (rho_inf - 2) rho))
    d       = r * mu_hat / (sqrt nu_hat + eps)   if t >= t*
              mu_hat                             otherwise
    p'      = p - lr * d
    v}

    where [t*] is the first step at which [rho] reaches [5]. [t*] depends on
    [b2] alone: it is found on the host in float64 and compared with the integer
    counter, so the switch is exact at any leaf dtype, and the choice between
    the two forms is a tensor [where] that traces under {!Rune.val-jit}.
    Defaults and the counter's role are {!adam_step}'s.

    Raises [Invalid_argument] as {!adam_step} does. *)

(** {1:lamb LAMB} *)

val lamb_init : 'p Nx.Ptree.t -> 'p -> 'p adam_state
(** [lamb_init] is {!adam_init}: LAMB shares Adam's state. *)

val lamb_step :
  'p Nx.Ptree.t ->
  lr:(float, 'b) Nx.t ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  ?weight_decay:float ->
  'p adam_state ->
  params:'p ->
  grads:'p ->
  'p * 'p adam_state
(** [lamb_step p ~lr st ~params ~grads] is [(params', st')] after one LAMB step
    (You et al., 2020), for large-batch training: {!adamw_step}'s update, scaled
    per leaf by LARS's trust ratio ({!lars_step}). With [d] Adam's
    bias-corrected direction:

    {v
    u  = d + weight_decay * p
    r  = |p| / (|u| + 1e-6), or 1 if |p| = 0 or |u| = 0
    p' = p - lr * r * u
    v}

    Defaults are {!adam_step}'s, and [weight_decay] defaults to [0.01].

    Raises [Invalid_argument] as {!adamw_step} does. *)

(** {1:rmsprop RMSprop} *)

type 'p rmsprop_state = { nu : 'p; velocity : 'p; step : Nx.int32_t }
(** The state for {!rmsprop_step}: the moving average of squared gradients and
    the momentum velocity, both with the shape of the parameters, and the number
    of completed steps as a scalar tensor. *)

module Rmsprop_state : Nx.Ptree.S with type 'p t = 'p rmsprop_state
(** The structure of RMSprop states. Its [walk] visits [nu] then [velocity] at
    those fields as positions of the parameter, then [step] at ["step"] as a
    fixed [int32] tensor. Transformations take {!rmsprop_ptree}. *)

val rmsprop_ptree : 'p Nx.Ptree.t -> 'p rmsprop_state Nx.Ptree.t
(** [rmsprop_ptree p] is [Nx.Ptree.nest (module Rmsprop_state) p]. *)

val rmsprop_init : 'p Nx.Ptree.t -> 'p -> 'p rmsprop_state
(** [rmsprop_init p params] is the initial state for optimizing [params]:
    all-zero averages and velocity, and [step = 0]. *)

val rmsprop_step :
  'p Nx.Ptree.t ->
  lr:(float, 'b) Nx.t ->
  ?decay:float ->
  ?eps:float ->
  ?momentum:float ->
  'p rmsprop_state ->
  params:'p ->
  grads:'p ->
  'p * 'p rmsprop_state
(** [rmsprop_step p ~lr st ~params ~grads] is [(params', st')] after one RMSprop
    step (Tieleman and Hinton, 2012). Per element:

    {v
    nu' = decay * nu + (1 - decay) * g^2
    v'  = momentum * v + g / (sqrt nu' + eps)
    p'  = p - lr * v'
    v}

    [decay] defaults to [0.9], [eps] to [1e-8], [momentum] to [0.], where the
    velocity is the last scaled gradient.

    Raises [Invalid_argument] if [decay] or [momentum] is outside \[[0];[1]\) or
    [eps] is not positive, or as {!section-structures} states if [grads],
    [st.nu] or [st.velocity] does not have [params]' skeleton. *)

(** {1:adagrad Adagrad} *)

type 'p adagrad_state = { sum_of_squares : 'p; step : Nx.int32_t }
(** The state for {!adagrad_step}: the sum of all squared gradients so far, with
    the shape of the parameters, and the number of completed steps as a scalar
    tensor. *)

module Adagrad_state : Nx.Ptree.S with type 'p t = 'p adagrad_state
(** The structure of Adagrad states. Its [walk] visits [sum_of_squares] at that
    field as a position of the parameter, then [step] at ["step"] as a fixed
    [int32] tensor. Transformations take {!adagrad_ptree}. *)

val adagrad_ptree : 'p Nx.Ptree.t -> 'p adagrad_state Nx.Ptree.t
(** [adagrad_ptree p] is [Nx.Ptree.nest (module Adagrad_state) p]. *)

val adagrad_init : 'p Nx.Ptree.t -> 'p -> 'p adagrad_state
(** [adagrad_init p params] is the initial state for optimizing [params]: an
    all-zero sum and [step = 0]. *)

val adagrad_step :
  'p Nx.Ptree.t ->
  lr:(float, 'b) Nx.t ->
  ?eps:float ->
  'p adagrad_state ->
  params:'p ->
  grads:'p ->
  'p * 'p adagrad_state
(** [adagrad_step p ~lr st ~params ~grads] is [(params', st')] after one Adagrad
    step (Duchi, Hazan and Singer, 2011). Per element:

    {v
    s' = s + g^2
    p' = p - lr * g / (sqrt s' + eps)
    v}

    [eps] defaults to [1e-8].

    Raises [Invalid_argument] if [eps] is not positive, or as
    {!section-structures} states if [grads] or [st.sum_of_squares] does not have
    [params]' skeleton. *)

(** {1:adan Adan} *)

type 'p adan_state = {
  mu : 'p;  (** Moving average of the gradients. *)
  delta : 'p;  (** Moving average of the differences of successive gradients. *)
  nu : 'p;
      (** Moving average of the squared look-ahead gradients
          [g + b2 * (g - prev_grads)]. *)
  prev_grads : 'p;  (** The gradients of the last step, zero initially. *)
  step : Nx.int32_t;  (** Completed steps, a scalar tensor. *)
}
(** The state for {!adan_step}. Every part but [step] has the shape of the
    parameters. *)

module Adan_state : Nx.Ptree.S with type 'p t = 'p adan_state
(** The structure of Adan states. Its [walk] visits [mu], [delta], [nu] then
    [prev_grads] at those fields as positions of the parameter, then [step] at
    ["step"] as a fixed [int32] tensor. Transformations take {!adan_ptree}. *)

val adan_ptree : 'p Nx.Ptree.t -> 'p adan_state Nx.Ptree.t
(** [adan_ptree p] is [Nx.Ptree.nest (module Adan_state) p]. *)

val adan_init : 'p Nx.Ptree.t -> 'p -> 'p adan_state
(** [adan_init p params] is the initial state for optimizing [params]: every
    part zero and [step = 0]. *)

val adan_step :
  'p Nx.Ptree.t ->
  lr:(float, 'b) Nx.t ->
  ?b1:float ->
  ?b2:float ->
  ?b3:float ->
  ?eps:float ->
  ?weight_decay:float ->
  'p adan_state ->
  params:'p ->
  grads:'p ->
  'p * 'p adan_state
(** [adan_step p ~lr st ~params ~grads] is [(params', st')] after one Adan step
    (Xie et al., 2022), an adaptive Nesterov momentum. Per element, with
    [dg = g - prev_grads]:

    {v
    mu'    = b1 * mu + (1 - b1) * g
    delta' = b2 * delta + (1 - b2) * dg
    nu'    = b3 * nu + (1 - b3) * (g + b2 * dg)^2
    d      = (mu' + b2 * delta') / (sqrt nu' + eps) + weight_decay * p
    p'     = p - lr * d
    v}

    and [prev_grads' = g]. There is no bias correction. [b1] defaults to [0.98],
    [b2] to [0.92], [b3] to [0.99], [eps] to [1e-8], [weight_decay] to [0.02];
    the weight decay is decoupled, as {!adamw_step}'s.

    Raises [Invalid_argument] if [b1], [b2] or [b3] is outside \[[0];[1]\),
    [eps] is not positive or [weight_decay] is negative, or as
    {!section-structures} states if [grads] or a part of [st] other than [step]
    does not have [params]' skeleton. *)

(** {1:lion Lion} *)

type 'p lion_state = { mu : 'p; step : Nx.int32_t }
(** The state for {!lion_step}: the moving average of the gradients, with the
    shape of the parameters, and the number of completed steps as a scalar
    tensor. *)

module Lion_state : Nx.Ptree.S with type 'p t = 'p lion_state
(** The structure of Lion states. Its [walk] visits [mu] at that field as a
    position of the parameter, then [step] at ["step"] as a fixed [int32]
    tensor. Transformations take {!lion_ptree}. *)

val lion_ptree : 'p Nx.Ptree.t -> 'p lion_state Nx.Ptree.t
(** [lion_ptree p] is [Nx.Ptree.nest (module Lion_state) p]. *)

val lion_init : 'p Nx.Ptree.t -> 'p -> 'p lion_state
(** [lion_init p params] is the initial state for optimizing [params]: an
    all-zero average and [step = 0]. *)

val lion_step :
  'p Nx.Ptree.t ->
  lr:(float, 'b) Nx.t ->
  ?b1:float ->
  ?b2:float ->
  'p lion_state ->
  params:'p ->
  grads:'p ->
  'p * 'p lion_state
(** [lion_step p ~lr st ~params ~grads] is [(params', st')] after one Lion step
    (Chen et al., 2023). Every element moves by exactly [lr], in the direction
    of the sign of an interpolation between the average and the gradient:

    {v
    p'  = p - lr * sign (b1 * mu + (1 - b1) * g)
    mu' = b2 * mu + (1 - b2) * g
    v}

    [b1] defaults to [0.9], [b2] to [0.99]. Since every step has the same size,
    Lion wants a smaller rate than Adam, typically 3 to 10 times.

    Raises [Invalid_argument] if [b1] or [b2] is outside \[[0];[1]\), or as
    {!section-structures} states if [grads] or [st.mu] does not have [params]'
    skeleton. *)

(** {1:adafactor Adafactor} *)

type 'p adafactor_state = {
  nu_row : 'p;
      (** For a factored leaf of shape [[...; m; n]], the moving average of the
          squared gradients' means over the last axis, of shape [[...; m; 1]]. A
          scalar zero for other leaves. *)
  nu_col : 'p;
      (** For a factored leaf, the moving average of the squared gradients'
          means over the second-to-last axis, of shape [[...; 1; n]]. A scalar
          zero for other leaves. *)
  nu : 'p;
      (** For a leaf that is not factored, the moving average of the squared
          gradients, of the leaf's shape. A scalar zero for factored leaves. *)
  step : Nx.int32_t;  (** Completed steps, a scalar tensor. *)
}
(** The state for {!adafactor_step}. A leaf of two or more axes is factored: its
    second moment is estimated from a row and a column statistic, in [m + n]
    numbers rather than [m * n]. {!adafactor_init} decides which leaves are
    factored, and the state records it. *)

module Adafactor_state : Nx.Ptree.S with type 'p t = 'p adafactor_state
(** The structure of Adafactor states. Its [walk] visits [nu_row], [nu_col] then
    [nu] at those fields as positions of the parameter, then [step] at ["step"]
    as a fixed [int32] tensor. Transformations take {!adafactor_ptree}. *)

val adafactor_ptree : 'p Nx.Ptree.t -> 'p adafactor_state Nx.Ptree.t
(** [adafactor_ptree p] is [Nx.Ptree.nest (module Adafactor_state) p]. *)

val adafactor_init : 'p Nx.Ptree.t -> ?factored:bool -> 'p -> 'p adafactor_state
(** [adafactor_init p params] is the initial state for optimizing [params]: zero
    statistics and [step = 0]. With [factored] (the default, [true]), leaves of
    two or more axes are factored; without, no leaf is. *)

val adafactor_step :
  'p Nx.Ptree.t ->
  lr:(float, 'b) Nx.t ->
  ?decay_rate:float ->
  ?eps:float ->
  ?clipping_threshold:float ->
  'p adafactor_state ->
  params:'p ->
  grads:'p ->
  'p * 'p adafactor_state
(** [adafactor_step p ~lr st ~params ~grads] is [(params', st')] after one
    Adafactor step (Shazeer and Stern, 2018): an adaptive step that keeps no
    first moment and, for a factored leaf of [m] rows and [n] columns, estimates
    the second moment from [m + n] numbers rather than [m * n]. With
    [t = st.step + 1], the averages decay by [b = 1 - t^(-decay_rate)], which
    rises towards [1]. Per element of a leaf that is not factored:

    {v
    nu' = b * nu + (1 - b) * g^2
    u   = g / (sqrt nu' + eps)
    v}

    and of a factored leaf, with [mean_r] and [mean_c] the means over the last
    and the second-to-last axis:

    {v
    nu_row' = b * nu_row + (1 - b) * mean_r (g^2)
    nu_col' = b * nu_col + (1 - b) * mean_c (g^2)
    u       = g / (sqrt (nu_row' * nu_col' / (mean_c nu_row' + eps)) + eps)
    v}

    The update is then clipped so that its root mean square over the leaf is at
    most [clipping_threshold]:

    {v p' = p - lr * u * min (1, clipping_threshold / rms u) v}

    [decay_rate] defaults to [0.8], [eps] to [1e-30], [clipping_threshold] to
    [1.0]; [infinity] disables the clipping. Adafactor is usually run with a
    rate that decays with the counter, which the step's [lr] derives in tensor
    arithmetic, here [1e-3 / sqrt t]:

    {[
    let t = Nx.cast Nx.float32 (Nx.add_s st.step 1l) in
    Vega.adafactor_step p ~lr:(Nx.mul_s (Nx.rsqrt t) 1e-3) st ~params ~grads
    ]}

    Raises [Invalid_argument] if [decay_rate], [eps] or [clipping_threshold] is
    not positive, or as {!section-structures} states if [grads] or a part of
    [st] other than [step] does not have [params]' skeleton. *)

(** {1:lbfgs L-BFGS}

    Limited-memory BFGS (Liu and Nocedal, 1989): a quasi-Newton method that
    builds its search direction from the last [history] pairs of parameter and
    gradient differences, then moves along it — by a line search when the step
    is left to choose its length, or by a fixed rate. It is the method of choice
    for deterministic objectives: full-batch fits, maximum a posteriori
    estimates, calibration, the second stage of training a physics-informed
    network. Minibatch noise defeats it.

    Unlike {!sgd_step} and {!adam_step}, a step here evaluates the objective
    itself, because a line search needs its value at trial points. The objective
    returns the value and the gradient at once — the type {!Rune.value_and_grad}
    yields — so an analytic gradient serves as well as a differentiated one:

    {[
    let objective = Rune.value_and_grad model loss in
    let st, status = Vega.minimize model objective params in
    st.params
    ]}

    The state carries the current point with its value and gradient, so an
    iteration costs the line search's trials and nothing more, and the history
    at a fixed memory size, so it is a parameter tree of static shape
    ({!lbfgs_ptree}). Every scalar the method keeps — the value, the curvature
    weights, the inner products of the two-loop recursion — is at the
    objective's dtype: a [float64] objective drives a [float64] line search.

    Under {!Rune.val-jit} only the fixed-rate form traces: a line search reads
    values on the host to decide its next trial, so it runs eagerly. *)

type ('p, 'v) lbfgs_state = {
  params : 'p;  (** The current point. *)
  value : (float, 'v) Nx.t;  (** The objective at [params], a scalar. *)
  grads : 'p;  (** The gradient at [params]. *)
  s : 'p;
      (** The last parameter differences, newest first, stacked along a new
          leading axis of length [history] on every leaf. *)
  y : 'p;  (** The matching gradient differences, laid out like [s]. *)
  rho : (float, 'v) Nx.t;
      (** [1 / (y . s)] per pair, shape [[history]]. [0] marks an empty slot, or
          a pair dropped for not having positive curvature; such slots do not
          enter the direction. *)
  step : Nx.int32_t;  (** Completed steps, a scalar tensor. *)
}

val lbfgs_ptree : 'p Nx.Ptree.t -> ('p, 'v) lbfgs_state Nx.Ptree.t
(** [lbfgs_ptree p] is the structure of an L-BFGS state over parameters of
    structure [p], for an objective of element type ['v]. It visits
    [params.]{e path} for each of [p]'s visits, [value], [grads.]{e path},
    [s.]{e path}, [y.]{e path}, [rho], then [step]. The state has no module of
    its own, since {!Nx.Ptree.S} has one type parameter and the state two. *)

val lbfgs_init :
  'p Nx.Ptree.t ->
  ?history:int ->
  ('p -> (float, 'v) Nx.t * 'p) ->
  'p ->
  ('p, 'v) lbfgs_state
(** [lbfgs_init p f params] is the initial state for minimizing [f] from
    [params]: it evaluates [f params] once and holds an empty history of
    [history] pairs (default [10]).

    Raises [Invalid_argument] if [history < 1]. *)

val lbfgs_step :
  'p Nx.Ptree.t ->
  ?lr:(float, 'b) Nx.t ->
  ?max_linesearch_steps:int ->
  ('p -> (float, 'v) Nx.t * 'p) ->
  ('p, 'v) lbfgs_state ->
  ('p, 'v) lbfgs_state
(** [lbfgs_step p f st] is the state after one L-BFGS step. The direction [d] is
    the two-loop recursion of Nocedal (1980) over the stored pairs applied to
    the negated gradient, with the initial inverse Hessian scaled by
    [(s . y) / (y . y)] of the newest pair; then the step moves to [p + a * d],
    evaluates [f] there, and pushes the new pair, dropping the oldest. A pair
    whose curvature [y . s] is not positive is kept out of the direction (its
    weight is [0]).

    Without [~lr], [a] satisfies the strong Wolfe conditions ([c1 = 1e-4],
    [c2 = 0.9]), found by bracketing from [a = 1] and zooming, with at most
    [max_linesearch_steps] (default [20]) evaluations of [f]. If the budget runs
    out, the step takes the lowest trial that decreased the value; if none did —
    the line search failed — it returns [st] unchanged, counter included.

    With [~lr], [a] is that scalar tensor and [f] is evaluated exactly once: the
    whole step is tensor arithmetic over [st], and traces under {!Rune.val-jit}.
    This is the form for training loops, preconditioning a fixed rate rather
    than searching a length.

    Raises [Invalid_argument] if [max_linesearch_steps < 1], or, naming the
    first differing path as {!Nx.Ptree.map2} does, if a gradient [f] returns
    does not have [st.params]' skeleton. *)

type status =
  | Converged  (** A tolerance of {!minimize} was met. *)
  | Max_iter_reached  (** The iteration budget ran out first. *)
  | Line_search_failed
      (** A step found no decrease along its direction: the point is at the
          precision limit of [f], or the gradient is inconsistent with it. *)

val minimize :
  'p Nx.Ptree.t ->
  ?history:int ->
  ?max_iter:int ->
  ?gtol:float ->
  ?ftol:float ->
  ?max_linesearch_steps:int ->
  ('p -> (float, 'v) Nx.t * 'p) ->
  'p ->
  ('p, 'v) lbfgs_state * status
(** [minimize p f params] runs {!lbfgs_step} from [lbfgs_init p f params] until
    a tolerance is met and returns the final state — its [params], [value] and
    [grads] are the result and its [step] the number of iterations — with the
    reason it stopped. It stops with {!Converged} when every gradient component
    is at most [gtol] in absolute value (default [1e-5]), checked before each
    step, or when a step decreases the value by no more than [ftol] relative to
    [max (|f|, |f'|, 1.)] (default [1e-9]); with {!Max_iter_reached} after
    [max_iter] steps (default [1000]); with {!Line_search_failed} when a step
    makes no progress. [history] and [max_linesearch_steps] are those of
    {!lbfgs_init} and {!lbfgs_step}.

    Raises [Invalid_argument] if [history < 1] or [max_iter < 0], if [gtol] or
    [ftol] is negative, or as {!lbfgs_step} does. *)
