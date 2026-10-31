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
    one function of [(params, state)] that one [Rune.jit] call compiles into a
    single program on any device, the state riding it as ordinary argument and
    result leaves:

    {[
    let model = Nx.Ptree.instantiate (module Model)
    let state = Nx.Ptree.pair model (Vega.adam_ptree model)
    let sched = Vega.Schedule.cosine_decay ~init_value:1e-3 ~decay_steps:1000 ()

    let step =
      Rune.jit
        Nx.Ptree.(tensor @-> tensor @-> state @-> returns (pair tensor state))
        (fun inputs targets (params, opt) ->
          let loss, grads =
            Rune.value_and_grad model (objective inputs targets) params
          in
          let grads = Vega.clip_by_global_norm model ~max_norm:1.0 grads in
          let params, opt =
            Vega.adamw_step model ~lr:sched opt ~params ~grads
          in
          (loss, (params, opt)))

    let loss, s = step inputs targets (Nx.Ptree.donate state s)
    ]}

    Hyperparameters that do not change across steps ([b1], [b2], [eps],
    [weight_decay], [max_norm]) are compile-time constants. Everything that does
    (the moments, the step counter, the learning rate) is a tensor leaf or
    derived from one, so the compiled program replays correctly on every call:
    no data transfers, no retracing. A caller that donates the state
    ({!Nx.Ptree.donate}) lets the step write the next one into its memory.

    Every function over parameters takes their structure first, a
    [('p, 'd) Nx.Ptree.t] built with [Nx.Ptree.instantiate], ['d] the set the
    parameters lie on; see {!section-structures} for what it checks.

    The optimizers are {!sgd_step}, {!lars_step}, {!adam_step}, {!adamw_step},
    {!radam_step}, {!lamb_step}, {!rmsprop_step}, {!adagrad_step}, {!adan_step},
    {!lion_step}, {!adafactor_step} and {!lbfgs_step}, each with its [*_init].
*)

(** {1:structures Parameters and states}

    A function over parameters walks its values with their structure [p]. Four
    rules hold:

    - {b One skeleton.} The parameters, the gradients and each part of a state
      that has the parameters' shape visit the same leaves and reports under [p]
      (the same {!Nx.Ptree.val-visits}) and have equal dtypes leaf by leaf.
      Every step raises [Invalid_argument] naming itself, the first path at
      which a value differs from the parameters, and what the value and the
      parameters hold there, for example
      ["Vega.adam_step: the root: length 1 in the gradients, length 2 in the
       parameters"] or
      ["Vega.adam_step: w: float32 in the gradients, float64 in the parameters"].
      A skeleton mismatch raises before any tensor is computed; a dtype mismatch
      raises at its leaf. The other functions that take two values raise as
      {!Nx.Ptree.map2} does.
    - {b Non-parameter leaves.} A structure may carry tensors that are not
      parameters: an {!Nx.Rng.key} threaded through a compiled step, a counter,
      a batch of indices. [Rune.grad] does not differentiate them, and a step
      passes every leaf whose dtype is neither a float nor a complex one
      through unchanged, in the parameters and in the state. So one structure
      serves the objective, the gradient and the update.
    - {b Complex leaves.} A complex parameter is two real coordinates per
      element, its real and imaginary parts, and a step moves them as it moves
      real parameters, each part with its own state: Adam's second moment of
      a complex leaf holds the real part's average in its real part and the
      imaginary part's in its imaginary part. The coordinates' gradient is the
      conjugate of the [Rune.grad] of a real objective, [∂f/∂x - i ∂f/∂y] for
      [z = x + iy], so a step descends along that conjugate: [sgd_step] moves
      [z] to [z - lr * conj g], and its velocity holds [conj g]. A norm over
      a leaf ({!lars_step}, {!lamb_step}, {!adafactor_step}'s clipping) sums
      both parts, and {!adafactor_step} factors each part as a leaf of the
      complex leaf's shape. The parts compute at float32 for complex64 and at
      float64 for complex128.
    - {b States are structures.} {!sgd_ptree}, {!adam_ptree}, {!rmsprop_ptree},
      {!adagrad_ptree}, {!adan_ptree}, {!lion_ptree}, {!adafactor_ptree} and
      {!lbfgs_ptree} are a state's structure over [p]. A state's leaf paths are
      its field name followed by the parameter's path ([mu.blocks.0.w]), and
      [step] for the counter, so a state is named in a compiled step's signature
      and saved with its paths as the names of its tensors.
    - {b Precision.} A step computes each leaf at float32, or at float64 for a
      float64 or complex128 leaf, and returns the parameters and the state at their own
      dtypes, so a float16 or bfloat16 leaf is rounded once, when stored.
      Constants that depend only on hyperparameters are computed on the host in
      float64; scalars of the step counter, once per step at the leaf's compute
      dtype, [1 - b^t] without cancellation. A low-precision state still rounds
      what it stores: a bfloat16 moving average whose decay is below half its
      spacing does not decay.
    - {b Placement.} A state lies with the parameters: its parts beside their
      leaves, its counter whole on each device of the parameters' set.
      Every [*_init] raises [Invalid_argument] naming itself if the parameters
      hold no array, which leaves no placement for the counter. *)

(** {1:schedules Learning-Rate Schedules}

    A schedule maps a step counter to a learning rate; it is a plain function
    from a scalar [int32] step tensor to a scalar [float32] rate tensor beside
    it. A step takes a schedule as [~lr] and applies it to its state's counter
    — tensor arithmetic, so the same code runs eagerly and inside a compiled
    step. *)

module Schedule = Schedule

(** {1:gradients Gradient Transformations}

    Pure functions on gradient structures, applied between the backward pass and
    the optimizer step. *)

val clip_by_global_norm : ('p, 'd) Nx.Ptree.t -> max_norm:float -> 'p -> 'p
(** [clip_by_global_norm p ~max_norm grads] scales [grads] so that their norm,
    {!Nx.Ptree.norm} at float32, does not exceed [max_norm]. Gradients within
    the bound (including all-zero gradients and a structure of no array) are
    returned unchanged; larger ones are scaled by [max_norm /. norm],
    preserving their direction. Leaves that are neither floats nor complex are
    returned unchanged.

    The scale factor is computed in float32 tensor arithmetic and selected with
    {!Nx.where} — no host read — so the transform traces under [Rune.jit] on any
    device and can sit between a jitted backward pass and a jitted optimizer
    step.

    Raises [Invalid_argument] if [max_norm <= 0.]. *)

val clip_by_value : ('p, 'd) Nx.Ptree.t -> max:float -> 'p -> 'p
(** [clip_by_value p ~max grads] clips every gradient element to the interval
    \[[-. max];[max]\], each part of a complex one.

    Raises [Invalid_argument] if [max <= 0.]. *)

(** {1:loss_scaling Loss Scaling}

    Float16 gradients underflow: activations and gradients that fit float16
    still produce per-element gradient contributions below [2^-24], which round
    to zero. Loss scaling multiplies the loss by a large factor before the
    backward pass, scaling every gradient with it, and divides the gradients
    back down in the optimizer step. A {!Loss_scale.dynamic} scale also adapts
    itself: a step whose gradients overflowed (hold a NaN or an infinity) is
    skipped and the scale backs off; long runs of finite steps grow it back.

    {!Loss_scale.step} wraps one optimizer step, here Adam's over the parameters
    and its state, with a scale made where the parameters lie
    ([Vega.Loss_scale.dynamic Nx.Host.on] on the host):

    {[
    let model_state = Nx.Ptree.pair model (Vega.adam_ptree model)

    let step ((params, opt), ls) =
      let objective p = Vega.Loss_scale.scale ls (loss p) in
      let _, grads = Rune.value_and_grad model objective params in
      Vega.Loss_scale.step model model_state ls ~grads
        (fun grads -> Vega.adam_step model ~lr opt ~params ~grads)
        (params, opt)
    ]}

    Bfloat16 shares float32's exponent range and needs no scale. Under
    [Loss_scale.static at 1.0], {!Loss_scale.step} only skips the steps whose
    gradients are not finite. *)

(** Loss scales for float16 training, after JAX's [jmp]. *)
module Loss_scale : sig
  type 'd t = { scale : 'd Nx.float32_t; good_steps : 'd Nx.int32_t }
  (** The type for loss scales on the set ['d]: the current scale factor and
      the number of consecutive finite steps since it last changed, both
      scalar tensors. [good_steps] is [-1] for a {!static} scale. Both are
      tensors so that a compiled step takes them as arguments and returns them
      ({!walk}); a captured float would be a constant of the program. *)

  val static : 'd Nx.Placement.t -> float -> 'd t
  (** [static at s] is the fixed scale [s] at [at], whole on each of its
      devices: {!val-step} never changes it.

      Raises [Invalid_argument] if [s] is not positive. *)

  val dynamic : ?init:float -> 'd Nx.Placement.t -> 'd t
  (** [dynamic at] is a fresh adaptive scale at [at], whole on each of its
      devices, adjusted by {!val-step}. [init] defaults to [32768.] ([2^15]).

      Raises [Invalid_argument] if [init] is not positive. *)

  val scale : 'd t -> (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t
  (** [scale ls x] is [x] times the current scale, at [x]'s dtype. Apply it to
      the loss, inside the differentiated objective. *)

  val step :
    ?growth_interval:int ->
    ?growth_factor:float ->
    ?backoff_factor:float ->
    ('p, 'd) Nx.Ptree.t ->
    ('s, 'd) Nx.Ptree.t ->
    'd t ->
    grads:'p ->
    ('p -> 's) ->
    's ->
    's * 'd t
  (** [step p s ls ~grads update x] is one optimizer step from [x] under the
      loss scale [ls], and the scale for the next step. [grads] are gradients of
      the scaled loss, with structure [p]; [x], typically the parameters and the
      optimizer state, has structure [s].

      [step] divides every leaf of [grads] by the current scale, at the leaf's
      dtype, and checks that every element of the result is finite.
      - If it is, the step is [update grads'] for the divided gradients
        [grads']. A {!dynamic} scale counts the finite step; on reaching
        [growth_interval] (default [2000]) consecutive finite steps it is
        multiplied by [growth_factor] (default [2.]) and the count restarts.
      - Otherwise the step is skipped: it is [x], every tensor of it, the
        optimizer's counters included. A {!dynamic} scale is multiplied by
        [backoff_factor] (default [0.5]) and the count restarts.

      A {!static} scale is returned unchanged. [step] is tensor arithmetic: it
      always computes [update grads'] and selects with {!Nx.where}, so it
      compiles under [Rune.jit] into one program whatever the gradients.

      Raises [Invalid_argument] if [growth_interval], [growth_factor] or
      [backoff_factor] is not positive, or as {!Nx.Ptree.map2} does if
      [update grads'] and [x] differ in structure. *)

  val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> 'd t -> 'e t
  (** [walk c ls] walks a loss scale at [c]'s path: the fixed tensors [scale]
      then [good_steps], at those fields, reporting nothing. Its structure is
      [Nx.Ptree.v Vega.Loss_scale.walk], and a structure holding one walks it
      with [Nx.Ptree.Walk.field c "loss_scale" Vega.Loss_scale.walk]. *)
end

(** {1:lr Learning Rates}

    A step's [~lr] is a function of the step counter, as a {!Schedule} is:
    the step applies it to its state's counter of completed steps and casts
    the scalar float tensor it gives to the dtype each leaf computes at
    ({!section-structures}). A constant rate is {!val-lr}; a rate exact at
    float64 is one at that dtype beside the counter,
    [fun step -> Nx.scalar_at (Nx.placement step) Nx.float64 0.1].
    Either is tensor arithmetic beside the counter, which is what makes the
    rate track correctly under [Rune.jit]. *)

val lr : float -> 'd Schedule.t
(** [lr v] is the constant rate [v], {!Schedule.constant}[ v], which serves
    any parameter dtype and set: [Vega.adam_step p ~lr:(Vega.lr 1e-3)]. *)

(** {1:sgd Stochastic Gradient Descent} *)

type ('p, 'd) sgd_state = { velocity : 'p; step : 'd Nx.int32_t }
(** The state for {!sgd_step}: the momentum velocity, with the shape of the
    parameters, and the number of completed steps as a scalar tensor. Every
    state carries its counter, so a {!Schedule} applies to [st.step] whichever
    optimizer is stepping. *)

module Sgd_state : Nx.Ptree.S with type ('p, 'd) t = ('p, 'd) sgd_state
(** The structure of SGD states. Its [walk] visits [velocity] at the field
    ["velocity"] as a position of the parameter, then [step] at ["step"] as a
    fixed [int32] tensor. Use it with {!Nx.Ptree.Payload} and {!Nx.Ptree.cast};
    transformations take {!sgd_ptree}. *)

val sgd_ptree : ('p, 'd) Nx.Ptree.t -> (('p, 'd) sgd_state, 'd) Nx.Ptree.t
(** [sgd_ptree p] is [Nx.Ptree.nest (module Sgd_state) p], the structure of an
    SGD state over parameters of structure [p]. It visits [velocity.]{e path}
    for each of [p]'s visits, in [p]'s order, then [step]. A compiled step names
    it beside the parameters:

    {[
    let state = Nx.Ptree.pair model (Vega.sgd_ptree model)
    ]} *)

val sgd_init : ('p, 'd) Nx.Ptree.t -> 'p -> ('p, 'd) sgd_state
(** [sgd_init p params] is the initial state for optimizing [params]: an
    all-zero velocity of [params]' shape and [step = 0]. *)

val sgd_step :
  ('p, 'd) Nx.Ptree.t ->
  lr:('d Nx.int32_t -> (float, 'b, 'd) Nx.t) ->
  ?momentum:float ->
  ('p, 'd) sgd_state ->
  params:'p ->
  grads:'p ->
  'p * ('p, 'd) sgd_state
(** [sgd_step p ~lr st ~params ~grads] is [(params', st')] after one step of
    gradient descent with heavy-ball momentum. Per element:

    {v
    v' = momentum * v + g
    p' = p - lr * v'
    v}

    [lr] gives the rate at the counter ({!section-lr}); [momentum] defaults to [0.], plain
    gradient descent: the velocity is then the last gradient, and the input
    velocity is not read at all. The counter advances by one. The whole step is
    tensor arithmetic over [(params, st)], so it traces under [Rune.jit].

    Raises [Invalid_argument] if [momentum] is outside \[[0];[1]\), or as
    {!section-structures} states if [grads], or [st.velocity] when [momentum] is
    not [0.], does not have [params]' skeleton. *)

(** {1:lars LARS} *)

val lars_init : ('p, 'd) Nx.Ptree.t -> 'p -> ('p, 'd) sgd_state
(** [lars_init] is {!sgd_init}: LARS keeps SGD's momentum velocity. *)

val lars_step :
  ('p, 'd) Nx.Ptree.t ->
  lr:('d Nx.int32_t -> (float, 'b, 'd) Nx.t) ->
  ?momentum:float ->
  ?weight_decay:float ->
  ?nesterov:bool ->
  ('p, 'd) sgd_state ->
  params:'p ->
  grads:'p ->
  'p * ('p, 'd) sgd_state
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

type ('p, 'd) adam_state = {
  mu : 'p;
      (** Exponential moving average of gradients (biased; the correction
          applies when computing the update), with the parameters' shape. *)
  nu : 'p;
      (** Exponential moving average of squared gradients, with the parameters'
          shape. *)
  step : 'd Nx.int32_t;
      (** Completed steps, a scalar tensor — not a host [int], which would be
          burned into a compiled trace as a constant and replayed stale. The
          steps derive their bias corrections from it, and it is the counter
          schedules take. *)
}

module Adam_state : Nx.Ptree.S with type ('p, 'd) t = ('p, 'd) adam_state
(** The structure of Adam states. Its [walk] visits [mu] then [nu] at those
    fields as positions of the parameter, then [step] at ["step"] as a fixed
    [int32] tensor. Transformations take {!adam_ptree}. *)

val adam_ptree : ('p, 'd) Nx.Ptree.t -> (('p, 'd) adam_state, 'd) Nx.Ptree.t
(** [adam_ptree p] is [Nx.Ptree.nest (module Adam_state) p], the structure of an
    Adam state over parameters of structure [p]. It visits [mu.]{e path} for
    each of [p]'s visits, in [p]'s order, then [nu.]{e path} likewise, then
    [step]. *)

val adam_init : ('p, 'd) Nx.Ptree.t -> 'p -> ('p, 'd) adam_state
(** [adam_init p params] is the initial state for optimizing [params]: all-zero
    moments and [step = 0]. *)

val adam_step :
  ('p, 'd) Nx.Ptree.t ->
  lr:('d Nx.int32_t -> (float, 'b, 'd) Nx.t) ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  ('p, 'd) adam_state ->
  params:'p ->
  grads:'p ->
  'p * ('p, 'd) adam_state
(** [adam_step p ~lr st ~params ~grads] is [(params', st')] after one Adam step
    (Kingma and Ba, 2015). Per element, with [t = st.step + 1]:

    {v
    mu' = b1 * mu + (1 - b1) * g
    nu' = b2 * nu + (1 - b2) * g^2
    d   = (mu' / (1 - b1^t)) / (sqrt (nu' / (1 - b2^t)) + eps)
    p'  = p - lr * d
    v}

    [lr] gives the rate at the counter ({!section-lr}). [b1] defaults to [0.9], [b2] to
    [0.999], [eps] to [1e-8]; they are compile-time constants, safe captures
    under [Rune.jit]. The bias corrections are derived from the state's counter
    in tensor arithmetic (see {!section-structures} for their precision), so the
    whole step traces and the returned state feeds the next call.

    Raises [Invalid_argument] if [b1] or [b2] is outside \[[0];[1]\) or [eps] is
    not positive, or as {!section-structures} states if [grads], [st.mu] or
    [st.nu] does not have [params]' skeleton. *)

val adamw_init : ('p, 'd) Nx.Ptree.t -> 'p -> ('p, 'd) adam_state
(** [adamw_init] is {!adam_init}: AdamW shares Adam's state. *)

val adamw_step :
  ('p, 'd) Nx.Ptree.t ->
  lr:('d Nx.int32_t -> (float, 'b, 'd) Nx.t) ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  ?weight_decay:float ->
  ('p, 'd) adam_state ->
  params:'p ->
  grads:'p ->
  'p * ('p, 'd) adam_state
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

val radam_init : ('p, 'd) Nx.Ptree.t -> 'p -> ('p, 'd) adam_state
(** [radam_init] is {!adam_init}: RAdam shares Adam's state. *)

val radam_step :
  ('p, 'd) Nx.Ptree.t ->
  lr:('d Nx.int32_t -> (float, 'b, 'd) Nx.t) ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  ('p, 'd) adam_state ->
  params:'p ->
  grads:'p ->
  'p * ('p, 'd) adam_state
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
    the two forms is a tensor [where] that traces under [Rune.jit]. Defaults and
    the counter's role are {!adam_step}'s.

    Raises [Invalid_argument] as {!adam_step} does. *)

(** {1:lamb LAMB} *)

val lamb_init : ('p, 'd) Nx.Ptree.t -> 'p -> ('p, 'd) adam_state
(** [lamb_init] is {!adam_init}: LAMB shares Adam's state. *)

val lamb_step :
  ('p, 'd) Nx.Ptree.t ->
  lr:('d Nx.int32_t -> (float, 'b, 'd) Nx.t) ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  ?weight_decay:float ->
  ('p, 'd) adam_state ->
  params:'p ->
  grads:'p ->
  'p * ('p, 'd) adam_state
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

type ('p, 'd) rmsprop_state = { nu : 'p; velocity : 'p; step : 'd Nx.int32_t }
(** The state for {!rmsprop_step}: the moving average of squared gradients and
    the momentum velocity, both with the shape of the parameters, and the number
    of completed steps as a scalar tensor. *)

module Rmsprop_state : Nx.Ptree.S with type ('p, 'd) t = ('p, 'd) rmsprop_state
(** The structure of RMSprop states. Its [walk] visits [nu] then [velocity] at
    those fields as positions of the parameter, then [step] at ["step"] as a
    fixed [int32] tensor. Transformations take {!rmsprop_ptree}. *)

val rmsprop_ptree : ('p, 'd) Nx.Ptree.t -> (('p, 'd) rmsprop_state, 'd) Nx.Ptree.t
(** [rmsprop_ptree p] is [Nx.Ptree.nest (module Rmsprop_state) p]. *)

val rmsprop_init : ('p, 'd) Nx.Ptree.t -> 'p -> ('p, 'd) rmsprop_state
(** [rmsprop_init p params] is the initial state for optimizing [params]:
    all-zero averages and velocity, and [step = 0]. *)

val rmsprop_step :
  ('p, 'd) Nx.Ptree.t ->
  lr:('d Nx.int32_t -> (float, 'b, 'd) Nx.t) ->
  ?decay:float ->
  ?eps:float ->
  ?momentum:float ->
  ('p, 'd) rmsprop_state ->
  params:'p ->
  grads:'p ->
  'p * ('p, 'd) rmsprop_state
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

type ('p, 'd) adagrad_state = { sum_of_squares : 'p; step : 'd Nx.int32_t }
(** The state for {!adagrad_step}: the sum of all squared gradients so far, with
    the shape of the parameters, and the number of completed steps as a scalar
    tensor. *)

module Adagrad_state : Nx.Ptree.S with type ('p, 'd) t = ('p, 'd) adagrad_state
(** The structure of Adagrad states. Its [walk] visits [sum_of_squares] at that
    field as a position of the parameter, then [step] at ["step"] as a fixed
    [int32] tensor. Transformations take {!adagrad_ptree}. *)

val adagrad_ptree : ('p, 'd) Nx.Ptree.t -> (('p, 'd) adagrad_state, 'd) Nx.Ptree.t
(** [adagrad_ptree p] is [Nx.Ptree.nest (module Adagrad_state) p]. *)

val adagrad_init : ('p, 'd) Nx.Ptree.t -> 'p -> ('p, 'd) adagrad_state
(** [adagrad_init p params] is the initial state for optimizing [params]: an
    all-zero sum and [step = 0]. *)

val adagrad_step :
  ('p, 'd) Nx.Ptree.t ->
  lr:('d Nx.int32_t -> (float, 'b, 'd) Nx.t) ->
  ?eps:float ->
  ('p, 'd) adagrad_state ->
  params:'p ->
  grads:'p ->
  'p * ('p, 'd) adagrad_state
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

type ('p, 'd) adan_state = {
  mu : 'p;  (** Moving average of the gradients. *)
  delta : 'p;  (** Moving average of the differences of successive gradients. *)
  nu : 'p;
      (** Moving average of the squared look-ahead gradients
          [g + b2 * (g - prev_grads)]. *)
  prev_grads : 'p;  (** The gradients of the last step, zero initially. *)
  step : 'd Nx.int32_t;  (** Completed steps, a scalar tensor. *)
}
(** The state for {!adan_step}. Every part but [step] has the shape of the
    parameters. *)

module Adan_state : Nx.Ptree.S with type ('p, 'd) t = ('p, 'd) adan_state
(** The structure of Adan states. Its [walk] visits [mu], [delta], [nu] then
    [prev_grads] at those fields as positions of the parameter, then [step] at
    ["step"] as a fixed [int32] tensor. Transformations take {!adan_ptree}. *)

val adan_ptree : ('p, 'd) Nx.Ptree.t -> (('p, 'd) adan_state, 'd) Nx.Ptree.t
(** [adan_ptree p] is [Nx.Ptree.nest (module Adan_state) p]. *)

val adan_init : ('p, 'd) Nx.Ptree.t -> 'p -> ('p, 'd) adan_state
(** [adan_init p params] is the initial state for optimizing [params]: every
    part zero and [step = 0]. *)

val adan_step :
  ('p, 'd) Nx.Ptree.t ->
  lr:('d Nx.int32_t -> (float, 'b, 'd) Nx.t) ->
  ?b1:float ->
  ?b2:float ->
  ?b3:float ->
  ?eps:float ->
  ?weight_decay:float ->
  ('p, 'd) adan_state ->
  params:'p ->
  grads:'p ->
  'p * ('p, 'd) adan_state
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

type ('p, 'd) lion_state = { mu : 'p; step : 'd Nx.int32_t }
(** The state for {!lion_step}: the moving average of the gradients, with the
    shape of the parameters, and the number of completed steps as a scalar
    tensor. *)

module Lion_state : Nx.Ptree.S with type ('p, 'd) t = ('p, 'd) lion_state
(** The structure of Lion states. Its [walk] visits [mu] at that field as a
    position of the parameter, then [step] at ["step"] as a fixed [int32]
    tensor. Transformations take {!lion_ptree}. *)

val lion_ptree : ('p, 'd) Nx.Ptree.t -> (('p, 'd) lion_state, 'd) Nx.Ptree.t
(** [lion_ptree p] is [Nx.Ptree.nest (module Lion_state) p]. *)

val lion_init : ('p, 'd) Nx.Ptree.t -> 'p -> ('p, 'd) lion_state
(** [lion_init p params] is the initial state for optimizing [params]: an
    all-zero average and [step = 0]. *)

val lion_step :
  ('p, 'd) Nx.Ptree.t ->
  lr:('d Nx.int32_t -> (float, 'b, 'd) Nx.t) ->
  ?b1:float ->
  ?b2:float ->
  ('p, 'd) lion_state ->
  params:'p ->
  grads:'p ->
  'p * ('p, 'd) lion_state
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

type ('p, 'd) adafactor_state = {
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
  step : 'd Nx.int32_t;  (** Completed steps, a scalar tensor. *)
}
(** The state for {!adafactor_step}. A leaf of two or more axes is factored: its
    second moment is estimated from a row and a column statistic, in [m + n]
    numbers rather than [m * n]. {!adafactor_init} decides which leaves are
    factored, and the state records it. *)

module Adafactor_state : Nx.Ptree.S with type ('p, 'd) t = ('p, 'd) adafactor_state
(** The structure of Adafactor states. Its [walk] visits [nu_row], [nu_col] then
    [nu] at those fields as positions of the parameter, then [step] at ["step"]
    as a fixed [int32] tensor. Transformations take {!adafactor_ptree}. *)

val adafactor_ptree : ('p, 'd) Nx.Ptree.t -> (('p, 'd) adafactor_state, 'd) Nx.Ptree.t
(** [adafactor_ptree p] is [Nx.Ptree.nest (module Adafactor_state) p]. *)

val adafactor_init : ('p, 'd) Nx.Ptree.t -> ?factored:bool -> 'p -> ('p, 'd) adafactor_state
(** [adafactor_init p params] is the initial state for optimizing [params]: zero
    statistics and [step = 0]. With [factored] (the default, [true]), leaves of
    two or more axes are factored; without, no leaf is. *)

val adafactor_step :
  ('p, 'd) Nx.Ptree.t ->
  lr:('d Nx.int32_t -> (float, 'b, 'd) Nx.t) ->
  ?decay_rate:float ->
  ?eps:float ->
  ?clipping_threshold:float ->
  ('p, 'd) adafactor_state ->
  params:'p ->
  grads:'p ->
  'p * ('p, 'd) adafactor_state
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
    rate that decays with the counter, a schedule of tensor arithmetic, here
    [1e-3 / sqrt t]:

    {[
    let lr step = Nx.mul_s (Nx.rsqrt (Nx.cast Nx.float32 (Nx.add_s step 1l))) 1e-3
    let st' = Vega.adafactor_step p ~lr st ~params ~grads
    ]}

    Raises [Invalid_argument] if [decay_rate], [eps] or [clipping_threshold] is
    not positive, or as {!section-structures} states if [grads] or a part of
    [st] other than [step] does not have [params]' skeleton. *)

(** {1:lbfgs L-BFGS}

    Limited-memory BFGS (Liu and Nocedal, 1989) at a fixed rate: the gradient
    preconditioned by the inverse Hessian that the last [history] pairs of
    parameter and gradient differences define. It is a training step: it takes
    the gradient at the state's parameters, as every step does, and moves by
    [lr] along the preconditioned direction. A deterministic problem, a minimum
    to a tolerance with its derivative, is [Jera.Minimize]'s, whose L-BFGS
    searches its step length.

    {[
    let st = Vega.lbfgs_init model params in
    let st =
      Vega.lbfgs_step model ~lr:(Vega.lr 0.1)
        (Rune.grad model loss st.params)
        st
    in
    st.params
    ]}

    The history is held at a fixed size, so the state is a parameter tree of
    static shape ({!lbfgs_ptree}) and the step traces under [Rune.jit]. Its
    vectors compute leaf by leaf as {!section-structures} states, a float16 or
    bfloat16 leaf at float32 and rounded once, when stored. The inner products
    of the two-loop recursion are at float64 when every float leaf is float64,
    and at float32 otherwise. *)

type ('p, 'd) lbfgs_state = {
  params : 'p;  (** The current point. *)
  grads : 'p;
      (** The gradient the last step received, at the point before [params];
          zeros before the first step. *)
  s : 'p;
      (** The last parameter differences, newest first, stacked along a new
          leading axis of length [history] on every leaf. The newest is the move
          the last step made. *)
  y : 'p;
      (** The matching gradient differences, laid out like [s]. The newest is
          zeros until the next step completes it. A pair whose curvature [y . s]
          is not positive, an empty slot among them, enters no direction. *)
  step : 'd Nx.int32_t;  (** Completed steps, a scalar tensor. *)
}

val lbfgs_ptree : ('p, 'd) Nx.Ptree.t -> (('p, 'd) lbfgs_state, 'd) Nx.Ptree.t
(** [lbfgs_ptree p] is the structure of an L-BFGS state over parameters of
    structure [p]. It visits [params.]{e path} for each of [p]'s visits, then
    [grads.]{e path}, [s.]{e path}, [y.]{e path} and [step]. *)

val lbfgs_init : ('p, 'd) Nx.Ptree.t -> ?history:int -> 'p -> ('p, 'd) lbfgs_state
(** [lbfgs_init p params] is the initial state at [params], with an empty
    history of [history] pairs (default [10]).

    Raises [Invalid_argument] if [history < 1]. *)

val lbfgs_step :
  ('p, 'd) Nx.Ptree.t -> lr:('d Nx.int32_t -> (float, 'b, 'd) Nx.t) -> 'p -> ('p, 'd) lbfgs_state -> ('p, 'd) lbfgs_state
(** [lbfgs_step p ~lr grads st] is the state after one step from [st.params],
    where the objective's gradient is [grads]. It completes the newest pair with
    [grads - st.grads], computes the direction [d] by the two-loop recursion of
    Nocedal (1980) applied to the negated gradient, with the initial inverse
    Hessian scaled by [(s . y) / (y . y)] of the newest pair, moves to
    [st.params + lr * d], and pushes the move as the newest pair, dropping the
    oldest. With no pair of positive curvature, [d] is [-grads].

    Raises [Invalid_argument] as {!section-structures} states if [grads],
    [st.grads], [st.s] or [st.y] does not have [st.params]' skeleton. *)
