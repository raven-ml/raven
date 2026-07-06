(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Gradient-descent optimizers over differentiable structures.

    Optimizer state has the shape of the parameters themselves: each algorithm
    keeps its per-parameter accumulators as values of the user's own structure
    type, in a small record the training loop threads explicitly
    ({!type:sgd_state}, {!type:adam_state}). Steps are pure {!Nx.Ptree.S}
    traversals — a step consumes a state and returns the next one — so a
    training step is an ordinary function of [(params, state)] and checkpointing
    an optimizer means saving a record of parameter-shaped values.

    There is no optimizer object; composition is function application. Transform
    gradients before the step (for example {!clip_by_global_norm}) and evaluate
    a {!type:schedule} at the loop's step counter for that step's learning rate:

    {[
      let sched = Optim.cosine_decay ~init:1e-3 ~steps:1000 () in
      let step k (params, st) =
        let loss, grads = Rune_next.value_and_grad (module P) f params in
        let grads = Optim.clip_by_global_norm (module P) ~max_norm:1.0 grads in
        let params, st =
          Optim.adamw_step (module P) ~lr:(sched k) st ~params ~grads
        in
        ((params, st), loss)
    ]} *)

(** {1:schedules Learning-rate schedules}

    A schedule maps a number of completed steps to a learning rate. It is a
    plain function: compose or define schedules directly, and pass [sched k] as
    a step function's [~lr]. *)

type schedule = int -> float
(** The type for learning-rate schedules. [sched k] is the learning rate of the
    step taken after [k] completed steps; [k = 0] is the first step. Schedules
    are defined for [k >= 0]. *)

val constant : float -> schedule
(** [constant lr] is the schedule that is [lr] at every step. *)

val exponential_decay : init:float -> rate:float -> steps:int -> schedule
(** [exponential_decay ~init ~rate ~steps] decays geometrically:
    [init *. (rate ** (float k /. float steps))]. The rate is [init] at step [0]
    and [init *. rate] at step [steps].

    Raises [Invalid_argument] if [steps <= 0]. *)

val cosine_decay : ?final:float -> init:float -> steps:int -> unit -> schedule
(** [cosine_decay ~init ~steps ()] follows a half cosine from [init] at step [0]
    down to [final] at step [steps], and is [final] from then on. [final]
    defaults to [0.].

    Raises [Invalid_argument] if [steps <= 0]. *)

val warmup_cosine :
  ?final:float -> peak:float -> warmup:int -> steps:int -> unit -> schedule
(** [warmup_cosine ~peak ~warmup ~steps ()] ramps linearly from [0.] at step [0]
    to [peak] at step [warmup], then follows {!cosine_decay} from [peak] down to
    [final] at step [steps]. [final] defaults to [0.].

    Raises [Invalid_argument] if [warmup < 0] or [steps <= warmup]. *)

(** {1:gradients Gradient transformations}

    Pure functions on gradient structures, applied between the backward pass and
    the optimizer step. *)

val global_norm : (module P : Nx.Ptree.S) -> P.t -> float
(** [global_norm (module P) grads] is the L2 norm of all leaves of [grads] taken
    together: [sqrt (sum of every element squared)]. *)

val clip_by_global_norm :
  (module P : Nx.Ptree.S) -> max_norm:float -> P.t -> P.t
(** [clip_by_global_norm (module P) ~max_norm grads] scales [grads] so that its
    {!global_norm} does not exceed [max_norm]. Gradients within the bound
    (including all-zero gradients) are returned unchanged; larger ones are
    scaled by [max_norm /. norm], preserving their direction.

    Raises [Invalid_argument] if [max_norm <= 0.]. *)

val clip_by_value : (module P : Nx.Ptree.S) -> max:float -> P.t -> P.t
(** [clip_by_value (module P) ~max grads] clips every gradient element to the
    interval \[[-. max];[max]\].

    Raises [Invalid_argument] if [max <= 0.]. *)

(** {1:sgd Stochastic gradient descent} *)

type 'p sgd_state = { velocity : 'p }
(** The state for {!sgd_step}: the momentum velocity, with the shape of the
    parameters. *)

val sgd_init : (module P : Nx.Ptree.S) -> P.t -> P.t sgd_state
(** [sgd_init (module P) params] is the initial state for optimizing [params]:
    an all-zero velocity of [params]' shape. *)

val sgd_step :
  (module P : Nx.Ptree.S) ->
  lr:float ->
  ?momentum:float ->
  P.t sgd_state ->
  params:P.t ->
  grads:P.t ->
  P.t * P.t sgd_state
(** [sgd_step (module P) ~lr st ~params ~grads] is [(params', st')] after one
    step of gradient descent with heavy-ball momentum. Per element:

    {v
    v' = momentum * v + g
    p' = p - lr * v'
    v}

    [momentum] defaults to [0.], plain gradient descent: the velocity is then
    the last gradient. *)

(** {1:adam Adam and AdamW} *)

type 'p adam_state = { mu : 'p; nu : 'p; step : int }
(** The state for {!adam_step} and {!adamw_step}. [mu] and [nu] are the
    exponential moving averages of gradients and of squared gradients (biased;
    steps apply the correction when computing the update), each with the
    parameters' shape. [step] is the number of completed steps. *)

val adam_init : (module P : Nx.Ptree.S) -> P.t -> P.t adam_state
(** [adam_init (module P) params] is the initial state for optimizing [params]:
    all-zero moments and [step = 0]. *)

val adam_step :
  (module P : Nx.Ptree.S) ->
  lr:float ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  P.t adam_state ->
  params:P.t ->
  grads:P.t ->
  P.t * P.t adam_state
(** [adam_step (module P) ~lr st ~params ~grads] is [(params', st')] after one
    Adam step (Kingma and Ba, 2015). Per element, with [t = st.step + 1]:

    {v
    mu' = b1 * mu + (1 - b1) * g
    nu' = b2 * nu + (1 - b2) * g^2
    d   = (mu' / (1 - b1^t)) / (sqrt (nu' / (1 - b2^t)) + eps)
    p'  = p - lr * d
    v}

    [b1] defaults to [0.9], [b2] to [0.999], [eps] to [1e-8]. *)

val adamw_init : (module P : Nx.Ptree.S) -> P.t -> P.t adam_state
(** [adamw_init] is {!adam_init}: AdamW shares Adam's state. *)

val adamw_step :
  (module P : Nx.Ptree.S) ->
  lr:float ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  ?weight_decay:float ->
  P.t adam_state ->
  params:P.t ->
  grads:P.t ->
  P.t * P.t adam_state
(** [adamw_step (module P) ~lr st ~params ~grads] is like {!adam_step} with
    decoupled weight decay (Loshchilov and Hutter, 2019): with [d] Adam's
    bias-corrected direction, the parameter update becomes

    {v p' = p - lr * (d + weight_decay * p) v}

    The decay applies to the parameters directly rather than through the
    adaptive scaling, so its effective strength does not depend on the gradient
    history. [weight_decay] defaults to [0.01]; with [weight_decay = 0.] the
    step is exactly {!adam_step}. *)
