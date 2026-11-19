(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Barlow Twins pretraining with mixup, the objective TESSERA's models are
    pretrained by: v1.0's, which v2's teacher is pretrained with too.

    A model learns from two {e views} of each pixel, each a draw of its
    observations ({!Pixels.draw}), and from their {e mixup}, the model's input
    [alpha · x₁ + (1 − alpha) · x₂] of the first view and the second view of
    another pixel ({!Model.mixup}). A {e projector} maps each view's embedding
    to [P] values: linear layers, each but the last followed by batch
    normalisation and ReLU. The objective's statistics are over a {e group}, the
    [B] pixels whose views it projects together: the projector's batch
    normalisation, which standardises each value over the group as
    [Nx.standardize ~axes:[0] ~eps:1e-5] before its scale and shift, and the
    cross-correlations below.

    For projections [z] and [z'] of a group ([[B; P]]), [w] is [z] with each
    column standardised by its mean and its deviation with [B − 1] degrees of
    freedom, at least [1e-9], and [c (z, z') = wᵀ w' / B], [P × P]. The
    {e Barlow Twins} loss (Zbontar et al. 2021) of views [z₁] and [z₂] is
    [Σᵢ (cᵢᵢ − 1)² + λ Σ_{i ≠ j} cᵢⱼ²], [c = c (z₁, z₂)]. The {e mixup} loss
    (Zhang et al. 2018) of the mixed view [z_m] of mixing weight [alpha] and
    pixel permutation [π], [z₂ₚ] the rows [π] of [z₂], is
    [μ (|c (z_m, z₁) − c_a|² + |c (z_m, z₂ₚ) − c_b|²)], Frobenius norms, with
    [c_a = alpha c (z₁, z₁) + (1 − alpha) c (z₂ₚ, z₁)] and
    [c_b = alpha c (z₁, z₂ₚ) + (1 − alpha) c (z₂ₚ, z₂ₚ)], the correlations a
    mixup linear in its inputs would have. The objective is their sum. The v1.0
    trainer's code weighs the mixup loss by [μ = 0.005], the product of its
    mixup weight [1] and [λ]; its paper writes [1/2].

    Every norm of a product [xᵀ y] ([B × P] each) is computed through the
    [B × B] Grams, [|xᵀ y|² = Σ (x xᵀ) ⊙ (y yᵀ)], when [2B ≤ P] (TESSERA v2,
    App. B.5), and through the [P × P] product otherwise. The Gram form costs
    [O (B² P)] where the product costs [O (B P²)].

    Each term is within a rounding bound of its exact value, proportional to its
    {e absolute form}, the same expression over [|w|]: the Gram form's
    off-diagonal sum, a difference, can cancel, so the product form takes over
    where [2B > P], within about twice the Gram form's cost.

    A training {e step} averages the objective over several groups, each a row
    of a {!Rune.scan}, clips the gradient's global norm and takes an AdamW step.
    A group's size is part of the objective: the mean over groups of [B] pixels
    is v1.0's mean over its devices, [B] pixels each. *)

(** {1:params Parameters} *)

type ('a, 'd) projector = {
  hidden :
    (('a, 'd) Kaun.Linear.t * ('d Nx.float32_t, 'd) Kaun.Batch_norm.t) list;
      (** Each a linear layer, then batch normalisation's scale and shift, then
          ReLU. *)
  out : ('a, 'd) Kaun.Linear.t;
}
(** The type for projectors. The products compute at their parameters' dtype, as
    {!Model}'s do; batch normalisation and the result are float32, and its
    scales and shifts are float32 fixed arrays at every dtype. *)

type ('a, 'd) params = {
  net : ('a, 'd) Model.net;
  projector : ('a, 'd) projector;
}
(** The type for what pretraining trains: a model's net and its projector. *)

module Params : Nx.Ptree.S with type ('a, 'd) t = ('a, 'd) params
(** Its walk reports the projector's length. *)

val projector : key:'d Nx.Rng.key -> int list -> ('d Nx.float32_t, 'd) projector
(** [projector ~key widths] is a projector from [List.hd widths] values through
    each next width, kaun's initialisers drawn from [key], batch normalisation
    at scale [1] and shift [0]. v1.0's is
    [[128; 16384; 16384; 16384; 16384; 16384; 16384]], six linear layers.

    Raises [Invalid_argument] if [widths] has fewer than two widths or one is
    not positive. *)

val load_projector : string -> (Nx.host Nx.float32_t, Nx.host) projector
(** [load_projector file] is the projector in the npz [file], in PyTorch's
    layout of a sequence [net] of linear, batch normalisation and ReLU modules:
    the modules [i] with parameters, [net.i.weight] and [net.i.bias], in the
    numeric order of [i] ([net.10] after [net.9]), are linear layers, a weight
    [[out; in]] read as stored, each but the last followed by a batch
    normalisation, its weight the scale and its bias the shift, all at float32.

    Raises [Failure] naming [file] for another sequence of modules. Raises
    [Failure] as {!Nx_io.Archive.float} does for a missing bias, and as
    {!Nx_io.load_npz} does. *)

val project :
  ((float, 's, 'd) Nx.t, 'd) projector -> 'd Nx.float32_t -> 'd Nx.float32_t
(** [project p e] is the projection [[B; P]] of the group of embeddings [e]
    ([[B; d]]), its batch normalisation over [e]'s rows. *)

(** {1:objective Objective} *)

type weights = { off_diagonal : float;  (** [λ] *) mixup : float  (** [μ] *) }
(** The type for the objective's weights. *)

val v1_weights : weights
(** [v1_weights] is [{ off_diagonal = 5e-3; mixup = 5e-3 }], the released v1.0
    run's. *)

type 'a terms = {
  on_diagonal : 'a;  (** [Σᵢ (cᵢᵢ − 1)²] *)
  off_diagonal : 'a;  (** [Σ_{i ≠ j} cᵢⱼ²] *)
  mixup : 'a;  (** [|c (z_m, z₁) − c_a|² + |c (z_m, z₂ₚ) − c_b|²] *)
}
(** The type for the objective's terms, before their weights: what a run watches
    to see the correlations reach the identity, and the mixup its targets. *)

val terms :
  alpha:(float, 's, 'd) Nx.t ->
  perm:'d Nx.int64_t ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t terms
(** [terms ~alpha ~perm z₁ z₂ z_m] is the terms of the objective of the views
    [z₁] and [z₂] and the mixed view [z_m], each [[B; P]], at [z₁]'s dtype.
    [alpha] is a scalar and [perm] ([[B]]) a permutation of the rows.

    Raises [Invalid_argument] if the projections differ in shape, if [B < 2],
    where the deviations and the batch normalisation are degenerate, or if
    [perm] is not [[B]]. *)

val objective : weights -> (float, 's, 'd) Nx.t terms -> (float, 's, 'd) Nx.t
(** [objective w t] is the Barlow Twins loss plus the mixup loss of the terms
    [t], [t.on_diagonal + λ t.off_diagonal + μ t.mixup]. *)

type 'd views = {
  a1 : 'd Batch.t;  (** The first views, as {!pack} lays them out. *)
  a2 : 'd Batch.t;  (** The second views of the same pixels. *)
  keys : 'd Nx.Rng.key;
      (** [[3; B; 2]]: each pixel's keys in the first, second and mixed view, as
          {!Model.train} takes them. *)
  alpha : 'd Nx.float32_t;  (** A scalar: the mixing weight. *)
  perm : 'd Nx.int64_t;
      (** [[B]]: the permutation whose pixel [π i] is mixed into pixel [i]. *)
}
(** The type for a group's views. *)

val pack : Nx.host Pixels.t -> Nx.host Batch.t
(** [pack v] is the batch of the views [v], each pixel [length] observations of
    each sensor as {!Pixels.draw} makes them: pixel [i] at place [i], alone in
    row [i], each observation a token of count [1].

    Raises [Invalid_argument] if [v] has no pixel, or if its rows are not all of
    one length. *)

val group :
  key:Nx.host Nx.Rng.key ->
  length:int ->
  first:int ->
  Nx.host Pixels.t ->
  Nx.host views
(** [group ~key ~length ~first px] is the views of a group of [px]'s [B] pixels,
    pixel [j] at stream position [first + j]: each of its two views
    {!Pixels.draw}s [length] observations of each sensor, and it has a dropout
    key in each of the three views, each a function of [key], the view and the
    position alone. [alpha] is uniform in \[[0]; [1]), Beta(1, 1), and [perm] a
    uniform permutation, both functions of [key] and [first]. A group's draws
    thus depend neither on how many groups a step has nor on the groups beside
    it.

    Raises [Invalid_argument] as {!Pixels.draw} and {!pack} do. *)

val loss :
  weights ->
  dropout:Model.dropout ->
  ('b, 'd) Model.t ->
  ((float, 's, 'd) Nx.t, 'd) params ->
  'd views ->
  'd Nx.float32_t
(** [loss w ~dropout m p v] is the {!objective} of the {!terms} of the group's
    views: the projections of {!Model.train}'s embeddings, by [m] with [p]'s
    net, of [v.a1], of [v.a2] and of their mixup, [v.a1] mixed with [v.a2]'s
    rows [v.perm] by [v.alpha]. Every place of [v.a1] and [v.a2] must hold a
    pixel, place [i] of both the same one's, since an empty place would enter
    the group's statistics.

    Raises [Invalid_argument] as {!Model.train} and {!terms} do. *)

(** {1:step Steps} *)

type ('a, 'd) group = 'd views
(** The type for a step's groups: their views stacked along axis [0]. *)

module Group : Nx.Ptree.S with type ('a, 'd) t = ('a, 'd) group
(** Views as structures, every array fixed. *)

val stack : 'd views list -> 'd views
(** [stack vs] is the views [vs], of one shape, each array stacked along a new
    axis [0], as {!Batch.stack} stacks batches.

    Raises [Invalid_argument] as {!Batch.stack} does. *)

type ('a, 'd) state = {
  model : ('a, 'd) Model.t;
  projector : ('a, 'd) projector;
  opt : (('a, 'd) params, 'd) Vega.adam_state;
}
(** The type for a pretraining's state: the model, with the statistics it keeps
    as they are, its projector and AdamW's state. *)

module State : Nx.Ptree.S with type ('a, 'd) t = ('a, 'd) state
(** States as structures: [model] as {!Model.walk} walks it at ["model"], then
    the projector at ["projector"], then AdamW's state at ["opt"]. *)

val init :
  ('d Nx.float32_t, 'd) Model.t ->
  ('d Nx.float32_t, 'd) projector ->
  ('d Nx.float32_t, 'd) state
(** [init m p] is the state before the first step. *)

val decays : Nx.Ptree.Path.t -> int array -> bool
(** [decays path shape] is [true] for the weights [w] of linear layers: the rule
    v1.0's trainer writes, a parameter of rank above [1], which in its model are
    those weights alone. *)

type recipe = {
  pixels : int;  (** A group's pixels, [B]. *)
  length : int;  (** Observations of each sensor a view. *)
  projector : int list;  (** The projector's widths after the model's. *)
  lr : float;  (** AdamW's peak rate. *)
  warmup : float;
      (** The share of the steps the rate rises over, linearly from [0], before
          it falls to [0] along a cosine ({!schedule}). *)
  weight_decay : float;
  clip : float;  (** The bound on the gradient's global norm. *)
  dropout : Model.dropout;
}
(** The type for pretraining recipes. *)

val warmup_steps : warmup:float -> steps:int -> int
(** [warmup_steps ~warmup ~steps] is [⌊warmup · steps⌋], the steps the rate of a
    run of [steps] steps rises over ({!schedule}), truncated as v1.0's trainer
    truncates it.

    Raises [Invalid_argument] if [warmup] is negative or not finite, if [steps]
    is negative, or if the product passes [max_int]. *)

val schedule : lr:float -> warmup:float -> steps:int -> 'd Vega.Schedule.t
(** [schedule ~lr ~warmup ~steps] is v1.0's trainer's rate over a run of [steps]
    steps (github.com/ucam-eo/tessera-util, commit 290feaf,
    [src/train_multi_gpu_FSDP_QAT.py], lines 179-191): with
    [w = warmup_steps ~warmup ~steps] warmup steps, the rate at step [s] (from
    [0]) is [lr · s / w] for [s < w], then
    [lr · (1 + cos (π · (s − w) / (steps − w))) / 2], [0] from [s = steps] on. A
    warmup of [0] steps starts at [lr]; one of [steps] or more stays at [lr]
    after it.

    Raises [Invalid_argument] as {!warmup_steps} does. *)

val v1_recipe : recipe
(** [v1_recipe] is the released v1.0 run's: groups of [2048], views of [40], the
    projector of six linear layers of [16384], AdamW at [2e-3] with [0.1]
    warmup, decay [0], the decay the run applied, clip [2.0], and dropout [0.1]
    at the encoder layers and [0.2] at the reducer, the rate v2's models fix. *)

val step :
  compute:(float, 'c) Nx.dtype ->
  weights ->
  dropout:Model.dropout ->
  lr:'d Vega.Schedule.t ->
  weight_decay:'d Vega.Schedule.t ->
  ?decays:(Nx.Ptree.Path.t -> int array -> bool) ->
  clip:float ->
  ('d Nx.float32_t, 'd) group * ('d Nx.float32_t, 'd) state ->
  'd Nx.float32_t terms * ('d Nx.float32_t, 'd) state
(** [step ~compute w ~dropout ~lr ~weight_decay ~decays ~clip (gs, st)] is one
    step over the groups [gs], stacked along axis [0]: the mean of the groups'
    {!terms}, whose {!objective} is the mean of their {!loss} up to rounding,
    and the state after
    - the gradient of each group's loss at [st]'s net and projector cast to
      [compute], summed in a {!Rune.scan} over the groups and divided by their
      number;
    - its global norm clipped to [clip] ({!Vega.clip_by_global_norm});
    - [Vega.adamw_step ~lr ~weight_decay ~decays], [decays] {!decays} by
      default.

    It is plain code over arrays, to compile with [Rune.jit] over {!Group} and
    {!State}, the state donated.

    Raises [Invalid_argument] as {!loss} does, and as [Vega.adamw_step] does. *)

(**/**)

type form =
  | Gram
  | Correlation  (** The norms through [B × B] Grams, or [P × P] products. *)

val terms_in :
  form ->
  alpha:(float, 's, 'd) Nx.t ->
  perm:'d Nx.int64_t ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t terms
(** [terms_in f] is {!terms} in the form [f], whichever [B] and [P] are. *)
