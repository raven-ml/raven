(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Matryoshka distillation of a student from a teacher's embeddings.

    A student learns to reconstruct the teacher's embedding [t] of each pixel
    from each of its Matryoshka prefixes (TESSERA v2 §5, eq. 2): its loss at a
    pixel is [Σ_k (1 − cos (h_k (s_{1:k}), t))] over its prefixes [k], [s] the
    student's embedding and [h_k] a {e head} of prefix [k], with equal weights
    (Kusupati et al. 2022, §3). A head is a stack of linear layers, ReLU
    between: one layer is the model card's "separate linear head per prefix",
    more the paper's MLP head. The heads are trained with the student and
    dropped after.

    A training {e step} sums the loss over the pixels of several {e groups},
    each a batch a scan computes in turn, divides by the number of pixels, clips
    the gradient's global norm and takes an AdamW step (v2 App. J, Table 9). The
    loss is a sum over pixels, so a step's gradient does not depend on how its
    pixels are cut into groups, up to rounding. *)

(** {1:params Parameters} *)

type ('a, 'd) head = int * ('a, 'd) Kaun.Linear.t list
(** The type for a prefix's head: the prefix's length and its layers. *)

type ('a, 'd) params = { net : ('a, 'd) Model.net; heads : ('a, 'd) head list }
(** The type for what a distillation trains: a student's net and its heads. *)

module Params : Nx.Ptree.S with type ('a, 'd) t = ('a, 'd) params
(** Its walk reports each head's prefix and the lists' lengths. *)

val heads :
  key:'d Nx.Rng.key ->
  ?hidden:int list ->
  prefixes:int list ->
  int ->
  ('d Nx.float32_t, 'd) head list
(** [heads ~key ~hidden ~prefixes d] is a head for each of [prefixes], to [d]
    values, each with a hidden layer of each width of [hidden] (default [[]]),
    kaun's initialisers drawn from [key].

    Raises [Invalid_argument] if a prefix or a width is not positive. *)

(** {1:loss Loss} *)

val cosines :
  ((float, 's, 'd) Nx.t, 'd) head list ->
  'd Nx.float32_t ->
  targets:(float, 't, 'd) Nx.t ->
  'd Nx.float32_t
(** [cosines hs e ~targets] is [[pixels; List.length hs]]: for each row [s] of
    [e] ([[pixels; dims]]) and each head [h_k] of [hs],
    [cos (h_k (s_{1:k}), t)], [t] the row's target ([[pixels; d]]). A head
    computes at its parameters' dtype, as {!Model}'s products do; its result,
    the targets and the cosine are float32, each vector divided by its norm or
    by [1e-8] if that is larger. Their mean over pixels is the prefixes'
    fidelity to the teacher (v2 App. K). *)

val loss :
  ?record:'d Model.record ->
  dropout:Model.dropout ->
  keys:'d Nx.Rng.key ->
  ('b, 'd) Model.t ->
  ((float, 's, 'd) Nx.t, 'd) params ->
  'd Batch.t ->
  targets:(float, 't, 'd) Nx.t ->
  'd Nx.float32_t
(** [loss ~dropout ~keys m p b ~targets] is the sum over [b]'s filled places of
    [Σ_k (1 − cos (h_k (s_{1:k}), t))] ({!cosines}), [s] the place's training
    embedding [Model.train ~dropout ~remat:false ~keys] by [m] with [p]'s net,
    and [t] its row of [targets]. An empty place adds [0]. [record] receives the
    embedding's intermediate values, as {!Model.train}'s does.

    Raises [Invalid_argument] as {!Model.train} does. *)

(** {1:step Steps} *)

type ('a, 'd) group = {
  batch : 'd Batch.t;
  keys : 'd Nx.Rng.key;  (** [[pixels; 2]], as {!Model.train} takes them. *)
  targets : 'a;  (** [[pixels; d]] *)
}
(** The type for a group: a batch, its places' keys and their targets. *)

module Group : Nx.Ptree.S with type ('a, 'd) t = ('a, 'd) group
(** Groups, and a step's groups stacked along axis [0], as structures: the
    batch's arrays and the keys are fixed arrays, the targets at the parameter,
    so a cast turns float16 targets into float32. *)

type ('a, 'd) state = {
  model : ('a, 'd) Model.t;
  heads : ('a, 'd) head list;
  opt : (('a, 'd) params, 'd) Vega.adam_state;
}
(** The type for a distillation's state: the student, with the statistics it
    keeps as they are, its heads and AdamW's state. *)

module State : Nx.Ptree.S with type ('a, 'd) t = ('a, 'd) state
(** States as structures: [model] as {!Model.walk} walks it at ["model"], then
    the heads at ["heads"], then AdamW's state at ["opt"]. *)

val init :
  ('d Nx.float32_t, 'd) Model.t ->
  ('d Nx.float32_t, 'd) head list ->
  ('d Nx.float32_t, 'd) state
(** [init m hs] is the state before the first step. *)

val step :
  compute:(float, 'c) Nx.dtype ->
  dropout:Model.dropout ->
  lr:'d Vega.Schedule.t ->
  weight_decay:'d Vega.Schedule.t ->
  ?decays:(Nx.Ptree.Path.t -> int array -> bool) ->
  clip:float ->
  ((float, 't, 'd) Nx.t, 'd) group * ('d Nx.float32_t, 'd) state ->
  'd Nx.float32_t * ('d Nx.float32_t, 'd) state
(** [step ~compute ~dropout ~lr ~weight_decay ~decays ~clip (gs, st)] is one
    step over the groups [gs], stacked along axis [0] ({!Batch.stack}): the
    loss's mean over the groups' filled places, and the state after
    - the gradient of each group's {!loss} at [st]'s net and heads cast to
      [compute], summed in a {!Rune.scan} over the groups and divided by the
      number of filled places;
    - its global norm clipped to [clip] ({!Vega.clip_by_global_norm});
    - [Vega.adamw_step ~b2:0.95 ~lr ~weight_decay ~decays].

    Groups that fill no place divide by [1]: the loss is [0], the gradient zero,
    and AdamW still steps its counter, its moments and its decay.

    It is plain code over arrays, to compile with [Rune.jit] over {!Group} and
    {!State}, the state donated.

    Raises [Invalid_argument] as {!loss} does, and as [Vega.adamw_step] does. *)
