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

val load_heads :
  string -> int list -> (Nx.host Nx.float32_t, Nx.host) head list
(** [load_heads file prefixes] is a head of one linear layer for each of
    [prefixes] from the npz [file], in PyTorch's layout: prefix [k]'s weight
    [k.weight], [[d; k]], read transposed, and its bias [k.bias], [[d]], at
    float32, every head of one [d].

    Raises [Failure] naming [file] and the entry if an entry is missing or of
    another shape, [d] the first prefix's, and as {!Nx_io.load_npz} does. *)

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
  dropout:Model.dropout ->
  keys:'d Nx.Rng.key ->
  ('b, 'd) Model.t ->
  ((float, 's, 'd) Nx.t, 'd) params ->
  'd Batch.t ->
  targets:(float, 't, 'd) Nx.t ->
  'd Nx.float32_t
(** [loss ~dropout ~keys m p b ~targets] is the sum over [b]'s filled places of
    [Σ_k (1 − cos (h_k (s_{1:k}), t))] ({!cosines}), [s] the place's training
    embedding [Model.train ~dropout ~keys] by [m] with [p]'s net,
    and [t] its row of [targets]. An empty place adds [0].

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

val targets :
  string ->
  Nx.host Nx.packed ->
  first:int ->
  Nx.host Nx.int64_t ->
  Nx.host Nx.float32_t
(** [targets file t ~first indices] is the rows [indices] of the targets file
    [file], [t] its {!Nx_io.load_npy}, one row a pixel of the corpus, that a
    run takes at the stream positions [first] to [first + k - 1], [k] the
    length of [indices], cast to float32. They are read as {!Pixels.window}
    reads pixels: by a take of [t] where those positions lie below [t]'s rows,
    the first epoch, and otherwise by {!Nx_io.load_npy_rows} of [file]. Only
    the window's rows are cast, so a float16 file is never cast whole. *)

val keys :
  key:Nx.host Nx.Rng.key -> first:int -> Nx.host Batch.t -> Nx.host Nx.Rng.key
(** [keys ~key ~first b] is the keys of [b]'s places, as {!Model.train} takes
    them, for [b]'s pixels at stream positions from [first]: place [k]'s key is
    [Nx.Rng.fold_in key (first + p)], [p] its pixel's index in [b], or [0] at an
    empty place. A pixel's masks are thus a function of [key] and its stream
    position: the same at that position in any group or batch shape, drawn anew
    at any other, as at a later step or epoch. *)

val shape : pixels:int -> slots:int -> Nx.host Pixels.t -> Batch.shape
(** [shape ~pixels ~slots px] is the shape a distillation of the corpus [px]
    packs its groups at: [pixels] pixels, at most [slots] a row, in rows of
    [256] tokens, of each sensor [⌈1.05 · m · pixels / 256⌉ + 1] rows, [m]
    the mean tokens a pixel of the duplicate-filled ({!Batch.expand} of
    {!Batch.v2_counts}) first [4096] pixels of [px]: their tokens with 5 % to
    spare, and a row more. As for {!Batch.typical}, a group whose pixels hold
    more stops before its pixels are full.

    Raises [Invalid_argument] if [px] has no pixel. *)

val groups :
  key:Nx.host Nx.Rng.key ->
  shape:Batch.shape ->
  groups:int ->
  first:int ->
  Nx.host Pixels.t ->
  Nx.host Nx.float32_t ->
  (Nx.host Nx.float32_t, Nx.host) group * int
(** [groups ~key ~shape ~groups ~first px targets] is a step's [groups]
    groups of the pixels [px] at stream positions from [first], stacked along
    axis [0] ({!Batch.stack}), and the number of [px]'s pixels they took.
    Group [g] is [Batch.pack shape] of [px] duplicate-filled ({!Batch.expand}
    of {!Batch.v2_counts}) from the pixel group [g − 1] stopped at, [0] for
    the first; its keys are {!val-keys} [~key ~first], and each place's
    target is its pixel's row of [targets] ([[n; d]], row [j] pixel [j]'s),
    row [0]'s at an empty place, which the loss leaves out. So the groups
    hold [px]'s pixels below the number taken, each with an observation
    once, in order.

    Raises [Invalid_argument] if [targets] does not have a row for each of
    [px]'s pixels, and as {!Batch.pack} does. *)

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

type recipe = {
  lr : float;  (** AdamW's peak rate. *)
  warmup : float;
      (** The share of the steps the rate rises over, linearly from [0], before
          it falls to [0] along a cosine, as {!Barlow.schedule} rounds it:
          v2's trainer, which would state its own rounding, is not public. *)
  weight_decay : float;
  clip : float;  (** The bound on the gradient's global norm. *)
}
(** The type for distillation recipes: a {!val-step}'s rates and bound. *)

val v2_recipe : recipe
(** [v2_recipe] is v2's distillation's (v2 App. J, Table 9, "Distillation
    configuration"): AdamW at a peak rate of [8e-4] after a linear warmup
    over [0.05] of the steps, then a cosine decay to [0], decay [1e-4], and
    the gradient's global norm clipped to [1.0]. Its dropout is
    {!Model.v2_dropout}, and its AdamW's [b2] the [0.95] {!val-step}
    applies. *)

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
