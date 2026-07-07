(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Batch normalization layers (Ioffe and Szegedy, 2015).

    A batch-norm layer is two structures: trainable parameters {!type:params}
    (the affine [gamma] and [beta], differentiated and optimized like any other
    parameters) and running statistics {!Stats.stats} (per-feature mean and
    variance, never differentiated, updated by every training forward). Both are
    plain records with structural traversals; this module is the {!Nx.Ptree.S}
    instance of its parameters and {!Stats} that of its statistics.

    {!apply} in training mode normalizes with the current batch's statistics and
    returns updated running statistics; in eval mode it normalizes with the
    running statistics and returns them unchanged. A training step threads the
    updated statistics out of the objective through {!Rune.value_and_grad_aux}'s
    auxiliary channel — they ride through differentiation undifferentiated:

    {[
    let step (params, stats, ostate) =
      let objective p =
        let pred, stats' = Model.forward p stats ~training:true x in
        (Loss.mse pred y, stats')
      in
      let loss, grads, stats' =
        Rune.value_and_grad_aux (module Model) objective params
      in
      let params, ostate =
        Vega.adam_step (module Model) ~lr:1e-3 ostate ~params ~grads
      in
      ((params, stats', ostate), Nx.item [] loss)
    ]}

    Evaluation reuses the same forward with [~training:false] and discards the
    returned statistics. Statistics checkpoint like parameters, under their own
    prefix:

    {[
    Checkpoint.concat
      [
        Checkpoint.of_params (module Model) ~prefix:"model" params;
        Checkpoint.of_params (module Model.Stats) ~prefix:"stats" stats;
      ]
    ]} *)

(** {1:params Parameters} *)

type 'b params = { gamma : (float, 'b) Nx.t; beta : (float, 'b) Nx.t }
(** The type for batch-norm parameters with float dtype layout ['b]: per-feature
    scale [gamma] and shift [beta], each of shape [[| features |]]. *)

type t = Nx.float32_elt params
(** The type for single-precision batch-norm parameters, the common case. *)

val map : ('a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) -> 'b params -> 'b params
(** [map f p] is [p] with [f] applied to [gamma] and [beta]. *)

val map2 :
  ('a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) ->
  'b params ->
  'b params ->
  'b params
(** [map2 f p q] combines [p] and [q] leafwise with [f]. *)

val iter : ('a 'c. ('a, 'c) Nx.t -> unit) -> 'b params -> unit
(** [iter f p] applies [f] to [gamma] and [beta], in that order. *)

val astype : (float, 'c) Nx.dtype -> 'b params -> 'c params
(** [astype dt p] is [p] with every parameter leaf cast to [dt]. Differentiable
    through Rune: gradients flow back at each original leaf's dtype, so an
    astype of float32 parameters inside a loss function yields float32
    gradients. *)

val names : 'b params -> string list
(** [names p] is [["gamma"; "beta"]], pairing leaves in traversal order (see
    {!Checkpoint.Named}). *)

(** {1:stats Running statistics} *)

(** Running statistics: the exponential moving averages of per-feature batch
    mean and (population) variance maintained by training-mode {!apply} and
    consumed by eval-mode {!apply}. Not parameters — never differentiate or
    optimize them; thread them through the training loop as auxiliary state. *)
module Stats : sig
  type 'b stats = { mean : (float, 'b) Nx.t; var : (float, 'b) Nx.t }
  (** The type for running statistics with float dtype layout ['b], each of
      shape [[| features |]]. *)

  type t = Nx.float32_elt stats
  (** The type for single-precision running statistics, the common case. The
      exponential moving average accumulates small increments, so keep
      statistics at float32 even when the parameters are half precision. *)

  val map : ('a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) -> 'b stats -> 'b stats
  (** [map f s] is [s] with [f] applied to [mean] and [var]. *)

  val map2 :
    ('a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) ->
    'b stats ->
    'b stats ->
    'b stats
  (** [map2 f s s'] combines [s] and [s'] leafwise with [f]. *)

  val iter : ('a 'c. ('a, 'c) Nx.t -> unit) -> 'b stats -> unit
  (** [iter f s] applies [f] to [mean] and [var], in that order. *)

  val astype : (float, 'c) Nx.dtype -> 'b stats -> 'c stats
  (** [astype dt s] is [s] with [mean] and [var] cast to [dt]. *)

  val names : 'b stats -> string list
  (** [names s] is [["mean"; "var"]], pairing leaves in traversal order (see
      {!Checkpoint.Named}). *)
end

(** {1:constructors Constructors} *)

val init : features:int -> t * Stats.t
(** [init ~features] is a fresh layer for [features]-dimensional features:
    parameters with [gamma] all ones and [beta] all zeros, and statistics with
    [mean] all zeros and [var] all ones (so an untrained layer's eval mode is
    close to the identity of a standardized input).

    Raises [Invalid_argument] if [features <= 0]. *)

(** {1:applying Applying} *)

val apply :
  ?axis:int ->
  ?momentum:float ->
  ?eps:float ->
  'b params ->
  'b Stats.stats ->
  training:bool ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t * 'b Stats.stats
(** [apply p stats ~training x] is [(y, stats')] where [y] normalizes [x] per
    feature and applies the affine transform:
    [y = gamma * (x - mean) / sqrt (var + eps) + beta].

    Features live on [axis] (default [-1], the last axis; negative counts from
    the end); statistics are taken over all other axes together. [axis] must
    have size [features].

    With [training = true], [mean] and [var] are the current batch's mean and
    population variance — gradients flow through them to [x] — and [stats'] is
    the updated running statistics [momentum * stats + (1 - momentum) * batch].
    The update is detached: no gradient flows through [stats'], so it can safely
    ride the auxiliary channel of {!Rune.value_and_grad_aux} (see the module
    preamble).

    With [training = false], [mean] and [var] are [stats] and [stats' == stats].

    For half and quarter precision inputs (float16, bfloat16, float8) the
    statistics and normalization are computed in a float32 island: [x] (and, in
    eval mode, [stats]) are upcast, normalized at float32, and the normalized
    values are cast back to [x]'s dtype before the [gamma]/[beta] affine
    transform. Float32 and float64 inputs use their own dtype throughout,
    exactly as if the island were absent.

    [momentum] defaults to [0.99] (per training step; unused in eval mode) and
    [eps] to [1e-5].

    Raises [Invalid_argument] if [x] has fewer than 2 axes, [axis] is out of
    bounds, the feature axis's size does not match [p], [momentum] is outside
    \[[0];[1]\], or [eps] is not positive. *)
