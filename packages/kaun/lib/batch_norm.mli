(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Batch normalization layers (Ioffe and Szegedy, 2015).

    A batch-norm layer is two structures: trainable parameters {!type:t} (the
    affine [gamma] and [beta], differentiated and optimized like any other
    parameters) and running statistics {!Stats.t} (per-feature mean and
    variance, never differentiated, updated by every training forward). Both are
    records with payload holes; this module is the {!Nx.Ptree.Uniform} instance
    of its parameters and {!Stats} that of its statistics.

    {!apply} in training mode normalizes with the current batch's statistics and
    returns updated running statistics; in eval mode it normalizes with the
    running statistics and returns them unchanged. A training step threads the
    updated statistics out of the objective through {!Rune.value_and_grad_aux}'s
    auxiliary channel — they ride through differentiation undifferentiated:

    {[
    let model = Kaun.ptree (module Model) in
    let step (params, stats, ostate) =
      let objective p =
        let pred, stats' = Model.forward p stats ~training:true x in
        (Loss.mse pred y, stats')
      in
      let loss, grads, stats' = Rune.value_and_grad_aux model objective params in
      let params, ostate =
        Vega.adam_step model ~lr:1e-3 ostate ~params ~grads
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

type 'a t = { gamma : 'a; beta : 'a }
(** The type for batch-norm parameters over payload ['a]: per-feature scale
    [gamma] and shift [beta], at tensor payloads each of shape
    [[| features |]]. *)

val map : ('a -> 'b) -> 'a t -> 'b t
(** [map f p] is [p] with [f] applied to [gamma] and [beta]. [map (Nx.cast dt)]
    converts a layer's precision; the cast is differentiable through Rune. *)

val map2 : ('a -> 'b -> 'c) -> 'a t -> 'b t -> 'c t
(** [map2 f p q] combines [p] and [q] leafwise with [f]. *)

val iter : ('a -> unit) -> 'a t -> unit
(** [iter f p] applies [f] to [gamma] and [beta], in that order. *)

val fold : (string -> 'acc -> 'a -> 'acc) -> 'acc -> 'a t -> 'acc
(** [fold f acc p] reduces [p] leafwise; leaf paths are ["gamma"] and
    ["beta"]. *)

val fold2 : (string -> 'acc -> 'a -> 'b -> 'acc) -> 'acc -> 'a t -> 'b t -> 'acc
(** [fold2 f acc p q] is like {!fold} across two layers. *)

val names : 'a t -> string t
(** [names p] is [{ gamma = "gamma"; beta = "beta" }]. *)

(** {1:stats Running statistics} *)

(** Running statistics: the exponential moving averages of per-feature batch
    mean and (population) variance maintained by training-mode {!apply} and
    consumed by eval-mode {!apply}. Not parameters — never differentiate or
    optimize them; thread them through the training loop as auxiliary state. *)
module Stats : sig
  type 'a t = { mean : 'a; var : 'a }
  (** The type for running statistics over payload ['a], at tensor payloads
      each of shape [[| features |]]. The exponential moving average
      accumulates small increments, so keep statistics at float32 even when
      the parameters are half precision. *)

  val map : ('a -> 'b) -> 'a t -> 'b t
  (** [map f s] is [s] with [f] applied to [mean] and [var]. *)

  val map2 : ('a -> 'b -> 'c) -> 'a t -> 'b t -> 'c t
  (** [map2 f s s'] combines [s] and [s'] leafwise with [f]. *)

  val iter : ('a -> unit) -> 'a t -> unit
  (** [iter f s] applies [f] to [mean] and [var], in that order. *)

  val fold : (string -> 'acc -> 'a -> 'acc) -> 'acc -> 'a t -> 'acc
  (** [fold f acc s] reduces [s] leafwise; leaf paths are ["mean"] and
      ["var"]. *)

  val fold2 :
    (string -> 'acc -> 'a -> 'b -> 'acc) -> 'acc -> 'a t -> 'b t -> 'acc
  (** [fold2 f acc s s'] is like {!fold} across two statistics. *)

  val names : 'a t -> string t
  (** [names s] is [{ mean = "mean"; var = "var" }]. *)
end

(** {1:constructors Constructors} *)

val init : features:int -> Nx.float32_t t * Nx.float32_t Stats.t
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
  (float, 'b) Nx.t t ->
  (float, 'b) Nx.t Stats.t ->
  training:bool ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t * (float, 'b) Nx.t Stats.t
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
