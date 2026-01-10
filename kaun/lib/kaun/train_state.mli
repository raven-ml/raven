(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Unified training state for Kaun. *)

module Snapshot = Checkpoint.Snapshot

type t = {
  step : int;
  params : Ptree.t;
  opt_state : Optimizer.state;
  rng : Rune.Rng.key;
  metrics : Metrics.Collection.t option;
}

val init :
  model:Layer.module_ ->
  optimizer:Optimizer.algorithm ->
  ?metrics:Metrics.Collection.t ->
  rngs:Rune.Rng.key ->
  dtype:(float, 'layout) Rune.dtype ->
  unit ->
  t
(** Initialize the training state by calling [model.init], seeding the
    optimizer, setting [step] to [0], and attaching optional metrics. *)

val create :
  ?step:int ->
  params:Ptree.t ->
  opt_state:Optimizer.state ->
  rng:Rune.Rng.key ->
  ?metrics:Metrics.Collection.t ->
  unit ->
  t
(** Assemble a state from individual components. *)

val apply_gradients : optimizer:Optimizer.algorithm -> grads:Ptree.t -> t -> t
(** Apply optimiser updates in-place to [params], refresh [opt_state], and bump
    [step] by one. *)

val next_rng : t -> Rune.Rng.key * t
(** Split the RNG, returning the next key alongside the updated state. *)

val reset_metrics : t -> t
(** Reset the attached metrics collection if present. *)

val update_metrics :
  t ->
  predictions:(float, 'layout) Rune.t ->
  targets:(_, 'layout) Rune.t ->
  ?loss:(float, 'layout) Rune.t ->
  ?weights:(float, 'layout) Rune.t ->
  unit ->
  unit
(** Update metrics with optional loss information; no-op when metrics are
    absent. *)

val compute_metrics : t -> (string * float) list
(** Compute current metric values or return [] if none attached. *)

val to_snapshot :
  ?encode_metrics:(Metrics.Collection.t -> Snapshot.t) -> t -> Snapshot.t
(** Encode the state into a checkpoint snapshot, tagging the schema for
    versioning. *)

val of_snapshot :
  optimizer:Optimizer.algorithm ->
  ?decode_metrics:(Snapshot.t -> (Metrics.Collection.t, string) result) ->
  Snapshot.t ->
  (t, string) result
(** Decode a snapshot back into a state using the supplied optimiser algorithm.
*)

val save :
  repository:Checkpoint.repository ->
  ?step:int ->
  ?tags:string list ->
  ?metadata:Checkpoint.metadata ->
  ?encode_metrics:(Metrics.Collection.t -> Snapshot.t) ->
  t ->
  (Checkpoint.manifest, Checkpoint.error) result
(** Persist the state into the repository under the ["state"] artifact slug. *)

val load :
  repository:Checkpoint.repository ->
  step:int ->
  optimizer:Optimizer.algorithm ->
  ?decode_metrics:(Snapshot.t -> (Metrics.Collection.t, string) result) ->
  unit ->
  (t, string) result
(** Restore a saved state for the requested step, assuming the caller supplies
    the matching optimiser algorithm. *)
