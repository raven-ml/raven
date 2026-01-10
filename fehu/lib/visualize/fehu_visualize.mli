(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Visualization sinks, encoders, and rollout helpers. *)

module Overlay = Overlay
module Video = Wrapper_video
module Sink = Sink

val push : Sink.t -> Fehu.Render.t -> unit
(** Push a single frame to the sink. *)

val push_many : Sink.t -> Fehu.Render.t array -> unit
(** Push multiple frames sequentially. *)

val record_rollout :
  env:('obs, 'act, Fehu.Render.t) Fehu.Env.t ->
  policy:('obs -> 'act * float option * float option) ->
  steps:int ->
  ?overlay:Overlay.t ->
  sink:Sink.t ->
  unit ->
  unit
(** Record a rollout from a single environment to the sink. *)

val record_evaluation :
  vec_env:('obs, 'act, Fehu.Render.t) Fehu.Vector_env.t ->
  policy:('obs array -> 'act array * float array option * float array option) ->
  n_episodes:int ->
  ?max_steps:int ->
  layout:[ `Single_each | `NxM_grid of int * int ] ->
  ?overlay:Overlay.t ->
  sink:Sink.t ->
  unit ->
  Fehu.Training.stats
(** Evaluate a vectorized policy while recording frames. *)
