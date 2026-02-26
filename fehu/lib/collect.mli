(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Trajectory collection from environments.

    Collects sequential agent-environment interactions into structure-of-arrays
    form for batch processing. Handles automatic resets on episode boundaries
    and records both the current and next observation at each timestep. *)

(** {1:types Types} *)

type ('obs, 'act) t = {
  observations : 'obs array;  (** States before each action. *)
  actions : 'act array;  (** Actions taken. *)
  rewards : float array;  (** Scalar rewards received. *)
  next_observations : 'obs array;  (** States after each action. *)
  terminated : bool array;  (** Natural episode endings. *)
  truncated : bool array;  (** Forced episode endings. *)
  infos : Info.t array;  (** Per-step metadata. *)
  log_probs : float array option;  (** Policy log-probabilities. *)
  values : float array option;  (** Value estimates. *)
}
(** The type for trajectories. All arrays have the same length. Optional fields
    are [None] when the policy does not provide them. *)

(** {1:accessors Accessors} *)

val length : ('obs, 'act) t -> int
(** [length traj] is the number of transitions in [traj]. *)

(** {1:combining Combining} *)

val concat : ('obs, 'act) t list -> ('obs, 'act) t
(** [concat trajs] concatenates [trajs] into a single trajectory. Optional
    fields are kept only if present in all inputs.

    Raises [Invalid_argument] if [trajs] is empty. *)

(** {1:collecting Collecting} *)

val rollout :
  ('obs, 'act, 'render) Env.t ->
  policy:('obs -> 'act * float option * float option) ->
  n_steps:int ->
  ('obs, 'act) t
(** [rollout env ~policy ~n_steps] collects [n_steps] transitions.

    Resets [env] at the start and automatically on episode boundaries
    (terminated or truncated). The [policy] receives the current observation and
    returns [(action, log_prob_opt, value_opt)]. *)

val episodes :
  ('obs, 'act, 'render) Env.t ->
  policy:('obs -> 'act * float option * float option) ->
  n_episodes:int ->
  ?max_steps:int ->
  unit ->
  ('obs, 'act) t list
(** [episodes env ~policy ~n_episodes ()] collects complete episodes, one
    trajectory per episode. Each episode runs until termination, truncation, or
    [max_steps] (default [1000]). *)
