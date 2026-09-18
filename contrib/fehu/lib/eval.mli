(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Policy evaluation.

    Runs a deterministic or stochastic policy over multiple episodes and reports
    summary statistics. *)

(** {1:types Types} *)

type stats = {
  mean_reward : float;  (** Mean total reward across episodes. *)
  std_reward : float;  (** Standard deviation of total rewards. *)
  mean_length : float;  (** Mean episode length in steps. *)
  n_episodes : int;  (** Number of episodes evaluated. *)
}
(** The type for evaluation statistics. *)

(** {1:running Running} *)

val run :
  ('obs, 'act, 'render) Env.t ->
  policy:('obs -> 'act) ->
  ?n_episodes:int ->
  ?max_steps:int ->
  unit ->
  stats
(** [run env ~policy ()] evaluates [policy] over [n_episodes] (default [10])
    episodes of at most [max_steps] (default [1000]) steps each. The environment
    is reset between episodes. *)
