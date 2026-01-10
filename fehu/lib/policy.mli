(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Helper combinators for building policies. *)

type ('obs, 'act) t = 'obs -> 'act * float option * float option
(** Policy returning an action with optional log prob and value estimate. *)

val deterministic : ('obs -> 'act) -> ('obs, 'act) t
(** Wrap a deterministic action function as a policy. *)

val random : ?rng:Rune.Rng.key -> ('obs, 'act, 'render) Env.t -> ('obs, 'act) t
(** Epsilon-free stochastic policy that samples uniformly from the action space.
    Reuses the environment RNG when [rng] is omitted. *)

val greedy_discrete :
  ('obs, Space.Discrete.element, 'render) Env.t ->
  score:('obs -> float array) ->
  ('obs, Space.Discrete.element) t
(** Build a greedy policy for discrete action spaces.

    The [score] function must return per-action scores (e.g., Q-values). The
    policy selects the highest scoring action, respecting the space's offset. *)
