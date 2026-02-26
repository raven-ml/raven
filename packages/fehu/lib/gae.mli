(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Generalized Advantage Estimation.

    Correctly handles the distinction between terminated and truncated episodes.
    On termination, the bootstrap value is zero. On truncation, the bootstrap
    value comes from [next_values]. *)

(** {1:gae GAE} *)

val compute :
  rewards:float array ->
  values:float array ->
  terminated:bool array ->
  truncated:bool array ->
  next_values:float array ->
  gamma:float ->
  lambda:float ->
  float array * float array
(** [compute ~rewards ~values ~terminated ~truncated ~next_values
      ~gamma ~lambda] is [(advantages, returns)].

    [next_values.(t)] is V(s_{{t+1}}). When [terminated.(t)] is
    [true], the bootstrap value is zero and the GAE trace resets.
    When [truncated.(t)] is [true], the bootstrap value is
    [next_values.(t)] and the trace resets for the new episode.
    Otherwise, continuation uses the next step's value.

    Raises [Invalid_argument] if array lengths differ. *)

val compute_from_values :
  rewards:float array ->
  values:float array ->
  terminated:bool array ->
  truncated:bool array ->
  last_value:float ->
  gamma:float ->
  lambda:float ->
  float array * float array
(** [compute_from_values ~rewards ~values ~terminated ~truncated ~last_value
     ~gamma ~lambda] is [(advantages, returns)].

    Convenience wrapper around {!compute} that builds [next_values] from
    [values] and [last_value]: [next_values.(t) = values.(t+1)] for [t < n-1],
    and [next_values.(n-1) = last_value].

    Raises [Invalid_argument] if array lengths differ. *)

(** {1:returns Monte Carlo returns} *)

val returns :
  rewards:float array ->
  terminated:bool array ->
  truncated:bool array ->
  gamma:float ->
  float array
(** [returns ~rewards ~terminated ~truncated ~gamma] computes discounted
    cumulative returns. The accumulation resets at terminal or truncated states.
*)

(** {1:normalize Normalization} *)

val normalize : ?eps:float -> float array -> float array
(** [normalize arr] is a copy of [arr] with zero mean and unit variance. [eps]
    (default [1e-8]) prevents division by zero. *)
