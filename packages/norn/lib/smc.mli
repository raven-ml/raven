(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Sequential Monte Carlo by tempering.

    The particles move from the prior to the posterior through the tempered
    densities [prior + β likelihood], [β] rising from [0] to [1], and the
    product of their mean incremental weights estimates the evidence.

    {b Temperatures.} From [β] the next temperature is [1] if the incremental
    weights [exp ((1 - β) L)] keep an effective sample size of half the
    particles; otherwise it is the [β'] above [β] and below [1] where they keep
    exactly half, found by {!Jera.Root.bracket}.

    {b Moves.} Each step is waste-free (Dau and Chopin 2022): it resamples
    [resampled] ancestors, [M], from the [N] particles systematically by their
    incremental weights and runs [N / M - 1] moves from each on the tempered
    density, keeping every state, so the [N] particles are [M] chains of [N / M]
    states, equally weighted. Moves are in the coordinates whitened by the
    reweighted particles' mean and covariance, by one of two kernels ({!move}):

    - [Hmc], Hamiltonian transitions of one step size and one length, the length
      jittered uniformly in [(0, 2)] times its value, as {!Hmc.step} takes them.
      Both start at [1] and are tuned between temperatures, which are close, and
      fixed within one. The step size takes one Robbins-Monro step on its log
      toward a mean acceptance of [0.8]: [log eps] gains the mean acceptance
      less [0.8]. The length takes one Adam step of the ChEES criterion per
      move, as {!Hmc.warmup} tunes it, one search across the run, each gradient
      estimated from the [M] chains. The densities need a gradient.
    - [Slice], hit-and-run slice steps (Neal 2003) along directions of unit
      length; nothing is tuned and the densities need no derivative.

    {b Error.} Each temperature's mean incremental weight has a variance
    estimated from its [M] chains read as one stationary chain: their pooled
    autocovariances summed by Geyer's initial monotone sequence (Dau and Chopin
    2022). {!Evidence.error} is the root of the relative variances summed over
    temperatures, whose errors are nearly independent when the chains are long
    beside their number.

    {b Densities.} [prior] and [likelihood] are densities over positions of the
    same structure, evaluated on the [M] chains of a move. The prior is
    normalised, so the evidence is the data's probability.

    {b Keys.} {!run}'s step [s] has the key [k_s = Nx.Rng.fold_in k s]: it
    resamples from [Nx.Rng.fold_in k_s 0], and its moves draw from
    [k_m = Nx.Rng.fold_in k_s 1]. Hamiltonian move [j] is the transition of key
    [Nx.Rng.fold_in k_m j], as {!Hmc.step} draws from its key; chain [i]'s slice
    move [j] draws from [Nx.Rng.fold_in] of row [i] of
    [Nx.Rng.split_batch ~n:M k_m] with [j]. *)

(** The type for the kernels that move the particles. *)
type move =
  | Hmc  (** Hamiltonian transitions; the densities need a gradient. *)
  | Slice  (** Hit-and-run slice steps; no derivative. *)

type ('u, 'f) state = private {
  particles : 'u;  (** The particles, on a leading axis of [N]. *)
  prior : (float, 'f) Nx.t;  (** Their prior log densities, shape [[N]]. *)
  likelihood : (float, 'f) Nx.t;  (** Their log likelihoods. *)
  beta : (float, 'f) Nx.t;  (** The temperature, a scalar in [[0, 1]]. *)
  log_evidence : (float, 'f) Nx.t;  (** [ln Z] of the tempered density. *)
  variance : (float, 'f) Nx.t;  (** The variance of [log_evidence]. *)
  steps : Nx.int32_t;  (** The temperatures taken. *)
  step_size : (float, 'f) Nx.t;
      (** The Hamiltonian moves' step size in whitened coordinates, a scalar. *)
  length : (float, 'f) Nx.t;
      (** Their trajectory length in whitened coordinates, a scalar. *)
  length_moment : (float, 'f) Nx.t;
      (** The second moment of the length's Adam search, a scalar. *)
  resampled : int;  (** [M], the chains of a step. *)
  move : move;  (** The kernel that moves the particles. *)
}
(** The type for states of tempering over positions ['u]. *)

val ptree : 'u Nx.Ptree.t -> ('u, 'f) state Nx.Ptree.t
(** [ptree u] is the structure of states over [u]. It reports [resampled]. *)

val init :
  'u Nx.Ptree.t ->
  ?move:move ->
  ?resampled:int ->
  prior:('u -> (float, 'f) Nx.t) ->
  likelihood:('u -> (float, 'f) Nx.t) ->
  'u ->
  ('u, 'f) state
(** [init u ?move ?resampled ~prior ~likelihood start] is the state at [β = 0]
    of the particles [start], [N] independent draws of the prior. [move]
    defaults to [Hmc]. [resampled] defaults to the largest divisor of [N] not
    above [sqrt N]: the evidence's error is consistent when the chains are long
    beside their number, [M] small beside [sqrt N] (Dau and Chopin 2022, theorem
    2), and a thousand particles make 25 chains of 40.

    Raises [Invalid_argument] if [resampled] does not divide [N] into chains of
    two states or more, or if a density does not return one log density per
    particle, each read from its own row. *)

val step :
  'u Nx.Ptree.t ->
  prior:('u -> (float, 'f) Nx.t) ->
  likelihood:('u -> (float, 'f) Nx.t) ->
  Nx.Rng.t ->
  ('u, 'f) state ->
  ('u, 'f) state
(** [step u ~prior ~likelihood k s] moves the particles of [s] to the next
    temperature. At [β = 1] it moves them on the posterior and leaves the
    evidence as it is.

    Raises [Invalid_argument] through {!Nx.check} if a density is NaN or [+inf]
    at a particle. *)

val evidence : ('u, 'f) state -> ('u, 'f) Evidence.t
(** [evidence s] is the evidence of the tempered density at [s.beta], the
    posterior's at [β = 1], with the particles as its sample. Its
    {!Evidence.stop} is [Converged] at [β = 1], else [Temperature s.beta]. *)

val run :
  'u Nx.Ptree.t ->
  ?move:move ->
  ?resampled:int ->
  budget:int ->
  prior:('u -> (float, 'f) Nx.t) ->
  likelihood:('u -> (float, 'f) Nx.t) ->
  Nx.Rng.t ->
  'u ->
  ('u, 'f) Evidence.t
(** [run u ?move ?resampled ~budget ~prior ~likelihood k start] steps from
    [init] until [β = 1], or for [budget] temperatures, and is the evidence
    then.

    Raises [Invalid_argument] as {!init} does, and if [budget < 1]. *)
