(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Nested sampling, by slice moves on the prior.

    Nested sampling (Skilling 2006) estimates the evidence [Z = ∫ L π] as a sum
    over shells of prior volume. It keeps [n] live points drawn from the prior;
    each step deletes the [batch] of lowest likelihood and replaces each by a
    draw from the prior above the highest deleted likelihood. Here a replacement
    starts at a surviving live point drawn at random and takes three hit-and-run
    slice moves (Neal 2003) per float element of a point on the prior,
    constrained above that likelihood, along directions of unit length in the
    coordinates whitened by the survivors' mean and covariance (nested slice
    sampling; Yallup et al. 2026). Fewer moves leave the replacements correlated
    with their starts: on a Gaussian in ten dimensions one move per element
    biases [ln Z] by [+0.39] nats, two by [+0.14], and three by no measurable
    amount. Likelihoods compare with a rank, uniform per point, as a second key,
    so ties keep the live set uniform (Fowlie, Handley and Su 2021).

    {b Volumes.} Deleting the [k] lowest of [n] live points shrinks the prior
    volume by the order statistics of [n] uniforms: the [j]-th deleted, from the
    lowest, lowers [ln X] by [1 / (n - j + 1)] in expectation. When the live
    points' largest possible contribution, the highest live likelihood times the
    volume left, would change [ln Z] by less than [tolerance], the run stops and
    adds them as dead points with their order statistics' volumes, the highest
    taking all the volume left. {!Evidence.error} is the spread of [ln Z] over
    100 simulated shrinkage sequences; [sqrt (H / n)] (Skilling 2006) is its
    limit at a constant count.

    {b Densities.} [prior] and [likelihood] are densities over positions of the
    same structure, evaluated on the [batch] points being replaced. The prior is
    normalised, so the evidence is the data's probability.

    {b Keys.} {!run}'s step [s] has the key
    [Nx.Rng.fold_in (Nx.Rng.fold_in k 0) s], and its {!evidence} the key
    [Nx.Rng.fold_in k 1]. A step's replacement [i] has row [i] of
    [Nx.Rng.split_batch ~n:batch] of the step's key. *)

type ('u, 'f) state = private {
  live : 'u;  (** The live points, on a leading axis of [n]. *)
  prior : (float, 'f) Nx.t;  (** Their prior log densities, shape [[n]]. *)
  likelihood : (float, 'f) Nx.t;  (** Their log likelihoods. *)
  rank : (float, 'f) Nx.t;  (** Their ranks, in [(0, 1)]. *)
  dead : 'u;
      (** The dead points, lowest first, on a leading axis of [budget * batch];
          the rows past [deaths] are padding. *)
  dead_likelihood : (float, 'f) Nx.t;
      (** Their log likelihoods, [-inf] in the padding. *)
  shrinkage : (float, 'f) Nx.t;
      (** The expected fall in [ln X] at each, [0] in the padding. *)
  deaths : Nx.int32_t;  (** The number of dead points. *)
  log_volume : (float, 'f) Nx.t;  (** The expected [ln X] left. *)
  log_evidence : (float, 'f) Nx.t;  (** [ln Z] of the dead points. *)
  steps : Nx.int32_t;
      (** The steps taken, at most [budget]: at [budget] the dead buffer is full
          and {!step} leaves the state as it is. *)
  tolerance : (float, 'f) Nx.t;  (** The stopping tolerance in [ln Z]. *)
  batch : int;  (** The points a step deletes. *)
  budget : int;  (** The steps the dead buffer holds. *)
}
(** The type for states of nested sampling over positions ['u]. *)

val ptree : 'u Nx.Ptree.t -> ('u, 'f) state Nx.Ptree.t
(** [ptree u] is the structure of states over [u]. It reports [batch] and
    [budget]. *)

val init :
  'u Nx.Ptree.t ->
  ?batch:int ->
  ?tolerance:float ->
  budget:int ->
  prior:('u -> (float, 'f) Nx.t) ->
  likelihood:('u -> (float, 'f) Nx.t) ->
  'u ->
  ('u, 'f) state
(** [init u ?batch ?tolerance ~budget ~prior ~likelihood live] is the state with
    the live points [live], [n] independent draws of the prior. [batch] defaults
    to [n / 2] and [tolerance] to [1e-3]; the dead buffer holds [budget] steps.

    Raises [Invalid_argument] if [batch < 1] or [batch >= n], [budget < 1],
    [tolerance] is not positive, or if a density does not return one log density
    per point, each read from its own row. *)

val step :
  'u Nx.Ptree.t ->
  prior:('u -> (float, 'f) Nx.t) ->
  likelihood:('u -> (float, 'f) Nx.t) ->
  Nx.Rng.t ->
  ('u, 'f) state ->
  ('u, 'f) state
(** [step u ~prior ~likelihood k s] deletes [s.batch] live points and replaces
    them, or is [s] if [s] has taken [s.budget] steps.

    Raises [Invalid_argument] through {!Nx.check} if [likelihood] is NaN or
    [+inf] at a point. *)

val evidence :
  'u Nx.Ptree.t -> Nx.Rng.t -> ('u, 'f) state -> ('u, 'f) Evidence.t
(** [evidence u k s] is the evidence of the dead points of [s] and of its live
    points added as dead, [k] simulating the shrinkage sequences of its error.
    Its {!Evidence.sample} is every dead point, padding included at weight zero,
    and its {!Evidence.stop} is [Converged], or [Remaining r] if the live points
    could still add [r] nats, the tolerance or more, to [ln Z]. *)

val run :
  'u Nx.Ptree.t ->
  ?batch:int ->
  ?tolerance:float ->
  budget:int ->
  prior:('u -> (float, 'f) Nx.t) ->
  likelihood:('u -> (float, 'f) Nx.t) ->
  Nx.Rng.t ->
  'u ->
  ('u, 'f) Evidence.t
(** [run u ?batch ?tolerance ~budget ~prior ~likelihood k live] steps from
    [init] until the live points' largest possible contribution changes [ln Z]
    by less than [tolerance], or for [budget] steps, and is the evidence then.
    At the default batch a step lowers [ln X] by about [0.69], so a posterior
    [H] nats from the prior needs about [H / 0.69] steps and a few more.

    Raises [Invalid_argument] as {!init} does. *)
