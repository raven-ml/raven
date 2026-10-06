(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Hamiltonian Monte Carlo over many chains, with one trajectory length.

    Every chain shares one step size, one trajectory length and one geometry
    ({!Gaussian}), and moves with unit metric in the geometry's whitened
    coordinates. A transition draws each chain's momentum, takes [n] leapfrog
    steps, and accepts the trajectory's end with the Metropolis probability
    [min (1, exp (H0 - H))]. [n] is the length, jittered uniformly in [(0, 2)]
    times its value by one draw the chains share, over the step size, rounded up
    and at most [1024], so every chain takes the same number of steps (Sountsov,
    Carroll and Hoffman 2024). A chain whose step leaves the reals or whose
    energy error exceeds [1000] diverges: it is held at its last point, the
    density evaluated again there, and rejected.

    {b Keys.} Transition [n] of a run has the key [k = Nx.Rng.fold_in run n].
    Chain [i] has row [i] of
    [Nx.Rng.split_batch ~n:chains (Nx.Rng.fold_in k 0)], so its randomness does
    not depend on the chain count; the jitter draws from [Nx.Rng.fold_in k 1].
    {!warmup} and {!sample} fold the state's draw counter into their key and
    advance it, so [a + b] draws are [a] draws then [b], and no two transitions
    of a run share a key. *)

type ('u, 'f) state = private {
  position : 'u;  (** Each chain's position. *)
  lp : (float, 'f) Nx.t;  (** The log density there, shape [[chain]]. *)
  grad : 'u;  (** Its gradient there. *)
  step_size : (float, 'f) Nx.t;  (** The step size, a scalar. *)
  length : (float, 'f) Nx.t;
      (** The trajectory length in whitened coordinates, a scalar. *)
  geometry : ('u, 'f) Gaussian.t;  (** The chains' Gaussian. *)
  stats : 'f Stats.t;  (** The last transition's statistics. *)
  draw : Nx.int32_t;  (** The number of transitions taken. *)
  accept : (float, 'f) Nx.t;  (** The acceptance warmup aims for. *)
}
(** The type for states of chains over positions ['u]. *)

type 'f stats = 'f Stats.t
(** The type for statistics of transitions. [acceptance] is the Metropolis
    probability of the trajectory's end, [0] at a divergence, and [saturated] is
    [false]. *)

val ptree : 'u Nx.Ptree.t -> ('u, 'f) state Nx.Ptree.t
(** [ptree u] is the structure of states over [u]. *)

val stats : (float, 'f) Nx.dtype -> 'f stats Nx.Ptree.t
(** [stats dtype] is {!Stats.ptree}[ dtype]. *)

val init :
  'u Nx.Ptree.t ->
  ?accept:float ->
  ?rank:int ->
  ?geometry:('u, 'f) Gaussian.t ->
  ('u -> (float, 'f) Nx.t) ->
  'u ->
  ('u, 'f) state
(** [init u ?accept ?rank ?geometry lp start] is the state of chains at [start],
    a position. [accept], the harmonic mean acceptance warmup aims for, defaults
    to [0.8], and [rank], the directions warmup fits beside the diagonal, to
    [2]; a rank above a chain's number of float elements is that number. The
    geometry is [geometry], or a diagonal centred on the chains' mean whose
    variance is the inverse of the gradient's root mean square over the chains
    at the start. The step size and the length are [1].

    Raises [Invalid_argument] if [accept] is not in [(0, 1)], [rank < 0], or if
    [lp] does not return one log density per chain, as in
    [Norn.Hmc.init: the density returned shape [4; 8] for a position of 4
     chains; a density returns one log density per chain, shape [4]]. With
    several chains it evaluates [lp] again on the chains reversed, and raises
    naming the first row whose log density differs. *)

val step :
  'u Nx.Ptree.t ->
  ('u -> (float, 'f) Nx.t) ->
  Nx.Rng.t ->
  ('u, 'f) state ->
  ('u, 'f) state
(** [step u lp k s] is one transition of every chain from [s].

    Raises [Invalid_argument] if [lp] is NaN or [+inf] at a finite position,
    naming the chain. *)

val warmup :
  'u Nx.Ptree.t ->
  ('u -> (float, 'f) Nx.t) ->
  Nx.Rng.t ->
  steps:int ->
  ('u, 'f) state ->
  ('u, 'f) state
(** [warmup u lp k ~steps s] is [s] after [steps] transitions that tune the step
    size, the length and the geometry, by Stan's schedule scaled to [steps]: an
    initial buffer, windows that double in length, and a final buffer.

    - Dual averaging (Nesterov 2009; Hoffman and Gelman 2014) sets the step size
      for a harmonic mean acceptance over the chains of [s.accept], restarting
      at each window. A chain that never accepts drives the mean to [0], so the
      step shrinks until every chain moves.
    - The length maximises the change in the chains' expected squared jumped
      distance (ChEES; Hoffman, Radul and Sountsov 2021) in whitened
      coordinates: each transition estimates the criterion's gradient in log
      length from the chains, weighted by acceptance, and Adam takes a step of
      [0.025] along it. The final length is the iterates' polynomial average.
    - At each window's end the geometry is refitted to every chain's draws and
      gradients of the window by the Fisher divergence (Seyboldt, Carlson and
      Carpenter 2026): a diagonal scale [sqrt (sd x / sd score)] per element,
      then its directions.

    Transition [n] of the run, counted by [s.draw], has the key
    [Nx.Rng.fold_in k n], as in {!sample}, so warmup and sampling may share a
    key.

    Raises [Invalid_argument] if [steps < 0]. *)

val sample :
  'u Nx.Ptree.t ->
  ('u -> (float, 'f) Nx.t) ->
  Nx.Rng.t ->
  draws:int ->
  ('u, 'f) state ->
  ('u, 'f) state * 'u Draws.t * 'f stats Draws.t
(** [sample u lp k ~draws s] is the state after [draws] transitions, their
    positions and their statistics. Transition [n] of the run, counted by
    [s.draw], has the key [Nx.Rng.fold_in k n].

    Raises [Invalid_argument] if [draws < 1]. *)
