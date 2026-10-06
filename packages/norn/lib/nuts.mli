(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The No-U-Turn sampler over many chains.

    Multinomial NUTS with biased progressive sampling and the generalised U-turn
    criterion (Betancourt 2017), checked around each merged subtree and between
    its two halves, as Stan has since 2019. A transition diverges when the
    energy error exceeds [1000] or a position is not finite.

    Every chain has its own step size and geometry ({!Gaussian}), and moves with
    unit metric in its geometry's whitened coordinates. The chains move in lock
    step: one loop over leapfrog steps, each one density evaluation over all
    chains, until every chain has turned, diverged or filled its depth. The tree
    is built iteratively (Phan, Pradhan and Jankowiak 2019), with a checkpoint
    per level. A chain that has stopped is held: the density is evaluated again
    at its last point and its results are discarded, so no density is evaluated
    at a point a chain did not reach.

    {b Keys.} A transition gives chain [i] row [i] of
    [Nx.Rng.split_batch ~n:chains k], so a chain's randomness does not depend on
    the chain count. {!warmup} and {!sample} fold the state's draw counter into
    their key and advance it, so [a + b] draws are [a] draws then [b], and no
    two transitions of a run share a key. *)

type ('u, 'f) state = private {
  position : 'u;  (** Each chain's position. *)
  lp : (float, 'f) Nx.t;  (** The log density there, shape [[chain]]. *)
  grad : 'u;  (** Its gradient there. *)
  step_size : (float, 'f) Nx.t;  (** Each chain's step size. *)
  geometry : ('u, 'f) Gaussian.t;
      (** Each chain's Gaussian, stacked on the chain axis. *)
  stats : 'f Stats.t;  (** The last transition's statistics. *)
  draw : Nx.int32_t;  (** The number of transitions taken. *)
  accept : (float, 'f) Nx.t;  (** The acceptance warmup aims for. *)
  max_depth : int;  (** The deepest tree a transition builds. *)
}
(** The type for states of chains over positions ['u]. *)

val ptree : 'u Nx.Ptree.t -> ('u, 'f) state Nx.Ptree.t
(** [ptree u] is the structure of states over [u]. It reports [max_depth]. *)

val stats : (float, 'f) Nx.dtype -> 'f Stats.t Nx.Ptree.t
(** [stats dtype] is {!Stats.ptree}[ dtype]. *)

val init :
  'u Nx.Ptree.t ->
  ?max_depth:int ->
  ?accept:float ->
  ?rank:int ->
  ?geometry:('u, 'f) Gaussian.t ->
  ('u -> (float, 'f) Nx.t) ->
  'u ->
  ('u, 'f) state
(** [init u ?max_depth ?accept ?rank ?geometry lp start] is the state of chains
    at [start], a position. [max_depth] defaults to [10], [accept], the
    acceptance warmup aims for, to [0.8], and [rank], the directions warmup fits
    beside the diagonal, to [0]. Each chain's geometry is [geometry], which
    seeds every chain, or a diagonal whose variance is the inverse of the
    absolute gradient at the start; its step size is [1].

    Raises [Invalid_argument] if [max_depth < 1], [accept] is not in [(0, 1)],
    [rank < 0], or if [lp] does not return one log density per chain, as in
    [Norn.Nuts.init: the density returned shape [4; 8] for a position of 4
     chains; a density returns one log density per chain, shape [4]]. With
    several chains it evaluates [lp] again on the chains reversed, and raises
    naming the first row whose log density differs: a chain's density reads only
    its own row. *)

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
(** [warmup u lp k ~steps s] is [s] after [steps] transitions that tune each
    chain's step size and geometry, by Stan's schedule scaled to [steps]: an
    initial buffer, windows that double in length, and a final buffer. Dual
    averaging (Nesterov 2009; Hoffman and Gelman 2014) sets each chain's step
    size for an acceptance of [s.accept], restarting at each window. At each
    window's end each chain's geometry is refitted to its window's draws and
    gradients by the Fisher divergence (Seyboldt, Carlson and Carpenter 2026): a
    diagonal scale [sqrt (sd x / sd score)] per element. Transition [n] of the
    run, counted by [s.draw], has the key [Nx.Rng.fold_in k n], as in {!sample},
    so warmup and sampling may share a key.

    Raises [Invalid_argument] if [steps < 0]. *)

val sample :
  'u Nx.Ptree.t ->
  ('u -> (float, 'f) Nx.t) ->
  Nx.Rng.t ->
  draws:int ->
  ('u, 'f) state ->
  ('u, 'f) state * 'u Draws.t * 'f Stats.t Draws.t
(** [sample u lp k ~draws s] is the state after [draws] transitions, their
    positions and their statistics. Transition [n] of the run, counted by
    [s.draw], has the key [Nx.Rng.fold_in k n].

    Raises [Invalid_argument] if [draws < 1]. *)
