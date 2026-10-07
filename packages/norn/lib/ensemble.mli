(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Ensemble sampling by the stretch move, without derivatives.

    Walkers lie on the chain axis in independent ensembles of equal size,
    consecutive on the axis. A transition splits each ensemble at random into
    two halves, which mixes faster than fixed halves, and updates them in turn:
    a walker [x] of the moving half draws a walker [w] of the other half and a
    stretch [z] of density proportional to [1 / sqrt z] on [[1 / 2, 2]], and
    moves to [w + z (x - w)] with probability [min (1, z^(d - 1) p(y) / p(x))],
    [d] its float elements (Goodman and Weare 2010). A walker's move depends
    only on the half that does not move, as detailed balance requires, and on no
    scale: the move is unchanged by an affine map of the coordinates, so a
    correlated posterior costs what an isotropic one does. Nothing is tuned.

    The density is evaluated once per transition on the moving half of every
    ensemble, a call of [walkers / 2] rows when ensembles have an even size.

    The walkers move along lines through each other, so they explore the region
    their spread spans. Walkers that start far from a narrow, correlated
    posterior, spread over a region of another shape, come back to it only over
    many transitions: start them where a warmed sampler or the posterior's
    approximation leaves them.

    {b Keys.} Transition [n] of a run has the key [k = Nx.Rng.fold_in run n],
    and walker [i] row [i] of [Nx.Rng.split_batch ~n:walkers k]; an ensemble's
    split draws from its first walker's key. So with ensembles of a fixed size a
    walker's randomness does not depend on the number of ensembles. {!warmup}
    and {!sample} fold the state's draw counter into their key and advance it,
    so [a + b] draws are [a] draws then [b]. *)

type 'f stats = {
  lp : (float, 'f) Nx.t;  (** The log density at the transition's end. *)
  acceptance : (float, 'f) Nx.t;
      (** The probability with which the walker's move was accepted. *)
}
(** The type for statistics of transitions, one element per walker. *)

type ('u, 'f) state = private {
  position : 'u;  (** Each walker's position. *)
  lp : (float, 'f) Nx.t;  (** The log density there, shape [[walker]]. *)
  stats : 'f stats;  (** The last transition's statistics. *)
  draw : Nx.int32_t;  (** The number of transitions taken. *)
  ensembles : int;  (** The number of ensembles. *)
}
(** The type for states of walkers over positions ['u]. *)

val ptree : 'u Nx.Ptree.t -> ('u, 'f) state Nx.Ptree.t
(** [ptree u] is the structure of states over [u]. It reports [ensembles]. *)

val stats : (float, 'f) Nx.dtype -> 'f stats Nx.Ptree.t
(** [stats dtype] is the structure of statistics at [dtype]. *)

val init :
  'u Nx.Ptree.t ->
  ?ensembles:int ->
  ('u -> (float, 'f) Nx.t) ->
  'u ->
  ('u, 'f) state
(** [init u ?ensembles lp start] is the state of walkers at [start], a position,
    in [ensembles] ensembles, by default [1].

    Raises [Invalid_argument] if [ensembles < 1], if the walkers do not split
    into [ensembles] ensembles of equal size, if an ensemble has fewer walkers
    than twice a walker's float elements, or if [lp] does not return one log
    density per walker, each read from its own row. *)

val step :
  'u Nx.Ptree.t ->
  ('u -> (float, 'f) Nx.t) ->
  Nx.Rng.t ->
  ('u, 'f) state ->
  ('u, 'f) state
(** [step u lp k s] is one transition of every walker from [s].

    Raises [Invalid_argument] if [lp] is NaN or [+inf] at a walker. *)

val warmup :
  'u Nx.Ptree.t ->
  ('u -> (float, 'f) Nx.t) ->
  Nx.Rng.t ->
  steps:int ->
  ('u, 'f) state ->
  ('u, 'f) state
(** [warmup u lp k ~steps s] is [s] after [steps] transitions, whose draws are
    discarded. Transition [n] of the run, counted by [s.draw], has the key
    [Nx.Rng.fold_in k n], as in {!sample}.

    Raises [Invalid_argument] if [steps < 0]. *)

val sample :
  'u Nx.Ptree.t ->
  ('u -> (float, 'f) Nx.t) ->
  Nx.Rng.t ->
  draws:int ->
  ('u, 'f) state ->
  ('u, 'f) state * 'u Draws.t * 'f stats Draws.t
(** [sample u lp k ~draws s] is the state after [draws] transitions, the
    walkers' positions and their statistics, each walker a chain of the draws.
    Transition [n] of the run, counted by [s.draw], has the key
    [Nx.Rng.fold_in k n].

    Raises [Invalid_argument] if [draws < 1]. *)
