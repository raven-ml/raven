(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Ensemble slice sampling, without derivatives.

    Walkers lie on the chain axis in independent ensembles of equal size,
    consecutive on the axis. A transition updates each ensemble in two halves
    (Karamanis and Beutler 2021): a half moves by slice sampling (Neal 2003)
    along directions drawn from the Gaussian of the other half, its walkers'
    mean and covariance, so a walker's move depends only on the half that does
    not move, as detailed balance requires. A direction has unit length in that
    Gaussian's whitened coordinates. Each walker draws a level under its log
    density and doubles a bracket toward random sides while an end lies in the
    slice, so a walker far from the others' scale reaches the slice in as many
    doublings as the log of the distance; shrinking keeps the walker in the
    bracket, takes a point only if the doubling could have produced the bracket
    from it, which keeps the move reversible, and ends, the walker staying, when
    the bracket is narrower than the dtype's resolution. Nothing is tuned.

    The density is evaluated on the moving half of every ensemble at once, once
    per trip of the slice loop: a call has [walkers / 2] rows when ensembles
    have an even size. A walker that has found its point repeats its last
    evaluation until the others have. A transition costs a varying number of
    evaluations, which {!stats} counts.

    {b Keys.} Transition [n] of a run has the key [k = Nx.Rng.fold_in run n],
    and walker [i] row [i] of [Nx.Rng.split_batch ~n:walkers k], so with
    ensembles of a fixed size a walker's randomness does not depend on the
    number of ensembles. {!warmup} and {!sample} fold the state's draw counter
    into their key and advance it, so [a + b] draws are [a] draws then [b]. *)

type 'f stats = {
  lp : (float, 'f) Nx.t;  (** The log density at the transition's end. *)
  evaluations : Nx.int32_t;
      (** The density evaluations of the walker's moves in the transition. *)
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
