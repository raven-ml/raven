(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Hamiltonian dynamics with unit metric, over chains.

    Chains move in whitened coordinates [z], on a density [lp_z] of them. *)

type ('u, 'f) point = { z : 'u; p : 'u; g : 'u; lp : (float, 'f) Nx.t }
(** A phase point of each chain: position, momentum, the density's gradient and
    value there. *)

val point_ptree : 'u Nx.Ptree.t -> ('u, 'f) point Nx.Ptree.t

val choose_point :
  'u Nx.Ptree.t ->
  Nx.bool_t ->
  ('u, 'f) point ->
  ('u, 'f) point ->
  ('u, 'f) point
(** [choose_point u mask a b] is [a] in the chains where [mask] holds. *)

val momentum : 'u Nx.Ptree.t -> Nx.Rng.t -> 'u -> 'u
(** [momentum u keys like] is a standard normal momentum per chain, chain [i]'s
    drawn from [fold_in keys.(i) 0]. *)

val kinetic : 'u Nx.Ptree.t -> (float, 'f) Nx.t -> 'u -> (float, 'f) Nx.t
(** [kinetic u like p] is each chain's [p · p / 2], at [like]'s dtype. *)

val max_energy_error : float
(** A transition diverges when its energy error exceeds this. *)

val leapfrog :
  string ->
  'u Nx.Ptree.t ->
  ('u -> (float, 'f) Nx.t) ->
  Nx.bool_t ->
  (float, 'f) Nx.t ->
  ('u, 'f) point ->
  ('u, 'f) point * Nx.bool_t
(** [leapfrog context u lp_z running h s] is one leapfrog step of [h], one per
    chain, from [s] with its cached gradient, by one density evaluation, and
    whether each chain's step stayed finite. A chain held by [running], or whose
    step left the reals, evaluates the density again at [s]. *)

val search :
  string ->
  'u Nx.Ptree.t ->
  ('u -> (float, 'f) Nx.t) ->
  reduce:((float, 'f) Nx.t -> (float, 'f) Nx.t) ->
  Nx.Rng.t ->
  ('u, 'f) point ->
  (float, 'f) Nx.t ->
  (float, 'f) Nx.t
(** [search context u lp_z ~reduce keys start eps] doubles or halves the step
    sizes [eps] until [reduce] of the chains' log acceptance of one leapfrog
    step from [start] crosses [log 0.8] (Stan's heuristic). [reduce] maps the
    chains' [[c]] log acceptances to [eps]'s shape. Trial [i]'s momentum of
    chain [j] draws from [fold_in (fold_in keys.(j) 2) i]. *)

(** {1:transitions Fixed-length transitions}

    Every chain shares one step size, one length and one Gaussian, and moves
    with unit metric in the Gaussian's whitened coordinates. *)

val max_steps : int
(** The longest trajectory, in leapfrog steps. *)

val color : 'u Nx.Ptree.t -> ('u, 'f) Gaussian.t -> 'u -> 'u
(** [color u g z] is {!Gaussian.color} of every chain's [z]. *)

val whiten : 'u Nx.Ptree.t -> ('u, 'f) Gaussian.t -> 'u -> 'u
(** [whiten u g x] is {!Gaussian.whiten} of every chain's [x]. *)

val to_whitened : 'u Nx.Ptree.t -> ('u, 'f) Gaussian.t -> 'u -> 'u -> 'u
(** [to_whitened u g z gx] is the gradient in whitened coordinates of the
    gradient [gx] at [color z]. *)

val chain_keys : Nx.Rng.t -> int -> Nx.Rng.t
(** [chain_keys k c] is row [i] of [split_batch ~n:c (fold_in k 0)] for chain
    [i]. *)

type ('u, 'f) transition = {
  position : 'u;  (** Each chain's position after the transition. *)
  lp : (float, 'f) Nx.t;  (** The density there. *)
  grad : 'u;  (** Its gradient there. *)
  alpha : (float, 'f) Nx.t;
      (** The Metropolis probability of the trajectory's end, [0] at a
          divergence. *)
  steps : Nx.int32_t;  (** Each chain's leapfrog steps. *)
  diverging : Nx.bool_t;
  energy : (float, 'f) Nx.t;  (** The Hamiltonian after the momentum draw. *)
  z0 : 'u;  (** The whitened start. *)
  z1 : 'u;  (** The whitened end of the trajectory. *)
  p1 : 'u;  (** Its momentum. *)
  time : (float, 'f) Nx.t;  (** The trajectory's duration. *)
}
(** The type for a transition of every chain. *)

val transition :
  string ->
  'u Nx.Ptree.t ->
  ('u -> (float, 'f) Nx.t) ->
  Nx.Rng.t ->
  ('u, 'f) Gaussian.t ->
  step_size:(float, 'f) Nx.t ->
  length:(float, 'f) Nx.t ->
  'u ->
  (float, 'f) Nx.t ->
  'u ->
  ('u, 'f) transition
(** [transition context u lp k g ~step_size ~length x l grad] is one transition
    of every chain from [x], at density [l] with gradient [grad]: [n] leapfrog
    steps of [step_size], [n] the [length] jittered uniformly in [(0, 2)] times
    its value by one draw the chains share, over the step size, rounded up and
    at most {!max_steps}; then a Metropolis choice between the trajectory's end
    and its start. A chain whose step leaves the reals or whose energy error
    exceeds {!max_energy_error} stops there and is rejected. Chain [i] draws
    from row [i] of {!chain_keys}[ k c], its momentum from [fold_in key 0] and
    its acceptance from [fold_in key 1]; the jitter from [fold_in k 1].

    Raises [Invalid_argument] naming [context] if [lp] is NaN or [+inf] at a
    finite position. *)

(** {1:chees ChEES}

    The length that maximises the change in expected squared jumped distance
    (Hoffman, Radul and Sountsov 2021), found by Adam on its log. *)

type 'f chees = {
  log_length : (float, 'f) Nx.t;  (** The current iterate. *)
  second : (float, 'f) Nx.t;  (** Adam's second moment. *)
  bar : (float, 'f) Nx.t;  (** The iterates' polynomial average. *)
  count : (float, 'f) Nx.t;  (** The steps taken. *)
}
(** The type for the search's state, every field a scalar. *)

val chees_ptree : unit -> 'f chees Nx.Ptree.t

val chees_gradient : 'u Nx.Ptree.t -> ('u, 'f) transition -> (float, 'f) Nx.t
(** [chees_gradient u t] is the acceptance-weighted mean over chains of the
    criterion's derivative in log length at the transition [t], zero when no
    chain accepts. *)

val chees_step : 'f chees -> (float, 'f) Nx.t -> (float, 'f) Nx.t -> 'f chees
(** [chees_step a eps g] is one Adam step of rate [0.025] up the gradient [g],
    the log length kept where a trajectory fits {!max_steps} steps of [eps]. *)
