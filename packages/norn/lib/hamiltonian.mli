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
