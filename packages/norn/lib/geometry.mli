(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Gaussians over structures, with the maps the samplers move through beside
    {!Gaussian}, which documents every value. *)

type ('u, 'f) t
(** The type for Gaussians. *)

val diagonal :
  'u Nx.Ptree.t -> (float, 'f) Nx.dtype -> mean:'u -> scale:'u -> ('u, 'f) t

val low_rank :
  'u Nx.Ptree.t ->
  mean:'u ->
  scale:'u ->
  directions:'u ->
  variances:(float, 'f) Nx.t ->
  ('u, 'f) t

val of_precision :
  'u Nx.Ptree.t ->
  (float, 'f) Nx.dtype ->
  ?low_rank:'u * (float, 'f) Nx.t ->
  mean:'u ->
  'u ->
  ('u, 'f) t

val color : 'u Nx.Ptree.t -> ('u, 'f) t -> 'u -> 'u
(** [color u g z] is [m + S A z], [A = I + Σ_j (sqrt v_j - 1) u_j u_jᵀ]:
    standard normal coordinates [z] mapped to [g], at one position. *)

val direction : 'u Nx.Ptree.t -> ('u, 'f) t -> 'u -> 'u
(** [direction u g z] is [S A z], {!color} without the mean: whitened directions
    [z] mapped to [g]'s. *)

val whiten : 'u Nx.Ptree.t -> ('u, 'f) t -> 'u -> 'u
(** [whiten u g x] is the inverse of {!color}. *)

val log_det : ('u, 'f) t -> 'u Nx.Ptree.t -> (float, 'f) Nx.t
(** [log_det g u] is [log |det|] of {!color}. *)

val rank : ('u, 'f) t -> int
(** [rank g] is the number of [g]'s directions, also of Gaussians stacked on a
    chain axis. *)

val sample : 'u Nx.Ptree.t -> Nx.Rng.t -> n:int -> ('u, 'f) t -> 'u
val log_density : 'u Nx.Ptree.t -> ('u, 'f) t -> 'u -> (float, 'f) Nx.t
val mean : 'u Nx.Ptree.t -> ('u, 'f) t -> 'u
val variance : 'u Nx.Ptree.t -> ('u, 'f) t -> 'u
val ptree : 'u Nx.Ptree.t -> ('u, 'f) t Nx.Ptree.t
val pp : 'u Nx.Ptree.t -> Format.formatter -> ('u, 'f) t -> unit
