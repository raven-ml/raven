(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The implementation of {!Norn.Gaussian}, with the flat rows the samplers
    move in. *)

type ('u, 'f) t

(** {1:constructors Constructors} *)

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

(** {1:eliminators Eliminators} *)

val sample : 'u Nx.Ptree.t -> Nx.Rng.t -> n:int -> ('u, 'f) t -> 'u

val log_density : 'u Nx.Ptree.t -> ('u, 'f) t -> 'u -> (float, 'f) Nx.t

val mean : 'u Nx.Ptree.t -> ('u, 'f) t -> 'u

val variance : 'u Nx.Ptree.t -> ('u, 'f) t -> 'u

val ptree : 'u Nx.Ptree.t -> ('u, 'f) t Nx.Ptree.t

val pp : 'u Nx.Ptree.t -> Format.formatter -> ('u, 'f) t -> unit

val rank : ('u, 'f) t -> int
(** [rank g] is the number of [g]'s directions, also of Gaussians stacked on a
    chain axis. *)

(** {1:flat Flat rows}

    The samplers move in whitened coordinates held as one row per chain
    ({!Rows.ravel}). Over rows [z] of shape [[c; d]], [m] is the mean, [S]
    the scale and [A] the matrix [I + Σ_j (sqrt v_j - 1) u_j u_jᵀ]. *)

type 'f flat
(** The type for Gaussians over rows of chains. *)

val flat : 'u Nx.Ptree.t -> 'u -> ('u, 'f) t -> 'f flat
(** [flat u x g] is [g] over the rows of [x], a position of chains. [g] is one
    Gaussian or Gaussians stacked on [x]'s chain axis: a tensor of [g] has the
    trailing axes of [x]'s at its path, after the chain axis for stacked
    Gaussians and, for the directions, after the axis of [k]. *)

val color_flat : 'f flat -> (float, 'f) Nx.t -> (float, 'f) Nx.t
(** [color_flat f z] is [m + S A z]. *)

val direction_flat : 'f flat -> (float, 'f) Nx.t -> (float, 'f) Nx.t
(** [direction_flat f z] is [S A z]. *)

val whiten_flat : 'f flat -> (float, 'f) Nx.t -> (float, 'f) Nx.t
(** [whiten_flat f x] is the [z] whose {!color_flat} is [x]. *)

val pull : 'f flat -> (float, 'f) Nx.t -> (float, 'f) Nx.t
(** [pull f gx] is [A S gx], the gradient [gx] at [color_flat f z] in
    whitened coordinates. *)

val push : 'f flat -> (float, 'f) Nx.t -> (float, 'f) Nx.t
(** [push f gz] is [S⁻¹ A⁻¹ gz], the whitened gradient [gz] as a gradient
    at [x]. *)
