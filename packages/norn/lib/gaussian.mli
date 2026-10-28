(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The implementation of {!Norn.Gaussian}, with the flat rows the samplers
    move in. *)

type ('u, 'f, 'd) t

(** {1:constructors Constructors} *)

val diagonal :
  ('u, 'd) Nx.Ptree.t -> (float, 'f) Nx.dtype -> mean:'u -> scale:'u -> ('u, 'f, 'd) t

val low_rank :
  ('u, 'd) Nx.Ptree.t ->
  mean:'u ->
  scale:'u ->
  directions:'u ->
  variances:(float, 'f, 'd) Nx.t ->
  ('u, 'f, 'd) t

val of_precision :
  ('u, 'd) Nx.Ptree.t ->
  (float, 'f) Nx.dtype ->
  ?low_rank:'u * (float, 'f, 'd) Nx.t ->
  mean:'u ->
  'u ->
  ('u, 'f, 'd) t

(** {1:eliminators Eliminators} *)

val sample : ('u, 'd) Nx.Ptree.t -> 'd Nx.Rng.key -> n:int -> ('u, 'f, 'd) t -> 'u

val log_density : ('u, 'd) Nx.Ptree.t -> ('u, 'f, 'd) t -> 'u -> (float, 'f, 'd) Nx.t

val mean : ('u, 'd) Nx.Ptree.t -> ('u, 'f, 'd) t -> 'u

val variance : ('u, 'd) Nx.Ptree.t -> ('u, 'f, 'd) t -> 'u

val ptree : ('u, 'd) Nx.Ptree.t -> (('u, 'f, 'd) t, 'd) Nx.Ptree.t

val pp : ('u, 'd) Nx.Ptree.t -> Format.formatter -> ('u, 'f, 'd) t -> unit

val rank : ('u, 'f, 'd) t -> int
(** [rank g] is the number of [g]'s directions, also of Gaussians stacked on a
    chain axis. *)

(** {1:flat Flat rows}

    The samplers move in whitened coordinates held as one row per chain
    ({!Rows.ravel}). Over rows [z] of shape [[c; d]], [m] is the mean, [S]
    the scale and [A] the matrix [I + Σ_j (sqrt v_j - 1) u_j u_jᵀ]. *)

type ('f, 'd) flat
(** The type for Gaussians over rows of chains. *)

val flat : ('u, 'd) Nx.Ptree.t -> 'u -> ('u, 'f, 'd) t -> ('f, 'd) flat
(** [flat u x g] is [g] over the rows of [x], a position of chains. [g] is one
    Gaussian or Gaussians stacked on [x]'s chain axis: a tensor of [g] has the
    trailing axes of [x]'s at its path, after the chain axis for stacked
    Gaussians and, for the directions, after the axis of [k]. *)

val color_flat : ('f, 'd) flat -> (float, 'f, 'd) Nx.t -> (float, 'f, 'd) Nx.t
(** [color_flat f z] is [m + S A z]. *)

val direction_flat : ('f, 'd) flat -> (float, 'f, 'd) Nx.t -> (float, 'f, 'd) Nx.t
(** [direction_flat f z] is [S A z]. *)

val whiten_flat : ('f, 'd) flat -> (float, 'f, 'd) Nx.t -> (float, 'f, 'd) Nx.t
(** [whiten_flat f x] is the [z] whose {!color_flat} is [x]. *)

val pull : ('f, 'd) flat -> (float, 'f, 'd) Nx.t -> (float, 'f, 'd) Nx.t
(** [pull f gx] is [A S gx], the gradient [gx] at [color_flat f z] in
    whitened coordinates. *)

val push : ('f, 'd) flat -> (float, 'f, 'd) Nx.t -> (float, 'f, 'd) Nx.t
(** [push f gz] is [S⁻¹ A⁻¹ gz], the whitened gradient [gz] as a gradient
    at [x]. *)
