(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Gaussians over structures, diagonal plus low rank.

    A Gaussian over a structure ['u] has a mean [m], a positive scale [S] per
    element, orthonormal directions [u_j] and variances [v_j > 0] along them:
    its covariance is [S (I + Σ_j (v_j - 1) u_j u_jᵀ) S], with the elements of
    every float tensor of ['u] taken together as one vector. No matrix over all
    the elements exists: each product is leafwise, then summed over leaves, at
    [O(k d)] for [k] directions and [d] elements.

    A Gaussian describes one position, so its tensors have no chain axis;
    {!sample} adds one and {!log_density} reads one. A sampler whose geometry
    differs per chain stacks Gaussians on the chain axis and applies chain [i]'s
    inside the map over chains.

    {b Preconditioning is a pullback.} The map [color z = m + S A z],
    [A = I + Σ_j (sqrt v_j - 1) u_j u_jᵀ], takes standard normal coordinates [z]
    to the Gaussian. A kernel moving with unit metric on
    [fun z -> lp (color z) + log |det|] moves with covariance [Σ] on [lp]. *)

type ('u, 'f) t = ('u, 'f) Geometry.t
(** The type for Gaussians over ['u] whose log densities have element type ['f].
*)

(** {1:constructors Constructors} *)

val diagonal :
  'u Nx.Ptree.t -> (float, 'f) Nx.dtype -> mean:'u -> scale:'u -> ('u, 'f) t
(** [diagonal u dtype ~mean ~scale] is the Gaussian of mean [mean] and standard
    deviation [scale], element by element. [scale]'s elements are positive. *)

val low_rank :
  'u Nx.Ptree.t ->
  mean:'u ->
  scale:'u ->
  directions:'u ->
  variances:(float, 'f) Nx.t ->
  ('u, 'f) t
(** [low_rank u ~mean ~scale ~directions ~variances] is the Gaussian whose
    directions are [directions], each tensor with a leading axis of [k], from
    [0] to the number of elements, orthonormalised, and whose variances along
    them are [variances], of shape [[k]].

    Raises [Invalid_argument] if [variances] is not of shape [[k]], and through
    {!Nx.check} if one is not positive and finite. *)

val of_precision :
  'u Nx.Ptree.t ->
  (float, 'f) Nx.dtype ->
  ?low_rank:'u * (float, 'f) Nx.t ->
  mean:'u ->
  'u ->
  ('u, 'f) t
(** [of_precision u dtype ?low_rank ~mean p] is the Gaussian of mean [mean]
    whose precision is the diagonal [p] plus, with [~low_rank:(w, l)], the sum
    of [l_j w_j w_jᵀ] over the directions [w_j], the leading axis of [w]'s
    tensors. The form is closed under inversion: a Laplace approximation is the
    precision at a mode.

    Raises [Invalid_argument] naming the path at an element of [p] that is not
    positive and finite, or at an [l_j] that is negative or not finite. *)

(** {1:eliminators Eliminators} *)

val sample : 'u Nx.Ptree.t -> Nx.Rng.t -> n:int -> ('u, 'f) t -> 'u
(** [sample u k ~n g] is [n] draws of [g] on a new leading axis.

    Raises [Invalid_argument] if [n < 0]. *)

val log_density : 'u Nx.Ptree.t -> ('u, 'f) t -> 'u -> (float, 'f) Nx.t
(** [log_density u g] is [g]'s log density at a position: one value per row of
    the leading axis. *)

val mean : 'u Nx.Ptree.t -> ('u, 'f) t -> 'u
(** [mean u g] is [g]'s mean. *)

val variance : 'u Nx.Ptree.t -> ('u, 'f) t -> 'u
(** [variance u g] is the diagonal of [g]'s covariance, element by element. *)

val ptree : 'u Nx.Ptree.t -> ('u, 'f) t Nx.Ptree.t
(** [ptree u] is the structure of Gaussians over [u]: [mean], [scale] and
    [directions], each walked as [u], and [variances]. *)

val pp : 'u Nx.Ptree.t -> Format.formatter -> ('u, 'f) t -> unit
(** [pp u ppf g] formats [g]'s dimension and rank. *)
