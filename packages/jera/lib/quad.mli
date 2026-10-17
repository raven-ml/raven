(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Integrals: the implementation of {!Jera.Quad}. *)

module Rule : sig
  type -'k t

  val gauss : int -> [ `Formula ] t
  val kronrod : int -> [ `Formula | `Embedded ] t

  val nodes :
    _ t -> (float, 'b) Nx.dtype -> (float, 'b) Nx.t * (float, 'b) Nx.t
end

module Range : sig
  type 'b t

  val v : (float, 'b) Nx.t -> (float, 'b) Nx.t -> 'b t
  val from : (float, 'b) Nx.t -> 'b t
  val line : (float, 'b) Nx.t -> 'b t
end

module Box : sig
  type 'b t

  val v : (float, 'b) Nx.t -> (float, 'b) Nx.t -> 'b t
end

type 'b integrand = (float, 'b) Nx.t -> (float, 'b) Nx.t

val fixed :
  [> `Formula ] Rule.t -> 'b integrand -> 'b Range.t -> (float, 'b) Nx.t

val cumulative :
  [> `Formula ] Rule.t -> 'b integrand -> (float, 'b) Nx.t -> (float, 'b) Nx.t

val adaptive :
  [> `Embedded ] Rule.t ->
  tol:Tol.t ->
  budget:int ->
  'b integrand ->
  'b Range.t ->
  (float, 'b) Nx.t Solution.t

val tanh_sinh :
  tol:Tol.t -> 'b integrand -> 'b Range.t -> (float, 'b) Nx.t Solution.t

val cubature :
  tol:Tol.t ->
  budget:int ->
  'b integrand ->
  'b Box.t ->
  (float, 'b) Nx.t Solution.t

val qmc :
  Nx.Rng.t ->
  tol:Tol.t ->
  budget:int ->
  'b integrand ->
  'b Box.t ->
  (float, 'b) Nx.t Solution.t
