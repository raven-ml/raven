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
    _ t ->
    (float, 'b) Nx.dtype ->
    (float, 'b, Nx.host) Nx.t * (float, 'b, Nx.host) Nx.t

  val nodes_at :
    'd Nx.Placement.t ->
    _ t ->
    (float, 'b) Nx.dtype ->
    (float, 'b, 'd) Nx.t * (float, 'b, 'd) Nx.t
end

module Range : sig
  type ('b, 'd) t

  val v : (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t -> ('b, 'd) t
  val from : (float, 'b, 'd) Nx.t -> ('b, 'd) t
  val line : (float, 'b, 'd) Nx.t -> ('b, 'd) t
end

module Box : sig
  type ('b, 'd) t

  val v : (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t -> ('b, 'd) t
end

type ('b, 'd) integrand = (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t

val fixed :
  [> `Formula ] Rule.t ->
  ('b, 'd) integrand ->
  ('b, 'd) Range.t ->
  (float, 'b, 'd) Nx.t

val cumulative :
  [> `Formula ] Rule.t ->
  ('b, 'd) integrand ->
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t

val adaptive :
  [> `Embedded ] Rule.t ->
  tol:Tol.t ->
  budget:int ->
  ('b, 'd) integrand ->
  ('b, 'd) Range.t ->
  ((float, 'b, 'd) Nx.t, 'd) Solution.t

val tanh_sinh :
  tol:Tol.t ->
  ('b, 'd) integrand ->
  ('b, 'd) Range.t ->
  ((float, 'b, 'd) Nx.t, 'd) Solution.t

val cubature :
  tol:Tol.t ->
  budget:int ->
  ('b, 'd) integrand ->
  ('b, 'd) Box.t ->
  ((float, 'b, 'd) Nx.t, 'd) Solution.t

val qmc :
  'd Nx.Rng.key ->
  tol:Tol.t ->
  budget:int ->
  ('b, 'd) integrand ->
  ('b, 'd) Box.t ->
  ((float, 'b, 'd) Nx.t, 'd) Solution.t
