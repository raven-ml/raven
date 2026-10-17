(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Zeros of systems: the implementation of {!Jera.System}. *)

type 'x t

val newton : derivative:('x -> 'x -> 'x) -> 'x t
val broyden : 'x t
val anderson : memory:int -> 'x t

val solve :
  'x Nx.Ptree.t ->
  'x t ->
  linear:'x Linear.t ->
  tol:Tol.t ->
  budget:int ->
  ('x -> 'x) ->
  'x ->
  'x Solution.t

val lanes :
  tol:Tol.t ->
  budget:int ->
  jacobian:((float, 'b) Nx.t -> (float, 'b) Nx.t) ->
  ((float, 'b) Nx.t -> (float, 'b) Nx.t) ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t Solution.t
