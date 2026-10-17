(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Minima: the implementation of {!Jera.Minimize}. *)

type ('x, 'f) t

val bfgs : linear:'x Linear.t -> ('x, 'x -> (float, 'b) Nx.t) t
val lbfgs : memory:int -> linear:'x Linear.t -> ('x, 'x -> (float, 'b) Nx.t) t
val newton : linear:'x Linear.t -> ('x, 'x -> (float, 'b) Nx.t) t

val levenberg_marquardt :
  'r Nx.Ptree.t -> linear:'x Linear.t -> ('x, 'x -> 'r) t

val nelder_mead : ('x, 'x -> (float, 'b) Nx.t) t

val solve :
  'x Nx.Ptree.t ->
  ('x, 'f) t ->
  ?within:'x * 'x ->
  tol:Tol.t ->
  budget:int ->
  'f ->
  'x ->
  'x Solution.t

val iterates : 'x Nx.Ptree.t -> ('x, 'f) t -> steps:int -> 'f -> 'x -> 'x

val bracket :
  tol:Tol.t ->
  ((float, 'b) Nx.t -> (float, 'b) Nx.t) ->
  lo:(float, 'b) Nx.t ->
  hi:(float, 'b) Nx.t ->
  (float, 'b) Nx.t Solution.t
