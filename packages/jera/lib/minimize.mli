(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Minima: the implementation of {!Jera.Minimize}. *)

type ('x, 'f, 'd) t

val bfgs : linear:'x Linear.t -> ('x, 'x -> (float, 'b, 'd) Nx.t, 'd) t

val lbfgs :
  memory:int -> linear:'x Linear.t -> ('x, 'x -> (float, 'b, 'd) Nx.t, 'd) t

val newton : linear:'x Linear.t -> ('x, 'x -> (float, 'b, 'd) Nx.t, 'd) t

val levenberg_marquardt :
  ('r, 'd) Nx.Ptree.t -> linear:'x Linear.t -> ('x, 'x -> 'r, 'd) t

val nelder_mead : ('x, 'x -> (float, 'b, 'd) Nx.t, 'd) t

val solve :
  ('x, 'd) Nx.Ptree.t ->
  ('x, 'f, 'd) t ->
  ?lo:'x ->
  ?hi:'x ->
  tol:Tol.t ->
  budget:int ->
  'f ->
  'x ->
  ('x, 'd) Solution.t

val iterates :
  ('x, 'd) Nx.Ptree.t -> ('x, 'f, 'd) t -> steps:int -> 'f -> 'x -> 'x

val bracket :
  tol:Tol.t ->
  ((float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t) ->
  lo:(float, 'b, 'd) Nx.t ->
  hi:(float, 'b, 'd) Nx.t ->
  ((float, 'b, 'd) Nx.t, 'd) Solution.t
