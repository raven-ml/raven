(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Piecewise functions: the implementation of {!Jera.Piecewise}, whose
    functions are the series of {!Cheb}, which {!Ode.path} builds too. *)

type ('v, 'b) t = ('v, 'b) Cheb.series

val v : 'v Nx.Ptree.t -> breaks:(float, 'b) Nx.t -> 'v -> ('v, 'b) t

type 'b ends = 'b Spline.ends

val linear : (float, 'b) Nx.t -> (float, 'b) Nx.t -> ((float, 'b) Nx.t, 'b) t

val cubic :
  'b ends -> (float, 'b) Nx.t -> (float, 'b) Nx.t -> ((float, 'b) Nx.t, 'b) t

val steffen : (float, 'b) Nx.t -> (float, 'b) Nx.t -> ((float, 'b) Nx.t, 'b) t

val hermite :
  (float, 'b) Nx.t ->
  values:(float, 'b) Nx.t ->
  slopes:(float, 'b) Nx.t ->
  ((float, 'b) Nx.t, 'b) t

val chebyshev :
  'v Nx.Ptree.t ->
  degree:int ->
  pieces:int ->
  ((float, 'b) Nx.t -> 'v) ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  ('v, 'b) t

val adapt :
  'v Nx.Ptree.t ->
  degree:int ->
  tol:Tol.t ->
  budget:int ->
  ((float, 'b) Nx.t -> 'v) ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  ('v, 'b) t Solution.t

val eval : ('v, 'b) t -> (float, 'b) Nx.t -> 'v

val eval_at :
  ('v, 'b) t -> (int64, Nx.int64_elt) Nx.t -> (float, 'b) Nx.t -> 'v

val derivative : ('v, 'b) t -> ('v, 'b) t
val integral : ('v, 'b) t -> ('v, 'b) t
val extend : [ `Hold | `Polynomial ] -> ('v, 'b) t -> ('v, 'b) t
val breaks : ('v, 'b) t -> (float, 'b) Nx.t
val coefficients : ('v, 'b) t -> 'v
val ptree : 'v Nx.Ptree.t -> (float, 'b) Nx.dtype -> ('v, 'b) t Nx.Ptree.t
