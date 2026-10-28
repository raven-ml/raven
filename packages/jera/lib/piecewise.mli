(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Piecewise functions: the implementation of {!Jera.Piecewise}, whose
    functions are the series of {!Cheb}, which {!Ode.path} builds too. *)

type ('v, 'b, 'd) t = ('v, 'b, 'd) Cheb.series

val v :
  ('v, 'd) Nx.Ptree.t -> breaks:(float, 'b, 'd) Nx.t -> 'v -> ('v, 'b, 'd) t

type ('b, 'd) ends = ('b, 'd) Spline.ends

val linear :
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t ->
  ((float, 'b, 'd) Nx.t, 'b, 'd) t

val cubic :
  ('b, 'd) ends ->
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t ->
  ((float, 'b, 'd) Nx.t, 'b, 'd) t

val steffen :
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t ->
  ((float, 'b, 'd) Nx.t, 'b, 'd) t

val hermite :
  (float, 'b, 'd) Nx.t ->
  values:(float, 'b, 'd) Nx.t ->
  slopes:(float, 'b, 'd) Nx.t ->
  ((float, 'b, 'd) Nx.t, 'b, 'd) t

val chebyshev :
  ('v, 'd) Nx.Ptree.t ->
  degree:int ->
  pieces:int ->
  ((float, 'b, 'd) Nx.t -> 'v) ->
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t ->
  ('v, 'b, 'd) t

val adapt :
  ('v, 'd) Nx.Ptree.t ->
  degree:int ->
  tol:Tol.t ->
  budget:int ->
  ((float, 'b, 'd) Nx.t -> 'v) ->
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t ->
  (('v, 'b, 'd) t, 'd) Solution.t

val eval : ('v, 'b, 'd) t -> (float, 'b, 'd) Nx.t -> 'v
val eval_at : ('v, 'b, 'd) t -> 'd Nx.int64_t -> (float, 'b, 'd) Nx.t -> 'v
val derivative : ('v, 'b, 'd) t -> ('v, 'b, 'd) t
val integral : ('v, 'b, 'd) t -> ('v, 'b, 'd) t
val extend : [ `Hold | `Polynomial ] -> ('v, 'b, 'd) t -> ('v, 'b, 'd) t
val breaks : ('v, 'b, 'd) t -> (float, 'b, 'd) Nx.t
val coefficients : ('v, 'b, 'd) t -> 'v

val ptree :
  ('v, 'd) Nx.Ptree.t -> (float, 'b) Nx.dtype -> (('v, 'b, 'd) t, 'd) Nx.Ptree.t
