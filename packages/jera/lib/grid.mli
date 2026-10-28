(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tensor-product series over grids: the implementation of {!Jera.Grid}, built
    from the series of {!Cheb} and the interpolants of {!Spline} one axis at a
    time. *)

type ('b, 'd) t

val linear :
  axes:(float, 'b, 'd) Nx.t list -> (float, 'b, 'd) Nx.t -> ('b, 'd) t

val cubic :
  [ `Natural | `Not_a_knot ] ->
  axes:(float, 'b, 'd) Nx.t list ->
  (float, 'b, 'd) Nx.t ->
  ('b, 'd) t

val chebyshev :
  degree:int ->
  pieces:int ->
  ((float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t) ->
  lo:(float, 'b, 'd) Nx.t ->
  hi:(float, 'b, 'd) Nx.t ->
  ('b, 'd) t

val eval : ('b, 'd) t -> (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t
val derivative : axis:int -> ('b, 'd) t -> ('b, 'd) t
val integral : axis:int -> ('b, 'd) t -> ('b, 'd) t
val ptree : (float, 'b) Nx.dtype -> (('b, 'd) t, 'd) Nx.Ptree.t
