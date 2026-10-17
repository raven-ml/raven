(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tensor-product series over grids: the implementation of {!Jera.Grid},
    built from the series of {!Cheb} and the interpolants of {!Spline} one
    axis at a time. *)

type 'b t

val linear : axes:(float, 'b) Nx.t list -> (float, 'b) Nx.t -> 'b t

val cubic :
  [ `Natural | `Not_a_knot ] ->
  axes:(float, 'b) Nx.t list ->
  (float, 'b) Nx.t ->
  'b t

val chebyshev :
  degree:int ->
  pieces:int ->
  ((float, 'b) Nx.t -> (float, 'b) Nx.t) ->
  lo:(float, 'b) Nx.t ->
  hi:(float, 'b) Nx.t ->
  'b t

val eval : 'b t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
val derivative : axis:int -> 'b t -> 'b t
val integral : axis:int -> 'b t -> 'b t
val ptree : (float, 'b) Nx.dtype -> 'b t Nx.Ptree.t
