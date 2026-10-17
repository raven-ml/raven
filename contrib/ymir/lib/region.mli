(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Regions: the implementation of {!Ymir.Region}, with the overlap that
    {!Observation.integrate} sums. *)

type ('w, 'e) t

val circle :
  ('w, Transform.plane) Transform.t ->
  radius:(float, 'e) Nx.t Ymir_units.Quantity.t ->
  ('w, 'e) t

val annulus :
  ('w, Transform.plane) Transform.t ->
  inner:(float, 'e) Nx.t Ymir_units.Quantity.t ->
  outer:(float, 'e) Nx.t Ymir_units.Quantity.t ->
  ('w, 'e) t

val ellipse :
  ('w, Transform.plane) Transform.t ->
  a:(float, 'e) Nx.t Ymir_units.Quantity.t ->
  b:(float, 'e) Nx.t Ymir_units.Quantity.t ->
  angle:(float, 'e) Nx.t Ymir_units.Quantity.t ->
  ('w, 'e) t

val polygon :
  (float, 'e) Nx.dtype -> ('w, Transform.plane) Transform.t -> 'w -> ('w, 'e) t

val weights : ('w, 'e) t -> ('w, 'e) Grid.t -> (float, 'e) Nx.t
val ptree : unit -> ('w, 'e) t Nx.Ptree.t

val overlap : ('w, 'e) t -> ('w, 'e) Grid.t -> 'e Overlap.t * (float, 'e) Nx.t
(** [overlap r g] is the overlap of each of [g]'s cells with [r],
    [[batch; h; w]], whose [weight] is [weights r g], and the shape's own
    area in its plane, [[batch]]. Both areas are in the square of the unit of
    the shape's sizes, or of its mapped vertices for a polygon.

    Raises as {!weights} does. *)
