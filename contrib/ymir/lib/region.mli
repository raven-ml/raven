(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Regions: the implementation of {!Ymir.Region}, with the overlap that
    {!Observation.integrate} sums. *)

type ('w, 'e, 'd) t

val circle :
  ('w, 'd Transform.plane, 'd) Transform.t ->
  radius:(float, 'e, 'd) Nx.t Ymir_units.Quantity.t ->
  ('w, 'e, 'd) t

val annulus :
  ('w, 'd Transform.plane, 'd) Transform.t ->
  inner:(float, 'e, 'd) Nx.t Ymir_units.Quantity.t ->
  outer:(float, 'e, 'd) Nx.t Ymir_units.Quantity.t ->
  ('w, 'e, 'd) t

val ellipse :
  ('w, 'd Transform.plane, 'd) Transform.t ->
  a:(float, 'e, 'd) Nx.t Ymir_units.Quantity.t ->
  b:(float, 'e, 'd) Nx.t Ymir_units.Quantity.t ->
  angle:(float, 'e, 'd) Nx.t Ymir_units.Quantity.t ->
  ('w, 'e, 'd) t

val polygon :
  (float, 'e) Nx.dtype -> ('w, 'd Transform.plane, 'd) Transform.t -> 'w -> ('w, 'e, 'd) t

val weights : ('w, 'e, 'd) t -> ('w, 'e, 'd) Grid.t -> (float, 'e, 'd) Nx.t
val ptree : unit -> (('w, 'e, 'd) t, 'd) Nx.Ptree.t

val overlap : ('w, 'e, 'd) t -> ('w, 'e, 'd) Grid.t -> ('e, 'd) Overlap.t * (float, 'e, 'd) Nx.t
(** [overlap r g] is the overlap of each of [g]'s cells with [r],
    [[batch; h; w]], whose [weight] is [weights r g], and the shape's own
    area in its plane, [[batch]]. Both areas are in the square of the unit of
    the shape's sizes, or of its mapped vertices for a polygon.

    Raises as {!weights} does. *)
