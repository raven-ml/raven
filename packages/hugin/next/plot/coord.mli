(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Coordinate systems. *)

module P2 := Hugin_next_gg.P2
module Box2 := Hugin_next_gg.Box2

type t = Cartesian of { aspect : float option }

val cartesian : ?aspect:float -> unit -> t
val equal : t -> t -> bool
val pp : Format.formatter -> t -> unit

(** {1:projections Projections} *)

type projection

val project : t -> Box2.t -> projection
(** [project c box] is the projection of [c] onto [box]. *)

val point : projection -> float -> float -> P2.t
val invert : projection -> P2.t -> (float * float) option
