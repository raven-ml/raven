(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Laid-out figures: panels and guides placed on the page. *)

module Box2 := Hugin_next_gg.Box2
module Ticks := Hugin_next_kit.Ticks

type panel = { id : Common.id; box : Box2.t; projection : Coord.projection }
type t

val layout : ?prev:t -> ?theme:Theme.t -> Size.t -> Resolved.t -> t
val size : t -> float * float
val panels : t -> panel list
val warnings : t -> Common.warning list

(** {1:parts Parts} *)

val resolved : t -> Resolved.t
val theme : t -> Theme.t
val coords : t -> (panel * Coord.t) list
val frozen : t -> Ticks.t array

val guides : t -> Guide.t list
(** [guides l] is the guides of [l] on the page, in drawing order. *)

val equal : t -> t -> bool
val pp : Format.formatter -> t -> unit
