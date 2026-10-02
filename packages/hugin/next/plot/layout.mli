(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Laid-out figures: panels, guides and titles placed on the page. *)

module Box2 := Hugin_next_gg.Box2

type t

val layout : ?prev:t -> ?theme:Theme.t -> Size.t -> Resolved.t -> t

type panel = { id : Common.id; box : Box2.t; projection : Coord.projection }

val size : t -> float * float
val panels : t -> panel list
val warnings : t -> Common.warning list
val equal : t -> t -> bool
val pp : Format.formatter -> t -> unit
