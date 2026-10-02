(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Drawings: laid-out figures painted at a density. *)

module Renderable := Hugin_next_vg.Renderable

type t

val draw : ?prev:t -> density:float -> Layout.t -> t
val renderable : t -> Renderable.t
val warnings : t -> Common.warning list
val equal : t -> t -> bool
val pp : Format.formatter -> t -> unit
