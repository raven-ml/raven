(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Figure sizes. *)

type t = Figure of float * float | Panels of float * float

val figure : float -> float -> t
val panels : float -> float -> t
val mm : float -> float
val dpi : float -> float
val equal : t -> t -> bool
val pp : Format.formatter -> t -> unit
