(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Holds: the implementation of {!Rig.Hold}. *)

type t = Def.hold

val make : (unit -> unit) -> t
