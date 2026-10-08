(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Programs, documented in rig.mli. *)

type t = Def.program

val load : Def.device -> string -> (t, string) result
val device : t -> Def.device
val entry : t -> string -> int option
