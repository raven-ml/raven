(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Images: the implementation of {!Rig.Image}. *)

type t = Def.image

val load : Def.device -> string -> (t, string) result
val device : t -> Def.device
val entry : t -> string -> int option
