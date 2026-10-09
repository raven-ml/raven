(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Claims: the implementation of {!Rig.Claim}. *)

val read : Def.buffer -> unit
val release : Def.buffer -> unit
val share : Def.buffer -> unit

type t

val with_ :
  read:Def.buffer list -> donate:Def.buffer list list -> (t -> 'a) -> 'a

val exclusive : t -> Def.buffer -> bool
val consume : t -> why:string -> Def.buffer -> Def.buffer
