(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Buffer.copy, documented in rig.mli. *)

val copy : src:Def.buffer -> dst:Def.buffer -> unit
val queued : Def.device -> string -> src:Def.buffer -> dst:Def.buffer -> unit
(* [queued d queue ~src ~dst] copies [src] into [dst], memory of [d], as a
   submission on [d]'s copy queue [queue], and waits for it. *)
