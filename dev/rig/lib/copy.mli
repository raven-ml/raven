(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Copies: the implementation of {!Rig.Buffer.copy}. *)

val copy : src:Def.buffer -> dst:Def.buffer -> unit

val queued : Def.device -> string -> src:Def.buffer -> dst:Def.buffer -> unit
(** [queued d queue ~src ~dst] copies [src] into [dst], memory of [d], as a
    submission on [d]'s copy queue [queue], and waits for it. It raises as
    {!Rig.submit} and {!Rig.wait}, and raises [Invalid_argument] if a buffer is
    held. *)
