(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Holds, documented in device_core.mli. *)

type t = Def.hold

val make : ?release:(unit -> unit) -> Def.buffer list -> t
