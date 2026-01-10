(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Backend interface for C implementation *)

include Nx_core.Backend_intf.S

val create_context : unit -> context
