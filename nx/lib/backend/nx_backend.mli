(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

include Nx_core.Backend_intf.S

(* TODO: [create_context : unit -> context] won't work for backends that need
   parameters (e.g. GPU device index, memory limits). We'll likely need
   [create_context : ?config:config -> unit -> context] or similar, with a
   default config for each backend. *)
val create_context : unit -> context
