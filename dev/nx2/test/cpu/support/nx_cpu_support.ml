(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external targets : unit -> string list = "nx_cpu_support_targets"
external current : unit -> string = "nx_cpu_support_current"
external use : string -> unit = "nx_cpu_support_use"

let with_target t f =
  let before = current () in
  use t;
  Fun.protect ~finally:(fun () -> use before) f

external copy : 'd -> 's -> int = "nx_cpu_copy"
external cast : 'd -> 's -> int = "nx_cpu_cast"
