(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type point = Sending | Sent | Arrived | Aborting

external arm : point -> unit = "rig_link_probe_arm"
external held : unit -> bool = "rig_link_probe_held"
external releases : unit -> int * int = "rig_link_probe_releases"
external resume : unit -> unit = "rig_link_probe_resume"
