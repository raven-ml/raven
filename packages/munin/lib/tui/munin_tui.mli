(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Experiment dashboard TUI.

    Terminal-based dashboard for monitoring runs tracked by {!Munin}. Displays
    live metrics, charts, and system resource usage. *)

val run :
  ?root:string -> ?experiment:string -> ?runs:string list -> unit -> unit
(** [run ()] launches the dashboard.

    - [root] defaults to [$RAVEN_TRACKING_DIR] or [$XDG_DATA_HOME/raven/munin].
    - [experiment] filters runs by experiment name.
    - [runs] is a list of run IDs to display; currently exactly one must be
      provided. When omitted, the most recent run is selected automatically. *)
