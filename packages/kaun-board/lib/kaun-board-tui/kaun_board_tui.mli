(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Training dashboard TUI.

    Terminal-based dashboard for monitoring training runs logged by
    {!Kaun_board.Log}. Displays live metrics, charts, and system resource usage.
*)

val run : ?base_dir:string -> ?runs:string list -> unit -> unit
(** [run ()] launches the dashboard.

    [base_dir] defaults to the value of [RAVEN_RUNS_DIR], or
    [XDG_CACHE_HOME/raven/runs] when unset. [runs] is a list of run IDs to
    display; currently exactly one must be provided. When [runs] is omitted, the
    most recent run in [base_dir] is selected automatically. *)
