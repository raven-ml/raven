(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Training dashboard for Kaun.

    A terminal-based dashboard for monitoring training runs logged by
    {!Kaun.Log}.

    {2 Usage}

    {[
      (* Launch dashboard for all runs *)
      Kaun_console.run ()

      (* Filter by experiment *)
      Kaun_console.run ~experiment:"mnist" ()

      (* Watch specific runs *)
      Kaun_console.run ~runs:["2026-01-19_14-30-22_mnist"] ()
    ]} *)

val run :
  ?base_dir:string ->
  ?experiment:string ->
  ?tags:string list ->
  ?runs:string list ->
  unit ->
  unit
(** [run ?base_dir ?experiment ?tags ?runs ()] launches the dashboard.

    @param base_dir Directory containing runs (default: ["./runs"])
    @param experiment Filter to runs with this experiment name (not yet implemented)
    @param tags Filter to runs containing all these tags (not yet implemented)
    @param runs Specific run IDs to display (currently requires exactly one) *)