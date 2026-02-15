(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Training run logging and discovery.

    This library provides persistent storage and discovery of machine learning
    training runs. Runs are stored as directories containing a JSON manifest and
    a JSONL event log.

    {1 Directory Structure}

    Each run is stored in [{base_dir}/{run_id}/] containing:
    - [run.json]: Manifest with run metadata (ID, timestamp, experiment, tags)
    - [events.jsonl]: Append-only log of training events

    {1 Quick Start}

    Create a run and log metrics:
    {[
      let run = Kaun_runlog.create_run ~experiment:"mnist" () in
      Run.append_event run
        (Event.Scalar { step = 0; epoch = None; tag = "loss"; value = 2.3 });
      Run.append_event run
        (Event.Scalar { step = 100; epoch = Some 1; tag = "loss"; value = 0.5 })
    ]}

    Discover and read existing runs:
    {[
      let runs = Kaun_runlog.discover () in
      List.iter (fun r -> Printf.printf "Run: %s\n" (Run.run_id r)) runs
    ]} *)

module Event = Event
(** Training event types. See {!Event}. *)

module Run = Run
(** Training run management. See {!Run}. *)

(** {1 Configuration} *)

val base_dir : unit -> string
(** [base_dir ()] returns the default base directory for training runs.

    Checks [RAVEN_RUNS_DIR] environment variable first, otherwise falls back to
    [XDG_CACHE_HOME/raven/runs] (or [~/.cache/raven/runs] if [XDG_CACHE_HOME] is
    not set). *)

(** {1 Discovery} *)

val discover : ?base_dir:string -> unit -> Run.t list
(** [discover ?base_dir ()] scans the base directory for training runs and
    returns them sorted by creation time (newest first).

    Returns an empty list if the base directory does not exist. Subdirectories
    that do not contain a valid [run.json] manifest are silently skipped.

    @param base_dir
      Directory containing run subdirectories. Defaults to [RAVEN_RUNS_DIR] if
      set, otherwise [XDG_CACHE_HOME/raven/runs]. *)

val latest : ?base_dir:string -> unit -> Run.t option
(** [latest ?base_dir ()] returns the most recent run, or [None] if no runs
    exist.

    @param base_dir See {!discover}. *)

(** {1 Creation} *)

val create_run :
  ?base_dir:string ->
  ?experiment:string ->
  ?tags:string list ->
  ?config:(string * Jsont.json) list ->
  unit ->
  Run.t
(** [create_run ?base_dir ?experiment ?tags ?config ()] creates a new run
    directory with a unique ID.

    Alias for {!Run.create}.

    @param base_dir
      Parent directory for runs. Defaults to [RAVEN_RUNS_DIR] if set, otherwise
      [XDG_CACHE_HOME/raven/runs].
    @param experiment Optional experiment name, appended to the run ID.
    @param tags Metadata tags stored in the run manifest.
    @param config Hyperparameters/configuration stored in the run manifest.
    @raise Sys_error if the directory or manifest cannot be created.
    @raise Unix.Unix_error if directory creation fails due to permissions. *)
