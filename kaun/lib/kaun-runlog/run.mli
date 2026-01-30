(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Training run management.

    A run represents a single training session, stored as a directory containing
    a JSON manifest ([run.json]) and an append-only event log ([events.jsonl]).

    Run IDs follow the format [YYYY-MM-DD_HH-MM-SS_XXXX] where [XXXX] is a
    random hex suffix. If an experiment name is provided, it is appended:
    [YYYY-MM-DD_HH-MM-SS_XXXX_experiment]. Timestamps use local time. *)

type t
(** A training run handle. The run directory and manifest are created eagerly
    on {!create}. *)

(** {1 Accessors} *)

val run_id : t -> string
(** [run_id t] returns the unique run identifier. *)

val created_at : t -> float
(** [created_at t] returns the Unix timestamp when the run was created. *)

val experiment_name : t -> string option
(** [experiment_name t] returns the experiment name, if specified at creation. *)

val tags : t -> string list
(** [tags t] returns the metadata tags associated with this run. *)

val config : t -> (string * Yojson.Safe.t) list
(** [config t] returns the hyperparameters/configuration for this run. *)

val dir : t -> string
(** [dir t] returns the absolute path to the run directory. *)

(** {1 Creation and Loading} *)

val create :
  ?base_dir:string ->
  ?experiment:string ->
  ?tags:string list ->
  ?config:(string * Yojson.Safe.t) list ->
  unit ->
  t
(** [create ?base_dir ?experiment ?tags ?config ()] creates a new run directory
    with a unique ID and writes the manifest.

    The run directory is created immediately, including any missing parent
    directories.

    @param base_dir Parent directory for runs. Defaults to [RAVEN_RUNS_DIR] if
    set, otherwise [XDG_CACHE_HOME/raven/runs].
    @param experiment Optional experiment name, appended to the run ID for
    easier identification.
    @param tags Metadata tags stored in the manifest. Defaults to [[]].
    @param config Hyperparameters/configuration stored in the manifest.
    @raise Sys_error if the directory or manifest cannot be created.
    @raise Unix.Unix_error if directory creation fails due to permissions. *)

val load : string -> t option
(** [load dir] loads a run from an existing directory.

    Returns [None] if [dir] does not contain a valid [run.json] manifest. *)

(** {1 Events} *)

val events : t -> Event.t list
(** [events t] reads all events from the run's event log.

    Returns an empty list if no events have been logged. Malformed JSON lines
    are silently skipped. *)

val append_event : t -> Event.t -> unit
(** [append_event t event] appends an event to the run's event log.

    Events are serialized as JSON and written as a single line to
    [events.jsonl]. The file is created if it does not exist.

    @raise Sys_error if the event log cannot be opened or written. *)

(** {1 Incremental Event Reading}

    For monitoring live training runs, use incremental reading to efficiently
    poll for new events without re-reading the entire log. *)

type event_stream
(** Handle for incremental event reading. Tracks file position and detects
    log rotation or truncation. *)

val open_events : t -> event_stream
(** [open_events t] opens the event log for incremental reading.

    The stream starts at position 0. The underlying file is opened lazily on
    the first call to {!read_events}. *)

val read_events : event_stream -> Event.t list
(** [read_events stream] reads events appended since the last call.

    Returns an empty list if no new events are available. Automatically detects
    and handles log rotation (file replaced) or truncation (file shortened) by
    resetting to the beginning.

    Malformed JSON lines are silently skipped. Incomplete lines at the end of
    the file are buffered until the next read. *)

val close_events : event_stream -> unit
(** [close_events stream] closes the event stream and releases file handles.

    Safe to call multiple times. *)
