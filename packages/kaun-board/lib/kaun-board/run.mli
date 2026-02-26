(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Training runs.

    A run is a directory on disk containing a [run.json] manifest and an
    append-only [events.jsonl] log. Runs are created by {!Log} and read back
    here for dashboards and analysis.

    Run IDs follow the format [YYYY-MM-DD_HH-MM-SS_XXXX] where [XXXX] is a
    random hex suffix. When an experiment name is given, it is appended:
    [YYYY-MM-DD_HH-MM-SS_XXXX_experiment]. *)

(** {1:types Types} *)

type t
(** The type for training run handles. *)

(** {1:accessors Accessors} *)

val run_id : t -> string
(** [run_id r] is [r]'s unique identifier. *)

val dir : t -> string
(** [dir r] is the absolute path to [r]'s directory. *)

val created_at : t -> float
(** [created_at r] is the Unix timestamp when [r] was created. *)

val experiment_name : t -> string option
(** [experiment_name r] is [r]'s experiment name, if any. *)

val tags : t -> string list
(** [tags r] is the metadata tags associated with [r]. *)

(** {1:constructors Constructors} *)

val create :
  ?base_dir:string ->
  ?experiment:string ->
  ?tags:string list ->
  ?config:(string * Jsont.json) list ->
  unit ->
  t
(** [create ()] is a new run with a unique ID and an on-disk directory
    containing a [run.json] manifest.

    [base_dir] defaults to the value of [RAVEN_RUNS_DIR], or
    [XDG_CACHE_HOME/raven/runs] when unset. [experiment] is appended to the run
    ID when given. [tags] defaults to [[]]. [config] defaults to [[]] and is
    written to the manifest as-is.

    Raises [Sys_error] if the directory or manifest cannot be created.

    See also {!load}. *)

val load : string -> t option
(** [load dir] is the run in [dir], if [dir] contains a valid [run.json]
    manifest, and [None] otherwise.

    See also {!create}. *)

(** {1:events Events} *)

val append_event : t -> Event.t -> unit
(** [append_event r ev] appends [ev] as a JSON line to [r]'s [events.jsonl]. *)

(** {1:reading Incremental reading} *)

type event_stream
(** The type for incremental event readers. Tracks the file position and detects
    log rotation or truncation. *)

val open_events : t -> event_stream
(** [open_events r] is a stream positioned at the beginning of [r]'s event log.

    See also {!close_events}. *)

val read_events : event_stream -> Event.t list
(** [read_events stream] is the list of events appended since the last call, or
    the empty list when no new data is available.

    Automatically resets to the beginning if the underlying file has been
    rotated or truncated. *)

val close_events : event_stream -> unit
(** [close_events stream] releases the stream's file descriptor. Safe to call
    multiple times. *)
