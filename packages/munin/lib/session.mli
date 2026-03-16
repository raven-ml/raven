(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Active run writers.

    A session is the append-only mutation boundary for a run. All writes go
    through the event log; no direct state mutation. *)

type t
(** The type for active run sessions. *)

(** {1:lifecycle Lifecycle} *)

val start :
  ?root:string ->
  experiment:string ->
  ?name:string ->
  ?group:string ->
  ?parent:Run.t ->
  ?tags:string list ->
  ?params:(string * Value.t) list ->
  ?notes:string ->
  ?capture_env:string list ->
  ?command:string list ->
  ?cwd:string ->
  ?hostname:string ->
  ?pid:int ->
  ?git_commit:string ->
  ?git_dirty:bool ->
  ?env:(string * string) list ->
  unit ->
  t
(** [start ~experiment ()] starts a new run session.

    Creates the store directory structure if it does not exist.

    - [root] defaults to [$RAVEN_TRACKING_DIR] or [$XDG_DATA_HOME/raven/munin].
    - [name] defaults to [None].
    - [tags] defaults to [[]].
    - [params] defaults to [[]].
    - [notes] defaults to [None].
    - [capture_env] is a list of environment variable names to capture into
      provenance. Defaults to [[]].
    - [command] defaults to [Sys.argv].
    - [cwd] defaults to [Sys.getcwd ()].
    - [hostname] defaults to [Unix.gethostname ()].
    - [pid] defaults to [Unix.getpid ()].
    - [git_commit] defaults to the HEAD commit detected from [cwd].
    - [git_dirty] defaults to the working tree status detected from [cwd].
    - [env] defaults to the variables captured via [capture_env]. *)

val with_run :
  ?root:string ->
  experiment:string ->
  ?name:string ->
  ?parent:Run.t ->
  ?tags:string list ->
  ?params:(string * Value.t) list ->
  ?notes:string ->
  ?capture_env:string list ->
  (t -> 'a) ->
  'a
(** [with_run ~experiment f] starts a run, calls [f], and finishes the run as
    [`finished] on success or [`failed] on exception. The exception is re-raised
    after the run is closed.

    Optional arguments default as in {!start}. *)

val resume : Run.t -> t
(** [resume run] reopens an unfinished run for additional logging.

    Raises [Invalid_argument] if [Run.resumable run] is [false]. *)

val run : t -> Run.t
(** [run t] is the current materialized view of the run.

    Raises [Failure] if the run manifest is missing. *)

val finish : ?status:[ `finished | `failed | `killed ] -> t -> unit -> unit
(** [finish t ()] closes the run with the given final status.

    [status] defaults to [`finished]. The trailing [unit] argument allows
    partial application as a finalizer (e.g.
    [Fun.protect ~finally:(finish session)]). Calling [finish] on an
    already-closed session is a no-op. *)

(** {1:scalars Scalars} *)

val log_metric : t -> step:int -> ?timestamp:float -> string -> float -> unit
(** [log_metric t ~step key value] appends a scalar metric sample.

    [timestamp] defaults to [Unix.gettimeofday ()]. Silently ignored if the
    session is closed. *)

val log_metrics :
  t -> step:int -> ?timestamp:float -> (string * float) list -> unit
(** [log_metrics t ~step pairs] appends multiple scalar metric samples
    atomically.

    [timestamp] defaults to [Unix.gettimeofday ()]. *)

(** {1:metric_defs Metric definitions} *)

val define_metric :
  t ->
  string ->
  ?summary:[ `Min | `Max | `Mean | `Last | `None ] ->
  ?step_metric:string ->
  ?goal:[ `Minimize | `Maximize ] ->
  unit ->
  unit
(** [define_metric t key ()] declares how a metric should be summarised and
    plotted.

    - [summary] controls how the run summary value is computed from history:
      [`Min] (best for loss), [`Max] (best for accuracy), [`Mean], [`Last]
      (default), [`None] (no auto-summary).
    - [step_metric] specifies another metric as x-axis (e.g. ["epoch"]).
      Defaults to [None].
    - [goal] declares whether lower ([`Minimize]) or higher ([`Maximize]) is
      better, used by the TUI for "best" badges and by comparisons. Defaults to
      [None]. *)

(** {1:media Media} *)

val log_media :
  t ->
  step:int ->
  key:string ->
  kind:[ `Image | `Audio | `Table | `File ] ->
  path:string ->
  unit
(** [log_media t ~step ~key ~kind ~path] copies [path] into the run's [media/]
    directory and appends a media event to the log.

    The file is stored at [<run_dir>/media/<key_path>_<step>.<ext>] where
    [<key_path>] preserves the key's slash-delimited hierarchy as directories.
    [kind] is metadata for renderers; the TUI ignores media events. Silently
    ignored if the session is closed.

    @raise Invalid_argument if [path] does not exist. *)

val log_table :
  t ->
  step:int ->
  key:string ->
  columns:string list ->
  rows:Value.t list list ->
  unit
(** [log_table t ~step ~key ~columns ~rows] stores a table as JSON in the run's
    [media/] directory and appends a media event with [kind = `Table].

    The JSON file has the structure [{"columns": [...], "rows": [...]}]. Useful
    for confusion matrices, per-class metrics, data samples. *)

(** {1:metadata Metadata} *)

val set_notes : t -> string option -> unit
(** [set_notes t note] replaces the run note. [None] clears it. *)

val set_summary : t -> (string * Value.t) list -> unit
(** [set_summary t values] merges summary values into the run. Later writes
    replace earlier values for the same key. *)

val add_tags : t -> string list -> unit
(** [add_tags t tags] appends tags to the run. Duplicate tags are ignored by
    readers. Empty lists are not written. *)

(** {1:artifacts Artifacts} *)

val log_artifact :
  t ->
  name:string ->
  kind:Artifact.kind ->
  path:string ->
  ?metadata:(string * Value.t) list ->
  ?aliases:string list ->
  unit ->
  Artifact.t
(** [log_artifact t ~name ~kind ~path ()] stores [path] as a versioned artifact,
    records it as an output of [t], and returns the created version.

    - [metadata] defaults to [[]].
    - [aliases] defaults to [[]].

    Raises [Failure] if the session is closed. Raises [Invalid_argument] if
    [path] does not exist. *)

val use_artifact : t -> Artifact.t -> unit
(** [use_artifact t artifact] records [artifact] as an input of [t]. *)
