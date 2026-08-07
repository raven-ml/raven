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
  ?store:Store.t ->
  ?name:string ->
  ?group:string ->
  ?parent:Run.t ->
  ?tags:string list ->
  ?params:(string * Value.t) list ->
  ?notes:string ->
  ?provenance:Provenance.t ->
  experiment:string ->
  unit ->
  t
(** [start ~experiment ()] starts a new run session in [store].

    - [store] defaults to [Store.open_ ()].
    - [name], [group], [parent], and [notes] default to [None].
    - [tags] and [params] default to [[]].
    - [provenance] defaults to [Provenance.detect ()]. To record environment
      variables, pass
      [~provenance:(Provenance.detect ~capture_env:["CUDA_VISIBLE_DEVICES"] ())].
*)

val with_run :
  ?store:Store.t ->
  ?name:string ->
  ?group:string ->
  ?parent:Run.t ->
  ?tags:string list ->
  ?params:(string * Value.t) list ->
  ?notes:string ->
  ?provenance:Provenance.t ->
  experiment:string ->
  (t -> 'a) ->
  'a
(** [with_run ~experiment f] starts a run, calls [f], and finishes the run as
    [`Finished] on success or [`Failed] on exception. The exception is re-raised
    after the run is closed.

    Optional arguments default as in {!start}. *)

val resume : Run.t -> t
(** [resume run] reopens an unfinished run for additional logging.

    Raises [Invalid_argument] if [Run.resumable run] is [false]. *)

val id : t -> string
(** [id t] is the unique run identifier. *)

val dir : t -> string
(** [dir t] is the absolute path to the run directory. *)

val run : t -> Run.t
(** [run t] is the current materialized view of the run.

    Raises [Failure] if the run manifest is missing. *)

val finish : ?status:[ `Finished | `Failed | `Killed ] -> t -> unit
(** [finish t] closes the run with the given final status.

    [status] defaults to [`Finished]. Calling [finish] on an already-closed
    session is a no-op. *)

(** {1:metrics Metrics} *)

val metric :
  t ->
  ?summary:Metric.summary ->
  ?goal:Metric.goal ->
  ?step_metric:Metric.t ->
  string ->
  Metric.t
(** [metric t key] declares the scalar metric [key] on [t] and is the handle
    that logs it, with {!Metric.log}. Declaring is the only way to name a
    metric, so a key is spelled exactly once.

    - [summary] controls how the run summary value is computed from the metric's
      history: [`Min] (best for loss), [`Max] (best for accuracy), [`Mean],
      [`Last], or [`None] for no auto-summary. Defaults to the mode [goal]
      implies — [`Min] for [`Minimize], [`Max] for [`Maximize] — or to [`Last]
      when [goal] is omitted.
    - [goal] declares whether lower ([`Minimize]) or higher ([`Maximize]) values
      are better, used by the TUI for "best" badges and by comparisons. Defaults
      to [None].
    - [step_metric] is another metric to plot [key] against instead of the step
      counter; pass its handle, which means declaring it first. An epoch axis is
      just a metric of its own. Defaults to [None].

    Declaring the same key twice replaces its definition; readers see the last
    one. Both handles stay valid and write to the same key. *)

val log_metrics :
  t -> step:int -> ?timestamp:float -> (Metric.t * float) list -> unit
(** [log_metrics t ~step pairs] appends one sample per metric of [pairs] at
    [step]. The samples are distinct entries in the event log sharing a single
    timestamp, so readers see them at the same instant.

    [timestamp] defaults to [Unix.gettimeofday ()]. Silently ignored if the
    session is closed. *)

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
