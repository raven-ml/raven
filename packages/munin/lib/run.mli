(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Persisted tracked runs.

    Runs are the durable tracked objects of Munin. They expose immutable
    manifest data together with materialized state rebuilt from the append-only
    event log. *)

(** {1:types Types} *)

type status =
  [ `Running  (** Run is actively logging. *)
  | `Finished  (** Run completed successfully. *)
  | `Failed  (** Run terminated with an error. *)
  | `Killed  (** Run was manually terminated. *) ]
(** The type for run status values. *)

type media_entry = {
  step : int;  (** Step counter at which the media was logged. *)
  timestamp : float;  (** Wall-clock time. *)
  kind : [ `Image | `Audio | `Table | `File ];  (** Media type for renderers. *)
  path : string;  (** Absolute path to stored file. *)
}
(** The type for media log entries. *)

type t
(** The type for run handles. Obtain one from {!Store.find_run},
    {!Store.list_runs}, {!Store.latest_run}, or {!Session.run}. *)

(** {1:identity Identity} *)

val id : t -> string
(** [id t] is the unique run identifier. *)

val dir : t -> string
(** [dir t] is the absolute path to the run directory. *)

val experiment : t -> string
(** [experiment t] is the containing experiment name. *)

val name : t -> string option
(** [name t] is the optional human-readable run name. *)

val group : t -> string option
(** [group t] is the optional run group for flat grouping (e.g. sweeps). *)

val parent_id : t -> string option
(** [parent_id t] is the parent run identifier, if any. *)

(** {1:status Status} *)

val started_at : t -> float
(** [started_at t] is the run start timestamp. *)

val ended_at : t -> float option
(** [ended_at t] is the run completion timestamp, if any. *)

val status : t -> status
(** [status t] is the current run status. *)

val resumable : t -> bool
(** [resumable t] is [true] iff [status t] is [`Running]. *)

(** {1:provenance Provenance} *)

val provenance : t -> Provenance.t
(** [provenance t] is the run provenance, as recorded when the run started. *)

val notes : t -> string option
(** [notes t] is the latest run note, if any. *)

(** {1:metadata Metadata} *)

val tags : t -> string list
(** [tags t] is the run tag list. *)

val params : t -> (string * Value.t) list
(** [params t] is the immutable run parameter set. *)

val find_param : t -> string -> Value.t option
(** [find_param t key] is the parameter value for [key], if present. *)

val summary : t -> (string * Value.t) list
(** [summary t] is the run summary map, sorted alphabetically by key. Later
    writes replace earlier values. *)

val find_summary : t -> string -> Value.t option
(** [find_summary t key] is the summary value for [key], if present. *)

(** {1:metrics Metrics} *)

(** Metrics are read back by key: keys come off disk, so there is nothing for a
    {!Metric.t} handle to check here. *)

val metric_keys : t -> string list
(** [metric_keys t] is the sorted list of metric keys observed in [t]. *)

val latest_metrics : t -> (string * Metric.sample) list
(** [latest_metrics t] is the latest sample per metric key, sorted
    alphabetically by key. *)

val metric_history : t -> string -> Metric.sample list
(** [metric_history t key] is the full history for [key] in chronological order.
    Returns the empty list if [key] has no samples. *)

val metric_defs : t -> (string * Metric.def) list
(** [metric_defs t] is the metric definitions declared via {!Session.metric},
    sorted alphabetically by key. *)

(** {1:media Media} *)

val media_keys : t -> string list
(** [media_keys t] is the sorted list of media keys logged in [t]. *)

val media_history : t -> string -> media_entry list
(** [media_history t key] is the media entries for [key] in chronological order.
    Returns the empty list if [key] has no entries. *)

(** {1:relations Relations} *)

val input_artifacts : t -> Artifact.t list
(** [input_artifacts t] is the list of artifacts consumed by [t]. *)

val output_artifacts : t -> Artifact.t list
(** [output_artifacts t] is the list of artifacts produced by [t]. *)

(**/**)

val status_of_string : string -> status
val load : root:string -> experiment:string -> id:string -> t option

val list :
  root:string ->
  experiment:string ->
  ?status:status ->
  ?tag:string ->
  ?parent:string ->
  ?group:string ->
  unit ->
  t list

val of_header :
  root:string ->
  id:string ->
  experiment:string ->
  name:string option ->
  group:string option ->
  parent_id:string option ->
  status:status ->
  tags:string list ->
  started_at:float ->
  t

(**/**)
