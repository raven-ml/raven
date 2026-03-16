(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Local tracking stores.

    A store is the local root directory containing experiments, runs, artifacts,
    and blobs. *)

type t
(** The type for store handles. *)

val open_ : ?root:string -> unit -> t
(** [open_ ()] opens the local tracking store, creating its root directories if
    needed.

    [root] defaults to [$RAVEN_TRACKING_DIR] or [$XDG_DATA_HOME/raven/munin]. *)

val root : t -> string
(** [root t] is the absolute store root path. *)

(** {1:runs Runs} *)

val list_experiments : t -> string list
(** [list_experiments t] is the experiment names stored in [t]. *)

val list_runs :
  t ->
  ?experiment:string ->
  ?status:Run.status ->
  ?tag:string ->
  ?parent:string ->
  ?group:string ->
  unit ->
  Run.t list
(** [list_runs t ()] is the persisted runs in [t], filtered when the optional
    selectors are provided. When [experiment] is omitted, searches across all
    experiments. Results are sorted by identifier descending (newest first). *)

val find_run : t -> string -> Run.t option
(** [find_run t id] is the run with identifier [id], if present. Searches across
    all experiments. *)

val latest_run :
  t ->
  ?experiment:string ->
  ?status:Run.status ->
  ?tag:string ->
  ?group:string ->
  unit ->
  Run.t option
(** [latest_run t ()] is the most recently started run matching the optional
    filters, by identifier ordering. *)

(** {1:artifacts Artifacts} *)

val find_artifact : t -> name:string -> version:string -> Artifact.t option
(** [find_artifact t ~name ~version] is the named artifact version or alias, if
    present. *)

val list_artifacts :
  t ->
  ?name:string ->
  ?kind:Artifact.kind ->
  ?alias:string ->
  ?producer_run:string ->
  ?consumer_run:string ->
  unit ->
  Artifact.t list
(** [list_artifacts t ()] is the stored artifacts in [t], filtered when the
    optional selectors are provided. *)

(** {1:maintenance Maintenance} *)

val delete_run : t -> Run.t -> unit
(** [delete_run t run] removes [run] and its event log from the store. Does not
    remove shared blobs. Removes the experiment directory if no runs remain. *)

val gc : t -> int
(** [gc t] removes unreferenced blobs. Returns the number removed. *)
