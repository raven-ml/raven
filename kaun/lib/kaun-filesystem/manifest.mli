(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Run manifest (run.json) operations.

    The manifest stores immutable run metadata written once at creation. *)

(** Manifest type - single source of truth for run metadata. *)
type t = {
  schema_version : int;
  (** Schema version for format compatibility. *)
  run_id : string;
  (** Unique run identifier (typically timestamp-based). *)
  created_at : float;
  experiment : string option;
  (** Experiment name, if specified. *)
  tags : string list;
  (** List of tags associated with this run. *)
  config : (string * Yojson.Basic.t) list;

}

val create :
  run_id:string ->
  ?experiment:string ->
  ?tags:string list ->
  ?config:(string * Yojson.Basic.t) list ->
  unit ->
  t
(** [create ~run_id ?experiment ?tags ?config ()] creates a new manifest.

    @param created_at is automatically set to current time. *)

val write : run_dir:string -> t -> unit
(** [write ~run_dir manifest] writes the manifest to [run.json] in the given
    directory. *)

val read : run_dir:string -> t option
(** [read ~run_dir] reads and parses the manifest from [run.json] in the given
    directory.

    Returns [None] if the manifest doesn't exist or cannot be parsed. *)

val manifest_path : run_dir:string -> string
(** [manifest_path ~run_dir] returns the path to the manifest file. *)

(** {1 Run ID and Directory Utilities} *)

val generate_run_id : ?experiment:string -> unit -> string
(** [generate_run_id ?experiment ()] generates a timestamp-based run ID.

    Format: [YYYY-MM-DD_HH-MM-SS] or [YYYY-MM-DD_HH-MM-SS_experiment] if
    experiment is provided. *)

val run_dir : base_dir:string -> run_id:string -> string
(** [run_dir ~base_dir ~run_id] returns the full path to a run directory. *)

val events_path : run_dir:string -> string
(** [events_path ~run_dir] returns the path to the events.jsonl file. *)

val ensure_run_dir : run_dir:string -> unit
(** [ensure_run_dir ~run_dir] creates the run directory and all parent
    directories if they don't exist. *)
