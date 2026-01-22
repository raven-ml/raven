(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Run discovery for Kaun training logs.

    Scans the runs directory, parses run manifests, and filters by experiment
    or tags. *)

(** Run metadata. *)
type run_info = {
  run_id : string;
  (** Unique run identifier (typically timestamp-based). *)
  created_at : float;
  (** Unix timestamp when the run was created. *)
  experiment : string option;
  (** Experiment name, if specified. *)
  tags : string list;
  (** List of tags associated with this run. *)
}

val parse_manifest : string -> run_info option
(** [parse_manifest run_dir] reads and parses the [run.json] manifest in the
    given directory.

    Returns [None] if the manifest doesn't exist or cannot be parsed. *)

val discover_runs :
  base_dir:string ->
  ?experiment:string ->
  ?tags:string list ->
  unit ->
  run_info list
(** [discover_runs ~base_dir ?experiment ?tags ()] scans the base directory for
    training runs and returns matching runs sorted by creation time (newest
    first).

    @param base_dir Directory containing run subdirectories (default: ["./runs"])
    @param experiment Filter to runs with this experiment name
    @param tags Filter to runs containing all these tags *)

val get_latest_run :
  base_dir:string ->
  ?experiment:string ->
  ?tags:string list ->
  unit ->
  run_info option
(** [get_latest_run ~base_dir ?experiment ?tags ()] returns the most recent run
    matching the given criteria.

    Returns [None] if no matching runs are found. *)
