(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type retention = { max_to_keep : int option; keep_every : int option }
type t

type error =
  | Io of string
  | Json of string
  | Corrupt of string
  | Not_found of string
  | Duplicate_slug of string
  | Invalid of string

val create : root:string -> ?retention:retention -> unit -> t
(** Create or open a repository rooted at [root]. The directory is created if it
    does not exist. *)

val root : t -> string
val retention : t -> retention

val write :
  step:int ->
  ?tags:string list ->
  ?metadata:(string * string) list ->
  artifacts:Artifact.t list ->
  t ->
  (Manifest.t, error) result
(** Write a new checkpoint version for the provided [step]. Existing checkpoints
    for the same step are replaced. Retention rules are enforced after writing.
*)

val manifest_path : t -> step:int -> string
val load_manifest : t -> step:int -> (Manifest.t, error) result

val load_artifact :
  t ->
  step:int ->
  artifact:Manifest.artifact_entry ->
  (Artifact.t, error) result

val read : t -> step:int -> (Manifest.t * Artifact.t list, error) result
(** Load the manifest and all artifacts for a given step. *)

val read_latest : t -> (Manifest.t * Artifact.t list, error) result
(** Load the most recent checkpoint. *)

val latest_step : t -> int option
val steps : t -> int list
val mem : t -> step:int -> bool
val delete : t -> step:int -> (unit, error) result

val load_artifact_snapshot :
  t -> step:int -> slug:string -> (Snapshot.t, error) result
