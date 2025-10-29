module Snapshot = Snapshot

type artifact_kind =
  | Params
  | Optimizer
  | Rng
  | Payload of string
  | Custom of string
  | Unknown of string

(* TODO: why do we have artifact and artifact_descriptor? seems redundant *)
type artifact

type artifact_descriptor = {
  kind : artifact_kind;
  label : string;
  slug : string;
}

type manifest = {
  version : int;
  step : int option;
  created_at : float;
  tags : string list;
  metadata : (string * string) list;
  artifacts : artifact_descriptor list;
}

type repository
type retention = { max_to_keep : int option; keep_every : int option }
type metadata = (string * string) list

type error =
  | Io of string
  | Json of string
  | Corrupt of string
  | Not_found of string
  | Duplicate_slug of string
  | Invalid of string

val error_to_string : error -> string

val artifact :
  ?label:string -> kind:artifact_kind -> snapshot:Snapshot.t -> unit -> artifact

val artifact_kind : artifact -> artifact_kind
val artifact_label : artifact -> string
val artifact_slug : artifact -> string
val artifact_snapshot : artifact -> Snapshot.t

val create_repository :
  directory:string -> ?retention:retention -> unit -> repository

val write :
  step:int ->
  ?tags:string list ->
  ?metadata:metadata ->
  artifacts:artifact list ->
  repository ->
  (manifest, error) result

val read : repository -> step:int -> (manifest * artifact list, error) result
val read_latest : repository -> (manifest * artifact list, error) result
val steps : repository -> int list
val latest_step : repository -> int option
val mem : repository -> step:int -> bool
val delete : repository -> step:int -> (unit, error) result

val filter_artifacts :
  ?kinds:artifact_kind list -> artifact list -> artifact list

val read_artifact_snapshot :
  repository -> step:int -> slug:string -> (Snapshot.t, error) result

val save_snapshot_file :
  path:string -> snapshot:Snapshot.t -> (unit, error) result

val load_snapshot_file : path:string -> (Snapshot.t, error) result
val save_params_file : path:string -> params:Ptree.t -> (unit, error) result
val load_params_file : path:string -> (Ptree.t, error) result
