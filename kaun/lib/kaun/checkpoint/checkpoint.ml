module Snapshot = Snapshot

type artifact_kind = Artifact.kind =
  | Params
  | Optimizer
  | Rng
  | Payload of string
  | Custom of string
  | Unknown of string

type artifact = Artifact.t

type artifact_descriptor = Manifest.artifact_entry = {
  kind : artifact_kind;
  label : string;
  slug : string;
}

type manifest = Manifest.t = {
  version : int;
  step : int option;
  created_at : float;
  tags : string list;
  metadata : (string * string) list;
  artifacts : artifact_descriptor list;
}

type repository = Repository.t

type retention = Repository.retention = {
  max_to_keep : int option;
  keep_every : int option;
}

type metadata = (string * string) list

type error = Repository.error =
  | Io of string
  | Json of string
  | Corrupt of string
  | Not_found of string
  | Duplicate_slug of string
  | Invalid of string

let error_to_string = function
  | Io msg -> msg
  | Json msg -> msg
  | Corrupt msg -> msg
  | Not_found msg -> msg
  | Duplicate_slug msg -> msg
  | Invalid msg -> msg

let artifact ?label ~kind ~snapshot () = Artifact.create ?label kind snapshot
let artifact_kind (artifact : artifact) = artifact.kind
let artifact_label (artifact : artifact) = artifact.label
let artifact_slug (artifact : artifact) = Artifact.slug artifact
let artifact_snapshot (artifact : artifact) = artifact.snapshot

let create_repository ~directory ?retention () =
  Repository.create ~root:directory ?retention ()

let write ~step ?tags ?metadata ~artifacts repository =
  Repository.write repository ~step ?tags ?metadata ~artifacts

let read repo ~step = Repository.read repo ~step
let read_latest repo = Repository.read_latest repo
let steps repo = Repository.steps repo
let latest_step repo = Repository.latest_step repo
let mem repo ~step = Repository.mem repo ~step
let delete repo ~step = Repository.delete repo ~step

let filter_artifacts ?kinds artifacts =
  match kinds with
  | None -> artifacts
  | Some kinds ->
      List.filter
        (fun artifact ->
          let kind = artifact_kind artifact in
          List.exists (fun target -> target = kind) kinds)
        artifacts

let read_artifact_snapshot repository ~step ~slug =
  Repository.load_artifact_snapshot repository ~step ~slug

let classify_snapshot_error path msg =
  let context = Printf.sprintf "Snapshot at %s: %s" path msg in
  if String.starts_with ~prefix:"Snapshot_store" msg then Corrupt context
  else Io context

let save_snapshot_file ~path ~snapshot =
  try
    let dir = Filename.dirname path in
    Util.mkdir_p dir;
    Snapshot_store.save ~base_path:path snapshot;
    Ok ()
  with
  | Sys_error msg -> Error (Io msg)
  | Failure msg -> Error (Invalid msg)

let load_snapshot_file ~path =
  match Snapshot_store.load ~base_path:path with
  | Ok snapshot -> Ok snapshot
  | Error msg -> Error (classify_snapshot_error path msg)

let write_snapshot_file_with ~path ~encode =
  save_snapshot_file ~path ~snapshot:(encode ())

let load_snapshot_file_with ~path ~decode =
  let ( let* ) = Result.bind in
  let* snapshot = load_snapshot_file ~path in
  match decode snapshot with
  | Ok value -> Ok value
  | Error msg -> Error (Invalid msg)

let save_params_file ~path ~params =
  let snapshot = Snapshot.ptree params in
  save_snapshot_file ~path ~snapshot

let load_params_file ~path =
  let ( let* ) = Result.bind in
  let* snapshot = load_snapshot_file ~path in
  match Snapshot.to_ptree snapshot with
  | Ok params -> Ok params
  | Error msg -> Error (Invalid msg)
