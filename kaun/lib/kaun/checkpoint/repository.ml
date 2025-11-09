type error =
  | Io of string
  | Json of string
  | Corrupt of string
  | Not_found of string
  | Duplicate_slug of string
  | Invalid of string

type retention = { max_to_keep : int option; keep_every : int option }
type t = { root : string; retention : retention }

open Artifact
open Manifest

let root repo = repo.root
let retention repo = repo.retention
let ( let* ) = Result.bind
let io msg = Error (Io msg)
let not_found msg = Error (Not_found msg)
let invalid msg = Error (Invalid msg)

let wrap_io f =
  try Ok (f ()) with
  | Sys_error msg -> io msg
  | Unix.Unix_error (err, fn, arg) ->
      let message =
        Printf.sprintf "%s(%s): %s" fn arg (Unix.error_message err)
      in
      io message

let remove_if_exists path =
  if Sys.file_exists path then wrap_io (fun () -> Util.remove_tree path)
  else Ok ()

let ensure_unique_slugs artifacts =
  let seen = Hashtbl.create (List.length artifacts) in
  let rec aux = function
    | [] -> Ok ()
    | artifact :: rest ->
        let slug = Artifact.slug artifact in
        if Hashtbl.mem seen slug then Error (Duplicate_slug slug)
        else (
          Hashtbl.add seen slug ();
          aux rest)
  in
  aux artifacts

let create ~root ?(retention = { max_to_keep = Some 5; keep_every = None }) () =
  Util.mkdir_p root;
  { root; retention }

let checkpoint_dir repo step =
  Filename.concat repo.root (Printf.sprintf "ckpt-%010d" step)

let manifest_path repo ~step =
  Filename.concat (checkpoint_dir repo step) "manifest.json"

let write_manifest path manifest =
  wrap_io (fun () -> Yojson.Basic.to_file path (Manifest.to_yojson manifest))

let load_manifest repo ~step =
  let path = manifest_path repo ~step in
  if not (Sys.file_exists path) then
    not_found (Printf.sprintf "Manifest not found for step %d" step)
  else
    try
      let json = Yojson.Basic.from_file path in
      match Manifest.of_yojson json with
      | Ok manifest -> Ok manifest
      | Error msg -> Error (Json msg)
    with
    | Sys_error msg -> Error (Io msg)
    | Yojson.Json_error msg -> Error (Json msg)

let artifact_base_dir base slug = Filename.concat base slug

let store_artifacts base artifacts =
  let rec aux = function
    | [] -> Ok ()
    | artifact :: rest -> (
        let slug = Artifact.slug artifact in
        let base_path = artifact_base_dir base slug in
        match
          try
            Snapshot_store.save ~base_path artifact.snapshot;
            Ok ()
          with
          | Sys_error msg -> io msg
          | Failure msg -> invalid msg
        with
        | Ok () -> aux rest
        | Error _ as err -> err)
  in
  aux artifacts

let artifact_entry_of_artifact (artifact : Artifact.t) =
  {
    kind = artifact.kind;
    label = artifact.label;
    slug = Artifact.slug artifact;
  }

let parse_step_name entry =
  let prefix = "ckpt-" in
  if
    String.length entry >= String.length prefix
    && String.sub entry 0 (String.length prefix) = prefix
  then
    let step_str =
      String.sub entry (String.length prefix)
        (String.length entry - String.length prefix)
    in
    try Some (int_of_string step_str) with Failure _ -> None
  else None

let steps repo =
  if Sys.file_exists repo.root then
    Sys.readdir repo.root |> Array.to_list
    |> List.filter_map parse_step_name
    |> List.sort compare
  else []

let retention_keep retention step =
  match retention.keep_every with
  | Some n when n > 0 && step mod n = 0 -> true
  | _ -> false

let cleanup repo =
  match repo.retention.max_to_keep with
  | None -> Ok ()
  | Some max when max <= 0 -> Ok ()
  | Some max ->
      let deletable =
        steps repo
        |> List.filter (fun step -> not (retention_keep repo.retention step))
      in
      let excess = List.length deletable - max in
      if excess <= 0 then Ok ()
      else
        let sorted = List.sort compare deletable in
        let rec take_first n lst acc =
          match (n, lst) with
          | 0, _ | _, [] -> List.rev acc
          | n, hd :: tl -> take_first (n - 1) tl (hd :: acc)
        in
        let to_delete = take_first excess sorted [] in
        List.fold_left
          (fun acc step ->
            let dir = checkpoint_dir repo step in
            let* () = acc in
            remove_if_exists dir)
          (Ok ()) to_delete

let classify_snapshot_error slug step msg =
  let context =
    Printf.sprintf "Failed to load artifact %s for step %d: %s" slug step msg
  in
  if String.starts_with ~prefix:"Snapshot_store" msg then Corrupt context
  else Io context

let load_artifact repo ~step ~artifact =
  let base = artifact_base_dir (checkpoint_dir repo step) artifact.slug in
  match Snapshot_store.load ~base_path:base with
  | Ok snapshot -> Ok { kind = artifact.kind; label = artifact.label; snapshot }
  | Error msg -> Error (classify_snapshot_error artifact.slug step msg)

let write ~step ?tags ?metadata ~artifacts repo =
  let final_dir = checkpoint_dir repo step in
  let tmp_dir =
    Printf.sprintf "%s.tmp-%d-%06x" final_dir (Unix.getpid ()) (Random.bits ())
  in
  let backup_dir = final_dir ^ ".old" in
  let cleanup_tmp () =
    let _ = remove_if_exists tmp_dir in
    ()
  in
  Fun.protect ~finally:cleanup_tmp (fun () ->
      let* () = ensure_unique_slugs artifacts in
      let* () = wrap_io (fun () -> Util.mkdir_p repo.root) in
      let* () = remove_if_exists tmp_dir in
      let* () = wrap_io (fun () -> Util.mkdir_p tmp_dir) in
      let* () = store_artifacts tmp_dir artifacts in
      let manifest =
        Manifest.create ?step:(Some step) ?tags ?metadata
          ~artifacts:(List.map artifact_entry_of_artifact artifacts)
          ()
      in
      let tmp_manifest = Filename.concat tmp_dir "manifest.json" in
      let* () = write_manifest tmp_manifest manifest in
      let rollback_backup () =
        let _ =
          if Sys.file_exists backup_dir then
            wrap_io (fun () -> Unix.rename backup_dir final_dir)
          else Ok ()
        in
        ()
      in
      let* () = remove_if_exists backup_dir in
      let* () =
        if Sys.file_exists final_dir then
          match wrap_io (fun () -> Unix.rename final_dir backup_dir) with
          | Ok () -> Ok ()
          | Error _ as err ->
              let _ = remove_if_exists backup_dir in
              err
        else Ok ()
      in
      match wrap_io (fun () -> Unix.rename tmp_dir final_dir) with
      | Error _ as err ->
          rollback_backup ();
          err
      | Ok () ->
          let* () = remove_if_exists backup_dir in
          let* () = cleanup repo in
          Ok manifest)

let latest_step repo =
  match List.rev (steps repo) with step :: _ -> Some step | [] -> None

let mem repo ~step = Sys.file_exists (checkpoint_dir repo step)

let read repo ~step =
  let* manifest = load_manifest repo ~step in
  let rec load_all acc = function
    | [] -> Ok (List.rev acc)
    | entry :: rest -> (
        match load_artifact repo ~step ~artifact:entry with
        | Ok artifact -> load_all (artifact :: acc) rest
        | Error _ as err -> err)
  in
  let* artifacts = load_all [] manifest.Manifest.artifacts in
  Ok (manifest, artifacts)

let read_latest repo =
  match latest_step repo with
  | None -> not_found "No checkpoints available"
  | Some step -> read repo ~step

let delete repo ~step =
  let dir = checkpoint_dir repo step in
  if Sys.file_exists dir then remove_if_exists dir else Ok ()

let load_artifact_snapshot repo ~step ~slug =
  let* manifest = load_manifest repo ~step in
  match
    List.find_opt (fun entry -> String.equal entry.slug slug) manifest.artifacts
  with
  | None ->
      not_found (Printf.sprintf "Artifact %s not found for step %d" slug step)
  | Some entry ->
      let* artifact = load_artifact repo ~step ~artifact:entry in
      Ok artifact.snapshot
