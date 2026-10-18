(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type status = [ `Running | `Finished | `Failed | `Killed ]

type entry = {
  experiment : string;
  name : string option;
  group : string option;
  parent_id : string option;
  status : status;
  tags : string list;
  started_at : float;
}

let index_path root = Filename.concat root "index.json"

let status_to_string = function
  | `Running -> "running"
  | `Finished -> "finished"
  | `Failed -> "failed"
  | `Killed -> "killed"

let entry_to_json entry =
  Json_utils.json_obj
    ([
       ("experiment", Jsont.Json.string entry.experiment);
       ("status", Jsont.Json.string (status_to_string entry.status));
       ("tags", Jsont.Json.list (List.map Jsont.Json.string entry.tags));
       ("started_at", Jsont.Json.number entry.started_at);
     ]
    @ (match entry.name with
      | Some n -> [ ("name", Jsont.Json.string n) ]
      | None -> [])
    @ (match entry.group with
      | Some g -> [ ("group", Jsont.Json.string g) ]
      | None -> [])
    @
    match entry.parent_id with
    | Some p -> [ ("parent_id", Jsont.Json.string p) ]
    | None -> [])

let entry_of_json json =
  match Json_utils.json_mem "experiment" json |> Json_utils.json_string with
  | None -> None
  | Some experiment ->
      let status : status =
        match Json_utils.json_mem "status" json |> Json_utils.json_string with
        | Some "finished" -> `Finished
        | Some "failed" -> `Failed
        | Some "killed" -> `Killed
        | _ -> `Running
      in
      let tags =
        Json_utils.json_mem "tags" json |> Json_utils.json_string_list
      in
      let started_at =
        Option.value
          (Json_utils.json_mem "started_at" json |> Json_utils.json_number)
          ~default:0.0
      in
      Some
        {
          experiment;
          name = Json_utils.json_mem "name" json |> Json_utils.json_string;
          group = Json_utils.json_mem "group" json |> Json_utils.json_string;
          parent_id =
            Json_utils.json_mem "parent_id" json |> Json_utils.json_string;
          status;
          tags;
          started_at;
        }

(* A change reads the index, changes it and writes it back under an exclusive
   lock on [index.lock], and a read holds a shared one, so changes from several
   processes apply one after another. A write goes to a temporary file that then
   replaces the index whole, so a reader never meets half an index. Windows also
   refuses to replace a file that a reader holds open, which the shared lock
   rules out. [Unix.lockf] locks belong to a process, and closing any of its
   descriptors on the file drops them, so the threads and domains of one process
   take turns under [mutex] first. *)

let lock_path root = Filename.concat root "index.lock"
let temp_path root = Filename.concat root "index.json.tmp"
let mutex = Mutex.create ()

(* [lock fd command] is [false] if the file system refuses locks: NFS without
   its lock daemon answers ENOLCK, Lustre without flock ENOSYS, others
   EOPNOTSUPP, EINVAL or an ENOTSUP the Unix library does not name. A wait
   that a signal interrupts is taken again. *)
let rec lock fd command =
  match Unix.lockf fd command 0 with
  | () -> true
  | exception Unix.Unix_error (EINTR, _, _) -> lock fd command
  | exception
      Unix.Unix_error
        ((ENOLCK | ENOSYS | EOPNOTSUPP | EINVAL | EUNKNOWNERR _), _, _) ->
      false

(* [with_lock root mode ~refused f] is [f ()] under the lock, or [refused ()]
   where the file system refuses it. *)
let with_lock root mode ~refused f =
  let flags, command =
    match mode with
    | `Read -> ([ Unix.O_RDONLY; O_CREAT; O_CLOEXEC ], Unix.F_RLOCK)
    | `Write -> ([ Unix.O_RDWR; O_CREAT; O_CLOEXEC ], Unix.F_LOCK)
  in
  Mutex.protect mutex @@ fun () ->
  let fd = Unix.openfile (lock_path root) flags 0o644 in
  Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
  if not (lock fd command) then refused ()
  else Fun.protect ~finally:(fun () -> Unix.lockf fd F_ULOCK 0) f

(* The entries of [index.json], none if it is missing or damaged. *)
let load root =
  let tbl = Hashtbl.create 64 in
  let path = index_path root in
  (if Sys.file_exists path then
     try
       Fs.read_file path |> Json_utils.json_of_string |> Json_utils.json_assoc
       |> List.iter (fun (id, value) ->
           match entry_of_json value with
           | Some entry -> Hashtbl.replace tbl id entry
           | None -> ())
     with _ -> ());
  tbl

let write root tbl =
  let entries =
    Hashtbl.to_seq tbl |> List.of_seq
    |> List.sort (fun (a, _) (b, _) -> String.compare b a)
    |> List.map (fun (id, entry) -> (id, entry_to_json entry))
  in
  let json = Json_utils.json_obj entries in
  Fs.write_file (temp_path root)
    (Json_utils.json_to_string ~pretty:true json ^ "\n");
  Unix.rename (temp_path root) (index_path root)

let entry_of_run run =
  {
    experiment = Run.experiment run;
    name = Run.name run;
    group = Run.group run;
    parent_id = Run.parent_id run;
    status = Run.status run;
    tags = Run.tags run;
    started_at = Run.started_at run;
  }

(* The runs on disk are the directories under [experiments/*/runs]; the index
   only saves reading their manifests. A run the index lacks, as a process
   killed between writing a run's manifest and its entry leaves, has its
   manifest read. An entry whose directory is gone is dropped. *)

let experiments_dir root = Filename.concat root "experiments"

let runs_dir root experiment =
  Filename.concat (Filename.concat (experiments_dir root) experiment) "runs"

let header root cached ~experiment ~id =
  match Hashtbl.find_opt cached id with
  | Some entry when String.equal entry.experiment experiment -> Some entry
  | _ -> Option.map entry_of_run (Run.load ~root ~experiment ~id)

let reconcile root cached =
  let tbl = Hashtbl.create 64 in
  Fs.list_dirs (experiments_dir root)
  |> List.iter (fun experiment ->
      Fs.list_dirs (runs_dir root experiment)
      |> List.iter (fun id ->
          match header root cached ~experiment ~id with
          | Some entry -> Hashtbl.replace tbl id entry
          | None -> ()));
  tbl

(* A read takes no lock where the file system refuses locks, and there a change
   removes the index, so a read meets no index and reads every manifest. A
   store whose lock file cannot be opened, as on a read-only file system with
   none yet, reads the same way. *)
let cached root =
  if not (Sys.file_exists (index_path root)) then Hashtbl.create 0
  else
    try
      with_lock root `Read ~refused:(fun () -> Hashtbl.create 0) (fun () ->
          load root)
    with Unix.Unix_error _ -> Hashtbl.create 0

let read root = reconcile root (cached root)

let find root ~id =
  let cached = cached root in
  Fs.list_dirs (experiments_dir root)
  |> List.find_map (fun experiment ->
      if not (Fs.is_directory (Filename.concat (runs_dir root experiment) id))
      then None
      else header root cached ~experiment ~id)

(* Where locks are refused, two changes could each write back an index that
   lacks the other's, so the index is removed instead and reads fall back to
   the manifests. *)
let unlink root =
  try Unix.unlink (index_path root)
  with Unix.Unix_error (ENOENT, _, _) -> ()

let modify root f =
  Fs.ensure_dir root;
  with_lock root `Write ~refused:(fun () -> unlink root) @@ fun () ->
  let tbl = load root in
  f tbl;
  write root (reconcile root tbl)

let add root ~id entry = modify root (fun tbl -> Hashtbl.replace tbl id entry)

let update_status root ~id status =
  modify root (fun tbl ->
      match Hashtbl.find_opt tbl id with
      | Some entry -> Hashtbl.replace tbl id { entry with status }
      | None -> ())

let remove root ~id = modify root (fun tbl -> Hashtbl.remove tbl id)
