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
  Fs.write_file (index_path root)
    (Json_utils.json_to_string ~pretty:true json ^ "\n")

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

let read root = reconcile root (load root)

let find root ~id =
  let cached = load root in
  Fs.list_dirs (experiments_dir root)
  |> List.find_map (fun experiment ->
      if not (Fs.is_directory (Filename.concat (runs_dir root experiment) id))
      then None
      else header root cached ~experiment ~id)

let modify root f =
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
