(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Run manifest operations. *)

type t = {
  schema_version : int;
  run_id : string;
  created_at : float;
  experiment : string option;
  tags : string list;
  config : (string * Yojson.Basic.t) list;
}

(* ───── JSON Helpers ───── *)

let get_string key = function
  | `Assoc fields -> (
      match List.assoc_opt key fields with
      | Some (`String s) -> Some s
      | _ -> None)
  | _ -> None

let get_float key = function
  | `Assoc fields -> (
      match List.assoc_opt key fields with
      | Some (`Float f) -> Some f
      | Some (`Int i) -> Some (float_of_int i)
      | _ -> None)
  | _ -> None

let get_string_list key = function
  | `Assoc fields -> (
      match List.assoc_opt key fields with
      | Some (`List items) ->
          List.filter_map (function `String s -> Some s | _ -> None) items
      | _ -> [])
  | _ -> []

(* ───── Manifest Operations ───── *)

let create ~run_id ?experiment ?tags ?(config = []) () =
  {
    schema_version = 1;
    run_id;
    created_at = Unix.gettimeofday ();
    experiment;
    tags = Option.value ~default:[] tags;
    config;
  }

let manifest_path ~run_dir = Filename.concat run_dir "run.json"

let write ~run_dir manifest =
  let path = manifest_path ~run_dir in
  let experiment_field =
    Option.map (fun e -> ("experiment", `String e)) manifest.experiment
    |> Option.to_list
  in
  let json =
    `Assoc
      ([
         ("schema_version", `Int manifest.schema_version);
         ("run_id", `String manifest.run_id);
         ("created_at", `Float manifest.created_at);
         ("tags", `List (List.map (fun t -> `String t) manifest.tags));
         ("config", `Assoc manifest.config);
       ]
      @ experiment_field)
  in
  let oc = open_out path in
  output_string oc (Yojson.Basic.pretty_to_string json);
  output_char oc '\n';
  close_out oc

let read ~run_dir =
  let path = manifest_path ~run_dir in
  if not (Sys.file_exists path) then None
  else
    try
      let json = Yojson.Basic.from_file path in
      match json with
      | `Assoc _ ->
          let run_id =
            match get_string "run_id" json with
            | Some s -> s
            | None -> Filename.basename run_dir
          in
          let created_at =
            match get_float "created_at" json with Some f -> f | None -> 0.0
          in
          let experiment = get_string "experiment" json in
          let tags = get_string_list "tags" json in
          let config =
            match json with
            | `Assoc fields -> (
                match List.assoc_opt "config" fields with
                | Some (`Assoc config_fields) -> config_fields
                | _ -> [])
            | _ -> []
          in
          Some
            {
              schema_version = 1;
              run_id;
              created_at;
              experiment;
              tags;
              config;
            }
      | _ -> None
    with _ -> None

(* ───── Run ID and Directory Utils ───── *)

let generate_run_id ?experiment () =
  let now = Unix.gettimeofday () in
  let tm = Unix.localtime now in
  let date =
    Printf.sprintf "%04d-%02d-%02d_%02d-%02d-%02d" (tm.Unix.tm_year + 1900)
      (tm.Unix.tm_mon + 1) tm.Unix.tm_mday tm.Unix.tm_hour tm.Unix.tm_min
      tm.Unix.tm_sec
  in
  Option.fold ~none:date ~some:(Printf.sprintf "%s_%s" date) experiment

let run_dir ~base_dir ~run_id = Filename.concat base_dir run_id

let events_path ~run_dir = Filename.concat run_dir "events.jsonl"

let ensure_run_dir ~run_dir =
  let rec ensure acc parts =
    match parts with
    | [] -> ()
    | part :: rest ->
        let next = if acc = "" then part else acc ^ "/" ^ part in
        if next <> "" && not (Sys.file_exists next) then Unix.mkdir next 0o755;
        ensure next rest
  in
  ensure "" (String.split_on_char '/' run_dir)
