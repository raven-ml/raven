(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Incremental JSONL event reader with position tracking. *)

(* ───── Event Types ───── *)

type scalar = { step : int; epoch : int; tag : string; value : float }
type event = Scalar of scalar | Unknown of Yojson.Basic.t

(* ───── Reader State ───── *)

type t = {
  file_path : string;
  mutable position : int64;
  mutable last_mtime : float;
  mutable channel : in_channel option;
}

(* ───── JSON Parsing Helpers ───── *)

let get_string key fields =
  match List.assoc_opt key fields with
  | Some (`String s) -> Some s
  | _ -> None

let get_int key fields =
  match List.assoc_opt key fields with
  | Some (`Int i) -> Some i
  | _ -> None

let get_float key fields =
  match List.assoc_opt key fields with
  | Some (`Float f) -> Some f
  | Some (`Int i) -> Some (float_of_int i)
  | _ -> None

let parse_event json =
  match json with
  | `Assoc fields -> (
      match get_string "type" fields with
      | Some "scalar" -> (
          match
            ( get_int "step" fields,
              get_string "tag" fields,
              get_float "value" fields )
          with
          | Some step, Some tag, Some value ->
              let epoch = Option.value ~default:0 (get_int "epoch" fields) in
              Scalar { step; epoch; tag; value }
          | _ -> Unknown json)
      | _ -> Unknown json)
  | _ -> Unknown json

let parse_event_line line =
  match Yojson.Basic.from_string line with
  | json -> Some (parse_event json)
  | exception Yojson.Json_error _ -> None

(* ───── Reader Implementation ───── *)

let create ~file_path =
  { file_path; position = 0L; last_mtime = 0.0; channel = None }

let close reader =
  match reader.channel with
  | Some ic ->
      close_in ic;
      reader.channel <- None
  | None -> ()

let reset reader =
  close reader;
  reader.position <- 0L;
  reader.last_mtime <- 0.0

let file_exists reader = Sys.file_exists reader.file_path

let ensure_channel reader =
  match reader.channel with
  | Some ic -> ic
  | None ->
      let ic = open_in reader.file_path in
      reader.channel <- Some ic;
      ic

let read_new reader =
  (* Check if file exists *)
  if not (file_exists reader) then []
  else
    try
      (* Check if file was modified *)
      let stat = Unix.stat reader.file_path in
      let file_size = Int64.of_int stat.Unix.st_size in

      (* If file was truncated or recreated, reset position *)
      if reader.position > file_size then (
        reset reader;
        reader.last_mtime <- stat.Unix.st_mtime);

      (* If file hasn't been modified and we're at the end, nothing new *)
      if
        stat.Unix.st_mtime <= reader.last_mtime
        && reader.position >= file_size
      then []
      else
        let ic = ensure_channel reader in

        (* Seek to last position *)
        let pos_int = Int64.to_int reader.position in
        seek_in ic pos_int;

        (* Read new lines *)
        let rec read_lines acc =
          try
            let line = input_line ic in
            let acc =
              match parse_event_line line with
              | Some event -> event :: acc
              | None -> acc
              (* Skip invalid lines *)
            in
            read_lines acc
          with End_of_file ->
            (* Update position and mtime *)
            let new_pos = pos_in ic in
            reader.position <- Int64.of_int new_pos;
            reader.last_mtime <- stat.Unix.st_mtime;
            List.rev acc
        in

        read_lines []
    with
    | Sys_error _ ->
        (* File disappeared or permission denied *)
        close reader;
        []
    | Unix.Unix_error _ ->
        (* Stat failed *)
        close reader;
        []
