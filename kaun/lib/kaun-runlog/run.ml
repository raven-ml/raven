(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* ───── Run Type ───── *)

type t = {
  run_id : string;
  created_at : float;
  experiment_name : string option;
  tags : string list;
  config : (string * Yojson.Safe.t) list;
  dir : string;
}

let run_id t = t.run_id
let created_at t = t.created_at
let experiment_name t = t.experiment_name
let tags t = t.tags
let config t = t.config
let dir t = t.dir

(* ───── Path Helpers ───── *)

let manifest_path dir = Filename.concat dir "run.json"
let events_path dir = Filename.concat dir "events.jsonl"

(* ───── Directory Creation ───── *)

let ensure_dir dir =
  let sep = Filename.dir_sep.[0] in
  let parts = String.split_on_char sep dir in
  (* Handle absolute paths: if dir starts with /, first part is empty *)
  let start, parts =
    match parts with
    | "" :: rest -> (Filename.dir_sep, rest)
    | parts -> ("", parts)
  in
  let rec loop acc parts =
    match parts with
    | [] -> ()
    | "" :: rest -> loop acc rest (* skip empty parts *)
    | part :: rest ->
        let next = if acc = "" then part else acc ^ Filename.dir_sep ^ part in
        if not (Sys.file_exists next) then Unix.mkdir next 0o755;
        loop next rest
  in
  loop start parts

(* ───── ID Generation ───── *)

let random_state = lazy (Random.State.make_self_init ())

let generate_id ?experiment () =
  let state = Lazy.force random_state in
  let now = Unix.gettimeofday () in
  let tm = Unix.localtime now in
  let date =
    Printf.sprintf "%04d-%02d-%02d_%02d-%02d-%02d" (tm.Unix.tm_year + 1900)
      (tm.Unix.tm_mon + 1) tm.Unix.tm_mday tm.Unix.tm_hour tm.Unix.tm_min
      tm.Unix.tm_sec
  in
  let suffix = Printf.sprintf "%04x" (Random.State.int state 0x10000) in
  let base = date ^ "_" ^ suffix in
  Option.fold ~none:base ~some:(Printf.sprintf "%s_%s" base) experiment

(* ───── Manifest I/O ───── *)

module Util = Yojson.Safe.Util

let load dir =
  let path = manifest_path dir in
  if not (Sys.file_exists path) then None
  else
    try
      let json = Yojson.Safe.from_file path in
      let run_id = Util.member "run_id" json |> Util.to_string in
      let created_at =
        Util.member "created_at" json
        |> Util.to_number_option |> Option.value ~default:0.0
      in
      let experiment_name =
        Util.member "experiment" json |> Util.to_string_option
      in
      let tags =
        Util.member "tags" json |> Util.to_list
        |> List.filter_map Util.to_string_option
      in
      let config =
        match Util.member "config" json with `Assoc pairs -> pairs | _ -> []
      in
      Some { run_id; created_at; experiment_name; tags; config; dir }
    with _ -> None

let write_manifest t =
  let experiment_field =
    Option.map (fun e -> ("experiment", `String e)) t.experiment_name
    |> Option.to_list
  in
  let config_field =
    match t.config with [] -> [] | pairs -> [ ("config", `Assoc pairs) ]
  in
  let json =
    `Assoc
      ([
         ("schema_version", `Int 1);
         ("run_id", `String t.run_id);
         ("created_at", `Float t.created_at);
         ("tags", `List (List.map (fun s -> `String s) t.tags));
       ]
      @ experiment_field @ config_field)
  in
  let path = manifest_path t.dir in
  let oc = open_out path in
  output_string oc (Yojson.Safe.pretty_to_string json);
  output_char oc '\n';
  close_out oc

let create ?base_dir ?experiment ?(tags = []) ?(config = []) () =
  let base = Option.value base_dir ~default:(Env.base_dir ()) in
  let run_id = generate_id ?experiment () in
  let dir = Filename.concat base run_id in
  ensure_dir dir;
  let t =
    {
      run_id;
      created_at = Unix.gettimeofday ();
      experiment_name = experiment;
      tags;
      config;
      dir;
    }
  in
  write_manifest t;
  t

(* ───── Event Writing ───── *)

let append_event t event =
  let path = events_path t.dir in
  let oc = open_out_gen [ Open_append; Open_creat ] 0o644 path in
  output_string oc (Yojson.Safe.to_string (Event.to_json event));
  output_char oc '\n';
  close_out oc

(* ───── Event Reading (batch) ───── *)

let events t =
  let path = events_path t.dir in
  if not (Sys.file_exists path) then []
  else
    let ic = open_in path in
    let rec read_lines acc =
      match input_line ic with
      | line -> (
          match Yojson.Safe.from_string line |> Event.of_json with
          | Ok ev -> read_lines (ev :: acc)
          | Error _ -> read_lines acc)
      | exception End_of_file ->
          close_in ic;
          List.rev acc
    in
    read_lines []

(* ───── Incremental Event Reading ───── *)

type file_id = int * int (* st_dev, st_ino *)

type event_stream = {
  path : string;
  mutable position : int64;
  mutable last_mtime : float;
  mutable file_id : file_id option;
  mutable channel : in_channel option;
  mutable pending : string;
}

let open_events t =
  {
    path = events_path t.dir;
    position = 0L;
    last_mtime = 0.0;
    file_id = None;
    channel = None;
    pending = "";
  }

let close_events stream =
  Option.iter
    (fun ic ->
      stream.channel <- None;
      try close_in ic with _ -> ())
    stream.channel

let reset_stream stream =
  close_events stream;
  stream.position <- 0L;
  stream.last_mtime <- 0.0;
  stream.file_id <- None;
  stream.pending <- ""

let ensure_channel stream =
  match stream.channel with
  | Some ic -> ic
  | None ->
      let ic = open_in_bin stream.path in
      stream.channel <- Some ic;
      (try
         let st = Unix.fstat (Unix.descr_of_in_channel ic) in
         stream.file_id <- Some (st.Unix.st_dev, st.Unix.st_ino)
       with _ -> ());
      ic

(* JSONL chunk parsing - handles incomplete lines *)

let is_whitespace c =
  match c with ' ' | '\t' | '\r' | '\n' -> true | _ -> false

let is_blank s =
  let len = String.length s in
  let rec loop i =
    if i >= len then true
    else if is_whitespace s.[i] then loop (i + 1)
    else false
  in
  loop 0

let parse_jsonl_chunk chunk =
  let len = String.length chunk in
  let rec scan i line_start acc =
    if i >= len then
      let pending =
        if line_start >= len then ""
        else String.sub chunk line_start (len - line_start)
      in
      (List.rev acc, pending)
    else if chunk.[i] = '\n' then
      let line_len = i - line_start in
      let line =
        if line_len <= 0 then ""
        else
          let effective_len =
            if line_len > 0 && chunk.[i - 1] = '\r' then line_len - 1
            else line_len
          in
          if effective_len <= 0 then ""
          else String.sub chunk line_start effective_len
      in
      let acc =
        if line = "" || is_blank line then acc
        else
          match Yojson.Safe.from_string line |> Event.of_json with
          | Ok ev -> ev :: acc
          | Error _ -> acc
      in
      scan (i + 1) (i + 1) acc
    else scan (i + 1) line_start acc
  in
  scan 0 0 []

let read_events stream =
  if not (Sys.file_exists stream.path) then (
    reset_stream stream;
    [])
  else
    try
      let st = Unix.LargeFile.stat stream.path in
      let path_id : file_id =
        (st.Unix.LargeFile.st_dev, st.Unix.LargeFile.st_ino)
      in
      let file_size = st.Unix.LargeFile.st_size in
      let mtime = st.Unix.LargeFile.st_mtime in

      (* Detect rotation/replacement *)
      let rotated =
        match stream.file_id with
        | None -> false
        | Some (dev, ino) -> dev <> fst path_id || ino <> snd path_id
      in

      (* Detect truncation *)
      let truncated = stream.position > file_size in

      if rotated || truncated then (
        reset_stream stream;
        stream.file_id <- Some path_id);

      (* Fast-path: nothing new *)
      if stream.position >= file_size && mtime <= stream.last_mtime then []
      else
        let ic = ensure_channel stream in
        LargeFile.seek_in ic stream.position;

        let buf = Bytes.create 65536 in
        let b = Buffer.create 65536 in
        let rec read_loop total =
          match input ic buf 0 (Bytes.length buf) with
          | 0 -> total
          | n ->
              Buffer.add_subbytes b buf 0 n;
              read_loop (total + n)
        in
        let bytes_read = read_loop 0 in

        stream.last_mtime <- mtime;

        if bytes_read = 0 then []
        else (
          stream.position <- Int64.add stream.position (Int64.of_int bytes_read);
          let data = Buffer.contents b in
          let chunk =
            if stream.pending = "" then data else stream.pending ^ data
          in
          let events, pending = parse_jsonl_chunk chunk in
          stream.pending <- pending;
          events)
    with
    | Sys_error _ ->
        close_events stream;
        []
    | Unix.Unix_error _ ->
        close_events stream;
        []
