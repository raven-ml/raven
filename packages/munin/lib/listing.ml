(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type status =
  [ `Running | `Silent | `Crashed | `Finished | `Failed of string | `Killed ]

type stamp = (float * int * int) option

type t = {
  id : string;
  dir : string;
  start : Record.start;
  started_at : float;
  tags : string list;
  notes : string option;
  diff : string option;
  status : status;
  ended_at : float option;
  updated_at : float;
  read_at : float;
  tags_stamp : stamp;
  notes_stamp : stamp;
}

let log dir = Filename.concat dir "events.jsonl"
let mark dir = Filename.concat dir ".deleted"

let deleted dir =
  Sys.file_exists (mark dir) && not (Sys.file_exists (log dir))

(* Status *)

(* A held lock proves a holder whatever the boots. A free lock proves the writer
   gone only to a reader under the writer's boot, whose kernel would show its
   lock. *)
let status ~held ~writer:(machine, boot) ~log_mtime last =
  match last with
  | Some (t, (#Record.status as s)) -> ((s :> status), Some t)
  | None ->
      let same_boot =
        match (boot, File.boot_id ()) with
        | Some b, Some b' -> b = b'
        | _ -> false
      in
      let rebooted =
        match (machine, File.machine_id (), File.boot_time ()) with
        | Some m, Some m', Some booted -> m = m' && log_mtime < booted
        | _ -> false
      in
      let s =
        match held with
        | Some true -> `Running
        | Some false when same_boot -> `Crashed
        | _ when rebooted -> `Crashed
        | _ -> `Silent
      in
      (s, None)

(* Tags and notes *)

let tags_dir dir = Filename.concat dir "tags"
let tag_file dir tag = Filename.concat (tags_dir dir) (File.digest_string tag)

let read_tag dir name =
  match File.read_opt (Filename.concat (tags_dir dir) name) with
  | Some tag when File.is_tag tag && File.digest_string tag = name -> Some tag
  | _ | (exception Sys_error _) -> None

let tags dir =
  File.entries (tags_dir dir)
  |> List.filter_map (read_tag dir)
  |> List.sort_uniq String.compare

let notes dir = File.read_opt (Filename.concat dir "notes")

(* A run without a [writer], as a run copied without it, reads silent. *)
let writer dir =
  match File.read_opt (Filename.concat dir "writer") with
  | None | (exception Sys_error _) -> (None, None)
  | Some s -> (
      match Json.of_string s with
      | Ok j ->
          let get k =
            match Json.mem k j with Some (Json.String v) -> Some v | _ -> None
          in
          (get "machine", get "boot")
      | Error _ -> (None, None))

(* Lines *)

(* A line is read in pieces that start small, as most lines are a few hundred
   bytes, and double. *)
let first_piece = 512
let max_piece = 65536
let next_piece len = Int.min max_piece (2 * len)

let first_line fd =
  (* [acc] holds the pieces read before, last first. *)
  let rec go pos len acc =
    let s = File.pread fd ~pos ~len in
    match String.index_opt s '\n' with
    | Some i ->
        let head = String.sub s 0 i in
        Some
          (if acc = [] then head else String.concat "" (List.rev (head :: acc)))
    | None when String.length s < len -> None
    | None -> go (pos + len) (next_piece len) (s :: acc)
  in
  match go 0 first_piece [] with
  | Some l when not (String.contains l '\000') -> Some l
  | _ -> None

let last_line fd ~size =
  (* The last line feed ends the last whole line. [line] is [None] until it is
     found, then the line's bytes from [pos] to it. *)
  let rec go pos len line =
    if pos <= 0 then line
    else
      let start = Int.max 0 (pos - len) in
      let s = File.pread fd ~pos:start ~len:(pos - start) in
      let len = next_piece len in
      match line with
      | None -> (
          match String.rindex_opt s '\n' with
          | None -> go start len None
          | Some lf -> (
              match String.rindex_from_opt s (lf - 1) '\n' with
              | Some i -> Some (String.sub s (i + 1) (lf - i - 1))
              | None -> go start len (Some (String.sub s 0 lf))))
      | Some l -> (
          match String.rindex_opt s '\n' with
          | Some i -> Some (String.sub s (i + 1) (String.length s - i - 1) ^ l)
          | None -> go start len (Some (s ^ l)))
  in
  go size first_piece None

type error = [ `Not_a_run | `Unreadable of string ]

let decode_start line =
  match Record.decode line with
  | Some (time, Record.Start s) when s.format = Store_format.version ->
      Ok (s, time)
  | _ -> (
      match Record.start_format line with
      | Some f when f <> Store_format.version ->
          Error
            (`Unreadable
               (Printf.sprintf "format %d; this munin reads %d" f
                  Store_format.version))
      | _ -> Error (`Unreadable "the first line is not a start record"))

let diff dir (start : Record.start) =
  match start.provenance.git with
  | Some { dirty = true; _ } ->
      let p = Filename.concat dir "git.diff" in
      if Sys.file_exists p then Some p else None
  | _ -> None

let ( let* ) = Result.bind

(* The coarsest time resolution munin assumes of a file system, in seconds: a
   whole second, as ext3, HFS+ and many NFS servers keep. *)
let resolution = 1.

let annotations ?prev dir =
  let tags_stamp = File.stamp (tags_dir dir) in
  let notes_stamp = File.stamp (Filename.concat dir "notes") in
  (* A file system keeps modification times to its own resolution, a whole
     second on many, so a time within one resolution of the earlier read may
     hide a later change with the same time. *)
  let fresh old now (p : t) =
    match (old, now) with
    | Some _, Some (t', _, _) -> old = now && t' < p.read_at -. resolution
    | None, None -> true
    | _ -> false
  in
  match prev with
  | Some p
    when fresh p.tags_stamp tags_stamp p && fresh p.notes_stamp notes_stamp p ->
      (p.tags, p.notes, tags_stamp, notes_stamp)
  | Some p ->
      let tags = if fresh p.tags_stamp tags_stamp p then p.tags else tags dir in
      let notes =
        if fresh p.notes_stamp notes_stamp p then p.notes else notes dir
      in
      (tags, notes, tags_stamp, notes_stamp)
  | None -> (tags dir, notes dir, tags_stamp, notes_stamp)

let read ?prev ~id dir =
  let read_at = Unix.gettimeofday () in
  let path = log dir in
  let ended =
    match prev with Some { ended_at = Some _; _ } -> true | _ -> false
  in
  match prev with
  | Some p when ended ->
      let tags, notes, tags_stamp, notes_stamp = annotations ~prev:p dir in
      Ok { p with tags; notes; tags_stamp; notes_stamp; read_at }
  | _ -> (
      (* A start claims the directory before it creates the log, and a
         deletion renames the log away before it removes the directory: a
         directory without a log holds no run. *)
      match File.open_read_opt path with
      | exception Sys_error e -> Error (`Unreadable e)
      | None -> Error `Not_a_run
      | Some fd ->
          Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
          let* start, started_at =
            match prev with
            | Some p -> Ok (p.start, p.started_at)
            | None -> (
                match first_line fd with
                | None -> Error `Not_a_run
                | Some line -> decode_start line)
          in
          (* The probe comes before the read of the last record. *)
          let held = File.probe fd in
          let st = File.fstat fd in
          let last =
            match last_line fd ~size:st.Unix.st_size with
            | None -> None
            | Some line -> (
                match Record.decode line with
                | Some (t, Record.End s) -> Some (t, s)
                | _ -> None)
          in
          let status, ended_at =
            status ~held:(Some held) ~writer:(writer dir)
              ~log_mtime:st.Unix.st_mtime last
          in
          let tags, notes, tags_stamp, notes_stamp = annotations ?prev dir in
          Ok
            {
              id;
              dir;
              start;
              started_at;
              tags;
              notes;
              diff = diff dir start;
              status;
              ended_at;
              updated_at = st.Unix.st_mtime;
              read_at;
              tags_stamp;
              notes_stamp;
            })
