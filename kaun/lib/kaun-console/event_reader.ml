(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type file_id = int * int  (* (st_dev, st_ino) *)

type t = {
  file_path : string;
  mutable position : int64;
  mutable last_mtime : float;
  mutable file_id : file_id option;
  mutable channel : in_channel option;
  mutable pending : string; (* trailing unterminated line fragment *)
}

let create ~file_path =
  {
    file_path;
    position = 0L;
    last_mtime = 0.0;
    file_id = None;
    channel = None;
    pending = "";
  }

let file_exists r = Sys.file_exists r.file_path

let close r =
  match r.channel with
  | None -> ()
  | Some ic ->
      r.channel <- None;
      (try close_in ic with _ -> ())

let reset r =
  close r;
  r.position <- 0L;
  r.last_mtime <- 0.0;
  r.file_id <- None;
  r.pending <- ""

let ensure_channel r =
  match r.channel with
  | Some ic -> ic
  | None ->
      let ic = open_in_bin r.file_path in
      r.channel <- Some ic;
      (* Record file identity from the actual opened descriptor. *)
      (try
         let st = Unix.fstat (Unix.descr_of_in_channel ic) in
         r.file_id <- Some (st.Unix.st_dev, st.Unix.st_ino)
       with _ ->
         ());
      ic

let read_new r =
  if not (file_exists r) then (
    (* If the file vanished, forget state so a future recreate is read from 0. *)
    reset r;
    []
  ) else
    try
      let st = Unix.LargeFile.stat r.file_path in
      let path_id : file_id = (st.Unix.LargeFile.st_dev, st.Unix.LargeFile.st_ino) in
      let file_size = st.Unix.LargeFile.st_size in
      let mtime = st.Unix.LargeFile.st_mtime in

      (* Detect rotation/replacement (inode change). *)
      let rotated =
        match r.file_id with
        | None -> false
        | Some (dev, ino) -> dev <> fst path_id || ino <> snd path_id
      in

      (* Detect truncation. *)
      let truncated = r.position > file_size in

      if rotated || truncated then (
        (* Drop pending fragment; old partial JSON cannot be completed safely. *)
        reset r;
        (* Keep identity so we don't immediately treat the same file as "rotated"
           after reopening. We still set last_mtime=0 to force a full read. *)
        r.file_id <- Some path_id;
      );

      (* Fast-path: nothing new. We still rely on size; mtime alone is not enough. *)
      if r.position >= file_size && mtime <= r.last_mtime then
        []
      else (
        let ic = ensure_channel r in

        (* Seek using 64-bit offsets. *)
        LargeFile.seek_in ic r.position;

        (* Read all currently available bytes. *)
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

        (* Update state even if we read 0 bytes: mtime may have advanced. *)
        r.last_mtime <- mtime;

        if bytes_read = 0 then
          []
        else (
          r.position <- Int64.add r.position (Int64.of_int bytes_read);

          let data = Buffer.contents b in
          let chunk =
            if r.pending = "" then data else r.pending ^ data
          in
          let events, pending = Event.parse_jsonl_chunk chunk in
          r.pending <- pending;
          events
        )
      )
    with
    | Sys_error _ ->
        (* Permission denied / disappeared mid-read *)
        close r;
        []
    | Unix.Unix_error _ ->
        close r;
        []
