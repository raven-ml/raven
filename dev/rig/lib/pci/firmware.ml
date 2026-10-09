(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let digest s = Digest.BLAKE256.to_hex (Digest.BLAKE256.string s)

(* A file's identity: rewriting it changes its size or its modification time,
   replacing it its inode. *)
type identity = { dev : int; ino : int; size : int; mtime : float }

let identity (st : Unix.stats) =
  { dev = st.st_dev; ino = st.st_ino; size = st.st_size; mtime = st.st_mtime }

let same a b =
  a.dev = b.dev && a.ino = b.ino && a.size = b.size
  && Float.equal a.mtime b.mtime

(* The images found, by file and pinned digest, with the identity their file had
   when it was read: while the file keeps it, a find gives the image again
   unread. *)
let found : (string * string, identity * string) Hashtbl.t = Hashtbl.create 16
let found_lock = Mutex.create ()

(* What [file] holds for the image of digest [pinned]: [`Absent] if it does not
   exist, [`Unreadable why] if it cannot be read, [why] the system's "FILE:
   cause". *)
let look file ~pinned =
  match Unix.stat file with
  | exception Unix.Unix_error ((ENOENT | ENOTDIR), _, _) -> `Absent
  | exception Unix.Unix_error (e, _, _) ->
      `Unreadable (strf "%s: %s" file (Unix.error_message e))
  | st -> (
      let id = identity st in
      let known =
        Mutex.protect found_lock (fun () ->
            Hashtbl.find_opt found (file, pinned))
      in
      match known with
      | Some (id', s) when same id' id -> `Image s
      | _ -> (
          match In_channel.with_open_bin file In_channel.input_all with
          | exception Sys_error why -> `Unreadable why
          | s when digest s = pinned ->
              Mutex.protect found_lock (fun () ->
                  Hashtbl.replace found (file, pinned) (id, s));
              `Image s
          | _ -> `Other))

type image = { path : string; contents : string }

let missing dirs name ~digest others unreadable =
  let others =
    match others with
    | [] -> ""
    | l -> "; another digest in " ^ String.concat ", " (List.rev l)
  in
  let unreadable =
    String.concat "" (List.rev_map (fun why -> "; reading " ^ why) unreadable)
  in
  strf "%s with digest %s is in none of [%s]%s%s" name digest
    (String.concat ", " dirs) others unreadable

let find dirs name ~digest:pinned =
  let rec go others unreadable = function
    | [] -> Error (missing dirs name ~digest:pinned others unreadable)
    | dir :: rest -> (
        let file = Filename.concat dir name in
        match look file ~pinned with
        | `Image contents -> Ok { path = file; contents }
        | `Other -> go (file :: others) unreadable rest
        | `Absent -> go others unreadable rest
        | `Unreadable why -> go others (why :: unreadable) rest)
  in
  go [] [] dirs
