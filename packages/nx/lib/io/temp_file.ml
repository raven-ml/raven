(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let remove_if_exists path = try Sys.remove path with Sys_error _ -> ()
let prng = Domain.DLS.new_key Random.State.make_self_init

(* A written file's permissions, less those the umask removes, as rig.disk
   creates a file. *)
let mode = 0o666

(* Created exclusively, a new file is never a link or a file another process put
   at its name. *)
let create = [ Unix.O_WRONLY; O_CREAT; O_EXCL; O_CLOEXEC ]

(* A name beside [path] for a temporary file, drawn at random. *)
let name path =
  let tag = Random.State.bits (Domain.DLS.get prng) land 0xffffff in
  Filename.concat (Filename.dirname path)
    (Printf.sprintf "%s.%06x.tmp" (Filename.basename path) tag)

(* A fresh empty file beside [path], and its descriptor. It is created with
   [Unix], so that a directory that cannot be written raises [Unix.Unix_error]
   as the other writes of this library do. *)
let sibling path =
  let rec go attempts =
    let name = name path in
    match Unix.openfile name create mode with
    | fd -> (name, fd)
    | exception Unix.Unix_error (EEXIST, _, _) when attempts > 1 ->
        go (attempts - 1)
  in
  go 1000

(* A save leaves, after a crash, the file it replaces or the whole new one:
   the new file is written beside [path], synced, then named [path], and the
   directory is synced. The order is SQLite's for its journal (os_unix.c,
   unixSync): the file's sync, then its directory's. A file that must not
   replace one is linked to [path], which fails if [path] exists, then its
   temporary name removed, as Git places an object (object-file.c,
   finalize_object_file).

     write temp --> sync temp --> rename / link to path --> sync directory *)

(* [sync fd] returns once the bytes written to [fd] are ordered before later
   changes to the file system. *)
external sync : Unix.file_descr -> unit = "caml_nx_io_sync"

(* [sync_dir dir] returns once the names put in [dir] outlast a crash. *)
external sync_dir : string -> unit = "caml_nx_io_sync_dir"

let existing path = Unix.Unix_error (EEXIST, "link", path)

let replace temp path =
  try Unix.rename temp path
  with Unix.Unix_error _ when Sys.win32 ->
    (* Windows refuses to replace a file while a view of its pages is mapped:
       a value loaded from [path] and borrowed may hold one until it is
       collected. *)
    Gc.full_major ();
    Unix.rename temp path

(* A file system without hard links, such as FAT, refuses [link]: the name is
   then checked and the file renamed, as Git does, so that a file put at
   [path] between the two is replaced. *)
let add temp path =
  match Unix.link temp path with
  | () -> Unix.unlink temp
  | exception Unix.Unix_error (EEXIST, _, _) -> raise (existing path)
  | exception Unix.Unix_error _ ->
      if Sys.file_exists path then raise (existing path);
      Unix.rename temp path

(* [place ~overwrite temp path] names the synced file [temp] [path], replacing
   what [path] names with [overwrite] and failing with [EEXIST] if it names
   something otherwise. A refused rename raises [Unix.Unix_error] for
   ["rename"] and leaves [temp]. *)
let place ~overwrite temp path =
  if overwrite then replace temp path else add temp path;
  sync_dir (Filename.dirname path)

(* [write ~overwrite path f] is [f fd] for [fd] a new file beside [path] that
   is then placed at [path] ({!place}). [f] writes through [fd] alone: the file
   is never reopened by its name, which a link put there meanwhile would
   redirect. A failed [f], sync, rename or link leaves no new file; a failed
   sync of the directory raises with the new file at [path]. *)
let write ~overwrite path f =
  (* An existing [path] fails before the file is written; the link that
     places it is the check that holds. *)
  if (not overwrite) && Sys.file_exists path then raise (existing path);
  let temp, fd = sibling path in
  match
    Fun.protect
      ~finally:(fun () -> Unix.close fd)
      (fun () ->
        f fd;
        sync fd);
    place ~overwrite temp path
  with
  | () -> ()
  | exception exn ->
      remove_if_exists temp;
      raise exn
