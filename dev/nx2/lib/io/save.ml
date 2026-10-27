(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A save leaves, after a crash, the file it replaces or the whole new one. It
   writes the new file beside [path] and syncs it, which orders the file's bytes
   before every later change to the file system. It then renames the file over
   [path] and syncs the directory, so that the rename outlasts a crash. The
   order is SQLite's for its journal (os_unix.c, unixSync): the file's sync,
   then its directory's.

   write temp --> sync temp --> rename to path --> sync directory

   A failure before the rename removes the new file. A refused rename keeps it,
   so that the bytes written are not lost, and names it in the error. *)

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

(* A fresh empty file beside [path], and its descriptor. *)
let sibling path =
  let rec go attempts =
    let name = name path in
    match Unix.openfile name create mode with
    | fd -> (name, fd)
    | exception Unix.Unix_error (EEXIST, _, _) when attempts > 1 ->
        go (attempts - 1)
  in
  go 1000

(* [sync fd] returns once the bytes written to [fd] are ordered before later
   changes to the file system. *)
external sync : Unix.file_descr -> unit = "caml_nx_io_sync"

(* [sync_dir dir] returns once the names put in [dir] outlast a crash. *)
external sync_dir : string -> unit = "caml_nx_io_sync_dir"

let replace temp path =
  try Unix.rename temp path
  with Unix.Unix_error _ when Sys.win32 ->
    (* Windows refuses to replace a file while a view of its pages is mapped: a
       value loaded from [path] may hold one until it is collected. *)
    Gc.full_major ();
    Unix.rename temp path

(* [written fd f] is [f fd] then a sync of [fd], and closes [fd] whatever
   happens; the first failure among the three raises. *)
let written fd f =
  match
    f fd;
    sync fd
  with
  | () -> Unix.close fd
  | exception exn ->
      (try Unix.close fd with Unix.Unix_error _ -> ());
      raise exn

let write path f =
  Fail.file path @@ fun () ->
  let temp, fd = sibling path in
  (match written fd f with
  | () -> ()
  | exception exn ->
      remove_if_exists temp;
      raise exn);
  (match replace temp path with
  | () -> ()
  | exception Unix.Unix_error (e, _, _) ->
      Fail.failf "cannot be replaced (%s); the new file is %s"
        (Unix.error_message e) temp);
  sync_dir (Filename.dirname path)

(* [string fd s] writes [s] to [fd]. *)
let string fd s = ignore (Unix.write_substring fd s 0 (String.length s))

(* [bytes fd b] writes [b] to [fd]. *)
let bytes fd b = ignore (Unix.write_bigarray fd b 0 (Bigarray.Array1.dim b))
