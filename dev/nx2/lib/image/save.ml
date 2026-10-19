(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A save leaves, after a crash, the file it replaces or the whole new one: the
   new file is written beside [path], synced, then named [path], and the
   directory is synced. The order is SQLite's for its journal (os_unix.c,
   unixSync): the file's sync, then its directory's. A file that must not
   replace one is linked to [path], which fails if [path] exists, then its
   temporary name removed, as Git places an object (object-file.c,
   finalize_object_file).

   write temp --> sync temp --> rename / link to path --> sync directory *)

let failf fmt = Printf.ksprintf failwith fmt

(* [file path f] is [f ()], whose failures raise [Failure] starting with
   [path]. *)
let file path f =
  try f () with
  | Failure m -> failwith (path ^ ": " ^ m)
  | Sys_error m -> failwith (path ^ ": " ^ m)
  | Unix.Unix_error (e, _, _) -> failwith (path ^ ": " ^ Unix.error_message e)

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
external sync : Unix.file_descr -> unit = "caml_nx_image_sync"

(* [sync_dir dir] returns once the names put in [dir] outlast a crash. *)
external sync_dir : string -> unit = "caml_nx_image_sync_dir"

let replace temp path =
  try Unix.rename temp path
  with Unix.Unix_error _ when Sys.win32 ->
    (* Windows refuses to replace a file while a view of its pages is mapped: a
       value loaded from [path] may hold one until it is collected. *)
    Gc.full_major ();
    Unix.rename temp path

(* A file system without hard links, such as FAT, refuses [link]: the name is
   then checked and the file renamed, as Git does, so that a file put at [path]
   between the two is replaced. *)
(* Once the link is made the file is at [path]: a temporary name left by a
   failed unlink is a second name for it. *)
let add temp path =
  let exists () = failf "the path names a file, and overwrite is false" in
  match Unix.link temp path with
  | () -> ( try Unix.unlink temp with Unix.Unix_error _ -> ())
  | exception Unix.Unix_error (EEXIST, _, _) -> exists ()
  | exception Unix.Unix_error _ ->
      if Sys.file_exists path then exists ();
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

let write ~overwrite path f =
  file path @@ fun () ->
  (* An existing [path] fails before the file is written; the link that places
     it is the check that holds. *)
  if (not overwrite) && Sys.file_exists path then
    failf "the path names a file, and overwrite is false";
  let temp, fd = sibling path in
  (match written fd f with
  | () -> ()
  | exception exn ->
      remove_if_exists temp;
      raise exn);
  (match if overwrite then replace temp path else add temp path with
  | () -> ()
  | exception Unix.Unix_error (e, "rename", _) when overwrite ->
      failf "cannot be replaced (%s); the new file is %s" (Unix.error_message e)
        temp
  | exception exn ->
      remove_if_exists temp;
      raise exn);
  sync_dir (Filename.dirname path)

(* [string fd s] writes [s] to [fd]. *)
let string fd s = ignore (Unix.write_substring fd s 0 (String.length s))
