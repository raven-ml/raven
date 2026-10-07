(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let remove_if_exists path = try Sys.remove path with Sys_error _ -> ()
let prng = Domain.DLS.new_key Random.State.make_self_init

(* A written file's permissions, less those the umask removes, as nx.device
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

(* [write ~overwrite path f] is [f fd] for [fd] a new file that becomes [path]:
   [path] itself, which must not exist, without [overwrite]; otherwise a sibling
   renamed over [path] once [f] returns. [f] writes through [fd] alone: the file
   is never reopened by its name, which a link put there meanwhile would
   redirect. A failed [f] leaves no file. *)
let write ~overwrite path f =
  let file, fd =
    if overwrite then sibling path else (path, Unix.openfile path create mode)
  in
  match
    Fun.protect ~finally:(fun () -> Unix.close fd) (fun () -> f fd);
    if overwrite then Unix.rename file path
  with
  | () -> ()
  | exception exn ->
      remove_if_exists file;
      raise exn
