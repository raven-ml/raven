(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* [sync_dir dir] puts on disk the entries renamed in [dir]. Windows has no
   directory sync. *)
let sync_dir dir =
  if not Sys.win32 then
    let fd = Unix.openfile dir [ Unix.O_RDONLY; Unix.O_CLOEXEC ] 0 in
    Fun.protect ~finally:(fun () -> Unix.close fd) (fun () -> Unix.fsync fd)

(* The file [path] names, through a link too, and the permissions it keeps when
   it exists. *)
let target path =
  match Unix.realpath path with
  | exception Unix.Unix_error (Unix.ENOENT, _, _) -> (path, None)
  | target ->
      (try Unix.access target [ Unix.W_OK ]
       with Unix.Unix_error (e, _, _) ->
         raise (Sys_error (path ^ ": " ^ Unix.error_message e)));
      (target, Some (Unix.stat target).st_perm)

let replace path text =
  let target, perm = target path in
  let dir = Filename.dirname target in
  let tmp, oc =
    Filename.open_temp_file ~mode:[ Open_binary ] ~perms:0o666 ~temp_dir:dir
      ("." ^ Filename.basename target ^ ".")
      ".tmp"
  in
  match
    Fun.protect
      ~finally:(fun () -> close_out_noerr oc)
      (fun () ->
        output_string oc text;
        flush oc;
        Option.iter (Unix.chmod tmp) perm;
        Unix.fsync (Unix.descr_of_out_channel oc));
    Unix.rename tmp target
  with
  | () -> sync_dir dir
  | exception e ->
      (try Sys.remove tmp with Sys_error _ -> ());
      raise e

let write path text =
  try replace path text
  with Unix.Unix_error (e, _, _) ->
    raise (Sys_error (path ^ ": " ^ Unix.error_message e))
