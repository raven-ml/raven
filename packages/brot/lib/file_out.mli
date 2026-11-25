(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Files written whole.

    {b Internal module.} *)

val write : string -> string -> unit
(** [write path text] replaces the file [path] names, through a link too, by one
    holding [text]. [text] goes to a file created exclusively beside it, synced
    to the disk, then renamed over it, and the directory is synced. A write that
    fails leaves the old file as it was and no other file; a crash leaves the
    old file or the new one, and may leave the temporary file, named
    [.<name>.<random>.tmp]. A new file has the permissions the umask allows; a
    replaced one keeps its own.

    Raises [Sys_error] if [path] names a read-only file or cannot be written. *)
