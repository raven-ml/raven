(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Saves that a crash leaves whole.

    A save writes a new file beside its path, syncs it, names it the path,
    and syncs the directory. After a crash the path names the file it held
    before or the whole new one. A save that must not replace a file links
    the new one to the path, which fails if the path exists. A file system
    without hard links checks the name and renames instead, so a file put
    at the path between the two is replaced. A save raises its own errors
    as [Unix.Unix_error]. *)

val write : overwrite:bool -> string -> (Unix.file_descr -> unit) -> unit
(** [write ~overwrite path f] calls [f fd] for [fd] a new file beside [path],
    then places that file at [path] as {!place} does. [f] must write through
    [fd] alone. If [f], the sync, the rename or the link fails, no new file
    remains. Raises [Unix.Unix_error] with [EEXIST] before [f] runs if
    [overwrite] is [false] and [path] exists, and with the new file at
    [path] if the directory's sync fails. *)

val name : string -> string
(** [name path] is a name beside [path] for a temporary file, drawn at
    random. *)

val place : overwrite:bool -> string -> string -> unit
(** [place ~overwrite temp path] names the synced file [temp] [path], then
    syncs [path]'s directory. With [overwrite] it replaces what [path] names.
    Otherwise it raises [Unix.Unix_error] with [EEXIST] if [path] exists. A
    refused rename raises [Unix.Unix_error] and leaves [temp]. *)

val remove_if_exists : string -> unit
(** [remove_if_exists path] removes the file [path], if it can. *)
