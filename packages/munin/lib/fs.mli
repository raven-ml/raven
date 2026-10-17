(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Files and directories of the store.

    Every function follows symbolic links and blocks for its I/O. None
    takes a lock or writes atomically, so a reader can see a file that
    another process is writing half done. A function that writes a file
    first creates its missing parent directories, as {!ensure_dir} does.
    Errors of the operating system raise [Sys_error] or [Unix.Unix_error],
    unless a value says otherwise. *)

val is_directory : string -> bool
(** [is_directory path] is [true] iff [path] names a directory. It is [false]
    where [Unix.stat] fails on [path], as for a missing one. *)

val ensure_dir : string -> unit
(** [ensure_dir path] creates the directory [path] and its missing parents,
    with permissions [0o755]. It ignores a directory that another process
    creates meanwhile, and does nothing if [path] exists, even as a
    file. *)

val list_dirs : string -> string list
(** [list_dirs path] is the names of the directories in [path], sorted by
    [String.compare]. It is [[]] if [path] is not a directory. *)

val read_file : string -> string
(** [read_file path] is the contents of [path]. *)

val write_file : string -> string -> unit
(** [write_file path text] makes [text] the contents of [path]. *)

val append_line : string -> string -> unit
(** [append_line path line] appends [line] and a newline to [path], which it
    creates with permissions [0o644] if it does not exist. [line] must hold
    no newline for the file to keep one line per call. *)

val copy_file : string -> string -> unit
(** [copy_file src dst] makes the contents of [dst] those of [src]. *)

val copy_tree : string -> string -> unit
(** [copy_tree src dst] copies the file or directory [src] to [dst], each
    directory's entries in [String.compare] order. *)

val remove_tree : string -> unit
(** [remove_tree path] removes the file or directory [path] and everything
    under it. Raises [Sys_error] if [path] does not exist. *)

val digest_path : string -> string
(** [digest_path path] is the hex BLAKE2b-256 digest of the file [path], or
    for a directory the digest of its manifest. The manifest has one line
    per entry under [path], in the order of {!copy_tree}, [dir:rel] for a
    directory and [file:rel:digest] for a file, where [rel] is the entry's
    path relative to [path]. Two trees with the same names and contents
    digest the same. *)

val path_size : string -> int
(** [path_size path] is the size in bytes of the file [path], or the sum of
    those of the files under the directory [path]. A file on which
    [Unix.stat] fails counts [0], as does a missing [path]. *)

val command_output : string -> string option
(** [command_output command] runs [command] through the shell, with its
    standard error discarded, and blocks until it exits. It is the lines of
    its standard output joined by newlines, without its last newline, or
    [None] if that is empty. The command's exit status is ignored. *)
