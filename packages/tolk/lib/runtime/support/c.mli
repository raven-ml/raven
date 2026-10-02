(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Shared libraries of the system.

    The compilers that run in the process load a vendor's library at their first
    compile. A program without the library still runs, and renders for its
    targets: only compiling fails. *)

val multiarch : string
(** [multiarch] is the name Linux distributions give the host's machine in
    library directories, as in ["x86_64-linux-gnu"]. *)

val findlib : ?extra_paths:string list -> string -> string list -> string option
(** [findlib ~extra_paths name paths] is the file of the library [name], the
    first of:
    - the file that the variable [NAME_PATH] names, where [NAME] is [name] in
      uppercase with its dashes replaced by underscores, if it is a file;
    - for each of [paths] in order: the path itself if it is absolute, when it
      is a file. Otherwise the library of that name in the first of these
      directories that has it: the directory [NAME_PATH] names, then on Windows
      those of [PATH]; elsewhere those of [LD_LIBRARY_PATH], [/usr/lib64],
      [/usr/lib] and [/usr/local/lib], then on macOS [/opt/homebrew/lib] and the
      framework of that name, system or private, and on Linux
      [/usr/lib/wsl/lib], [/lib], [/lib64] and [/lib/]{!multiarch}; then
      [extra_paths] (default [[]]).

    The library [p] in a directory is, on Windows, [p.dll]; on macOS, the first
    of [libp.dylib], [p.dylib] and [p] (a framework's binary is a symbolic
    link); elsewhere, a file [libp.so] or [libp.so.V], with [V] made of digits
    and dots, that starts as an ELF file does, the first in the order of their
    names. It is [None] if there is none. *)

val identity : string option -> string
(** [identity file] names the library [file], as {!findlib} finds it: its path,
    size and time of last modification, or its path alone if it cannot be read,
    as a library of macOS's shared cache, and ["none"] for [None]. A library
    that changes changes its identity, so that what was compiled with it can be
    told from what another compiles. *)
