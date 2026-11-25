(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Files written whole, and downloads into them with curl.

    {b Internal library} of kaun.hf and kaun.datasets.

    A file is written to a temporary file created exclusively beside its path,
    through the descriptor that created it, so a name reopened cannot follow a
    link put there meanwhile. It is synced, then renamed to its path: the path
    holds the whole file or what it held before, even when the process dies
    midway or the system loses power. A process that dies midway leaves its
    temporary file, under a random name that later writes skip. *)

val mkdir_p : string -> unit
(** [mkdir_p path] creates the directory [path] and its missing parents, with
    permissions [0o755].

    Raises [Failure] if a component of [path] exists and is not a directory, and
    [Unix.Unix_error] if one cannot be created. *)

val write : string -> (Unix.file_descr -> unit) -> unit
(** [write path f] gives [f] the descriptor of a file created beside [path], in
    [path]'s directory, which must exist. Once [f] returns, the file is synced
    and renamed to [path]. A caller writes a [path] it found absent, so a
    [path] that exists when the rename fails was written meanwhile, by another
    process: it is kept, and the file removed. Windows refuses the rename while
    that process holds [path] open. On any failure, [f]'s included, the file
    is removed where it can be, and [path] is as it was.

    Raises what [f] raises, and [Unix.Unix_error] if the file cannot be created,
    synced or renamed. *)

val max_header_bytes : int
(** [max_header_bytes] is [1024], the most bytes of header lines {!file}
    sends. *)

val file :
  ?headers:(string * string) list -> ?sha256:string -> url:string -> string -> unit
(** [file ~url path] downloads [url] to [path] through {!write}, creating
    [path]'s directory. curl, the first [curl] ([curl.exe] on Windows) that
    this process may execute (on Windows, read) in [PATH]'s directories, a
    relative or empty one taken from the current directory, runs by that path
    without a shell as [curl -q -L --fail -s url], [-q] first so that it reads
    none of the user's curl configuration files, and writes to the file's
    descriptor. With:

    - [headers], sent as HTTP headers. curl reads them on its standard input
      ([-H @-]), never on its command line, which every user of the machine
      can read: at most {!max_header_bytes} of [name: value] lines, each
      ended by a newline.
    - [sha256], 64 lowercase hexadecimal digits: the download is hashed as it
      is written (FIPS 180-4) and refused unless its digest is [sha256].

    Raises, before any path is made, [Invalid_argument] if the headers' lines
    exceed {!max_header_bytes}, and [Failure] naming [url] if [url] holds a
    NUL, which no command line carries. Raises [Failure] naming [url] if curl
    is missing or cannot be run, exits other than with code [0], or gives a
    file of another digest, naming both; or if [path]'s directory or file
    cannot be made, synced or renamed. [path] is then as it was. *)
