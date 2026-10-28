(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The cache, download and unpacking steps the dataset loaders share.

    Downloads run [curl] and extraction runs [tar], each found on [PATH] and run
    without a shell. Each blocks until its program exits. Progress is logged to
    the [kaun.datasets] source. *)

val mkdir_p : string -> unit
(** [mkdir_p path] creates the directory [path] and its missing parents, with
    permissions [0o755].

    Raises [Failure] if a component of [path] exists and is not a directory, and
    [Unix.Unix_error] if one cannot be created. *)

val get_cache_dir : ?getenv:(string -> string option) -> string -> string
(** [get_cache_dir name] is the directory of the dataset [name], with a trailing
    separator. It is [datasets/name] under [$RAVEN_CACHE_ROOT], else under
    [$XDG_CACHE_HOME/raven], else under [.cache/raven] in [$HOME] or
    [$USERPROFILE]. An empty variable counts as unset. [getenv] reads the
    variables and defaults to [Sys.getenv_opt]. It creates nothing.

    Raises [Failure] if the four variables are unset. *)

val ensure_file : string -> string -> unit
(** [ensure_file url dest] downloads [url] to [dest], creating [dest]'s
    directory, unless [dest] exists.

    Raises [Failure] if [curl] is missing or the download fails, in which case
    [dest] is removed. Raises as {!mkdir_p} does. *)

val ensure_decompressed_gz : gz_path:string -> target_path:string -> bool
(** [ensure_decompressed_gz ~gz_path ~target_path] is [true] if [target_path]
    exists, or once [gz_path] has been decompressed to it. It is [false] if
    neither file exists. [target_path] is written whole or not at all.

    Raises [Failure] naming [gz_path] if it is not gzip data or fails a
    checksum, [Sys_error] if [gz_path] cannot be read, and [Unix.Unix_error] if
    [target_path] cannot be written. *)

val ensure_extracted_tar_gz :
  tar_gz_path:string -> target_dir:string -> check_file:string -> bool
(** [ensure_extracted_tar_gz ~tar_gz_path ~target_dir ~check_file] is [true] if
    [check_file] exists, or once [tar] has extracted [tar_gz_path] into
    [target_dir], which it creates, and exited with code [0]. It is [false] if
    the archive is missing, [tar] is missing or [tar] fails. It does not check
    that [check_file] exists after extraction.

    Raises as {!mkdir_p} does. *)
