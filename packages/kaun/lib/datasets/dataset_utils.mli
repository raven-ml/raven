(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The cache, download and unpacking steps the dataset loaders share.

    Downloads run [curl] and extraction runs [tar], each found on [PATH] and run
    without a shell; curl reads none of the user's curl configuration files.
    Each blocks until its program exits. Progress is logged to
    the [kaun.datasets] source. *)

val get_cache_dir : ?getenv:(string -> string option) -> string -> string
(** [get_cache_dir name] is the directory of the dataset [name], with a trailing
    separator. It is [datasets/name] under [$RAVEN_CACHE_ROOT], else under
    [$XDG_CACHE_HOME/raven], else under [.cache/raven] in [$HOME] or
    [$USERPROFILE]. An empty variable counts as unset. [getenv] reads the
    variables and defaults to [Sys.getenv_opt]. It creates nothing.

    Raises [Failure] if the four variables are unset. *)

val ensure_file : string -> string -> unit
(** [ensure_file url dest] downloads [url] to [dest] unless [dest] exists, as
    {!Kaun_download.file} downloads: [dest] holds a whole download or
    nothing, even when the process dies midway.

    Raises [Failure] naming [url] as {!Kaun_download.file} does, leaving [dest]
    as it was. *)

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

    Raises as {!Kaun_download.mkdir_p} does. *)
