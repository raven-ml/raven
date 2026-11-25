(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The cache, download and unpacking steps the dataset loaders share.

    Downloads run [curl], found on [PATH] and run without a shell, which reads
    none of the user's curl configuration files and blocks until it exits.
    Archives are decompressed and read in this process. Progress is logged to
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

    Raises [Failure] naming [gz_path] if it is not gzip data, fails a
    checksum or a read of it fails, and [Unix.Unix_error] if [gz_path] cannot
    be opened or [target_path] cannot be written. *)

val read_tar_gz : string -> (string -> int -> (int -> string) -> unit) -> unit
(** [read_tar_gz path f] reads the gzip file [path], which holds a ustar
    archive (POSIX pax, "ustar Interchange Format"), and applies
    [f name size take] to each regular file in the archive's order: [name] is
    its path in the archive, [size] its length in bytes, and [take n] its next
    [n] bytes, fewer once [size] are taken. The bytes [f] does not take are
    skipped, and so are directories, links, special files and pax global
    headers. It then reads [path] to its end: the bytes [f] took are verified
    only once [read_tar_gz] returns, so a caller keeps nothing [f] built from
    them before.

    Raises [Failure] naming [path] if it cannot be read, is not gzip data,
    fails gzip's checksum or length, or is truncated; if a header fails its
    checksum or holds a numeric field that is not octal, as GNU's base-256
    sizes of 8 GiB and more are not; or if a member is of another type, such
    as a pax extended header or a GNU long name, which it does not read.
    Raises what [f] raises, unchanged. *)
