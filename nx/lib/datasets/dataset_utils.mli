(** Utilities shared by dataset loaders. *)

(** {1:cache Cache paths} *)

val get_cache_dir : ?getenv:(string -> string option) -> string -> string
(** [get_cache_dir ?getenv dataset_name] is the cache directory for
    [dataset_name] under the [datasets] scope.

    [getenv] defaults to [Sys.getenv_opt]. *)

(** {1:download Download and extraction} *)

val download_file : string -> string -> unit
(** [download_file url dest_path] downloads [url] to [dest_path].

    Parent directories are created if needed.

    Raises [Failure] on download errors. *)

val ensure_file : string -> string -> unit
(** [ensure_file url dest_path] ensures [dest_path] exists, downloading from
    [url] if missing.

    Raises [Failure] on download errors. *)

val ensure_extracted_archive :
  url:string ->
  archive_path:string ->
  extract_dir:string ->
  check_file:string ->
  unit
(** [ensure_extracted_archive ~url ~archive_path ~extract_dir ~check_file]
    ensures [check_file] exists under [extract_dir], downloading and extracting
    [archive_path] when needed.

    Only [.tar.gz] archives are supported.

    Raises [Failure] on download, extraction, or validation errors. *)

val ensure_decompressed_gz : gz_path:string -> target_path:string -> bool
(** [ensure_decompressed_gz ~gz_path ~target_path] ensures [target_path] exists
    by decompressing [gz_path] when needed.

    Returns [true] if [target_path] exists after the call, [false] if [gz_path]
    does not exist.

    Raises [Failure] on decompression errors. *)

(** {1:parsing Parsing helpers} *)

val parse_float_cell : context:(unit -> string) -> string -> float
(** [parse_float_cell ~context s] parses [s] as a float.

    Raises [Failure] with contextual information if parsing fails. *)

val parse_int_cell : context:(unit -> string) -> string -> int
(** [parse_int_cell ~context s] parses [s] as an integer.

    Raises [Failure] with contextual information if parsing fails. *)

(** {1:filesystem Filesystem helpers} *)

val mkdir_p : string -> unit
(** [mkdir_p path] creates [path] and missing parent directories.

    If [path] already exists as a directory, the call is a no-op. *)

(** {1:csv CSV loading} *)

val load_csv :
  ?separator:char ->
  ?has_header:bool ->
  string ->
  string list * string list list
(** [load_csv ?separator ?has_header path] reads a CSV file.

    [separator] defaults to [','] and [has_header] defaults to [false].

    Returns [(header, rows)], where [header] is [[]] when [has_header] is
    [false]. *)
