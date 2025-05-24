(** Utilities for downloading and managing datasets. *)

val get_cache_dir : string -> string
(** Return the platform-specific cache directory path for the given dataset.

    The default location is "~/.cache/ocaml-nx/datasets/[dataset_name]/".

    {2 Parameters}
    - dataset_name: the name of the dataset.

    {2 Returns}
    - the cache directory path, including trailing slash. *)

val download_file : string -> string -> unit
(** Download a file from a URL to a destination path.

    Creates parent directories as needed, downloads the file from [url], and
    saves it to [dest_path].

    {2 Parameters}
    - url: the source URL of the file.
    - dest_path: local path to save the downloaded file.

    {2 Raises}
    - [Failure] on download or write error. *)

val ensure_file : string -> string -> unit
(** Ensure a file exists at the given path, downloading if necessary.

    Checks if [dest_path] exists. If not, downloads the file from [url].

    {2 Parameters}
    - url: the source URL of the file.
    - dest_path: local path to ensure the file exists.

    {2 Raises}
    - [Failure] on download or write error. *)

val ensure_extracted_archive :
  url:string ->
  archive_path:string ->
  extract_dir:string ->
  check_file:string ->
  unit
(** Ensure an archive is downloaded, extracted, and a file exists.

    Checks if [check_file] (relative to [extract_dir]) exists. If not, downloads
    the archive from [url] to [archive_path], extracts it into [extract_dir],
    and verifies [check_file] is present. Currently supports only .tar.gz
    archives.

    {2 Parameters}
    - url: the source URL of the archive.
    - archive_path: local path for the downloaded archive.
    - extract_dir: directory to extract the archive into.
    - check_file: relative path under [extract_dir] to verify extraction.

    {2 Raises}
    - [Failure] on download, extraction, or missing [check_file]. *)

val ensure_decompressed_gz : gz_path:string -> target_path:string -> bool
(** Ensure a gzip-compressed file is decompressed to a target path.

    If [target_path] exists, does nothing and returns [true]. Otherwise, if
    [gz_path] exists, decompresses it to [target_path].

    {2 Parameters}
    - gz_path: the path to the .gz file to decompress.
    - target_path: the destination path for the decompressed file.

    {2 Returns}
    - [true] if [target_path] exists after the operation.
    - [false] if [gz_path] does not exist.

    {2 Raises}
    - [Failure] on gzip decompression error. *)

val parse_float_cell : context:(unit -> string) -> string -> float
(** Parse a CSV cell as a float.

    Attempts to convert [value] to a float. On failure, raises [Failure] with a
    descriptive message including [context ()].

    {2 Parameters}
    - context: a function returning context information for error messages.
    - value: the string to parse as a float.

    {2 Returns}
    - the parsed float.

    {2 Raises}
    - [Failure] if [value] cannot be parsed as a float. *)

val parse_int_cell : context:(unit -> string) -> string -> int
(** Parse a CSV cell as an integer.

    Attempts to convert [value] to an int. On failure, raises [Failure] with a
    descriptive message including [context ()].

    {2 Parameters}
    - context: a function returning context information for error messages.
    - value: the string to parse as an int.

    {2 Returns}
    - the parsed integer.

    {2 Raises}
    - [Failure] if [value] cannot be parsed as an int. *)

val mkdir_p : string -> unit
(** Recursively create a directory and its parents.

    Creates the directory at [path], along with any missing parent directories.
    If [path] already exists as a directory, does nothing.

    {2 Parameters}
    - path: the directory path to create.

    {2 Raises}
    - [Unix.Unix_error] if creation fails for other reasons. *)
