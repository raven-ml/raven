(** File I/O utilities *)

val read_lines : ?buffer_size:int -> string -> string list
(** [read_lines ?buffer_size filename] efficiently reads all lines from a file.

    @param buffer_size Size of the read buffer in bytes (default: 65536)
    @return List of lines without trailing newlines
    @raise Sys_error if file cannot be opened or read

    Features:
    - Efficient buffered reading for large files
    - Automatic resource cleanup on errors
    - Windows/Unix line ending compatibility
    - Memory-efficient for files with many lines *)

val read_lines_lazy : ?buffer_size:int -> string -> string Seq.t
(** [read_lines_lazy ?buffer_size filename] returns a lazy sequence of lines.

    @param buffer_size Size of the read buffer in bytes (default: 65536)
    @return Lazy sequence of lines that are read on-demand
    @raise Sys_error if file cannot be opened

    Use this for very large files to avoid loading everything into memory. The
    file is automatically closed when the sequence is fully consumed or when an
    error occurs. *)

val write_lines : ?append:bool -> string -> string list -> unit
(** [write_lines ?append filename lines] writes lines to a file.

    @param append If true, append to existing file (default: false)
    @param filename Target file path
    @param lines List of lines to write (newlines are added automatically)
    @raise Sys_error if file cannot be written *)
