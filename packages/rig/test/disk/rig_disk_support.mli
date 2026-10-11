(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the disk's suites ask of the system. *)

val set_open_files : int -> int
(** [set_open_files n] sets this process's soft limit of open files to [n]: [0],
    the errno, or [-1] on Windows, which has no such limit. *)

val open_files : unit -> int
(** [open_files ()] is this process's soft limit of open files, at most
    [max_int], or [-1] on Windows. *)

val set_file_size : int -> int
(** [set_file_size n] sets this process's soft limit of a file's size to [n]
    bytes and ignores [SIGXFSZ], so growing a file past it fails with [EFBIG]:
    [0], the errno, or [-1] on Windows, which has no such limit. *)

val drop_pages : string -> int
(** [drop_pages path] writes the dirty pages of the file at [path] to storage
    and drops its cached pages, so that the next read reaches storage: [0], or
    the errno. On Linux only; [-1] elsewhere. *)

val heap_bytes : unit -> int option
(** [heap_bytes ()] is the bytes the C heap holds allocated, as its allocator
    counts them, or [None] where it does not say. *)

val descriptors : unit -> int option
(** [descriptors ()] is the number of file descriptors the process holds open,
    or [None] on Windows. *)
