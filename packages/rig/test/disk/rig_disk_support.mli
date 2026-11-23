(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the disk's suites ask of the system. *)

val set_open_files : int -> int
(** [set_open_files n] sets this process's soft limit of open files to [n]: [0],
    the errno, or [-1] on Windows, which has no such limit. *)

val set_file_size : int -> int
(** [set_file_size n] sets this process's soft limit of a file's size to [n]
    bytes and ignores [SIGXFSZ], so growing a file past it fails with [EFBIG]:
    [0], the errno, or [-1] on Windows, which has no such limit. *)

val drop_pages : string -> int
(** [drop_pages path] writes the dirty pages of the file at [path] to storage
    and drops its cached pages, so that the next read reaches storage: [0], or
    the errno. On Linux only; [-1] elsewhere. *)

val identified : string -> bool
(** [identified path] is whether the system gives the file at [path] the
    identity by which the disk reopens a file, as [rig_disk.mli] names it: a
    file handle on Linux, an APFS or HFS+ volume on macOS, a file ID on Windows.
    Where it gives none, the disk keeps the file's descriptor until its buffers
    are collected. *)

val fault_reads : fail:int -> wait:float -> unit
(** [fault_reads ~fail ~wait] makes each run of every later read of runs of the
    disk wait [wait] seconds before it reads, and the run of index [fail] of
    each read fail with an I/O error without reading, as storage that fails a
    read does ([rig_disk_fault_runs] in [rig_disk_stubs.c]). [~fail:(-1)
    ~wait:0.] restores the disk's reads. Nothing on Windows. *)

val hold_pool : unit -> unit
(** [hold_pool ()] runs a compute job of two chunks on two threads of the
    host's pool ([rig_pool_run] in [rig_pool.h]), whose every chunk lasts until
    {!release_pool}. A domain that calls it holds the pool's compute workers
    until then. *)

val pool_held : unit -> bool
(** [pool_held ()] is whether a chunk of {!hold_pool}'s job began. *)

val release_pool : unit -> unit
(** [release_pool ()] ends {!hold_pool}'s job. *)

val descriptors : unit -> int option
(** [descriptors ()] is the number of file descriptors the process holds open,
    or [None] on Windows. *)
