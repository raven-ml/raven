(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Files, locks, syncs and the system's identities, for the store.

    Every function that touches the file system raises [Sys_error] with the path
    and the system's message when the system refuses it, and never
    [Unix.Unix_error]. Descriptors are opened close-on-exec. *)

(** {1:names Names} *)

val is_text : string -> bool
(** [is_text s] is [true] iff [s] is a text ({!Store_format.section-names}). *)

val utf8 : string -> string
(** [utf8 s] is [s] with each byte that does not start a valid UTF-8 sequence
    replaced by U+FFFD, as text from the system may hold. *)

val is_tag : string -> bool
(** [is_tag s] is [true] iff [s] is a tag. *)

val is_segment : string -> bool
(** [is_segment s] is [true] iff [s] is a key segment. *)

val is_key : string -> bool
(** [is_key s] is [true] iff [s] is a key. *)

(** {1:system The system} *)

val pid : unit -> int

val monotonic : unit -> float
(** [monotonic ()] is seconds on a clock that never steps, from an arbitrary
    origin. *)

val uname : unit -> string * string
(** [uname ()] is the system's name and release, and its architecture. *)

val cpu_model : unit -> string option
val boot_id : unit -> string option
val boot_time : unit -> float option

val machine_id : unit -> string option
(** [machine_id ()] is the machine identity ({!Store_format.section-status}),
    computed once. *)

(** {1:digests BLAKE2b-256} *)

type hasher

val hasher : ?key:string -> unit -> hasher
val feed : hasher -> bytes -> int -> int -> unit

val hex : hasher -> string
(** [hex h] is [h]'s digest in lowercase hexadecimal. [h] is spent. *)

val digest_string : string -> string
(** [digest_string s] is the hexadecimal digest of [s]. *)

(** {1:locks Locks} *)

val try_lock : Unix.file_descr -> [ `Exclusive | `Shared ] -> bool
(** [try_lock fd k] takes a lock of kind [k] without waiting, and is [false] if
    a holder refuses it. *)

val lock : Unix.file_descr -> [ `Exclusive | `Shared ] -> unit
(** [lock fd k] waits for a lock of kind [k]. *)

val unlock : Unix.file_descr -> unit

val with_lock : string -> [ `Exclusive | `Shared ] -> (unit -> 'a) -> 'a
(** [with_lock path k f] is [f ()], run holding a lock of kind [k] on the file
    [path], which it creates if it does not exist. It waits for the lock. *)

val take_writer : Unix.file_descr -> bool
(** [take_writer fd] takes the writer's lock on the log [fd], waiting out
    readers' probes, and is [false] if a writer holds it
    ({!Store_format.section-resume}). *)

val probe : Unix.file_descr -> bool
(** [probe fd] is [true] iff a process holds the lock of the log open as [fd],
    taking a shared lock without waiting and releasing it. *)

val repeat : ('a -> string) -> 'a list -> string option
(** [repeat key l] is the first key of an element of [l] that an earlier element
    has, if any. It allocates nothing. *)

(** {1:files Files} *)

val sync : Unix.file_descr -> unit
(** [sync fd] writes [fd]'s data to the storage, with [F_FULLFSYNC] on macOS. *)

val sync_dir : string -> unit
(** [sync_dir d] syncs the directory [d]'s entries. It does nothing on Windows.
*)

val create_flags : Unix.open_flag list
(** [create_flags] opens a new file for writing, failing if it exists,
    close-on-exec and share-delete, as every file munin writes is opened. *)

val open_read : string -> Unix.file_descr

val open_read_opt : string -> Unix.file_descr option
(** [open_read_opt p] is [open_read p], or [None] if [p] does not exist. *)

val read_all : string -> string

val read_opt : string -> string option
(** [read_opt p] is [p]'s content, or [None] if [p] does not exist. *)

val fstat : Unix.file_descr -> Unix.stats

val pread : Unix.file_descr -> pos:int -> len:int -> string
(** [pread fd ~pos ~len] reads up to [len] bytes at [pos]. *)

val write_all : Unix.file_descr -> string -> unit

val copy : ?dst:string -> string -> string * int
(** [copy ?dst src] is the hexadecimal digest of the file [src]'s bytes and
    their count, read through one buffer. With [dst], it also copies them to the
    new file [dst], digesting the bytes it writes, and syncs it. *)

val mkdirs : string -> unit
(** [mkdirs d] creates [d] and its missing parents. *)

val mkdir : string -> unit
(** [mkdir d] creates [d], if it does not exist, and none of its parents. *)

val entries : string -> string list
(** [entries d] is the names in [d] that do not start with [.], sorted, or [[]]
    if [d] does not exist. *)

val temp_name : string -> string
(** [temp_name dir] is a fresh [.]-named path in [dir]. *)

val replace : ?parents:bool -> string -> string -> unit
(** [replace path data] writes [data] to a [.]-named file beside [path], syncs
    it and renames it over [path]. It creates [path]'s missing directories
    unless [parents] is [false] (defaults to [true]). *)

val remove_tree : string -> unit
(** [remove_tree p] removes the file or directory [p], if it exists. An entry
    that another process removes meanwhile is not an error. On Windows, it
    blocks briefly while another handle holds an entry. *)

val stamp : string -> (float * int * int) option
(** [stamp p] is [p]'s modification time, size and inode, if it exists. *)

val mtime : string -> float option

(** {1:processes Processes} *)

val output : string array -> string option
(** [output argv] is the standard output of the program [argv.(0)], run with
    [argv] and no shell, if it exits with status [0]. Its standard error is
    discarded. *)
