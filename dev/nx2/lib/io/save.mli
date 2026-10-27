(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Saves that leave, after a crash, the old file or the whole new one. {!Nx_io}
    states what a save promises, and [save.ml] how it keeps it. *)

val write : string -> (Unix.file_descr -> unit) -> unit
(** [write path f] calls [f fd] on a new file beside [path]. It then syncs the
    file, renames it to [path] and syncs the directory. [write] owns [fd] and
    closes it once [f] returns or raises, so [f] must not close it. [f] must
    write through [fd] alone, since a link put at the new file's name would
    redirect a reopen.

    A [Failure], [Sys_error] or [Unix.Unix_error] from [f] or from the file's
    sync raises [Failure] naming [path], and an [Invalid_argument] from [f]
    passes unchanged. Both leave no new file. A refused rename raises [Failure]
    naming [path] and keeps the new file, which it names. A failed sync of the
    directory raises [Failure] naming [path], with the new file at [path]. *)

val string : Unix.file_descr -> string -> unit
(** [string fd s] writes [s] to [fd]. *)

val bytes :
  Unix.file_descr ->
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  unit
(** [bytes fd b] writes [b] to [fd]. *)
