(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Saves that leave, after a crash, the old file or the whole new one. *)

val write : overwrite:bool -> string -> (Unix.file_descr -> unit) -> unit
(** [write ~overwrite path f] is [f fd] for [fd] a new file beside [path], which
    is then synced and named [path], replacing what [path] names with
    [overwrite] and failing if it names something otherwise, and the directory
    synced. [f] writes through [fd] alone: the file is never reopened by its
    name, which a link put there meanwhile would redirect.

    A failure of [f] or of the sync, rename or link raises [Failure] naming
    [path] and leaves no new file, but a refused rename keeps the new file and
    names it. A failed sync of the directory raises with the new file at [path].
*)

val string : Unix.file_descr -> string -> unit
(** [string fd s] writes [s] to [fd]. *)

val bytes :
  Unix.file_descr ->
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  unit
(** [bytes fd b] writes [b] to [fd]. *)
