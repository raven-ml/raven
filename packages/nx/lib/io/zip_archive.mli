(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** ZIP archives of NPY entries, the container of NPZ files.

    A reader accepts stored and DEFLATE entries, data descriptors, UTF-8
    names and ZIP64 sizes and offsets, and refuses multi-disk and encrypted
    archives. It validates the central directory and each local header,
    and an entry's CRC-32 when the entry is read. It refuses duplicate
    names, and names that are empty, absolute, hold [\\] or NUL, have [:]
    as their second character (as a drive letter does), have an empty, [.]
    or [..] component, or are invalid UTF-8 (ASCII without the UTF-8
    flag). A malformed or refused archive raises
    [Failure]. An entry [name] is stored as the file [name.npy]. *)

type in_file
(** The type for archives open for reading. *)

type out_file
(** The type for archives being written. *)

val open_in : string -> in_file
(** [open_in path] is the archive in the file at [path], read where its
    bytes lie. Raises [Sys_error] if the file cannot be opened or mapped,
    and [Failure] if the archive is malformed. *)

val npy_entries : in_file -> string list
(** [npy_entries a] is the names of [a]'s [.npy] entries without the
    suffix, in the order of the central directory. *)

val read_npy : in_file -> string -> Npy.packed
(** [read_npy a name] is the tensor of [a]'s entry [name ^ ".npy"]. A
    stored entry is copied out of the archive; an inflated one is a view of
    its inflated bytes where {!Npy.view_or_materialize} allows. Raises
    [Not_found] if [a] has no such entry, [Failure] if the entry is
    malformed or its CRC-32 or sizes disagree, and {!Npy.Read_error} if
    its NPY header is malformed. *)

val open_out : Unix.file_descr -> out_file
(** [open_out fd] is an empty archive written to [fd]. [fd] must be at the
    start of its file, since the archive records offsets from there. The
    caller closes [fd] after {!close_out}. *)

val add_npy : out_file -> string -> Npy.packed -> unit
(** [add_npy a name p] writes [p] to [a] as the entry [name ^ ".npy"],
    compressed with DEFLATE if that shrinks its first 64 KiB, and stored
    otherwise. Raises [Invalid_argument] if [a] is closed, if [name] is
    empty, unsafe or invalid UTF-8, or as {!Npy.encode} does; [Failure] if
    [a] has an entry [name]; and [Unix.Unix_error] if a write fails. *)

val close_out : out_file -> unit
(** [close_out a] writes [a]'s central directory and end record, and does
    nothing if [a] is closed. *)
