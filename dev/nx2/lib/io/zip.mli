(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Zip archives of npy files, as npz holds them. *)

(** {1:reading Reading} *)

type in_file
(** The type for archives read from a file. *)

val open_in : Rig.Buffer.t -> in_file
(** [open_in file] is the archive of the disk buffer [file], whose central
    directory and local headers it checks.

    Raises [Failure] if they are malformed. *)

val npy_entries : in_file -> string list
(** [npy_entries a] is the names of [a]'s entries that end in [.npy], without
    the suffix, in the central directory's order. *)

val read_npy : in_file -> string -> Nx.host Nx.packed
(** [read_npy a name] is the array of [a]'s entry [name] and [.npy]: a value
    over the archive's bytes for a stored entry, over inflated bytes for a
    compressed one.

    Raises [Failure] if there is no such entry, its checksum does not match, or
    its data is malformed. *)

(** {1:writing Writing} *)

type out_file
(** The type for archives being written. *)

val open_out : Unix.file_descr -> out_file
(** [open_out fd] is an empty archive written to [fd]. *)

val valid_name : string -> bool
(** [valid_name name] is [true] iff [name] is valid UTF-8, relative, and free of
    a backslash, NUL bytes, drives and empty, [.] and [..] components. *)

val add_npy : out_file -> string -> string -> Storage.bytes -> unit
(** [add_npy a name header data] writes the npy stream [header] then [data] as
    [a]'s entry [name] and [.npy], compressed if that shrinks its first 64 KiB.
    [name] is valid and not yet in [a]. *)

val close_out : out_file -> unit
(** [close_out a] writes [a]'s central directory. *)
