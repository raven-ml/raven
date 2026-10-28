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
    directory and local headers it checks: their bounds, their agreement, and
    that no two entries share a name or overlap.

    Raises [Failure] if they are malformed. *)

val read_npys : in_file -> (string * Nx.host Nx.packed) list
(** [read_npys a] is the arrays of [a]'s entries whose names end in [.npy],
    named without the suffix, in the central directory's order: a value over the
    archive's bytes for a stored entry, over inflated bytes for a compressed
    one. Other entries are not read.

    Raises [Failure] if such an entry is encrypted, compressed by a method other
    than DEFLATE, or has a flag npz does not use, or if its checksum does not
    match or its data is malformed. *)

(** {1:writing Writing} *)

type out_file
(** The type for archives being written. *)

val open_out : Unix.file_descr -> out_file
(** [open_out fd] is an empty archive written to [fd]. [fd] stays its caller's,
    and {!close_out} leaves it open. *)

val valid_name : string -> bool
(** [valid_name name] is [true] iff [name] is valid UTF-8, relative, and free of
    a backslash, NUL bytes, drives and empty, [.] and [..] components. *)

val add_npy : out_file -> string -> string -> Storage.bytes -> unit
(** [add_npy a name header data] writes the npy stream [header] then [data] as
    [a]'s entry [name] and [.npy], compressed if that shrinks its first 64 KiB.
    [name] must pass {!valid_name} and must not be in [a]. *)

val close_out : out_file -> unit
(** [close_out a] writes [a]'s central directory, and leaves [a]'s descriptor
    open. *)
