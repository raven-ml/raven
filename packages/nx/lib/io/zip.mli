(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Zip archives, as npz files hold them. *)

(** {1:reading Reading}

    A read checks an entry against the CRC-32 the archive states for it,
    unless that is 0, which PyTorch states when told to compute none
    ([Nx_io]'s preamble). *)

type in_file
(** The type for archives read from a file. *)

type entry
(** The type for an archive's entries. *)

val open_in : Rig.Buffer.t -> in_file
(** [open_in file] is the archive of the disk buffer [file], whose central
    directory and local headers it checks: their bounds, their agreement, and
    that no two entries share a name or overlap.

    Raises [Failure] if they are malformed. *)

val entries : in_file -> entry list
(** [entries a] is [a]'s entries, in the central directory's order. *)

val find : in_file -> string -> entry option
(** [find a name] is [a]'s entry [name], if any. *)

val name : entry -> string
(** [name e] is [e]'s name. *)

(** The type for an entry's bytes: a view of the archive's bytes for a stored
    entry, and new host memory nothing else reads for a compressed one. *)
type data = Stored of Rig.Buffer.t | Inflated of Rig.Buffer.t

val read : in_file -> entry -> data
(** [read a e] is [e]'s bytes, once checked against its CRC-32.

    Raises [Failure] naming [e] if it is encrypted, compressed by a method
    other than DEFLATE, or has a flag other than a data descriptor's, UTF-8's
    and DEFLATE's options, or if its checksum does not match or its DEFLATE
    data is malformed. *)

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
    [name] must pass {!valid_name} and must not be in [a]. Once it raises, [a]
    takes no more entries. *)

val close_out : out_file -> unit
(** [close_out a] writes [a]'s central directory, and leaves [a]'s descriptor
    open. *)
