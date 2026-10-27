(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Files as bytes, and values over them.

    A value over a file's pages keeps the host's borrow of the file's buffer,
    which keeps the file mapped while the value is reachable
    ({!Rig_disk.section-borrows}). rig.disk may close the file's descriptor
    meanwhile, and reopens it by path for a later copy
    ({!Rig_disk.section-descriptors}). *)

type bytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

(** {1:files Files} *)

val open_file : string -> Rig.Buffer.t
(** [open_file path] is the bytes of the file [path] on the disk.

    Raises [Sys_error] starting with [path] if it cannot be opened. *)

val string : Rig.Buffer.t -> pos:int -> int -> string
(** [string b ~pos n] is the [n] bytes of [b] from byte [pos].

    Raises [Failure] if they do not lie in [b]. *)

val pages : Rig.Buffer.t -> Rig.Buffer.t
(** [pages file] is [file]'s bytes on the host. They are the file's pages,
    mapped into the host, or a copy where the system cannot map the file. *)

val read : Rig.Buffer.t -> (bytes -> 'a) -> 'a
(** [read b f] is [f] of [b]'s bytes, [b] on the host ({!Rig.Buffer.read}). *)

(** {1:loading Loading} *)

val check_shape : int array -> unit
(** [check_shape s] is [()] if a layout has the shape [s].

    Raises [Failure] naming [s] otherwise. *)

val alignment : ('v, 's) Nx_array.Dtype.t -> int
(** [alignment dt] is the alignment of [dt]'s elements in bytes. *)

val value :
  Rig.Buffer.t ->
  ('v, 's) Nx_array.Dtype.t ->
  Nx_array.Layout.t ->
  ('v, 's, Nx.host) Nx.t
(** [value b dt l] is the host value of [dt] elements laid out by [l] over [b],
    bytes of a file or of host memory, in the host's byte order. Where [b]
    starts on a multiple of [alignment dt] and the host maps it, the value reads
    [b]'s own memory, and its donation writes its result elsewhere
    ({!Nx.Repr.of_array}). Otherwise it reads a copy of [b]'s bytes, which it
    owns ({!Nx.Repr.unsafe_of_array}).

    Raises [Invalid_argument] as {!Nx_array.v} does. *)

val copied :
  Rig.Buffer.t ->
  ('v, 's) Nx_array.Dtype.t ->
  Nx_array.Layout.t ->
  ('v, 's, Nx.host) Nx.t
(** [copied b dt l] is {!value} over a copy of [b]'s bytes in fresh host memory,
    which the value owns ({!Nx.Repr.unsafe_of_array}): for bytes nothing else
    may read once loaded, such as an inflated entry's.

    Raises [Invalid_argument] as {!Nx_array.v} does. *)

val byteswap : ('v, 's, Nx.host) Nx.t -> ('v, 's, Nx.host) Nx.t
(** [byteswap x] is [x] with the bytes of each element reversed, those of each
    component for a complex dtype. *)

val little :
  Rig.Buffer.t ->
  ('v, 's) Nx_array.Dtype.t ->
  Nx_array.Layout.t ->
  ('v, 's, Nx.host) Nx.t
(** [little b dt l] is {!value} of little-endian elements: swapped on a
    big-endian host. *)

(** {1:saving Saving} *)

val with_bytes : by:string -> ('v, 's, 'd) Nx.t -> (bytes -> 'a) -> 'a
(** [with_bytes ~by x f] is [f b], [b] the bytes of [x]'s elements in C order,
    little-endian, read on the host for the extent of [f] ({!Nx.Repr.read},
    {!Nx_array.read}), which leaves [x]'s memory unshared. [x]'s dtype must be
    of a byte or more, and [f] must not keep [b].

    Raises [Invalid_argument] naming [by] if [x] is dead or traced, or its
    memory is held exclusive, and a failed check pending on [x] as
    {!Nx.Repr.read} does. *)
