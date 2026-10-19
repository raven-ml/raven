(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Files as bytes, and values over them. *)

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
(** [pages file] is [file]'s bytes on the host: the host's borrow of its pages,
    or a copy where the system cannot map it. *)

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
    bytes of a file or of host memory, in the host's byte order. It reads [b]'s
    own memory where [b] starts on a multiple of [alignment dt] and the host
    maps it, and a copy of [b]'s bytes otherwise.

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
    {!Nx_array.read}), which leaves [x]'s memory unshared. [x]'s dtype is of a
    byte or more. [f] must not keep [b].

    Raises [Invalid_argument] naming [by] if [x] is dead or traced. *)
