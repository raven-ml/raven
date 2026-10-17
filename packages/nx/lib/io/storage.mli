(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Values' elements in host buffers, and host buffers as bytes. *)

val claiming : Rig.Buffer.t list -> (unit -> 'a) -> 'a
(** [claiming buffers f] is [f ()] run with [buffers] under read claims, so
    that no compiled call lends their memory while [f] reads them. *)

val reading : by:string -> ('a, 'b) Nx.t -> (Rig.Buffer.t -> 'c) -> 'c
(** [reading ~by x f] is [f b] for [b] a host buffer of the elements of [x]
    in C order, under a read claim while [f] runs. [b] is [x]'s storage when
    the elements are one run of it on the host, and a copy otherwise, so
    [f] must not write [b]. [by] is the public function that reads, which
    errors name. *)

val read_bytes :
  Rig.Buffer.t ->
  ((int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  'a) ->
  'a
(** [read_bytes b f] is [f] of the bytes of [b], once its last write is
    done ({!Rig.Buffer.read}). *)

val write_bytes :
  Rig.Buffer.t ->
  ((int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  'a) ->
  'a
(** [write_bytes b f] is [f] of the bytes of [b], once every use of [b] is
    done ({!Rig.Buffer.write}). *)

val file : string -> Rig.Buffer.t
(** [file path] is the bytes of the file at [path] where they lie, as the
    host's borrow of its pages. Raises [Sys_error] if the file cannot be
    opened or mapped. *)

val mapped :
  Rig.Buffer.t ->
  ('a, 'b) Nx_dtype.t ->
  int array ->
  off:int ->
  len:int ->
  Nx.packed
(** [mapped file dtype shape ~off ~len] is the value of [shape] over the
    [len] bytes at byte [off] of [file], read as little-endian elements of
    [dtype]. On a little-endian host it lies over [file]'s bytes. On a
    big-endian host it is a copy in the host's byte order. *)
