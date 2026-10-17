(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NumPy's NPY format: headers, and tensors read from and written to it.

    A reader accepts header versions 1.0, 2.0 and 3.0, with a dictionary of
    exactly the keys [descr], [fortran_order] and [shape] and a header of at
    most 1 MiB. It reads the dtypes [f2], [f4], [f8], the signed and unsigned
    integers of 1, 2, 4 and 8 bytes, [c8], [c16] and [b1], in either byte
    order. A writer writes C order, in the host's byte order, with a header
    padded so that the data starts at a multiple of 64 bytes, at version 1.0
    or at 2.0 when the header needs it. A malformed or unsupported stream
    raises {!Read_error}. *)

exception Read_error of string
(** The exception for a stream that is malformed or that the reader does not
    support, with a message that says why. *)

type bytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for streams' bytes. *)

type packed_kind =
  | K : ('a, 'b) Nx_dtype.t -> packed_kind  (** The type for dtypes. *)

type header = {
  kind : packed_kind;
  shape : int array;
  fortran_order : bool;
  swap_endian : bool;
      (** [true] iff the elements' byte order is not the host's. *)
  elements : int;
  element_size : int;  (** In bytes. *)
  data_offset : int;  (** The data's first byte, from the stream's. *)
  data_size : int;  (** In bytes, [elements * element_size]. *)
}
(** The type for parsed headers. *)

type packed =
  | P : ('a, 'b) Nx_dtype.t * Rig.Buffer.t * int array -> packed
      (** The type for tensors: a dtype, a host buffer of the elements in C
          order, and a shape. *)

type encoded = { header : string; data : bytes }
(** The type for encoded tensors: the stream's header and its data, whose
    concatenation is the stream. *)

val parse_header : bytes -> off:int -> len:int -> header
(** [parse_header src ~off ~len] is the header of the stream of [len] bytes
    at byte [off] of [src]. It checks that the header fits in [len] bytes,
    and leaves the data's size to its caller. Raises [Invalid_argument] if
    the span is not in [src], and {!Read_error} if the header is malformed
    or unsupported. *)

val materialize : header -> src:bytes -> off:int -> packed
(** [materialize h ~src ~off] is the tensor of [h] whose data starts at
    byte [off] of [src], copied into a buffer of its own in C order and in
    the host's byte order. [src] must hold [h.data_size] bytes from
    [off]. *)

val view_or_materialize : header -> src:bytes -> off:int -> packed
(** [view_or_materialize h ~src ~off] is {!materialize}[ h ~src ~off], or a
    view of [src] without a copy when the data is in C order and in the
    host's byte order and [off] is a multiple of the element size. *)

val read_copy : string -> packed
(** [read_copy path] is the tensor of the NPY file at [path], copied as
    {!materialize} does. Raises {!Read_error} if the data's size is not
    the header's, and [Sys_error] if the file cannot be read. *)

val no_bit : string
(** [no_bit] is the message that refuses to save a [bit] tensor as NPY. *)

val encode : packed -> (encoded -> 'a) -> 'a
(** [encode p f] is [f e] for [e] the encoding of [p], whose [data] is valid
    while [f] runs. Raises [Invalid_argument] if [p]'s dtype has no NPY
    code: [bit], [bfloat16], the [float8] dtypes, [int4] and [uint4]. *)

val write : packed -> Unix.file_descr -> unit
(** [write p fd] writes the encoding of [p] to [fd]. Raises what
    {!encode} raises, and [Unix.Unix_error] if a write fails. *)
