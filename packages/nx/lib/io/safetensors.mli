(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The SafeTensors format: headers, parsed and laid out.

    A file is an 8-byte little-endian header length, the header, and the
    tensors' bytes. The header is a UTF-8 JSON object of the tensors by
    name, each with its dtype, shape and the offsets of its bytes from the
    header's end, and an optional ["__metadata__"] object of strings. The
    tensors' bytes are contiguous, in the order of their offsets, and each
    tensor's size is that of its dtype and shape. *)

type error
(** The type for header errors. *)

val string_of_error : error -> string
(** [string_of_error e] is [e]'s message. *)

(** The type for dtypes. [F4], [F6_E2M3] and [F6_E3M2] have elements
    narrower than a byte. *)
type dtype =
  | BOOL
  | F4
  | F6_E2M3
  | F6_E3M2
  | U8
  | I8
  | F8_E5M2
  | F8_E4M3
  | F8_E8M0
  | I16
  | U16
  | F16
  | BF16
  | I32
  | U32
  | F32
  | F64
  | I64
  | U64

type tensor_info = {
  dtype : dtype;
  shape : int list;
  data_offsets : int * int;
      (** The first byte and the byte after the last, from the header's
          end. *)
}
(** The type for a tensor's entry in a header. *)

type metadata = {
  metadata_kv : (string * string) list option;
      (** The ["__metadata__"] object, if any. *)
  tensors : tensor_info array;  (** In the order of their offsets. *)
  index_map : (string, int) Hashtbl.t;
      (** Each tensor's name, to its index in [tensors]. *)
}
(** The type for parsed headers. *)

type tensor_view
(** The type for tensors to write, by their dtype, shape and size. *)

val tensor_view_new :
  dtype:dtype -> shape:int list -> nbytes:int -> (tensor_view, error) result
(** [tensor_view_new ~dtype ~shape ~nbytes] is a tensor to write of [nbytes]
    bytes. It is [Error] if a dimension is negative, if the size overflows,
    if the elements do not end on a byte, or if [nbytes] is not the size of
    [dtype] and [shape]. *)

val max_header_size : int
(** [max_header_size] is [100_000_000], the longest header, in bytes, that
    a reader accepts. *)

val header_len_bytes : int
(** [header_len_bytes] is [8], the size of the header length that starts a
    file. *)

val parse_header : string -> (metadata * int, error) result
(** [parse_header h] is the header [h], read without its length prefix,
    and the size of the tensors' bytes it describes. It is [Error] if [h]
    is not valid UTF-8 or JSON, names a tensor twice, has an unknown dtype,
    or if the tensors' offsets or sizes disagree. *)

val layout :
  (string * tensor_view) list ->
  (string * string) list option ->
  (string * (string * int) list, error) result
(** [layout data kv] is the header of a file of [data]'s tensors and the
    metadata [kv], length prefix included and padded with spaces to a
    multiple of 8 bytes, and each tensor's name with the offset of its
    bytes from the header's end. Tensors are laid out by dtype, in the
    order [U64], [I64], [F64], [F32], [U32], [I32], [BF16], [F16], [U16],
    [I16], [F8_E8M0], [F8_E4M3], [F8_E5M2], [I8], [U8], [F6_E3M2],
    [F6_E2M3], [F4], [BOOL], then by name. The names of [data] must be
    distinct. *)
