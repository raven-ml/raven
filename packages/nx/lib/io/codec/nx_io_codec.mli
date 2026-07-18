(*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  --------------------------------------------------------------------------*)

(** Private codec primitives for [Nx_io].

    All bigarray offsets and lengths are byte counts. The functions validate
    every span before entering C. *)

type bytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

val crc32 : _ Bigarray.Array1.t -> off:int -> len:int -> int32
(** [crc32 buf ~off ~len] is the ZIP/gzip CRC-32 of the byte span. *)

val adler32 : _ Bigarray.Array1.t -> off:int -> len:int -> int32
(** [adler32 buf ~off ~len] is the zlib Adler-32 of the byte span. *)

val blit_bytes :
  src:_ Bigarray.Array1.t ->
  src_off:int ->
  dst:_ Bigarray.Array1.t ->
  dst_off:int ->
  len:int ->
  unit
(** [blit_bytes ~src ~src_off ~dst ~dst_off ~len] copies a byte span. *)

val byteswap : _ Bigarray.Array1.t -> element_size:int -> elements:int -> unit
(** [byteswap buf ~element_size ~elements] reverses the bytes of every element
    in place. *)

val reorder_fortran_to_c :
  src:_ Bigarray.Array1.t ->
  src_off:int ->
  dst:_ Bigarray.Array1.t ->
  shape:int array ->
  element_size:int ->
  unit
(** [reorder_fortran_to_c ~src ~src_off ~dst ~shape ~element_size] copies a
    Fortran-contiguous tensor into C-contiguous order. *)

val write_all :
  Unix.file_descr -> _ Bigarray.Array1.t -> off:int -> len:int -> unit
(** [write_all fd buf ~off ~len] writes the complete byte span to [fd]. *)

val inflate_raw_prefix : bytes -> off:int -> len:int -> max_output:int -> bytes
(** [inflate_raw_prefix src ~off ~len ~max_output] inflates at most [max_output]
    bytes from a raw DEFLATE stream. It is intended for bounded format probing;
    reaching [max_output] before the stream ends is success. *)

val inflate_raw_into :
  bytes ->
  src_off:int ->
  src_len:int ->
  skip:int ->
  dst:_ Bigarray.Array1.t ->
  dst_off:int ->
  dst_len:int ->
  int32
(** [inflate_raw_into src ~src_off ~src_len ~skip ~dst ~dst_off ~dst_len]
    inflates a complete raw DEFLATE stream. The first [skip] output bytes are
    validated but discarded and the next [dst_len] bytes are written into [dst].
    The stream must produce exactly [skip + dst_len] bytes. The result is the
    CRC-32 of the complete uncompressed stream. *)

val inflate_raw_to_fd :
  Unix.file_descr ->
  bytes ->
  src_off:int ->
  src_len:int ->
  output_size:int ->
  int32
(** [inflate_raw_to_fd fd src ~src_off ~src_len ~output_size] inflates a raw
    DEFLATE stream directly to [fd]. The output size is checked exactly and the
    result is the CRC-32 of the uncompressed stream. *)

val inflate_raw_member_to_fd :
  Unix.file_descr -> bytes -> src_off:int -> src_len:int -> int * int * int32
(** [inflate_raw_member_to_fd fd src ~src_off ~src_len] inflates the first raw
    DEFLATE stream in the span without consuming trailing bytes. The result is
    [(compressed_size, output_size, crc32)]. *)

type deflate_stats = { crc32 : int32; input_size : int; output_size : int }

val store_to_fd :
  Unix.file_descr ->
  prefix:string ->
  _ Bigarray.Array1.t ->
  off:int ->
  len:int ->
  deflate_stats
(** [store_to_fd fd ~prefix src ~off ~len] writes [prefix] and the byte span
    unchanged to [fd]. The result describes the written stream. *)

val deflate_raw :
  prefix:string -> _ Bigarray.Array1.t -> off:int -> len:int -> bytes
(** [deflate_raw ~prefix src ~off ~len] compresses [prefix] followed by the byte
    span as a raw DEFLATE stream. *)

val deflate_raw_to_fd :
  Unix.file_descr ->
  prefix:string ->
  _ Bigarray.Array1.t ->
  off:int ->
  len:int ->
  deflate_stats
(** [deflate_raw_to_fd fd ~prefix src ~off ~len] writes a raw DEFLATE stream to
    [fd]. The result describes the compressed and uncompressed streams. *)
