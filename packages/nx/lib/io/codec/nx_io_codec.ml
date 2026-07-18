(*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  --------------------------------------------------------------------------*)

type bytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external crc32 : _ Bigarray.Array1.t -> off:int -> len:int -> int32
  = "caml_nx_io_crc32"

external adler32 : _ Bigarray.Array1.t -> off:int -> len:int -> int32
  = "caml_nx_io_adler32"

external blit_bytes_native :
  _ Bigarray.Array1.t -> int -> _ Bigarray.Array1.t -> int -> int -> unit
  = "caml_nx_io_blit_bytes"

let blit_bytes ~src ~src_off ~dst ~dst_off ~len =
  blit_bytes_native src src_off dst dst_off len

external byteswap_native : _ Bigarray.Array1.t -> int -> int -> unit
  = "caml_nx_io_byteswap"

let byteswap buf ~element_size ~elements =
  byteswap_native buf element_size elements

external reorder_fortran_to_c_native :
  _ Bigarray.Array1.t -> int -> _ Bigarray.Array1.t -> int array -> int -> unit
  = "caml_nx_io_reorder_fortran_to_c"

let reorder_fortran_to_c ~src ~src_off ~dst ~shape ~element_size =
  reorder_fortran_to_c_native src src_off dst shape element_size

external write_all :
  Unix.file_descr -> _ Bigarray.Array1.t -> off:int -> len:int -> unit
  = "caml_nx_io_write_all"

external inflate_raw_prefix :
  bytes -> off:int -> len:int -> max_output:int -> bytes
  = "caml_nx_io_inflate_raw_prefix"

external inflate_raw_into_native :
  bytes -> int -> int -> int -> _ Bigarray.Array1.t -> int -> int -> int32
  = "caml_nx_io_inflate_raw_into_bytecode" "caml_nx_io_inflate_raw_into"

let inflate_raw_into src ~src_off ~src_len ~skip ~dst ~dst_off ~dst_len =
  inflate_raw_into_native src src_off src_len skip dst dst_off dst_len

external inflate_raw_to_fd_native :
  Unix.file_descr -> bytes -> int -> int -> int -> int32
  = "caml_nx_io_inflate_raw_to_fd"

let inflate_raw_to_fd fd src ~src_off ~src_len ~output_size =
  inflate_raw_to_fd_native fd src src_off src_len output_size

external inflate_raw_member_to_fd_native :
  Unix.file_descr -> bytes -> int -> int -> int * int * int32
  = "caml_nx_io_inflate_raw_member_to_fd"

let inflate_raw_member_to_fd fd src ~src_off ~src_len =
  inflate_raw_member_to_fd_native fd src src_off src_len

type deflate_stats = { crc32 : int32; input_size : int; output_size : int }

external store_to_fd_native :
  Unix.file_descr ->
  string ->
  _ Bigarray.Array1.t ->
  int ->
  int ->
  int32 * int * int = "caml_nx_io_store_to_fd"

let store_to_fd fd ~prefix src ~off ~len =
  let crc32, input_size, output_size =
    store_to_fd_native fd prefix src off len
  in
  { crc32; input_size; output_size }

external deflate_raw_native :
  string -> _ Bigarray.Array1.t -> int -> int -> bytes
  = "caml_nx_io_deflate_raw"

let deflate_raw ~prefix src ~off ~len = deflate_raw_native prefix src off len

external deflate_raw_to_fd_native :
  Unix.file_descr ->
  string ->
  _ Bigarray.Array1.t ->
  int ->
  int ->
  int32 * int * int = "caml_nx_io_deflate_raw_to_fd"

let deflate_raw_to_fd fd ~prefix src ~off ~len =
  let crc32, input_size, output_size =
    deflate_raw_to_fd_native fd prefix src off len
  in
  { crc32; input_size; output_size }
