(*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  --------------------------------------------------------------------------*)

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
