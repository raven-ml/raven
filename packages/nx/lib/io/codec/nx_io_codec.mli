(*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  --------------------------------------------------------------------------*)

(** Private codec primitives for [Nx_io].

    All bigarray offsets and lengths are byte counts. The functions validate
    every span before entering C. *)

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
