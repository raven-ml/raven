(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Rice coding of integers (FITS 4.0 §10.4.1), as cfitsio writes it.

    A stream is the first value as is, then blocks of differences between
    neighbours, mapped to unsigned numbers and Rice-coded with a split
    chosen per block. Values are [width] bytes wide, where [width] is 1, 2 or
    4, and arithmetic is modulo the width. *)

val signed : int -> int -> int
(** [signed w v] reads the [8w]-bit pattern [v] as a signed integer, except
    for bytes ([w = 1]), which are unsigned. *)

val decode :
  width:int ->
  block:int ->
  Checksum.bigbytes ->
  int ->
  int ->
  (int, Bigarray.int_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  int ->
  unit
(** [decode ~width ~block src off len out n] decodes [n] values in blocks of
    [block] from the [len] bytes of [src] at [off] into [out]. Those bytes
    lie in [src] and [out] holds at least [n] values, which [decode] does
    not check. Bytes after the last value are ignored. Raises {!Err.Fail} if
    the stream ends before [n] values or a block's split is past the
    width's. *)

val encode :
  width:int ->
  block:int ->
  (int, Bigarray.int_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  int ->
  string
(** [encode ~width ~block v n] codes the first [n] values of [v] in blocks
    of [block], each wrapped to [width] bytes, as cfitsio's [fits_rcomp]
    does. *)
