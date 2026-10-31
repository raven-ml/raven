(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** IRAF's PLIO line lists (FITS 4.0 §10.4.3): run-length coded masks of
    integers.

    A list is big-endian 16-bit words: a header, then instructions of a
    clear sign bit, a 3-bit opcode and 12 bits of data that output runs of
    zeros or of a high value, or set, raise or lower that value, which
    starts at 1. The header is three words with the list's length third, or
    a version below 1 third, its length second and the list's length in
    the fourth and fifth, low 15 bits first. The length counts the header's
    words. Values past the list's end are zeros. *)

val decode :
  Checksum.bigbytes ->
  int ->
  int ->
  int ->
  until:int ->
  (int -> int -> int -> unit) ->
  unit
(** [decode src off len n ~until fill] runs the list of [n] values in the
    [len] bytes of [src] at [off], calling [fill x m v] for each run of [m]
    nonzero values [v] from value [x]: the values no run reports are zeros.
    It stops at value [until], and runs from there on may go unreported.
    Those bytes lie in [src], which [decode] does not check. Instructions
    after the [n]th value are ignored. Raises {!Err.Fail} if the header is
    short or gives more words than [len] bytes hold, an SH before value
    [until] lacks its second word, or a word before it has its sign bit
    set. *)
