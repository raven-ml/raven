(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Ones' complement sums of big-endian 32-bit words (FITS 4.0 §4.4.2.7,
    Appendix J), and the [CHECKSUM] text that makes an HDU sum to −0.

    A sum is an [int] from [0] to {!mask}, a 32-bit sum whose carries wrap
    around to bit 0. *)

type bigbytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for byte bigarrays. *)

val mask : int
(** [mask] is [0xFFFFFFFF], the sum −0. *)

val add : int -> int -> int
(** [add a b] is the ones' complement sum of the sums [a] and [b]. *)

val bigbytes : int -> bigbytes -> int -> int -> int
(** [bigbytes s b off len] is [s] plus the words of the [len] bytes of [b]
    from [off]. [len] is a multiple of 4 and the range lies in [b], which
    [bigbytes] does not check. *)

val string : int -> string -> int
(** [string s str] is [s] plus the words of [str], whose length is a
    multiple of 4, which [string] does not check. *)

(** {1:running Running sums} *)

type summer
(** The type for running sums of a data unit's words, fed in pieces of any
    length. *)

val summer : unit -> summer
(** [summer ()] is an empty running sum, whose first byte fed starts a
    word. *)

val at : int -> summer
(** [at off] is a running sum of bytes that start at byte [off] of a data
    unit, which may fall inside a word. *)

val feed : summer -> bigbytes -> int -> int -> unit
(** [feed s b off len] adds the [len] bytes of [b] from [off] to [s]. *)

val total : summer -> int
(** [total s] is [s]'s sum, a last partial word padded with zeros. *)

(** {1:checksum Checksum} *)

val checksum : int -> string
(** [checksum s] is the 16-character [CHECKSUM] text that makes an HDU
    whose other bytes sum to [s] sum to −0, given that its [CHECKSUM] field
    held ['0000000000000000'] in that sum. *)
