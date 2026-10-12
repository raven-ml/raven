(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Fixture files and corpora shared by the compress suites. *)

type bigbytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

val read : string -> string
(** [read name] is the contents of the fixture file [name]. *)

val bigbytes_of_string : string -> bigbytes
val string_of_bigbytes : bigbytes -> string

val zeros : int -> bigbytes
(** [zeros n] is [n] zero bytes. *)

(** {1:corpora Corpora}

    The bytes that [generate.py] compresses into the fixtures. *)

val lines : int -> string
(** [lines n] is ["line 0\n"] to ["line n-1\n"], concatenated. *)

val text : string
(** [text] is [lines 8000]: longer than a deflate block and window. *)

val columns : string
(** [columns] is 2048 rows of an increasing int64 key and a float64 price, as a
    Parquet page holds them. *)

val runs : string
(** [runs] is 100 runs of 1000 equal bytes. *)

val random : string
(** [random] is 20000 pseudo-random bytes. *)

val long : ?size:int Windtrap.Gen.t -> char -> char -> string Windtrap.Gen.t
(** [long ~size lo hi] draws strings of [size] bytes (defaults to 60000 to
    200000), each from [lo] to [hi], made from a drawn seed in one pass. It
    shrinks their length and seed. A string drawn a character at a time costs
    a shrink tree for each, which made a property of 1000 such cases run a
    minute. *)
