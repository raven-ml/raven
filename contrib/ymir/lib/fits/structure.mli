(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The structural keywords the writer owns, and how they enter a header.

    A record that holds a keyword's computed value stays at its place with
    its text. The others are printed at the standard's positions, the
    mandatory keywords first and in order (FITS 4.0 §4.4.1), the rest after
    them, and the caller's differing copies go. Raises [Invalid_argument]
    where {!Header.set} does. *)

(** {1:entries Entries} *)

type entry
(** The type for a structural keyword with its computed value: the records
    that print it, and a test of whether a record holds that value. *)

val int : string -> int -> entry
(** [int k n] is [k] with the integer value [n]. *)

val bool : string -> bool -> entry
(** [bool k b] is [k] with the logical value [b]. *)

val string : string -> string -> entry
(** [string k s] is [k] with the string value [s]. *)

val decimal : string -> string -> entry
(** [decimal k t] is [k] with the number of the real text [t], as
    [BZERO = 9223372036854775808]. A record holds it when its value writes
    the same decimal ({!Decimal.equal}). *)

val verbatim : string -> string list -> entry
(** [verbatim k rs] is [k] with the records [rs], held elsewhere; a record
    holds it when it equals [rs]'s first. *)

val apply :
  owned:(string -> bool) -> prefix:entry list -> others:entry list ->
  Header.t -> Header.t
(** [apply ~owned ~prefix ~others h] is [h] with the structure [prefix] at
    its start, in order, and [others], placed where [h] holds them or after
    the prefix. Every other record of a keyword that [owned] accepts, that
    is {!mandatory} or that is a {!checksums} keyword is dropped. *)

val image_prefix : primary:bool -> bitpix:int -> axes:int array -> entry list
(** [image_prefix ~primary ~bitpix ~axes] is the mandatory keywords of an
    image of [BITPIX] [bitpix] and [NAXISn] [axes], in file order: a
    primary's, or an [IMAGE] extension's. *)

(** {1:classes Keyword classes} *)

val mandatory : string -> bool
(** [mandatory k] is [true] iff [k] is a mandatory keyword of a primary or
    an extension (§4.4.1), [GROUPS] and [TFIELDS] included. *)

val checksums : string -> bool
(** [checksums k] is [true] iff [k] is [DATASUM] or [CHECKSUM]. *)

val image_owned : string -> bool
(** [image_owned k] is [true] iff [k] is [BZERO] or [BSCALE]. *)

val column_numbered : string -> bool
(** [column_numbered k] is [true] iff [k] is a column-numbered keyword: one
    of FITS 4.0 Table 8, or the column form of a WCS keyword (Table 26), with
    its alternate letter. *)

val table_owned : string -> bool
(** [table_owned k] is [true] iff [k] is [THEAP] or column-numbered. *)

val tile_owned : string -> bool
(** [tile_owned k] is [true] iff [k] is a tile-compression keyword. *)

val is_digit : char -> bool
(** [is_digit c] is [true] iff [c] is in ['0'] to ['9']. *)
